//! Abduction–action–prediction engine.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use causal_core::{ExecutionContext, Intervention, VariableId};
use causal_data::{TableView, TabularData};
use causal_graph::DenseNodeId;
use causal_model::{
    CompiledCausalModel, InterventionOverlay, MechanismSlot, MechanismWorkspace, ParentBatch,
    ValueBatchMut, evaluate_column, infer_noise_column, sample_stochastic, soft_to_slot,
};

use crate::error::CounterfactualError;

/// Policy for missing factual columns during abduction.
///
/// Distinct from [`causal_data::MissingPolicy`] (sample construction).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AbductionMissingPolicy {
    /// Fail if a required factual column is absent.
    Error,
    /// Zero-fill absent columns and mark them as assumed noise.
    ZeroFill,
}

impl AbductionMissingPolicy {
    /// Whether absent columns are zero-filled.
    #[must_use]
    pub const fn allows_missing(self) -> bool {
        matches!(self, Self::ZeroFill)
    }
}

impl From<bool> for AbductionMissingPolicy {
    fn from(allow_missing: bool) -> Self {
        if allow_missing { Self::ZeroFill } else { Self::Error }
    }
}

/// How exogenous noise was obtained.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NoiseInferenceKind {
    /// Exact inversion of invertible structural assignments.
    Invertible,
    /// Posterior / sampled noise (not used in base path).
    PosteriorNoise,
    /// Assumed independent noise draws (no abduction).
    AssumedNoise,
}

/// Exogenous state for one or more factual units (columnar).
#[derive(Clone, Debug)]
pub struct ExogenousPosterior {
    /// Noise values: `noise[node * n_units + unit]`.
    pub noise: Arc<[f64]>,
    /// Units.
    pub n_units: usize,
    /// Nodes.
    pub n_nodes: usize,
    /// Inference kind.
    pub kind: NoiseInferenceKind,
    /// Per-node flag: factual column was absent and zero-filled under [`AbductionMissingPolicy::ZeroFill`]
    /// (`true` ⇒ that node's factual values must not be treated as observed).
    pub assumed_columns: Arc<[bool]>,
}

/// One counterfactual world request.
#[derive(Clone, Debug)]
pub struct CounterfactualWorld {
    /// Unit row indices into the factual table (`None` = all rows).
    pub unit_rows: Option<Arc<[usize]>>,
    /// Interventions applied after abduction.
    pub interventions: Arc<[Intervention]>,
}

/// Compiled plan: abduction gather order shared across worlds.
#[derive(Clone, Debug)]
pub struct CompiledCounterfactualPlan {
    /// Topological order (from model).
    pub node_order: Arc<[DenseNodeId]>,
}

/// Counterfactual engine over an invertible SCM.
#[derive(Clone, Debug)]
pub struct CounterfactualEngine {
    /// Fitted invertible model (shared; clone of the engine is cheap).
    pub model: Arc<CompiledCausalModel>,
    /// Compiled plan.
    pub compiled: CompiledCounterfactualPlan,
}

impl CounterfactualEngine {
    /// Own the model behind an Arc.
    #[must_use]
    pub fn new(model: CompiledCausalModel) -> Self {
        Self::shared(Arc::new(model))
    }

    /// Share an existing Arc-backed model (no clone of model contents).
    #[must_use]
    pub fn shared(model: Arc<CompiledCausalModel>) -> Self {
        let compiled = CompiledCounterfactualPlan { node_order: Arc::clone(&model.node_order) };
        Self { model, compiled }
    }

    /// Clone model into a new Arc (prefer [`Self::shared`] when already Arc-backed).
    #[must_use]
    pub fn from_ref(model: &CompiledCausalModel) -> Self {
        Self::shared(Arc::new(model.clone()))
    }

    /// Abduce exogenous noise once from factual data (shared across worlds).
    ///
    /// When `allow_missing` is true and a **whole variable column** is absent from
    /// `data`, that node's factual values are filled with zeros and
    /// [`NoiseInferenceKind::AssumedNoise`] is set for the posterior (column-level,
    /// not per-cell — tabular float columns are all-or-nothing here).
    ///
    /// # Errors
    ///
    /// Missing data (when not allowed) or non-invertible mechanisms.
    pub fn abduct(
        &self,
        data: &TabularData,
        missing: AbductionMissingPolicy,
    ) -> Result<ExogenousPosterior, CounterfactualError> {
        let n = data.row_count();
        let n_nodes = self.model.n_nodes();
        let mut noise = vec![0.0; n * n_nodes];
        let mut kind = NoiseInferenceKind::Invertible;
        let mut assumed_columns = vec![false; n_nodes];
        let mut ws = MechanismWorkspace::default();

        // Load factual values.
        let mut values = vec![0.0; n * n_nodes];
        for (i, &var) in self.model.output_layout.variables.iter().enumerate() {
            match data.float64_values(var) {
                Ok(col) => values[i * n..(i + 1) * n].copy_from_slice(&col[..n]),
                Err(e) => {
                    if missing.allows_missing() {
                        kind = NoiseInferenceKind::AssumedNoise;
                        assumed_columns[i] = true;
                        values[i * n..(i + 1) * n].fill(0.0);
                        let _ = e;
                    } else {
                        return Err(CounterfactualError::MissingFactual {
                            message: format!("variable {var}: {e}"),
                        });
                    }
                }
            }
        }

        for gather in self.model.parent_gathers.iter() {
            let node = gather.child;
            let idx = node.as_usize();
            ws.prepare(n, gather.n_parents().max(1));
            gather.gather(&values, n, &mut ws.parents);
            let parent_owned = ws.parents[..gather.n_parents().saturating_mul(n)].to_vec();
            let parents =
                ParentBatch { n_rows: n, n_parents: gather.n_parents(), values: &parent_owned };
            let y = &values[idx * n..(idx + 1) * n];
            let out = &mut noise[idx * n..(idx + 1) * n];
            match self.model.mechanisms.get(node) {
                MechanismSlot::LinearGaussian { .. }
                | MechanismSlot::HierarchicalLinear { .. }
                | MechanismSlot::Bvar { .. }
                | MechanismSlot::Constant { .. } => {
                    infer_noise_column(self.model.mechanisms.get(node), y, parents, out)?;
                }
                other => {
                    return Err(CounterfactualError::model_msg(format!(
                        "abduction requires invertible mechanism, got {other:?}"
                    )));
                }
            }
        }

        Ok(ExogenousPosterior {
            noise: Arc::from(noise),
            n_units: n,
            n_nodes,
            kind,
            assumed_columns: Arc::from(assumed_columns),
        })
    }

    /// Predict counterfactual outcomes for worlds sharing abduced noise.
    ///
    /// # Errors
    ///
    /// Overlay / evaluation failures.
    pub fn predict(
        &self,
        exo: &ExogenousPosterior,
        worlds: &[CounterfactualWorld],
        outcomes: &[VariableId],
        allow_nested: bool,
        ws: &mut MechanismWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CounterfactualResult, CounterfactualError> {
        if worlds.is_empty() {
            return Err(CounterfactualError::model_msg("no worlds"));
        }
        let n_units = exo.n_units;
        let n_nodes = exo.n_nodes;
        let n_worlds = worlds.len();
        let mut outcome_dense = Vec::with_capacity(outcomes.len());
        for &o in outcomes {
            outcome_dense.push(self.model.dense_of(o).ok_or_else(|| {
                CounterfactualError::model_msg(format!("outcome {o} not in model"))
            })?);
        }

        // Columnar: values[world * (n_nodes * n_units) + node * n_units + unit]
        let mut all = vec![0.0; n_worlds * n_nodes * n_units];
        let mut notes = Vec::new();
        notes.push(Arc::from(format!("noise_inference={:?}", exo.kind)));
        let mut rng = ctx.rng.stream(0xCF_01);

        for (wi, world) in worlds.iter().enumerate() {
            if !allow_nested {
                for iv in world.interventions.iter() {
                    if matches!(iv, Intervention::Sequence(_)) {
                        return Err(CounterfactualError::NestedNotAllowed);
                    }
                }
            }
            let overlay =
                InterventionOverlay::from_interventions(&self.model, &world.interventions)?;
            let unit_filter: Option<&[usize]> = world.unit_rows.as_ref().map(AsRef::as_ref);

            let mut values_buf = vec![0.0; n_units * n_nodes];
            let mut values = ValueBatchMut::new(n_units, n_nodes, &mut values_buf)?;

            for gather in self.model.parent_gathers.iter() {
                let node = gather.child;
                let idx = node.as_usize();
                ws.prepare(n_units, gather.n_parents().max(1));
                gather.gather(values.values, n_units, &mut ws.parents);
                let parent_owned =
                    ws.parents[..gather.n_parents().saturating_mul(n_units)].to_vec();
                let parents = ParentBatch {
                    n_rows: n_units,
                    n_parents: gather.n_parents(),
                    values: &parent_owned,
                };
                let out = values.column_mut(idx)?;
                if let Some(v) = overlay.hard_set[idx] {
                    out.fill(v);
                    continue;
                }
                if let Some(policy) = &overlay.stochastic[idx] {
                    sample_stochastic(policy, n_units, &mut rng, out)?;
                    if overlay.shifts[idx] != 0.0 {
                        for x in out.iter_mut() {
                            *x += overlay.shifts[idx];
                        }
                    }
                    continue;
                }
                let noise_col = &exo.noise[idx * n_units..(idx + 1) * n_units];
                let slot = if let Some(soft) = &overlay.soft[idx] {
                    soft_to_slot(soft, gather.n_parents())?
                } else {
                    self.model.mechanisms.get(node).clone()
                };
                evaluate_column(&slot, parents, noise_col, out, ws)?;
                if overlay.shifts[idx] != 0.0 {
                    for x in out.iter_mut() {
                        *x += overlay.shifts[idx];
                    }
                }
            }

            let base = wi * n_nodes * n_units;
            for node in 0..n_nodes {
                let src = &values_buf[node * n_units..(node + 1) * n_units];
                let dest = base + node * n_units;
                if let Some(rows) = unit_filter {
                    let mut tmp = vec![f64::NAN; n_units];
                    for &r in rows {
                        if r < n_units {
                            tmp[r] = src[r];
                        }
                    }
                    all[dest..dest + n_units].copy_from_slice(&tmp);
                } else {
                    all[dest..dest + n_units].copy_from_slice(src);
                }
            }
        }

        Ok(CounterfactualResult {
            values: Arc::from(all),
            n_worlds,
            n_units,
            n_nodes,
            outcomes: Arc::from(outcome_dense),
            noise_kind: exo.kind,
            notes,
        })
    }

    /// Individual treatment effect: E[Y_{do(a)} - Y_{do(c)} | U] per unit (point CF).
    ///
    /// # Errors
    ///
    /// Prediction failures.
    pub fn individual_treatment_effect(
        &self,
        exo: &ExogenousPosterior,
        outcome: VariableId,
        active: Intervention,
        control: Intervention,
        ws: &mut MechanismWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Arc<[f64]>, CounterfactualError> {
        let worlds = [
            CounterfactualWorld { unit_rows: None, interventions: Arc::from([active]) },
            CounterfactualWorld { unit_rows: None, interventions: Arc::from([control]) },
        ];
        let res = self.predict(exo, &worlds, &[outcome], false, ws, ctx)?;
        let o = self.model.dense_of(outcome).ok_or_else(|| {
            CounterfactualError::model_msg(format!("unknown outcome variable {outcome}"))
        })?;
        let mut ite = vec![0.0; exo.n_units];
        for u in 0..exo.n_units {
            let ya = res.get(0, o, u);
            let yc = res.get(1, o, u);
            ite[u] = ya - yc;
        }
        Ok(Arc::from(ite))
    }
}

/// Counterfactual result tensor (world × node × unit), columnar.
#[derive(Clone, Debug)]
pub struct CounterfactualResult {
    /// Flat storage.
    pub values: Arc<[f64]>,
    /// Worlds.
    pub n_worlds: usize,
    /// Units.
    pub n_units: usize,
    /// Nodes.
    pub n_nodes: usize,
    /// Requested outcome dense ids.
    pub outcomes: Arc<[DenseNodeId]>,
    /// Noise inference kind (visible assumption).
    pub noise_kind: NoiseInferenceKind,
    /// Notes.
    pub notes: Vec<Arc<str>>,
}

impl CounterfactualResult {
    /// Value at world/node/unit.
    #[must_use]
    pub fn get(&self, world: usize, node: DenseNodeId, unit: usize) -> f64 {
        let i = world * self.n_nodes * self.n_units + node.as_usize() * self.n_units + unit;
        self.values.get(i).copied().unwrap_or(f64::NAN)
    }

    /// Streaming mean of an outcome across units for one world (no full retain required).
    #[must_use]
    pub fn streaming_outcome_mean(&self, world: usize, outcome: DenseNodeId) -> f64 {
        let mut sum = 0.0;
        let mut n = 0usize;
        for u in 0..self.n_units {
            let v = self.get(world, outcome, u);
            if v.is_finite() {
                sum += v;
                n += 1;
            }
        }
        sum / n.max(1) as f64
    }

    /// Borrowed outcome column for one world (`length = n_units`).
    ///
    /// # Errors
    ///
    /// World out of range.
    pub fn outcome_column(
        &self,
        world: usize,
        outcome: DenseNodeId,
    ) -> Result<&[f64], CounterfactualError> {
        if world >= self.n_worlds {
            return Err(CounterfactualError::model_msg("world index out of range"));
        }
        let start = world * self.n_nodes * self.n_units + outcome.as_usize() * self.n_units;
        Ok(&self.values[start..start + self.n_units])
    }
}

/// Equivalence: streaming mean matches mean of retained draws for the same result.
#[must_use]
pub fn streaming_matches_retained(
    result: &CounterfactualResult,
    world: usize,
    outcome: DenseNodeId,
) -> bool {
    let stream = result.streaming_outcome_mean(world, outcome);
    let mut sum = 0.0;
    let mut n = 0usize;
    for u in 0..result.n_units {
        let v = result.get(world, outcome, u);
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    let retained = sum / n.max(1) as f64;
    (stream - retained).abs() < 1e-12
}

/// Simultaneous / nested hard interventions under invertible additive-noise SCMs.
///
/// When `outer` and `inner` target **disjoint** variables, they are composed into one
/// counterfactual world (later hard sets override earlier ones). Overlapping targets
/// are rejected (fail-closed): true re-abduction under conflicting nested assignments
/// is not identifiable from invertible additive noise alone without additional structure.
///
/// # Errors
///
/// Engine failures, unknown outcome, or overlapping intervention targets.
pub fn simultaneous_hard_counterfactual(
    engine: &CounterfactualEngine,
    data: &TabularData,
    outer: &[Intervention],
    inner: &[Intervention],
    outcome: VariableId,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, CounterfactualError> {
    for o in outer {
        let Some(ov) = o.primary_variable() else {
            continue;
        };
        for i in inner {
            if i.primary_variable() == Some(ov) {
                return Err(CounterfactualError::model_msg(format!(
                    "overlapping hard intervention on {ov}: nested composition requires \
                     disjoint targets under invertible additive-noise assumptions"
                )));
            }
        }
    }
    let exo = engine.abduct(data, AbductionMissingPolicy::Error)?;
    let mut combined = outer.to_vec();
    combined.extend_from_slice(inner);
    let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from(combined) };
    let res = engine.predict(&exo, &[world], &[outcome], true, ws, ctx)?;
    let o = engine.model.dense_of(outcome).ok_or_else(|| {
        CounterfactualError::model_msg(format!("unknown outcome variable {outcome}"))
    })?;
    Ok(res.streaming_outcome_mean(0, o))
}

/// Nested counterfactual under invertible additive-noise assumptions.
///
/// Twin-network composition for forms like `Y_{x, M_{x'}}`:
/// 1. Abduct exogenous noise once from factual data.
/// 2. Evaluate the **outer** world (`outer` interventions, typically `do(X=x')`).
/// 3. Freeze every node that is **not** a primary target of `inner` at its outer
///    counterfactual value (this captures mediators `M_{x'}`).
/// 4. Evaluate the **inner** world with those freezes plus `inner` interventions
///    (typically `do(X=x)`), sharing the same abduced noise.
///
/// Overlapping primary targets are allowed: inner overrides outer on shared
/// targets (e.g. treatment), while non-targeted nodes stay at outer values.
/// Fail-closed when `outer`/`inner` contain non-hard interventions other than
/// `Set` / `Shift` (soft/stochastic nested forms need extra structure).
///
/// # Errors
///
/// Unsupported intervention kinds, abduction/predict failures.
pub fn nested_hard_counterfactual(
    engine: &CounterfactualEngine,
    data: &TabularData,
    outer: &[Intervention],
    inner: &[Intervention],
    outcome: VariableId,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, CounterfactualError> {
    for iv in outer.iter().chain(inner.iter()) {
        match iv {
            Intervention::Set { .. } | Intervention::Shift { .. } => {}
            Intervention::Sequence(_) => {
                return Err(CounterfactualError::model_msg(
                    "nested_hard_counterfactual does not accept Sequence inside outer/inner; \
                     pass flat Set/Shift slices",
                ));
            }
            _ => {
                return Err(CounterfactualError::model_msg(
                    "nested counterfactuals currently support Set and Shift only",
                ));
            }
        }
    }

    let exo = engine.abduct(data, AbductionMissingPolicy::Error)?;
    let outer_world =
        CounterfactualWorld { unit_rows: None, interventions: Arc::from(outer.to_vec()) };
    let outer_res = engine.predict(&exo, &[outer_world], &[], true, ws, ctx)?;

    let n_nodes = exo.n_nodes;
    let n_units = exo.n_units;
    let outcome_dense = engine.model.dense_of(outcome).ok_or_else(|| {
        CounterfactualError::model_msg(format!("unknown outcome variable {outcome}"))
    })?;

    // Primary variables targeted by inner interventions (not frozen).
    let mut inner_targets = vec![false; n_nodes];
    for iv in inner {
        if let Some(v) = iv.primary_variable() {
            if let Some(d) = engine.model.dense_of(v) {
                inner_targets[d.as_usize()] = true;
            }
        }
    }

    // Freeze non-inner-target nodes at outer counterfactual values.
    // Intervention::Set is scalar, so when unit values differ we evaluate unit-wise.
    let mut freeze_ivs: Vec<Intervention> = Vec::new();
    let mut need_unitwise = false;
    for node in 0..n_nodes {
        if inner_targets[node] || node == outcome_dense.as_usize() {
            continue;
        }
        let start = node * n_units;
        let col = &outer_res.values[start..start + n_units];
        let first = col.first().copied().unwrap_or(0.0);
        if col.iter().any(|&v| (v - first).abs() > 1e-12) {
            need_unitwise = true;
            break;
        }
        let var = engine.model.output_layout.variables[node];
        freeze_ivs.push(Intervention::set(var, causal_core::Value::f64(first)));
    }

    if need_unitwise {
        let mut sum = 0.0;
        let mut count = 0usize;
        for u in 0..n_units {
            let mut unit_freeze = Vec::new();
            for node in 0..n_nodes {
                if inner_targets[node] || node == outcome_dense.as_usize() {
                    continue;
                }
                let start = node * n_units;
                let v = outer_res.values[start + u];
                let var = engine.model.output_layout.variables[node];
                unit_freeze.push(Intervention::set(var, causal_core::Value::f64(v)));
            }
            let mut combined = unit_freeze;
            combined.extend_from_slice(inner);
            let world = CounterfactualWorld {
                unit_rows: Some(Arc::from([u])),
                interventions: Arc::from(combined),
            };
            let res = engine.predict(&exo, &[world], &[outcome], true, ws, ctx)?;
            let v = res.get(0, outcome_dense, u);
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        if count == 0 {
            return Err(CounterfactualError::model_msg("nested CF produced no finite outcomes"));
        }
        return Ok(sum / count as f64);
    }

    let mut combined = freeze_ivs;
    combined.extend_from_slice(inner);
    let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from(combined) };
    let res = engine.predict(&exo, &[world], &[outcome], true, ws, ctx)?;
    Ok(res.streaming_outcome_mean(0, outcome_dense))
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, Intervention, InterventionSequence, MeasurementSpec, RoleHint,
        SequencedIntervention, SmallRoleSet, TemporalPolicy, ToleranceClass, Value, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{
        CompiledMechanismStore, MechanismRegistry, MechanismSlot, SelectionPolicy,
    };

    fn unit_normal(rng: &mut causal_core::CausalRng) -> f64 {
        let u1 = rng.next_f64().clamp(1e-12, 1.0);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn toy() -> (CounterfactualEngine, TabularData) {
        let n = 20usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            t[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            y[i] = 2.0 * t[i] + 0.1 * (i as f64);
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        (CounterfactualEngine::new(compiled.with_mechanisms(store)), data)
    }

    #[test]
    fn ite_and_streaming_equivalence() {
        let (engine, data) = toy();
        let exo = engine.abduct(&data, AbductionMissingPolicy::Error).unwrap();
        assert_eq!(exo.kind, NoiseInferenceKind::Invertible);
        let mut ws = MechanismWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let ite = engine
            .individual_treatment_effect(
                &exo,
                y,
                Intervention::set(t, Value::f64(1.0)),
                Intervention::set(t, Value::f64(0.0)),
                &mut ws,
                &ctx,
            )
            .unwrap();
        let mean_ite = ite.iter().sum::<f64>() / ite.len() as f64;
        assert!((mean_ite - 2.0).abs() < 0.15, "mean_ite={mean_ite}");

        let worlds = [CounterfactualWorld {
            unit_rows: None,
            interventions: Arc::from([Intervention::set(t, Value::f64(1.0))]),
        }];
        let res = engine.predict(&exo, &worlds, &[y], false, &mut ws, &ctx).unwrap();
        assert!(streaming_matches_retained(&res, 0, DenseNodeId::from_raw(1)));
        assert!(res.notes.iter().any(|n| n.contains("noise_inference")));
    }

    #[test]
    fn abduction_predicts_factual_and_ite_variance_finite() {
        let (engine, data) = toy();
        let exo = engine.abduct(&data, AbductionMissingPolicy::Error).unwrap();
        assert_eq!(exo.kind, NoiseInferenceKind::Invertible);
        let mut ws = MechanismWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let t_vals = data.float64_values(t).unwrap();
        let y_vals = data.float64_values(y).unwrap();

        // Abduced noise + factual treatment level reproduces observed y per unit.
        for u in 0..exo.n_units {
            let world = CounterfactualWorld {
                unit_rows: Some(Arc::from([u])),
                interventions: Arc::from([Intervention::set(t, Value::f64(t_vals[u]))]),
            };
            let res = engine.predict(&exo, &[world], &[y], false, &mut ws, &ctx).unwrap();
            let pred = res.get(0, DenseNodeId::from_raw(1), u);
            assert!(
                (pred - y_vals[u]).abs() < 1e-8,
                "unit {u}: pred={pred} factual={}",
                y_vals[u]
            );
        }

        let ite = engine
            .individual_treatment_effect(
                &exo,
                y,
                Intervention::set(t, Value::f64(1.0)),
                Intervention::set(t, Value::f64(0.0)),
                &mut ws,
                &ctx,
            )
            .unwrap();
        let mean = ite.iter().sum::<f64>() / ite.len() as f64;
        assert!((mean - 2.0).abs() < 0.15, "mean_ite={mean}");
        let var = ite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ite.len() as f64;
        assert!(var.is_finite() && var > 0.0, "ite variance={var}");

        let do_world =
            CounterfactualWorld { unit_rows: None, interventions: Arc::from([Intervention::set(t, Value::f64(1.0))]) };
        let shift_world = CounterfactualWorld {
            unit_rows: None,
            interventions: Arc::from([Intervention::shift(t, Value::f64(0.5))]),
        };
        let res_do = engine.predict(&exo, &[do_world], &[y], false, &mut ws, &ctx).unwrap();
        let res_shift = engine.predict(&exo, &[shift_world], &[y], false, &mut ws, &ctx).unwrap();
        assert!(res_do.streaming_outcome_mean(0, DenseNodeId::from_raw(1)).is_finite());
        assert!(res_shift.streaming_outcome_mean(0, DenseNodeId::from_raw(1)).is_finite());
    }

    /// Multi-world predict: streaming means match retained materialization for each
    /// intervention level (do(1), do(0), shift), beyond the single-world check in
    /// `ite_and_streaming_equivalence`.
    #[test]
    fn multi_world_streaming_matches_retained() {
        let (engine, data) = toy();
        let exo = engine.abduct(&data, AbductionMissingPolicy::Error).unwrap();
        assert_eq!(exo.kind, NoiseInferenceKind::Invertible);
        let mut ws = MechanismWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let outcome = DenseNodeId::from_raw(1);

        let worlds = [
            CounterfactualWorld {
                unit_rows: None,
                interventions: Arc::from([Intervention::set(t, Value::f64(1.0))]),
            },
            CounterfactualWorld {
                unit_rows: None,
                interventions: Arc::from([Intervention::set(t, Value::f64(0.0))]),
            },
            CounterfactualWorld {
                unit_rows: None,
                interventions: Arc::from([Intervention::shift(t, Value::f64(0.5))]),
            },
        ];
        let res = engine.predict(&exo, &worlds, &[y], false, &mut ws, &ctx).unwrap();
        assert_eq!(res.n_worlds, 3);
        for wi in 0..res.n_worlds {
            assert!(
                streaming_matches_retained(&res, wi, outcome),
                "world {wi}: streaming mean diverged from retained draws"
            );
            assert!(res.streaming_outcome_mean(wi, outcome).is_finite());
        }
        // Distinct hard interventions should yield distinct retained means on this SCM.
        let m1 = res.streaming_outcome_mean(0, outcome);
        let m0 = res.streaming_outcome_mean(1, outcome);
        assert!((m1 - m0 - 2.0).abs() < 0.15, "E[Y|do(1)]-E[Y|do(0)]={}", m1 - m0);
    }

    /// Invertible linear SEM with pinned β: mean ITE must recover structural β
    /// (`ToleranceClass::StableFloat` under exact mechanisms; MonteCarlo floor if fit noise).
    #[test]
    fn random_linear_sem_mean_ite_matches_structural_beta() {
        let mut rng = ExecutionContext::for_tests(0xCF_B7).rng.stream(0x1E_E7);
        for trial in 0..12 {
            let beta = 0.5 + 2.5 * rng.next_f64(); // [0.5, 3.0]
            let n = 32 + (rng.next_u64() % 48) as usize;
            let sigma_y = 0.05 + 0.2 * rng.next_f64();

            let mut b = CausalSchemaBuilder::new();
            b.add_variable(
                "t",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            let schema = b.build().unwrap();
            let mut t = vec![0.0; n];
            let mut y = vec![0.0; n];
            for i in 0..n {
                t[i] = unit_normal(&mut rng);
                y[i] = beta * t[i] + sigma_y * unit_normal(&mut rng);
            }
            let validity = ValidityBitmap::all_valid(n);
            let cols = vec![
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                        .unwrap(),
                ),
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
                ),
            ];
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables(2);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            let compiled = CompiledCausalModel::compile(g).unwrap();
            let store = CompiledMechanismStore {
                slots: Arc::from([
                    MechanismSlot::LinearGaussian {
                        intercept: 0.0,
                        coeffs: Arc::from([]),
                        sigma: 1.0,
                    },
                    MechanismSlot::LinearGaussian {
                        intercept: 0.0,
                        coeffs: Arc::from([beta]),
                        sigma: sigma_y,
                    },
                ]),
            };
            let engine = CounterfactualEngine::new(compiled.with_mechanisms(store));
            let exo = engine.abduct(&data, AbductionMissingPolicy::Error).unwrap();
            assert_eq!(exo.kind, NoiseInferenceKind::Invertible);
            let mut ws = MechanismWorkspace::default();
            let ctx = ExecutionContext::for_tests(1);
            let ite = engine
                .individual_treatment_effect(
                    &exo,
                    VariableId::from_raw(1),
                    Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                    Intervention::set(VariableId::from_raw(0), Value::f64(0.0)),
                    &mut ws,
                    &ctx,
                )
                .unwrap();
            let mean_ite = ite.iter().sum::<f64>() / ite.len() as f64;
            assert!(
                ToleranceClass::StableFloat.close(mean_ite, beta)
                    || ToleranceClass::MonteCarlo.close(mean_ite, beta),
                "trial {trial}: mean_ite={mean_ite} beta={beta} n={n}"
            );
        }
    }

    #[test]
    fn nested_sequence_refused_when_not_allowed() {
        let (engine, data) = toy();
        let exo = engine.abduct(&data, AbductionMissingPolicy::Error).unwrap();
        let mut ws = MechanismWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let nested = Intervention::sequence(InterventionSequence::new(vec![SequencedIntervention::new(
            Intervention::set(t, Value::f64(1.0)),
            TemporalPolicy::pulse(0),
        )]));
        let world =
            CounterfactualWorld { unit_rows: None, interventions: Arc::from([nested]) };
        let err = engine.predict(&exo, &[world], &[y], false, &mut ws, &ctx).unwrap_err();
        assert_eq!(err, CounterfactualError::NestedNotAllowed);
    }

    #[test]
    fn overlapping_simultaneous_interventions_fail_closed() {
        let (engine, data) = toy();
        let mut ws = MechanismWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let err = simultaneous_hard_counterfactual(
            &engine,
            &data,
            &[Intervention::set(t, Value::f64(1.0))],
            &[Intervention::set(t, Value::f64(0.0))],
            y,
            &mut ws,
            &ctx,
        )
        .unwrap_err();
        assert!(
            matches!(err, CounterfactualError::Model(_)),
            "overlapping targets must fail closed, got {err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("overlapping"), "message={msg}");
    }

    #[test]
    fn nested_hard_allows_overlapping_treatment_with_mediator_freeze() {
        // T -> M -> Y, Y also <- T. Nested Y_{x=1, M_{x=0}} should freeze M at do(T=0).
        let n = 40usize;
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("m", RoleHint::Context),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut tv = vec![0.0; n];
        let mut mv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            tv[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            mv[i] = 0.7 * tv[i] + 0.05 * (i as f64);
            yv[i] = 0.5 * tv[i] + 1.5 * mv[i] + 0.01 * (i as f64);
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(tv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(mv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let compiled = causal_model::CompiledCausalModel::compile(g).unwrap();
        // Force invertible linear mechanisms (T is binary → auto-assign prefers Discrete).
        let store = CompiledMechanismStore {
            slots: Arc::from(vec![
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([]),
                    sigma: 0.05,
                },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([0.7]),
                    sigma: 0.05,
                },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([0.5, 1.5]),
                    sigma: 0.05,
                },
            ]),
        };
        let engine = CounterfactualEngine::new(compiled.with_mechanisms(store));
        let mut ws = MechanismWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(2);
        // Y_{1, M_0}: outer do(T=0), inner do(T=1) with M frozen.
        let nested = nested_hard_counterfactual(
            &engine,
            &data,
            &[Intervention::set(t, Value::f64(0.0))],
            &[Intervention::set(t, Value::f64(1.0))],
            y,
            &mut ws,
            &ctx,
        )
        .unwrap();
        assert!(nested.is_finite(), "nested={nested}");
        // Pure do(T=1) should differ when M responds to T.
        let exo = engine.abduct(&data, AbductionMissingPolicy::Error).unwrap();
        let world = CounterfactualWorld {
            unit_rows: None,
            interventions: Arc::from([Intervention::set(t, Value::f64(1.0))]),
        };
        let pure = engine
            .predict(&exo, &[world], &[y], true, &mut ws, &ctx)
            .unwrap()
            .streaming_outcome_mean(0, DenseNodeId::from_raw(2));
        assert!(
            (nested - pure).abs() > 1e-3,
            "nested={nested} should differ from pure do(T=1)={pure}"
        );
    }

    /// Streaming ≡ retained under random unit counts and multi-world intervention sets.
    #[test]
    fn random_multi_world_streaming_matches_retained() {
        let mut rng = ExecutionContext::for_tests(0xCF_57).rng.stream(0x57_EA);
        for trial in 0..8 {
            let n = 16 + (rng.next_u64() % 40) as usize;
            let n_worlds = 2 + (rng.next_u64() % 4) as usize; // 2..=5
            let mut b = CausalSchemaBuilder::new();
            b.add_variable(
                "t",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            let schema = b.build().unwrap();
            let mut t = vec![0.0; n];
            let mut y = vec![0.0; n];
            for i in 0..n {
                t[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
                y[i] = 2.0 * t[i] + 0.1 * (i as f64);
            }
            let validity = ValidityBitmap::all_valid(n);
            let cols = vec![
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone())
                        .unwrap(),
                ),
                OwnedColumn::Float64(
                    Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
                ),
            ];
            let data =
                TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
            let mut g = Dag::with_variables(2);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            let compiled = CompiledCausalModel::compile(g).unwrap();
            let (store, _) = MechanismRegistry::standard()
                .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
                .unwrap();
            let engine = CounterfactualEngine::new(compiled.with_mechanisms(store));
            let exo = engine.abduct(&data, AbductionMissingPolicy::Error).unwrap();
            let mut ws = MechanismWorkspace::default();
            let ctx = ExecutionContext::for_tests(1);
            let tid = VariableId::from_raw(0);
            let yid = VariableId::from_raw(1);
            let outcome = DenseNodeId::from_raw(1);
            let worlds: Vec<CounterfactualWorld> = (0..n_worlds)
                .map(|wi| {
                    let level = wi as f64 * 0.5;
                    CounterfactualWorld {
                        unit_rows: None,
                        interventions: Arc::from([Intervention::set(tid, Value::f64(level))]),
                    }
                })
                .collect();
            let res = engine.predict(&exo, &worlds, &[yid], false, &mut ws, &ctx).unwrap();
            assert_eq!(res.n_worlds, n_worlds);
            assert_eq!(res.n_units, n);
            for wi in 0..n_worlds {
                assert!(
                    streaming_matches_retained(&res, wi, outcome),
                    "trial {trial} world {wi}: streaming ≠ retained (n={n}, worlds={n_worlds})"
                );
            }
        }
    }
}
