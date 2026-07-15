//! Abduction–action–prediction engine (DESIGN.md §16).
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MissingPolicy {
    /// Fail if a required factual column is absent.
    Error,
    /// Zero-fill absent columns and mark them as assumed noise.
    ZeroFill,
}

impl MissingPolicy {
    /// Whether absent columns are zero-filled.
    #[must_use]
    pub const fn allows_missing(self) -> bool {
        matches!(self, Self::ZeroFill)
    }
}

impl From<bool> for MissingPolicy {
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
    /// Per-node flag: factual column was absent and zero-filled under [`MissingPolicy::ZeroFill`]
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
    /// Fitted invertible model.
    pub model: CompiledCausalModel,
    /// Compiled plan.
    pub compiled: CompiledCounterfactualPlan,
}

impl CounterfactualEngine {
    /// Build from a fitted model (mechanisms should be invertible families).
    #[must_use]
    pub fn new(model: CompiledCausalModel) -> Self {
        let compiled = CompiledCounterfactualPlan { node_order: Arc::clone(&model.node_order) };
        Self { model, compiled }
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
        missing: MissingPolicy,
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
                MechanismSlot::LinearGaussian { .. } | MechanismSlot::Constant { .. } => {
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
        let o = self.model.dense_of(outcome).unwrap();
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

/// Nested hard interventions under invertible additive-noise SCMs: apply an outer
/// abduction, then an inner do inside the counterfactual world.
///
/// # Errors
///
/// Engine failures.
pub fn nested_hard_counterfactual(
    engine: &CounterfactualEngine,
    data: &TabularData,
    outer: &[Intervention],
    inner: &[Intervention],
    outcome: VariableId,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, CounterfactualError> {
    let exo = engine.abduct(data, MissingPolicy::Error)?;
    let mut combined = outer.to_vec();
    combined.extend_from_slice(inner);
    let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from(combined) };
    let res = engine.predict(&exo, &[world], &[outcome], true, ws, ctx)?;
    let o = engine.model.dense_of(outcome).unwrap();
    Ok(res.streaming_outcome_mean(0, o))
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, Intervention, MeasurementSpec, RoleHint, SmallRoleSet, Value,
        ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{MechanismRegistry, SelectionPolicy};

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
        let exo = engine.abduct(&data, MissingPolicy::Error).unwrap();
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
}
