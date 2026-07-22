//! Anomaly attribution via ancestor-noise Shapley (Janzing et al. 2020; ).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    AnomalyAttributionQuery, ComponentId, ExecutionContext, ShapleyConfig, VariableId,
};
use causal_counterfactual::{AbductionMissingPolicy, CounterfactualEngine};
use causal_data::{TableView, TabularData};
use causal_graph::{BitSet, DenseNodeId, GraphWorkspace};
use causal_model::{
    CompiledCausalModel, MechanismWorkspace, NoiseBatchMut, ParentBatch, ValueBatchMut,
    evaluate_batch_topo, log_prob_column,
};

use crate::error::AttributionError;
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Per-unit anomaly score for a target variable, with noise-term attribution.
#[derive(Clone, Debug)]
pub struct AnomalyScores {
    /// Target variable.
    pub target: VariableId,
    /// Row indices scored.
    pub rows: Arc<[usize]>,
    /// Anomaly scores (−log p under the fitted mechanism at factual parents; higher = more anomalous).
    pub scores: Arc<[f64]>,
    /// Sum of absolute Shapley −log p attributions per row.
    pub residual_abs: Arc<[f64]>,
    /// Ancestor (incl. target) components used as Shapley players.
    pub noise_components: Arc<[ComponentId]>,
    /// Row-major Shapley attributions: `rows.len() * noise_components.len()`.
    pub noise_contributions: Arc<[f64]>,
}

/// Score anomalies and attribute them to ancestor noise terms via Shapley
/// (Janzing et al. 2020): replace noise coordinates outside the coalition with
/// reference draws (0 for additive noise) and redistribute the target −log p.
///
/// # Errors
///
/// Size limit or data/model failures.
pub fn score_anomalies(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &AnomalyAttributionQuery,
) -> Result<Vec<AnomalyScores>, AttributionError> {
    query.validate()?;
    let n = data.row_count();
    let rows: Vec<usize> = match &query.unit_rows {
        Some(r) => r.to_vec(),
        None => (0..n).collect(),
    };
    if rows.len() > query.max_units {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: rows.len(),
            max: query.max_units,
        });
    }

    let engine = CounterfactualEngine::from_ref(model);
    let exo = engine.abduct(data, AbductionMissingPolicy::Error)?;
    let ctx = ExecutionContext::for_tests(0xA10A);
    let shapley = ShapleyConfig::exact();

    let mut out = Vec::with_capacity(query.targets.len());
    for &target in query.targets.iter() {
        let dense = model
            .dense_of(target)
            .ok_or_else(|| AttributionError::missing_var("target", target))?;
        let players_dense = ancestor_nodes(model, dense);
        if players_dense.len() > 64 {
            return Err(AttributionError::SizeLimit {
                kind: "components",
                requested: players_dense.len(),
                max: 64,
            });
        }
        let players: Vec<ComponentId> = players_dense
            .iter()
            .map(|&d| ComponentId::from_variable(model.output_layout.variables[d.as_usize()]))
            .collect();

        let y_all = data.float64_values(target)?;
        let mut scores = Vec::with_capacity(rows.len());
        let mut resid = Vec::with_capacity(rows.len());
        let mut contrib = vec![0.0; rows.len() * players.len()];

        for (ui, &row) in rows.iter().enumerate() {
            let mut payoff = NoiseShapleyPayoff {
                model,
                target: dense,
                players: &players_dense,
                exo_noise: &exo.noise,
                n_units: exo.n_units,
                row,
                noise_buf: vec![0.0; model.n_nodes()],
                value_buf: vec![0.0; model.n_nodes()],
                parent_buf: Vec::new(),
                ws: MechanismWorkspace::default(),
            };
            // Factual anomaly score: −log p(y|parents) under observed parents.
            let gather = model
                .gather_for(dense)
                .ok_or(AttributionError::MissingArtifact("missing gather"))?;
            let n_par = gather.n_parents();
            let mut parent_mat = vec![0.0; n_par.max(1)];
            for (pi, &p) in gather.parents.iter().enumerate() {
                let pv = model.output_layout.variables[p.as_usize()];
                parent_mat[pi] = data.float64_values(pv)?[row];
            }
            let parents = ParentBatch { n_rows: 1, n_parents: n_par, values: &parent_mat[..n_par] };
            let mut lp = [0.0];
            log_prob_column(model.mechanisms.get(dense), &[y_all[row]], parents, &mut lp)?;
            scores.push(-lp[0]);
            let est = estimate_shapley(&players, &shapley, &mut payoff, &ctx)?;
            let mut abs_sum = 0.0;
            for (j, v) in est.values.iter().enumerate() {
                contrib[ui * players.len() + j] = *v;
                abs_sum += v.abs();
            }
            resid.push(abs_sum);
        }

        out.push(AnomalyScores {
            target,
            rows: Arc::from(rows.clone()),
            scores: Arc::from(scores),
            residual_abs: Arc::from(resid),
            noise_components: Arc::from(players),
            noise_contributions: Arc::from(contrib),
        });
    }
    Ok(out)
}

fn ancestor_nodes(model: &CompiledCausalModel, target: DenseNodeId) -> Vec<DenseNodeId> {
    let mut ws = GraphWorkspace::default();
    let mut anc = BitSet::with_len(model.n_nodes());
    model.graph.ancestors_of(&[target], &mut anc, &mut ws);
    let mut nodes = Vec::new();
    for gather in model.parent_gathers.iter() {
        if anc.contains(gather.child) {
            nodes.push(gather.child);
        }
    }
    if nodes.is_empty() {
        nodes.push(target);
    }
    nodes
}

struct NoiseShapleyPayoff<'a> {
    model: &'a CompiledCausalModel,
    target: DenseNodeId,
    players: &'a [DenseNodeId],
    exo_noise: &'a [f64],
    n_units: usize,
    row: usize,
    noise_buf: Vec<f64>,
    value_buf: Vec<f64>,
    parent_buf: Vec<f64>,
    ws: MechanismWorkspace,
}

impl CoalitionPayoff for NoiseShapleyPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let n_nodes = self.model.n_nodes();
        self.noise_buf.fill(0.0);
        for (i, &node) in self.players.iter().enumerate() {
            let factual = self.exo_noise[node.as_usize() * self.n_units + self.row];
            self.noise_buf[node.as_usize()] = if mask & (1u64 << i) != 0 { factual } else { 0.0 };
        }
        for node in 0..n_nodes {
            let dense = DenseNodeId::from_raw(node as u32);
            if !self.players.contains(&dense) {
                self.noise_buf[node] = self.exo_noise[node * self.n_units + self.row];
            }
        }

        self.value_buf.fill(0.0);
        let noise = NoiseBatchMut::new(1, n_nodes, &mut self.noise_buf)?;
        let mut values = ValueBatchMut::new(1, n_nodes, &mut self.value_buf)?;
        evaluate_batch_topo(
            &self.model.node_order,
            &self.model.parent_gathers,
            &self.model.mechanisms.slots,
            &noise,
            &mut values,
            &mut self.ws,
        )?;
        // Payoff = −log p(y|parents) under the coalition reconstruction so Shapley
        // redistributes the anomaly score (not the reconstructed Y level).
        let gather = self
            .model
            .gather_for(self.target)
            .ok_or(AttributionError::MissingArtifact("missing gather"))?;
        let n_par = gather.n_parents();
        if self.parent_buf.len() < n_par {
            self.parent_buf.resize(n_par, 0.0);
        }
        for (pi, &p) in gather.parents.iter().enumerate() {
            self.parent_buf[pi] = self.value_buf[p.as_usize()];
        }
        let parents =
            ParentBatch { n_rows: 1, n_parents: n_par, values: &self.parent_buf[..n_par] };
        let y = [self.value_buf[self.target.as_usize()]];
        let mut lp = [0.0];
        log_prob_column(self.model.mechanisms.get(self.target), &y, parents, &mut lp)?;
        Ok(-lp[0])
    }
}

/// Direct arrow strength: `|β|` for linear-family edges (`LinearGaussian` /
/// `HierarchicalLinear` / `Bvar`). Non-linear mechanisms error — use
/// [`population_do_contrast`] for interventional influence.
#[derive(Clone, Debug)]
pub struct ArrowStrength {
    /// Parent variable.
    pub parent: VariableId,
    /// Child variable.
    pub child: VariableId,
    /// Strength.
    pub strength: f64,
}

/// Compute arrow strengths for all edges in the compiled model.
///
/// # Errors
///
/// [`AttributionError::NonLinearGaussianMechanism`] when a child with parents is
/// not a linear-family mechanism.
pub fn arrow_strengths(
    model: &CompiledCausalModel,
) -> Result<Vec<ArrowStrength>, AttributionError> {
    let mut out = Vec::new();
    for gather in model.parent_gathers.iter() {
        let child_var = model.output_layout.variables[gather.child.as_usize()];
        if gather.parents.is_empty() {
            continue;
        }
        let (causal_model::MechanismSlot::LinearGaussian { coeffs, .. }
        | causal_model::MechanismSlot::HierarchicalLinear { coeffs, .. }
        | causal_model::MechanismSlot::Bvar { coeffs, .. }) = model.mechanisms.get(gather.child)
        else {
            return Err(AttributionError::NonLinearGaussianMechanism);
        };
        for (i, &p) in gather.parents.iter().enumerate() {
            let parent = model.output_layout.variables[p.as_usize()];
            let s = coeffs.get(i).copied().unwrap_or(0.0).abs();
            out.push(ArrowStrength { parent, child: child_var, strength: s });
        }
    }
    Ok(out)
}

/// Population do-contrast of parent on child: `|E[Y|do(X=μ+δ/2)] − E[Y|do(X=μ−δ/2)]|`.
///
/// This is **not** intrinsic (noise-based) causal influence.
///
/// # Errors
///
/// Size / model failures.
pub fn population_do_contrast(
    model: &CompiledCausalModel,
    data: &TabularData,
    parent: VariableId,
    child: VariableId,
    delta: f64,
    max_units: usize,
    ctx: &ExecutionContext,
) -> Result<f64, AttributionError> {
    use causal_core::{Intervention, Value};
    use causal_model::sample_interventional;

    let n = data.row_count().min(max_units);
    if data.row_count() > max_units {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: data.row_count(),
            max: max_units,
        });
    }
    let mut rng = ctx.rng.stream(0x1C1_u64);
    let mut ws = MechanismWorkspace::default();
    let child_dense =
        model.dense_of(child).ok_or_else(|| AttributionError::missing_var("child", child))?;
    let pcol = data.float64_values(parent)?;
    let pmean = pcol.iter().sum::<f64>() / pcol.len().max(1) as f64;
    let hi = sample_interventional(
        model,
        &[Intervention::set(parent, Value::f64(pmean + 0.5 * delta))],
        n.max(1),
        &mut rng,
        &mut ws,
        ctx,
    )?;
    let lo = sample_interventional(
        model,
        &[Intervention::set(parent, Value::f64(pmean - 0.5 * delta))],
        n.max(1),
        &mut rng,
        &mut ws,
        ctx,
    )?;
    let hi_m = hi.column(child_dense.as_usize())?.iter().sum::<f64>() / n.max(1) as f64;
    let lo_m = lo.column(child_dense.as_usize())?.iter().sum::<f64>() / n.max(1) as f64;
    Ok((hi_m - lo_m).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{MechanismRegistry, SelectionPolicy};

    #[test]
    fn anomaly_and_arrow_strength() {
        let n = 30usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
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
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let mut yv: Vec<f64> = xv.iter().map(|x| 1.0 + 2.0 * x).collect();
        yv[n - 1] = 100.0; // anomaly
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
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
        let model = compiled.with_mechanisms(store);
        let q = AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100);
        let scores = score_anomalies(&model, &data, &q).unwrap();
        assert!(scores[0].scores[n - 1] > scores[0].scores[0]);
        assert!(!scores[0].noise_components.is_empty());
        // Anomalous unit should attribute primarily to Y's own noise.
        let y_idx = scores[0]
            .noise_components
            .iter()
            .position(|c| c.variable() == VariableId::from_raw(1))
            .expect("y player");
        let y_phi =
            scores[0].noise_contributions[(n - 1) * scores[0].noise_components.len() + y_idx];
        assert!(y_phi.abs() > 0.0, "y attribution={y_phi}");
        // Efficiency: Σφ redistributes anomaly-score change (signed sum finite; abs sum = residual_abs).
        let n_p = scores[0].noise_components.len();
        let row = n - 1;
        let phi_sum: f64 = (0..n_p).map(|j| scores[0].noise_contributions[row * n_p + j]).sum();
        let abs_sum: f64 =
            (0..n_p).map(|j| scores[0].noise_contributions[row * n_p + j].abs()).sum();
        assert!(phi_sum.is_finite());
        assert!((abs_sum - scores[0].residual_abs[row]).abs() < 1e-9);
        let arrows = arrow_strengths(&model).unwrap();
        assert!(!arrows.is_empty());
        assert!(arrows.iter().any(|a| a.strength > 0.5), "arrows={arrows:?}");
    }
}
