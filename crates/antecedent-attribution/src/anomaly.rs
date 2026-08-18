//! Anomaly attribution via ancestor-noise Shapley (Budhathoki, Minorics, Bloebaum & Janzing 2022).
//!
//! # Scoring
//!
//! The anomaly score is the information-theoretic (IT) score of Budhathoki, Janzing,
//! Bloebaum & Ng (ICML 2022): `−log P(τ(Y) ≥ τ(y))`, the negative log of a **tail
//! probability** under the target's own marginal, with `τ` a two-sided outlierness
//! measure. See the internal `OutlierTail` accumulator.
//!
//! It is deliberately not `−log p(y | parents)`, the negative log **density**, which this
//! module previously returned. A density is not a probability: it is unbounded, it carries
//! the units of `1/y`, and it moves under any rescaling of the target. Two points equally
//! extreme in their own distributions score differently purely because of the fitted
//! `σ` — the normalizer contributes `−ln σ`, so the same standardized residual shifts by
//! `ln(100) ≈ 4.6` nats between `σ = 0.01` and `σ = 1`. A tail probability has none of
//! those defects: it is in `(0, 1]`, dimensionless, and invariant to any monotone
//! reparameterization of the target.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AnomalyAttributionQuery, ComponentId, ExecutionContext, ShapleyConfig, VariableId,
};
use antecedent_counterfactual::{AbductionMissingPolicy, CounterfactualEngine};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{BitSet, DenseNodeId, GraphWorkspace};
use antecedent_model::{
    CompiledCausalModel, MechanismWorkspace, NoiseBatchMut, ValueBatchMut, evaluate_batch_topo,
};

use crate::error::AttributionError;
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Empirical two-sided outlier tail for one target's marginal.
///
/// Scores a value by how far into the tail its outlierness falls:
/// `−log P(τ(Y) ≥ τ(y))` with `τ(y) = |y − median|`, estimated from the observed values.
///
/// Two-sided because "anomalous" here means unusually far from the bulk in either
/// direction; a one-sided tail would score a large negative excursion as perfectly
/// ordinary. The median is the centre rather than the mean so that the outliers being
/// scored do not drag the reference point toward themselves.
///
/// The reference location and scale are the **median** and `1.4826 · MAD`, both robust: the
/// very outliers being scored must not drag the reference toward themselves, which a mean and
/// a sample SD would let them do.
///
/// The tail is evaluated on a Gaussian reference rather than by empirical rank. A rank-based
/// tail `(1 + k) / (1 + n)` is bounded by `ln(1 + n)` and takes only `n + 1` distinct values,
/// which is fatal here for two reasons: the genuine outlier saturates against the merely
/// edge-of-range observations (in a 30-row linear ramp with one 100× excursion, the smallest
/// ordinary value scores 2.34 against the outlier's 2.74), and every coalition reconstruction
/// lands in the same rank bucket, so all Shapley differences collapse to exactly zero. A
/// continuous tail keeps the score monotone in outlierness and keeps the attribution alive.
struct OutlierTail {
    center: f64,
    scale: f64,
}

impl OutlierTail {
    fn from_reference(values: &[f64]) -> Self {
        let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return Self { center: 0.0, scale: 1.0 };
        }
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let center = median_of_sorted(&finite);
        let mut dev: Vec<f64> = finite.iter().map(|v| (v - center).abs()).collect();
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // 1.4826 · MAD is the consistent estimator of σ for a Gaussian.
        let mad = median_of_sorted(&dev);
        let scale = if mad > 0.0 {
            1.482_602_218_505_602 * mad
        } else {
            // Degenerate bulk (≥ half the values identical). Fall back to the mean absolute
            // deviation so a genuine excursion is still scored rather than divided by zero.
            let mean_abs = dev.iter().sum::<f64>() / dev.len() as f64;
            if mean_abs > 0.0 { mean_abs } else { 1.0 }
        };
        Self { center, scale }
    }

    /// `−log P(|Z| ≥ |z|)` for the standardized deviation `z = (y − center) / scale`.
    ///
    /// Zero for a value sitting at the centre, growing without bound into the tail — and
    /// dimensionless, so rescaling the target leaves it unchanged.
    fn score(&self, y: f64) -> f64 {
        if !y.is_finite() {
            return 0.0;
        }
        let z = ((y - self.center) / self.scale).abs();
        // Two-sided Gaussian tail. `norm_sf` underflows to 0 past z ≈ 38, so switch to the
        // log-space asymptotic `2Φ(−z) ≈ 2φ(z)/z` there instead of returning `+∞`.
        let two_sided = 2.0 * antecedent_kernels::norm_sf(z);
        if two_sided > f64::MIN_POSITIVE {
            -two_sided.ln()
        } else {
            0.5 * z * z + z.ln() + 0.5 * std::f64::consts::TAU.ln() - std::f64::consts::LN_2
        }
    }
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
}

/// Per-unit anomaly score for a target variable, with noise-term attribution.
#[derive(Clone, Debug)]
pub struct AnomalyScores {
    /// Target variable.
    pub target: VariableId,
    /// Row indices scored.
    pub rows: Arc<[usize]>,
    /// IT scores: `−log P(|Z| ≥ |z|)` for the standardized deviation of the value from the
    /// target's robust marginal centre. Higher = more anomalous; `0` at the centre. A tail
    /// probability, not a density, so the scale of the target does not affect it.
    pub scores: Arc<[f64]>,
    /// Sum of absolute Shapley IT-score attributions per row.
    pub residual_abs: Arc<[f64]>,
    /// Ancestor (incl. target) components used as Shapley players.
    pub noise_components: Arc<[ComponentId]>,
    /// Row-major Shapley attributions: `rows.len() * noise_components.len()`.
    pub noise_contributions: Arc<[f64]>,
}

/// Score anomalies and attribute them to ancestor noise terms via Shapley
/// (Budhathoki, Minorics, Bloebaum & Janzing 2022): replace noise coordinates outside the coalition with
/// reference draws (0 for additive noise) and redistribute the target's IT score
/// (see the module docs for why that is a tail probability and not a density).
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

        // Reference distribution for the IT score: the target's own observed marginal.
        let tail = OutlierTail::from_reference(&y_all);

        for (ui, &row) in rows.iter().enumerate() {
            let mut payoff = NoiseShapleyPayoff {
                model,
                target: dense,
                players: &players_dense,
                exo_noise: &exo.noise,
                n_units: exo.n_units,
                row,
                tail: &tail,
                noise_buf: vec![0.0; model.n_nodes()],
                value_buf: vec![0.0; model.n_nodes()],
                ws: MechanismWorkspace::default(),
            };
            // Factual IT score: how far into the target's marginal tail this value falls.
            scores.push(tail.score(y_all[row]));
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
    tail: &'a OutlierTail,
    noise_buf: Vec<f64>,
    value_buf: Vec<f64>,
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
        // Payoff = the IT score of the *reconstructed* target value, scored against the same
        // marginal tail as the factual value. Shapley then redistributes the anomaly score
        // itself rather than the reconstructed Y level, and every coalition value is
        // commensurable because they all use one fixed reference distribution.
        Ok(self.tail.score(self.value_buf[self.target.as_usize()]))
    }
}

/// Arrow strength for a linear-family edge: the variance the edge contributes,
/// `β² · Var(parent)` (Janzing et al. 2013, *Quantifying causal influences*, Section 6,
/// "Causal strength for linear structural equations").
///
/// This is the first-order (small-`β²·Var(parent)/Var(child)`) approximation of that
/// section's exact causal-strength formula
/// `CS = −½·log(1 − β²·Var(parent)/Var(child))`, not the log/KL expression itself — the
/// log form is recovered from this variance term only in the limit where the edge's
/// contributed variance is small relative to the child's total variance.
///
/// Not `|β|`, which this returned previously. A bare coefficient is not a measure of
/// influence, because it says nothing about how much the parent actually varies: a parent
/// with `β = 2.5` and `Var = 0.001` moves its child by almost nothing, while `β = 0.75`
/// with `Var = 100` dominates it — yet `|β|` ranks them the other way round, off by a
/// factor of ~9000 in that example. `β²·Var(parent)` is the variance of the child that
/// flows through the edge, which is what "strength" is asking for.
///
/// `Var(parent)` is implied by the model, not measured from data: variances propagate in
/// topological order from the roots' own noise. This is exact when the child's parents are
/// mutually uncorrelated; with correlated parents it remains the standard linear-Gaussian
/// arrow strength, reporting each edge's own contribution and not the cross terms.
///
/// Non-linear mechanisms error — use [`population_do_contrast`] for interventional
/// influence.
#[derive(Clone, Debug)]
pub struct ArrowStrength {
    /// Parent variable.
    pub parent: VariableId,
    /// Child variable.
    pub child: VariableId,
    /// `β² · Var(parent)`.
    pub strength: f64,
    /// The edge coefficient itself, retained because it carries the *sign* and direction
    /// of the effect that the (non-negative) strength deliberately discards.
    pub coefficient: f64,
}

/// Model-implied variance of every node, indexed by dense id.
///
/// Linear-family mechanisms compose as `Var(j) = Σ_i Σ_l β_i β_l Cov(pa_i, pa_l) + σ_j²`, so
/// the full covariance has to be carried along in topological order — the diagonal alone is
/// not enough once a node has two parents that share an ancestor.
fn model_implied_variances(model: &CompiledCausalModel) -> Result<Vec<f64>, AttributionError> {
    use antecedent_model::MechanismSlot;

    let n = model.n_nodes();
    let mut cov = vec![0.0; n * n];
    let mut settled: Vec<DenseNodeId> = Vec::with_capacity(n);

    for &node in model.node_order.iter() {
        let j = node.as_usize();
        let gather =
            model.gather_for(node).ok_or(AttributionError::MissingArtifact("missing gather"))?;

        // (parent dense index, coefficient) pairs plus this node's own noise variance.
        let (betas, own_var): (Vec<(usize, f64)>, f64) = match model.mechanisms.get(node) {
            MechanismSlot::LinearGaussian { coeffs, sigma, .. }
            | MechanismSlot::HierarchicalLinear { coeffs, sigma, .. }
            | MechanismSlot::Bvar { coeffs, sigma, .. } => (
                gather
                    .parents
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (p.as_usize(), coeffs.get(i).copied().unwrap_or(0.0)))
                    .collect(),
                sigma * sigma,
            ),
            // Deterministic: contributes no variance and depends on nothing.
            MechanismSlot::Constant { .. } => (Vec::new(), 0.0),
            // An unconditional categorical ignores its parents entirely, so it behaves as an
            // independent draw with the variance of its own support.
            MechanismSlot::Discrete { support, probs, logit_coeffs: None } => {
                let total: f64 = probs.iter().sum();
                if total <= 0.0 || !total.is_finite() {
                    return Err(AttributionError::NonLinearGaussianMechanism);
                }
                let mean: f64 = support.iter().zip(probs.iter()).map(|(s, p)| s * p / total).sum();
                let var: f64 = support
                    .iter()
                    .zip(probs.iter())
                    .map(|(s, p)| (s - mean) * (s - mean) * p / total)
                    .sum();
                (Vec::new(), var)
            }
            // Parent-conditional discrete, GP, LGSSM, unfitted: no closed-form variance.
            _ => return Err(AttributionError::NonLinearGaussianMechanism),
        };

        // Cross-covariances with everything already settled (all parents are among them,
        // since `node_order` is topological).
        for &prev in &settled {
            let k = prev.as_usize();
            let c: f64 = betas.iter().map(|&(p, b)| b * cov[p * n + k]).sum();
            cov[j * n + k] = c;
            cov[k * n + j] = c;
        }

        let mut var = own_var;
        for &(p, bp) in &betas {
            for &(q, bq) in &betas {
                var += bp * bq * cov[p * n + q];
            }
        }
        cov[j * n + j] = var;
        settled.push(node);
    }

    Ok((0..n).map(|i| cov[i * n + i]).collect())
}

/// Compute arrow strengths for all edges in the compiled model.
///
/// # Errors
///
/// [`AttributionError::NonLinearGaussianMechanism`] when a child with parents is not a
/// linear-family mechanism, or when any node's variance is not available in closed form
/// (the strength of an edge depends on its parent's variance, so an unusable mechanism
/// anywhere upstream makes the answer unavailable rather than approximate).
pub fn arrow_strengths(
    model: &CompiledCausalModel,
) -> Result<Vec<ArrowStrength>, AttributionError> {
    let variances = model_implied_variances(model)?;
    let mut out = Vec::new();
    for gather in model.parent_gathers.iter() {
        let child_var = model.output_layout.variables[gather.child.as_usize()];
        if gather.parents.is_empty() {
            continue;
        }
        let (antecedent_model::MechanismSlot::LinearGaussian { coeffs, .. }
        | antecedent_model::MechanismSlot::HierarchicalLinear { coeffs, .. }
        | antecedent_model::MechanismSlot::Bvar { coeffs, .. }) =
            model.mechanisms.get(gather.child)
        else {
            return Err(AttributionError::NonLinearGaussianMechanism);
        };
        for (i, &p) in gather.parents.iter().enumerate() {
            let parent = model.output_layout.variables[p.as_usize()];
            let beta = coeffs.get(i).copied().unwrap_or(0.0);
            let strength = beta * beta * variances[p.as_usize()];
            out.push(ArrowStrength { parent, child: child_var, strength, coefficient: beta });
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
    use antecedent_core::{Intervention, Value};
    use antecedent_model::sample_interventional;

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
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_model::{
        CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
        SelectionPolicy,
    };
    use serde::Deserialize;

    /// Build the anomaly fixture with the outcome scaled by `y_scale`.
    fn scaled_anomaly_fixture(n: usize, y_scale: f64) -> (CompiledCausalModel, TabularData) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let mut yv: Vec<f64> = xv.iter().map(|x| (1.0 + 2.0 * x) * y_scale).collect();
        yv[n - 1] = 100.0 * y_scale; // same anomaly, same standardized deviation
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
        (compiled.with_mechanisms(store), data)
    }

    /// The IT score must be invariant to the target's scale.
    ///
    /// This is the property that separates a tail probability from a density and the reason
    /// the finding was raised. `−log p(y | parents)` includes the normalizer `−ln σ`, so
    /// multiplying the outcome by 100 shifts every score by `ln(100) ≈ 4.605` even though
    /// nothing about the data's shape — or which unit is anomalous — has changed. A tail
    /// probability is dimensionless and cannot move.
    #[test]
    fn it_score_is_invariant_to_target_scale() {
        let n = 30usize;
        let q = AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100);

        let (m1, d1) = scaled_anomaly_fixture(n, 1.0);
        let (m2, d2) = scaled_anomaly_fixture(n, 100.0);
        let s1 = score_anomalies(&m1, &d1, &q).unwrap();
        let s2 = score_anomalies(&m2, &d2, &q).unwrap();

        for row in 0..n {
            let a = s1[0].scores[row];
            let b = s2[0].scores[row];
            assert!(
                (a - b).abs() < 1e-6 * a.abs().max(1.0),
                "row {row}: score moved under a pure rescaling of Y ({a} vs {b}); a density \
                 would shift by ln(100) = {}",
                100.0_f64.ln()
            );
        }

        // And the score must still separate the anomaly from the bulk, not merely be stable.
        assert!(
            s1[0].scores[n - 1] > 10.0 * s1[0].scores[0],
            "anomaly {} should dwarf an ordinary row {}",
            s1[0].scores[n - 1],
            s1[0].scores[0]
        );
    }

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

    #[derive(Deserialize)]
    struct ArrowFixture {
        parents: Vec<FixtureParent>,
        child: FixtureChild,
        edges: Vec<ExpectedEdge>,
        tolerance: FixtureTolerance,
    }

    #[derive(Deserialize)]
    struct FixtureParent {
        raw: u32,
        sigma: f64,
        variance: f64,
    }

    #[derive(Deserialize)]
    struct FixtureChild {
        sigma: f64,
    }

    #[derive(Deserialize)]
    struct FixtureTolerance {
        absolute: f64,
    }

    #[derive(Deserialize)]
    struct ExpectedEdge {
        parent_raw: u32,
        child_raw: u32,
        coefficient: f64,
        strength: f64,
    }

    /// Arrow strength is `β²·Var(parent)`, and the fixture is built so `|β|` ranks the two
    /// edges the wrong way round.
    ///
    /// Parent 0 has the smaller coefficient (0.75 against 2.5) but a variance four orders of
    /// magnitude larger (100 against 0.001), so it genuinely dominates the child by ~9000×.
    /// Ranking by `|β|` puts parent 1 first, which is exactly backwards. The fixture's
    /// previous revision could not catch this: it built both parents as `Constant{0.0}`,
    /// which has no variance at all, so every edge's true strength was 0 and only the
    /// coefficient was observable.
    #[test]
    fn arrow_strength_is_variance_weighted_not_bare_coefficient() {
        let fixture: ArrowFixture = serde_json::from_str(include_str!(
            "../../../conformance/attribution/arrow_strength/expected.json"
        ))
        .unwrap();
        let mut graph = Dag::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let coefficients: Vec<f64> = fixture.edges.iter().map(|edge| edge.coefficient).collect();
        // Roots are linear-Gaussian with no parents, so each carries variance sigma^2.
        let root = |sigma: f64| MechanismSlot::LinearGaussian {
            intercept: 0.0,
            coeffs: Arc::from([]),
            sigma,
        };
        let model = compiled.with_mechanisms(CompiledMechanismStore {
            slots: Arc::from([
                root(fixture.parents[0].sigma),
                root(fixture.parents[1].sigma),
                MechanismSlot::LinearGaussian {
                    intercept: 1.0,
                    coeffs: Arc::from(coefficients),
                    sigma: fixture.child.sigma,
                },
            ]),
        });

        // The propagated variances must match the pinned closed form first; the strengths
        // depend on them.
        let variances = model_implied_variances(&model).unwrap();
        for p in &fixture.parents {
            let got = variances[p.raw as usize];
            assert!(
                (got - p.variance).abs() < fixture.tolerance.absolute,
                "Var(node {}) = {got}, expected {}",
                p.raw,
                p.variance
            );
        }

        let actual = arrow_strengths(&model).unwrap();
        assert_eq!(actual.len(), fixture.edges.len());
        for expected in &fixture.edges {
            let got = actual
                .iter()
                .find(|edge| {
                    edge.parent == VariableId::from_raw(expected.parent_raw)
                        && edge.child == VariableId::from_raw(expected.child_raw)
                })
                .unwrap();
            assert!(
                (got.strength - expected.strength).abs() < fixture.tolerance.absolute,
                "edge {}→{} strength {} != expected {}",
                expected.parent_raw,
                expected.child_raw,
                got.strength,
                expected.strength
            );
            assert!((got.coefficient - expected.coefficient).abs() < fixture.tolerance.absolute);
        }

        // The ranking must follow influence, not coefficient magnitude.
        let by_parent = |raw: u32| {
            actual.iter().find(|e| e.parent == VariableId::from_raw(raw)).expect("edge present")
        };
        let (e0, e1) = (by_parent(0), by_parent(1));
        assert!(
            e0.strength > e1.strength,
            "parent 0 (beta={}, var=100) must outrank parent 1 (beta={}, var=0.001); got {} vs {}",
            e0.coefficient,
            e1.coefficient,
            e0.strength,
            e1.strength
        );
        assert!(
            e0.coefficient.abs() < e1.coefficient.abs(),
            "fixture must keep |beta| pointing the other way, or it proves nothing"
        );
    }

    /// Variance must compose through a chain, not just come off the roots.
    ///
    /// `x → m → y` with `Var(x) = 4`, `m = 3x + N(0, 1)` gives `Var(m) = 9·4 + 1 = 37`, so the
    /// `m → y` edge with `β = 2` has strength `4·37 = 148`. A diagonal-only propagation that
    /// forgot to carry `m`'s inherited variance would report `4·1 = 4`.
    #[test]
    fn variance_propagates_through_a_chain() {
        let mut graph = Dag::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let model = compiled.with_mechanisms(CompiledMechanismStore {
            slots: Arc::from([
                MechanismSlot::LinearGaussian { intercept: 0.0, coeffs: Arc::from([]), sigma: 2.0 },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([3.0]),
                    sigma: 1.0,
                },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([2.0]),
                    sigma: 1.0,
                },
            ]),
        });
        let variances = model_implied_variances(&model).unwrap();
        assert!((variances[0] - 4.0).abs() < 1e-12, "Var(x)={}", variances[0]);
        assert!((variances[1] - 37.0).abs() < 1e-12, "Var(m)={}", variances[1]);
        assert!((variances[2] - 149.0).abs() < 1e-12, "Var(y)={}", variances[2]);

        let arrows = arrow_strengths(&model).unwrap();
        let m_to_y = arrows
            .iter()
            .find(|e| e.parent == VariableId::from_raw(1) && e.child == VariableId::from_raw(2))
            .unwrap();
        assert!((m_to_y.strength - 148.0).abs() < 1e-12, "m→y strength={}", m_to_y.strength);
    }
}
