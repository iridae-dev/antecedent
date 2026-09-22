//! `DirectLiNGAM` → static [`Dag`] (Shimizu et al. 2011).
//!
//! Causal-order search by residual–predictor independence, then pruning of the OLS
//! coefficients on the *standardised* scale (so the cut-off does not depend on units).
//! Required edges constrain the order search (a required parent is placed before its
//! child). Does not use ICA or the Meek/PC orientation stack.
//!
//! Deviation from the paper: independence is scored with distance correlation
//! (Székely, Rizzo & Bakirov 2007) rather than the paper's kernel-based mutual-
//! information statistic `T_kernel` (Section 3.2, Eqs. 9-14). The causal-order
//! search procedure itself follows the paper; only the independence measure differs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::manual_let_else, clippy::too_many_lines)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DagReview, DenseNodeId, NodeRef};
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, distance_correlation,
};

use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pc::collect_float_columns;
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink,
    discovery_assumptions,
};

/// `(parent index, coefficient)` pairs of one regression target.
type ParentCoefs = Vec<(usize, f64)>;

/// Static `DirectLiNGAM` discovery result (`Dag` evidence + review).
pub type StaticDagDiscoveryResult = DiscoveryResult<Dag, DagReview>;

/// `DirectLiNGAM` over tabular (non-temporal) continuous data.
#[derive(Clone, Debug)]
pub struct DirectLingam {
    /// Constraints / max parents / forbidden edges.
    pub constraints: DiscoveryConstraints,
    /// Prune threshold on the absolute *standardised* coefficient
    /// `|β| · sd(parent) / sd(child)` after order search (unit-free, in `[0, 1]` for a single
    /// predictor). Required edges are never pruned.
    pub prune_threshold: f64,
}

impl Default for DirectLingam {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectLingam {
    /// Default `DirectLiNGAM` (`prune_threshold = 0.05`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: DiscoveryConstraints {
                temporal: crate::constraints::TemporalConstraints {
                    max_lag: Lag::CONTEMPORANEOUS,
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                ..DiscoveryConstraints::default()
            },
            prune_threshold: 0.05,
        }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Prune threshold on the absolute standardised OLS coefficient.
    #[must_use]
    pub fn with_prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }

    /// Run `DirectLiNGAM`.
    ///
    /// # Errors
    ///
    /// Data, numerical, or graph failures.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticDagDiscoveryResult, DiscoveryError> {
        let _ = workspace;
        self.constraints.validate()?;
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "DirectLiNGAM requires at least one variable",
            });
        }

        let col_owned = collect_float_columns(data, variables)?;
        let n = col_owned[0].len();
        if n < 3 {
            return Err(DiscoveryError::stats_msg("insufficient rows for DirectLiNGAM"));
        }
        for c in &col_owned {
            if c.len() != n {
                return Err(DiscoveryError::data_msg("column length mismatch"));
            }
            require_finite_column(c)?;
        }

        let p = variables.len();
        // Working residual matrix (column-major via Vec per variable), centered.
        let mut cols: Vec<Vec<f64>> = col_owned
            .iter()
            .map(|c| {
                let mut v = c.to_vec();
                center_inplace(&mut v);
                v
            })
            .collect();

        let mut remaining: Vec<usize> = (0..p).collect();
        let mut order: Vec<usize> = Vec::with_capacity(p);

        // Required contemporaneous edges as index pairs `(parent, child)`.
        let required_pairs: Vec<(usize, usize)> = self
            .constraints
            .required
            .iter()
            .filter(|r| {
                r.source_lag == Lag::CONTEMPORANEOUS && r.target_lag == Lag::CONTEMPORANEOUS
            })
            .filter_map(|r| {
                let si = variables.iter().position(|v| *v == r.source)?;
                let ti = variables.iter().position(|v| *v == r.target)?;
                (si != ti).then_some((si, ti))
            })
            .collect();

        while remaining.len() > 1 {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let mut best_j = remaining[0];
            let mut best_score = f64::INFINITY;
            // A variable whose required parent is still unplaced cannot be exogenous yet:
            // ordering it first would silently contradict the constraint.
            let eligible: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&j| !required_pairs.iter().any(|&(a, b)| b == j && remaining.contains(&a)))
                .collect();
            if eligible.is_empty() {
                return Err(DiscoveryError::unsupported(
                    "required edges form a cycle: no causal order satisfies them",
                ));
            }
            for &j in &eligible {
                if ctx.cancellation.is_cancelled() {
                    return Err(DiscoveryError::Cancelled);
                }
                let mut score = 0.0;
                for &i in &remaining {
                    if i == j {
                        continue;
                    }
                    let resid = regress_residual(&cols[i], &cols[j]);
                    let d = distance_correlation(&resid, &cols[j]);
                    if !d.is_finite() {
                        return Err(DiscoveryError::stats_msg(
                            "DirectLiNGAM independence score is non-finite",
                        ));
                    }
                    score += d;
                }
                if score < best_score {
                    best_score = score;
                    best_j = j;
                }
            }
            if !best_score.is_finite() {
                return Err(DiscoveryError::stats_msg(
                    "DirectLiNGAM causal-order search produced a non-finite score",
                ));
            }
            // Residualize remaining on chosen exogenous.
            for &i in &remaining {
                if i == best_j {
                    continue;
                }
                cols[i] = regress_residual(&cols[i], &cols[best_j]);
            }
            remaining.retain(|&i| i != best_j);
            order.push(best_j);
        }
        if let Some(last) = remaining.pop() {
            order.push(last);
        }

        // Rebuild original centered columns for pruning and residual checks.
        let orig: Vec<Vec<f64>> = col_owned
            .iter()
            .map(|c| {
                let mut v = c.to_vec();
                center_inplace(&mut v);
                v
            })
            .collect();

        refuse_gaussian_consistent_residuals(&orig, &order)?;

        let max_parents = self.constraints.max_parents.unwrap_or(p.saturating_sub(1));
        let mut dag = Dag::empty();
        for &v in variables {
            dag.add_node(NodeRef::Static(v))?;
        }

        let mut edge_coefs: Vec<(usize, usize, f64)> = Vec::new();
        let backend = FaerBackend;
        let mut ls_ws = LeastSquaresWorkspace::default();
        // Standard deviations for the unit-free coefficient scale.
        let sd: Vec<f64> =
            orig.iter().map(|c| (c.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt()).collect();
        let standardised = |par: usize, child: usize, beta: f64| {
            if sd[child] > 0.0 { beta * sd[par] / sd[child] } else { 0.0 }
        };
        // Keep required parents unconditionally, then the largest standardised
        // coefficients above the threshold up to `max_parents`.
        let select = |child: usize, coefs: &[(usize, f64)]| -> Vec<(usize, f64)> {
            let (mut kept, mut optional): (ParentCoefs, ParentCoefs) =
                coefs.iter().copied().partition(|&(par, _)| required_pairs.contains(&(par, child)));
            optional.retain(|&(par, beta)| {
                standardised(par, child, beta).abs() >= self.prune_threshold
            });
            optional.sort_by(|a, b| {
                standardised(b.0, child, b.1)
                    .abs()
                    .partial_cmp(&standardised(a.0, child, a.1).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            optional.truncate(max_parents.saturating_sub(kept.len()));
            kept.extend(optional);
            kept
        };

        for (pos, &child) in order.iter().enumerate() {
            if pos == 0 {
                continue;
            }
            let preds: Vec<usize> = order[..pos]
                .iter()
                .copied()
                .filter(|&par| !forbidden_edge(&self.constraints, variables, par, child))
                .collect();
            if preds.is_empty() {
                continue;
            }
            // OLS child ~ preds (no intercept — already centered).
            let k = preds.len();
            let mut x = vec![0.0; n * k];
            for (c, &par) in preds.iter().enumerate() {
                for r in 0..n {
                    x[c * n + r] = orig[par][r];
                }
            }
            let coefs: Vec<(usize, f64)> =
                if let Ok(f) = backend.least_squares(&x, n, k, &orig[child], &mut ls_ws) {
                    preds.iter().enumerate().map(|(i, &par)| (par, f.coefficients[i])).collect()
                } else {
                    // Fall back to pairwise coefficients on rank failure.
                    preds
                        .iter()
                        .map(|&par| (par, simple_regression_coef(&orig[child], &orig[par])))
                        .collect()
                };
            for (par, beta) in select(child, &coefs) {
                edge_coefs.push((par, child, beta));
            }
        }

        for &(par, child, _) in &edge_coefs {
            let from = DenseNodeId::from_raw(crate::indexing::dense_u32(par));
            let to = DenseNodeId::from_raw(crate::indexing::dense_u32(child));
            if dag.children(from).contains(&to) {
                continue;
            }
            // Respect causal order: parent must precede child.
            let po = order.iter().position(|&x| x == par);
            let co = order.iter().position(|&x| x == child);
            if let (Some(pi), Some(ci)) = (po, co) {
                if pi < ci {
                    let _ = dag.insert_directed(from, to);
                }
            }
        }

        let edge_evidence: Vec<EdgeEvidence> = edge_coefs
            .iter()
            .filter_map(|&(par, child, beta)| {
                let po = order.iter().position(|&x| x == par)?;
                let co = order.iter().position(|&x| x == child)?;
                if po >= co {
                    return None;
                }
                Some(EdgeEvidence {
                    link: LaggedLink {
                        source: variables[par],
                        source_lag: Lag::CONTEMPORANEOUS,
                        target: variables[child],
                        target_lag: Lag::CONTEMPORANEOUS,
                    },
                    statistic: Some(beta),
                    p_value: None,
                    adjusted_p_value: None,
                    interval: None,
                    separating_set: None,
                    provenance: if required_pairs.contains(&(par, child)) {
                        Arc::from([Arc::from("direct_lingam"), Arc::from("required")])
                    } else {
                        Arc::from([Arc::from("direct_lingam")])
                    },
                })
            })
            .collect();

        let links: Vec<ScoredLink> = edge_evidence
            .iter()
            .map(|e| ScoredLink {
                link: e.link,
                statistic: e.statistic.unwrap_or(0.0),
                p_value: f64::NAN,
                adjusted_p_value: None,
            })
            .collect();

        let edge_count = dag.edges().count();
        let review = DagReview::from_dag(dag.clone(), "direct_lingam");
        let evidence = GraphEvidence {
            graph: dag,
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(links),
            source: EvidenceSource::Discovery { algorithm: Arc::from("direct_lingam") },
        };

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from("direct_lingam"),
                config: Arc::from(format!(
                    "prune_threshold={} order={order:?}",
                    self.prune_threshold
                )),
            },
            assumptions: discovery_assumptions("direct_lingam", true),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics: Vec::<DiscoveryDiagnostic>::new(),
            performance: DiscoveryPerformanceRecord {
                ci_tests: 0,
                links_retained: u64::try_from(edge_count).unwrap_or(u64::MAX),
                targets: u64::try_from(p).unwrap_or(u64::MAX),
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets: crate::result::PcSepsets::default(),
        })
    }
}

fn forbidden_edge(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    par: usize,
    child: usize,
) -> bool {
    let Some(&src) = variables.get(par) else {
        return true;
    };
    let Some(&tgt) = variables.get(child) else {
        return true;
    };
    let link = LaggedLink {
        source: src,
        source_lag: Lag::CONTEMPORANEOUS,
        target: tgt,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    constraints.is_forbidden(link) || constraints.tier_forbids(src, tgt)
}

fn require_finite_column(col: &[f64]) -> Result<(), DiscoveryError> {
    if col.iter().any(|x| !x.is_finite()) {
        return Err(DiscoveryError::data_msg(
            "DirectLiNGAM refuses non-finite (NaN/Inf) input; causal order is undefined",
        ));
    }
    Ok(())
}

/// Refuse when every exogenous residual under the estimated order is consistent
/// with a Gaussian law (Jarque–Bera at α=0.05). Under joint Gaussianity the
/// `LiNGAM` order is not identifiable, so returning an order would be silent fiction.
fn refuse_gaussian_consistent_residuals(
    centered: &[Vec<f64>],
    order: &[usize],
) -> Result<(), DiscoveryError> {
    let n = centered.first().map_or(0, Vec::len);
    if n < 8 || order.is_empty() {
        // Too few rows for a reliable moment test; still require at least one
        // variable with non-zero excess kurtosis as a weak gate.
        let any_heavy = centered.iter().any(|c| excess_kurtosis(c).abs() > 0.5);
        if !any_heavy {
            return Err(DiscoveryError::stats_msg(
                "DirectLiNGAM refuses data consistent with Gaussian errors; order is unidentifiable",
            ));
        }
        return Ok(());
    }

    let mut any_non_gaussian = false;
    for (pos, &idx) in order.iter().enumerate() {
        let resid = if pos == 0 {
            centered[idx].clone()
        } else {
            residual_on_predecessors(&centered[idx], centered, &order[..pos])
        };
        if !residual_looks_gaussian(&resid) {
            any_non_gaussian = true;
            break;
        }
    }
    if !any_non_gaussian {
        return Err(DiscoveryError::stats_msg(
            "DirectLiNGAM refuses residuals consistent with Gaussian errors (Jarque–Bera); \
             causal order is unidentifiable under Gaussian noise",
        ));
    }
    Ok(())
}

fn residual_on_predecessors(y: &[f64], cols: &[Vec<f64>], preds: &[usize]) -> Vec<f64> {
    if preds.is_empty() {
        return y.to_vec();
    }
    let n = y.len();
    let k = preds.len();
    let mut x = vec![0.0; n * k];
    for (c, &par) in preds.iter().enumerate() {
        for r in 0..n {
            x[c * n + r] = cols[par][r];
        }
    }
    let backend = FaerBackend;
    let mut ls_ws = LeastSquaresWorkspace::default();
    if let Ok(fit) = backend.least_squares(&x, n, k, y, &mut ls_ws) {
        let mut resid = y.to_vec();
        for r in 0..n {
            let mut pred = 0.0;
            for (c, coef) in fit.coefficients.iter().enumerate() {
                pred += coef * x[c * n + r];
            }
            resid[r] -= pred;
        }
        return resid;
    }
    // Pairwise residualize when the joint fit fails.
    let mut resid = y.to_vec();
    for &par in preds {
        resid = regress_residual(&resid, &cols[par]);
    }
    resid
}

/// Jarque–Bera normality gate: `JB = n/6 (S² + K²/4)` with asymptotic χ²(2).
/// Critical value at α=0.05 is 5.991; below that the residual is treated as
/// Gaussian-consistent.
fn residual_looks_gaussian(resid: &[f64]) -> bool {
    let n = resid.len();
    if n < 8 {
        return excess_kurtosis(resid).abs() <= 0.5;
    }
    let nf = n as f64;
    let mean = resid.iter().sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &x in resid {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    if m2 <= 1e-15 {
        // Degenerate residual — treat as Gaussian-consistent (no identifiable signal).
        return true;
    }
    let skew = m3 / m2.powf(1.5);
    let excess_kurt = m4 / (m2 * m2) - 3.0;
    let jb = nf / 6.0 * (skew * skew + excess_kurt * excess_kurt / 4.0);
    jb < 5.991
}

fn excess_kurtosis(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &v in x {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m4 /= nf;
    if m2 <= 1e-15 {
        return 0.0;
    }
    m4 / (m2 * m2) - 3.0
}

fn center_inplace(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

/// Residual of `y` after simple OLS on centered `x` (no intercept).
fn regress_residual(y: &[f64], x: &[f64]) -> Vec<f64> {
    let beta = simple_regression_coef(y, x);
    y.iter().zip(x.iter()).map(|(yi, xi)| yi - beta * xi).collect()
}

fn simple_regression_coef(y: &[f64], x: &[f64]) -> f64 {
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (yi, xi) in y.iter().zip(x.iter()) {
        sxx += xi * xi;
        sxy += xi * yi;
    }
    if sxx <= 1e-15 { 0.0 } else { sxy / sxx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };

    /// Non-Gaussian SEM: X0 → X1 → X2 with Laplace-like noise.
    fn lingam_chain(n: usize) -> (TabularData, Vec<VariableId>) {
        lingam_chain_scaled(n, [1.0, 1.0, 1.0])
    }

    /// [`lingam_chain`] with column `i` expressed in units `scale[i]` times larger.
    fn lingam_chain_scaled(n: usize, scale: [f64; 3]) -> (TabularData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..3 {
            b.add_variable(
                format!("x{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut x0 = vec![0.0; n];
        let mut x1 = vec![0.0; n];
        let mut x2 = vec![0.0; n];
        for i in 0..n {
            // Heavy-tailed noise via mixture of uniforms (non-Gaussian).
            let u = ((i as f64 * 0.137) % 1.0) - 0.5;
            let v = ((i as f64 * 0.271) % 1.0) - 0.5;
            let w = ((i as f64 * 0.419) % 1.0) - 0.5;
            let e0 = u * u * u * 4.0;
            let e1 = v * v * v * 4.0;
            let e2 = w * w * w * 4.0;
            x0[i] = e0;
            x1[i] = 0.9 * x0[i] + e1;
            x2[i] = 0.9 * x1[i] + e2;
        }
        for i in 0..n {
            x0[i] *= scale[0];
            x1[i] *= scale[1];
            x2[i] *= scale[2];
        }
        let owned = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x0),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(x1),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(x2),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        let data = TabularData::new(storage);
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        (data, vars)
    }

    #[test]
    fn direct_lingam_recovers_chain_order_edges() {
        let (data, vars) = lingam_chain(500);
        let alg = DirectLingam::new().with_prune_threshold(0.2);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "direct_lingam");
        assert!(!result.assumptions.is_empty());
        let g = &result.evidence.graph;
        let d = |i: u32| DenseNodeId::from_raw(i);
        assert!(
            g.children(d(0)).contains(&d(1)),
            "expected 0→1, edges={:?}",
            g.edges().collect::<Vec<_>>()
        );
        assert!(
            g.children(d(1)).contains(&d(2)),
            "expected 1→2, edges={:?}",
            g.edges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn direct_lingam_refuses_nan_column() {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..3 {
            b.add_variable(
                format!("x{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let n = 40;
        let mut x0 = vec![0.0; n];
        let mut x1 = vec![0.0; n];
        let mut x2 = vec![0.0; n];
        for i in 0..n {
            let u = ((i as f64 * 0.137) % 1.0) - 0.5;
            x0[i] = u * u * u * 4.0;
            x1[i] = 0.9 * x0[i] + (((i as f64 * 0.271) % 1.0) - 0.5).powi(3) * 4.0;
            x2[i] = 0.9 * x1[i] + (((i as f64 * 0.419) % 1.0) - 0.5).powi(3) * 4.0;
        }
        x1[7] = f64::NAN;
        let owned = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x0),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(x1),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(x2),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        let data = TabularData::new(storage);
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        let alg = DirectLingam::new();
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let err = alg.run(&data, &vars, &mut ws, &ctx).expect_err("NaN must refuse");
        let msg = err.to_string();
        assert!(msg.contains("non-finite") || msg.contains("NaN"), "unexpected error: {msg}");
    }

    #[test]
    fn dcor_self_near_one() {
        let x: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.1).collect();
        let d = distance_correlation(&x, &x);
        assert!(d > 0.99, "dCor(x,x)={d}");
    }

    /// Rescaling a column must not change the graph: the same chain expressed with the cause in
    /// units 10⁴ times larger and the effect 10⁴ times smaller still has 0→1 and 1→2.
    /// (On raw coefficients `0.9 · 10⁻⁴ / 10⁴ ≈ 9·10⁻⁹` falls under any fixed cut-off.)
    #[test]
    fn direct_lingam_pruning_is_unit_free() {
        let (data, vars) = lingam_chain_scaled(500, [1.0e4, 1.0, 1.0e-4]);
        let alg = DirectLingam::new().with_prune_threshold(0.2);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        let d = |i: u32| DenseNodeId::from_raw(i);
        assert!(g.children(d(0)).contains(&d(1)), "edges={:?}", g.edges().collect::<Vec<_>>());
        assert!(g.children(d(1)).contains(&d(2)), "edges={:?}", g.edges().collect::<Vec<_>>());
    }

    /// A required edge that contradicts the data-driven order constrains the order instead of
    /// being dropped: the required parent is placed before its child and the edge is kept,
    /// with its fitted coefficient (not a placeholder) as evidence.
    #[test]
    fn direct_lingam_required_edge_constrains_order() {
        let (data, vars) = lingam_chain(2000);
        let pair = &vars[..2];
        let mut alg = DirectLingam::new().with_prune_threshold(0.2);
        // Truth is x0 → x1; require the reverse.
        alg.constraints.required = Arc::from([LaggedLink {
            source: pair[1],
            source_lag: Lag::CONTEMPORANEOUS,
            target: pair[0],
            target_lag: Lag::CONTEMPORANEOUS,
        }]);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = alg.run(&data, pair, &mut ws, &ctx).unwrap();
        let d = |i: u32| DenseNodeId::from_raw(i);
        assert!(result.evidence.graph.children(d(1)).contains(&d(0)));
        assert!(!result.evidence.graph.children(d(0)).contains(&d(1)));
        let ev = result
            .evidence
            .edge_evidence
            .iter()
            .find(|e| e.link.source == pair[1] && e.link.target == pair[0])
            .expect("required edge reported");
        assert!(ev.provenance.iter().any(|p| &**p == "required"));
        assert_ne!(ev.statistic, Some(1.0), "no placeholder coefficient");
    }

    #[test]
    fn direct_lingam_refuses_cyclic_required_edges() {
        let (data, vars) = lingam_chain(500);
        let pair = &vars[..2];
        let mut alg = DirectLingam::new();
        let link = |a: usize, b: usize| LaggedLink {
            source: pair[a],
            source_lag: Lag::CONTEMPORANEOUS,
            target: pair[b],
            target_lag: Lag::CONTEMPORANEOUS,
        };
        alg.constraints.required = Arc::from([link(0, 1), link(1, 0)]);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        assert!(alg.run(&data, pair, &mut ws, &ctx).is_err());
    }
}
