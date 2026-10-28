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

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DagReview, DenseNodeId, NodeRef};
#[cfg(test)]
use antecedent_stats::distance_correlation;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, dcor::DistanceY};

use crate::constraints::{DiscoveryConstraints, KnowledgeScope};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pc::collect_float_columns;
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink,
    discovery_assumptions,
};

/// Identity of this algorithm in its records.
const ALGORITHM: &str = crate::pipeline::StaticAlgorithm::DirectLingam.as_str();

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
    #[allow(
        clippy::too_many_lines,
        reason = "one algorithm pipeline over shared mutable graph state; the phases are marked in place"
    )]
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        _workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticDagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        self.constraints
            .check_background_knowledge(KnowledgeScope::ScoreSearch, variables)
            .map_err(DiscoveryError::Knowledge)?;
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
        // Centred columns (kept for the coefficient fits) and their standard deviations.
        let centered: Vec<Vec<f64>> = col_owned
            .iter()
            .map(|c| {
                let mut v = c.to_vec();
                center_inplace(&mut v);
                v
            })
            .collect();
        let sd: Vec<f64> = centered
            .iter()
            .map(|c| (c.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt())
            .collect();
        for (i, raw) in col_owned.iter().enumerate() {
            // A column is constant when its spread is rounding noise on its own magnitude.
            let magnitude = raw.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if !(sd[i] > 64.0 * f64::EPSILON * magnitude) {
                return Err(DiscoveryError::data_msg(format!(
                    "DirectLiNGAM refuses constant variable {}: a constant has no causal \
                     order",
                    variables[i].raw()
                )));
            }
        }
        // Working residual matrix, standardised so the degeneracy tests below are relative to
        // each variable's own scale: rescaling a column cannot make it look degenerate.
        let cols: Vec<Vec<f64>> =
            centered.iter().zip(&sd).map(|(c, &s)| c.iter().map(|v| v / s).collect()).collect();

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

        let order = causal_order(cols, &required_pairs, ctx)?;

        let gaussian = gaussian_consistent_residuals(&centered, &sd, &order)?;
        let mut diagnostics = Vec::new();
        if gaussian.len() >= 2 {
            let names: Vec<String> = gaussian
                .iter()
                .map(|&i| {
                    data.schema()
                        .get(variables[i])
                        .map_or_else(|_| format!("{:?}", variables[i]), |v| v.name.to_string())
                })
                .collect();
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("direct_lingam.gaussian_residuals"),
                message: Arc::from(format!(
                    "{} exogenous residuals are consistent with Gaussian errors (Jarque-Bera, \
                     alpha 0.05): {}. LiNGAM identifies the causal order only when at most one \
                     error is Gaussian, so the relative order of these variables, and the \
                     direction of edges between them, is not identified",
                    names.len(),
                    names.join(", ")
                )),
            });
        }

        let mut dag = Dag::empty();
        for &v in variables {
            dag.add_node(NodeRef::Static(v))?;
        }

        let edge_coefs = self.fit_edges(&centered, &sd, &order, &required_pairs, variables)?;

        // Every coefficient edge runs from a predecessor in the causal order (the regressors of
        // `child` are `order[..pos]`), so each is a forward edge.
        for &(par, child, _) in &edge_coefs {
            let from = DenseNodeId::from_raw(crate::indexing::dense_u32(par));
            let to = DenseNodeId::from_raw(crate::indexing::dense_u32(child));
            if dag.children(from).contains(&to) {
                continue;
            }
            insert_forward(&mut dag, from, to)?;
        }

        let edge_evidence: Vec<EdgeEvidence> = edge_coefs
            .iter()
            .map(|&(par, child, beta)| EdgeEvidence {
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
                    Arc::from([Arc::from(ALGORITHM), Arc::from("required")])
                } else {
                    Arc::from([Arc::from(ALGORITHM)])
                },
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
        let review = DagReview::from_dag(dag.clone(), ALGORITHM);
        let evidence = GraphEvidence {
            graph: dag,
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(links),
            source: EvidenceSource::Discovery { algorithm: Arc::from(ALGORITHM) },
        };

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from(ALGORITHM),
                config: Arc::from(format!(
                    "prune_threshold={} order={order:?}{}",
                    self.prune_threshold,
                    self.constraints.knowledge_config()
                )),
            },
            assumptions: discovery_assumptions(ALGORITHM, true),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics,
            performance: DiscoveryPerformanceRecord {
                ci_tests: 0,
                links_retained: crate::pipeline::count_u64(edge_count)?,
                targets: crate::pipeline::count_u64(p)?,
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets: crate::result::PcSepsets::default(),
        })
    }
}

/// Insert a forward edge of a causal order. A directed cycle is refused and is
/// not dropped.
fn insert_forward(dag: &mut Dag, from: DenseNodeId, to: DenseNodeId) -> Result<(), DiscoveryError> {
    dag.insert_directed(from, to).map_err(DiscoveryError::from)
}

impl DirectLingam {
    /// The pruned `(parent, child, coefficient)` edges of the causal `order`: each variable is
    /// regressed on its allowed predecessors, required parents are kept unconditionally, then
    /// the largest standardised coefficients above the threshold up to `max_parents`.
    fn fit_edges(
        &self,
        centered: &[Vec<f64>],
        sd: &[f64],
        order: &[usize],
        required_pairs: &[(usize, usize)],
        variables: &[VariableId],
    ) -> Result<Vec<(usize, usize, f64)>, DiscoveryError> {
        let p = variables.len();
        let n = centered.first().map_or(0, Vec::len);
        let max_parents = self.constraints.max_parents.unwrap_or(p.saturating_sub(1));
        let mut edge_coefs: Vec<(usize, usize, f64)> = Vec::new();
        let mut ls_ws = LeastSquaresWorkspace::default();
        // `sd` gives the unit-free coefficient scale.
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
                standardised(b.0, child, b.1).abs().total_cmp(&standardised(a.0, child, a.1).abs())
            });
            optional.truncate(max_parents.saturating_sub(kept.len()));
            kept.extend(optional);
            kept
        };

        let index = crate::constraints::ConstraintIndex::compile(&self.constraints);
        for (pos, &child) in order.iter().enumerate() {
            if pos == 0 {
                continue;
            }
            let preds: Vec<usize> = order[..pos]
                .iter()
                .copied()
                .filter(|&par| {
                    !index.is_forbidden(
                        &self.constraints,
                        crate::constraints::contemp_link(variables[par], variables[child]),
                    )
                })
                .collect();
            if preds.is_empty() {
                continue;
            }
            // OLS child ~ preds (no intercept — already centered).
            let k = preds.len();
            let mut x = vec![0.0; n * k];
            for (c, &par) in preds.iter().enumerate() {
                for r in 0..n {
                    x[c * n + r] = centered[par][r];
                }
            }
            let f = predecessor_regression(&x, n, k, &centered[child], &mut ls_ws)?;
            let coefs: Vec<(usize, f64)> =
                preds.iter().enumerate().map(|(i, &par)| (par, f.coefficients[i])).collect();
            for (par, beta) in select(child, &coefs) {
                edge_coefs.push((par, child, beta));
            }
        }
        Ok(edge_coefs)
    }
}

/// The causal order by DirectLiNGAM's exogeneity search over the working `cols` (standardised
/// columns, consumed as they are residualized): repeatedly the variable whose residuals on
/// every other remaining variable are least dependent on it goes first, the rest are regressed
/// on it, and the search continues. A variable whose required parent is still unplaced is not
/// eligible.
///
/// # Errors
///
/// Cancellation, required edges that form a cycle, or a non-finite independence score.
fn causal_order(
    mut cols: Vec<Vec<f64>>,
    required_pairs: &[(usize, usize)],
    ctx: &ExecutionContext,
) -> Result<Vec<usize>, DiscoveryError> {
    let p = cols.len();
    let n = cols.first().map_or(0, Vec::len);
    let mut remaining: Vec<usize> = (0..p).collect();
    let mut in_remaining = vec![true; p];
    // Residual buffer reused by every pairwise score.
    let mut resid = vec![0.0; n];
    // Cross products of the current working columns, `gram[i * p + j] = <cols[i], cols[j]>`,
    // formed once per round: every ordered pair's slope (and the residualization after the
    // round) reads it instead of re-summing the two columns.
    let mut gram = vec![0.0; p * p];
    let mut order: Vec<usize> = Vec::with_capacity(p);
    while remaining.len() > 1 {
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }
        fill_gram(&mut gram, &cols, &remaining);
        let mut best_j = remaining[0];
        let mut best_score = f64::INFINITY;
        // A variable whose required parent is still unplaced cannot be exogenous yet:
        // ordering it first would silently contradict the constraint.
        let eligible: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&j| !required_pairs.iter().any(|&(a, b)| b == j && in_remaining[a]))
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
            // `cols[j]` is residualized after this round, so the cache dies with `j`.
            let y = DistanceY::new(&cols[j]);
            for &i in &remaining {
                if i == j {
                    continue;
                }
                let beta = slope_from_gram(gram[i * p + j], gram[j * p + j], n);
                residual_into(&cols[i], &cols[j], beta, &mut resid);
                let d = y.correlate(&resid);
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
        let exogenous = std::mem::take(&mut cols[best_j]);
        for &i in &remaining {
            if i != best_j {
                let beta = slope_from_gram(gram[i * p + best_j], gram[best_j * p + best_j], n);
                for (yi, xi) in cols[i].iter_mut().zip(&exogenous) {
                    *yi -= beta * xi;
                }
            }
        }
        remaining.retain(|&i| i != best_j);
        in_remaining[best_j] = false;
        order.push(best_j);
    }
    if let Some(last) = remaining.pop() {
        order.push(last);
    }
    Ok(order)
}

fn require_finite_column(col: &[f64]) -> Result<(), DiscoveryError> {
    if col.iter().any(|x| !x.is_finite()) {
        return Err(DiscoveryError::data_msg(
            "DirectLiNGAM refuses non-finite (NaN/Inf) input; causal order is undefined",
        ));
    }
    Ok(())
}

/// Variables whose exogenous residual under the estimated order is consistent with a Gaussian
/// law (Jarque–Bera at α=0.05), in causal order.
///
/// Refuses when every residual is: under joint Gaussianity the `LiNGAM` order is not
/// identifiable at all, so returning an order would be silent fiction. Two or more
/// Gaussian-consistent residuals leave only their relative order unidentified; the caller
/// reports that rather than refusing.
fn gaussian_consistent_residuals(
    centered: &[Vec<f64>],
    scales: &[f64],
    order: &[usize],
) -> Result<Vec<usize>, DiscoveryError> {
    let n = centered.first().map_or(0, Vec::len);
    if n < 8 || order.is_empty() {
        // Too few rows for a reliable moment test; still require at least one
        // variable with non-zero excess kurtosis as a weak gate.
        let any_heavy =
            centered.iter().zip(scales).any(|(c, &scale)| excess_kurtosis(c, scale).abs() > 0.5);
        if !any_heavy {
            return Err(DiscoveryError::stats_msg(
                "DirectLiNGAM refuses data consistent with Gaussian errors; order is unidentifiable",
            ));
        }
        return Ok(Vec::new());
    }

    let mut gaussian = Vec::new();
    for (pos, &idx) in order.iter().enumerate() {
        let resid = if pos == 0 {
            centered[idx].clone()
        } else {
            residual_on_predecessors(&centered[idx], centered, &order[..pos])?
        };
        if residual_looks_gaussian(&resid, scales[idx]) {
            gaussian.push(idx);
        }
    }
    if gaussian.len() == order.len() {
        return Err(DiscoveryError::stats_msg(
            "DirectLiNGAM refuses residuals consistent with Gaussian errors (Jarque–Bera); \
             causal order is unidentifiable under Gaussian noise",
        ));
    }
    Ok(gaussian)
}

/// Regress a variable on its causal predecessors (column-major `x`, `n × k`).
///
/// Full-rank designs use the QR solve. When collinear predecessors make the design rank
/// deficient, the individual coefficients are not identified; the minimum-norm solution keeps
/// the joint fit (its residual is the exact projection residual) and spreads the effect over
/// the collinear columns, instead of marginal pairwise slopes that double count it.
fn predecessor_regression(
    x: &[f64],
    n: usize,
    k: usize,
    y: &[f64],
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<antecedent_stats::LeastSquaresFit, DiscoveryError> {
    let backend = FaerBackend;
    match backend.least_squares(x, n, k, y, ls_ws) {
        Err(antecedent_stats::StatsError::RankDeficient { .. }) => {
            backend.min_norm_least_squares(x, n, k, y)
        }
        other => other,
    }
    .map_err(|e| {
        DiscoveryError::stats_msg(format!(
            "DirectLiNGAM cannot regress a variable on its causal predecessors ({e})"
        ))
    })
}

fn residual_on_predecessors(
    y: &[f64],
    cols: &[Vec<f64>],
    preds: &[usize],
) -> Result<Vec<f64>, DiscoveryError> {
    if preds.is_empty() {
        return Ok(y.to_vec());
    }
    let n = y.len();
    let k = preds.len();
    let mut x = vec![0.0; n * k];
    for (c, &par) in preds.iter().enumerate() {
        for r in 0..n {
            x[c * n + r] = cols[par][r];
        }
    }
    let mut ls_ws = LeastSquaresWorkspace::default();
    let fit = predecessor_regression(&x, n, k, y, &mut ls_ws)?;
    let mut resid = y.to_vec();
    for r in 0..n {
        let mut pred = 0.0;
        for (c, coef) in fit.coefficients.iter().enumerate() {
            pred += coef * x[c * n + r];
        }
        resid[r] -= pred;
    }
    Ok(resid)
}

/// Jarque–Bera normality gate: `JB = n/6 (S² + K²/4)` with asymptotic χ²(2).
/// Critical value at α=0.05 is 5.991; below that the residual is treated as
/// Gaussian-consistent.
///
/// This is a screening heuristic, not a calibrated test. The χ²(2) reference is an asymptotic
/// result and the statistic converges to it slowly (the sample skewness and kurtosis are far
/// from their normal limits for moderate `n`), so the realised size at `n` in the tens to low
/// hundreds is not 5%. Below 8 rows the moment estimates are too unstable for the statistic
/// at all and a fixed excess-kurtosis bound of 0.5 stands in. The verdict only feeds the
/// `direct_lingam.gaussian_residuals` diagnostic and the refusal of data that look entirely
/// Gaussian; it never changes the estimated order.
///
/// `scale` is the standard deviation of the variable the residual belongs to: a residual whose
/// variance is below `1e-15` of that variable's own is degenerate whatever the units.
fn residual_looks_gaussian(resid: &[f64], scale: f64) -> bool {
    let n = resid.len();
    if n < 8 {
        return excess_kurtosis(resid, scale).abs() <= 0.5;
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
    if m2 <= 1e-15 * scale * scale {
        // Degenerate residual — treat as Gaussian-consistent (no identifiable signal).
        return true;
    }
    let skew = m3 / m2.powf(1.5);
    let excess_kurt = m4 / (m2 * m2) - 3.0;
    let jb = nf / 6.0 * (skew * skew + excess_kurt * excess_kurt / 4.0);
    jb < 5.991
}

/// Excess kurtosis of `x`, `0` when its variance is below `1e-15` of `scale²`.
fn excess_kurtosis(x: &[f64], scale: f64) -> f64 {
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
    if m2 <= 1e-15 * scale * scale {
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

/// Fill the entries of `gram` among the `active` columns with their cross products.
///
/// Each unordered pair is summed once, in row order; `<a, b>` and `<b, a>` are the same sum, so
/// the slopes read from it are the ones a column-by-column regression computes.
fn fill_gram(gram: &mut [f64], cols: &[Vec<f64>], active: &[usize]) {
    let p = cols.len();
    for (position, &i) in active.iter().enumerate() {
        for &j in &active[position..] {
            let mut sum = 0.0;
            for (a, b) in cols[i].iter().zip(&cols[j]) {
                sum += a * b;
            }
            gram[i * p + j] = sum;
            gram[j * p + i] = sum;
        }
    }
}

/// Residual `y - beta * x`, written into `out`.
fn residual_into(y: &[f64], x: &[f64], beta: f64, out: &mut [f64]) {
    for ((o, yi), xi) in out.iter_mut().zip(y).zip(x) {
        *o = yi - beta * xi;
    }
}

/// Least-squares slope of `y` on `x` from `sxy = <x, y>` and `sxx = <x, x>` over `n` rows; `0`
/// when `x` has no variance.
///
/// The columns it sees are standardised (and their residuals), so `sxx` is compared with the
/// `n` it would have at unit variance: `x` is degenerate when it has lost all but `1e-15` of
/// the variance it started with.
fn slope_from_gram(sxy: f64, sxx: f64, n: usize) -> f64 {
    if sxx <= 1e-15 * n as f64 { 0.0 } else { sxy / sxx }
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

    #[test]
    fn a_two_cycle_is_not_inserted_and_a_chain_is() {
        let mut dag = Dag::with_variables(2);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        insert_forward(&mut dag, a, b).unwrap();
        let err = insert_forward(&mut dag, b, a).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("condensation"), "{text}");
        assert!(text.contains("sigma_separates"), "{text}");
        assert!(dag.children(b).is_empty(), "the return edge was not dropped into the DAG");
        assert_eq!(dag.topological_order().unwrap(), vec![a, b]);
    }

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

    /// A table of float columns `x0, x1, ...`.
    fn table(columns: Vec<Vec<f64>>) -> (TabularData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..columns.len() {
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
        let owned = columns
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                let n = c.len();
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Arc::from(c),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        let data = TabularData::new(storage);
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        (data, vars)
    }

    /// Two Gaussian exogenous errors: their relative order is not identified. The run still
    /// returns a graph, but must say which variables' order it cannot vouch for.
    #[test]
    fn direct_lingam_reports_two_gaussian_consistent_residuals() {
        let n = 500;
        // The two exogenous errors are Gaussian by construction: each is the exact normal
        // quantile sequence at (i + 1/2) / n, put in a different deterministic order by a
        // stride coprime to n, so the Jarque-Bera verdict on each does not depend on a draw.
        // The third error is strongly skewed (centred exponential quantiles in a third
        // order), so the non-Gaussian variable is identified with high power at n = 500.
        let quantiles = |stride: usize, f: &dyn Fn(f64) -> f64| -> Vec<f64> {
            (0..n).map(|i| f((((i * stride) % n) as f64 + 0.5) / n as f64)).collect()
        };
        let x0 = quantiles(7, &antecedent_kernels::norm_inv);
        let x1 = quantiles(13, &antecedent_kernels::norm_inv);
        let skew = quantiles(11, &|p: f64| -(1.0 - p).ln() - 1.0);
        let x2: Vec<f64> = (0..n).map(|i| x0[i] + x1[i] + 2.0 * skew[i]).collect();
        let (data, vars) = table(vec![x0, x1, x2]);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = DirectLingam::new().run(&data, &vars, &mut ws, &ctx).unwrap();
        let diag = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "direct_lingam.gaussian_residuals")
            .unwrap_or_else(|| panic!("{:?}", result.diagnostics));
        assert!(diag.message.contains("x0") && diag.message.contains("x1"), "{}", diag.message);
        // The non-Gaussian chain carries no such warning.
        let (data, vars) = lingam_chain(500);
        let result = DirectLingam::new().run(&data, &vars, &mut ws, &ctx).unwrap();
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    }

    /// An exactly collinear pair (`x1 = 2·x0`) leaves the joint regression of `x2` on both
    /// rank deficient. The minimum-norm solve keeps the joint fit: the pair's coefficients
    /// jointly reproduce the least-squares slope `s` of `x2` on `x0` (`β0 + 2·β1 = s`), split
    /// in the minimum-norm ratio `β1 = 2·β0`, instead of the pairwise slopes `s` and `s/2`,
    /// which together double count the effect.
    #[test]
    fn direct_lingam_resolves_rank_deficient_regressions_by_minimum_norm() {
        let n = 400;
        let cube = |k: f64| -> Vec<f64> {
            (0..n).map(|i| (((i as f64 * k) % 1.0) - 0.5).powi(3) * 4.0).collect()
        };
        let x0 = cube(0.137);
        let x1: Vec<f64> = x0.iter().map(|v| 2.0 * v).collect();
        let e2 = cube(0.419);
        let x2: Vec<f64> = (0..n).map(|i| 0.9 * x0[i] + e2[i]).collect();
        let centred = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / n as f64;
            v.iter().map(|a| a - m).collect::<Vec<f64>>()
        };
        let (c0, c2) = (centred(&x0), centred(&x2));
        let slope = c0.iter().zip(&c2).map(|(a, b)| a * b).sum::<f64>()
            / c0.iter().map(|a| a * a).sum::<f64>();
        let (data, vars) = table(vec![x0, x1, x2]);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = DirectLingam::new().run(&data, &vars, &mut ws, &ctx).unwrap();
        let coef = |src: usize| {
            result
                .evidence
                .edge_evidence
                .iter()
                .find(|e| e.link.source == vars[src] && e.link.target == vars[2])
                .and_then(|e| e.statistic)
                .unwrap_or(0.0)
        };
        let (b0, b1) = (coef(0), coef(1));
        assert!((b0 + 2.0 * b1 - slope).abs() < 1e-9, "b0={b0} b1={b1} slope={slope}");
        assert!((b1 - 2.0 * b0).abs() < 1e-9, "b0={b0} b1={b1}");
        assert!((b0 - slope).abs() > 0.1, "pairwise slope reported: {b0}");
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

#[cfg(test)]
#[path = "lingam_oracle_tests.rs"]
mod oracle_tests;
