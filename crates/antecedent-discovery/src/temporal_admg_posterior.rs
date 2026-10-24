//! Posterior over a declared family of lagged temporal-ADMG templates.
//!
//! Each candidate template ([`TemporalAdmgAtom`]) is scored by the Gaussian BIC of the ordinary
//! Markov model of a stationary latent-confounded process, conditional on the past:
//!
//! ```text
//! X_t = B X_t + Γ Z_t + ε_t,    ε_t ~ N(0, Ω),    Z_t = (X_{t-1}, …, X_{t-L}),
//! ```
//!
//! where `B_ij ≠ 0` only for a contemporaneous directed edge `X_j → X_i` (acyclic), `Γ` carries
//! the lagged edges, and `Ω_ij ≠ 0` only for `i = j` or a contemporaneous latent-confounder
//! edge `X_i ↔ X_j`. The maximum-likelihood fit with bidirected edges is iterative conditional
//! fitting (Drton and Richardson 2004): each node in turn is regressed on its parents, its lagged
//! causes and the residuals of its spouses adjusted for the rest of the error covariance.
//!
//! A lagged latent-confounder edge `X_i(t-ℓ) ↔ X_j(t)` and the lagged directed edge
//! `X_i(t-ℓ) → X_j(t)` give the same conditional likelihood given the past (both are one free
//! coefficient of the regression of `X_j(t)` on `X_i(t-ℓ)`), so the data do not tell them apart:
//! such candidates score equal and share the posterior mass in proportion to their prior. The
//! same holds for `X_i → X_j` against `X_i ↔ X_j` between two variables with no other edge. The
//! posterior is the exact BIC posterior over the declared family, with an optional prior over it;
//! it is a model average over the hypotheses the caller declares, not a search of the space of all
//! templates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TableView, TimeSeriesData};
use antecedent_prob::kish_ess;

use crate::adjacency_mask::{AdjacencyMask, has_edge};
use crate::error::DiscoveryError;
use crate::graph_posterior::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, accumulate_marginals,
    analytic_graph_diagnostics, normalize_log_weights,
};
use crate::temporal_admg_atoms::{TemporalAdmgAtom, has_lagged_edge};

/// Most variables a candidate family may span (the posterior packing's own bound).
const MAX_VARIABLES: usize = crate::adjacency_mask::MAX_POSTERIOR_NODES;
/// Sweeps of iterative conditional fitting before the fit is taken as converged.
const MAX_SWEEPS: usize = 500;
/// Relative log-likelihood change below which a sweep ends the fit.
const FIT_TOLERANCE: f64 = 1e-11;

/// Exact BIC posterior over a declared family of lagged temporal-ADMG templates.
#[derive(Clone, Debug)]
pub struct TemporalAdmgPosterior {
    /// Longest lag the family reaches (`≥ 1`).
    pub max_lag: u32,
}

impl TemporalAdmgPosterior {
    /// A posterior over templates that reach back at most `max_lag` periods.
    #[must_use]
    pub fn new(max_lag: u32) -> Self {
        Self { max_lag: max_lag.max(1) }
    }

    /// Score every candidate on `data` and normalize.
    ///
    /// `variables` names the variables of the masks by position. `log_prior` is the optional
    /// log prior mass of each candidate (uniform when omitted).
    ///
    /// # Errors
    ///
    /// An empty family, a candidate that is not an acyclic ADMG over `variables` within
    /// `max_lag` lags, a prior of another length, a series too short for the lag window, a
    /// cancellation, or no candidate with a finite score.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        candidates: &[TemporalAdmgAtom],
        log_prior: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let p = variables.len();
        if p == 0 || p > MAX_VARIABLES {
            return Err(DiscoveryError::unsupported(
                "a temporal ADMG posterior covers between one and sixteen variables",
            ));
        }
        if candidates.is_empty() {
            return Err(DiscoveryError::unsupported("empty candidate family"));
        }
        if log_prior.is_some_and(|prior| prior.len() != candidates.len()) {
            return Err(DiscoveryError::unsupported(
                "log prior and candidate family differ in length",
            ));
        }
        if (self.max_lag as usize).saturating_mul(p * p) > 64 {
            return Err(DiscoveryError::unsupported(
                "the lag space exceeds the 64 bits of a lag word (max_lag * n_vars^2 <= 64)",
            ));
        }
        for candidate in candidates {
            // Acyclicity and the lag bound are the template's own.
            candidate.to_template(p, self.max_lag, variables)?;
        }
        let design = Design::build(data, variables, self.max_lag)?;
        let mut kept = Vec::new();
        let mut log_weights = Vec::new();
        let mut rejected = 0u64;
        for (index, candidate) in candidates.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let Some(bic) = score(&design, candidate, p, self.max_lag) else {
                rejected += 1;
                continue;
            };
            let prior = log_prior.map_or(0.0, |prior| prior[index]);
            kept.push(*candidate);
            log_weights.push(-0.5 * bic + prior);
        }
        if kept.is_empty() {
            return Err(DiscoveryError::unsupported(
                "no candidate template has a finite score on this series",
            ));
        }
        let weights = normalize_log_weights(&log_weights)?;
        let ess = kish_ess(&weights);
        let masks: Vec<AdjacencyMask> = kept.iter().map(|atom| atom.contemporaneous).collect();
        let (edge, orient) = accumulate_marginals(p, &weights, &masks);
        let cells = self.max_lag as usize * p * p;
        let mut lagged = vec![0.0; cells];
        for (atom, &weight) in kept.iter().zip(&weights) {
            for (bit, cell) in lagged.iter_mut().enumerate() {
                if (atom.lagged_directed >> bit) & 1 == 1 {
                    *cell += weight;
                }
            }
        }
        let diagnostics = analytic_graph_diagnostics(kept.len(), ess);
        Ok(GraphPosterior::from_atoms(
            p,
            weights,
            masks,
            edge,
            orient,
            SamplingSummary { ess, diagnostics, rejected_invalid: rejected },
        )?
        .with_atom_kind(GraphPosteriorAtomKind::TemporalAdmg)
        .with_lagged_marginals(self.max_lag, lagged)?
        .with_lag_masks(kept.iter().map(|atom| atom.lagged_directed).collect::<Vec<_>>())?
        .with_lag_bidirected_masks(
            kept.iter().map(|atom| atom.lagged_bidirected).collect::<Vec<_>>(),
        )?
        .with_algorithm(Arc::<str>::from("temporal_admg_posterior")))
    }
}

/// The centered series, lag-aligned: column `j` of `x` is `X_j(t)` and column
/// `(lag - 1) * p + i` of `z` is `X_i(t - lag)`, for the rows `t = max_lag ..`.
struct Design {
    n: usize,
    x: Vec<Vec<f64>>,
    z: Vec<Vec<f64>>,
}

impl Design {
    fn build(
        data: &TimeSeriesData,
        variables: &[VariableId],
        max_lag: u32,
    ) -> Result<Self, DiscoveryError> {
        let p = variables.len();
        let lags = max_lag as usize;
        let length = data.row_count();
        let n = length.saturating_sub(lags);
        if n < 2 * (p * (lags + 1) + 1) {
            return Err(DiscoveryError::data_msg(
                "the series is too short for the temporal ADMG window of this family",
            ));
        }
        let mut x = vec![Vec::with_capacity(n); p];
        let mut z = vec![Vec::with_capacity(n); p * lags];
        for (column, &variable) in variables.iter().enumerate() {
            let series = data.float64_values(variable).map_err(DiscoveryError::from)?;
            if series.len() != length {
                return Err(DiscoveryError::data_msg("series length mismatch"));
            }
            x[column] = series[lags..].to_vec();
            for lag in 1..=lags {
                z[(lag - 1) * p + column] = series[lags - lag..length - lag].to_vec();
            }
        }
        for column in x.iter_mut().chain(z.iter_mut()) {
            let mean = column.iter().sum::<f64>() / n as f64;
            for value in column.iter_mut() {
                *value -= mean;
            }
        }
        Ok(Self { n, x, z })
    }
}

/// Dense Cholesky factor (lower, row-major) of a symmetric positive-definite matrix.
fn cholesky(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[i * d + j];
            for k in 0..j {
                sum -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if !(sum > 1e-300) || !sum.is_finite() {
                    return None;
                }
                l[i * d + i] = sum.sqrt();
            } else {
                l[i * d + j] = sum / l[j * d + j];
            }
        }
    }
    Some(l)
}

/// Solve `L Lᵀ x = b`.
fn cholesky_solve(l: &[f64], d: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; d];
    for i in 0..d {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * d + k] * y[k];
        }
        y[i] = sum / l[i * d + i];
    }
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut sum = y[i];
        for k in i + 1..d {
            sum -= l[k * d + i] * x[k];
        }
        x[i] = sum / l[i * d + i];
    }
    x
}

/// Inverse and log-determinant of a symmetric positive-definite matrix.
fn invert_spd(a: &[f64], d: usize) -> Option<(Vec<f64>, f64)> {
    let l = cholesky(a, d)?;
    let mut inverse = vec![0.0; d * d];
    let mut unit = vec![0.0; d];
    for column in 0..d {
        unit.iter_mut().for_each(|v| *v = 0.0);
        unit[column] = 1.0;
        let solved = cholesky_solve(&l, d, &unit);
        for (row, value) in solved.into_iter().enumerate() {
            inverse[row * d + column] = value;
        }
    }
    let log_det = 2.0 * (0..d).map(|i| l[i * d + i].ln()).sum::<f64>();
    Some((inverse, log_det))
}

/// Least squares of `y` on `columns`: the coefficients and the residual sum of squares.
fn least_squares(columns: &[&[f64]], y: &[f64]) -> Option<(Vec<f64>, f64)> {
    let m = columns.len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
    let mut gram = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for i in 0..m {
        rhs[i] = dot(columns[i], y);
        for j in 0..=i {
            let value = dot(columns[i], columns[j]);
            gram[i * m + j] = value;
            gram[j * m + i] = value;
        }
    }
    let coefficients = if m == 0 {
        Vec::new()
    } else {
        let l = cholesky(&gram, m)?;
        cholesky_solve(&l, m, &rhs)
    };
    let mut rss = 0.0;
    for (t, &target) in y.iter().enumerate() {
        let fitted: f64 = columns.iter().zip(&coefficients).map(|(column, c)| c * column[t]).sum();
        rss += (target - fitted) * (target - fitted);
    }
    rss.is_finite().then_some((coefficients, rss))
}

/// One node's regressors: contemporaneous parents (variable index) and lagged causes (column of
/// `z`), and its spouses.
struct NodeSpec {
    parents: Vec<usize>,
    lagged: Vec<usize>,
    spouses: Vec<usize>,
}

/// The fit's free parameters.
struct Fit {
    parent_coefficients: Vec<Vec<f64>>,
    lagged_coefficients: Vec<Vec<f64>>,
    omega: Vec<f64>,
}

/// Residuals `ε_j = X_j - Σ B_jk X_k - Σ Γ_j· Z` of node `j`.
fn residuals(design: &Design, spec: &NodeSpec, fit: &Fit, j: usize) -> Vec<f64> {
    let mut e = design.x[j].clone();
    for (&parent, &c) in spec.parents.iter().zip(&fit.parent_coefficients[j]) {
        for (t, value) in e.iter_mut().enumerate() {
            *value -= c * design.x[parent][t];
        }
    }
    for (&column, &c) in spec.lagged.iter().zip(&fit.lagged_coefficients[j]) {
        for (t, value) in e.iter_mut().enumerate() {
            *value -= c * design.z[column][t];
        }
    }
    e
}

/// The Gaussian BIC of `candidate` on the design, `None` when the fit is degenerate.
fn score(design: &Design, candidate: &TemporalAdmgAtom, p: usize, max_lag: u32) -> Option<f64> {
    let mut specs: Vec<NodeSpec> = Vec::with_capacity(p);
    let mut parameters = p; // the error variances
    for j in 0..p {
        let parents: Vec<usize> = (0..p)
            .filter(|&i| {
                i != j
                    && has_edge(candidate.contemporaneous, p, i, j)
                    && !has_edge(candidate.contemporaneous, p, j, i)
            })
            .collect();
        let spouses: Vec<usize> = (0..p)
            .filter(|&i| {
                i != j
                    && has_edge(candidate.contemporaneous, p, i, j)
                    && has_edge(candidate.contemporaneous, p, j, i)
            })
            .collect();
        let mut lagged = Vec::new();
        for lag in 1..=max_lag {
            for i in 0..p {
                // A lagged latent-confounder edge and a lagged directed edge are one free
                // coefficient of the regression on the past.
                if has_lagged_edge(candidate.lagged_directed, p, lag, i, j)
                    || has_lagged_edge(candidate.lagged_bidirected, p, lag, i, j)
                {
                    lagged.push((lag as usize - 1) * p + i);
                }
            }
        }
        parameters += parents.len() + lagged.len() + spouses.iter().filter(|&&s| s > j).count();
        specs.push(NodeSpec { parents, lagged, spouses });
    }
    let n = design.n;
    let nf = n as f64;
    let mut fit = Fit {
        parent_coefficients: vec![Vec::new(); p],
        lagged_coefficients: vec![Vec::new(); p],
        omega: vec![0.0; p * p],
    };
    // Start from the per-node regressions that ignore the latent-confounder edges.
    for (j, spec) in specs.iter().enumerate() {
        let columns: Vec<&[f64]> = spec
            .parents
            .iter()
            .map(|&k| design.x[k].as_slice())
            .chain(spec.lagged.iter().map(|&k| design.z[k].as_slice()))
            .collect();
        let (coefficients, rss) = least_squares(&columns, &design.x[j])?;
        fit.parent_coefficients[j] = coefficients[..spec.parents.len()].to_vec();
        fit.lagged_coefficients[j] = coefficients[spec.parents.len()..].to_vec();
        fit.omega[j * p + j] = rss / nf;
    }
    let mut previous = f64::NEG_INFINITY;
    let mut current = log_likelihood(design, &specs, &fit)?;
    // Without a latent-confounder edge the regressions above are already the maximum.
    let has_spouses = specs.iter().any(|spec| !spec.spouses.is_empty());
    for _ in 0..if has_spouses { MAX_SWEEPS } else { 0 } {
        if (current - previous).abs() <= FIT_TOLERANCE * (1.0 + current.abs()) {
            break;
        }
        previous = current;
        for v in 0..p {
            update_node(design, &specs, &mut fit, v)?;
        }
        current = log_likelihood(design, &specs, &fit)?;
    }
    Some(-2.0 * current + parameters as f64 * nf.ln())
}

/// Log-likelihood of the design at the current parameters.
fn log_likelihood(design: &Design, specs: &[NodeSpec], fit: &Fit) -> Option<f64> {
    let p = specs.len();
    let (inverse, log_det) = invert_spd(&fit.omega, p)?;
    let columns: Vec<Vec<f64>> = (0..p).map(|j| residuals(design, &specs[j], fit, j)).collect();
    let mut quadratic = 0.0;
    for t in 0..design.n {
        for a in 0..p {
            for b in 0..p {
                quadratic += columns[a][t] * inverse[a * p + b] * columns[b][t];
            }
        }
    }
    let nf = design.n as f64;
    let value = -0.5 * (nf * (p as f64 * (2.0 * std::f64::consts::PI).ln() + log_det) + quadratic);
    value.is_finite().then_some(value)
}

/// One conditional update of node `v` (iterative conditional fitting): regress `X_v` on its
/// parents, its lagged causes and the spouses' residuals adjusted for the rest of the error
/// covariance; the coefficient on a spouse's adjusted residual is the covariance with it.
fn update_node(design: &Design, specs: &[NodeSpec], fit: &mut Fit, v: usize) -> Option<()> {
    let p = specs.len();
    let spec = &specs[v];
    if spec.spouses.is_empty() {
        let columns: Vec<&[f64]> = spec
            .parents
            .iter()
            .map(|&k| design.x[k].as_slice())
            .chain(spec.lagged.iter().map(|&k| design.z[k].as_slice()))
            .collect();
        let (coefficients, rss) = least_squares(&columns, &design.x[v])?;
        fit.parent_coefficients[v] = coefficients[..spec.parents.len()].to_vec();
        fit.lagged_coefficients[v] = coefficients[spec.parents.len()..].to_vec();
        fit.omega[v * p + v] = rss / design.n as f64;
        return Some(());
    }
    let others: Vec<usize> = (0..p).filter(|&k| k != v).collect();
    let q = others.len();
    let mut sub = vec![0.0; q * q];
    for (a, &i) in others.iter().enumerate() {
        for (b, &j) in others.iter().enumerate() {
            sub[a * q + b] = fit.omega[i * p + j];
        }
    }
    let (inverse, _) = invert_spd(&sub, q)?;
    let other_residuals: Vec<Vec<f64>> =
        others.iter().map(|&k| residuals(design, &specs[k], fit, k)).collect();
    let position = |node: usize| others.iter().position(|&k| k == node);
    // u_s = Σ_k ε_k K_{k s}: the spouse's residual adjusted for the remaining error covariance.
    let mut adjusted: Vec<Vec<f64>> = Vec::with_capacity(spec.spouses.len());
    for &spouse in &spec.spouses {
        let s = position(spouse)?;
        let mut u = vec![0.0; design.n];
        for (k, residual) in other_residuals.iter().enumerate() {
            let weight = inverse[k * q + s];
            for (t, value) in u.iter_mut().enumerate() {
                *value += weight * residual[t];
            }
        }
        adjusted.push(u);
    }
    let columns: Vec<&[f64]> = spec
        .parents
        .iter()
        .map(|&k| design.x[k].as_slice())
        .chain(spec.lagged.iter().map(|&k| design.z[k].as_slice()))
        .chain(adjusted.iter().map(Vec::as_slice))
        .collect();
    let (coefficients, rss) = least_squares(&columns, &design.x[v])?;
    let parent_count = spec.parents.len();
    let lagged_count = spec.lagged.len();
    fit.parent_coefficients[v] = coefficients[..parent_count].to_vec();
    fit.lagged_coefficients[v] = coefficients[parent_count..parent_count + lagged_count].to_vec();
    let lambda = &coefficients[parent_count + lagged_count..];
    let mut spouse_term = 0.0;
    for (a, &sa) in spec.spouses.iter().enumerate() {
        for (b, &sb) in spec.spouses.iter().enumerate() {
            spouse_term += lambda[a] * inverse[position(sa)? * q + position(sb)?] * lambda[b];
        }
    }
    fit.omega[v * p + v] = rss / design.n as f64 + spouse_term;
    for (a, &spouse) in spec.spouses.iter().enumerate() {
        fit.omega[v * p + spouse] = lambda[a];
        fit.omega[spouse * p + v] = lambda[a];
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency_mask::set_edge;
    use crate::temporal_admg_atoms::with_lagged_edge;

    struct Normals(u64);

    impl Normals {
        fn unit(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (a, b) = (self.unit(), self.unit());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
    }

    fn variables(n: u32) -> Vec<VariableId> {
        (0..n).map(VariableId::from_raw).collect()
    }

    fn series(columns: &[Vec<f64>]) -> TimeSeriesData {
        let names = ["a", "b", "c", "d"];
        TimeSeriesData::from_f64_columns(
            columns.iter().enumerate().map(|(i, column)| (names[i], column.as_slice())),
            1,
        )
        .unwrap()
    }

    /// `x_t = u_t + e1`, `y_t = u_t + e2` with a hidden common cause `u`: a latent-confounder
    /// pair, no directed edge.
    fn confounded_pair(len: usize, seed: u64) -> TimeSeriesData {
        let mut noise = Normals(seed);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for _ in 0..len {
            let u = noise.normal();
            x.push(u + 0.6 * noise.normal());
            y.push(u + 0.6 * noise.normal());
        }
        series(&[x, y])
    }

    fn contemporaneous(edges: &[(usize, usize)], n: usize) -> AdjacencyMask {
        edges.iter().fold(AdjacencyMask::EMPTY, |m, &(a, b)| set_edge(m, n, a, b, true))
    }

    fn atom(mask: AdjacencyMask) -> TemporalAdmgAtom {
        TemporalAdmgAtom { contemporaneous: mask, lagged_directed: 0, lagged_bidirected: 0 }
    }

    #[test]
    fn iterative_conditional_fitting_reaches_the_saturated_likelihood_of_a_covariance_edge() {
        // Two variables joined by a latent-confounder edge span every covariance: the fit must
        // reach the closed-form saturated log-likelihood -n/2 (p ln 2π + ln|S| + p).
        let data = confounded_pair(4000, 3);
        let design = Design::build(&data, &variables(2), 1).unwrap();
        let both = atom(contemporaneous(&[(0, 1), (1, 0)], 2));
        let n = design.n as f64;
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>() / n;
        let s = [
            dot(&design.x[0], &design.x[0]),
            dot(&design.x[0], &design.x[1]),
            dot(&design.x[1], &design.x[1]),
        ];
        let saturated = -0.5
            * n
            * (2.0 * (2.0 * std::f64::consts::PI).ln() + (s[0] * s[2] - s[1] * s[1]).ln() + 2.0);
        let bic = score(&design, &both, 2, 1).unwrap();
        let parameters = 2.0 + 1.0;
        let expected = -2.0 * saturated + parameters * n.ln();
        assert!((bic - expected).abs() < 1e-6 * expected.abs(), "{bic} vs {expected}");
    }

    #[test]
    fn a_directed_edge_and_a_latent_confounder_edge_between_two_variables_tie() {
        let data = confounded_pair(3000, 5);
        let family = [
            atom(contemporaneous(&[(0, 1)], 2)),
            atom(contemporaneous(&[(1, 0)], 2)),
            atom(contemporaneous(&[(0, 1), (1, 0)], 2)),
            atom(AdjacencyMask::EMPTY),
        ];
        let ctx = ExecutionContext::for_tests(1);
        let posterior =
            TemporalAdmgPosterior::new(1).run(&data, &variables(2), &family, None, &ctx).unwrap();
        // The three dependent templates are one likelihood; the empty template is far worse.
        let w = &posterior.weights;
        assert!((w[0] - w[1]).abs() < 1e-6 && (w[0] - w[2]).abs() < 1e-6, "{w:?}");
        assert!(w[3] < 1e-6, "{w:?}");
        assert!(posterior.is_lagged_temporal_admg());
        assert_eq!(posterior.atom_kind, GraphPosteriorAtomKind::TemporalAdmg);
        assert_eq!(posterior.temporal_admg_atom(2).unwrap(), family[2]);
    }

    #[test]
    fn a_lagged_cause_with_data_support_takes_the_posterior_mass() {
        // y_t = 0.8 x_{t-1} + noise: the template with the lagged edge beats the one without.
        let mut noise = Normals(11);
        let len = 3000;
        let x: Vec<f64> = (0..len).map(|_| noise.normal()).collect();
        let y: Vec<f64> = (0..len)
            .map(|t| if t == 0 { 0.0 } else { 0.8 * x[t - 1] } + 0.5 * noise.normal())
            .collect();
        let data = series(&[x, y]);
        let lagged = with_lagged_edge(0, 2, 1, 0, 1, true).unwrap();
        let with_edge = TemporalAdmgAtom { lagged_directed: lagged, ..atom(AdjacencyMask::EMPTY) };
        let without = atom(AdjacencyMask::EMPTY);
        let ctx = ExecutionContext::for_tests(2);
        let posterior = TemporalAdmgPosterior::new(1)
            .run(&data, &variables(2), &[with_edge, without], None, &ctx)
            .unwrap();
        assert!(posterior.weights[0] > 0.999, "{:?}", posterior.weights);
        // The lagged marginal of x(t-1) -> y(t) is the mass of the first template.
        let marginals = posterior.lagged_edge_marginals.as_ref().unwrap();
        assert!((marginals[1] - posterior.weights[0]).abs() < 1e-12);
    }

    #[test]
    fn a_lagged_latent_confounder_edge_scores_as_the_lagged_directed_edge() {
        let mut noise = Normals(13);
        let len = 2000;
        let x: Vec<f64> = (0..len).map(|_| noise.normal()).collect();
        let y: Vec<f64> = (0..len)
            .map(|t| if t == 0 { 0.0 } else { 0.7 * x[t - 1] } + 0.5 * noise.normal())
            .collect();
        let data = series(&[x, y]);
        let bit = with_lagged_edge(0, 2, 1, 0, 1, true).unwrap();
        let directed = TemporalAdmgAtom { lagged_directed: bit, ..atom(AdjacencyMask::EMPTY) };
        let confounded = TemporalAdmgAtom { lagged_bidirected: bit, ..atom(AdjacencyMask::EMPTY) };
        let ctx = ExecutionContext::for_tests(3);
        let posterior = TemporalAdmgPosterior::new(1)
            .run(&data, &variables(2), &[directed, confounded], None, &ctx)
            .unwrap();
        assert!((posterior.weights[0] - posterior.weights[1]).abs() < 1e-9);
        assert_eq!(posterior.temporal_admg_atom(1).unwrap(), confounded);
    }

    #[test]
    fn a_prior_over_the_family_moves_the_tied_mass() {
        let data = confounded_pair(2000, 7);
        let family =
            [atom(contemporaneous(&[(0, 1)], 2)), atom(contemporaneous(&[(0, 1), (1, 0)], 2))];
        let prior = [0.0, 3.0_f64.ln()];
        let ctx = ExecutionContext::for_tests(4);
        let posterior = TemporalAdmgPosterior::new(1)
            .run(&data, &variables(2), &family, Some(&prior), &ctx)
            .unwrap();
        assert!((posterior.weights[1] / posterior.weights[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn an_empty_family_and_a_mismatched_prior_are_refused() {
        let data = confounded_pair(500, 9);
        let ctx = ExecutionContext::for_tests(5);
        assert!(TemporalAdmgPosterior::new(1).run(&data, &variables(2), &[], None, &ctx).is_err());
        assert!(
            TemporalAdmgPosterior::new(1)
                .run(&data, &variables(2), &[atom(AdjacencyMask::EMPTY)], Some(&[0.0, 1.0]), &ctx)
                .is_err()
        );
    }
}
