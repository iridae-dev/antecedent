//! Directed-edge probabilities given a Gaussian table with one partially observed
//! column. The missingness law must be ignorable for the structural model.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_kernels::{
    back_solve_into, chol_solve_into, cholesky_spd_into, forward_solve_into, sample_inv_gamma,
    standard_normal,
};
use antecedent_state::GraphScoreFamily;
use antecedent_stats::special::ln_gamma;

use crate::error::DiscoveryError;
use crate::exact_marginals::exact_dag_marginals;
use crate::graph_posterior::GraphPrior;

/// Numerical estimates under the signed-uniform, zero-intercept, unit-noise model.
#[derive(Clone, Debug)]
pub struct ObservedMarginals {
    /// Directed edge probabilities in row-major variable order.
    pub edge_marginals: Vec<f64>,
    /// Log estimated observed-data graph normalizer under the supplied graph weights.
    pub log_evidence: f64,
    /// Number of completed tables used; zero for fully observed data.
    pub draws_used: usize,
    /// Importance effective sample size, in completed tables; zero for fully observed data.
    pub effective_draws: f64,
    /// Largest estimated Monte Carlo standard error across the edge probabilities.
    pub max_edge_mcse: f64,
}

/// Integrate the last column's absent cells. Each completed table contributes its
/// full graph normalizer divided by the *joint* proposal density of its missing
/// cells. Flat averaging would instead condition on the imputation law. The other
/// columns must be complete and in the order given by `variables`.
///
/// # Errors
/// Refuses unsupported shapes, invalid inputs, unusable proposal fits, cancellation,
/// and unresolved integration. The error diagnostic is empirical, not a guaranteed
/// absolute error bound. Starting at 64 draws, the fixed schedule quadruples the sample until
/// the effective sample size reaches 16 and the maximum edge MCSE is at most 0.01. The last
/// check uses the supplied draw limit. The seed fixes the entire sequence of completions.
pub fn signed_uniform_observed_marginals(
    table: &TabularData,
    variables: &[VariableId],
    prior: &GraphPrior,
    seed: u64,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<ObservedMarginals, DiscoveryError> {
    let n = variables.len();
    let rows = table.row_count();
    if !(2..=6).contains(&n) || rows < 2 || !(32..=65536).contains(&draws) {
        return Err(DiscoveryError::unsupported(
            "observed marginals require 2–6 variables, at least two rows, and 32–65536 draws",
        ));
    }
    let response_id = variables[n - 1];
    let predictors = variables[..n - 1]
        .iter()
        .map(|&id| table.float64_slice(id))
        .collect::<Result<Vec<_>, _>>()?;
    let raw = table.float64_slice(response_id)?;
    if predictors.iter().any(|v| v.iter().any(|x| !x.is_finite()))
        || raw.iter().any(|x| x.is_infinite())
    {
        return Err(DiscoveryError::data_msg(
            "only the last graph column may have missing cells; infinite values are invalid",
        ));
    }
    let response: Vec<_> = raw.iter().map(|&v| (!v.is_nan()).then_some(v)).collect();
    if variables.iter().enumerate().any(|(i, id)| variables[..i].contains(id)) {
        return Err(DiscoveryError::data_msg("observed graph variables must be distinct"));
    }
    prior.validate_for(variables)?;
    prior.constraints.validate()?;
    let missing: Vec<_> = (0..rows).filter(|&i| response[i].is_none()).collect();
    let score = |data: &TabularData| {
        exact_dag_marginals(data, variables, prior, GraphScoreFamily::GaussianSignedUniform, ctx)
    };
    if missing.is_empty() {
        let s = score(table)?;
        return Ok(ObservedMarginals {
            edge_marginals: s.edge_marginals.to_vec(),
            log_evidence: s.log_evidence,
            draws_used: 0,
            effective_draws: 0.0,
            max_edge_mcse: 0.0,
        });
    }
    let Proposal { chol, all_chol, means, scale, shape, log_constant } =
        Proposal::fit(&predictors, &response, &missing)?;
    let p = predictors.len();
    let mut scratch = vec![0.; p];
    let mut rhs = vec![0.; p];
    let mut completed = raw.to_vec();
    let half_dim = 0.5 * missing.len() as f64;
    let mut rng = CausalRng::from_seed(seed);
    let mut scored = Vec::with_capacity(draws);
    let mut delta = vec![0.; p];
    let mut checkpoint = 64.min(draws);
    for _ in 0..draws {
        let sigma = sample_inv_gamma(shape, scale, &mut rng).sqrt();
        for v in &mut scratch {
            *v = standard_normal(&mut rng);
        }
        back_solve_into(&chol, p, &scratch, &mut delta);
        rhs.fill(0.);
        let mut ss = 0.;
        for (&i, &mu) in missing.iter().zip(&means) {
            let r = sigma
                * (standard_normal(&mut rng)
                    + predictors.iter().zip(&delta).map(|(v, b)| v[i] * b).sum::<f64>());
            completed[i] = mu + r;
            ss += r * r;
            for j in 0..p {
                rhs[j] += predictors[j][i] * r;
            }
        }
        // Matrix determinant lemma and Woodbury avoid a missing_count squared matrix.
        forward_solve_into(&all_chol, p, &rhs, &mut scratch);
        let quad = ss - scratch.iter().map(|v| v * v).sum::<f64>();
        if !quad.is_finite() || quad < 0.0 {
            return Err(DiscoveryError::stats_msg(
                "non-finite or negative graph completion quadratic",
            ));
        }
        let log_q = log_constant - (shape + half_dim) * (quad / (2.0 * scale)).ln_1p();
        let s = score(&table.with_replaced_float(response_id, completed.as_slice().into())?)?;
        scored.push((s.log_evidence - log_q, s.edge_marginals));
        if scored.len() == checkpoint {
            let summary = summarize(&scored, n);
            if summary.effective_draws >= 16.0
                && summary.max_edge_mcse <= 0.01
                && summary.log_evidence.is_finite()
            {
                return Ok(summary);
            }
            if checkpoint == draws {
                return Err(DiscoveryError::stats_msg(format!(
                    "observed-data integration unresolved after {draws} draws (ESS {:.2}, maximum edge MCSE {:.5})",
                    summary.effective_draws, summary.max_edge_mcse
                )));
            }
            checkpoint = (checkpoint * 4).min(draws);
        }
    }
    Err(DiscoveryError::stats_msg("empty observed-data integration"))
}

struct Proposal {
    chol: Vec<f64>,
    all_chol: Vec<f64>,
    means: Vec<f64>,
    scale: f64,
    shape: f64,
    log_constant: f64,
}

impl Proposal {
    fn fit(
        predictors: &[&[f64]],
        response: &[Option<f64>],
        missing: &[usize],
    ) -> Result<Self, DiscoveryError> {
        // Draw a whole completion from the posterior predictive of a *proposal*
        // regression: beta | variance ~ N(0, variance I), variance ~ IG(1,1).
        // Uncertainty in the shared coefficients correlates the missing cells.
        // The proposal cancels in the importance ratio and does not specify a graph.
        let p = predictors.len();
        let mut precision = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        let mut observed = 0_usize;
        for i in 0..p {
            precision[i * p + i] = 1.0;
        }
        for (row, y) in response.iter().enumerate() {
            if let Some(y) = y {
                observed += 1;
                for i in 0..p {
                    rhs[i] += predictors[i][row] * y;
                    for j in 0..p {
                        precision[i * p + j] += predictors[i][row] * predictors[j][row];
                    }
                }
            }
        }
        let factor = |matrix: &[f64]| {
            let mut l = vec![0.0; p * p];
            cholesky_spd_into(matrix, p, &mut l).map_err(|_| {
                DiscoveryError::stats_msg("ill-conditioned graph completion proposal")
            })?;
            Ok::<_, DiscoveryError>(l)
        };
        let chol = factor(&precision)?;
        let mut beta = vec![0.0; p];
        let mut scratch = vec![0.0; p];
        chol_solve_into(&chol, p, &rhs, &mut scratch, &mut beta);
        let mean = |row: usize| predictors.iter().zip(&beta).map(|(v, b)| v[row] * b).sum::<f64>();
        let scale = 1.0
            + 0.5
                * (beta.iter().map(|x| x * x).sum::<f64>()
                    + response
                        .iter()
                        .enumerate()
                        .filter_map(|(r, y)| y.map(|v| (v - mean(r)).powi(2)))
                        .sum::<f64>());
        let shape = 1.0 + 0.5 * observed as f64;
        for &row in missing {
            for i in 0..p {
                for j in 0..p {
                    precision[i * p + j] += predictors[i][row] * predictors[j][row];
                }
            }
        }
        let all_chol = factor(&precision)?;
        let half_log_det =
            (0..p).map(|i| (all_chol[i * p + i] / chol[i * p + i]).ln()).sum::<f64>();
        let half_dim = 0.5 * missing.len() as f64;
        let log_constant = ln_gamma(shape + half_dim)
            - ln_gamma(shape)
            - half_dim * (std::f64::consts::TAU * scale).ln()
            - half_log_det;
        let means: Vec<_> = missing.iter().map(|&row| mean(row)).collect();
        Ok(Self { chol, all_chol, means, scale, shape, log_constant })
    }
}

fn summarize(scored: &[(f64, std::sync::Arc<[f64]>)], n: usize) -> ObservedMarginals {
    let draws = scored.len();
    let peak = scored.iter().map(|(w, _)| *w).fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = scored.iter().map(|(w, _)| (w - peak).exp()).collect();
    let total = weights.iter().sum::<f64>();
    let mut edge = vec![0.0; n * n];
    for ((_, m), w) in scored.iter().zip(&weights) {
        for (out, x) in edge.iter_mut().zip(m.iter()) {
            *out += w * x / total;
        }
    }
    let ess = total * total / weights.iter().map(|w| w * w).sum::<f64>();
    let mcse = (0..edge.len())
        .map(|j| {
            scored
                .iter()
                .zip(&weights)
                .map(|((_, m), w)| (w * (m[j] - edge[j]) / total).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    ObservedMarginals {
        edge_marginals: edge,
        log_evidence: peak + (total / draws as f64).ln(),
        draws_used: draws,
        effective_draws: ess,
        max_edge_mcse: mcse,
    }
}
