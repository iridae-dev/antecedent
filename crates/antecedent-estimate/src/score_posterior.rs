//! Bayesian-bootstrap posterior of the mean of per-subject scores.
//!
//! The row law is given a Dirichlet-process prior in the limit of vanishing concentration,
//! which makes the posterior weights on the observed rows Dirichlet(1, …, 1) (Rubin 1981).
//! A score `ψ(Oᵢ)` that is a known function of one subject's record has a posterior mean
//! functional `Σᵢ wᵢ ψ(Oᵢ)` under those weights, and every output here is that functional's
//! draws. Several score columns share one weight vector per draw, so a contrast of two scores
//! is the contrast of their draws.
//!
//! What the posterior is a posterior of: the population mean of the score *as a fixed function
//! of the record*. That mean is the causal target when the score is unbiased for it. A score
//! built from a known randomization is unbiased whatever frozen outcome model it carries, so
//! its posterior is a posterior of the target. A score that carries fitted propensities is
//! unbiased only up to the second-order product of the nuisance errors; its draws condition on
//! the cross-fit nuisance fits and do not carry their sampling uncertainty. Callers state
//! which of the two they are in; this module cannot tell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use crate::EstimationError;
use crate::util::{count, dot, mean, sample_std};

/// Key of the weight stream: one stream per draw, shared by every score column.
const WEIGHT_STREAM: u64 = 0x5C0E_B007_0000_0000;

/// Draw count and central coverage of a score posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScorePosteriorOptions {
    /// Posterior draws; at least two.
    pub draws: usize,
    /// Central posterior interval coverage in `(0, 1)`.
    pub coverage: f64,
}

impl Default for ScorePosteriorOptions {
    fn default() -> Self {
        Self { draws: 2000, coverage: 0.95 }
    }
}

/// Posterior of the mean of one score column.
#[derive(Clone, Debug, PartialEq)]
pub struct ScorePosterior {
    /// Plain mean of the observed scores; the posterior centre in the large-draw limit.
    pub score_mean: f64,
    /// Mean of the posterior draws.
    pub posterior_mean: f64,
    /// Standard deviation of the posterior draws.
    pub standard_deviation: f64,
    /// Equal-tailed central interval of the draws.
    pub interval: (f64, f64),
    /// One draw per shared weight vector.
    pub draws: Vec<f64>,
}

fn refused(message: &'static str) -> EstimationError {
    EstimationError::data_msg(message)
}

/// Posterior draws of the mean of each score column.
///
/// All columns are indexed by the same subjects and share the weight vector of each draw.
///
/// # Errors
/// No column, fewer than two subjects, columns of unequal length, a non-finite score, fewer
/// than two draws, a coverage outside `(0, 1)`, or cancellation.
pub fn bayesian_bootstrap_score_posteriors(
    columns: &[&[f64]],
    options: ScorePosteriorOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<ScorePosterior>, EstimationError> {
    let Some(first) = columns.first() else {
        return Err(refused("score posterior requires at least one score column"));
    };
    let n = first.len();
    if n < 2 || columns.iter().any(|column| column.len() != n) {
        return Err(refused("score posterior requires equal columns of at least two subjects"));
    }
    if columns.iter().any(|column| column.iter().any(|score| !score.is_finite())) {
        return Err(refused("score posterior requires finite scores"));
    }
    if options.draws < 2
        || !options.coverage.is_finite()
        || options.coverage <= 0.0
        || options.coverage >= 1.0
    {
        return Err(refused("score posterior requires two draws and a coverage in (0, 1)"));
    }
    let _admitted = crate::util::admit_array::<f64>(ctx, options.draws, columns.len())?;
    let mut stores: Vec<Vec<f64>> =
        columns.iter().map(|_| Vec::with_capacity(options.draws)).collect();
    for draw in 0..options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported("score posterior cancelled"));
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Bayesian, WEIGHT_STREAM | draw as u64);
        // Rescaled to mean one, so `dot(w, s) / n` is the weighted score mean.
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, n, true);
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(EstimationError::stats_msg("invalid Bayesian-bootstrap row weights"));
        }
        for (store, column) in stores.iter_mut().zip(columns) {
            store.push(dot(&weights, column) / count(n));
        }
    }
    Ok(stores
        .into_iter()
        .zip(columns)
        .map(|(draws, column)| summarize(draws, mean(column), options.coverage))
        .collect())
}

pub(crate) fn summarize(draws: Vec<f64>, score_mean: f64, coverage: f64) -> ScorePosterior {
    let mut sorted = draws.clone();
    sorted.sort_by(f64::total_cmp);
    let interval = antecedent_stats::equal_tail_interval_sorted(
        &sorted,
        coverage,
        antecedent_stats::QuantileRule::ExchangeableRank,
    );
    ScorePosterior {
        score_mean,
        posterior_mean: mean(&draws),
        standard_deviation: sample_std(&draws),
        interval,
        draws,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(draws: usize) -> ScorePosteriorOptions {
        ScorePosteriorOptions { draws, coverage: 0.95 }
    }

    fn posterior(columns: &[&[f64]], draws: usize, seed: u64) -> Vec<ScorePosterior> {
        let ctx = ExecutionContext::for_tests(seed);
        bayesian_bootstrap_score_posteriors(columns, options(draws), &ctx).expect("posterior")
    }

    #[test]
    fn two_subjects_give_the_uniform_posterior() {
        // Dirichlet(1, 1) on scores (0, 1): the weighted mean is the second weight, Uniform(0, 1).
        let scores = [0.0, 1.0];
        let result = &posterior(&[&scores], 20_000, 7)[0];
        assert!((result.score_mean - 0.5).abs() < 1.0e-12);
        assert!((result.posterior_mean - 0.5).abs() < 0.01);
        assert!((result.standard_deviation - (1.0_f64 / 12.0).sqrt()).abs() < 0.01);
        assert!((result.interval.0 - 0.025).abs() < 0.01, "{:?}", result.interval);
        assert!((result.interval.1 - 0.975).abs() < 0.01, "{:?}", result.interval);
    }

    #[test]
    fn posterior_variance_matches_the_dirichlet_moment() {
        // Var(sum w s) = (1/n sum s^2 - mean^2) / (n + 1) for w ~ Dirichlet(1, ..., 1).
        let scores = [1.0, -2.0, 0.5, 3.0, 2.5, -1.0, 0.0, 4.0, 1.5, -0.5];
        let n = count(scores.len());
        let centre = scores.iter().sum::<f64>() / n;
        let spread = scores.iter().map(|s| s * s).sum::<f64>() / n - centre * centre;
        let expected_sd = (spread / (n + 1.0)).sqrt();
        let result = &posterior(&[&scores], 8_000, 11)[0];
        assert!((result.score_mean - centre).abs() < 1.0e-12);
        let relative = (result.standard_deviation - expected_sd).abs() / expected_sd;
        assert!(relative < 0.05, "sd {} expected {expected_sd}", result.standard_deviation);
        assert!((result.posterior_mean - centre).abs() < 5.0 * expected_sd / 8_000.0_f64.sqrt());
    }

    #[test]
    fn columns_share_each_draws_weights() {
        let first = [0.5, 1.5, -1.0, 2.0, 0.25, 3.0];
        let second: Vec<f64> = first.iter().map(|s| 2.0 * s + 1.0).collect();
        let result = posterior(&[&first, &second], 200, 3);
        for (a, b) in result[0].draws.iter().zip(&result[1].draws) {
            assert!((b - (2.0 * a + 1.0)).abs() < 1.0e-9);
        }
    }

    #[test]
    fn constant_scores_have_a_degenerate_posterior() {
        let scores = [2.5; 6];
        let result = &posterior(&[&scores], 50, 5)[0];
        assert!(result.draws.iter().all(|draw| (draw - 2.5).abs() < 1.0e-9));
    }

    #[test]
    fn the_same_seed_repeats_the_draws() {
        let scores = [0.0, 1.0, 4.0, 2.0];
        assert_eq!(posterior(&[&scores], 30, 9), posterior(&[&scores], 30, 9));
    }

    #[test]
    fn invalid_input_is_refused() {
        let ctx = ExecutionContext::for_tests(1);
        let good = [1.0, 2.0, 3.0];
        let short = [1.0, 2.0];
        let bad = [1.0, f64::NAN, 3.0];
        let run = |columns: &[&[f64]], options: ScorePosteriorOptions| {
            bayesian_bootstrap_score_posteriors(columns, options, &ctx)
        };
        assert!(run(&[], options(10)).is_err());
        assert!(run(&[&[1.0]], options(10)).is_err());
        assert!(run(&[&good, &short], options(10)).is_err());
        assert!(run(&[&bad], options(10)).is_err());
        assert!(run(&[&good], options(1)).is_err());
        assert!(run(&[&good], ScorePosteriorOptions { draws: 10, coverage: 1.0 }).is_err());
        assert!(run(&[&good], ScorePosteriorOptions { draws: 10, coverage: f64::NAN }).is_err());
    }
}
