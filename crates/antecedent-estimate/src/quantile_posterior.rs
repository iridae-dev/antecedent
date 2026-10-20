//! Bayesian posterior for the quantiles of randomized potential outcomes.
//!
//! A quantile of `Y(a)` is a functional of the whole outcome law, so a posterior for it needs
//! a model of that law and not only of a mean response. The likelihood here is stated: within
//! each randomized arm the outcomes fall into the `K + 1` bins cut by the declared threshold
//! grid, independently, with one multinomial probability vector per arm, and that vector has
//! a symmetric `Dirichlet` prior. The posterior is `Dirichlet(prior + counts)` in closed
//! form. The distribution function at the grid is the cumulative sum of a Dirichlet draw,
//! and the quantile is read by the same piecewise-linear inversion of the grid distribution
//! function that the frequentist cell uses ([`crate::quantile::invert_cdf_quantile`]), so the
//! two answer the same target.
//!
//! The grid is part of the model: like the frequentist cell, inference conditions on it, and
//! the within-bin law is the linear interpolation of the distribution function. The arms are
//! independent under randomization, so the posterior of the quantile treatment effect is the
//! difference of independent arm posteriors, taken draw by draw. A draw for which the
//! `tau`-quantile falls outside the grid distribution function (or where the local density is
//! too small to invert) is unresolved and is counted. More than one percent unresolved
//! refuses the posterior: the grid does not support that quantile.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;
use antecedent_kernels::sample_gamma;
use antecedent_stats::{QuantileRule, equal_tail_interval_sorted};

use crate::error::EstimationError;
use crate::quantile::invert_cdf_quantile;

/// Fewest posterior draws accepted.
pub const MIN_QUANTILE_POSTERIOR_DRAWS: usize = 200;
/// Most posterior draws one request may ask for.
pub const MAX_QUANTILE_POSTERIOR_DRAWS: usize = 200_000;
/// Largest share of unresolved draws a posterior may carry before it is refused.
pub const MAX_UNRESOLVED_QUANTILE_SHARE: f64 = 0.01;

/// Posterior of the `tau`-quantile of each arm and of the treated-minus-control difference.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantilePosterior {
    /// Quantile level.
    pub tau: f64,
    /// Closed-form posterior mean of the control grid distribution function.
    pub control_cdf_mean: Vec<f64>,
    /// Closed-form posterior mean of the treated grid distribution function.
    pub treated_cdf_mean: Vec<f64>,
    /// Resolved posterior draws of the control quantile, aligned with the other arrays.
    pub control_draws: Vec<f64>,
    /// Resolved posterior draws of the treated quantile.
    pub treated_draws: Vec<f64>,
    /// Posterior mean of the control quantile over resolved draws.
    pub control_quantile_mean: f64,
    /// Posterior mean of the treated quantile over resolved draws.
    pub treated_quantile_mean: f64,
    /// Posterior mean of the treated-minus-control quantile difference.
    pub difference_mean: f64,
    /// Equal-tailed 95% credible interval of the control quantile.
    pub control_interval_95: [f64; 2],
    /// Equal-tailed 95% credible interval of the treated quantile.
    pub treated_interval_95: [f64; 2],
    /// Equal-tailed 95% credible interval of the difference.
    pub difference_interval_95: [f64; 2],
    /// Posterior probability that the treated quantile exceeds the control quantile.
    pub probability_treated_quantile_higher: f64,
    /// Draws requested.
    pub draws_requested: usize,
    /// Draws whose quantile was resolved in both arms.
    pub draws_resolved: usize,
}

/// Posterior of the quantile treatment effect of a randomized two-arm trial.
///
/// `thresholds` is a strictly increasing grid of at least two cut points; `bin_concentration`
/// is the symmetric Dirichlet concentration given to every one of the `K + 1` bins.
///
/// # Errors
///
/// Misaligned or non-finite rows, a missing arm, an invalid grid, level, concentration or
/// draw count, or a grid that leaves more than one percent of the draws unresolved.
pub fn randomized_quantile_posterior(
    outcome: &[f64],
    treated: &[bool],
    thresholds: &[f64],
    bin_concentration: f64,
    tau: f64,
    draws: usize,
    seed: u64,
) -> Result<QuantilePosterior, EstimationError> {
    validate(outcome, treated, thresholds, bin_concentration, tau, draws)?;
    let control_counts = bin_counts(outcome, treated, false, thresholds);
    let treated_counts = bin_counts(outcome, treated, true, thresholds);
    let control_alpha: Vec<f64> = control_counts.iter().map(|c| bin_concentration + c).collect();
    let treated_alpha: Vec<f64> = treated_counts.iter().map(|c| bin_concentration + c).collect();
    let mut rng = CausalRng::from_seed(seed);
    let mut control_draws = Vec::with_capacity(draws);
    let mut treated_draws = Vec::with_capacity(draws);
    for _ in 0..draws {
        let control = draw_quantile(&control_alpha, thresholds, tau, &mut rng);
        let active = draw_quantile(&treated_alpha, thresholds, tau, &mut rng);
        if let (Some(control), Some(active)) = (control, active) {
            control_draws.push(control);
            treated_draws.push(active);
        }
    }
    let resolved = control_draws.len();
    #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^52")]
    let unresolved_share = (draws - resolved) as f64 / draws as f64;
    if unresolved_share > MAX_UNRESOLVED_QUANTILE_SHARE {
        return Err(EstimationError::unsupported(
            "quantile posterior refused: the threshold grid does not bracket the quantile in \
             enough posterior draws; widen the grid or move the level inside its range",
        ));
    }
    let differences: Vec<f64> =
        treated_draws.iter().zip(&control_draws).map(|(a, c)| a - c).collect();
    #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^52")]
    let count = resolved as f64;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / count;
    #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^52")]
    let higher = differences.iter().filter(|value| **value > 0.0).count() as f64;
    Ok(QuantilePosterior {
        tau,
        control_cdf_mean: cdf_mean(&control_alpha),
        treated_cdf_mean: cdf_mean(&treated_alpha),
        control_quantile_mean: mean(&control_draws),
        treated_quantile_mean: mean(&treated_draws),
        difference_mean: mean(&differences),
        control_interval_95: interval_95(&control_draws),
        treated_interval_95: interval_95(&treated_draws),
        difference_interval_95: interval_95(&differences),
        probability_treated_quantile_higher: higher / count,
        control_draws,
        treated_draws,
        draws_requested: draws,
        draws_resolved: resolved,
    })
}

fn interval_95(values: &[f64]) -> [f64; 2] {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lower, upper) = equal_tail_interval_sorted(&sorted, 0.95, QuantileRule::ExchangeableRank);
    [lower, upper]
}

fn validate(
    outcome: &[f64],
    treated: &[bool],
    thresholds: &[f64],
    bin_concentration: f64,
    tau: f64,
    draws: usize,
) -> Result<(), EstimationError> {
    if outcome.is_empty()
        || outcome.len() != treated.len()
        || outcome.iter().any(|value| !value.is_finite())
    {
        return Err(EstimationError::data_msg(
            "outcome and arm must be aligned, nonempty, and finite",
        ));
    }
    if treated.iter().all(|arm| *arm) || treated.iter().all(|arm| !*arm) {
        return Err(EstimationError::data_msg("both randomized arms require observed units"));
    }
    if thresholds.len() < 2
        || thresholds.iter().any(|value| !value.is_finite())
        || thresholds.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(EstimationError::data_msg(
            "the threshold grid needs at least two finite, strictly increasing cut points",
        ));
    }
    if !bin_concentration.is_finite() || bin_concentration <= 0.0 {
        return Err(EstimationError::data_msg(
            "the Dirichlet bin concentration must be finite and positive",
        ));
    }
    if !(tau > 0.0 && tau < 1.0) {
        return Err(EstimationError::unsupported("quantile level must lie in (0, 1)"));
    }
    if !(MIN_QUANTILE_POSTERIOR_DRAWS..=MAX_QUANTILE_POSTERIOR_DRAWS).contains(&draws) {
        return Err(EstimationError::data_msg("quantile posterior draws must be 200 to 200000"));
    }
    Ok(())
}

/// Counts in the bins `(-inf, t_0], (t_0, t_1], .., (t_{K-1}, inf)` for one arm.
fn bin_counts(outcome: &[f64], treated: &[bool], arm: bool, thresholds: &[f64]) -> Vec<f64> {
    let mut counts = vec![0.0; thresholds.len() + 1];
    for (value, member) in outcome.iter().zip(treated) {
        if *member == arm {
            let bin = thresholds.partition_point(|cut| cut < value);
            counts[bin] += 1.0;
        }
    }
    counts
}

/// Posterior mean of the grid distribution function: cumulative Dirichlet mean.
fn cdf_mean(alpha: &[f64]) -> Vec<f64> {
    let total: f64 = alpha.iter().sum();
    let mut running = 0.0;
    alpha[..alpha.len() - 1]
        .iter()
        .map(|a| {
            running += a;
            running / total
        })
        .collect()
}

/// One draw of the `tau`-quantile, or `None` when the grid cannot resolve it.
fn draw_quantile(alpha: &[f64], thresholds: &[f64], tau: f64, rng: &mut CausalRng) -> Option<f64> {
    let gammas: Vec<f64> = alpha.iter().map(|a| sample_gamma(*a, 1.0, rng)).collect();
    let total: f64 = gammas.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mut running = 0.0;
    let mut cdf: Vec<f64> = gammas[..gammas.len() - 1]
        .iter()
        .map(|g| {
            running += g / total;
            running.min(1.0)
        })
        .collect();
    // Cumulative sums of positive terms are already nondecreasing; the clamp only guards
    // against a rounding overshoot of one.
    for value in &mut cdf {
        *value = value.clamp(0.0, 1.0);
    }
    let influence = vec![vec![0.0]; thresholds.len()];
    invert_cdf_quantile(thresholds, &cdf, &influence, tau).ok().map(|(quantile, _, _)| quantile)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<bool>) {
        let control = [0.4, 0.8, 1.1, 1.3, 1.7, 2.2, 2.4, 2.9, 3.3, 3.6, 4.1, 4.6];
        let active = [0.9, 1.4, 1.8, 2.1, 2.6, 2.8, 3.1, 3.4, 3.9, 4.3, 4.4, 4.8];
        let mut outcome = control.to_vec();
        outcome.extend(active);
        let mut treated = vec![false; control.len()];
        treated.extend(vec![true; active.len()]);
        (outcome, treated)
    }

    const GRID: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    #[test]
    fn posterior_cdf_mean_and_quantile_probabilities_match_the_dirichlet_reference() {
        // The reference is the Dirichlet marginal: F(t_j) is Beta(A_j, total - A_j), and
        // under the piecewise-linear inversion Q <= t_j exactly when F(t_j) >= tau.
        // Counts, cumulative prior-plus-count masses and the Beta tail probabilities were
        // computed independently with scipy.
        let (outcome, treated) = data();
        let posterior =
            randomized_quantile_posterior(&outcome, &treated, &GRID, 0.5, 0.5, 40_000, 9).unwrap();
        // Control counts per bin (-inf,0],(0,1],(1,2],(2,3],(3,4],(4,5],(5,inf): 0,2,3,3,2,2,0.
        // Prior 0.5 per bin gives total 15.5 and cumulative masses 0.5, 3, 6.5, 10, 12.5, 15.
        let expected = [0.5, 3.0, 6.5, 10.0, 12.5, 15.0].map(|a| a / 15.5);
        for (got, want) in posterior.control_cdf_mean.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
        let share_below = |draws: &[f64], cut: f64| {
            draws.iter().filter(|value| **value <= cut).count() as f64 / draws.len() as f64
        };
        // Control: F(2) ~ Beta(6.5, 9), F(3) ~ Beta(10, 5.5). Treated: F(3) ~ Beta(8, 7.5).
        let tolerance = 0.012;
        assert!(
            (share_below(&posterior.control_draws, 2.0) - BETA_TAIL_CONTROL_2).abs() < tolerance
        );
        assert!(
            (share_below(&posterior.control_draws, 3.0) - BETA_TAIL_CONTROL_3).abs() < tolerance
        );
        assert!(
            (share_below(&posterior.treated_draws, 3.0) - BETA_TAIL_TREATED_3).abs() < tolerance
        );
        assert_eq!(posterior.draws_requested, 40_000);
        assert_eq!(posterior.draws_resolved, posterior.control_draws.len());
        assert!(posterior.difference_interval_95[0] < posterior.difference_mean);
        assert!(posterior.difference_mean < posterior.difference_interval_95[1]);
    }

    // Values produced by `scipy.stats.beta.sf(0.5, a, b)` for the parameters in the test.
    const BETA_TAIL_CONTROL_2: f64 = 0.256_553_5;
    const BETA_TAIL_CONTROL_3: f64 = 0.881_777_0;
    const BETA_TAIL_TREATED_3: f64 = 0.551_911_9;

    #[test]
    fn vague_prior_median_tracks_the_empirical_piecewise_linear_quantile() {
        // With a tiny concentration the posterior mass sits on the empirical bin masses, so
        // the posterior median of the control quantile is the inversion of the empirical
        // grid distribution function, which the frequentist inversion computes directly.
        let (outcome, treated) = data();
        let posterior =
            randomized_quantile_posterior(&outcome, &treated, &GRID, 0.01, 0.5, 20_000, 4).unwrap();
        let empirical: Vec<f64> = GRID
            .iter()
            .map(|cut| {
                let arm: Vec<f64> = outcome
                    .iter()
                    .zip(&treated)
                    .filter_map(|(value, arm)| (!*arm).then_some(*value))
                    .collect();
                arm.iter().filter(|value| **value <= *cut).count() as f64 / arm.len() as f64
            })
            .collect();
        let (reference, _, _) =
            invert_cdf_quantile(&GRID, &empirical, &vec![vec![0.0]; GRID.len()], 0.5).unwrap();
        let mut sorted = posterior.control_draws.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        assert!((median - reference).abs() < 0.2, "{median} vs {reference}");
    }

    #[test]
    fn a_level_the_grid_cannot_bracket_and_invalid_requests_are_refused() {
        let (outcome, treated) = data();
        let tail = randomized_quantile_posterior(&outcome, &treated, &GRID, 0.5, 0.999, 400, 1)
            .unwrap_err();
        assert!(tail.to_string().contains("threshold grid"));
        let level = randomized_quantile_posterior(&outcome, &treated, &GRID, 0.5, 1.0, 400, 1);
        assert!(level.is_err());
        let prior = randomized_quantile_posterior(&outcome, &treated, &GRID, 0.0, 0.5, 400, 1);
        assert!(prior.is_err());
        let unsorted =
            randomized_quantile_posterior(&outcome, &treated, &[1.0, 0.0], 0.5, 0.5, 400, 1);
        assert!(unsorted.is_err());
        let one_arm = vec![false; outcome.len()];
        assert!(
            randomized_quantile_posterior(&outcome, &one_arm, &GRID, 0.5, 0.5, 400, 1).is_err()
        );
    }
}
