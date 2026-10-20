//! Uncertainty for the observational cross-fit sequential contrast.
//!
//! [`crate::sequential_crossfit`] estimates the contrast of two static two-decision regimes
//! from observed data with fitted propensities and iterated conditional means. The score is the
//! sequential augmented inverse-probability score. Identification is assumed, not derived:
//! sequential conditional exchangeability given the measured history, positivity at each
//! decision over the observed histories, consistency, no interference, and no unmeasured
//! confounder of a later decision that the first action affected and the history omits.
//!
//! Two intervals over those scores:
//!
//! * [`sequential_contrast_wald_interval`] is the normal interval of the cross-fit score mean.
//!   It rests on the product rate of the propensity and conditional-mean errors being
//!   `o(n^-1/2)` and on independent subjects. Its repeated-sampling calibration is unmeasured.
//! * [`bayesian_sequential_contrast`] is the Bayesian-bootstrap posterior of the mean score
//!   ([`crate::score_posterior`]). It is a modular posterior: the cross-fit propensity and
//!   conditional-mean fits are held at their fitted values in each score, so the draws carry
//!   the subject-law uncertainty of the orthogonal score and not the sampling uncertainty of the
//!   nuisance fits. The score is orthogonal, so that omission is second order, and the posterior
//!   centres on the same functional as the frequentist estimate. The prior is the vanishing-
//!   concentration Dirichlet-process prior on the subject law; it does not identify the target,
//!   the stated identification assumptions do.
//!
//! The in-sample nested regression pseudo-outcome of the point estimator is retained as is.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use crate::EstimationError;
use crate::score_posterior::{
    ScorePosterior, ScorePosteriorOptions, bayesian_bootstrap_score_posteriors,
};
use crate::sequential_crossfit::SequentialContrast;
use crate::util::count;

/// Normal interval of the cross-fit contrast score mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequentialContrastInterval {
    /// Cross-fit contrast of the active regime minus the control regime.
    pub contrast: f64,
    /// Standard deviation of the contrast scores over `sqrt(n)`.
    pub standard_error: f64,
    /// Central normal interval at `coverage`.
    pub interval: (f64, f64),
    /// Nominal central coverage.
    pub coverage: f64,
    /// Whether repeated-sampling calibration of this interval has been measured. It has not.
    pub calibration_measured: bool,
}

/// Wald interval of a cross-fit sequential contrast.
///
/// # Errors
/// A coverage outside `(0, 1)`, fewer than two scores, a non-finite score, or a contrast with
/// no spread (a constant score has no estimate of its own variance).
pub fn sequential_contrast_wald_interval(
    contrast: &SequentialContrast,
    coverage: f64,
) -> Result<SequentialContrastInterval, EstimationError> {
    if !coverage.is_finite() || coverage <= 0.0 || coverage >= 1.0 {
        return Err(EstimationError::data_msg("sequential interval coverage must be in (0, 1)"));
    }
    let scores = &contrast.contrast_scores;
    if scores.len() < 2 || scores.iter().any(|score| !score.is_finite()) {
        return Err(EstimationError::data_msg(
            "sequential interval needs two finite contrast scores",
        ));
    }
    let n = count(scores.len());
    let centre = scores.iter().sum::<f64>() / n;
    let sum_of_squares =
        scores.iter().map(|score| (score - centre) * (score - centre)).sum::<f64>();
    let standard_error = (sum_of_squares / ((n - 1.0) * n)).sqrt();
    if !standard_error.is_finite() || standard_error <= 0.0 {
        return Err(EstimationError::data_msg(
            "sequential contrast scores have no spread; a constant score has no interval",
        ));
    }
    let z = antecedent_stats::special::normal_ppf(0.5 + coverage / 2.0);
    Ok(SequentialContrastInterval {
        contrast: centre,
        standard_error,
        interval: (centre - z * standard_error, centre + z * standard_error),
        coverage,
        calibration_measured: false,
    })
}

/// Joint modular posterior of the two regime values and their contrast.
///
/// The three posteriors share every draw's weights, so `contrast.draws[k]` is
/// `active.draws[k] - control.draws[k]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialContrastPosterior {
    /// Active-regime value.
    pub active: ScorePosterior,
    /// Control-regime value.
    pub control: ScorePosterior,
    /// Active minus control.
    pub contrast: ScorePosterior,
}

/// Bayesian-bootstrap posterior of the cross-fit sequential contrast scores.
///
/// # Errors
/// The option refusals of [`bayesian_bootstrap_score_posteriors`] and score columns of unequal
/// length.
pub fn bayesian_sequential_contrast(
    contrast: &SequentialContrast,
    options: ScorePosteriorOptions,
    ctx: &ExecutionContext,
) -> Result<SequentialContrastPosterior, EstimationError> {
    let mut posteriors = bayesian_bootstrap_score_posteriors(
        &[&contrast.active_scores, &contrast.control_scores, &contrast.contrast_scores],
        options,
        ctx,
    )?
    .into_iter();
    match (posteriors.next(), posteriors.next(), posteriors.next()) {
        (Some(active), Some(control), Some(contrast)) => {
            Ok(SequentialContrastPosterior { active, control, contrast })
        }
        _ => Err(EstimationError::stats_msg("sequential posterior lost a score column")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(active: &[f64], control: &[f64]) -> SequentialContrast {
        let contrast_scores: Vec<f64> = active.iter().zip(control).map(|(a, c)| a - c).collect();
        let n = count(active.len());
        SequentialContrast {
            contrast: contrast_scores.iter().sum::<f64>() / n,
            active_mean: active.iter().sum::<f64>() / n,
            control_mean: control.iter().sum::<f64>() / n,
            unmeasured_score_scale: 0.0,
            evaluated_rows: active.len() as u64,
            minimum_action_probability: 0.5,
            active_scores: active.to_vec(),
            control_scores: control.to_vec(),
            contrast_scores,
            fits_computed: 0,
            fits_reused: 0,
        }
    }

    #[test]
    fn wald_interval_matches_the_hand_calculation() {
        // Contrast scores 1..8: mean 4.5, sample variance 6, se = sqrt(6 / 8).
        let active = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let control = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result =
            sequential_contrast_wald_interval(&scored(&active, &control), 0.95).expect("interval");
        assert!((result.contrast - 4.5).abs() < 1.0e-12);
        assert!((result.standard_error - 0.75_f64.sqrt()).abs() < 1.0e-12);
        let half = 1.959_963_984_540_054 * 0.75_f64.sqrt();
        assert!((result.interval.0 - (4.5 - half)).abs() < 1.0e-6);
        assert!((result.interval.1 - (4.5 + half)).abs() < 1.0e-6);
        assert!(!result.calibration_measured);
    }

    #[test]
    fn a_constant_score_and_a_bad_coverage_are_refused() {
        let flat = scored(&[3.0, 3.0, 3.0], &[1.0, 1.0, 1.0]);
        assert!(sequential_contrast_wald_interval(&flat, 0.95).is_err());
        let varied = scored(&[3.0, 4.0, 5.0], &[1.0, 1.0, 1.0]);
        assert!(sequential_contrast_wald_interval(&varied, 1.0).is_err());
        assert!(sequential_contrast_wald_interval(&varied, f64::NAN).is_err());
    }

    #[test]
    fn posterior_shares_weights_and_matches_the_dirichlet_moment() {
        let active = [1.0, 6.0, -2.0, 4.0, 3.0, 9.0, 0.5, 2.0];
        let control = [0.0, 2.0, 1.0, -1.0, 2.5, 3.0, 0.5, 1.0];
        let input = scored(&active, &control);
        let ctx = ExecutionContext::for_tests(31);
        let options = ScorePosteriorOptions { draws: 8_000, coverage: 0.95 };
        let result = bayesian_sequential_contrast(&input, options, &ctx).expect("posterior");
        for ((a, c), d) in
            result.active.draws.iter().zip(&result.control.draws).zip(&result.contrast.draws)
        {
            assert!((d - (a - c)).abs() < 1.0e-9);
        }
        let n = count(input.contrast_scores.len());
        let centre = input.contrast;
        let spread = input.contrast_scores.iter().map(|s| s * s).sum::<f64>() / n - centre * centre;
        let expected = (spread / (n + 1.0)).sqrt();
        assert!((result.contrast.standard_deviation - expected).abs() < 0.05 * expected);
        assert!((result.contrast.score_mean - centre).abs() < 1.0e-12);
    }
}
