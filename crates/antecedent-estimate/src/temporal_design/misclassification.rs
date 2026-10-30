//! Excursion contrasts of a misclassified binary outcome.
//!
//! The recorded outcome `Y*` is a misclassified version of a binary outcome `Y` with
//! sensitivity `se = P(Y* = 1 | Y = 1)` and specificity `sp = P(Y* = 0 | Y = 0)`, the same in
//! every arm, unit and period (non-differential error), learned on an independent validation
//! sample of `(true positive, false negative)` and `(true negative, false positive)` counts.
//! Then `E[Y* | a] = (se + sp - 1) E[Y | a] + (1 - sp)`, so with `J = se + sp - 1 > 0`:
//!
//! * an arm mean of the true outcome is `(mu* - (1 - sp)) / J`;
//! * a contrast (coefficients summing to zero) is `theta* / J`.
//!
//! The frequentist covariance adds the delta-method contribution of the validation counts,
//! independent of the panel, to the panel covariance scaled by `1 / J^2`. The posterior draws
//! `se` and `sp` from their Beta posteriors under uniform priors, one pair per draw, and applies
//! the correction to that draw. A Youden index `J` near zero leaves the true outcome
//! unidentified, and the correction refuses it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use super::{PosteriorSummary, SamplingSummary, invalid, summarize_draws};
use crate::error::EstimationError;
use crate::util::refuse;

/// Smallest Youden index `se + sp - 1` the correction accepts.
pub const MIN_YOUDEN_INDEX: f64 = 0.1;
/// Stream key of the posterior sensitivity and specificity draws.
const VALIDATION_STREAM: u64 = 0x7E5D_5E5B_0000_0000;

/// Validation-sample counts of a misclassified binary outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidationCounts {
    /// Recorded positive among true positives.
    pub true_positive: u64,
    /// Recorded negative among true positives.
    pub false_negative: u64,
    /// Recorded negative among true negatives.
    pub true_negative: u64,
    /// Recorded positive among true negatives.
    pub false_positive: u64,
}

impl ValidationCounts {
    fn sensitivity(&self) -> (f64, f64) {
        let total = self.true_positive as f64 + self.false_negative as f64;
        let value = self.true_positive as f64 / total;
        (value, value * (1.0 - value) / total)
    }

    fn specificity(&self) -> (f64, f64) {
        let total = self.true_negative as f64 + self.false_positive as f64;
        let value = self.true_negative as f64 / total;
        (value, value * (1.0 - value) / total)
    }

    fn validate(&self) -> Result<(), EstimationError> {
        if self.true_positive == 0 && self.false_negative == 0
            || self.true_negative == 0 && self.false_positive == 0
        {
            return Err(invalid("the validation sample needs true positives and true negatives"));
        }
        Ok(())
    }
}

/// Whether a coordinate name is an arm mean (rather than a contrast).
#[must_use]
pub fn is_mean_coordinate(name: &str) -> bool {
    name.starts_with("mean:")
        || matches!(name, "treated_mean" | "control_mean" | "active_mean" | "reference_mean")
}

fn youden_refusal(youden: f64) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "temporal_design.misclassification_unidentified",
        &format!(
            "the validation sample's Youden index se + sp - 1 = {youden} is below {MIN_YOUDEN_INDEX}, \
             so the recorded outcome carries too little information about the true one. Next \
             attempt: a more accurate outcome measurement or a larger validation sample"
        ),
    )
}

/// Correct a frequentist summary of a binary recorded outcome.
///
/// The band of the recorded-outcome summary is dropped: its multiplier process does not carry
/// the validation sample. Holm-adjusted p-values are recomputed from the corrected standard
/// errors when `family` has more than one coordinate.
///
/// # Errors
/// An empty validation margin or a Youden index below [`MIN_YOUDEN_INDEX`].
pub fn correct_sampling(
    summary: &mut SamplingSummary,
    coordinates: &[String],
    family: &[usize],
    counts: ValidationCounts,
) -> Result<f64, EstimationError> {
    counts.validate()?;
    let (se, var_se) = counts.sensitivity();
    let (sp, var_sp) = counts.specificity();
    let youden = se + sp - 1.0;
    if youden.is_nan() || youden < MIN_YOUDEN_INDEX {
        return Err(youden_refusal(youden));
    }
    let k = coordinates.len();
    let offset: Vec<f64> =
        coordinates.iter().map(|name| f64::from(u8::from(is_mean_coordinate(name)))).collect();
    let corrected: Vec<f64> =
        (0..k).map(|j| (summary.estimate[j] - offset[j] * (1.0 - sp)) / youden).collect();
    let mut covariance = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            let panel = summary.covariance[a * k + b] / (youden * youden);
            let sensitivity = corrected[a] * corrected[b] / (youden * youden) * var_se;
            let specificity = (offset[a] - corrected[a]) * (offset[b] - corrected[b])
                / (youden * youden)
                * var_sp;
            covariance[a * k + b] = panel + sensitivity + specificity;
        }
    }
    let standard_error: Vec<f64> = (0..k).map(|j| covariance[j * k + j].max(0.0).sqrt()).collect();
    let z = antecedent_stats::NORMAL_Q975;
    summary.lower = corrected.iter().zip(&standard_error).map(|(e, s)| e - z * s).collect();
    summary.upper = corrected.iter().zip(&standard_error).map(|(e, s)| e + z * s).collect();
    summary.estimate = corrected;
    summary.standard_error = standard_error;
    summary.covariance = covariance;
    summary.band = None;
    let holm = (family.len() > 1).then(|| super::holm_adjusted(summary, family));
    summary.holm_adjusted_p = holm;
    summary.method = "cluster_robust_ratio_normal_with_validation_delta";
    Ok(youden)
}

/// Correct posterior draws of a binary recorded outcome, one sensitivity and specificity draw
/// per posterior draw.
///
/// # Errors
/// An empty validation margin, a Youden index below [`MIN_YOUDEN_INDEX`] at the validation
/// estimate, a non-positive drawn index, or cancellation.
pub fn correct_posterior(
    posterior: PosteriorSummary,
    coordinates: &[String],
    counts: ValidationCounts,
    ctx: &ExecutionContext,
) -> Result<PosteriorSummary, EstimationError> {
    counts.validate()?;
    let (se, _) = counts.sensitivity();
    let (sp, _) = counts.specificity();
    if se + sp - 1.0 < MIN_YOUDEN_INDEX {
        return Err(youden_refusal(se + sp - 1.0));
    }
    let method = posterior.method;
    let mut draws = posterior.draws;
    let n = draws.first().map_or(0, Vec::len);
    for s in 0..n {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Bayesian, VALIDATION_STREAM | s as u64);
        let beta = |a: f64, b: f64, rng: &mut antecedent_core::CausalRng| {
            let x = antecedent_kernels::sample_gamma(a, 1.0, rng);
            let y = antecedent_kernels::sample_gamma(b, 1.0, rng);
            x / (x + y)
        };
        let se_draw =
            beta(counts.true_positive as f64 + 1.0, counts.false_negative as f64 + 1.0, &mut rng);
        let sp_draw =
            beta(counts.true_negative as f64 + 1.0, counts.false_positive as f64 + 1.0, &mut rng);
        let youden = se_draw + sp_draw - 1.0;
        if youden.is_nan() || youden <= 0.0 {
            return Err(youden_refusal(youden));
        }
        for (column, name) in draws.iter_mut().zip(coordinates) {
            let offset = f64::from(u8::from(is_mean_coordinate(name)));
            column[s] = (column[s] - offset * (1.0 - sp_draw)) / youden;
        }
    }
    let mut summary = summarize_draws(draws, method);
    summary.method = "bayesian_bootstrap_with_beta_validation_draws";
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary() -> SamplingSummary {
        SamplingSummary {
            estimate: vec![0.18, 0.5, 0.32],
            standard_error: vec![0.02, 0.01, 0.01],
            lower: vec![0.0; 3],
            upper: vec![0.0; 3],
            covariance: vec![4e-4, 1e-4, -1e-4, 1e-4, 1e-4, 0.0, -1e-4, 0.0, 1e-4],
            band: None,
            holm_adjusted_p: None,
            smoothing_bias_bound: None,
            method: "test",
        }
    }

    #[test]
    fn coordinate_grammar_and_large_validation_counts_preserve_correction() {
        let names: Vec<String> = ["contrast:active-control_mean", "mean:custom", "treated_mean"]
            .map(str::to_owned)
            .to_vec();
        assert!(!is_mean_coordinate(&names[0]));
        assert!(names[1..].iter().all(|name| is_mean_coordinate(name)));
        assert!(
            is_mean_coordinate("control_mean")
                && is_mean_coordinate("active_mean")
                && is_mean_coordinate("reference_mean")
        );
        let counts = ValidationCounts {
            true_positive: 90,
            false_negative: 10,
            true_negative: 80,
            false_positive: 20,
        };
        let mut result = summary();
        correct_sampling(&mut result, &names, &[0], counts).unwrap();
        assert!((result.estimate[0] - 0.18 / 0.7).abs() < 1e-12);
        assert!((result.estimate[1] - (0.5 - 0.2) / 0.7).abs() < 1e-12);
        let huge = ValidationCounts {
            true_positive: u64::MAX,
            false_negative: 1,
            true_negative: u64::MAX,
            false_positive: 1,
        };
        assert!(correct_sampling(&mut summary(), &names, &[0], huge).is_ok());
        let empty = ValidationCounts { true_positive: 0, false_negative: 0, ..huge };
        assert!(correct_sampling(&mut summary(), &names, &[0], empty).is_err());
        let original = summarize_draws(vec![vec![0.18, 0.19, 0.2]], "test");
        let ctx = ExecutionContext::for_tests(7);
        let a = correct_posterior(original.clone(), &[names[0].clone()], counts, &ctx).unwrap();
        let b = correct_posterior(original, &["contrast:active-control".to_owned()], counts, &ctx)
            .unwrap();
        assert_eq!(a.draws, b.draws);
    }

    #[test]
    fn contrasts_scale_and_means_shift_by_the_validation_rates() {
        let names: Vec<String> =
            ["excursion_effect", "treated_mean", "control_mean"].map(str::to_owned).to_vec();
        let counts = ValidationCounts {
            true_positive: 90,
            false_negative: 10,
            true_negative: 80,
            false_positive: 20,
        };
        let mut corrected = summary();
        let youden = correct_sampling(&mut corrected, &names, &[0], counts).unwrap();
        assert!((youden - 0.7).abs() < 1e-12);
        assert!((corrected.estimate[0] - 0.18 / 0.7).abs() < 1e-12);
        assert!((corrected.estimate[1] - (0.5 - 0.2) / 0.7).abs() < 1e-12);
        assert!(corrected.standard_error[0] > 0.02 / 0.7);
        let blind = ValidationCounts {
            true_positive: 50,
            false_negative: 50,
            true_negative: 52,
            false_positive: 48,
        };
        assert!(correct_sampling(&mut summary(), &names, &[0], blind).is_err());
    }
}
