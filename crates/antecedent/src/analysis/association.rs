//! The result of the associational route and the proper scores of its predictive model.
//!
//! An association describes the observed law of the columns. It is never the effect of an
//! intervention, ranks no action, and carries no coverage record: its standard errors, spreads
//! and posterior intervals are labelled `unmeasured`. A predictive model is scored on held-out
//! cases with the proper scores of the predictive-assessment contract (the Gaussian logarithmic
//! score and CRPS for a real outcome, the Brier score for a 0/1 outcome, the Poisson logarithmic
//! score and the finite-law CRPS for a count), against the training-mean baseline on the same
//! cases. A score is a point value; no interval is published for it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::{AssociationEstimate, HeldOutCase, OutcomeKind};
use antecedent_validate::predictive_assessment::{
    brier_binary, crps_gaussian, crps_weighted_atoms, log_score_gaussian,
};

/// Calibration label of every association result.
pub const ASSOCIATION_CALIBRATION: &str = "unmeasured";
/// Most atoms a count forecast's finite law keeps.
const MAX_COUNT_ATOMS: usize = 20_000;

/// Mean held-out proper score of a predictive model and of the training-mean baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionScores {
    /// Held-out cases scored.
    pub cases: usize,
    /// Name of the proper score (`log_score_gaussian`, `brier_binary`, `poisson_log_score`).
    pub score: &'static str,
    /// Weighted mean score of the model (lower is better).
    pub mean_score: f64,
    /// Weighted mean score of the training-mean baseline on the same cases.
    pub baseline_score: f64,
    /// Weighted mean CRPS of the model, for a real or count outcome.
    pub mean_crps: Option<f64>,
}

/// An association result: the estimate, the scores of a predictive model, and the refusal of
/// the causal query it follows.
#[derive(Clone, Debug, PartialEq)]
pub struct AssociationPayload {
    /// The description and the evidence behind it.
    pub estimate: AssociationEstimate,
    /// Held-out scores, when the analysis is a predictive model.
    pub scores: Option<PredictionScores>,
    /// Reason code of the refused causal query this association follows.
    pub refused_after: Option<Arc<str>>,
}

impl AssociationPayload {
    /// The label of the uncertainty the result reports: `posterior`, `resampling_standard_error`,
    /// `analytic_standard_error` or `none`. No label claims a calibration.
    #[must_use]
    pub fn uncertainty_label(&self) -> &'static str {
        if self.estimate.posterior.is_some() {
            "posterior"
        } else if self.estimate.resampling.is_some() {
            "resampling_standard_error"
        } else if self.estimate.terms.iter().any(|term| term.standard_error.is_some()) {
            "analytic_standard_error"
        } else {
            "none"
        }
    }
}

fn weighted_mean(values: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut sum, mut total) = (0.0, 0.0);
    for (value, weight) in values {
        sum += value * weight;
        total += weight;
    }
    sum / total
}

/// `ln(k!)` for a small non-negative integer.
fn ln_factorial(k: u64) -> f64 {
    (2..=k).map(|j| (j as f64).ln()).sum()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a Poisson count outcome is a non-negative integer far below u64::MAX"
)]
fn count_of(outcome: f64) -> u64 {
    outcome.round().max(0.0) as u64
}

fn poisson_log_score(rate: f64, outcome: f64) -> f64 {
    rate - outcome * rate.ln() + ln_factorial(count_of(outcome))
}

/// The finite law of a Poisson forecast: atoms `(k, P(K = k))` up to the point the tail mass is
/// negligible, renormalised to sum to one.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the upper atom is a small positive count bounded by MAX_COUNT_ATOMS"
)]
fn poisson_atoms(rate: f64, outcome: f64) -> Vec<(f64, f64)> {
    let upper = ((rate + 12.0 * rate.sqrt() + 20.0).ceil() as usize)
        .max(count_of(outcome) as usize + 1)
        .min(MAX_COUNT_ATOMS);
    let mut probability = (-rate).exp();
    let mut atoms = Vec::with_capacity(upper);
    let mut total = 0.0;
    for k in 0..upper {
        atoms.push((k as f64, probability));
        total += probability;
        probability *= rate / (k as f64 + 1.0);
    }
    for atom in &mut atoms {
        atom.1 /= total;
    }
    atoms
}

fn score_case(kind: OutcomeKind, case: &HeldOutCase, baseline: bool) -> (f64, Option<f64>) {
    let (mean, sd) = if baseline {
        (case.baseline, case.baseline_sd)
    } else {
        (case.predicted, case.predicted_sd)
    };
    match kind {
        OutcomeKind::Continuous => (
            log_score_gaussian(mean, sd, case.outcome),
            Some(crps_gaussian(mean, sd, case.outcome)),
        ),
        OutcomeKind::Binary => (brier_binary(mean, case.outcome > 0.5), None),
        OutcomeKind::Count => (
            poisson_log_score(mean, case.outcome),
            Some(crps_weighted_atoms(&poisson_atoms(mean, case.outcome), case.outcome)),
        ),
    }
}

/// Score the held-out cases of a predictive analysis.
///
/// # Panics
///
/// Never in practice: every case weight is finite and non-negative and at least one case has
/// positive weight (the estimator refuses otherwise).
#[must_use]
pub(crate) fn score_predictions(kind: OutcomeKind, cases: &[HeldOutCase]) -> PredictionScores {
    let score = match kind {
        OutcomeKind::Continuous => "log_score_gaussian",
        OutcomeKind::Binary => "brier_binary",
        OutcomeKind::Count => "poisson_log_score",
    };
    let model = |case: &HeldOutCase| score_case(kind, case, false);
    let baseline = |case: &HeldOutCase| score_case(kind, case, true);
    let mean_score = weighted_mean(cases.iter().map(|c| (model(c).0, c.weight)));
    let baseline_score = weighted_mean(cases.iter().map(|c| (baseline(c).0, c.weight)));
    let mean_crps = (kind != OutcomeKind::Binary)
        .then(|| weighted_mean(cases.iter().map(|c| (model(c).1.unwrap_or(f64::NAN), c.weight))));
    PredictionScores { cases: cases.len(), score, mean_score, baseline_score, mean_crps }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(outcome: f64, predicted: f64, baseline: f64) -> HeldOutCase {
        HeldOutCase {
            outcome,
            weight: 1.0,
            predicted,
            predicted_sd: 1.0,
            baseline,
            baseline_sd: 1.0,
        }
    }

    #[test]
    fn a_gaussian_score_is_the_negative_log_density_and_a_better_forecast_scores_lower() {
        let cases = vec![case(1.0, 1.0, 0.0), case(2.0, 2.0, 0.0)];
        let scores = score_predictions(OutcomeKind::Continuous, &cases);
        // -ln N(0; 0, 1) = 0.5 ln(2 pi).
        let at_mode = 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((scores.mean_score - at_mode).abs() < 1e-12);
        assert!(scores.mean_score < scores.baseline_score);
        assert_eq!(scores.score, "log_score_gaussian");
        assert!(scores.mean_crps.unwrap() < 0.25);
    }

    #[test]
    fn a_brier_score_and_a_poisson_score_match_their_closed_forms() {
        let binary = vec![case(1.0, 0.8, 0.5), case(0.0, 0.2, 0.5)];
        let scores = score_predictions(OutcomeKind::Binary, &binary);
        assert!((scores.mean_score - 0.04).abs() < 1e-12);
        assert!((scores.baseline_score - 0.25).abs() < 1e-12);
        assert!(scores.mean_crps.is_none());
        let counts = vec![case(2.0, 2.0, 1.0)];
        let scores = score_predictions(OutcomeKind::Count, &counts);
        // -ln Poisson(2; 2) = 2 - 2 ln 2 + ln 2.
        let expected = 2.0 - 2.0 * 2.0_f64.ln() + 2.0_f64.ln();
        assert!((scores.mean_score - expected).abs() < 1e-12);
        assert!(scores.mean_crps.unwrap() > 0.0);
        assert_eq!(scores.score, "poisson_log_score");
    }
}
