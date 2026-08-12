//! Trial-to-target transport estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use crate::EstimationError;

/// Overlap diagnostics for one transport nuisance mechanism.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportOverlapDiagnostic {
    /// Minimum probability.
    pub probability_min: f64,
    /// Maximum probability.
    pub probability_max: f64,
    /// Kish effective sample size of weights using this mechanism.
    pub effective_sample_size: f64,
    /// Count of weights greater than ten.
    pub extreme_weight_count: usize,
}

/// Separate overlap reports for trial selection and randomized treatment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportOverlapReport {
    /// Trial participation / selection overlap.
    pub selection: TransportOverlapDiagnostic,
    /// Within-trial treatment overlap.
    pub treatment: TransportOverlapDiagnostic,
}

/// Trial-to-target IPW and optional augmented-IPW contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportEffectEstimate {
    /// Selection-odds weighted IPW effect.
    pub ipw: f64,
    /// Augmented estimate when both target potential-outcome regressions were supplied.
    pub aipw: Option<f64>,
    /// Distinct overlap diagnostics.
    pub overlap: TransportOverlapReport,
}

/// An augmented transported mean response evaluated on a caller-specified grid.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportResponseGridEstimate {
    /// Treatment/intervention grid, preserved in caller order.
    pub grid: Arc<[f64]>,
    /// Transported target-population mean response at each grid point.
    pub mean: Arc<[f64]>,
    /// Kish effective sample size of the source correction weights at each grid point.
    pub source_effective_sample_size: Arc<[f64]>,
    /// Trial-selection overlap, kept separate from treatment-grid support.
    pub selection_overlap: TransportOverlapDiagnostic,
}

/// Evaluate a caller-specified augmented trial-to-target response equation over a grid.
///
/// This function composes the trial-generalization augmented formula of Dahabreh et al. (2019,
/// Web Appendix 2) with a grid-local source correction. `target_regression[g * n + i]` is the
/// source outcome regression at intervention `grid[g]`, evaluated for target row `i`.
/// `observed_regression[i]` is the same regression at the observed source treatment.
/// `response_weight[g * n + i]` is a caller-constructed treatment-density/local
/// smoothing equivalent weight for source row `i`. Signed weights are accepted, as required by
/// local-linear smoothers near boundaries. This primitive evaluates the supplied equation; it
/// does not claim that the composition itself inherits either component method's
/// double-robustness or inference theorem.
///
/// The implementation intentionally does not estimate either nuisance or choose a bandwidth:
/// doing so here would conflate structural transport, treatment support, and nuisance learning.
/// Target-row outcomes and response weights are ignored, but must still be finite so malformed
/// rectangular inputs cannot pass silently.
///
/// # Errors
///
/// Returns [`EstimationError`] for empty/mismatched inputs, non-finite values, invalid selection
/// probabilities, non-finite response weights, or samples without both source and target rows.
#[allow(clippy::too_many_arguments)]
pub fn transport_augmented_response_grid(
    outcome: &[f64],
    trial: &[bool],
    selection_probability: &[f64],
    grid: &[f64],
    observed_regression: &[f64],
    target_regression: &[f64],
    response_weight: &[f64],
) -> Result<TransportResponseGridEstimate, EstimationError> {
    let n = outcome.len();
    let cells = n
        .checked_mul(grid.len())
        .ok_or_else(|| EstimationError::data_msg("transport response grid dimensions overflow"))?;
    if n == 0
        || grid.is_empty()
        || trial.len() != n
        || selection_probability.len() != n
        || observed_regression.len() != n
        || target_regression.len() != cells
        || response_weight.len() != cells
    {
        return Err(EstimationError::data_msg("transport response grid input length mismatch"));
    }
    if grid.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::data_msg("transport response grid must be finite"));
    }
    let target_n = trial.iter().filter(|&&source| !source).count();
    if target_n == 0 || !trial.iter().any(|&source| source) {
        return Err(EstimationError::data_msg(
            "transport requires source-trial and target-population rows",
        ));
    }
    let mut selection_weights = Vec::new();
    for i in 0..n {
        let probability = selection_probability[i];
        if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
            return Err(EstimationError::data_msg(
                "selection probabilities must lie strictly inside (0,1)",
            ));
        }
        if trial[i] && (!observed_regression[i].is_finite() || !outcome[i].is_finite()) {
            return Err(EstimationError::data_msg(
                "source outcomes and observed-treatment regressions must be finite",
            ));
        }
        if trial[i] {
            selection_weights.push((1.0 - probability) / probability);
        }
    }

    let target_n_float = target_n as f64;
    let mut mean = Vec::with_capacity(grid.len());
    let mut effective_sample_size = Vec::with_capacity(grid.len());
    for grid_index in 0..grid.len() {
        let row = grid_index * n;
        let mut total = 0.0;
        let mut correction_weights = Vec::new();
        for i in 0..n {
            let prediction = target_regression[row + i];
            let local_weight = response_weight[row + i];
            if !prediction.is_finite() || !local_weight.is_finite() {
                return Err(EstimationError::data_msg(
                    "transport grid regressions and response weights must be finite",
                ));
            }
            if trial[i] {
                let selection_odds = (1.0 - selection_probability[i]) / selection_probability[i];
                let weight = selection_odds * local_weight;
                total += weight * (outcome[i] - observed_regression[i]);
                correction_weights.push(weight);
            } else {
                total += prediction;
            }
        }
        mean.push(total / target_n_float);
        effective_sample_size.push(signed_weight_effective_sample_size(&correction_weights));
    }
    Ok(TransportResponseGridEstimate {
        grid: grid.to_vec().into(),
        mean: mean.into(),
        source_effective_sample_size: effective_sample_size.into(),
        selection_overlap: diagnostic(selection_probability, &selection_weights),
    })
}

/// Transport a randomized binary-treatment contrast from trial participants to nonparticipants.
///
/// `trial[i]` is true for source-trial rows. `selection_probability` is `P(S=1|X)` and
/// `treatment_probability` is `P(A=1|X,S=1)`. AIPW requires both `mu0` and `mu1`, interpreted as
/// trial outcome regressions evaluated for every source and target row.
pub fn trial_to_target_effect(
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    outcome_regressions: Option<(&[f64], &[f64])>,
) -> Result<TransportEffectEstimate, EstimationError> {
    let n = outcome.len();
    if n == 0
        || treatment.len() != n
        || trial.len() != n
        || selection_probability.len() != n
        || treatment_probability.len() != n
    {
        return Err(EstimationError::data_msg("transport input length mismatch"));
    }
    if let Some((mu0, mu1)) = outcome_regressions {
        if mu0.len() != n || mu1.len() != n {
            return Err(EstimationError::data_msg("transport outcome-regression length mismatch"));
        }
    }
    let target_n = trial.iter().filter(|&&source| !source).count();
    if target_n == 0 || !trial.iter().any(|&source| source) {
        return Err(EstimationError::data_msg(
            "transport requires source-trial and target-population rows",
        ));
    }
    let mut ipw_sum = 0.0;
    let mut augmentation_sum = 0.0;
    let mut selection_weights = Vec::new();
    let mut treatment_weights = Vec::new();
    for i in 0..n {
        let s = selection_probability[i];
        let e = treatment_probability[i];
        if !s.is_finite() || !e.is_finite() || s <= 0.0 || s >= 1.0 || e <= 0.0 || e >= 1.0 {
            return Err(EstimationError::data_msg(
                "selection and treatment probabilities must lie strictly inside (0,1)",
            ));
        }
        if trial[i] {
            if !outcome[i].is_finite() {
                return Err(EstimationError::data_msg("trial outcomes must be finite"));
            }
            let selection_odds = (1.0 - s) / s;
            let arm = if treatment[i] { e } else { 1.0 - e };
            let sign = if treatment[i] { 1.0 } else { -1.0 };
            let weight = selection_odds / arm;
            ipw_sum += sign * weight * outcome[i];
            selection_weights.push(selection_odds);
            treatment_weights.push(1.0 / arm);
            if let Some((mu0, mu1)) = outcome_regressions {
                let mu = if treatment[i] { mu1[i] } else { mu0[i] };
                if !mu.is_finite() {
                    return Err(EstimationError::data_msg("outcome regressions must be finite"));
                }
                augmentation_sum += sign * weight * (outcome[i] - mu);
            }
        } else if let Some((mu0, mu1)) = outcome_regressions {
            if !mu0[i].is_finite() || !mu1[i].is_finite() {
                return Err(EstimationError::data_msg("outcome regressions must be finite"));
            }
            augmentation_sum += mu1[i] - mu0[i];
        }
    }
    let target_n = target_n as f64;
    Ok(TransportEffectEstimate {
        ipw: ipw_sum / target_n,
        aipw: outcome_regressions.map(|_| augmentation_sum / target_n),
        overlap: TransportOverlapReport {
            selection: diagnostic(selection_probability, &selection_weights),
            treatment: diagnostic(treatment_probability, &treatment_weights),
        },
    })
}

fn diagnostic(probabilities: &[f64], weights: &[f64]) -> TransportOverlapDiagnostic {
    TransportOverlapDiagnostic {
        probability_min: probabilities.iter().copied().fold(f64::INFINITY, f64::min),
        probability_max: probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        effective_sample_size: kish_effective_sample_size(weights),
        extreme_weight_count: weights.iter().filter(|&&weight| weight > 10.0).count(),
    }
}

fn kish_effective_sample_size(weights: &[f64]) -> f64 {
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|weight| weight * weight).sum();
    if sum_sq > 0.0 { sum * sum / sum_sq } else { 0.0 }
}

fn signed_weight_effective_sample_size(weights: &[f64]) -> f64 {
    let absolute_sum: f64 = weights.iter().map(|weight| weight.abs()).sum();
    let sum_sq: f64 = weights.iter().map(|weight| weight * weight).sum();
    if sum_sq > 0.0 { absolute_sum * absolute_sum / sum_sq } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_trial_and_target_laws_recover_randomized_effect() {
        let result = trial_to_target_effect(
            &[1.0, 3.0, 0.0, 0.0],
            &[false, true, false, false],
            &[true, true, false, false],
            &[0.5; 4],
            &[0.5; 4],
            Some((&[1.0; 4], &[3.0; 4])),
        )
        .unwrap();
        assert!((result.ipw - 2.0).abs() < 1e-12);
        assert!((result.aipw.unwrap() - 2.0).abs() < 1e-12);
        assert!((result.overlap.selection.probability_min - 0.5).abs() < f64::EPSILON);
        assert!((result.overlap.treatment.probability_min - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn matches_frozen_trial_transport_equation_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/response/trial_transport/expected.json"
        ))
        .unwrap();
        let inputs = &fixture["inputs"];
        let numbers = |field: &serde_json::Value| {
            field
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let booleans = |field: &serde_json::Value| {
            field
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_bool().unwrap())
                .collect::<Vec<_>>()
        };
        let outcome = numbers(&inputs["outcome"]);
        let treatment = booleans(&inputs["treatment"]);
        let trial = booleans(&inputs["trial"]);
        let selection = numbers(&inputs["selection_probability"]);
        let treatment_probability = numbers(&inputs["treatment_probability"]);
        let mu0 = numbers(&inputs["mu0"]);
        let mu1 = numbers(&inputs["mu1"]);
        let result = trial_to_target_effect(
            &outcome,
            &treatment,
            &trial,
            &selection,
            &treatment_probability,
            Some((&mu0, &mu1)),
        )
        .unwrap();
        let expected = &fixture["expected"];
        let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
        assert!((result.ipw - expected["ipw"].as_f64().unwrap()).abs() <= atol);
        assert!((result.aipw.unwrap() - expected["aipw"].as_f64().unwrap()).abs() <= atol);
        assert!(
            (result.overlap.selection.effective_sample_size
                - expected["selection_effective_sample_size"].as_f64().unwrap())
            .abs()
                <= atol
        );
        assert!(
            (result.overlap.treatment.effective_sample_size
                - expected["treatment_effective_sample_size"].as_f64().unwrap())
            .abs()
                <= atol
        );
    }

    #[test]
    fn selection_and_treatment_positivity_are_checked_separately() {
        let err = trial_to_target_effect(
            &[1.0, 0.0],
            &[true, false],
            &[true, false],
            &[0.0, 0.5],
            &[0.5, 0.5],
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("selection and treatment"));
    }

    #[test]
    fn transported_response_grid_evaluates_every_requested_intervention() {
        let estimate = transport_augmented_response_grid(
            &[1.0, 3.0, 0.0, 0.0],
            &[true, true, false, false],
            &[0.5; 4],
            &[0.0, 1.0],
            &[1.0, 3.0, 0.0, 0.0],
            &[
                1.0, 1.0, 1.0, 1.0, // target predictions at a=0
                3.0, 3.0, 3.0, 3.0, // target predictions at a=1
            ],
            &[1.0; 8],
        )
        .unwrap();
        assert_eq!(&*estimate.grid, &[0.0, 1.0]);
        assert_eq!(&*estimate.mean, &[1.0, 3.0]);
        assert_eq!(&*estimate.source_effective_sample_size, &[2.0, 2.0]);
    }

    #[test]
    fn transported_response_grid_adds_source_residual_correction() {
        let estimate = transport_augmented_response_grid(
            &[2.0, 4.0, 0.0],
            &[true, true, false],
            &[0.5; 3],
            &[0.0],
            &[1.0, 3.0, 0.0],
            &[0.0, 0.0, 5.0],
            &[0.5, 0.5, 0.0],
        )
        .unwrap();
        assert!((estimate.mean[0] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn transported_response_grid_accepts_signed_local_linear_equivalent_weights() {
        let estimate = transport_augmented_response_grid(
            &[2.0, 4.0, 0.0],
            &[true, true, false],
            &[0.5; 3],
            &[0.0],
            &[1.0, 3.0, 0.0],
            &[0.0, 0.0, 5.0],
            &[-0.25, 0.75, 0.0],
        )
        .unwrap();
        assert!((estimate.mean[0] - 5.5).abs() < f64::EPSILON);
        assert!((estimate.source_effective_sample_size[0] - 1.6).abs() < f64::EPSILON);
    }
}
