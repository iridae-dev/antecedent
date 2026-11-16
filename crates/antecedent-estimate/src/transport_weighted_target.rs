//! Trial-to-target transport toward a weighted target population.
//!
//! The target is the non-participant rows reweighted by `v(x)`, a function of covariates (a
//! kernel neighbourhood, or declared row weights). The effect for that population is
//! `E[(1 - S) v(X) tau(X)] / E[(1 - S) v(X)]`; the trial rows reach it through the same
//! selection odds as the unweighted contrast, multiplied by `v`:
//!
//! ```text
//! ipw  = sum_{S=1} v_i sign_i (1 - s_i)/(s_i arm_i) Y_i / sum_{S=0} v_i
//! aipw = [ sum_{S=1} v_i sign_i (1 - s_i)/(s_i arm_i) (Y_i - mu_i)
//!          + sum_{S=0} v_i (mu1_i - mu0_i) ] / sum_{S=0} v_i
//! ```
//!
//! Two overlap gates guard it, both the shared [`antecedent_prob::check_weights_overlap`]:
//! the weights over the target rows (the target must rest on enough of the target sample) and
//! the weights `odds_i v_i` the trial rows carry toward it (the trial must cover the target).
//! With `v` constant the estimate is the unweighted contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_identify::TransportIdentification;
use antecedent_prob::{TargetOverlap, check_weights_overlap};

use super::{
    TransportEffectEstimate, TransportOverlapReport, diagnostic,
    require_dahabreh_compatible_formula, validate_trial_to_target_inputs,
};
use crate::EstimationError;

/// The transported contrast for a weighted target, with its overlap and standard errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedTargetTransportEstimate {
    /// The IPW and optional AIPW contrast for the weighted target, with the selection and
    /// treatment overlap of the trial rows.
    pub effect: TransportEffectEstimate,
    /// Delta-method standard error of the IPW contrast, conditional on the supplied
    /// probabilities and the fixed target weights.
    pub ipw_se: f64,
    /// The same for the AIPW contrast.
    pub aipw_se: Option<f64>,
    /// Overlap of the target weights over the target rows.
    pub target_overlap: TargetOverlap,
    /// Overlap of the weights the trial rows carry toward the target.
    pub trial_overlap: TargetOverlap,
}

/// Transport a randomized binary-treatment contrast from trial participants to the weighted
/// target.
///
/// `target_weights` holds one finite non-negative weight per row (trial and target rows), a
/// function of covariates the transport identification licenses. Everything else is as for
/// [`super::trial_to_target_effect`].
///
/// # Errors
///
/// The refusals of [`super::trial_to_target_effect`]; a weight vector of the wrong length or
/// with a non-finite or negative entry; and `transport_target_overlap_failed` when either
/// overlap gate fails.
#[allow(
    clippy::too_many_arguments,
    reason = "the seven inputs of `trial_to_target_effect`, the weights and the floor"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one weighted transport contrast with its two overlap gates and both standard errors"
)]
pub fn trial_to_weighted_target_effect(
    identification: &TransportIdentification,
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    outcome_regressions: Option<(&[f64], &[f64])>,
    target_weights: &[f64],
    min_ess_fraction: f64,
) -> Result<WeightedTargetTransportEstimate, EstimationError> {
    require_dahabreh_compatible_formula(identification, "weighted-target trial-to-target effect")?;
    let n = outcome.len();
    if target_weights.len() != n {
        return Err(EstimationError::data_msg("target weights need one entry per row"));
    }
    if let Some((mu0, mu1)) = outcome_regressions
        && (mu0.len() != n || mu1.len() != n)
    {
        return Err(EstimationError::data_msg("transport outcome-regression length mismatch"));
    }
    validate_trial_to_target_inputs(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    if target_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(EstimationError::data_msg("target weights must be finite and >= 0"));
    }

    let mut target_rows = Vec::new();
    let mut trial_rows = Vec::new();
    for i in 0..n {
        if trial[i] {
            let odds = (1.0 - selection_probability[i]) / selection_probability[i];
            trial_rows.push(odds * target_weights[i]);
        } else {
            target_rows.push(target_weights[i]);
        }
    }
    let target_overlap = check_weights_overlap(&target_rows, min_ess_fraction)?;
    let trial_overlap = check_weights_overlap(&trial_rows, min_ess_fraction)?;
    let denominator: f64 = target_rows.iter().sum();

    let mut ipw_sum = 0.0;
    let mut augmentation_sum = 0.0;
    let mut selection_weights = Vec::new();
    let mut treatment_weights = Vec::new();
    let mut trial_selection_probabilities = Vec::new();
    let mut trial_treatment_probabilities = Vec::new();
    // Per-row numerators of the two contrasts, for the delta-method standard errors.
    let mut ipw_terms = vec![0.0; n];
    let mut aipw_terms = vec![0.0; n];
    for i in 0..n {
        let v = target_weights[i];
        if trial[i] {
            let s = selection_probability[i];
            let e = treatment_probability[i];
            let odds = (1.0 - s) / s;
            let arm = if treatment[i] { e } else { 1.0 - e };
            let sign = if treatment[i] { 1.0 } else { -1.0 };
            let weight = odds / arm;
            let ipw_term = v * sign * weight * outcome[i];
            ipw_sum += ipw_term;
            ipw_terms[i] = ipw_term;
            selection_weights.push(odds);
            treatment_weights.push(1.0 / arm);
            trial_selection_probabilities.push(s);
            trial_treatment_probabilities.push(e);
            if let Some((mu0, mu1)) = outcome_regressions {
                let mu = if treatment[i] { mu1[i] } else { mu0[i] };
                if !mu.is_finite() {
                    return Err(EstimationError::data_msg("outcome regressions must be finite"));
                }
                let term = v * sign * weight * (outcome[i] - mu);
                augmentation_sum += term;
                aipw_terms[i] = term;
            }
        } else if let Some((mu0, mu1)) = outcome_regressions {
            if !mu0[i].is_finite() || !mu1[i].is_finite() {
                return Err(EstimationError::data_msg("outcome regressions must be finite"));
            }
            let term = v * (mu1[i] - mu0[i]);
            augmentation_sum += term;
            aipw_terms[i] = term;
        }
    }
    let ipw = ipw_sum / denominator;
    let aipw = outcome_regressions.map(|_| augmentation_sum / denominator);
    let standard_error = |terms: &[f64], estimate: f64| -> f64 {
        let sum_sq: f64 = (0..n)
            .map(|i| {
                let centred = terms[i] - if trial[i] { 0.0 } else { estimate * target_weights[i] };
                centred * centred
            })
            .sum();
        sum_sq.sqrt() / denominator
    };
    // Target rows are centred at `estimate * v`: the denominator `sum_{S=0} v` is the only
    // estimated nuisance when the probabilities and weights are taken as given.
    let ipw_se = standard_error(&ipw_terms, ipw);
    let aipw_se = aipw.map(|value| standard_error(&aipw_terms, value));
    Ok(WeightedTargetTransportEstimate {
        effect: TransportEffectEstimate {
            ipw,
            aipw,
            overlap: TransportOverlapReport {
                selection: diagnostic(&trial_selection_probabilities, &selection_weights),
                treatment: diagnostic(&trial_treatment_probabilities, &treatment_weights),
            },
        },
        ipw_se,
        aipw_se,
        target_overlap,
        trial_overlap,
    })
}

/// Bayesian-bootstrap draws of the trial-to-target IPW contrast for a weighted target.
///
/// Each draw puts a `Dirichlet(1, ..., 1)` law on the observed trial rows and keeps the
/// supplied probabilities, the target weights and the target rows fixed:
/// `draw = n_trial * sum_r w_r v_r score_r / (sum_r w_r * sum_{S=0} v)`, with `score_r` the
/// signed inverse-odds score of the row. It propagates the uncertainty in the trial outcome
/// law conditional on the target; with constant weights it is the unweighted construction.
/// Both overlap gates apply, as for the point estimate.
///
/// # Errors
///
/// The refusals of [`trial_to_weighted_target_effect`], fewer than two draws, or a trial
/// without observed units in both arms.
#[allow(
    clippy::too_many_arguments,
    reason = "the inputs of the weighted contrast and its draw specification"
)]
pub fn trial_to_weighted_target_bayesian_bootstrap(
    identification: &TransportIdentification,
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    target_weights: &[f64],
    min_ess_fraction: f64,
    sampling: crate::util::DrawSpec,
) -> Result<Vec<f64>, EstimationError> {
    let crate::util::DrawSpec { draws: n_draws, seed } = sampling;
    require_dahabreh_compatible_formula(identification, "Bayesian weighted-target effect")?;
    let n = outcome.len();
    if target_weights.len() != n {
        return Err(EstimationError::data_msg("target weights need one entry per row"));
    }
    validate_trial_to_target_inputs(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    if target_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(EstimationError::data_msg("target weights must be finite and >= 0"));
    }
    if n_draws < 2 {
        return Err(EstimationError::data_msg(
            "Bayesian weighted-target effect requires at least two draws",
        ));
    }
    let source_rows: Vec<usize> = (0..n).filter(|&i| trial[i]).collect();
    if !source_rows.iter().any(|&i| treatment[i]) || !source_rows.iter().any(|&i| !treatment[i]) {
        return Err(EstimationError::data_msg(
            "Bayesian weighted-target effect requires both observed treatment arms",
        ));
    }
    let target_rows: Vec<f64> = (0..n).filter(|&i| !trial[i]).map(|i| target_weights[i]).collect();
    let trial_rows: Vec<f64> = source_rows
        .iter()
        .map(|&i| (1.0 - selection_probability[i]) / selection_probability[i] * target_weights[i])
        .collect();
    check_weights_overlap(&target_rows, min_ess_fraction)?;
    check_weights_overlap(&trial_rows, min_ess_fraction)?;
    let denominator: f64 = target_rows.iter().sum();
    let scores: Vec<f64> = source_rows
        .iter()
        .map(|&i| {
            let s = selection_probability[i];
            let e = treatment_probability[i];
            let arm = if treatment[i] { e } else { 1.0 - e };
            let sign = if treatment[i] { 1.0 } else { -1.0 };
            target_weights[i] * sign * ((1.0 - s) / s) * outcome[i] / arm
        })
        .collect();
    let mut rng = antecedent_core::CausalRng::from_seed(seed);
    let _admitted = crate::util::admit_bound_array::<f64>(n_draws, 1)?;
    let mut draws = Vec::with_capacity(n_draws);
    for _ in 0..n_draws {
        let weights =
            antecedent_kernels::bayesian_bootstrap_weights(&mut rng, source_rows.len(), false);
        let normalizer: f64 = weights.iter().sum();
        if !normalizer.is_finite() || normalizer <= 0.0 {
            return Err(EstimationError::stats_msg(
                "Bayesian weighted-target row weights failed to normalize",
            ));
        }
        let numerator: f64 = weights.iter().zip(&scores).map(|(w, score)| w * score).sum();
        let draw = numerator * source_rows.len() as f64 / (normalizer * denominator);
        if !draw.is_finite() {
            return Err(EstimationError::stats_msg(
                "Bayesian weighted-target effect has a non-finite draw",
            ));
        }
        draws.push(draw);
    }
    Ok(draws)
}
