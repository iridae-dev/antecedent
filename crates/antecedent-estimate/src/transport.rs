//! Trial-to-target transport estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_identify::{TransportFormula, TransportIdentification};

use crate::EstimationError;

/// Dahabreh-style trial-to-target IPW/AIPW and the composed response-grid primitive implement
/// the algebra of *direct* transport and *standardization* over a trial-selection mechanism.
/// A recursive factorization certificate is a different identifying formula; evaluating the
/// Dahabreh functional against it would manufacture a number the certificate did not license.
fn require_dahabreh_compatible_formula(
    identification: &TransportIdentification,
    stage: &str,
) -> Result<(), EstimationError> {
    match identification {
        TransportIdentification::NotCertified(certificate) => {
            Err(EstimationError::not_certified(stage, &certificate.reason, &certificate.message))
        }
        TransportIdentification::MissingEvidence(certificate) => {
            Err(EstimationError::not_certified(stage, &certificate.reason, &certificate.message))
        }
        TransportIdentification::Transportable {
            formula: TransportFormula::Direct(_) | TransportFormula::Standardize { .. },
            ..
        } => Ok(()),
        TransportIdentification::Transportable {
            formula: TransportFormula::RecursiveFactorization { .. },
            certificate,
        } => Err(EstimationError::NotCertified {
            message: format!(
                "{stage} refused: certificate rule '{}' yields a recursive factorization, \
                 which this Dahabreh-style estimator does not evaluate; identification and \
                 estimation stay separate, and recursive factorization remains identify-only \
                 in this release",
                certificate.rule
            ),
        }),
    }
}

/// Overlap diagnostics for one transport nuisance mechanism.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TransportOverlapDiagnostic {
    /// Minimum probability.
    pub probability_min: f64,
    /// Maximum probability.
    pub probability_max: f64,
    /// Kish effective sample size of weights using this mechanism.
    pub effective_sample_size: f64,
    /// Count of weights greater than [`EXTREME_WEIGHT_THRESHOLD`]. A diagnostic tripwire, not
    /// an inferential cutoff.
    pub extreme_weight_count: usize,
}

/// Separate overlap reports for trial selection and randomized treatment.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
/// `identification` must be the [`TransportIdentification`] produced for this query by
/// `antecedent_identify::TransportIdentifier::identify`. A transported estimate is only
/// meaningful once a sound formula has been certified for the query being evaluated here;
/// evaluating this grid against an uncertified query would silently manufacture a number for
/// a quantity that was never shown to be identified. When `identification` is
/// [`TransportIdentification::NotCertified`], or when it certifies a
/// [`TransportFormula::RecursiveFactorization`] that this Dahabreh-style grid does not
/// evaluate, this function returns an error rather than an estimate.
///
/// # Errors
///
/// Returns [`EstimationError`] for empty/mismatched inputs, non-finite values, invalid selection
/// probabilities, non-finite response weights, samples without both source and target rows, an
/// `identification` that is [`TransportIdentification::NotCertified`], or a recursive-
/// factorization certificate.
#[allow(clippy::too_many_arguments)]
pub fn transport_augmented_response_grid(
    identification: &TransportIdentification,
    outcome: &[f64],
    trial: &[bool],
    selection_probability: &[f64],
    grid: &[f64],
    observed_regression: &[f64],
    target_regression: &[f64],
    response_weight: &[f64],
) -> Result<TransportResponseGridEstimate, EstimationError> {
    require_dahabreh_compatible_formula(identification, "transport response grid")?;
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
    let mut trial_selection_probabilities = Vec::new();
    for i in 0..n {
        let probability = selection_probability[i];
        if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
            return Err(EstimationError::data_msg(
                "selection probabilities must lie strictly inside (0,1)",
            ));
        }
        // Checked unconditionally (not just for trial[i] rows): the doc contract promises
        // that target-row outcomes/regressions must still be finite so malformed rectangular
        // inputs cannot pass silently, even though their values are otherwise unused below.
        if !observed_regression[i].is_finite() || !outcome[i].is_finite() {
            return Err(EstimationError::data_msg(
                "outcomes and observed-treatment regressions must be finite for every row",
            ));
        }
        if trial[i] {
            selection_weights.push((1.0 - probability) / probability);
            trial_selection_probabilities.push(probability);
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
        selection_overlap: diagnostic(&trial_selection_probabilities, &selection_weights),
    })
}

/// Transport a randomized binary-treatment contrast from trial participants to nonparticipants.
///
/// `trial[i]` is true for source-trial rows. `selection_probability` is `P(S=1|X)` and
/// `treatment_probability` is `P(A=1|X,S=1)`. AIPW requires both `mu0` and `mu1`, interpreted as
/// trial outcome regressions evaluated for every source and target row.
///
/// `identification` must be the [`TransportIdentification`] produced for this query by
/// `antecedent_identify::TransportIdentifier::identify`. Estimating a transported contrast
/// without a positive identification certificate would report a number for a quantity that was
/// never shown to be identified, so when `identification` is
/// [`TransportIdentification::NotCertified`] this function returns an error carrying the
/// certificate's `reason` and `message` instead of an estimate.
///
/// # Errors
///
/// Returns [`EstimationError`] for empty/mismatched inputs, out-of-range probabilities, samples
/// without both source and target rows, an `identification` that is
/// [`TransportIdentification::NotCertified`], or a [`TransportFormula::RecursiveFactorization`]
/// certificate (Dahabreh IPW/AIPW does not evaluate that formula).
pub fn trial_to_target_effect(
    identification: &TransportIdentification,
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    outcome_regressions: Option<(&[f64], &[f64])>,
) -> Result<TransportEffectEstimate, EstimationError> {
    require_dahabreh_compatible_formula(identification, "trial-to-target effect")?;
    let n = outcome.len();
    if let Some((mu0, mu1)) = outcome_regressions {
        if mu0.len() != n || mu1.len() != n {
            return Err(EstimationError::data_msg("transport outcome-regression length mismatch"));
        }
    }
    let target_n = validate_trial_to_target_inputs(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    let mut ipw_sum = 0.0;
    let mut augmentation_sum = 0.0;
    let mut selection_weights = Vec::new();
    let mut treatment_weights = Vec::new();
    let mut trial_selection_probabilities = Vec::new();
    let mut trial_treatment_probabilities = Vec::new();
    for i in 0..n {
        let s = selection_probability[i];
        let e = treatment_probability[i];
        // Ranges and finiteness were checked by `validate_trial_to_target_inputs`; `e` is
        // P(A=1|X,S=1), defined and read only on trial rows.
        if trial[i] {
            let selection_odds = (1.0 - s) / s;
            let arm = if treatment[i] { e } else { 1.0 - e };
            let sign = if treatment[i] { 1.0 } else { -1.0 };
            let weight = selection_odds / arm;
            ipw_sum += sign * weight * outcome[i];
            selection_weights.push(selection_odds);
            treatment_weights.push(1.0 / arm);
            trial_selection_probabilities.push(s);
            trial_treatment_probabilities.push(e);
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
            selection: diagnostic(&trial_selection_probabilities, &selection_weights),
            treatment: diagnostic(&trial_treatment_probabilities, &treatment_weights),
        },
    })
}

/// Bayesian-bootstrap draws for the identified trial-to-target IPW contrast.
///
/// Each draw puts a shared `Dirichlet(1, ..., 1)` law on the observed trial rows,
/// keeping the trial sample size, target sample size, supplied selection odds,
/// and supplied treatment probabilities fixed. It propagates uncertainty in the
/// empirical trial outcome law, conditional on these supplied probabilities;
/// it does not propagate uncertainty from fitting either probability model or
/// from sampling the target rows. The identifying certificate is checked anew.
///
/// # Errors
///
/// Uncertified identification, invalid inputs, fewer than two draws, or a trial
/// without observed units in both treatment arms.
pub fn trial_to_target_bayesian_bootstrap(
    identification: &TransportIdentification,
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    n_draws: usize,
    seed: u64,
) -> Result<Vec<f64>, EstimationError> {
    require_dahabreh_compatible_formula(identification, "Bayesian trial-to-target effect")?;
    let target_n = validate_trial_to_target_inputs(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    if n_draws < 2 {
        return Err(EstimationError::data_msg(
            "Bayesian trial-to-target effect requires at least two draws",
        ));
    }
    let source_rows: Vec<usize> = trial
        .iter()
        .enumerate()
        .filter_map(|(index, is_trial)| is_trial.then_some(index))
        .collect();
    if !source_rows.iter().any(|&i| treatment[i]) || !source_rows.iter().any(|&i| !treatment[i]) {
        return Err(EstimationError::data_msg(
            "Bayesian trial-to-target effect requires both observed treatment arms",
        ));
    }
    let scores: Vec<f64> = source_rows
        .iter()
        .map(|&i| {
            let s = selection_probability[i];
            let e = treatment_probability[i];
            let arm = if treatment[i] { e } else { 1.0 - e };
            let sign = if treatment[i] { 1.0 } else { -1.0 };
            sign * ((1.0 - s) / s) * outcome[i] / arm
        })
        .collect();
    let mut rng = CausalRng::from_seed(seed);
    let mut draws = Vec::with_capacity(n_draws);
    for _ in 0..n_draws {
        let weights: Vec<f64> =
            source_rows.iter().map(|_| -rng.next_f64().max(f64::MIN_POSITIVE).ln()).collect();
        let normalizer: f64 = weights.iter().sum();
        if !normalizer.is_finite() || normalizer <= 0.0 {
            return Err(EstimationError::stats_msg(
                "Bayesian trial-to-target row weights failed to normalize",
            ));
        }
        let numerator: f64 = weights.iter().zip(&scores).map(|(w, score)| w * score).sum();
        let draw = numerator * source_rows.len() as f64 / (normalizer * target_n as f64);
        if !draw.is_finite() {
            return Err(EstimationError::stats_msg(
                "Bayesian trial-to-target effect has a non-finite draw",
            ));
        }
        draws.push(draw);
    }
    Ok(draws)
}

/// Checks shared by [`trial_to_target_effect`] and [`trial_to_target_ipw_se`]: equal
/// non-empty lengths, both source and target rows, finite trial outcomes, and
/// probabilities strictly inside (0, 1) (the treatment probability only on trial rows,
/// where it is defined). Returns the number of target rows.
fn validate_trial_to_target_inputs(
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
) -> Result<usize, EstimationError> {
    let n = outcome.len();
    if n == 0
        || treatment.len() != n
        || trial.len() != n
        || selection_probability.len() != n
        || treatment_probability.len() != n
    {
        return Err(EstimationError::data_msg("transport input length mismatch"));
    }
    let target_n = trial.iter().filter(|&&source| !source).count();
    if target_n == 0 || !trial.iter().any(|&source| source) {
        return Err(EstimationError::data_msg(
            "transport requires source-trial and target-population rows",
        ));
    }
    for i in 0..n {
        let (s, e) = (selection_probability[i], treatment_probability[i]);
        let treatment_probability_out_of_range =
            trial[i] && (!e.is_finite() || e <= 0.0 || e >= 1.0);
        if !s.is_finite() || s <= 0.0 || s >= 1.0 || treatment_probability_out_of_range {
            return Err(EstimationError::data_msg(
                "selection and treatment probabilities must lie strictly inside (0,1)",
            ));
        }
        if trial[i] && !outcome[i].is_finite() {
            return Err(EstimationError::data_msg("trial outcomes must be finite"));
        }
    }
    Ok(target_n)
}

/// Standard error of the trial-to-target IPW contrast of [`trial_to_target_effect`].
///
/// The IPW contrast is a ratio of two sample means over all `n` rows,
/// `ipw = mean(ψ) / mean(1 − S)` with `ψ_i = S_i·(±1)·(1 − s_i)/(s_i·e_i^±)·Y_i`.
/// With the selection and treatment probabilities known (the licensed
/// contract) the rows are iid and the only estimated denominator is the
/// target share, so the influence of row `i` is `(ψ_i − ipw·(1 − S_i)) /
/// mean(1 − S)` and the delta-method SE is
/// `sqrt(Σ_i (ψ_i − ipw·(1 − S_i))²) / n_target`.
///
/// The SE conditions on the probabilities as given: it carries no term for their
/// estimation, so it is design-based only when they are known (or fixed by design) and
/// is not guaranteed conservative for fitted probabilities. Inputs are those passed to
/// [`trial_to_target_effect`]; `ipw` is its [`TransportEffectEstimate::ipw`].
///
/// # Errors
///
/// Returns [`EstimationError`] under the same input conditions as
/// [`trial_to_target_effect`] (lengths, source and target rows, finite trial outcomes,
/// probabilities strictly inside (0, 1)) or for a non-finite `ipw`.
pub fn trial_to_target_ipw_se(
    outcome: &[f64],
    treatment: &[bool],
    trial: &[bool],
    selection_probability: &[f64],
    treatment_probability: &[f64],
    ipw: f64,
) -> Result<f64, EstimationError> {
    let n = outcome.len();
    let target_n = validate_trial_to_target_inputs(
        outcome,
        treatment,
        trial,
        selection_probability,
        treatment_probability,
    )?;
    if !ipw.is_finite() {
        return Err(EstimationError::data_msg("transport IPW estimate must be finite"));
    }
    let mut sum_sq = 0.0;
    for i in 0..n {
        let influence = if trial[i] {
            let s = selection_probability[i];
            let e = treatment_probability[i];
            let arm = if treatment[i] { e } else { 1.0 - e };
            let sign = if treatment[i] { 1.0 } else { -1.0 };
            sign * (1.0 - s) / s / arm * outcome[i]
        } else {
            -ipw
        };
        sum_sq += influence * influence;
    }
    Ok(sum_sq.sqrt() / target_n as f64)
}

/// Weight magnitude above which [`TransportOverlapDiagnostic::extreme_weight_count`] flags a
/// row. This is a diagnostic threshold meant to draw a reviewer's eye to poor overlap; it is not
/// an inferential cutoff and does not itself clip, trim, or otherwise change any estimate.
/// Weight magnitude above which a transport weight counts as extreme in
/// diagnostics. A tripwire for reporting, not an inferential cutoff — no
/// trimming or clipping is applied at this value.
pub const EXTREME_WEIGHT_THRESHOLD: f64 = 10.0;

fn diagnostic(probabilities: &[f64], weights: &[f64]) -> TransportOverlapDiagnostic {
    TransportOverlapDiagnostic {
        probability_min: probabilities.iter().copied().fold(f64::INFINITY, f64::min),
        probability_max: probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        effective_sample_size: kish_effective_sample_size(weights),
        extreme_weight_count: weights
            .iter()
            .filter(|&&weight| weight > EXTREME_WEIGHT_THRESHOLD)
            .count(),
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

/// Evaluate a checked, catalog-bound transport functional against supplied exact laws.
/// The returned distribution carries no sampling standard errors or intervals.
///
/// # Errors
/// Inconsistent evidence/provider metadata, missing support, invalid mass, or budgets.
pub fn evaluate_exact_transport(
    functional: &antecedent_identify::BoundTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactDistribution, antecedent_expr::EvalError> {
    prepare_exact_transport(functional, data, request, limits, ctx)?.evaluate(ctx)
}

/// Validate exact providers and compile without evaluating any probabilities.
///
/// # Errors
/// Provider contract, coverage, or resource limit violation.
pub fn prepare_exact_transport(
    functional: &antecedent_identify::BoundTransportFunctional,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_expr::ExactEvaluationPlan, antecedent_expr::EvalError> {
    use antecedent_core::{DistributionAvailability, VariableDomain};
    use antecedent_expr::{EvalError, ExactEvaluationPlan, LawTolerance};
    let treatments = &functional.derivation().query().treatments;
    if request.entries().len() != treatments.len()
        || treatments.iter().any(|v| request.get(*v).is_none())
    {
        return Err(EvalError::ProviderKind(
            "exact request must bind precisely the certified treatment coordinates",
        ));
    }
    let catalog = functional.catalog();
    for law in data.laws() {
        let regime = catalog
            .regimes
            .iter()
            .find(|r| r.id == law.regime() && r.population.as_ref() == law.population())
            .ok_or(EvalError::ProviderKind("exact provider names an unknown evidence regime"))?;
        if !regime.evidence_kind.can_satisfy_factor()
            || !matches!(regime.distribution, DistributionAvailability::Joint)
            || !regime.conditioned_on.is_empty()
            || law.interventions().len() != regime.interventions.len()
            || !law.interventions().iter().all(|a| regime.interventions.contains(&a.variable))
            || law.axes().iter().any(|axis| !regime.measured.contains(&axis.variable))
            || regime.intervention_values.iter().any(|required| {
                !law.interventions()
                    .iter()
                    .any(|a| a.variable == required.variable && a.value == required.value)
            })
        {
            return Err(EvalError::ProviderKind(
                "exact provider disagrees with its evidence regime",
            ));
        }
        for binding in catalog.bindings.iter().filter(|b| b.regime == regime.id) {
            if binding.snapshot_identity.as_ref() != law.snapshot_identity() {
                return Err(EvalError::ProviderKind(
                    "exact provider snapshot does not match catalog binding",
                ));
            }
        }
        for axis in law.axes() {
            for coordinate in catalog
                .environments
                .iter()
                .flat_map(|env| env.variables.iter())
                .filter(|c| c.variable == axis.variable)
            {
                let valid = match coordinate.domain {
                    VariableDomain::Unspecified => true,
                    VariableDomain::Continuous => false,
                    VariableDomain::Binary => {
                        axis.values.len() == 2
                            && [0.0, 1.0]
                                .iter()
                                .all(|level| axis.values.iter().any(|v| v.as_f64() == Some(*level)))
                    }
                    VariableDomain::Categorical { cardinality } => {
                        usize::try_from(cardinality).ok() == Some(axis.values.len())
                            && (0..cardinality).all(|level| {
                                axis.values.iter().any(|v| v.as_f64() == Some(f64::from(level)))
                            })
                    }
                    VariableDomain::Count => axis
                        .values
                        .iter()
                        .all(|v| v.as_f64().is_some_and(|v| v >= 0.0 && v.fract() == 0.0)),
                };
                if !valid {
                    return Err(EvalError::ProviderKind(
                        "exact provider domain disagrees with evidence coordinates",
                    ));
                }
            }
        }
    }
    ExactEvaluationPlan::compile(
        functional.arena(),
        functional.root(),
        data,
        functional.derivation().query().outcomes.clone(),
        request,
        limits,
        LawTolerance::default(),
        ctx,
    )
}

#[cfg(test)]
mod tests {
    use antecedent_identify::{
        NotCertifiedCertificate, PopulationFactor, TransportCertificate, TransportFormula,
    };

    use super::*;

    /// A minimal positive certificate, standing in for whatever
    /// `TransportIdentifier::identify` actually certified for a query. The estimators below
    /// only branch on which enum variant this is, so the certificate's content is otherwise
    /// arbitrary.
    fn certified_identification() -> TransportIdentification {
        TransportIdentification::Transportable {
            formula: TransportFormula::Direct(PopulationFactor {
                regime: None,
                population: Arc::from("source"),
                variables: Arc::from([]),
                conditioned_on: Arc::from([]),
                interventions: Arc::from([]),
            }),
            certificate: TransportCertificate {
                rule: Arc::from("transport.sid.direct"),
                selection_targets: Arc::from([]),
                premises: Arc::from([]),
            },
        }
    }

    /// A refusal certificate carrying a distinctive reason/message so tests can assert both
    /// are surfaced in the returned error rather than swallowed.
    fn not_certified_identification() -> TransportIdentification {
        TransportIdentification::NotCertified(NotCertifiedCertificate {
            reason: Arc::from("transport.test.refused"),
            witness: Arc::from([]),
            message: Arc::from("test-fixture refusal explaining why identification failed"),
        })
    }

    #[test]
    fn identical_trial_and_target_laws_recover_randomized_effect() {
        let result = trial_to_target_effect(
            &certified_identification(),
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

    /// Hand-computed ratio-of-means SE: weights `(1−s)/(s·e) = 2`, so
    /// `ψ = (6, −2)` on the trial rows, `ipw = (6 − 2)/2 = 2`, target-row
    /// influences `−2`, `se = sqrt(36 + 4 + 4 + 4)/2 = √48/2`.
    #[test]
    fn trial_to_target_ipw_se_is_the_ratio_of_means_delta_method() {
        let (outcome, treatment, trial) =
            ([3.0, 1.0, 0.0, 0.0], [true, false, false, false], [true, true, false, false]);
        let effect = trial_to_target_effect(
            &certified_identification(),
            &outcome,
            &treatment,
            &trial,
            &[0.5; 4],
            &[0.5; 4],
            None,
        )
        .unwrap();
        assert!((effect.ipw - 2.0).abs() < 1e-12);
        let se = trial_to_target_ipw_se(&outcome, &treatment, &trial, &[0.5; 4], &[0.5; 4], 2.0)
            .unwrap();
        assert!((se - 48.0_f64.sqrt() / 2.0).abs() < 1e-12, "se={se}");
        assert!(
            trial_to_target_ipw_se(&outcome, &treatment, &[true; 4], &[0.5; 4], &[0.5; 4], 2.0)
                .is_err()
        );
    }

    #[test]
    fn bayesian_trial_transport_matches_two_row_dirichlet_law() {
        // With one source row in each arm, equal supplied probabilities and
        // outcomes (2, 0), the draw is exactly 4U for U ~ Beta(1, 1).
        // Thus E[effect] = 2 and Var(effect) = 4/3 independently of the
        // implementation's random-number generation and weight normalization.
        let source = [true, true, false, false];
        let treatment = [true, false, false, false];
        let outcome = [2.0, 0.0, 0.0, 0.0];
        let draw = || {
            trial_to_target_bayesian_bootstrap(
                &certified_identification(),
                &outcome,
                &treatment,
                &source,
                &[0.5; 4],
                &[0.5; 4],
                10_000,
                41,
            )
            .unwrap()
        };
        let draws = draw();
        assert_eq!(draws, draw());
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        assert!((mean - 2.0).abs() < 0.08, "mean={mean}");
        assert!((variance - 4.0 / 3.0).abs() < 0.08, "variance={variance}");
        assert!(draws.iter().all(|&x| (0.0..=4.0).contains(&x)));
        assert!(
            trial_to_target_bayesian_bootstrap(
                &certified_identification(),
                &outcome,
                &[true; 4],
                &source,
                &[0.5; 4],
                &[0.5; 4],
                100,
                41,
            )
            .is_err()
        );
    }

    #[test]
    fn trial_to_target_ipw_se_shares_the_effect_input_validation() {
        let (outcome, treatment, trial) =
            ([3.0, 1.0, 0.0, 0.0], [true, false, false, false], [true, true, false, false]);
        let se = |outcome: &[f64], selection: &[f64], ipw: f64| {
            trial_to_target_ipw_se(outcome, &treatment, &trial, selection, &[0.5; 4], ipw)
        };
        assert!(se(&outcome, &[0.5; 4], 2.0).is_ok());
        // Out-of-range or non-finite selection probabilities would give inf/NaN silently.
        assert!(se(&outcome, &[0.5, 0.0, 0.5, 0.5], 2.0).is_err());
        assert!(se(&outcome, &[0.5, 0.5, 1.0, 0.5], 2.0).is_err());
        assert!(se(&outcome, &[0.5, f64::NAN, 0.5, 0.5], 2.0).is_err());
        assert!(se(&[f64::NAN, 1.0, 0.0, 0.0], &[0.5; 4], 2.0).is_err());
        assert!(se(&outcome, &[0.5; 4], f64::NAN).is_err());
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
            &certified_identification(),
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
            &certified_identification(),
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
    fn trial_to_target_effect_refuses_uncertified_identification() {
        let err = trial_to_target_effect(
            &not_certified_identification(),
            &[1.0, 0.0],
            &[true, false],
            &[true, false],
            &[0.5, 0.5],
            &[0.5, 0.5],
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("test-fixture refusal explaining why identification failed")
        );
    }

    fn recursive_factorization_identification() -> TransportIdentification {
        TransportIdentification::Transportable {
            formula: TransportFormula::RecursiveFactorization {
                sum_out: Arc::from([]),
                factors: Arc::from([PopulationFactor {
                    regime: None,
                    population: Arc::from("target"),
                    variables: Arc::from([]),
                    conditioned_on: Arc::from([]),
                    interventions: Arc::from([]),
                }]),
            },
            certificate: TransportCertificate {
                rule: Arc::from("transport.sid.singleton_c_components"),
                selection_targets: Arc::from([]),
                premises: Arc::from([]),
            },
        }
    }

    #[test]
    fn trial_to_target_effect_refuses_recursive_factorization_certificate() {
        // A recursive-factorization certificate is Transportable, so the NotCertified gate
        // alone would let the Dahabreh functional run. That functional is not the identifying
        // formula the certificate licensed — the same class of hole as estimating under
        // NotCertified, one layer down.
        let err = trial_to_target_effect(
            &recursive_factorization_identification(),
            &[1.0, 3.0, 0.0, 0.0],
            &[false, true, false, false],
            &[true, true, false, false],
            &[0.5; 4],
            &[0.5; 4],
            None,
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("recursive factorization"),
            "refusal must name the formula kind: {message}"
        );
        assert!(
            message.contains("transport.sid.singleton_c_components"),
            "refusal must name the certificate rule: {message}"
        );
    }

    #[test]
    fn transported_response_grid_refuses_recursive_factorization_certificate() {
        let err = transport_augmented_response_grid(
            &recursive_factorization_identification(),
            &[1.0, 3.0, 0.0, 0.0],
            &[true, true, false, false],
            &[0.5; 4],
            &[0.0, 1.0],
            &[1.0, 3.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
            &[1.0; 8],
        )
        .unwrap_err();
        assert!(err.to_string().contains("recursive factorization"));
    }

    #[test]
    fn transported_response_grid_evaluates_every_requested_intervention() {
        let estimate = transport_augmented_response_grid(
            &certified_identification(),
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
    fn transported_response_grid_refuses_uncertified_identification() {
        let err = transport_augmented_response_grid(
            &not_certified_identification(),
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
        .unwrap_err();
        assert!(
            err.to_string().contains("test-fixture refusal explaining why identification failed")
        );
    }

    #[test]
    fn transported_response_grid_adds_source_residual_correction() {
        let estimate = transport_augmented_response_grid(
            &certified_identification(),
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
            &certified_identification(),
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

    #[test]
    fn transported_response_grid_rejects_non_finite_target_outcome() {
        // Target-row outcomes are never read by the formula, but the documented contract
        // requires them to still be finite so a malformed rectangular input (e.g. a
        // mis-joined column) cannot pass silently.
        let err = transport_augmented_response_grid(
            &certified_identification(),
            &[1.0, 3.0, f64::NAN, 0.0],
            &[true, true, false, false],
            &[0.5; 4],
            &[0.0, 1.0],
            &[1.0, 3.0, 1.0, 1.0],
            &[
                1.0, 1.0, 1.0, 1.0, // target predictions at a=0
                3.0, 3.0, 3.0, 3.0, // target predictions at a=1
            ],
            &[1.0; 8],
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be finite"));
    }

    #[test]
    fn overlap_probability_range_is_restricted_to_trial_rows() {
        // Target rows carry a much wider (but never-realized) selection probability, and a
        // treatment probability that is meaningless off-trial (here deliberately outside
        // (0,1), and accepted precisely because target rows are not range-checked). The
        // reported min/max must come only from the trial rows that actually produced weights.
        let result = trial_to_target_effect(
            &certified_identification(),
            &[1.0, 3.0, 0.0, 0.0],
            &[false, true, false, false],
            &[true, true, false, false],
            &[0.4, 0.6, 0.01, 0.99],
            &[0.5, 0.5, -1.0, 2.0],
            None,
        )
        .unwrap();
        assert!((result.overlap.selection.probability_min - 0.4).abs() < 1e-12);
        assert!((result.overlap.selection.probability_max - 0.6).abs() < 1e-12);
        assert!((result.overlap.treatment.probability_min - 0.5).abs() < 1e-12);
        assert!((result.overlap.treatment.probability_max - 0.5).abs() < 1e-12);
    }
}
