//! Numerical replay of licensed randomized-arm IPW pseudo-posteriors.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::randomized_posterior::{
    ArmContrastRequest, ArmNormalPrior, ArmOutcomeModel, RandomizedEstimand, RandomizedPosterior,
    posterior_arm_contrasts_with_draws,
};
use antecedent_io_core::distribution_replay::ReplayLimits;
use antecedent_io_core::randomized_ipow_wire::{ESTIMATOR, RandomizedIpowWire, TARGET};
use antecedent_io_core::{AnalysisResultWire, IoError, PosteriorRowWire};

/// Replay the observation-model fit, its influence-corrected pseudo-outcomes, and arm law.
///
/// # Errors
/// Unsupported or non-estimable missingness, an invalid prior, or failed posterior construction.
pub fn replay(source: &RandomizedIpowWire, draws: usize) -> Result<RandomizedPosterior, IoError> {
    let outcomes: Vec<f64> = source.outcomes.iter().map(|y| y.unwrap_or(f64::NAN)).collect();
    let observed: Vec<bool> = source.outcomes.iter().map(Option::is_some).collect();
    let pseudo = antecedent_estimate::randomized_missing::responder_pseudo_outcomes(
        &outcomes,
        &observed,
        &source.assignment,
        &source.covariates,
        source.covariate_ids.len(),
    )
    .map_err(|e| IoError::Convert(e.to_string()))?;
    posterior_arm_contrasts_with_draws(
        &ArmContrastRequest {
            outcome: &pseudo,
            assignment: &source.assignment,
            probabilities: &source.probabilities,
            model: ArmOutcomeModel::Gaussian(ArmNormalPrior {
                mean: source.prior[0],
                pseudo_units: source.prior[1],
                variance_shape: source.prior[2],
                variance_scale: source.prior[3],
            }),
            contrasts: &source.contrasts,
            estimand: RandomizedEstimand::Superpopulation,
            credible_level: 0.95,
        },
        draws,
    )
    .map_err(|e| IoError::Convert(e.to_string()))
}

/// Build the retained pointwise pseudo-posterior claim from an executed arm law.
#[must_use]
pub fn posterior_row(
    source: RandomizedIpowWire,
    posterior: &RandomizedPosterior,
    coordinates: Vec<String>,
) -> PosteriorRowWire {
    let draws = posterior.contrasts.first().map_or(0, |c| c.draws.len());
    let mut row = PosteriorRowWire::new(ESTIMATOR, TARGET.to_owned(), coordinates, draws)
        .with_intervals(
            posterior.contrasts.iter().map(|c| c.mean).collect(),
            &posterior.contrasts.iter().map(|c| c.interval).collect::<Vec<_>>(),
        );
    row.licensed = source.licensed(row.coordinates.len());
    row.randomized_ipow = Some(source);
    row
}

fn verify(body: &AnalysisResultWire, limits: &ReplayLimits) -> Result<(), IoError> {
    let Some(row) = &body.posterior_row else {
        return Ok(());
    };
    let Some(source) = &row.randomized_ipow else {
        return Ok(());
    };
    let fail = || {
        IoError::Convert("randomized IPW pseudo-posterior does not replay from its source observations and declared prior".into())
    };
    let bytes = body.posterior_artifact.as_ref().ok_or_else(fail)?;
    let (meta, stored_draws) = antecedent_io_core::decode_causal_posterior_bytes(bytes)?;
    if source.outcomes.len().saturating_mul(source.covariate_ids.len().max(1)) > limits.max_cells
        || row.draws > limits.max_cells
        || meta.n_draws as usize != row.draws
        || source.contrasts.len().saturating_mul(row.draws) > limits.max_cells
    {
        return Err(fail());
    }
    // Charge the bounded 50-step logistic fit and dense score solve before running it.
    let columns = source.covariate_ids.len().saturating_add(1);
    let work = source
        .outcomes
        .len()
        .saturating_mul(columns.saturating_mul(columns))
        .saturating_add(columns.saturating_pow(3))
        .saturating_mul(50)
        .saturating_add(row.draws);
    if work > limits.max_operations {
        return Err(fail());
    }
    let posterior = replay(source, row.draws)?;
    let expected = posterior_row(source.clone(), &posterior, row.coordinates.clone());
    // Re-execution uses the same retained numerical construction. Summary comparisons allow
    // only rounding; no statistical tolerance admits a changed prior or source table.
    let close = |a: f64, b: f64| {
        a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs()))
    };
    let arrays_match =
        |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| close(*a, *b));
    let expected_draws: Vec<f64> =
        posterior.contrasts.iter().flat_map(|c| c.draws.iter().copied()).collect();
    if meta.quantities
        != row
            .coordinates
            .iter()
            .map(|name| antecedent_io_core::PosteriorQuantityWire::Effect { name: name.clone() })
            .collect::<Vec<_>>()
        || meta.backend_id != "bayesian.randomized_arm_ipow_modular"
        || !meta.converged
        || meta.unidentified_mass != 0.0
        || meta.subsampled_out_mass != 0.0
        || !row.licensed
        || row.covariance != expected.covariance
        || row.exact_mean != expected.exact_mean
        || row.exact_covariance != expected.exact_covariance
        || row.extras != expected.extras
        || row.vectors != expected.vectors
        || row.estimator != expected.estimator
        || row.target != expected.target
        || !arrays_match(&row.mean, &expected.mean)
        || !arrays_match(&row.lower, &expected.lower)
        || !arrays_match(&row.upper, &expected.upper)
        || !arrays_match(&stored_draws, &expected_draws)
        || body.estimate.is_none_or(|point| !close(point, expected.mean[0]))
        || body
            .standard_error
            .is_none_or(|sd| !close(sd, posterior.contrasts[0].standard_deviation))
    {
        return Err(fail());
    }
    Ok(())
}

/// Decode a composite result and replay any licensed IPW pseudo-posterior from source rows.
///
/// # Errors
/// Invalid container, unsupported source, or a posterior that disagrees with re-execution.
pub fn decode_analysis_result_artifact(
    bytes: &[u8],
) -> Result<
    (
        antecedent_io_core::EncodedArtifact,
        antecedent_io_core::AnalysisResultHeader,
        AnalysisResultWire,
    ),
    IoError,
> {
    let decoded = antecedent_io_core::decode_analysis_result_artifact(bytes)?;
    verify(&decoded.2, &ReplayLimits::default())?;
    Ok(decoded)
}

/// Consume a contracted result, including numerical replay of its licensed IPW construction.
///
/// # Errors
/// Invalid artifact, contract, or a pseudo-posterior that fails replay.
pub fn consume_analysis_result(
    bytes: &[u8],
) -> Result<antecedent_io_core::AnalysisResultConsumption, IoError> {
    consume_analysis_result_within(bytes, &ReplayLimits::default())
}

/// Consume under caller-provided replay bounds.
///
/// # Errors
/// As [`consume_analysis_result`], or work exceeding the supplied bounds.
pub fn consume_analysis_result_within(
    bytes: &[u8],
    limits: &ReplayLimits,
) -> Result<antecedent_io_core::AnalysisResultConsumption, IoError> {
    let mut consumed = antecedent_io_core::consume_analysis_result_within(bytes, limits)?;
    verify(&consumed.body, limits)?;
    promote_replayed_ipow(&mut consumed);
    Ok(consumed)
}

fn promote_replayed_ipow(consumed: &mut antecedent_io_core::AnalysisResultConsumption) {
    if consumed.body.posterior_row.as_ref().is_some_and(|row| row.randomized_ipow.is_some())
        && consumed.acceptance.recognized
        && consumed.contract.is_some()
    {
        let missing =
            antecedent_io_core::ContractReason::DependenciesCheckedBayesianRandomizedOperation
                .as_str();
        consumed.acceptance.unresolved = consumed
            .acceptance
            .unresolved
            .iter()
            .filter(|reason| reason.as_ref() != missing)
            .cloned()
            .collect();
        if consumed.acceptance.unresolved.is_empty() {
            consumed.acceptance.verified_references = true;
            consumed.acceptance.restriction = if consumed.acceptance.claim_present {
                None
            } else {
                Some("verified_program_without_claim".into())
            };
            let mut operations = vec![
                "store".into(),
                "forward".into(),
                "read_body".into(),
                "verify_contract".into(),
            ];
            if consumed.acceptance.claim_present {
                operations.push("inspect_claim".into());
            }
            consumed.acceptance.supported_operations = operations.into();
        }
    }
}

/// Accept an IPW claim only after replay, preserving the receiver's capability restrictions.
///
/// # Errors
/// Invalid artifact or numerical replay disagreement.
pub fn accept_claim(
    bytes: &[u8],
    profile: &antecedent_contract::ConsumerProfile,
) -> Result<
    (antecedent_io_core::AnalysisResultConsumption, antecedent_contract::HandoffReceipt),
    IoError,
> {
    let (mut consumed, mut receipt) = antecedent_io_core::accept_claim(bytes, profile)?;
    verify(&consumed.body, &ReplayLimits::default())?;
    if profile.understands("verify_contract") && profile.understands("inspect_claim") {
        promote_replayed_ipow(&mut consumed);
        if consumed.acceptance.accepts_as_claim() {
            receipt =
                antecedent_contract::HandoffReceipt::lossless(receipt.input, profile.id.clone());
        }
        consumed.acceptance.supported_operations = consumed
            .acceptance
            .supported_operations
            .iter()
            .filter(|operation| profile.understands(operation))
            .cloned()
            .collect();
    }
    Ok((consumed, receipt))
}
