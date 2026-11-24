//! Marginal sensitivity model (MSM) bounds, tipping point and the sensitivity artifact.
//!
//! The facade routes and owns no scientific rule of its own:
//!
//! * [`msm_sensitivity`] runs the exact-input sharp bounds and tipping point of
//!   `antecedent-validate` ([`msm_ate_sensitivity`]);
//! * [`msm_sensitivity_artifact`] adapts the result into the composition-ready
//!   [`SensitivityArtifact`] that `sensitivity_decision` evaluates;
//! * [`msm_sensitivity_artifact_bytes`] exports that artifact through the bounded
//!   sectioned container.
//!
//! Every number is an assumption range or an identified value. The sampling interval is
//! withheld (`sampling interval: not reported`) and a requested sampling composition refuses
//! with `not_supported` / `msm_sensitivity.composition_not_supported`. Calibration is
//! `not_measured`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_io::IoError;
pub use antecedent_io::msm_sensitivity_adapter::{
    MSM_COORDINATE_ID, MSM_COORDINATE_SCALE, MSM_SOURCE_KIND, MsmOutcomeLaw, MsmPoint,
    MsmSensitivityError, MsmSensitivityResult, MsmSensitivitySpec, MsmStratum, MsmTipping,
    MsmTippingDirection,
};
pub use antecedent_io::sensitivity_artifact::{ActionUtility, SensitivityArtifact};
pub use antecedent_validate::msm_sensitivity::{
    MSM_ASSUMPTION_RANGE_CLAIM, MSM_FAMILY, MSM_INTERPRETATION, MSM_MAX_GRID_POINTS,
    MSM_MAX_LAMBDA, MSM_MAX_OUTCOME_ATOMS, MSM_MAX_STRATA, MSM_MAX_TOLERANCE, MSM_METHOD,
    MSM_MIN_TOLERANCE, MSM_NO_INTERVAL_METHOD, MSM_NORMALIZATION, MSM_PERTURBATION_SCALE,
    MSM_SAMPLING_STATEMENT, MSM_TARGET, MsmSensitivityUncertainty,
};

/// Marginal sensitivity model bounds of the stratified ATE over a `Lambda` grid, with
/// an optional tipping point against a declared threshold.
///
/// The strata (exact masses, propensities and outcome laws) are validated first;
/// nothing is sampled. Every grid point is the sharp assumption range at that
/// `Lambda`; the ranges nest, so the bounds widen with `Lambda`. A declared sampling
/// composition refuses.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_io::msm_sensitivity_adapter::msm_ate_sensitivity`].
///
/// # Errors
///
/// A typed `MsmSensitivityError` carrying its reason code and `msm_sensitivity.*`
/// detail: `composition_not_supported`, `lambda_below_one`, `lambda_range_empty`,
/// `positivity`, `stratum_mass`, `outcome_law`, `invalid_threshold`,
/// `invalid_tolerance` or `bounds_exceeded`.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MarginalSensitivity on a two-level average-effect study: it builds the empirical strata of the study's adjustment set (mass, propensity and each arm's outcome law, as Python msm_sensitivity.from_analysis does) and calls msm_ate_sensitivity"
)]
pub fn msm_ate_sensitivity(
    strata: &[antecedent_validate::MsmStratum],
    spec: &antecedent_validate::MsmSensitivitySpec,
) -> Result<antecedent_validate::MsmSensitivityResult, antecedent_validate::MsmSensitivityError> {
    antecedent_io::msm_sensitivity_adapter::msm_ate_sensitivity(strata, spec)
}

/// Sharp ATE bounds of the stratified MSM at one `Lambda`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_io::msm_sensitivity_adapter::msm_ate_bounds_at`].
///
/// # Errors
///
/// A typed `MsmSensitivityError`: `Lambda` below one or not finite
/// (`msm_sensitivity.lambda_below_one`), a propensity outside `(0, 1)`
/// (`msm_sensitivity.positivity`), malformed masses or outcome laws, or exceeded bounds.
#[deprecated(
    since = "3.1.0",
    note = "use PreparedStudy::sensitivity_range with SensitivityRangeRequest::MarginalSensitivity; every grid point of its result is msm_ate_bounds_at at that Lambda on the study's empirical strata"
)]
pub fn msm_ate_bounds_at(
    strata: &[antecedent_validate::MsmStratum],
    lambda: f64,
) -> Result<antecedent_validate::MsmPoint, antecedent_validate::MsmSensitivityError> {
    antecedent_io::msm_sensitivity_adapter::msm_ate_bounds_at(strata, lambda)
}

/// Sharp ATE bounds of the stratified marginal sensitivity model over a `Lambda` grid with an
/// optional tipping point.
///
/// # Errors
///
/// The typed [`MsmSensitivityError`] of the contract (`composition_not_supported`,
/// `lambda_below_one`, `lambda_range_empty`, `positivity`, `stratum_mass`, `outcome_law`,
/// `invalid_threshold`, `invalid_tolerance`, `bounds_exceeded`).
pub fn msm_sensitivity(
    strata: &[MsmStratum],
    spec: &MsmSensitivitySpec,
) -> Result<MsmSensitivityResult, MsmSensitivityError> {
    antecedent_validate::msm_ate_sensitivity(strata, spec)
}

/// The sensitivity artifact of a result.
///
/// `effect` is the scientific coordinate of the ATE, `point_quantities` are optional
/// assumption-dependent point quantities (one value per grid point) and `actions` the compared
/// actions' utilities.
///
/// # Errors
///
/// Any refusal of the adapter (`invalid_surface`, `wrong_contract`).
pub fn msm_sensitivity_artifact(
    result: &MsmSensitivityResult,
    effect: &ScientificQuantity,
    point_quantities: &[(ScientificQuantity, Vec<f64>)],
    actions: Vec<ActionUtility>,
    causal_contract_id: &str,
) -> Result<SensitivityArtifact, IoError> {
    antecedent_io::msm_sensitivity_adapter::msm_sensitivity_artifact(
        result,
        effect,
        point_quantities,
        actions,
        causal_contract_id,
    )
}

/// The artifact of a result, serialized through the bounded sectioned container.
///
/// # Errors
///
/// Any adapter refusal, or a serialization failure.
pub fn msm_sensitivity_artifact_bytes(
    result: &MsmSensitivityResult,
    effect: &ScientificQuantity,
    point_quantities: &[(ScientificQuantity, Vec<f64>)],
    actions: Vec<ActionUtility>,
    causal_contract_id: &str,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    msm_sensitivity_artifact(result, effect, point_quantities, actions, causal_contract_id)?
        .to_bytes(artifact_id)
}
