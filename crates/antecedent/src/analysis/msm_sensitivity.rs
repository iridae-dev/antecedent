//! 2.3 B3: marginal sensitivity model (MSM) bounds, tipping point and the F17 artifact.
//!
//! The facade routes and owns no scientific rule of its own:
//!
//! * [`msm_sensitivity`] runs the exact-input sharp bounds and tipping point of
//!   `antecedent-validate` ([`msm_ate_sensitivity`]);
//! * [`msm_sensitivity_artifact`] adapts the result into the composition-ready F17
//!   [`SensitivityArtifact`] that `sensitivity_decision` evaluates;
//! * [`msm_sensitivity_artifact_bytes`] exports that artifact through the bounded
//!   sectioned container.
//!
//! Every number is an assumption range or an identified value. The sampling interval is
//! withheld (`sampling interval: not reported`) and a requested sampling composition refuses
//! with `cell_not_licensed` / `msm_sensitivity.composition_not_licensed`. Calibration is
//! unmeasured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ScientificQuantity;
use antecedent_io::IoError;
pub use antecedent_io::msm_sensitivity_adapter::{
    MSM_COORDINATE_ID, MSM_COORDINATE_SCALE, MSM_SOURCE_KIND, MsmOutcomeLaw, MsmPoint,
    MsmSensitivityError, MsmSensitivityResult, MsmSensitivitySpec, MsmStratum, MsmTipping,
    MsmTippingDirection, msm_ate_bounds_at, msm_ate_sensitivity,
};
pub use antecedent_io::sensitivity_artifact::{ActionUtility, SensitivityArtifact};
pub use antecedent_validate::msm_sensitivity::{
    MSM_ASSUMPTION_RANGE_CLAIM, MSM_FAMILY, MSM_INTERPRETATION, MSM_INTERVAL_WITHHELD,
    MSM_MAX_GRID_POINTS, MSM_MAX_LAMBDA, MSM_MAX_OUTCOME_ATOMS, MSM_MAX_STRATA, MSM_MAX_TOLERANCE,
    MSM_METHOD, MSM_MIN_TOLERANCE, MSM_NORMALIZATION, MSM_PERTURBATION_SCALE,
    MSM_SAMPLING_STATEMENT, MSM_TARGET, MsmSensitivityUncertainty,
};

/// Sharp ATE bounds of the stratified marginal sensitivity model over a `Lambda` grid with an
/// optional tipping point.
///
/// # Errors
///
/// The typed [`MsmSensitivityError`] of the contract (`composition_not_licensed`,
/// `lambda_below_one`, `lambda_range_empty`, `positivity`, `stratum_mass`, `outcome_law`,
/// `invalid_threshold`, `invalid_tolerance`, `bounds_exceeded`).
pub fn msm_sensitivity(
    strata: &[MsmStratum],
    spec: &MsmSensitivitySpec,
) -> Result<MsmSensitivityResult, MsmSensitivityError> {
    msm_ate_sensitivity(strata, spec)
}

/// The F17 sensitivity artifact of a result.
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
