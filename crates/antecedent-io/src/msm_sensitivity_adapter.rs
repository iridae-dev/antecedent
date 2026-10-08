//! 2.3 B3: adapter from a marginal sensitivity model result into the F17
//! [`SensitivityArtifact`].
//!
//! The assumption coordinate is `Lambda` (the odds-ratio bound, scale
//! `odds_ratio_bound`, declared range `[1, lambda_max]`); the one surface quantity is
//! the effect, with the sharp `[lower, upper]` ATE bounds at each grid `Lambda`. The
//! decision composition ([`SensitivityArtifact::decision_outcome`] here, and
//! `evaluate_sensitivity_decision` in `antecedent-design`) then classifies the actions
//! over the declared `Lambda` range as an invariant action, an assumption-dependent
//! switch or no robust action.
//!
//! The three uncertainty kinds stay apart: the surface is an assumption range, no
//! partial-identification bound is declared (the `Lambda = 1` value is a point
//! identification, available as the first grid point), and the sampling interval is
//! withheld exactly as the result declares it (`sampling interval: not reported`).
//! No sampling composition is expressible: a composed interval is refused by the
//! artifact (`composition_not_licensed`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ScientificQuantity;
pub use antecedent_validate::{
    MsmOutcomeLaw, MsmPoint, MsmSensitivityError, MsmSensitivityResult, MsmSensitivitySpec,
    MsmStratum, MsmTipping, MsmTippingDirection, msm_ate_bounds_at, msm_ate_sensitivity,
};

use crate::error::IoError;
use crate::quantity_wire::ScientificQuantityWire;
use crate::sensitivity_artifact::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, PointSupport,
    SENSITIVITY_INFERENCE_CLAIM, SamplingStatus, SensitivityArtifact, SensitivityParts,
    SensitivityProvenance, SourceTipping, SurfaceQuantity, UncertaintyRelationship,
};

/// Producer kind recorded in the artifact's provenance.
pub const MSM_SOURCE_KIND: &str = "msm_sensitivity_2_3";
/// Coordinate identity of the perturbation.
pub const MSM_COORDINATE_ID: &str = "msm_lambda";
/// Coordinate scale of the perturbation.
pub const MSM_COORDINATE_SCALE: &str = "odds_ratio_bound";

/// `detail` is the full literal detail (`sensitivity_decision_composition.<slot>`), so the
/// refusal inventory can read each one.
fn refused(code: &'static str, detail: &str, text: &str) -> IoError {
    IoError::Refused { code, message: format!("{detail}: {text}") }
}

fn invalid_effect() -> IoError {
    refused(
        antecedent_core::reason_code!("invalid_argument"),
        "sensitivity_decision_composition.invalid_surface",
        "a surface quantity coordinate is invalid",
    )
}

/// The effect surface (the two ATE bounds per grid `Lambda`) followed by the declared
/// point quantities (lower equal to upper, one value per grid point).
fn surfaces(
    result: &MsmSensitivityResult,
    effect: &ScientificQuantity,
    point_quantities: &[(ScientificQuantity, Vec<f64>)],
) -> Result<Vec<SurfaceQuantity>, IoError> {
    effect.validate().map_err(|_| invalid_effect())?;
    let mut quantities = vec![SurfaceQuantity {
        quantity: ScientificQuantityWire::from(effect),
        lower: result.grid.iter().map(|point| point.lower).collect(),
        upper: result.grid.iter().map(|point| point.upper).collect(),
    }];
    for (quantity, values) in point_quantities {
        quantity.validate().map_err(|_| invalid_effect())?;
        quantities.push(SurfaceQuantity {
            quantity: ScientificQuantityWire::from(quantity),
            lower: values.clone(),
            upper: values.clone(),
        });
    }
    Ok(quantities)
}

/// Build the F17 artifact of a marginal sensitivity result.
///
/// `effect` is the scientific coordinate of the ATE (its units are the utility units
/// the `actions` read). `point_quantities` are optional assumption-dependent point
/// quantities (for example a cost that depends on the assumed `Lambda`), each with one
/// value per result grid point. `actions` are the compared actions' utilities.
///
/// # Errors
///
/// A result that is not an assumption range, a grid that is empty or not inside
/// `[1, lambda_max]` (`invalid_surface`), an invalid effect coordinate, or any
/// [`SensitivityArtifact::new`] refusal (fewer than two actions or units that differ:
/// `wrong_contract`).
pub fn msm_sensitivity_artifact(
    result: &MsmSensitivityResult,
    effect: &ScientificQuantity,
    point_quantities: &[(ScientificQuantity, Vec<f64>)],
    actions: Vec<ActionUtility>,
    causal_contract_id: &str,
) -> Result<SensitivityArtifact, IoError> {
    if result.inference_claim != SENSITIVITY_INFERENCE_CLAIM {
        return Err(refused(
            antecedent_core::reason_code!("decision_contract_unsatisfied"),
            "sensitivity_decision_composition.wrong_contract",
            "only an assumption-range result can be adapted",
        ));
    }
    let quantities = surfaces(result, effect, point_quantities)?;
    let grid: Vec<f64> = result.grid.iter().map(|point| point.lambda).collect();
    let n = grid.len();
    SensitivityArtifact::new(SensitivityParts {
        coordinate: AssumptionCoordinate {
            id: MSM_COORDINATE_ID.into(),
            scale: MSM_COORDINATE_SCALE.into(),
            units: "dimensionless".into(),
            minimum: 1.0,
            maximum: result.lambda_max,
        },
        grid,
        support: vec![PointSupport::Supported; n],
        quantities,
        actions,
        uncertainty: UncertaintyRelationship {
            assumption_range: AssumptionRangeStatement {
                kind: "assumption_range".into(),
                interpretation: result.interpretation.into(),
            },
            identified_bound: None,
            sampling: SamplingStatus::Withheld {
                reason_code: result.uncertainty.reason_code.into(),
                detail: result.uncertainty.detail.into(),
            },
        },
        provenance: SensitivityProvenance {
            source_kind: MSM_SOURCE_KIND.into(),
            query_binding: format!(
                "{}:strata={}:lambda_max={}",
                result.family, result.strata, result.lambda_max
            ),
            provider_snapshot: "exact_population_inputs".into(),
            source_regime: effect.regime_id.clone(),
            method: result.method.into(),
            causal_contract_id: causal_contract_id.into(),
            decision_threshold: result.decision_threshold,
            source_tipping: result
                .tipping
                .iter()
                .map(|tipping| SourceTipping {
                    factor: MSM_COORDINATE_ID.into(),
                    status: tipping.status.name().into(),
                    lower: tipping.bracket.map(|b| b.lower),
                    upper: tipping.bracket.map(|b| b.upper),
                    analytic: None,
                })
                .collect(),
        },
    })
}
