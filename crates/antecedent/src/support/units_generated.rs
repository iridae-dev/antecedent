//! The support unit each compiled route runs under, generated from
//! `parity/compiled_units.toml` by `scripts/generate_compiled_units.py`. Do not edit.
//!
//! The table is keyed by the `PlanRoute::as_str` string, so it does not depend on the
//! route enum.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "a unit constant is read once a module binds its route")]

use super::unit::UnitId;

pub(crate) const COMPOSITION_BUNDLE: UnitId = UnitId::new("composition.bundle");
pub(crate) const ESTIMATE_LATENT_SOFT_INTERVENTION: UnitId =
    UnitId::new("estimate.latent_soft_intervention");
pub(crate) const ESTIMATE_OBSERVATIONAL_POLICY_VALUE: UnitId =
    UnitId::new("estimate.observational_policy_value");
pub(crate) const FORECAST_EVENT_PROCESS: UnitId = UnitId::new("forecast.event_process");
pub(crate) const JOINT_RESPONSE_INTERVAL_READINGS: UnitId =
    UnitId::new("joint_response.interval_readings");
pub(crate) const MNAR_FIXED_SHIFT_COMPILED_CANDIDATE: UnitId =
    UnitId::new("mnar_fixed_shift.compiled_candidate");

/// The unit whose support status gates the compiled route `route`, by its
/// `PlanRoute::as_str` name.
#[must_use]
pub(crate) fn unit_of(route: &str) -> Option<UnitId> {
    match route {
        "AssessedBridgePosteriorNodeDecisionBundleRecomputation"
        | "BridgeBundleRecomputation"
        | "BundleRecomputation"
        | "CyclicPosteriorBundleRecomputation"
        | "CyclicPosteriorNodeBundleRecomputation"
        | "CyclicTargetFeedbackNodeDecisionBundleRecomputation"
        | "GraphImputationPredictiveNodeDecisionBundleRecomputation"
        | "GraphImputationTransportNodeBundleRecomputation"
        | "JointTransportEffectNodeBundleRecomputation"
        | "JointTransportPredictiveNodeDecisionBundleRecomputation"
        | "PosteriorDecisionBundleRecomputation"
        | "SequentialPolicyReliabilityNodeBundleRecomputation"
        | "TrialIpwNodeBundleRecomputation"
        | "TrialIpwPredictiveNodeDecisionBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "LatentSoftIntervention" => Some(ESTIMATE_LATENT_SOFT_INTERVENTION),
        "PolicySubsampling" => Some(ESTIMATE_OBSERVATIONAL_POLICY_VALUE),
        "EventProcess" => Some(FORECAST_EVENT_PROCESS),
        "JointDesign"
        | "JointPanelExperiment"
        | "JointPanelImputation"
        | "JointPanelProgramSearch"
        | "JointPanelProgramSelection"
        | "JointPanelResponse"
        | "JointPanelTimeVarying" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "MnarFixedShift" => Some(MNAR_FIXED_SHIFT_COMPILED_CANDIDATE),
        _ => None,
    }
}
