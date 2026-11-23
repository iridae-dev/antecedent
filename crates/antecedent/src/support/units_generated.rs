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
        "AssessedBridgePosteriorNodeDecisionBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "BridgeBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "BundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "CyclicPosteriorBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "CyclicPosteriorNodeBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "CyclicTargetFeedbackNodeDecisionBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "EventProcess" => Some(FORECAST_EVENT_PROCESS),
        "GraphImputationPredictiveNodeDecisionBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "GraphImputationTransportNodeBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "JointDesign" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "JointPanelExperiment" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "JointPanelImputation" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "JointPanelProgramSearch" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "JointPanelProgramSelection" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "JointPanelResponse" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "JointPanelTimeVarying" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "JointTransportEffectNodeBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "JointTransportPredictiveNodeDecisionBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "LatentSoftIntervention" => Some(ESTIMATE_LATENT_SOFT_INTERVENTION),
        "MnarFixedShift" => Some(MNAR_FIXED_SHIFT_COMPILED_CANDIDATE),
        "PolicySubsampling" => Some(ESTIMATE_OBSERVATIONAL_POLICY_VALUE),
        "PosteriorDecisionBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "SequentialPolicyReliabilityNodeBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "TrialIpwNodeBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        "TrialIpwPredictiveNodeDecisionBundleRecomputation" => Some(COMPOSITION_BUNDLE),
        _ => None,
    }
}
