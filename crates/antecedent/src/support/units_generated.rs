//! The support unit each compiled route runs under, generated from
//! `parity/compiled_units.toml` by `scripts/generate_compiled_units.py`. Do not edit.
//!
//! The table is keyed by the `PlanRoute::as_str` string, so it does not depend on the
//! route enum.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "a unit constant is read once a module binds its route")]

use super::unit::UnitId;

pub(crate) const CATEGORICAL_TREATMENT_COMPILED_CANDIDATE: UnitId =
    UnitId::new("categorical_treatment.compiled_candidate");
pub(crate) const COMPOSITION_BUNDLE: UnitId = UnitId::new("composition.bundle");
pub(crate) const CYCLIC_GENERAL_SYSTEMS: UnitId = UnitId::new("cyclic.general_systems");
pub(crate) const DECISION_RELIABILITY_SAMPLED_DECISION: UnitId =
    UnitId::new("decision_reliability.sampled_decision");
pub(crate) const ESTIMATE_LATENT_SOFT_INTERVENTION: UnitId =
    UnitId::new("estimate.latent_soft_intervention");
pub(crate) const ESTIMATE_OBSERVATIONAL_POLICY_VALUE: UnitId =
    UnitId::new("estimate.observational_policy_value");
pub(crate) const FORECAST_EVENT_PROCESS: UnitId = UnitId::new("forecast.event_process");
pub(crate) const GRAPH_IMPUTATION_COMPILED_CANDIDATE: UnitId =
    UnitId::new("graph_imputation.compiled_candidate");
pub(crate) const JOINT_RESPONSE_INTERVAL_READINGS: UnitId =
    UnitId::new("joint_response.interval_readings");
pub(crate) const MNAR_FIXED_SHIFT_COMPILED_CANDIDATE: UnitId =
    UnitId::new("mnar_fixed_shift.compiled_candidate");
pub(crate) const OBSERVATION_QUERY_FINITE_LAW_CONDITIONING: UnitId =
    UnitId::new("observation_query.finite_law_conditioning");

/// The unit whose support status gates the compiled route `route`, by its
/// `PlanRoute::as_str` name.
#[must_use]
pub(crate) fn unit_of(route: &str) -> Option<UnitId> {
    match route {
        "CategoricalPosteriorCandidate" => Some(CATEGORICAL_TREATMENT_COMPILED_CANDIDATE),
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
        "CyclicFamily" => Some(CYCLIC_GENERAL_SYSTEMS),
        "SequentialPolicyReliability" => Some(DECISION_RELIABILITY_SAMPLED_DECISION),
        "LatentSoftIntervention" => Some(ESTIMATE_LATENT_SOFT_INTERVENTION),
        "ObservationalPolicy" | "PolicySubsampling" => Some(ESTIMATE_OBSERVATIONAL_POLICY_VALUE),
        "EventProcess" => Some(FORECAST_EVENT_PROCESS),
        "GraphImputation" => Some(GRAPH_IMPUTATION_COMPILED_CANDIDATE),
        "JointDesign"
        | "JointPanelExperiment"
        | "JointPanelImputation"
        | "JointPanelProgramSearch"
        | "JointPanelProgramSelection"
        | "JointPanelResponse"
        | "JointPanelTimeVarying" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "MnarFixedShift" => Some(MNAR_FIXED_SHIFT_COMPILED_CANDIDATE),
        "DependentMeasuredObservation" | "JointMeasuredObservation" => {
            Some(OBSERVATION_QUERY_FINITE_LAW_CONDITIONING)
        }
        _ => None,
    }
}
