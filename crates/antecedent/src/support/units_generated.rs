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
pub(crate) const COMPOSITION_PROPAGATION_CHAIN: UnitId =
    UnitId::new("composition.propagation_chain");
pub(crate) const COMPOSITION_SAMPLING_ASSUMPTION_UNION: UnitId =
    UnitId::new("composition.sampling_assumption_union");
pub(crate) const CYCLIC_ACTION_WORLD_DECISIONS: UnitId =
    UnitId::new("cyclic.action_world_decisions");
pub(crate) const CYCLIC_COMPONENT_POSTERIOR_DECISIONS: UnitId =
    UnitId::new("cyclic.component_posterior_decisions");
pub(crate) const CYCLIC_GENERAL_SYSTEMS: UnitId = UnitId::new("cyclic.general_systems");
pub(crate) const DECISION_RELIABILITY_SAMPLED_DECISION: UnitId =
    UnitId::new("decision_reliability.sampled_decision");
pub(crate) const DESIGN_EXPERIMENT_PLANNING: UnitId = UnitId::new("design.experiment_planning");
pub(crate) const ESTIMATE_LATENT_SOFT_INTERVENTION: UnitId =
    UnitId::new("estimate.latent_soft_intervention");
pub(crate) const ESTIMATE_OBSERVATIONAL_POLICY_VALUE: UnitId =
    UnitId::new("estimate.observational_policy_value");
pub(crate) const FORECAST_EVENT_PROCESS: UnitId = UnitId::new("forecast.event_process");
pub(crate) const GRAPH_IMPUTATION_COMPILED_CANDIDATE: UnitId =
    UnitId::new("graph_imputation.compiled_candidate");
pub(crate) const GRAPH_IMPUTATION_COMPILED_DECISION: UnitId =
    UnitId::new("graph_imputation.compiled_decision");
pub(crate) const JOINT_RESPONSE_INTERVAL_READINGS: UnitId =
    UnitId::new("joint_response.interval_readings");
pub(crate) const MNAR_FIXED_SHIFT_COMPILED_CANDIDATE: UnitId =
    UnitId::new("mnar_fixed_shift.compiled_candidate");
pub(crate) const OBSERVATION_QUERY_FINITE_LAW_CONDITIONING: UnitId =
    UnitId::new("observation_query.finite_law_conditioning");
pub(crate) const TRANSPORT_BRIDGE_DECISION: UnitId = UnitId::new("transport.bridge_decision");
pub(crate) const TRANSPORT_BRIDGE_PREMISE_ASSESSMENT: UnitId =
    UnitId::new("transport.bridge_premise_assessment");
pub(crate) const TRANSPORT_JOINT_TRANSPORT_DECISION: UnitId =
    UnitId::new("transport.joint_transport_decision");
pub(crate) const TRANSPORT_TEMPORAL_PATH_SPECIFIC: UnitId =
    UnitId::new("transport.temporal_path_specific");
pub(crate) const TRANSPORT_TRIAL_IPW_DECISION: UnitId = UnitId::new("transport.trial_ipw_decision");

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
        "PropagationChain" => Some(COMPOSITION_PROPAGATION_CHAIN),
        "ClusterSamplingAssumptionUnion"
        | "IidSamplingAssumptionUnion"
        | "MovingBlockSamplingAssumptionUnion"
        | "SamplingAssumptionUnion" => Some(COMPOSITION_SAMPLING_ASSUMPTION_UNION),
        "CyclicActionWorldDecision"
        | "CyclicPosteriorActionDecision"
        | "CyclicTargetFeedbackDecision" => Some(CYCLIC_ACTION_WORLD_DECISIONS),
        "BoundedComponentPosteriorDecision" | "JointComponentPosteriorDecision" => {
            Some(CYCLIC_COMPONENT_POSTERIOR_DECISIONS)
        }
        "CyclicFamily" => Some(CYCLIC_GENERAL_SYSTEMS),
        "SequentialPolicyReliability" => Some(DECISION_RELIABILITY_SAMPLED_DECISION),
        "ExperimentPlanning" => Some(DESIGN_EXPERIMENT_PLANNING),
        "LatentSoftIntervention" => Some(ESTIMATE_LATENT_SOFT_INTERVENTION),
        "ObservationalPolicy" | "PolicySubsampling" => Some(ESTIMATE_OBSERVATIONAL_POLICY_VALUE),
        "EventProcess" => Some(FORECAST_EVENT_PROCESS),
        "GraphImputation" => Some(GRAPH_IMPUTATION_COMPILED_CANDIDATE),
        "GraphImputationBernoulliPredictiveClusterBootstrapDecision"
        | "GraphImputationBernoulliPredictiveWorldDecision"
        | "GraphImputationConfoundedKernelTransportClusterBootstrapDecision"
        | "GraphImputationConfoundedTransportClusterBootstrapDecision"
        | "GraphImputationConfoundedTransportDecision"
        | "GraphImputationEffectDecision"
        | "GraphImputationGaussianPredictiveClusterBootstrapDecision"
        | "GraphImputationGaussianTargetPredictiveWorldDecision"
        | "GraphImputationKernelTransportClusterBootstrapDecision"
        | "GraphImputationKernelTransportDecision"
        | "GraphImputationMarTransportDecision"
        | "GraphImputationTransportClusterBootstrapDecision"
        | "GraphImputationTransportDecision" => Some(GRAPH_IMPUTATION_COMPILED_DECISION),
        "JointDesign"
        | "JointPanelExperiment"
        | "JointPanelImputation"
        | "JointPanelProgramSearch"
        | "JointPanelProgramSelection"
        | "JointPanelResponse"
        | "JointPanelTimeVarying" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "MnarFixedShift" => Some(MNAR_FIXED_SHIFT_COMPILED_CANDIDATE),
        "DependentMeasuredObservation"
        | "FittedDependentMeasuredObservation"
        | "JointMeasuredObservation" => Some(OBSERVATION_QUERY_FINITE_LAW_CONDITIONING),
        "AssessedBridgeTransportDecision" | "BridgeTransportDecision" => {
            Some(TRANSPORT_BRIDGE_DECISION)
        }
        "BridgeTransportPremiseAssessment" => Some(TRANSPORT_BRIDGE_PREMISE_ASSESSMENT),
        "JointTransportEffectDecision" | "JointTransportPredictiveDecision" => {
            Some(TRANSPORT_JOINT_TRANSPORT_DECISION)
        }
        "TemporalPathTransport" => Some(TRANSPORT_TEMPORAL_PATH_SPECIFIC),
        "TrialIpwPredictiveDecision" | "TrialIpwTransportDecision" => {
            Some(TRANSPORT_TRIAL_IPW_DECISION)
        }
        _ => None,
    }
}
