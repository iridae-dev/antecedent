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
pub(crate) const COMPOSITION_BOUNDARY: UnitId = UnitId::new("composition.boundary");
pub(crate) const COMPOSITION_BUNDLE: UnitId = UnitId::new("composition.bundle");
pub(crate) const COMPOSITION_PROPAGATION_CHAIN: UnitId =
    UnitId::new("composition.propagation_chain");
pub(crate) const COMPOSITION_SAMPLING_ASSUMPTION_UNION: UnitId =
    UnitId::new("composition.sampling_assumption_union");
pub(crate) const COUNTERFACTUAL_TRANSPORT_GENERAL: UnitId =
    UnitId::new("counterfactual.transport_general");
pub(crate) const COUNTERFACTUAL_TRANSPORTED_ADDITIVE: UnitId =
    UnitId::new("counterfactual.transported_additive");
pub(crate) const CYCLIC_ACTION_WORLD_DECISIONS: UnitId =
    UnitId::new("cyclic.action_world_decisions");
pub(crate) const CYCLIC_COMPONENT_POSTERIOR_DECISIONS: UnitId =
    UnitId::new("cyclic.component_posterior_decisions");
pub(crate) const CYCLIC_GENERAL_SYSTEMS: UnitId = UnitId::new("cyclic.general_systems");
pub(crate) const DECISION_CONTRACT: UnitId = UnitId::new("decision.contract");
pub(crate) const DECISION_FUNCTIONAL_INVERSE: UnitId = UnitId::new("decision.functional_inverse");
pub(crate) const DECISION_FUNCTIONALS: UnitId = UnitId::new("decision.functionals");
pub(crate) const DECISION_SENSITIVITY_COMPOSITION: UnitId =
    UnitId::new("decision.sensitivity_composition");
pub(crate) const DECISION_STRUCTURAL_ROBUSTNESS: UnitId =
    UnitId::new("decision.structural_robustness");
pub(crate) const DECISION_RELIABILITY_RANKING_STATEMENT: UnitId =
    UnitId::new("decision_reliability.ranking_statement");
pub(crate) const DECISION_RELIABILITY_SAMPLED_DECISION: UnitId =
    UnitId::new("decision_reliability.sampled_decision");
pub(crate) const DESIGN_ADAPTIVE_POLICY_SEARCH: UnitId =
    UnitId::new("design.adaptive_policy_search");
pub(crate) const DESIGN_EVIDENCE_OBLIGATIONS: UnitId = UnitId::new("design.evidence_obligations");
pub(crate) const DESIGN_EVIDENCE_PLAN: UnitId = UnitId::new("design.evidence_plan");
pub(crate) const DESIGN_EVSI: UnitId = UnitId::new("design.evsi");
pub(crate) const DESIGN_EXPERIMENT_PLANNING: UnitId = UnitId::new("design.experiment_planning");
pub(crate) const DESIGN_IDENTIFICATION_REPAIR: UnitId = UnitId::new("design.identification_repair");
pub(crate) const DESIGN_PRIOR_SIGNAL_PLANNING: UnitId = UnitId::new("design.prior_signal_planning");
pub(crate) const DESIGN_RANKING_ARTIFACT: UnitId = UnitId::new("design.ranking_artifact");
pub(crate) const DESIGN_STUDY_PLANNING: UnitId = UnitId::new("design.study_planning");
pub(crate) const ESTIMATE_DESCRIPTIVE_COMPARISON: UnitId =
    UnitId::new("estimate.descriptive_comparison");
pub(crate) const ESTIMATE_LATENT_SOFT_INTERVENTION: UnitId =
    UnitId::new("estimate.latent_soft_intervention");
pub(crate) const ESTIMATE_MATCHED_CASE_CONTROL_ODDS_RATIO: UnitId =
    UnitId::new("estimate.matched_case_control_odds_ratio");
pub(crate) const ESTIMATE_OBSERVATIONAL_POLICY_VALUE: UnitId =
    UnitId::new("estimate.observational_policy_value");
pub(crate) const ESTIMATE_TEMPORAL_LINEAR_PREDICTION: UnitId =
    UnitId::new("estimate.temporal_linear_prediction");
pub(crate) const FORECAST_EVENT_PROCESS: UnitId = UnitId::new("forecast.event_process");
pub(crate) const GCM_ATTRIBUTE_MECHANISM_REFIT: UnitId =
    UnitId::new("gcm.attribute.mechanism_refit");
pub(crate) const GCM_MODEL_GRAPH_FALSIFICATION: UnitId =
    UnitId::new("gcm.model.graph_falsification");
pub(crate) const GCM_SIMULATE_MECHANISM_REFIT: UnitId = UnitId::new("gcm.simulate.mechanism_refit");
pub(crate) const GRAPH_IMPUTATION_COMPILED_CANDIDATE: UnitId =
    UnitId::new("graph_imputation.compiled_candidate");
pub(crate) const GRAPH_IMPUTATION_COMPILED_DECISION: UnitId =
    UnitId::new("graph_imputation.compiled_decision");
pub(crate) const IDENTIFY_MIXED_SOURCE_PROOF_SEARCH: UnitId =
    UnitId::new("identify.mixed_source_proof_search");
pub(crate) const JOINT_RESPONSE_INTERVAL_READINGS: UnitId =
    UnitId::new("joint_response.interval_readings");
pub(crate) const MNAR_FIXED_SHIFT_COMPILED_CANDIDATE: UnitId =
    UnitId::new("mnar_fixed_shift.compiled_candidate");
pub(crate) const OBSERVATION_QUERY_FINITE_LAW_CONDITIONING: UnitId =
    UnitId::new("observation_query.finite_law_conditioning");
pub(crate) const RECOVERY_BINARY_OBSERVATION: UnitId = UnitId::new("recovery.binary_observation");
pub(crate) const RECOVERY_CHAIN_OBSERVATION: UnitId = UnitId::new("recovery.chain_observation");
pub(crate) const TRANSPORT_ADMG_CONDITIONAL: UnitId = UnitId::new("transport.admg_conditional");
pub(crate) const TRANSPORT_BRIDGE_DECISION: UnitId = UnitId::new("transport.bridge_decision");
pub(crate) const TRANSPORT_BRIDGE_PREMISE_ASSESSMENT: UnitId =
    UnitId::new("transport.bridge_premise_assessment");
pub(crate) const TRANSPORT_CLASSICAL_IDENTIFICATION: UnitId =
    UnitId::new("transport.classical_identification");
pub(crate) const TRANSPORT_JOINT_TRANSPORT_DECISION: UnitId =
    UnitId::new("transport.joint_transport_decision");
pub(crate) const TRANSPORT_MULTI_SOURCE_LIMITED_EXPERIMENT: UnitId =
    UnitId::new("transport.multi_source_limited_experiment");
pub(crate) const TRANSPORT_TEMPORAL_PATH_SPECIFIC: UnitId =
    UnitId::new("transport.temporal_path_specific");
pub(crate) const TRANSPORT_TRIAL_IPW_DECISION: UnitId = UnitId::new("transport.trial_ipw_decision");
pub(crate) const TRANSPORT_Z_IDENTIFICATION: UnitId = UnitId::new("transport.z_identification");
pub(crate) const VALIDATE_DISCOVERY_STABILITY: UnitId = UnitId::new("validate.discovery_stability");

/// The unit whose support status gates the compiled route `route`, by its
/// `PlanRoute::as_str` name.
#[must_use]
#[allow(clippy::too_many_lines, reason = "one arm per bound compiled route, generated")]
pub(crate) fn unit_of(route: &str) -> Option<UnitId> {
    match route {
        "CategoricalPosteriorCandidate" => Some(CATEGORICAL_TREATMENT_COMPILED_CANDIDATE),
        "SupportedInputDecision" => Some(COMPOSITION_BOUNDARY),
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
        "CounterfactualTransportIdentification" => Some(COUNTERFACTUAL_TRANSPORT_GENERAL),
        "TransportedPathSpecific" => Some(COUNTERFACTUAL_TRANSPORTED_ADDITIVE),
        "CyclicActionWorldDecision"
        | "CyclicPosteriorActionDecision"
        | "CyclicTargetFeedbackDecision" => Some(CYCLIC_ACTION_WORLD_DECISIONS),
        "BoundedComponentPosteriorDecision" | "JointComponentPosteriorDecision" => {
            Some(CYCLIC_COMPONENT_POSTERIOR_DECISIONS)
        }
        "CyclicFamily" | "LocalCyclicCallback" => Some(CYCLIC_GENERAL_SYSTEMS),
        "ContractLawDecision" => Some(DECISION_CONTRACT),
        "InverseQueryDecision" => Some(DECISION_FUNCTIONAL_INVERSE),
        "DecisionFunctionalValue" => Some(DECISION_FUNCTIONALS),
        "SensitivityDecision" => Some(DECISION_SENSITIVITY_COMPOSITION),
        "IdentifiedSetDecision" | "SuppliedClaimDecision" => Some(DECISION_STRUCTURAL_ROBUSTNESS),
        "DecisionDrawsRanking" => Some(DECISION_RELIABILITY_RANKING_STATEMENT),
        "SequentialPolicyReliability" => Some(DECISION_RELIABILITY_SAMPLED_DECISION),
        "AdaptivePolicySearch" => Some(DESIGN_ADAPTIVE_POLICY_SEARCH),
        "EvidenceObligations" => Some(DESIGN_EVIDENCE_OBLIGATIONS),
        "StructureIdentification" => Some(DESIGN_EVIDENCE_PLAN),
        "ChoiceStudyRanking" | "FiniteActionDecision" => Some(DESIGN_EVSI),
        "ExperimentPlanning" => Some(DESIGN_EXPERIMENT_PLANNING),
        "IdentificationRepair" | "ZTransportEvidencePlanning" => Some(DESIGN_IDENTIFICATION_REPAIR),
        "PriorSignalAdaptation" => Some(DESIGN_PRIOR_SIGNAL_PLANNING),
        "DesignRanking" => Some(DESIGN_RANKING_ARTIFACT),
        "StudyAdditionPlanning" => Some(DESIGN_STUDY_PLANNING),
        "DescriptiveComparison" => Some(ESTIMATE_DESCRIPTIVE_COMPARISON),
        "LatentSoftIntervention" => Some(ESTIMATE_LATENT_SOFT_INTERVENTION),
        "MatchedCaseControl" => Some(ESTIMATE_MATCHED_CASE_CONTROL_ODDS_RATIO),
        "ObservationalPolicy" | "PolicySubsampling" => Some(ESTIMATE_OBSERVATIONAL_POLICY_VALUE),
        "TemporalLinearPrediction" => Some(ESTIMATE_TEMPORAL_LINEAR_PREDICTION),
        "EventProcess" => Some(FORECAST_EVENT_PROCESS),
        "FittedModelAttribution" => Some(GCM_ATTRIBUTE_MECHANISM_REFIT),
        "GraphFalsification" => Some(GCM_MODEL_GRAPH_FALSIFICATION),
        "FittedModelSimulation" => Some(GCM_SIMULATE_MECHANISM_REFIT),
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
        "MixedSourceIdentification" => Some(IDENTIFY_MIXED_SOURCE_PROOF_SEARCH),
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
        "ObservationRecoveryIdentification" => Some(RECOVERY_BINARY_OBSERVATION),
        "ChainRecoveryIdentification" => Some(RECOVERY_CHAIN_OBSERVATION),
        "AdmgConditionalTransportIdentification" => Some(TRANSPORT_ADMG_CONDITIONAL),
        "AssessedBridgeTransportDecision" | "BridgeTransportDecision" => {
            Some(TRANSPORT_BRIDGE_DECISION)
        }
        "BridgeTransportPremiseAssessment" => Some(TRANSPORT_BRIDGE_PREMISE_ASSESSMENT),
        "ClassicalTransportIdentification" => Some(TRANSPORT_CLASSICAL_IDENTIFICATION),
        "JointTransportEffectDecision" | "JointTransportPredictiveDecision" => {
            Some(TRANSPORT_JOINT_TRANSPORT_DECISION)
        }
        "MzTransportIdentification" => Some(TRANSPORT_MULTI_SOURCE_LIMITED_EXPERIMENT),
        "TemporalPathTransport" => Some(TRANSPORT_TEMPORAL_PATH_SPECIFIC),
        "TrialIpwPredictiveDecision" | "TrialIpwTransportDecision" => {
            Some(TRANSPORT_TRIAL_IPW_DECISION)
        }
        "ZTransportIdentification" => Some(TRANSPORT_Z_IDENTIFICATION),
        "DiscoveryValidation" => Some(VALIDATE_DISCOVERY_STABILITY),
        _ => None,
    }
}
