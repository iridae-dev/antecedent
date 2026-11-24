//! The support unit each compiled route runs under, generated from
//! `parity/compiled_units.toml` by `scripts/generate_compiled_units.py`. Do not edit.
//!
//! The table is keyed by the `PlanRoute::as_str` string, so it does not depend on the
//! route enum.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "a unit constant is read once a module binds its route")]

use super::unit::UnitId;

pub(crate) const ADAPTIVE_INFERENCE_UNSUPPORTED_GUARANTEES: UnitId =
    UnitId::new("adaptive_inference.unsupported_guarantees");
pub(crate) const ASSOCIATION_CONJUGATE_COUNT_POSTERIOR: UnitId =
    UnitId::new("association.conjugate_count_posterior");
pub(crate) const CATEGORICAL_TREATMENT_COMPILED_CANDIDATE: UnitId =
    UnitId::new("categorical_treatment.compiled_candidate");
pub(crate) const CHOICE_INFERENCE_INTERVAL_CANDIDATES: UnitId =
    UnitId::new("choice_inference.interval_candidates");
pub(crate) const COHORT_DECISION_INTERVAL: UnitId = UnitId::new("cohort_decision.interval");
pub(crate) const COMPOSITION_BOUNDARY: UnitId = UnitId::new("composition.boundary");
pub(crate) const COMPOSITION_BUNDLE: UnitId = UnitId::new("composition.bundle");
pub(crate) const COMPOSITION_PROPAGATION_CHAIN: UnitId =
    UnitId::new("composition.propagation_chain");
pub(crate) const COMPOSITION_SAMPLING_ASSUMPTION_UNION: UnitId =
    UnitId::new("composition.sampling_assumption_union");
pub(crate) const COUNTERFACTUAL_TRANSPORT_GENERAL: UnitId =
    UnitId::new("counterfactual.transport_general");
pub(crate) const COUNTERFACTUAL_TRANSPORT_SAMPLED: UnitId =
    UnitId::new("counterfactual.transport_sampled");
pub(crate) const COUNTERFACTUAL_TRANSPORT_SAMPLED_INTERVAL: UnitId =
    UnitId::new("counterfactual.transport_sampled_interval");
pub(crate) const COUNTERFACTUAL_TRANSPORTED_ADDITIVE: UnitId =
    UnitId::new("counterfactual.transported_additive");
pub(crate) const CYCLIC_ACTION_WORLD_DECISIONS: UnitId =
    UnitId::new("cyclic.action_world_decisions");
pub(crate) const CYCLIC_COMPONENT_POSTERIOR_DECISIONS: UnitId =
    UnitId::new("cyclic.component_posterior_decisions");
pub(crate) const CYCLIC_EXTERNAL_MECHANISMS: UnitId = UnitId::new("cyclic.external_mechanisms");
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
pub(crate) const DISCOVERY_PER_POPULATION: UnitId = UnitId::new("discovery.per_population");
pub(crate) const DISCOVERY_SAMPLE_SPLIT_EFFECT: UnitId =
    UnitId::new("discovery.sample_split_effect");
pub(crate) const DISCOVERY_SAMPLE_SPLIT_PARENT_SET_EFFECT: UnitId =
    UnitId::new("discovery.sample_split_parent_set_effect");
pub(crate) const ESTIMATE_DESCRIPTIVE_COMPARISON: UnitId =
    UnitId::new("estimate.descriptive_comparison");
pub(crate) const ESTIMATE_DOSE_GRID_BIAS_AWARE: UnitId =
    UnitId::new("estimate.dose_grid_bias_aware");
pub(crate) const ESTIMATE_JOINT_CELLS_CALLER_LEARNERS: UnitId =
    UnitId::new("estimate.joint_cells_caller_learners");
pub(crate) const ESTIMATE_LATENT_SOFT_INTERVENTION: UnitId =
    UnitId::new("estimate.latent_soft_intervention");
pub(crate) const ESTIMATE_MATCHED_CASE_CONTROL_ODDS_RATIO: UnitId =
    UnitId::new("estimate.matched_case_control_odds_ratio");
pub(crate) const ESTIMATE_OBSERVATIONAL_POLICY_VALUE: UnitId =
    UnitId::new("estimate.observational_policy_value");
pub(crate) const ESTIMATE_TEMPORAL_LINEAR_PREDICTION: UnitId =
    UnitId::new("estimate.temporal_linear_prediction");
pub(crate) const FORECAST_EVENT_PROCESS: UnitId = UnitId::new("forecast.event_process");
pub(crate) const FORECAST_HAWKES_EXCITATION: UnitId = UnitId::new("forecast.hawkes_excitation");
pub(crate) const FORECAST_HAWKES_PARAMETER_BRIDGE: UnitId =
    UnitId::new("forecast.hawkes_parameter_bridge");
pub(crate) const FORECAST_MARKED_EVENT_PROCESS: UnitId =
    UnitId::new("forecast.marked_event_process");
pub(crate) const GCM_ATTRIBUTE_MECHANISM_REFIT: UnitId =
    UnitId::new("gcm.attribute.mechanism_refit");
pub(crate) const GCM_MODEL_GRAPH_FALSIFICATION: UnitId =
    UnitId::new("gcm.model.graph_falsification");
pub(crate) const GCM_SIMULATE_MECHANISM_REFIT: UnitId = UnitId::new("gcm.simulate.mechanism_refit");
pub(crate) const GRAPH_IMPUTATION_COMPILED_CANDIDATE: UnitId =
    UnitId::new("graph_imputation.compiled_candidate");
pub(crate) const GRAPH_IMPUTATION_COMPILED_DECISION: UnitId =
    UnitId::new("graph_imputation.compiled_decision");
pub(crate) const IDENTIFY_BINARY_NESTED_MARKOV: UnitId =
    UnitId::new("identify.binary_nested_markov");
pub(crate) const IDENTIFY_BINARY_NESTED_MARKOV_FISHER: UnitId =
    UnitId::new("identify.binary_nested_markov_fisher");
pub(crate) const IDENTIFY_MIXED_SOURCE_PROOF_SEARCH: UnitId =
    UnitId::new("identify.mixed_source_proof_search");
pub(crate) const JOINT_RESPONSE_INTERVAL_READINGS: UnitId =
    UnitId::new("joint_response.interval_readings");
pub(crate) const MEASUREMENT_DETECTION_DEPENDENT_PREVALENCE: UnitId =
    UnitId::new("measurement.detection_dependent_prevalence");
pub(crate) const MEASUREMENT_EXPOSURE_MISCLASSIFICATION: UnitId =
    UnitId::new("measurement.exposure_misclassification");
pub(crate) const MEASUREMENT_MEASURED_EXPOSURE_EFFECT: UnitId =
    UnitId::new("measurement.measured_exposure_effect");
pub(crate) const MEASUREMENT_MODEL_CONTRACT: UnitId = UnitId::new("measurement.model_contract");
pub(crate) const MEASUREMENT_REPEAT_CODINGS_LATENT_CLASS: UnitId =
    UnitId::new("measurement.repeat_codings_latent_class");
pub(crate) const MEASUREMENT_SOFT_EXPOSURE: UnitId = UnitId::new("measurement.soft_exposure");
pub(crate) const MNAR_FIXED_SHIFT_COMPILED_CANDIDATE: UnitId =
    UnitId::new("mnar_fixed_shift.compiled_candidate");
pub(crate) const OBSERVATION_QUERY_CALIBRATED_PROBABILITY: UnitId =
    UnitId::new("observation_query.calibrated_probability");
pub(crate) const OBSERVATION_QUERY_FINITE_LAW_CONDITIONING: UnitId =
    UnitId::new("observation_query.finite_law_conditioning");
pub(crate) const PRIOR_COMPOSED_NAMED_COORDINATES: UnitId =
    UnitId::new("prior.composed_named_coordinates");
pub(crate) const PRIOR_EXTERNAL_COMPOSITION: UnitId = UnitId::new("prior.external_composition");
pub(crate) const RECOVERY_BINARY_OBSERVATION: UnitId = UnitId::new("recovery.binary_observation");
pub(crate) const RECOVERY_CHAIN_OBSERVATION: UnitId = UnitId::new("recovery.chain_observation");
pub(crate) const RECOVERY_SAMPLED_OBSERVATION: UnitId = UnitId::new("recovery.sampled_observation");
pub(crate) const SENSITIVITY_JOINT_MECHANISM: UnitId = UnitId::new("sensitivity.joint_mechanism");
pub(crate) const SENSITIVITY_MATCHED_HIDDEN_BIAS: UnitId =
    UnitId::new("sensitivity.matched_hidden_bias");
pub(crate) const SENSITIVITY_SUPPLIED_ESTIMATE_EVALUE: UnitId =
    UnitId::new("sensitivity.supplied_estimate_evalue");
pub(crate) const TEMPORAL_DEPENDENT_INTERVAL: UnitId = UnitId::new("temporal.dependent_interval");
pub(crate) const TRANSPORT_ADMG_CONDITIONAL: UnitId = UnitId::new("transport.admg_conditional");
pub(crate) const TRANSPORT_AUGMENTED_RESPONSE_GRID: UnitId =
    UnitId::new("transport.augmented_response_grid");
pub(crate) const TRANSPORT_BRIDGE_DECISION: UnitId = UnitId::new("transport.bridge_decision");
pub(crate) const TRANSPORT_BRIDGE_PREMISE_ASSESSMENT: UnitId =
    UnitId::new("transport.bridge_premise_assessment");
pub(crate) const TRANSPORT_CLASSICAL_IDENTIFICATION: UnitId =
    UnitId::new("transport.classical_identification");
pub(crate) const TRANSPORT_COVARIATE_SHIFT_COHORT_EFFECT: UnitId =
    UnitId::new("transport.covariate_shift_cohort_effect");
pub(crate) const TRANSPORT_CPDAG_CLASS_ENVELOPE: UnitId =
    UnitId::new("transport.cpdag_class_envelope");
pub(crate) const TRANSPORT_FINITE_SCENARIO_ENVELOPE: UnitId =
    UnitId::new("transport.finite_scenario_envelope");
pub(crate) const TRANSPORT_JOINT_BAYESIAN: UnitId = UnitId::new("transport.joint_bayesian");
pub(crate) const TRANSPORT_JOINT_TRANSPORT_DECISION: UnitId =
    UnitId::new("transport.joint_transport_decision");
pub(crate) const TRANSPORT_LEARNED_JOINT: UnitId = UnitId::new("transport.learned_joint");
pub(crate) const TRANSPORT_MEASUREMENT_MODEL: UnitId = UnitId::new("transport.measurement_model");
pub(crate) const TRANSPORT_MULTI_SOURCE_LIMITED_EXPERIMENT: UnitId =
    UnitId::new("transport.multi_source_limited_experiment");
pub(crate) const TRANSPORT_SHARED_SCENARIO_COVARIANCE: UnitId =
    UnitId::new("transport.shared_scenario_covariance");
pub(crate) const TRANSPORT_SINGLE_SOURCE_Z_TRANSPORT: UnitId =
    UnitId::new("transport.single_source_z_transport");
pub(crate) const TRANSPORT_SMOOTHED_DOSE_CALLER_LEARNERS: UnitId =
    UnitId::new("transport.smoothed_dose_caller_learners");
pub(crate) const TRANSPORT_TEMPORAL_PATH_SPECIFIC: UnitId =
    UnitId::new("transport.temporal_path_specific");
pub(crate) const TRANSPORT_TRIAL_AIPW_CALLER_LEARNERS: UnitId =
    UnitId::new("transport.trial_aipw_caller_learners");
pub(crate) const TRANSPORT_TRIAL_IPW_DECISION: UnitId = UnitId::new("transport.trial_ipw_decision");
pub(crate) const TRANSPORT_Z_IDENTIFICATION: UnitId = UnitId::new("transport.z_identification");
pub(crate) const VALIDATE_DISCOVERY_STABILITY: UnitId = UnitId::new("validate.discovery_stability");

/// The unit whose support status gates the compiled route `route`, by its
/// `PlanRoute::as_str` name.
#[must_use]
#[allow(clippy::too_many_lines, reason = "one arm per bound compiled route, generated")]
pub(crate) fn unit_of(route: &str) -> Option<UnitId> {
    match route {
        "AdaptiveResultInterval" => Some(ADAPTIVE_INFERENCE_UNSUPPORTED_GUARANTEES),
        "PowerPriorBetaSelection" | "PowerPriorGammaSelection" => {
            Some(ASSOCIATION_CONJUGATE_COUNT_POSTERIOR)
        }
        "CategoricalPosteriorCandidate" => Some(CATEGORICAL_TREATMENT_COMPILED_CANDIDATE),
        "ChoiceResultInterval" => Some(CHOICE_INFERENCE_INTERVAL_CANDIDATES),
        "CohortMeanInterval" => Some(COHORT_DECISION_INTERVAL),
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
        "SampledCounterfactualTransport" => Some(COUNTERFACTUAL_TRANSPORT_SAMPLED),
        "SampledCounterfactualTransportInterval" => Some(COUNTERFACTUAL_TRANSPORT_SAMPLED_INTERVAL),
        "TransportedPathSpecific" => Some(COUNTERFACTUAL_TRANSPORTED_ADDITIVE),
        "CyclicActionWorldDecision"
        | "CyclicPosteriorActionDecision"
        | "CyclicTargetFeedbackDecision" => Some(CYCLIC_ACTION_WORLD_DECISIONS),
        "BoundedComponentPosteriorDecision" | "JointComponentPosteriorDecision" => {
            Some(CYCLIC_COMPONENT_POSTERIOR_DECISIONS)
        }
        "ExternalFeedbackEquilibrium" | "ExternalFeedbackFit" => Some(CYCLIC_EXTERNAL_MECHANISMS),
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
        "PerPopulationStructureSearch" => Some(DISCOVERY_PER_POPULATION),
        "SampleSplitEffect" => Some(DISCOVERY_SAMPLE_SPLIT_EFFECT),
        "SampleSplitParentSetEffect" => Some(DISCOVERY_SAMPLE_SPLIT_PARENT_SET_EFFECT),
        "DescriptiveComparison" => Some(ESTIMATE_DESCRIPTIVE_COMPARISON),
        "DoseGridBiasAware" => Some(ESTIMATE_DOSE_GRID_BIAS_AWARE),
        "JointCellsLearnerFit" => Some(ESTIMATE_JOINT_CELLS_CALLER_LEARNERS),
        "LatentSoftIntervention" => Some(ESTIMATE_LATENT_SOFT_INTERVENTION),
        "MatchedCaseControl" => Some(ESTIMATE_MATCHED_CASE_CONTROL_ODDS_RATIO),
        "ObservationalPolicy" | "PolicySubsampling" => Some(ESTIMATE_OBSERVATIONAL_POLICY_VALUE),
        "TemporalLinearPrediction" => Some(ESTIMATE_TEMPORAL_LINEAR_PREDICTION),
        "EventProcess" => Some(FORECAST_EVENT_PROCESS),
        "HawkesExcitation" => Some(FORECAST_HAWKES_EXCITATION),
        "HawkesParameterBridge" => Some(FORECAST_HAWKES_PARAMETER_BRIDGE),
        "MarkedEventProcess" => Some(FORECAST_MARKED_EVENT_PROCESS),
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
        "NestedMarkovPosterior" => Some(IDENTIFY_BINARY_NESTED_MARKOV),
        "NestedMarkovFisherInterval" => Some(IDENTIFY_BINARY_NESTED_MARKOV_FISHER),
        "MixedSourceIdentification" => Some(IDENTIFY_MIXED_SOURCE_PROOF_SEARCH),
        "JointDesign"
        | "JointPanelExperiment"
        | "JointPanelImputation"
        | "JointPanelProgramSearch"
        | "JointPanelProgramSelection"
        | "JointPanelResponse"
        | "JointPanelTimeVarying" => Some(JOINT_RESPONSE_INTERVAL_READINGS),
        "DetectionDependentPrevalenceCorrection" => {
            Some(MEASUREMENT_DETECTION_DEPENDENT_PREVALENCE)
        }
        "ExposureMisclassificationCorrection" => Some(MEASUREMENT_EXPOSURE_MISCLASSIFICATION),
        "MeasuredExposureImputation" | "MeasuredExposurePooling" => {
            Some(MEASUREMENT_MEASURED_EXPOSURE_EFFECT)
        }
        "PooledDifferentialTransport" => Some(MEASUREMENT_MODEL_CONTRACT),
        "RepeatCodingsLatentClass" => Some(MEASUREMENT_REPEAT_CODINGS_LATENT_CLASS),
        "SoftExposureCorrection" => Some(MEASUREMENT_SOFT_EXPOSURE),
        "MnarFixedShift" => Some(MNAR_FIXED_SHIFT_COMPILED_CANDIDATE),
        "CountFittedInterval" => Some(OBSERVATION_QUERY_CALIBRATED_PROBABILITY),
        "DependentMeasuredObservation"
        | "FittedDependentMeasuredObservation"
        | "JointMeasuredObservation" => Some(OBSERVATION_QUERY_FINITE_LAW_CONDITIONING),
        "PowerPriorLinearSelection" => Some(PRIOR_COMPOSED_NAMED_COORDINATES),
        "ExternalPriorComposition" | "ExternalPriorTransport" => Some(PRIOR_EXTERNAL_COMPOSITION),
        "ObservationRecoveryIdentification" => Some(RECOVERY_BINARY_OBSERVATION),
        "ChainObservationRecovery" | "ChainRecoveryIdentification" => {
            Some(RECOVERY_CHAIN_OBSERVATION)
        }
        "MeasuredSampledObservationRecovery" | "SampledObservationRecovery" => {
            Some(RECOVERY_SAMPLED_OBSERVATION)
        }
        "ZTransportMechanismSensitivity" => Some(SENSITIVITY_JOINT_MECHANISM),
        "MatchedDesignSensitivity" => Some(SENSITIVITY_MATCHED_HIDDEN_BIAS),
        "SuppliedEstimateEValue" => Some(SENSITIVITY_SUPPLIED_ESTIMATE_EVALUE),
        "TemporalDependentInterval" => Some(TEMPORAL_DEPENDENT_INTERVAL),
        "AdmgConditionalTransportIdentification" => Some(TRANSPORT_ADMG_CONDITIONAL),
        "TransportResponseGrid" => Some(TRANSPORT_AUGMENTED_RESPONSE_GRID),
        "AssessedBridgeTransportDecision" | "BridgeTransportDecision" => {
            Some(TRANSPORT_BRIDGE_DECISION)
        }
        "BridgeTransportPremiseAssessment" => Some(TRANSPORT_BRIDGE_PREMISE_ASSESSMENT),
        "ClassicalTransportIdentification" => Some(TRANSPORT_CLASSICAL_IDENTIFICATION),
        "TransportedCohortEffect" => Some(TRANSPORT_COVARIATE_SHIFT_COHORT_EFFECT),
        "DiscoveredClassTransport" => Some(TRANSPORT_CPDAG_CLASS_ENVELOPE),
        "TransportScenarioSet" => Some(TRANSPORT_FINITE_SCENARIO_ENVELOPE),
        "JointBayesianTransport" => Some(TRANSPORT_JOINT_BAYESIAN),
        "JointTransportEffectDecision" | "JointTransportPredictiveDecision" => {
            Some(TRANSPORT_JOINT_TRANSPORT_DECISION)
        }
        "LearnedJointTransport" => Some(TRANSPORT_LEARNED_JOINT),
        "MeasurementModelTransport" => Some(TRANSPORT_MEASUREMENT_MODEL),
        "MzTransportIdentification" => Some(TRANSPORT_MULTI_SOURCE_LIMITED_EXPERIMENT),
        "SharedScenarioCovariance" => Some(TRANSPORT_SHARED_SCENARIO_COVARIANCE),
        "ZTransportEstimate" => Some(TRANSPORT_SINGLE_SOURCE_Z_TRANSPORT),
        "SmoothedDoseLearnerTransport" => Some(TRANSPORT_SMOOTHED_DOSE_CALLER_LEARNERS),
        "TemporalPathTransport" => Some(TRANSPORT_TEMPORAL_PATH_SPECIFIC),
        "TrialAipwLearnerTransport" => Some(TRANSPORT_TRIAL_AIPW_CALLER_LEARNERS),
        "TrialIpwPredictiveDecision" | "TrialIpwTransportDecision" => {
            Some(TRANSPORT_TRIAL_IPW_DECISION)
        }
        "ZTransportIdentification" => Some(TRANSPORT_Z_IDENTIFICATION),
        "DiscoveryValidation" => Some(VALIDATE_DISCOVERY_STABILITY),
        _ => None,
    }
}
