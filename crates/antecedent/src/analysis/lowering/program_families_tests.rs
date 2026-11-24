//! Pins every program family's stable names, plan route and family, so a change to the
//! program-family table that would move an identity byte fails here.
//!
//! The literals are the names the hand-written arms carried before the table generated
//! them. `ProgramCell::as_str` and `PlanRoute::as_str` feed claim ids, contract seals and
//! evidence manifests: a literal here changes only with a re-earn of every record it names.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use crate::analysis::lowering::PlanRoute as R;
use crate::planner::lowered::ProgramCell as C;
use crate::planner::plan_view::RouteFamily as F;
use crate::support::unit::UnitId;
use crate::support::units_generated::unit_of;

/// Every program family, in table order: its cell, its plan route, the stable name both
/// write (the two spellings agree for every family) and the family that lowers it.
const PINNED: &[(C, R, &str, F)] = &[
    (C::ObservationalPolicy, R::ObservationalPolicy, "ObservationalPolicy", F::DesignBased),
    (
        C::JointMeasuredObservation,
        R::JointMeasuredObservation,
        "JointMeasuredObservation",
        F::Structural,
    ),
    (
        C::DependentMeasuredObservation,
        R::DependentMeasuredObservation,
        "DependentMeasuredObservation",
        F::Structural,
    ),
    (
        C::FittedDependentMeasuredObservation,
        R::FittedDependentMeasuredObservation,
        "FittedDependentMeasuredObservation",
        F::Structural,
    ),
    (
        C::CyclicActionWorldDecision,
        R::CyclicActionWorldDecision,
        "CyclicActionWorldDecision",
        F::Cyclic,
    ),
    (
        C::CyclicPosteriorActionDecision,
        R::CyclicPosteriorActionDecision,
        "CyclicPosteriorActionDecision",
        F::Cyclic,
    ),
    (
        C::CyclicTargetFeedbackDecision,
        R::CyclicTargetFeedbackDecision,
        "CyclicTargetFeedbackDecision",
        F::Cyclic,
    ),
    (
        C::BridgeTransportDecision,
        R::BridgeTransportDecision,
        "BridgeTransportDecision",
        F::Structural,
    ),
    (
        C::BridgeTransportPremiseAssessment,
        R::BridgeTransportPremiseAssessment,
        "BridgeTransportPremiseAssessment",
        F::Structural,
    ),
    (
        C::AssessedBridgeTransportDecision,
        R::AssessedBridgeTransportDecision,
        "AssessedBridgeTransportDecision",
        F::Structural,
    ),
    (
        C::JointTransportEffectDecision,
        R::JointTransportEffectDecision,
        "JointTransportEffectDecision",
        F::Structural,
    ),
    (
        C::TrialIpwTransportDecision,
        R::TrialIpwTransportDecision,
        "TrialIpwTransportDecision",
        F::Structural,
    ),
    (
        C::TrialIpwPredictiveDecision,
        R::TrialIpwPredictiveDecision,
        "TrialIpwPredictiveDecision",
        F::Structural,
    ),
    (
        C::JointTransportPredictiveDecision,
        R::JointTransportPredictiveDecision,
        "JointTransportPredictiveDecision",
        F::Structural,
    ),
    (C::TemporalPathTransport, R::TemporalPathTransport, "TemporalPathTransport", F::Temporal),
    (
        C::SamplingAssumptionUnion,
        R::SamplingAssumptionUnion,
        "SamplingAssumptionUnion",
        F::PropagationChain,
    ),
    (
        C::IidSamplingAssumptionUnion,
        R::IidSamplingAssumptionUnion,
        "IidSamplingAssumptionUnion",
        F::PropagationChain,
    ),
    (
        C::ClusterSamplingAssumptionUnion,
        R::ClusterSamplingAssumptionUnion,
        "ClusterSamplingAssumptionUnion",
        F::PropagationChain,
    ),
    (
        C::MovingBlockSamplingAssumptionUnion,
        R::MovingBlockSamplingAssumptionUnion,
        "MovingBlockSamplingAssumptionUnion",
        F::PropagationChain,
    ),
    (C::PolicySubsampling, R::PolicySubsampling, "PolicySubsampling", F::DesignBased),
    (
        C::SequentialPolicyReliability,
        R::SequentialPolicyReliability,
        "SequentialPolicyReliability",
        F::DesignBased,
    ),
    (C::BundleRecomputation, R::BundleRecomputation, "BundleRecomputation", F::PropagationChain),
    (
        C::BridgeBundleRecomputation,
        R::BridgeBundleRecomputation,
        "BridgeBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::CyclicPosteriorBundleRecomputation,
        R::CyclicPosteriorBundleRecomputation,
        "CyclicPosteriorBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::CyclicPosteriorNodeBundleRecomputation,
        R::CyclicPosteriorNodeBundleRecomputation,
        "CyclicPosteriorNodeBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::TrialIpwNodeBundleRecomputation,
        R::TrialIpwNodeBundleRecomputation,
        "TrialIpwNodeBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::GraphImputationTransportNodeBundleRecomputation,
        R::GraphImputationTransportNodeBundleRecomputation,
        "GraphImputationTransportNodeBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::JointTransportEffectNodeBundleRecomputation,
        R::JointTransportEffectNodeBundleRecomputation,
        "JointTransportEffectNodeBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::JointTransportPredictiveNodeDecisionBundleRecomputation,
        R::JointTransportPredictiveNodeDecisionBundleRecomputation,
        "JointTransportPredictiveNodeDecisionBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::TrialIpwPredictiveNodeDecisionBundleRecomputation,
        R::TrialIpwPredictiveNodeDecisionBundleRecomputation,
        "TrialIpwPredictiveNodeDecisionBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::CyclicTargetFeedbackNodeDecisionBundleRecomputation,
        R::CyclicTargetFeedbackNodeDecisionBundleRecomputation,
        "CyclicTargetFeedbackNodeDecisionBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::GraphImputationPredictiveNodeDecisionBundleRecomputation,
        R::GraphImputationPredictiveNodeDecisionBundleRecomputation,
        "GraphImputationPredictiveNodeDecisionBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::AssessedBridgePosteriorNodeDecisionBundleRecomputation,
        R::AssessedBridgePosteriorNodeDecisionBundleRecomputation,
        "AssessedBridgePosteriorNodeDecisionBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::SequentialPolicyReliabilityNodeBundleRecomputation,
        R::SequentialPolicyReliabilityNodeBundleRecomputation,
        "SequentialPolicyReliabilityNodeBundleRecomputation",
        F::PropagationChain,
    ),
    (
        C::JointComponentPosteriorDecision,
        R::JointComponentPosteriorDecision,
        "JointComponentPosteriorDecision",
        F::PropagationChain,
    ),
    (
        C::BoundedComponentPosteriorDecision,
        R::BoundedComponentPosteriorDecision,
        "BoundedComponentPosteriorDecision",
        F::PropagationChain,
    ),
    (
        C::PosteriorDecisionBundleRecomputation,
        R::PosteriorDecisionBundleRecomputation,
        "PosteriorDecisionBundleRecomputation",
        F::PropagationChain,
    ),
    (C::GraphImputation, R::GraphImputation, "GraphImputation", F::GraphImputation),
    (
        C::GraphImputationEffectDecision,
        R::GraphImputationEffectDecision,
        "GraphImputationEffectDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationBernoulliPredictiveWorldDecision,
        R::GraphImputationBernoulliPredictiveWorldDecision,
        "GraphImputationBernoulliPredictiveWorldDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationGaussianTargetPredictiveWorldDecision,
        R::GraphImputationGaussianTargetPredictiveWorldDecision,
        "GraphImputationGaussianTargetPredictiveWorldDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationBernoulliPredictiveClusterBootstrapDecision,
        R::GraphImputationBernoulliPredictiveClusterBootstrapDecision,
        "GraphImputationBernoulliPredictiveClusterBootstrapDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationGaussianPredictiveClusterBootstrapDecision,
        R::GraphImputationGaussianPredictiveClusterBootstrapDecision,
        "GraphImputationGaussianPredictiveClusterBootstrapDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationTransportDecision,
        R::GraphImputationTransportDecision,
        "GraphImputationTransportDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationTransportClusterBootstrapDecision,
        R::GraphImputationTransportClusterBootstrapDecision,
        "GraphImputationTransportClusterBootstrapDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationConfoundedTransportDecision,
        R::GraphImputationConfoundedTransportDecision,
        "GraphImputationConfoundedTransportDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationConfoundedTransportClusterBootstrapDecision,
        R::GraphImputationConfoundedTransportClusterBootstrapDecision,
        "GraphImputationConfoundedTransportClusterBootstrapDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationKernelTransportDecision,
        R::GraphImputationKernelTransportDecision,
        "GraphImputationKernelTransportDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationKernelTransportClusterBootstrapDecision,
        R::GraphImputationKernelTransportClusterBootstrapDecision,
        "GraphImputationKernelTransportClusterBootstrapDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationConfoundedKernelTransportClusterBootstrapDecision,
        R::GraphImputationConfoundedKernelTransportClusterBootstrapDecision,
        "GraphImputationConfoundedKernelTransportClusterBootstrapDecision",
        F::GraphImputation,
    ),
    (
        C::GraphImputationMarTransportDecision,
        R::GraphImputationMarTransportDecision,
        "GraphImputationMarTransportDecision",
        F::GraphImputation,
    ),
    (
        C::CategoricalPosteriorCandidate,
        R::CategoricalPosteriorCandidate,
        "CategoricalPosteriorCandidate",
        F::CategoricalPosteriorCandidate,
    ),
    (C::MnarFixedShift, R::MnarFixedShift, "MnarFixedShift", F::DesignBased),
    (C::LatentSoftIntervention, R::LatentSoftIntervention, "LatentSoftIntervention", F::Structural),
    (C::EventProcess, R::EventProcess, "EventProcess", F::Structural),
    (C::JointPanelResponse, R::JointPanelResponse, "JointPanelResponse", F::JointPanel),
    (C::JointPanelTimeVarying, R::JointPanelTimeVarying, "JointPanelTimeVarying", F::JointPanel),
    (C::JointPanelImputation, R::JointPanelImputation, "JointPanelImputation", F::JointPanel),
    (
        C::JointPanelProgramSearch,
        R::JointPanelProgramSearch,
        "JointPanelProgramSearch",
        F::JointPanel,
    ),
    (
        C::JointPanelProgramSelection,
        R::JointPanelProgramSelection,
        "JointPanelProgramSelection",
        F::JointPanel,
    ),
    (C::JointPanelExperiment, R::JointPanelExperiment, "JointPanelExperiment", F::JointPanel),
    (C::JointDesign, R::JointDesign, "JointDesign", F::JointPanel),
    (C::DiscoveryValidation, R::DiscoveryValidation, "DiscoveryValidation", F::Discovery),
    (C::GraphFalsification, R::GraphFalsification, "GraphFalsification", F::Structural),
    (
        C::TransportedPathSpecific,
        R::TransportedPathSpecific,
        "TransportedPathSpecific",
        F::Structural,
    ),
    (C::FittedModelAttribution, R::FittedModelAttribution, "FittedModelAttribution", F::Structural),
    (C::FittedModelSimulation, R::FittedModelSimulation, "FittedModelSimulation", F::Structural),
    (C::LocalCyclicCallback, R::LocalCyclicCallback, "LocalCyclicCallback", F::Cyclic),
    (
        C::SuppliedClaimDecision,
        R::SuppliedClaimDecision,
        "SuppliedClaimDecision",
        F::PropagationChain,
    ),
    (
        C::IdentifiedSetDecision,
        R::IdentifiedSetDecision,
        "IdentifiedSetDecision",
        F::PropagationChain,
    ),
    (C::ContractLawDecision, R::ContractLawDecision, "ContractLawDecision", F::PropagationChain),
    (
        C::DecisionFunctionalValue,
        R::DecisionFunctionalValue,
        "DecisionFunctionalValue",
        F::PropagationChain,
    ),
    (
        C::SupportedInputDecision,
        R::SupportedInputDecision,
        "SupportedInputDecision",
        F::PropagationChain,
    ),
    (C::SensitivityDecision, R::SensitivityDecision, "SensitivityDecision", F::PropagationChain),
    (C::InverseQueryDecision, R::InverseQueryDecision, "InverseQueryDecision", F::PropagationChain),
    (C::DecisionDrawsRanking, R::DecisionDrawsRanking, "DecisionDrawsRanking", F::PropagationChain),
    (C::PriorSignalAdaptation, R::PriorSignalAdaptation, "PriorSignalAdaptation", F::Planning),
    (C::EvidenceObligations, R::EvidenceObligations, "EvidenceObligations", F::Planning),
    (C::FiniteActionDecision, R::FiniteActionDecision, "FiniteActionDecision", F::PropagationChain),
    (C::AdaptivePolicySearch, R::AdaptivePolicySearch, "AdaptivePolicySearch", F::Planning),
    (C::IdentificationRepair, R::IdentificationRepair, "IdentificationRepair", F::Planning),
    (C::StudyAdditionPlanning, R::StudyAdditionPlanning, "StudyAdditionPlanning", F::Planning),
    (
        C::ZTransportEvidencePlanning,
        R::ZTransportEvidencePlanning,
        "ZTransportEvidencePlanning",
        F::Planning,
    ),
    (C::DesignRanking, R::DesignRanking, "DesignRanking", F::Planning),
    (C::ChoiceStudyRanking, R::ChoiceStudyRanking, "ChoiceStudyRanking", F::Planning),
    (
        C::StructureIdentification,
        R::StructureIdentification,
        "StructureIdentification",
        F::Planning,
    ),
    (
        C::ClassicalTransportIdentification,
        R::ClassicalTransportIdentification,
        "ClassicalTransportIdentification",
        F::DesignBased,
    ),
    (
        C::ZTransportIdentification,
        R::ZTransportIdentification,
        "ZTransportIdentification",
        F::DesignBased,
    ),
    (
        C::MzTransportIdentification,
        R::MzTransportIdentification,
        "MzTransportIdentification",
        F::DesignBased,
    ),
    (
        C::MixedSourceIdentification,
        R::MixedSourceIdentification,
        "MixedSourceIdentification",
        F::DesignBased,
    ),
    (
        C::AdmgConditionalTransportIdentification,
        R::AdmgConditionalTransportIdentification,
        "AdmgConditionalTransportIdentification",
        F::DesignBased,
    ),
    (
        C::CounterfactualTransportIdentification,
        R::CounterfactualTransportIdentification,
        "CounterfactualTransportIdentification",
        F::DesignBased,
    ),
    (
        C::ObservationRecoveryIdentification,
        R::ObservationRecoveryIdentification,
        "ObservationRecoveryIdentification",
        F::DesignBased,
    ),
    (
        C::ChainRecoveryIdentification,
        R::ChainRecoveryIdentification,
        "ChainRecoveryIdentification",
        F::DesignBased,
    ),
    (C::DescriptiveComparison, R::DescriptiveComparison, "DescriptiveComparison", F::DesignBased),
    (C::MatchedCaseControl, R::MatchedCaseControl, "MatchedCaseControl", F::DesignBased),
    (
        C::TemporalLinearPrediction,
        R::TemporalLinearPrediction,
        "TemporalLinearPrediction",
        F::Associational,
    ),
    (C::SampleSplitEffect, R::SampleSplitEffect, "SampleSplitEffect", F::Structural),
    (
        C::SampleSplitParentSetEffect,
        R::SampleSplitParentSetEffect,
        "SampleSplitParentSetEffect",
        F::Structural,
    ),
    (
        C::NestedMarkovFisherInterval,
        R::NestedMarkovFisherInterval,
        "NestedMarkovFisherInterval",
        F::Structural,
    ),
    (C::NestedMarkovPosterior, R::NestedMarkovPosterior, "NestedMarkovPosterior", F::Structural),
    (C::LearnedJointTransport, R::LearnedJointTransport, "LearnedJointTransport", F::Structural),
    (C::JointBayesianTransport, R::JointBayesianTransport, "JointBayesianTransport", F::Structural),
    (
        C::SharedScenarioCovariance,
        R::SharedScenarioCovariance,
        "SharedScenarioCovariance",
        F::Structural,
    ),
    (
        C::ExternalFeedbackEquilibrium,
        R::ExternalFeedbackEquilibrium,
        "ExternalFeedbackEquilibrium",
        F::Cyclic,
    ),
    (
        C::SampledCounterfactualTransport,
        R::SampledCounterfactualTransport,
        "SampledCounterfactualTransport",
        F::Structural,
    ),
    (
        C::SampledCounterfactualTransportInterval,
        R::SampledCounterfactualTransportInterval,
        "SampledCounterfactualTransportInterval",
        F::Structural,
    ),
    (
        C::SampledObservationRecovery,
        R::SampledObservationRecovery,
        "SampledObservationRecovery",
        F::Structural,
    ),
    (
        C::MeasuredSampledObservationRecovery,
        R::MeasuredSampledObservationRecovery,
        "MeasuredSampledObservationRecovery",
        F::Structural,
    ),
    (
        C::ChainObservationRecovery,
        R::ChainObservationRecovery,
        "ChainObservationRecovery",
        F::Structural,
    ),
    (
        C::ExposureMisclassificationCorrection,
        R::ExposureMisclassificationCorrection,
        "ExposureMisclassificationCorrection",
        F::Associational,
    ),
    (
        C::SoftExposureCorrection,
        R::SoftExposureCorrection,
        "SoftExposureCorrection",
        F::Associational,
    ),
    (
        C::DetectionDependentPrevalenceCorrection,
        R::DetectionDependentPrevalenceCorrection,
        "DetectionDependentPrevalenceCorrection",
        F::Associational,
    ),
    (
        C::RepeatCodingsLatentClass,
        R::RepeatCodingsLatentClass,
        "RepeatCodingsLatentClass",
        F::Associational,
    ),
    (
        C::MeasuredExposureImputation,
        R::MeasuredExposureImputation,
        "MeasuredExposureImputation",
        F::PropagationChain,
    ),
    (
        C::MeasuredExposurePooling,
        R::MeasuredExposurePooling,
        "MeasuredExposurePooling",
        F::PropagationChain,
    ),
    (
        C::PowerPriorBetaSelection,
        R::PowerPriorBetaSelection,
        "PowerPriorBetaSelection",
        F::Associational,
    ),
    (
        C::PowerPriorGammaSelection,
        R::PowerPriorGammaSelection,
        "PowerPriorGammaSelection",
        F::Associational,
    ),
    (
        C::PowerPriorLinearSelection,
        R::PowerPriorLinearSelection,
        "PowerPriorLinearSelection",
        F::Associational,
    ),
    (C::ZTransportEstimate, R::ZTransportEstimate, "ZTransportEstimate", F::DesignBased),
    (
        C::ExternalPriorComposition,
        R::ExternalPriorComposition,
        "ExternalPriorComposition",
        F::Associational,
    ),
    (
        C::ExternalPriorTransport,
        R::ExternalPriorTransport,
        "ExternalPriorTransport",
        F::Associational,
    ),
    (C::TransportResponseGrid, R::TransportResponseGrid, "TransportResponseGrid", F::Structural),
    (C::TransportScenarioSet, R::TransportScenarioSet, "TransportScenarioSet", F::Structural),
    (
        C::MeasurementModelTransport,
        R::MeasurementModelTransport,
        "MeasurementModelTransport",
        F::Structural,
    ),
    (
        C::PooledDifferentialTransport,
        R::PooledDifferentialTransport,
        "PooledDifferentialTransport",
        F::Structural,
    ),
    (
        C::TransportedCohortEffect,
        R::TransportedCohortEffect,
        "TransportedCohortEffect",
        F::Structural,
    ),
    (C::DoseGridBiasAware, R::DoseGridBiasAware, "DoseGridBiasAware", F::DesignBased),
    (
        C::SmoothedDoseLearnerTransport,
        R::SmoothedDoseLearnerTransport,
        "SmoothedDoseLearnerTransport",
        F::DesignBased,
    ),
    (C::ExternalFeedbackFit, R::ExternalFeedbackFit, "ExternalFeedbackFit", F::Cyclic),
    (
        C::AdaptiveResultInterval,
        R::AdaptiveResultInterval,
        "AdaptiveResultInterval",
        F::DesignBased,
    ),
    (C::ChoiceResultInterval, R::ChoiceResultInterval, "ChoiceResultInterval", F::DesignBased),
    (C::CohortMeanInterval, R::CohortMeanInterval, "CohortMeanInterval", F::Structural),
    (C::CountFittedInterval, R::CountFittedInterval, "CountFittedInterval", F::Structural),
    (
        C::TemporalDependentInterval,
        R::TemporalDependentInterval,
        "TemporalDependentInterval",
        F::Temporal,
    ),
    (
        C::MatchedDesignSensitivity,
        R::MatchedDesignSensitivity,
        "MatchedDesignSensitivity",
        F::DesignBased,
    ),
    (
        C::SuppliedEstimateEValue,
        R::SuppliedEstimateEValue,
        "SuppliedEstimateEValue",
        F::PropagationChain,
    ),
    (
        C::ZTransportMechanismSensitivity,
        R::ZTransportMechanismSensitivity,
        "ZTransportMechanismSensitivity",
        F::Structural,
    ),
    (
        C::DiscoveredClassTransport,
        R::DiscoveredClassTransport,
        "DiscoveredClassTransport",
        F::DesignBased,
    ),
    (
        C::PerPopulationStructureSearch,
        R::PerPopulationStructureSearch,
        "PerPopulationStructureSearch",
        F::Discovery,
    ),
    (
        C::TrialAipwLearnerTransport,
        R::TrialAipwLearnerTransport,
        "TrialAipwLearnerTransport",
        F::DesignBased,
    ),
    (C::JointCellsLearnerFit, R::JointCellsLearnerFit, "JointCellsLearnerFit", F::DesignBased),
    (C::MarkedEventProcess, R::MarkedEventProcess, "MarkedEventProcess", F::Structural),
    (C::HawkesExcitation, R::HawkesExcitation, "HawkesExcitation", F::Structural),
    (C::HawkesParameterBridge, R::HawkesParameterBridge, "HawkesParameterBridge", F::Structural),
    (C::TradeoffLawDecision, R::TradeoffLawDecision, "TradeoffLawDecision", F::PropagationChain),
    (
        C::TradeoffCertifiedLawDecision,
        R::TradeoffCertifiedLawDecision,
        "TradeoffCertifiedLawDecision",
        F::PropagationChain,
    ),
    (
        C::TradeoffStructuralDecision,
        R::TradeoffStructuralDecision,
        "TradeoffStructuralDecision",
        F::PropagationChain,
    ),
    (
        C::TradeoffCertifiedStructuralDecision,
        R::TradeoffCertifiedStructuralDecision,
        "TradeoffCertifiedStructuralDecision",
        F::PropagationChain,
    ),
    (
        C::PreferenceCompatibleSet,
        R::PreferenceCompatibleSet,
        "PreferenceCompatibleSet",
        F::Associational,
    ),
    (C::PreferenceChoiceModel, R::PreferenceChoiceModel, "PreferenceChoiceModel", F::Associational),
    (
        C::PreferenceAttachedDecision,
        R::PreferenceAttachedDecision,
        "PreferenceAttachedDecision",
        F::PropagationChain,
    ),
    (
        C::CertifiedPreferenceAttachedDecision,
        R::CertifiedPreferenceAttachedDecision,
        "CertifiedPreferenceAttachedDecision",
        F::PropagationChain,
    ),
    (
        C::ContractAttachedDecision,
        R::ContractAttachedDecision,
        "ContractAttachedDecision",
        F::PropagationChain,
    ),
    (
        C::JointScenarioAttachedDecision,
        R::JointScenarioAttachedDecision,
        "JointScenarioAttachedDecision",
        F::PropagationChain,
    ),
    (C::EvsiAttachedDecision, R::EvsiAttachedDecision, "EvsiAttachedDecision", F::PropagationChain),
];

#[test]
fn every_program_family_keeps_its_pinned_names_route_and_family() {
    for &(cell, route, name, family) in PINNED {
        assert_eq!(cell.as_str(), name, "program cell name of {cell:?}");
        assert_eq!(route.as_str(), name, "plan route name of {route:?}");
        assert_eq!(cell.plan_route(), route, "plan route of {cell:?}");
        assert_eq!(route.family(), family, "family of {route:?}");
        assert_eq!(cell.cyclic_family(), None, "{name} is not a cyclic family");
        assert_eq!(cell.planned_design(), None, "{name} is not an experiment plan");
        assert_eq!(cell.chain_reading(), None, "{name} is not a propagation chain");
    }
}

#[test]
fn the_generated_family_list_is_exactly_the_pinned_list() {
    let generated: Vec<C> = C::FAMILIES.iter().map(|&(cell, _)| cell).collect();
    let pinned: Vec<C> = PINNED.iter().map(|&(cell, ..)| cell).collect();
    assert_eq!(generated, pinned, "the table and the pinned list name the same families in order");

    let mut names = BTreeSet::new();
    for &(_, _, name, _) in PINNED {
        assert!(names.insert(name), "{name} names one program family");
    }
    for &(cell, unit) in C::FAMILIES {
        assert_eq!(unit, unit.trim(), "support unit key of {cell:?} carries no padding");
    }
}

#[test]
fn the_named_cells_keep_their_pinned_names_route_and_family() {
    let cyclic = C::CyclicFamily("system");
    assert_eq!(cyclic.as_str(), "CyclicFamily");
    assert_eq!(cyclic.plan_route(), R::CyclicFamily("system"));
    assert_eq!(R::CyclicFamily("system").as_str(), "CyclicFamily");
    assert_eq!(R::CyclicFamily("system").family(), F::Cyclic);
    assert_eq!(cyclic.cyclic_family(), Some("system"));
    assert_eq!(cyclic.planned_design(), None);
    assert_eq!(cyclic.chain_reading(), None);

    let planning = C::Planning("cluster");
    assert_eq!(planning.as_str(), "ExperimentPlanning");
    assert_eq!(planning.plan_route(), R::Planning("cluster"));
    assert_eq!(R::Planning("cluster").as_str(), "ExperimentPlanning");
    assert_eq!(R::Planning("cluster").family(), F::Planning);
    assert_eq!(planning.cyclic_family(), None);
    assert_eq!(planning.planned_design(), Some("cluster"));
    assert_eq!(planning.chain_reading(), None);

    let chain = C::PropagationChain("spread");
    assert_eq!(chain.as_str(), "PropagationChain");
    assert_eq!(chain.plan_route(), R::PropagationChain("spread"));
    assert_eq!(R::PropagationChain("spread").as_str(), "PropagationChain");
    assert_eq!(R::PropagationChain("spread").family(), F::PropagationChain);
    assert_eq!(chain.cyclic_family(), None);
    assert_eq!(chain.planned_design(), None);
    assert_eq!(chain.chain_reading(), Some("spread"));
}

/// The routes that run under no support unit, each with the reason no record owns it. Every
/// program family and named route is bound by a `[[route]]` row of
/// `parity/compiled_units.toml` now, so none is listed; a route added without a unit fails here
/// until it is bound or listed (with an `[[unbound]]` row) together.
const UNBOUND: &[&str] = &[];

/// The `lowering` of every `[[unbound]]` row of `parity/compiled_units.toml`.
fn listed_unbound() -> BTreeSet<&'static str> {
    let text = include_str!("../../../../../parity/compiled_units.toml");
    let mut inside = false;
    let mut listed = BTreeSet::new();
    for line in text.lines() {
        if line.starts_with('[') {
            inside = line.trim() == "[[unbound]]";
        } else if inside && let Some(rest) = line.strip_prefix("lowering = \"") {
            listed.extend(rest.split('"').next());
        }
    }
    listed
}

#[test]
fn every_program_family_and_named_route_runs_under_a_unit_or_is_listed_unbound() {
    let named = [C::CyclicFamily("system"), C::Planning("cluster"), C::PropagationChain("spread")];
    let families = C::FAMILIES.iter().map(|&(cell, column)| (cell.as_str(), Some(column)));
    let routes = families.chain(named.iter().map(|cell| (cell.as_str(), None)));
    let mut seen = BTreeSet::new();
    for (route, column) in routes {
        seen.insert(route);
        let bound = unit_of(route);
        if let Some(column) = column {
            assert_eq!(
                bound.map_or("", UnitId::as_str),
                column,
                "{route}: the unit column and the generated table name the same unit"
            );
        }
        if UNBOUND.contains(&route) {
            assert!(bound.is_none(), "{route} is listed unbound but runs under {bound:?}");
        } else {
            assert!(
                bound.is_some(),
                "{route} runs under no support unit and is not listed unbound"
            );
        }
    }
    let listed: BTreeSet<&str> = UNBOUND.iter().copied().collect();
    assert_eq!(listed.len(), UNBOUND.len(), "a route is listed unbound twice");
    assert!(listed.is_subset(&seen), "an unbound entry names no program family or named route");
    assert_eq!(listed, listed_unbound(), "the unbound list and parity/compiled_units.toml agree");
}
