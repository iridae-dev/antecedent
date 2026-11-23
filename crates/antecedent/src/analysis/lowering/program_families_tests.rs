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
