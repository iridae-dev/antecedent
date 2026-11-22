//! Lower an independent experiment that conditions a retained joint panel or design law, and
//! the constrained decision and next-study value read from the conditioned law.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CancellationPoint, CanonicalWriter, EvidenceClaims, FrozenFitReuse, Operator,
    OperatorDeclaration, OperatorEffect, OperatorGraph, OperatorPorts, OperatorRole,
    PhysicalEffect, PhysicalOperator, SemanticDigest,
};

use super::design::refuse;
use crate::analysis::joint_panel::compiled::support_status;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, JointPanelLoweredParts,
    JointPanelRoute, LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramIdentityScope, ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::support::CellStatus;

/// What an experiment program evaluates instead of one quoted design matrix.
const DESIGN_REASON: &str =
    "importance conditioning of the retained joint draws; the original fits stay frozen";

/// The operation a joint-panel experiment program compiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExperimentOperation {
    /// Condition a retained panel law on one independent experiment, including the update
    /// for a declared future study.
    Condition,
    /// Condition a retained panel law jointly on several retained experiments.
    JointCondition,
    /// Condition a retained joint-design law on an experiment.
    LawCondition,
    /// Condition a native joint-design law on an experiment.
    NativeCondition,
    /// Condition a retained panel law on compiled excursion evidence.
    Excursion,
    /// Choose an action from a conditioned law under a constrained decision contract.
    Decide,
    /// Rank next studies or samples by their expected information value.
    RankNext,
}

impl ExperimentOperation {
    /// Stable `snake_case` name.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Condition => "condition",
            Self::JointCondition => "joint_condition",
            Self::LawCondition => "law_condition",
            Self::NativeCondition => "native_condition",
            Self::Excursion => "excursion",
            Self::Decide => "decide",
            Self::RankNext => "rank_next",
        }
    }

    /// Whether the operation conditions a retained law on new experiment evidence, rather
    /// than reading a law an earlier update conditioned.
    const fn updates(self) -> bool {
        !matches!(self, Self::Decide | Self::RankNext)
    }

    /// The graph class of the law the operation reads: a fitted panel model or a design law.
    const fn graph_class(self) -> &'static str {
        match self {
            Self::LawCondition | Self::NativeCondition => "design_native",
            Self::Condition
            | Self::JointCondition
            | Self::Excursion
            | Self::Decide
            | Self::RankNext => "joint_panel_model",
        }
    }

    /// Name of the evaluator that realizes the operation over the retained aligned draws.
    const fn evaluator(self) -> &'static str {
        match self {
            Self::Condition | Self::LawCondition | Self::NativeCondition | Self::Excursion => {
                "frozen_joint_importance_v1"
            }
            Self::JointCondition => "frozen_joint_importance_joint_gaussian_v1",
            Self::Decide => "weighted_law_decision_v1",
            Self::RankNext => "weighted_preposterior_v1",
        }
    }

    /// The claim an execution of the operation publishes.
    const fn claim(self) -> ClaimDemand {
        match self {
            Self::Decide => ClaimDemand {
                uncertainty: UncertaintyReading::Posterior,
                kind: Some(ClaimKind::PolicyValue),
            },
            Self::RankNext => ClaimDemand {
                uncertainty: UncertaintyReading::MonteCarlo,
                kind: Some(ClaimKind::PolicyValue),
            },
            Self::Condition
            | Self::JointCondition
            | Self::LawCondition
            | Self::NativeCondition
            | Self::Excursion => ClaimDemand {
                uncertainty: UncertaintyReading::Posterior,
                kind: Some(ClaimKind::Response),
            },
        }
    }
}

/// Lower the program of an experiment operation (`JointPanelExperiment`).
///
/// The program is a child of `parent`: the panel response or joint design whose law is
/// conditioned, or the conditioned law a decision or ranking reads. The six layers:
///
/// 1. the parent's scientific target, as conditioned by the experiment: a propagated
///    functional whose query binds the operation, `parent` and `experiment`;
/// 2. identification carried from the parent; updating does not discharge an identification
///    obligation, so the program is conditional on the parent model together with the
///    declared experiment likelihood and mapping;
/// 3. the importance conditioning of the parent's retained joint law, with the original
///    fitting modules frozen; a decision evaluates the conditioned law, and a ranking
///    simulates the conditioning each candidate study would cause;
/// 4. the parent program and the experiment as trial evidence in a chained composition, with
///    `decision` for a decision (its contract, actions and constraints) or a ranking (its
///    information value);
/// 5. the read of `experiment_bytes` and the reweighting of the retained draws;
/// 6. the response claim of an update, or the policy-value claim of a decision or ranking.
///
/// `experiment` is the identity of the independent experiment evidence: its likelihood,
/// mapping and declared origins. For a decision it is the conditioned law the decision reads,
/// and for a ranking that law together with the candidate studies' sampling models and
/// regimes. A ranking evaluates a caller utility, so its program identifies that utility by
/// its declaration only.
///
/// # Errors
/// A program layer the planner refuses to declare.
pub(crate) fn lower(
    operation: ExperimentOperation,
    parent: SemanticDigest,
    experiment: SemanticDigest,
    experiment_bytes: u64,
    decision: Option<DecisionProgram>,
) -> Result<LoweredProgram, CausalError> {
    let route = JointPanelRoute::Experiment;
    let status = support_status(route);
    let mut question = CanonicalWriter::new("antecedent.lowering.joint_panel.experiment.v1");
    question.text(operation.as_str()).bytes(parent.as_bytes()).bytes(experiment.as_bytes());
    let query = question.finish();
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: if matches!(operation, ExperimentOperation::RankNext) {
            ProgramIdentityScope::RuntimeCallbackDeclarationOnly
        } else {
            ProgramIdentityScope::Semantic
        },
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            query,
        },
        plan_id: Arc::from(format!("joint_panel_experiment.{}", operation.as_str())),
        data: DataClassification::Panel,
        graph_class: operation.graph_class(),
        query_variables: Arc::from([]),
        source_digest: Some(*experiment.as_bytes()),
        support_coordinate: None,
        supported: status == CellStatus::Supported,
        support_status: Some(status),
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model: query },
    ));
    let shape = DesignShape::not_design_based(DESIGN_REASON);
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::DeclaredPosterior { evaluator: Arc::from(operation.evaluator()) },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design: shape,
        operators: Arc::from(program_operators(operation)?),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::ChainSource, digest: Some(*parent.as_bytes()) },
            EvidenceSourceRef { role: EvidenceRole::Trial, digest: Some(*experiment.as_bytes()) },
        ]),
        composition: Composition::Chained { stages: if decision.is_some() { 3 } else { 2 } },
        decision,
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from(physical_operators(operation, experiment_bytes)?),
        design: shape,
    };
    Ok(JointPanelLoweredParts {
        route,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim: operation.claim(),
    }
    .finish())
}

/// The statistical operators of `operation`.
///
/// Every operation reuses the parent's frozen fit. An update reads the experiment's
/// sufficient statistic as a new observation, transports it to the parent's target through
/// the declared bridge and conditions the frozen law on it. A ranking conditions the law on
/// each simulated study outcome; a decision only evaluates the law.
fn program_operators(operation: ExperimentOperation) -> Result<Vec<Operator>, CausalError> {
    let frozen = Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::ReuseFrozenFit,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::FitHandle]),
        )
        .with_frozen_fit_reuse(FrozenFitReuse::Allowed),
    )
    .map_err(refuse)?;
    if operation.updates() {
        let read = Operator::new(
            OperatorDeclaration::new(
                OperatorEffect::Read,
                OperatorGraph::Execution,
                OperatorPorts::new([OperatorRole::Population], [OperatorRole::Outcome]),
            )
            .with_evidence_claims(EvidenceClaims::observation_read()),
        )
        .map_err(refuse)?;
        let transport = Operator::new(OperatorDeclaration::new(
            OperatorEffect::Transport,
            OperatorGraph::Causal,
            OperatorPorts::new([OperatorRole::Outcome], [OperatorRole::TransportedLaw]),
        ))
        .map_err(refuse)?;
        let condition = Operator::new(OperatorDeclaration::new(
            OperatorEffect::Condition,
            OperatorGraph::Causal,
            OperatorPorts::new(
                [OperatorRole::FitHandle, OperatorRole::TransportedLaw],
                [OperatorRole::Conditioning],
            ),
        ))
        .map_err(refuse)?;
        return Ok(vec![read, frozen, transport, condition]);
    }
    if matches!(operation, ExperimentOperation::RankNext) {
        let condition = Operator::new(OperatorDeclaration::new(
            OperatorEffect::Condition,
            OperatorGraph::Causal,
            OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::Conditioning]),
        ))
        .map_err(refuse)?;
        return Ok(vec![frozen, condition]);
    }
    Ok(vec![frozen])
}

/// The physical operators of `operation`: the read of `bytes`, the reuse of the frozen draws,
/// and the transport and conditioning an update or ranking runs. A ranking's conditioning
/// checks cancellation at every simulated replicate.
fn physical_operators(
    operation: ExperimentOperation,
    bytes: u64,
) -> Result<Vec<PhysicalOperator>, CausalError> {
    let mut operators = vec![
        PhysicalOperator::declare(PhysicalEffect::Read)
            .estimates(ByteEstimates::absent().with_output(bytes))
            .build()
            .map_err(refuse)?,
        PhysicalOperator::declare(PhysicalEffect::ReuseFrozenFit).build().map_err(refuse)?,
    ];
    if operation.updates() {
        for effect in [PhysicalEffect::Transport, PhysicalEffect::Condition] {
            operators.push(PhysicalOperator::declare(effect).build().map_err(refuse)?);
        }
    } else if matches!(operation, ExperimentOperation::RankNext) {
        operators.push(
            PhysicalOperator::declare(PhysicalEffect::Condition)
                .cancellation_point(CancellationPoint::Resample)
                .build()
                .map_err(refuse)?,
        );
    }
    Ok(operators)
}

#[cfg(test)]
mod tests {
    use antecedent_core::SemanticDigest;

    use super::{ExperimentOperation, lower};
    use crate::planner::lowered::{EvidenceRole, UncertaintyReading};

    const OPERATIONS: [ExperimentOperation; 7] = [
        ExperimentOperation::Condition,
        ExperimentOperation::JointCondition,
        ExperimentOperation::LawCondition,
        ExperimentOperation::NativeCondition,
        ExperimentOperation::Excursion,
        ExperimentOperation::Decide,
        ExperimentOperation::RankNext,
    ];

    #[test]
    fn every_operation_cites_its_parent_and_the_experiment_as_trial_evidence() {
        let parent = SemanticDigest::from_bytes([1; 32]);
        let experiment = SemanticDigest::from_bytes([2; 32]);
        let mut identities = std::collections::BTreeSet::new();
        for operation in OPERATIONS {
            let program = lower(operation, parent, experiment, 64, None).unwrap();
            let sources = program.evidence().sources();
            assert_eq!(sources.len(), 2, "{}", operation.as_str());
            assert_eq!(sources[0].role(), EvidenceRole::ChainSource);
            assert_eq!(sources[0].digest(), Some(*parent.as_bytes()));
            assert_eq!(sources[1].role(), EvidenceRole::Trial);
            assert_eq!(sources[1].digest(), Some(*experiment.as_bytes()));
            assert!(identities.insert(*program.identity().as_bytes()), "{}", operation.as_str());
        }
    }

    #[test]
    fn a_changed_parent_or_experiment_changes_the_program_identity() {
        let parent = SemanticDigest::from_bytes([1; 32]);
        let experiment = SemanticDigest::from_bytes([2; 32]);
        let other = SemanticDigest::from_bytes([3; 32]);
        let operation = ExperimentOperation::Condition;
        let base = lower(operation, parent, experiment, 64, None).unwrap().identity();
        assert_ne!(base, lower(operation, other, experiment, 64, None).unwrap().identity());
        assert_ne!(base, lower(operation, parent, other, 64, None).unwrap().identity());
        // The read quote is execution accounting, outside the program identity.
        assert_eq!(base, lower(operation, parent, experiment, 128, None).unwrap().identity());
    }

    #[test]
    fn a_ranking_reads_its_value_as_monte_carlo_and_an_update_as_a_posterior() {
        let digest = SemanticDigest::from_bytes([4; 32]);
        let rank = lower(ExperimentOperation::RankNext, digest, digest, 0, None).unwrap();
        assert_eq!(rank.claim().uncertainty(), UncertaintyReading::MonteCarlo);
        assert_eq!(rank.semantic_identity(), None);
        let update = lower(ExperimentOperation::Condition, digest, digest, 0, None).unwrap();
        assert_eq!(update.claim().uncertainty(), UncertaintyReading::Posterior);
        assert_eq!(update.semantic_identity(), Some(update.identity()));
    }
}
