//! Experiment, measurement, and decision primitives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::type_complexity,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::unnecessary_literal_bound
)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

pub mod candidate;
pub mod composition_boundary;
pub mod composition_bundle;
pub mod decision;
pub mod decision_adapters;
pub mod decision_artifact;
pub mod decision_contract;
pub mod decision_eval;
pub mod decision_refusal;
pub mod decision_robust_artifact;
pub mod decision_robustness;
pub mod decision_structural;
pub mod decision_structural_artifact;
pub mod design_ranking_artifact;
pub mod error;
pub mod evsi;
pub mod inverse_query;
pub mod inverse_query_artifact;
pub mod objective;
pub mod obligation_adapters;
mod plan_common;
pub mod preposterior;
pub mod prior_signal;
pub mod proposal_receipt;
pub mod ranker;
pub mod ranking;
pub mod repair;
pub mod repair_artifact;
pub mod result;
pub mod sensitivity_decision;
pub mod signal;
pub mod study_candidate;
pub mod study_plan_artifact;
pub mod study_planner;
pub mod transport_planner;
pub mod z_transport_planner;

/// The plan the proposals are read from, owned by the identification crate that
/// runs it; re-exported so the facade reaches it through the design crate.
pub use antecedent_identify::{
    StudyPlan, StudyPlanLimits, StudyPlanRoute, StudyPlanStop, StudyProposal, StudySubsetOutcome,
};
pub use candidate::{
    CandidateDesign, DesignCost, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan,
};
pub use decision::{
    AffineUtility, DecisionConstraint, DecisionEvaluation, DecisionProblem, DecisionProblemId,
    Utility, evaluate_decision,
};
pub use decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion,
    DecisionFunctional, HardConstraint, SourceMode, SourceRepresentation, SourceRequirement,
    StructuralPolicy, Tail, UtilityExpr,
};
pub use decision_structural_artifact::StructuralResultArtifact;
pub use error::DesignError;
pub use objective::DesignObjective;
pub use preposterior::{
    BinomialSignal, DecisionPrior, DecisionSignal, GaussianMeanSignal, PreposteriorAnalysis,
};
pub use ranker::{
    DecisionRegistry, DesignConstraints, DesignEvaluationContext, DesignRankConfig, DesignRanker,
    EffectWidthContext, EnvironmentGramSpec, InterventionDesignEffect, MeasureColumnSpec,
    ModelLoglikDraws,
};
pub use repair::{
    BackdoorRepairFamily, FamilyVerdict, RepairClassification, RepairError, RepairFamily,
    RepairLimits, RepairObjective, RepairOutcome, RepairReceipt, RepairReport,
    TransportRepairFamily, repair,
};
pub use repair_artifact::{
    REPAIR_ARTIFACT_KIND, REPAIR_ARTIFACT_VERSION, RepairArtifactError, RepairConsumeLimits,
    RepairFamilyRef, RepairReportArtifact,
};
pub use result::{ConstraintViolation, DesignRanking, RankedCandidate, ScoreEvaluation};
pub use study_candidate::{
    DurableStudyCandidate, ExpectedEvidence, StudyCandidateError, StudyCostDeclaration, StudyKind,
    UnitRules,
};
pub use study_plan_artifact::{
    STUDY_PLAN_ARTIFACT_KIND, STUDY_PLAN_ARTIFACT_VERSION, StudyBaseFailureWire,
    StudyCandidateWire, StudyDerivationWire, StudyPlanArtifactWire, StudyPlanConsumeLimits,
    StudyPlanDataWire, StudyPlanLimitsWire, StudyPlanOutcomeWire, StudyPlanPremisesWire,
    StudyPlanStopWire, StudyProposalWire, StudyRepairWire, StudySubsetWire,
};
pub use study_planner::{
    StudyArrival, StudyArrivalDecision, StudyCandidate, StudyCost, StudyPlanError,
    StudyPlanProposal, StudyPlanResult, plan_studies,
};
pub use transport_planner::{
    TransportCandidateAssessment, TransportCandidateOutcome, TransportEvidenceCandidate,
    TransportPlanResult, TransportPlanSpec, TransportPlanningError, TransportProposal,
    plan_transport_evidence,
};
pub use z_transport_planner::{
    ZTransportArrival, ZTransportCandidateAssessment, ZTransportCandidateOutcome,
    ZTransportFailureSnapshot, ZTransportFailureSnapshotWire, ZTransportFailureStatus,
    ZTransportPlanResult, ZTransportPlanSpec, ZTransportPlanningError, ZTransportProposal,
    ZTransportProposalWire, ZTransportQueryWire, plan_z_transport_evidence,
    propose_z_transport_evidence, snapshot_z_transport_failure, validate_z_transport_candidate,
};
