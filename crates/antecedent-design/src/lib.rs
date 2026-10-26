//! Experiment, measurement, and decision primitives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod candidate;
pub mod candidate_generation;
pub mod cohort_decision_input;
pub mod composition_boundary;
pub mod composition_bundle;
pub mod composition_verifiers;
pub mod confounder_value;
pub mod decision;
pub mod decision_adapters;
pub mod decision_artifact;
pub mod decision_contract;
pub mod decision_eval;
pub mod decision_evaluation;
pub mod decision_refusal;
pub mod decision_reliability;
pub mod decision_robust_artifact;
pub mod decision_robustness;
pub mod decision_structural;
pub mod decision_structural_artifact;
pub mod design_ranking_artifact;
pub mod error;
pub mod evidence_plan;
pub mod evidence_plan_artifact;
pub mod evsi;
pub mod failure_routes;
pub mod functional_source;
pub mod identification_repair;
pub mod inverse_query;
pub mod inverse_query_artifact;
pub mod joint_coupling;
pub mod objective;
pub mod obligation_adapters;
mod plan_common;
pub mod planning;
pub mod preposterior;
pub mod prior_signal;
pub mod program_search;
pub mod proposal_receipt;
mod rank_rows;
pub mod ranker;
pub mod ranking;
pub mod repair;
pub mod repair_artifact;
pub mod result;
pub mod rollout_artifact;
pub mod scientific_source;
pub mod sensitivity_decision;
pub mod sequential_evaluation;
pub mod sequential_evaluation_artifact;
pub mod signal;
pub(crate) mod structure_aggregator;
pub mod structure_identification;
pub mod study_candidate;
pub mod study_plan_artifact;
pub mod study_planner;
pub mod transport_planner;
pub mod z_transport_planner;
pub mod z_transport_repair;
pub use z_transport_repair::ZTransportRepairFamily;

/// The plan the proposals are read from, owned by the identification crate that
/// runs it; re-exported so the facade reaches it through the design crate.
pub use antecedent_identify::{
    StudyPlan, StudyPlanLimits, StudyPlanRoute, StudyPlanStop, StudyProposal, StudySubsetOutcome,
};
pub use candidate::{
    CandidateDesign, DesignCost, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan,
};
pub use candidate_generation::{
    CandidateGeneration, CandidateTemplate, StudyDeclarations, UnmatchedObligation,
    UnverifiedStudy, VerifiedUnlocks, generate_candidates, verified_unlocks,
};
pub use confounder_value::{
    ConfounderPriorDeclaration, ConfounderValueError, ConfounderValueFrame, GridUtility,
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
pub use evidence_plan::{
    EvidencePlan, EvidencePlanError, OptionValue, PlanCost, PlanInputs, PlanOption, PlanOutcome,
    PlanRoute, PlanValue, PlanValueBasis, PlanValueRow, RouteBudget, UnverifiedRoute,
};
pub use evidence_plan_artifact::{
    EvidencePlanArtifact, EvidencePlanArtifactError, EvidencePlanReadLimits,
};
pub use failure_routes::{FailureRoute, FailureRoutes, failure_routes};
pub use identification_repair::{
    FrontDoorRepairFamily, HedgeRepairFamily, InstrumentRepairFamily, LatentCause,
    SensitivityFinding, SensitivityMethod, SensitivityRepairFamily, latent_expansion,
};
pub use objective::{DesignObjective, ObjectiveChannel, ObjectiveScorer};
pub use preposterior::{
    BinomialSignal, DecisionPrior, DecisionSignal, GaussianMeanSignal, PreposteriorAnalysis,
};
pub use ranker::{
    DecisionRegistry, DesignConstraints, DesignEvaluationContext, DesignRankConfig, DesignRanker,
    EffectWidthContext, EnvironmentGramSpec, InterventionDesignEffect, MeasureColumnSpec,
    ModelLoglikDraws, ScoringChannels,
};
pub use repair::{
    BackdoorRepairFamily, FailedContract, FamilyVerdict, RepairClassification, RepairError, RepairFamily,
    RepairLimits, RepairObjective, RepairOutcome, RepairReceipt, RepairReport,
    TransportRepairFamily, repair,
};
pub use repair_artifact::{
    RepairArtifactError, RepairConsumeLimits, RepairFamilyRef, RepairReadLimits,
    RepairReportArtifact,
};
pub use result::{ConstraintViolation, RankedCandidate, ScoreEvaluation, ScoredDesignRanking};
pub use structure_identification::{
    GraphPosteriorSample, StructureIdentificationError, identified_under_intervention, mutilate,
    structure_posterior,
};
pub use study_candidate::{
    DurableStudyCandidate, ExpectedEvidence, StudyCandidateError, StudyCostDeclaration, StudyKind,
    UnitRules,
};
pub use study_plan_artifact::{
    StudyBaseFailureWire, StudyCandidateWire, StudyDerivationWire, StudyPlanArtifactWire,
    StudyPlanConsumeLimits, StudyPlanDataWire, StudyPlanLimitsWire, StudyPlanOutcomeWire,
    StudyPlanPremisesWire, StudyPlanStopWire, StudyProposalWire, StudyRepairWire, StudySubsetWire,
};
pub use study_planner::{
    StudyArrival, StudyArrivalDecision, StudyCandidate, StudyCost, StudyPlanError,
    StudyPlanProposal, StudyPlanResult, plan_studies,
};
pub use transport_planner::TransportEvidenceCandidate;
pub use z_transport_planner::{
    ZTransportArrival, ZTransportCandidateAssessment, ZTransportCandidateOutcome,
    ZTransportFailureSnapshot, ZTransportFailureSnapshotWire, ZTransportFailureStatus,
    ZTransportPlanResult, ZTransportPlanSpec, ZTransportPlanningError, ZTransportProposal,
    ZTransportProposalWire, ZTransportQueryWire, plan_z_transport_evidence,
    snapshot_z_transport_failure,
};

pub mod adaptive_trial_transport;
pub mod decision_transported_source;
pub mod inverse_forward_source;
pub mod inverse_observational_scenarios;
/// Independently consumed original source diagnostics and semantic mappings.
pub mod source_evidence;
pub mod source_projection_artifact;
pub mod temporal_policy_transport;
pub mod transport_sensitivity;
pub mod transported_bridge;
pub mod transported_decision_artifact;
pub mod transported_policy;
pub mod transported_synthesis;
