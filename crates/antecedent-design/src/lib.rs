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
pub mod decision;
pub mod error;
pub mod objective;
pub mod preposterior;
pub mod ranker;
pub mod result;
pub mod transport_planner;
pub mod z_transport_planner;

pub use candidate::{
    CandidateDesign, DesignCost, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan,
};
pub use decision::{
    AffineUtility, DecisionConstraint, DecisionEvaluation, DecisionProblem, DecisionProblemId,
    Utility, evaluate_decision,
};
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
pub use result::{ConstraintViolation, DesignRanking, RankedCandidate, ScoreEvaluation};
pub use transport_planner::{
    TransportCandidateAssessment, TransportCandidateOutcome, TransportEvidenceCandidate,
    TransportPlanResult, TransportPlanSpec, TransportPlanningError, TransportProposal,
    plan_transport_evidence,
};
pub use z_transport_planner::{
    ZTransportArrival, ZTransportCandidateAssessment, ZTransportCandidateOutcome,
    ZTransportFailureSnapshot, ZTransportFailureSnapshotWire, ZTransportFailureStatus,
    ZTransportPlanResult, ZTransportPlanningError, ZTransportProposal, ZTransportProposalWire,
    ZTransportQueryWire, plan_z_transport_evidence, propose_z_transport_evidence,
    snapshot_z_transport_failure, validate_z_transport_candidate,
};
