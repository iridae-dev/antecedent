//! Experiment, measurement, and decision primitives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::type_complexity,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::unnecessary_literal_bound
)]

pub mod candidate;
pub mod decision;
pub mod error;
pub mod objective;
pub mod ranker;
pub mod result;

pub use candidate::{
    CandidateDesign, DesignCost, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan,
};
pub use decision::{
    DecisionConstraint, DecisionEvaluation, DecisionProblem, DecisionProblemId, Utility,
    evaluate_decision,
};
pub use error::DesignError;
pub use objective::DesignObjective;
pub use ranker::{
    DecisionRegistry, DesignConstraints, DesignEvaluationContext, DesignRankConfig, DesignRanker,
    EffectWidthContext, InterventionDesignEffect, MeasureColumnSpec, ModelLoglikDraws,
};
pub use result::{ConstraintViolation, DesignRanking, RankedCandidate};
