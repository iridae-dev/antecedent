//! experiment / measurement design facade helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use crate::error::CausalError;

pub use antecedent_design::{
    CandidateDesign, ConstraintViolation, DecisionConstraint, DecisionEvaluation, DecisionProblem,
    DecisionProblemId, DesignConstraints, DesignCost, DesignError, DesignEvaluationContext,
    DesignObjective, DesignRankConfig, DesignRanker, DesignRanking, EffectWidthContext,
    EnvironmentGramSpec, EnvironmentPlan, ExperimentPlan, InterventionDesignEffect,
    MeasureColumnSpec, MeasurementPlan, ModelLoglikDraws, RankedCandidate, SamplingPlan, Utility,
    evaluate_decision,
};

/// Rank candidate designs under an objective.
///
/// # Errors
///
/// Propagates design evaluation failures.
pub fn rank_designs<A, O>(
    ranker: &DesignRanker,
    objective: &DesignObjective,
    candidates: &[CandidateDesign],
    eval: &DesignEvaluationContext<'_, A, O>,
    ctx: &ExecutionContext,
) -> Result<DesignRanking, CausalError>
where
    A: Clone,
    O: Clone,
{
    ranker.rank(objective, candidates, eval, ctx).map_err(CausalError::from)
}
