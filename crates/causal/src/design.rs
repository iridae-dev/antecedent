//! experiment / measurement design facade helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::ExecutionContext;
use causal_design::{
    CandidateDesign, DesignEvaluationContext, DesignObjective, DesignRanker, DesignRanking,
};

use crate::error::AnalysisError;

/// Rank candidate designs under an objective (DESIGN.md §19).
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
) -> Result<DesignRanking, AnalysisError>
where
    A: Clone,
    O: Clone,
{
    ranker.rank(objective, candidates, eval, ctx).map_err(AnalysisError::from)
}
