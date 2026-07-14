//! Design ranking results and diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{MonteCarloBudget, MonteCarloError};

use crate::candidate::CandidateDesign;

/// Why a candidate was filtered before ranking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintViolation {
    /// Candidate index in the original list.
    pub candidate_index: usize,
    /// Constraint name.
    pub constraint: Arc<str>,
    /// Detail.
    pub detail: Arc<str>,
}

/// Scored candidate with Monte Carlo uncertainty.
#[derive(Clone, Debug)]
pub struct RankedCandidate {
    /// Index into the original candidate slice.
    pub candidate_index: usize,
    /// Candidate design (cloned for convenience).
    pub candidate: CandidateDesign,
    /// Objective score (higher is better for all objectives after signing).
    pub score: f64,
    /// Monte Carlo error on the score.
    pub monte_carlo: MonteCarloError,
    /// Rank position (0 = best); ties broken by lower index when scores overlap within stderr.
    pub rank: usize,
    /// Whether rank is uncertain relative to neighbors given MC error.
    pub rank_uncertain: bool,
}

/// Full ranking output.
#[derive(Clone, Debug)]
pub struct DesignRanking {
    /// Ranked candidates (best first).
    pub ranked: Arc<[RankedCandidate]>,
    /// Candidates filtered by hard constraints (never silently dropped).
    pub violations: Arc<[ConstraintViolation]>,
    /// Compute budget consumed.
    pub budget: MonteCarloBudget,
    /// Adaptive MC stopped early.
    pub early_stopped: bool,
}
