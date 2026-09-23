//! Design ranking results and diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{MonteCarloBudget, MonteCarloError};

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

/// How a candidate's score was computed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScoreEvaluation {
    /// The implemented functional evaluated exactly (finite enumeration or closed
    /// form); `monte_carlo.stderr` is zero.
    Exact,
    /// A Monte Carlo estimate of the implemented functional; `monte_carlo` carries
    /// its standard error.
    MonteCarlo,
}

impl ScoreEvaluation {
    /// Stable lowercase label (`exact` / `monte_carlo`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::MonteCarlo => "monte_carlo",
        }
    }
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
    /// Whether this candidate is not separated from the next-ranked one: the gap between
    /// their mean scores is within the sequentially adjusted critical value times the
    /// standard error of the *paired* per-draw difference (shared-CRN replicates). An exact
    /// tie (zero gap) is always uncertain; the last-ranked candidate has no neighbour and is
    /// `false`.
    pub rank_uncertain: bool,
    /// Mathematics actually scored (see [`crate::objective::DesignObjective::implemented_functional`]).
    pub implemented_functional: Arc<str>,
    /// Whether `score` is the exact value of the functional or a Monte Carlo estimate.
    pub evaluation: ScoreEvaluation,
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
    /// Adaptive MC stopped before `max_batches`.
    ///
    /// After every batch from `min_batches` on, each adjacent pair among the top three is
    /// checked: sampling stops once every such pair either has an adjusted CI half-width
    /// within `rank_uncertainty_threshold` or is separated by more than that half-width. The
    /// half-width uses `z = Φ⁻¹(1 − 0.025 / looks)` (Bonferroni over the batch counts at
    /// which the rule is evaluated) and the standard error of the paired difference, so
    /// repeated looking does not inflate the false-separation rate; `rank_uncertain` uses
    /// the same `z`.
    pub early_stopped: bool,
}
