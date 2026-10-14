//! Ranking and optimality statements over a finite candidate set.
//!
//! A statement here is a Monte Carlo estimate of a probability *under the supplied law*
//! over the candidates *as supplied*: that an action is best, in the top `k`, or better
//! than a baseline. It is labelled with its paradigm (posterior probability, or
//! sampling frequency of a simulated design, or Monte Carlo noise only) and as an
//! uncalibrated diagnostic. It is not a confidence statement about the candidate set
//! the search could have produced, and no interval is built from it.
//!
//! Draws are paired by index only when the carrier declared one shared simulation; that
//! is checked against the stream identities by [`DecisionDraws::validate`] and licensed
//! by the propagation rule (a nonlinear functional of independently drawn candidates is
//! refused, never paired by index). Ties split their probability equally, so the best
//! probabilities of all candidates sum to one at every draw.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

use super::ReliabilityError;
use super::draws::{CandidateCoupling, CandidateOrigin, DecisionDraws, SourceParadigm};
use super::mc_error::{BatchPlan, Functional, McEstimate, estimate, series_stats};

/// Most candidate-by-draw cells a ranking will hold (two `u32` ranks per cell).
pub const MAX_RANKING_CELLS: usize = 1 << 24;

/// Calibration standing of a statement. There is one value: every ranking and
/// optimality statement is uncalibrated until a coverage record measures it at its
/// coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    /// A Monte Carlo diagnostic with no measured calibration.
    Uncalibrated,
}

/// What the candidate set is, for the purposes of the statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateScope {
    /// The set was fixed before evaluation; the statement is over exactly this set.
    EnumeratedSet,
    /// The set is the retained part of an adaptive search; the statement is over the
    /// retained candidates only and is conditional on the search having kept them.
    AdaptiveRetainedSet,
}

/// The paradigm, standing and scope that travel with every ranking statement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatementLabel {
    /// The law the probabilities are under.
    pub paradigm: SourceParadigm,
    /// Always uncalibrated.
    pub calibration: CalibrationStatus,
    /// How the candidates' draws were coupled.
    pub coupling: CandidateCoupling,
    /// Whether the set was enumerated or adaptively retained.
    pub scope: CandidateScope,
    /// Identity of the candidate set (and its search record).
    pub candidate_set_digest: String,
    /// Candidates in the statement.
    pub n_candidates: u64,
    /// Candidates the search considered (equals `n_candidates` for an enumerated set).
    pub n_considered: u64,
}

/// What to rank.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankingRequest {
    /// Candidate every other candidate is compared with, when asked for.
    pub baseline: Option<String>,
    /// Top-`k` sizes to report, each in `1..=n_candidates`.
    pub top_k: Vec<u32>,
    /// Batch plan for dependent streams.
    pub batch_plan: BatchPlan,
}

impl Default for RankingRequest {
    fn default() -> Self {
        Self { baseline: None, top_k: Vec::new(), batch_plan: BatchPlan::SquareRoot }
    }
}

/// Probability that a candidate is among the best `k`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopKEstimate {
    /// The set size.
    pub k: u32,
    /// The estimate.
    pub estimate: McEstimate,
}

/// One candidate's ranking statements.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRankStatement {
    /// Candidate identity.
    pub id: String,
    /// Expected utility, for reference.
    pub expected_utility: McEstimate,
    /// Probability of being the (tie-shared) best candidate.
    pub probability_best: McEstimate,
    /// Probability of being in the top `k`, for each requested `k`.
    pub probability_top_k: Vec<TopKEstimate>,
    /// Probability of exceeding the baseline (ties count one half); `None` for the
    /// baseline itself or when no baseline was requested.
    pub probability_better_than_baseline: Option<McEstimate>,
}

/// Ranking statements for a finite candidate set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankingReport {
    /// Paradigm, standing and scope.
    pub label: StatementLabel,
    /// Baseline, when requested.
    pub baseline: Option<String>,
    /// Candidate with the largest estimated expected utility (ties: smallest id).
    pub best_by_expected_utility: String,
    /// One entry per candidate, in canonical (id) order.
    pub statements: Vec<CandidateRankStatement>,
}

/// Rank the candidates of a carrier.
///
/// # Errors
/// Anything [`DecisionDraws::validate`] refuses, fewer than two candidates, an unknown
/// baseline, a `k` outside the candidate count, a set larger than
/// [`MAX_RANKING_CELLS`], or cancellation.
pub fn rank(
    draws: &DecisionDraws,
    request: &RankingRequest,
    ctx: &ExecutionContext,
) -> Result<RankingReport, ReliabilityError> {
    draws.validate()?;
    rank_validated(draws, request, ctx)
}

/// [`rank`] for a carrier that has already been validated.
#[allow(
    clippy::too_many_lines,
    reason = "one pass over candidates assembling the per-candidate statements"
)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "candidate counts are bounded by the carrier limits; ties are exact equality of finite draws"
)]
pub(crate) fn rank_validated(
    draws: &DecisionDraws,
    request: &RankingRequest,
    ctx: &ExecutionContext,
) -> Result<RankingReport, ReliabilityError> {
    let (n_candidates, n) = (draws.n_candidates(), draws.n_draws());
    if n_candidates < 2 {
        return Err(ReliabilityError::invalid(
            "ranking_needs_two_candidates",
            "a ranking statement compares at least two candidates",
        ));
    }
    // Rank statements pair the candidates' draws by index: the propagation rule decides
    // whether the declared dependence licenses that.
    draws.license_index_pairing()?;
    if n_candidates.checked_mul(n).is_none_or(|c| c > MAX_RANKING_CELLS) {
        return Err(ReliabilityError::RouteNotSupported {
            message: format!("ranking holds at most {MAX_RANKING_CELLS} candidate-draw cells"),
        });
    }
    let order = draws.canonical_order();
    if let Some(&k) = request.top_k.iter().find(|&&k| k == 0 || k as usize > n_candidates) {
        return Err(ReliabilityError::invalid(
            "top_k",
            format!("k = {k} outside 1..={n_candidates}"),
        ));
    }
    let baseline = match &request.baseline {
        Some(id) => Some(
            order
                .iter()
                .position(|&i| draws.candidates[i].id == *id)
                .ok_or_else(|| ReliabilityError::invalid("baseline", "unknown baseline id"))?,
        ),
        None => None,
    };
    let columns: Vec<&[f64]> =
        order.iter().map(|&i| draws.candidates[i].utilities.as_slice()).collect();
    let (greater, equal) = rank_tables(&columns, n, ctx)?;
    let weights = draws.weights.as_deref();
    let dependence = draws.within_stream;
    let plan = request.batch_plan;
    let probability = |series: &[f64]| -> Result<McEstimate, ReliabilityError> {
        let stats = series_stats(series, weights, dependence, plan)?;
        Ok(McEstimate {
            value: stats.mean,
            standard_error: stats.standard_error,
            effective_sample_size: stats.ess,
            n_draws: n as u64,
            method: stats.method,
        })
    };
    let mut statements = Vec::with_capacity(n_candidates);
    for (j, &source) in order.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(ReliabilityError::Cancelled);
        }
        let g = &greater[j * n..(j + 1) * n];
        let e = &equal[j * n..(j + 1) * n];
        let best: Vec<f64> = (0..n).map(|i| top_share(g[i], e[i], 1)).collect();
        let mut top_k = Vec::with_capacity(request.top_k.len());
        for &k in &request.top_k {
            let series: Vec<f64> = (0..n).map(|i| top_share(g[i], e[i], k)).collect();
            top_k.push(TopKEstimate { k, estimate: probability(&series)? });
        }
        let versus = match baseline {
            Some(b) if b != j => {
                let series: Vec<f64> = columns[j]
                    .iter()
                    .zip(columns[b])
                    .map(|(u, v)| {
                        if u > v {
                            1.0
                        } else if u < v {
                            0.0
                        } else {
                            0.5
                        }
                    })
                    .collect();
                Some(probability(&series)?)
            }
            _ => None,
        };
        let utility = estimate(
            &draws.candidates[source].utilities,
            weights,
            dependence,
            plan,
            &Functional::ExpectedUtility,
        )?
        .estimate;
        statements.push(CandidateRankStatement {
            id: draws.candidates[source].id.clone(),
            expected_utility: utility,
            probability_best: probability(&best)?,
            probability_top_k: top_k,
            probability_better_than_baseline: versus,
        });
    }
    let best_by_expected_utility = statements
        .iter()
        .fold(None::<&CandidateRankStatement>, |best, s| match best {
            Some(b) if b.expected_utility.value >= s.expected_utility.value => Some(b),
            _ => Some(s),
        })
        .map(|s| s.id.clone())
        .ok_or_else(|| ReliabilityError::invalid("ranking_needs_two_candidates", "empty"))?;
    Ok(RankingReport {
        label: StatementLabel {
            paradigm: draws.source,
            calibration: CalibrationStatus::Uncalibrated,
            coupling: draws.coupling,
            scope: match draws.origin {
                CandidateOrigin::Enumerated => CandidateScope::EnumeratedSet,
                CandidateOrigin::AdaptivelyGenerated => CandidateScope::AdaptiveRetainedSet,
            },
            candidate_set_digest: draws.candidate_set_digest(),
            n_candidates: n_candidates as u64,
            n_considered: draws.n_considered(),
        },
        baseline: request.baseline.clone(),
        best_by_expected_utility,
        statements,
    })
}

/// Share of the top-`k` slots a candidate takes at one draw when `greater` candidates
/// are strictly better and `equal` (itself included) tie with it.
fn top_share(greater: u32, equal: u32, k: u32) -> f64 {
    if greater >= k { 0.0 } else { f64::from(equal.min(k - greater)) / f64::from(equal) }
}

/// Per candidate and draw: how many candidates are strictly greater, and how many are
/// equal (itself included). Tables are candidate-major.
#[allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "candidate counts are bounded by the carrier limits; ties are exact equality of finite draws"
)]
fn rank_tables(
    columns: &[&[f64]],
    n: usize,
    ctx: &ExecutionContext,
) -> Result<(Vec<u32>, Vec<u32>), ReliabilityError> {
    let n_candidates = columns.len();
    let mut greater = vec![0_u32; n_candidates * n];
    let mut equal = vec![0_u32; n_candidates * n];
    let mut order: Vec<usize> = (0..n_candidates).collect();
    for i in 0..n {
        if i % 1024 == 0 && ctx.cancellation.is_cancelled() {
            return Err(ReliabilityError::Cancelled);
        }
        order.sort_unstable_by(|&a, &b| columns[b][i].total_cmp(&columns[a][i]));
        let mut start = 0;
        while start < n_candidates {
            let mut end = start + 1;
            while end < n_candidates && columns[order[end]][i] == columns[order[start]][i] {
                end += 1;
            }
            for &j in &order[start..end] {
                greater[j * n + i] = start as u32;
                equal[j * n + i] = (end - start) as u32;
            }
            start = end;
        }
    }
    Ok((greater, equal))
}
