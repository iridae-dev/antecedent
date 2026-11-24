//! Decision value of further choice studies: the entry through which hosts reach the EVSI ranking
//! of [`antecedent_design::choice_evsi`]. The posterior it values comes from the choice inference
//! ([`super::choice_inference::ChoiceInferenceSession::rank_studies`]).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_design::choice_evsi::{
    ChoiceEvsiError, EvsiMethod, MAX_EXACT_COMPOSITIONS, StudyCandidate, StudyRank,
};

/// Rank candidate studies with caller-owned admission and cancellation.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::choice_evsi::rank_studies_with_context`].
///
/// # Errors
/// Invalid inputs, an excessive cumulative work request, cancellation or resource admission failure.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ChoiceStudyRanking route, which calls this kernel function with the same inputs"
)]
pub fn rank_studies_with_context(
    alpha: &[f64],
    values: &[f64],
    candidates: &[antecedent_design::choice_evsi::StudyCandidate],
    seed: u64,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    Vec<antecedent_design::choice_evsi::StudyRank>,
    antecedent_design::choice_evsi::ChoiceEvsiError,
> {
    antecedent_design::choice_evsi::rank_studies_with_context(alpha, values, candidates, seed, ctx)
}

/// Rank candidate studies by the expected value of sample information net of cost.
///
/// `alpha` is the Dirichlet posterior of the shares, as the choice inference reports it.
/// The terminal decision picks the option with the largest `value_k * E[share_k]`. The
/// application owns the candidate designs, the acquisition of observations and the authorization
/// of fieldwork; this computes their decision value under the posterior.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::choice_evsi::rank_studies`].
///
/// # Errors
/// Values or candidates outside their domain.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ChoiceStudyRanking route, which calls this kernel function with the same inputs"
)]
pub fn rank_studies(
    alpha: &[f64],
    values: &[f64],
    candidates: &[antecedent_design::choice_evsi::StudyCandidate],
    seed: u64,
) -> Result<
    Vec<antecedent_design::choice_evsi::StudyRank>,
    antecedent_design::choice_evsi::ChoiceEvsiError,
> {
    antecedent_design::choice_evsi::rank_studies(alpha, values, candidates, seed)
}

/// Rank studies of one new respondent who makes one or two repeated choices.
///
/// Each predictive outcome is a complete respondent count vector. The hierarchical posterior is
/// refitted for every outcome, retaining dependence within that respondent. The outcome mass is
/// the multinomial coefficient times the ratio of the refitted and current marginal likelihoods.
/// This is deterministic quadrature, not a calibrated interval or a guarantee of fieldwork value.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::choice_evsi::rank_hierarchical_studies`].
///
/// # Errors
/// Invalid inputs, unsupported design, a quadrature or mass check failure, cancellation or admission.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ChoiceStudyRanking route, which calls this kernel function with the same inputs"
)]
pub fn rank_hierarchical_studies(
    prior: &antecedent_stats::adaptive_inference::hierarchical_choice::HierarchicalChoicePrior,
    respondents: &[Vec<u32>],
    values: &[f64],
    candidates: &[antecedent_design::choice_evsi::StudyCandidate],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    Vec<antecedent_design::choice_evsi::StudyRank>,
    antecedent_design::choice_evsi::ChoiceEvsiError,
> {
    antecedent_design::choice_evsi::rank_hierarchical_studies(
        prior,
        respondents,
        values,
        candidates,
        ctx,
    )
}
