//! Derived intervals and confidence sets for ranking and optimality statements.
//!
//! The derivation answers one question for a finite candidate set: *given the supplied
//! law and the candidates as supplied, how far can the reported ranking probabilities,
//! utilities and the selected decision's value be trusted to move?* It keeps the three
//! uncertainties as three separate components and never adds them:
//!
//! * **source-distribution uncertainty** is carried over from the report unchanged. A
//!   probability that a candidate is best is a probability *under* the supplied law; the
//!   law's own spread does not shrink with draws and no interval here covers a wrong law;
//! * the **numerical component** is the Monte Carlo error: a Wald interval for utilities
//!   and differences, and a Wilson interval at the effective sample size for ranking
//!   probabilities. The effective sample size is the one the estimator reported, so
//!   common random numbers, serial dependence (batch means) and importance weights are
//!   already accounted for. Below [`MIN_NORMAL_ESS`] the normal approximation is not
//!   derived and the request refuses;
//! * the **selection component** is made only of statements that stay valid when the
//!   decision was chosen by the data: a confidence set for the best candidate
//!   (multiple comparisons with the best, with Bonferroni critical values that hold for
//!   any dependence between candidates), and a simultaneous interval for the selected
//!   candidate's value whose extra width over the ordinary interval is reported
//!   separately, next to the held-out interval when the decision was re-evaluated.
//!
//! Assumptions, each recorded in [`DerivedUncertainty::assumptions`]: Monte Carlo errors
//! are approximately normal at the reported effective sample size, the estimated
//! standard errors are treated as known, and every statement is about the value under
//! the supplied law over the retained candidates. Candidates that an adaptive search
//! considered but did not retain are counted, never covered: the confidence set is for
//! the best *retained* candidate.
//!
//! The derivation is **unmeasured**: it carries [`DerivedStanding::DerivedUnmeasured`]
//! and [`DerivedUncertainty::publish`] refuses with `cell_not_licensed` until a coverage
//! record measures the route at its coordinate (candidate-set size, Monte Carlo budget,
//! selection rule).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_kernels::norm_inv;
use serde::{Deserialize, Serialize};

use super::ReliabilityError;
use super::calibration::{CalibrationCoordinate, PublishedRoute, license_route};
use super::draws::{CandidateCoupling, DecisionDraws};
use super::mc_error::{McEstimate, PairCoupling, estimate_difference};
use super::report::{
    ReliabilityReport, ReliabilityRequest, SelectionUncertainty, SourceUncertainty,
    ValidationPlan, assess,
};
use super::selection::split_draws;

/// Body version of [`DerivedUncertainty`].
pub const DERIVED_UNCERTAINTY_VERSION: u32 = 1;
/// Smallest effective sample size at which the normal approximation is derived.
pub const MIN_NORMAL_ESS: f64 = 30.0;
/// Most candidates for which every pairwise paired comparison is computed.
pub const MAX_PAIRED_COMPARISON_CANDIDATES: usize = 64;
/// Most candidate-pair-draw cells the paired comparisons may touch.
pub const MAX_PAIRED_COMPARISON_CELLS: usize = 1 << 27;
/// Largest nominal level a request may ask for (the tail mass must stay representable).
pub const MAX_INTERVAL_LEVEL: f64 = 0.9999;
/// Smallest nominal level a request may ask for.
pub const MIN_INTERVAL_LEVEL: f64 = 0.5;

/// Whether the numerical intervals are individually or jointly at the nominal level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalMultiplicity {
    /// Each interval is at the nominal level on its own.
    Marginal,
    /// Bonferroni over every reported numerical interval: all hold jointly, for any
    /// dependence between them.
    Simultaneous,
}

/// What to derive.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalRequest {
    /// Nominal two-sided level, in `[0.5, 0.9999]`.
    pub level: f64,
    /// Marginal or simultaneous numerical intervals. The selection component is always
    /// simultaneous over the candidate set.
    pub multiplicity: IntervalMultiplicity,
}

impl IntervalRequest {
    /// Marginal numerical intervals at `level`.
    #[must_use]
    pub const fn marginal(level: f64) -> Self {
        Self { level, multiplicity: IntervalMultiplicity::Marginal }
    }

    fn validate(&self) -> Result<(), ReliabilityError> {
        if self.level.is_finite()
            && self.level >= MIN_INTERVAL_LEVEL
            && self.level <= MAX_INTERVAL_LEVEL
        {
            Ok(())
        } else {
            Err(ReliabilityError::invalid("interval_level", "level must lie in [0.5, 0.9999]"))
        }
    }

    fn alpha(&self) -> f64 {
        1.0 - self.level
    }
}

/// Standing of every interval here: derived, not measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivedStanding {
    /// The construction is derived under stated assumptions and has no coverage record
    /// at its coordinate.
    DerivedUnmeasured,
}

/// How a numerical interval was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalIntervalMethod {
    /// Estimate plus or minus `z` standard errors.
    WaldNormal,
    /// Wilson score interval with the effective sample size as the sample size.
    WilsonEffectiveSample,
}

/// A numerical (Monte Carlo) interval for the value implied by the supplied law.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalInterval {
    /// The point estimate.
    pub estimate: f64,
    /// Its standard error.
    pub standard_error: f64,
    /// The effective sample size the interval used.
    pub effective_sample_size: f64,
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
    /// Two-sided level this interval was built at (after any multiplicity adjustment).
    pub level: f64,
    /// Construction.
    pub method: NumericalIntervalMethod,
}

/// Which ranking quantity a probability interval is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbabilityQuantity {
    /// The candidate is (tie-shared) best.
    Best,
    /// The candidate is among the best `k`.
    TopK {
        /// Set size.
        k: u32,
    },
    /// The candidate beats the baseline (ties count one half).
    BetterThanBaseline,
}

/// One ranking probability with its numerical interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbabilityInterval {
    /// Candidate identity.
    pub id: String,
    /// Which probability.
    pub quantity: ProbabilityQuantity,
    /// Its interval.
    pub interval: NumericalInterval,
}

/// A candidate's utility interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityInterval {
    /// Candidate identity.
    pub id: String,
    /// Its interval under the selection functional.
    pub interval: NumericalInterval,
}

/// Numerical (Monte Carlo) component.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalComponent {
    /// Multiplicity the intervals were built under.
    pub multiplicity: IntervalMultiplicity,
    /// Number of intervals the multiplicity adjusted for.
    pub intervals_reported: u64,
    /// Whether candidates shared noise (common random numbers) in these intervals.
    pub shared_noise: bool,
    /// Smallest effective sample size among the reported intervals.
    pub minimum_effective_sample_size: f64,
    /// Every candidate's functional, canonical order.
    pub utilities: Vec<UtilityInterval>,
    /// The decided candidate minus each other candidate, paired when noise is shared.
    pub differences_vs_decision: Vec<UtilityInterval>,
    /// Ranking probabilities, when the request asked for them.
    pub probabilities: Vec<ProbabilityInterval>,
}

/// How a confidence set for the best candidate was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceSetBasis {
    /// Marginal standard errors with a Bonferroni critical value over the candidates:
    /// valid for any dependence, conservative under common random numbers.
    MarginalBonferroni,
    /// Every pair compared on its own paired (or independent-stream) difference error,
    /// Bonferroni over the competitors of the best: sharper under common random numbers.
    PairedComparisonsWithBest,
}

/// Confidence set that contains the best retained candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimalityConfidenceSet {
    /// Construction.
    pub basis: ConfidenceSetBasis,
    /// Retained candidates not excluded, canonical order.
    pub members: Vec<String>,
    /// Retained candidates excluded, canonical order.
    pub excluded: Vec<String>,
    /// Nominal probability that the best retained candidate (of the supplied law) is a
    /// member.
    pub level: f64,
    /// The critical value the exclusion rule used.
    pub critical_value: f64,
    /// Candidates the search considered but did not retain. The set says nothing about
    /// them.
    pub unretained_candidates: u64,
}

/// Interval for the selected candidate's value that stays valid after selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionCorrectedInterval {
    /// The selected candidate.
    pub selected_id: String,
    /// Its unadjusted (in-sample) value.
    pub in_sample: f64,
    /// Its standard error on the selection draws.
    pub standard_error: f64,
    /// Candidates the selection considered (the Bonferroni count).
    pub n_considered: u64,
    /// Half-width of the ordinary two-sided interval at the nominal level (the numerical
    /// component on its own).
    pub numerical_half_width: f64,
    /// Additional half-width the selection costs: the interval must hold for whichever
    /// candidate the data picked.
    pub selection_extra_half_width: f64,
    /// Simultaneous lower end, `in_sample - numerical - extra`.
    pub lower: f64,
    /// Simultaneous upper end.
    pub upper: f64,
    /// Critical value used (`z` at tail mass `alpha / (2 n_considered)`).
    pub critical_value: f64,
    /// Bound on the expected upward bias of the maximum, from the report.
    pub expected_bias_bound: Option<f64>,
    /// Interval from the independent re-evaluation, when the decision was re-evaluated:
    /// unbiased for the selected candidate and purely numerical.
    pub held_out: Option<NumericalInterval>,
}

/// Selection component.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionComponent {
    /// Confidence set for the best retained candidate.
    pub confidence_set: OptimalityConfidenceSet,
    /// The selection-aware interval; `None` for a decision fixed independently of the
    /// draws.
    pub corrected: Option<SelectionCorrectedInterval>,
}

/// The three components, derived and unmeasured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedUncertainty {
    /// Body version.
    pub version: u32,
    /// What was requested.
    pub request: IntervalRequest,
    /// Digest of the supplied selection draws.
    pub draws_digest: String,
    /// Identity of the candidate set and its search record.
    pub candidate_set_digest: String,
    /// The decided candidate.
    pub decided_id: String,
    /// Component one, copied from the report: spread under the supplied law.
    pub source: SourceUncertainty,
    /// Component two: numerical error.
    pub numerical: NumericalComponent,
    /// Component three: selection-induced uncertainty.
    pub selection: SelectionComponent,
    /// Always derived and unmeasured.
    pub standing: DerivedStanding,
    /// Coordinate a coverage record would have to measure.
    pub coordinate: CalibrationCoordinate,
    /// Assumptions the derivation rests on.
    pub assumptions: Vec<String>,
}

impl DerivedUncertainty {
    /// Ask to publish a route from this derivation.
    ///
    /// # Errors
    /// [`ReliabilityError::CellNotLicensed`] until a coverage record measures the route
    /// at [`Self::coordinate`].
    pub fn publish(&self, route: PublishedRoute) -> Result<(), ReliabilityError> {
        license_route(route, &self.coordinate)
    }

    /// One combined uncertainty number.
    ///
    /// # Errors
    /// Always: the three components are never summed.
    pub fn combined_uncertainty(&self) -> Result<f64, ReliabilityError> {
        Err(ReliabilityError::CombinedUncertainty)
    }

    /// Re-derive from the draws and compare. A fresh consumer does not trust the stored
    /// intervals.
    ///
    /// # Errors
    /// Anything [`derive_uncertainty`] refuses, or a derivation that differs.
    pub fn verify(
        &self,
        draws: &DecisionDraws,
        validation: Option<&DecisionDraws>,
        request: &ReliabilityRequest,
        ctx: &ExecutionContext,
    ) -> Result<(), ReliabilityError> {
        let recomputed = derive_uncertainty(draws, validation, request, &self.request, ctx)?;
        if recomputed == *self {
            Ok(())
        } else {
            Err(ReliabilityError::invalid(
                "derived_uncertainty",
                "stored derivation does not recompute from the draws",
            ))
        }
    }
}

/// Upper-tail normal critical value at one-sided tail mass `tail`.
fn critical(tail: f64) -> f64 {
    -norm_inv(tail)
}

fn require_normal_ess(what: &'static str, ess: f64) -> Result<(), ReliabilityError> {
    if ess.is_finite() && ess >= MIN_NORMAL_ESS {
        Ok(())
    } else {
        Err(ReliabilityError::EffectiveSampleSizeTooSmall {
            what,
            effective_sample_size: ess,
            minimum: MIN_NORMAL_ESS,
        })
    }
}

fn wald(
    what: &'static str,
    estimate: &McEstimate,
    tail: f64,
    level: f64,
) -> Result<NumericalInterval, ReliabilityError> {
    require_normal_ess(what, estimate.effective_sample_size)?;
    let half = critical(tail) * estimate.standard_error;
    Ok(NumericalInterval {
        estimate: estimate.value,
        standard_error: estimate.standard_error,
        effective_sample_size: estimate.effective_sample_size,
        lower: estimate.value - half,
        upper: estimate.value + half,
        level,
        method: NumericalIntervalMethod::WaldNormal,
    })
}

/// Wilson score interval with the effective sample size as the sample size. The
/// per-draw series is bounded in `[0, 1]`, so its variance is at most `p (1 - p)` and the
/// interval is not narrower than the series' own error warrants.
fn wilson(
    what: &'static str,
    estimate: &McEstimate,
    tail: f64,
    level: f64,
) -> Result<NumericalInterval, ReliabilityError> {
    let n = estimate.effective_sample_size;
    require_normal_ess(what, n)?;
    let p = estimate.value.clamp(0.0, 1.0);
    let z = critical(tail);
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denominator;
    let half = z / denominator * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    Ok(NumericalInterval {
        estimate: estimate.value,
        standard_error: estimate.standard_error,
        effective_sample_size: n,
        lower: (centre - half).max(0.0),
        upper: (centre + half).min(1.0),
        level,
        method: NumericalIntervalMethod::WilsonEffectiveSample,
    })
}

/// Members of the best-candidate set by multiple comparisons with the best: candidate
/// `j` is excluded when some candidate `i` beats it by more than `z` standard errors of
/// their (paired or independent-stream) difference. The true best is wrongly excluded
/// only if some competitor appears better than it by chance, a union of `K - 1` events.
fn paired_confidence_set(
    draws: &DecisionDraws,
    request: &ReliabilityRequest,
    one_sided_tail: f64,
    ctx: &ExecutionContext,
) -> Result<Vec<String>, ReliabilityError> {
    let pair_coupling = if draws.coupling == CandidateCoupling::SharedSimulation {
        PairCoupling::SharedStream
    } else {
        PairCoupling::IndependentStreams
    };
    let z = critical(one_sided_tail);
    let order = draws.canonical_order();
    let mut members = Vec::new();
    for &j in &order {
        if ctx.cancellation.is_cancelled() {
            return Err(ReliabilityError::Cancelled);
        }
        let mut excluded = false;
        for &i in &order {
            if i == j {
                continue;
            }
            let difference = estimate_difference(
                &draws.candidates[i].utilities,
                &draws.candidates[j].utilities,
                draws.weights.as_deref(),
                draws.within_stream,
                request.batch_plan,
                &request.functional,
                pair_coupling,
            )?;
            let beats = if difference.standard_error > 0.0 {
                difference.value > z * difference.standard_error
            } else {
                difference.value > 0.0
            };
            if beats {
                excluded = true;
                break;
            }
        }
        if !excluded {
            members.push(draws.candidates[j].id.clone());
        }
    }
    Ok(members)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "candidate and interval counts are bounded by the carrier limits"
)]
fn confidence_set(
    report: &ReliabilityReport,
    selection_draws: &DecisionDraws,
    request: &ReliabilityRequest,
    alpha: f64,
    ctx: &ExecutionContext,
) -> Result<OptimalityConfidenceSet, ReliabilityError> {
    let retained = selection_draws.n_candidates();
    let unretained = selection_draws.n_considered().saturating_sub(retained as u64);
    let ids: Vec<String> = selection_draws
        .canonical_order()
        .iter()
        .map(|&i| selection_draws.candidates[i].id.clone())
        .collect();
    let make = |basis: ConfidenceSetBasis, members: Vec<String>, critical_value: f64| {
        let excluded = ids.iter().filter(|id| !members.contains(id)).cloned().collect();
        OptimalityConfidenceSet {
            basis,
            members,
            excluded,
            level: 1.0 - alpha,
            critical_value,
            unretained_candidates: unretained,
        }
    };
    if retained < 2 {
        return Ok(make(ConfidenceSetBasis::MarginalBonferroni, ids.clone(), 0.0));
    }
    let cells = retained
        .checked_mul(retained)
        .and_then(|pairs| pairs.checked_mul(selection_draws.n_draws()));
    let paired = retained <= MAX_PAIRED_COMPARISON_CANDIDATES
        && cells.is_some_and(|c| c <= MAX_PAIRED_COMPARISON_CELLS);
    if paired {
        let tail = alpha / (retained as f64 - 1.0);
        let members = paired_confidence_set(selection_draws, request, tail, ctx)?;
        return Ok(make(ConfidenceSetBasis::PairedComparisonsWithBest, members, critical(tail)));
    }
    // Dependence-free: every candidate within `z` standard errors of its own value
    // simultaneously (two-sided Bonferroni), so the best can be no lower than the
    // largest lower end and no candidate whose upper end is below it can be best.
    let tail = alpha / (2.0 * retained as f64);
    let z = critical(tail);
    let estimates = &report.monte_carlo_error.candidates;
    let floor = estimates
        .iter()
        .map(|c| c.estimate.value - z * c.estimate.standard_error)
        .fold(f64::NEG_INFINITY, f64::max);
    let members: Vec<String> = estimates
        .iter()
        .filter(|c| c.estimate.value + z * c.estimate.standard_error >= floor)
        .map(|c| c.id.clone())
        .collect();
    Ok(make(ConfidenceSetBasis::MarginalBonferroni, members, z))
}

fn numerical_component(
    report: &ReliabilityReport,
    request: &IntervalRequest,
) -> Result<NumericalComponent, ReliabilityError> {
    let monte_carlo = &report.monte_carlo_error;
    let probabilities_count = report.ranking.as_ref().map_or(0, |ranking| {
        ranking
            .statements
            .iter()
            .map(|s| {
                1 + s.probability_top_k.len()
                    + usize::from(s.probability_better_than_baseline.is_some())
            })
            .sum::<usize>()
    });
    let reported =
        monte_carlo.candidates.len() + monte_carlo.differences.len() + probabilities_count;
    let alpha = request.alpha();
    #[allow(
        clippy::cast_precision_loss,
        reason = "interval counts are bounded by the carrier limits"
    )]
    let per_interval_alpha = match request.multiplicity {
        IntervalMultiplicity::Marginal => alpha,
        IntervalMultiplicity::Simultaneous => alpha / reported.max(1) as f64,
    };
    let tail = per_interval_alpha / 2.0;
    let level = 1.0 - per_interval_alpha;
    let mut minimum_ess = f64::INFINITY;

    let mut utilities = Vec::with_capacity(monte_carlo.candidates.len());
    for candidate in &monte_carlo.candidates {
        let interval = wald("candidate_functional", &candidate.estimate, tail, level)?;
        minimum_ess = minimum_ess.min(interval.effective_sample_size);
        utilities.push(UtilityInterval { id: candidate.id.clone(), interval });
    }
    let mut differences = Vec::with_capacity(monte_carlo.differences.len());
    for entry in &monte_carlo.differences {
        let difference = &entry.difference;
        // A shared-stream quantile difference has no per-draw decomposition: its batch
        // standard error is used with the smaller of the two marginal sample sizes.
        let ess = difference.effective_sample_size.unwrap_or_else(|| {
            monte_carlo.decision.effective_sample_size.min(
                monte_carlo
                    .candidates
                    .iter()
                    .find(|c| c.id == entry.id)
                    .map_or(f64::INFINITY, |c| c.estimate.effective_sample_size),
            )
        });
        let as_estimate = McEstimate {
            value: difference.value,
            standard_error: difference.standard_error,
            effective_sample_size: ess,
            n_draws: monte_carlo.decision.n_draws,
            method: difference.method,
        };
        let interval = wald("difference_to_decision", &as_estimate, tail, level)?;
        minimum_ess = minimum_ess.min(interval.effective_sample_size);
        differences.push(UtilityInterval { id: entry.id.clone(), interval });
    }
    let mut probabilities = Vec::with_capacity(probabilities_count);
    if let Some(ranking) = &report.ranking {
        for statement in &ranking.statements {
            let mut push = |quantity: ProbabilityQuantity,
                            estimate: &McEstimate,
                            what: &'static str|
             -> Result<(), ReliabilityError> {
                let interval = wilson(what, estimate, tail, level)?;
                minimum_ess = minimum_ess.min(interval.effective_sample_size);
                probabilities.push(ProbabilityInterval {
                    id: statement.id.clone(),
                    quantity,
                    interval,
                });
                Ok(())
            };
            push(ProbabilityQuantity::Best, &statement.probability_best, "probability_best")?;
            for top in &statement.probability_top_k {
                push(
                    ProbabilityQuantity::TopK { k: top.k },
                    &top.estimate,
                    "probability_top_k",
                )?;
            }
            if let Some(versus) = &statement.probability_better_than_baseline {
                push(
                    ProbabilityQuantity::BetterThanBaseline,
                    versus,
                    "probability_better_than_baseline",
                )?;
            }
        }
    }
    Ok(NumericalComponent {
        multiplicity: request.multiplicity,
        intervals_reported: reported as u64,
        shared_noise: monte_carlo.streams.shared_noise,
        minimum_effective_sample_size: if minimum_ess.is_finite() { minimum_ess } else { 0.0 },
        utilities,
        differences_vs_decision: differences,
        probabilities,
    })
}

#[allow(
    clippy::cast_precision_loss,
    reason = "candidate counts are bounded by the carrier limits"
)]
fn corrected_interval(
    report: &ReliabilityReport,
    request: &IntervalRequest,
) -> Result<Option<SelectionCorrectedInterval>, ReliabilityError> {
    let SelectionUncertainty::Selected(assessment) = &report.selection_uncertainty else {
        return Ok(None);
    };
    let alpha = request.alpha();
    let in_sample = assessment.in_sample;
    require_normal_ess("selected_candidate", in_sample.effective_sample_size)?;
    let n_considered = assessment.n_considered.max(1);
    let ordinary = critical(alpha / 2.0);
    let simultaneous = critical(alpha / (2.0 * n_considered as f64));
    let numerical_half_width = ordinary * in_sample.standard_error;
    let selection_extra_half_width = (simultaneous - ordinary) * in_sample.standard_error;
    let held_out = match &assessment.held_out {
        Some(held) => {
            Some(wald("held_out_selected", &held.held_out.estimate, alpha / 2.0, request.level)?)
        }
        None => None,
    };
    Ok(Some(SelectionCorrectedInterval {
        selected_id: report.decided_id.clone(),
        in_sample: in_sample.value,
        standard_error: in_sample.standard_error,
        n_considered,
        numerical_half_width,
        selection_extra_half_width,
        lower: in_sample.value - numerical_half_width - selection_extra_half_width,
        upper: in_sample.value + numerical_half_width + selection_extra_half_width,
        critical_value: simultaneous,
        expected_bias_bound: assessment.winners_curse.as_ref().map(|b| b.bound),
        held_out,
    }))
}

fn assumptions() -> Vec<String> {
    [
        "monte_carlo_error_is_normal_at_the_reported_effective_sample_size",
        "estimated_standard_errors_are_treated_as_known",
        "statements_are_for_the_value_under_the_supplied_law_not_for_a_wrong_law",
        "statements_are_over_the_retained_candidates_unretained_candidates_are_counted_not_covered",
        "selection_component_is_valid_for_any_dependence_between_candidates",
        "components_are_reported_separately_and_never_summed",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Derive the three components of uncertainty for a decision over a finite candidate set.
///
/// `validation` carries independent draws when `request` names a provided validation
/// stream.
///
/// # Errors
/// A level outside `[0.5, 0.9999]`, anything [`assess`] refuses (undeclared dependence,
/// an adaptive candidate set without its search record, non-independent validation, a
/// combined-uncertainty request), an effective sample size below [`MIN_NORMAL_ESS`], or
/// cancellation.
pub fn derive_uncertainty(
    draws: &DecisionDraws,
    validation: Option<&DecisionDraws>,
    request: &ReliabilityRequest,
    interval: &IntervalRequest,
    ctx: &ExecutionContext,
) -> Result<DerivedUncertainty, ReliabilityError> {
    interval.validate()?;
    let report = assess(draws, validation, request, ctx)?;
    let selection_draws = match &request.validation {
        ValidationPlan::SplitSelectionDraws { selection_fraction, gap } => {
            split_draws(draws, *selection_fraction, *gap)?.0
        }
        _ => draws.clone(),
    };
    let numerical = numerical_component(&report, interval)?;
    let confidence = confidence_set(&report, &selection_draws, request, interval.alpha(), ctx)?;
    let corrected = corrected_interval(&report, interval)?;
    Ok(DerivedUncertainty {
        version: DERIVED_UNCERTAINTY_VERSION,
        request: *interval,
        draws_digest: report.draws_digest.clone(),
        candidate_set_digest: report.candidate_set_digest.clone(),
        decided_id: report.decided_id.clone(),
        source: report.source_uncertainty.clone(),
        numerical,
        selection: SelectionComponent { confidence_set: confidence, corrected },
        standing: DerivedStanding::DerivedUnmeasured,
        coordinate: report.coordinate.clone(),
        assumptions: assumptions(),
    })
}
