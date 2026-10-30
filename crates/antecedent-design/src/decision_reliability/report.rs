//! The reliability report: three uncertainties, separately.
//!
//! * **Source-distribution uncertainty**: the spread of the decision's utility under the
//!   supplied source law (posterior spread or sampling spread). It is a property of the
//!   law, not of the computation, and it does not shrink with more draws. When the law
//!   is declared known there is none to report.
//! * **Monte Carlo (numerical) error**: how far the reported numbers are from what the
//!   supplied law implies, because only finitely many draws were used. It shrinks with
//!   the effective sample size.
//! * **Selection-induced uncertainty**: how far the in-sample maximum over noisy
//!   candidates overstates the selected decision's value, with a bound and a held-out
//!   re-evaluation.
//!
//! The three are fields of [`ReliabilityReport`] and are never summed. A request for a
//! combined number refuses ([`UncertaintyReporting::Combined`], or
//! [`ReliabilityReport::combined_uncertainty`]): a combined statement is a whole-method
//! claim that needs its own calibration at the coordinate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;

use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

use super::ReliabilityError;
use super::calibration::{CalibrationCoordinate, ClosedRoute, closed_routes};
use super::draws::{
    CandidateCoupling, CandidateOrigin, DecisionDraws, SelectionRule, SourceParadigm,
    WithinStreamDependence,
};
use super::mc_error::{BatchPlan, DifferenceEstimate, Functional, McEstimate, PairCoupling};
use super::ranking::{CalibrationStatus, RankingReport, RankingRequest, rank_validated};
use super::selection::{
    CandidateEstimate, HeldOutAssessment, SelectionStatus, ValidationKind, ValueLabel,
    WinnersCurseBound, held_out_evaluation, noise_dependence_in_context, select_validated,
    split_draws_in_context, winners_curse_bound,
};

/// Report body version.
pub const RELIABILITY_REPORT_VERSION: u32 = 1;

/// Whether the caller wants the components separately or one number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyReporting {
    /// Three components, never summed. The only licensed value.
    Separate,
    /// One combined uncertainty: refused.
    Combined,
}

/// How the reported decision was arrived at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionChoice {
    /// The decision is the candidate maximizing the functional on the supplied draws.
    SelectedByRule,
    /// The decision was fixed independently of the draws (by identity).
    PreSpecified(String),
}

/// How the selected decision is validated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationPlan {
    /// No re-evaluation; the value stays in-sample.
    NoValidation,
    /// Re-evaluate on a separately supplied independent carrier.
    ProvidedStream {
        /// What kind of independent draws they are.
        kind: ValidationKind,
        /// Draws demanded between ranges of one shared stream.
        gap: u64,
    },
    /// Select on a leading segment of the supplied draws, validate on a later one.
    SplitSelectionDraws {
        /// Share of draws that select, in `(0, 1)`.
        selection_fraction: f64,
        /// Draws dropped between the segments.
        gap: u64,
    },
}

/// What to assess.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReliabilityRequest {
    /// The utility functional the decision maximizes and the report evaluates.
    pub functional: Functional,
    /// How the decision was arrived at.
    pub decision: DecisionChoice,
    /// Batch plan for dependent streams (also used by the ranking).
    pub batch_plan: BatchPlan,
    /// Independent validation.
    pub validation: ValidationPlan,
    /// Ranking statements to include, if any.
    pub ranking: Option<RankingRequest>,
    /// Separate components only.
    pub reporting: UncertaintyReporting,
}

impl ReliabilityRequest {
    /// Expected-utility maximization with no validation and no ranking.
    #[must_use]
    pub fn expected_utility() -> Self {
        Self {
            functional: Functional::ExpectedUtility,
            decision: DecisionChoice::SelectedByRule,
            batch_plan: BatchPlan::SquareRoot,
            validation: ValidationPlan::NoValidation,
            ranking: None,
            reporting: UncertaintyReporting::Separate,
        }
    }
}

/// Kind of claim a report makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimKind {
    /// Point estimates, standard errors and diagnostics; no interval or probability
    /// guarantee.
    PointDiagnostic,
}

/// Standing of everything in a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimLabel {
    /// Point diagnostics only.
    pub kind: ClaimKind,
    /// Uncalibrated at every coordinate until a coverage record measures it.
    pub calibration: CalibrationStatus,
}

/// Spread of one candidate's utility under the supplied source law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSpread {
    /// Candidate identity.
    pub id: String,
    /// Mean utility under the law (importance-weighted when weights were supplied).
    pub mean: f64,
    /// Standard deviation of the utility under the law.
    pub standard_deviation: f64,
    /// Standard deviation of `utility(decided) - utility(this)`; available on a shared
    /// simulation only, `None` otherwise and for the decided candidate itself.
    pub difference_standard_deviation: Option<f64>,
}

/// Source-distribution uncertainty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceUncertainty {
    /// Spread of utilities under the posterior or sampling law, one entry per candidate
    /// in canonical order. Conditional on the supplied law being the right one.
    Spread {
        /// The law the spread is under.
        paradigm: SourceParadigm,
        /// Per-candidate spread.
        candidates: Vec<SourceSpread>,
    },
    /// The law was declared known; the draws carry numerical error only.
    KnownLaw,
}

/// One candidate's difference from the decided candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferenceVsDecision {
    /// The other candidate.
    pub id: String,
    /// `functional(decided) - functional(other)` with its numerical error.
    pub difference: DifferenceEstimate,
}

/// Accounting of the streams the draws came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamAccounting {
    /// Distinct stream identities across candidates.
    pub distinct_streams: u64,
    /// Whether the candidates shared noise (common random numbers).
    pub shared_noise: bool,
    /// Draws per candidate.
    pub n_draws: u64,
    /// Candidates.
    pub n_candidates: u64,
    /// Kish effective sample size of the importance weights, when supplied.
    pub weights_kish_ess: Option<f64>,
}

/// Monte Carlo (numerical) error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonteCarloErrorReport {
    /// Functional evaluated.
    pub functional: Functional,
    /// Every candidate's estimate on the selection draws, canonical order.
    pub candidates: Vec<CandidateEstimate>,
    /// The decided candidate's estimate (in-sample when it was selected).
    pub decision: McEstimate,
    /// The decided candidate against each other candidate, canonical order.
    pub differences: Vec<DifferenceVsDecision>,
    /// Stream and weight accounting.
    pub streams: StreamAccounting,
}

/// Selection-induced uncertainty of a selected decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionAssessment {
    /// Rule that chose the decision.
    pub rule: SelectionRule,
    /// Candidates the selection considered.
    pub n_considered: u64,
    /// The selected candidate's unadjusted value on the selection draws.
    pub in_sample: McEstimate,
    /// Always [`ValueLabel::InSample`].
    pub in_sample_label: ValueLabel,
    /// How far the value has been checked for selection bias.
    pub status: SelectionStatus,
    /// Bound on the expected upward bias of the in-sample maximum.
    pub winners_curse: Option<WinnersCurseBound>,
    /// Re-evaluation on independent draws.
    pub held_out: Option<HeldOutAssessment>,
}

/// Selection-induced uncertainty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionUncertainty {
    /// The decision was fixed independently of the draws; no selection effect exists.
    NotSelected,
    /// The decision was chosen among candidates on the supplied draws.
    Selected(Box<SelectionAssessment>),
}

/// The dependence the claim rests on, stated explicitly rather than assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependenceStatement {
    /// How the candidates' draws depend on each other.
    pub coupling: CandidateCoupling,
    /// Serial dependence within each candidate's stream.
    pub within_stream: WithinStreamDependence,
    /// Earlier evaluations of these candidates on these draws.
    pub repeated_evaluations: u64,
    /// Whether the candidate set came from an adaptive search.
    pub candidates_adaptively_generated: bool,
    /// Whether validation draws were checked independent of the selection draws;
    /// `None` when there was no validation.
    pub validation_independent_of_selection: Option<bool>,
}

/// The decision-reliability report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReliabilityReport {
    /// Report body version.
    pub version: u32,
    /// Utility unit.
    pub utility_unit: String,
    /// Producer label.
    pub producer: String,
    /// Source paradigm.
    pub source_paradigm: SourceParadigm,
    /// Declared coupling of the candidates.
    pub coupling: CandidateCoupling,
    /// Declared serial dependence.
    pub within_stream: WithinStreamDependence,
    /// Identity of the candidate set and its search record.
    pub candidate_set_digest: String,
    /// Digest of the supplied draws.
    pub draws_digest: String,
    /// Digest of the supplied validation draws, when any.
    pub validation_digest: Option<String>,
    /// Functional evaluated.
    pub functional: Functional,
    /// The reported decision.
    pub decided_id: String,
    /// The dependence the claim rests on.
    pub dependence: DependenceStatement,
    /// Component one: spread under the supplied source law.
    pub source_uncertainty: SourceUncertainty,
    /// Component two: numerical error.
    pub monte_carlo_error: MonteCarloErrorReport,
    /// Component three: selection-induced uncertainty.
    pub selection_uncertainty: SelectionUncertainty,
    /// Ranking statements, when requested.
    pub ranking: Option<RankingReport>,
    /// Standing of every number above.
    pub label: ClaimLabel,
    /// Coordinate the report sits at.
    pub coordinate: CalibrationCoordinate,
    /// Published routes that are closed at that coordinate, with their refusals.
    pub closed_routes: Vec<ClosedRoute>,
}

impl ReliabilityReport {
    /// One combined uncertainty number.
    ///
    /// # Errors
    /// Always: the three components are never summed.
    pub fn combined_uncertainty(&self) -> Result<f64, ReliabilityError> {
        Err(ReliabilityError::CombinedUncertainty)
    }
}

fn weighted_spread(values: &[f64], weights: Option<&[f64]>) -> (f64, f64) {
    if let Some(w) = weights {
        let sum: f64 = w.iter().sum();
        let mean = values.iter().zip(w).map(|(u, wi)| u * wi).sum::<f64>() / sum;
        let squares: f64 = values.iter().zip(w).map(|(u, wi)| wi * (u - mean).powi(2)).sum();
        (mean, (squares / sum).sqrt())
    } else {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|u| (u - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }
}

fn source_uncertainty(
    draws: &DecisionDraws,
    decided: &str,
    ctx: &ExecutionContext,
) -> Result<SourceUncertainty, ReliabilityError> {
    if draws.source == SourceParadigm::KnownLaw {
        return Ok(SourceUncertainty::KnownLaw);
    }
    let weights = draws.weights.as_deref();
    let decided_draws = draws.candidate(decided).map(|c| c.utilities.as_slice());
    let mut candidates = Vec::with_capacity(draws.n_candidates());
    for &i in &draws.canonical_order() {
        let candidate = &draws.candidates[i];
        let (mean, sd) = weighted_spread(&candidate.utilities, weights);
        let paired_available =
            candidate.id != decided && draws.coupling == CandidateCoupling::SharedSimulation;
        let difference_standard_deviation = match decided_draws {
            Some(d) if paired_available => {
                let _paired = super::workspace::reserve::<f64>(ctx, d.len())?;
                let paired: Vec<f64> =
                    d.iter().zip(&candidate.utilities).map(|(a, b)| a - b).collect();
                Some(weighted_spread(&paired, weights).1)
            }
            _ => None,
        };
        candidates.push(SourceSpread {
            id: candidate.id.clone(),
            mean,
            standard_deviation: sd,
            difference_standard_deviation,
        });
    }
    Ok(SourceUncertainty::Spread { paradigm: draws.source, candidates })
}

fn stream_accounting(draws: &DecisionDraws) -> StreamAccounting {
    let mut streams: Vec<&super::stream::StreamId> =
        draws.candidates.iter().map(|c| &c.segment.stream).collect();
    streams.sort();
    streams.dedup();
    StreamAccounting {
        distinct_streams: streams.len() as u64,
        shared_noise: draws.n_candidates() > 1
            && draws.coupling == CandidateCoupling::SharedSimulation,
        n_draws: draws.n_draws() as u64,
        n_candidates: draws.n_candidates() as u64,
        weights_kish_ess: draws.weights.as_deref().map(antecedent_kernels::kish_ess),
    }
}

fn selection_rule(draws: &DecisionDraws, functional: &Functional) -> SelectionRule {
    if let Some(record) = &draws.search_record {
        return record.selection_rule.clone();
    }
    match functional {
        Functional::ExpectedUtility => SelectionRule::MaximizeExpectedUtility,
        Functional::ThresholdProbability { .. } => SelectionRule::MaximizeThresholdProbability,
        Functional::Quantile { .. } => SelectionRule::MaximizeQuantile,
        Functional::Cvar { .. } => SelectionRule::MaximizeCvar,
    }
}

/// Assess the reliability of a decision.
///
/// `draws` carries the candidates and the draws that selected the decision;
/// `validation` carries the independent draws when [`ValidationPlan::ProvidedStream`]
/// is requested.
///
/// # Errors
/// A combined-uncertainty request, any carrier defect or undeclared dependence, an
/// adaptive candidate set without its search record, validation draws that are not
/// independent, a pre-specified decision that is not a candidate (or that follows a
/// search), unsupported weighted risk measures, or cancellation.
#[allow(
    clippy::too_many_lines,
    reason = "one linear assembly of the three components and their diagnostics"
)]
pub fn assess(
    draws: &DecisionDraws,
    validation: Option<&DecisionDraws>,
    request: &ReliabilityRequest,
    ctx: &ExecutionContext,
) -> Result<ReliabilityReport, ReliabilityError> {
    if request.reporting == UncertaintyReporting::Combined {
        return Err(ReliabilityError::CombinedUncertainty);
    }
    request.functional.validate()?;
    draws.validate()?;
    if ctx.cancellation.is_cancelled() {
        return Err(ReliabilityError::Cancelled);
    }
    let plan = request.batch_plan;
    let split = if let (ValidationPlan::SplitSelectionDraws { selection_fraction, gap }, None) =
        (&request.validation, validation)
    {
        Some(split_draws_in_context(draws, *selection_fraction, *gap, ctx)?)
    } else {
        None
    };
    let (selection_draws, validation_draws, kind, gap) = match (&request.validation, validation) {
        (ValidationPlan::NoValidation, None) => (Cow::Borrowed(draws), None, None, 0),
        (ValidationPlan::ProvidedStream { kind, gap }, Some(v)) => {
            (Cow::Borrowed(draws), Some(Cow::Borrowed(v)), Some(*kind), *gap)
        }
        (ValidationPlan::SplitSelectionDraws { gap, .. }, None) => {
            let parts = split
                .as_ref()
                .ok_or_else(|| ReliabilityError::invalid("split", "split ownership missing"))?;
            (
                Cow::Borrowed(&parts.selection),
                Some(Cow::Borrowed(&parts.validation)),
                Some(ValidationKind::SplitOfSelectionStream),
                *gap,
            )
        }
        _ => {
            return Err(ReliabilityError::invalid(
                "validation_plan",
                "validation draws are supplied exactly when a provided stream is requested",
            ));
        }
    };
    let in_sample = select_validated(&selection_draws, &request.functional, plan, Some(ctx))?;
    let (decided_id, selected) = match &request.decision {
        DecisionChoice::SelectedByRule => (in_sample.selected_id.clone(), true),
        DecisionChoice::PreSpecified(id) => {
            if draws.candidate(id).is_none() {
                return Err(ReliabilityError::invalid(
                    "decision",
                    "pre-specified id is not a candidate",
                ));
            }
            if draws.origin == CandidateOrigin::AdaptivelyGenerated {
                return Err(ReliabilityError::invalid(
                    "decision",
                    "a decision found by an adaptive search is not pre-specified",
                ));
            }
            if validation_draws.is_some() {
                return Err(ReliabilityError::invalid(
                    "validation_plan",
                    "a pre-specified decision has no selection effect to validate",
                ));
            }
            (id.clone(), false)
        }
    };
    let decision_estimate = in_sample
        .estimates
        .iter()
        .find(|c| c.id == decided_id)
        .map(|c| c.estimate)
        .ok_or_else(|| ReliabilityError::invalid("decision", "decided candidate missing"))?;
    let pair_coupling = if selection_draws.coupling == CandidateCoupling::SharedSimulation {
        PairCoupling::SharedStream
    } else {
        PairCoupling::IndependentStreams
    };
    let decided_series = selection_draws
        .candidate(&decided_id)
        .map(|c| c.utilities.as_slice())
        .ok_or_else(|| ReliabilityError::invalid("decision", "decided candidate missing"))?;
    let mut differences = Vec::new();
    for &i in &selection_draws.canonical_order() {
        let other = &selection_draws.candidates[i];
        if other.id == decided_id {
            continue;
        }
        if ctx.cancellation.is_cancelled() {
            return Err(ReliabilityError::Cancelled);
        }
        differences.push(DifferenceVsDecision {
            id: other.id.clone(),
            difference: super::mc_error::estimate_difference_in_context(
                decided_series,
                &other.utilities,
                selection_draws.weights.as_deref(),
                selection_draws.within_stream,
                plan,
                &request.functional,
                pair_coupling,
                ctx,
            )?,
        });
    }
    let held_out_checked = validation_draws.is_some();
    let selection_uncertainty = if selected {
        let _estimates = super::workspace::reserve::<McEstimate>(ctx, in_sample.estimates.len())?;
        let estimates: Vec<McEstimate> = in_sample.estimates.iter().map(|c| c.estimate).collect();
        let noise = noise_dependence_in_context(&selection_draws, ctx)?;
        let curse = {
            let _ses = super::workspace::reserve::<f64>(ctx, estimates.len())?;
            let _groups = super::workspace::reserve::<(f64, u64)>(ctx, estimates.len())?;
            winners_curse_bound(
                &estimates,
                selection_draws.n_considered(),
                noise,
                decision_estimate.value,
            )?
        };
        let held_out = match (&validation_draws, kind) {
            (Some(v), Some(kind)) => {
                let slots = super::workspace::estimate_slots(
                    v.n_draws(),
                    v.within_stream,
                    plan,
                    &request.functional,
                )?;
                let _held_out = super::workspace::reserve::<f64>(ctx, slots)?;
                Some(held_out_evaluation(
                    &selection_draws,
                    &decided_id,
                    v,
                    &request.functional,
                    plan,
                    kind,
                    gap,
                )?)
            }
            _ => None,
        };
        let status = match (&held_out, kind) {
            (Some(_), Some(ValidationKind::SplitOfSelectionStream)) => {
                SelectionStatus::SampleSplitValidated
            }
            (Some(_), _) => SelectionStatus::HeldOutValidated,
            (None, _) => SelectionStatus::WinnersCurseBounded,
        };
        SelectionUncertainty::Selected(Box::new(SelectionAssessment {
            rule: selection_rule(draws, &request.functional),
            n_considered: selection_draws.n_considered(),
            in_sample: decision_estimate,
            in_sample_label: ValueLabel::InSample,
            status,
            winners_curse: Some(curse),
            held_out,
        }))
    } else {
        SelectionUncertainty::NotSelected
    };
    let ranking = match &request.ranking {
        Some(ranking_request) => {
            let mut ranking_request = ranking_request.clone();
            ranking_request.batch_plan = plan;
            Some(rank_validated(&selection_draws, &ranking_request, ctx)?)
        }
        None => None,
    };
    let coordinate = CalibrationCoordinate {
        candidate_set_size: selection_draws.n_considered(),
        mc_budget: selection_draws.n_draws() as u64,
        selection_rule: selection_rule(draws, &request.functional).kind(),
        functional: request.functional.kind().to_owned(),
        coupling: selection_draws.coupling,
        paradigm: selection_draws.source,
        within_stream: selection_draws.within_stream,
        importance_weighted: selection_draws.weights.is_some(),
        adaptively_generated: selection_draws.origin == CandidateOrigin::AdaptivelyGenerated,
    };
    Ok(ReliabilityReport {
        version: RELIABILITY_REPORT_VERSION,
        utility_unit: draws.utility_unit.clone(),
        producer: draws.producer.clone(),
        source_paradigm: draws.source,
        coupling: draws.coupling,
        within_stream: draws.within_stream,
        candidate_set_digest: draws.candidate_set_digest(),
        draws_digest: draws.digest_hex(),
        validation_digest: validation.map(DecisionDraws::digest_hex),
        functional: request.functional.clone(),
        decided_id: decided_id.clone(),
        dependence: DependenceStatement {
            coupling: draws.coupling,
            within_stream: draws.within_stream,
            repeated_evaluations: draws.repeated_evaluations,
            candidates_adaptively_generated: draws.origin == CandidateOrigin::AdaptivelyGenerated,
            validation_independent_of_selection: held_out_checked.then_some(true),
        },
        source_uncertainty: source_uncertainty(&selection_draws, &decided_id, ctx)?,
        monte_carlo_error: MonteCarloErrorReport {
            functional: request.functional.clone(),
            candidates: in_sample.estimates,
            decision: decision_estimate,
            differences,
            streams: stream_accounting(&selection_draws),
        },
        selection_uncertainty,
        ranking,
        label: ClaimLabel {
            kind: ClaimKind::PointDiagnostic,
            calibration: CalibrationStatus::Uncalibrated,
        },
        closed_routes: closed_routes(&coordinate),
        coordinate,
    })
}

#[cfg(test)]
mod source_spread_tests {
    use super::weighted_spread;

    #[test]
    fn equal_weights_and_unweighted_describe_the_same_empirical_law() {
        let values = [0.0, 2.0];
        assert_eq!(weighted_spread(&values, None), (1.0, 1.0));
        assert_eq!(weighted_spread(&values, Some(&[1.0, 1.0])), (1.0, 1.0));
        assert_eq!(weighted_spread(&values, Some(&[20.0, 20.0])), (1.0, 1.0));
        assert_eq!(weighted_spread(&[7.0], None), (7.0, 0.0));
        assert_eq!(weighted_spread(&[7.0], Some(&[3.0])), (7.0, 0.0));
    }
}
