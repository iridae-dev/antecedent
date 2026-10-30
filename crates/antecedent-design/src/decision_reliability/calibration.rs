//! Calibration coordinates, licensed routes and the known-optimum simulation design.
//!
//! A ranking or optimality probability, or an interval for a selected decision, is a
//! claim about the *whole procedure*: the candidate-set size, the Monte Carlo budget and
//! the selection rule together. Component Monte Carlo error alone does not cover the
//! selected decision. A route is licensed at a coordinate exactly when a current,
//! attesting, non-boundary coverage record measured that construction and the
//! coordinate lies inside the range it measured ([`license_route_at`] resolves the
//! record through the shipped coverage table). At any other coordinate the route is
//! closed with a typed `cell_not_licensed` refusal and only point diagnostics are
//! published.
//!
//! [`KnownOptimumDesign`] is the data-generating process the measurement uses: a
//! finite candidate set with known true utilities (so the true optimum, the true value
//! of the selected candidate and the true ranking are known), noisy Monte Carlo
//! utilities on a selection stream, and an independent validation stream.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_io_core::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};
use antecedent_io_core::contract_section::CalibrationStatus;
use antecedent_kernels::standard_normal;
use serde::{Deserialize, Serialize};

use super::ReliabilityError;
use super::draws::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws, SourceParadigm,
    WithinStreamDependence,
};
use super::ranking::RankingRequest;
use super::report::{
    DecisionChoice, ReliabilityRequest, SelectionUncertainty, UncertaintyReporting, ValidationPlan,
    assess,
};
use super::selection::ValidationKind;
use super::stream::{DrawSegment, StreamId, StreamPurpose};

/// The coordinate at which a published route would be calibrated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationCoordinate {
    /// Candidates the selection considered.
    pub candidate_set_size: u64,
    /// Monte Carlo draws per candidate.
    pub mc_budget: u64,
    /// Selection rule family.
    pub selection_rule: String,
    /// Functional family.
    pub functional: String,
    /// How the candidates' draws were coupled.
    pub coupling: CandidateCoupling,
    /// Paradigm of the source law.
    pub paradigm: SourceParadigm,
    /// Serial dependence of the draws within a candidate's stream. A report written
    /// before this field existed reads as `unknown`, which no record covers.
    #[serde(default = "unknown_within_stream")]
    pub within_stream: WithinStreamDependence,
    /// Whether importance weights were applied. Absent reads as weighted, which no
    /// record covers.
    #[serde(default = "assume_flagged")]
    pub importance_weighted: bool,
    /// Whether the candidates came from an adaptive search. Absent reads as adaptive,
    /// which no record covers.
    #[serde(default = "assume_flagged")]
    pub adaptively_generated: bool,
}

const fn unknown_within_stream() -> WithinStreamDependence {
    WithinStreamDependence::Unknown
}

const fn assume_flagged() -> bool {
    true
}

impl CalibrationCoordinate {
    /// Human-readable coordinate.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "candidate_set_size={}/mc_budget={}/selection_rule={}/functional={}/coupling={}/\
             paradigm={}/within_stream={}/importance_weighted={}/adaptively_generated={}",
            self.candidate_set_size,
            self.mc_budget,
            self.selection_rule,
            self.functional,
            self.coupling.as_str(),
            self.paradigm.as_str(),
            self.within_stream.as_str(),
            self.importance_weighted,
            self.adaptively_generated
        )
    }
}

/// A published inferential route that needs a coverage record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublishedRoute {
    /// A probability that a candidate is best, top-k or better than a baseline,
    /// published with a guarantee.
    RankingProbability,
    /// A confidence or credible set that contains the best candidate.
    OptimalityConfidenceSet,
    /// An interval for a candidate's utility from the Monte Carlo error.
    UtilityInterval,
    /// An interval for the selected decision's value that accounts for selection.
    SelectionCorrectedInterval,
}

impl PublishedRoute {
    /// Every published route.
    pub const ALL: [Self; 4] = [
        Self::RankingProbability,
        Self::OptimalityConfidenceSet,
        Self::UtilityInterval,
        Self::SelectionCorrectedInterval,
    ];

    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RankingProbability => "ranking_probability",
            Self::OptimalityConfidenceSet => "optimality_confidence_set",
            Self::UtilityInterval => "utility_interval",
            Self::SelectionCorrectedInterval => "selection_corrected_interval",
        }
    }
}

/// A route that is closed at a coordinate, with its refusal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosedRoute {
    /// The route.
    pub route: PublishedRoute,
    /// Coordinate it would be calibrated at.
    pub coordinate: String,
    /// Registered reason code of the refusal.
    pub code: String,
    /// What would open it.
    pub remedy: String,
}

/// Query axis name every decision-reliability coverage record carries.
pub const RECORD_QUERY: &str = "DecisionReliability";
/// Nominal level [`license_route`] resolves at.
pub const DEFAULT_LICENSE_LEVEL: f64 = 0.95;

impl PublishedRoute {
    /// Estimator label and interval method of the record that licenses the route.
    #[must_use]
    pub const fn record_construction(self) -> (&'static str, &'static str) {
        match self {
            Self::RankingProbability => {
                ("decision_reliability_ranking_probability", "wilson_effective_sample")
            }
            Self::OptimalityConfidenceSet => {
                ("decision_reliability_optimality_confidence_set", "bonferroni_confidence_set")
            }
            Self::UtilityInterval => ("decision_reliability_utility_interval", "wald_normal"),
            Self::SelectionCorrectedInterval => {
                ("decision_reliability_selection_corrected_interval", "simultaneous_wald_normal")
            }
        }
    }
}

/// Candidate-set size bands a coverage record is keyed by, inclusive on both ends.
///
/// A record measured at several sizes inside a band licenses every size of the band,
/// and only because each size visited passed; a size above the last band has no
/// record and is closed. The width of a band is what the measurement has to visit: a
/// selection correction grows with the logarithm of the count, and the best-candidate
/// set switches from paired comparisons to the dependence-free construction above
/// 64 candidates, so no band spans that switch.
pub const CANDIDATE_COUNT_BANDS: [(u64, u64); 4] = [(2, 4), (5, 16), (17, 64), (65, 100)];

/// The band of [`CANDIDATE_COUNT_BANDS`] containing `count`, if any.
#[must_use]
pub fn candidate_count_band(count: u64) -> Option<(u64, u64)> {
    CANDIDATE_COUNT_BANDS.into_iter().find(|&(low, high)| (low..=high).contains(&count))
}

/// The `functional` label a record is keyed under at a coordinate: the band of the
/// candidate-set size, the selection rule and the functional family are part of what
/// was measured. `None` when the size lies in no measured band.
#[must_use]
pub fn banded_record_functional(coordinate: &CalibrationCoordinate) -> Option<String> {
    let (low, high) = candidate_count_band(coordinate.candidate_set_size)?;
    Some(format!(
        "candidates={low}-{high};rule={};functional={}",
        coordinate.selection_rule, coordinate.functional
    ))
}

/// The `functional` label of a record that measured one exact candidate count: the
/// candidate-set size, the selection rule and the functional family. Records measured
/// now key by band ([`banded_record_functional`]); this label still keys the records
/// that were measured at one count.
#[must_use]
pub fn record_functional(coordinate: &CalibrationCoordinate) -> String {
    format!(
        "candidates={};rule={};functional={}",
        coordinate.candidate_set_size, coordinate.selection_rule, coordinate.functional
    )
}

/// The calibration basis of `route` at `coordinate` and nominal `level`: the key a
/// coverage record must equal and the execution scope (the per-candidate Monte Carlo
/// budget is the row count) it must cover. `None` when the coordinate carries a
/// declaration no record can cover (undeclared serial dependence, weights, adaptive
/// candidates, an unknown coupling or a candidate-set size outside every measured band).
#[must_use]
pub fn calibration_basis(
    route: PublishedRoute,
    coordinate: &CalibrationCoordinate,
    level: f64,
) -> Option<CalibrationBasisWire> {
    basis_with_functional(route, coordinate, level, banded_record_functional(coordinate)?)
}

/// The basis under the key of records that measured one exact candidate count. Such a
/// record licenses exactly that count; a measurement now keys by band
/// ([`calibration_basis`]).
fn exact_count_basis(
    route: PublishedRoute,
    coordinate: &CalibrationCoordinate,
    level: f64,
) -> Option<CalibrationBasisWire> {
    basis_with_functional(route, coordinate, level, record_functional(coordinate))
}

fn basis_with_functional(
    route: PublishedRoute,
    coordinate: &CalibrationCoordinate,
    level: f64,
    functional: String,
) -> Option<CalibrationBasisWire> {
    if coordinate.within_stream == WithinStreamDependence::Unknown
        || coordinate.importance_weighted
        || coordinate.adaptively_generated
        || coordinate.coupling == CandidateCoupling::Unknown
    {
        return None;
    }
    let (estimator, interval_method) = route.record_construction();
    Some(CalibrationBasisWire {
        key: CalibrationKeyWire {
            validation: "not_applicable".into(),
            query: RECORD_QUERY.to_owned(),
            graph_class: "NotApplicable".to_owned(),
            structure: "fixed".to_owned(),
            modality: "decision_draws".to_owned(),
            inference: "Frequentist".to_owned(),
            estimator: estimator.to_owned(),
            interval_method: interval_method.to_owned(),
            se_kind: "monte_carlo".to_owned(),
            dependence: format!(
                "{}.{}",
                coordinate.coupling.as_str(),
                coordinate.within_stream.as_str()
            ),
            posterior: String::new(),
            functional,
            level,
            identification: "point".to_owned(),
        },
        scope: CalibrationScopeWire {
            row_count: coordinate.mc_budget,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
    })
}

/// Ask for a published route at a coordinate, at [`DEFAULT_LICENSE_LEVEL`].
///
/// # Errors
/// [`ReliabilityError::CellNotLicensed`] unless a coverage record measured this route
/// at this coordinate.
pub fn license_route(
    route: PublishedRoute,
    coordinate: &CalibrationCoordinate,
) -> Result<(), ReliabilityError> {
    license_route_at(route, coordinate, DEFAULT_LICENSE_LEVEL)
}

/// Ask for a published route at a coordinate and nominal level.
///
/// The route is licensed only when the shipped coverage table holds a current,
/// attesting, non-boundary record whose construction equals [`calibration_basis`] (or,
/// for a record measured at one exact count, that count) and whose measured sample-size
/// range contains the coordinate's Monte Carlo budget.
/// Nothing a caller passes can open a route.
///
/// # Errors
/// [`ReliabilityError::CellNotLicensed`] for every other coordinate.
pub fn license_route_at(
    route: PublishedRoute,
    coordinate: &CalibrationCoordinate,
    level: f64,
) -> Result<(), ReliabilityError> {
    let licensed =
        [calibration_basis(route, coordinate, level), exact_count_basis(route, coordinate, level)]
            .into_iter()
            .flatten()
            .any(|basis| calibration_slot(&basis).status == CalibrationStatus::Calibrated);
    if licensed {
        Ok(())
    } else {
        Err(ReliabilityError::CellNotLicensed {
            route: route.as_str(),
            coordinate: coordinate.describe(),
        })
    }
}

/// Every route that is closed at `coordinate`, with the refusal it would return.
#[must_use]
pub fn closed_routes(coordinate: &CalibrationCoordinate) -> Vec<ClosedRoute> {
    PublishedRoute::ALL
        .into_iter()
        .filter(|&route| license_route(route, coordinate).is_err())
        .map(|route| ClosedRoute {
            route,
            coordinate: coordinate.describe(),
            code: antecedent_core::reason_code!("cell_not_licensed").to_owned(),
            remedy: "point diagnostics only: a coverage record at this candidate-set size, \
                     Monte Carlo budget and selection rule would be needed"
                .to_owned(),
        })
        .collect()
}

/// How the simulated candidates' noise is coupled.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseCoupling {
    /// All candidates evaluated on one shared stream; `correlation` in `[0, 1]` is the
    /// share of each candidate's noise variance that comes from a component common to
    /// every candidate (common random numbers).
    SharedStream {
        /// Correlation of the candidates' per-draw noise.
        correlation: f64,
    },
    /// Each candidate on its own stream, declared independent.
    SeparateStreams,
}

/// A finite candidate set with known true utilities.
#[derive(Clone, Debug, PartialEq)]
pub struct KnownOptimumDesign {
    /// True expected utility of each candidate.
    pub true_means: Vec<f64>,
    /// Per-draw standard deviation of every candidate's utility.
    pub noise_sd: f64,
    /// Draws per candidate on each of the selection and validation streams.
    pub n_draws: usize,
    /// Noise coupling.
    pub coupling: NoiseCoupling,
}

impl KnownOptimumDesign {
    /// Identity of candidate `j`.
    #[must_use]
    pub fn candidate_id(j: usize) -> String {
        format!("c{j:05}")
    }

    /// Index of the true optimum (ties: smallest index).
    #[must_use]
    pub fn true_optimum(&self) -> usize {
        let mut best = 0;
        for (j, &m) in self.true_means.iter().enumerate() {
            if m > self.true_means[best] {
                best = j;
            }
        }
        best
    }
}

/// Simulate one stream's worth of candidate utilities under a design.
///
/// # Errors
/// An empty or non-finite design, a correlation outside `[0, 1]`, or a carrier defect.
pub fn simulate_selection_draws(
    design: &KnownOptimumDesign,
    stream: &StreamId,
) -> Result<DecisionDraws, ReliabilityError> {
    if design.true_means.is_empty()
        || design.true_means.iter().any(|m| !m.is_finite())
        || !design.noise_sd.is_finite()
        || design.noise_sd < 0.0
    {
        return Err(ReliabilityError::invalid("design", "known-optimum design is not finite"));
    }
    let n = design.n_draws;
    let k = design.true_means.len();
    // Known-optimum calibration harness: the design declares `n_draws` and runs on no study context.
    let mut columns = vec![vec![0.0_f64; n]; k];
    let (coupling, segments) = match design.coupling {
        NoiseCoupling::SharedStream { correlation } => {
            if !(0.0..=1.0).contains(&correlation) || !correlation.is_finite() {
                return Err(ReliabilityError::invalid("design", "correlation outside [0, 1]"));
            }
            let mut rng = stream.rng();
            let (common, own) = (correlation.sqrt(), (1.0 - correlation).sqrt());
            for i in 0..n {
                let c = standard_normal(&mut rng);
                for (j, column) in columns.iter_mut().enumerate() {
                    let e = standard_normal(&mut rng);
                    column[i] = design.true_means[j] + design.noise_sd * (common * c + own * e);
                }
            }
            (CandidateCoupling::SharedSimulation, vec![DrawSegment::whole(stream.clone(), n); k])
        }
        NoiseCoupling::SeparateStreams => {
            let mut segments = Vec::with_capacity(k);
            for (j, column) in columns.iter_mut().enumerate() {
                let own = StreamId::new(
                    stream.run_seed,
                    stream.purpose,
                    stream.replicate,
                    format!("{}/candidate/{j}", stream.label),
                );
                let mut rng = own.rng();
                for slot in column.iter_mut() {
                    *slot = design.true_means[j] + design.noise_sd * standard_normal(&mut rng);
                }
                segments.push(DrawSegment::whole(own, n));
            }
            (CandidateCoupling::DeclaredIndependent, segments)
        }
    };
    let candidates = columns
        .into_iter()
        .zip(segments)
        .enumerate()
        .map(|(j, (utilities, segment))| CandidateDraws {
            id: KnownOptimumDesign::candidate_id(j),
            utilities,
            segment,
        })
        .collect();
    Ok(DecisionDraws {
        utility_unit: "utility".into(),
        producer: "known_optimum_design".into(),
        source: SourceParadigm::KnownLaw,
        coupling,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates,
    })
}

/// What one replicate of the known-optimum design produced, against the known truth.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicateOutcome {
    /// Index of the candidate selected on the selection stream.
    pub selected_index: usize,
    /// Index of the true optimum.
    pub true_optimum_index: usize,
    /// Whether the selected candidate is a true optimum.
    pub selection_correct: bool,
    /// The unadjusted (in-sample) value of the selected candidate.
    pub in_sample_value: f64,
    /// Held-out value of the selected candidate on the validation stream.
    pub held_out_value: f64,
    /// True utility of the selected candidate.
    pub true_value_of_selected: f64,
    /// `in_sample_value - true_value_of_selected`.
    pub in_sample_bias: f64,
    /// `held_out_value - true_value_of_selected`.
    pub held_out_bias: f64,
    /// The winner's-curse bound reported for this replicate.
    pub winners_curse_bound: f64,
    /// Estimated probability that the selected candidate is best (uncalibrated).
    pub probability_best_of_selected: f64,
}

/// Run one replicate: select on a selection stream, re-evaluate on an independent
/// validation stream, and compare with the known truth.
///
/// Streams are named by `(run_seed, purpose, replicate)`; replicates never share noise.
///
/// # Errors
/// Design or carrier defects, or cancellation.
pub fn simulate_replicate(
    design: &KnownOptimumDesign,
    run_seed: u64,
    replicate: u64,
    ctx: &ExecutionContext,
) -> Result<ReplicateOutcome, ReliabilityError> {
    let selection_stream =
        StreamId::new(run_seed, StreamPurpose::Selection, replicate, "known_optimum");
    let validation_stream =
        StreamId::new(run_seed, StreamPurpose::Validation, replicate, "known_optimum");
    let selection = simulate_selection_draws(design, &selection_stream)?;
    let validation = simulate_selection_draws(design, &validation_stream)?;
    let request = ReliabilityRequest {
        decision: DecisionChoice::SelectedByRule,
        validation: ValidationPlan::ProvidedStream {
            kind: ValidationKind::IndependentSimulationStream,
            gap: 0,
        },
        // Rank probabilities are nonlinear functionals of the candidates' joint law;
        // independently drawn candidates have none to pair, so no rank statement is made.
        ranking: matches!(design.coupling, NoiseCoupling::SharedStream { .. })
            .then(RankingRequest::default),
        reporting: UncertaintyReporting::Separate,
        ..ReliabilityRequest::expected_utility()
    };
    let report = assess(&selection, Some(&validation), &request, ctx)?;
    let SelectionUncertainty::Selected(assessment) = &report.selection_uncertainty else {
        return Err(ReliabilityError::invalid("design", "selection was expected"));
    };
    let selected_index = (0..design.true_means.len())
        .find(|&j| KnownOptimumDesign::candidate_id(j) == report.decided_id)
        .ok_or_else(|| ReliabilityError::invalid("design", "selected id unknown"))?;
    let optimum = design.true_optimum();
    let truth = design.true_means[selected_index];
    let held_out = assessment
        .held_out
        .as_ref()
        .map(|h| h.held_out.estimate.value)
        .ok_or_else(|| ReliabilityError::invalid("design", "validation was expected"))?;
    let bound = assessment.winners_curse.as_ref().map_or(f64::NAN, |b| b.bound);
    let p_best = report
        .ranking
        .as_ref()
        .and_then(|r| r.statements.iter().find(|s| s.id == report.decided_id))
        .map_or(f64::NAN, |s| s.probability_best.value);
    Ok(ReplicateOutcome {
        selected_index,
        true_optimum_index: optimum,
        selection_correct: design.true_means[selected_index] >= design.true_means[optimum],
        in_sample_value: assessment.in_sample.value,
        held_out_value: held_out,
        true_value_of_selected: truth,
        in_sample_bias: assessment.in_sample.value - truth,
        held_out_bias: held_out - truth,
        winners_curse_bound: bound,
        probability_best_of_selected: p_best,
    })
}
