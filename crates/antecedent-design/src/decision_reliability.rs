//! Reliability of sampled and adaptively selected decisions.
//!
//! Computing a decision from a supplied law is different from establishing that the
//! decision is reliable. This module takes per-draw utilities of a finite candidate set
//! (a [`DecisionDraws`] carrier) and reports, as point diagnostics:
//!
//! * three uncertainties **separately** and never summed: spread under the supplied
//!   source law, Monte Carlo (numerical) error, and selection-induced uncertainty
//!   ([`ReliabilityReport`]); a request for one combined number is refused;
//! * Monte Carlo standard errors with an effective sample size for utility means,
//!   utility differences, threshold probabilities and risk measures (quantile via
//!   Maritz-Jarrett, `CVaR` via its influence function), accounting for common random
//!   numbers, serial dependence (batch means) and importance weights (Kish);
//! * ranking statements over the *finite* candidate set (best, top-k, better than a
//!   baseline), labelled with their paradigm and as uncalibrated diagnostics, refused
//!   when the candidates were generated adaptively without a [`CandidateSearchRecord`];
//! * selection-aware evaluation: the unadjusted maximum is labelled `in_sample`, with
//!   a winner's-curse bound, a held-out re-evaluation on an independent stream and
//!   sample splitting;
//! * the stream identities (semantic coordinates) that selected and validated, and the
//!   artifact `decision_reliability_v1` that a consumer recomputes from the draws.
//!
//! * a **derived** uncertainty method ([`interval`]): numerical intervals at the
//!   effective sample size, a confidence set for the best candidate and a
//!   selection-aware interval for the selected value, with the three components kept
//!   separate and labelled derived-unmeasured;
//! * the producer connection ([`producer`]): the cohort-decision and
//!   sequential-evaluation outputs reach the carrier only as per-draw utilities from a
//!   declared joint law, and refuse with a typed reason while they are point-only.
//!
//! No ranking probability or interval is published here. Every such route is closed
//! with a typed `cell_not_licensed` refusal until it is measured at its coordinate
//! (candidate-set size, Monte Carlo budget, selection rule); the simulation design
//! that will measure it lives in [`calibration`] and [`coverage_design`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod artifact;
pub mod calibration;
pub mod coverage_design;
pub mod draws;
pub mod interval;
pub mod mc_error;
pub mod producer;
pub mod ranking;
pub mod report;
pub mod selection;
pub mod stream;

use antecedent_contract::ExternalRefusal;

use crate::signal::RefusalBuilder;

pub use artifact::{
    DECISION_RELIABILITY_KIND, DecisionReliabilityArtifact, DecisionReliabilityExpectation,
    MAX_DECISION_RELIABILITY_BYTES, consume,
};
pub use calibration::{
    CalibrationCoordinate, ClosedRoute, KnownOptimumDesign, NoiseCoupling, PublishedRoute,
    ReplicateOutcome, closed_routes, license_route, simulate_replicate, simulate_selection_draws,
};
pub use coverage_design::{CoverageCell, TruthKind, coverage_design};
pub use draws::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, CandidateSearchRecord, DecisionDraws,
    SelectionRule, SourceParadigm, WithinStreamDependence,
};
pub use interval::{
    ConfidenceSetBasis, DERIVED_UNCERTAINTY_VERSION, DerivedStanding, DerivedUncertainty,
    IntervalMultiplicity, IntervalRequest, MIN_NORMAL_ESS, NumericalComponent, NumericalInterval,
    NumericalIntervalMethod, OptimalityConfidenceSet, ProbabilityInterval, ProbabilityQuantity,
    SelectionComponent, SelectionCorrectedInterval, UtilityInterval, derive_uncertainty,
};
pub use mc_error::{
    BatchPlan, DifferenceEstimate, Functional, FunctionalEstimate, McEstimate, PairCoupling,
    RiskTail, StandardErrorMethod, ThresholdSide, estimate, estimate_difference,
};
pub use producer::{
    CohortUtilityDraws, JointLawDeclaration, JointLawStanding, OffPolicyReturnDraws,
    PointOnlyOutput, ProducerCarrier, ProducerKind, ProducerOutput, RolloutNoise,
    RolloutUtilityDraws, reach_carrier,
};
pub use ranking::{
    CalibrationStatus, CandidateRankStatement, CandidateScope, RankingReport, RankingRequest,
    StatementLabel, TopKEstimate, rank,
};
pub use report::{
    ClaimKind, ClaimLabel, DecisionChoice, DependenceStatement, DifferenceVsDecision,
    MonteCarloErrorReport, ReliabilityReport, ReliabilityRequest, SelectionAssessment,
    SelectionUncertainty, SourceSpread, SourceUncertainty, StreamAccounting, UncertaintyReporting,
    ValidationPlan, assess,
};
pub use selection::{
    BoundKind, CandidateEstimate, HeldOutAssessment, InSampleSelection, NoiseDependence,
    SelectionStatus, ValidationKind, ValidationRecord, ValueLabel, WinnersCurseBound,
    held_out_evaluation, noise_dependence, select_in_sample, split_draws, winners_curse_bound,
};
pub use stream::{DECISION_RELIABILITY_STREAM_DOMAIN, DrawSegment, StreamId, StreamPurpose};

/// Most candidates one carrier may hold.
pub const MAX_CANDIDATES: usize = 65_536;
/// Most candidate-by-draw cells one carrier may hold (bounds memory and work).
pub const MAX_CELLS: usize = 1 << 26;
/// Fewest draws any estimate needs; below this no standard error is defined.
pub const MIN_DRAWS: usize = 4;

/// Why a reliability request was refused or could not complete.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ReliabilityError {
    /// A blank identity, mismatched length, non-finite value or unusable declaration.
    InvalidInput {
        /// Namespaced slot, `family.slot` without the module prefix.
        slot: &'static str,
        /// What is wrong.
        message: String,
    },
    /// The candidates came from an adaptive search and no search record was supplied.
    SearchRecordRequired {
        /// Candidate-set size supplied.
        candidates: usize,
    },
    /// The search record contradicts the supplied candidate set.
    SearchRecordInconsistent {
        /// What disagrees.
        message: String,
    },
    /// The validation draws are not independent of the selection draws.
    ValidationNotIndependent {
        /// Semantic coordinates of the selection segment that overlaps or shares noise.
        selection: String,
        /// Semantic coordinates of the validation segment.
        validation: String,
    },
    /// Candidates are compared by draw index without an aligned joint law.
    JointLawRequired {
        /// Why the pairing is not licensed.
        reason: &'static str,
    },
    /// Dependence between candidates, draws or repeated evaluations is unknown.
    DependenceUnknown {
        /// Which dependence is undeclared.
        what: &'static str,
    },
    /// One combined uncertainty number was requested.
    CombinedUncertainty,
    /// A published probability or interval route is closed pending measurement.
    CellNotLicensed {
        /// Route name.
        route: &'static str,
        /// Coordinate the route would be calibrated at.
        coordinate: String,
    },
    /// A producer handed over a point-only output: there is no law and no draws to take.
    ProducerPointOnly {
        /// The producing route.
        producer: ProducerKind,
        /// What the producer actually holds.
        held: &'static str,
    },
    /// Per-draw utilities disagree with the point value the producer reported.
    ProducerMeanDisagrees {
        /// Action or policy identity.
        id: String,
        /// The producer's reported value.
        reported: f64,
        /// Mean of the supplied draws.
        draws_mean: f64,
        /// Standard error of that mean.
        standard_error: f64,
    },
    /// The effective sample size is too small for the normal approximation.
    EffectiveSampleSizeTooSmall {
        /// Which estimate.
        what: &'static str,
        /// Its effective sample size.
        effective_sample_size: f64,
        /// The smallest the derivation accepts.
        minimum: f64,
    },
    /// The combination of estimator and declarations has no route.
    RouteNotSupported {
        /// What is outside the boundary.
        message: String,
    },
    /// A numerical failure (degenerate variance, non-finite result).
    Numerical {
        /// Context.
        message: String,
    },
    /// Cancellation was observed at a work boundary.
    Cancelled,
}

impl std::fmt::Display for ReliabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let refusal = self.to_refusal();
        write!(f, "{}: {}", refusal.code, refusal.detail)?;
        if let Some(offending) = &refusal.offending {
            write!(f, " ({offending})")?;
        }
        Ok(())
    }
}

impl std::error::Error for ReliabilityError {}

fn builder(code: &'static str, detail: &str) -> RefusalBuilder {
    RefusalBuilder::new(code, "declare", detail)
}

/// The refusal for a producer that holds only a point value, with the remedy for its route.
fn producer_point_only_refusal(producer: ProducerKind, held: &str) -> ExternalRefusal {
    match producer {
        ProducerKind::CohortMeanDecision => builder(
            antecedent_core::reason_code!("joint_law_required"),
            "decision_reliability.producer_point_only",
        )
        .offending(format!("{}: holds {held}", producer.as_str()))
        .remedy(
            "supply per-draw cohort utilities from one aligned joint law (for example \
             posterior draws of the effect model's coefficients mapped through the \
             cohort) with the stream they came from",
        )
        .build(),
        ProducerKind::SequentialRollout | ProducerKind::SequentialOffPolicy => builder(
            antecedent_core::reason_code!("diagnostic_not_available"),
            "decision_reliability.producer_point_only",
        )
        .offending(format!("{}: holds {held}", producer.as_str()))
        .remedy(
            "supply the per-rollout utilities or per-trajectory weighted returns the \
             estimate was computed from; a mean and its standard error carry neither \
             draws nor a stream",
        )
        .build(),
    }
}

fn producer_mean_disagrees_refusal(
    id: &str,
    reported: f64,
    draws_mean: f64,
    standard_error: f64,
) -> ExternalRefusal {
    builder(
        antecedent_core::reason_code!("distribution_meaning_mismatch"),
        "decision_reliability.producer_mean_disagrees",
    )
    .offending(format!(
        "{id}: reported {reported} but the supplied draws average {draws_mean} \
         (standard error {standard_error})"
    ))
    .remedy("supply draws of the same quantity the producer reported, or drop the check")
    .build()
}

fn effective_sample_size_refusal(
    what: &str,
    effective_sample_size: f64,
    minimum: f64,
) -> ExternalRefusal {
    builder(
        antecedent_core::reason_code!("route_not_supported"),
        "decision_reliability.effective_sample_size_too_small",
    )
    .offending(format!("{what}: effective sample size {effective_sample_size:.1}"))
    .expected(format!("effective sample size of at least {minimum}"))
    .remedy("draw more, or reduce serial dependence, before deriving intervals")
    .build()
}

impl ReliabilityError {
    pub(crate) fn invalid(slot: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidInput { slot, message: message.into() }
    }

    /// Structured refusal with a registered code, a namespaced detail and a remedy.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        match self {
            Self::InvalidInput { slot, message } => {
                builder(invalid, "decision_reliability.invalid_input")
                    .offending(format!("{slot}: {message}"))
                    .build()
            }
            Self::SearchRecordRequired { candidates } => builder(
                antecedent_core::reason_code!("candidate_search_record_required"),
                "decision_reliability.search_record_required",
            )
            .offending(format!("{candidates} adaptively generated candidates"))
            .remedy(
                "supply a CandidateSearchRecord naming the generator, its budget, the number \
                 evaluated and the selection rule, or enumerate the candidate set up front",
            )
            .build(),
            Self::SearchRecordInconsistent { message } => {
                builder(invalid, "decision_reliability.search_record_inconsistent")
                    .offending(message.clone())
                    .remedy("the record must cover every retained candidate within its budget")
                    .build()
            }
            Self::ValidationNotIndependent { selection, validation } => builder(
                antecedent_core::reason_code!("validation_stream_not_independent"),
                "decision_reliability.validation_not_independent",
            )
            .expected(format!("a stream or draw range disjoint from {selection}"))
            .supplied(validation.clone())
            .remedy(
                "re-evaluate on a stream with a different semantic coordinate, or split the \
                 draws into disjoint ranges separated by a gap longer than the autocorrelation",
            )
            .build(),
            Self::JointLawRequired { reason } => builder(
                antecedent_core::reason_code!("joint_law_required"),
                "decision_reliability.joint_law_required",
            )
            .offending(*reason)
            .remedy(
                "supply draws from one shared simulation (identical stream segments) or \
                 declare the candidates independent",
            )
            .build(),
            Self::DependenceUnknown { what } => builder(
                antecedent_core::reason_code!("sampling_dependence_unknown"),
                "decision_reliability.dependence_unknown",
            )
            .offending(*what)
            .remedy("declare the dependence; unknown dependence is never taken as independence")
            .build(),
            Self::CombinedUncertainty => builder(
                antecedent_core::reason_code!("uncertainty_components_not_combined"),
                "decision_reliability.combined_uncertainty",
            )
            .remedy(
                "read the three components separately; a combined statement needs a whole-procedure \
                 calibration at the coordinate (candidate-set size, Monte Carlo budget, selection rule)",
            )
            .build(),
            Self::CellNotLicensed { route, coordinate } => builder(
                antecedent_core::reason_code!("cell_not_licensed"),
                "decision_reliability.cell_not_licensed",
            )
            .offending(format!("{route} at {coordinate}"))
            .remedy(
                "the route is point-only until a coverage record measures it at this \
                 coordinate; use the diagnostic fields",
            )
            .build(),
            Self::ProducerPointOnly { producer, held } => {
                producer_point_only_refusal(*producer, held)
            }
            Self::ProducerMeanDisagrees { id, reported, draws_mean, standard_error } => {
                producer_mean_disagrees_refusal(id, *reported, *draws_mean, *standard_error)
            }
            Self::EffectiveSampleSizeTooSmall { what, effective_sample_size, minimum } => {
                effective_sample_size_refusal(what, *effective_sample_size, *minimum)
            }
            Self::RouteNotSupported { message } => builder(
                antecedent_core::reason_code!("route_not_supported"),
                "decision_reliability.route_not_supported",
            )
            .offending(message.clone())
            .build(),
            Self::Numerical { message } => builder(invalid, "decision_reliability.numerical")
                .offending(message.clone())
                .build(),
            Self::Cancelled => builder(
                antecedent_core::reason_code!("cancelled_no_claim"),
                "decision_reliability.cancelled",
            )
            .build(),
        }
    }

    /// Registered runtime reason code.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        self.to_refusal().code
    }
}
