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
//! No ranking probability or interval is published here. Every such route is closed
//! with a typed `cell_not_licensed` refusal until it is measured at its coordinate
//! (candidate-set size, Monte Carlo budget, selection rule); the simulation design
//! that will measure it lives in [`calibration`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod artifact;
pub mod calibration;
pub mod draws;
pub mod mc_error;
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
pub use draws::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, CandidateSearchRecord, DecisionDraws,
    SelectionRule, SourceParadigm, WithinStreamDependence,
};
pub use mc_error::{
    BatchPlan, DifferenceEstimate, Functional, FunctionalEstimate, McEstimate, PairCoupling,
    RiskTail, StandardErrorMethod, ThresholdSide, estimate, estimate_difference,
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
