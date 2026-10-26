//! Reliability of sampled and adaptively selected decisions as a facade object.
//!
//! A decision computed from a supplied law is not thereby reliable. A
//! [`PreparedDecisionReliability`] takes per-draw utilities of a finite candidate set (a
//! [`DecisionDraws`] carrier fed by the cohort-decision and sequential-evaluation
//! routes, or by any producer that declares its dependence) and reports, separately and
//! never summed:
//!
//! * **source-distribution uncertainty**: the spread of the utility under the supplied
//!   posterior or sampling law (none when the law is declared known);
//! * **Monte Carlo error**: standard errors and effective sample sizes of utility
//!   means, differences, threshold probabilities and risk measures, accounting for
//!   common random numbers, serial dependence and importance weights;
//! * **selection-induced uncertainty**: the unadjusted maximum labelled `in_sample`,
//!   with a winner's-curse bound, a held-out re-evaluation on an independent stream or
//!   a sample split, and the identity of the streams that selected and validated.
//!
//! Everything is a point diagnostic labelled uncalibrated. Ranking and optimality
//! probabilities (best, top-k, better than a baseline) are Monte Carlo estimates under
//! the supplied law over the candidate set as supplied; they refuse adaptively
//! generated candidates without a [`CandidateSearchRecord`]. Publishing any ranking
//! probability or interval ([`PreparedDecisionReliability::publish`]) refuses with
//! `cell_not_licensed` until a coverage record measures it at its coordinate
//! (candidate-set size, Monte Carlo budget, selection rule).
//!
//! # Lifecycle
//!
//! [`PreparedDecisionReliability::export`] writes `decision_reliability_v1`;
//! [`consume_decision_reliability_artifact`] recomputes every number from the carried
//! draws and refuses any edited draw, stream coordinate, declaration or claimed value,
//! and a consistently resealed forgery when the consumer retained the draws digest or
//! the selecting and validating streams. New draws are a new assessment: nothing is
//! refreshed in place.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use super::design_exports::design_crate::decision_reliability::{
    BatchPlan, CalibrationCoordinate, CandidateCoupling, CandidateDraws, CandidateOrigin,
    CandidateSearchRecord, DecisionChoice, DecisionDraws, DecisionReliabilityArtifact,
    DecisionReliabilityExpectation, DrawSegment, Functional, KnownOptimumDesign, NoiseCoupling,
    PublishedRoute, RankingRequest, ReliabilityReport, ReliabilityRequest, RiskTail, SelectionRule,
    SelectionUncertainty, SourceParadigm, SourceUncertainty, StreamId, StreamPurpose,
    ThresholdSide, UncertaintyReporting, ValidationKind, ValidationPlan, WithinStreamDependence,
};
pub use super::design_exports::design_crate::decision_reliability::{
    CohortUtilityDraws, ConfidenceSetBasis, CoverageCell, DERIVED_UNCERTAINTY_VERSION,
    DerivedStanding, DerivedUncertainty, IntervalMultiplicity, IntervalRequest,
    JointLawDeclaration, JointLawStanding, MIN_NORMAL_ESS, NumericalComponent, NumericalInterval,
    NumericalIntervalMethod, OffPolicyReturnDraws, OptimalityConfidenceSet, PointOnlyOutput,
    ProbabilityInterval, ProbabilityQuantity, ProducerCarrier, ProducerKind, ProducerOutput,
    RolloutNoise, RolloutUtilityDraws, SelectionComponent, SelectionCorrectedInterval, TruthKind,
    UtilityInterval,
};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    ReliabilityError, assess, consume, coverage_design, derive_uncertainty, license_route,
    reach_carrier,
};
use antecedent_io::IoError;

/// One refusal as plain data: registered reason code, namespaced
/// `decision_reliability.*` detail and the explanation of what would license the request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionReliabilityRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `decision_reliability.<rule>`.
    pub detail: String,
    /// What was refused and what would license it.
    pub message: String,
    /// What the caller can change to proceed, when the refusal names it.
    pub remedy: Option<&'static str>,
}

impl DecisionReliabilityRefusal {
    fn from_error(error: &ReliabilityError) -> Self {
        let refusal = error.to_refusal();
        let mut message = refusal.detail.clone();
        if let Some(offending) = &refusal.offending {
            message.push_str(&format!(": {offending}"));
        }
        if let Some(expected) = &refusal.expected {
            message.push_str(&format!("; expected {expected}"));
        }
        if let Some(remedy) = refusal.remedy {
            message.push_str(&format!("; {remedy}"));
        }
        Self { code: refusal.code, detail: refusal.detail, message, remedy: refusal.remedy }
    }

    fn from_io(error: &IoError) -> Self {
        if let IoError::Admission(error) = error {
            let cancelled = *error == antecedent_core::execution::AdmissionError::Cancelled;
            return Self {
                code: if cancelled {
                    antecedent_core::reason_code!("cancelled_no_claim")
                } else {
                    antecedent_core::reason_code!("invalid_argument")
                },
                detail: if cancelled {
                    "decision_reliability.cancelled"
                } else {
                    "decision_reliability.memory_budget_exceeded"
                }
                .to_owned(),
                message: error.to_string(),
                remedy: None,
            };
        }
        // The loader's own kind-mismatch detail survives; every other artifact
        // failure keeps the generic detail.
        if let IoError::Refused { code, message } = error {
            if let Some(text) =
                message.strip_prefix("decision_reliability.artifact_kind_mismatch: ")
            {
                return Self {
                    code: *code,
                    detail: "decision_reliability.artifact_kind_mismatch".to_owned(),
                    message: text.to_owned(),
                    remedy: None,
                };
            }
        }
        Self {
            code: error.reason_code().unwrap_or(antecedent_core::reason_code!("invalid_argument")),
            detail: "decision_reliability.artifact".to_owned(),
            message: error.to_string(),
            remedy: None,
        }
    }
}

/// What an inspection of a reliability assessment shows.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionReliabilityInspection {
    /// The reported decision.
    pub decided_id: String,
    /// Standing of every number: always an uncalibrated point diagnostic.
    pub standing: &'static str,
    /// Source-distribution uncertainty, summarized.
    pub source_uncertainty: String,
    /// Monte Carlo error of the decision's value, summarized.
    pub monte_carlo_error: String,
    /// Selection-induced uncertainty, summarized.
    pub selection_uncertainty: String,
    /// Routes that are closed at this coordinate, by name.
    pub closed_routes: Vec<&'static str>,
    /// The coordinate a calibration would have to measure.
    pub coordinate: String,
}

/// A decision-reliability assessment with the draws it was computed from.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedDecisionReliability {
    artifact: DecisionReliabilityArtifact,
}

impl PreparedDecisionReliability {
    /// Assess the reliability of a decision from its draws.
    ///
    /// `validation` carries independent draws when the request names a provided
    /// validation stream.
    ///
    /// # Errors
    /// A combined-uncertainty request, undeclared or contradicted dependence, an
    /// adaptive candidate set without its search record, validation draws that are not
    /// independent of the selection draws, or cancellation.
    pub fn assess(
        draws: DecisionDraws,
        validation: Option<DecisionDraws>,
        request: ReliabilityRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, DecisionReliabilityRefusal> {
        // Refuse before the artifact is built so the refusal keeps its own code.
        assess(&draws, validation.as_ref(), &request, ctx)
            .map_err(|error| DecisionReliabilityRefusal::from_error(&error))?;
        let artifact = DecisionReliabilityArtifact::new(draws, validation, request, ctx)
            .map_err(|error| DecisionReliabilityRefusal::from_io(&error))?;
        Ok(Self { artifact })
    }

    /// The report: three uncertainty components, ranking statements and closed routes.
    #[must_use]
    pub const fn report(&self) -> &ReliabilityReport {
        &self.artifact.report
    }

    /// The artifact (draws, request and report) the assessment is built on.
    #[must_use]
    pub const fn artifact(&self) -> &DecisionReliabilityArtifact {
        &self.artifact
    }

    /// One combined uncertainty number.
    ///
    /// # Errors
    /// Always `uncertainty_components_not_combined`.
    pub fn combined_uncertainty(&self) -> Result<f64, DecisionReliabilityRefusal> {
        self.artifact
            .report
            .combined_uncertainty()
            .map_err(|error| DecisionReliabilityRefusal::from_error(&error))
    }

    /// Publish a ranking or optimality probability, or an interval, as a calibrated
    /// claim at this assessment's coordinate.
    ///
    /// # Errors
    /// `cell_not_licensed` until a coverage record measures the route at the
    /// coordinate; the diagnostics in [`Self::report`] remain available.
    pub fn publish(&self, route: PublishedRoute) -> Result<(), DecisionReliabilityRefusal> {
        license_route(route, &self.artifact.report.coordinate)
            .map_err(|error| DecisionReliabilityRefusal::from_error(&error))
    }

    /// Plain-data summary for inspection.
    #[must_use]
    pub fn inspect(&self) -> DecisionReliabilityInspection {
        let report = &self.artifact.report;
        let source = match &report.source_uncertainty {
            SourceUncertainty::KnownLaw => {
                "none: the law was declared known; draws carry numerical error only".to_owned()
            }
            SourceUncertainty::Spread { paradigm, candidates } => format!(
                "spread of utility under the supplied {} law over {} candidate(s)",
                paradigm.as_str(),
                candidates.len()
            ),
        };
        let mc = &report.monte_carlo_error.decision;
        let selection = match &report.selection_uncertainty {
            SelectionUncertainty::NotSelected => {
                "none: the decision was fixed independently of the draws".to_owned()
            }
            SelectionUncertainty::Selected(assessment) => format!(
                "in_sample maximum over {} considered candidate(s); status {:?}",
                assessment.n_considered, assessment.status
            ),
        };
        DecisionReliabilityInspection {
            decided_id: report.decided_id.clone(),
            standing: "uncalibrated_point_diagnostic",
            source_uncertainty: source,
            monte_carlo_error: format!(
                "value {} with standard error {} (effective sample size {:.1}, {:?})",
                mc.value, mc.standard_error, mc.effective_sample_size, mc.method
            ),
            selection_uncertainty: selection,
            closed_routes: report.closed_routes.iter().map(|r| r.route.as_str()).collect(),
            coordinate: report.coordinate.describe(),
        }
    }

    /// Write `decision_reliability_v1`.
    ///
    /// # Errors
    /// A blank artifact identity or an artifact that no longer recomputes.
    pub fn export(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, DecisionReliabilityRefusal> {
        self.artifact
            .to_bytes(artifact_id, ctx)
            .map_err(|error| DecisionReliabilityRefusal::from_io(&error))
    }
}

impl PreparedDecisionReliability {
    /// Assess a producer's carrier, honouring what the producer allows it to claim.
    ///
    /// A carrier whose noise is shared only for variance reduction refuses ranking
    /// requests (`joint_law_required`); the exported artifact carries the draws and the
    /// request as accepted.
    ///
    /// # Errors
    /// Anything [`ProducerCarrier::assess`] or [`Self::assess`] refuses.
    pub fn assess_producer(
        carrier: &ProducerCarrier,
        validation: Option<DecisionDraws>,
        request: ReliabilityRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, DecisionReliabilityRefusal> {
        carrier
            .assess(validation.as_ref(), &request, ctx)
            .map_err(|error| DecisionReliabilityRefusal::from_error(&error))?;
        let artifact =
            DecisionReliabilityArtifact::new(carrier.draws().clone(), validation, request, ctx)
                .map_err(|error| DecisionReliabilityRefusal::from_io(&error))?;
        Ok(Self { artifact })
    }

    /// Derive the three uncertainty components for this assessment's decision: source
    /// distribution, Monte Carlo error at the effective sample size, and selection
    /// (confidence set for the best candidate and a selection-corrected interval).
    ///
    /// The result is labelled [`DerivedStanding::DerivedUnmeasured`]; it is recomputed
    /// from the carried draws, so nothing here is stored beside the artifact.
    ///
    /// # Errors
    /// A level outside `[0.5, 0.9999]`, an effective sample size below
    /// [`MIN_NORMAL_ESS`], or anything the assessment itself refuses.
    pub fn derive_uncertainty(
        &self,
        interval: &IntervalRequest,
        ctx: &ExecutionContext,
    ) -> Result<DerivedUncertainty, DecisionReliabilityRefusal> {
        derive_uncertainty(
            &self.artifact.draws,
            self.artifact.validation.as_ref(),
            &self.artifact.request,
            interval,
            ctx,
        )
        .map_err(|error| DecisionReliabilityRefusal::from_error(&error))
    }

    /// Re-derive `derived` from this assessment's draws and compare.
    ///
    /// # Errors
    /// A derivation that does not recompute, or anything [`Self::derive_uncertainty`]
    /// refuses.
    pub fn verify_derived(
        &self,
        derived: &DerivedUncertainty,
        ctx: &ExecutionContext,
    ) -> Result<(), DecisionReliabilityRefusal> {
        derived
            .verify(
                &self.artifact.draws,
                self.artifact.validation.as_ref(),
                &self.artifact.request,
                ctx,
            )
            .map_err(|error| DecisionReliabilityRefusal::from_error(&error))
    }
}

/// Derive the three uncertainty components directly from draws.
///
/// # Errors
/// As [`PreparedDecisionReliability::derive_uncertainty`], plus anything
/// [`PreparedDecisionReliability::assess`] refuses.
pub fn derive_decision_uncertainty(
    draws: &DecisionDraws,
    validation: Option<&DecisionDraws>,
    request: &ReliabilityRequest,
    interval: &IntervalRequest,
    ctx: &ExecutionContext,
) -> Result<DerivedUncertainty, DecisionReliabilityRefusal> {
    derive_uncertainty(draws, validation, request, interval, ctx)
        .map_err(|error| DecisionReliabilityRefusal::from_error(&error))
}

/// Derive the three components for a producer's carrier; a marginal-only carrier refuses
/// ranking requests.
///
/// # Errors
/// As [`ProducerCarrier::derive_uncertainty`].
pub fn derive_producer_uncertainty(
    carrier: &ProducerCarrier,
    validation: Option<&DecisionDraws>,
    request: &ReliabilityRequest,
    interval: &IntervalRequest,
    ctx: &ExecutionContext,
) -> Result<DerivedUncertainty, DecisionReliabilityRefusal> {
    carrier
        .derive_uncertainty(validation, request, interval, ctx)
        .map_err(|error| DecisionReliabilityRefusal::from_error(&error))
}

/// Take a cohort-decision or sequential-evaluation output to the reliability carrier.
///
/// # Errors
/// A point-only output (`joint_law_required` for a cohort mean,
/// `diagnostic_not_available` for a rollout or off-policy mean), cohort utilities
/// without a declared shared law, a reported value the draws disagree with, or a carrier
/// defect.
pub fn reach_decision_carrier(
    output: ProducerOutput,
) -> Result<ProducerCarrier, DecisionReliabilityRefusal> {
    reach_carrier(output).map_err(|error| DecisionReliabilityRefusal::from_error(&error))
}

/// Ask to publish a route from a derivation.
///
/// # Errors
/// `cell_not_licensed` until a coverage record measures the route at the derivation's
/// coordinate.
pub fn publish_derived_route(
    derived: &DerivedUncertainty,
    route: PublishedRoute,
) -> Result<(), DecisionReliabilityRefusal> {
    derived.publish(route).map_err(|error| DecisionReliabilityRefusal::from_error(&error))
}

/// The coverage-measurement design for every derived interval and set, as data. Nothing
/// is measured here.
#[must_use]
pub fn decision_reliability_coverage_design() -> Vec<CoverageCell> {
    coverage_design()
}

/// Consume `decision_reliability_v1` bytes (recomputing the report from the carried
/// draws) and derive the three uncertainty components from those draws: a fresh consumer
/// never trusts a stored interval.
///
/// # Errors
/// As [`consume_decision_reliability_artifact`] and
/// [`PreparedDecisionReliability::derive_uncertainty`].
pub fn consume_derived_uncertainty(
    bytes: &[u8],
    expected: &DecisionReliabilityExpectation,
    interval: &IntervalRequest,
    ctx: &ExecutionContext,
) -> Result<DerivedUncertainty, DecisionReliabilityRefusal> {
    consume_decision_reliability_artifact(bytes, expected, ctx)?.derive_uncertainty(interval, ctx)
}

/// Independently consume `decision_reliability_v1` bytes: every number is recomputed
/// from the carried draws and compared bit for bit.
///
/// # Errors
/// A damaged container, an edited draw, stream, declaration or claim, or a retained
/// expectation that differs.
pub fn consume_decision_reliability_artifact(
    bytes: &[u8],
    expected: &DecisionReliabilityExpectation,
    ctx: &ExecutionContext,
) -> Result<PreparedDecisionReliability, DecisionReliabilityRefusal> {
    // A readable container of another kind is a recognised artifact this consumer
    // does not read: name both kinds and the consumer that does.
    if let Some(kind) = antecedent_design::inverse_query_artifact::recognised_artifact_kind(bytes) {
        if kind != antecedent_design::decision_reliability::DECISION_RELIABILITY_KIND {
            let message = format!(
                "decision_reliability.artifact_kind_mismatch: expected {}, supplied {kind}; {}",
                antecedent_design::decision_reliability::DECISION_RELIABILITY_KIND,
                antecedent_design::inverse_query_artifact::artifact_kind_remedy(&kind),
            );
            return Err(DecisionReliabilityRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "decision_reliability.artifact_kind_mismatch".to_owned(),
                message,
                remedy: Some(antecedent_design::inverse_query_artifact::artifact_kind_remedy(
                    &kind,
                )),
            });
        }
    }
    let artifact = consume(bytes, expected, ctx)
        .map_err(|error| DecisionReliabilityRefusal::from_io(&error))?;
    Ok(PreparedDecisionReliability { artifact })
}
