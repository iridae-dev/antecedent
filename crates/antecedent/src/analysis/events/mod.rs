//! Compiler-owned forecasts of timestamped events and their composed world responses.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::{ExecutionContract, ProviderObjectIdentity, ProviderVerificationReceipt};
use antecedent_core::execution::{AdmissionError, AdmissionLease, RngFactory};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use serde::{Deserialize, Serialize};

use super::program_handle::{CompiledProgramHandle, ProgramPublication, gate_program};
use crate::Study;
use crate::planner::lowered::{EventProcessRoute, LoweredProgram};
use crate::planner::{PlanStage, PlanView};

mod assessment;
mod challenge;
mod discovery;
mod dynamics;
mod hawkes;
mod marked;
mod offspring;
mod question;
mod release;
#[cfg(test)]
mod structure_tests;
mod study;

pub use antecedent_contract::bridge::BridgeContract;
pub use assessment::{
    CountCoverageCandidate, ForecastAssessment, ForecastAssessmentRequest, ForecastOrigin,
    ForecastScore, PreparedForecastAssessment, PublishedCountInterval, ReliabilityBin,
};
pub use challenge::{
    DiagnosticStatus, ExcitationChallenge, ExcitationReading, MarkedExcitationChallenge,
    MarkedReleaseComparisonRequest, PreparedExcitationChallenge, PreparedMarkedChallenge,
    PreparedReleaseComparison, ReleaseChoice, ReleaseComparison, ReleaseComparisonRequest,
};
pub use discovery::{MemberReading, StructuralReading};
pub use dynamics::{
    EventBinding, EventDecision, EventWorldRequest, GridDifference, PathFunctional, PathOutput,
    PreparedEventWorlds,
};
pub use hawkes::{
    BaselineFeature, ExcitationOffspring, HawkesFit, HawkesFitSpec, HawkesParameters,
};
pub use marked::{
    MAX_PROTOTYPES, MarkPrototype, MarkedArrival, MarkedFit, MarkedFitSpec, MarkedHistory,
    MarkedOffspring, MarkedParameters, PreparedMarkedDraw, PreparedMarkedFit,
    PreparedMarkedLikelihood,
};
pub use offspring::{PreparedExcitationOffspring, PreparedMarkedOffspring};
pub use question::{
    AcceptedExcitation, Continuation, DeclaredQuantity, ExcitationSubject, ExcludedShock,
    LagCountQuery, PreparedExcitationQuestion, PreparedShock, ReleaseAction, ReleaseDraw,
    parameter_bridge,
};
pub use release::{
    PreparedReleaseLaw, PreparedReleaseQuestion, QuestionChallenge, ReleaseQuestion,
    ReleaseQuestionOutcome,
};
pub use study::{
    EventBelief, EventHypothesis, EventObservation, EventStudyRequest, EventStudyValue,
    MemberReleaseStudy, MemberStudyValue, PreparedEventStudy, PreparedEventUpdate,
    PreparedMemberReleaseStudy,
};

pub(crate) const CAPABILITY: &str =
    crate::support::units_generated::FORECAST_EVENT_PROCESS.as_str();
pub(crate) const COUNT_INTERVAL: &str = "forecast.event_count_interval";
/// The support unit of the finite-rank marked kernel routes.
pub(crate) const MARKED_UNIT: &str =
    crate::support::units_generated::FORECAST_MARKED_EVENT_PROCESS.as_str();
/// The support unit of the offspring, immigrant-challenge and release-comparison routes.
pub(crate) const EXCITATION_UNIT: &str =
    crate::support::units_generated::FORECAST_HAWKES_EXCITATION.as_str();
/// The support unit of the parameter-intervention bridge and what it composes.
pub(crate) const BRIDGE_UNIT: &str =
    crate::support::units_generated::FORECAST_HAWKES_PARAMETER_BRIDGE.as_str();
const MAX_BYTES: u64 = 64 * 1024 * 1024;
/// Operation charges for one event-process invocation, shared with the Hawkes work counter.
pub(super) const OPERATION_BUDGET: usize = 250_000_000;
pub(super) const MAX_EVENTS: usize = 100_000;
const MAX_PATHS: usize = 4096;

/// Largest dense event catalog.
///
/// A likelihood build charges `(MAX_EVENTS + 1) * n²` operations. This is the
/// largest `n` for which that build uses at most half the operation budget, so
/// gradient evaluations still fit in the same invocation. The work counter
/// refuses a fit or forecast that exceeds the budget even inside this cap.
/// The bound is the dense per-pair kernel's arithmetic, not a property of the
/// Hawkes process.
pub const MAX_EVENT_TYPES: usize = max_dense_types(MAX_EVENTS, OPERATION_BUDGET);

/// Largest `n` with `(max_events + 1) * n² <= budget / 2`.
#[must_use]
pub const fn max_dense_types(max_events: usize, budget: usize) -> usize {
    let allowance = budget / 2;
    let rows = max_events.saturating_add(1);
    let mut n = 1usize;
    loop {
        let next = n + 1;
        if next > 4096 {
            return n;
        }
        let cost = rows.saturating_mul(next).saturating_mul(next);
        if cost > allowance {
            return n;
        }
        n = next;
    }
}

/// A typed refusal from event compilation or execution.
#[derive(Debug, thiserror::Error)]
pub enum EventError {
    /// The declaration, lineage or schedule violates the event contract.
    #[error("events.invalid_request: {0}")]
    Invalid(String),
    /// Stability could not be certified inside the stationary domain.
    #[error("events.unstable: cannot certify spectral radius below one")]
    Unstable,
    /// Fit failed or identification is too weak for the requested operation.
    #[error("events.fit_failed: {0}")]
    Fit(&'static str),
    /// An event, operation or retained-resource bound was exceeded.
    #[error("events.budget: event operation or resource limit exceeded")]
    Budget,
    /// A numerical condition prevents a complete law.
    #[error("events.numerical: {0}")]
    Numerical(&'static str),
    /// A provider claim lacks exact-request verification.
    #[error("events.unverified_source: external event paths lack exact-request verification")]
    Unverified,
    /// An interval has no matching release evidence.
    #[error("events.interval_unmeasured: no matching measured count coverage")]
    IntervalUnmeasured,
    /// The solver failed on an explicitly identified path and world.
    #[error("events.incomplete_law: path {path}, world {world}: {detail}")]
    IncompleteLaw {
        /// Retained path index.
        path: usize,
        /// Failing action world.
        world: String,
        /// Original validation or solver message.
        detail: String,
    },
    /// Native compiler or retained execution failure.
    #[error(transparent)]
    Compiler(#[from] crate::error::CausalError),
    /// The parameter-intervention bridge refused the structural reading of an excitation mass.
    #[error("{detail}: {message}")]
    BridgeRefused {
        /// Registered `events` refusal detail of the bridge.
        detail: &'static str,
        /// Registered runtime reason code.
        code: &'static str,
        /// What refused, and the declaration or evidence that would support it.
        message: String,
    },
}

impl EventError {
    /// Stable route-specific reason, when the error originated in this compiler route.
    #[must_use]
    pub fn detail(&self) -> Option<&'static str> {
        use antecedent_core::reason_details::events;
        match self {
            Self::Invalid(_) => Some(events::INVALID_REQUEST),
            Self::Unstable => Some(events::UNSTABLE),
            Self::Fit(_) => Some(events::FIT_FAILED),
            Self::Budget => Some(events::BUDGET),
            Self::Numerical(_) => Some(events::NUMERICAL),
            Self::Unverified => Some(events::UNVERIFIED_SOURCE),
            Self::IntervalUnmeasured => Some(events::INTERVAL_UNMEASURED),
            Self::IncompleteLaw { .. } => Some(events::INCOMPLETE_LAW),
            Self::Compiler(_) => None,
            Self::BridgeRefused { detail, .. } => Some(*detail),
        }
    }

    /// Stable top-level refusal code for the route failure.
    #[must_use]
    pub fn reason_code(&self) -> Option<&'static str> {
        use antecedent_core::reason_code;
        Some(match self {
            Self::Invalid(_) | Self::Unstable | Self::Numerical(_) | Self::IncompleteLaw { .. } => {
                reason_code!("invalid_argument")
            }
            Self::Budget => reason_code!("execution_resource_limit"),
            Self::Fit(_) => reason_code!("dynamics_not_identified"),
            Self::Unverified => reason_code!("external_verification_failed"),
            Self::IntervalUnmeasured => reason_code!("uncertainty_representation_insufficient"),
            Self::Compiler(_) => return None,
            Self::BridgeRefused { code, .. } => code,
        })
    }

    pub(super) fn invalid(error: impl std::fmt::Display) -> Self {
        Self::Invalid(error.to_string())
    }

    /// A refusal of the parameter-intervention bridge under its registered detail and code.
    pub(super) fn bridge(
        detail: &'static str,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self::BridgeRefused { detail, code, message: message.into() }
    }
}

pub(super) fn poll(ctx: &ExecutionContext) -> Result<(), EventError> {
    if ctx.cancellation.is_cancelled() {
        return Err(crate::error::CausalError::Cancelled { stage: "event process" }.into());
    }
    Ok(())
}

pub(super) fn admit(
    ctx: &ExecutionContext,
    scratch: u64,
    retained: u64,
) -> Result<AdmissionLease, EventError> {
    let bytes = scratch.checked_add(retained).ok_or(EventError::Budget)?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        AdmissionError::Cancelled => {
            crate::error::CausalError::Cancelled { stage: "event process admission" }.into()
        }
        _ => EventError::Budget,
    })
}

pub(super) fn digest(tag: &str, value: &impl Serialize) -> Result<SemanticDigest, EventError> {
    let encoded = serde_json::to_vec(value).map_err(EventError::invalid)?;
    if encoded.len() as u64 > MAX_BYTES {
        return Err(EventError::Budget);
    }
    let mut writer = CanonicalWriter::new(tag);
    writer.bytes(&encoded);
    Ok(SemanticDigest::from_bytes(writer.finish()))
}

pub(super) fn release_supported(id: &str) -> bool {
    crate::support::capability_supported(id)
}

/// A closed-open timestamp interval `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventWindow {
    /// Included start.
    pub start: f64,
    /// Excluded end.
    pub end: f64,
}

impl EventWindow {
    pub(super) fn validate(self) -> Result<(), EventError> {
        if !self.start.is_finite()
            || !self.end.is_finite()
            || self.end <= self.start
            || !(self.end - self.start).is_finite()
        {
            return Err(EventError::invalid("invalid event window"));
        }
        Ok(())
    }
}

/// One arrival in an ordered, named catalog.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arrival {
    /// Timestamp in the history's clock.
    pub time: f64,
    /// Index in the declared event catalog.
    pub kind: usize,
}

/// A complete observation window with explicit left conditioning.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventHistory {
    /// Unique event names in fixed order.
    pub catalog: Vec<String>,
    /// Unit of timestamp, decay and rate conversions.
    pub time_unit: String,
    /// Named origin for absolute time.
    pub origin: String,
    /// Stable source or data collection identity.
    pub source: String,
    /// Start of the supplied complete history. Excitation is declared empty there.
    pub empty_origin: f64,
    /// Complete observation window used for likelihood.
    pub window: EventWindow,
    /// Observed arrivals between the empty origin and window start.
    pub left: Vec<Arrival>,
    /// Arrivals in the observation window.
    pub events: Vec<Arrival>,
    /// Explicit gaps. Incomplete windows cannot condition a native forecast.
    #[serde(default)]
    pub gaps: Vec<EventWindow>,
}

impl EventHistory {
    fn validate(&self, native: bool) -> Result<(), EventError> {
        self.window.validate()?;
        let valid_name = |s: &str| !s.trim().is_empty() && s.len() <= 256;
        if self.catalog.is_empty()
            || self.catalog.len() > MAX_EVENT_TYPES
            || self.catalog.iter().collect::<BTreeSet<_>>().len() != self.catalog.len()
            || self.catalog.iter().any(|kind| !valid_name(kind))
            || [&self.time_unit, &self.origin, &self.source].iter().any(|name| !valid_name(name))
            || self.left.len().saturating_add(self.events.len()) > MAX_EVENTS
            || !self.empty_origin.is_finite()
            || self.empty_origin > self.window.start
            || !(self.window.end - self.empty_origin).is_finite()
            || !self.gaps.is_empty()
        {
            return Err(EventError::invalid(
                "invalid event catalog, clock or incomplete event history",
            ));
        }
        validate_arrivals(
            &self.left,
            self.empty_origin,
            self.window.start,
            self.catalog.len(),
            native,
        )?;
        validate_arrivals(
            &self.events,
            self.window.start,
            self.window.end,
            self.catalog.len(),
            native,
        )
    }

    pub(super) fn prefix(&self) -> Vec<Arrival> {
        self.left.iter().chain(&self.events).copied().collect()
    }
}

#[allow(clippy::float_cmp, reason = "tied native arrivals are detected by exact time equality")]
fn validate_arrivals(
    arrivals: &[Arrival],
    start: f64,
    end: f64,
    types: usize,
    strict: bool,
) -> Result<(), EventError> {
    if arrivals
        .iter()
        .any(|e| !e.time.is_finite() || e.time < start || e.time >= end || e.kind >= types)
        || arrivals.windows(2).any(|w| w[0].time > w[1].time || (strict && w[0].time == w[1].time))
    {
        return Err(EventError::invalid("out-of-window, unordered or tied native arrivals"));
    }
    Ok(())
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "ProviderObjectIdentity", deny_unknown_fields)]
#[allow(
    clippy::struct_field_names,
    reason = "the fields mirror the provider object identity wire names"
)]
struct EventProviderIdentityWire {
    provider_id: String,
    object_id: String,
    version_id: String,
    snapshot_id: String,
    request_id: String,
}

/// Source of a joint event path law.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventSource {
    /// Explicit finite, equally weighted paths, retaining their declared pairing.
    Fixture {
        /// One ordered event list per path.
        paths: Vec<Vec<Arrival>>,
    },
    /// Equally weighted paths from a separately checked provider.
    External {
        /// Exact provider object key.
        #[serde(with = "EventProviderIdentityWire")]
        identity: ProviderObjectIdentity,
        /// One ordered event list per path.
        paths: Vec<Vec<Arrival>>,
    },
    /// Forecast at fixed, known parameters.
    Declared {
        /// Fixed parameter model.
        parameters: HawkesParameters,
    },
    /// Estimate parameters and optionally propagate a recursive refit law.
    Fitted {
        /// Fixed-decay likelihood scope.
        fit: HawkesFitSpec,
        /// Whole-history refits. Zero selects plug-in prediction.
        #[serde(default)]
        refits: usize,
    },
}

/// A complete forecast statement and operational limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventForecastRequest {
    /// Complete history up to forecast start.
    pub history: EventHistory,
    /// Horizon, with start equal to history end.
    pub forecast: EventWindow,
    /// Source law for forecast arrivals.
    pub source: EventSource,
    /// Number of retained paths.
    pub paths: usize,
    /// Maximum events in any one sampled history or forecast.
    pub max_events: usize,
    /// Independent forecast and refit stream seed.
    pub seed: u64,
}

impl EventForecastRequest {
    #[allow(
        clippy::float_cmp,
        reason = "the forecast must start exactly at the observation cutoff"
    )]
    pub(super) fn validate(&self) -> Result<(), EventError> {
        self.forecast.validate()?;
        validate_budget(self.paths, self.max_events)?;
        self.history.validate(matches!(
            &self.source,
            EventSource::Fitted { .. } | EventSource::Declared { .. }
        ))?;
        if self.forecast.start != self.history.window.end {
            return Err(EventError::invalid("forecast start must equal observation cutoff"));
        }
        match &self.source {
            EventSource::Fixture { paths } | EventSource::External { paths, .. } => {
                if paths.len() != self.paths
                    || paths.iter().any(|path| path.len() > self.max_events)
                {
                    return Err(EventError::invalid(
                        "external or fixture path cardinality mismatch",
                    ));
                }
                for path in paths {
                    validate_arrivals(
                        path,
                        self.forecast.start,
                        self.forecast.end,
                        self.history.catalog.len(),
                        false,
                    )?;
                }
                Ok(())
            }
            EventSource::Declared { parameters } => parameters.validate(self.history.catalog.len()),
            EventSource::Fitted { fit, refits } => {
                if *refits != 0
                    && (*refits < 8 || *refits > 128 || !self.paths.is_multiple_of(*refits))
                {
                    return Err(EventError::invalid("use 8–128 equal-size refit blocks or zero"));
                }
                fit.validate(self.history.catalog.len())
            }
        }
    }

    /// Digest the exact payload that an external provider must bind as its request identifier.
    ///
    /// # Errors
    /// A native source, invalid history or oversized path declaration.
    pub fn provider_request_id(&self) -> Result<String, EventError> {
        self.validate()?;
        let EventSource::External { paths, .. } = &self.source else {
            return Err(EventError::invalid("expected external source"));
        };
        Ok(digest(
            "antecedent.events.provider.request.v1",
            &(&self.history, self.forecast, paths, self.max_events, self.seed),
        )?
        .to_hex())
    }
}

pub(super) fn validate_budget(paths: usize, max_events: usize) -> Result<(), EventError> {
    if !(1..=MAX_PATHS).contains(&paths)
        || !(1..=MAX_EVENTS).contains(&max_events)
        || paths.saturating_mul(max_events) > 2_000_000
    {
        return Err(EventError::Budget);
    }
    Ok(())
}

/// Meaning of the parameter values used to generate paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventLawMeaning {
    /// Explicit fixture scenarios.
    Fixture,
    /// Independently attested external joint paths.
    VerifiedExternal,
    /// Conditional on fixed model parameters.
    Fixed,
    /// Conditional on estimated point parameters.
    PlugIn,
    /// Predictive propagation through a recursive parametric refit distribution.
    RecursiveRefit,
    /// Conditional on an explicitly weighted finite parameter belief.
    WeightedBelief,
    /// Continuation after one added event, once a parameter bridge has accepted the direct mass.
    ///
    /// These paths are not an observational forecast.
    Intervened,
}

/// Immutable forecast compiled by the causal compiler.
pub struct PreparedEventForecast {
    request: Arc<EventForecastRequest>,
    program: Arc<LoweredProgram>,
    provider_receipt: Option<String>,
    _lease: AdmissionLease,
}

/// Retained compiled event paths, with source and operation lineage.
#[derive(Clone)]
pub struct EventLaw {
    history: Arc<EventHistory>,
    window: EventWindow,
    paths: Arc<Vec<Vec<Arrival>>>,
    weights: Option<Arc<Vec<f64>>>,
    path_left: Option<Arc<Vec<Vec<Arrival>>>>,
    meaning: EventLawMeaning,
    identity: SemanticDigest,
    provider: Option<ProviderObjectIdentity>,
    provider_receipt: Option<String>,
    model_fit: Option<Arc<HawkesFit>>,
    refit_parameters: Option<Arc<Vec<HawkesParameters>>>,
    release_supported: bool,
    _lease: Arc<AdmissionLease>,
}

impl Study {
    /// Lower an event forecast to a native plan before fitting or sampling.
    ///
    /// # Errors
    /// Ineligible history, unsupported source, unverified provider or resource refusal.
    pub fn prepare_event_forecast(
        request: EventForecastRequest,
        receipt: Option<&ProviderVerificationReceipt>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEventForecast, EventError> {
        super::lowering::refuse_reentry(ctx)?;
        poll(ctx)?;
        request.validate()?;
        let provider_receipt = if let EventSource::External { identity, .. } = &request.source {
            let request_id = request.provider_request_id()?;
            if !identity.is_complete() || identity.request_id != request_id {
                return Err(EventError::Unverified);
            }
            // Verification must cover this event payload and this operation.
            let contract = external_execution_contract(identity.clone());
            let receipt =
                receipt.filter(|r| r.covers_execution(&contract)).ok_or(EventError::Unverified)?;
            Some(receipt.digest())
        } else {
            if receipt.is_some() {
                return Err(EventError::invalid("receipt belongs to an external event source"));
            }
            None
        };
        let source = digest("antecedent.events.forecast.v1", &(&request, &provider_receipt))?;
        let bytes = history_bytes(&request.history)
            + match &request.source {
                EventSource::Fixture { paths } | EventSource::External { paths, .. } => {
                    paths.iter().map(|p| p.len() as u64 * 16 + 24).sum()
                }
                _ => 32768,
            };
        let lease = admit(ctx, 0, bytes)?;
        let operation = match &request.source {
            EventSource::Fitted { refits: 0, .. } => "fit_and_forecast",
            EventSource::Fitted { .. } => "refit_and_forecast",
            _ => "forecast",
        };
        let program = Arc::new(super::lowering::events::lower(
            EventProcessRoute::Forecast,
            source,
            bytes,
            operation,
            CAPABILITY,
            None,
            None,
        )?);
        Ok(PreparedEventForecast {
            request: Arc::new(request),
            program,
            provider_receipt,
            _lease: lease,
        })
    }
}

fn history_bytes(history: &EventHistory) -> u64 {
    (history.events.len() + history.left.len()) as u64 * 16 + 8192
}

impl PreparedEventForecast {
    /// Stable identity of the prepared forecast.
    pub fn program_identity(&self) -> SemanticDigest {
        self.program.identity()
    }

    /// Inspect without fitting or forecasting.
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.program), PlanStage::Prepared)
    }

    /// Replay the forecast only when its compiled identity matches the consumer's expectation.
    ///
    /// # Errors
    /// Changed program identity or an execution refusal.
    pub fn replay(
        &self,
        expected: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<EventLaw, EventError> {
        if expected != self.program_identity() {
            return Err(EventError::invalid("replay program identity changed"));
        }
        self.execute(ctx)
    }

    /// Execute and retain the source-bound path law.
    ///
    /// # Errors
    /// Resource exhaustion, failed fit/refit, cancellation, or numerical failure.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<EventLaw, EventError> {
        let (scoped, publication) = scoped_execution(ctx, self)?;
        let ctx = &scoped;
        publication.finish(self.sample_retained(ctx))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one pass from source sampling through retained paths"
    )]
    fn sample_retained(&self, ctx: &ExecutionContext) -> Result<EventLaw, EventError> {
        let request = &self.request;
        let history = &request.history;
        let retained = path_bytes(request.paths, request.max_events)
            + history_bytes(history)
            + 128 * (history.catalog.len() * (history.catalog.len() + 1) * 16) as u64;
        let lease = Arc::new(admit(ctx, 0, retained)?);
        let scratch = history_bytes(history)
            + (history.events.len() as u64 + request.max_events as u64)
                * (history.catalog.len() as u64 * 8 + 40);
        let _scratch = admit(ctx, scratch, 0)?;
        let mut work = hawkes::Work::new(ctx);
        let rng = RngFactory::from_seed(request.seed);
        let prefix = history.prefix();
        let mut fit = None;
        let mut refit_parameters = None;
        let mut provider = None;
        let native_paths = |parameters: &HawkesParameters, work: &mut hawkes::Work<'_>| {
            (0..request.paths)
                .map(|i| {
                    hawkes::sample(
                        history,
                        &prefix,
                        parameters,
                        request.forecast,
                        (&rng, hawkes::FORECAST, i as u64),
                        request.max_events,
                        work,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let (paths, meaning) = match &request.source {
            EventSource::Fixture { paths } => (paths.clone(), EventLawMeaning::Fixture),
            EventSource::External { identity, paths } => {
                provider = Some(identity.clone());
                (paths.clone(), EventLawMeaning::VerifiedExternal)
            }
            EventSource::Declared { parameters } => {
                (native_paths(parameters, &mut work)?, EventLawMeaning::Fixed)
            }
            EventSource::Fitted { fit: spec, refits } => {
                let model = hawkes::fit(history, spec, &mut work)?;
                let paths = if *refits == 0 {
                    native_paths(&model.parameters, &mut work)?
                } else {
                    if model.weak_identification
                        || !model.active_columns.is_empty()
                        || !model.zero_edges.is_empty()
                    {
                        return Err(EventError::Fit(
                            "recursive refits require an interior identified original fit",
                        ));
                    }
                    let mut fitted = Vec::with_capacity(*refits);
                    let mut paths = Vec::with_capacity(request.paths);
                    let mut replicate = history.clone();
                    for i in 0..*refits {
                        replicate.events = hawkes::sample(
                            history,
                            &history.left,
                            &model.parameters,
                            history.window,
                            (&rng, hawkes::REFIT, i as u64),
                            request.max_events,
                            &mut work,
                        )?;
                        let draw = hawkes::fit(&replicate, spec, &mut work)?;
                        for j in 0..request.paths / *refits {
                            paths.push(hawkes::sample(
                                history,
                                &prefix,
                                &draw.parameters,
                                request.forecast,
                                (
                                    &rng,
                                    hawkes::FORECAST,
                                    (i * (request.paths / *refits) + j) as u64,
                                ),
                                request.max_events,
                                &mut work,
                            )?);
                        }
                        fitted.push(draw.parameters);
                    }
                    refit_parameters = Some(Arc::new(fitted));
                    paths
                };
                fit = Some(Arc::new(model));
                (
                    paths,
                    if *refits == 0 {
                        EventLawMeaning::PlugIn
                    } else {
                        EventLawMeaning::RecursiveRefit
                    },
                )
            }
        };
        Ok(EventLaw {
            history: Arc::new(history.clone()),
            window: request.forecast,
            paths: Arc::new(paths),
            weights: None,
            path_left: None,
            meaning,
            identity: self.program.identity(),
            provider,
            provider_receipt: self.provider_receipt.clone(),
            model_fit: fit,
            refit_parameters,
            release_supported: release_supported(CAPABILITY),
            _lease: lease,
        })
    }
}

impl EventLaw {
    /// Source-bound compiler program digest.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// Meaning of the forecast law.
    #[must_use]
    pub fn meaning(&self) -> EventLawMeaning {
        self.meaning
    }

    /// Complete retained paths in pairing order.
    #[must_use]
    pub fn paths(&self) -> &[Vec<Arrival>] {
        &self.paths
    }

    /// Forecast clock interval.
    #[must_use]
    pub fn window(&self) -> EventWindow {
        self.window
    }

    /// Observation and clock declaration bound to the forecast.
    #[must_use]
    pub fn history(&self) -> &EventHistory {
        &self.history
    }

    /// Fit diagnostics, if the forecast estimated a model.
    #[must_use]
    pub fn fit(&self) -> Option<&HawkesFit> {
        self.model_fit.as_deref()
    }

    /// Recursive refit parameters in path-block order, when requested.
    pub fn refit_parameters(&self) -> Option<&[HawkesParameters]> {
        self.refit_parameters.as_deref().map(Vec::as_slice)
    }

    /// Exact provider verification receipt digest when present.
    #[must_use]
    pub fn provider_receipt(&self) -> Option<&str> {
        self.provider_receipt.as_deref()
    }

    /// Whether the base event computation has earned release support.
    #[must_use]
    pub fn release_supported(&self) -> bool {
        self.release_supported
    }

    /// Row masses for a weighted law. Equal masses are implicit when absent.
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref().map(Vec::as_slice)
    }
}

pub(super) fn path_bytes(paths: usize, max_events: usize) -> u64 {
    paths as u64 * (max_events as u64 * 32 + 64)
}

/// The provider execution contract a consumer must verify for these joint paths.
#[must_use]
pub fn external_execution_contract(identity: ProviderObjectIdentity) -> ExecutionContract {
    ExecutionContract {
        identity,
        deterministic: true,
        reports_uncertainty: false,
        artifact_blake3: None,
    }
}

struct PreparedKernel(Arc<LoweredProgram>);

/// The support unit a route's programs run under, as the lowering annotates it.
const fn route_unit(route: EventProcessRoute) -> &'static str {
    match route {
        EventProcessRoute::Forecast => CAPABILITY,
        EventProcessRoute::Marked => MARKED_UNIT,
        EventProcessRoute::Excitation => EXCITATION_UNIT,
        EventProcessRoute::ParameterBridge => BRIDGE_UNIT,
    }
}

impl PreparedKernel {
    fn new(
        source: SemanticDigest,
        operation: &'static str,
        capability: &'static str,
    ) -> Result<Self, EventError> {
        Ok(Self(Arc::new(super::lowering::events::lower(
            EventProcessRoute::Forecast,
            source,
            0,
            operation,
            capability,
            None,
            None,
        )?)))
    }

    /// A program of `route`, read under that route's support unit.
    fn on(
        route: EventProcessRoute,
        source: SemanticDigest,
        operation: &'static str,
    ) -> Result<Self, EventError> {
        Ok(Self(Arc::new(super::lowering::events::lower(
            route,
            source,
            0,
            operation,
            route_unit(route),
            None,
            None,
        )?)))
    }

    fn composed(
        source: SemanticDigest,
        parent: SemanticDigest,
        operation: &'static str,
        contract: Option<&antecedent_design::DecisionContract>,
    ) -> Result<Self, EventError> {
        Self::composed_on(EventProcessRoute::Forecast, source, parent, operation, contract)
    }

    /// A program of `route` composed on the parent program `parent`.
    fn composed_on(
        route: EventProcessRoute,
        source: SemanticDigest,
        parent: SemanticDigest,
        operation: &'static str,
        contract: Option<&antecedent_design::DecisionContract>,
    ) -> Result<Self, EventError> {
        Ok(Self(Arc::new(super::lowering::events::lower(
            route,
            source,
            0,
            operation,
            route_unit(route),
            Some(parent),
            contract,
        )?)))
    }

    fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.0), PlanStage::Prepared)
    }

    fn identity(&self) -> SemanticDigest {
        self.0.identity()
    }
}

impl CompiledProgramHandle for PreparedKernel {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.0
    }
}

impl CompiledProgramHandle for PreparedEventForecast {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.program
    }
}

/// Enter the dispatch of `handle`'s program and pass the support gate of the unit its route
/// runs under: a release build refuses a route whose unit is not supported, a development
/// build records what a release build refuses and runs the route.
fn scoped_execution(
    ctx: &ExecutionContext,
    handle: &impl CompiledProgramHandle,
) -> Result<(ExecutionContext, ProgramPublication), EventError> {
    super::lowering::refuse_reentry(ctx)?;
    let scoped = ctx.enter_lowered_dispatch(handle.program().identity());
    poll(&scoped)?;
    let publication = gate_program(&scoped, handle).map_err(crate::error::CausalError::from)?;
    Ok((scoped, publication))
}

/// Compiled conditional likelihood of the declared history at fixed parameters.
pub struct PreparedEventLikelihood {
    history: EventHistory,
    parameters: HawkesParameters,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

impl Study {
    /// Compile the conditional likelihood before evaluating any intensity.
    ///
    /// # Errors
    /// Invalid history, unstable parameters or resource refusal.
    pub fn prepare_event_likelihood(
        history: EventHistory,
        parameters: HawkesParameters,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEventLikelihood, EventError> {
        super::lowering::refuse_reentry(ctx)?;
        history.validate(true)?;
        parameters.validate(history.catalog.len())?;
        let lease = admit(ctx, 0, history_bytes(&history))?;
        let source = digest("antecedent.events.likelihood.v1", &(&history, &parameters))?;
        let plan = PreparedKernel::new(source, "likelihood", CAPABILITY)?;
        Ok(PreparedEventLikelihood { history, parameters, plan, _lease: lease })
    }
}

impl PreparedEventLikelihood {
    /// Inspect the compiled likelihood plan.
    pub fn plan_view(&self) -> PlanView {
        self.plan.plan_view()
    }

    /// Evaluate the conditional log likelihood. Negative infinity means zero probability density.
    ///
    /// # Errors
    /// Resource, cancellation or unsupported release scope.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<f64, EventError> {
        let (scoped, publication) = scoped_execution(ctx, &self.plan)?;
        let _scratch = admit(&scoped, history_bytes(&self.history) * 12, 0)?;
        publication.finish(hawkes::log_likelihood(
            &self.history,
            &self.parameters,
            &mut hawkes::Work::new(&scoped),
        ))
    }
}

#[cfg(test)]
mod limits {
    use super::{
        EventHistory, EventWindow, MAX_EVENT_TYPES, MAX_EVENTS, OPERATION_BUDGET, max_dense_types,
    };

    fn history(types: usize) -> EventHistory {
        EventHistory {
            catalog: (0..types).map(|i| format!("type-{i}")).collect(),
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: vec![],
            gaps: vec![],
        }
    }

    #[test]
    fn dense_catalog_cap_is_half_the_likelihood_build() {
        let n = MAX_EVENT_TYPES;
        let build = (MAX_EVENTS + 1) * n * n;
        assert!(build <= OPERATION_BUDGET / 2, "{build}");
        let next = (MAX_EVENTS + 1) * (n + 1) * (n + 1);
        assert!(next > OPERATION_BUDGET / 2, "{next}");
        assert!(n > 16);
        assert_eq!(n, max_dense_types(MAX_EVENTS, OPERATION_BUDGET));
        history(n).validate(true).unwrap();
        history(17).validate(true).unwrap();
        assert!(history(n + 1).validate(true).is_err());
    }
}

#[cfg(test)]
mod site_tests {
    use super::{
        Arrival, EventForecastRequest, EventHistory, EventSource, EventWindow,
        ProviderObjectIdentity, digest,
    };
    use crate::identity_laws::{bit_exact, injective, ordered_laws};

    fn history(catalog: &[&str], events: Vec<Arrival>, start: f64) -> EventHistory {
        EventHistory {
            catalog: catalog.iter().map(|name| (*name).to_owned()).collect(),
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: start,
            window: EventWindow { start, end: 10.0 },
            left: vec![],
            events,
            gaps: vec![],
        }
    }

    /// The source digest of a history the native route accepts, `None` when it refuses it.
    fn source(history: &EventHistory) -> Option<[u8; 32]> {
        history.validate(true).ok()?;
        digest("antecedent.events.site", history).ok().map(|digest| *digest.as_bytes())
    }

    #[test]
    fn site_events_source_digest_orders_the_catalog_and_the_arrivals() {
        // An arrival names its type by catalog index, so the catalog is ordered.
        ordered_laws(&["a", "b", "c"], |catalog| {
            source(&history(catalog, vec![Arrival { time: 1.0, kind: 0 }], 0.0))
        });
        // Arrivals are read in time order, which validation enforces: another order is refused.
        let arrivals = vec![Arrival { time: 1.0, kind: 0 }, Arrival { time: 2.0, kind: 1 }];
        assert!(source(&history(&["a", "b"], arrivals.clone(), 0.0)).is_some());
        let mut reversed = arrivals;
        reversed.reverse();
        assert!(source(&history(&["a", "b"], reversed, 0.0)).is_none());
        // Times are written exactly; the encoding frames and escapes every text.
        bit_exact(|start| source(&history(&["a"], Vec::new(), start)));
        let parts = [("ab", "c"), ("a", "bc"), ("a\",\"origin\":\"b", "c")];
        injective(
            &parts,
            |pair| *pair,
            |pair| {
                let mut declared = history(&["a"], Vec::new(), 0.0);
                declared.time_unit = pair.0.to_owned();
                declared.origin = pair.1.to_owned();
                source(&declared)
            },
        );
    }

    /// Tied arrivals are admitted only for fixture and external sources, whose history no kernel
    /// here reads: an external provider receives the payload as declared and binds its request
    /// identifier to it, so the order of tied arrivals is part of the request it answered.
    #[test]
    fn site_events_external_request_binds_tied_arrivals_in_declared_order() {
        let tied = vec![Arrival { time: 1.0, kind: 0 }, Arrival { time: 1.0, kind: 1 }];
        let request = |events: Vec<Arrival>| {
            let mut declared = history(&["a", "b"], events, 0.0);
            declared.window = EventWindow { start: 0.0, end: 2.0 };
            EventForecastRequest {
                history: declared,
                forecast: EventWindow { start: 2.0, end: 3.0 },
                source: EventSource::External {
                    identity: ProviderObjectIdentity {
                        provider_id: "provider".into(),
                        object_id: "paths".into(),
                        version_id: "v".into(),
                        snapshot_id: "snapshot".into(),
                        request_id: "pending".into(),
                    },
                    paths: vec![vec![]],
                },
                paths: 1,
                max_events: 8,
                seed: 1,
            }
        };
        // The native routes refuse the tie, so no likelihood or sampler reads its order.
        assert!(request(tied.clone()).history.validate(true).is_err());
        ordered_laws(&tied, |events| request(events.to_vec()).provider_request_id().ok());
    }
}
