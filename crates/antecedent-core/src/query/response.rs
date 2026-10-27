//! Continuous causal-response queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::{InterventionFault, ResponseFault};
use std::sync::Arc;

use crate::canonical_encode::{
    CanonicalEncode, count, encode_list, encode_option, encode_set, encode_u32s,
    encode_variable_set, encode_variables,
};
use crate::identity::CanonicalWriter;
use crate::intervention::TemporalPolicy;
use crate::{InitialLaw, Intervention, TargetPopulation, Value, VariableId};

use super::QueryError;
use super::functional::OutcomeFunctional;

/// Default dose-by-horizon cell count an executor materializes when the caller
/// supplies no [`crate::ExecutionContext::memory_limit_bytes`].
///
/// The query contract does not enforce this value: a temporal response query
/// is valid for any grid and horizon list whose cell count fits in `usize`,
/// and the resource bound is the execution memory budget. Executors that
/// materialize the five cell-scaled `f64` values (mean, lower, upper, dose,
/// horizon) use this as their fallback ceiling only when no budget is declared.
pub const MAX_TEMPORAL_RESPONSE_CELLS: usize = 1_000_000;

/// Licensed temporal-response query policy, for language facades.
///
/// [`TemporalResponseSpec::validate`] is the semantic authority. Bindings may
/// duplicate checks for early errors but must read these values rather than
/// defining them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalResponseLicense {
    /// Largest horizon count the query contract accepts. The contract has no
    /// fixed count: the execution memory budget bounds materialization, so this
    /// is `usize::MAX`.
    pub max_horizons: usize,
    /// Largest dose-by-horizon cell count the query contract accepts. The
    /// contract has no fixed count (see `max_horizons`), so this is `usize::MAX`.
    pub max_cells: usize,
    /// Wire tags this spec accepts (`pulse`, `sustained`).
    pub allowed_policies: &'static [&'static str],
    /// Default wire tag when the caller omits `policy`.
    pub default_policy: &'static str,
    /// Default `treatment_lag` (policy origin `-lag`) for Pulse / `ResponseCurve`.
    pub default_treatment_lag: u32,
}

/// Declared delayed-feedback mechanism of a temporal response.
///
/// The dynamics `s <- a s + b u + e` are declared, not fitted: the coefficients,
/// the innovation variance, and the state at the origin are the caller's
/// statement about the system. `delay_ticks` is the feedback delay, the ticks
/// before the action re-enters the state. It is not the action lag, which is the
/// negative start of the policy. `level` is the dose the pulse or sustained
/// policy holds. [`TemporalResponseSpec::validate`] checks the numbers are
/// finite, the delay is at least one tick, and the variance is not negative.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackDeclaration {
    /// Ticks before the action re-enters the state (at least 1).
    pub delay_ticks: u32,
    /// Dose the policy holds.
    pub level: f64,
    /// Autoregressive coefficient `a`.
    pub autoregression: f64,
    /// Coefficient `b` on the delayed action.
    pub input_gain: f64,
    /// Variance of the transition's innovation (not negative).
    pub innovation_variance: f64,
    /// Law whose level at the origin is `initial_state`.
    pub initial_law: InitialLaw,
    /// Level of `initial_law` at the origin.
    pub initial_state: f64,
}

/// Every field is written; reals as their exact bits.
impl CanonicalEncode for FeedbackDeclaration {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer
            .word(u64::from(self.delay_ticks))
            .real(self.level)
            .real(self.autoregression)
            .real(self.input_gain)
            .real(self.innovation_variance);
        self.initial_law.encode(writer);
        writer.real(self.initial_state);
    }
}

impl FeedbackDeclaration {
    /// Validate delay, finiteness, and variance.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidResponse`] for a zero delay, a non-finite number, or
    /// a negative innovation variance.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.delay_ticks == 0 {
            return Err(QueryError::InvalidResponse(ResponseFault::FeedbackDelay));
        }
        let numbers = [
            self.level,
            self.autoregression,
            self.input_gain,
            self.innovation_variance,
            self.initial_state,
        ];
        if numbers.iter().any(|value| !value.is_finite()) {
            return Err(QueryError::InvalidResponse(ResponseFault::FeedbackNotFinite));
        }
        if self.innovation_variance < 0.0 {
            return Err(QueryError::InvalidResponse(ResponseFault::FeedbackNegativeVariance));
        }
        Ok(())
    }

    /// Hard assignment of `variable` to this declaration's dose.
    ///
    /// The dose is [`Intervention::Set`]. It is not a shift. Timing stays on the
    /// response's [`TemporalPolicy`].
    #[must_use]
    pub fn dose(&self, variable: VariableId) -> Intervention {
        Intervention::set(variable, Value::f64(self.level))
    }
}

/// Temporal attachment for a continuous-response query (ADR 0021).
///
/// When present, the query is a temporal cell: dose × horizon surfaces for
/// [`ResponseFunctional::MeanCurve`], or horizon-indexed intervention responses.
/// Absence means a static response cell.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalResponseSpec {
    /// Outcome evaluation horizons in time steps after the policy origin (each ≥ 1).
    /// Strictly increasing; at least one entry.
    pub horizons: Arc<[u32]>,
    /// Temporal intervention policy. Licensed 0.7 cells are Pulse and
    /// single-step Sustained; Dynamic is refused here (use `TemporalEffectQuery`).
    pub policy: TemporalPolicy,
    /// Optional max history lag (steps) when unfolding; `None` = planner default.
    pub max_history_lag: Option<u32>,
    /// Declared delayed-feedback mechanism. `None` publishes the fitted
    /// dose-by-horizon surface.
    pub feedback: Option<FeedbackDeclaration>,
}

/// Every field is written. `horizons` is the ordered grid the results are
/// indexed by.
impl CanonicalEncode for TemporalResponseSpec {
    fn encode(&self, writer: &mut CanonicalWriter) {
        encode_u32s(writer, &self.horizons);
        self.policy.encode(writer);
        writer.opt_word(self.max_history_lag.map(u64::from));
        encode_option(writer, self.feedback.as_ref());
    }
}

impl TemporalResponseSpec {
    /// Wire tag for [`crate::intervention::TemporalPolicy::Pulse`].
    pub const POLICY_PULSE: &'static str = "pulse";
    /// Wire tag for [`crate::intervention::TemporalPolicy::Sustained`].
    pub const POLICY_SUSTAINED: &'static str = "sustained";
    /// Policies [`Self::validate`] accepts on a response cell.
    pub const ALLOWED_POLICIES: &'static [&'static str] =
        &[Self::POLICY_PULSE, Self::POLICY_SUSTAINED];
    /// Default wire tag when a facade omits `policy`.
    pub const DEFAULT_POLICY: &'static str = Self::POLICY_PULSE;
    /// Default `treatment_lag` (policy origin `-lag`).
    pub const DEFAULT_TREATMENT_LAG: u32 = 1;

    /// Machine-readable license for facades. Python must not hard-code these.
    #[must_use]
    pub const fn license() -> TemporalResponseLicense {
        TemporalResponseLicense {
            max_horizons: usize::MAX,
            max_cells: usize::MAX,
            allowed_policies: Self::ALLOWED_POLICIES,
            default_policy: Self::DEFAULT_POLICY,
            default_treatment_lag: Self::DEFAULT_TREATMENT_LAG,
        }
    }

    /// Parse a licensed response-policy wire tag at treatment offset `at`.
    ///
    /// Sustained is the licensed single-step window `[at, at]`.
    ///
    /// # Errors
    ///
    /// Unknown tag (including `dynamic`).
    pub fn parse_policy(tag: &str, at: i32) -> Result<TemporalPolicy, QueryError> {
        match tag {
            Self::POLICY_PULSE => Ok(TemporalPolicy::pulse(at)),
            Self::POLICY_SUSTAINED => Ok(TemporalPolicy::sustained(at, at)),
            other => Err(QueryError::InvalidResponse(ResponseFault::UnsupportedPolicy {
                got: other.into(),
            })),
        }
    }

    /// Construct a temporal attachment after validating horizons and policy.
    ///
    /// # Errors
    ///
    /// Empty/non-increasing horizons, zero horizon, or invalid policy.
    pub fn new(
        horizons: impl Into<Arc<[u32]>>,
        policy: TemporalPolicy,
        max_history_lag: Option<u32>,
    ) -> Result<Self, QueryError> {
        let horizons = horizons.into();
        let spec = Self { horizons, policy, max_history_lag, feedback: None };
        spec.validate()?;
        Ok(spec)
    }

    /// Declare the delayed-feedback mechanism the response is asked under.
    ///
    /// # Errors
    ///
    /// The declaration's validation errors.
    pub fn with_feedback(mut self, feedback: FeedbackDeclaration) -> Result<Self, QueryError> {
        self.feedback = Some(feedback);
        self.validate()?;
        Ok(self)
    }

    /// Validate horizons and nested policy.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidResponse`] or temporal-policy errors.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.horizons.is_empty() {
            return Err(QueryError::InvalidResponse(ResponseFault::HorizonsEmpty));
        }
        if self.horizons.iter().any(|h| *h == 0) {
            return Err(QueryError::NonPositiveHorizon);
        }
        if self.horizons.windows(2).any(|w| w[0] >= w[1]) {
            return Err(QueryError::InvalidResponse(ResponseFault::HorizonsOrder));
        }
        self.policy.validate().map_err(|e| match e {
            crate::intervention::InterventionError::InvalidTemporalWindow { from, until } => {
                QueryError::InvalidTemporalWindow { from, until }
            }
            other => QueryError::InvalidIntervention(InterventionFault::Rejected(other)),
        })?;
        match &self.policy {
            TemporalPolicy::Pulse { .. } | TemporalPolicy::Sustained { .. } => {}
            TemporalPolicy::Dynamic { .. } => {
                return Err(QueryError::InvalidResponse(ResponseFault::DynamicPolicyNotACell));
            }
        }
        if let Some(feedback) = &self.feedback {
            feedback.validate()?;
        }
        Ok(())
    }

    /// Largest requested horizon (guaranteed ≥ 1 after validation).
    #[must_use]
    pub fn max_horizon(&self) -> u32 {
        self.horizons.last().copied().unwrap_or(1)
    }

    /// Treatment-time origin under the attached policy.
    ///
    /// # Errors
    ///
    /// Empty dynamic schedule.
    pub fn treatment_offset(&self) -> Result<i32, QueryError> {
        match &self.policy {
            TemporalPolicy::Pulse { at } => Ok(*at),
            TemporalPolicy::Sustained { from, .. } => Ok(*from),
            TemporalPolicy::Dynamic { active_at, .. } => {
                active_at.first().copied().ok_or(QueryError::DynamicPolicyHasNoTreatmentOffset)
            }
        }
    }
}

/// Treatment dimension up to which an estimator may tabulate an explicit
/// evaluation grid.
///
/// The query contract does not enforce this value: a response query may name
/// any number of treatments, and a gradient of `k` treatments is `k` partial
/// derivatives of one fit. Estimators that materialize a full tensor grid use
/// it as their own guard.
pub const MAX_NONPARAMETRIC_RESPONSE_DIM: usize = 2;

/// Points at which a continuous response is evaluated.
#[derive(Clone, Debug, PartialEq)]
pub enum GridSpec {
    /// Explicit, strictly increasing finite values.
    Values(Arc<[f64]>),
    /// Inclusive evenly spaced grid.
    Linspace {
        /// First point.
        start: f64,
        /// Last point.
        end: f64,
        /// Number of points, at least two.
        points: usize,
    },
}

/// The grid as spelled: explicit values in order, or the linspace triple. A
/// linspace and the explicit values it would materialize encode differently.
impl CanonicalEncode for GridSpec {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Values(values) => {
                writer.word(1).reals(values);
            }
            Self::Linspace { start, end, points } => {
                writer.word(2).real(*start).real(*end).word(count(*points));
            }
        }
    }
}

impl GridSpec {
    /// Number of grid points (not validated).
    #[must_use]
    pub fn point_count(&self) -> usize {
        match self {
            Self::Values(values) => values.len(),
            Self::Linspace { points, .. } => *points,
        }
    }

    /// The `index`-th grid point of a validated grid, computed without
    /// materializing the grid; `None` past the end. Streaming and tiled
    /// consumers use this instead of [`Self::values`].
    #[must_use]
    pub fn value_at(&self, index: usize) -> Option<f64> {
        match self {
            Self::Values(values) => values.get(index).copied(),
            Self::Linspace { start, end, points } => {
                if index >= *points || *points < 2 {
                    return None;
                }
                let step = (end - start) / (*points - 1) as f64;
                Some(start + index as f64 * step)
            }
        }
    }

    /// Materialize the grid after validation.
    ///
    /// The allocation is requested fallibly, so a grid the allocator cannot
    /// satisfy is a typed refusal rather than an abort; callers with a declared
    /// memory budget should check [`Self::point_count`] against it first or
    /// stream through [`Self::value_at`].
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidResponse`] when the grid is invalid or its
    /// allocation fails.
    pub fn values(&self) -> Result<Vec<f64>, QueryError> {
        self.validate()?;
        match self {
            Self::Values(values) => Ok(values.to_vec()),
            Self::Linspace { start, end, points } => {
                let mut out = Vec::new();
                out.try_reserve_exact(*points).map_err(|_| {
                    QueryError::InvalidResponse(ResponseFault::LinspaceExceedsMemory)
                })?;
                let step = (end - start) / (*points - 1) as f64;
                out.extend((0..*points).map(|i| start + i as f64 * step));
                Ok(out)
            }
        }
    }

    /// Validate finiteness, size, and ordering.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidResponse`] when values are non-finite, unordered, or undersized.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::Values(values) => {
                if values.len() < 2 {
                    return Err(QueryError::InvalidResponse(ResponseFault::GridTooShort));
                }
                if values.iter().any(|v| !v.is_finite()) || values.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(QueryError::InvalidResponse(ResponseFault::GridValuesOrder));
                }
            }
            Self::Linspace { start, end, points } => {
                if !start.is_finite() || !end.is_finite() || start >= end || *points < 2 {
                    return Err(QueryError::InvalidResponse(ResponseFault::LinspaceBounds));
                }
            }
        }
        Ok(())
    }
}

/// Domain of a scalar continuous intervention.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousDomain {
    /// Intervened variable.
    pub variable: VariableId,
    /// Evaluation grid.
    pub grid: GridSpec,
}

impl CanonicalEncode for ContinuousDomain {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.variable.encode(writer);
        self.grid.encode(writer);
    }
}

impl ContinuousDomain {
    /// Construct a continuous intervention domain.
    #[must_use]
    pub fn new(variable: VariableId, grid: GridSpec) -> Self {
        Self { variable, grid }
    }
}

/// Weighting law for an average derivative effect.
#[derive(Clone, Debug, PartialEq)]
pub enum DerivativeWeighting {
    /// Average over the observed treatment/covariate law.
    Observed,
    /// Uniform weighting over a supplied finite interval.
    Uniform {
        /// Inclusive lower endpoint.
        lower: f64,
        /// Inclusive upper endpoint.
        upper: f64,
    },
    /// Caller-supplied row weights, normalized by the estimator.
    Custom(Arc<[f64]>),
}

/// Scale on which a derivative is reported.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DerivativeScale {
    /// `dm/da`.
    Identity,
    /// `a * dm/da`.
    LogTreatment,
    /// `(1/m) * dm/da`.
    LogOutcome,
    /// `(a/m) * dm/da` (elasticity).
    LogLog,
}

/// How a scientific outcome entered the observed dataset.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ObservationSpec {
    /// Outcome is completely observed.
    Complete,
    /// Right-censored continuous outcome.
    RightCensored {
        /// Scientific latent outcome.
        latent: VariableId,
        /// Recorded minimum of the latent outcome and censoring value.
        observed: VariableId,
        /// Censoring value.
        censoring: VariableId,
        /// Event/uncensored indicator.
        event: VariableId,
    },
    /// Left-censored continuous outcome.
    LeftCensored {
        /// Scientific latent outcome.
        latent: VariableId,
        /// Recorded maximum of the latent outcome and censoring value.
        observed: VariableId,
        /// Censoring value.
        censoring: VariableId,
        /// Event/uncensored indicator.
        event: VariableId,
    },
    /// Interval-censored continuous outcome.
    IntervalCensored {
        /// Scientific latent outcome.
        latent: VariableId,
        /// Observed lower endpoint.
        lower: VariableId,
        /// Observed upper endpoint.
        upper: VariableId,
    },
    /// Sampling truncation with optional row-specific bounds.
    Truncated {
        /// Scientific latent outcome.
        latent: VariableId,
        /// Recorded outcome among sampled units.
        observed: VariableId,
        /// Optional lower truncation bound.
        lower: Option<VariableId>,
        /// Optional upper truncation bound.
        upper: Option<VariableId>,
    },
    /// Outcome observed only when an indicator is one.
    Selected {
        /// Scientific latent outcome.
        latent: VariableId,
        /// Recorded outcome (valid only on selected rows).
        observed: VariableId,
        /// Observation/selection indicator.
        indicator: VariableId,
    },
}

/// Explicit claim about an observation mechanism.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ObservationAssumption {
    /// Observation/censoring is independent after conditioning on these variables.
    IndependentGiven(Arc<[VariableId]>),
    /// Observation is independent of the latent outcome after conditioning.
    OutcomeIndependentGiven(Arc<[VariableId]>),
    /// Named structural observation model.
    Structural(Arc<str>),
}

/// A response functional, distinct from the estimator used to learn it.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseFunctional {
    /// `a -> E[Y | do(A=a)]`.
    MeanCurve {
        /// Outcome.
        outcome: VariableId,
        /// Scalar continuous treatment domain.
        treatment: ContinuousDomain,
    },
    /// Scalar weighted average derivative effect.
    AverageDerivative {
        /// Outcome.
        outcome: VariableId,
        /// Treatment.
        treatment: VariableId,
        /// Target weighting law.
        weighting: DerivativeWeighting,
    },
    /// Local derivative of a response representation.
    PointDerivative {
        /// Outcome.
        outcome: VariableId,
        /// Treatment.
        treatment: VariableId,
        /// Evaluation point.
        at: f64,
        /// Derivative order (one or two).
        order: u8,
        /// Reporting scale.
        scale: DerivativeScale,
    },
    /// Directional derivative for a vector intervention.
    DirectionalDerivative {
        /// Outcomes.
        outcomes: Arc<[VariableId]>,
        /// Treatments.
        treatments: Arc<[VariableId]>,
        /// Evaluation point in treatment order.
        at: Arc<[f64]>,
        /// Direction in treatment order.
        direction: Arc<[f64]>,
    },
    /// Low-dimensional response Jacobian.
    Jacobian {
        /// Outcomes.
        outcomes: Arc<[VariableId]>,
        /// Treatments.
        treatments: Arc<[VariableId]>,
        /// Evaluation point in treatment order.
        at: Arc<[f64]>,
        /// Reporting scale.
        scale: DerivativeScale,
    },
    /// Mean response under an existing intervention or joint intervention.
    InterventionResponse {
        /// Outcome.
        outcome: VariableId,
        /// Intervention set.
        interventions: Arc<[Intervention]>,
    },
}

/// Custom weights are row- or grid-aligned and keep their order.
impl CanonicalEncode for DerivativeWeighting {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Observed => {
                writer.word(1);
            }
            Self::Uniform { lower, upper } => {
                writer.word(2).real(*lower).real(*upper);
            }
            Self::Custom(weights) => {
                writer.word(3).reals(weights);
            }
        }
    }
}

impl CanonicalEncode for DerivativeScale {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let tag = match self {
            Self::Identity => 1,
            Self::LogTreatment => 2,
            Self::LogOutcome => 3,
            Self::LogLog => 4,
        };
        writer.word(tag);
    }
}

/// Every variable keeps its role; an absent truncation bound is a presence word.
impl CanonicalEncode for ObservationSpec {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Complete => {
                writer.word(1);
            }
            Self::RightCensored { latent, observed, censoring, event } => {
                writer.word(2);
                latent.encode(writer);
                observed.encode(writer);
                censoring.encode(writer);
                event.encode(writer);
            }
            Self::LeftCensored { latent, observed, censoring, event } => {
                writer.word(3);
                latent.encode(writer);
                observed.encode(writer);
                censoring.encode(writer);
                event.encode(writer);
            }
            Self::IntervalCensored { latent, lower, upper } => {
                writer.word(4);
                latent.encode(writer);
                lower.encode(writer);
                upper.encode(writer);
            }
            Self::Truncated { latent, observed, lower, upper } => {
                writer.word(5);
                latent.encode(writer);
                observed.encode(writer);
                encode_option(writer, lower.as_ref());
                encode_option(writer, upper.as_ref());
            }
            Self::Selected { latent, observed, indicator } => {
                writer.word(6);
                latent.encode(writer);
                observed.encode(writer);
                indicator.encode(writer);
            }
        }
    }
}

/// Conditioning variables are a set and are encoded in ascending id order.
impl CanonicalEncode for ObservationAssumption {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::IndependentGiven(variables) => {
                writer.word(1);
                encode_variable_set(writer, variables);
            }
            Self::OutcomeIndependentGiven(variables) => {
                writer.word(2);
                encode_variable_set(writer, variables);
            }
            Self::Structural(claim) => {
                writer.word(3).text(claim);
            }
        }
    }
}

/// Multi-coordinate functionals keep every list in order: evaluation points
/// and directions are aligned with `treatments`, and `outcomes` index the
/// result. `interventions` keep their declared order.
impl CanonicalEncode for ResponseFunctional {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::MeanCurve { outcome, treatment } => {
                writer.word(1);
                outcome.encode(writer);
                treatment.encode(writer);
            }
            Self::AverageDerivative { outcome, treatment, weighting } => {
                writer.word(2);
                outcome.encode(writer);
                treatment.encode(writer);
                weighting.encode(writer);
            }
            Self::PointDerivative { outcome, treatment, at, order, scale } => {
                writer.word(3);
                outcome.encode(writer);
                treatment.encode(writer);
                writer.real(*at).word(u64::from(*order));
                scale.encode(writer);
            }
            Self::DirectionalDerivative { outcomes, treatments, at, direction } => {
                writer.word(4);
                encode_variables(writer, outcomes);
                encode_variables(writer, treatments);
                writer.reals(at).reals(direction);
            }
            Self::Jacobian { outcomes, treatments, at, scale } => {
                writer.word(5);
                encode_variables(writer, outcomes);
                encode_variables(writer, treatments);
                writer.reals(at);
                scale.encode(writer);
            }
            Self::InterventionResponse { outcome, interventions } => {
                writer.word(6);
                outcome.encode(writer);
                encode_list(writer, interventions);
            }
        }
    }
}

impl ResponseFunctional {
    /// Treatment variable ids in query order.
    #[must_use]
    pub fn treatment_ids(&self) -> Vec<VariableId> {
        match self {
            Self::MeanCurve { treatment, .. } => vec![treatment.variable],
            Self::AverageDerivative { treatment, .. } | Self::PointDerivative { treatment, .. } => {
                vec![*treatment]
            }
            Self::DirectionalDerivative { treatments, .. } | Self::Jacobian { treatments, .. } => {
                treatments.to_vec()
            }
            Self::InterventionResponse { interventions, .. } => {
                let mut ids = Vec::new();
                for intervention in interventions.iter() {
                    for variable in intervention.target_variables() {
                        if !ids.contains(&variable) {
                            ids.push(variable);
                        }
                    }
                }
                ids
            }
        }
    }

    /// Outcome variable ids in query order.
    #[must_use]
    pub fn outcome_ids(&self) -> Vec<VariableId> {
        match self {
            Self::MeanCurve { outcome, .. }
            | Self::AverageDerivative { outcome, .. }
            | Self::PointDerivative { outcome, .. }
            | Self::InterventionResponse { outcome, .. } => vec![*outcome],
            Self::DirectionalDerivative { outcomes, .. } | Self::Jacobian { outcomes, .. } => {
                outcomes.to_vec()
            }
        }
    }

    /// First treatment/outcome pair, when the functional names at least one of each.
    #[must_use]
    pub fn primary_pair(&self) -> Option<(VariableId, VariableId)> {
        let treatment = self.treatment_ids().into_iter().next()?;
        let outcome = self.outcome_ids().into_iter().next()?;
        Some((treatment, outcome))
    }
}

/// Complete continuous-response query.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseQuery {
    /// Requested functional.
    pub functional: ResponseFunctional,
    /// Target population.
    pub target_population: TargetPopulation,
    /// Observation mechanism (complete by default).
    pub observation: ObservationSpec,
    /// Caller-declared observation assumptions. Empty means none.
    pub observation_assumptions: Arc<[ObservationAssumption]>,
    /// Optional temporal attachment (ADR 0021). `None` = static response cell.
    pub temporal: Option<TemporalResponseSpec>,
    /// Outcome functional (mean or exceedance). Identification is unchanged.
    pub outcome_functional: OutcomeFunctional,
}

/// Every field is written. `observation_assumptions` are a conjunction of
/// claims (each one is appended to the assumption set) and are encoded as an
/// unordered collection.
impl CanonicalEncode for ResponseQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.functional.encode(writer);
        self.target_population.encode(writer);
        self.observation.encode(writer);
        encode_set(writer, &self.observation_assumptions);
        encode_option(writer, self.temporal.as_ref());
        self.outcome_functional.encode(writer);
    }
}

impl ResponseQuery {
    /// Construct a completely observed response query.
    #[must_use]
    pub fn new(functional: ResponseFunctional) -> Self {
        Self {
            functional,
            target_population: TargetPopulation::AllObserved,
            observation: ObservationSpec::Complete,
            observation_assumptions: Arc::from([]),
            temporal: None,
            outcome_functional: OutcomeFunctional::Mean,
        }
    }

    /// Set the outcome functional. Identification uses the same adjustment set.
    #[must_use]
    pub fn with_outcome_functional(mut self, functional: OutcomeFunctional) -> Self {
        self.outcome_functional = functional;
        self
    }

    /// Attach an explicit observation process and its assumptions.
    #[must_use]
    pub fn with_observation(
        mut self,
        observation: ObservationSpec,
        assumptions: impl Into<Arc<[ObservationAssumption]>>,
    ) -> Self {
        self.observation = observation;
        self.observation_assumptions = assumptions.into();
        self
    }

    /// Attach a temporal dose-over-horizon / policy-path specification.
    #[must_use]
    pub fn with_temporal(mut self, temporal: TemporalResponseSpec) -> Self {
        self.temporal = Some(temporal);
        self
    }

    /// Whether this query is a temporal response cell.
    #[must_use]
    pub const fn is_temporal(&self) -> bool {
        self.temporal.is_some()
    }

    /// Set the target population.
    #[must_use]
    pub fn with_target_population(mut self, target: TargetPopulation) -> Self {
        self.target_population = target;
        self
    }

    /// Validate dimensions, finite values, scales, and intervention targets.
    ///
    /// # Errors
    ///
    /// [`QueryError`] when variables, dimensions, values, or observation semantics are invalid.
    pub fn validate(&self) -> Result<(), QueryError> {
        self.outcome_functional.validate()?;
        self.validate_temporal_attachment()?;
        match &self.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                if *outcome == treatment.variable {
                    return Err(QueryError::TreatmentEqualsOutcome { id: *outcome });
                }
                treatment.grid.validate()?;
            }
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                if outcome == treatment {
                    return Err(QueryError::TreatmentEqualsOutcome { id: *outcome });
                }
                match weighting {
                    DerivativeWeighting::Uniform { lower, upper }
                        if !lower.is_finite() || !upper.is_finite() || lower >= upper =>
                    {
                        return Err(QueryError::InvalidResponse(
                            ResponseFault::UniformWeightBounds,
                        ));
                    }
                    DerivativeWeighting::Custom(weights)
                        if weights.is_empty()
                            || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
                            || weights.iter().all(|w| *w == 0.0) =>
                    {
                        return Err(QueryError::InvalidResponse(ResponseFault::CustomWeights));
                    }
                    _ => {}
                }
            }
            ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => {
                if outcome == treatment {
                    return Err(QueryError::TreatmentEqualsOutcome { id: *outcome });
                }
                if !at.is_finite() || !matches!(order, 1 | 2) {
                    return Err(QueryError::InvalidResponse(ResponseFault::PointDerivative));
                }
                if matches!(scale, DerivativeScale::LogTreatment | DerivativeScale::LogLog)
                    && *at <= 0.0
                {
                    return Err(QueryError::InvalidResponse(ResponseFault::LogScalePoint));
                }
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                if !response_sets_are_distinct(outcomes, treatments)
                    || at.len() != treatments.len()
                    || direction.len() != treatments.len()
                    || at.iter().chain(direction.iter()).any(|v| !v.is_finite())
                    || direction.iter().all(|v| *v == 0.0)
                {
                    return Err(QueryError::InvalidResponse(ResponseFault::DirectionalDimensions));
                }
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                if !response_sets_are_distinct(outcomes, treatments)
                    || at.len() != treatments.len()
                    || at.iter().any(|v| !v.is_finite())
                {
                    return Err(QueryError::InvalidResponse(ResponseFault::JacobianDimensions));
                }
                if matches!(scale, DerivativeScale::LogTreatment | DerivativeScale::LogLog)
                    && at.iter().any(|v| *v <= 0.0)
                {
                    return Err(QueryError::InvalidResponse(ResponseFault::LogScaleJacobian));
                }
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                if interventions.is_empty() {
                    return Err(QueryError::InvalidResponse(ResponseFault::NoIntervention));
                }
                for intervention in interventions.iter() {
                    intervention.validate().map_err(|e| {
                        QueryError::InvalidIntervention(InterventionFault::Rejected(e))
                    })?;
                    if intervention.target_variables().contains(outcome) {
                        return Err(QueryError::TreatmentEqualsOutcome { id: *outcome });
                    }
                }
            }
        }
        self.target_population.validate()?;
        self.outcome_functional.validate()?;
        Ok(())
    }

    fn validate_temporal_attachment(&self) -> Result<(), QueryError> {
        if let Some(temporal) = &self.temporal {
            temporal.validate()?;
            match &self.functional {
                ResponseFunctional::MeanCurve { treatment, .. } => {
                    // The contract bounds the cell count only by what `usize`
                    // can address; the execution memory budget bounds
                    // materialization.
                    treatment.grid.point_count().checked_mul(temporal.horizons.len()).ok_or(
                        QueryError::InvalidResponse(ResponseFault::DoseHorizonCellOverflow),
                    )?;
                }
                ResponseFunctional::InterventionResponse { .. }
                | ResponseFunctional::AverageDerivative { .. }
                | ResponseFunctional::PointDerivative { .. }
                | ResponseFunctional::DirectionalDerivative { .. }
                | ResponseFunctional::Jacobian { .. } => {}
            }
        }
        Ok(())
    }

    /// Licensed static pairs that may ride a temporal curve. Anything else
    /// refuses with [`TEMPORAL_OBSERVATION_UNLICENSED`].
    ///
    /// # Errors
    ///
    /// Unlicensed mechanism/assumption pair on a temporal attachment.
    pub fn require_licensed_temporal_observation(&self) -> Result<(), QueryError> {
        if self.temporal.is_none() {
            return Ok(());
        }
        match &self.observation {
            ObservationSpec::Complete => {
                if self.observation_assumptions.is_empty() {
                    Ok(())
                } else {
                    Err(QueryError::InvalidResponse(ResponseFault::TemporalObservationUnlicensed))
                }
            }
            ObservationSpec::Selected { .. } => match single_observation_assumption(self) {
                Some(ObservationAssumption::OutcomeIndependentGiven(_)) => Ok(()),
                _ => Err(QueryError::InvalidResponse(ResponseFault::TemporalObservationUnlicensed)),
            },
            ObservationSpec::RightCensored { .. } | ObservationSpec::LeftCensored { .. } => {
                match single_observation_assumption(self) {
                    Some(ObservationAssumption::IndependentGiven(_)) => Ok(()),
                    Some(ObservationAssumption::OutcomeIndependentGiven(vars))
                        if vars.is_empty() =>
                    {
                        Ok(())
                    }
                    _ => Err(QueryError::InvalidResponse(
                        ResponseFault::TemporalObservationUnlicensed,
                    )),
                }
            }
            ObservationSpec::IntervalCensored { .. } | ObservationSpec::Truncated { .. } => {
                Err(QueryError::InvalidResponse(ResponseFault::TemporalObservationUnlicensed))
            }
        }
    }
}

/// Stable compile/execute refusal for an unlicensed temporal observation pair.
pub const TEMPORAL_OBSERVATION_UNLICENSED: &str =
    "temporal response observation pair is not licensed";

fn single_observation_assumption(query: &ResponseQuery) -> Option<&ObservationAssumption> {
    if query.observation_assumptions.len() != 1 {
        return None;
    }
    query.observation_assumptions.first()
}

fn response_sets_are_distinct(outcomes: &[VariableId], treatments: &[VariableId]) -> bool {
    !outcomes.is_empty()
        && !treatments.is_empty()
        && !outcomes.iter().any(|outcome| treatments.contains(outcome))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_response_license_is_the_facade_contract() {
        let license = TemporalResponseSpec::license();
        assert_eq!(license.max_horizons, usize::MAX);
        assert_eq!(license.max_cells, usize::MAX);
        assert_eq!(license.allowed_policies, TemporalResponseSpec::ALLOWED_POLICIES);
        assert_eq!(license.default_policy, TemporalResponseSpec::POLICY_PULSE);
        assert!(license.allowed_policies.contains(&license.default_policy));
        assert_eq!(license.default_treatment_lag, TemporalResponseSpec::DEFAULT_TREATMENT_LAG);
        let at = -i32::try_from(license.default_treatment_lag).unwrap();
        assert!(TemporalResponseSpec::parse_policy(license.default_policy, at).is_ok());
        assert!(TemporalResponseSpec::parse_policy("dynamic", at).is_err());
        let many: Vec<u32> = (1..=100_000).collect();
        assert!(TemporalResponseSpec::new(many, TemporalPolicy::pulse(at), None).is_ok());
    }

    #[test]
    fn temporal_response_spec_refuses_dynamic_policy() {
        let err = TemporalResponseSpec::new(
            vec![1u32],
            TemporalPolicy::dynamic(crate::DynamicRuleId::from_raw(0), [0]),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, QueryError::InvalidResponse(_)));
        assert!(err.to_string().contains("pulse or sustained"));
    }

    #[test]
    fn temporal_surface_has_no_fixed_cell_count_but_refuses_overflow() {
        let surface = |points: usize, horizons: usize| {
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Linspace { start: 0.0, end: 1.0, points },
                ),
            })
            .with_temporal(
                TemporalResponseSpec::new(
                    (1..=u32::try_from(horizons).unwrap()).collect::<Vec<_>>(),
                    TemporalPolicy::pulse(-1),
                    None,
                )
                .unwrap(),
            )
        };
        assert!(surface(2_000, 1_024).validate().is_ok());
        let error = surface(usize::MAX, 2).validate().unwrap_err();
        assert!(error.to_string().contains("cell count overflow"));
    }

    #[test]
    fn linspace_within_cap_validates_and_materializes() {
        let grid = GridSpec::Linspace { start: 0.0, end: 1.0, points: 5 };
        assert!(grid.validate().is_ok());
        assert_eq!(grid.values().unwrap().len(), 5);
    }

    #[test]
    fn linspace_streams_and_materializes_the_same_points() {
        let grid = GridSpec::Linspace { start: -1.0, end: 2.0, points: 7 };
        let values = grid.values().unwrap();
        assert_eq!(values.len(), grid.point_count());
        for (i, value) in values.iter().enumerate() {
            assert_eq!(grid.value_at(i), Some(*value));
        }
        assert_eq!(grid.value_at(7), None);
    }

    #[test]
    fn linspace_beyond_addressable_memory_is_a_typed_refusal() {
        let grid = GridSpec::Linspace { start: 0.0, end: 1.0, points: usize::MAX };
        assert!(grid.validate().is_ok());
        assert!(matches!(grid.values(), Err(QueryError::InvalidResponse(_))));
    }

    #[test]
    fn response_functional_primary_pair_matches_treatment_and_outcome_ids() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let functional = ResponseFunctional::AverageDerivative {
            outcome,
            treatment,
            weighting: DerivativeWeighting::Observed,
        };
        assert_eq!(functional.treatment_ids(), vec![treatment]);
        assert_eq!(functional.outcome_ids(), vec![outcome]);
        assert_eq!(functional.primary_pair(), Some((treatment, outcome)));
    }

    #[test]
    fn linspace_point_count_far_beyond_u32_capacity_is_a_budget_matter_not_a_validation_cap() {
        // A grid is described by its endpoints and a count and is never materialised by
        // validation, so a very large count is well-formed; materialising it is what the
        // memory budget governs.
        let grid = GridSpec::Linspace { start: 0.0, end: 1.0, points: 4_000_000_000 };
        assert!(grid.validate().is_ok());
        assert_eq!(grid.point_count(), 4_000_000_000);
    }

    fn temporal_curve() -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap(),
        )
    }

    #[test]
    fn temporal_observation_license_matches_static_1_3_pairs() {
        let query = temporal_curve();
        assert!(query.validate().is_ok());
        assert!(query.require_licensed_temporal_observation().is_ok());

        let selected = query.clone().with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)]))],
        );
        assert!(selected.validate().is_ok());
        assert!(selected.require_licensed_temporal_observation().is_ok());

        let right = query.clone().with_observation(
            ObservationSpec::RightCensored {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                censoring: VariableId::from_raw(2),
                event: VariableId::from_raw(3),
            },
            [ObservationAssumption::IndependentGiven(Arc::from([]))],
        );
        assert!(right.require_licensed_temporal_observation().is_ok());

        let cox = query.clone().with_observation(
            ObservationSpec::LeftCensored {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                censoring: VariableId::from_raw(2),
                event: VariableId::from_raw(3),
            },
            [ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(0)]))],
        );
        assert!(cox.require_licensed_temporal_observation().is_ok());

        let interval = query.with_observation(
            ObservationSpec::IntervalCensored {
                latent: VariableId::from_raw(1),
                lower: VariableId::from_raw(1),
                upper: VariableId::from_raw(2),
            },
            [ObservationAssumption::IndependentGiven(Arc::from([]))],
        );
        assert!(interval.validate().is_ok());
        let err = interval.require_licensed_temporal_observation().unwrap_err();
        assert!(err.to_string().contains(TEMPORAL_OBSERVATION_UNLICENSED));
    }
}
