//! Continuous causal-response queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::intervention::TemporalPolicy;
use crate::{Intervention, TargetPopulation, VariableId};

use super::QueryError;

/// Maximum number of discrete horizons a temporal response may request.
pub const MAX_TEMPORAL_RESPONSE_HORIZONS: usize = 512;

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
}

impl TemporalResponseSpec {
    /// Construct a temporal attachment after validating horizons and policy.
    ///
    /// # Errors
    ///
    /// Empty/non-increasing/oversized horizons, zero horizon, or invalid policy.
    pub fn new(
        horizons: impl Into<Arc<[u32]>>,
        policy: TemporalPolicy,
        max_history_lag: Option<u32>,
    ) -> Result<Self, QueryError> {
        let horizons = horizons.into();
        let spec = Self { horizons, policy, max_history_lag };
        spec.validate()?;
        Ok(spec)
    }

    /// Validate horizons and nested policy.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidResponse`] or temporal-policy errors.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.horizons.is_empty() {
            return Err(QueryError::InvalidResponse(
                "temporal response requires at least one horizon".into(),
            ));
        }
        if self.horizons.len() > MAX_TEMPORAL_RESPONSE_HORIZONS {
            return Err(QueryError::InvalidResponse(
                "temporal response horizon count exceeds the materialization cap".into(),
            ));
        }
        if self.horizons.iter().any(|h| *h == 0) {
            return Err(QueryError::NonPositiveHorizon);
        }
        if self.horizons.windows(2).any(|w| w[0] >= w[1]) {
            return Err(QueryError::InvalidResponse(
                "temporal response horizons must be strictly increasing".into(),
            ));
        }
        self.policy.validate().map_err(|e| match e {
            crate::intervention::InterventionError::InvalidTemporalWindow { from, until } => {
                QueryError::InvalidTemporalWindow { from, until }
            }
            other => QueryError::InvalidIntervention(other.to_string()),
        })?;
        match &self.policy {
            TemporalPolicy::Pulse { .. } | TemporalPolicy::Sustained { .. } => {}
            TemporalPolicy::Dynamic { .. } => {
                return Err(QueryError::InvalidResponse(
                    "temporal response policy must be pulse or sustained; \
                     Dynamic is a TemporalEffect spelling, not a ResponseCurve cell"
                        .into(),
                ));
            }
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

/// Maximum treatment dimension for an explicitly gridded non-parametric surface.
pub const MAX_NONPARAMETRIC_RESPONSE_DIM: usize = 2;

/// Maximum number of points a [`GridSpec::Linspace`] may materialize.
///
/// `GridSpec::values` previously only checked that `points` fit in a `u32`, so e.g.
/// `Linspace { points: 4_000_000_000 }` passed validation and then tried to allocate a
/// multi-gigabyte `Vec<f64>` (8 bytes/point). This cap keeps materialization bounded to
/// something a caller could plausibly intend as an evaluation grid; 1,000,000 points is already
/// far beyond what any of this crate's response/derivative estimators need per grid.
pub const MAX_MATERIALIZED_GRID_POINTS: usize = 1_000_000;

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

impl GridSpec {
    /// Materialize the grid after validation.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidResponse`] when the grid is invalid or too large to materialize.
    pub fn values(&self) -> Result<Vec<f64>, QueryError> {
        self.validate()?;
        Ok(match self {
            Self::Values(values) => values.to_vec(),
            Self::Linspace { start, end, points } => {
                let points = u32::try_from(*points).map_err(|_| {
                    QueryError::InvalidResponse("linspace point count exceeds u32 capacity".into())
                })?;
                let step = (end - start) / f64::from(points - 1);
                (0..points).map(|i| start + f64::from(i) * step).collect()
            }
        })
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
                    return Err(QueryError::InvalidResponse(
                        "a response grid requires at least two points".into(),
                    ));
                }
                if values.iter().any(|v| !v.is_finite()) || values.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(QueryError::InvalidResponse(
                        "response-grid values must be finite and strictly increasing".into(),
                    ));
                }
            }
            Self::Linspace { start, end, points } => {
                if !start.is_finite() || !end.is_finite() || start >= end || *points < 2 {
                    return Err(QueryError::InvalidResponse(
                        "linspace requires finite start < end and at least two points".into(),
                    ));
                }
                if *points > MAX_MATERIALIZED_GRID_POINTS {
                    return Err(QueryError::InvalidResponse(
                        "linspace point count is too large to materialize".into(),
                    ));
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
                interventions.iter().filter_map(Intervention::primary_variable).collect()
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
        }
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
                            "uniform derivative weighting requires finite lower < upper".into(),
                        ));
                    }
                    DerivativeWeighting::Custom(weights)
                        if weights.is_empty()
                            || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
                            || weights.iter().all(|w| *w == 0.0) =>
                    {
                        return Err(QueryError::InvalidResponse(
                            "custom derivative weights must be finite, non-negative, and non-zero"
                                .into(),
                        ));
                    }
                    _ => {}
                }
            }
            ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => {
                if outcome == treatment {
                    return Err(QueryError::TreatmentEqualsOutcome { id: *outcome });
                }
                if !at.is_finite() || !matches!(order, 1 | 2) {
                    return Err(QueryError::InvalidResponse(
                        "point derivative requires a finite point and order one or two".into(),
                    ));
                }
                if matches!(scale, DerivativeScale::LogTreatment | DerivativeScale::LogLog)
                    && *at <= 0.0
                {
                    return Err(QueryError::InvalidResponse(
                        "log-treatment derivative scales require a positive treatment point".into(),
                    ));
                }
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                if !response_sets_are_distinct(outcomes, treatments)
                    || at.len() != treatments.len()
                    || direction.len() != treatments.len()
                    || at.iter().chain(direction.iter()).any(|v| !v.is_finite())
                    || direction.iter().all(|v| *v == 0.0)
                {
                    return Err(QueryError::InvalidResponse(
                        "directional derivative dimensions/values are inconsistent".into(),
                    ));
                }
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                if !response_sets_are_distinct(outcomes, treatments)
                    || at.len() != treatments.len()
                    || at.iter().any(|v| !v.is_finite())
                {
                    return Err(QueryError::InvalidResponse(
                        "Jacobian dimensions/values are inconsistent".into(),
                    ));
                }
                if matches!(scale, DerivativeScale::LogTreatment | DerivativeScale::LogLog)
                    && at.iter().any(|v| *v <= 0.0)
                {
                    return Err(QueryError::InvalidResponse(
                        "log-treatment Jacobians require positive treatment coordinates".into(),
                    ));
                }
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                if interventions.is_empty() {
                    return Err(QueryError::InvalidResponse(
                        "intervention response requires at least one intervention".into(),
                    ));
                }
                for intervention in interventions.iter() {
                    intervention
                        .validate()
                        .map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
                    if intervention.primary_variable() == Some(*outcome) {
                        return Err(QueryError::TreatmentEqualsOutcome { id: *outcome });
                    }
                }
            }
        }
        self.target_population.validate()?;
        Ok(())
    }

    fn validate_temporal_attachment(&self) -> Result<(), QueryError> {
        if let Some(temporal) = &self.temporal {
            temporal.validate()?;
            match &self.functional {
                ResponseFunctional::MeanCurve { .. }
                | ResponseFunctional::InterventionResponse { .. } => {}
                _ => {
                    return Err(QueryError::InvalidResponse(
                        "temporal attachment is licensed only for MeanCurve and InterventionResponse"
                            .into(),
                    ));
                }
            }
            if self.observation != ObservationSpec::Complete {
                return Err(QueryError::InvalidResponse(
                    "temporal response requires complete observation in 0.7".into(),
                ));
            }
        }
        Ok(())
    }
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
    fn linspace_within_cap_validates_and_materializes() {
        let grid = GridSpec::Linspace { start: 0.0, end: 1.0, points: 5 };
        assert!(grid.validate().is_ok());
        assert_eq!(grid.values().unwrap().len(), 5);
    }

    #[test]
    fn linspace_beyond_materialization_cap_is_rejected() {
        // Before the fix, only `u32::try_from(points)` was checked, so a huge-but-u32-valid
        // point count (here, well over MAX_MATERIALIZED_GRID_POINTS but still far under
        // u32::MAX) would sail through validation and then try to allocate an
        // unreasonably large `Vec<f64>`.
        let grid =
            GridSpec::Linspace { start: 0.0, end: 1.0, points: MAX_MATERIALIZED_GRID_POINTS + 1 };
        let err = grid.validate().unwrap_err();
        assert!(matches!(err, QueryError::InvalidResponse(_)));
        assert!(grid.values().is_err());
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
    fn linspace_point_count_far_beyond_u32_capacity_is_still_rejected_by_the_cap() {
        // Guards the original bug report directly: a `points` value so large the old
        // `u32::try_from` guard alone would have rejected it, but only after already deciding
        // the input was otherwise well-formed. The size cap must reject it first and for the
        // documented "too large to materialize" reason.
        let grid = GridSpec::Linspace { start: 0.0, end: 1.0, points: 4_000_000_000 };
        let err = grid.validate().unwrap_err();
        assert!(matches!(err, QueryError::InvalidResponse(_)));
    }
}
