//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::VariableId;
use crate::intervention::{Intervention, TemporalPolicy};
use crate::value::Value;

use super::TargetPopulation;
use super::error::QueryError;

/// Temporal effect query over a discrete horizon.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalEffectQuery {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Temporal intervention policy.
    pub policy: TemporalPolicy,
    /// Control intervention level on the treatment variable.
    pub control: Intervention,
    /// Active intervention level on the treatment variable.
    pub active: Intervention,
    /// Outcome horizon in time steps after the policy origin (must be ≥ 1).
    pub horizon_steps: u32,
    /// Optional max history lag (steps) to retain when unfolding; `None` = planner default.
    pub max_history_lag: Option<u32>,
    /// Target population.
    pub target_population: TargetPopulation,
}

impl TemporalEffectQuery {
    /// Pulse intervention at step 0 with active float level; control is 0.0.
    #[must_use]
    pub fn pulse(treatment: VariableId, outcome: VariableId, active_level: f64) -> Self {
        Self {
            treatment,
            outcome,
            policy: TemporalPolicy::pulse(0),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(active_level)),
            horizon_steps: 1,
            max_history_lag: None,
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Sustained intervention on `[0, until]` with active float level; control is 0.0.
    #[must_use]
    pub fn sustained(
        treatment: VariableId,
        outcome: VariableId,
        until: i32,
        active_level: f64,
    ) -> Self {
        Self {
            treatment,
            outcome,
            policy: TemporalPolicy::sustained(0, until),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(active_level)),
            horizon_steps: 1,
            max_history_lag: None,
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Set outcome evaluation horizon in time steps.
    #[must_use]
    pub const fn with_horizon_steps(mut self, horizon_steps: u32) -> Self {
        self.horizon_steps = horizon_steps;
        self
    }

    /// Set optional max history lag for unfolding.
    #[must_use]
    pub const fn with_max_history_lag(mut self, max_history_lag: Option<u32>) -> Self {
        self.max_history_lag = max_history_lag;
        self
    }

    /// Replace the temporal policy.
    #[must_use]
    pub fn with_policy(mut self, policy: TemporalPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set target population.
    #[must_use]
    pub fn with_target_population(mut self, population: TargetPopulation) -> Self {
        self.target_population = population;
        self
    }

    /// Validate treatment/outcome, interventions, policy, and horizon.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError`] on inconsistent configuration.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.outcome {
            return Err(QueryError::TreatmentEqualsOutcome { id: self.treatment });
        }
        if self.horizon_steps == 0 {
            return Err(QueryError::NonPositiveHorizon);
        }
        self.policy.validate().map_err(|e| match e {
            crate::intervention::InterventionError::InvalidTemporalWindow { from, until } => {
                QueryError::InvalidTemporalWindow { from, until }
            }
            other => QueryError::InvalidIntervention(other.to_string()),
        })?;
        self.target_population.validate()?;
        let control_var =
            self.control.primary_variable().ok_or(QueryError::AmbiguousInterventionTarget)?;
        if control_var != self.treatment {
            return Err(QueryError::InterventionVariableMismatch {
                expected: self.treatment,
                got: control_var,
            });
        }
        let active_var =
            self.active.primary_variable().ok_or(QueryError::AmbiguousInterventionTarget)?;
        if active_var != self.treatment {
            return Err(QueryError::InterventionVariableMismatch {
                expected: self.treatment,
                got: active_var,
            });
        }
        Ok(())
    }

    /// Treatment time offset for Pulse `at` / Sustained `from` / Dynamic first active step.
    #[must_use]
    pub fn treatment_offset(&self) -> i32 {
        self.try_treatment_offset().unwrap_or_default()
    }

    /// Treatment time offset when the policy has a defined origin.
    ///
    /// # Errors
    ///
    /// [`QueryError::DynamicPolicyHasNoTreatmentOffset`] for an empty dynamic schedule.
    pub fn try_treatment_offset(&self) -> Result<i32, QueryError> {
        match &self.policy {
            TemporalPolicy::Pulse { at } => Ok(*at),
            TemporalPolicy::Sustained { from, .. } => Ok(*from),
            TemporalPolicy::Dynamic { active_at, .. } => {
                active_at.first().copied().ok_or(QueryError::DynamicPolicyHasNoTreatmentOffset)
            }
        }
    }

    /// Outcome evaluation offset: `horizon_steps - 1` (absolute from window origin).
    #[must_use]
    pub fn outcome_offset(&self) -> i32 {
        i32::try_from(self.horizon_steps.saturating_sub(1)).unwrap_or(i32::MAX)
    }
}
