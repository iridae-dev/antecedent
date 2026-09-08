//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;
use crate::intervention::Intervention;
use crate::value::Value;

use super::error::QueryError;

#[derive(Clone, Debug, PartialEq)]
/// Counterfactual query over factual observations and interventions.
pub struct CounterfactualQuery {
    /// Outcome variable(s) to predict under the counterfactual world.
    pub outcomes: Arc<[VariableId]>,
    /// Interventions defining the counterfactual (active) world after abduction.
    pub interventions: Arc<[Intervention]>,
    /// Control world for a two-world ITE (typically a hard set of treatment to 0).
    pub control: Intervention,
    /// When true, allow nested counterfactual interventions under invertible SCMs.
    pub allow_nested: bool,
}

impl CounterfactualQuery {
    /// Hard set of the first intervention's target to `0.0`.
    ///
    /// [`Self::new`] uses this; format-0.4 query decode uses it when the wire omits `control`.
    #[must_use]
    pub fn default_control(interventions: &[Intervention]) -> Intervention {
        let variable = interventions
            .first()
            .and_then(Intervention::primary_variable)
            .unwrap_or(VariableId::from_raw(0));
        Intervention::set(variable, Value::f64(0.0))
    }

    /// Construct a single-outcome counterfactual query.
    ///
    /// Control defaults to a hard set of the first intervention's target to `0.0`.
    #[must_use]
    pub fn new(outcome: VariableId, interventions: impl Into<Arc<[Intervention]>>) -> Self {
        let interventions = interventions.into();
        let control = Self::default_control(&interventions);
        Self { outcomes: Arc::from([outcome]), interventions, control, allow_nested: false }
    }

    /// Replace the control world (ITE baseline).
    #[must_use]
    pub fn with_control(mut self, control: Intervention) -> Self {
        self.control = control;
        self
    }

    /// Hard-set the control level on the current control (or first intervention) target.
    #[must_use]
    pub fn with_control_level(self, level: f64) -> Self {
        let variable = self
            .control
            .primary_variable()
            .or_else(|| self.interventions.first().and_then(Intervention::primary_variable))
            .unwrap_or(VariableId::from_raw(0));
        self.with_control(Intervention::set(variable, Value::f64(level)))
    }

    /// Enable nested interventions where the model supports them.
    #[must_use]
    pub const fn with_nested(mut self, allow_nested: bool) -> Self {
        self.allow_nested = allow_nested;
        self
    }

    /// Validate interventions.
    ///
    /// # Errors
    ///
    /// Empty outcomes, invalid interventions, or a control that targets a different
    /// variable than the first active intervention.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcomes.is_empty() {
            return Err(QueryError::EmptyCounterfactualOutcomes);
        }
        for iv in self.interventions.iter() {
            iv.validate().map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
        }
        self.control.validate().map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
        if let Some(active_var) =
            self.interventions.first().and_then(Intervention::primary_variable)
        {
            let control_var =
                self.control.primary_variable().ok_or(QueryError::AmbiguousInterventionTarget)?;
            if control_var != active_var {
                return Err(QueryError::InterventionVariableMismatch {
                    expected: active_var,
                    got: control_var,
                });
            }
        }
        Ok(())
    }
}
