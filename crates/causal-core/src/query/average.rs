//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;
use crate::intervention::Intervention;
use crate::value::Value;

use super::TargetPopulation;
use super::error::QueryError;

/// Average treatment effect (ATE / ATT-style) query.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct AverageEffectQuery {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Optional effect modifiers .
    pub effect_modifiers: Arc<[VariableId]>,
    /// Control intervention level (typically treatment = 0).
    pub control: Intervention,
    /// Active intervention level (typically treatment = 1).
    pub active: Intervention,
    /// Target population.
    pub target_population: TargetPopulation,
}

impl AverageEffectQuery {
    /// Full constructor (required outside this crate because the type is `#[non_exhaustive]`).
    #[must_use]
    pub fn new(
        treatment: VariableId,
        outcome: VariableId,
        effect_modifiers: impl Into<Arc<[VariableId]>>,
        control: Intervention,
        active: Intervention,
        target_population: TargetPopulation,
    ) -> Self {
        Self {
            treatment,
            outcome,
            effect_modifiers: effect_modifiers.into(),
            control,
            active,
            target_population,
        }
    }

    /// ATE for binary treatment coded as 0/1 on `treatment`.
    ///
    /// # Examples
    ///
    /// ```
    /// use causal_core::{AverageEffectQuery, VariableId};
    ///
    /// let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    /// assert_eq!(q.treatment, VariableId::from_raw(0));
    /// ```
    #[must_use]
    pub fn binary_ate(treatment: VariableId, outcome: VariableId) -> Self {
        Self::new(
            treatment,
            outcome,
            Arc::from([]),
            Intervention::set(treatment, Value::f64(0.0)),
            Intervention::set(treatment, Value::f64(1.0)),
            TargetPopulation::AllObserved,
        )
    }

    /// ATE with explicit control/active float levels.
    #[must_use]
    pub fn with_levels(
        treatment: VariableId,
        outcome: VariableId,
        control_level: f64,
        active_level: f64,
    ) -> Self {
        Self {
            treatment,
            outcome,
            effect_modifiers: Arc::from([]),
            control: Intervention::set(treatment, Value::f64(control_level)),
            active: Intervention::set(treatment, Value::f64(active_level)),
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Attach effect modifiers (IDs already resolved).
    #[must_use]
    pub fn with_effect_modifiers(mut self, modifiers: impl Into<Arc<[VariableId]>>) -> Self {
        self.effect_modifiers = modifiers.into();
        self
    }

    /// Set target population.
    #[must_use]
    pub fn with_target_population(mut self, population: TargetPopulation) -> Self {
        self.target_population = population;
        self
    }

    /// Validate that interventions target the treatment variable.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError`] when interventions are inconsistent.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.outcome {
            return Err(QueryError::TreatmentEqualsOutcome { id: self.treatment });
        }
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
        if self.effect_modifiers.iter().any(|m| *m == self.treatment || *m == self.outcome) {
            return Err(QueryError::ModifierOverlapsTreatmentOrOutcome);
        }
        self.target_population.validate()?;
        Ok(())
    }
}
