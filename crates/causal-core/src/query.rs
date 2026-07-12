//! Typed causal queries (DESIGN.md §8).
//!
//! Hot paths bind [`VariableId`]s; names are resolved only at API boundaries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::{EnvironmentId, VariableId};
use crate::intervention::Intervention;
use crate::value::Value;

/// Target population for an effect query (DESIGN.md §8.2).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TargetPopulation {
    /// All observed units.
    AllObserved,
    /// Treated units only.
    Treated,
    /// Untreated units only.
    Untreated,
    /// Environment-restricted population.
    Environment(EnvironmentId),
}

/// Average treatment effect (ATE / ATT-style) query.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AverageEffectQuery {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Optional effect modifiers (Phase 1: empty / unused by estimators).
    pub effect_modifiers: Arc<[VariableId]>,
    /// Control intervention level (typically treatment = 0).
    pub control: Intervention,
    /// Active intervention level (typically treatment = 1).
    pub active: Intervention,
    /// Target population.
    pub target_population: TargetPopulation,
}

impl AverageEffectQuery {
    /// ATE for binary treatment coded as 0/1 on `treatment`.
    #[must_use]
    pub fn binary_ate(treatment: VariableId, outcome: VariableId) -> Self {
        Self {
            treatment,
            outcome,
            effect_modifiers: Arc::from([]),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            target_population: TargetPopulation::AllObserved,
        }
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
        if self.control.primary_variable() != self.treatment {
            return Err(QueryError::InterventionVariableMismatch {
                expected: self.treatment,
                got: self.control.primary_variable(),
            });
        }
        if self.active.primary_variable() != self.treatment {
            return Err(QueryError::InterventionVariableMismatch {
                expected: self.treatment,
                got: self.active.primary_variable(),
            });
        }
        if self.effect_modifiers.iter().any(|m| *m == self.treatment || *m == self.outcome) {
            return Err(QueryError::ModifierOverlapsTreatmentOrOutcome);
        }
        Ok(())
    }
}

/// Top-level causal query enum.
///
/// Variants beyond [`CausalQuery::AverageEffect`] are reserved; Phase 1 facades
/// must reject them rather than inventing behavior.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CausalQuery {
    /// Average / population effect.
    AverageEffect(AverageEffectQuery),
}

impl CausalQuery {
    /// Construct an average-effect query.
    #[must_use]
    pub fn average_effect(query: AverageEffectQuery) -> Self {
        Self::AverageEffect(query)
    }

    /// Whether this query is supported by the Phase 1 static ATE path.
    #[must_use]
    pub const fn is_phase1_ate(&self) -> bool {
        matches!(self, Self::AverageEffect(_))
    }
}

/// Errors from query construction or validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueryError {
    /// Treatment and outcome are the same variable.
    TreatmentEqualsOutcome {
        /// Shared id.
        id: VariableId,
    },
    /// Intervention does not target the declared treatment.
    InterventionVariableMismatch {
        /// Expected treatment id.
        expected: VariableId,
        /// Actual intervention target.
        got: VariableId,
    },
    /// Effect modifier overlaps treatment or outcome.
    ModifierOverlapsTreatmentOrOutcome,
}

impl core::fmt::Display for QueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TreatmentEqualsOutcome { id } => {
                write!(f, "treatment and outcome are the same variable {id}")
            }
            Self::InterventionVariableMismatch { expected, got } => {
                write!(f, "intervention targets {got}, expected treatment {expected}")
            }
            Self::ModifierOverlapsTreatmentOrOutcome => {
                write!(f, "effect modifier overlaps treatment or outcome")
            }
        }
    }
}

impl std::error::Error for QueryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_ate_binds_ids_not_names() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let q = AverageEffectQuery::binary_ate(t, y);
        q.validate().unwrap();
        assert_eq!(q.treatment, t);
        assert_eq!(q.outcome, y);
        assert_eq!(q.target_population, TargetPopulation::AllObserved);
        match &q.control {
            Intervention::Set { variable, value } => {
                assert_eq!(*variable, t);
                assert_eq!(*value, Value::f64(0.0));
            }
        }
    }

    #[test]
    fn rejects_treatment_equals_outcome() {
        let id = VariableId::from_raw(0);
        let q = AverageEffectQuery::binary_ate(id, id);
        assert!(matches!(q.validate(), Err(QueryError::TreatmentEqualsOutcome { .. })));
    }

    #[test]
    fn causal_query_phase1_flag() {
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        assert!(q.is_phase1_ate());
    }
}
