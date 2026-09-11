//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;
use crate::intervention::Intervention;
use crate::value::Value;

use super::AverageEffectQuery;
use super::TargetPopulation;
use super::error::QueryError;
use super::response::MAX_TEMPORAL_RESPONSE_HORIZONS;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
/// Which mediation contrast to identify / estimate (linear SEM path).
pub enum MediationContrast {
    /// Total effect (direct + mediated).
    Total,
    /// Natural direct path-sum that avoids declared mediators.
    ///
    /// Static `mediation.linear` aliases this to [`NaturalDirect`]. It is not a
    /// general controlled direct effect that holds mediators fixed under
    /// treatment–mediator interaction.
    Direct,
    /// Mediated / indirect effect (path through mediators).
    Mediated,
    /// Natural direct effect (linear SEM: coincides with controlled direct under linearity).
    NaturalDirect,
    /// Natural indirect effect (linear SEM: coincides with mediated under linearity).
    NaturalIndirect,
}

/// Mediation query: treatment → mediators → outcome.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MediationQuery {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Mediator set (non-empty).
    pub mediators: Arc<[VariableId]>,
    /// Contrast of interest.
    pub contrast: MediationContrast,
    /// Control intervention level.
    pub control: Intervention,
    /// Active intervention level.
    pub active: Intervention,
    /// Target population.
    pub target_population: TargetPopulation,
    /// Outcome horizons in steps after a Pulse origin at 0 (each ≥ 1).
    ///
    /// Strictly increasing; at least one entry. [`Self::binary`] defaults to
    /// `[1]` so existing single-horizon callers stay valid. Identification
    /// and estimation click `I(h)` for each requested horizon separately.
    pub horizons: Arc<[u32]>,
}

impl MediationQuery {
    /// Linear mediation with binary 0/1 treatment levels.
    #[must_use]
    pub fn binary(
        treatment: VariableId,
        outcome: VariableId,
        mediators: impl Into<Arc<[VariableId]>>,
        contrast: MediationContrast,
    ) -> Self {
        Self {
            treatment,
            outcome,
            mediators: mediators.into(),
            contrast,
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            target_population: TargetPopulation::AllObserved,
            horizons: Arc::from([1]),
        }
    }

    /// Replace requested horizons after validating the same rules as
    /// [`super::TemporalResponseSpec`] (nonempty, each ≥ 1, strictly
    /// increasing, count ≤ [`MAX_TEMPORAL_RESPONSE_HORIZONS`]).
    ///
    /// # Errors
    ///
    /// Empty, zero, non-increasing, or oversized horizon lists.
    pub fn with_horizons(mut self, horizons: impl Into<Arc<[u32]>>) -> Result<Self, QueryError> {
        self.horizons = horizons.into();
        self.validate_horizons()?;
        Ok(self)
    }

    fn validate_horizons(&self) -> Result<(), QueryError> {
        if self.horizons.is_empty() {
            return Err(QueryError::InvalidResponse(
                "temporal mediation requires at least one horizon".into(),
            ));
        }
        if self.horizons.len() > MAX_TEMPORAL_RESPONSE_HORIZONS {
            return Err(QueryError::InvalidResponse(
                "temporal mediation horizon count exceeds the materialization cap".into(),
            ));
        }
        if self.horizons.iter().any(|h| *h == 0) {
            return Err(QueryError::NonPositiveHorizon);
        }
        if self.horizons.windows(2).any(|w| w[0] >= w[1]) {
            return Err(QueryError::InvalidResponse(
                "temporal mediation horizons must be strictly increasing".into(),
            ));
        }
        Ok(())
    }

    /// Validate ids and interventions.
    ///
    /// # Errors
    ///
    /// Empty mediators, overlaps, or inconsistent interventions.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.outcome {
            return Err(QueryError::TreatmentEqualsOutcome { id: self.treatment });
        }
        if self.mediators.is_empty() {
            return Err(QueryError::EmptyMediators);
        }
        if self.mediators.iter().any(|&m| m == self.treatment || m == self.outcome) {
            return Err(QueryError::MediatorOverlapsTreatmentOrOutcome);
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
        self.target_population.validate()?;
        self.validate_horizons()?;
        Ok(())
    }
}

/// Conditional average effect given effect modifiers.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ConditionalEffectQuery {
    /// Inner ATE-style query; `effect_modifiers` must be non-empty.
    pub inner: AverageEffectQuery,
}

impl ConditionalEffectQuery {
    /// Wrap an ATE query that already carries modifiers.
    ///
    /// # Errors
    ///
    /// Empty effect modifiers.
    pub fn try_new(inner: AverageEffectQuery) -> Result<Self, QueryError> {
        if inner.effect_modifiers.is_empty() {
            return Err(QueryError::EmptyEffectModifiers);
        }
        inner.validate()?;
        Ok(Self { inner })
    }

    /// Validate.
    ///
    /// # Errors
    ///
    /// Empty modifiers or invalid inner query.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.inner.effect_modifiers.is_empty() {
            return Err(QueryError::EmptyEffectModifiers);
        }
        self.inner.validate()
    }
}
