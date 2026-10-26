//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, encode_u32s, encode_variable_set};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::Intervention;
use crate::value::Value;

use super::AverageEffectQuery;
use super::TargetPopulation;
use super::error::QueryError;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
/// Which mediation contrast to identify / estimate (linear SEM path).
pub enum MediationContrast {
    /// Total effect (direct + mediated).
    Total,
    /// Natural direct path-sum that avoids declared mediators.
    ///
    /// Static `mediation.linear` aliases this to [`Self::NaturalDirect`]. It is not a
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

impl CanonicalEncode for MediationContrast {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let tag = match self {
            Self::Total => 1,
            Self::Direct => 2,
            Self::Mediated => 3,
            Self::NaturalDirect => 4,
            Self::NaturalIndirect => 5,
        };
        writer.word(tag);
    }
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
    /// `[1]` so existing single-horizon callers stay valid. Each entry is
    /// identified as `I(h)` and retained on the mediation grid; a single
    /// horizon also fills the scalar compatibility fields.
    pub horizons: Arc<[u32]>,
}

/// Every field is written. `mediators` is a set and is encoded in ascending id
/// order; `horizons` is an ordered grid the results are indexed by.
impl CanonicalEncode for MediationQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.treatment.encode(writer);
        self.outcome.encode(writer);
        encode_variable_set(writer, &self.mediators);
        self.contrast.encode(writer);
        self.control.encode(writer);
        self.active.encode(writer);
        self.target_population.encode(writer);
        encode_u32s(writer, &self.horizons);
    }
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
    /// increasing).
    ///
    /// # Errors
    ///
    /// Empty, zero, or non-increasing horizon lists.
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

/// The wrapped average-effect query; the variant tag of
/// [`super::CausalQuery`] separates it from a plain average effect.
impl CanonicalEncode for ConditionalEffectQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.inner.encode(writer);
    }
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
