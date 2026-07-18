//! Typed causal queries (DESIGN.md §8).
//!
//! Hot paths bind [`VariableId`]s; names are resolved only at API boundaries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod average;
mod attribution;
mod counterfactual;
mod distribution;
mod error;
mod mediation;
mod target;
mod temporal;

pub use crate::intervention::TemporalPolicy;

pub use average::AverageEffectQuery;
pub use attribution::{
    AllocationMethod, AnomalyAttributionQuery, AttributionComponents, ChangeAttributionQuery,
    MechanismChangeQuery, OrderedFloatBits, PopulationSelector, ShapleyConfig, ShapleyMode,
    UnitChangeQuery,
};
pub use counterfactual::CounterfactualQuery;
pub use distribution::{InterventionalDistributionQuery, PathSpecificEffectQuery};
pub use error::QueryError;
pub use mediation::{ConditionalEffectQuery, MediationContrast, MediationQuery};
pub use target::{PredicateExpr, TargetPopulation};
pub use temporal::TemporalEffectQuery;


/// Top-level causal query enum.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CausalQuery {
    /// Average / population effect (static).
    AverageEffect(AverageEffectQuery),
    /// Temporal effect over a discrete horizon.
    TemporalEffect(TemporalEffectQuery),
    /// Counterfactual / unit-level what-if query .
    Counterfactual(CounterfactualQuery),
    /// Anomaly attribution for one or more units .
    AnomalyAttribution(AnomalyAttributionQuery),
    /// Distribution / population change attribution .
    ChangeAttribution(ChangeAttributionQuery),
    /// Mechanism-change detection — not attribution .
    MechanismChange(MechanismChangeQuery),
    /// Per-unit change attribution .
    UnitChange(UnitChangeQuery),
    /// Mediation (direct / mediated / natural effects).
    Mediation(MediationQuery),
    /// Conditional average effect given modifiers.
    ConditionalEffect(ConditionalEffectQuery),
    /// Interventional distribution P(Y | do(...)).
    Distribution(InterventionalDistributionQuery),
    /// Path-specific effect / contribution.
    PathSpecific(PathSpecificEffectQuery),
}

impl CausalQuery {
    /// Construct an average-effect query.
    #[must_use]
    pub fn average_effect(query: AverageEffectQuery) -> Self {
        Self::AverageEffect(query)
    }

    /// Construct a temporal-effect query.
    #[must_use]
    pub fn temporal_effect(query: TemporalEffectQuery) -> Self {
        Self::TemporalEffect(query)
    }

    /// Construct a counterfactual query.
    #[must_use]
    pub fn counterfactual(query: CounterfactualQuery) -> Self {
        Self::Counterfactual(query)
    }

    /// Construct an anomaly attribution query.
    #[must_use]
    pub fn anomaly_attribution(query: AnomalyAttributionQuery) -> Self {
        Self::AnomalyAttribution(query)
    }

    /// Construct a change attribution query.
    #[must_use]
    pub fn change_attribution(query: ChangeAttributionQuery) -> Self {
        Self::ChangeAttribution(query)
    }

    /// Construct a mechanism-change detection query.
    #[must_use]
    pub fn mechanism_change(query: MechanismChangeQuery) -> Self {
        Self::MechanismChange(query)
    }

    /// Construct a unit-change attribution query.
    #[must_use]
    pub fn unit_change(query: UnitChangeQuery) -> Self {
        Self::UnitChange(query)
    }

    /// Construct a mediation query.
    #[must_use]
    pub fn mediation(query: MediationQuery) -> Self {
        Self::Mediation(query)
    }

    /// Construct a conditional-effect query.
    #[must_use]
    pub fn conditional_effect(query: ConditionalEffectQuery) -> Self {
        Self::ConditionalEffect(query)
    }

    /// Construct an interventional-distribution query.
    #[must_use]
    pub fn distribution(query: InterventionalDistributionQuery) -> Self {
        Self::Distribution(query)
    }

    /// Construct a path-specific effect query.
    #[must_use]
    pub fn path_specific(query: PathSpecificEffectQuery) -> Self {
        Self::PathSpecific(query)
    }

    /// Whether this query is the static ATE path.
    #[must_use]
    pub const fn is_static_ate(&self) -> bool {
        matches!(self, Self::AverageEffect(_))
    }

    /// Whether this query is a temporal effect.
    #[must_use]
    pub const fn is_temporal_effect(&self) -> bool {
        matches!(self, Self::TemporalEffect(_))
    }

    /// Whether this query is counterfactual.
    #[must_use]
    pub const fn is_counterfactual(&self) -> bool {
        matches!(self, Self::Counterfactual(_))
    }

    /// Whether this query is mediation.
    #[must_use]
    pub const fn is_mediation(&self) -> bool {
        matches!(self, Self::Mediation(_))
    }

    /// Whether this query is a conditional effect.
    #[must_use]
    pub const fn is_conditional_effect(&self) -> bool {
        matches!(self, Self::ConditionalEffect(_))
    }

    /// Whether this query is an interventional distribution.
    #[must_use]
    pub const fn is_distribution(&self) -> bool {
        matches!(self, Self::Distribution(_))
    }

    /// Whether this query is path-specific.
    #[must_use]
    pub const fn is_path_specific(&self) -> bool {
        matches!(self, Self::PathSpecific(_))
    }

    /// Validate the inner query.
    ///
    /// # Errors
    ///
    /// Propagates inner [`QueryError`].
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::AverageEffect(q) => q.validate(),
            Self::TemporalEffect(q) => q.validate(),
            Self::Counterfactual(q) => q.validate(),
            Self::AnomalyAttribution(q) => q.validate(),
            Self::ChangeAttribution(q) => q.validate(),
            Self::MechanismChange(q) => q.validate(),
            Self::UnitChange(q) => q.validate(),
            Self::Mediation(q) => q.validate(),
            Self::ConditionalEffect(q) => q.validate(),
            Self::Distribution(q) => q.validate(),
            Self::PathSpecific(q) => q.validate(),
        }
    }
}

/// Errors from query construction or validation.

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
