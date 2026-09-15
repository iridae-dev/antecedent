//! Four typed reasoning slots: Identification, Support, Uncertainty, Assumptions.
//!
//! Structured state is authoritative. Unknown fields are
//! [`SlotAvailability::Unavailable`], never empty objects that imply success.
//! Priors, validation, and successful estimation never upgrade identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::identification::IdentificationStatus;
use crate::obligation::ObligationRecord;

/// Whether a semantic field is present.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SlotAvailability<T> {
    /// Inspectable value.
    Available(T),
    /// Explicitly unavailable; `reason` is a stable id, not prose.
    Unavailable {
        /// Stable reason id (`not_prepared`, `execution_specific`, …).
        reason: Arc<str>,
    },
}

impl<T> SlotAvailability<T> {
    /// Construct an unavailable slot.
    #[must_use]
    pub fn unavailable(reason: impl Into<Arc<str>>) -> Self {
        Self::Unavailable { reason: reason.into() }
    }

    /// Whether a value is present.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }

    /// Borrow the value when present.
    #[must_use]
    pub const fn as_ref(&self) -> Option<&T> {
        match self {
            Self::Available(value) => Some(value),
            Self::Unavailable { .. } => None,
        }
    }

    /// Map a present value, or format `unavailable:<reason>`.
    #[must_use]
    pub fn label(&self, available: impl FnOnce(&T) -> String) -> String {
        match self {
            Self::Available(value) => available(value),
            Self::Unavailable { reason } => format!("unavailable:{reason}"),
        }
    }
}

/// Identification slot: status, masses, and search scope.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct IdentificationSlot {
    /// Overall identification status.
    pub status: IdentificationStatus,
    /// Fraction of structural mass that is identified and evaluable.
    pub identified_mass: f64,
    /// Fraction structurally unidentified.
    pub unidentified_mass: f64,
    /// Fraction identified but not evaluable by the selected estimator.
    pub unevaluable_mass: f64,
    /// Fraction left out of a capped search / Interactive subsample.
    pub incomplete_search_mass: f64,
    /// Whether reported mass covers the full class.
    pub full_mass_scope: bool,
    /// Interpretation of atom weights, when this is a mixture.
    pub weight_basis: Option<Arc<str>>,
    /// Whether identification search was capped before a determination.
    pub search_capped: bool,
}

impl IdentificationSlot {
    /// Fully specified identification slot.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        status: IdentificationStatus,
        identified_mass: f64,
        unidentified_mass: f64,
        unevaluable_mass: f64,
        incomplete_search_mass: f64,
        full_mass_scope: bool,
        weight_basis: Option<Arc<str>>,
        search_capped: bool,
    ) -> Self {
        Self {
            status,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
            incomplete_search_mass,
            full_mass_scope,
            weight_basis,
            search_capped,
        }
    }

    /// Single-atom identified result (DAG / ADMG point identification).
    #[must_use]
    pub fn identified_singleton(status: IdentificationStatus) -> Self {
        let identified = !matches!(status, IdentificationStatus::NotIdentified);
        Self {
            status,
            identified_mass: if identified { 1.0 } else { 0.0 },
            unidentified_mass: if identified { 0.0 } else { 1.0 },
            unevaluable_mass: 0.0,
            incomplete_search_mass: 0.0,
            full_mass_scope: true,
            weight_basis: None,
            search_capped: false,
        }
    }
}

/// Matrix licensing versus empirical support. They stay distinct.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct SupportSlot {
    /// Support-matrix evidence status (`licensed`, `not_applicable`, `refused`).
    pub matrix_status: Arc<str>,
    /// Actual matrix coordinate, when classified.
    pub matrix_coordinate: Option<Arc<str>>,
    /// Empirical support status, when evaluated.
    pub empirical: SlotAvailability<Arc<str>>,
}

impl SupportSlot {
    /// Construct a support slot.
    #[must_use]
    pub fn new(
        matrix_status: impl Into<Arc<str>>,
        matrix_coordinate: Option<Arc<str>>,
        empirical: SlotAvailability<Arc<str>>,
    ) -> Self {
        Self { matrix_status: matrix_status.into(), matrix_coordinate, empirical }
    }
}

/// Source of a reported uncertainty number.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum UncertaintySource {
    /// Sampling / replicate uncertainty.
    Sampling,
    /// Parameter uncertainty.
    Parameter,
    /// Structural / orientation uncertainty.
    Structural,
    /// Identification / partial-identification bounds.
    Identification,
    /// Mechanism uncertainty.
    Mechanism,
    /// Regime uncertainty.
    Regime,
    /// Measurement uncertainty.
    Measurement,
}

impl UncertaintySource {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sampling => "sampling",
            Self::Parameter => "parameter",
            Self::Structural => "structural",
            Self::Identification => "identification",
            Self::Mechanism => "mechanism",
            Self::Regime => "regime",
            Self::Measurement => "measurement",
        }
    }
}

/// One reported uncertainty component. Not a generic interval reducer.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct UncertaintyComponent {
    /// Source of this component.
    pub source: UncertaintySource,
    /// Reported target (`se`, `credible_interval`, `identified_set`, …).
    pub target: Arc<str>,
    /// Whether this component is omitted / unresolved.
    pub omitted: bool,
}

impl UncertaintyComponent {
    /// Construct one uncertainty component.
    #[must_use]
    pub fn new(source: UncertaintySource, target: impl Into<Arc<str>>, omitted: bool) -> Self {
        Self { source, target: target.into(), omitted }
    }
}

/// Uncertainty slot: sources and reported targets.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct UncertaintySlot {
    /// Declared components. Empty only when the slot is unavailable.
    pub components: Arc<[UncertaintyComponent]>,
}

impl UncertaintySlot {
    /// Construct from components.
    #[must_use]
    pub fn new(components: impl Into<Arc<[UncertaintyComponent]>>) -> Self {
        Self { components: components.into() }
    }
}

/// Assumption / obligation slot.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct AssumptionSlot {
    /// Scoped obligations, including uncheckable and not-run checks.
    pub obligations: Arc<[ObligationRecord]>,
}

impl AssumptionSlot {
    /// Construct from obligations.
    #[must_use]
    pub fn new(obligations: impl Into<Arc<[ObligationRecord]>>) -> Self {
        Self { obligations: obligations.into() }
    }
}

/// Compact four-slot view. Structured state is authoritative.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ReasoningView {
    /// Identification.
    pub identification: SlotAvailability<IdentificationSlot>,
    /// Support (matrix and empirical).
    pub support: SlotAvailability<SupportSlot>,
    /// Uncertainty.
    pub uncertainty: SlotAvailability<UncertaintySlot>,
    /// Assumptions and obligations.
    pub assumptions: SlotAvailability<AssumptionSlot>,
}

impl ReasoningView {
    /// Fully specified four-slot view.
    #[must_use]
    pub const fn new(
        identification: SlotAvailability<IdentificationSlot>,
        support: SlotAvailability<SupportSlot>,
        uncertainty: SlotAvailability<UncertaintySlot>,
        assumptions: SlotAvailability<AssumptionSlot>,
    ) -> Self {
        Self { identification, support, uncertainty, assumptions }
    }

    /// Inspection before identification: ID and uncertainty unavailable.
    #[must_use]
    pub fn structural(support: SupportSlot, assumptions: AssumptionSlot) -> Self {
        Self {
            identification: SlotAvailability::unavailable("not_prepared"),
            support: SlotAvailability::Available(support),
            uncertainty: SlotAvailability::unavailable("execution_specific"),
            assumptions: SlotAvailability::Available(assumptions),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identification::IdentificationStatus;

    #[test]
    fn unavailable_is_not_an_empty_success() {
        let view = ReasoningView::structural(
            SupportSlot {
                matrix_status: Arc::from("licensed"),
                matrix_coordinate: None,
                empirical: SlotAvailability::unavailable("not_evaluated"),
            },
            AssumptionSlot { obligations: Arc::from([]) },
        );
        assert!(!view.identification.is_available());
        assert!(view.support.is_available());
        assert!(!view.uncertainty.is_available());
        match &view.identification {
            SlotAvailability::Unavailable { reason } => assert_eq!(&**reason, "not_prepared"),
            SlotAvailability::Available(_) => panic!("identification must stay unavailable"),
        }
    }

    #[test]
    fn singleton_not_identified_does_not_report_unit_identified_mass() {
        let slot = IdentificationSlot::identified_singleton(IdentificationStatus::NotIdentified);
        assert!((slot.identified_mass - 0.0).abs() < f64::EPSILON);
        assert!((slot.unidentified_mass - 1.0).abs() < f64::EPSILON);
    }
}
