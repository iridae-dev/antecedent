//! Identified estimand types shared by identify and estimate (DESIGN.md §3.2 / §10.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::VariableId;

use crate::ExprId;

/// One identified estimand.
///
/// Backdoor estimands use [`Self::adjustment_set`]; IV estimands populate
/// [`Self::instruments`]; front-door estimands populate [`Self::mediators`].
/// Unused role slices are empty.
#[derive(Clone, Debug)]
pub struct IdentifiedEstimand {
    /// Method tag (e.g. `backdoor.adjustment`, `frontdoor`, `iv`).
    pub method: Arc<str>,
    /// Adjustment set (dense variable ids). Empty when not an adjustment estimand.
    pub adjustment_set: Arc<[VariableId]>,
    /// Instrument variables (dense ids). Empty unless IV.
    pub instruments: Arc<[VariableId]>,
    /// Mediator variables for front-door / two-stage. Empty unless front-door.
    pub mediators: Arc<[VariableId]>,
    /// Functional expression id in `arena`.
    pub functional: ExprId,
}

impl IdentifiedEstimand {
    /// Backdoor-style estimand with an adjustment set and empty IV/mediator roles.
    #[must_use]
    pub fn backdoor(
        method: impl Into<Arc<str>>,
        adjustment_set: Arc<[VariableId]>,
        functional: ExprId,
    ) -> Self {
        Self {
            method: method.into(),
            adjustment_set,
            instruments: Arc::from([]),
            mediators: Arc::from([]),
            functional,
        }
    }

    /// IV estimand with instruments and empty adjustment/mediators.
    #[must_use]
    pub fn instrumental(
        method: impl Into<Arc<str>>,
        instruments: Arc<[VariableId]>,
        functional: ExprId,
    ) -> Self {
        Self {
            method: method.into(),
            adjustment_set: Arc::from([]),
            instruments,
            mediators: Arc::from([]),
            functional,
        }
    }

    /// Front-door estimand with mediators and empty adjustment/instruments.
    #[must_use]
    pub fn frontdoor(
        method: impl Into<Arc<str>>,
        mediators: Arc<[VariableId]>,
        functional: ExprId,
    ) -> Self {
        Self {
            method: method.into(),
            adjustment_set: Arc::from([]),
            instruments: Arc::from([]),
            mediators,
            functional,
        }
    }
}
