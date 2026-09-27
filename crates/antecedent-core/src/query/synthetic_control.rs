//! Retained synthetic-control design metadata.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// A balanced unit-by-period panel with one treated unit and donor units.
///
/// The causal interpretation requires a valid convex donor counterfactual,
/// no anticipation, no interference, and no concurrent treated-unit shock.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticControlQuery {
    /// Continuous outcome column.
    pub outcome: VariableId,
    /// Unit identity in row order.
    pub units: Arc<[Arc<str>]>,
    /// Calendar period in row order.
    pub periods: Arc<[i64]>,
    /// The single treated unit.
    pub treated_unit: Arc<str>,
    /// First treated period; all earlier periods fit donor weights.
    pub intervention_period: i64,
}

impl SyntheticControlQuery {
    /// Bind the row-aligned design at study preparation.
    #[must_use]
    pub fn new(
        outcome: VariableId,
        units: impl Into<Arc<[Arc<str>]>>,
        periods: impl Into<Arc<[i64]>>,
        treated_unit: impl Into<Arc<str>>,
        intervention_period: i64,
    ) -> Self {
        Self { outcome, units: units.into(), periods: periods.into(), treated_unit: treated_unit.into(), intervention_period }
    }

    /// Validate frozen metadata before compiling the analysis.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.units.is_empty() || self.units.len() != self.periods.len()
            || self.treated_unit.trim().is_empty() || self.intervention_period <= 0
            || self.units.iter().any(|unit| unit.trim().is_empty())
            || self.periods.iter().any(|period| *period <= 0)
        {
            return Err(QueryError::InvalidRandomizedEffect(
                "synthetic control requires aligned nonempty units, positive periods, and a treated unit with a positive intervention period".into(),
            ));
        }
        Ok(())
    }
}
