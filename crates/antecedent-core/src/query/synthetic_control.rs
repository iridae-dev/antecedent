//! Retained synthetic-control design metadata.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Fitted synthetic-panel contrast.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SyntheticPanelMethod {
    /// Post-treatment treated-versus-convex-donor outcome contrast.
    #[default]
    Control,
    /// Difference in differences with convex donor and pre-period weights.
    DifferenceInDifferences,
}

/// A balanced unit-by-period panel with one treated unit and donor units.
///
/// The causal interpretation requires a valid donor counterfactual or
/// synthetic-DiD untreated trend, no anticipation, no interference, and no
/// concurrent treated-unit shock. Exact unit-randomization inference further
/// requires a declared uniform single-treated-unit assignment.
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
    /// Distinguishes synthetic control from synthetic difference in differences.
    pub method: SyntheticPanelMethod,
    /// Declare uniform assignment of exactly one treated unit for a sharp-null test.
    pub uniform_unit_randomization: bool,
    /// Prespecified constant additive effect for an exact sharp-null test.
    /// Requires declared uniform assignment; absent means the zero-effect null.
    pub sharp_null_effect: Option<f64>,
    /// Optional donor-trained ridge outcome-model correction to the simplex fit.
    pub augmentation_ridge: Option<f64>,
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
        Self { outcome, units: units.into(), periods: periods.into(), treated_unit: treated_unit.into(), intervention_period, method: SyntheticPanelMethod::Control, uniform_unit_randomization: false, sharp_null_effect: None, augmentation_ridge: None }
    }

    /// Use the same frozen panel for a synthetic difference-in-differences contrast.
    #[must_use]
    pub fn difference_in_differences(mut self) -> Self {
        self.method = SyntheticPanelMethod::DifferenceInDifferences;
        self
    }

    /// Request exhaustive unit-level Fisher randomization inference.
    #[must_use]
    pub fn with_uniform_unit_randomization(mut self) -> Self {
        self.uniform_unit_randomization = true;
        self
    }

    /// Test a prespecified constant additive post-treatment effect under a
    /// declared uniform choice of one treated unit.
    #[must_use]
    pub fn with_sharp_null_effect(mut self, effect: f64) -> Self {
        self.sharp_null_effect = Some(effect);
        self
    }

    /// Correct the simplex contrast with a donor-trained ridge outcome model.
    #[must_use]
    pub fn with_augmentation(mut self, ridge_penalty: f64) -> Self {
        self.augmentation_ridge = Some(ridge_penalty);
        self
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
        if let Some(ridge) = self.augmentation_ridge {
            if !ridge.is_finite() || ridge <= 0.0 {
                return Err(QueryError::InvalidRandomizedEffect(
                    "augmented synthetic control requires a finite positive ridge penalty".into(),
                ));
            }
            if self.method != SyntheticPanelMethod::Control {
                return Err(QueryError::InvalidRandomizedEffect(
                    "augmented synthetic control cannot be combined with synthetic DiD".into(),
                ));
            }
        }
        if let Some(effect) = self.sharp_null_effect {
            if !effect.is_finite() || !self.uniform_unit_randomization {
                return Err(QueryError::InvalidRandomizedEffect(
                    "a finite sharp-null effect requires declared uniform single-unit randomization".into(),
                ));
            }
        }
        Ok(())
    }
}
