//! Retained synthetic-control design metadata.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::PanelDesignFault;
use super::QueryError;
use crate::canonical_encode::{CanonicalEncode, encode_i64s, flag, i64_bits};
use crate::identity::CanonicalWriter;
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

impl CanonicalEncode for SyntheticPanelMethod {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let tag = match self {
            Self::Control => 1,
            Self::DifferenceInDifferences => 2,
        };
        writer.word(tag);
    }
}

/// Interval construction of a synthetic rollout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SyntheticInterval {
    /// Conformal inference by inverting a permutation test over time (one treated unit).
    #[default]
    Conformal,
    /// Placebo variance from reassigning the adoptions to never-treated units.
    PlaceboVariance,
}

impl SyntheticInterval {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conformal => "conformal",
            Self::PlaceboVariance => "placebo_variance",
        }
    }
}

/// The adopting units and interval construction of a synthetic panel with intervals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntheticRollout {
    /// Every adopting unit with its first treated period; it lists the query's treated unit at
    /// its intervention period.
    pub adoptions: Arc<[(Arc<str>, i64)]>,
    /// Interval construction.
    pub interval: SyntheticInterval,
    /// Placebo replicates; zero for a conformal interval.
    pub replicates: u32,
}

/// Every field is written; adoptions keep their declared order.
impl CanonicalEncode for SyntheticRollout {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(self.adoptions.len() as u64);
        for (unit, period) in self.adoptions.iter() {
            writer.text(unit).word(i64_bits(*period));
        }
        let interval = match self.interval {
            SyntheticInterval::Conformal => 1,
            SyntheticInterval::PlaceboVariance => 2,
        };
        writer.word(interval).word(u64::from(self.replicates));
    }
}

/// Marker word ahead of the rollout field, written only when it is present so a query without
/// it keeps its canonical bytes.
const ROLLOUT_FIELD: u64 = 0x5359_4e52_4f4c_4c00;

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
    /// Adopting units and interval construction of a synthetic panel with intervals.
    pub rollout: Option<SyntheticRollout>,
}

/// Every field is written; `units` and `periods` are row-aligned and keep
/// their order.
impl CanonicalEncode for SyntheticControlQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        writer.texts(&self.units);
        encode_i64s(writer, &self.periods);
        writer.text(&self.treated_unit).word(i64_bits(self.intervention_period));
        self.method.encode(writer);
        writer
            .word(flag(self.uniform_unit_randomization))
            .opt_real(self.sharp_null_effect)
            .opt_real(self.augmentation_ridge);
        if let Some(rollout) = &self.rollout {
            writer.word(ROLLOUT_FIELD);
            rollout.encode(writer);
        }
    }
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
        Self {
            outcome,
            units: units.into(),
            periods: periods.into(),
            treated_unit: treated_unit.into(),
            intervention_period,
            method: SyntheticPanelMethod::Control,
            uniform_unit_randomization: false,
            sharp_null_effect: None,
            augmentation_ridge: None,
            rollout: None,
        }
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

    /// Report conformal intervals for the single treated unit.
    #[must_use]
    pub fn with_conformal_intervals(mut self) -> Self {
        self.rollout = Some(SyntheticRollout {
            adoptions: Arc::from([(Arc::clone(&self.treated_unit), self.intervention_period)]),
            interval: SyntheticInterval::Conformal,
            replicates: 0,
        });
        self
    }

    /// Fit every adopting unit of a staggered rollout against the never-treated donors and
    /// report placebo-variance intervals from `replicates` placebo reassignments.
    /// `adoptions` must list the query's treated unit at its intervention period.
    #[must_use]
    pub fn with_staggered_adoptions(
        mut self,
        adoptions: impl Into<Arc<[(Arc<str>, i64)]>>,
        replicates: u32,
    ) -> Self {
        self.rollout = Some(SyntheticRollout {
            adoptions: adoptions.into(),
            interval: SyntheticInterval::PlaceboVariance,
            replicates,
        });
        self
    }

    /// Validate frozen metadata before compiling the analysis.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidRandomizedEffect`] when the units and
    /// periods are misaligned or empty, the treated unit is blank, the
    /// intervention period is not positive, an augmentation ridge is not finite
    /// and positive or is combined with synthetic `DiD`, or a sharp-null effect is
    /// non-finite or declared without uniform single-unit randomization.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.units.is_empty()
            || self.units.len() != self.periods.len()
            || self.treated_unit.trim().is_empty()
            || self.intervention_period <= 0
            || self.units.iter().any(|unit| unit.trim().is_empty())
            || self.periods.iter().any(|period| *period <= 0)
        {
            return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::SyntheticControlPanel));
        }
        if let Some(ridge) = self.augmentation_ridge {
            if !ridge.is_finite() || ridge <= 0.0 {
                return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::AugmentedRidge));
            }
            if self.method != SyntheticPanelMethod::Control {
                return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::AugmentedWithDid));
            }
        }
        if let Some(effect) = self.sharp_null_effect {
            if !effect.is_finite() || !self.uniform_unit_randomization {
                return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::SharpNullRandomization));
            }
        }
        if let Some(rollout) = &self.rollout {
            let units = rollout
                .adoptions
                .iter()
                .map(|(unit, _)| unit)
                .collect::<std::collections::BTreeSet<_>>();
            if rollout.adoptions.is_empty()
                || units.len() != rollout.adoptions.len()
                || rollout
                    .adoptions
                    .iter()
                    .any(|(unit, period)| unit.trim().is_empty() || *period <= 0)
                || !rollout.adoptions.iter().any(|(unit, period)| {
                    *unit == self.treated_unit && *period == self.intervention_period
                })
            {
                return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::SyntheticAdoptions));
            }
            if self.uniform_unit_randomization || self.augmentation_ridge.is_some() {
                return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::IntervalWithRandomizationTest));
            }
            match rollout.interval {
                SyntheticInterval::Conformal
                    if rollout.adoptions.len() != 1
                        || self.method != SyntheticPanelMethod::Control =>
                {
                    return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::ConformalIntervalDesign));
                }
                SyntheticInterval::PlaceboVariance if rollout.replicates < 2 => {
                    return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::PlaceboReplicates));
                }
                _ => {}
            }
        }
        Ok(())
    }
}
