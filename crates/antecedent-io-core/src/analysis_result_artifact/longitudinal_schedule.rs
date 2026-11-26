//! Portable paired schedule scores and learned schedule confidence sets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::longitudinal_policy::{
    LearnedLongitudinalSchedule, LongitudinalScheduleComparison,
};
use serde::{Deserialize, Serialize};

/// Realized simultaneous comparison between two prespecified schedules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalScheduleContrastWire {
    /// Evaluated schedule code.
    pub schedule: u8,
    /// Reference schedule code.
    pub reference: u8,
    /// Mean difference of paired scores.
    pub difference: f64,
    /// Subject score standard error.
    pub standard_error: f64,
    /// Simultaneous interval coordinate.
    pub interval: [f64; 2],
}

/// Learning outputs. The selected value and its interval live on the typed regime result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalScheduleLearningWire {
    /// Selected by the prespecified training rule.
    pub selected_schedule: u8,
    /// Every schedule compatible with being optimal after simultaneous comparisons.
    pub optimum_set: Vec<u8>,
    /// Training-refit counts, in schedule order.
    pub refit_selections: Vec<(u8, usize)>,
    /// Variability of refit-held-out values, as a diagnostic.
    pub refit_value_sd: f64,
}

/// Policy family and simultaneous contrasts, with optional learned schedule outputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalScheduleResultWire {
    /// Absolute schedule values, in declared schedule order.
    pub values: Vec<(u8, f64)>,
    /// Comparison coordinates sharing the same subject-level multipliers.
    pub contrasts: Vec<LongitudinalScheduleContrastWire>,
    /// Maximum absolute multiplier quantile.
    pub critical_value: f64,
    /// Number of evaluation subjects.
    pub evaluation_subjects: usize,
    /// Present for a held-out trained schedule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning: Option<LongitudinalScheduleLearningWire>,
}

impl From<&LongitudinalScheduleComparison> for LongitudinalScheduleResultWire {
    fn from(value: &LongitudinalScheduleComparison) -> Self {
        Self {
            values: value.values.clone(),
            contrasts: value
                .contrasts
                .iter()
                .map(|contrast| LongitudinalScheduleContrastWire {
                    schedule: contrast.schedule,
                    reference: contrast.reference,
                    difference: contrast.difference,
                    standard_error: contrast.standard_error,
                    interval: contrast.interval,
                })
                .collect(),
            critical_value: value.critical_value,
            evaluation_subjects: value.evaluation_subjects,
            learning: None,
        }
    }
}

impl From<&LearnedLongitudinalSchedule> for LongitudinalScheduleResultWire {
    fn from(value: &LearnedLongitudinalSchedule) -> Self {
        let mut wire = Self::from(&value.comparison);
        wire.learning = Some(LongitudinalScheduleLearningWire {
            selected_schedule: value.selected_schedule,
            optimum_set: value.optimum_set.clone(),
            refit_selections: value.refit_selections.clone(),
            refit_value_sd: value.refit_value_sd,
        });
        wire
    }
}
