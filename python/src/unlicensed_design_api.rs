//! Result sections of the observational sequential contrast and the within-period history match.
//!
//! Both are unlicensed and uncalibrated: the sections carry no support status, the retained scores
//! or matched pairs a consumer recomputes from, and the standing of the reported uncertainty.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;

/// Observational two-decision contrast of two static regimes, with its cross-fit scores.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct SequentialContrastSection {
    pub active_regime: (u8, u8),
    pub control_regime: (u8, u8),
    pub contrast: f64,
    pub active_mean: f64,
    pub control_mean: f64,
    pub evaluated_rows: u64,
    pub minimum_action_probability: f64,
    pub min_probability: f64,
    pub standard_error: Option<f64>,
    pub interval: Option<(f64, f64)>,
    pub coverage: f64,
    pub active_scores: Vec<f64>,
    pub control_scores: Vec<f64>,
    pub contrast_scores: Vec<f64>,
    pub uncertainty: String,
    pub fits_computed: u64,
    pub fits_reused: u64,
}

impl From<&antecedent::SequentialContrastEstimate> for SequentialContrastSection {
    fn from(value: &antecedent::SequentialContrastEstimate) -> Self {
        Self {
            active_regime: (value.active_regime[0], value.active_regime[1]),
            control_regime: (value.control_regime[0], value.control_regime[1]),
            contrast: value.contrast,
            active_mean: value.active_mean,
            control_mean: value.control_mean,
            evaluated_rows: value.evaluated_rows,
            minimum_action_probability: value.minimum_action_probability,
            min_probability: value.min_probability,
            standard_error: value.standard_error,
            interval: value.interval.map(|bounds| (bounds[0], bounds[1])),
            coverage: value.coverage,
            active_scores: value.active_scores.to_vec(),
            control_scores: value.control_scores.to_vec(),
            contrast_scores: value.contrast_scores.to_vec(),
            uncertainty: value.uncertainty.to_string(),
            fits_computed: value.fits_computed,
            fits_reused: value.fits_reused,
        }
    }
}

/// Within-period history-matched contrast with its matched pairs.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct HistoryMatchSection {
    pub contrast: f64,
    pub standard_error: Option<f64>,
    pub weight_total: f64,
    pub matched_queries: Vec<usize>,
    pub matched_donors: Vec<usize>,
    pub differences: Vec<f64>,
    pub pair_weights: Vec<f64>,
    pub unmatched_treated: Vec<usize>,
    pub periods_matched: usize,
    pub caliper: Option<f64>,
    pub uncertainty: String,
}

impl From<&antecedent::HistoryMatchEstimate> for HistoryMatchSection {
    fn from(value: &antecedent::HistoryMatchEstimate) -> Self {
        Self {
            contrast: value.contrast,
            standard_error: value.standard_error,
            weight_total: value.weight_total,
            matched_queries: value.matched_queries.to_vec(),
            matched_donors: value.matched_donors.to_vec(),
            differences: value.differences.to_vec(),
            pair_weights: value.pair_weights.to_vec(),
            unmatched_treated: value.unmatched_treated.to_vec(),
            periods_matched: value.periods_matched,
            caliper: value.caliper,
            uncertainty: value.uncertainty.to_string(),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SequentialContrastSection>()?;
    m.add_class::<HistoryMatchSection>()
}
