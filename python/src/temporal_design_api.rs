//! Result section of the sequentially randomized unit-period designs and regime survival.
//!
//! The section is unlicensed and uncalibrated: it carries every named coordinate with its
//! estimate (or posterior mean), standard error (or posterior standard deviation), pointwise
//! interval, the simultaneous band and Holm family when the design has them, the worst-case
//! smoothing bias of a dose grid, and the standing of the reported uncertainty.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;

/// Excursion contrasts of a sequentially randomized design, or static-regime survival.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct TemporalDesignSection {
    pub design: String,
    pub functional: String,
    pub coordinates: Vec<String>,
    pub estimate: Vec<f64>,
    pub standard_error: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub covariance: Option<Vec<f64>>,
    pub band_coordinates: Vec<usize>,
    pub band_critical_value: Option<f64>,
    pub band_lower: Option<Vec<f64>>,
    pub band_upper: Option<Vec<f64>>,
    pub holm_adjusted_p: Option<Vec<f64>>,
    pub smoothing_bias_bound: Option<Vec<f64>>,
    pub pairs: u64,
    pub clusters: u64,
    pub unpaired: u64,
    pub minimum_probability: f64,
    pub replicates: Option<(u64, u64)>,
    pub draws: Option<u64>,
    pub uncertainty: String,
    pub calibration: String,
    pub licensed: bool,
}

impl From<&antecedent::TemporalDesignEstimate> for TemporalDesignSection {
    fn from(value: &antecedent::TemporalDesignEstimate) -> Self {
        Self {
            design: value.design.to_string(),
            functional: value.functional.to_string(),
            coordinates: value.coordinates.iter().map(ToString::to_string).collect(),
            estimate: value.estimate.to_vec(),
            standard_error: value.standard_error.to_vec(),
            lower: value.lower.to_vec(),
            upper: value.upper.to_vec(),
            covariance: value.covariance.as_ref().map(|values| values.to_vec()),
            band_coordinates: value.band_coordinates.to_vec(),
            band_critical_value: value.band_critical_value,
            band_lower: value.band_lower.as_ref().map(|values| values.to_vec()),
            band_upper: value.band_upper.as_ref().map(|values| values.to_vec()),
            holm_adjusted_p: value.holm_adjusted_p.as_ref().map(|values| values.to_vec()),
            smoothing_bias_bound: value.smoothing_bias_bound.as_ref().map(|values| values.to_vec()),
            pairs: value.pairs,
            clusters: value.clusters,
            unpaired: value.unpaired,
            minimum_probability: value.minimum_probability,
            replicates: value.replicates,
            draws: value.draws,
            uncertainty: value.uncertainty.to_string(),
            calibration: value.calibration.to_string(),
            licensed: value.licensed,
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TemporalDesignSection>()
}
