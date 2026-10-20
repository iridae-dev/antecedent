//! Retained posterior of an unlicensed Bayesian randomized-design row.
//!
//! Python sees the Rust result as is: coordinates, means, equal-tailed 95% credible intervals,
//! the covariance of the joint draws (row-major), the exact Gaussian moments when the residual
//! variance is known, and the named scalar and vector summaries. The row is never licensed and
//! its calibration is `unmeasured`.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use pyo3::prelude::*;

/// Posterior of an unlicensed Bayesian randomized-design row from Study execution.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PosteriorRowSection {
    pub estimator: String,
    pub target: String,
    pub uncertainty: String,
    pub calibration: String,
    pub licensed: bool,
    pub coordinates: Vec<String>,
    pub mean: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub covariance: Option<Vec<f64>>,
    pub exact_mean: Option<Vec<f64>>,
    pub exact_covariance: Option<Vec<f64>>,
    pub draws: usize,
    pub extras: BTreeMap<String, f64>,
    pub vectors: BTreeMap<String, Vec<f64>>,
}

impl From<&antecedent::PosteriorRowEstimate> for PosteriorRowSection {
    fn from(value: &antecedent::PosteriorRowEstimate) -> Self {
        Self {
            estimator: value.estimator.clone(),
            target: value.target.clone(),
            uncertainty: value.uncertainty.clone(),
            calibration: value.calibration.clone(),
            licensed: value.licensed,
            coordinates: value.coordinates.clone(),
            mean: value.mean.clone(),
            lower: value.lower.clone(),
            upper: value.upper.clone(),
            covariance: value.covariance.clone(),
            exact_mean: value.exact_mean.clone(),
            exact_covariance: value.exact_covariance.clone(),
            draws: value.draws,
            extras: value.extras.clone(),
            vectors: value.vectors.clone(),
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PosteriorRowSection>()
}
