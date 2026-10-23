//! Result section of the binary-instrument bounds route.
//!
//! The bounds are computed by the compiled `IvBounds` study (`Study.compile` and
//! `prepare_iv_bounds`); this module only carries the identified set and the evidence behind it to
//! Python. The result is unlicensed and uncalibrated, and it is never a point.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;

/// Sampling interval about the identified set.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct IvBoundsIntervalSection {
    pub level: f64,
    pub lower: f64,
    pub upper: f64,
    pub lower_se: f64,
    pub upper_se: f64,
    pub critical_value: f64,
    pub width_retained: bool,
    pub replicates: usize,
    pub infeasible: usize,
    pub method: String,
}

/// Posterior summary of the endpoints under the Bayesian bootstrap.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct IvBoundsPosteriorSection {
    pub n_draws: usize,
    pub infeasible_draws: usize,
    pub lower_mean: f64,
    pub upper_mean: f64,
    pub lower_sd: f64,
    pub upper_sd: f64,
}

/// Sharp bounds on the average effect of a binary treatment under a binary instrument.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct IvBoundsSection {
    pub lower: f64,
    pub upper: f64,
    /// Observed law, instrument arm major; cell order `(0,0) (1,0) (0,1) (1,1)`.
    pub cells: Vec<f64>,
    pub arm_weights: Vec<f64>,
    pub rows: usize,
    pub rows_dropped: usize,
    pub interval: Option<IvBoundsIntervalSection>,
    pub posterior: Option<IvBoundsPosteriorSection>,
}

impl From<&antecedent::IvBoundsEstimate> for IvBoundsSection {
    fn from(value: &antecedent::IvBoundsEstimate) -> Self {
        Self {
            lower: value.lower,
            upper: value.upper,
            cells: value.law.cells.iter().flatten().copied().collect(),
            arm_weights: value.arm_weights.to_vec(),
            rows: value.rows,
            rows_dropped: value.rows_dropped,
            interval: value.interval.map(|interval| IvBoundsIntervalSection {
                level: interval.level,
                lower: interval.lower,
                upper: interval.upper,
                lower_se: interval.lower_se,
                upper_se: interval.upper_se,
                critical_value: interval.critical_value,
                width_retained: interval.width_retained,
                replicates: interval.replicates,
                infeasible: interval.infeasible,
                method: interval.method.as_str().to_owned(),
            }),
            posterior: value.posterior.map(|posterior| IvBoundsPosteriorSection {
                n_draws: posterior.n_draws,
                infeasible_draws: posterior.infeasible_draws,
                lower_mean: posterior.lower_mean,
                upper_mean: posterior.upper_mean,
                lower_sd: posterior.lower_sd,
                upper_sd: posterior.upper_sd,
            }),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IvBoundsIntervalSection>()?;
    m.add_class::<IvBoundsPosteriorSection>()?;
    m.add_class::<IvBoundsSection>()
}
