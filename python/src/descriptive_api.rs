//! Python binding for the descriptive raw-versus-adjusted comparison and the declared
//! reporting-scale transform (2.2 E7).
use crate::transport_common::execution_context;
use antecedent_estimate::{
    AdjustedEstimate, Availability, EstimationError, MeanPair, RawContrast, ReportingScale,
    ReportingTransform, compare_raw_adjusted, raw_contrast, refuse_column_attribution,
    refuse_transform_interval,
};
use pyo3::prelude::*;

fn estimation_err(error: EstimationError) -> PyErr {
    match error {
        EstimationError::Refused { code, message } => crate::refusal(code, message),
        other => crate::value_err(other.to_string()),
    }
}

/// `(count, mean, sum of squared deviations)` of one arm.
type Arm = (usize, f64, f64);

/// The raw contrast beside an adjusted estimate:
/// `(active arm, control arm, raw difference, raw Welch SE or None, gap = raw - adjusted)`.
type Comparison = (Arm, Arm, f64, Option<f64>, f64);

fn arm(summary: &antecedent_estimate::ArmSummary) -> Arm {
    (summary.n, summary.mean, summary.m2)
}

fn raw_se(raw: &RawContrast) -> Option<f64> {
    raw.se.available().copied()
}

/// Raw versus adjusted contrast of the same declared estimand coding. The adjusted estimate
/// declares its scale, levels and population; any coding other than a 1-versus-0 difference
/// of means on the all-observed population is refused before data are read.
#[pyfunction]
#[pyo3(signature=(outcome, treatment, adjusted_estimate, *, adjusted_se=None, scale, active, control, all_observed, cancel=None))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn raw_vs_adjusted(
    py: Python<'_>,
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    adjusted_estimate: f64,
    adjusted_se: Option<f64>,
    scale: &str,
    active: f64,
    control: f64,
    all_observed: bool,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Comparison> {
    let adjusted = AdjustedEstimate {
        estimate: adjusted_estimate,
        se: adjusted_se,
        active,
        control,
        scale: ReportingScale::parse(scale).map_err(estimation_err)?,
        all_observed_population: all_observed,
    };
    let ctx = execution_context(0, None, cancel);
    crate::detach_catch(py, move || {
        let comparison =
            compare_raw_adjusted(&outcome, &treatment, adjusted, &ctx).map_err(estimation_err)?;
        let raw = &comparison.raw;
        Ok((arm(&raw.active), arm(&raw.control), raw.difference, raw_se(raw), comparison.gap))
    })
}

/// A column attribution of the gap is always refused.
#[pyfunction]
fn attribute_gap_to_columns() -> PyResult<()> {
    Err(estimation_err(refuse_column_attribution()))
}

/// `(values, gradients, covariance)` of a transformed family; the covariance is column-major
/// `k x k` or `None` when the source carries no joint covariance.
type Transformed = (Vec<f64>, Vec<(f64, f64)>, Option<Vec<f64>>);

fn transformed(transform: &ReportingTransform) -> Transformed {
    (
        transform.contrasts.iter().map(|c| c.value).collect(),
        transform.contrasts.iter().map(|c| (c.gradient[0], c.gradient[1])).collect(),
        match &transform.covariance {
            Availability::Available(values) => Some(values.clone()),
            Availability::Unavailable(_) => None,
        },
    )
}

fn parse_scales(scales: &[String]) -> PyResult<Vec<ReportingScale>> {
    scales.iter().map(|name| ReportingScale::parse(name).map_err(estimation_err)).collect()
}

/// Put a mean-outcome pair on the declared scales. `covariance` is
/// `(var(active), cov, var(control))` as the source published it, or `None`. An interval
/// request is refused before any work.
#[pyfunction]
#[pyo3(signature=(mean_active, mean_control, scales, *, covariance=None, interval=false))]
#[allow(clippy::needless_pass_by_value)]
fn transform_mean_pair(
    mean_active: f64,
    mean_control: f64,
    scales: Vec<String>,
    covariance: Option<(f64, f64, f64)>,
    interval: bool,
) -> PyResult<Transformed> {
    if interval {
        return Err(estimation_err(refuse_transform_interval()));
    }
    let pair = MeanPair::new(mean_active, mean_control, covariance.map(|(a, c, b)| [a, c, b]));
    let transform = antecedent_estimate::transform_mean_pair(&pair, &parse_scales(&scales)?)
        .map_err(estimation_err)?;
    Ok(transformed(&transform))
}

/// Put the raw arm means of a binary-treatment sample on the declared scales, with the
/// independent-groups covariance of the two arm means. Returns the arm summaries and the
/// transformed family.
#[pyfunction]
#[pyo3(signature=(outcome, treatment, scales, *, interval=false, cancel=None))]
#[allow(clippy::needless_pass_by_value)]
fn transform_raw_arms(
    py: Python<'_>,
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    scales: Vec<String>,
    interval: bool,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Arm, Arm, Transformed)> {
    if interval {
        return Err(estimation_err(refuse_transform_interval()));
    }
    let scales = parse_scales(&scales)?;
    let ctx = execution_context(0, None, cancel);
    crate::detach_catch(py, move || {
        let raw = raw_contrast(&outcome, &treatment, &ctx).map_err(estimation_err)?;
        let transform = antecedent_estimate::transform_mean_pair(&raw.mean_pair(), &scales)
            .map_err(estimation_err)?;
        Ok((arm(&raw.active), arm(&raw.control), transformed(&transform)))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(raw_vs_adjusted, module)?)?;
    module.add_function(wrap_pyfunction!(attribute_gap_to_columns, module)?)?;
    module.add_function(wrap_pyfunction!(transform_mean_pair, module)?)?;
    module.add_function(wrap_pyfunction!(transform_raw_arms, module)?)?;
    Ok(())
}
