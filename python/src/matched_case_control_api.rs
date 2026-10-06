//! Python binding for the matched case-control conditional odds ratio (2.2 E7).
use crate::transport_common::execution_context;
use antecedent_estimate::{
    EstimationError, conditional_odds_ratio, parse_matched_estimand, parse_matched_sampling,
    refuse_matched_interval,
};
use pyo3::prelude::*;

fn estimation_err(error: EstimationError) -> PyErr {
    match error {
        EstimationError::Refused { code, message } => crate::refusal(code, message),
        other => crate::value_err(other.to_string()),
    }
}

/// Conditional odds ratio of matched sets: `(log odds ratio, odds ratio, sets, informative,
/// exposure-concordant, outcome-degenerate, singleton)`. The declared sampling design and
/// estimand are checked, and a requested interval refused, before any data is read.
#[pyfunction]
#[pyo3(signature=(stratum, case, exposed, *, sampling, estimand, interval=false, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn matched_case_control_odds_ratio(
    py: Python<'_>,
    stratum: Vec<String>,
    case: Vec<f64>,
    exposed: Vec<f64>,
    sampling: &str,
    estimand: &str,
    interval: bool,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(f64, f64, usize, usize, usize, usize, usize)> {
    parse_matched_sampling(sampling).map_err(estimation_err)?;
    parse_matched_estimand(estimand).map_err(estimation_err)?;
    if interval {
        return Err(estimation_err(refuse_matched_interval()));
    }
    let ctx = execution_context(0, None, cancel);
    crate::detach_catch(py, move || {
        let fitted =
            conditional_odds_ratio(&stratum, &case, &exposed, &ctx).map_err(estimation_err)?;
        let c = fitted.counts;
        Ok((
            fitted.log_odds_ratio,
            fitted.odds_ratio,
            c.total,
            c.informative,
            c.exposure_concordant,
            c.outcome_degenerate,
            c.singleton,
        ))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(matched_case_control_odds_ratio, module)?)?;
    Ok(())
}
