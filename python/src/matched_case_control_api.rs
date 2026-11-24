//! Python binding for the matched case-control conditional odds ratio.
//!
//! The odds ratio runs through its compiled program (`MatchedCaseControl` family,
//! `antecedent::analysis::descriptive_programs`), which calls the same kernel with the same
//! inputs and returns its result unchanged; the kernel's refusals reach Python as before.
use crate::transport_common::execution_context;
use antecedent::analysis::descriptive_programs::{
    DescriptiveOutcome, DescriptiveProgramError, DescriptiveRequest, compile_descriptive,
};
use antecedent::estimate::{EstimationError, parse_matched_sampling};
use pyo3::prelude::*;

fn estimation_err(error: EstimationError) -> PyErr {
    match error {
        EstimationError::Refused { code, message } => crate::refusal(code, message),
        other => crate::value_err(other.to_string()),
    }
}

fn program_err(error: DescriptiveProgramError) -> PyErr {
    match error {
        DescriptiveProgramError::Estimation(error) => estimation_err(error),
        DescriptiveProgramError::NotSupported(refusal) => crate::not_supported_err(&refusal),
        other => crate::refusal("public_builder_reentry", other.to_string()),
    }
}

/// The odds-ratio interval with its calibration slot as JSON.
type MatchedInterval = (f64, f64, String);

/// The matched odds ratio as returned to Python.
type MatchedOddsRatio = (f64, f64, usize, usize, usize, usize, usize, Option<MatchedInterval>);

/// Conditional odds ratio of matched sets: `(log odds ratio, odds ratio, sets, informative,
/// exposure-concordant, outcome-degenerate, singleton, interval)`. The declared sampling
/// design and estimand are checked before any data is read. A `level` adds the Wald interval
/// of the odds ratio `(lower, upper, calibration slot JSON)`; no coverage record measures
/// it, so the slot is `not_measured`.
#[pyfunction]
#[pyo3(signature=(stratum, case, exposed, *, sampling, estimand, level=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn matched_case_control_odds_ratio(
    py: Python<'_>,
    stratum: Vec<String>,
    case: Vec<f64>,
    exposed: Vec<f64>,
    sampling: &str,
    estimand: &str,
    level: Option<f64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<MatchedOddsRatio> {
    parse_matched_sampling(sampling).map_err(estimation_err)?;
    estimand.parse::<antecedent::declared::MatchedEstimand>().map_err(estimation_err)?;
    if level.is_some_and(|level| !(level > 0.0 && level < 1.0)) {
        return Err(crate::with_reason_code(
            crate::value_err(
                "matched_case_control.invalid_level: level must lie strictly between 0 and 1",
            ),
            antecedent::reason::reason_code!("invalid_argument"),
        ));
    }
    let ctx = execution_context(0, None, cancel);
    crate::detach_catch(py, move || {
        let request = DescriptiveRequest::MatchedOddsRatio { stratum, case, exposed };
        let outcome = compile_descriptive(request, &ctx)
            .and_then(|program| program.evaluate(&ctx))
            .map_err(program_err)?;
        let DescriptiveOutcome::MatchedOddsRatio(fitted) = outcome else {
            return Err(crate::value_err(
                "the compiled matched case-control program returned another result kind",
            ));
        };
        let c = fitted.counts;
        let interval = level.and_then(|level| fitted.odds_ratio_interval(level)).map(|(lo, hi)| {
            let slot = antecedent::io::contract_section::CalibrationSlotWire::not_measured(
                antecedent::io::calibration::CONSTRUCTION_NOT_MEASURED,
            );
            (lo, hi, serde_json::to_string(&slot).unwrap_or_default())
        });
        Ok((
            fitted.log_odds_ratio,
            fitted.odds_ratio,
            c.total,
            c.informative,
            c.exposure_concordant,
            c.outcome_degenerate,
            c.singleton,
            interval,
        ))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(matched_case_control_odds_ratio, module)?)?;
    Ok(())
}
