//! The compiled bias-aware dose grid.
//!
//! [`evaluate_dose_grid_bias_aware_compiled`] takes the request `evaluate_dose_grid_functional`
//! takes, with a declared curvature bound, and runs it through
//! `Study::seal_dose_grid_bias_aware`: the request is checked as that entry checks it (every
//! refusal, limit and the stored premises are unchanged), the `DoseGridBiasAware` program
//! evaluates the kernel once under the band's declared seed, and the portable artifact is sealed
//! from the program's fit. The report is that entry's report with the program's route, identity
//! and the estimator key its measured coverage is recorded under.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::dose_grid_functional::DoseGridRequestWire;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{
    EvaluatePayload, MAX_JSON_BYTES, dose_context, execution_error, refusal_json, report_json,
};
use crate::bridge::check_declared_size;
use crate::{detach_catch, value_err};

/// Run a bias-aware row through its compiled program and seal it:
/// `(report json, artifact bytes, refusal json)`.
///
/// The arguments are those of `evaluate_dose_grid_functional`; `request_json` must declare
/// `bias_aware`. The report is that entry's report with a `program` section: the plan route,
/// the lowered program identity and the calibration estimator key.
#[pyfunction]
#[pyo3(signature=(request_json, dose, outcome, *, sampling_weights=None, cluster=None, serial_lag=None, threads=None, max_bytes=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "the request, the table, its declarations and the controls"
)]
pub(super) fn evaluate_dose_grid_bias_aware_compiled(
    py: Python<'_>,
    request_json: &str,
    dose: Vec<f64>,
    outcome: Vec<f64>,
    sampling_weights: Option<Vec<f64>>,
    cluster: Option<Vec<u32>>,
    serial_lag: Option<u64>,
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<EvaluatePayload> {
    check_declared_size(request_json, MAX_JSON_BYTES, "dose grid")?;
    let mut request: DoseGridRequestWire = serde_json::from_str(request_json)
        .map_err(|e| value_err(format!("dose_grid.invalid_request: {e}")))?;
    if request.bias_aware.is_none() {
        return Err(value_err(format!(
            "{}: the compiled bias-aware row needs a curvature bound",
            antecedent::reason::reason_details::dose_grid::INVALID_REQUEST
        )));
    }
    request.dose = dose;
    request.outcome = outcome;
    request.sampling_weights = sampling_weights;
    request.cluster = cluster;
    request.serial_lag = serial_lag;
    let ctx = dose_context(threads, max_bytes, cancel)?;
    let payload = detach_catch(py, move || {
        let sealed = match Study::seal_dose_grid_bias_aware(request, &ctx) {
            Ok(sealed) => sealed,
            Err(error) => {
                return match refusal_json(&error, "dose_grid") {
                    Some(json) => Ok((None, None, Some(json))),
                    None => Err(execution_error(&error)),
                };
            }
        };
        let mut report: serde_json::Value = serde_json::from_str(&report_json(&sealed.sealed)?)
            .map_err(|e| crate::serialization_err(e.to_string()))?;
        report["program"] = serde_json::json!({
            "route": sealed.compiled.plan_view().route().as_str(),
            "lowered_program_identity": sealed.run.lowered_program_identity.to_hex(),
            "calibration_estimator": sealed.run.calibration_estimator,
        });
        let bytes = sealed.sealed.export().map_err(|e| crate::bridge::serialization(&e))?;
        Ok((Some(report.to_string()), Some(bytes), None))
    })?;
    let (report, bytes, refusal) = payload;
    Ok((report, bytes.map(|b| PyBytes::new(py, &b).unbind()), refusal))
}
