//! The compiled bias-aware dose grid.
//!
//! [`evaluate_dose_grid_bias_aware_compiled`] takes the request `evaluate_dose_grid_functional`
//! takes, with a declared curvature bound. It seals the portable artifact exactly as that entry
//! does (every refusal, limit and the stored premises are unchanged), binds the sealed premises
//! and rows into the `DoseGridBiasAware` program family
//! (`Study::compile_dose_grid_bias_aware`) and executes it under the band's declared seed. The
//! program evaluates the same kernel the artifact stores; the report is returned only when every
//! row value, standard error, worst-case bias, bias-aware interval, band bound and critical value
//! of the program equals the sealed one bit for bit, with the program's route, identity and the
//! estimator key its measured coverage is recorded under.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::compiled_dose_grid_bias_aware::{
    DoseGridBiasAwareRequest, DoseGridBiasAwareRun,
};
use antecedent::analysis::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridFunctional, DoseGridRequest, DoseGridRequestWire,
    DoseGridResultWire,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{
    EvaluatePayload, MAX_JSON_BYTES, dose_context, execution_error, refusal_json, report_json,
};
use crate::bridge::check_declared_size;
use crate::{detach_catch, value_err};

/// Whether two slices hold the same values bit for bit.
fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Whether two optional slices hold the same values bit for bit.
fn same_optional_bits(left: Option<&[f64]>, right: Option<&[f64]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => same_bits(left, right),
        _ => false,
    }
}

/// The compiled request of a sealed bias-aware request: the stored premises and rows.
fn compiled_request(wire: &DoseGridRequestWire) -> PyResult<DoseGridBiasAwareRequest> {
    let invalid = |what: &str| value_err(format!("dose_grid.invalid_request: {what}"));
    let declared = wire.bias_aware.ok_or_else(|| invalid("no curvature bound is declared"))?;
    let design = match wire.design.as_str() {
        "randomized_dose" => DoseDesign::RandomizedDose,
        "observational" => DoseDesign::Observational,
        _ => return Err(invalid("design spelling")),
    };
    let functional = match (wire.functional.kind.as_str(), wire.functional.from, wire.functional.to)
    {
        ("level", None, None) => DoseFunctional::Level,
        ("derivative", None, None) => DoseFunctional::Derivative,
        ("contrast", Some(from), Some(to)) => DoseFunctional::Contrast { from, to },
        _ => return Err(invalid("functional kind")),
    };
    let replicates =
        usize::try_from(declared.band_replicates).map_err(|_| invalid("band replicates"))?;
    let row = DoseGridRequest {
        dose: &wire.dose,
        outcome: &wire.outcome,
        grid: &wire.grid,
        bandwidth: wire.bandwidth,
        bandwidth_range: wire.bandwidth_range,
        minimum_local_ess: wire.minimum_local_ess,
        functional,
        claims: ClaimSet {
            pointwise_level: wire.claims.pointwise_level,
            derivative: wire.claims.derivative,
            simultaneous_band: wire.claims.simultaneous_band,
        },
        design,
    };
    let mut request = DoseGridBiasAwareRequest::new(&row, declared.curvature_bound, replicates);
    if let Some(weights) = &wire.sampling_weights {
        request = request.sampling_weights(weights.clone());
    }
    if let Some(clusters) = &wire.cluster {
        request = request.clusters(clusters.clone());
    }
    if let Some(lag) = wire.serial_lag {
        request = request.serial_lag(usize::try_from(lag).map_err(|_| invalid("serial lag"))?);
    }
    Ok(request)
}

/// Whether the program's row and bias-aware claims are the sealed ones, bit for bit.
fn agrees(run: &DoseGridBiasAwareRun, sealed: &DoseGridResultWire) -> bool {
    let fit = &run.fit;
    let Some(section) = sealed.bias_aware.as_ref() else { return false };
    let lower: Vec<f64> = fit.intervals.iter().map(|interval| interval.lower).collect();
    let upper: Vec<f64> = fit.intervals.iter().map(|interval| interval.upper).collect();
    let bias_aware = section.curvature_bound.to_bits() == fit.curvature_bound.to_bits()
        && same_bits(&section.smoothing_bias_bound, &fit.smoothing_bias_bound)
        && same_bits(&section.lower, &lower)
        && same_bits(&section.upper, &upper)
        && same_optional_bits(
            section.band_lower.as_deref(),
            fit.band.as_ref().map(|band| band.lower.as_slice()),
        )
        && same_optional_bits(
            section.band_upper.as_deref(),
            fit.band.as_ref().map(|band| band.upper.as_slice()),
        )
        && section.band_critical_value.map(f64::to_bits)
            == fit.band_critical_value.map(f64::to_bits)
        && u64::try_from(fit.band_replicates).ok() == Some(section.band_replicates);
    let row = &fit.row;
    let point = |values: (f64, f64, f64, f64), wire: (f64, f64, f64, f64)| {
        same_bits(&[values.0, values.1, values.2, values.3], &[wire.0, wire.1, wire.2, wire.3])
    };
    let levels = row.levels.len() == sealed.levels.len()
        && row.levels.iter().zip(&sealed.levels).all(|(level, wire)| {
            point(
                (level.level, level.standard_error, level.interval.lower, level.interval.upper),
                (wire.value, wire.standard_error, wire.lower, wire.upper),
            )
        });
    let derivatives = row.derivatives.len() == sealed.derivatives.len()
        && row.derivatives.iter().zip(&sealed.derivatives).all(|(derivative, wire)| {
            point(
                (
                    derivative.derivative,
                    derivative.standard_error,
                    derivative.interval.lower,
                    derivative.interval.upper,
                ),
                (wire.value, wire.standard_error, wire.lower, wire.upper),
            )
        });
    let contrast = match (&row.contrast, &sealed.contrast) {
        (None, None) => true,
        (Some(contrast), Some(wire)) => point(
            (
                contrast.estimate,
                contrast.standard_error,
                contrast.interval.lower,
                contrast.interval.upper,
            ),
            (wire.estimate, wire.standard_error, wire.lower, wire.upper),
        ),
        _ => false,
    };
    bias_aware && levels && derivatives && contrast
}

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
        let done = match DoseGridFunctional::evaluate_with_context(request, &ctx) {
            Ok(done) => done,
            Err(error) => {
                return match refusal_json(&error, "dose_grid") {
                    Some(json) => Ok((None, None, Some(json))),
                    None => Err(execution_error(&error)),
                };
            }
        };
        let stored = done.request();
        let band_seed = stored.bias_aware.map_or(0, |declared| declared.band_seed);
        let compiled = Study::compile_dose_grid_bias_aware(compiled_request(stored)?)
            .map_err(crate::py_err)?;
        // The band is drawn from its declared seed, as the sealed artifact draws it.
        let mut band_ctx = ctx.clone();
        band_ctx.rng = antecedent::execution::RngFactory::from_seed(band_seed);
        let run = compiled.execute(&band_ctx).map_err(crate::py_err)?;
        if !agrees(&run, done.result()) {
            return Err(crate::serialization_err(
                "dose_grid: the compiled bias-aware program and the sealed row disagree",
            ));
        }
        let mut report: serde_json::Value = serde_json::from_str(&report_json(&done)?)
            .map_err(|e| crate::serialization_err(e.to_string()))?;
        report["program"] = serde_json::json!({
            "route": compiled.plan_view().route().as_str(),
            "lowered_program_identity": run.lowered_program_identity.to_hex(),
            "calibration_estimator": run.calibration_estimator,
        });
        let bytes = done.export().map_err(|e| crate::bridge::serialization(&e))?;
        Ok((Some(report.to_string()), Some(bytes), None))
    })?;
    let (report, bytes, refusal) = payload;
    Ok((report, bytes.map(|b| PyBytes::new(py, &b).unbind()), refusal))
}
