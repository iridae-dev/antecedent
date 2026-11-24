//! The compiled measured whole-unit interval of the finite two-step response.
//!
//! `temporal_dependent_interval_compiled` validates the arguments exactly as
//! `temporal_dependent_interval_measured` does, binds them into the
//! `TemporalDependentInterval` program family and executes it under the same context. The family
//! builds the same whole-unit candidate and replays it with the same measured consumer, so the
//! numbers and refusals are identical.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::IntervalEstimand;
use antecedent::Study;
use antecedent::analysis::compiled_temporal_interval::TemporalDependentIntervalRequest;
use antecedent::estimate::temporal_dependent_interval::{DependentIntervalConfig, IntervalMethod};
use pyo3::prelude::*;

use super::{
    LawTuple, Outcome, UnitTuple, produce, spec_of, temporal_dependent_interval_scope, units_of,
};

/// Measured whole-unit interval through the compiled `TemporalDependentInterval` program.
#[pyfunction]
#[pyo3(signature = (snapshot_id, units, sequence, estimand, fixed_state, law, point, replicates, seed, level, method, min_units, max_failed_fraction, *, memory_limit_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
pub(super) fn temporal_dependent_interval_compiled(
    py: Python<'_>,
    snapshot_id: &str,
    units: Option<Vec<UnitTuple>>,
    sequence: (u32, u32),
    estimand: &str,
    fixed_state: Option<u32>,
    law: Option<LawTuple>,
    point: Option<u32>,
    replicates: usize,
    seed: u64,
    level: f64,
    method: &str,
    min_units: usize,
    max_failed_fraction: f64,
    memory_limit_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> Outcome<crate::measured_inference_api::NativeMeasuredInference> {
    crate::bridge::check_count("replicates", replicates as u64, crate::bridge::MAX_RESAMPLES)?;
    let numerical_bytes = units
        .as_ref()
        .map_or(0, |u| u.iter().fold(0_usize, |total, (_, h)| total.saturating_add(h.len())))
        .saturating_mul(128);
    crate::measured_inference_api::before_work(
        0,
        numerical_bytes,
        memory_limit_bytes,
        cancel.as_ref(),
    )?;
    // Preserve complete original-domain validation before scoped inference.
    let refusal = temporal_dependent_interval_scope(
        snapshot_id,
        units.clone(),
        sequence,
        estimand,
        fixed_state,
        law.clone(),
        point,
        replicates,
        seed,
        level,
        method,
        min_units,
        max_failed_fraction,
    )?;
    if let Some(refusal) = refusal {
        return Ok((None, Some(refusal)));
    }
    let method = match method {
        "studentized" => IntervalMethod::Studentized,
        _ => return Err(crate::value_err("invalid interval method")),
    };
    let estimand = match (estimand, fixed_state) {
        ("observed_initial_state", None) => IntervalEstimand::Observed,
        ("fixed_initial_state", Some(s0)) => IntervalEstimand::FixedState(s0),
        ("marginalized_initial_state", None) => IntervalEstimand::Marginalized(
            spec_of(law, point).map_err(|e| crate::value_err(e.to_string()))?,
        ),
        _ => return Err(crate::value_err("invalid interval estimand")),
    };
    let config =
        DependentIntervalConfig { replicates, seed, level, method, min_units, max_failed_fraction };
    let ctx = crate::measured_inference_api::context(seed, memory_limit_bytes, cancel);
    let request = TemporalDependentIntervalRequest::new(
        snapshot_id,
        units_of(units),
        [sequence.0, sequence.1],
        estimand,
        config,
    );
    crate::detach_catch(py, move || {
        let compiled = Study::compile_temporal_dependent_interval(request)
            .map_err(|error| crate::value_err(error.to_string()))?;
        let run = match compiled.execute(&ctx) {
            Ok(run) => run,
            Err(cause) => return produce(Err(cause)),
        };
        crate::measured_inference_api::check_cancelled(&ctx)?;
        Ok((
            Some(crate::measured_inference_api::NativeMeasuredInference::from_io(run.measured)?),
            None,
        ))
    })
}
