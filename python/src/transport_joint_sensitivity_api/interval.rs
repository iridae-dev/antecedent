//! Separately versioned endpoint interval artifact bindings.
use super::{PreparedZTransportStage, build_spec, execution_context, joint_py_err, to_py_json};
use pyo3::prelude::*;
use serde_json::json;

/// Export a separately versioned interval, preserving the existing range contract.
#[pyfunction]
#[pyo3(signature=(stage, factors, *, seed, replicates=199, level=0.95, decision_threshold=None, total_budget=None, tolerance=1e-9, frontier_points=17, max_operations=100_000, max_depth=64, max_memory_bytes=67_108_864, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one bounded interval artifact request")]
fn export_z_transport_joint_interval(
    py: Python<'_>,
    stage: PyRef<'_, PreparedZTransportStage>,
    factors: Vec<(String, f64)>,
    seed: u64,
    replicates: u32,
    level: f64,
    decision_threshold: Option<f64>,
    total_budget: Option<f64>,
    tolerance: f64,
    frontier_points: usize,
    max_operations: usize,
    max_depth: usize,
    max_memory_bytes: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    let spec = build_spec(
        factors,
        decision_threshold,
        total_budget,
        tolerance,
        frontier_points,
        max_operations,
        max_depth,
        max_memory_bytes,
    )?;
    let baseline = stage.raw_export("estimate before exporting an interval artifact")?;
    let ctx = stage.ctx(memory_bytes, cancel);
    drop(stage);
    let bytes = crate::detach_catch(py, move || {
        antecedent::JointSensitivityIntervalArtifact::checked(
            baseline, &spec, seed, replicates, level, &ctx,
        )
        .and_then(|wire| wire.export())
        .map_err(joint_py_err)
    })?;
    Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
}

/// Replay the numerical interval and independently resolve current calibration.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_replicates=2000, max_bytes=67_108_864, max_search_operations=100_000, max_search_depth=64, max_memory_bytes=67_108_864, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one bounded interval consumer request")]
fn consume_z_transport_joint_interval(
    py: Python<'_>,
    artifact: &[u8],
    max_replicates: u32,
    max_bytes: usize,
    max_search_operations: usize,
    max_search_depth: usize,
    max_memory_bytes: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Py<PyAny>> {
    let bytes = artifact.to_vec();
    let consumed = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        antecedent::JointSensitivityIntervalArtifact::consume_with_limits(
            &bytes,
            antecedent::JointIntervalConsumeLimits {
                max_replicates,
                max_bytes,
                range: antecedent::JointSensitivityConsumeLimits {
                    max_operations: max_search_operations,
                    max_depth: max_search_depth,
                    max_memory_bytes,
                    ..Default::default()
                },
                ..Default::default()
            },
            &ctx,
        )
        .map_err(joint_py_err)
    })?;
    let value = json!({"version":consumed.artifact.version,"digest":consumed.artifact.digest,
        "procedure":consumed.artifact.procedure,"seed":consumed.artifact.seed,
        "deviation":consumed.artifact.range.body.perturbation,"numerical":consumed.artifact.numerical,
        "recorded_calibration":consumed.artifact.recorded_calibration,"calibration":consumed.calibration});
    to_py_json(py, &value)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_z_transport_joint_interval, module)?)?;
    module.add_function(wrap_pyfunction!(consume_z_transport_joint_interval, module)?)?;
    Ok(())
}
