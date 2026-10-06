//! Python bindings for 2.2B X3 joint mechanism deviations of the registered
//! surrogate z formula. Every function here backs one routed wrapper in
//! `antecedent.transport._joint_sensitivity` and is reached only through it.
use antecedent::{JointSensitivityArtifactWire, JointSensitivityConsumeLimits};
use antecedent_core::reason_code;
use antecedent_io::IoError;
use antecedent_io::z_transport_artifact::ZTransportConsumeLimits;
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityLimits, TippingBracket,
};
use pyo3::prelude::*;
use serde_json::json;

use crate::transport_common::execution_context;
use crate::transport_z_api::{PreparedZTransportStage, to_py_json};

/// A reason-coded refusal as the Python exception a caller sees, always
/// carrying `reason_code`.
fn joint_py_err(error: IoError) -> PyErr {
    match error {
        IoError::Refused { code, message } => {
            let base = if code == reason_code!("transport_budget_cancel") {
                if message.contains("cancel") {
                    crate::CausalCancelledError::new_err(message)
                } else {
                    crate::CausalResourceError::new_err(message)
                }
            } else if code == reason_code!("invalid_argument") {
                crate::value_err(message)
            } else {
                return crate::refusal(code, message);
            };
            crate::with_reason_code(base, code)
        }
        other => crate::transport_common::error(other),
    }
}

/// Parse the declared factors; a name outside the vocabulary refuses.
fn parse_factors(factors: Vec<(String, f64)>) -> PyResult<Vec<JointFactorBound>> {
    factors
        .into_iter()
        .map(|(name, max_fraction)| {
            JointFactor::from_name(&name)
                .map(|factor| JointFactorBound { factor, max_fraction })
                .ok_or_else(|| {
                    joint_py_err(IoError::Refused {
                        code: reason_code!("invalid_argument"),
                        message: format!("joint_sensitivity.unknown_factor: {name:?}"),
                    })
                })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_spec(
    factors: Vec<(String, f64)>,
    decision_threshold: Option<f64>,
    total_budget: Option<f64>,
    tolerance: f64,
    frontier_points: usize,
    max_operations: usize,
    max_depth: usize,
    max_memory_bytes: u64,
) -> PyResult<JointDeviationSpec> {
    Ok(JointDeviationSpec {
        factors: parse_factors(factors)?,
        total_budget,
        decision_threshold,
        tolerance,
        frontier_points,
        limits: JointSensitivityLimits {
            operations: max_operations,
            depth: max_depth,
            memory_bytes: max_memory_bytes,
        },
    })
}

fn bracket_json(bracket: Option<TippingBracket>) -> serde_json::Value {
    bracket.map_or(
        serde_json::Value::Null,
        |b| json!({"lower": b.lower, "upper": b.upper, "iterations": b.iterations}),
    )
}

fn result_json(result: &JointMechanismSensitivityResult) -> serde_json::Value {
    json!({
        "status": "available",
        "estimand": "target active-minus-control mean outcome response",
        "inference_claim": result.inference_claim,
        "interpretation": result.interpretation,
        "baseline": result.baseline,
        "assumption_range": {"minimum": result.range.minimum, "maximum": result.range.maximum},
        "minimizing_outcome_by_stratum": result.range.minimizing_outcome_by_stratum,
        "maximizing_outcome_by_stratum": result.range.maximizing_outcome_by_stratum,
        "minimizing_parent_level": result.range.minimizing_parent_level,
        "maximizing_parent_level": result.range.maximizing_parent_level,
        "factors": result.factors.iter().map(|b| json!({"factor": b.factor.name(), "max_fraction": b.max_fraction})).collect::<Vec<_>>(),
        "decision_threshold": result.decision_threshold,
        "axis_tipping": result.axis_tipping.iter().map(|a| json!({
            "factor": a.factor.name(), "analytic": a.analytic, "status": a.status.name(), "bracket": bracket_json(a.bracket),
        })).collect::<Vec<_>>(),
        "frontier": result.frontier.iter().map(|p| json!({
            "parent_fraction": p.parent_fraction, "status": p.status.name(), "bracket": bracket_json(p.bracket),
        })).collect::<Vec<_>>(),
        "unresolved_detail": result.unresolved_detail,
        "receipt": {
            "method": result.receipt.method,
            "fraction_box": result.receipt.fraction_box,
            "tolerance": result.receipt.tolerance,
            "operations_limit": result.receipt.limits.operations,
            "depth_limit": result.receipt.limits.depth,
            "memory_limit_bytes": result.receipt.memory_limit_bytes,
            "operations_consumed": result.receipt.operations_consumed,
            "depth_reached": result.receipt.depth_reached,
            "live_state_bytes": result.receipt.live_state_bytes,
            "stop": result.receipt.stop.map(antecedent_core::SearchStop::code),
            "explored": result.receipt.explored,
            "unevaluated": result.receipt.unevaluated,
        },
        "uncertainty": {
            "status": "withheld",
            "method": result.uncertainty.method,
            "coverage_target": result.uncertainty.coverage_target,
            "reason_code": result.uncertainty.reason_code,
            "detail": result.uncertainty.detail,
        },
        "baseline_binding": {
            "query": result.query_binding,
            "source_regime": result.source_regime.raw(),
            "provider_snapshot": result.provider_snapshot,
        },
    })
}

/// Evaluate the exact joint assumption range on the stage's retained laws.
#[pyfunction]
#[pyo3(signature=(stage, factors, *, decision_threshold=None, total_budget=None, tolerance=1e-9, frontier_points=17, max_operations=100_000, max_depth=64, max_memory_bytes=67_108_864, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
fn z_transport_joint_sensitivity(
    py: Python<'_>,
    stage: PyRef<'_, PreparedZTransportStage>,
    factors: Vec<(String, f64)>,
    decision_threshold: Option<f64>,
    total_budget: Option<f64>,
    tolerance: f64,
    frontier_points: usize,
    max_operations: usize,
    max_depth: usize,
    max_memory_bytes: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Py<PyAny>> {
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
    let ctx = stage.ctx(memory_bytes, cancel);
    let prepared = stage.prepared().clone();
    drop(stage);
    let result = crate::detach_catch(py, move || {
        prepared.joint_mechanism_sensitivity(&spec, &ctx).map_err(joint_py_err)
    })?;
    to_py_json(py, &result_json(&result))
}

/// Export the last execution's baseline with a checked joint analysis as a
/// version 3 artifact.
#[pyfunction]
#[pyo3(signature=(stage, factors, *, decision_threshold=None, total_budget=None, tolerance=1e-9, frontier_points=17, max_operations=100_000, max_depth=64, max_memory_bytes=67_108_864, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
fn export_z_transport_joint_sensitivity(
    py: Python<'_>,
    stage: PyRef<'_, PreparedZTransportStage>,
    factors: Vec<(String, f64)>,
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
    let baseline = stage.raw_export("estimate before exporting a joint sensitivity artifact")?;
    let ctx = stage.ctx(memory_bytes, cancel);
    drop(stage);
    let bytes = crate::detach_catch(py, move || {
        JointSensitivityArtifactWire::checked(baseline, &spec, &ctx)
            .and_then(|wire| wire.export())
            .map_err(joint_py_err)
    })?;
    Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
}

/// Independently verify a version 3 artifact under the consumer's limits and
/// return its replayed body with both digests.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_operations=10_000_000, max_depth=256, max_support_rows=None, max_laws=None, max_law_cells=None, max_search_operations=100_000, max_search_depth=64, max_memory_bytes=67_108_864, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
fn consume_z_transport_joint_sensitivity_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_operations: usize,
    max_depth: usize,
    max_support_rows: Option<usize>,
    max_laws: Option<usize>,
    max_law_cells: Option<usize>,
    max_search_operations: usize,
    max_search_depth: usize,
    max_memory_bytes: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Py<PyAny>> {
    let bytes = artifact.to_vec();
    let defaults = ZTransportConsumeLimits::default();
    let limits = JointSensitivityConsumeLimits {
        baseline: ZTransportConsumeLimits {
            evaluation: antecedent_expr::ExactEvaluationLimits {
                operations: max_operations,
                depth: max_depth,
            },
            max_support_rows: max_support_rows.unwrap_or(defaults.max_support_rows),
            max_laws: max_laws.unwrap_or(defaults.max_laws),
            max_law_cells: max_law_cells.unwrap_or(defaults.max_law_cells),
        },
        max_operations: max_search_operations,
        max_depth: max_search_depth,
        max_memory_bytes,
    };
    let wire = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        JointSensitivityArtifactWire::consume_with_limits(&bytes, limits, &ctx)
            .map_err(joint_py_err)
    })?;
    let mut value =
        serde_json::to_value(&wire.body).map_err(crate::transport_common::serialization_error)?;
    if let Some(map) = value.as_object_mut() {
        map.insert("version".into(), json!(wire.version));
        map.insert("premises_digest".into(), json!(wire.premises_digest));
        map.insert("data_digest".into(), json!(wire.data_digest));
        map.insert("status".into(), json!("available"));
    }
    to_py_json(py, &value)
}

/// The closed sampling-uncertainty interval route: always refuses.
#[pyfunction]
#[pyo3(signature=(stage, factors, *, decision_threshold=None, memory_bytes=None, cancel=None))]
#[doc(hidden)]
fn z_transport_joint_sensitivity_interval(
    stage: PyRef<'_, PreparedZTransportStage>,
    factors: Vec<(String, f64)>,
    decision_threshold: Option<f64>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<()> {
    let mut spec = JointDeviationSpec::new(parse_factors(factors)?);
    spec.decision_threshold = decision_threshold;
    let ctx = stage.ctx(memory_bytes, cancel);
    match stage.prepared().joint_mechanism_sensitivity_interval(&spec, &ctx) {
        Ok(never) => match never {},
        Err(error) => Err(joint_py_err(error)),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(z_transport_joint_sensitivity, module)?)?;
    module.add_function(wrap_pyfunction!(export_z_transport_joint_sensitivity, module)?)?;
    module
        .add_function(wrap_pyfunction!(consume_z_transport_joint_sensitivity_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(z_transport_joint_sensitivity_interval, module)?)?;
    Ok(())
}
