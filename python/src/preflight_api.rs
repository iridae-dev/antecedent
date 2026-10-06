//! Python bindings for the E1 preflight, rank-drop and cost surface.
//!
//! Reports cross the boundary as JSON text (the Rust report types are `Serialize`) and the
//! Python wrapper parses them into frozen dataclasses, so the two languages share one shape.
//! The fit-free report ([`antecedent::PreflightReport`]) and the fitted diagnostics
//! ([`antecedent::NuisanceFitDiagnostics`]) are separate calls, as they are in Rust.

use std::sync::Arc;

use antecedent::{
    CausalError, ColumnPriority, EstimatorId, PreflightInput, RankDropPolicy,
    estimate_with_rank_drop, fit_diagnostics_design, plan_rank_drop, preflight_design,
};
use antecedent_core::{CausalSchema, VariableId};
use antecedent_data::TableView;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;

use crate::ate_api::PyPreparedBatch;
use crate::prepared_api::PyPreparedAnalysis;
use crate::{
    detach_catch, py_err, py_execution_context, resolve_user_threads, tabular_from_py_columns,
};

fn to_json<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Resolve a declared priority (column names, highest first) against the frozen schema.
fn policy_for(schema: &CausalSchema, priority: Option<Vec<String>>) -> PyResult<RankDropPolicy> {
    let priority = match priority {
        None => ColumnPriority::AdjustmentOrder,
        Some(names) => ColumnPriority::Declared(
            names
                .iter()
                .map(|name| schema.id_of(name))
                .collect::<Result<Vec<VariableId>, _>>()
                .map_err(|e| py_err(CausalError::from(e)))?,
        ),
    };
    Ok(RankDropPolicy { priority })
}

/// Structured refusal fields as JSON (`null` and empty lists for absent entries).
pub(crate) fn refusal_fields_value(
    fields: &antecedent_estimate::RefusalFields,
) -> serde_json::Value {
    serde_json::json!({
        "stage": fields.stage,
        "subject": fields.subject,
        "reason": fields.reason,
        "arm_ess": fields.arm_ess.iter().map(|(l, e)| (l, e.0)).collect::<Vec<_>>(),
        "propensity_min": fields.propensity_min.map(|v| v.0),
        "propensity_max": fields.propensity_max.map(|v| v.0),
        "propensity_quantiles": fields
            .propensity_quantiles
            .iter()
            .map(|(p, v)| (p.0, v.0))
            .collect::<Vec<_>>(),
        "cluster_count": fields.cluster_count,
        "cluster_minimum": fields.cluster_minimum,
        "numerical_rank": fields.numerical_rank,
        "design_columns": fields.design_columns,
        "implicated_columns": fields.implicated_columns,
        "remedy": fields.remedy,
        "glm_iterations": fields.glm_iterations,
        "boundary_margin": fields.boundary_margin.map(|v| v.0),
        "boundary_count": fields.boundary_count,
    })
}

/// Structured refusal fields as a Python dict (`None` and empty lists for absent entries).
pub(crate) fn refusal_fields_dict<'py>(
    py: Python<'py>,
    fields: &antecedent_estimate::RefusalFields,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?.call_method1("loads", (refusal_fields_value(fields).to_string(),))
}

/// Attach a refusal's structured fields (`refusal_fields`) to the raised exception.
///
/// Additive: set only when the refusal carries them, so every other exception reads the
/// `refusal_fields = None` class default registered with the errors.
pub(crate) fn with_refusal_fields(
    err: PyErr,
    fields: Option<&antecedent_estimate::RefusalFields>,
) -> PyErr {
    if let Some(fields) = fields {
        Python::attach(|py| {
            if let Ok(dict) = refusal_fields_dict(py, fields) {
                let _ = err.value(py).setattr("refusal_fields", dict);
            }
        });
    }
    err
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Fit-free preflight report of this plan, as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_json(&self, py: Python<'_>, seed: u64, threads: Option<u32>) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose(&ctx).map_err(py_err)?)
        })
    }

    /// Propensity-fit diagnostics (fit-requiring), as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_fit_json(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose_fit(&ctx).map_err(py_err)?)
        })
    }

    /// Opt-in rank-drop plan under a declared priority (`None` = adjustment-set order).
    #[pyo3(signature = (priority=None, *, seed=1, threads=None))]
    fn plan_rank_drop_json(
        &self,
        py: Python<'_>,
        priority: Option<Vec<String>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        let policy = policy_for(inner.schema(), priority)?;
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.plan_rank_drop(&policy, &ctx).map_err(py_err)?)
        })
    }

    /// Planning-time cost counts, as JSON.
    fn estimate_cost_json(&self) -> PyResult<String> {
        to_json(&self.inner.estimate_cost().map_err(py_err)?)
    }
}

#[pymethods]
impl PyPreparedBatch {
    /// Fit-free preflight report of every plan, as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_json(&self, py: Python<'_>, seed: u64, threads: Option<u32>) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose(&ctx).map_err(py_err)?)
        })
    }

    /// Propensity-fit diagnostics of every plan (fit-requiring), as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_fit_json(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose_fit(&ctx).map_err(py_err)?)
        })
    }

    /// Opt-in rank-drop plans under a declared priority (`None` = adjustment-set order).
    #[pyo3(signature = (priority=None, *, seed=1, threads=None))]
    fn plan_rank_drop_json(
        &self,
        py: Python<'_>,
        priority: Option<Vec<String>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        let policy = {
            let first = inner
                .plans()
                .first()
                .ok_or_else(|| PyValueError::new_err("a prepared batch holds no plans"))?;
            policy_for(first.schema(), priority)?
        };
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.plan_rank_drop(&policy, &ctx).map_err(py_err)?)
        })
    }

    /// Planning-time cost counts of every plan and their totals, as JSON.
    fn estimate_cost_json(&self) -> PyResult<String> {
        to_json(&self.inner.estimate_cost().map_err(py_err)?)
    }
}

/// A table and the resolved columns of a binary-effect design.
struct BoundDesign {
    data: antecedent_data::TabularData,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: Vec<VariableId>,
}

impl BoundDesign {
    fn bind(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: &str,
        outcome: &str,
        adjustment: &[String],
    ) -> PyResult<Self> {
        let (data, _) = tabular_from_py_columns(py, names, columns)?;
        let lookup =
            |name: &str| data.schema().id_of(name).map_err(|e| py_err(CausalError::from(e)));
        let treatment = lookup(treatment)?;
        let outcome = lookup(outcome)?;
        let adjustment = adjustment.iter().map(|n| lookup(n)).collect::<PyResult<Vec<_>>>()?;
        Ok(Self { data, treatment, outcome, adjustment })
    }

    fn input(&self, control: f64, active: f64) -> PreflightInput<'_> {
        PreflightInput::binary_effect(
            &self.data,
            self.treatment,
            self.outcome,
            &self.adjustment,
            control,
            active,
        )
    }
}

/// Fit-free preflight of a declared binary-effect design, before any study is prepared (a
/// design a fit would refuse cannot be prepared, so this is the entry point for it).
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn preflight_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(&preflight_design(&design.input(control, active), &ctx).map_err(py_err)?)
    })
}

/// Propensity-fit diagnostics of a declared binary-effect design (fit-requiring).
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn fit_diagnostics_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(&fit_diagnostics_design(&design.input(control, active), &ctx).map_err(py_err)?)
    })
}

/// Rank-drop plan of a declared binary-effect design under a declared priority.
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, priority=None, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn rank_drop_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    priority: Option<Vec<String>>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    let policy = policy_for(design.data.schema(), priority)?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(&plan_rank_drop(&design.input(control, active), &policy, &ctx).map_err(py_err)?)
    })
}

/// Estimate a declared binary-effect design on the span-preserving reduced adjustment set of
/// a rank-drop plan (`estimator` is a licensed estimator id, e.g. `aipw`).
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, estimator, priority=None, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn estimate_with_rank_drop_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    estimator: &str,
    priority: Option<Vec<String>>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    let policy = policy_for(design.data.schema(), priority)?;
    let estimator =
        estimator.parse::<EstimatorId>().map_err(|e| PyValueError::new_err(e.to_string()))?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(
            &estimate_with_rank_drop(&design.input(control, active), &policy, estimator, &ctx)
                .map_err(py_err)?,
        )
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preflight_json, module)?)?;
    module.add_function(wrap_pyfunction!(fit_diagnostics_json, module)?)?;
    module.add_function(wrap_pyfunction!(rank_drop_json, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_with_rank_drop_json, module)?)?;
    Ok(())
}
