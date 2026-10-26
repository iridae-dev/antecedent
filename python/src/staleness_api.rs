//! Python bindings for staleness and drift diagnostics.
//!
//! Python builds the policy as a JSON object and reads the report as JSON; every statistic,
//! threshold, multiplicity rule, recommendation and identity is computed in Rust
//! ([`antecedent::analysis::staleness`]). A diagnostic that could not run raises the ordinary
//! causal error; data the plan cannot be applied to is a `refuse` report, not an exception.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::staleness::{
    StalenessPolicy, assess_artifact_staleness, staleness_policy_from_json, staleness_policy_json,
    staleness_report_json,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::prepared_api::PyPreparedAnalysis;
use crate::{
    detach_catch, py_err, py_execution_context, resolve_user_threads, tabular_from_py_columns,
};

fn policy_of(policy_json: Option<&str>) -> PyResult<StalenessPolicy> {
    policy_json.map_or_else(
        || Ok(StalenessPolicy::default()),
        |text| staleness_policy_from_json(text).map_err(py_err),
    )
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Compare new columns with the data this study retains; the report as JSON.
    #[pyo3(signature = (names, columns, *, policy_json=None, seed=1, threads=None))]
    fn staleness(
        &self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        policy_json: Option<String>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let policy = policy_of(policy_json.as_deref())?;
        let (data, _) = tabular_from_py_columns(names, columns)?;
        let study = std::sync::Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let report = study.staleness(&data, &policy, &ctx).map_err(py_err)?;
            Ok(staleness_report_json(&report).to_string())
        })
    }

    /// The sealed staleness baseline of the data this study retains.
    #[pyo3(signature = (*, policy_json=None, seed=1, threads=None))]
    fn export_staleness_baseline<'py>(
        &self,
        py: Python<'py>,
        policy_json: Option<String>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let policy = policy_of(policy_json.as_deref())?;
        let study = std::sync::Arc::clone(&self.inner);
        let bytes = detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let baseline = study.staleness_baseline(&policy, &ctx).map_err(py_err)?;
            baseline
                .export()
                .map_err(|error| crate::value_err(format!("staleness baseline: {error}")))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// The roles a staleness baseline of this study records, as JSON.
    fn staleness_plan_json(&self) -> String {
        let plan = self.inner.staleness_plan();
        serde_json::json!({
            "plan_identity": plan.plan_identity,
            "treatment": plan.treatment,
            "outcome": plan.outcome,
            "adjustment": plan.adjustment,
            "bayesian": plan.bayesian,
        })
        .to_string()
    }
}

/// Compare new columns with an exported staleness baseline; the report as JSON.
#[pyfunction]
#[pyo3(signature = (artifact, names, columns, *, policy_json=None, seed=1, threads=None))]
fn staleness_from_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    policy_json: Option<String>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let policy = policy_of(policy_json.as_deref())?;
    let (data, _) = tabular_from_py_columns(names, columns)?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        let report = assess_artifact_staleness(&artifact, &data, &policy, &ctx).map_err(py_err)?;
        Ok(staleness_report_json(&report).to_string())
    })
}

/// Validate a policy object and return it complete (every field and its identity) as JSON.
#[pyfunction]
#[pyo3(signature = (policy_json=None))]
fn staleness_policy(policy_json: Option<String>) -> PyResult<String> {
    Ok(staleness_policy_json(&policy_of(policy_json.as_deref())?).to_string())
}

/// Independently verify a baseline and expose bounded metadata, never its profile arrays.
#[pyfunction]
#[pyo3(signature = (artifact, *, expected_identity=None))]
fn consume_staleness_baseline(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: Option<String>,
) -> PyResult<String> {
    detach_catch(py, move || {
        let ctx = py_execution_context(1, resolve_user_threads(None));
        crate::measured_inference_api::check_cancelled(&ctx)?;
        let baseline =
            antecedent_io_core::staleness_artifact::StalenessBaselineWire::consume_admitted(
                artifact, &ctx,
            )
            .map_err(|error| match error {
                antecedent_io_core::IoError::StreamCancelled => {
                    py_err(antecedent::CausalError::Cancelled {
                        stage: "consume_staleness_baseline",
                    })
                }
                antecedent_io_core::IoError::StreamBudgetRefused { message } => {
                    crate::CausalResourceError::new_err(message)
                }
                other => crate::bridge::serialization(other),
            })?;
        // Keep the admitted owner through verification, expected-identity checking
        // and all reads. No decoded profile can escape without its lease.
        let wire = baseline.wire();
        crate::measured_inference_api::check_cancelled(&ctx)?;
        if expected_identity.as_ref().is_some_and(|expected| expected != &wire.seal) {
            return Err(crate::value_err("staleness baseline identity differs from expectation"));
        }
        Ok(serde_json::json!({
            "identity": wire.seal,
            "plan_identity": wire.plan_identity,
            "snapshot": wire.snapshot,
            "policy_identity": wire.policy,
            "rows": wire.rows,
            "complete_rows": wire.complete_rows,
            "columns": wire.columns.len(),
            "version": wire.version,
        })
        .to_string())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(consume_staleness_baseline, module)?)?;
    module.add_function(wrap_pyfunction!(staleness_from_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(staleness_policy, module)?)?;
    Ok(())
}
