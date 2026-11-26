//! Inspectable compact-model refit plans for Python.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::cohort_decision::{NativeCohortFit, NativeFitRows};
use antecedent::analysis::composition::{BundleError, BundleStage, CompiledCompactModelRefit};
use antecedent::execution::AdmissionRefusal;
use antecedent::prelude::ExecutionContext;
use pyo3::prelude::*;
use serde::Deserialize;

use super::error_refusal;
use crate::PyCancellationToken;
use crate::chain_decision_api::context;

type Checked<T> = (Option<T>, Option<String>);
type RefitResult = (String, Vec<u8>, String);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompactRefitTraining {
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    features: Vec<Vec<f64>>,
    feature_ids: Vec<String>,
}

fn fault(stage: BundleStage, reason: impl ToString, node: Option<&str>) -> BundleError {
    BundleError::Refused { stage, reason: reason.to_string(), offending: node.map(str::to_owned) }
}

pub(super) fn training(
    training_json: &str,
    node: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<NativeFitRows, BundleError> {
    if training_json.len() > 64 * 1024 * 1024 {
        return Err(fault(
            BundleStage::Oversized,
            "compact refit training exceeds the JSON bound",
            node,
        ));
    }
    let _lease = ctx.acquire_admission(0, (training_json.len() as u64).saturating_mul(2)).map_err(
        |error| {
            let admission = AdmissionRefusal::from(error);
            let stage = if admission.is_cancelled() {
                BundleStage::ExecutionCancelled
            } else {
                BundleStage::ResourceRefused
            };
            fault(stage, admission, node)
        },
    )?;
    let rows = serde_json::from_str::<CompactRefitTraining>(training_json)
        .map_err(|error| fault(BundleStage::SwappedEvidence, error, node))?;
    Ok(NativeFitRows {
        outcome: rows.outcome,
        treatment: rows.treatment,
        features: rows.features,
        feature_ids: rows.feature_ids,
    })
}

pub(super) fn artifact(
    fit: &NativeCohortFit,
    node: Option<&str>,
) -> Result<(String, Vec<u8>), BundleError> {
    let export = fit.export();
    export
        .to_bytes("refitted-compact-model")
        .map(|bytes| (export.identity().to_owned(), bytes))
        .map_err(|error| fault(BundleStage::TamperedQuantity, error, node))
}

#[pyclass(name = "CompactModelRefitPlanHandle", skip_from_py_object)]
struct PyCompactModelRefitPlan {
    plan: CompiledCompactModelRefit,
    node: Option<String>,
}

#[pymethods]
impl PyCompactModelRefitPlan {
    #[getter]
    fn program_identity(&self) -> String {
        self.plan.lowered().identity().to_hex()
    }

    fn plan_view_json(&self) -> String {
        self.plan.plan_view().to_json().to_string()
    }

    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.plan.plan_view())
    }

    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Checked<RefitResult>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let result = crate::detach_catch(py, || {
            Ok(self.plan.execute(&ctx).and_then(|run| {
                let (identity, bytes) = artifact(&run.fit, self.node.as_deref())?;
                Ok((identity, bytes, run.lowered_program_identity.to_hex()))
            }))
        })?;
        Ok(match result {
            Ok(receipt) => (Some(receipt), None),
            Err(error) => (None, Some(error_refusal(&error))),
        })
    }
}

#[pyfunction]
#[pyo3(signature = (data, expected_identity, learner, training_json, *, model_node=None, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments)]
fn composition_compile_compact_refit(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    learner: &str,
    training_json: &str,
    model_node: Option<&str>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Checked<PyCompactModelRefitPlan>> {
    let ctx = context(seed, threads, cancel, memory_bytes);
    let result = crate::detach_catch(py, || {
        Ok(training(training_json, model_node, &ctx).and_then(|rows| {
            if let Some(node) = model_node {
                Study::compile_compact_bundle_refit(
                    data,
                    expected_identity,
                    node,
                    rows,
                    learner,
                    &ctx,
                )
            } else {
                Study::compile_compact_model_refit(data, expected_identity, rows, learner, &ctx)
            }
        }))
    })?;
    Ok(match result {
        Ok(plan) => {
            (Some(PyCompactModelRefitPlan { plan, node: model_node.map(str::to_owned) }), None)
        }
        Err(error) => (None, Some(error_refusal(&error))),
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompactModelRefitPlan>()?;
    m.add_function(wrap_pyfunction!(composition_compile_compact_refit, m)?)
}
