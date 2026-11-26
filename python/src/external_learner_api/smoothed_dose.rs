//! Python learners on the compiled smoothed dose-response transport.
//!
//! [`smoothed_dose_learner_transport`] declares two registered Python providers as the
//! outcome-curve and source-membership learners of a smoothed dose-response transport and runs
//! it as the compiled `SmoothedDoseLearnerTransport` program
//! (`Study::compile_smoothed_dose_learners`): the transport is decided on the declared selection
//! diagram and every fold is fitted by the caller's providers. The route evaluates the grid from
//! portable fold models, so each provider fit returns its model as a portable prediction map
//! (the provider's model artifact, decoded by `antecedent::analysis::portable_learner_fit`); the
//! fit receipt of every fold is returned.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, atomic::Ordering};

use antecedent::Study;
use antecedent::analysis::compiled_smoothed_dose_learners::SmoothedDoseLearnersRequest;
use antecedent::analysis::declared_learners::LearnerFactories;
use antecedent::analysis::portable_learner_fit::portable_fit;
use antecedent::estimate::SmoothedDoseOptions;
use antecedent::estimate::external_learner::{
    DesignView, FittedPredictor, LearnError, LearnerCapabilities, LearnerFactory,
    LearnerProvenance, PredictionTask, TargetView,
};
use antecedent::prelude::ExecutionContext;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::{
    CallbackState, PythonReceipt, callback_bytes, callback_context, fit_columns, learner_weights,
};

/// A registered Python provider whose fits are portable prediction maps.
struct PortablePythonFactory {
    learner: Py<PyAny>,
    task: PredictionTask,
    weights: bool,
    state: Arc<CallbackState>,
}

impl LearnerFactory for PortablePythonFactory {
    fn task(&self) -> PredictionTask {
        self.task
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: self.task == PredictionTask::Regression,
            binary_probability: self.task == PredictionTask::BinaryProbability,
            sample_weights: self.weights,
            ..LearnerCapabilities::none()
        }
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        let bytes = callback_bytes(x, 1 + usize::from(weights.is_some()))?;
        let _hold = ctx.acquire_admission(0, bytes).map_err(LearnError::Admission)?;
        let id = self.state.next.fetch_add(1, Ordering::Relaxed);
        let map =
            Python::attach(|py| -> PyResult<String> {
                let fitted = fit_columns(py, self.learner.bind(py), x, y, weights, bytes)?;
                let map: String =
                    fitted.getattr("model")?.call_method1("decode", ("utf-8",))?.extract()?;
                let receipt = fitted.getattr("fit_receipt")?.unbind();
                self.state.receipts.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(
                    (id, receipt, pyo3::types::PyNone::get(py).to_owned().into_any().unbind()),
                );
                Ok(map)
            })
            .map_err(|e| self.state.fail(e))?;
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        portable_fit(
            &map,
            x.ncols(),
            LearnerProvenance {
                spec: "external_provider".into(),
                implementation: format!("python_receipt:{id}"),
                version: "1".into(),
            },
        )
    }
}

/// Outcome and treatment factories whose fits are portable prediction maps, plus the
/// receipt log those fits append to.
pub(super) fn portable_factories(
    py: Python<'_>,
    outcome: Py<PyAny>,
    treatment: Py<PyAny>,
    identity: String,
) -> PyResult<(LearnerFactories, Arc<CallbackState>)> {
    let outcome_weights = learner_weights(py, &outcome, "regression")?;
    let treatment_weights = learner_weights(py, &treatment, "binary_probability")?;
    let portable =
        |obj: &Py<PyAny>| -> PyResult<bool> { obj.bind(py).getattr("portable")?.extract() };
    let declared = portable(&outcome)? && portable(&treatment)?;
    let state = Arc::new(CallbackState::default());
    let learners = LearnerFactories::new(
        Arc::new(PortablePythonFactory {
            learner: outcome,
            task: PredictionTask::Regression,
            weights: outcome_weights,
            state: Arc::clone(&state),
        }),
        Arc::new(PortablePythonFactory {
            learner: treatment,
            task: PredictionTask::BinaryProbability,
            weights: treatment_weights,
            state: Arc::clone(&state),
        }),
        identity,
    )
    .map_err(crate::py_err)?
    .declared_portability(declared);
    Ok((learners, state))
}

/// What `smoothed_dose_learner_transport` returns: the compiled run as JSON and one
/// `(fit id, fit receipt, None)` per provider fit.
type SmoothedDoseLearnersPayload = (String, Vec<PythonReceipt>);

/// Run the compiled smoothed dose-response transport with two registered Python providers as
/// its learners under `identity`.
///
/// `graph`, `selections`, `query`, `data` and `options` are what `prepare_smoothed_dose` takes;
/// `outcome` is a `regression` learner and `membership` a `binary_probability` learner. The JSON
/// holds the kernel's estimate (without its fold models), the route, the lowered program
/// identity, the claim and the calibration status.
#[pyfunction]
#[pyo3(signature = (graph, selections, query, data, options, outcome, membership, *, identity, seed=1, cancel=None, memory_limit_bytes=None))]
#[allow(clippy::too_many_arguments)]
pub(super) fn smoothed_dose_learner_transport(
    py: Python<'_>,
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    query: &str,
    data: &Bound<'_, PyAny>,
    options: &str,
    outcome: Py<PyAny>,
    membership: Py<PyAny>,
    identity: String,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
    memory_limit_bytes: Option<u64>,
) -> PyResult<SmoothedDoseLearnersPayload> {
    let (learners, state) = portable_factories(py, outcome, membership, identity)?;
    let (diagram, query) = crate::smoothed_dose_api::build(graph, &selections, query)?;
    let input = crate::smoothed_dose_api::parse_input(data, &graph.names)?;
    let options: SmoothedDoseOptions =
        serde_json::from_str(options).map_err(|e| crate::value_err(e.to_string()))?;
    let names = graph.names.clone();
    let ctx = callback_context(seed, cancel, memory_limit_bytes)?;
    let result: PyResult<String> = py.detach(|| {
        let compiled = Study::compile_smoothed_dose_learners(
            SmoothedDoseLearnersRequest::new(diagram, query, input, options, learners),
            &ctx,
        )
        .map_err(crate::transport_common::error)?;
        let run = compiled.execute(&ctx).map_err(crate::transport_common::error)?;
        let mut estimate = serde_json::to_value(&run.estimate)
            .map_err(|e| crate::serialization_err(e.to_string()))?;
        // The fold models and per-row scores stay native; the grid, diagnostics, overlap,
        // provenance and folds travel.
        if let Some(map) = estimate.as_object_mut() {
            map.remove("models");
        }
        if let Some(grid) = estimate.get_mut("grid").and_then(serde_json::Value::as_array_mut) {
            for point in grid.iter_mut().filter_map(serde_json::Value::as_object_mut) {
                point.remove("row_scores");
            }
        }
        Ok(serde_json::json!({
            "estimate": estimate,
            "route": compiled.plan_view().route().as_str(),
            "program_identity": run.lowered_program_identity.to_hex(),
            "claim": run.claim,
            "calibration": run.calibration,
            "variable_names": names,
        })
        .to_string())
    });
    state.finish(result)
}
