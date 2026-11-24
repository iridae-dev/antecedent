//! Process-local cyclic callback execution.
//!
//! Each Python mechanism is a declared batch callback that provides its derivatives; the system
//! runs through the compiled `LocalCyclicCallback` program
//! (`antecedent::analysis::compiled_local_cyclic::compile_local_cyclic`), which binds the
//! declarations, gates the run on the `cyclic.general_systems` unit and executes the
//! `LocalCyclicModel` Newton solve and abduction unchanged.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::compiled_local_cyclic::{
    DeclaredCyclicMechanism, LocalCyclicCompileError, LocalCyclicRun, compile_local_cyclic,
};
use antecedent::execution::{
    BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec,
};

use super::{
    Bound, MAX_CYCLIC_ARTIFACT_BYTES, Py, PyAny, PyAnyMethods, PyModule, PyModuleMethods, PyResult,
    Python, execution_context, input_admission, operation_error, pyfunction, wrap_pyfunction,
};

/// A Python mechanism `f(inputs) -> (value, derivatives)` as a declared runtime callback.
struct PythonCyclicCallback {
    spec: CallbackSpec,
    callback: Py<PyAny>,
}

impl std::fmt::Debug for PythonCyclicCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonCyclicCallback")
            .field("name", &self.spec.name)
            .finish_non_exhaustive()
    }
}

impl PythonCyclicCallback {
    fn new(id: &str, reads: &[usize], callback: Py<PyAny>) -> Self {
        let spec = CallbackSpec {
            name: id.to_owned(),
            inputs: reads.iter().map(|read| format!("x{read}")).collect::<Vec<_>>().into(),
            output: CallbackOutput::Real,
            dependencies: Arc::from([]),
            implementation: CallbackImplementation::Runtime,
        };
        Self { spec, callback }
    }
}

impl BatchCallback for PythonCyclicCallback {
    fn spec(&self) -> &CallbackSpec {
        &self.spec
    }

    fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
        for (row, slot) in out.iter_mut().enumerate() {
            let point: Vec<f64> = inputs.iter().map(|column| column[row]).collect();
            *slot = self.value_and_derivatives(&point)?.0;
        }
        Ok(())
    }

    fn value_and_derivatives(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), CallbackError> {
        Python::attach(|py| {
            self.callback.bind(py).call1((inputs.to_vec(),))?.extract::<(f64, Vec<f64>)>()
        })
        .map_err(|error| {
            CallbackError::Failed(crate::interrupt::callback_failure(
                "cyclic callback raised",
                error,
            ))
        })
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalRequest {
    mechanisms: Vec<(String, Vec<usize>)>,
    noise: Vec<f64>,
    clamps: Vec<antecedent::gcm::nonlinear_equilibrium::Clamp>,
    initial: Vec<f64>,
    factual: Option<Vec<f64>>,
    policy: antecedent::gcm::local_cyclic::LocalCyclicPolicy,
}
/// Process-local execution retains no executable artifact representation.
#[pyfunction]
#[pyo3(signature = (request_json, callbacks, *, max_bytes=None, cancel=None))]
fn execute_local_cyclic(
    py: Python<'_>,
    request_json: &str,
    callbacks: Vec<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    if request_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("local cyclic request limit"));
    }
    let request: LocalRequest =
        serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
    if callbacks.len() != request.mechanisms.len() {
        return Err(crate::value_err("local callback count"));
    }
    let mechanisms: Vec<DeclaredCyclicMechanism> = request
        .mechanisms
        .into_iter()
        .zip(callbacks)
        .map(|((id, reads), callback)| {
            let declared = PythonCyclicCallback::new(&id, &reads, callback);
            DeclaredCyclicMechanism { id, reads, callback: Arc::new(declared) }
        })
        .collect();
    let run = LocalCyclicRun {
        noise: request.noise,
        clamps: request.clamps,
        initial: request.initial,
        factual: request.factual,
        policy: request.policy,
    };
    crate::detach_catch(py, move || {
        let compiled =
            compile_local_cyclic(&mechanisms, &run, &ctx).map_err(|error| match error {
                LocalCyclicCompileError::Model(error) => operation_error(error, &ctx),
                other => operation_error(other, &ctx),
            })?;
        serde_json::to_string(&serde_json::json!({
            "inspection": compiled.inspection(),
            "result": compiled.result(),
            "counterfactual": compiled.counterfactual(),
        }))
        .map_err(|e| {
            operation_error(
                antecedent::gcm::nonlinear_equilibrium::EquilibriumError::Invalid(e.to_string()),
                &ctx,
            )
        })
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(execute_local_cyclic, m)?)
}
