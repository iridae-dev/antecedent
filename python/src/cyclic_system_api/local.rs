//! Process-local cyclic callback execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

struct PythonLocalMechanism {
    id: String,
    reads: Vec<usize>,
    callback: Py<PyAny>,
}
impl antecedent::gcm::local_cyclic::LocalCyclicMechanism for PythonLocalMechanism {
    fn id(&self) -> &str {
        &self.id
    }
    fn reads(&self) -> &[usize] {
        &self.reads
    }
    fn evaluate(
        &self,
        inputs: &[f64],
    ) -> Result<(f64, Vec<f64>), antecedent::gcm::nonlinear_equilibrium::EquilibriumError> {
        Python::attach(|py| {
            self.callback.bind(py).call1((inputs.to_vec(),))?.extract::<(f64, Vec<f64>)>()
        })
        .map_err(|error| {
            antecedent::gcm::nonlinear_equilibrium::EquilibriumError::CallbackFailed {
                mechanism: self.id.clone(),
                failure: crate::interrupt::callback_failure("cyclic callback raised", error),
            }
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
    let model = antecedent::gcm::local_cyclic::LocalCyclicModel {
        mechanisms: request
            .mechanisms
            .into_iter()
            .zip(callbacks)
            .map(|((id, reads), callback)| {
                Box::new(PythonLocalMechanism { id, reads, callback })
                    as Box<dyn antecedent::gcm::local_cyclic::LocalCyclicMechanism>
            })
            .collect(),
    };
    crate::detach_catch(py, move || {
        let execute = || {
            let inspection = model.inspect(&request.clamps)?;
            let result = model.solve(
                &request.noise,
                &request.clamps,
                &request.initial,
                request.policy,
                &ctx,
            )?;
            let counterfactual = request
                .factual
                .as_ref()
                .map(|factual| {
                    let shared = model.abduce(factual, request.policy, &ctx)?;
                    model.solve(&shared, &request.clamps, factual, request.policy, &ctx)
                })
                .transpose()?;
            serde_json::to_string(&serde_json::json!({"inspection":inspection,"result":result,"counterfactual":counterfactual}))
                .map_err(|e| antecedent::gcm::nonlinear_equilibrium::EquilibriumError::Invalid(e.to_string()))
        };
        execute().map_err(|e| operation_error(e, &ctx))
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(execute_local_cyclic, m)?)
}
