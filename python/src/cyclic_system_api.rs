// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nonlinear cyclic systems: bounded Rust execution, fitting and artifact replay.
use antecedent_core::ExecutionContext;
use antecedent_io::cyclic_system_artifact::{
    CyclicExecutionRequest, MAX_CYCLIC_ARTIFACT_BYTES, consume_cyclic_artifact,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn execution_context(
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx = crate::py_execution_context_cancel(0, 1, cancel);
    ctx.memory.hard_limit_bytes = max_bytes;
    ctx
}

fn input_admission(
    json: &str,
    ctx: &ExecutionContext,
) -> PyResult<antecedent_core::execution::AdmissionLease> {
    if json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("cyclic input byte limit"));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(crate::value_err("cancelled"));
    }
    ctx.acquire_admission(0, (json.len() as u64).saturating_mul(4))
        .map_err(|e| crate::value_err(e.to_string()))
}

fn parse_request(json: &str) -> PyResult<CyclicExecutionRequest> {
    if json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("cyclic request exceeds byte limit"));
    }
    serde_json::from_str(json).map_err(|e| crate::value_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn execute_cyclic_system(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(String, Py<PyBytes>)> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    let request = parse_request(request_json)?;
    let (report, bytes) = py
        .detach(move || request.export("cyclic_equilibrium", &ctx))
        .map_err(|e| crate::value_err(e.to_string()))?;
    let json =
        serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))?;
    Ok((json, PyBytes::new(py, &bytes).unbind()))
}

#[pyfunction]
#[pyo3(signature = (bytes, expected_identity, *, max_bytes=None, cancel=None))]
fn consume_cyclic_system(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::serialization_err("cyclic artifact byte limit"));
    }
    let report = py
        .detach(|| consume_cyclic_artifact(bytes, expected_identity, &ctx))
        .map_err(|e| crate::serialization_err(e.to_string()))?;
    serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (request_json, prior_sd, chains, warmup, draws, proposal_scale, seed, *, max_bytes=None, cancel=None))]
fn cyclic_parameter_posterior(
    py: Python<'_>,
    request_json: &str,
    prior_sd: f64,
    chains: usize,
    warmup: usize,
    draws: usize,
    proposal_scale: f64,
    seed: u64,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let mut ctx = execution_context(max_bytes, cancel);
    ctx.rng = antecedent_core::RngFactory::from_seed(seed);
    let _input = input_admission(request_json, &ctx)?;
    let request = parse_request(request_json)?;
    let fit = request.fit.as_ref().ok_or_else(|| {
        crate::value_err("posterior requires retained data/parameter coordinates")
    })?;
    let report = py
        .detach(|| {
            antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior(
                &request.model,
                &fit.parameters,
                &fit.rows,
                prior_sd,
                antecedent_prob::metropolis::MetropolisOptions {
                    chains,
                    warmup,
                    draws,
                    proposal_scale,
                },
                &ctx,
            )
        })
        .map_err(|e| crate::value_err(e.to_string()))?;
    serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCyclicEvidence>()?;
    m.add_function(wrap_pyfunction!(consume_cyclic_evidence, m)?)?;
    m.add_function(wrap_pyfunction!(execute_local_cyclic, m)?)?;
    m.add_class::<PyPreparedCyclicSystem>()?;
    m.add_class::<PyPreparedCyclicFamily>()?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_family, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cyclic_system, m)?)?;
    m.add_function(wrap_pyfunction!(execute_cyclic_system, m)?)?;
    m.add_function(wrap_pyfunction!(execute_cyclic_family, m)?)?;
    m.add_function(wrap_pyfunction!(consume_cyclic_family, m)?)?;
    m.add_function(wrap_pyfunction!(consume_cyclic_system, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_parameter_posterior, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn execute_cyclic_family(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(String, Py<PyBytes>)> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    if request_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("cyclic family request limit"));
    }
    let request: antecedent_io::cyclic_family_artifact::CyclicFamilyRequest =
        serde_json::from_str(request_json).map_err(|e| crate::value_err(e.to_string()))?;
    let (report, bytes) =
        py.detach(|| request.export(&ctx)).map_err(|e| crate::value_err(e.to_string()))?;
    Ok((
        serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))?,
        PyBytes::new(py, &bytes).unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (bytes, expected_identity, *, max_bytes=None, cancel=None))]
fn consume_cyclic_family(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let report = py
        .detach(|| {
            antecedent_io::cyclic_family_artifact::consume_cyclic_family(
                bytes,
                expected_identity,
                &ctx,
            )
        })
        .map_err(|e| crate::serialization_err(e.to_string()))?;
    serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
}

#[pyclass(name = "_PreparedCyclicSystem")]
pub(crate) struct PyPreparedCyclicSystem {
    inner: antecedent::analysis::PreparedStudy<antecedent::analysis::cyclic::CyclicPreparedState>,
}

#[pymethods]
impl PyPreparedCyclicSystem {
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(String, Py<PyBytes>)> {
        let ctx = execution_context(max_bytes, cancel);
        let (report, bytes) = py
            .detach(|| self.inner.export("cyclic_equilibrium", &ctx))
            .map_err(|e| crate::value_err(e.to_string()))?;
        Ok((
            serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))?,
            PyBytes::new(py, &bytes).unbind(),
        ))
    }
    fn inspect(&self) -> PyResult<String> {
        let inspected = self.inner.inspect().map_err(|e| crate::value_err(e.to_string()))?;
        Ok(serde_json::json!({"identity":inspected.identity,"fits_performed":inspected.fits_performed,"solve_blocks":inspected.solve_blocks,"execution_plans":inspected.execution_plans}).to_string())
    }
    fn retarget(&self, worlds_json: &str, noise_json: &str, initial_json: &str) -> PyResult<Self> {
        for json in [worlds_json, noise_json, initial_json] {
            if json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
                return Err(crate::value_err("cyclic retarget input limit"));
            }
        }
        let worlds =
            serde_json::from_str(worlds_json).map_err(|e| crate::value_err(e.to_string()))?;
        let noise =
            serde_json::from_str(noise_json).map_err(|e| crate::value_err(e.to_string()))?;
        let initial =
            serde_json::from_str(initial_json).map_err(|e| crate::value_err(e.to_string()))?;
        let inner = self
            .inner
            .retarget_worlds(worlds, noise, initial)
            .map_err(|e| crate::value_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn prepare_cyclic_system(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedCyclicSystem> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    let request = parse_request(request_json)?;
    let inner = py
        .detach(|| antecedent::analysis::Study::prepare_cyclic(request, &ctx))
        .map_err(|e| crate::value_err(e.to_string()))?;
    Ok(PyPreparedCyclicSystem { inner })
}

#[pyclass(name = "_PreparedCyclicFamily")]
pub(crate) struct PyPreparedCyclicFamily {
    inner: antecedent::analysis::PreparedStudy<
        antecedent::analysis::cyclic::CyclicFamilyPreparedState,
    >,
}
#[pymethods]
impl PyPreparedCyclicFamily {
    #[pyo3(signature = (*, max_bytes=None, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(String, Py<PyBytes>)> {
        let ctx = execution_context(max_bytes, cancel);
        let (report, artifact) =
            py.detach(|| self.inner.export(&ctx)).map_err(|e| crate::value_err(e.to_string()))?;
        Ok((
            serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))?,
            PyBytes::new(py, &artifact).unbind(),
        ))
    }
    #[pyo3(signature=(schema_json,selectors_json,*,max_bytes=None,cancel=None))]
    fn evidence(
        &self,
        py: Python<'_>,
        schema_json: &str,
        selectors_json: &str,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyCyclicEvidence> {
        let ctx = execution_context(max_bytes, cancel);
        let _schema_input = input_admission(schema_json, &ctx)?;
        let _selectors_input = input_admission(selectors_json, &ctx)?;
        if schema_json.len() > MAX_CYCLIC_ARTIFACT_BYTES
            || selectors_json.len() > MAX_CYCLIC_ARTIFACT_BYTES
        {
            return Err(crate::value_err("cyclic evidence input byte limit"));
        }
        let schema =
            serde_json::from_str(schema_json).map_err(|e| crate::value_err(e.to_string()))?;
        let selectors =
            serde_json::from_str(selectors_json).map_err(|e| crate::value_err(e.to_string()))?;
        let inner = py
            .detach(|| self.inner.cyclic_evidence(schema, selectors, &ctx))
            .map_err(|e| crate::value_err(e.to_string()))?;
        Ok(PyCyclicEvidence { inner })
    }
    fn inspect(&self) -> String {
        let inspection = self.inner.inspect();
        serde_json::json!({"identity":inspection.identity,"family":inspection.family,"preparation_evaluations":inspection.preparation_evaluations,"retained_bytes":inspection.retained_bytes}).to_string()
    }
}
#[pyfunction]
#[pyo3(signature = (request_json, *, max_bytes=None, cancel=None))]
fn prepare_cyclic_family(
    py: Python<'_>,
    request_json: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedCyclicFamily> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    if request_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("cyclic family request limit"));
    }
    let request =
        serde_json::from_str(request_json).map_err(|e| crate::value_err(e.to_string()))?;
    let inner = py
        .detach(|| antecedent::analysis::Study::prepare_cyclic_family(&request, &ctx))
        .map_err(|e| crate::value_err(e.to_string()))?;
    Ok(PyPreparedCyclicFamily { inner })
}

struct PythonLocalMechanism {
    id: String,
    reads: Vec<usize>,
    callback: Py<PyAny>,
}
impl antecedent_model::local_cyclic::LocalCyclicMechanism for PythonLocalMechanism {
    fn id(&self) -> &str {
        &self.id
    }
    fn reads(&self) -> &[usize] {
        &self.reads
    }
    fn evaluate(
        &self,
        inputs: &[f64],
    ) -> Result<(f64, Vec<f64>), antecedent_model::nonlinear_equilibrium::EquilibriumError> {
        Python::attach(|py| {
            self.callback.bind(py).call1((inputs.to_vec(),))?.extract::<(f64, Vec<f64>)>()
        })
        .map_err(|error| {
            antecedent_model::nonlinear_equilibrium::EquilibriumError::Invalid(format!(
                "local callback {}: {}",
                self.id,
                crate::interrupt::callback_failure("cyclic callback raised", error)
            ))
        })
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalRequest {
    mechanisms: Vec<(String, Vec<usize>)>,
    noise: Vec<f64>,
    clamps: Vec<antecedent_model::nonlinear_equilibrium::Clamp>,
    initial: Vec<f64>,
    factual: Option<Vec<f64>>,
    policy: antecedent_model::local_cyclic::LocalCyclicPolicy,
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
        serde_json::from_str(request_json).map_err(|e| crate::value_err(e.to_string()))?;
    if callbacks.len() != request.mechanisms.len() {
        return Err(crate::value_err("local callback count"));
    }
    let model = antecedent_model::local_cyclic::LocalCyclicModel {
        mechanisms: request
            .mechanisms
            .into_iter()
            .zip(callbacks)
            .map(|((id, reads), callback)| {
                Box::new(PythonLocalMechanism { id, reads, callback })
                    as Box<dyn antecedent_model::local_cyclic::LocalCyclicMechanism>
            })
            .collect(),
    };
    py.detach(move || {
        let inspection = model.inspect(&request.clamps)?;
        let result = model.solve(&request.noise, &request.clamps, &request.initial, request.policy, &ctx)?;
        let counterfactual = request.factual.as_ref().map(|factual| {
            let shared = model.abduce(factual, request.policy, &ctx)?;
            model.solve(&shared, &request.clamps, factual, request.policy, &ctx)
        }).transpose()?;
        serde_json::to_string(&serde_json::json!({"inspection":inspection,"result":result,"counterfactual":counterfactual}))
            .map_err(|e| antecedent_model::nonlinear_equilibrium::EquilibriumError::Invalid(e.to_string()))
    }).map_err(|e| crate::value_err(e.to_string()))
}

#[pyclass(name = "_CyclicEvidence")]
struct PyCyclicEvidence {
    inner: antecedent::analysis::cyclic_evidence::CyclicEvidence,
}
#[pymethods]
impl PyCyclicEvidence {
    fn report(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.report())
            .map_err(|e| crate::serialization_err(e.to_string()))
    }
    #[pyo3(signature=(*,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = execution_context(max_bytes, cancel);
        let bytes =
            py.detach(|| self.inner.export(&ctx)).map_err(|e| crate::value_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
    #[pyo3(signature=(*,max_bytes=None,cancel=None))]
    fn joint_law(
        &self,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<crate::distribution_api::PyJointDistributionArtifact> {
        let ctx = execution_context(max_bytes, cancel);
        if ctx.cancellation.is_cancelled() {
            return Err(crate::value_err("cancelled"));
        }
        let samples =
            self.inner.report().columns.iter().map(|c| c.samples.len() as u64).sum::<u64>();
        let _lease = ctx
            .acquire_admission(0, samples.saturating_mul(32))
            .map_err(|e| crate::value_err(e.to_string()))?;
        Ok(crate::distribution_api::PyJointDistributionArtifact {
            artifact: self.inner.joint_law().map_err(|e| crate::value_err(e.to_string()))?,
            source_claim: None,
        })
    }
    fn decide_enclosures(&self, contract_json: &str) -> PyResult<String> {
        if contract_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("decision contract byte limit"));
        }
        let contract = antecedent_design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| crate::value_err(e.to_string()))?;
        self.inner
            .decide_enclosures(&contract)
            .map(|v| v.to_string())
            .map_err(|e| crate::value_err(e.to_string()))
    }
    fn decide_means(&self, contract_json: &str) -> PyResult<String> {
        if contract_json.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(crate::value_err("decision contract byte limit"));
        }
        let contract = antecedent_design::decision_artifact::contract_from_json(contract_json)
            .map_err(|e| crate::value_err(e.to_string()))?;
        self.inner
            .decide_means(&contract)
            .map(|v| v.to_string())
            .map_err(|e| crate::value_err(e.to_string()))
    }
}
#[pyfunction]
#[pyo3(signature=(bytes,expected_identity,*,max_bytes=None,cancel=None))]
fn consume_cyclic_evidence(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyCyclicEvidence> {
    let ctx = execution_context(max_bytes, cancel);
    let inner = py
        .detach(|| {
            antecedent::analysis::cyclic_evidence::CyclicEvidence::consume(
                bytes,
                expected_identity,
                &ctx,
            )
        })
        .map_err(|e| crate::serialization_err(e.to_string()))?;
    Ok(PyCyclicEvidence { inner })
}
