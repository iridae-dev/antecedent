//! Typed compiler owners for joint scenario data and robust readouts.
use super::{
    Bound, IoError, Prepared, PyAnyMethods, PyBytes, PyDictMethods, PyErr,
    PyJointDistributionArtifact, PyModule, PyModuleMethods, PyRef, PyResult, Python,
    admissible_contract_from_json_refusal, check_size, external_refusal_json, prepare, pyclass,
    pyfunction, pymethods, serialization, structural_json, wrap_pyfunction,
};
use crate::prepared_api::PyPreparedAnalysis;
use antecedent::execution::SearchStop;
use antecedent::prelude::ExecutionContext;
use antecedent::{
    BoundDecisionContract, CompiledRobustDecision, CompiledRobustResult, JointScenarioCarrier,
};
use std::{io::Write, sync::Arc};

fn io_error(error: IoError) -> PyErr {
    // This closed producer preserves the evaluator's registered detail as the
    // prefix of Refused.message. The typed variant/code decides refusal kind;
    // reading the prefix here only restores the presentation field.
    if let IoError::Refused { code, message } = error {
        return Python::attach(|py| {
            let build = || -> PyResult<PyErr> {
                let fields = pyo3::types::PyDict::new(py);
                fields.set_item("code", code)?;
                fields.set_item("stage", "decision")?;
                fields.set_item(
                    "detail",
                    message.split_once(':').map_or(message.as_str(), |(detail, _)| detail),
                )?;
                fields.set_item("message", &message)?;
                let class = py.import("antecedent.decision")?.getattr("DecisionRefusal")?;
                Ok(PyErr::from_value(class.call1((fields,))?))
            };
            build().unwrap_or_else(|error| error)
        });
    }
    crate::transport_common::error(error)
}
fn causal_error(error: antecedent::CausalError) -> PyErr {
    match error {
        antecedent::CausalError::Serialization(error) => io_error(error),
        other => crate::py_err(other),
    }
}
fn context(
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    crate::transport_common::execution_context(0, memory_bytes, cancel)
}
fn stop(value: Option<&str>) -> PyResult<Option<SearchStop>> {
    value
        .map(|s| match s {
            "operations" | "search.operations" => Ok(SearchStop::Operations),
            "depth" | "search.depth" => Ok(SearchStop::Depth),
            "memory" | "search.memory" => Ok(SearchStop::Memory),
            "cancelled" | "search.cancelled" => Ok(SearchStop::Cancelled),
            _ => Err(crate::value_err(
                "search_stop must be operations, depth, memory, cancelled or None",
            )),
        })
        .transpose()
}
struct Counter(u64);
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("scenario input size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn input_bytes(sources: &[Option<PyRef<'_, PyJointDistributionArtifact>>]) -> PyResult<u64> {
    let mut counter = Counter(4096);
    for source in sources.iter().flatten() {
        serde_json::to_writer(&mut counter, source.inner().metadata()).map_err(serialization)?;
        counter.0 = counter
            .0
            .checked_add(
                (source.inner().draws().len() as u64)
                    .checked_mul(8)
                    .ok_or_else(|| crate::value_err("scenario input size overflow"))?,
            )
            .ok_or_else(|| crate::value_err("scenario input size overflow"))?;
    }
    counter.0.checked_mul(2).ok_or_else(|| crate::value_err("scenario input size overflow"))
}

#[pyclass(name = "JointScenarioCarrier", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyJointScenarioCarrier {
    inner: JointScenarioCarrier,
}
#[pymethods]
impl PyJointScenarioCarrier {
    #[getter]
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    #[getter]
    fn causal_target(&self) -> &str {
        self.inner.causal_target()
    }
    #[getter]
    fn causal_identification(&self) -> &str {
        self.inner.causal_identification()
    }
    #[getter]
    fn joint_structure(&self) -> &str {
        self.inner.joint_structure()
    }
    #[getter]
    fn complete(&self) -> bool {
        self.inner.complete()
    }
    #[getter]
    fn search_stop(&self) -> Option<&str> {
        self.inner.search_stop()
    }
    #[getter]
    fn n_scenarios(&self) -> usize {
        self.inner.len()
    }
    #[pyo3(signature=(*,memory_bytes=None,cancel=None))]
    fn export<'py>(
        &self,
        py: Python<'py>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let source = self.inner.clone();
        let bytes = crate::detach_catch(py, move || {
            source.export(&context(memory_bytes, cancel)).map_err(io_error)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}

#[pyclass(name = "CompiledRobustReadout")]
pub(crate) struct PyCompiledRobustReadout {
    _inner: Arc<CompiledRobustResult>,
    json: String,
}
#[pymethods]
impl PyCompiledRobustReadout {
    fn json(&self) -> &str {
        &self.json
    }
}

fn readout(
    plan: &CompiledRobustDecision,
    result: CompiledRobustResult,
) -> PyResult<PyCompiledRobustReadout> {
    let mut value: serde_json::Value =
        serde_json::from_str(&plan.source().result_json(&result).map_err(io_error)?)
            .map_err(serialization)?;
    let map =
        value.as_object_mut().ok_or_else(|| crate::value_err("robust readout is not an object"))?;
    map.insert("structural".into(), structural_json(&result.decision().structural));
    map.insert("program_identity".into(), serde_json::json!(plan.program_identity()));
    map.insert(
        "bound_contract_identity".into(),
        serde_json::json!(plan.decision_contract().identity()),
    );
    let json = serde_json::to_string(&value).map_err(serialization)?;
    Ok(PyCompiledRobustReadout { _inner: Arc::new(result), json })
}

#[pyclass(name = "CompiledRobustDecision", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCompiledRobustDecision {
    inner: Arc<CompiledRobustDecision>,
}
#[pymethods]
impl PyCompiledRobustDecision {
    #[getter]
    fn program_identity(&self) -> &str {
        self.inner.program_identity()
    }
    #[getter]
    fn source_identity(&self) -> &str {
        self.inner.source().identity()
    }
    fn describe(&self) -> String {
        serde_json::json!({"program_identity":self.inner.program_identity(),"source_identity":self.inner.source().identity(),
            "carrier":"identified_set","complete":self.inner.source().complete(),"search_stop":self.inner.source().search_stop(),
            "n_scenarios":self.inner.source().len(),"joint_structure":self.inner.source().joint_structure(),
            "bound_contract_identity":self.inner.decision_contract().identity(),"standing":self.inner.standing()}).to_string()
    }
    #[pyo3(signature=(*,memory_bytes=None,cancel=None))]
    fn evaluate(
        &self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyCompiledRobustReadout> {
        let plan = Arc::clone(&self.inner);
        crate::detach_catch(py, move || {
            let result = plan.execute(&context(memory_bytes, cancel)).map_err(causal_error)?;
            readout(&plan, result)
        })
    }
    #[pyo3(signature=(*,memory_bytes=None,cancel=None))]
    fn export<'py>(
        &self,
        py: Python<'py>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let plan = Arc::clone(&self.inner);
        let bytes = crate::detach_catch(py, move || {
            plan.export(&context(memory_bytes, cancel)).map_err(causal_error)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(artifact,*,memory_bytes=None,cancel=None))]
    fn consume(
        &self,
        py: Python<'_>,
        artifact: &[u8],
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyCompiledRobustReadout> {
        let plan = Arc::clone(&self.inner);
        let ctx = context(memory_bytes, cancel);
        let admission = ctx
            .acquire_admission_product(0, artifact.len() as u64, 2)
            .map_err(|e| io_error(IoError::Admission(e)))?;
        let bytes = artifact.to_vec();
        crate::detach_catch(py, move || {
            let _admission = admission;
            let result = plan.consume(&bytes, &ctx).map_err(causal_error)?;
            readout(&plan, result)
        })
    }
}

#[pyfunction]
#[pyo3(signature=(study,declaration_json,claims_json,sources,joint_structure,exhaustive,*,search_stop=None,memory_bytes=None,cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one explicit bounded joint source request")]
fn joint_scenario_carrier_for_study(
    py: Python<'_>,
    study: PyRef<'_, PyPreparedAnalysis>,
    declaration_json: &str,
    claims_json: &str,
    sources: Vec<Option<PyRef<'_, PyJointDistributionArtifact>>>,
    joint_structure: &str,
    exhaustive: bool,
    search_stop: Option<&str>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<PyJointScenarioCarrier>, Option<String>)> {
    check_size(declaration_json)?;
    check_size(claims_json)?;
    let ctx = context(memory_bytes, cancel);
    let source_bytes = input_bytes(&sources)?;
    let json_bytes = (declaration_json.len() as u64)
        .checked_add(claims_json.len() as u64)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| source_bytes.checked_add(n))
        .ok_or_else(|| crate::value_err("scenario declaration size overflow"))?;
    let admission =
        ctx.acquire_admission(0, json_bytes).map_err(|e| io_error(IoError::Admission(e)))?;
    let Prepared { adapted, .. } =
        match prepare(declaration_json, "finite_scenarios", claims_json, &sources) {
            Ok(value) => value,
            Err(refusal) => return Ok((None, Some(refusal))),
        };
    let causal = study.inner.contract().map_err(causal_error)?;
    let target = causal.identities.target.to_hex();
    let identification = causal.identities.identification.to_hex();
    let joint_structure = joint_structure.to_owned();
    let search_stop = stop(search_stop)?;
    drop(study);
    drop(sources);
    let inner = crate::detach_catch(py, move || {
        let _admission = admission;
        JointScenarioCarrier::checked(
            &target,
            &identification,
            &joint_structure,
            exhaustive,
            search_stop,
            adapted,
            &ctx,
        )
        .map_err(io_error)
    })?;
    Ok((Some(PyJointScenarioCarrier { inner }), None))
}

#[pyfunction]
#[pyo3(signature=(artifact,expected_identity,*,memory_bytes=None,cancel=None))]
fn consume_joint_scenario_carrier(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: &str,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyJointScenarioCarrier> {
    let ctx = context(memory_bytes, cancel);
    let admission = ctx
        .acquire_admission_product(0, artifact.len() as u64, 2)
        .map_err(|e| io_error(IoError::Admission(e)))?;
    let bytes = artifact.to_vec();
    let expected = expected_identity.to_owned();
    let inner = crate::detach_catch(py, move || {
        let _admission = admission;
        JointScenarioCarrier::consume(&bytes, &expected, &ctx).map_err(io_error)
    })?;
    Ok(PyJointScenarioCarrier { inner })
}

#[pyfunction]
#[pyo3(signature=(study,declaration_json,source,*,memory_bytes=None,cancel=None))]
fn compile_robust_for_study(
    py: Python<'_>,
    study: PyRef<'_, PyPreparedAnalysis>,
    declaration_json: &str,
    source: PyRef<'_, PyJointScenarioCarrier>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<PyCompiledRobustDecision>, Option<String>)> {
    check_size(declaration_json)?;
    let ctx = context(memory_bytes, cancel);
    let admission = ctx
        .acquire_admission_product(0, declaration_json.len() as u64, 8)
        .map_err(|e| io_error(IoError::Admission(e)))?;
    let contract = match admissible_contract_from_json_refusal(declaration_json) {
        Ok(c) => c,
        Err(r) => return Ok((None, Some(external_refusal_json(&r)))),
    };
    let study = Arc::clone(&study.inner);
    let source = source.inner.clone();
    let result = crate::detach_catch(py, move || {
        let _admission = admission;
        let causal = study.contract().map_err(causal_error)?;
        let bound = match BoundDecisionContract::bind(&causal, contract) {
            Ok(b) => b,
            Err(r) => return Ok(Err(external_refusal_json(&r))),
        };
        study.compile_for_robust_decision(bound, source, &ctx).map(Ok).map_err(causal_error)
    })?;
    match result {
        Ok(inner) => Ok((Some(PyCompiledRobustDecision { inner: Arc::new(inner) }), None)),
        Err(refusal) => Ok((None, Some(refusal))),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyJointScenarioCarrier>()?;
    module.add_class::<PyCompiledRobustDecision>()?;
    module.add_class::<PyCompiledRobustReadout>()?;
    module.add_function(wrap_pyfunction!(joint_scenario_carrier_for_study, module)?)?;
    module.add_function(wrap_pyfunction!(consume_joint_scenario_carrier, module)?)?;
    module.add_function(wrap_pyfunction!(compile_robust_for_study, module)?)?;
    Ok(())
}
