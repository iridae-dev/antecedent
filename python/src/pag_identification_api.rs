//! Typed host bridge for direct static PAG identification.

use antecedent::analysis::pag_identification::{
    CompiledPagIdentification, PagIdError, PagIdentificationArtifact, PagIdentificationProblem,
    PagReplayError, compile_pag_identification, consume_pag_identification,
};
use antecedent::estimate::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, LawTolerance,
};
use antecedent::graph::{DenseNodeId, MiddleMark};
use antecedent::prelude::{Value, VariableId};
use antecedent::reason::reason_code;
use antecedent::schema::RegimeId;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::json;

use crate::PyCancellationToken;
use crate::bridge::RefusalJson;
use crate::chain_decision_api::context;
use crate::graphs::Pag;

type Refused<T> = (Option<T>, Option<String>);

fn fail(error: &PagReplayError) -> String {
    let (code, detail) = match error {
        PagReplayError::Identification(
            PagIdError::Budget(_) | PagIdError::Depth { .. } | PagIdError::Admission(_),
        ) => (reason_code!("route_not_supported"), "pag_identification.resource_exhausted"),
        PagReplayError::Identification(PagIdError::Cancelled) => {
            (reason_code!("cancelled_no_claim"), "pag_identification.cancelled")
        }
        PagReplayError::Unsupported(why) => (why.code(), "pag_identification.not_supported"),
        PagReplayError::Evaluation(_) => {
            (reason_code!("invalid_argument"), "pag_identification.exact_evaluation_refused")
        }
        _ => (reason_code!("invalid_argument"), "pag_identification.invalid_declaration"),
    };
    RefusalJson::new(code, "pag_identification", detail, &error.to_string()).json()
}

#[pyclass(name = "DirectPagPlanHandle", skip_from_py_object)]
pub(crate) struct PyDirectPagPlan {
    plan: CompiledPagIdentification,
}

#[pyclass(name = "DirectPagResultHandle", skip_from_py_object)]
pub(crate) struct PyDirectPagResult {
    artifact: PagIdentificationArtifact,
}

#[pymethods]
impl PyDirectPagPlan {
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

    #[pyo3(signature = (*, steps=1_048_576, retain_resource_stop=false, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute(
        &self,
        py: Python<'_>,
        steps: u64,
        retain_resource_stop: bool,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Refused<PyDirectPagResult>> {
        let ctx = context(seed, threads, cancel, memory_bytes).with_id_work_limit(steps);
        let plan = &self.plan;
        let run = crate::detach_catch(py, move || {
            Ok(if retain_resource_stop {
                plan.execute_retaining_stop(&ctx)
            } else {
                plan.execute(&ctx)
            })
        })?;
        Ok(match run {
            Ok(artifact) => (Some(PyDirectPagResult { artifact }), None),
            Err(error) => (None, Some(fail(&error))),
        })
    }
}

#[pymethods]
impl PyDirectPagResult {
    #[getter]
    fn identity(&self) -> &str {
        self.artifact.identity()
    }

    #[getter]
    fn work(&self) -> u64 {
        self.artifact.work()
    }

    fn proof_json(&self) -> PyResult<String> {
        serde_json::to_string(self.artifact.proof()).map_err(crate::value_err_display)
    }

    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.artifact
            .to_bytes()
            .map(|bytes| PyBytes::new(py, &bytes))
            .map_err(crate::value_err_display)
    }

    #[pyo3(signature = (levels, probabilities, bindings, *, operations=10_000_000, seed=1, threads=None, cancel=None, memory_bytes=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the declared law, concrete query and execution controls"
    )]
    fn evaluate(
        &self,
        py: Python<'_>,
        levels: Vec<Vec<i64>>,
        probabilities: Vec<f64>,
        bindings: Vec<(String, i64)>,
        operations: usize,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Refused<String>> {
        let problem = self.artifact.problem();
        if levels.len() != problem.names.len() {
            return Err(crate::value_err("exact PAG evaluation needs levels for every variable"));
        }
        let mut assignment = Assignment::new();
        for (name, value) in bindings {
            let id = locate(&problem.names, &name)?;
            if assignment.get(id).is_some() {
                return Err(crate::value_err("duplicate exact PAG assignment"));
            }
            assignment.set(id, Value::Int64(value));
        }
        let axes = levels
            .into_iter()
            .enumerate()
            .map(|(index, values)| {
                let variable = u32::try_from(index)
                    .map(VariableId::from_raw)
                    .map_err(crate::value_err_display)?;
                Ok(DiscreteAxis {
                    variable,
                    values: values.into_iter().map(Value::Int64).collect::<Vec<_>>().into(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let artifact = &self.artifact;
        let ctx = context(seed, threads, cancel, memory_bytes);
        let evaluated = crate::detach_catch(py, move || {
            let law = ExactDiscreteLaw::try_new(
                problem.population.as_str(),
                RegimeId::from_raw(0),
                [],
                axes,
                probabilities,
                "declared-pag-observational-law",
                LawTolerance::default(),
            )
            .map_err(crate::value_err_display)?;
            Ok(artifact.evaluate(
                law,
                &assignment,
                ExactEvaluationLimits { operations, ..ExactEvaluationLimits::default() },
                &ctx,
            ))
        })?;
        Ok(match evaluated {
            Ok(law) => {
                let mut atoms = Vec::with_capacity(law.atoms.len());
                for row in &*law.atoms {
                    let values = row
                        .iter()
                        .map(|value| match value {
                            Value::Int64(level) => Ok(*level),
                            _ => Err(crate::value_err("unexpected exact PAG axis value")),
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    atoms.push(values);
                }
                let names: Vec<_> =
                    law.outcomes.iter().map(|id| &problem.names[id.raw() as usize]).collect();
                (
                    Some(
                        json!({
                            "outcomes": names,
                            "atoms": atoms,
                            "probabilities": law.probabilities.as_ref(),
                        })
                        .to_string(),
                    ),
                    None,
                )
            }
            Err(error) => (None, Some(fail(&error))),
        })
    }
}

fn locate(names: &[String], name: &str) -> PyResult<VariableId> {
    names
        .iter()
        .position(|n| n == name)
        .and_then(|i| u32::try_from(i).ok())
        .map(VariableId::from_raw)
        .ok_or_else(|| crate::value_err(format!("unknown PAG variable {name:?}")))
}

#[pyfunction]
fn direct_pag_compile(
    pag: PyRef<'_, Pag>,
    actions: Vec<String>,
    outcomes: Vec<String>,
    given: Vec<String>,
    population: String,
) -> PyResult<Refused<PyDirectPagPlan>> {
    // PagWire carries endpoint marks only. Check the source before encoding it.
    for index in 0..pag.pag.node_count() {
        let node = DenseNodeId::from_raw(u32::try_from(index).map_err(crate::value_err_display)?);
        if pag.pag.neighbors(node).any(|(neighbor, _, _)| {
            node < neighbor
                && pag
                    .pag
                    .edge_between(node, neighbor)
                    .is_some_and(|edge| edge.middle != MiddleMark::Empty)
        }) {
            return Ok((
                None,
                Some(fail(&PagReplayError::Invalid(
                    "the oracle PAG contains unfinished middle marks",
                ))),
            ));
        }
    }
    let ids = |values: Vec<String>| {
        let mut ids = values
            .iter()
            .map(|value| locate(&pag.names, value).map(VariableId::raw))
            .collect::<PyResult<Vec<_>>>()?;
        ids.sort_unstable();
        Ok::<_, PyErr>(ids)
    };
    let problem = PagIdentificationProblem {
        graph: antecedent::io::pag_to_wire(&pag.pag).map_err(crate::value_err_display)?,
        names: pag.names.clone(),
        actions: ids(actions)?,
        outcomes: ids(outcomes)?,
        given: ids(given)?,
        population,
    };
    Ok(match compile_pag_identification(problem) {
        Ok(plan) => (Some(PyDirectPagPlan { plan }), None),
        Err(error) => (None, Some(fail(&error))),
    })
}

#[pyfunction]
#[pyo3(signature = (data, expected_result, expected_program, *, steps=1_048_576, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments, reason = "the retained artifact and execution controls")]
fn direct_pag_consume(
    py: Python<'_>,
    data: &[u8],
    expected_result: &str,
    expected_program: &str,
    steps: u64,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyDirectPagResult>> {
    let program = crate::digest::digest(expected_program, "expected_program")?;
    let ctx = context(seed, threads, cancel, memory_bytes).with_id_work_limit(steps);
    let result = crate::detach_catch(py, move || {
        Ok(consume_pag_identification(data, expected_result, program, &ctx))
    })?;
    Ok(match result {
        Ok(artifact) => (Some(PyDirectPagResult { artifact }), None),
        Err(error) => (None, Some(fail(&error))),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDirectPagPlan>()?;
    module.add_class::<PyDirectPagResult>()?;
    module.add_function(wrap_pyfunction!(direct_pag_compile, module)?)?;
    module.add_function(wrap_pyfunction!(direct_pag_consume, module)?)
}
