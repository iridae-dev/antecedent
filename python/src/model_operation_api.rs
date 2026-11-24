//! The native side of the `antecedent.analyze` query forms `ArrowStrengths`,
//! `PopulationContrast`, `DoSampling`, `FittedModelEvaluation`, `InterventionalMean` and
//! `SimultaneousCounterfactual`.
//!
//! The first four are operations of a prepared study on its own fitted model
//! (`PreparedStudy::model_operation`, program families `FittedModelAttribution` /
//! `FittedModelSimulation`); the last two run the compiled `Simulate` route. Each fits the
//! standard mechanism program on the supplied graph (the fit `fit_gcm` makes with no selection
//! keyword). The seeds are chosen so that a call with
//! `seed` reproduces the numbers of the corresponding `FittedGcm` method run with the same seed:
//! the population do-contrast and the model evaluation take `seed` itself, and the do-samplers and
//! the interventional mean take the seed of the stream the method drew from.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent::analysis::model_operation::ModelOperationAnswer;
use antecedent::execution::{CausalRng, RngFactory, StreamDomain};
use antecedent::prelude::{ExecutionContext, Intervention, TabularData, VariableId};
use antecedent::query::{
    ArrowStrengthsTarget, CausalQuery, DoSamplerMethod, DoSamplerTarget, ModelEvaluationTarget,
    ModelOperation, PopulationDoContrastTarget, SimulateQuery, SimulationTarget,
};
use antecedent::{MechanismProgram, SimulationOutcome, Study};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::NumericColumn;
use crate::compiled_route_api::{ModelSource, RouteHandle, RouteRecord, outcome_mismatch};
use crate::gcm_api::PyFittedGcm;
use crate::gcm_estimand_api::{
    PyArrowStrength, PyDoSample, PyModelEvaluation, PyPopulationDoContrast, hard_interventions,
};
use crate::{
    dag_from_named_edges, detach_catch, py_err, py_execution_context, py_msg, resolve_user_threads,
    tabular_from_numpy_receipt,
};

/// Stream index of `FittedGcm.interventional_mean`.
const INTERVENTIONAL_MEAN_STREAM: u64 = 0x1A1E;
/// Stream index of `FittedGcm.do_sample`.
const DO_SAMPLE_STREAM: u64 = 0xD05A;

/// The seed whose generator starts where the counterfactual stream `index` of `seed` starts.
fn stream_seed(seed: u64, index: u64) -> u64 {
    let stream = RngFactory::from_seed(seed).stream_for(StreamDomain::Counterfactual, index);
    // `CausalRng::from_seed(s)` starts at `s ^ from_seed(0).state()`.
    stream.state() ^ CausalRng::from_seed(0).state()
}

/// The answer of a fitted-model operation and the compiled route that produced it.
#[pyclass(skip_from_py_object, name = "ModelOperationAnswer")]
pub struct PyModelOperationAnswer {
    /// The operation: `arrow_strengths`, `population_do_contrast`, `do_sample`,
    /// `model_evaluation`, `interventional_mean` or `simultaneous_counterfactual`.
    #[pyo3(get)]
    pub kind: String,
    /// The engine's answer: a list of `ArrowStrength`, a `PopulationDoContrast`, a `DoSample`,
    /// a `ModelEvaluation`, or a float for the two means.
    #[pyo3(get)]
    pub value: Py<PyAny>,
    /// The compiled `Simulate` route the two means ran under (contract, claim, inspection
    /// record); `None` for an operation of a prepared study, which reports its program instead.
    #[pyo3(get)]
    pub route: RouteHandle,
    /// The program family the answer ran under.
    #[pyo3(get)]
    pub program_route: String,
    /// Hex identity of the lowered program the answer ran under.
    #[pyo3(get)]
    pub program_identity: Option<String>,
}

#[pymethods]
impl PyModelOperationAnswer {
    fn __repr__(&self) -> String {
        format!("ModelOperationAnswer(kind={:?})", self.kind)
    }
}

/// The table, graph and run settings of one operation.
struct Input {
    data: TabularData,
    edges: Vec<(String, String)>,
    seed: u64,
    threads: Option<u32>,
}

impl Input {
    fn load(
        names: &[String],
        columns: Vec<NumericColumn<'_>>,
        edges: Vec<(String, String)>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Self> {
        let loaded = tabular_from_numpy_receipt(names, &columns)?;
        drop(columns);
        Ok(Self { data: loaded.data, edges, seed, threads })
    }

    fn id(&self, name: &str) -> PyResult<VariableId> {
        self.data.schema().id_of(name).map_err(py_err)
    }

    fn source(self) -> PyResult<(ModelSource, ExecutionContext, u64)> {
        let ctx = py_execution_context(self.seed, resolve_user_threads(self.threads));
        let source = ModelSource {
            dag: dag_from_named_edges(self.data.schema(), &self.edges)?,
            data: Arc::new(self.data),
            accepted: None,
            program: MechanismProgram::standard(),
        };
        Ok((source, ctx, self.seed))
    }
}

fn answer(
    py: Python<'_>,
    kind: &str,
    value: Py<PyAny>,
    route: &RouteRecord,
) -> PyResult<PyModelOperationAnswer> {
    Ok(PyModelOperationAnswer {
        kind: kind.to_owned(),
        value,
        route: crate::compiled_route_api::route_object(py, Some(route))?,
        program_route: "GcmSimulateDag".to_owned(),
        program_identity: route.execution.lowered_identity().map(|digest| digest.to_hex()),
    })
}

/// An operation answered by a prepared study, with the program it ran under.
struct Operated {
    answer: ModelOperationAnswer,
    route: String,
    identity: String,
}

fn operated(
    py: Python<'_>,
    kind: &str,
    value: Py<PyAny>,
    run: &Operated,
) -> PyModelOperationAnswer {
    let _ = py;
    PyModelOperationAnswer {
        kind: kind.to_owned(),
        value,
        route: None,
        program_route: run.route.clone(),
        program_identity: Some(run.identity.clone()),
    }
}

/// The study whose fitted model an operation reads: the supplied rows and DAG under the standard
/// mechanism program. Its own question is an empty interventional sample; the operation does not
/// read it, and it neither changes the study nor enters the operation's answer.
fn run_operation(input: Input, operation: &ModelOperation) -> PyResult<Operated> {
    let (source, ctx, _) = input.source()?;
    let carrier =
        CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
            interventions: Arc::from([]),
            draws: 1,
            seed: 0,
        }));
    let prepared = Study::tabular((*source.data).clone())
        .graph(source.dag.clone())
        .query(carrier)
        .mechanisms(source.program.clone())
        .build()
        .map_err(py_err)?
        .prepare(&ctx)
        .map_err(py_err)?;
    let compiled = prepared.model_operation(operation, &ctx).map_err(py_err)?;
    let route = compiled.lowered().route().to_owned();
    let identity = compiled.lowered().identity().to_hex();
    Ok(Operated { answer: compiled.into_answer(), route, identity })
}

fn name_of(names: &[String], id: VariableId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
}

/// The model-implied strength of every edge, through the compiled `Attribute` route.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, *, seed=1, threads=None))]
fn model_operation_arrow_strengths(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyModelOperationAnswer> {
    let input = Input::load(&names, columns, edges, seed, threads)?;
    let run = detach_catch(py, move || {
        run_operation(input, &ModelOperation::ArrowStrengths(ArrowStrengthsTarget))
    })?;
    let ModelOperationAnswer::ArrowStrengths(strengths) = &run.answer else {
        return Err(outcome_mismatch());
    };
    let items = strengths
        .iter()
        .map(|s| {
            Py::new(
                py,
                PyArrowStrength {
                    parent: name_of(&names, s.parent),
                    child: name_of(&names, s.child),
                    strength: s.strength,
                    variance_contribution: s.variance_contribution,
                    coefficient: s.coefficient,
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    let value = PyList::new(py, items)?.into_any().unbind();
    Ok(operated(py, "arrow_strengths", value, &run))
}

/// The population do-contrast of `parent` on `child`, through the compiled `Attribute` route.
/// Both arms draw from the attribution stream of `seed`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, parent, child, delta, *, max_units=None, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "the query's fields and the run settings")]
fn model_operation_population_contrast(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    parent: String,
    child: String,
    delta: f64,
    max_units: Option<usize>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyModelOperationAnswer> {
    crate::require_finite([("delta", delta)])?;
    if delta == 0.0 {
        return Err(crate::value_err("delta must be nonzero"));
    }
    let input = Input::load(&names, columns, edges, seed, threads)?;
    let (parent_id, child_id) = (input.id(&parent)?, input.id(&child)?);
    let run = detach_catch(py, move || {
        let target = PopulationDoContrastTarget {
            parent: parent_id,
            child: child_id,
            delta,
            max_units,
            seed: input.seed,
        };
        run_operation(input, &ModelOperation::PopulationDoContrast(target))
    })?;
    let ModelOperationAnswer::PopulationDoContrast(result) = &run.answer else {
        return Err(outcome_mismatch());
    };
    let value = Py::new(
        py,
        PyPopulationDoContrast {
            contrast: result.contrast,
            stderr: result.stderr,
            parent,
            child,
            delta,
        },
    )?
    .into_any();
    Ok(operated(py, "population_do_contrast", value, &run))
}

/// A named do-sampler for `outcome` under hard `interventions`, through the compiled `Simulate`
/// route. The kernel-density and Metropolis-Hastings samplers start from the stream
/// `FittedGcm.do_sample` draws from under `seed`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, method, outcome, interventions, n=1000, *, bandwidth=None, proposal_sd=None, burn_in=None, thin=None, root_match_tolerance=None, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "one keyword per sampler control")]
fn model_operation_do_sample(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    method: String,
    outcome: String,
    interventions: HashMap<String, f64>,
    n: usize,
    bandwidth: Option<f64>,
    proposal_sd: Option<f64>,
    burn_in: Option<usize>,
    thin: Option<usize>,
    root_match_tolerance: Option<f64>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyModelOperationAnswer> {
    crate::require_finite(interventions.iter().map(|(name, level)| (name.as_str(), *level)))?;
    let applicable = |name: &str, given: bool, methods: &[&str]| -> PyResult<()> {
        if given && !methods.contains(&method.as_str()) {
            return Err(crate::value_err(format!(
                "{name} does not apply to the {method:?} do-sampler"
            )));
        }
        Ok(())
    };
    applicable("bandwidth", bandwidth.is_some(), &["kde"])?;
    applicable("proposal_sd", proposal_sd.is_some(), &["mcmc"])?;
    applicable("burn_in", burn_in.is_some(), &["mcmc"])?;
    applicable("thin", thin.is_some(), &["mcmc"])?;
    applicable("root_match_tolerance", root_match_tolerance.is_some(), &["weighting"])?;
    let sampler = match method.as_str() {
        "weighting" => {
            if interventions.len() != 1 {
                return Err(py_msg(
                    "the weighting do-sampler targets one hard intervention do(T = t); pass \
                     exactly one",
                ));
            }
            DoSamplerMethod::Weighting { root_match_tolerance }
        }
        "kde" => DoSamplerMethod::Kde { bandwidth },
        "mcmc" => DoSamplerMethod::Mcmc { proposal_sd, burn_in, thin },
        other => {
            return Err(py_msg(format!(
                "unknown do-sampler {other:?}; choose \"weighting\", \"kde\" or \"mcmc\""
            )));
        }
    };
    let input = Input::load(&names, columns, edges, seed, threads)?;
    let outcome_id = input.id(&outcome)?;
    let ints = hard_interventions(&input.data, &interventions)?;
    let run = detach_catch(py, move || {
        let target = DoSamplerTarget {
            method: sampler,
            outcome: outcome_id,
            interventions: Arc::from(ints),
            draws: n,
            seed: stream_seed(input.seed, DO_SAMPLE_STREAM),
        };
        run_operation(input, &ModelOperation::DoSample(target))
    })?;
    let ModelOperationAnswer::DoSample(result) = &run.answer else {
        return Err(outcome_mismatch());
    };
    let value = Py::new(py, PyDoSample::from_result(result.clone()))?.into_any();
    Ok(operated(py, "do_sample", value, &run))
}

/// The fitted model's falsification evaluation, through the compiled `Simulate` route.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, *, alpha=0.05, n_permutations=20, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "the query's fields and the run settings")]
fn model_operation_evaluation(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    alpha: f64,
    n_permutations: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyModelOperationAnswer> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(crate::value_err(format!("alpha must lie in (0, 1), got {alpha}")));
    }
    let input = Input::load(&names, columns, edges, seed, threads)?;
    let run = detach_catch(py, move || {
        let target = ModelEvaluationTarget { alpha, n_permutations, seed: input.seed };
        run_operation(input, &ModelOperation::ModelEvaluation(target))
    })?;
    let ModelOperationAnswer::ModelEvaluation(report) = &run.answer else {
        return Err(outcome_mismatch());
    };
    let value = Py::new(
        py,
        PyModelEvaluation {
            in_sample_loglik: report.in_sample_loglik,
            zero_probability_rows: report.zero_probability_rows,
            mean_abs_residual: report.mean_abs_residual,
            residual_independence_p: report.residual_independence_p.to_vec(),
            local_markov_p: report.local_markov_p.to_vec(),
            permutation_loglik: report.permutation_loglik,
            falsified: report.falsified,
            alpha: report.alpha,
            notes: report.notes.iter().map(ToString::to_string).collect(),
        },
    )?
    .into_any();
    Ok(operated(py, "model_evaluation", value, &run))
}

/// The mean of `outcome` over `n` interventional draws, through the compiled `Simulate` route
/// (the draws of an interventional sample, averaged as the direct kernel averages them). The
/// draws start from the stream `FittedGcm.interventional_mean` draws from under `seed`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, interventions, n, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "the query's fields and the run settings")]
fn model_operation_interventional_mean(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    outcome: String,
    interventions: HashMap<String, f64>,
    n: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyModelOperationAnswer> {
    crate::require_finite(interventions.iter().map(|(name, level)| (name.as_str(), *level)))?;
    let input = Input::load(&names, columns, edges, seed, threads)?;
    let outcome_id = input.id(&outcome)?;
    let ints = hard_interventions(&input.data, &interventions)?;
    let (mean, route) = detach_catch(py, move || {
        let (source, ctx, seed) = input.source()?;
        let target = SimulationTarget::InterventionalSample {
            interventions: Arc::from(ints),
            draws: n,
            seed: stream_seed(seed, INTERVENTIONAL_MEAN_STREAM),
        };
        let (simulated, route) = source.simulate(target, &ctx)?;
        let SimulationOutcome::Sample(batch) = simulated else {
            return Err(outcome_mismatch());
        };
        // The column of the outcome, in the model's dense order (the schema order here).
        let column = batch.column(outcome_id.as_usize()).map_err(py_err)?;
        #[allow(clippy::cast_precision_loss, reason = "a draw count")]
        let mean = column.iter().sum::<f64>() / column.len().max(1) as f64;
        Ok::<_, PyErr>((mean, route))
    })?;
    answer(py, "interventional_mean", mean.into_pyobject(py)?.into_any().unbind(), &route)
}

/// The mean of `outcome` when the `outer` and `inner` hard interventions hold simultaneously on
/// the abduced factual noise, through the compiled `Simulate` route. Overlapping sets are refused
/// when the query is declared.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, outer, inner, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "the query's fields and the run settings")]
fn model_operation_simultaneous_counterfactual(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    outcome: String,
    outer: HashMap<String, f64>,
    inner: HashMap<String, f64>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyModelOperationAnswer> {
    let input = Input::load(&names, columns, edges, seed, threads)?;
    let outcome_id = input.id(&outcome)?;
    let outer = hard_interventions(&input.data, &outer)?;
    let inner = hard_interventions(&input.data, &inner)?;
    let query = SimulateQuery::simultaneous_world(&outer, &inner, vec![outcome_id])
        .map_err(|e| py_err(antecedent::CausalError::from(e)))?;
    let (mean, route) = detach_catch(py, move || {
        let (source, ctx, _) = input.source()?;
        let (simulated, route) = source.simulate(query.target, &ctx)?;
        let SimulationOutcome::Counterfactual(values) = simulated else {
            return Err(outcome_mismatch());
        };
        let dense = antecedent::graph::DenseNodeId::from_raw(outcome_id.raw());
        let mean = values.outcome_mean(0, dense);
        if mean.n_used != mean.n_units {
            return Err(crate::value_err(
                "some units have no finite counterfactual outcome; their mean would average a \
                 different population",
            ));
        }
        Ok::<_, PyErr>((mean.mean, route))
    })?;
    answer(py, "simultaneous_counterfactual", mean.into_pyobject(py)?.into_any().unbind(), &route)
}

#[pymethods]
impl PyFittedGcm {
    /// Whether `antecedent.analyze` on this model's rows and graph refits exactly these
    /// mechanisms: the model was fitted with the standard candidate set and selection rule, on a
    /// supplied graph, with no caller-wrapped mechanism.
    #[getter]
    fn analyze_refits_this_model(&self) -> bool {
        self.accepted.is_none() && self.program.label() == MechanismProgram::standard().label()
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModelOperationAnswer>()?;
    m.add_function(wrap_pyfunction!(model_operation_arrow_strengths, m)?)?;
    m.add_function(wrap_pyfunction!(model_operation_population_contrast, m)?)?;
    m.add_function(wrap_pyfunction!(model_operation_do_sample, m)?)?;
    m.add_function(wrap_pyfunction!(model_operation_evaluation, m)?)?;
    m.add_function(wrap_pyfunction!(model_operation_interventional_mean, m)?)?;
    m.add_function(wrap_pyfunction!(model_operation_simultaneous_counterfactual, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_seed_restarts_the_stream_it_names() {
        for seed in [0, 1, 5, u64::MAX] {
            let mut stream = RngFactory::from_seed(seed)
                .stream_for(StreamDomain::Counterfactual, INTERVENTIONAL_MEAN_STREAM);
            let mut restarted = CausalRng::from_seed(stream_seed(seed, INTERVENTIONAL_MEAN_STREAM));
            for _ in 0..4 {
                assert_eq!(stream.next_u64(), restarted.next_u64());
            }
        }
    }
}
