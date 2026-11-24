//! Compiled transport routes for Python: the augmented transported response grid on supplied
//! nuisances and the pooled-differential measurement transport.
//!
//! Each entry compiles the question into its program family (`TransportResponseGrid`,
//! `PooledDifferentialTransport`) and returns a handle whose `execute` runs the same kernel the
//! direct entries run; Python owns no number.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::compiled_measurement_transport::{
    CompiledPooledDifferentialTransport, PooledDifferentialTransportRequest,
};
use antecedent::analysis::transport_response_grid::{
    CompiledTransportResponseGrid, TransportResponseGridRequest,
};
use antecedent::analysis::transport_supplied_nuisance::TransportResponseGridColumns;
use antecedent::analysis::transported_measurement::{PooledRate, PopulationValidation};
use antecedent::prelude::{TabularData, VariableId};
use pyo3::prelude::*;

use crate::graphs::Admg;
use crate::pooled_measurement_api::PooledDifferentialSection;
use crate::transport_interference_api::{ResponseArgs, transport_query};

/// A compiled augmented transported response grid.
#[pyclass(name = "NativeCompiledTransportResponseGrid", skip_from_py_object)]
pub(crate) struct NativeCompiledTransportResponseGrid {
    inner: Arc<CompiledTransportResponseGrid>,
}

#[pymethods]
impl NativeCompiledTransportResponseGrid {
    /// The plan route of the compiled program.
    #[getter]
    fn route(&self) -> String {
        self.inner.plan_view().route().as_str().to_owned()
    }

    /// Identity of the lowered program, as lowercase hex.
    #[getter]
    fn program_identity(&self) -> String {
        self.inner.lowered().identity().to_hex()
    }

    /// Evaluate the grid; the result is the JSON the direct entry returned.
    #[pyo3(signature = (*, seed=0, cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let mut ctx = crate::py_execution_context(seed, 1);
        if cancel.is_some() {
            crate::apply_cancel(&mut ctx, cancel);
        }
        let inner = Arc::clone(&self.inner);
        let run = crate::detach_catch(py, move || inner.execute(&ctx).map_err(crate::py_err))?;
        let estimate = &run.estimate;
        Ok(serde_json::json!({
            "grid": estimate.grid.as_ref(), "mean": estimate.mean.as_ref(),
            "source_effective_sample_size": estimate.source_effective_sample_size.as_ref(),
            "selection_overlap": {
                "probability_min": estimate.selection_overlap.probability_min,
                "probability_max": estimate.selection_overlap.probability_max,
                "effective_sample_size": estimate.selection_overlap.effective_sample_size,
                "extreme_weight_count": estimate.selection_overlap.extreme_weight_count,
            },
            "claim": run.claim,
            "calibration": run.calibration,
            "program_identity": run.lowered_program_identity.to_hex(),
        })
        .to_string())
    }
}

/// Compile an augmented transported response grid: the transport question on the named
/// selection diagram, the columns of `data` (one per graph node, in node order) and the
/// supplied regressions and response weights, named by column.
#[pyfunction]
#[pyo3(signature = (
    graph, selections, source_population, target_population, source_experiments,
    kind, treatments, outcomes, columns, trial, selection_probability, observed_regression,
    grid_regressions, response_weights, *, grid=None, at=None, direction=None, order=1,
    scale="identity", weighting="observed", seed=0
))]
#[allow(clippy::too_many_arguments)]
fn compile_transport_response_grid(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    source_population: String,
    target_population: String,
    source_experiments: Vec<String>,
    kind: String,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    columns: Vec<Vec<f64>>,
    trial: String,
    selection_probability: String,
    observed_regression: String,
    grid_regressions: Vec<String>,
    response_weights: Vec<String>,
    grid: Option<Vec<f64>>,
    at: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    order: u8,
    scale: &str,
    weighting: &str,
    seed: u64,
) -> PyResult<NativeCompiledTransportResponseGrid> {
    let names = graph.names.clone();
    if columns.len() != names.len() {
        return Err(crate::value_err(format!(
            "the data has {} columns and the graph {} nodes; supply one column per node, in \
             node order",
            columns.len(),
            names.len()
        )));
    }
    let id = |name: &str| -> PyResult<VariableId> {
        names
            .iter()
            .position(|candidate| candidate == name)
            .and_then(|index| u32::try_from(index).ok())
            .map(VariableId::from_raw)
            .ok_or_else(|| crate::value_err(format!("column {name:?} is not a graph node")))
    };
    let resolve = |list: &[String]| -> PyResult<Arc<[VariableId]>> {
        list.iter().map(|name| id(name)).collect::<PyResult<Vec<_>>>().map(Into::into)
    };
    let query = transport_query(
        ResponseArgs {
            kind,
            treatments,
            outcomes,
            grid,
            at,
            direction,
            order,
            scale: scale.to_owned(),
            weighting: weighting.to_owned(),
        },
        source_population,
        target_population,
        &source_experiments,
        resolve,
    )?;
    let diagram = antecedent::graph::SelectionDiagram::try_new(
        graph.aligned_to_names(&names)?,
        resolve(&selections)?,
    )
    .map_err(crate::py_err)?;
    let pairs: Vec<(&str, &[f64])> =
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)).collect();
    let data = TabularData::from_f64_columns(pairs).map_err(crate::py_err)?;
    let selection = id(&selection_probability)?;
    let trial_spec = antecedent::TransportTrialSpec {
        trial: id(&trial)?,
        selection_probability: selection,
        // The grid reads no treatment probability; the selection column stands in, unread.
        treatment_probability: selection,
    };
    let nuisance = TransportResponseGridColumns::new(
        id(&observed_regression)?,
        resolve(&grid_regressions)?,
        resolve(&response_weights)?,
    );
    let request = TransportResponseGridRequest::new(diagram, query, data, trial_spec, nuisance);
    let ctx = crate::py_execution_context(seed, 1);
    let compiled = crate::detach_catch(py, move || {
        antecedent::Study::compile_transport_response_grid(request, &ctx).map_err(crate::py_err)
    })?;
    Ok(NativeCompiledTransportResponseGrid { inner: Arc::new(compiled) })
}

/// A compiled pooled-differential measurement transport.
#[pyclass(name = "NativeCompiledPooledDifferentialTransport", skip_from_py_object)]
pub(crate) struct NativeCompiledPooledDifferentialTransport {
    inner: Arc<CompiledPooledDifferentialTransport>,
}

const fn rate(value: &PooledRate) -> (f64, f64, f64, f64, f64) {
    (
        value.mean_logit,
        value.mean_variance,
        value.between_population_variance,
        value.target_rate,
        value.target_rate_sd,
    )
}

#[pymethods]
impl NativeCompiledPooledDifferentialTransport {
    /// The plan route of the compiled program.
    #[getter]
    fn route(&self) -> String {
        self.inner.plan_view().route().as_str().to_owned()
    }

    /// Identity of the lowered program, as lowercase hex.
    #[getter]
    fn program_identity(&self) -> String {
        self.inner.lowered().identity().to_hex()
    }

    /// Pool the rates and draw the corrected prevalence under `seed` and `threads`, the
    /// execution context the direct entry builds.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<PooledDifferentialSection> {
        if threads == Some(0) {
            return Err(crate::value_err("threads must be positive"));
        }
        let ctx = antecedent::prelude::ExecutionContext::production(
            seed,
            threads.unwrap_or_else(antecedent::execution::default_user_threads),
        );
        let inner = Arc::clone(&self.inner);
        crate::detach_catch(py, move || {
            let fit = inner.execute(&ctx).map_err(crate::py_err)?.correction;
            Ok(PooledDifferentialSection {
                target_population: fit.target_population.clone(),
                prevalence: fit.prevalence,
                interval: fit.interval,
                sensitivity: rate(&fit.sensitivity),
                specificity: rate(&fit.specificity),
                discarded_draws: fit.discarded_draws,
                truncated_draws: fit.truncated_draws,
                draws: fit.draws,
                calibration: fit.calibration.to_owned(),
            })
        })
    }
}

type Counts = (u64, u64, u64, u64);

fn validation(population: String, counts: Counts) -> PopulationValidation {
    PopulationValidation {
        population,
        true_positive: counts.0,
        false_negative: counts.1,
        true_negative: counts.2,
        false_positive: counts.3,
    }
}

/// Compile a pooled-differential measurement transport of the target's recorded prevalence.
#[pyfunction]
#[pyo3(signature = (sources, target_population, observed_positive, observed_total, *,
    target_validation=None, draws=4000))]
fn compile_pooled_differential_transport(
    sources: Vec<(String, u64, u64, u64, u64)>,
    target_population: String,
    observed_positive: u64,
    observed_total: u64,
    target_validation: Option<Counts>,
    draws: usize,
) -> PyResult<NativeCompiledPooledDifferentialTransport> {
    let sources: Vec<PopulationValidation> = sources
        .into_iter()
        .map(|(population, tp, fn_, tn, fp)| validation(population, (tp, fn_, tn, fp)))
        .collect();
    let mut request = PooledDifferentialTransportRequest::new(
        sources,
        target_population.clone(),
        observed_positive,
        observed_total,
        draws,
    );
    if let Some(counts) = target_validation {
        request = request.with_target_validation(validation(target_population, counts));
    }
    let compiled =
        antecedent::Study::compile_pooled_differential_transport(request).map_err(crate::py_err)?;
    Ok(NativeCompiledPooledDifferentialTransport { inner: Arc::new(compiled) })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeCompiledTransportResponseGrid>()?;
    m.add_class::<NativeCompiledPooledDifferentialTransport>()?;
    m.add_function(wrap_pyfunction!(compile_transport_response_grid, m)?)?;
    m.add_function(wrap_pyfunction!(compile_pooled_differential_transport, m)?)
}
