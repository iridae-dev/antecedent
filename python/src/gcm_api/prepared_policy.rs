//! Prepared policy lifecycle over the retained GCM fit.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, Bound, Dag, FitCache, FittedGcm, HashMap, Mutex, Ordering, Py, PyAny, PyDict, PyFittedGcm,
    PyModule, PyModuleMethods, PyResult, Python, Selection, TableView, TabularData, VariableId,
    dag_from_edges, detach_catch, py_err, py_execution_context, pyclass, pymethods,
};

struct PolicyMaterialization {
    fitted: Arc<FittedGcm>,
    _retention: antecedent::execution::Retention,
}

/// Checked topology and program over owned observations, before statistical fitting.
/// Forks share successful fitted ownership; rebinding owns new rows and clears that fit.
#[pyclass(name = "_PreparedPolicyAnalysis", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyPreparedPolicyAnalysis {
    data: Arc<TabularData>,
    names: Arc<[String]>,
    graph: Dag,
    selection: Arc<Selection>,
    template: antecedent::gcm::PreparedPolicy,
    fitted: Arc<FitCache<PolicyMaterialization>>,
}

#[pymethods]
impl PyPreparedPolicyAnalysis {
    #[new]
    #[pyo3(signature = (names, columns, edges, policies, outcome, *, candidates=None, constraints=None, splits="shuffled", purge=0))]
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        policies: &str,
        outcome: &str,
        candidates: Option<Vec<String>>,
        constraints: Option<Bound<'_, PyDict>>,
        splits: &str,
        purge: usize,
    ) -> PyResult<Self> {
        let data = crate::tabular_from_py_columns(names.clone(), columns)?;
        data.schema().id_of(outcome).map_err(py_err)?;
        let selection = Selection::parse(candidates, constraints.as_ref(), &data, splits, purge)?;
        // Lower expressions and their dependency ordering now, without mechanisms or samples.
        let interventions = antecedent::io::policy_program_artifact::interventions_from_named_json(
            policies,
            &|name| data.schema().id_of(name).ok().map(VariableId::raw),
        )
        .map_err(crate::intervention_program_api::io_err)?;
        detach_catch(py, move || {
            let graph = dag_from_edges(&data, &edges)?;
            let topology =
                antecedent::gcm::CompiledCausalModel::compile(graph.clone()).map_err(py_err)?;
            let template = antecedent::gcm::PreparedPolicy::prepare(&topology, interventions)
                .map_err(py_err)?;
            Ok(Self {
                data: Arc::new(data),
                names: Arc::from(names),
                graph,
                selection: Arc::new(selection),
                template,
                fitted: Arc::new(FitCache::default()),
            })
        })
    }

    fn fork(&self) -> Self {
        self.clone()
    }

    fn bind(&self, names: Vec<String>, columns: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        if names.as_slice() != self.names.as_ref() {
            return Err(crate::value_err("policy observations must preserve variables and order"));
        }
        let data = crate::tabular_from_py_columns(names, columns)?;
        Ok(Self {
            data: Arc::new(data),
            names: Arc::clone(&self.names),
            graph: self.graph.clone(),
            selection: Arc::clone(&self.selection),
            template: self.template.clone(),
            fitted: Arc::new(FitCache::default()),
        })
    }

    #[getter]
    fn identity(&self) -> Option<String> {
        antecedent::io::policy_program_artifact::policy_program_identity(
            self.template.interventions(),
        )
        .ok()
        .map(|digest| antecedent::identity::to_hex(&digest))
    }

    fn export_program<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = antecedent::io::policy_program_artifact::encode_policy_program(
            self.template.interventions(),
        )
        .map_err(crate::intervention_program_api::io_err)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Deterministic physical work dimensions, without fitting or drawing.
    fn quote(&self, draws: usize) -> PyResult<HashMap<String, u64>> {
        let width =
            u64::try_from(self.names.len()).map_err(|error| crate::value_err(error.to_string()))?;
        let values = |rows: usize| -> PyResult<u64> {
            u64::try_from(rows)
                .ok()
                .and_then(|rows| rows.checked_mul(width))
                .and_then(|cells| cells.checked_mul(8))
                .ok_or_else(|| {
                    crate::CausalResourceError::new_err("policy work dimensions overflow")
                })
        };
        Ok(HashMap::from([
            ("fit_workspace_bytes".into(), values(self.data.row_count())?),
            ("draw_matrix_bytes".into(), values(draws)?),
        ]))
    }

    #[getter]
    fn fit_materialized(&self) -> bool {
        self.fitted.published.load(Ordering::Acquire)
    }

    #[pyo3(signature = (seed=1, threads=None))]
    fn materialize(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<(PyFittedGcm, crate::intervention_program_api::PyPreparedPolicy)> {
        let fitted = detach_catch(py, || {
            let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
            // Admission also checks cancellation on a reused fit, before publishing a run.
            let admission_error = |error: antecedent::execution::AdmissionRefusal| match error {
                antecedent::execution::AdmissionRefusal::Cancelled => {
                    py_err(antecedent::CausalError::Cancelled { stage: "policy_fit" })
                }
                error @ antecedent::execution::AdmissionRefusal::Refused(_) => {
                    crate::CausalResourceError::new_err(error.to_string())
                }
            };
            let _check = antecedent::execution::retain_bounded(&ctx, 0, std::time::Duration::ZERO)
                .map_err(admission_error)?;
            self.fitted.materialize(
                &ctx,
                || {
                    let bytes = (self.data.row_count() as u64)
                        .checked_mul(self.names.len() as u64)
                        .and_then(|cells| cells.checked_mul(8))
                        .ok_or_else(|| {
                            crate::CausalResourceError::new_err("policy fit workspace overflow")
                        })?;
                    let workspace = antecedent::execution::retain_bounded(
                        &ctx,
                        bytes,
                        std::time::Duration::ZERO,
                    )
                    .map_err(admission_error)?;
                    let selection = if self.selection.is_standard() {
                        None
                    } else {
                        Some((&self.selection.registry, &self.selection.options))
                    };
                    let fitted = antecedent::gcm::fit_gcm_polled(
                        self.graph.clone(),
                        &self.data,
                        selection,
                        &ctx,
                    )
                    .map_err(py_err)?;
                    let _completed =
                        antecedent::execution::retain_bounded(&ctx, 0, std::time::Duration::ZERO)
                            .map_err(admission_error)?;
                    Ok(PolicyMaterialization { fitted: Arc::new(fitted), _retention: workspace })
                },
                ctx.progress.is_some(),
                |fraction| {
                    if let Some(progress) = &ctx.progress {
                        progress.report(fraction, "policy_fit");
                    }
                },
                || {},
            )
        })?;
        let program = self.selection.program()?.with_prefit(Arc::clone(&fitted.fitted), &self.data);
        let model = PyFittedGcm {
            inner: Arc::clone(&fitted.fitted),
            names: Arc::clone(&self.names),
            data: Arc::clone(&self.data),
            program: Arc::new(program),
            accepted: None,
            last_route: Arc::new(Mutex::new(None)),
        };
        let core = self.template.rebind(&model.inner.model).map_err(py_err)?;
        let native_model = Py::new(py, model)?;
        let policy = crate::intervention_program_api::PyPreparedPolicy::from_template(
            native_model.clone_ref(py),
            core,
            self.identity(),
        );
        // Return a second header on the same native storage so result routes are independent.
        let model = native_model.bind(py).borrow();
        let header = PyFittedGcm {
            inner: Arc::clone(&model.inner),
            names: Arc::clone(&model.names),
            data: Arc::clone(&model.data),
            program: Arc::clone(&model.program),
            accepted: None,
            last_route: Arc::clone(&model.last_route),
        };
        Ok((header, policy))
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreparedPolicyAnalysis>()
}
