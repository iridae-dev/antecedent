//! GCM mechanism selection, constraints and fit entry points.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AcceptedGraph, Arc, Bound, Dag, MechanismAssignment, MechanismFamily, MechanismProgram,
    MechanismRegistry, NodeFamilyConstraint, PyAny, PyAnyMethods, PyDict, PyDictMethods,
    PyFittedGcm, PyModule, PyModuleMethods, PyRef, PyResult, Python, SelectionOptions, TableView,
    TabularData, ValidationSplits, VariableId, dag_from_edges, detach_catch, py_err, py_msg,
    pyfunction, var_name, wrap_pyfunction,
};

fn family_ids(families: &[MechanismFamily]) -> Vec<&'static str> {
    families.iter().map(|f| f.id()).collect()
}

pub(super) fn selection_record<'py>(
    py: Python<'py>,
    names: &[String],
    a: &MechanismAssignment,
) -> PyResult<Bound<'py, PyDict>> {
    let record = PyDict::new(py);
    record.set_item("variable", var_name(names, a.variable))?;
    record.set_item("provenance_id", "estimate.gcm.fit")?;
    record.set_item("selected", a.selected.id())?;
    record.set_item("typing", a.typing.id())?;
    record.set_item("noise_inference", a.selected.noise_inference_mode().id())?;
    record.set_item("attribution_compatible", a.selected.attribution_compatible())?;
    let provenance = &a.provenance;
    record.set_item("rule", provenance.rule.id())?;
    record.set_item("splits", provenance.splits.id())?;
    record.set_item(
        "purge",
        match provenance.splits {
            ValidationSplits::TimeOrdered { purge } => Some(purge),
            _ => None,
        },
    )?;
    record.set_item("folds", provenance.folds)?;
    record.set_item("n_rows", provenance.n_rows)?;
    record.set_item("seed", provenance.seed)?;
    record.set_item("candidate_set", family_ids(&provenance.candidate_set))?;
    let constraint = match &provenance.constraint {
        None => None,
        Some(c) => {
            let d = PyDict::new(py);
            d.set_item("allowed", c.allowed.as_deref().map(family_ids))?;
            d.set_item("excluded", family_ids(&c.excluded))?;
            d.set_item("required", c.required.map(MechanismFamily::id))?;
            Some(d)
        }
    };
    record.set_item("constraint", constraint)?;
    let candidates = a
        .candidates
        .iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("family", c.family.id())?;
            d.set_item("score", c.score)?;
            d.set_item("held_out_log_likelihood", c.held_out_log_likelihood)?;
            d.set_item("complexity_penalty", c.complexity_penalty)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    record.set_item("candidates", candidates)?;
    let failed = a
        .failed_families
        .iter()
        .map(|(family, reason)| {
            let d = PyDict::new(py);
            d.set_item("family", family.id())?;
            d.set_item("reason", reason)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    record.set_item("failed", failed)?;
    Ok(record)
}

/// Families a candidate token list names for one node constraint field.
fn constraint_families(tokens: &[String]) -> PyResult<Vec<MechanismFamily>> {
    let mut out: Vec<MechanismFamily> = Vec::new();
    for token in tokens {
        for family in antecedent::gcm::candidate_families(token).map_err(py_err)? {
            if !out.contains(&family) {
                out.push(family);
            }
        }
    }
    Ok(out)
}

/// Per-node constraints from `{node: {"allowed": [...], "excluded": [...], "required": id}}`,
/// in node-name order so the resolved options do not depend on dict iteration order.
fn parse_constraints(
    constraints: Option<&Bound<'_, PyDict>>,
    data: &TabularData,
) -> PyResult<Vec<(VariableId, NodeFamilyConstraint)>> {
    let mut entries: Vec<(String, Bound<'_, PyDict>)> = Vec::new();
    if let Some(map) = constraints {
        for (node, spec) in map.iter() {
            entries.push((node.extract()?, spec.cast::<PyDict>().map_err(py_msg)?.clone()));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::with_capacity(entries.len());
    for (node, spec) in entries {
        let variable = data.schema().id_of(&node).map_err(py_err)?;
        let mut constraint = NodeFamilyConstraint::default();
        for key in spec.keys() {
            let key: String = key.extract()?;
            let value = spec
                .get_item(&key)?
                .ok_or_else(|| py_msg(format!("constraint for `{node}`: missing `{key}`")))?;
            match key.as_str() {
                "allowed" => {
                    let tokens: Vec<String> = value.extract()?;
                    constraint.allowed = Some(Arc::from(constraint_families(&tokens)?));
                }
                "excluded" => {
                    let tokens: Vec<String> = value.extract()?;
                    constraint.excluded = Arc::from(constraint_families(&tokens)?);
                }
                "required" => {
                    let token: String = value.extract()?;
                    let families = antecedent::gcm::candidate_families(&token).map_err(py_err)?;
                    let [family] = families.as_slice() else {
                        return Err(py_msg(format!(
                            "constraint for `{node}`: `required` must name one family id, \
                             `{token}` names {}",
                            families.len()
                        )));
                    };
                    constraint.required = Some(*family);
                }
                other => {
                    return Err(py_msg(format!(
                        "constraint for `{node}`: unknown key `{other}` (expected allowed, \
                         excluded or required)"
                    )));
                }
            }
        }
        out.push((variable, constraint));
    }
    Ok(out)
}

fn parse_splits(splits: &str, purge: usize) -> PyResult<ValidationSplits> {
    match (splits, purge) {
        ("shuffled", 0) => Ok(ValidationSplits::Shuffled),
        ("clustered", 0) => Ok(ValidationSplits::Clustered),
        ("time_ordered", purge) => Ok(ValidationSplits::TimeOrdered { purge }),
        ("shuffled" | "clustered", _) => {
            Err(py_msg(format!("purge applies only to splits=\"time_ordered\", not `{splits}`")))
        }
        (other, _) => Err(py_msg(format!(
            "unknown splits `{other}` (expected shuffled, time_ordered or clustered)"
        ))),
    }
}

/// Fit a graphical causal model; return a reusable [`FittedGcm`].
///
/// With no keyword it fits the standard candidate set. `candidates` (family ids or
/// the aliases `linear`, `logistic`, `additive_spline`, `interactions`, `random_forest`,
/// `gradient_boosted_trees`, `tree_ensembles`, `tree_ensemble_classifier`) replaces the
/// candidate set; `constraints` narrows it per node; `splits` declares how rows are
/// cut for the held-out scores (`shuffled`, `time_ordered` with an optional `purge`,
/// or `clustered` by the `UnitId` column). The fit runs serially; `threads` sizes the execution
/// context, whose cancellation (Ctrl-C) the fit polls.
#[pyfunction]
#[pyo3(
    name = "fit_gcm",
    signature = (names, columns, edges, *, threads=None, candidates=None, constraints=None, splits="shuffled", purge=0)
)]
#[allow(clippy::too_many_arguments, reason = "keyword-only selection options of one Python call")]
fn fit_gcm_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    threads: Option<u32>,
    candidates: Option<Vec<String>>,
    constraints: Option<Bound<'_, PyDict>>,
    splits: &str,
    purge: usize,
) -> PyResult<PyFittedGcm> {
    let data = crate::tabular_from_py_columns(names.clone(), columns)?;
    let selection = Selection::parse(candidates, constraints.as_ref(), &data, splits, purge)?;
    let ctx = crate::py_execution_context_cancel(0, crate::resolve_user_threads(threads), None);
    detach_catch(py, move || {
        let g = dag_from_edges(&data, &edges)?;
        selection.fit(g, data, names, None, &ctx)
    })
}

/// The candidate set and selection rule of one fit, and the mechanism program that states them.
pub(super) struct Selection {
    candidates: Option<Vec<String>>,
    pub(super) registry: MechanismRegistry,
    pub(super) options: SelectionOptions,
}

impl Selection {
    pub(super) fn parse(
        candidates: Option<Vec<String>>,
        constraints: Option<&Bound<'_, PyDict>>,
        data: &TabularData,
        splits: &str,
        purge: usize,
    ) -> PyResult<Self> {
        let registry = match &candidates {
            None => MechanismRegistry::standard(),
            Some(tokens) => {
                let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
                antecedent::gcm::registry_from_candidates(&tokens).map_err(py_err)?
            }
        };
        let options = SelectionOptions {
            constraints: parse_constraints(constraints, data)?,
            splits: parse_splits(splits, purge)?,
        };
        Ok(Self { candidates, registry, options })
    }

    pub(super) fn is_standard(&self) -> bool {
        self.candidates.is_none() && self.options == SelectionOptions::default()
    }

    pub(super) fn program(&self) -> PyResult<MechanismProgram> {
        match &self.candidates {
            None if self.is_standard() => Ok(MechanismProgram::standard()),
            None => Ok(MechanismProgram::standard().with_options(self.options.clone())),
            Some(tokens) => {
                let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
                MechanismProgram::from_candidates(&tokens, self.options.clone()).map_err(py_err)
            }
        }
    }

    /// Fit on `graph` and keep the program, so every later route reuses this fit.
    fn fit(
        &self,
        graph: Dag,
        data: TabularData,
        names: Vec<String>,
        accepted: Option<AcceptedGraph>,
        ctx: &antecedent::prelude::ExecutionContext,
    ) -> PyResult<PyFittedGcm> {
        let selection =
            if self.is_standard() { None } else { Some((&self.registry, &self.options)) };
        let fitted =
            antecedent::gcm::fit_gcm_polled(graph, &data, selection, ctx).map_err(py_err)?;
        Ok(PyFittedGcm::new(fitted, names, data, self.program()?, accepted))
    }
}

/// Fit a graphical causal model on the structure a compiled discovery returned.
///
/// `discovery` is a result of a `discover_*` entry point (PC, GES, LiNGAM, NOTEARS, ...).
/// Its route accepts the structure through the review gate as a fully oriented DAG (pending
/// directed edges are accepted; undirected or circle marks refuse), and the model is fitted
/// on that accepted DAG. Every attribution and simulation on the returned model states the
/// discovery's provenance (method, inference, query, data snapshot, program and claim) in
/// its contract. Returns `(fitted, edges)`, `edges` the accepted directed edges by name.
#[pyfunction]
#[pyo3(
    name = "fit_gcm_accepted",
    signature = (names, columns, discovery, *, threads=None, candidates=None, constraints=None, splits="shuffled", purge=0)
)]
#[allow(clippy::too_many_arguments, reason = "keyword-only selection options of one Python call")]
fn fit_gcm_accepted_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    discovery: PyRef<'_, crate::discovery_api::PcmciDiscoveryResult>,
    threads: Option<u32>,
    candidates: Option<Vec<String>>,
    constraints: Option<Bound<'_, PyDict>>,
    splits: &str,
    purge: usize,
) -> PyResult<(PyFittedGcm, Vec<(String, String)>)> {
    let route = discovery
        .route
        .clone()
        .ok_or_else(|| crate::value_err("this discovery result has no compiled route to accept"))?;
    drop(discovery);
    let data = crate::tabular_from_py_columns(names.clone(), columns)?;
    let selection = Selection::parse(candidates, constraints.as_ref(), &data, splits, purge)?;
    let ctx = crate::py_execution_context_cancel(0, crate::resolve_user_threads(threads), None);
    detach_catch(py, move || {
        let accepted = route.execution.accept_dag(&route.ctx).map_err(py_err)?;
        let dag = accepted
            .as_dag()
            .cloned()
            .ok_or_else(|| crate::value_err("the accepted structure is not a DAG"))?;
        if dag.node_count() != data.schema().len() {
            return Err(crate::value_err(format!(
                "the discovery ran on {} variables but {} columns were supplied; pass the \
                 columns the discovery read, in the same order",
                dag.node_count(),
                data.schema().len()
            )));
        }
        let edges: Vec<(String, String)> = dag
            .directed_edge_pairs_sorted()
            .into_iter()
            .map(|(from, to)| {
                (
                    var_name(&names, VariableId::from_raw(from)),
                    var_name(&names, VariableId::from_raw(to)),
                )
            })
            .collect();
        let fitted = selection.fit(dag, data, names, Some(accepted), &ctx)?;
        Ok((fitted, edges))
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_gcm_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gcm_accepted_py, m)?)?;
    Ok(())
}
