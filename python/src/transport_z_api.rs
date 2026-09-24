//! Python bindings for the graph-specific point-only z-transport route.
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{ExecutionContext, Value, VariableId};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ZTransportQuery, ZTransportResult, bind_z_transport_catalog, identify_z_transport_surrogate,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::BTreeMap;
use std::sync::Arc;

fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn resolve(names: &[String], name: &str) -> PyResult<VariableId> {
    let index = names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| error(format!("unknown variable {name}")))?;
    Ok(VariableId::from_raw(u32::try_from(index).map_err(error)?))
}

#[pyclass(skip_from_py_object)]
struct ZTransportStage {
    result: ZTransportResult,
    graph: Admg,
    diagram: SelectionDiagram,
    query: ZTransportQuery,
}

#[pymethods]
impl ZTransportStage {
    #[getter]
    fn outcome(&self) -> &'static str {
        match &self.result {
            ZTransportResult::Identified(_) => "identified",
            ZTransportResult::NotCertified { .. } => "not_certified",
        }
    }
    #[getter]
    fn reason(&self) -> Option<&'static str> {
        match &self.result {
            ZTransportResult::Identified(_) => None,
            ZTransportResult::NotCertified { reason } => Some(reason),
        }
    }

    #[pyo3(signature=(catalog, laws, assignments, *, max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000, memory_bytes=None))]
    fn prepare_exact(
        &self,
        py: Python<'_>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        max_support_rows: usize,
        memory_bytes: Option<u64>,
    ) -> PyResult<PreparedZTransportStage> {
        let ZTransportResult::Identified(proof) = &self.result else {
            return Err(error(format!(
                "zTR refused: {}",
                self.reason().unwrap_or("not certified")
            )));
        };
        let catalog = parse_catalog(catalog, &self.graph)?;
        let data = parse_z_data(laws, &catalog, &self.graph, max_support_rows)?;
        let request = Assignment::from_pairs(
            assignments
                .into_iter()
                .map(|(name, value)| Ok((resolve(&self.graph.names, &name)?, Value::f64(value))))
                .collect::<PyResult<Vec<_>>>()?,
        );
        let diagram = self.diagram.clone();
        let query = self.query.clone();
        let proof = proof.as_ref().clone();
        let named_graph = self.graph.clone();
        crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory_bytes;
            let functional =
                bind_z_transport_catalog(&diagram, &query, &proof, &catalog).map_err(error)?;
            let inner = antecedent::StudyBuilder::z_transport(
                diagram,
                functional,
                data,
                request,
                ExactEvaluationLimits { operations: max_operations, depth: max_depth },
                &ctx,
            )
            .map_err(error)?;
            Ok(PreparedZTransportStage {
                inner,
                graph: named_graph,
                catalog,
                last: None,
                max_support_rows,
                memory_bytes,
            })
        })
    }
}

#[pyclass(skip_from_py_object)]
struct PreparedZTransportStage {
    inner: antecedent::PreparedZTransport,
    graph: Admg,
    catalog: antecedent_core::EvidenceCatalog,
    last: Option<antecedent::ZTransportResult>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl PreparedZTransportStage {
    fn estimate(&mut self, py: Python<'_>) -> PyResult<String> {
        let inner = self.inner.clone();
        let memory = self.memory_bytes;
        let result = crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory;
            inner.estimate(&ctx).map_err(error)
        })?;
        let names = &self.graph.names;
        let query = self.inner.functional().derivation().query();
        let distribution = result.distribution();
        let payload = serde_json::json!({
            "status":"available",
            "scope":"registered_surrogate_z_transport_sound_incomplete",
            "outcomes":query.outcomes.iter().map(|v| &names[v.as_usize()]).collect::<Vec<_>>(),
            "atoms":distribution.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "probabilities":distribution.probabilities.as_ref(),
            "factor_support":distribution.support.iter().map(|s| serde_json::json!({
                "expression":s.expression.raw(), "status":s.status, "denominator":s.denominator,
                "assignment":s.assignment.iter().map(|(v,x)| (names[v.as_usize()].as_str(), x.as_f64())).collect::<BTreeMap<_,_>>()
            })).collect::<Vec<_>>(),
            "interval":{"available":false,"reason":"no_interval_reported"}
        }).to_string();
        self.last = Some(result);
        Ok(payload)
    }

    fn refresh(&mut self, py: Python<'_>, laws: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = parse_z_data(laws, &self.catalog, &self.graph, self.max_support_rows)?;
        let inner = self.inner.clone();
        let memory = self.memory_bytes;
        self.inner = crate::detach_catch(py, move || {
            let mut ctx = ExecutionContext::production_default(0);
            ctx.memory.hard_limit_bytes = memory;
            inner.refresh(data, &ctx).map_err(error)
        })?;
        self.last = None;
        Ok(())
    }

    #[getter]
    fn interval_type(&self) -> &'static str {
        "no_interval_reported"
    }
}

#[pyfunction]
#[pyo3(signature=(graph, selections, source, target, outcomes, treatments, controllable, experiment_assignment))]
#[allow(clippy::too_many_arguments)]
fn identify_z_transport_stage(
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    source: String,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    controllable: Vec<String>,
    experiment_assignment: BTreeMap<String, f64>,
) -> PyResult<ZTransportStage> {
    let coordinates = |variables: Vec<String>| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(&graph.names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let experiment_assignment = experiment_assignment
        .into_iter()
        .map(|(name, value)| {
            Ok(antecedent_core::InterventionAssignment {
                variable: resolve(&graph.names, &name)?,
                value: Value::f64(value),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let query = ZTransportQuery {
        outcomes: coordinates(outcomes)?,
        treatments: coordinates(treatments)?,
        controllable: coordinates(controllable)?,
        experiment_assignment: experiment_assignment.into(),
        source: source.into(),
        target: target.into(),
    };
    let diagram =
        SelectionDiagram::try_new(graph.aligned_to_names(&graph.names)?, coordinates(selections)?)
            .map_err(error)?;
    let named_graph = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    let result = identify_z_transport_surrogate(
        &diagram,
        &query, // This implementation is a sound/incomplete registered specialization.
    )
    .map_err(error)?;
    Ok(ZTransportStage { result, graph: named_graph, diagram, query })
}

fn parse_z_data(
    laws: &Bound<'_, PyAny>,
    catalog: &antecedent_core::EvidenceCatalog,
    graph: &Admg,
    max_support_rows: usize,
) -> PyResult<ExactTransportData> {
    let mut tables = Vec::new();
    for table in laws.try_iter()? {
        let table = table?;
        let population: String = table.getattr("population")?.extract()?;
        let label: String = table.getattr("regime")?.extract()?;
        let regime = catalog
            .regimes
            .iter()
            .find(|r| {
                r.label.as_deref() == Some(label.as_str()) && r.population.as_ref() == population
            })
            .ok_or_else(|| error("exact zTR law names an unknown population/regime"))?;
        let axes: Vec<(String, Vec<f64>)> = table.getattr("axes")?.extract()?;
        let axes = axes
            .into_iter()
            .map(|(name, values)| {
                Ok(DiscreteAxis {
                    variable: resolve(&graph.names, &name)?,
                    values: values.into_iter().map(Value::f64).collect(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let interventions: Vec<(String, f64)> = table.getattr("interventions")?.extract()?;
        let interventions = interventions
            .into_iter()
            .map(|(name, value)| {
                Ok(InterventionAssignment {
                    variable: resolve(&graph.names, &name)?,
                    value: Value::f64(value),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let probabilities: Vec<f64> = table.getattr("probabilities")?.extract()?;
        let snapshot: String = table.getattr("snapshot_identity")?.extract()?;
        let absolute: f64 = table.getattr("absolute_tolerance")?.extract()?;
        let relative: f64 = table.getattr("relative_tolerance")?.extract()?;
        let mut law = ExactDiscreteLaw::try_new(
            population,
            regime.id,
            interventions,
            axes,
            probabilities,
            snapshot,
            LawTolerance { absolute, relative },
        )
        .map_err(error)?;
        if let Ok(counts) = table.getattr("empirical_counts") {
            if !counts.is_none() {
                law = law.with_empirical_counts(counts.extract()?).map_err(error)?;
            }
        }
        tables.push(law);
    }
    ExactTransportData::try_new(tables, max_support_rows).map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ZTransportStage>()?;
    module.add_class::<PreparedZTransportStage>()?;
    module.add_function(wrap_pyfunction!(identify_z_transport_stage, module)?)?;
    Ok(())
}
