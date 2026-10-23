//! Compiled structure-discovery, attribution and simulation routes for the Python bindings.
//!
//! Every discovery, fitted-model attribution and simulation binding runs through
//! [`antecedent::CompiledExecution`]: the study is compiled to a lowered program,
//! prepared and estimated once, and the answer the engine returned is handed back
//! unchanged together with a [`PyCompiledRoute`] that states the contract the
//! execution ran under, the claim sealed over it and the cheap inspection record.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent::{
    AcceptedGraph, AttributionOutcome, CausalError as RustCausalError, CompiledExecution,
    DiscoveredStructure, DiscoverySpec, MechanismProgram, SimulationOutcome, Study,
};
use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, ExecutionContext, SimulateQuery,
    SimulationTarget, VariableId,
};
use antecedent_data::{MultiEnvironmentData, TabularData, TimeSeriesData};
use antecedent_graph::Dag;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::py_err;

/// Provenance entries of an accepted structure, by key.
type Provenance = HashMap<String, String>;

/// The data a structure search reads.
pub(crate) enum DiscoverInput {
    /// Static rows.
    Tabular(TabularData),
    /// One series.
    Series(TimeSeriesData),
    /// The units of a panel, each its own series.
    Panel(Vec<TimeSeriesData>),
    /// Several environments' series.
    MultiEnv(MultiEnvironmentData),
}

/// The structure a fitted-model route fits its mechanisms on.
pub(crate) enum ModelGraph {
    /// A supplied DAG.
    Dag(Dag),
    /// A DAG accepted from a compiled discovery, with its provenance.
    Accepted(AcceptedGraph),
}

/// An executed compiled route and the context it ran under.
#[derive(Clone)]
pub(crate) struct RouteRecord {
    pub(crate) execution: Arc<CompiledExecution>,
    pub(crate) ctx: ExecutionContext,
}

impl std::fmt::Debug for RouteRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouteRecord")
            .field("program", &self.execution.lowered_identity().map(|d| d.to_hex()))
            .finish_non_exhaustive()
    }
}

/// Compile, prepare and run the structure search `spec` states over `input`.
///
/// Returns the search engine's own result and the executed route.
///
/// # Errors
///
/// Build, compile, preparation and search refusals.
pub(crate) fn run_discovery(
    spec: DiscoverySpec,
    input: DiscoverInput,
    variables: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<(DiscoveredStructure, RouteRecord), RustCausalError> {
    let builder = match input {
        DiscoverInput::Tabular(data) => Study::tabular(data),
        DiscoverInput::Series(series) => Study::series(series),
        DiscoverInput::Panel(units) => Study::panel(antecedent::panel_of_units(units)?),
        DiscoverInput::MultiEnv(multi) => Study::series_multi(multi),
    };
    let study = builder.discovery(spec, variables.to_vec()).build()?;
    let execution = CompiledExecution::run(&study, ctx)?;
    let structure =
        execution.result().discovery().map(|payload| payload.structure.clone()).ok_or_else(
            || RustCausalError::Compile {
                message: "a compiled structure search returned no structure".to_owned(),
            },
        )?;
    Ok((structure, RouteRecord { execution: Arc::new(execution), ctx: ctx.clone() }))
}

/// Compile, prepare and run the attribution or simulation `query` on mechanisms
/// `program` fits on `graph`.
///
/// # Errors
///
/// Build, compile, preparation, fit and engine refusals.
pub(crate) fn run_model(
    data: TabularData,
    graph: ModelGraph,
    query: CausalQuery,
    program: MechanismProgram,
    ctx: &ExecutionContext,
) -> Result<RouteRecord, RustCausalError> {
    let builder = Study::tabular(data);
    let builder = match graph {
        ModelGraph::Dag(dag) => builder.graph(dag),
        ModelGraph::Accepted(accepted) => builder.graph(accepted),
    };
    let study = builder.query(query).mechanisms(program).build()?;
    let execution = CompiledExecution::run(&study, ctx)?;
    Ok(RouteRecord { execution: Arc::new(execution), ctx: ctx.clone() })
}

/// What a fitted-model route runs on: the rows, the structure the mechanisms are
/// fitted on and the program that fits them (with the fit already made on those rows).
#[derive(Clone)]
pub(crate) struct ModelSource {
    pub(crate) data: Arc<TabularData>,
    pub(crate) dag: Dag,
    /// The structure accepted from a compiled discovery, when the model was fitted on one;
    /// its provenance reaches every contract of a route run on it.
    pub(crate) accepted: Option<AcceptedGraph>,
    pub(crate) program: MechanismProgram,
}

impl ModelSource {
    fn graph(&self) -> ModelGraph {
        match &self.accepted {
            Some(accepted) => ModelGraph::Accepted(accepted.clone()),
            None => ModelGraph::Dag(self.dag.clone()),
        }
    }

    /// Run the attribution `target` states through the compiled route.
    pub(crate) fn attribute(
        &self,
        target: AttributeTarget,
        ctx: &ExecutionContext,
    ) -> PyResult<(AttributionOutcome, RouteRecord)> {
        let query = CausalQuery::Attribute(AttributeQuery::new(target));
        let record =
            run_model((*self.data).clone(), self.graph(), query, self.program.clone(), ctx)
                .map_err(py_err)?;
        let outcome = record
            .execution
            .result()
            .attribution()
            .map(|payload| payload.outcome.clone())
            .ok_or_else(|| crate::value_err("the compiled attribution returned no attribution"))?;
        Ok((outcome, record))
    }

    /// Run the simulation `target` states through the compiled route.
    pub(crate) fn simulate(
        &self,
        target: SimulationTarget,
        ctx: &ExecutionContext,
    ) -> PyResult<(SimulationOutcome, RouteRecord)> {
        let query = CausalQuery::Simulate(SimulateQuery::new(target));
        let record =
            run_model((*self.data).clone(), self.graph(), query, self.program.clone(), ctx)
                .map_err(py_err)?;
        let outcome = record
            .execution
            .result()
            .simulation()
            .map(|payload| payload.outcome.clone())
            .ok_or_else(|| crate::value_err("the compiled simulation returned no simulation"))?;
        Ok((outcome, record))
    }
}

/// The refusal for a route that returned another outcome than its target states.
pub(crate) fn outcome_mismatch() -> PyErr {
    crate::value_err("the compiled route returned an outcome its target does not state")
}

fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(64), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// The structure-provenance keys of a contract, prefixed `structure_provenance.`.
fn provenance_map(provenance: &antecedent::StructureProvenance) -> HashMap<String, String> {
    let mut out = HashMap::new();
    out.insert("method".to_owned(), provenance.method.to_string());
    out.insert("inference".to_owned(), provenance.inference.to_string());
    out.insert("discovered_class".to_owned(), provenance.discovered_class.to_string());
    out.insert("query".to_owned(), hex(&provenance.query));
    out.insert("search_bounds".to_owned(), provenance.search_bounds.to_string());
    out.insert("data_snapshot".to_owned(), hex(&provenance.data_snapshot));
    out.insert("program".to_owned(), hex(&provenance.program));
    out.insert("claim".to_owned(), hex(&provenance.claim));
    out.insert("review".to_owned(), provenance.review.to_string());
    out.insert("resolved_marks".to_owned(), provenance.resolved_marks.to_string());
    out.insert("digest".to_owned(), hex(&provenance.digest()));
    out
}

impl RouteRecord {
    fn contract_map(&self, contract: &antecedent::CausalContract) -> HashMap<String, String> {
        let prepared = self.execution.prepared();
        let mut out = crate::prepared_api::contract_to_map(
            contract,
            prepared.schema(),
            prepared.query(),
            prepared.population_registry(),
        );
        out.insert("query_kind".to_owned(), contract.query_kind.to_string());
        out.insert("inference".to_owned(), contract.inference.to_string());
        if let Some(identity) = self.execution.lowered_identity() {
            out.insert("lowered_program".to_owned(), identity.to_hex());
        }
        if let Some(provenance) = &contract.structure_provenance {
            for (key, value) in provenance_map(provenance) {
                out.insert(format!("structure_provenance.{key}"), value);
            }
        }
        out
    }

    /// The executed contract as a flat mapping.
    pub(crate) fn contract(&self) -> PyResult<HashMap<String, String>> {
        let (contract, _) = self.execution.contract_and_claim(&self.ctx).map_err(py_err)?;
        Ok(self.contract_map(&contract))
    }

    /// The claim sealed over the execution as a flat mapping.
    pub(crate) fn claim(&self) -> PyResult<HashMap<String, String>> {
        let (_, claim) = self.execution.contract_and_claim(&self.ctx).map_err(py_err)?;
        let mut out = HashMap::new();
        out.insert("claim_id".to_owned(), claim.claim_id.to_hex());
        out.insert("kind".to_owned(), claim.kind.as_str().to_owned());
        if let Some(value) = claim.value {
            out.insert("value".to_owned(), value.to_string());
        }
        out.insert("calibration_status".to_owned(), claim.calibration.status.to_string());
        if let Some(reason) = &claim.calibration.reason {
            out.insert("calibration_reason".to_owned(), reason.to_string());
        }
        Ok(out)
    }

    /// The cheap inspection record as a flat mapping.
    pub(crate) fn inspect(&self) -> PyResult<HashMap<String, String>> {
        let contract = self.execution.inspect().map_err(py_err)?;
        Ok(self.contract_map(&contract))
    }

    /// The variable names of the study's schema, in column order.
    fn names(&self) -> Vec<String> {
        self.execution
            .prepared()
            .schema()
            .variables()
            .iter()
            .map(|variable| variable.name.to_string())
            .collect()
    }
}

/// An executed compiled route: the contract it ran under, its claim and its inspection record.
#[pyclass(name = "CompiledRoute", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyCompiledRoute {
    pub(crate) record: RouteRecord,
}

impl PyCompiledRoute {
    pub(crate) fn new(record: RouteRecord) -> Self {
        Self { record }
    }
}

#[pymethods]
impl PyCompiledRoute {
    /// The contract the execution ran under (identities, reasoning slots, query kind,
    /// structure source, inference, and the structure provenance when the structure was
    /// accepted from a compiled discovery).
    #[getter]
    fn contract(&self) -> PyResult<HashMap<String, String>> {
        self.record.contract()
    }

    /// The claim sealed over the execution.
    #[getter]
    fn claim(&self) -> PyResult<HashMap<String, String>> {
        self.record.claim()
    }

    /// Cheap inspection of the compiled study, without identification products.
    fn inspect(&self) -> PyResult<HashMap<String, String>> {
        self.record.inspect()
    }

    /// Hex identity of the lowered program the execution ran.
    #[getter]
    fn program_identity(&self) -> Option<String> {
        self.record.execution.lowered_identity().map(|digest| digest.to_hex())
    }

    /// The provenance of the structure the study ran on, when it was accepted from a
    /// compiled discovery; `None` for a supplied structure or a discovery itself.
    fn structure_provenance(&self) -> PyResult<Option<HashMap<String, String>>> {
        let (contract, _) =
            self.record.execution.contract_and_claim(&self.record.ctx).map_err(py_err)?;
        Ok(contract.structure_provenance.as_deref().map(provenance_map))
    }

    /// Accept a discovered structure through its review gate and return the graph.
    ///
    /// `accept_discovered=True` accepts pending directed edges; CPDAG and PAG marks stay
    /// class information. Returns `(graph, provenance)`: the provenance the accepted
    /// structure carries into every study prepared on it.
    #[pyo3(signature = (*, accept_discovered=true))]
    fn accepted_graph(
        &self,
        py: Python<'_>,
        accept_discovered: bool,
    ) -> PyResult<(Py<PyAny>, Option<Provenance>)> {
        let record = self.record.clone();
        let accepted = crate::detach_catch(py, move || {
            record.execution.accept(accept_discovered, &record.ctx).map_err(py_err)
        })?;
        let provenance = accepted.provenance().map(provenance_map);
        let names = self.record.names();
        let graph = crate::discovery_api::accepted_graph_object(py, &accepted, &names)?;
        Ok((graph, provenance))
    }

    fn __repr__(&self) -> String {
        format!("CompiledRoute(program={})", self.program_identity().unwrap_or_default())
    }
}

/// The Python handle of a result's compiled route, `None` for a result built without one.
pub(crate) type RouteHandle = Option<Py<PyCompiledRoute>>;

/// The `route` a result gains from its compiled route.
pub(crate) fn route_object(py: Python<'_>, route: Option<&RouteRecord>) -> PyResult<RouteHandle> {
    route.map(|record| Py::new(py, PyCompiledRoute::new(record.clone()))).transpose()
}

/// The contract of a result's route, refusing a result built without one.
pub(crate) fn require_route(route: Option<&RouteRecord>) -> PyResult<&RouteRecord> {
    route.ok_or_else(|| {
        crate::value_err("this result was not produced by a compiled route; it has no contract")
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompiledRoute>()?;
    Ok(())
}
