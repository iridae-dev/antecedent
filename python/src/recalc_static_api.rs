//! Checked bindings for finite ADMG response and multi-source transport recalculation.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::analysis::recalc_receipt::{RecalcRunError, UtilitySpec};
use antecedent::analysis::recalc_static::{
    MzRecalcRequest, MzRecalcSession, StaticRecalcOutcome, StaticResponseRequest,
    StaticResponseSession, execute_mz_with_receipt, execute_static_response_with_receipt,
};
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RequestSupport, ResumeContext, RetargetSupport, StageIdentities,
};
use antecedent_core::{SearchLimits, VariableId};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_identify::{MzTransportQuery, ZTransportSourceSpec};
use antecedent_io::recalc_receipt_artifact::{
    CapabilitiesWire, DeclaredStageWire, RecalcReceiptArtifact, ResumeWire, identities_from_wire,
    identities_to_wire, plan_to_wire,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use serde::Deserialize;

use crate::graphs::Admg;
use crate::recalc_api::{
    RunPayload, artifact_invalid, context, counts_wire, invalid, parse_json, plan_refusal_json,
    refusal_json,
};
use crate::recalc_bounds::{check_columns, sequence_item, sequence_len};
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, frame_named_artifact, parse_law_table, resolve,
};
use crate::transport_interference_api::parse_catalog;
use crate::{detach_catch, py_msg};

const MAX_VARIABLES: usize = 12;
const MAX_GRID: usize = 64;
const MAX_LAWS: usize = 64;
const MAX_LAW_CELLS: usize = 1_000_000;
const MZ_PREFIX: &[u8] = b"ANTECEDENT-MZ-TRANSPORT\x01";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseWire {
    treatment: String,
    outcome: String,
    support: Vec<f64>,
    actions: Vec<f64>,
    baseline: usize,
    active: usize,
    benefit_per_unit: f64,
    cost: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceWire {
    population: String,
    controllable: Vec<String>,
    selections: Vec<String>,
    experiment_assignment: BTreeMap<String, f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportWire {
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    sources: Vec<SourceWire>,
    assignments: Vec<BTreeMap<String, f64>>,
    baseline: usize,
    active: usize,
    benefit_per_unit: f64,
    cost: f64,
    search_operations: usize,
    search_depth: usize,
    evaluation_operations: usize,
    evaluation_depth: usize,
}
fn response_request(
    names: Vec<String>,
    columns: &[PyReadonlyArray1<'_, f64>],
    graph: &Admg,
    text: &str,
) -> PyResult<StaticResponseRequest> {
    check_columns(&names, columns)?;
    let wire: ResponseWire = parse_json(text, "static response")?;
    if names.len() > MAX_VARIABLES || wire.support.len() > MAX_GRID || wire.actions.len() > MAX_GRID
    {
        return Err(invalid(
            "recalc.limits_exceeded",
            "static response exceeds variable/grid limits",
        ));
    }
    if names.is_empty()
        || names.len() != columns.len()
        || names.iter().enumerate().any(|(i, n)| n.trim().is_empty() || names[..i].contains(n))
    {
        return Err(invalid("recalc.static_invalid_data", "unique named columns required"));
    }
    Ok(StaticResponseRequest {
        treatment: resolve(&names, &wire.treatment)?,
        outcome: resolve(&names, &wire.outcome)?,
        graph: graph.aligned_to_names(&names)?,
        columns: names.into_iter().zip(columns.iter().map(|c| c.as_array().to_vec())).collect(),
        support: wire.support,
        actions: wire.actions,
        baseline: wire.baseline,
        active: wire.active,
        utility: UtilitySpec { benefit_per_unit: wire.benefit_per_unit, cost: wire.cost },
    })
}
fn inspect_laws(laws: &Bound<'_, PyAny>, catalog: &Bound<'_, PyAny>) -> PyResult<()> {
    let len = sequence_len(laws, "recalc.static_invalid_data")?;
    if len > MAX_LAWS {
        return Err(invalid("recalc.limits_exceeded", "at most64 exact laws"));
    }
    let mut cells = 0usize;
    for i in 0..len {
        let law = sequence_item(laws, i)?;
        let probabilities = law.getattr("probabilities")?;
        let n = sequence_len(&probabilities, "recalc.static_invalid_data")?;
        cells = cells
            .checked_add(n)
            .ok_or_else(|| invalid("recalc.limits_exceeded", "law size overflow"))?;
        if cells > MAX_LAW_CELLS {
            return Err(invalid("recalc.limits_exceeded", "law cells exceed1million"));
        }
        let counts = law.getattr("empirical_counts")?;
        if !counts.is_none() && sequence_len(&counts, "recalc.static_invalid_data")? > MAX_LAW_CELLS
        {
            return Err(invalid("recalc.limits_exceeded", "count cells exceed1million"));
        }
        let axes = law.getattr("axes")?;
        let axes_len = sequence_len(&axes, "recalc.static_invalid_data")?;
        if axes_len > MAX_VARIABLES {
            return Err(invalid("recalc.limits_exceeded", "too many law axes"));
        }
        for a in 0..axes_len {
            let axis = sequence_item(&axes, a)?;
            let values = sequence_item(&axis, 1)?;
            if sequence_len(&values, "recalc.static_invalid_data")? > MAX_GRID {
                return Err(invalid("recalc.limits_exceeded", "axis levels exceed64"));
            }
        }
    }
    for (name, limit) in [("regimes", MAX_LAWS), ("bindings", MAX_LAWS), ("environments", 5)] {
        if sequence_len(&catalog.getattr(name)?, "recalc.mz_invalid_request")? > limit {
            return Err(invalid("recalc.limits_exceeded", "catalog exceeds declared limits"));
        }
    }
    Ok(())
}
fn transport_request(
    graph: &Admg,
    catalog: &Bound<'_, PyAny>,
    laws: &Bound<'_, PyAny>,
    text: &str,
) -> PyResult<MzRecalcRequest> {
    let wire: TransportWire = parse_json(text, "multi-source request")?;
    if graph.names.len() > MAX_VARIABLES || wire.assignments.len() > MAX_GRID {
        return Err(invalid("recalc.limits_exceeded", "transport exceeds variable/request limits"));
    }
    if wire.sources.len() > 4
        || wire.search_operations > 4096
        || wire.search_depth > 24
        || wire.evaluation_operations > 10_000_000
        || wire.evaluation_depth > 256
    {
        return Err(invalid(
            "recalc.limits_exceeded",
            "transport exceeds source/search/evaluation limits",
        ));
    }
    inspect_laws(laws, catalog)?;
    let ids = |names: &[String]| -> PyResult<Arc<[VariableId]>> {
        names.iter().map(|n| resolve(&graph.names, n)).collect::<PyResult<Vec<_>>>().map(Arc::from)
    };
    let sources = wire
        .sources
        .into_iter()
        .map(|source| -> PyResult<_> {
            Ok(ZTransportSourceSpec {
                population: source.population.into(),
                controllable: ids(&source.controllable)?,
                selection_targets: ids(&source.selections)?,
                experiment_assignment: source
                    .experiment_assignment
                    .into_iter()
                    .map(|(name, value)| {
                        Ok(antecedent_core::InterventionAssignment {
                            variable: resolve(&graph.names, &name)?,
                            value: antecedent_core::Value::f64(value),
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?
                    .into(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let catalog = parse_catalog(catalog, graph)?;
    let tables = (0..sequence_len(laws, "recalc.static_invalid_data")?)
        .map(|i| parse_law_table(&sequence_item(laws, i)?, &catalog, graph, RegimeCheck::Strict))
        .collect::<PyResult<Vec<_>>>()?;
    let data = antecedent_expr::ExactTransportData::try_new(tables, MAX_LAW_CELLS)
        .map_err(crate::transport_common::error)?;
    Ok(MzRecalcRequest {
        graph: graph.aligned_to_names(&graph.names)?,
        variable_names: graph.names.clone(),
        query: MzTransportQuery {
            outcomes: ids(&wire.outcomes)?,
            treatments: ids(&wire.treatments)?,
            target: wire.target.into(),
            sources: sources.into(),
        },
        catalog,
        data,
        requests: wire
            .assignments
            .into_iter()
            .map(|assignment| assignment_from_pairs(&graph.names, assignment))
            .collect::<PyResult<Vec<_>>>()?,
        search: SearchLimits { operations: wire.search_operations, depth: wire.search_depth },
        limits: ExactEvaluationLimits {
            operations: wire.evaluation_operations,
            depth: wire.evaluation_depth,
        },
        baseline: wire.baseline,
        active: wire.active,
        utility: UtilitySpec { benefit_per_unit: wire.benefit_per_unit, cost: wire.cost },
    })
}
fn caps(boundary: Boundary) -> RecalcCapabilities {
    RecalcCapabilities {
        boundary,
        request: RequestSupport::OnGrid,
        retarget: RetargetSupport::NotDeclared,
    }
}
fn resume(previous: &str, resume: &str) -> PyResult<(StageIdentities, ResumeContext)> {
    let ids: Vec<DeclaredStageWire> = parse_json(previous, "previous")?;
    let ids = identities_from_wire(&ids).map_err(|e| artifact_invalid(&e))?;
    let wire: ResumeWire = parse_json(resume, "resume")?;
    Ok((
        ids,
        ResumeContext {
            portable_fit: wire.portable_fit,
            portable_scores: wire.portable_scores,
            supplied_data: wire.supplied_data,
            supplied_provider: wire.supplied_provider,
            scores_snapshot_bound: wire.scores_snapshot_bound,
        },
    ))
}
fn static_error(error: &RecalcRunError) -> String {
    match error {
        RecalcRunError::Refused(plan) => plan_refusal_json(plan),
        RecalcRunError::NoLiveState(stage) => refusal_json(
            antecedent_core::reason_code!("score_table_unavailable"),
            &stage.label(),
            "recalc.no_live_state",
            Some(&stage.label()),
            None,
            &error.to_string(),
            None,
        ),
        RecalcRunError::Request(detail) => {
            let (code, stage) = match *detail {
                "recalc.cancelled" => {
                    (antecedent_core::reason_code!("cancelled_no_claim"), "score_artifact")
                }
                "recalc.static_action_out_of_support" | "recalc.mz_action_out_of_support" => {
                    (antecedent_core::reason_code!("route_not_supported"), "treatment_grid")
                }
                "recalc.static_invalid_query"
                | "recalc.static_invalid_contrast"
                | "recalc.mz_invalid_request" => {
                    (antecedent_core::reason_code!("invalid_argument"), "query")
                }
                "recalc.static_graph_unsupported" => {
                    (antecedent_core::reason_code!("route_not_supported"), "graph")
                }
                "recalc.static_invalid_data" => {
                    (antecedent_core::reason_code!("invalid_argument"), "data_snapshot")
                }
                "recalc.static_required_factor_missing" => {
                    (antecedent_core::reason_code!("route_not_supported"), "score_artifact")
                }
                "recalc.memory_budget_exceeded" => {
                    (antecedent_core::reason_code!("invalid_argument"), "score_artifact")
                }
                _ => (antecedent_core::reason_code!("invalid_argument"), "score_artifact"),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        RecalcRunError::Execution(cause) => {
            use antecedent::CausalError;
            use antecedent_identify::IdentificationError;
            use antecedent_io::IoError;
            let (code, detail, stage) = match cause.peeled() {
                CausalError::Estimate(
                    antecedent_estimate::EstimationError::Refused { code, .. }
                    | antecedent_estimate::EstimationError::RefusedWithFields { code, .. },
                ) if *code == "invalid_argument" => (
                    antecedent_core::reason_code!("invalid_argument"),
                    "recalc.static_invalid_factor",
                    "score_artifact",
                ),
                CausalError::Estimate(
                    antecedent_estimate::EstimationError::Refused { code, .. }
                    | antecedent_estimate::EstimationError::RefusedWithFields { code, .. },
                ) if *code == "cancelled_no_claim" => (
                    antecedent_core::reason_code!("cancelled_no_claim"),
                    "recalc.cancelled",
                    "score_artifact",
                ),
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_missing_evidence" =>
                {
                    (
                        antecedent_core::reason_code!("transport_missing_evidence"),
                        "recalc.static_transport_missing_evidence",
                        "identification",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_not_certified" =>
                {
                    (
                        antecedent_core::reason_code!("transport_not_certified"),
                        "recalc.static_transport_not_certified",
                        "identification",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_budget_cancel" =>
                {
                    (
                        antecedent_core::reason_code!("transport_budget_cancel"),
                        "recalc.static_transport_budget_cancel",
                        "identification",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "cancelled_no_claim" =>
                {
                    (
                        antecedent_core::reason_code!("cancelled_no_claim"),
                        "recalc.cancelled",
                        "score_artifact",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "invalid_argument" =>
                {
                    (
                        antecedent_core::reason_code!("invalid_argument"),
                        "recalc.static_invalid_factor",
                        "score_artifact",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_numerical_failure" =>
                {
                    (
                        antecedent_core::reason_code!("transport_numerical_failure"),
                        "recalc.static_transport_numerical_failure",
                        "score_artifact",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_unsupported_evaluator" =>
                {
                    (
                        antecedent_core::reason_code!("transport_unsupported_evaluator"),
                        "recalc.static_transport_unsupported_evaluator",
                        "score_artifact",
                    )
                }
                CausalError::NotIdentified { .. } => (
                    antecedent_core::reason_code!("effect_not_identified"),
                    "recalc.static_not_identified",
                    "identification",
                ),
                CausalError::Identify(IdentificationError::MissingEvidence { .. }) => (
                    antecedent_core::reason_code!("transport_missing_evidence"),
                    "recalc.static_transport_missing_evidence",
                    "identification",
                ),
                CausalError::Identify(
                    IdentificationError::NotCertified { .. }
                    | IdentificationError::InvalidDerivation { .. },
                ) => (
                    antecedent_core::reason_code!("transport_not_certified"),
                    "recalc.static_transport_not_certified",
                    "identification",
                ),
                CausalError::Identify(
                    IdentificationError::Cancelled | IdentificationError::Budget { .. },
                ) => (
                    antecedent_core::reason_code!("transport_budget_cancel"),
                    "recalc.static_transport_budget_cancel",
                    "identification",
                ),
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_missing_provider" =>
                {
                    (
                        antecedent_core::reason_code!("transport_missing_provider"),
                        "recalc.static_transport_missing_provider",
                        "score_artifact",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_support_failure" =>
                {
                    (
                        antecedent_core::reason_code!("transport_support_failure"),
                        "recalc.static_transport_support_failure",
                        "score_artifact",
                    )
                }
                CausalError::Serialization(IoError::Refused { code, .. })
                    if *code == "transport_proven_non_transportable" =>
                {
                    (
                        antecedent_core::reason_code!("transport_proven_non_transportable"),
                        "recalc.static_transport_non_transportable",
                        "identification",
                    )
                }
                _ => (
                    antecedent_core::reason_code!("route_not_supported"),
                    "recalc.static_engine_refused",
                    "score_artifact",
                ),
            };
            refusal_json(code, stage, detail, None, None, &cause.to_string(), None)
        }
        RecalcRunError::Receipt(_) => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            "score_artifact",
            "recalc.static_receipt_invalid",
            None,
            None,
            &error.to_string(),
            None,
        ),
    }
}
fn payload(
    previous: &StageIdentities,
    requested: &StageIdentities,
    capabilities: &RecalcCapabilities,
    outcome: StaticRecalcOutcome,
) -> PyResult<RunPayload> {
    let counts = outcome
        .receipt
        .entries()
        .iter()
        .map(|e| (e.stage, counts_wire(&e.counts)))
        .collect::<BTreeMap<_, _>>();
    let artifact =
        RecalcReceiptArtifact::seal(previous, requested, capabilities, &counts).map_err(py_msg)?;
    if artifact.plan() != &outcome.plan {
        return Err(py_msg("static receipt differs from executed plan"));
    }
    let bytes = artifact
        .to_bytes(&format!("recalc-static-{}", &artifact.receipt_identity()[..16]))
        .map_err(py_msg)?;
    let json = serde_json::json!({"plan":plan_to_wire(&outcome.plan),"receipt":artifact.meta(),"means":outcome.means,"contrast":outcome.contrast,"decision":{"net_benefit":outcome.decision.net_benefit,"treat":outcome.decision.treat}});
    Ok((Some(json.to_string()), Some(bytes), None))
}

#[pyclass(name = "StaticResponseSessionHandle")]
struct PyStaticSession {
    inner: StaticResponseSession,
}
#[pymethods]
impl PyStaticSession {
    #[new]
    fn new() -> Self {
        Self { inner: StaticResponseSession::new() }
    }
    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let (ids, context) = resume(previous_json, resume_json)?;
        Ok(Self { inner: StaticResponseSession::resume(ids, context) })
    }
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(py_msg)
    }
    fn capabilities_json(&self) -> PyResult<String> {
        serde_json::to_string(&CapabilitiesWire::from_capabilities(&caps(self.inner.boundary())))
            .map_err(py_msg)
    }
    #[pyo3(signature=(names,columns,graph,specification,*,seed=1,threads=None))]
    fn plan(
        &self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        graph: PyRef<'_, Admg>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let request = response_request(names, &columns, &graph, specification)?;
        serde_json::to_string(&plan_to_wire(&self.inner.plan(&request, &context(seed, threads))))
            .map_err(py_msg)
    }
    #[pyo3(signature=(names,columns,graph,specification,*,seed=1,threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        graph: PyRef<'_, Admg>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<RunPayload> {
        let request = response_request(names, &columns, &graph, specification)?;
        drop(columns);
        let previous = self.inner.identities().clone();
        let capabilities = caps(self.inner.boundary());
        let mut session = std::mem::take(&mut self.inner);
        let ctx = context(seed, threads);
        let (session, ran) = detach_catch(py, move || {
            let ran = execute_static_response_with_receipt(&mut session, &request, &ctx);
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ok(outcome) => payload(&previous, self.inner.identities(), &capabilities, outcome),
            Err(error) => Ok((None, None, Some(static_error(&error)))),
        }
    }
    #[pyo3(signature=(*,seed=1,threads=None))]
    fn export_result(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<(Option<Vec<u8>>, Option<String>)> {
        let ctx = context(seed, threads);
        detach_catch(py, || match self.inner.export_result(&ctx) {
            Ok(bytes) => Ok((Some(bytes), None)),
            Err(error) => Ok((None, Some(static_error(&error)))),
        })
    }
}
#[pyclass(name = "MultiSourceSessionHandle")]
struct PyMzSession {
    inner: MzRecalcSession,
    names: Vec<String>,
}
#[pymethods]
impl PyMzSession {
    #[new]
    fn new() -> Self {
        Self { inner: MzRecalcSession::new(), names: Vec::new() }
    }
    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let (ids, context) = resume(previous_json, resume_json)?;
        Ok(Self { inner: MzRecalcSession::resume(ids, context), names: Vec::new() })
    }
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(py_msg)
    }
    fn capabilities_json(&self) -> PyResult<String> {
        serde_json::to_string(&CapabilitiesWire::from_capabilities(&caps(self.inner.boundary())))
            .map_err(py_msg)
    }
    #[pyo3(signature=(graph,catalog,laws,specification,*,seed=1,threads=None))]
    fn plan(
        &self,
        graph: PyRef<'_, Admg>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let request = transport_request(&graph, catalog, laws, specification)?;
        serde_json::to_string(&plan_to_wire(&self.inner.plan(&request, &context(seed, threads))))
            .map_err(py_msg)
    }
    #[pyo3(signature=(graph,catalog,laws,specification,*,seed=1,threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        py: Python<'_>,
        graph: PyRef<'_, Admg>,
        catalog: &Bound<'_, PyAny>,
        laws: &Bound<'_, PyAny>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<RunPayload> {
        let request = transport_request(&graph, catalog, laws, specification)?;
        let names = request.variable_names.clone();
        let previous = self.inner.identities().clone();
        let capabilities = caps(self.inner.boundary());
        let mut session = std::mem::take(&mut self.inner);
        let ctx = context(seed, threads);
        let (session, ran) = detach_catch(py, move || {
            let ran = execute_mz_with_receipt(&mut session, &request, &ctx);
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ok(outcome) => {
                self.names = names;
                payload(&previous, self.inner.identities(), &capabilities, outcome)
            }
            Err(error) => Ok((None, None, Some(static_error(&error)))),
        }
    }
    #[pyo3(signature=(*,seed=1,threads=None))]
    fn export_result(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<(Option<Vec<u8>>, Option<String>)> {
        let ctx = context(seed, threads);
        detach_catch(py, || match self.inner.export_result(&ctx) {
            Ok(bytes) => Ok((Some(frame_named_artifact(MZ_PREFIX, &self.names, bytes)?), None)),
            Err(error) => Ok((None, Some(static_error(&error)))),
        })
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStaticSession>()?;
    module.add_class::<PyMzSession>()
}
