//! Python bindings for the identification-repair facade (B0.1: F9/F10/F13).
//!
//! Python builds the declarations (a failed transport or back-door contract and
//! durable study candidates, all spelled in variable names); the evidence
//! obligations, the theorem-specific re-checks, the classification, the search
//! receipt and the `repair_search_receipt_v1` artifact are Rust's. A refusal
//! comes back as the registered `CausalUnsupportedError` with its reason code and
//! stable detail in the message.
use crate::graphs::{Admg, Dag};
use crate::transport_common::{
    execution_context, frame_named_artifact, resolve, serialization_error, unframe_named_artifact,
};
use crate::transport_interference_api::parse_catalog;
use crate::transport_z_api::{intervention_assignments, to_py_json};
use antecedent::analysis::repair::{
    BackdoorRepairFamily, DurableStudyCandidate, EvidenceObligation, ExpectedEvidence,
    RepairArtifactError, RepairConsumeLimits, RepairError, RepairFamily, RepairFamilyRef,
    RepairLimits, RepairObjective, RepairReport, RepairReportArtifact, StudyCostDeclaration,
    StudyKind, TransportRepairFamily, UnitRules, ZTransportRepairFamily, repair_contract,
};
use antecedent_core::assumption::{AssumptionSource, AssumptionStatus};
use antecedent_core::{
    DistributionAvailability, ObligationKind, ObligationRecord, ObligationScope, SearchLimits,
    VariableId,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{ClassicalTransportQuery, SidLimits};
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Magic prefix of a portable repair report: the design crate's container
/// bytes, framed with the variable names the repair was declared under.
pub(crate) const REPAIR_PREFIX: &[u8] = b"ANTECEDENT-REPAIR\x01";

fn repair_error(error: &RepairError) -> PyErr {
    let mut text = format!("{}: {}", error.detail, error.message);
    if let Some(receipt) = &error.receipt {
        text.push_str("; ");
        text.push_str(&receipt.summary());
    }
    crate::refusal(error.code, text)
}

fn artifact_error(error: &RepairArtifactError) -> PyErr {
    crate::refusal(error.code, format!("{}: {}", error.detail, error.message))
}

fn name_of(names: &[String], variable: VariableId) -> String {
    names.get(variable.as_usize()).cloned().unwrap_or_else(|| format!("v{}", variable.raw()))
}

fn names_of(names: &[String], variables: &[VariableId]) -> Vec<String> {
    variables.iter().map(|v| name_of(names, *v)).collect()
}

/// A failed contract of either family.
#[derive(Clone)]
enum Contract {
    Transport(Box<TransportRepairFamily>),
    Backdoor(Box<BackdoorRepairFamily>),
    ZTransport(Box<ZTransportRepairFamily>),
}

impl Contract {
    fn family(&self) -> &dyn RepairFamily {
        match self {
            Self::Transport(family) => &**family,
            Self::Backdoor(family) => &**family,
            Self::ZTransport(family) => &**family,
        }
    }

    fn reference(&self) -> RepairFamilyRef<'_> {
        match self {
            Self::Transport(family) => RepairFamilyRef::Transport(family),
            Self::Backdoor(family) => RepairFamilyRef::Backdoor(family),
            Self::ZTransport(family) => RepairFamilyRef::ZTransport(family),
        }
    }
}

fn obligation_json(obligation: &EvidenceObligation, names: &[String]) -> serde_json::Value {
    serde_json::json!({
        "quantities": obligation.quantities.iter().map(|(variable, quantity)| {
            (names[variable.raw() as usize].clone(), antecedent_io::quantity_wire::ScientificQuantityWire::from(quantity))
        }).collect::<BTreeMap<_, _>>(),
        "id": obligation.id.as_ref(),
        "kind": obligation.kind.as_str(),
        "scope": obligation.scope.as_str(),
        "variables": names_of(names, &obligation.variables),
        "population": obligation.population.as_deref(),
        "regime": {
            "interventions": names_of(names, &obligation.regime.interventions),
            "conditioned_on": names_of(names, &obligation.regime.conditioned_on),
            "joint": obligation.regime.joint,
        },
        "reason": obligation.reason.as_ref(),
        "required_slots": obligation.required_slots.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        "min_additional_samples": obligation.min_additional_samples,
        "provenance": {
            "family": obligation.provenance.family.as_ref(),
            "source": obligation.provenance.source.as_ref(),
            "proof_step": obligation.provenance.proof_step.as_deref(),
        },
        "satisfiable_by_study": obligation.satisfiable_by_study(),
    })
}

/// `repaired`, `exhausted` (a budget stop with nothing certified: never a
/// verdict) or `none_certified` (nothing certified within the declared
/// candidates: never an impossibility claim).
fn outcome_name(report: &RepairReport) -> &'static str {
    match report.status() {
        None => "repaired",
        Some((_, "identification_repair.budget")) => "exhausted",
        Some(_) => "none_certified",
    }
}

fn report_json(
    report: &RepairReport,
    names: &[String],
    declared: &[(String, String)],
) -> serde_json::Value {
    let labels: BTreeMap<&str, &str> =
        declared.iter().map(|(id, label)| (id.as_str(), label.as_str())).collect();
    let status = report.status();
    let receipt = &report.receipt;
    serde_json::json!({
        "family": report.family,
        "contract": report.contract,
        "objective": match report.objective {
            RepairObjective::MinimizeCost => "minimize_cost",
            RepairObjective::MinimizeSampleBudget => "minimize_sample_budget",
        },
        "outcome": outcome_name(report),
        "reason": status.map(|(code, _)| code),
        "detail": status.map(|(_, detail)| detail),
        "inference_claim": "none",
        "obligations": report.obligations.iter().map(|o| obligation_json(o, names)).collect::<Vec<_>>(),
        "declared": declared.iter().map(|(id, label)| serde_json::json!({
            "semantic_id": id, "label": label,
        })).collect::<Vec<_>>(),
        "outcomes": report.outcomes.iter().map(|o| serde_json::json!({
            "candidates": o.candidates.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            "labels": o.candidates.iter()
                .map(|id| labels.get(id.as_ref()).copied().unwrap_or(id.as_ref()))
                .collect::<Vec<&str>>(),
            "classification": o.classification.as_str(),
            "cost_units": o.cost_units,
            "cost_unit": o.cost_unit_label.as_deref(),
            "sample_budget": o.sample_budget,
            "reasons": o.reasons,
            "addressed": o.addressed.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            "unmet": o.unmet.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            "derivation": o.derivation.as_ref().map(|d| serde_json::json!({
                "checker": d.checker, "steps": d.steps, "verified": d.verified,
            })),
        })).collect::<Vec<_>>(),
        "ranked": report.ranked_sufficient,
        "receipt": {
            "operations_limit": receipt.limits.search.operations,
            "depth_limit": receipt.limits.search.depth,
            "memory_limit_bytes": receipt.memory_limit_bytes,
            "operations_consumed": receipt.operations_consumed,
            "depth_reached": receipt.depth_reached,
            "explored": receipt.explored,
            "unevaluated": receipt.unevaluated,
            "unevaluated_total": receipt.unevaluated_total,
            "dominated_skipped": receipt.dominated_skipped,
            "beyond_declared_depth": receipt.beyond_declared_depth,
            "stop": receipt.stop.as_ref().map(|s| s.stop.code()),
        },
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceDecl {
    population: String,
    interventions: Vec<String>,
    levels: BTreeMap<String, f64>,
    conditioned_on: Vec<String>,
    measured: Vec<String>,
    joint: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateDecl {
    label: String,
    kind: String,
    population: String,
    interventions: Vec<String>,
    measured: Vec<String>,
    joint: bool,
    sample_size: u64,
    recruitment: String,
    timing: String,
    unit: String,
    cluster: Option<String>,
    whole_cluster_sampling: bool,
    cost: u64,
    cost_unit: String,
    sample_budget: u64,
    feasible: bool,
    notes: Vec<String>,
    evidence: Vec<EvidenceDecl>,
    provider: Option<String>,
}

fn coordinates(names: &[String], variables: &[String]) -> PyResult<Arc<[VariableId]>> {
    variables.iter().map(|name| resolve(names, name)).collect::<PyResult<Vec<_>>>().map(Arc::from)
}

fn candidate_from_decl(names: &[String], decl: CandidateDecl) -> PyResult<DurableStudyCandidate> {
    let kind = match decl.kind.as_str() {
        "experiment" => StudyKind::Experiment,
        "observation" => StudyKind::Observation,
        "sample_increase" => StudyKind::SampleIncrease,
        other => {
            return Err(crate::value_err(format!(
                "unknown study kind {other:?}; use experiment, observation or sample_increase"
            )));
        }
    };
    let mut evidence = Vec::with_capacity(decl.evidence.len());
    for e in decl.evidence {
        let measured = coordinates(names, &e.measured)?;
        evidence.push(ExpectedEvidence {
            population: e.population.into(),
            interventions: coordinates(names, &e.interventions)?,
            intervention_values: intervention_assignments(names, e.levels)?,
            conditioned_on: coordinates(names, &e.conditioned_on)?,
            distribution: if e.joint {
                DistributionAvailability::Joint
            } else {
                DistributionAvailability::SeparateMarginals { variables: Arc::clone(&measured) }
            },
            measured,
        });
    }
    Ok(DurableStudyCandidate {
        label: decl.label.into(),
        kind,
        population: decl.population.into(),
        interventions: coordinates(names, &decl.interventions)?,
        measured: coordinates(names, &decl.measured)?,
        joint_measurement: decl.joint,
        sample_size: decl.sample_size,
        recruitment: decl.recruitment.into(),
        timing: decl.timing.into(),
        unit_rules: UnitRules {
            unit: decl.unit.into(),
            cluster: decl.cluster.map(Arc::from),
            whole_cluster_sampling: decl.whole_cluster_sampling,
        },
        cost: StudyCostDeclaration {
            units: decl.cost,
            unit_label: decl.cost_unit.into(),
            sample_budget: decl.sample_budget,
        },
        feasible: decl.feasible,
        feasibility_notes: decl.notes.into_iter().map(Arc::<str>::from).collect(),
        expected_evidence: evidence.into(),
        external_provider: decl.provider.map(Arc::from),
    })
}

pub(crate) fn parse_candidates(
    names: &[String],
    json: &str,
) -> PyResult<Vec<DurableStudyCandidate>> {
    let decls: Vec<CandidateDecl> = serde_json::from_str(json)
        .map_err(|e| crate::value_err(format!("invalid candidate declaration: {e}")))?;
    decls.into_iter().map(|decl| candidate_from_decl(names, decl)).collect()
}

fn declared_labels(candidates: &[DurableStudyCandidate]) -> Vec<(String, String)> {
    candidates.iter().map(|c| (c.semantic_id().to_string(), c.label.to_string())).collect()
}

/// A failed contract with its unresolved evidence obligations.
#[pyclass(skip_from_py_object)]
pub(crate) struct RepairContractStage {
    contract: Contract,
    names: Vec<String>,
}

impl RepairContractStage {
    /// The frozen failed contract, for re-identification on arrived evidence in
    /// `composition_api`.
    pub(crate) fn family_ref(&self) -> RepairFamilyRef<'_> {
        self.contract.reference()
    }

    /// The variable names the contract was declared under.
    pub(crate) fn variable_names(&self) -> &[String] {
        &self.names
    }
}

#[pymethods]
impl RepairContractStage {
    /// `transport` or `backdoor`.
    #[getter]
    fn family(&self) -> &'static str {
        self.contract.family().family_id()
    }

    /// The failed contract's stable identity.
    #[getter]
    fn contract_id(&self) -> String {
        self.contract.family().contract_id()
    }

    /// The unresolved obligations, each with its source proof step.
    #[pyo3(signature = (quantities_json=None))]
    fn obligations(&self, py: Python<'_>, quantities_json: Option<&str>) -> PyResult<Py<PyAny>> {
        let declared: BTreeMap<String, antecedent_io::quantity_wire::ScientificQuantityWire> =
            serde_json::from_str(quantities_json.unwrap_or("{}"))
                .map_err(|error| crate::value_err(error.to_string()))?;
        let mut coordinates = BTreeMap::new();
        for (name, wire) in declared {
            let variable =
                self.names.iter().position(|candidate| candidate == &name).ok_or_else(|| {
                    crate::value_err(format!("unknown scientific coordinate variable {name}"))
                })?;
            let quantity =
                antecedent_core::ScientificQuantity::try_from(wire).map_err(crate::value_err)?;
            let variable = u32::try_from(variable).map_err(|_| {
                crate::value_err("scientific coordinate variable index exceeds u32")
            })?;
            coordinates.insert(VariableId::from_raw(variable), quantity);
        }
        let mut values = Vec::new();
        for obligation in self.contract.family().unresolved_obligations() {
            let obligation = if coordinates.is_empty() {
                obligation
            } else {
                let selected = coordinates
                    .iter()
                    .filter(|(variable, _)| obligation.variables.contains(variable))
                    .map(|(variable, quantity)| (*variable, quantity.clone()))
                    .collect();
                obligation
                    .with_quantities(selected)
                    .map_err(|error| crate::value_err(error.to_string()))?
            };
            values.push(obligation_json(&obligation, &self.names));
        }
        to_py_json(py, &serde_json::Value::Array(values))
    }

    /// Apply each candidate (and each bounded subset of candidates) as
    /// hypothetical evidence and re-run the contract's own theorem checker.
    #[pyo3(signature = (candidates_json, objective, *, max_operations, max_depth, memory_bytes=None, cancel=None))]
    fn repair(
        &self,
        py: Python<'_>,
        candidates_json: &str,
        objective: &str,
        max_operations: usize,
        max_depth: usize,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<RepairStage> {
        let objective = match objective {
            "minimize_cost" => RepairObjective::MinimizeCost,
            "minimize_sample_budget" => RepairObjective::MinimizeSampleBudget,
            other => {
                return Err(crate::value_err(format!(
                    "unknown objective {other:?}; use minimize_cost or minimize_sample_budget"
                )));
            }
        };
        let candidates = parse_candidates(&self.names, candidates_json)?;
        let limits = RepairLimits {
            search: SearchLimits { operations: max_operations, depth: max_depth },
            memory_limit_bytes: memory_bytes
                .unwrap_or_else(|| RepairLimits::default().memory_limit_bytes),
        };
        let contract = self.contract.clone();
        let declared = declared_labels(&candidates);
        let (report, artifact) = crate::detach_catch(py, move || {
            let ctx = execution_context(0, memory_bytes, cancel);
            let report = repair_contract(contract.family(), &candidates, objective, limits, &ctx)
                .map_err(|e| repair_error(&e))?;
            let artifact = RepairReportArtifact::build(contract.reference(), &candidates, &report);
            Ok((report, artifact))
        })?;
        Ok(RepairStage { report, names: self.names.clone(), declared, artifact })
    }
}

/// A finished repair search: the classification table, the ranked verified
/// subsets, the search receipt and the exportable artifact.
#[pyclass(skip_from_py_object)]
struct RepairStage {
    report: RepairReport,
    names: Vec<String>,
    declared: Vec<(String, String)>,
    artifact: Result<RepairReportArtifact, RepairArtifactError>,
}

#[pymethods]
impl RepairStage {
    /// `repaired`, `none_certified` (nothing certified within the declared
    /// candidates; never an impossibility claim) or `exhausted` (a budget stop
    /// with nothing certified; never a verdict).
    #[getter]
    fn outcome(&self) -> &'static str {
        outcome_name(&self.report)
    }

    /// The whole report as a dictionary spelled in variable names.
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(py, &report_json(&self.report, &self.names, &self.declared))
    }

    /// Export the report as a portable, independently replayable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let artifact = self.artifact.as_ref().map_err(artifact_error)?;
        let bytes = artifact.to_bytes("repair-report").map_err(|e| artifact_error(&e))?;
        let framed = frame_named_artifact(REPAIR_PREFIX, &self.names, bytes)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
}

/// Freeze a failed catalog-aware classical-transport contract.
#[pyfunction]
#[pyo3(signature = (graph, selections, source, target, outcomes, treatments, catalog, *, max_steps=100_000, max_depth=256, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn repair_transport_contract(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    selections: Vec<String>,
    source: String,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    catalog: &Bound<'_, PyAny>,
    max_steps: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<RepairContractStage> {
    let names = graph.names.clone();
    let query = ClassicalTransportQuery {
        outcomes: coordinates(&names, &outcomes)?,
        treatments: coordinates(&names, &treatments)?,
        source: source.into(),
        target: target.into(),
    };
    let diagram = SelectionDiagram::try_new(
        graph.aligned_to_names(&names)?,
        coordinates(&names, &selections)?,
    )
    .map_err(|e| crate::value_err(e.to_string()))?;
    let named = Admg { admg: graph.admg.clone(), names: names.clone() };
    let base = parse_catalog(catalog, &named)?;
    let family = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        TransportRepairFamily::try_new(
            diagram,
            query,
            base,
            SidLimits { steps: max_steps, depth: max_depth },
            &ctx,
        )
        .map_err(|e| repair_error(&e))
    })?;
    Ok(RepairContractStage { contract: Contract::Transport(Box::new(family)), names })
}

/// Restore the existing z-transport failure snapshot as a repair contract.
#[pyfunction]
#[pyo3(signature = (names, failure_snapshot, *, max_steps=100_000, max_depth=256, memory_bytes=None, cancel=None))]
fn repair_z_transport_contract(
    py: Python<'_>,
    names: Vec<String>,
    failure_snapshot: &[u8],
    max_steps: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<RepairContractStage> {
    let wire: antecedent_design::ZTransportFailureSnapshotWire =
        serde_json::from_slice(failure_snapshot).map_err(serialization_error)?;
    check_names(&names, wire.graph.node_count)?;
    let family = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let limits = SidLimits { steps: max_steps, depth: max_depth };
        let snapshot = antecedent_design::ZTransportFailureSnapshot::from_wire(&wire, limits, &ctx)
            .map_err(|e| {
                crate::refusal(
                    "invalid_argument",
                    format!("identification_repair.invalid_request: {e}"),
                )
            })?;
        ZTransportRepairFamily::try_new(snapshot, limits).map_err(|e| repair_error(&e))
    })?;
    Ok(RepairContractStage { contract: Contract::ZTransport(Box::new(family)), names })
}

/// Freeze a failed back-door contract: `observed` are the variables currently
/// measured jointly with treatment and outcome in `population`, and each
/// `(id, description, required_check)` assumption is an unresolved
/// `establish_assumption` obligation that no study satisfies.
#[pyfunction]
#[pyo3(signature = (graph, treatment, outcome, population, observed, assumptions))]
fn repair_backdoor_contract(
    graph: PyRef<'_, Dag>,
    treatment: &str,
    outcome: &str,
    population: String,
    observed: Vec<String>,
    assumptions: Vec<(String, String, Option<String>)>,
) -> PyResult<RepairContractStage> {
    let names = graph.names.clone();
    let records = assumptions
        .into_iter()
        .map(|(id, description, check)| {
            let record = ObligationRecord::new(
                id,
                ObligationScope::Program,
                AssumptionSource::UserDeclared,
                ObligationKind::CheckNotRun,
                AssumptionStatus::Declared,
                description,
            );
            match check {
                Some(check) => record.with_required_check(check),
                None => record,
            }
        })
        .collect::<Vec<_>>();
    let family = BackdoorRepairFamily::try_new(
        graph.dag.clone(),
        resolve(&names, treatment)?,
        resolve(&names, outcome)?,
        population,
        coordinates(&names, &observed)?.iter().copied(),
        &records,
    )
    .map_err(|e| repair_error(&e))?;
    Ok(RepairContractStage { contract: Contract::Backdoor(Box::new(family)), names })
}

fn check_names(names: &[String], node_count: u32) -> PyResult<()> {
    if names.len() != node_count as usize
        || names.iter().collect::<BTreeSet<_>>().len() != names.len()
    {
        return Err(serialization_error("invalid repair artifact coordinate names"));
    }
    Ok(())
}

/// Replay an exported repair report under its stored limits and return it.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_operations, max_depth, memory_bytes=None, cancel=None))]
fn consume_repair_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<RepairStage> {
    let (names, bytes) = unframe_named_artifact(REPAIR_PREFIX, artifact, "repair")?;
    let limits = RepairConsumeLimits {
        search: SearchLimits { operations: max_operations, depth: max_depth },
        memory_limit_bytes: memory_bytes
            .unwrap_or_else(|| RepairConsumeLimits::default().memory_limit_bytes),
    };
    crate::detach_catch(py, move || {
        let stored = RepairReportArtifact::from_bytes(&bytes).map_err(|e| artifact_error(&e))?;
        let node_count = match (
            &stored.premises.transport,
            &stored.premises.backdoor,
            &stored.premises.z_transport,
        ) {
            (Some(t), None, None) => t.graph.node_count,
            (None, Some(b), None) => b.graph.node_count,
            (None, None, Some(z)) => z.snapshot.graph.node_count,
            _ => return Err(serialization_error("invalid repair artifact premises")),
        };
        check_names(&names, node_count)?;
        let ctx = execution_context(0, memory_bytes, cancel);
        let report = stored.consume(limits, &ctx).map_err(|e| artifact_error(&e))?;
        let declared =
            stored.candidates().map_err(|e| artifact_error(&e)).map(|c| declared_labels(&c))?;
        Ok(RepairStage { report, names, declared, artifact: Ok(stored) })
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RepairContractStage>()?;
    module.add_class::<RepairStage>()?;
    module.add_function(wrap_pyfunction!(repair_transport_contract, module)?)?;
    module.add_function(wrap_pyfunction!(repair_backdoor_contract, module)?)?;
    module.add_function(wrap_pyfunction!(repair_z_transport_contract, module)?)?;
    module.add_function(wrap_pyfunction!(consume_repair_artifact, module)?)?;
    Ok(())
}
