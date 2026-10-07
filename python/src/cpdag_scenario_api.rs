//! Python bindings for the DAG completions of a CPDAG and the shared-data
//! covariance of scenario estimates (2.3A X2).
//!
//! Python builds the declarations; every identity, check and refusal rule is
//! Rust's. A coded refusal comes back as structured JSON
//! (`{"code", "detail", "message"}`) for the Python layer to raise as its own
//! typed exception; failures that are not refusals (cancellation, resource
//! budgets, serialization) raise the ordinary transport exceptions.
use crate::graphs::{Admg, Cpdag};
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact, parse_laws,
    resolve, resolve_variable_ids, serialization_error, unframe_named_artifact,
};
use crate::transport_interference_api::parse_catalog;
use antecedent_core::{NodeRef, SearchLimits, Value, VariableDomain, VariableId, reason_code};
use antecedent_estimate::cpdag_scenarios::CpdagScenarioReport;
use antecedent_estimate::scenario_covariance::ScenarioCovariance;
use antecedent_expr::ExactEvaluationLimits;
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::cpdag_completion::{
    CompletionEvidenceBinding, CpdagCompletionInput, CpdagEvidence, enumerate_cpdag_completions,
};
use antecedent_identify::sid::scenarios::ScenarioCoordinate;
use antecedent_io::IoError;
use antecedent_io::cpdag_completion_artifact::{CpdagCompletionArtifactWire, CpdagConsumeLimits};
use antecedent_io::scenario_covariance_artifact::{
    CovarianceSpec, ScenarioCovarianceArtifactWire, ScenarioCovarianceConsumeLimits,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

const CPDAG_PREFIX: &[u8] = b"ANTECEDENT-CPDAG-SCENARIOS\x01";
const COVARIANCE_PREFIX: &[u8] = b"ANTECEDENT-SCENARIO-COVARIANCE\x01";
const CPDAG_SCOPE: &str = "cpdag_completion_scenarios_structural_envelope";
const COVARIANCE_SCOPE: &str = "shared_row_covariance_point_only";
/// Largest spec JSON the covariance bridge accepts.
const MAX_SPEC_JSON_BYTES: usize = 8 * 1024 * 1024;

/// One declared coordinate: `(name, domain, cardinality, unit)`.
type CoordinateTuple = (String, String, Option<u32>, Option<String>);
/// One evidence declaration: `(completion id, completion edges, evidence identity,
/// catalog, certified-for id)`. A shared declaration names no completion.
type EvidenceTuple<'py> =
    (Option<String>, Option<Vec<(String, String)>>, String, Bound<'py, PyAny>, Option<String>);
/// The refusal JSON of a coded refusal.
type Refusal = String;

fn refusal_json(code: &str, message: &str) -> Refusal {
    let (detail, text) = match message.split_once(": ") {
        Some((head, rest))
            if head.contains('.')
                && head.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') =>
        {
            (head, rest)
        }
        _ => ("", message),
    };
    serde_json::json!({ "code": code, "detail": detail, "message": text }).to_string()
}

/// A coded refusal as JSON; anything else (cancellation, resource budgets,
/// serialization) as the ordinary transport exception.
fn refusal_or_error(e: IoError) -> PyResult<Refusal> {
    match e {
        IoError::Refused { code, message } if code != reason_code!("transport_budget_cancel") => {
            Ok(refusal_json(code, &message))
        }
        IoError::UnsupportedVersion { .. } => Err(serialization_error(e)),
        other => Err(error(other)),
    }
}

fn argument_refusal(code: &str, detail: &str, message: &str) -> Refusal {
    refusal_json(code, &format!("{detail}: {message}"))
}

/// Resolve declared coordinates against the shared variable names.
fn parse_coordinates(
    coordinates: &[CoordinateTuple],
    names: &[String],
) -> PyResult<Arc<[ScenarioCoordinate]>> {
    coordinates
        .iter()
        .map(|(name, domain, cardinality, unit)| {
            let variable = names.iter().position(|n| n == name).ok_or_else(|| {
                crate::with_reason_code(
                    crate::value_err(format!(
                        "scenarios.coordinate_mismatch: unknown variable {name}"
                    )),
                    reason_code!("schema_mismatch"),
                )
            })?;
            let domain = match (domain.as_str(), cardinality) {
                ("unspecified", None) => VariableDomain::Unspecified,
                ("continuous", None) => VariableDomain::Continuous,
                ("binary", None) => VariableDomain::Binary,
                ("count", None) => VariableDomain::Count,
                ("categorical", Some(n)) => VariableDomain::Categorical { cardinality: *n },
                _ => return Err(crate::value_err(format!("invalid domain for {name}"))),
            };
            Ok(ScenarioCoordinate {
                variable: VariableId::from_raw(
                    u32::try_from(variable).map_err(|e| crate::value_err(e.to_string()))?,
                ),
                name: Arc::from(name.as_str()),
                domain,
                unit: unit.as_deref().map(Arc::from),
            })
        })
        .collect()
}

/// A scenario row's identification status in the shared transport vocabulary.
fn identification_status(status: &str) -> &'static str {
    antecedent_core::TransportOutcomeKind::from_identification_status(status)
        .unwrap_or(antecedent_core::TransportOutcomeKind::Identified)
        .as_str()
}

fn cpdag_report_json(report: &CpdagScenarioReport, names: &[String]) -> serde_json::Value {
    let name =
        |v: &VariableId| names.get(v.as_usize()).cloned().unwrap_or_else(|| v.raw().to_string());
    let results = report.report.as_ref().map_or(&[][..], |r| r.scenarios.as_slice());
    let point = |id: &str| {
        results.iter().find(|s| &*s.name == id).and_then(|s| s.distribution.as_ref()).map(|d| {
            serde_json::json!({
                "outcomes": d.outcomes.iter().map(name).collect::<Vec<_>>(),
                "atoms": d.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "probabilities": d.probabilities.as_ref(),
                "means": d.outcomes.iter().map(|o| (name(o), d.mean(*o).ok())).collect::<std::collections::BTreeMap<_, _>>(),
            })
        })
    };
    serde_json::json!({
        "status": "available",
        "scope": CPDAG_SCOPE,
        "cpdag_identity": report.cpdag_identity.as_ref(),
        "completions": report.completions.iter().map(|c| serde_json::json!({
            "id": c.id.as_ref(),
            "edges": c.edges.iter().map(|(p, ch)| [name(p), name(ch)]).collect::<Vec<_>>(),
            "status": c.status,
            "identification_status": identification_status(c.status),
            "detail": c.detail.as_ref().map(|d| resolve_variable_ids(d, names)),
            "evidence_identity": c.evidence_identity.as_ref().map(ToString::to_string),
            "point": point(&c.id),
        })).collect::<Vec<_>>(),
        "counts": {
            "identified": report.identified,
            "unidentified": report.unidentified,
            "unevaluated": report.unevaluated,
            "not_enumerated": report.not_enumerated,
            "total": report.total(),
        },
        "masses": report.masses().iter().map(|m| serde_json::json!({
            "status": m.status, "count": m.count,
        })).collect::<Vec<_>>(),
        "envelope": report.report.as_ref().and_then(|r| r.envelope.as_ref()).map(|e| serde_json::json!({
            "scenarios": e.scenarios.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "means": e.means.iter().map(|m| serde_json::json!({
                "outcome": name(&m.outcome),
                "lower": m.lower,
                "upper": m.upper,
                "lower_scenario": m.lower_scenario.as_ref(),
                "upper_scenario": m.upper_scenario.as_ref(),
            })).collect::<Vec<_>>(),
            "atoms": e.atoms,
            "interpretation": e.interpretation,
        })),
        "receipt": report.receipt.as_ref().map(|r| serde_json::json!({
            "stop": r.stop.code(),
            "operations_limit": r.operations_limit,
            "operations_consumed": r.operations_consumed,
            "explored": r.explored,
            "unevaluated": r.unevaluated,
            "not_enumerated": report.not_enumerated,
        })),
        "exportable": report.is_exportable(),
    })
}

/// A decided, compiled and evaluated set of CPDAG completions.
#[pyclass(skip_from_py_object)]
struct CpdagScenarioRun {
    inner: antecedent::PreparedCpdagCompletionScenarios,
    report: CpdagScenarioReport,
    names: Vec<String>,
    payload: String,
}

#[pymethods]
impl CpdagScenarioRun {
    /// The report as JSON.
    #[getter]
    fn payload_json(&self) -> String {
        self.payload.clone()
    }

    /// The report as a framed, independently consumable artifact, or the
    /// structured refusal (a report cut short by cancellation never exports).
    fn export<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Option<Bound<'py, PyBytes>>, Option<Refusal>)> {
        match self.inner.export(&self.report) {
            Ok(raw) => {
                let framed = frame_named_artifact(CPDAG_PREFIX, &self.names, raw)?;
                Ok((Some(PyBytes::new(py, &framed)), None))
            }
            Err(e) => Ok((None, Some(refusal_or_error(e)?))),
        }
    }
}

/// One completion: its sorted edge list and its identity.
type CompletionIds = Vec<(Vec<(VariableId, VariableId)>, Arc<str>)>;

/// Completion ids by their sorted edge lists, for evidence declared by edges.
fn completion_ids(cpdag: &antecedent_graph::Cpdag) -> Result<CompletionIds, Refusal> {
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let limits = SearchLimits { operations: 10_000_000, depth: 64 };
    let enumeration =
        enumerate_cpdag_completions(&CpdagCompletionInput::new(cpdag.clone()), limits, &ctx)
            .map_err(|r| refusal_json(r.code, &format!("{}: {}", r.detail, r.message)))?;
    Ok(enumeration
        .completions
        .iter()
        .map(|c| (c.edges.to_vec(), Arc::clone(&c.identity)))
        .collect())
}

fn evidence_from_python(
    mode: &str,
    declared: &[EvidenceTuple<'_>],
    catalogs: Vec<antecedent_core::EvidenceCatalog>,
    names: &[String],
    cpdag: &antecedent_graph::Cpdag,
) -> PyResult<Result<CpdagEvidence, Refusal>> {
    match mode {
        "shared" => {
            let (Some(catalog), true) = (catalogs.into_iter().next(), declared.len() == 1) else {
                return Err(crate::value_err("shared evidence declares exactly one catalog"));
            };
            let (completion, edges, identity, _, _) = &declared[0];
            if completion.is_some() || edges.is_some() {
                return Err(crate::value_err("shared evidence names no completion"));
            }
            Ok(Ok(CpdagEvidence::Shared {
                evidence_identity: Arc::from(identity.as_str()),
                catalog,
            }))
        }
        "per_completion" => {
            let table = if declared.iter().any(|d| d.1.is_some()) {
                match completion_ids(cpdag) {
                    Ok(table) => table,
                    Err(refusal) => return Ok(Err(refusal)),
                }
            } else {
                Vec::new()
            };
            let mut bindings = Vec::with_capacity(declared.len());
            for ((completion, edges, identity, _, certified), catalog) in
                declared.iter().zip(catalogs)
            {
                let id: Arc<str> = match (completion, edges) {
                    (Some(id), None) => Arc::from(id.as_str()),
                    (None, Some(edges)) => {
                        let mut wanted = edges
                            .iter()
                            .map(|(a, b)| Ok((resolve(names, a)?, resolve(names, b)?)))
                            .collect::<PyResult<Vec<_>>>()?;
                        wanted.sort_unstable();
                        match table.iter().find(|(known, _)| *known == wanted) {
                            Some((_, id)) => Arc::clone(id),
                            None => {
                                return Ok(Err(argument_refusal(
                                    reason_code!("invalid_argument"),
                                    antecedent_identify::sid::cpdag_completion::CPDAG_EVIDENCE_MISMATCH_DETAIL,
                                    "the declared edges are not a completion of this CPDAG",
                                )));
                            }
                        }
                    }
                    _ => {
                        return Err(crate::value_err(
                            "an evidence binding names its completion by id or by edges, not both",
                        ));
                    }
                };
                bindings.push(CompletionEvidenceBinding {
                    certified_for: certified.as_deref().map_or_else(|| Arc::clone(&id), Arc::from),
                    completion: id,
                    evidence_identity: Arc::from(identity.as_str()),
                    catalog,
                });
            }
            Ok(Ok(CpdagEvidence::PerCompletion(bindings)))
        }
        _ => Err(crate::value_err("evidence mode is shared or per_completion")),
    }
}

/// Accept an `ExactTransportData` wrapper or a sequence of laws.
fn law_sequence<'py>(laws: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if laws.hasattr("laws")? { laws.getattr("laws") } else { Ok(laws.clone()) }
}

/// Enumerate the completions of a CPDAG, bind evidence to each, decide and
/// compile them against exact laws, and evaluate. Returns the run or a refusal.
#[pyfunction]
#[pyo3(signature=(cpdag, coordinates, outcomes, treatments, source, target, evidence_mode, evidence, laws, assignments, *, max_steps=100_000, max_depth=256, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn cpdag_completion_scenarios_stage(
    py: Python<'_>,
    cpdag: PyRef<'_, Cpdag>,
    coordinates: Vec<CoordinateTuple>,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    source: String,
    target: String,
    evidence_mode: &str,
    evidence: Vec<EvidenceTuple<'_>>,
    laws: &Bound<'_, PyAny>,
    assignments: std::collections::BTreeMap<String, f64>,
    max_steps: usize,
    max_depth: usize,
    max_operations: usize,
    max_evaluation_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<CpdagScenarioRun>, Option<Refusal>)> {
    let names = cpdag.names.clone();
    let aligned = names.len() == cpdag.cpdag.node_count()
        && cpdag
            .cpdag
            .nodes()
            .iter()
            .enumerate()
            .all(|(i, node)| matches!(node, NodeRef::Static(v) if v.as_usize() == i));
    if !aligned {
        return Err(crate::value_err("the CPDAG's variable ids must match its name positions"));
    }
    let named = Admg { admg: antecedent_graph::Admg::empty(), names: names.clone() };
    let coordinates = parse_coordinates(&coordinates, &names)?;
    let resolved = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables.iter().map(|n| resolve(&names, n)).collect::<PyResult<Vec<_>>>().map(Arc::from)
    };
    let query = ClassicalTransportQuery {
        outcomes: resolved(&outcomes)?,
        treatments: resolved(&treatments)?,
        source: source.into(),
        target: target.into(),
    };
    if laws.hasattr("samples")? {
        return Ok((
            None,
            Some(argument_refusal(
                reason_code!("route_not_supported"),
                "cpdag_scenarios.exact_laws_only",
                "the completion route evaluates supplied exact laws; counted laws are not licensed",
            )),
        ));
    }
    let catalogs =
        evidence.iter().map(|d| parse_catalog(&d.3, &named)).collect::<PyResult<Vec<_>>>()?;
    let Some(first) = catalogs.first() else {
        return Err(crate::value_err("at least one evidence binding is needed to read the laws"));
    };
    let check = if catalogs.len() == 1 { RegimeCheck::Strict } else { RegimeCheck::Lenient };
    let data = parse_laws(&law_sequence(laws)?, first, &named, max_support_rows, check)?;
    let request = assignment_from_pairs(&names, assignments)?;
    let bound =
        match evidence_from_python(evidence_mode, &evidence, catalogs, &names, &cpdag.cpdag)? {
            Ok(bound) => bound,
            Err(refusal) => return Ok((None, Some(refusal))),
        };
    let input = CpdagCompletionInput::new(cpdag.cpdag.clone());
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let budget = SearchLimits { operations: max_steps, depth: max_depth };
        let evaluation =
            ExactEvaluationLimits { operations: max_operations, depth: max_evaluation_depth };
        let prepared = antecedent::StudyBuilder::cpdag_completion_scenarios(
            input,
            coordinates,
            query,
            bound,
            budget,
            data,
            request,
            evaluation,
            &ctx,
        )
        .and_then(|prepared| prepared.estimate(&ctx).map(|report| (prepared, report)));
        match prepared {
            Ok(done) => Ok(Ok(done)),
            Err(e) => refusal_or_error(e).map(Err),
        }
    })?;
    match outcome {
        Ok((inner, report)) => {
            let payload = cpdag_report_json(&report, &names).to_string();
            Ok((Some(CpdagScenarioRun { inner, report, names, payload }), None))
        }
        Err(refusal) => Ok((None, Some(refusal))),
    }
}

/// Independently replay a framed completion artifact under the consumer's limits.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_steps=100_000, max_depth=256, max_operations=10_000_000, max_evaluation_depth=256, max_support_rows=1_000_000, max_laws=256, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_cpdag_scenarios_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_steps: usize,
    max_depth: usize,
    max_operations: usize,
    max_evaluation_depth: usize,
    max_support_rows: usize,
    max_laws: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<String>, Option<Refusal>)> {
    let (names, bytes) = unframe_named_artifact(CPDAG_PREFIX, artifact, "cpdag scenario")?;
    let limits = CpdagConsumeLimits {
        search_budget: SearchLimits { operations: max_steps, depth: max_depth },
        evaluation: ExactEvaluationLimits {
            operations: max_operations,
            depth: max_evaluation_depth,
        },
        max_support_rows,
        max_laws,
    };
    crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let (wire, report) =
            match CpdagCompletionArtifactWire::consume_with_limits(&bytes, limits, &ctx) {
                Ok(done) => done,
                Err(e) => return refusal_or_error(e).map(|r| (None, Some(r))),
            };
        // The frame's names must be exactly the ones the identity binds: the
        // report is read with them.
        let bound = wire.coordinates.len() == names.len()
            && wire.coordinates.iter().all(|c| {
                usize::try_from(c.variable).ok().and_then(|i| names.get(i)) == Some(&c.name)
            });
        if !bound {
            return Err(serialization_error(
                "cpdag scenario artifact names disagree with its coordinate schema",
            ));
        }
        let mut payload = cpdag_report_json(&report, &names);
        payload["premises_digest"] = wire.premises_digest.clone().into();
        payload["data_digest"] = wire.data_digest.clone().into();
        Ok((Some(payload.to_string()), None))
    })
}

fn covariance_json(cov: &ScenarioCovariance) -> serde_json::Value {
    let k = cov.dimension();
    serde_json::json!({
        "status": "available",
        "scope": COVARIANCE_SCOPE,
        "claim": "point_only",
        "interpretation": cov.interpretation,
        "method": cov.method.label(),
        "scenario_ids": cov.scenario_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "means": cov.means,
        "covariance": (0..k).map(|i| (0..k).map(|j| cov.entry(i, j)).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "n_rows": cov.n_rows,
        "replicates_total": cov.replicates_total,
        "replicates_used": cov.replicates_used,
        "failed_replicates": cov.failed_replicates,
        "failed_mass": cov.failed_mass,
        "seed": cov.seed,
        "replicate_digest": cov.replicate_digest,
        "row_identity_digest": cov.row_identity_digest,
        "snapshot_digest": cov.snapshot_digest.as_ref(),
    })
}

/// A computed shared-data covariance with the declarations it came from.
#[pyclass(skip_from_py_object)]
struct ScenarioCovarianceRun {
    spec: CovarianceSpec,
    covariance: ScenarioCovariance,
    payload: String,
}

#[pymethods]
impl ScenarioCovarianceRun {
    /// The covariance as JSON.
    #[getter]
    fn payload_json(&self) -> String {
        self.payload.clone()
    }

    /// The covariance as an independently recomputable artifact, or the refusal.
    fn export<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Option<Bound<'py, PyBytes>>, Option<Refusal>)> {
        match antecedent::export_scenario_covariance(&self.spec, &self.covariance) {
            Ok(raw) => {
                let mut framed = COVARIANCE_PREFIX.to_vec();
                framed.extend(raw);
                Ok((Some(PyBytes::new(py, &framed)), None))
            }
            Err(e) => Ok((None, Some(refusal_or_error(e)?))),
        }
    }
}

/// Compute the joint covariance of declared scenario estimates over one row table.
#[pyfunction]
#[pyo3(signature=(spec_json, *, memory_bytes=None, cancel=None))]
fn scenario_shared_covariance_stage(
    py: Python<'_>,
    spec_json: &str,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<ScenarioCovarianceRun>, Option<Refusal>)> {
    if spec_json.len() > MAX_SPEC_JSON_BYTES {
        return Err(crate::value_err("the covariance declaration is too large"));
    }
    let spec: CovarianceSpec = match serde_json::from_str(spec_json) {
        Ok(spec) => spec,
        Err(e) => {
            return Ok((
                None,
                Some(argument_refusal(
                    reason_code!("invalid_argument"),
                    "scenario_covariance.invalid_declaration",
                    &e.to_string(),
                )),
            ));
        }
    };
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        match antecedent::StudyBuilder::scenario_shared_covariance(&spec, &ctx) {
            Ok(covariance) => Ok(Ok((spec, covariance))),
            Err(e) => refusal_or_error(e).map(Err),
        }
    })?;
    match outcome {
        Ok((spec, covariance)) => {
            let payload = covariance_json(&covariance).to_string();
            Ok((Some(ScenarioCovarianceRun { spec, covariance, payload }), None))
        }
        Err(refusal) => Ok((None, Some(refusal))),
    }
}

/// Independently recompute a covariance artifact under the consumer's limits.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_rows=10_000, max_columns=64, max_replicates=2000, max_compositions=4_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_scenario_covariance_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_rows: usize,
    max_columns: usize,
    max_replicates: usize,
    max_compositions: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<String>, Option<Refusal>)> {
    let bytes = artifact
        .strip_prefix(COVARIANCE_PREFIX)
        .ok_or_else(|| serialization_error("invalid scenario covariance artifact format"))?
        .to_vec();
    let limits =
        ScenarioCovarianceConsumeLimits { max_rows, max_columns, max_replicates, max_compositions };
    crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        match ScenarioCovarianceArtifactWire::consume_with_limits(&bytes, limits, &ctx) {
            Ok((wire, covariance)) => {
                let mut payload = covariance_json(&covariance);
                payload["premises_digest"] = wire.premises_digest.clone().into();
                payload["data_digest"] = wire.data_digest.clone().into();
                Ok((Some(payload.to_string()), None))
            }
            Err(e) => refusal_or_error(e).map(|r| (None, Some(r))),
        }
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CpdagScenarioRun>()?;
    module.add_class::<ScenarioCovarianceRun>()?;
    module.add_function(wrap_pyfunction!(cpdag_completion_scenarios_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_cpdag_scenarios_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(scenario_shared_covariance_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_scenario_covariance_artifact, module)?)?;
    Ok(())
}
