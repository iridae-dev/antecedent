//! Python bindings for general finite counterfactual transport on exact laws.
//!
//! A decision is taken once from a selection diagram, declared levels, a
//! nested or conditional counterfactual event over worlds, per-source
//! selection targets and controllable variables, and a catalog of evidence
//! regimes. An identified decision is a frozen derivation that is evaluated on
//! exact finite laws without searching again. The route is point-only: exact
//! laws carry no sampling uncertainty and no interval is published. Sampled
//! evidence is evaluated against the same derivation by
//! `transport_counterfactual_sampled_api`.
use crate::graphs::Admg;
use crate::transport_common::{
    RegimeCheck, error, execution_context, parse_laws, resolve, resolve_variable_ids,
};
use crate::transport_interference_api::parse_catalog;
use crate::transport_z_api::to_py_json;
use antecedent::{
    CounterfactualTransportConsumption, CounterfactualTransportObstructionReport,
    CounterfactualTransportOptions, CounterfactualTransportPreparation,
    CounterfactualTransportResult, PreparedCounterfactualTransport,
    consume_counterfactual_transport_artifact, prepare_counterfactual_transport,
};
use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, EdgeRoute, EvidenceCatalog, ExogenousCoupling,
    SearchLimits, VariableId, WorldId, WorldSpec,
};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, CounterfactualTransportProblem,
    CounterfactualTransportQuery, CounterfactualTransportSource, counterfactual_transport_refusal,
    identify_counterfactual_transport,
};
use antecedent_io::IoError;
use antecedent_io::counterfactual_transport_artifact::CounterfactualTransportConsumeLimits;
use antecedent_io::counterfactual_transport_wire::{
    CounterfactualTransportArtifactWire, CounterfactualTransportExpectedIdentity,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Exact finite laws only: the point has no sampling uncertainty.
const EXACT_SCOPE: &str = "counterfactual_transport_exact_finite_laws_point_only";

/// One Python world: `(interventions, routes (parent, child, source world))`.
type WorldTuple = (BTreeMap<String, f64>, Vec<(String, String, usize)>);
/// One Python atom: `(world index, variable, level)`.
type AtomTuple = (usize, String, f64);
/// One Python source: `(population, controllable, selection targets)`.
type SourceTuple = (String, Vec<String>, Vec<String>);

pub(crate) fn variable_name(names: &[String], id: VariableId) -> String {
    names.get(id.raw() as usize).cloned().unwrap_or_else(|| id.raw().to_string())
}

pub(crate) fn regime_label(catalog: &EvidenceCatalog, id: antecedent_core::RegimeId) -> String {
    catalog
        .regimes
        .iter()
        .find(|r| r.id == id)
        .and_then(|r| r.label.as_deref())
        .map_or_else(|| id.raw().to_string(), ToString::to_string)
}

/// A refusal of the route as the record's frozen `(reason code,
/// counterfactual_transport.* detail)` pair; budgets and cancellation and
/// errors the route does not own keep the shared identification mapping.
fn identification_refusal(e: antecedent_identify::IdentificationError) -> PyErr {
    if e.is_budget_or_cancel() {
        return error(e);
    }
    match counterfactual_transport_refusal(&e) {
        Some((code, detail)) => {
            let text = e.to_string();
            let message = if text.starts_with(detail) { text } else { format!("{detail}: {text}") };
            crate::refusal(code, message)
        }
        None => error(e),
    }
}

pub(crate) fn io_refusal(code: &'static str, message: String) -> PyErr {
    error(IoError::Refused { code, message })
}

fn interventions_json(interventions: &[(VariableId, f64)], names: &[String]) -> serde_json::Value {
    interventions
        .iter()
        .map(|(v, level)| (variable_name(names, *v), serde_json::json!(level)))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

fn derivation_json(
    derivation: &CounterfactualTransportDerivation,
    names: &[String],
) -> serde_json::Value {
    serde_json::json!({
        "contract": derivation.contract,
        "canonical_text": derivation.canonical_text(),
        "populations": derivation.populations.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "rules": derivation.rules,
        "conditioning_mass_checked": derivation.conditioning_mass_checked,
        "cited": derivation.cited.iter().map(|c| serde_json::json!({
            "population": c.population.as_ref(),
            "regime": regime_label(&derivation.catalog, c.regime),
            "interventions": c.interventions.iter().map(|v| variable_name(names, *v)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "ignored": derivation.ignored.iter().map(|(id, why)| serde_json::json!({
            "regime": regime_label(&derivation.catalog, *id),
            "reason": why,
        })).collect::<Vec<_>>(),
        "search_receipt": {
            "operations_limit": derivation.search.operations_limit,
            "depth_limit": derivation.search.depth_limit,
            "memory_limit_bytes": derivation.search.memory_limit_bytes,
            "operations_consumed": derivation.search.operations_consumed,
            "depth_reached": derivation.search.depth_reached,
            "stages": derivation.search.stages,
        },
    })
}

fn point_json(result: &CounterfactualTransportResult, names: &[String]) -> serde_json::Value {
    let derivation = result.derivation();
    serde_json::json!({
        "scope": EXACT_SCOPE,
        "probability": result.probability(),
        "numerator": result.numerator(),
        "denominator": result.denominator(),
        "conditioning_mass": result.conditioning_mass(),
        "conditioning_mass_checked": result.conditioning_mass_checked(),
        "cells_read": result.cells_read(),
        "uncertainty": serde_json::Value::Null,
        "cited_laws": result.cited_laws().iter().map(|law| serde_json::json!({
            "population": law.population.as_ref(),
            "regime": regime_label(&derivation.catalog, law.regime),
            "interventions": interventions_json(&law.interventions, names),
            "snapshot_identity": law.snapshot_identity,
            "content_identity": law.content_identity,
        })).collect::<Vec<_>>(),
        "derivation": derivation_json(derivation, names),
    })
}

fn obstruction_json(
    report: &CounterfactualTransportObstructionReport,
    names: &[String],
) -> serde_json::Value {
    let obstruction = report.obstruction();
    let witness = serde_json::to_value(&obstruction.witness).unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "outcome": "proven_non_transportable",
        "reason": "transport_proven_non_transportable",
        "detail": "counterfactual_transport.checked_obstruction",
        "explanation": obstruction.reason,
        "witness": witness,
        "witness_clique": obstruction.witness.clique.iter()
            .map(|v| variable_name(names, VariableId::from_raw(*v)))
            .collect::<Vec<_>>(),
        "check": {
            "cells_compared": obstruction.check.cells_compared,
            "first_value": obstruction.check.first_value,
            "second_value": obstruction.check.second_value,
        },
    })
}

fn missing_regimes_text(
    missing: &antecedent_identify::CounterfactualTransportMissing,
    names: &[String],
) -> String {
    missing
        .missing
        .iter()
        .map(|m| {
            if m.interventions.is_empty() {
                format!("the observational law of population {}", m.population)
            } else {
                let vars: Vec<String> =
                    m.interventions.iter().map(|v| variable_name(names, *v)).collect();
                format!("the experiment on [{}] in population {}", vars.join(", "), m.population)
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The payload and `(reason, detail, text)` of a typed decision that is not an
/// identification or an obstruction.
fn unprepared_payload(
    decision: &CounterfactualTransportDecision,
    names: &[String],
) -> Option<(serde_json::Value, &'static str, String, String)> {
    let code = decision.reason_code()?;
    let detail = decision.detail_code().unwrap_or_default();
    let (mut payload, text) = match decision {
        CounterfactualTransportDecision::MissingEvidence(missing) => {
            let regimes = missing_regimes_text(missing, names);
            let message = resolve_variable_ids(&missing.detail, names);
            (
                serde_json::json!({
                    "outcome": "missing_evidence",
                    "missing_regimes": missing.missing.iter().map(|m| serde_json::json!({
                        "population": m.population.as_ref(),
                        "interventions": m.interventions.iter()
                            .map(|v| variable_name(names, *v)).collect::<Vec<_>>(),
                    })).collect::<Vec<_>>(),
                    "message": message,
                }),
                format!("{message}; supply a catalog regime and an exact law for {regimes}"),
            )
        }
        CounterfactualTransportDecision::UnsupportedScope(unsupported) => (
            serde_json::json!({"outcome": "unsupported_scope", "message": unsupported.message}),
            unsupported.message.clone(),
        ),
        CounterfactualTransportDecision::NotCertified(inspection) => (
            serde_json::json!({
                "outcome": "not_certified",
                "failures": inspection.failures,
                "witness_search": inspection.witness,
                "stages": inspection.stages,
            }),
            format!(
                "the bounded search found neither a derivation nor a verified two-model \
                 witness, which is not a proof of non-transportability ({}; {})",
                inspection.failures.join("; "),
                inspection.witness
            ),
        ),
        CounterfactualTransportDecision::Exhausted(receipt) => (
            serde_json::json!({
                "outcome": "exhausted",
                "limits_receipt": {
                    "stop": receipt.stop.code(),
                    "operations_limit": receipt.operations_limit,
                    "depth_limit": receipt.depth_limit,
                    "memory_limit_bytes": receipt.memory_limit_bytes,
                    "operations_consumed": receipt.operations_consumed,
                    "depth_reached": receipt.depth_reached,
                    "explored": receipt.explored,
                    "unevaluated": receipt.unevaluated,
                },
            }),
            format!("the search stopped before deciding: {}", receipt.summary()),
        ),
        CounterfactualTransportDecision::Identified(_)
        | CounterfactualTransportDecision::ProvenNonTransportable(_) => return None,
    };
    payload["reason"] = code.into();
    payload["detail"] = detail.into();
    Some((payload, code, detail.to_string(), text))
}

/// A decision of one request: an identified derivation, a proven obstruction,
/// or the typed reason it is neither.
#[pyclass(skip_from_py_object)]
struct CounterfactualTransportDecisionNative {
    status: &'static str,
    payload: serde_json::Value,
    /// `(reason code, detail, text)` of preparing a decision that is not identified.
    refusal: Option<(&'static str, String, String)>,
    prepared: Option<PreparedCounterfactualTransport>,
    obstruction: Option<CounterfactualTransportObstructionReport>,
    graph: Option<Admg>,
    catalog: Option<EvidenceCatalog>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl CounterfactualTransportDecisionNative {
    /// `identified`, `proven_non_transportable`, `missing_evidence`,
    /// `not_certified` or `budget_cancel`.
    #[getter]
    fn identification_status(&self) -> &'static str {
        self.status
    }

    /// The decision in variable names and catalog regime ids.
    fn decision(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut value = self.payload.clone();
        value["identification_status"] = self.status.into();
        to_py_json(py, &value)
    }

    /// The prepared derivation of an identified decision; any other decision
    /// raises its typed reason.
    fn prepared(&self) -> PyResult<PreparedCounterfactualTransportNative> {
        if let (Some(prepared), Some(graph), Some(catalog)) =
            (&self.prepared, &self.graph, &self.catalog)
        {
            Ok(PreparedCounterfactualTransportNative {
                inner: prepared.clone(),
                graph: graph.clone(),
                catalog: catalog.clone(),
                max_support_rows: self.max_support_rows,
                memory_bytes: self.memory_bytes,
            })
        } else {
            {
                let (code, detail, text) = self.refusal.clone().unwrap_or((
                    antecedent_core::reason_code!("transport_not_certified"),
                    String::new(),
                    String::from("only an identified decision can be prepared"),
                ));
                Err(crate::refusal(
                    code,
                    format!("{detail}: {text}; only an identified decision can be prepared"),
                ))
            }
        }
    }

    /// The proven obstruction as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let report = self.obstruction.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: only a proven obstruction exports without evidence; \
                 evaluate an identified decision and export its result",
            )
        })?;
        let bytes = report.export_artifact().map_err(error)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    /// The expected `(premises digest, data digest)` of the obstruction artifact.
    fn identity(&self) -> PyResult<(Option<String>, Option<String>)> {
        let report = self.obstruction.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: only a proven obstruction has an identity here",
            )
        })?;
        let bytes = report.export_artifact().map_err(error)?;
        let wire = CounterfactualTransportArtifactWire::decode(&bytes).map_err(|e| {
            let (code, message) = e.refusal();
            io_refusal(code, message)
        })?;
        let identity = wire.identity();
        Ok((identity.premises_digest, identity.data_digest))
    }
}

/// An identified derivation with its catalog; evaluation never searches.
#[pyclass(skip_from_py_object)]
pub(crate) struct PreparedCounterfactualTransportNative {
    pub(crate) inner: PreparedCounterfactualTransport,
    pub(crate) graph: Admg,
    pub(crate) catalog: EvidenceCatalog,
    max_support_rows: usize,
    pub(crate) memory_bytes: Option<u64>,
}

#[pymethods]
impl PreparedCounterfactualTransportNative {
    /// The frozen derivation: canonical text, rules, cited regimes, receipt.
    fn derivation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(py, &derivation_json(self.inner.derivation(), &self.graph.names))
    }

    /// Whether evidence is bound (a `refresh` has run).
    #[getter]
    fn bound(&self) -> bool {
        self.inner.data().is_some()
    }

    /// Evaluate the frozen derivation on `laws` without binding them.
    #[pyo3(signature=(laws, *, memory_bytes=None, cancel=None))]
    fn evaluate(
        &self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<CounterfactualTransportResultNative> {
        let data = parse_laws(
            laws,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let memory = memory_bytes.or(self.memory_bytes);
        let inner = self.inner.clone();
        let result = crate::detach_catch(py, move || {
            let ctx = execution_context(0, memory, cancel);
            inner.evaluate(&data, &ctx).map_err(error)
        })?;
        Ok(CounterfactualTransportResultNative { inner: result, names: self.graph.names.clone() })
    }

    /// A prepared value bound to `laws`; this value is untouched, and a refused
    /// refresh raises. Laws of another population or regime are refused.
    #[pyo3(signature=(laws, *, memory_bytes=None, cancel=None))]
    fn refresh(
        &self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedCounterfactualTransportNative> {
        let data = parse_laws(
            laws,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let memory = memory_bytes.or(self.memory_bytes);
        let inner = self.inner.clone();
        let refreshed = crate::detach_catch(py, move || {
            let ctx = execution_context(0, memory, cancel);
            inner.refresh(data, &ctx).map_err(error)
        })?;
        Ok(PreparedCounterfactualTransportNative {
            inner: refreshed,
            graph: self.graph.clone(),
            catalog: self.catalog.clone(),
            max_support_rows: self.max_support_rows,
            memory_bytes: self.memory_bytes,
        })
    }

    /// The numerical state of the bound evidence.
    #[pyo3(signature=(*, memory_bytes=None, cancel=None))]
    fn estimate(
        &self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<CounterfactualTransportResultNative> {
        let memory = memory_bytes.or(self.memory_bytes);
        let inner = self.inner.clone();
        let result = crate::detach_catch(py, move || {
            let ctx = execution_context(0, memory, cancel);
            inner.estimate(&ctx).cloned().map_err(error)
        })?;
        Ok(CounterfactualTransportResultNative { inner: result, names: self.graph.names.clone() })
    }
}

/// The exact point of one evaluation with everything needed to export it.
#[pyclass(skip_from_py_object)]
struct CounterfactualTransportResultNative {
    inner: CounterfactualTransportResult,
    names: Vec<String>,
}

#[pymethods]
impl CounterfactualTransportResultNative {
    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(py, &point_json(&self.inner, &self.names))
    }

    /// The point as a `counterfactual_transport_v1` artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let bytes = self.inner.export_artifact().map_err(error)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    /// `(premises digest, data digest)`, independent of the bytes.
    fn identity(&self) -> PyResult<(Option<String>, Option<String>)> {
        let identity = self.inner.identity().map_err(error)?;
        Ok((identity.premises_digest, identity.data_digest))
    }
}

// ---------------------------------------------------------------------------
// Parsing.
// ---------------------------------------------------------------------------

fn world_id(index: usize) -> PyResult<WorldId> {
    u8::try_from(index).map(WorldId::new).map_err(error)
}

fn parse_atoms(names: &[String], atoms: Vec<AtomTuple>) -> PyResult<Vec<CounterfactualEvent>> {
    atoms
        .into_iter()
        .map(|(world, variable, level)| {
            CounterfactualEvent::new(world_id(world)?, resolve(names, &variable)?, level)
                .map_err(error)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Decide a counterfactual transport request once.
///
/// `worlds` are `(interventions, routes)`, with routes `(parent, child,
/// source world index)`; atoms are `(world index, variable, level)`; sources
/// are `(population, controllable, selection targets)`; `levels` declares every
/// variable's finite levels.
#[pyfunction]
#[pyo3(signature=(graph, levels, worlds, event, given, target, sources, catalog, *, max_operations=None, max_depth=None, max_eval_operations=None, max_eval_depth=None, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::too_many_lines,
    reason = "one flat decode of each declared argument into the typed problem, then a single identification call"
)]
#[doc(hidden)]
fn identify_counterfactual_transport_native(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    levels: Vec<(String, Vec<f64>)>,
    worlds: Vec<WorldTuple>,
    event: Vec<AtomTuple>,
    given: Vec<AtomTuple>,
    target: String,
    sources: Vec<SourceTuple>,
    catalog: &Bound<'_, PyAny>,
    max_operations: Option<usize>,
    max_depth: Option<usize>,
    max_eval_operations: Option<usize>,
    max_eval_depth: Option<usize>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CounterfactualTransportDecisionNative> {
    let names = graph.names.clone();
    let named = Admg { admg: graph.admg.clone(), names: names.clone() };
    let shared = graph.aligned_to_names(&names)?;
    let declared = levels
        .into_iter()
        .map(|(name, values)| Ok((resolve(&names, &name)?, values)))
        .collect::<PyResult<Vec<_>>>()?;
    let problem = CounterfactualTransportProblem::from_admg(&shared, &declared)
        .map_err(identification_refusal)?;
    let mut specs = Vec::with_capacity(worlds.len());
    for (interventions, routes) in worlds {
        let interventions = interventions
            .into_iter()
            .map(|(name, level)| Ok((resolve(&names, &name)?, level)))
            .collect::<PyResult<Vec<_>>>()?;
        let routes = routes
            .into_iter()
            .map(|(parent, child, source)| {
                Ok(EdgeRoute {
                    parent: resolve(&names, &parent)?,
                    child: resolve(&names, &child)?,
                    source: world_id(source)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        specs.push(WorldSpec::new(interventions, routes).map_err(error)?);
    }
    let event_query = CounterfactualEventQuery::new(
        specs,
        ExogenousCoupling::SharedLatentExogenous,
        parse_atoms(&names, event)?,
        parse_atoms(&names, given)?,
    )
    .map_err(error)?;
    let coordinates = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(&names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let mut declared_sources = Vec::with_capacity(sources.len());
    for (population, controllable, selections) in &sources {
        declared_sources.push(CounterfactualTransportSource {
            population: Arc::from(population.as_str()),
            selection_targets: coordinates(selections)?,
            controllable: coordinates(controllable)?,
        });
    }
    let query = CounterfactualTransportQuery {
        event: event_query,
        target: Arc::from(target.as_str()),
        sources: declared_sources.into(),
    };
    let catalog = parse_catalog(catalog, &named)?;
    let defaults = ExactEvaluationLimits::default();
    let options = CounterfactualTransportOptions {
        search: SearchLimits {
            operations: max_operations
                .unwrap_or(COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS.operations),
            depth: max_depth.unwrap_or(COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS.depth),
        },
        evaluation: ExactEvaluationLimits {
            operations: max_eval_operations.unwrap_or(defaults.operations),
            depth: max_eval_depth.unwrap_or(defaults.depth),
        },
        variable_names: names.clone(),
    };
    let search = options.search;
    let (decision_catalog, decision_names) = (catalog.clone(), names.clone());
    let decided = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        match prepare_counterfactual_transport(
            &problem,
            &query,
            &decision_catalog,
            None,
            &options,
            &ctx,
        ) {
            Ok(CounterfactualTransportPreparation::Identified(prepared)) => {
                let mut payload = serde_json::json!({"outcome": "identified"});
                let derivation = derivation_json(prepared.derivation(), &decision_names);
                payload["derivation"] = derivation;
                Ok(Decided {
                    status: "identified",
                    payload,
                    refusal: None,
                    prepared: Some(*prepared),
                    obstruction: None,
                })
            }
            Ok(CounterfactualTransportPreparation::ProvenNonTransportable(report)) => {
                let payload = obstruction_json(&report, &decision_names);
                Ok(Decided {
                    status: "proven_non_transportable",
                    refusal: Some((
                        antecedent_core::reason_code!("transport_proven_non_transportable"),
                        "counterfactual_transport.checked_obstruction".to_string(),
                        report.reason().to_string(),
                    )),
                    payload,
                    prepared: None,
                    obstruction: Some(*report),
                })
            }
            Err(IoError::Refused { code, message })
                if [
                    antecedent_core::reason_code!("transport_missing_evidence"),
                    antecedent_core::reason_code!("transport_not_certified"),
                    antecedent_core::reason_code!("route_not_supported"),
                    antecedent_core::reason_code!("transport_budget_cancel"),
                ]
                .contains(&code) =>
            {
                // The refusal carries the facade's text; the typed decision
                // carries the structure (missing regimes, receipt, stages).
                match identify_counterfactual_transport(
                    &problem,
                    &query,
                    &decision_catalog,
                    search,
                    &ctx,
                ) {
                    Ok(decision) => match unprepared_payload(&decision, &decision_names) {
                        Some((payload, code, detail, text)) => Ok(Decided {
                            status: decision.identification_status().as_str(),
                            payload,
                            refusal: Some((code, detail, text)),
                            prepared: None,
                            obstruction: None,
                        }),
                        None => Err(io_refusal(code, message)),
                    },
                    Err(e) if e.is_budget_or_cancel() => {
                        let payload = serde_json::json!({
                            "outcome": "exhausted",
                            "reason": code,
                            "detail": "counterfactual_transport.budget",
                            "message": message,
                            "limits_receipt": {
                                "operations_limit": search.operations,
                                "depth_limit": search.depth,
                            },
                        });
                        Ok(Decided {
                            status: "budget_cancel",
                            payload,
                            refusal: Some((
                                code,
                                "counterfactual_transport.budget".to_string(),
                                message,
                            )),
                            prepared: None,
                            obstruction: None,
                        })
                    }
                    Err(e) => Err(identification_refusal(e)),
                }
            }
            Err(other) => Err(error(other)),
        }
    })?;
    Ok(CounterfactualTransportDecisionNative {
        status: decided.status,
        payload: decided.payload,
        refusal: decided.refusal,
        prepared: decided.prepared,
        obstruction: decided.obstruction,
        graph: Some(named),
        catalog: Some(catalog),
        max_support_rows,
        memory_bytes,
    })
}

struct Decided {
    status: &'static str,
    payload: serde_json::Value,
    refusal: Option<(&'static str, String, String)>,
    prepared: Option<PreparedCounterfactualTransport>,
    obstruction: Option<CounterfactualTransportObstructionReport>,
}

/// Independently recheck an exact artifact under the consumer's limits:
/// re-decide, require the identical derivation, re-evaluate the point or
/// re-verify the obstruction's witness.
///
/// Returns `(kind, result, obstruction)` with exactly one of the last two set.
#[pyfunction]
#[pyo3(signature=(artifact, *, expected_premises_digest=None, expected_data_digest=None, max_search_operations=None, max_search_depth=None, max_operations=None, max_depth=None, max_support_rows=None, max_laws=None, max_law_cells=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
fn consume_counterfactual_transport_artifact_native(
    py: Python<'_>,
    artifact: &[u8],
    expected_premises_digest: Option<String>,
    expected_data_digest: Option<String>,
    max_search_operations: Option<usize>,
    max_search_depth: Option<usize>,
    max_operations: Option<usize>,
    max_depth: Option<usize>,
    max_support_rows: Option<usize>,
    max_laws: Option<usize>,
    max_law_cells: Option<usize>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(
    String,
    Option<CounterfactualTransportResultNative>,
    Option<CounterfactualTransportDecisionNative>,
)> {
    let defaults = CounterfactualTransportConsumeLimits::default();
    let limits = CounterfactualTransportConsumeLimits {
        search: SearchLimits {
            operations: max_search_operations.unwrap_or(defaults.search.operations),
            depth: max_search_depth.unwrap_or(defaults.search.depth),
        },
        evaluation: ExactEvaluationLimits {
            operations: max_operations.unwrap_or(defaults.evaluation.operations),
            depth: max_depth.unwrap_or(defaults.evaluation.depth),
        },
        max_support_rows: max_support_rows.unwrap_or(defaults.max_support_rows),
        max_laws: max_laws.unwrap_or(defaults.max_laws),
        max_law_cells: max_law_cells.unwrap_or(defaults.max_law_cells),
        ..defaults
    };
    let expected = CounterfactualTransportExpectedIdentity {
        premises_digest: expected_premises_digest,
        data_digest: expected_data_digest,
    };
    let bytes = artifact.to_vec();
    let consumption = crate::detach_catch(py, {
        let bytes = bytes.clone();
        move || {
            let ctx = execution_context(0, memory_bytes, cancel);
            consume_counterfactual_transport_artifact(&bytes, &expected, limits, &ctx)
                .map_err(error)
        }
    })?;
    // The consumer verified the bytes; the names are the mapping its identity binds.
    let names = CounterfactualTransportArtifactWire::decode(&bytes)
        .map(|wire| wire.variable_names)
        .map_err(|e| {
            let (code, message) = e.refusal();
            io_refusal(code, message)
        })?;
    match consumption {
        CounterfactualTransportConsumption::Identified(result) => Ok((
            "identified".to_string(),
            Some(CounterfactualTransportResultNative { inner: *result, names }),
            None,
        )),
        CounterfactualTransportConsumption::ProvenNonTransportable(report) => {
            let payload = obstruction_json(&report, &names);
            Ok((
                "proven_non_transportable".to_string(),
                None,
                Some(CounterfactualTransportDecisionNative {
                    status: "proven_non_transportable",
                    payload,
                    refusal: Some((
                        antecedent_core::reason_code!("transport_proven_non_transportable"),
                        "counterfactual_transport.checked_obstruction".to_string(),
                        report.reason().to_string(),
                    )),
                    prepared: None,
                    obstruction: Some(*report),
                    graph: None,
                    catalog: None,
                    max_support_rows: 0,
                    memory_bytes: None,
                }),
            ))
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CounterfactualTransportDecisionNative>()?;
    module.add_class::<PreparedCounterfactualTransportNative>()?;
    module.add_class::<CounterfactualTransportResultNative>()?;
    module.add_function(wrap_pyfunction!(identify_counterfactual_transport_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        consume_counterfactual_transport_artifact_native,
        module
    )?)?;
    Ok(())
}
