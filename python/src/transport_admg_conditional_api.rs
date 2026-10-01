//! Python bindings for ADMG conditional transport (2.2B B1, X2).
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact, parse_laws,
    resolve, serialization_error, unframe_named_artifact,
};
use crate::transport_z_api::to_py_json;
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{EvidenceCatalog, SearchLimits, SearchReceipt, Value, VariableId};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ClassicalTransportQuery, ConditionalTransportDecision, ConditionalTransportQuery,
    IdentificationError, admg_conditional_refusal, decide_admg_conditional_transport,
};
use antecedent_io::IoError;
use antecedent_io::admg_conditional_transport_artifact::{
    AdmgConditionalArtifactWire, AdmgConditionalConsumeLimits,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Magic prefix of a portable conditional artifact: the io crate's versioned
/// CBOR wire, framed with the variable names it was built under.
const PREFIX: &[u8] = b"ANTECEDENT-ADMG-CONDITIONAL\x01";

/// Sound, incomplete conditional transport over the classical complete-source family.
const SCOPE: &str = "admg_conditional_transport_sound_incomplete";

fn names_of(names: &[String], variables: &[VariableId]) -> Vec<String> {
    variables
        .iter()
        .map(|v| names.get(v.as_usize()).cloned().unwrap_or_else(|| format!("v{}", v.raw())))
        .collect()
}

fn receipt_json(receipt: &SearchReceipt) -> serde_json::Value {
    serde_json::json!({
        "stop": receipt.stop.code(),
        "operations_limit": receipt.operations_limit,
        "depth_limit": receipt.depth_limit,
        "memory_limit_bytes": receipt.memory_limit_bytes,
        "operations_consumed": receipt.operations_consumed,
        "depth_reached": receipt.depth_reached,
        "explored": receipt.explored,
        "unevaluated": receipt.unevaluated,
    })
}

/// The decision in variable names.
fn decision_json(decision: &ConditionalTransportDecision, names: &[String]) -> serde_json::Value {
    let reduced = |q: &ClassicalTransportQuery| {
        serde_json::json!({
            "outcomes": names_of(names, &q.outcomes),
            "treatments": names_of(names, &q.treatments),
        })
    };
    let mut payload = match decision {
        ConditionalTransportDecision::Identified(bound) => {
            let derivation = bound.derivation();
            serde_json::json!({
                "outcome": "identified",
                "moved": names_of(names, derivation.moves()),
                "remaining": names_of(names, derivation.remaining()),
                "reduced_query": reduced(derivation.reduced_query()),
                "rules": derivation.joint().rules(),
                "cited_leaves": bound
                    .cited_leaves()
                    .into_iter()
                    .map(|(population, regime)| serde_json::json!({"population": population.as_ref(), "regime": regime}))
                    .collect::<Vec<_>>(),
            })
        }
        ConditionalTransportDecision::MissingEvidence { derivation, obligations } => {
            serde_json::json!({
                "outcome": "missing_evidence",
                "moved": names_of(names, derivation.moves()),
                "remaining": names_of(names, derivation.remaining()),
                "reduced_query": reduced(derivation.reduced_query()),
                "obligations": obligations
                    .iter()
                    .map(|o| crate::transport_common::resolve_variable_ids(o, names))
                    .collect::<Vec<_>>(),
            })
        }
        ConditionalTransportDecision::NotCertified(inspection) => serde_json::json!({
            "outcome": "not_certified",
            "moved": names_of(names, &inspection.moves),
            "remaining": names_of(names, &inspection.remaining),
            "stages": inspection
                .stages
                .iter()
                .map(|s| serde_json::json!({"stage": s.stage, "outcome": s.outcome}))
                .collect::<Vec<_>>(),
            // Inspection only: the reduced joint's verified s-hedge. Lifting it to
            // the conditional query is paper-inherited, so it is no impossibility claim.
            "candidate": inspection.candidate.as_ref().map(|candidate| {
                let record = candidate.s_hedge().to_record();
                let forest = |forest: &antecedent_identify::sid::SelectionForestRecord| {
                    serde_json::json!({
                        "nodes": names_of(names, &forest.nodes.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>()),
                        "directed": forest.directed.iter().map(|(a, b)| [names_of(names, &[VariableId::from_raw(*a)]), names_of(names, &[VariableId::from_raw(*b)])].concat()).collect::<Vec<_>>(),
                        "bidirected": forest.bidirected.iter().map(|(a, b)| [names_of(names, &[VariableId::from_raw(*a)]), names_of(names, &[VariableId::from_raw(*b)])].concat()).collect::<Vec<_>>(),
                    })
                };
                serde_json::json!({
                    "reduced_query": reduced(candidate.reduced_query()),
                    "larger": forest(&record.larger),
                    "smaller": forest(&record.smaller),
                    "proof": false,
                })
            }),
        }),
        ConditionalTransportDecision::Exhausted(receipt) => serde_json::json!({
            "outcome": "exhausted",
            "receipt": receipt_json(receipt),
        }),
    };
    payload["reason_code"] = decision.reason_code().into();
    payload["detail"] = decision.detail_code().into();
    payload
}

fn point_json(
    distribution: &antecedent_expr::ExactDistribution,
    names: &[String],
) -> serde_json::Value {
    serde_json::json!({
        "outcomes": names_of(names, &distribution.outcomes),
        "atoms": distribution.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "probabilities": distribution.probabilities.as_ref(),
    })
}

/// Every request's conditional point and outcome means; request 0 at the top level.
fn points_json(
    requests: &[Assignment],
    distributions: &[antecedent_expr::ExactDistribution],
    names: &[String],
) -> serde_json::Value {
    let mut payload = point_json(&distributions[0], names);
    payload["requests"] = requests
        .iter()
        .zip(distributions)
        .map(|(request, distribution)| {
            let mut entry = point_json(distribution, names);
            entry["assignment"] = request
                .entries()
                .iter()
                .map(|(v, x)| (names[v.as_usize()].clone(), serde_json::json!(x.as_f64())))
                .collect::<serde_json::Map<_, _>>()
                .into();
            entry["means"] = distribution
                .outcomes
                .iter()
                .map(|v| {
                    (names[v.as_usize()].clone(), serde_json::json!(distribution.mean(*v).ok()))
                })
                .collect::<serde_json::Map<_, _>>()
                .into();
            entry
        })
        .collect::<Vec<_>>()
        .into();
    payload["interval"] = serde_json::json!({"available": false, "status": "point_only"});
    payload["status"] = "available".into();
    payload["scope"] = SCOPE.into();
    payload
}

/// One request mapping, or a sequence of them.
fn parse_requests(names: &[String], assignments: &Bound<'_, PyAny>) -> PyResult<Vec<Assignment>> {
    if let Ok(single) = assignments.extract::<BTreeMap<String, f64>>() {
        return Ok(vec![assignment_from_pairs(names, single)?]);
    }
    let many = assignments.extract::<Vec<BTreeMap<String, f64>>>().map_err(|_| {
        crate::value_err(
            "assignments must be a mapping of variable names to levels, or a sequence of them",
        )
    })?;
    if many.is_empty() {
        return Err(crate::value_err("assignments must name at least one request"));
    }
    many.into_iter().map(|pairs| assignment_from_pairs(names, pairs)).collect()
}

/// An identification error of this route as its `(reason code, detail)` refusal.
fn identification_error(e: IdentificationError) -> PyErr {
    match admg_conditional_refusal(&e) {
        Some((code, detail)) => {
            let text = e.to_string();
            crate::refusal(
                code,
                if text.starts_with(detail) { text } else { format!("{detail}: {text}") },
            )
        }
        None => error(e),
    }
}

/// An io error: a typed artifact refusal keeps its `(reason code, detail)` pair.
fn io_error(e: IoError) -> PyErr {
    match e {
        IoError::AdmgConditional(inner) => {
            let (code, detail) = inner.refusal();
            let text = inner.to_string();
            crate::refusal(
                code,
                if text.starts_with(detail) { text } else { format!("{detail}: {text}") },
            )
        }
        other => error(other),
    }
}

/// A bounded conditional transport decision against one catalog.
#[pyclass(skip_from_py_object)]
struct AdmgConditionalTransportStage {
    decision: ConditionalTransportDecision,
    graph: Admg,
    diagram: SelectionDiagram,
    catalog: EvidenceCatalog,
    search: SearchLimits,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl AdmgConditionalTransportStage {
    /// `identified`, `missing_evidence`, `not_certified` or `exhausted`.
    #[getter]
    #[doc(hidden)]
    fn outcome(&self) -> &'static str {
        match &self.decision {
            ConditionalTransportDecision::Identified(_) => "identified",
            ConditionalTransportDecision::MissingEvidence { .. } => "missing_evidence",
            ConditionalTransportDecision::NotCertified(_) => "not_certified",
            ConditionalTransportDecision::Exhausted(_) => "exhausted",
        }
    }

    /// The decision in variable names.
    #[doc(hidden)]
    fn decision(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(py, &decision_json(&self.decision, &self.graph.names))
    }

    /// Prepare exact-law evaluation for one request mapping (binding the
    /// treatments and conditioned variables), or a sequence of them.
    #[pyo3(signature=(laws, assignments, *, max_operations=10_000_000, max_depth=256, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_exact(
        &self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        assignments: &Bound<'_, PyAny>,
        max_operations: usize,
        max_depth: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedAdmgConditionalTransportStage> {
        let ConditionalTransportDecision::Identified(bound) = &self.decision else {
            let code = self
                .decision
                .reason_code()
                .unwrap_or(antecedent_core::reason_code!("transport_not_certified"));
            let detail = self.decision.detail_code().unwrap_or_default();
            return Err(crate::refusal(
                code,
                format!("{detail}: only an identified decision can be prepared; see decision()"),
            ));
        };
        let data = parse_laws(
            laws,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let requests = parse_requests(&self.graph.names, assignments)?;
        let (diagram, functional, search, memory_bytes) =
            (self.diagram.clone(), bound.as_ref().clone(), self.search, self.memory_bytes);
        let inner = crate::detach_catch(py, move || {
            let ctx = execution_context(seed, memory_bytes, cancel);
            let limits = ExactEvaluationLimits { operations: max_operations, depth: max_depth };
            antecedent::StudyBuilder::admg_conditional_transport(
                diagram, functional, search, data, requests, limits, &ctx,
            )
            .map_err(io_error)
        })?;
        Ok(PreparedAdmgConditionalTransportStage {
            inner,
            graph: self.graph.clone(),
            catalog: self.catalog.clone(),
            last: None,
            max_support_rows: self.max_support_rows,
            memory_bytes: self.memory_bytes,
            seed,
        })
    }

    /// Counted laws are not licensed on this route (no coverage record exists),
    /// so preparation refuses: exact-law points only.
    #[pyo3(signature=(laws, assignments, *, max_operations=10_000_000, max_depth=256, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments, clippy::unused_self, unused_variables)]
    fn prepare_empirical(
        &self,
        laws: &Bound<'_, PyAny>,
        assignments: &Bound<'_, PyAny>,
        max_operations: usize,
        max_depth: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedAdmgConditionalTransportStage> {
        Err(crate::refusal(
            antecedent_core::reason_code!("cell_not_licensed"),
            "admg_transport.interval_withheld: counted laws are not licensed on the conditional route; it publishes exact-law points only",
        ))
    }
}

/// A prepared conditional formula; estimation evaluates the frozen plans.
#[pyclass(skip_from_py_object)]
struct PreparedAdmgConditionalTransportStage {
    inner: antecedent::PreparedAdmgConditionalTransport,
    graph: Admg,
    catalog: EvidenceCatalog,
    last: Option<antecedent::AdmgConditionalResult>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    seed: u64,
}

#[pymethods]
impl PreparedAdmgConditionalTransportStage {
    /// Evaluate every prepared request. Exact laws are point-only:
    /// `interval["available"]` is always false.
    #[pyo3(signature=(*, memory_bytes=None, cancel=None, seed=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        if let Some(seed) = seed {
            self.seed = seed;
        }
        let ctx = execution_context(self.seed, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        let result = crate::detach_catch(py, move || inner.estimate(&ctx).map_err(io_error))?;
        let mut payload =
            points_json(self.inner.requests(), result.distributions(), &self.graph.names);
        payload["seed"] = self.seed.into();
        self.last = Some(result);
        Ok(payload.to_string())
    }

    /// Replace laws for the snapshots the frozen catalog binds; the proof is
    /// unchanged and the last result is cleared.
    #[pyo3(signature=(laws, *, memory_bytes=None, cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let data = parse_laws(
            laws,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let ctx = execution_context(self.seed, memory_bytes.or(self.memory_bytes), cancel);
        let inner = self.inner.clone();
        self.inner = crate::detach_catch(py, move || inner.refresh(data, &ctx).map_err(io_error))?;
        self.last = None;
        Ok(())
    }

    /// The last execution as a framed artifact a consumer re-derives and replays.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: estimate before exporting a conditional transport artifact",
            )
        })?;
        let raw = result.export_named(&self.inner, &self.graph.names).map_err(io_error)?;
        let framed = frame_named_artifact(PREFIX, &self.graph.names, raw)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }

    /// The retained plan: moves, remaining set, reduced query and request count.
    #[doc(hidden)]
    fn plan(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = &self.graph.names;
        let derivation = self.inner.functional().derivation();
        to_py_json(
            py,
            &serde_json::json!({
                "moved": names_of(names, derivation.moves()),
                "remaining": names_of(names, derivation.remaining()),
                "reduced_outcomes": names_of(names, &derivation.reduced_query().outcomes),
                "reduced_treatments": names_of(names, &derivation.reduced_query().treatments),
                "compiled_plans": self.inner.plans().len(),
            }),
        )
    }

    #[getter]
    #[doc(hidden)]
    fn seed(&self) -> u64 {
        self.seed
    }
}

/// Decide a bounded ADMG conditional transport query `P*(outcomes | do(treatments),
/// conditioned_on)` against a catalog, once; every preparation reuses it.
#[pyfunction]
#[doc(hidden)]
#[pyo3(signature=(graph, source, target, selections, outcomes, treatments, conditioned_on, catalog, *, max_operations=4096, max_depth=24, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn identify_admg_conditional_transport_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    source: String,
    target: String,
    selections: Vec<String>,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    conditioned_on: Vec<String>,
    catalog: &Bound<'_, PyAny>,
    max_operations: usize,
    max_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<AdmgConditionalTransportStage> {
    // A name the graph does not carry is this route's invalid query, with its
    // recorded (reason code, detail) pair, like every other invalid coordinate.
    let coordinates = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| {
                resolve(&graph.names, name).map_err(|_| {
                    crate::refusal(
                        antecedent_core::reason_code!("invalid_argument"),
                        format!(
                            "admg_transport.invalid_query: {name:?} is not a variable of the graph"
                        ),
                    )
                })
            })
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let query = ConditionalTransportQuery {
        base: ClassicalTransportQuery {
            outcomes: coordinates(&outcomes)?,
            treatments: coordinates(&treatments)?,
            source: source.into(),
            target: target.into(),
        },
        conditioned_on: coordinates(&conditioned_on)?,
    };
    let named = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    let catalog = parse_catalog(catalog, &named)?;
    let diagram = SelectionDiagram::try_new(
        graph.aligned_to_names(&graph.names)?,
        coordinates(&selections)?.to_vec(),
    )
    .map_err(|e| crate::value_err(e.to_string()))?;
    let search = SearchLimits { operations: max_operations, depth: max_depth };
    let (decision_diagram, decision_catalog) = (diagram.clone(), catalog.clone());
    let decision = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        decide_admg_conditional_transport(
            &decision_diagram,
            &query,
            &decision_catalog,
            search,
            &ctx,
        )
        .map_err(identification_error)
    })?;
    Ok(AdmgConditionalTransportStage {
        decision,
        graph: named,
        diagram,
        catalog,
        search,
        max_support_rows,
        memory_bytes,
    })
}

/// Re-check a framed conditional artifact under the consumer's limits and
/// recompute every point (same search and evaluator as the producer).
#[pyfunction]
#[pyo3(signature=(artifact, *, max_search_operations=4096, max_search_depth=24, max_operations=10_000_000, max_depth=256, max_support_rows=None, max_laws=None, max_law_cells=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_admg_conditional_transport_artifact(
    py: Python<'_>,
    artifact: &[u8],
    max_search_operations: usize,
    max_search_depth: usize,
    max_operations: usize,
    max_depth: usize,
    max_support_rows: Option<usize>,
    max_laws: Option<usize>,
    max_law_cells: Option<usize>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(PREFIX, artifact, "admg conditional transport")?;
    let defaults = AdmgConditionalConsumeLimits::default();
    let limits = AdmgConditionalConsumeLimits {
        search: SearchLimits { operations: max_search_operations, depth: max_search_depth },
        evaluation: ExactEvaluationLimits { operations: max_operations, depth: max_depth },
        max_support_rows: max_support_rows.unwrap_or(defaults.max_support_rows),
        max_laws: max_laws.unwrap_or(defaults.max_laws),
        max_law_cells: max_law_cells.unwrap_or(defaults.max_law_cells),
        ..defaults
    };
    crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        let consumed = AdmgConditionalArtifactWire::consume_with_limits(&bytes, limits, &ctx)
            .map_err(io_error)?;
        crate::transport_exact_api::validate_artifact_names(
            &names,
            consumed.diagram.causal_graph(),
        )?;
        consumed
            .wire
            .check_variable_names(&names)
            .map_err(|e| io_error(IoError::AdmgConditional(e)))?;
        let mut payload = points_json(&consumed.requests, &consumed.distributions, &names);
        payload["proof"] =
            serde_json::to_value(&consumed.wire.proof).map_err(serialization_error)?;
        payload["premises_digest"] = consumed.wire.premises_digest.clone().into();
        payload["data_digest"] = consumed.wire.data_digest.clone().into();
        Ok(payload.to_string())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AdmgConditionalTransportStage>()?;
    module.add_class::<PreparedAdmgConditionalTransportStage>()?;
    module.add_function(wrap_pyfunction!(identify_admg_conditional_transport_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_admg_conditional_transport_artifact, module)?)?;
    Ok(())
}
