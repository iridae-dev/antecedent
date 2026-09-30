//! Python bindings for bounded mixed-source proof search (X9).
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact, parse_laws,
    resolve, serialization_error, unframe_named_artifact,
};
use crate::transport_z_api::{intervention_assignments, to_py_json};
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, RegimeId, SearchLimits, SearchReceipt, Value, VariableId,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_identify::{
    MixedQuantity, MixedSourceDecision, MixedSourceDerivation, MixedSourceQuery,
    ZTransportSourceSpec, bind_mixed_source_catalog, decide_mixed_source, render_quantity,
};
use antecedent_io::mixed_source_artifact::{MixedSourceArtifactWire, MixedSourceConsumeLimits};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Magic prefix of a portable mixed-source artifact: the io crate's versioned
/// CBOR wire, framed with the variable names it was built under.
const MIXED_SOURCE_PREFIX: &[u8] = b"ANTECEDENT-MIXED-SOURCE\x01";

/// Sound, incomplete bounded proof search over the target population's studies.
const SCOPE: &str = "mixed_source_proof_search_sound_incomplete";

/// One Python source: `(population, controllable, experiment_assignment, selections)`.
type SourceTuple = (String, Vec<String>, BTreeMap<String, f64>, Vec<String>);

fn regime_label(catalog: &EvidenceCatalog, id: RegimeId) -> String {
    catalog
        .regimes
        .iter()
        .find(|r| r.id == id)
        .and_then(|r| r.label.as_deref())
        .map_or_else(|| id.raw().to_string(), ToString::to_string)
}

fn names_of(names: &[String], variables: &[VariableId]) -> Vec<String> {
    variables
        .iter()
        .map(|v| names.get(v.as_usize()).cloned().unwrap_or_else(|| format!("v{}", v.raw())))
        .collect()
}

fn quantity_json(quantity: &MixedQuantity, names: &[String]) -> serde_json::Value {
    serde_json::json!({
        "text": render_quantity(quantity, names),
        "variables": names_of(names, &quantity.y),
        "intervened": names_of(names, &quantity.intervened),
        "conditioned": names_of(names, &quantity.conditioned),
    })
}

fn stages_json(stages: &[antecedent_identify::MixedStageRecord]) -> Vec<serde_json::Value> {
    stages.iter().map(|s| serde_json::json!({"stage": s.stage, "outcome": s.outcome})).collect()
}

fn receipt_json(receipt: &SearchReceipt, memory_bytes: Option<u64>) -> serde_json::Value {
    serde_json::json!({
        "stop": receipt.stop.code(),
        "operations_limit": receipt.operations_limit,
        "depth_limit": receipt.depth_limit,
        "memory_limit_bytes": memory_bytes,
        "operations_consumed": receipt.operations_consumed,
        "depth_reached": receipt.depth_reached,
        "explored": receipt.explored,
        "unevaluated": receipt.unevaluated,
    })
}

/// The proof of one derivation: every step with its premises and source
/// distribution, the compact proof graph, and the cited regimes.
fn derivation_json(
    derivation: &MixedSourceDerivation,
    catalog: &EvidenceCatalog,
    names: &[String],
) -> serde_json::Value {
    let steps = derivation
        .steps()
        .iter()
        .enumerate()
        .map(|(index, step)| {
            let quantity = MixedQuantity {
                y: step.y.clone(),
                intervened: step.intervened.clone(),
                conditioned: step.conditioned.clone(),
            };
            serde_json::json!({
                "step": index,
                "rule": step.rule.as_str(),
                "premises": step.premises,
                "moved": names_of(names, &step.params),
                "quantity": render_quantity(&quantity, names),
                "source": step.source.as_ref().map(|leaf| serde_json::json!({
                    "regime": regime_label(catalog, leaf.regime),
                    "study": leaf.study.as_ref(),
                    "population": leaf.population.as_ref(),
                    "snapshot": leaf.snapshot.as_deref(),
                    "identity": leaf.identity,
                })),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "rule_set": derivation.rule_set(),
        "steps": steps,
        "proof_graph": derivation.proof_graph(names),
        "cited_regimes": derivation
            .cited_regimes()
            .iter()
            .map(|id| regime_label(catalog, *id))
            .collect::<Vec<_>>(),
    })
}

/// JSON for one decision, spelled in variable names and catalog regime labels.
fn decision_json(
    decision: &MixedSourceDecision,
    catalog: &EvidenceCatalog,
    names: &[String],
    memory_bytes: Option<u64>,
) -> serde_json::Value {
    match decision {
        MixedSourceDecision::Identified { derivation, alternatives, .. } => {
            let mut payload = derivation_json(derivation, catalog, names);
            payload["outcome"] = "identified".into();
            payload["stages"] = stages_json(derivation.stages()).into();
            payload["search"] = serde_json::json!({
                "operations_to_proof": derivation.summary().operations_to_proof,
                "generation": derivation.summary().generation,
            });
            payload["alternatives"] = alternatives
                .iter()
                .map(|alternative| derivation_json(alternative, catalog, names))
                .collect::<Vec<_>>()
                .into();
            payload
        }
        MixedSourceDecision::NamedRoute { route, stages } => serde_json::json!({
            "outcome": "named_route",
            "reason": "route_not_supported",
            "route": route,
            "stages": stages_json(stages),
        }),
        MixedSourceDecision::MissingEvidence(missing) => serde_json::json!({
            "outcome": "missing_evidence",
            "reason": "transport_missing_evidence",
            "missing_leaves": missing.leaves.iter().map(|leaf| serde_json::json!({
                "step": leaf.step,
                "regime": regime_label(catalog, leaf.regime),
                "study": leaf.study.as_ref(),
                "variables": names_of(names, &leaf.variables),
                "intervened": names_of(names, &leaf.intervened),
                "supplied_as_separate_marginals": names_of(names, &leaf.marginals),
            })).collect::<Vec<_>>(),
            "proof_graph": missing.proof_graph(names),
            "stages": stages_json(&missing.stages),
        }),
        MixedSourceDecision::NotCertified(inspection) => serde_json::json!({
            "outcome": "not_certified",
            "reason": "transport_not_certified",
            "goal": quantity_json(&inspection.goal, names),
            "frontier": inspection.frontier.iter().map(|q| quantity_json(q, names)).collect::<Vec<_>>(),
            "quantities": inspection.quantities,
            "generations": inspection.generations,
            "operations": inspection.operations,
            "rule_counts": inspection.rule_counts.iter().map(|(rule, count)| ((*rule).to_string(), *count)).collect::<BTreeMap<_, _>>(),
            "excluded": inspection.exclusions.iter().map(|e| serde_json::json!({
                "regime": regime_label(catalog, e.regime),
                "reason": e.reason,
            })).collect::<Vec<_>>(),
            "stages": stages_json(&inspection.stages),
        }),
        MixedSourceDecision::Exhausted(receipt) => serde_json::json!({
            "outcome": "exhausted",
            "reason": "transport_budget_cancel",
            "limits_receipt": receipt_json(receipt, memory_bytes),
        }),
    }
}

fn point_json(
    distribution: &antecedent_expr::ExactDistribution,
    names: &[String],
) -> serde_json::Value {
    serde_json::json!({
        "outcomes": distribution.outcomes.iter().map(|v| &names[v.as_usize()]).collect::<Vec<_>>(),
        "atoms": distribution.atoms.iter().map(|row| row.iter().map(Value::as_f64).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "probabilities": distribution.probabilities.as_ref(),
    })
}

/// Request 0's point at the top level and every request's point and outcome means
/// under `requests`.
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
    // The route publishes points only: exact laws carry no sampling uncertainty.
    payload["interval"] = serde_json::json!({"available": false, "status": "point_only"});
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

/// The typed refusal of preparing a decision that did not identify: the
/// decision's own frozen (reason code, `mixed_search.*` detail) pair.
fn not_identified(decision: &MixedSourceDecision) -> PyErr {
    let code =
        decision.reason_code().unwrap_or(antecedent_core::reason_code!("transport_not_certified"));
    let detail = decision.detail_code().unwrap_or_default();
    let route = match decision {
        MixedSourceDecision::NamedRoute { route, .. } => format!(" (use the {route} route)"),
        _ => String::new(),
    };
    crate::refusal(
        code,
        format!("{detail}: only an identified decision can be prepared{route}; see decision()"),
    )
}

/// A decision error: a declared bound the query exceeds, or a model artifact
/// offered as an experimental law, is `route_not_supported` naming the detail;
/// anything else keeps its identification mapping.
fn decision_error(e: antecedent_identify::IdentificationError) -> PyErr {
    match e {
        antecedent_identify::IdentificationError::UnsupportedInput { code } => {
            crate::refusal(antecedent_core::reason_code!("route_not_supported"), code)
        }
        other => error(other),
    }
}

/// A bounded mixed-source decision against one catalog of the target's studies.
#[pyclass(skip_from_py_object)]
struct MixedSourceStage {
    decision: MixedSourceDecision,
    graph: Admg,
    shared: antecedent_graph::Admg,
    catalog: EvidenceCatalog,
    search: SearchLimits,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl MixedSourceStage {
    /// `identified`, `named_route`, `missing_evidence`, `not_certified` or `exhausted`.
    #[getter]
    #[doc(hidden)]
    fn outcome(&self) -> &'static str {
        match &self.decision {
            MixedSourceDecision::Identified { .. } => "identified",
            MixedSourceDecision::NamedRoute { .. } => "named_route",
            MixedSourceDecision::MissingEvidence(_) => "missing_evidence",
            MixedSourceDecision::NotCertified(_) => "not_certified",
            MixedSourceDecision::Exhausted(_) => "exhausted",
        }
    }

    /// The decision in variable names and catalog regime labels.
    #[doc(hidden)]
    fn decision(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(
            py,
            &decision_json(&self.decision, &self.catalog, &self.graph.names, self.memory_bytes),
        )
    }

    /// Prepare exact-law evaluation of the identified formula for one request
    /// mapping, or a sequence of them.
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
    ) -> PyResult<PreparedMixedSourceStage> {
        let MixedSourceDecision::Identified { derivation, .. } = &self.decision else {
            return Err(not_identified(&self.decision));
        };
        let data = parse_laws(
            laws,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let requests = parse_requests(&self.graph.names, assignments)?;
        let (shared, catalog, derivation, search, memory_bytes) = (
            self.shared.clone(),
            self.catalog.clone(),
            derivation.as_ref().clone(),
            self.search,
            self.memory_bytes,
        );
        let inner = crate::detach_catch(py, move || {
            let ctx = execution_context(seed, memory_bytes, cancel);
            let functional =
                bind_mixed_source_catalog(&shared, &derivation, &catalog).map_err(error)?;
            let limits = ExactEvaluationLimits { operations: max_operations, depth: max_depth };
            antecedent::StudyBuilder::mixed_source(
                shared, functional, search, data, requests, limits, &ctx,
            )
            .map_err(error)
        })?;
        Ok(PreparedMixedSourceStage {
            inner,
            graph: self.graph.clone(),
            catalog: self.catalog.clone(),
            last: None,
            max_support_rows: self.max_support_rows,
            memory_bytes: self.memory_bytes,
            seed,
        })
    }

    /// Counted laws are not licensed on this route: no sampling theory backs an
    /// empirical plug-in of a searched formula, so preparation refuses.
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
    ) -> PyResult<PreparedMixedSourceStage> {
        Err(crate::refusal(
            antecedent_core::reason_code!("cell_not_licensed"),
            "counted laws are not licensed on the mixed-source route: it publishes exact-law points only",
        ))
    }
}

/// A prepared mixed-source formula; estimation evaluates the frozen plan without searching.
#[pyclass(skip_from_py_object)]
struct PreparedMixedSourceStage {
    inner: antecedent::PreparedMixedSource,
    graph: Admg,
    catalog: EvidenceCatalog,
    last: Option<antecedent::MixedSourceResult>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    seed: u64,
}

#[pymethods]
impl PreparedMixedSourceStage {
    /// Evaluate the frozen formula for every prepared request. Exact laws are
    /// point-only: `interval["available"]` is always false.
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
        let result = crate::detach_catch(py, move || inner.estimate(&ctx).map_err(error))?;
        let mut payload =
            points_json(self.inner.requests(), result.distributions(), &self.graph.names);
        payload["status"] = "available".into();
        payload["scope"] = SCOPE.into();
        payload["seed"] = self.seed.into();
        self.last = Some(result);
        Ok(payload.to_string())
    }

    /// Replace laws for the snapshots the frozen catalog binds; the proof is
    /// unchanged and the last claim is cleared. Other evidence identity refuses.
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
        self.inner = crate::detach_catch(py, move || inner.refresh(data, &ctx).map_err(error))?;
        self.last = None;
        Ok(())
    }

    /// The last execution as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: estimate before exporting a mixed-source artifact",
            )
        })?;
        let raw = result.export_named(&self.inner, &self.graph.names).map_err(error)?;
        let framed = frame_named_artifact(MIXED_SOURCE_PREFIX, &self.graph.names, raw)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }

    /// The retained, compiled plan: the frozen proof, its source-named leaves and
    /// one compiled evaluation plan per request.
    #[doc(hidden)]
    fn plan(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = &self.graph.names;
        let functional = self.inner.functional();
        let requests = self
            .inner
            .requests()
            .iter()
            .map(|request| {
                request
                    .entries()
                    .iter()
                    .map(|(v, x)| (names[v.as_usize()].clone(), serde_json::json!(x.as_f64())))
                    .collect::<serde_json::Map<_, _>>()
            })
            .collect::<Vec<_>>();
        let mut payload = derivation_json(functional.derivation(), &self.catalog, names);
        payload["requests"] = requests.into();
        payload["compiled_plans"] = self.inner.plans().len().into();
        to_py_json(py, &payload)
    }

    /// Studies the prepared formula cites, in canonical order.
    #[getter]
    #[doc(hidden)]
    fn cited_studies(&self) -> Vec<String> {
        let mut studies = self
            .inner
            .functional()
            .cited_sources()
            .into_iter()
            .map(|(_, study, _)| study.to_string())
            .collect::<Vec<_>>();
        studies.sort();
        studies.dedup();
        studies
    }

    #[getter]
    #[doc(hidden)]
    fn seed(&self) -> u64 {
        self.seed
    }
}

/// Decide a bounded mixed-source query against a catalog.
///
/// `sources` are `(population, controllable, experiment_assignment, selections)`
/// of the optional theorem-scoped z / mz routes that run first. The decision is
/// taken once here; every preparation made from the stage reuses it and never
/// searches again.
#[pyfunction]
#[doc(hidden)]
#[pyo3(signature=(graph, target, outcomes, treatments, sources, catalog, *, max_operations=20_000, max_depth=16, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn identify_mixed_source_transport_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    sources: Vec<SourceTuple>,
    catalog: &Bound<'_, PyAny>,
    max_operations: usize,
    max_depth: usize,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<MixedSourceStage> {
    let coordinates = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(&graph.names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let mut specs = Vec::with_capacity(sources.len());
    for (population, controllable, assignment, selections) in sources {
        specs.push(ZTransportSourceSpec {
            population: population.into(),
            controllable: coordinates(&controllable)?,
            experiment_assignment: intervention_assignments(&graph.names, assignment)?,
            selection_targets: coordinates(&selections)?,
        });
    }
    let query = MixedSourceQuery {
        outcomes: coordinates(&outcomes)?,
        treatments: coordinates(&treatments)?,
        target: target.into(),
        sources: specs.into(),
    };
    let named = Admg { admg: graph.admg.clone(), names: graph.names.clone() };
    let catalog = parse_catalog(catalog, &named)?;
    let shared = graph.aligned_to_names(&graph.names)?;
    let search = SearchLimits { operations: max_operations, depth: max_depth };
    let (decision_graph, decision_catalog) = (shared.clone(), catalog.clone());
    let decision = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        decide_mixed_source(&decision_graph, &query, &decision_catalog, search, &ctx)
            .map_err(decision_error)
    })?;
    Ok(MixedSourceStage {
        decision,
        graph: named,
        shared,
        catalog,
        search,
        max_support_rows,
        memory_bytes,
    })
}

/// Independently recheck a framed mixed-source artifact under the consumer's
/// limits: re-decide, re-check every proof step, re-bind every source-named leaf
/// and recompute the point.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_search_operations=20_000, max_search_depth=16, max_operations=10_000_000, max_depth=256, max_support_rows=None, max_laws=None, max_law_cells=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_mixed_source_artifact(
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
    let (names, bytes) = unframe_named_artifact(MIXED_SOURCE_PREFIX, artifact, "mixed-source")?;
    let defaults = MixedSourceConsumeLimits::default();
    let limits = MixedSourceConsumeLimits {
        search: SearchLimits { operations: max_search_operations, depth: max_search_depth },
        evaluation: ExactEvaluationLimits { operations: max_operations, depth: max_depth },
        max_support_rows: max_support_rows.unwrap_or(defaults.max_support_rows),
        max_laws: max_laws.unwrap_or(defaults.max_laws),
        max_law_cells: max_law_cells.unwrap_or(defaults.max_law_cells),
        ..defaults
    };
    crate::detach_catch(py, move || {
        let ctx: ExecutionContext = execution_context(0, memory_bytes, cancel);
        let consumed =
            MixedSourceArtifactWire::consume_with_limits(&bytes, limits, &ctx).map_err(error)?;
        let graph = antecedent_io::admg_from_wire(&consumed.wire.graph).map_err(error)?;
        crate::transport_exact_api::validate_artifact_names(&names, &graph)?;
        // The frame's names must be the mapping the verified identity binds.
        consumed.wire.check_variable_names(&names).map_err(error)?;
        let mut payload = points_json(&consumed.requests, &consumed.distributions, &names);
        payload["status"] = "available".into();
        payload["scope"] = SCOPE.into();
        payload["proof"] =
            serde_json::to_value(&consumed.wire.proof).map_err(serialization_error)?;
        payload["premises_digest"] = consumed.wire.premises_digest.clone().into();
        payload["data_digest"] = consumed.wire.data_digest.clone().into();
        payload["cited_sources"] = consumed
            .wire
            .bindings
            .iter()
            .map(|(regime, study, population)| {
                serde_json::json!({"regime": regime, "study": study, "population": population})
            })
            .collect::<Vec<_>>()
            .into();
        Ok(payload.to_string())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MixedSourceStage>()?;
    module.add_class::<PreparedMixedSourceStage>()?;
    module.add_function(wrap_pyfunction!(identify_mixed_source_transport_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_mixed_source_artifact, module)?)?;
    Ok(())
}
