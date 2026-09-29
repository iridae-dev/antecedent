//! Python bindings for bounded multi-source limited-experiment (`TR^mz`) transport.
use crate::transport_common::{
    RegimeCheck, assignment_from_pairs, error, execution_context, frame_named_artifact, parse_laws,
    resolve, resolve_variable_ids, serialization_error, unframe_named_artifact,
};
use crate::transport_z_api::{intervention_assignments, to_py_json};
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{EvidenceCatalog, ExecutionContext, SearchLimits, Value, VariableId};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_identify::{
    MzTransportDecision, MzTransportQuery, MzTransportRoute, ZTransportSourceSpec,
    bind_mz_transport_catalog, decide_mz_transport,
};
use antecedent_io::mz_transport_artifact::{
    MzTransportArtifactWire, MzTransportConsumeLimits, MzUncertaintyWire,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Magic prefix of a portable mz-transport artifact: the io crate's versioned
/// CBOR wire, framed with the variable names it was built under.
const MZ_TRANSPORT_PREFIX: &[u8] = b"ANTECEDENT-MZ-TRANSPORT\x01";

/// Sound, incomplete bounded multi-source limited-experiment transport.
const SCOPE: &str = "multi_source_z_transport_cited_joints_sound_incomplete";

/// One Python source: `(population, controllable, experiment_assignment, selections)`.
type SourceTuple = (String, Vec<String>, BTreeMap<String, f64>, Vec<String>);

fn names_of(names: &[String], raw: &[u32]) -> Vec<String> {
    raw.iter().map(|v| names[*v as usize].clone()).collect()
}

fn regime_label(catalog: &EvidenceCatalog, id: antecedent_core::RegimeId) -> String {
    catalog
        .regimes
        .iter()
        .find(|r| r.id == id)
        .and_then(|r| r.label.as_deref())
        .map_or_else(|| id.raw().to_string(), ToString::to_string)
}

/// JSON for one decision, spelled in variable names and catalog regime ids.
fn decision_json(
    decision: &MzTransportDecision,
    catalog: &EvidenceCatalog,
    names: &[String],
    memory_bytes: Option<u64>,
) -> serde_json::Value {
    let stages = |stages: &[antecedent_identify::MzStageRecord]| {
        stages
            .iter()
            .map(|s| serde_json::json!({"stage": s.stage, "outcome": s.outcome}))
            .collect::<Vec<_>>()
    };
    match decision {
        MzTransportDecision::Identified { derivation, cited } => {
            let (route, sources) = match derivation.route() {
                MzTransportRoute::TargetOnly => ("target_only", Vec::new()),
                MzTransportRoute::SingleSource { population } => {
                    ("single_source", vec![population.to_string()])
                }
                MzTransportRoute::Combined { populations } => {
                    ("combined", populations.iter().map(ToString::to_string).collect())
                }
            };
            serde_json::json!({
                "outcome": "identified",
                "route": route,
                "sources": sources,
                "cited_regimes": cited.iter().map(|id| regime_label(catalog, *id)).collect::<Vec<_>>(),
                "rules": derivation.rules(),
                "stages": stages(derivation.stages()),
            })
        }
        MzTransportDecision::ProvenNonTransportable(obstruction) => serde_json::json!({
            "outcome": "proven_non_transportable",
            "reason": "transport_proven_non_transportable",
            "c0": names_of(names, obstruction.c0()),
            "sources": obstruction.sources().iter().map(|(population, active, separated)| serde_json::json!({
                "population": population,
                "active_controllable": names_of(names, active),
                "selection_separated": separated,
            })).collect::<Vec<_>>(),
        }),
        MzTransportDecision::MissingEvidence { derivation, detail } => serde_json::json!({
            "outcome": "missing_evidence",
            "reason": "transport_missing_evidence",
            "detail": resolve_variable_ids(detail, names),
            "formula_certified": derivation.is_some(),
        }),
        MzTransportDecision::NotCertified(inspection) => serde_json::json!({
            "outcome": "not_certified",
            "reason": "transport_not_certified",
            "stages": stages(&inspection.stages),
            "explored_rules": inspection.explored_rules,
            "steps_explored": inspection.steps_explored,
            "depth_reached": inspection.depth_reached,
        }),
        MzTransportDecision::Exhausted(receipt) => serde_json::json!({
            "outcome": "exhausted",
            "reason": "transport_budget_cancel",
            "limits_receipt": {
                "stop": receipt.stop.code(),
                "operations_limit": receipt.operations_limit,
                "depth_limit": receipt.depth_limit,
                "memory_limit_bytes": memory_bytes,
                "operations_consumed": receipt.operations_consumed,
                "depth_reached": receipt.depth_reached,
                "explored": receipt.explored,
                "unevaluated": receipt.unevaluated,
            },
        }),
    }
}

fn uncertainty_json(
    uncertainty: &MzUncertaintyWire,
    names: &[String],
    seed: u64,
) -> serde_json::Value {
    serde_json::json!({
        "available": !uncertainty.mean_intervals.is_empty(),
        "status": uncertainty.status,
        "reason": uncertainty.reason,
        "method": uncertainty.method,
        "coverage_target": uncertainty.coverage_target,
        "seed": seed,
        "replicates_requested": uncertainty.replicates_requested,
        "replicates_ok": uncertainty.replicates_ok,
        "replicates_failed": uncertainty.replicates_failed,
        "mean_intervals": uncertainty.mean_intervals.iter().map(|(v, lo, hi)| serde_json::json!({
            "outcome": names[*v as usize], "lower": lo, "upper": hi,
        })).collect::<Vec<_>>(),
    })
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

/// A bounded multi-source decision against one catalog of target and source evidence.
#[pyclass(skip_from_py_object)]
struct MultiSourceZTransportStage {
    decision: MzTransportDecision,
    graph: Admg,
    shared: antecedent_graph::Admg,
    catalog: EvidenceCatalog,
    search: SearchLimits,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
}

impl MultiSourceZTransportStage {
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        empirical: bool,
        max_operations: usize,
        max_depth: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedMultiSourceZTransportStage> {
        let MzTransportDecision::Identified { derivation, .. } = &self.decision else {
            return Err(crate::refusal(
                antecedent_core::reason_code!("transport_not_certified"),
                "mz_transport.not_identified: only an identified decision can be prepared",
            ));
        };
        let data = parse_laws(
            laws,
            &self.catalog,
            &self.graph,
            self.max_support_rows,
            RegimeCheck::Strict,
        )?;
        let request = assignment_from_pairs(&self.graph.names, assignments)?;
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
                bind_mz_transport_catalog(&shared, &derivation, &catalog).map_err(error)?;
            let limits = ExactEvaluationLimits { operations: max_operations, depth: max_depth };
            let build = if empirical {
                antecedent::StudyBuilder::mz_transport_empirical
            } else {
                antecedent::StudyBuilder::mz_transport
            };
            build(shared, functional, search, data, request, limits, &ctx).map_err(error)
        })?;
        Ok(PreparedMultiSourceZTransportStage {
            inner,
            graph: self.graph.clone(),
            catalog: self.catalog.clone(),
            last: None,
            max_support_rows: self.max_support_rows,
            memory_bytes: self.memory_bytes,
            seed,
        })
    }
}

#[pymethods]
impl MultiSourceZTransportStage {
    /// `identified`, `proven_non_transportable`, `missing_evidence`,
    /// `not_certified` or `exhausted`.
    #[getter]
    fn outcome(&self) -> &'static str {
        match &self.decision {
            MzTransportDecision::Identified { .. } => "identified",
            MzTransportDecision::ProvenNonTransportable(_) => "proven_non_transportable",
            MzTransportDecision::MissingEvidence { .. } => "missing_evidence",
            MzTransportDecision::NotCertified(_) => "not_certified",
            MzTransportDecision::Exhausted(_) => "exhausted",
        }
    }

    /// The decision in variable names and catalog regime ids.
    fn decision(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(
            py,
            &decision_json(&self.decision, &self.catalog, &self.graph.names, self.memory_bytes),
        )
    }

    /// Prepare exact-law evaluation of the identified formula.
    #[pyo3(signature=(laws, assignments, *, max_operations=10_000_000, max_depth=256, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_exact(
        &self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedMultiSourceZTransportStage> {
        self.prepare(py, laws, assignments, false, max_operations, max_depth, seed, cancel)
    }

    /// Prepare an empirical plug-in evaluation; every law must carry counts.
    #[pyo3(signature=(laws, assignments, *, max_operations=10_000_000, max_depth=256, seed=0, cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_empirical(
        &self,
        py: Python<'_>,
        laws: &Bound<'_, PyAny>,
        assignments: BTreeMap<String, f64>,
        max_operations: usize,
        max_depth: usize,
        seed: u64,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PreparedMultiSourceZTransportStage> {
        self.prepare(py, laws, assignments, true, max_operations, max_depth, seed, cancel)
    }
}

/// A prepared mz formula; estimation evaluates the frozen plan without searching.
#[pyclass(skip_from_py_object)]
struct PreparedMultiSourceZTransportStage {
    inner: antecedent::PreparedMzTransport,
    graph: Admg,
    catalog: EvidenceCatalog,
    last: Option<antecedent::MzTransportResult>,
    max_support_rows: usize,
    memory_bytes: Option<u64>,
    seed: u64,
}

#[pymethods]
impl PreparedMultiSourceZTransportStage {
    /// Evaluate the frozen formula. Exact laws are point-only; empirical tables
    /// attach a nominal joint bootstrap or the reason it is withheld.
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
        let names = &self.graph.names;
        let mut payload = point_json(result.distribution(), names);
        payload["status"] = "available".into();
        payload["scope"] = SCOPE.into();
        payload["seed"] = self.seed.into();
        payload["interval"] = uncertainty_json(result.uncertainty(), names, self.seed);
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
                "transport.no_execution_claim: estimate before exporting an mz-transport artifact",
            )
        })?;
        let raw = result.export(&self.inner).map_err(error)?;
        let framed = frame_named_artifact(MZ_TRANSPORT_PREFIX, &self.graph.names, raw)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }

    /// Sources the prepared formula cites, in canonical order.
    #[getter]
    fn cited_sources(&self) -> Vec<String> {
        let mut sources = self
            .inner
            .functional()
            .cited_populations()
            .into_iter()
            .map(|(_, p)| p.to_string())
            .collect::<Vec<_>>();
        sources.sort();
        sources.dedup();
        sources
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }
}

/// Decide a bounded multi-source limited-experiment query against a catalog.
///
/// `sources` are `(population, controllable, experiment_assignment, selections)`.
/// The decision is taken once here; every preparation made from the stage
/// reuses it and never searches again.
#[pyfunction]
#[pyo3(signature=(graph, target, outcomes, treatments, sources, catalog, *, max_operations=4096, max_depth=24, max_support_rows=1_000_000, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn identify_multi_source_z_transport_stage(
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
) -> PyResult<MultiSourceZTransportStage> {
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
    let query = MzTransportQuery {
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
        decide_mz_transport(&decision_graph, &query, &decision_catalog, search, &ctx).map_err(error)
    })?;
    Ok(MultiSourceZTransportStage {
        decision,
        graph: named,
        shared,
        catalog,
        search,
        max_support_rows,
        memory_bytes,
    })
}

/// Independently recheck a framed mz artifact under the consumer's limits:
/// re-decide, re-bind, recompute the point and recheck interval bookkeeping.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_search_operations=4096, max_search_depth=24, max_operations=10_000_000, max_depth=256, max_support_rows=None, max_laws=None, max_law_cells=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_multi_source_z_transport_artifact(
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
    let (names, bytes) = unframe_named_artifact(MZ_TRANSPORT_PREFIX, artifact, "mz-transport")?;
    let defaults = MzTransportConsumeLimits::default();
    let limits = MzTransportConsumeLimits {
        search: SearchLimits { operations: max_search_operations, depth: max_search_depth },
        evaluation: ExactEvaluationLimits { operations: max_operations, depth: max_depth },
        max_support_rows: max_support_rows.unwrap_or(defaults.max_support_rows),
        max_laws: max_laws.unwrap_or(defaults.max_laws),
        max_law_cells: max_law_cells.unwrap_or(defaults.max_law_cells),
    };
    crate::detach_catch(py, move || {
        let ctx: ExecutionContext = execution_context(0, memory_bytes, cancel);
        let consumed =
            MzTransportArtifactWire::consume_with_limits(&bytes, limits, &ctx).map_err(error)?;
        let graph = antecedent_io::admg_from_wire(&consumed.wire.graph).map_err(error)?;
        crate::transport_exact_api::validate_artifact_names(&names, &graph)?;
        let mut payload = point_json(&consumed.distribution, &names);
        payload["status"] = "available".into();
        payload["scope"] = SCOPE.into();
        payload["proof"] =
            serde_json::to_value(&consumed.wire.proof).map_err(serialization_error)?;
        payload["premises_digest"] = consumed.wire.premises_digest.clone().into();
        payload["cited_sources"] = consumed
            .wire
            .bindings
            .iter()
            .map(|(_, p)| p.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into();
        // Bookkeeping was rechecked against the licensing decision; the seeded
        // bootstrap itself is not re-run.
        payload["interval"] = uncertainty_json(&consumed.wire.uncertainty, &names, 0);
        Ok(payload.to_string())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MultiSourceZTransportStage>()?;
    module.add_class::<PreparedMultiSourceZTransportStage>()?;
    module.add_function(wrap_pyfunction!(identify_multi_source_z_transport_stage, module)?)?;
    module.add_function(wrap_pyfunction!(consume_multi_source_z_transport_artifact, module)?)?;
    Ok(())
}
