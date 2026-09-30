//! Python bindings for X6 study planning over the X1 (mz) and X9 (mixed-source)
//! catalogs.
use crate::transport_common::{
    execution_context, frame_named_artifact, resolve, serialization_error, unframe_named_artifact,
};
use crate::transport_z_api::{intervention_assignments, to_py_json};
use crate::{graphs::Admg, transport_interference_api::parse_catalog};
use antecedent_core::{EvidenceCatalog, RegimeId, SearchLimits, VariableId};
use antecedent_design::{
    StudyArrivalDecision, StudyCandidate, StudyCost, StudyPlanArtifactWire, StudyPlanConsumeLimits,
    StudyPlanError, StudyPlanResult, plan_studies,
};
use antecedent_identify::{
    MixedSourceDecision, MixedSourceQuery, MzTransportDecision, MzTransportQuery,
    STUDY_PLAN_MEMORY_BYTES, StudyPlan, StudyPlanLimits, StudyPlanRoute, StudyPlanStop,
    StudySubsetOutcome, ZTransportSourceSpec,
};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Magic prefix of a portable study plan: the design crate's CBOR wire, framed
/// with the variable names it was planned under.
const STUDY_PLAN_PREFIX: &[u8] = b"ANTECEDENT-STUDY-PLAN\x01";

/// One Python source: `(population, controllable, experiment_assignment, selections)`.
type SourceTuple = (String, Vec<String>, BTreeMap<String, f64>, Vec<String>);

/// One Python candidate: `(id, population, interventions, levels, measured,
/// recruitment, cost_units, sample_budget, requires, conflicts, feasibility)`.
type CandidateTuple = (
    String,
    String,
    Vec<String>,
    Option<Vec<BTreeMap<String, f64>>>,
    Vec<String>,
    String,
    u64,
    u64,
    Vec<String>,
    Vec<String>,
    Vec<String>,
);

/// The Python exception of a refused plan, proposal or arrival: its registered
/// reason code and `study_plan.*` detail.
fn plan_error(error: StudyPlanError) -> PyErr {
    crate::refusal(error.code, format!("{}: {}", error.detail, error.message))
}

fn names_of(names: &[String], variables: &[VariableId]) -> Vec<String> {
    variables
        .iter()
        .map(|v| names.get(v.as_usize()).cloned().unwrap_or_else(|| format!("v{}", v.raw())))
        .collect()
}

/// Every regime label the plan can cite: the base catalog's and each compiled
/// candidate's.
fn labels(result: &StudyPlanResult) -> BTreeMap<RegimeId, String> {
    result
        .catalog()
        .regimes
        .iter()
        .chain(result.compiled().iter().flat_map(|c| c.delta.proposed_regimes.iter()))
        .map(|r| (r.id, r.label.as_deref().map_or_else(|| r.id.raw().to_string(), str::to_owned)))
        .collect()
}

fn plan_json(result: &StudyPlanResult, names: &[String]) -> serde_json::Value {
    let plan: &StudyPlan = result.plan();
    let label = labels(result);
    let regime = |id: &RegimeId| label.get(id).cloned().unwrap_or_else(|| id.raw().to_string());
    let deliver = |candidates: &[Arc<str>]| {
        result
            .compiled()
            .iter()
            .filter(|c| candidates.contains(&c.id))
            .flat_map(|c| c.delta.proposed_regimes.iter())
            .map(|r| {
                serde_json::json!({
                    "regime": regime(&r.id),
                    "population": r.population.as_ref(),
                    "interventions": names_of(names, &r.interventions),
                    "levels": r.intervention_values.iter().map(|a| {
                        (names_of(names, &[a.variable])[0].clone(), a.value.as_f64())
                    }).collect::<BTreeMap<_, _>>(),
                    "measured": names_of(names, &r.measured),
                    "study": r.study.as_deref(),
                })
            })
            .collect::<Vec<_>>()
    };
    let status = plan.status();
    serde_json::json!({
        "route": plan.route.name(),
        "outcome": match status {
            None => "sufficient",
            Some((_, "study_plan.budget")) => "exhausted",
            Some(_) => "none_certified",
        },
        "reason": status.map(|(code, _)| code),
        "detail": status.map(|(_, detail)| detail),
        "inference_claim": "none",
        "failure": {
            "code": plan.failure.code,
            "detail": plan.failure.detail,
            "facts": plan.failure.facts,
        },
        "subsets": plan.subsets.iter().map(|s| serde_json::json!({
            "candidates": s.candidates.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            "cost_units": s.cost_units,
            "sample_budget": s.sample_budget,
            "status": s.outcome.status(),
            "detail": match &s.outcome {
                StudySubsetOutcome::Insufficient { detail, .. } => Some((*detail).to_owned()),
                StudySubsetOutcome::Refused { message } => Some(message.clone()),
                _ => None,
            },
        })).collect::<Vec<_>>(),
        "proposals": plan.proposals.iter().enumerate().map(|(rank, p)| serde_json::json!({
            "rank": rank,
            "candidates": p.candidates.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            "cost_units": p.cost_units,
            "sample_budget": p.sample_budget,
            "stage": p.stage,
            "cited_regimes": p.cited_regimes.iter().map(regime).collect::<Vec<_>>(),
            "uncited_candidates": p.uncited_candidates.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            "decision_operations": p.decision_operations,
            "deliver": deliver(&p.candidates),
            "repairs": p.repairs.iter().map(|r| serde_json::json!({
                "candidate": r.candidate.as_ref(),
                "regime": regime(&r.regime),
                "population": r.population.as_ref(),
                "intervened": names_of(names, &r.intervened),
                "factors": r.factors.iter().map(|f| serde_json::json!({
                    "variables": names_of(names, &f.variables),
                    "conditioned_on": names_of(names, &f.conditioned_on),
                })).collect::<Vec<_>>(),
                "required_margin": names_of(names, &r.required_margin),
                "declared_margin": names_of(names, &r.declared_margin),
                "proof_steps": r.proof_steps,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "minimal": plan.minimal,
        "stop": plan.stop.as_ref().map(|stop| match stop {
            StudyPlanStop::Budget(receipt) => serde_json::json!({
                "kind": "budget",
                "stop": receipt.stop.code(),
                "operations_limit": receipt.operations_limit,
                "depth_limit": receipt.depth_limit,
                "memory_limit_bytes": receipt.memory_limit_bytes,
                "operations_consumed": receipt.operations_consumed,
                "depth_reached": receipt.depth_reached,
                "explored": receipt.explored,
                "unevaluated": receipt.unevaluated,
            }),
            StudyPlanStop::ProposalCap => serde_json::json!({"kind": "proposal_cap"}),
        }),
        "operations_consumed": plan.operations_consumed,
        "limits": {
            "operations": plan.limits.search.operations,
            "depth": plan.limits.search.depth,
            "memory_limit_bytes": plan.limits.memory_limit_bytes,
        },
    })
}

/// A finished study plan: frozen failure, every subset, ranked proposals.
#[pyclass(skip_from_py_object)]
struct StudyPlanStage {
    result: StudyPlanResult,
    names: Vec<String>,
    graph: Admg,
}

#[pymethods]
impl StudyPlanStage {
    /// `sufficient`, `none_certified` (nothing certified within the universe;
    /// never an impossibility claim) or `exhausted` (a budget stop before any
    /// proposal).
    #[getter]
    fn outcome(&self) -> &'static str {
        match self.result.plan().status() {
            None => "sufficient",
            Some((_, "study_plan.budget")) => "exhausted",
            Some(_) => "none_certified",
        }
    }

    /// The whole plan as a dictionary spelled in variable names and regime labels.
    fn plan(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py_json(py, &plan_json(&self.result, &self.names))
    }

    /// Accept arriving evidence for the proposal at `rank`: the catalog must
    /// keep the frozen base and hold every proposed regime (by its label) as
    /// available evidence of exactly the proposed shape, bound to
    /// `provider_snapshot`; the public route then decides the real catalog.
    fn receive(
        &self,
        py: Python<'_>,
        rank: usize,
        catalog: &Bound<'_, PyAny>,
        provider_snapshot: String,
    ) -> PyResult<Py<PyAny>> {
        let proposal = self
            .result
            .proposal(rank)
            .ok_or_else(|| crate::value_err(format!("the plan has no proposal of rank {rank}")))?;
        let actual = relabel(&parse_catalog(catalog, &self.graph)?, &labels(&self.result))?;
        let label = labels(&self.result);
        let regime = |id: &RegimeId| label.get(id).cloned().unwrap_or_else(|| id.raw().to_string());
        let arrival = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, None);
            proposal.receive(&actual, provider_snapshot, &ctx).map_err(plan_error)
        })?;
        let payload = match &arrival.decision {
            StudyArrivalDecision::Mz(decision) => match decision.as_ref() {
                MzTransportDecision::Identified { cited, .. } => serde_json::json!({
                    "outcome": "identified",
                    "route": "mz",
                    "cited_regimes": cited.iter().map(regime).collect::<Vec<_>>(),
                }),
                _ => serde_json::json!({"outcome": "not_identified"}),
            },
            StudyArrivalDecision::Mixed(decision) => match decision.as_ref() {
                MixedSourceDecision::Identified { cited, .. } => serde_json::json!({
                    "outcome": "identified",
                    "route": "mixed",
                    "cited_regimes": cited.iter().map(regime).collect::<Vec<_>>(),
                }),
                MixedSourceDecision::NamedRoute { route, .. } => serde_json::json!({
                    "outcome": "named_route",
                    "route": route,
                }),
                _ => serde_json::json!({"outcome": "not_identified"}),
            },
        };
        let mut payload = payload;
        payload["provider_snapshot"] = arrival.provider_snapshot.as_ref().into();
        to_py_json(py, &payload)
    }

    /// Export the plan as a portable, independently replayable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let wire = self.result.to_artifact().map_err(plan_error)?;
        let bytes = antecedent_io::to_cbor(&wire).map_err(serialization_error)?;
        let framed = frame_named_artifact(STUDY_PLAN_PREFIX, &self.names, bytes)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
}

/// Map an arriving catalog's regimes onto the plan's regime ids by label (the
/// Python catalog numbers regimes by sorted label, so ids alone do not travel).
fn relabel(
    actual: &EvidenceCatalog,
    known: &BTreeMap<RegimeId, String>,
) -> PyResult<EvidenceCatalog> {
    let by_label = known.iter().map(|(id, l)| (l.clone(), *id)).collect::<BTreeMap<_, _>>();
    let mut next = known.keys().map(|id| id.raw() + 1).max().unwrap_or(0);
    let mut map = BTreeMap::new();
    let mut regimes = Vec::with_capacity(actual.regimes.len());
    for regime in actual.regimes.iter() {
        let id = if let Some(id) = regime.label.as_deref().and_then(|l| by_label.get(l)) {
            *id
        } else {
            next += 1;
            RegimeId::from_raw(next)
        };
        map.insert(regime.id, id);
        let mut regime = regime.clone();
        regime.id = id;
        regimes.push(regime);
    }
    let bindings = actual
        .bindings
        .iter()
        .map(|b| {
            let mut b = b.clone();
            b.regime = map.get(&b.regime).copied().unwrap_or(b.regime);
            b
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new(
        Arc::clone(&actual.environments),
        regimes,
        bindings,
        actual.target_sampling,
    )
    .map_err(|e| crate::value_err(e.to_string()))
}

/// Plan the cheapest subset of at most three declared studies that would make
/// a failed X1 (`route="mz"`) or X9 (`route="mixed"`) decision identify.
#[pyfunction]
#[doc(hidden)]
#[pyo3(signature = (graph, route, target, outcomes, treatments, sources, catalog, candidates, *, max_operations, max_depth, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn plan_studies_stage(
    py: Python<'_>,
    graph: PyRef<'_, Admg>,
    route: &str,
    target: String,
    outcomes: Vec<String>,
    treatments: Vec<String>,
    sources: Vec<SourceTuple>,
    catalog: &Bound<'_, PyAny>,
    candidates: Vec<CandidateTuple>,
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<StudyPlanStage> {
    let names = graph.names.clone();
    let coordinates = |variables: &[String]| -> PyResult<Arc<[VariableId]>> {
        variables
            .iter()
            .map(|name| resolve(&names, name))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    let mut specs = Vec::with_capacity(sources.len());
    for (population, controllable, assignment, selections) in sources {
        specs.push(ZTransportSourceSpec {
            population: population.into(),
            controllable: coordinates(&controllable)?,
            experiment_assignment: intervention_assignments(&names, assignment)?,
            selection_targets: coordinates(&selections)?,
        });
    }
    let (outcomes, treatments) = (coordinates(&outcomes)?, coordinates(&treatments)?);
    let route = match route {
        "mz" => StudyPlanRoute::Mz(MzTransportQuery {
            outcomes,
            treatments,
            target: target.into(),
            sources: specs.into(),
        }),
        "mixed" => StudyPlanRoute::Mixed(MixedSourceQuery {
            outcomes,
            treatments,
            target: target.into(),
            sources: specs.into(),
        }),
        other => return Err(crate::value_err(format!("unknown route {other:?}; use mz or mixed"))),
    };
    let mut declared = Vec::with_capacity(candidates.len());
    for (
        id,
        population,
        on,
        levels,
        measured,
        recruitment,
        units,
        samples,
        requires,
        conflicts,
        notes,
    ) in candidates
    {
        let levels = match levels {
            None => None,
            Some(levels) => Some(
                levels
                    .into_iter()
                    .map(|level| intervention_assignments(&names, level))
                    .collect::<PyResult<Vec<_>>>()?
                    .into(),
            ),
        };
        let arcs = |v: Vec<String>| v.into_iter().map(Arc::<str>::from).collect::<Arc<[_]>>();
        declared.push(StudyCandidate {
            id: id.into(),
            population: population.into(),
            interventions: coordinates(&on)?,
            levels,
            measured: coordinates(&measured)?,
            recruitment: recruitment.into(),
            cost: StudyCost { units, sample_budget: samples },
            requires: arcs(requires),
            conflicts: arcs(conflicts),
            feasibility_constraints: arcs(notes),
        });
    }
    let named = Admg { admg: graph.admg.clone(), names: names.clone() };
    let base = parse_catalog(catalog, &named)?;
    let shared = graph.aligned_to_names(&names)?;
    let limits = StudyPlanLimits {
        search: SearchLimits { operations: max_operations, depth: max_depth },
        memory_limit_bytes: STUDY_PLAN_MEMORY_BYTES,
    };
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, memory_bytes, cancel);
        plan_studies(&shared, &route, &base, &declared, limits, &ctx).map_err(plan_error)
    })?;
    Ok(StudyPlanStage { result, names, graph: named })
}

/// Replay an exported study plan under its stored limits and return the plan.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_operations, max_depth, memory_bytes=None, cancel=None))]
fn replay_study_plan(
    py: Python<'_>,
    artifact: &[u8],
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(STUDY_PLAN_PREFIX, artifact, "study-plan")?;
    let wire: StudyPlanArtifactWire =
        antecedent_io::from_cbor(&bytes).map_err(serialization_error)?;
    let limits = StudyPlanConsumeLimits {
        search: SearchLimits { operations: max_operations, depth: max_depth },
        memory_limit_bytes: STUDY_PLAN_MEMORY_BYTES,
    };
    crate::detach_catch(py, move || {
        let graph =
            antecedent_io::admg_from_wire(&wire.premises.graph).map_err(serialization_error)?;
        crate::transport_exact_api::validate_artifact_names(&names, &graph)?;
        let ctx = execution_context(0, memory_bytes, cancel);
        let consumed = wire.consume_with_limits(limits, &ctx).map_err(plan_error)?;
        let mut payload = plan_json(&consumed, &names);
        payload["premises_digest"] = wire.premises_digest.clone().into();
        payload["data_digest"] = wire.data_digest.clone().into();
        payload["plan_digest"] = wire.plan_digest.clone().into();
        Ok(payload.to_string())
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StudyPlanStage>()?;
    module.add_function(wrap_pyfunction!(plan_studies_stage, module)?)?;
    module.add_function(wrap_pyfunction!(replay_study_plan, module)?)?;
    Ok(())
}
