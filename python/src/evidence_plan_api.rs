//! Python bindings for candidate generation and evidence plans.
//!
//! Python declares the failed identification (a graph, a treatment, an outcome, the
//! variables measured today and, optionally, a sensitivity finding) and the studies
//! that might close the gap, all spelled in variable names. The theorem routes, the
//! obligations, the unfinished study templates, each route's theorem check, the
//! value evaluation, the ordering and the `evidence_plan_v1` artifact are Rust's. A
//! refusal comes back as the registered `CausalUnsupportedError` with its reason
//! code and stable detail in the message.
use crate::graphs::{Admg, Dag};
use crate::repair_api::{obligation_json, parse_candidates};
use crate::transport_common::{
    execution_context, frame_named_artifact, resolve, serialization_error, unframe_named_artifact,
};
use antecedent::analysis::evidence_plan::{
    EvidencePlanFacadeError, EvidencePlanRequest, EvidenceValueRequest, plan_over_routes,
    variable_names,
};
use antecedent::analysis::fallback::failure_next_actions;
use antecedent_core::{AssumptionSet, AverageEffectQuery, CausalQuery, SearchLimits, VariableId};
use antecedent_design::structure_identification::{
    GraphIdentFlag, GraphPosteriorSample, identified_under_intervention, structure_posterior,
};
use antecedent_design::{
    CandidateGeneration, CandidateTemplate, EvidencePlan, EvidencePlanArtifact,
    EvidencePlanArtifactError, FailureRoutes, PlanOption, RepairConsumeLimits, RepairLimits,
    RepairObjective, SensitivityFinding, SensitivityMethod, failure_routes, generate_candidates,
};
use antecedent_graph::Admg as CoreAdmg;
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};
use pyo3::prelude::*;
use serde_json::{Value, json};
use std::sync::Arc;

/// Magic prefix of a portable evidence plan: the design crate's container bytes,
/// framed with the variable names the plan was declared under.
pub(crate) const EVIDENCE_PLAN_PREFIX: &[u8] = b"ANTECEDENT-EVIDENCE-PLAN\x01";

fn facade_error(error: &EvidencePlanFacadeError) -> PyErr {
    // Every variant but the value evaluation's renders its own detail first.
    let text = match error {
        EvidencePlanFacadeError::Value(_) => format!("{}: {error}", error.detail()),
        _ => error.to_string(),
    };
    crate::refusal(error.code(), text)
}

fn artifact_error(error: &EvidencePlanArtifactError) -> PyErr {
    crate::refusal(error.code, format!("{}: {}", error.detail, error.message))
}

fn names_of(names: &[String], variables: &[VariableId]) -> Vec<String> {
    variables
        .iter()
        .map(|v| names.get(v.as_usize()).cloned().unwrap_or_else(|| format!("v{}", v.raw())))
        .collect()
}

fn ids_of(ids: &[Arc<str>]) -> Vec<String> {
    ids.iter().map(ToString::to_string).collect()
}

fn template_json(template: &CandidateTemplate, names: &[String]) -> Value {
    let mut label = format!("{} in {}", template.kind.as_str(), template.population);
    if !template.interventions.is_empty() {
        label.push_str(&format!(
            " setting [{}]",
            names_of(names, &template.interventions).join(", ")
        ));
    }
    if !template.measured.is_empty() {
        label.push_str(&format!(" measuring [{}]", names_of(names, &template.measured).join(", ")));
    }
    json!({
        "label": label,
        "kind": template.kind.as_str(),
        "population": template.population.as_ref(),
        "interventions": names_of(names, &template.interventions),
        "measured": names_of(names, &template.measured),
        "joint": template.joint_measurement,
        "evidence": template.expected_evidence.iter().map(|e| json!({
            "population": e.population.as_ref(),
            "interventions": names_of(names, &e.interventions),
            "levels": {},
            "conditioned_on": names_of(names, &e.conditioned_on),
            "measured": names_of(names, &e.measured),
            "joint": matches!(e.distribution, antecedent_core::DistributionAvailability::Joint),
        })).collect::<Vec<_>>(),
        "addresses": ids_of(&template.addresses),
        "routes": ids_of(&template.routes),
        "minimum_sample_size": template.minimum_sample_size,
    })
}

fn generation_json(generation: &CandidateGeneration, names: &[String]) -> Value {
    json!({
        "templates": generation.templates.iter().map(|t| template_json(t, names)).collect::<Vec<_>>(),
        "unmatched": generation.unmatched.iter().map(|u| json!({
            "id": u.id.as_ref(),
            "kind": u.kind.as_str(),
            "route": u.route.as_ref(),
            "reason": u.reason,
        })).collect::<Vec<_>>(),
    })
}

fn option_json(option: &PlanOption) -> Value {
    json!({
        "rank": option.rank,
        "route": option.route,
        "members": option.members.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        "labels": option.labels.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        "gate": option.gate.as_str(),
        "reasons": option.reasons,
        "derivation": option.derivation.as_ref().map(|d| json!({
            "checker": d.checker, "steps": d.steps, "verified": d.verified,
        })),
        "resolves": option.resolves.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        "unresolved": option.unresolved.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        "assumptions_remaining": option.assumptions_remaining.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        "cost": {
            "units": option.cost.units,
            "unit": option.cost.unit_label.as_ref(),
            "sample_budget": option.cost.sample_budget,
        },
        "value": option.value.as_ref().map(|v| json!({
            "evsi": v.evsi,
            "evpi": v.evpi,
            "mc_stderr": v.mc_stderr,
            "integration": v.integration,
            "study_cost_utility": v.study_cost_utility,
            "net_value": v.net_value,
            "provider_trust": v.provider_trust,
        })),
        "value_note": option.value_note,
        "rank_uncertain": option.rank_uncertain,
    })
}

fn plan_json(plan: &EvidencePlan, names: &[String]) -> Value {
    let status = plan.status();
    json!({
        "outcome": plan.outcome.as_str(),
        "reason": status.map(|(code, _)| code),
        "detail": status.map(|(_, detail)| detail),
        "inference_claim": "none",
        "ranking_guarantee": plan.ranking_guarantee(),
        "calibration": plan.calibration(),
        "cost_unit": plan.cost_unit.as_ref(),
        "basis": plan.basis.as_str(),
        "cost_map": plan.cost_map.as_ref().map(|m| json!({
            "cost_unit": m.cost_unit, "utility_unit": m.utility_unit,
            "utility_per_cost": m.utility_per_cost,
        })),
        "utility_unit": plan.utility_unit,
        "value_identity": plan.value_identity,
        "identity": plan.identity(),
        "routes": plan.routes.iter().map(|r| json!({
            "family": r.family,
            "contract": r.contract,
            "obligations": r.obligations.iter().map(|o| obligation_json(o, names)).collect::<Vec<_>>(),
            "budget": {
                "unevaluated_total": r.budget.unevaluated_total,
                "stop": r.budget.stop,
                "beyond_declared_depth": r.budget.beyond_declared_depth,
            },
        })).collect::<Vec<_>>(),
        "unverified": plan.unverified.iter().map(|u| json!({
            "route": u.route.as_ref(),
            "obligations": u.obligations.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            "reason": u.reason.as_ref(),
        })).collect::<Vec<_>>(),
        "options": plan.options.iter().map(option_json).collect::<Vec<_>>(),
    })
}

/// The variables whose measurement or setting a verified study relies on: the ranker's
/// unlock list, derived from the repair reports instead of declared by hand.
fn unlocks_json(
    reports: &[antecedent_design::RepairReport],
    candidates: &[antecedent_design::DurableStudyCandidate],
    observed: &[VariableId],
    names: &[String],
) -> Value {
    let mut variables: Vec<VariableId> = Vec::new();
    let mut verified: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for report in reports {
        let unlocks = antecedent_design::verified_unlocks(report, candidates, observed);
        variables.extend(unlocks.variables);
        verified.extend(unlocks.verified);
    }
    variables.sort_unstable_by_key(|v| v.raw());
    variables.dedup();
    json!({
        "variables": names_of(names, &variables),
        "ids": variables.iter().map(|v| v.raw()).collect::<Vec<_>>(),
        "verified": candidates.iter().enumerate()
            .filter(|(i, _)| verified.contains(i))
            .map(|(_, c)| c.label.as_ref())
            .collect::<Vec<&str>>(),
        "unverified": candidates.iter().enumerate()
            .filter(|(i, _)| !verified.contains(i))
            .map(|(_, c)| c.label.as_ref())
            .collect::<Vec<&str>>(),
    })
}

/// A failed identification and the theorem routes a study could repair.
#[pyclass(skip_from_py_object)]
pub(crate) struct FailureStage {
    routes: FailureRoutes,
    observed: Vec<VariableId>,
    names: Vec<String>,
    population: String,
    treatment: VariableId,
    outcome: VariableId,
}

#[pymethods]
impl FailureStage {
    /// `failure`.
    #[getter]
    fn family(&self) -> &'static str {
        "failure"
    }

    /// The failed contract's stable identity.
    #[getter]
    fn contract_id(&self) -> String {
        format!(
            "failure:{}:treatment={}:outcome={}",
            self.population,
            self.treatment.raw(),
            self.outcome.raw()
        )
    }

    /// The theorem routes a study could repair, then the routes no checker decides.
    #[getter]
    fn routes(&self) -> Vec<String> {
        let mut out: Vec<String> =
            self.routes.routes.iter().map(|r| r.family().family_id().to_owned()).collect();
        out.extend(self.routes.unverified().iter().map(|u| u.route.to_string()));
        out
    }

    /// Every variable name a candidate may use: the graph's, the unmeasured common
    /// causes of its bidirected edges and the sensitivity confounder.
    #[getter]
    fn variable_names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// The unresolved obligations of every route, each with its proof step.
    ///
    /// Scientific coordinates are declared on a single-route contract; here a
    /// non-empty declaration refuses.
    #[pyo3(signature = (quantities_json=None))]
    fn obligations(&self, py: Python<'_>, quantities_json: Option<&str>) -> PyResult<Py<PyAny>> {
        if let Some(text) = quantities_json {
            let declared: serde_json::Map<String, Value> = serde_json::from_str(text)
                .map_err(|e| crate::value_err(format!("invalid coordinate declaration: {e}")))?;
            if !declared.is_empty() {
                return Err(crate::value_err(
                    "declare scientific coordinates on a single-route contract, not on a failure",
                ));
            }
        }
        let values: Vec<Value> =
            self.routes.obligations().iter().map(|o| obligation_json(o, &self.names)).collect();
        crate::transport_z_api::to_py_json(py, &Value::Array(values))
    }

    /// The evidence-gathering next actions those obligations imply: `measure_variable`,
    /// `run_experiment`, `find_instrument` and `collect_panel`. An assumption obligation
    /// yields no action, because no study satisfies it.
    fn next_actions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let actions = failure_next_actions(&self.routes.obligations(), &self.observed, &self.names);
        let values: Vec<Value> = actions
            .iter()
            .map(|a| json!({ "id": a.id.as_ref(), "description": a.description.as_ref() }))
            .collect();
        crate::transport_z_api::to_py_json(py, &Value::Array(values))
    }

    /// The unfinished studies those obligations determine.
    fn templates(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let generation = generate_candidates(&self.routes.obligations());
        crate::transport_z_api::to_py_json(py, &generation_json(&generation, &self.names))
    }

    /// Verify every declared study on every route and order the options.
    #[pyo3(signature = (candidates_json, objective, *, max_operations, max_depth, memory_bytes=None, value_json=None, cancel=None))]
    #[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
    fn plan(
        &self,
        py: Python<'_>,
        candidates_json: &str,
        objective: &str,
        max_operations: usize,
        max_depth: usize,
        memory_bytes: Option<u64>,
        value_json: Option<&str>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<EvidencePlanStage> {
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
        let value: Option<EvidenceValueRequest> = value_json
            .map(|text| {
                serde_json::from_str(text)
                    .map_err(|e| crate::value_err(format!("invalid value declaration: {e}")))
            })
            .transpose()?;
        let request = EvidencePlanRequest {
            candidates,
            objective,
            limits: RepairLimits {
                search: SearchLimits { operations: max_operations, depth: max_depth },
                memory_limit_bytes: memory_bytes
                    .unwrap_or_else(|| RepairLimits::default().memory_limit_bytes),
            },
            sensitivity: None,
            value,
        };
        let routes = self.routes.clone();
        let names = self.names.clone();
        let observed = self.observed.clone();
        let (plan, generation, artifact, unlocks) = crate::detach_catch(py, move || {
            let ctx = execution_context(0, memory_bytes, cancel);
            let result = plan_over_routes(&routes, names.clone(), &request, &ctx)
                .map_err(|e| facade_error(&e))?;
            let artifact = result.artifact().map(Clone::clone).map_err(Clone::clone);
            let unlocks = unlocks_json(&result.reports, &request.candidates, &observed, &names);
            Ok((result.plan, result.generation, artifact, unlocks))
        })?;
        Ok(EvidencePlanStage {
            plan,
            names: self.names.clone(),
            generation: Some(generation),
            unlocks: Some(unlocks),
            artifact,
        })
    }
}

fn finding(sensitivity: Option<(String, f64, bool)>) -> PyResult<Option<SensitivityFinding>> {
    sensitivity
        .map(|(method, strength, robust)| {
            let method = SensitivityMethod::from_name(&method).ok_or_else(|| {
                crate::value_err(format!(
                    "unknown sensitivity method {method:?}; use e_value, omitted_variable_bias or tipping"
                ))
            })?;
            Ok(SensitivityFinding { method, strength, robust })
        })
        .transpose()
}

fn failure_stage(
    graph: CoreAdmg,
    graph_names: Vec<String>,
    treatment: &str,
    outcome: &str,
    population: String,
    observed: &[String],
    sensitivity: Option<(String, f64, bool)>,
) -> PyResult<FailureStage> {
    let t = resolve(&graph_names, treatment)?;
    let y = resolve(&graph_names, outcome)?;
    let observed_ids =
        observed.iter().map(|name| resolve(&graph_names, name)).collect::<PyResult<Vec<_>>>()?;
    // A plan is for a failure: the complete ID algorithm must not identify the effect.
    match antecedent_design::structure_identification::identifies(&graph, t, y) {
        Ok(GraphIdentFlag::Unidentified) => {}
        Ok(GraphIdentFlag::Identified) => {
            return Err(crate::refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "identification_repair.not_a_failure: the effect is identified by the graph; \
                 there is no gap to plan for",
            ));
        }
        Err(error) => {
            return Err(crate::refusal(error.code, format!("{}: {}", error.detail, error.message)));
        }
    }
    let result = IdentificationResult::not_identified(
        CausalQuery::average_effect(AverageEffectQuery::binary_ate(t, y)),
        DerivationTrace::default(),
        AssumptionSet::new(),
        IdentificationPerformanceRecord::default(),
    );
    let finding = finding(sensitivity)?;
    let routes = failure_routes(&result, &graph, &observed_ids, &population, finding.as_ref())
        .map_err(|e| crate::refusal(e.code, format!("{}: {}", e.detail, e.message)))?;
    let study_names: Vec<Arc<str>> = graph_names.iter().map(|n| Arc::from(n.as_str())).collect();
    let names = variable_names(&study_names, &routes, &graph);
    Ok(FailureStage { routes, observed: observed_ids, names, population, treatment: t, outcome: y })
}

/// Freeze the failed identification of a directed graph.
#[pyfunction]
#[pyo3(signature = (graph, treatment, outcome, population, observed, sensitivity=None))]
fn evidence_failure_dag(
    graph: PyRef<'_, Dag>,
    treatment: &str,
    outcome: &str,
    population: String,
    observed: Vec<String>,
    sensitivity: Option<(String, f64, bool)>,
) -> PyResult<FailureStage> {
    let admg = CoreAdmg::from_dag(&graph.dag).map_err(|e| crate::value_err(e.to_string()))?;
    failure_stage(admg, graph.names.clone(), treatment, outcome, population, &observed, sensitivity)
}

/// Freeze the failed identification of a graph with bidirected edges.
#[pyfunction]
#[pyo3(signature = (graph, treatment, outcome, population, observed, sensitivity=None))]
fn evidence_failure_admg(
    graph: PyRef<'_, Admg>,
    treatment: &str,
    outcome: &str,
    population: String,
    observed: Vec<String>,
    sensitivity: Option<(String, f64, bool)>,
) -> PyResult<FailureStage> {
    let names = graph.names.clone();
    let aligned = graph.aligned_to_names(&names)?;
    failure_stage(aligned, names, treatment, outcome, population, &observed, sensitivity)
}

/// A finished plan: the ordered options, the routes behind them and the artifact.
#[pyclass(skip_from_py_object)]
pub(crate) struct EvidencePlanStage {
    plan: EvidencePlan,
    names: Vec<String>,
    generation: Option<CandidateGeneration>,
    unlocks: Option<Value>,
    artifact: Result<EvidencePlanArtifact, EvidencePlanArtifactError>,
}

#[pymethods]
impl EvidencePlanStage {
    /// `closing`, `none_certified` or `budget_exhausted`.
    #[getter]
    fn outcome(&self) -> &'static str {
        self.plan.outcome.as_str()
    }

    /// The whole plan as a dictionary spelled in variable names.
    fn report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut report = plan_json(&self.plan, &self.names);
        if let Value::Object(map) = &mut report {
            if let Some(generation) = &self.generation {
                map.insert("generation".to_owned(), generation_json(generation, &self.names));
            }
            if let Some(unlocks) = &self.unlocks {
                map.insert("unlocks".to_owned(), unlocks.clone());
            }
        }
        crate::transport_z_api::to_py_json(py, &report)
    }

    /// Export the plan as a portable, independently replayable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let artifact = self.artifact.as_ref().map_err(artifact_error)?;
        let bytes = artifact.to_bytes("evidence-plan").map_err(|e| artifact_error(&e))?;
        let framed = frame_named_artifact(EVIDENCE_PLAN_PREFIX, &self.names, bytes)?;
        Ok(pyo3::types::PyBytes::new(py, &framed).unbind())
    }
}

/// Replay an exported evidence plan under the consumer's limits and return it.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_operations, max_depth, memory_bytes=None, cancel=None))]
fn consume_evidence_plan(
    py: Python<'_>,
    artifact: &[u8],
    max_operations: usize,
    max_depth: usize,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<EvidencePlanStage> {
    let (names, bytes) = unframe_named_artifact(EVIDENCE_PLAN_PREFIX, artifact, "evidence plan")?;
    let limits = RepairConsumeLimits {
        search: SearchLimits { operations: max_operations, depth: max_depth },
        memory_limit_bytes: memory_bytes
            .unwrap_or_else(|| RepairConsumeLimits::default().memory_limit_bytes),
    };
    crate::detach_catch(py, move || {
        let stored = EvidencePlanArtifact::from_bytes(&bytes).map_err(|e| artifact_error(&e))?;
        let ctx = execution_context(0, memory_bytes, cancel);
        let plan = stored.consume(limits, &ctx).map_err(|e| artifact_error(&e))?;
        Ok(EvidencePlanStage { plan, names, generation: None, unlocks: None, artifact: Ok(stored) })
    })
}

/// The identification flags of a weighted graph posterior, derived by running the
/// complete ID algorithm on each structure (and, for an experiment that sets
/// `interventions`, on each mutilated structure).
#[pyfunction]
#[pyo3(signature = (graphs, weights, keys, treatment, outcome, interventions=None))]
fn identify_structure_posterior(
    graphs: Vec<PyRef<'_, Admg>>,
    weights: Vec<f64>,
    keys: Vec<u64>,
    treatment: &str,
    outcome: &str,
    interventions: Option<Vec<String>>,
) -> PyResult<(Vec<bool>, Option<Vec<bool>>)> {
    if graphs.is_empty() || graphs.len() != weights.len() || graphs.len() != keys.len() {
        return Err(crate::value_err("declare one weight and one key per structure"));
    }
    let names = graphs[0].names.clone();
    let t = resolve(&names, treatment)?;
    let y = resolve(&names, outcome)?;
    let mut samples = Vec::with_capacity(graphs.len());
    for ((graph, weight), key) in graphs.iter().zip(&weights).zip(&keys) {
        samples.push(GraphPosteriorSample {
            weight: *weight,
            key: *key,
            graph: graph.aligned_to_names(&names)?,
        });
    }
    let refuse = |e: antecedent_design::StructureIdentificationError| {
        crate::refusal(e.code, format!("{}: {}", e.detail, e.message))
    };
    let posterior = structure_posterior(&samples, t, y).map_err(refuse)?;
    let identified =
        posterior.identified.iter().map(|flag| *flag == GraphIdentFlag::Identified).collect();
    let under = interventions
        .map(|targets| {
            let targets =
                targets.iter().map(|name| resolve(&names, name)).collect::<PyResult<Vec<_>>>()?;
            identified_under_intervention(&samples, t, y, &targets)
                .map(|flags| {
                    flags.iter().map(|flag| *flag == GraphIdentFlag::Identified).collect::<Vec<_>>()
                })
                .map_err(refuse)
        })
        .transpose()?;
    Ok((identified, under))
}

/// Reason codes of the identification-class refusals the opt-in descriptive fallback converts.
#[pyfunction]
fn identification_refusal_codes() -> Vec<String> {
    antecedent::IDENTIFICATION_REFUSAL_CODES.iter().map(|code| (*code).to_owned()).collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FailureStage>()?;
    module.add_class::<EvidencePlanStage>()?;
    module.add_function(wrap_pyfunction!(evidence_failure_dag, module)?)?;
    module.add_function(wrap_pyfunction!(evidence_failure_admg, module)?)?;
    module.add_function(wrap_pyfunction!(consume_evidence_plan, module)?)?;
    module.add_function(wrap_pyfunction!(identify_structure_posterior, module)?)?;
    module.add_function(wrap_pyfunction!(identification_refusal_codes, module)?)?;
    Ok(())
}
