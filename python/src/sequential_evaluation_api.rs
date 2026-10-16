//! Bounded Python bridge for sequential evaluation of a supplied contingent policy.
//!
//! Python builds the declarations (a finite model, a policy table, logged
//! trajectories, comparisons) as JSON; the enumeration, the importance
//! weighting, the rollouts, the value-of-information cross-check, the belief
//! session, the artifact and every refusal are Rust's. A refusal comes back as
//! structured JSON (`code`, `detail`, `offending`, `remedy`) for the Python layer
//! to raise as its own exception type. A malformed declaration is an input error.

use antecedent::analysis::sequential_evaluation as seq;
use antecedent_core::reason_code;
use antecedent_design::sequential_evaluation_artifact::StepWire;
use antecedent_io::IoError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::bridge::{RefusalJson, check_declared_size, external_refusal_json, serialization};
use crate::measurement_model_api::{model_json as channel_json, parse_model as parse_channel};
use crate::transport_common::execution_context;

/// Largest declaration this bridge parses, in bytes of JSON text.
const MAX_DECLARATION_BYTES: usize = 128 * 1024 * 1024;

type Pair = (Option<String>, Option<String>);

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DelayedIn {
    delay: usize,
    utility: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionIn {
    label: String,
    role: String,
    utility_cost: f64,
    resource_use: f64,
    transition: Vec<f64>,
    channel: Option<Json>,
    immediate_utility: Vec<f64>,
    delayed: Option<DelayedIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryIn {
    label: String,
    states: Vec<usize>,
    upper: Option<f64>,
    lower: Option<f64>,
    stop_action: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OverrideIn {
    history: Vec<(usize, usize)>,
    allowed: Vec<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelIn {
    label: String,
    population: String,
    regime: String,
    period: String,
    state_variable: String,
    states: Vec<String>,
    initial: Vec<f64>,
    horizon: usize,
    actions: Vec<ActionIn>,
    terminal_utility: Vec<f64>,
    censoring: String,
    resource_budget: Option<f64>,
    stopping: Vec<BoundaryIn>,
    support_default: Vec<bool>,
    support_overrides: Vec<OverrideIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionIn {
    history: Vec<(usize, usize)>,
    actions: Vec<(usize, f64)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyIn {
    version: String,
    decisions: Vec<DecisionIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoggedStepIn {
    action: usize,
    observation: Option<usize>,
    probability: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrajectoryIn {
    id: String,
    steps: Vec<LoggedStepIn>,
    outcome_utility: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VoiIn {
    collect_action: usize,
    commit_actions: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestIn {
    target_population: Option<String>,
    target_period: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedIn {
    model: Option<String>,
    policy: Option<String>,
    evaluation: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutIn {
    seed: u64,
    rollouts: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SpecIn {
    Contingent { policy: PolicyIn },
    ReweightedStaticScores { scores: Vec<f64>, weights: Vec<f64> },
    LearnedFromEvaluationData { rule: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComparisonIn {
    label: String,
    kind: String,
    policy: SpecIn,
}

fn declaration<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    check_declared_size(text, MAX_DECLARATION_BYTES, what)?;
    serde_json::from_str(text)
        .map_err(|error| crate::value_err(format!("invalid {what} declaration: {error}")))
}

pub(crate) fn model_of(text: &str) -> PyResult<seq::FiniteSequentialModel> {
    let wire: ModelIn = declaration(text, "sequential model")?;
    let mut actions = Vec::with_capacity(wire.actions.len());
    for action in wire.actions {
        let role = seq::ActionRole::from_name(&action.role)
            .ok_or_else(|| crate::value_err(format!("unknown action role `{}`", action.role)))?;
        let channel = match &action.channel {
            Some(value) => Some(seq::ObservationChannel { measurement: parse_channel(value)? }),
            None => None,
        };
        actions.push(seq::ActionSpec {
            label: action.label,
            role,
            utility_cost: action.utility_cost,
            resource_use: action.resource_use,
            transition: action.transition,
            channel,
            immediate_utility: action.immediate_utility,
            delayed: action
                .delayed
                .map(|d| seq::DelayedOutcome { delay: d.delay, utility: d.utility }),
        });
    }
    let censoring = seq::CensoringRule::from_name(&wire.censoring)
        .ok_or_else(|| crate::value_err(format!("unknown censoring rule `{}`", wire.censoring)))?;
    let mut support = seq::ActionSupport::by_action(wire.support_default);
    for item in wire.support_overrides {
        support = support.with_history(key_of(&item.history), item.allowed);
    }
    Ok(seq::FiniteSequentialModel {
        label: wire.label,
        population: wire.population,
        regime: wire.regime,
        period: wire.period,
        state_variable: wire.state_variable,
        states: wire.states,
        initial: wire.initial,
        horizon: wire.horizon,
        actions,
        terminal_utility: wire.terminal_utility,
        censoring,
        resource_budget: wire.resource_budget,
        stopping: wire
            .stopping
            .into_iter()
            .map(|b| seq::StoppingBoundary {
                label: b.label,
                states: b.states,
                upper: b.upper,
                lower: b.lower,
                stop_action: b.stop_action,
            })
            .collect(),
        support,
    })
}

fn key_of(history: &[(usize, usize)]) -> seq::HistoryKey {
    seq::HistoryKey::from_steps(
        history.iter().map(|&(action, observation)| seq::HistoryStep { action, observation }),
    )
}

fn policy_from(wire: PolicyIn) -> Result<seq::ContingentPolicy, seq::SequentialEvaluationError> {
    let mut policy = seq::ContingentPolicy::new(wire.version);
    for decision in wire.decisions {
        policy = policy.decide_stochastic(key_of(&decision.history), decision.actions)?;
    }
    Ok(policy)
}

fn trajectories_of(text: &str) -> PyResult<Vec<seq::LoggedTrajectory>> {
    let wire: Vec<TrajectoryIn> = declaration(text, "logged trajectories")?;
    Ok(wire
        .into_iter()
        .map(|t| seq::LoggedTrajectory {
            id: t.id,
            steps: t
                .steps
                .into_iter()
                .map(|s| seq::LoggedStep {
                    action: s.action,
                    observation: s.observation,
                    assignment_probability: s.probability,
                })
                .collect(),
            outcome_utility: t.outcome_utility,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Refusals and results
// ---------------------------------------------------------------------------

fn refusal(error: &seq::SequentialEvaluationError) -> String {
    external_refusal_json(&error.to_refusal())
}

fn io_refusal(error: &IoError) -> String {
    match error {
        IoError::Refused { code, message } => {
            RefusalJson::from_coded_message(code, "sequential_evaluation", message).json()
        }
        other => RefusalJson::new(
            reason_code!("invalid_argument"),
            "sequential_evaluation",
            "sequential_evaluation.malformed",
            &other.to_string(),
        )
        .json(),
    }
}

fn history_json(history: &seq::HistoryKey) -> Json {
    Json::Array(history.steps().iter().map(|s| json!([s.action, s.observation])).collect())
}

fn distribution_json(actions: &[(usize, f64)]) -> Json {
    Json::Array(actions.iter().map(|(a, p)| json!([a, p])).collect())
}

fn policy_json(policy: &seq::ContingentPolicy) -> Json {
    json!({
        "version": policy.version(),
        "decisions": policy
            .entries()
            .map(|(history, actions)| json!({
                "history": history_json(history),
                "actions": distribution_json(actions),
            }))
            .collect::<Vec<_>>(),
    })
}

fn evaluation_json(e: &seq::PolicyEvaluation) -> Json {
    let histories: Vec<Json> = e
        .histories
        .iter()
        .map(|h| {
            let boundary = match h.kind {
                seq::NodeKind::ForcedStop { boundary } => Json::from(boundary),
                _ => Json::Null,
            };
            json!({
                "history": history_json(&h.history),
                "time": h.time,
                "kind": h.kind.as_str(),
                "boundary": boundary,
                "reach_probability": h.reach_probability,
                "belief": h.belief,
                "realized_utility": h.realized_utility,
                "continuation_value": h.continuation_value,
                "forfeited_outcomes": h.forfeited_outcomes,
            })
        })
        .collect();
    let decisions: Vec<Json> = e
        .decisions
        .iter()
        .map(|d| {
            json!({
                "history": history_json(&d.history),
                "time": d.time,
                "forced_by": d.forced_by,
                "actions": distribution_json(&d.actions),
                "resource_used": d.resource_used,
            })
        })
        .collect();
    json!({
        "value": e.value,
        "policy_version": e.policy_version,
        "policy_identity": e.policy_identity,
        "model_identity": e.model_identity,
        "histories": histories,
        "decisions": decisions,
        "nodes_visited": e.nodes_visited,
        "accounting_residual": e.accounting_residual,
        "measurement_identities": e
            .measurement_identities
            .iter()
            .map(|(action, identity)| json!([action, identity]))
            .collect::<Vec<_>>(),
        "assumptions": {
            "exchangeability": e.assumptions.exchangeability,
            "consistency": e.assumptions.consistency,
            "positivity": e.assumptions.positivity,
        },
        "inference": e.inference,
    })
}

fn limits_of(max_nodes: Option<u64>) -> seq::EvaluationLimits {
    let mut limits = seq::EvaluationLimits::default();
    if let Some(nodes) = max_nodes {
        limits.max_nodes = nodes;
    }
    limits
}

fn ok_json(value: &Json) -> Pair {
    (Some(value.to_string()), None)
}

fn refused_pair(text: String) -> Pair {
    (None, Some(text))
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Validate a model and return its identity.
#[pyfunction]
fn sequential_model_check(model_json: &str) -> PyResult<Pair> {
    let model = model_of(model_json)?;
    Ok(match model.validate() {
        Ok(()) => ok_json(&json!({ "identity": model.identity_hex() })),
        Err(error) => refused_pair(refusal(&error)),
    })
}

/// Evaluate a fixed policy by exact enumeration.
#[pyfunction]
#[pyo3(signature = (model_json, policy_json, max_nodes=None, *, cancel=None))]
fn sequential_evaluate(
    py: Python<'_>,
    model_json: &str,
    policy_json: &str,
    max_nodes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Pair> {
    let model = model_of(model_json)?;
    let wire: PolicyIn = declaration(policy_json, "sequential policy")?;
    let policy = match policy_from(wire) {
        Ok(policy) => policy,
        Err(error) => return Ok(refused_pair(refusal(&error))),
    };
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(seq::evaluate_policy(&model, &policy, limits_of(max_nodes), &ctx))
    })?;
    Ok(match outcome {
        Ok(evaluation) => ok_json(&evaluation_json(&evaluation)),
        Err(error) => refused_pair(refusal(&error)),
    })
}

/// Evaluate a policy from logged trajectories under a logging policy.
#[pyfunction]
#[pyo3(signature = (model_json, policy_json, behaviour_json, log_json, max_nodes=None, *, cancel=None))]
fn sequential_off_policy(
    py: Python<'_>,
    model_json: &str,
    policy_json: &str,
    behaviour_json: &str,
    log_json: &str,
    max_nodes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Pair> {
    let model = model_of(model_json)?;
    let target: PolicyIn = declaration(policy_json, "sequential policy")?;
    let behaviour: PolicyIn = declaration(behaviour_json, "logging policy")?;
    let log = trajectories_of(log_json)?;
    let (target, behaviour) = match (policy_from(target), policy_from(behaviour)) {
        (Ok(target), Ok(behaviour)) => (target, behaviour),
        (Err(error), _) | (_, Err(error)) => return Ok(refused_pair(refusal(&error))),
    };
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let mut ledger = seq::EvidenceLedger::new();
        Ok(seq::evaluate_off_policy(
            &model,
            &target,
            &behaviour,
            &log,
            &mut ledger,
            limits_of(max_nodes),
            &ctx,
        ))
    })?;
    Ok(match outcome {
        Ok(e) => ok_json(&json!({
            "ordinary": e.ordinary,
            "self_normalized": e.self_normalized,
            "effective_sample_size": e.effective_sample_size,
            "max_weight": e.max_weight,
            "trajectories": e.trajectories,
            "uncalibrated_standard_error": e.uncalibrated_standard_error,
            "policy_version": e.policy_version,
            "policy_identity": e.policy_identity,
            "behaviour_version": e.behaviour_version,
            "behaviour_identity": e.behaviour_identity,
            "model_identity": e.model_identity,
            "positivity_histories_checked": e.positivity_histories_checked,
            "records_counted": e.records_counted,
            "reused_recorded_log": e.reused_recorded_log,
            "ledger_state": e.ledger_state,
            "inference": e.inference,
        })),
        Err(error) => refused_pair(refusal(&error)),
    })
}

/// Simulate a policy on common random streams.
#[pyfunction]
#[pyo3(signature = (model_json, policy_json, seed, rollouts, *, cancel=None))]
fn sequential_rollout(
    py: Python<'_>,
    model_json: &str,
    policy_json: &str,
    seed: u64,
    rollouts: u64,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Pair> {
    let model = model_of(model_json)?;
    let wire: PolicyIn = declaration(policy_json, "sequential policy")?;
    let policy = match policy_from(wire) {
        Ok(policy) => policy,
        Err(error) => return Ok(refused_pair(refusal(&error))),
    };
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(seq::rollout_policy(&model, &policy, seq::RolloutConfig { seed, rollouts }, &ctx))
    })?;
    Ok(match outcome {
        Ok(e) => ok_json(&json!({
            "mean": e.mean,
            "monte_carlo_standard_error": e.monte_carlo_standard_error,
            "rollouts": e.rollouts,
            "seed": e.seed,
            "stream_scheme": e.stream_scheme,
            "error_label": e.error_label,
            "policy_version": e.policy_version,
            "policy_identity": e.policy_identity,
            "model_identity": e.model_identity,
            "inference": e.inference,
        })),
        Err(error) => refused_pair(refusal(&error)),
    })
}

fn voi_json(check: &seq::VoiCheck) -> Json {
    json!({
        "collect_action": check.collect_action,
        "prior_value": check.prior_value,
        "expected_value_of_sample_information": check.expected_value_of_sample_information,
        "cost": check.cost,
        "net_value_of_collecting": check.net_value_of_collecting,
        "predicted_policy_value": check.predicted_policy_value,
        "enumerated_policy_value": check.enumerated_policy_value,
        "ranking_evaluation": check.ranking_evaluation,
        "implemented_functional": check.implemented_functional,
        "bayes_after_observation": policy_json(&check.bayes_after_observation),
    })
}

/// Score the value of collecting one observation through the design ranker.
#[pyfunction]
#[pyo3(signature = (model_json, voi_request_json, max_nodes=None, *, cancel=None))]
fn sequential_value_of_evidence(
    py: Python<'_>,
    model_json: &str,
    voi_request_json: &str,
    max_nodes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Pair> {
    let model = model_of(model_json)?;
    let voi: VoiIn = declaration(voi_request_json, "value-of-information request")?;
    let request =
        seq::VoiRequest { collect_action: voi.collect_action, commit_actions: voi.commit_actions };
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(seq::value_of_collecting_evidence(&model, &request, limits_of(max_nodes), &ctx))
    })?;
    Ok(match outcome {
        Ok(check) => ok_json(&voi_json(&check)),
        Err(error) => refused_pair(refusal(&error)),
    })
}

/// Evaluate named comparisons, optionally with the value-of-information check.
#[pyfunction]
#[pyo3(signature = (model_json, comparisons_json, voi_request_json=None, request_json=None, max_nodes=None, *, cancel=None))]
fn sequential_compare(
    py: Python<'_>,
    model_json: &str,
    comparisons_json: &str,
    voi_request_json: Option<&str>,
    request_json: Option<&str>,
    max_nodes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Pair> {
    let model = model_of(model_json)?;
    let wire: Vec<ComparisonIn> = declaration(comparisons_json, "comparisons")?;
    let voi = match voi_request_json {
        Some(text) => {
            let voi: VoiIn = declaration(text, "value-of-information request")?;
            Some(seq::VoiRequest {
                collect_action: voi.collect_action,
                commit_actions: voi.commit_actions,
            })
        }
        None => None,
    };
    let request = match request_json {
        Some(text) => {
            let request: RequestIn = declaration(text, "request")?;
            seq::SequentialRequest {
                target_population: request.target_population,
                target_period: request.target_period,
            }
        }
        None => seq::SequentialRequest::default(),
    };
    let mut specs = Vec::with_capacity(wire.len());
    for item in wire {
        let kind = seq::ComparisonKind::from_name(&item.kind)
            .ok_or_else(|| crate::value_err(format!("unknown comparison kind `{}`", item.kind)))?;
        let policy = match item.policy {
            SpecIn::Contingent { policy } => match policy_from(policy) {
                Ok(policy) => seq::PolicySpec::Contingent(policy),
                Err(error) => return Ok(refused_pair(refusal(&error))),
            },
            SpecIn::ReweightedStaticScores { scores, weights } => {
                seq::PolicySpec::ReweightedStaticScores { scores, weights }
            }
            SpecIn::LearnedFromEvaluationData { rule } => {
                seq::PolicySpec::LearnedFromEvaluationData { rule }
            }
        };
        specs.push(seq::ComparisonSpec { label: item.label, kind, policy });
    }
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(seq::compare_policies(&model, specs, voi.as_ref(), &request, limits_of(max_nodes), &ctx))
    })?;
    Ok(match outcome {
        Ok(report) => {
            let rows: Vec<Json> = report
                .rows
                .iter()
                .map(|row| {
                    json!({
                        "label": row.label,
                        "kind": row.kind.as_str(),
                        "policy_version": row.policy_version,
                        "policy_identity": row.policy_identity,
                        "value": row.value,
                        "difference_from_first": row.difference_from_first,
                        "evaluation": evaluation_json(&row.evaluation),
                    })
                })
                .collect();
            ok_json(&json!({
                "rows": rows,
                "best": report.best,
                "value_of_information": report.value_of_information.as_ref().map(voi_json),
                "model_identity": report.model_identity,
                "inference": report.inference,
            }))
        }
        Err(error) => refused_pair(refusal(&error)),
    })
}

/// Check that a request is about the model's own population and period.
#[pyfunction]
fn sequential_check_request(model_json: &str, request_json: &str) -> PyResult<Option<String>> {
    let model = model_of(model_json)?;
    let request: RequestIn = declaration(request_json, "request")?;
    let request = seq::SequentialRequest {
        target_population: request.target_population,
        target_period: request.target_period,
    };
    Ok(request.check(&model).err().map(|error| refusal(&error)))
}

/// Resolve what a caller offers as a policy: only a fixed contingent policy is one.
#[pyfunction]
fn sequential_resolve_policy(spec_json: &str) -> PyResult<Pair> {
    let spec: SpecIn = declaration(spec_json, "policy")?;
    let spec = match spec {
        SpecIn::Contingent { policy } => match policy_from(policy) {
            Ok(policy) => seq::PolicySpec::Contingent(policy),
            Err(error) => return Ok(refused_pair(refusal(&error))),
        },
        SpecIn::ReweightedStaticScores { scores, weights } => {
            seq::PolicySpec::ReweightedStaticScores { scores, weights }
        }
        SpecIn::LearnedFromEvaluationData { rule } => {
            seq::PolicySpec::LearnedFromEvaluationData { rule }
        }
    };
    Ok(match spec.into_contingent() {
        Ok(policy) => ok_json(&policy_json(&policy)),
        Err(error) => refused_pair(refusal(&error)),
    })
}

/// The closed interval route: always the typed refusal.
#[pyfunction]
fn sequential_request_interval() -> String {
    match seq::request_interval() {
        Err(error) => refusal(&error),
        Ok(never) => match never {},
    }
}

/// The uniform of one semantic stream identity.
#[pyfunction]
fn sequential_stream_uniform(seed: u64, rollout: u64, time: u64, role: &str) -> PyResult<f64> {
    let role = match role {
        "initial_state" => seq::StreamRole::InitialState,
        "transition" => seq::StreamRole::Transition,
        "observation" => seq::StreamRole::Observation,
        "policy_choice" => seq::StreamRole::PolicyChoice,
        other => return Err(crate::value_err(format!("unknown stream role `{other}`"))),
    };
    Ok(seq::stream_uniform(seed, rollout, time, role))
}

// ---------------------------------------------------------------------------
// Belief session
// ---------------------------------------------------------------------------

/// One realized history: the Bayes belief and the once-only evidence record.
#[pyclass(name = "SequentialBeliefSession", skip_from_py_object)]
pub(crate) struct PyBeliefSession {
    tracker: seq::BeliefTracker,
    ledger: seq::EvidenceLedger,
}

#[pymethods]
impl PyBeliefSession {
    /// Start a session at the model's prior.
    #[staticmethod]
    fn start(model_json: &str, session: &str) -> PyResult<(Option<Self>, Option<String>)> {
        let model = model_of(model_json)?;
        Ok(match seq::BeliefTracker::new(&model, session) {
            Ok(tracker) => (Some(Self { tracker, ledger: seq::EvidenceLedger::new() }), None),
            Err(error) => (None, Some(refusal(&error))),
        })
    }

    /// Record one action and the observation it produced.
    fn observe(&mut self, action: usize, symbol: usize, record: &str) -> Pair {
        match self.tracker.observe(&mut self.ledger, action, symbol, record) {
            Ok(receipt) => ok_json(&json!({
                "record_id": receipt.record_id,
                "time": receipt.time,
                "action": receipt.action,
                "symbol": receipt.symbol,
                "symbol_label": receipt.symbol_label,
                "probability": receipt.probability,
                "belief": receipt.belief,
                "measurement_identity": receipt.measurement_identity,
                "action_record": receipt.action_record,
                "observation_record": receipt.observation_record,
                "parameter_uncertainty_propagated": receipt.parameter_uncertainty_propagated,
            })),
            Err(error) => refused_pair(refusal(&error)),
        }
    }

    /// The current posterior.
    #[getter]
    fn belief(&self) -> Vec<f64> {
        self.tracker.belief().to_vec()
    }

    /// The realized history so far, as `[[action, observation], ...]` JSON.
    #[getter]
    fn history_json(&self) -> String {
        history_json(self.tracker.history()).to_string()
    }

    /// The decision time of the next action.
    #[getter]
    fn time(&self) -> usize {
        self.tracker.time()
    }

    /// Resource consumed so far.
    #[getter]
    fn resource_used(&self) -> f64 {
        self.tracker.resource_used()
    }

    /// Identity of the model the session updates under.
    #[getter]
    fn model_identity(&self) -> String {
        self.tracker.model_identity().to_owned()
    }

    /// Digest of the evidence ledger state.
    #[getter]
    fn ledger_state(&self) -> String {
        self.ledger.state_digest().to_hex()
    }

    /// Number of ledger sources (one per recorded observation).
    #[getter]
    fn recorded(&self) -> usize {
        self.ledger.sources().len()
    }
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// A sealed, replayable sequential evaluation.
#[pyclass(name = "SequentialEvaluationArtifact", skip_from_py_object)]
pub(crate) struct PyEvaluationArtifact {
    artifact: seq::SequentialEvaluationArtifact,
}

#[pymethods]
impl PyEvaluationArtifact {
    /// Evaluate and seal a policy, with an optional off-policy estimate and an
    /// optional rollout diagnostic.
    #[staticmethod]
    #[pyo3(signature = (model_json, policy_json, off_policy_json=None, rollout_json=None, *, cancel=None))]
    fn seal(
        py: Python<'_>,
        model_json: &str,
        policy_json: &str,
        off_policy_json: Option<&str>,
        rollout_json: Option<&str>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<Self>, Option<String>)> {
        let model = model_of(model_json)?;
        let wire: PolicyIn = declaration(policy_json, "sequential policy")?;
        let policy = match policy_from(wire) {
            Ok(policy) => policy,
            Err(error) => return Ok((None, Some(refusal(&error)))),
        };
        let off_policy = match off_policy_json {
            Some(text) => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct OffIn {
                    behaviour: PolicyIn,
                    trajectories: Vec<TrajectoryIn>,
                }
                let wire: OffIn = declaration(text, "off-policy inputs")?;
                let behaviour = match policy_from(wire.behaviour) {
                    Ok(policy) => policy,
                    Err(error) => return Ok((None, Some(refusal(&error)))),
                };
                let trajectories = wire
                    .trajectories
                    .into_iter()
                    .map(|t| seq::LoggedTrajectory {
                        id: t.id,
                        steps: t
                            .steps
                            .into_iter()
                            .map(|s| seq::LoggedStep {
                                action: s.action,
                                observation: s.observation,
                                assignment_probability: s.probability,
                            })
                            .collect(),
                        outcome_utility: t.outcome_utility,
                    })
                    .collect();
                Some(seq::OffPolicyInput { behaviour, trajectories })
            }
            None => None,
        };
        let rollout = match rollout_json {
            Some(text) => {
                let wire: RolloutIn = declaration(text, "rollout")?;
                Some(seq::RolloutConfig { seed: wire.seed, rollouts: wire.rollouts })
            }
            None => None,
        };
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(seq::seal(&model, &policy, off_policy.as_ref(), rollout, &ctx))
        })?;
        Ok(match outcome {
            Ok(artifact) => (Some(Self { artifact }), None),
            Err(error) => (None, Some(io_refusal(&error))),
        })
    }

    /// Consume an artifact by re-evaluation; `expected_json` holds identities the
    /// consumer retained independently of the bytes.
    #[staticmethod]
    #[pyo3(signature = (data, expected_json=None, *, cancel=None))]
    fn consume(
        py: Python<'_>,
        data: &[u8],
        expected_json: Option<&str>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<Self>, Option<String>)> {
        let expected = match expected_json {
            Some(text) => {
                let wire: ExpectedIn = declaration(text, "expected identities")?;
                Some(seq::ExpectedEvaluation {
                    model: wire.model,
                    policy: wire.policy,
                    evaluation: wire.evaluation,
                })
            }
            None => None,
        };
        let bytes = data.to_vec();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(seq::consume(&bytes, expected.as_ref(), &ctx))
        })?;
        Ok(match outcome {
            Ok(artifact) => (Some(Self { artifact }), None),
            Err(error) => (None, Some(io_refusal(&error))),
        })
    }

    /// The exact policy value.
    #[getter]
    fn value(&self) -> f64 {
        self.artifact.value()
    }

    /// The identities (model, policy, logging policy, whole evaluation), as JSON.
    #[getter]
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(self.artifact.identities()).map_err(serialization)
    }

    /// The receipts and estimates with plain numbers, as JSON.
    #[getter]
    fn summary_json(&self) -> String {
        let body = &self.artifact.body;
        let steps = |history: &[StepWire]| -> Json {
            Json::Array(history.iter().map(|s| json!([s.action, s.observation])).collect())
        };
        let histories: Vec<Json> = body
            .exact
            .histories
            .iter()
            .map(|h| {
                json!({
                    "history": steps(&h.history),
                    "time": h.time,
                    "kind": h.kind,
                    "boundary": h.boundary,
                    "reach_probability": f64::from_bits(h.reach_bits),
                    "belief": h.belief_bits.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>(),
                    "realized_utility": f64::from_bits(h.realized_bits),
                    "continuation_value": f64::from_bits(h.continuation_bits),
                    "forfeited_outcomes": h.forfeited,
                })
            })
            .collect();
        let decisions: Vec<Json> = body
            .exact
            .decisions
            .iter()
            .map(|d| {
                json!({
                    "history": steps(&d.history),
                    "time": d.time,
                    "forced_by": d.forced_by,
                    "actions": d.actions.iter()
                        .map(|a| json!([a.action, f64::from_bits(a.probability_bits)]))
                        .collect::<Vec<_>>(),
                    "resource_used": f64::from_bits(d.resource_bits),
                })
            })
            .collect();
        json!({
            "version": body.version,
            "inference": body.inference,
            "policy_version": body.policy.version,
            "value": f64::from_bits(body.exact.value_bits),
            "nodes_visited": body.exact.nodes_visited,
            "histories": histories,
            "decisions": decisions,
            "off_policy": body.off_policy.as_ref().map(|o| json!({
                "ordinary": f64::from_bits(o.ordinary_bits),
                "self_normalized": f64::from_bits(o.self_normalized_bits),
                "effective_sample_size": f64::from_bits(o.effective_sample_size_bits),
                "max_weight": f64::from_bits(o.max_weight_bits),
                "uncalibrated_standard_error": o.standard_error_bits.map(f64::from_bits),
                "trajectories": o.trajectories.len(),
                "records_counted": o.records_counted,
                "ledger_state": o.ledger_state,
            })),
            "rollout": body.rollout.as_ref().map(|r| json!({
                "seed": r.seed,
                "rollouts": r.rollouts,
                "stream_scheme": r.stream_scheme,
                "mean": f64::from_bits(r.mean_bits),
                "monte_carlo_standard_error": f64::from_bits(r.standard_error_bits),
            })),
        })
        .to_string()
    }

    /// The model declaration the artifact replays (the plain-number form), as JSON.
    #[getter]
    fn model_json(&self) -> PyResult<String> {
        let model = &self.artifact.body.model;
        let real = |bits: &[u64]| -> Vec<f64> { bits.iter().map(|b| f64::from_bits(*b)).collect() };
        let mut actions = Vec::with_capacity(model.actions.len());
        for action in &model.actions {
            let channel = match &action.channel {
                Some(wire) => {
                    let parsed = antecedent_io::measurement_model_wire::model_from_wire(wire)
                        .map_err(serialization)?;
                    channel_json(&parsed)
                }
                None => Json::Null,
            };
            actions.push(json!({
                "label": action.label,
                "role": action.role,
                "utility_cost": f64::from_bits(action.utility_cost_bits),
                "resource_use": f64::from_bits(action.resource_use_bits),
                "transition": real(&action.transition_bits),
                "channel": channel,
                "immediate_utility": real(&action.immediate_bits),
                "delayed": action.delayed.as_ref().map(|d| json!({
                    "delay": d.delay,
                    "utility": real(&d.utility_bits),
                })),
            }));
        }
        let text = json!({
            "label": model.label,
            "population": model.population,
            "regime": model.regime,
            "period": model.period,
            "state_variable": model.state_variable,
            "states": model.states,
            "initial": real(&model.initial_bits),
            "horizon": model.horizon,
            "actions": actions,
            "terminal_utility": real(&model.terminal_bits),
            "censoring": model.censoring,
            "resource_budget": model.resource_budget_bits.map(f64::from_bits),
            "stopping": model.stopping.iter().map(|b| json!({
                "label": b.label,
                "states": b.states,
                "upper": b.upper_bits.map(f64::from_bits),
                "lower": b.lower_bits.map(f64::from_bits),
                "stop_action": b.stop_action,
            })).collect::<Vec<_>>(),
            "support_default": model.support_default,
            "support_overrides": model.support_overrides.iter().map(|o| json!({
                "history": o.history.iter().map(|s| json!([s.action, s.observation])).collect::<Vec<_>>(),
                "allowed": o.allowed,
            })).collect::<Vec<_>>(),
        });
        Ok(text.to_string())
    }

    /// The stored policy declaration, as JSON.
    #[getter]
    fn policy_json(&self) -> String {
        let policy = &self.artifact.body.policy;
        json!({
            "version": policy.version,
            "decisions": policy.decisions.iter().map(|d| json!({
                "history": d.history.iter().map(|s| json!([s.action, s.observation])).collect::<Vec<_>>(),
                "actions": d.actions.iter()
                    .map(|a| json!([a.action, f64::from_bits(a.probability_bits)]))
                    .collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
        .to_string()
    }

    /// Serialize through the bounded sectioned container.
    fn export<'py>(&self, py: Python<'py>, artifact_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.artifact.to_bytes(artifact_id).map_err(serialization)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBeliefSession>()?;
    m.add_class::<PyEvaluationArtifact>()?;
    m.add_function(wrap_pyfunction!(sequential_model_check, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_off_policy, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_value_of_evidence, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_compare, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_check_request, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_resolve_policy, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_request_interval, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_stream_uniform, m)?)?;
    Ok(())
}
