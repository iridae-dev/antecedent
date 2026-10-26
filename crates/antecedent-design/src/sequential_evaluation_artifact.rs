//! `SequentialEvaluationArtifact` (`sequential_evaluation_v1`).
//!
//! A durable record of one sequential policy evaluation: the finite model
//! with the bound measurement models, the policy version, the exact
//! enumeration's value with its history and decision receipts, and, when
//! present, the off-policy estimate with the logged trajectories and the
//! logging policy, and the rollout diagnostic with its seed.
//!
//! Every real travels as its IEEE-754 bit pattern. A consumer trusts none of
//! the stored numbers: it rebuilds the model, policy, logged data and rollout
//! configuration, re-runs the exact enumeration, the importance weighting
//! (recording the log once in a fresh evidence ledger) and the rollouts, and
//! refuses unless every stored number, receipt and identity equals the
//! recomputation bit for bit. A resealed edit of a value, a receipt, a policy
//! entry or a model law is therefore refused, and a consumer that retained an
//! identity independently refuses a consistently resealed artifact of another
//! evaluation.
//!
//! The artifact carries no interval, posterior or coverage statement
//! (`point_only`); the rollout error and the importance-weighted standard
//! error are labelled uncalibrated diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_core::{CanonicalWriter, ExecutionContext, blake3_hex};
use antecedent_io_core::convert::{from_cbor, to_cbor};
use antecedent_io_core::error::IoError;
use antecedent_io_core::measurement_model_wire::{
    MeasurementModelWire, model_from_wire, model_to_wire,
};
use serde::{Deserialize, Serialize};

use crate::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, DecisionReceipt,
    DelayedOutcome, EvaluationLimits, FiniteSequentialModel, HistoryKey, HistoryReceipt,
    HistoryStep, LoggedStep, LoggedTrajectory, MAX_LOGGED_TRAJECTORIES, NodeKind,
    ObservationChannel, RolloutConfig, SEQUENTIAL_EVALUATION_INFERENCE, SequentialEvaluationError,
    StoppingBoundary, evaluate_off_policy, evaluate_policy, rollout_policy,
};

/// The artifact major version this reader writes and accepts.
pub const SEQUENTIAL_EVALUATION_ARTIFACT_VERSION: u16 = 1;
/// The artifact kind and feature marker.
pub const SEQUENTIAL_EVALUATION_ARTIFACT_FEATURE: &str = "sequential_evaluation_v1";
/// Most bytes an artifact may occupy.
pub const MAX_SEQUENTIAL_EVALUATION_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Most history receipts an artifact carries.
pub const MAX_ARTIFACT_RECEIPTS: usize = 100_000;

const ARTIFACT_KIND: &str = "sequential_evaluation_v1";
const SECTION: &str = "policy_evaluation.body";
const NOTE: &str = "sequential policy evaluation: model, policy, history and decision receipts";

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// One history cycle on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepWire {
    /// Action index.
    pub action: u64,
    /// Observation symbol.
    pub observation: u64,
}

/// One action probability on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionProbabilityWire {
    /// Action index.
    pub action: u64,
    /// Probability bits.
    pub probability_bits: u64,
}

/// One policy decision on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyEntryWire {
    /// The information set.
    pub history: Vec<StepWire>,
    /// The action distribution.
    pub actions: Vec<ActionProbabilityWire>,
}

/// A policy on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyWire {
    /// Version label.
    pub version: String,
    /// Decisions in canonical history order.
    pub decisions: Vec<PolicyEntryWire>,
}

/// A delayed outcome on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedWire {
    /// Decision times of delay.
    pub delay: u64,
    /// Utility bits by state.
    pub utility_bits: Vec<u64>,
}

/// An action on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionWire {
    /// Label.
    pub label: String,
    /// `continue` or `commit`.
    pub role: String,
    /// Cost bits.
    pub utility_cost_bits: u64,
    /// Resource-use bits.
    pub resource_use_bits: u64,
    /// Row-major transition bits.
    pub transition_bits: Vec<u64>,
    /// The bound measurement model.
    pub channel: Option<MeasurementModelWire>,
    /// Immediate utility bits by state.
    pub immediate_bits: Vec<u64>,
    /// Delayed outcome.
    pub delayed: Option<DelayedWire>,
}

/// A stopping boundary on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryWire {
    /// Label.
    pub label: String,
    /// Monitored states.
    pub states: Vec<u64>,
    /// Upper threshold bits.
    pub upper_bits: Option<u64>,
    /// Lower threshold bits.
    pub lower_bits: Option<u64>,
    /// Commit action taken on crossing.
    pub stop_action: u64,
}

/// A per-history support override on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportOverrideWire {
    /// History.
    pub history: Vec<StepWire>,
    /// Support per action.
    pub allowed: Vec<bool>,
}

/// The model on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelWire {
    /// Display label.
    pub label: String,
    /// Population.
    pub population: String,
    /// Regime.
    pub regime: String,
    /// Period.
    pub period: String,
    /// State variable identity.
    pub state_variable: String,
    /// State names.
    pub states: Vec<String>,
    /// Initial law bits.
    pub initial_bits: Vec<u64>,
    /// Horizon.
    pub horizon: u64,
    /// Actions.
    pub actions: Vec<ActionWire>,
    /// Terminal utility bits.
    pub terminal_bits: Vec<u64>,
    /// `refuse` or `forfeit`.
    pub censoring: String,
    /// Budget bits.
    pub resource_budget_bits: Option<u64>,
    /// Boundaries.
    pub stopping: Vec<BoundaryWire>,
    /// Default support.
    pub support_default: Vec<bool>,
    /// Support overrides.
    pub support_overrides: Vec<SupportOverrideWire>,
}

/// A history receipt on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryReceiptWire {
    /// History.
    pub history: Vec<StepWire>,
    /// Decision time.
    pub time: u64,
    /// `decision`, `forced_stop` or `horizon`.
    pub kind: String,
    /// Boundary index of a forced stop.
    pub boundary: Option<u64>,
    /// Reach probability bits.
    pub reach_bits: u64,
    /// Belief bits.
    pub belief_bits: Vec<u64>,
    /// Realized utility bits.
    pub realized_bits: u64,
    /// Continuation value bits.
    pub continuation_bits: u64,
    /// Forfeited outcomes.
    pub forfeited: u64,
}

/// A decision receipt on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionReceiptWire {
    /// History.
    pub history: Vec<StepWire>,
    /// Decision time.
    pub time: u64,
    /// Forcing boundary.
    pub forced_by: Option<u64>,
    /// Action distribution.
    pub actions: Vec<ActionProbabilityWire>,
    /// Resource-used bits.
    pub resource_bits: u64,
}

/// The exact enumeration on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactWire {
    /// Value bits.
    pub value_bits: u64,
    /// Histories enumerated.
    pub nodes_visited: u64,
    /// Accounting residual bits.
    pub accounting_residual_bits: u64,
    /// History receipts.
    pub histories: Vec<HistoryReceiptWire>,
    /// Decision receipts.
    pub decisions: Vec<DecisionReceiptWire>,
    /// Measurement-model identities by action.
    pub measurement_identities: Vec<(u64, String)>,
}

/// One logged step on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggedStepWire {
    /// Action index.
    pub action: u64,
    /// Observation symbol.
    pub observation: Option<u64>,
    /// Assignment probability bits.
    pub probability_bits: u64,
}

/// One logged trajectory on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryWire {
    /// Identity.
    pub id: String,
    /// Steps.
    pub steps: Vec<LoggedStepWire>,
    /// Outcome utility bits.
    pub outcome_bits: u64,
}

/// The off-policy estimate and its inputs on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OffPolicyWire {
    /// The logging policy.
    pub behaviour: PolicyWire,
    /// The logged trajectories.
    pub trajectories: Vec<TrajectoryWire>,
    /// Ordinary estimate bits.
    pub ordinary_bits: u64,
    /// Self-normalized estimate bits.
    pub self_normalized_bits: u64,
    /// Effective sample size bits.
    pub effective_sample_size_bits: u64,
    /// Largest weight bits.
    pub max_weight_bits: u64,
    /// Uncalibrated standard error bits.
    pub standard_error_bits: Option<u64>,
    /// Histories checked for positivity.
    pub positivity_histories_checked: u64,
    /// Records counted once.
    pub records_counted: u64,
    /// Ledger state digest after recording.
    pub ledger_state: String,
}

/// The rollout diagnostic on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutWire {
    /// Master seed.
    pub seed: u64,
    /// Rollouts.
    pub rollouts: u64,
    /// Stream scheme.
    pub stream_scheme: String,
    /// Mean bits.
    pub mean_bits: u64,
    /// Standard error bits.
    pub standard_error_bits: u64,
}

/// The identities of one evaluation, as lowercase hex.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationIdentities {
    /// Model identity.
    pub model: String,
    /// Policy identity.
    pub policy: String,
    /// Logging policy identity, when an off-policy estimate is carried.
    pub behaviour: Option<String>,
    /// Identity of the whole evaluation.
    pub evaluation: String,
}

/// The sealed body.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialEvaluationBody {
    /// Body schema version.
    pub version: u32,
    /// The only claim carried: `point_only`.
    pub inference: String,
    /// The model.
    pub model: ModelWire,
    /// The evaluated policy.
    pub policy: PolicyWire,
    /// The exact enumeration.
    pub exact: ExactWire,
    /// The off-policy estimate, when carried.
    pub off_policy: Option<OffPolicyWire>,
    /// The rollout diagnostic, when carried.
    pub rollout: Option<RolloutWire>,
    /// Identities.
    pub identities: EvaluationIdentities,
    /// Digest of the body with this field blank.
    pub digest: String,
}

/// Identities a consumer retained independently of the bytes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpectedEvaluation {
    /// Expected model identity.
    pub model: Option<String>,
    /// Expected policy identity.
    pub policy: Option<String>,
    /// Expected identity of the whole evaluation.
    pub evaluation: Option<String>,
}

/// Off-policy inputs to carry in an artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct OffPolicyInput {
    /// The logging policy.
    pub behaviour: ContingentPolicy,
    /// The logged trajectories.
    pub trajectories: Vec<LoggedTrajectory>,
}

/// A sealed, replayable sequential evaluation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SequentialEvaluationArtifact {
    /// The sealed body.
    pub body: SequentialEvaluationBody,
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

fn refused(detail: &'static str, text: impl AsRef<str>) -> IoError {
    crate::plan_common::refused_with(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        text.as_ref(),
    )
}

fn identity_refused(field: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("route_not_supported"),
        message: format!("sequential_evaluation.identity_mismatch: {field} changed"),
    }
}

fn evaluation_refused(error: &SequentialEvaluationError) -> IoError {
    IoError::Refused { code: error.code, message: format!("{}: {}", error.detail, error.message) }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reals(values: &[u64]) -> Vec<f64> {
    values.iter().map(|b| f64::from_bits(*b)).collect()
}

fn index(value: u64, what: &str) -> Result<usize, IoError> {
    usize::try_from(value)
        .map_err(|_| refused("sequential_evaluation.malformed", format!("{what} is out of range")))
}

fn steps_to_wire(history: &HistoryKey) -> Vec<StepWire> {
    history
        .steps()
        .iter()
        .map(|step| StepWire { action: step.action as u64, observation: step.observation as u64 })
        .collect()
}

fn steps_from_wire(steps: &[StepWire]) -> Result<HistoryKey, IoError> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        out.push(HistoryStep {
            action: index(step.action, "a history action")?,
            observation: index(step.observation, "a history observation")?,
        });
    }
    Ok(HistoryKey::from_steps(out))
}

fn actions_to_wire(actions: &[(usize, f64)]) -> Vec<ActionProbabilityWire> {
    actions
        .iter()
        .map(|(action, p)| ActionProbabilityWire {
            action: *action as u64,
            probability_bits: p.to_bits(),
        })
        .collect()
}

fn policy_to_wire(policy: &ContingentPolicy) -> PolicyWire {
    PolicyWire {
        version: policy.version().to_owned(),
        decisions: policy
            .entries()
            .map(|(history, actions)| PolicyEntryWire {
                history: steps_to_wire(history),
                actions: actions_to_wire(actions),
            })
            .collect(),
    }
}

fn policy_from_wire(wire: &PolicyWire) -> Result<ContingentPolicy, IoError> {
    let mut policy = ContingentPolicy::new(wire.version.clone());
    for entry in &wire.decisions {
        let history = steps_from_wire(&entry.history)?;
        let mut distribution = Vec::with_capacity(entry.actions.len());
        for item in &entry.actions {
            distribution.push((
                index(item.action, "a policy action")?,
                f64::from_bits(item.probability_bits),
            ));
        }
        policy = policy
            .decide_stochastic(history, distribution)
            .map_err(|error| evaluation_refused(&error))?;
    }
    Ok(policy)
}

fn model_to_wire_body(model: &FiniteSequentialModel) -> ModelWire {
    ModelWire {
        label: model.label.clone(),
        population: model.population.clone(),
        regime: model.regime.clone(),
        period: model.period.clone(),
        state_variable: model.state_variable.clone(),
        states: model.states.clone(),
        initial_bits: bits(&model.initial),
        horizon: model.horizon as u64,
        actions: model
            .actions
            .iter()
            .map(|action| ActionWire {
                label: action.label.clone(),
                role: action.role.as_str().to_owned(),
                utility_cost_bits: action.utility_cost.to_bits(),
                resource_use_bits: action.resource_use.to_bits(),
                transition_bits: bits(&action.transition),
                channel: action.channel.as_ref().map(|c| model_to_wire(&c.measurement)),
                immediate_bits: bits(&action.immediate_utility),
                delayed: action
                    .delayed
                    .as_ref()
                    .map(|d| DelayedWire { delay: d.delay as u64, utility_bits: bits(&d.utility) }),
            })
            .collect(),
        terminal_bits: bits(&model.terminal_utility),
        censoring: model.censoring.as_str().to_owned(),
        resource_budget_bits: model.resource_budget.map(f64::to_bits),
        stopping: model
            .stopping
            .iter()
            .map(|boundary| BoundaryWire {
                label: boundary.label.clone(),
                states: boundary.states.iter().map(|s| *s as u64).collect(),
                upper_bits: boundary.upper.map(f64::to_bits),
                lower_bits: boundary.lower.map(f64::to_bits),
                stop_action: boundary.stop_action as u64,
            })
            .collect(),
        support_default: model.support.default_allowed().to_vec(),
        support_overrides: model
            .support
            .overrides()
            .map(|(history, allowed)| SupportOverrideWire {
                history: steps_to_wire(history),
                allowed: allowed.to_vec(),
            })
            .collect(),
    }
}

fn model_from_wire_body(wire: &ModelWire) -> Result<FiniteSequentialModel, IoError> {
    let mut actions = Vec::with_capacity(wire.actions.len());
    for action in &wire.actions {
        let role = ActionRole::from_name(&action.role)
            .ok_or_else(|| refused("sequential_evaluation.malformed", "unknown action role"))?;
        let channel = match &action.channel {
            Some(measurement) => {
                Some(ObservationChannel { measurement: model_from_wire(measurement)? })
            }
            None => None,
        };
        let delayed = match &action.delayed {
            Some(delayed) => Some(DelayedOutcome {
                delay: index(delayed.delay, "a delay")?,
                utility: reals(&delayed.utility_bits),
            }),
            None => None,
        };
        actions.push(ActionSpec {
            label: action.label.clone(),
            role,
            utility_cost: f64::from_bits(action.utility_cost_bits),
            resource_use: f64::from_bits(action.resource_use_bits),
            transition: reals(&action.transition_bits),
            channel,
            immediate_utility: reals(&action.immediate_bits),
            delayed,
        });
    }
    let mut stopping = Vec::with_capacity(wire.stopping.len());
    for boundary in &wire.stopping {
        let mut states = Vec::with_capacity(boundary.states.len());
        for state in &boundary.states {
            states.push(index(*state, "a boundary state")?);
        }
        stopping.push(StoppingBoundary {
            label: boundary.label.clone(),
            states,
            upper: boundary.upper_bits.map(f64::from_bits),
            lower: boundary.lower_bits.map(f64::from_bits),
            stop_action: index(boundary.stop_action, "a stop action")?,
        });
    }
    let mut support = ActionSupport::by_action(wire.support_default.clone());
    for item in &wire.support_overrides {
        support = support.with_history(steps_from_wire(&item.history)?, item.allowed.clone());
    }
    Ok(FiniteSequentialModel {
        label: wire.label.clone(),
        population: wire.population.clone(),
        regime: wire.regime.clone(),
        period: wire.period.clone(),
        state_variable: wire.state_variable.clone(),
        states: wire.states.clone(),
        initial: reals(&wire.initial_bits),
        horizon: index(wire.horizon, "the horizon")?,
        actions,
        terminal_utility: reals(&wire.terminal_bits),
        censoring: CensoringRule::from_name(&wire.censoring)
            .ok_or_else(|| refused("sequential_evaluation.malformed", "unknown censoring rule"))?,
        resource_budget: wire.resource_budget_bits.map(f64::from_bits),
        stopping,
        support,
    })
}

fn trajectories_to_wire(log: &[LoggedTrajectory]) -> Vec<TrajectoryWire> {
    log.iter()
        .map(|trajectory| TrajectoryWire {
            id: trajectory.id.clone(),
            steps: trajectory
                .steps
                .iter()
                .map(|step| LoggedStepWire {
                    action: step.action as u64,
                    observation: step.observation.map(|o| o as u64),
                    probability_bits: step.assignment_probability.to_bits(),
                })
                .collect(),
            outcome_bits: trajectory.outcome_utility.to_bits(),
        })
        .collect()
}

fn trajectories_from_wire(wire: &[TrajectoryWire]) -> Result<Vec<LoggedTrajectory>, IoError> {
    let mut out = Vec::with_capacity(wire.len());
    for trajectory in wire {
        let mut steps = Vec::with_capacity(trajectory.steps.len());
        for step in &trajectory.steps {
            steps.push(LoggedStep {
                action: index(step.action, "a logged action")?,
                observation: step
                    .observation
                    .map(|o| index(o, "a logged observation"))
                    .transpose()?,
                assignment_probability: f64::from_bits(step.probability_bits),
            });
        }
        out.push(LoggedTrajectory {
            id: trajectory.id.clone(),
            steps,
            outcome_utility: f64::from_bits(trajectory.outcome_bits),
        });
    }
    Ok(out)
}

fn kind_to_wire(kind: NodeKind) -> (String, Option<u64>) {
    match kind {
        NodeKind::ForcedStop { boundary } => (kind.as_str().to_owned(), Some(boundary as u64)),
        other => (other.as_str().to_owned(), None),
    }
}

fn history_receipt_wire(receipt: &HistoryReceipt) -> HistoryReceiptWire {
    let (kind, boundary) = kind_to_wire(receipt.kind);
    HistoryReceiptWire {
        history: steps_to_wire(&receipt.history),
        time: receipt.time as u64,
        kind,
        boundary,
        reach_bits: receipt.reach_probability.to_bits(),
        belief_bits: bits(&receipt.belief),
        realized_bits: receipt.realized_utility.to_bits(),
        continuation_bits: receipt.continuation_value.to_bits(),
        forfeited: receipt.forfeited_outcomes as u64,
    }
}

fn decision_receipt_wire(receipt: &DecisionReceipt) -> DecisionReceiptWire {
    DecisionReceiptWire {
        history: steps_to_wire(&receipt.history),
        time: receipt.time as u64,
        forced_by: receipt.forced_by.map(|b| b as u64),
        actions: actions_to_wire(&receipt.actions),
        resource_bits: receipt.resource_used.to_bits(),
    }
}

fn evaluation_identity(body: &SequentialEvaluationBody) -> String {
    let mut writer = CanonicalWriter::new("antecedent.sequential_evaluation.evaluation");
    writer
        .word(u64::from(body.version))
        .text(&body.identities.model)
        .text(&body.identities.policy)
        .opt_text(body.identities.behaviour.as_deref())
        .word(body.exact.value_bits)
        .word(body.exact.nodes_visited)
        .word(body.exact.histories.len() as u64)
        .word(body.exact.decisions.len() as u64);
    for receipt in &body.exact.histories {
        writer.word(receipt.reach_bits).word(receipt.realized_bits).word(receipt.continuation_bits);
    }
    match &body.off_policy {
        Some(off) => {
            writer
                .word(1)
                .word(off.ordinary_bits)
                .word(off.self_normalized_bits)
                .word(off.effective_sample_size_bits)
                .word(off.max_weight_bits)
                .opt_word(off.standard_error_bits)
                .text(&off.ledger_state);
        }
        None => {
            writer.word(0);
        }
    }
    match &body.rollout {
        Some(rollout) => {
            writer
                .word(1)
                .word(rollout.seed)
                .word(rollout.rollouts)
                .word(rollout.mean_bits)
                .word(rollout.standard_error_bits);
        }
        None => {
            writer.word(0);
        }
    }
    writer.finish_hex()
}

// ---------------------------------------------------------------------------
// Building and verifying
// ---------------------------------------------------------------------------

fn build_body(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    off_policy: Option<&OffPolicyInput>,
    rollout: Option<RolloutConfig>,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<SequentialEvaluationBody, SequentialEvaluationError> {
    let exact = evaluate_policy(model, policy, limits, ctx)?;
    let off = match off_policy {
        Some(input) => {
            let mut ledger = EvidenceLedger::new();
            let estimate = evaluate_off_policy(
                model,
                policy,
                &input.behaviour,
                &input.trajectories,
                &mut ledger,
                limits,
                ctx,
            )?;
            Some(OffPolicyWire {
                behaviour: policy_to_wire(&input.behaviour),
                trajectories: trajectories_to_wire(&input.trajectories),
                ordinary_bits: estimate.ordinary.to_bits(),
                self_normalized_bits: estimate.self_normalized.to_bits(),
                effective_sample_size_bits: estimate.effective_sample_size.to_bits(),
                max_weight_bits: estimate.max_weight.to_bits(),
                standard_error_bits: estimate.uncalibrated_standard_error.map(f64::to_bits),
                positivity_histories_checked: estimate.positivity_histories_checked,
                records_counted: estimate.records_counted,
                ledger_state: estimate.ledger_state,
            })
        }
        None => None,
    };
    let rolled = match rollout {
        Some(config) => {
            let estimate = rollout_policy(model, policy, config, ctx)?;
            Some(RolloutWire {
                seed: estimate.seed,
                rollouts: estimate.rollouts,
                stream_scheme: estimate.stream_scheme.to_owned(),
                mean_bits: estimate.mean.to_bits(),
                standard_error_bits: estimate.monte_carlo_standard_error.to_bits(),
            })
        }
        None => None,
    };
    let mut body = SequentialEvaluationBody {
        version: u32::from(SEQUENTIAL_EVALUATION_ARTIFACT_VERSION),
        inference: SEQUENTIAL_EVALUATION_INFERENCE.to_owned(),
        model: model_to_wire_body(model),
        policy: policy_to_wire(policy),
        exact: ExactWire {
            value_bits: exact.value.to_bits(),
            nodes_visited: exact.nodes_visited,
            accounting_residual_bits: exact.accounting_residual.to_bits(),
            histories: exact.histories.iter().map(history_receipt_wire).collect(),
            decisions: exact.decisions.iter().map(decision_receipt_wire).collect(),
            measurement_identities: exact
                .measurement_identities
                .iter()
                .map(|(action, identity)| (*action as u64, identity.clone()))
                .collect(),
        },
        off_policy: off,
        rollout: rolled,
        identities: EvaluationIdentities {
            model: exact.model_identity.clone(),
            policy: exact.policy_identity.clone(),
            behaviour: off_policy.map(|input| input.behaviour.identity_hex()),
            evaluation: String::new(),
        },
        digest: String::new(),
    };
    body.identities.evaluation = evaluation_identity(&body);
    Ok(body)
}

impl SequentialEvaluationArtifact {
    /// Evaluate and seal a policy, with an optional off-policy estimate from
    /// logged trajectories and an optional rollout diagnostic.
    ///
    /// # Errors
    /// Any evaluation refusal, a body over the receipt bounds, or an
    /// encoding failure.
    pub fn new(
        model: &FiniteSequentialModel,
        policy: &ContingentPolicy,
        off_policy: Option<&OffPolicyInput>,
        rollout: Option<RolloutConfig>,
        limits: EvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let body = build_body(model, policy, off_policy, rollout, limits, ctx)
            .map_err(|error| evaluation_refused(&error))?;
        let mut artifact = Self { body };
        artifact.check_bounds()?;
        artifact.seal()?;
        Ok(artifact)
    }

    /// Identities of the sealed evaluation.
    #[must_use]
    pub fn identities(&self) -> &EvaluationIdentities {
        &self.body.identities
    }

    /// The exact value, from its stored bit pattern.
    #[must_use]
    pub fn value(&self) -> f64 {
        f64::from_bits(self.body.exact.value_bits)
    }

    fn check_bounds(&self) -> Result<(), IoError> {
        let too_many_trajectories = self
            .body
            .off_policy
            .as_ref()
            .is_some_and(|off| off.trajectories.len() > MAX_LOGGED_TRAJECTORIES);
        if self.body.exact.histories.len() > MAX_ARTIFACT_RECEIPTS
            || self.body.exact.decisions.len() > MAX_ARTIFACT_RECEIPTS
            || self.body.policy.decisions.len() > crate::sequential_evaluation::MAX_POLICY_ENTRIES
            || too_many_trajectories
        {
            return Err(IoError::TooLarge);
        }
        Ok(())
    }

    /// Canonical digest of the body with the digest field blank.
    ///
    /// # Errors
    /// Wire encoding failure.
    pub fn recomputed_digest(&self) -> Result<String, IoError> {
        let mut body = self.body.clone();
        body.digest = String::new();
        Ok(blake3_hex(&to_cbor(&body)?))
    }

    /// Seal the body. Sealing alone does not establish its validity.
    ///
    /// # Errors
    /// Wire encoding failure.
    pub fn seal(&mut self) -> Result<(), IoError> {
        self.body.digest = self.recomputed_digest()?;
        Ok(())
    }

    /// Re-run the whole evaluation from the stored declarations and require
    /// every stored number, receipt and identity to match bit for bit.
    ///
    /// # Errors
    /// A wrong version or digest, a changed retained identity, an evaluation
    /// refusal on replay, or any stored value that does not replay.
    pub fn verify(
        &self,
        expected: Option<&ExpectedEvaluation>,
        ctx: &ExecutionContext,
    ) -> Result<(), IoError> {
        let body = &self.body;
        if body.version != u32::from(SEQUENTIAL_EVALUATION_ARTIFACT_VERSION) {
            return Err(IoError::UnsupportedVersion { version: body.version });
        }
        if body.inference != SEQUENTIAL_EVALUATION_INFERENCE {
            return Err(refused("sequential_evaluation.version", "unsupported inference claim"));
        }
        if body.digest != self.recomputed_digest()? {
            return Err(identity_refused("body digest"));
        }
        self.check_bounds()?;
        if let Some(expected) = expected {
            for (field, wanted, stored) in [
                ("model identity", &expected.model, &body.identities.model),
                ("policy identity", &expected.policy, &body.identities.policy),
                ("evaluation identity", &expected.evaluation, &body.identities.evaluation),
            ] {
                if wanted.as_ref().is_some_and(|wanted| wanted != stored) {
                    return Err(identity_refused(field));
                }
            }
        }
        let model = model_from_wire_body(&body.model)?;
        let policy = policy_from_wire(&body.policy)?;
        let off_policy = match &body.off_policy {
            Some(off) => Some(OffPolicyInput {
                behaviour: policy_from_wire(&off.behaviour)?,
                trajectories: trajectories_from_wire(&off.trajectories)?,
            }),
            None => None,
        };
        let rollout = body
            .rollout
            .as_ref()
            .map(|rollout| RolloutConfig { seed: rollout.seed, rollouts: rollout.rollouts });
        let fresh = build_body(
            &model,
            &policy,
            off_policy.as_ref(),
            rollout,
            EvaluationLimits::default(),
            ctx,
        )
        .map_err(|error| evaluation_refused(&error))?;
        let differs = |detail: &'static str, what: &str| {
            Err(refused(detail, format!("the stored {what} does not replay")))
        };
        if fresh.model != body.model {
            return differs("sequential_evaluation.model_mismatch", "model");
        }
        if fresh.policy != body.policy {
            return differs("sequential_evaluation.policy_mismatch", "policy");
        }
        if fresh.exact.value_bits != body.exact.value_bits {
            return differs("sequential_evaluation.value_mismatch", "policy value");
        }
        if fresh.exact.histories != body.exact.histories {
            return differs("sequential_evaluation.history_receipt_mismatch", "history receipts");
        }
        if fresh.exact.decisions != body.exact.decisions {
            return differs("sequential_evaluation.decision_receipt_mismatch", "decision receipts");
        }
        if fresh.exact != body.exact {
            return differs("sequential_evaluation.exact_mismatch", "enumeration");
        }
        if fresh.off_policy != body.off_policy {
            return differs("sequential_evaluation.off_policy_mismatch", "off-policy estimate");
        }
        if fresh.rollout != body.rollout {
            return differs("sequential_evaluation.rollout_mismatch", "rollout diagnostic");
        }
        if fresh.identities != body.identities {
            return Err(identity_refused("identities"));
        }
        Ok(())
    }

    /// Write a validated bounded checksummed container.
    ///
    /// # Errors
    /// An empty id, an invalid body, or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        if artifact_id.trim().is_empty() {
            return Err(refused(
                "sequential_evaluation.artifact_id_missing",
                "an artifact needs a non-blank id",
            ));
        }
        self.check_bounds()?;
        if self.body.digest != self.recomputed_digest()? {
            return Err(identity_refused("body digest"));
        }
        let body = to_cbor(self)?;
        antecedent_io_core::encode_container(
            &antecedent_io_core::ArtifactContainer {
                kind: ARTIFACT_KIND,
                note: NOTE,
                max_bytes: MAX_SEQUENTIAL_EVALUATION_ARTIFACT_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[antecedent_io_core::OutSection {
                id: SECTION,
                content_type: antecedent_io_core::CONTENT_CBOR,
                bytes: &body,
            }],
        )
        .map_err(|error| match error {
            antecedent_io_core::CodecError::Layout => refused(
                "sequential_evaluation.artifact_kind_mismatch",
                "unsupported container layout",
            ),
            other => other.into_io("sequential evaluation artifact"),
        })
    }

    /// Decode a bounded container; verification remains explicit.
    ///
    /// # Errors
    /// Corruption, another kind, version or byte limits refuse before replay.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IoError> {
        let parts = antecedent_io_core::decode_container(
            ARTIFACT_KIND,
            MAX_SEQUENTIAL_EVALUATION_ARTIFACT_BYTES,
            &[SECTION],
            bytes,
        )
        .map_err(|error| match error {
            antecedent_io_core::CodecError::Layout => refused(
                "sequential_evaluation.artifact_kind_mismatch",
                "unsupported container layout",
            ),
            other => other.into_io("sequential evaluation artifact"),
        })?;
        let Ok([body]) = <[Vec<u8>; 1]>::try_from(parts) else {
            return Err(refused(
                "sequential_evaluation.artifact_kind_mismatch",
                "unexpected section count",
            ));
        };
        let artifact: Self = from_cbor(&body)?;
        artifact.check_bounds()?;
        Ok(artifact)
    }
}

/// Decode and independently re-evaluate an artifact.
///
/// `expected` holds identities the consumer retained independently of the
/// bytes.
///
/// # Errors
/// Corruption, a changed retained identity, an evaluation refusal on replay,
/// or any stored number that does not replay bit for bit.
pub fn consume(
    bytes: &[u8],
    expected: Option<&ExpectedEvaluation>,
    ctx: &ExecutionContext,
) -> Result<SequentialEvaluationArtifact, IoError> {
    let artifact = SequentialEvaluationArtifact::from_bytes(bytes)?;
    artifact.verify(expected, ctx)?;
    Ok(artifact)
}
