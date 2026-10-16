//! The sequential g-formula: exact enumeration of the histories a fixed
//! policy reaches in a dynamic finite model.
//!
//! The value of a policy `pi` is
//!
//! ```text
//! V(pi) = sum over histories h reachable under pi of  P(h | pi) * realized(h)
//! ```
//!
//! where the history law factorizes through the declared interventional laws
//! (transition and observation per action) and `pi`'s own action
//! probabilities, and `realized(h)` is the utility realized at the history:
//! outcomes falling due, immediate utility, action cost and, at a commit, the
//! committed action's delayed outcome and those still pending, evolved under
//! the commit's transition. Every quantity conditional on a history is
//! computed from the Bayes belief given that history alone, so a decision can
//! depend on nothing but the information set.
//!
//! Identification premises, declared and not testable from the data:
//!
//! * exchangeability: the per-action laws are interventional laws given the
//!   information set;
//! * consistency: the realized outcome under an action is the model's
//!   outcome for that action;
//! * positivity: every action the policy takes in a reachable history is
//!   supported there ([`super::ActionSupport`]), checked on every reachable
//!   history.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use super::belief::{branches, evolve};
use super::model::{ActionRole, CensoringRule, Compiled, dot};
use super::policy::{ContingentPolicy, HistoryKey};
use super::{
    FiniteSequentialModel, SEQUENTIAL_EVALUATION_INFERENCE, SequentialResult, cancelled,
    limit_exceeded,
};

/// Bounds on one enumeration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluationLimits {
    /// Most histories (nodes) the enumeration may visit.
    pub max_nodes: u64,
    /// Most history receipts to retain.
    pub max_receipts: usize,
}

impl Default for EvaluationLimits {
    fn default() -> Self {
        Self { max_nodes: 2_000_000, max_receipts: 100_000 }
    }
}

/// How a history node ended its cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NodeKind {
    /// The policy was consulted.
    Decision,
    /// A declared stopping boundary was crossed and forced a commit.
    ForcedStop {
        /// Index of the boundary.
        boundary: usize,
    },
    /// The horizon was reached without a commit.
    Horizon,
}

impl NodeKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::ForcedStop { .. } => "forced_stop",
            Self::Horizon => "horizon",
        }
    }
}

/// The receipt of one reachable history.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryReceipt {
    /// The observed history.
    pub history: HistoryKey,
    /// Decision time (the number of completed cycles).
    pub time: usize,
    /// How the node ended its cycle.
    pub kind: NodeKind,
    /// Probability of reaching the history under the policy.
    pub reach_probability: f64,
    /// Posterior over the latent states given the history.
    pub belief: Vec<f64>,
    /// Expected utility realized at this node (excluding successors).
    pub realized_utility: f64,
    /// Expected utility from this node onward, given the history.
    pub continuation_value: f64,
    /// Outcomes still pending at the horizon and forfeited here.
    pub forfeited_outcomes: usize,
}

/// The receipt of one decision (a policy or forced choice).
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionReceipt {
    /// The information set the decision was made in.
    pub history: HistoryKey,
    /// Decision time.
    pub time: usize,
    /// Boundary that forced the decision; `None` when the policy chose.
    pub forced_by: Option<usize>,
    /// Action probabilities taken, by action index.
    pub actions: Vec<(usize, f64)>,
    /// Resource consumed before this decision.
    pub resource_used: f64,
}

/// The named identification premises a value is conditional on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequentialAssumptions {
    /// Exchangeability premise.
    pub exchangeability: &'static str,
    /// Consistency premise.
    pub consistency: &'static str,
    /// Positivity premise.
    pub positivity: &'static str,
}

impl SequentialAssumptions {
    /// The premises every value of this module is conditional on.
    pub const DECLARED: Self = Self {
        exchangeability: "the declared per-action transition and observation laws are the interventional laws of the dynamic model given the information set; no unmeasured cause of both an action choice and later states",
        consistency: "the outcome realized under an action in the data-generating process is the declared outcome of that action in the model",
        positivity: "every action the policy takes with positive probability in a reachable history is supported there by the declared support evidence",
    };
}

/// The result of evaluating a fixed policy by exact enumeration.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyEvaluation {
    /// Expected total utility of the policy.
    pub value: f64,
    /// Policy version label.
    pub policy_version: String,
    /// Policy identity (hex).
    pub policy_identity: String,
    /// Model identity (hex).
    pub model_identity: String,
    /// Every reachable history, in canonical pre-order.
    pub histories: Vec<HistoryReceipt>,
    /// Every decision, in the same order.
    pub decisions: Vec<DecisionReceipt>,
    /// Histories enumerated.
    pub nodes_visited: u64,
    /// `|sum reach * realized - value|`: an internal accounting check.
    pub accounting_residual: f64,
    /// Measurement-model identities of the channels used, by action index.
    pub measurement_identities: Vec<(usize, String)>,
    /// The premises the value is conditional on.
    pub assumptions: SequentialAssumptions,
    /// The inferential claim: always `point_only`.
    pub inference: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    action: usize,
    due: usize,
}

struct Node<'n> {
    history: &'n HistoryKey,
    belief: &'n [f64],
    pending: &'n [Pending],
    reach: f64,
    resource: f64,
}

struct Walker<'a> {
    compiled: &'a Compiled,
    policy: &'a ContingentPolicy,
    behaviour: Option<&'a ContingentPolicy>,
    ctx: &'a ExecutionContext,
    limits: EvaluationLimits,
    nodes: u64,
    histories: Vec<HistoryReceipt>,
    decisions: Vec<DecisionReceipt>,
    accounting: f64,
}

fn delayed_utility(compiled: &Compiled, action: usize) -> &[f64] {
    match &compiled.actions[action].delayed {
        Some((_, utility)) => utility.as_slice(),
        None => &[],
    }
}

/// Outcomes realized after a commit: the committed action's own delayed
/// outcome and every pending one, with the latent state evolving under the
/// commit action's transition and no further observation.
pub(crate) fn commit_tail(
    compiled: &Compiled,
    action: usize,
    belief: &[f64],
    time: usize,
    pending: &[(usize, usize)],
) -> f64 {
    let spec = &compiled.actions[action];
    let mut items: Vec<(usize, &[f64])> = Vec::new();
    if let Some((delay, utility)) = &spec.delayed {
        items.push((*delay, utility.as_slice()));
    }
    for &(pending_action, due) in pending {
        items.push((due - time, delayed_utility(compiled, pending_action)));
    }
    let longest = items.iter().map(|(steps, _)| *steps).max().unwrap_or(0);
    let mut tail = 0.0;
    let mut current = belief.to_vec();
    for step in 1..=longest {
        current = evolve(spec, &current, compiled.states);
        for (steps, utility) in &items {
            if *steps == step {
                tail += dot(&current, utility);
            }
        }
    }
    tail
}

impl Walker<'_> {
    fn incomplete(history: &HistoryKey) -> super::SequentialEvaluationError {
        refusal!(
            "invalid_argument",
            "sequential_evaluation.policy_incomplete",
            format!("the policy declares no decision for the reachable history {history}"),
            "declare an action for every history the policy can reach; a missing history is not given a default",
        )
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one flat pass over a node: due credits, boundary, then each supported action"
    )]
    fn visit(&mut self, node: &Node<'_>) -> SequentialResult<f64> {
        self.nodes += 1;
        if self.nodes > self.limits.max_nodes {
            return Err(limit_exceeded(
                "the number of enumerated histories",
                self.limits.max_nodes,
            ));
        }
        if (self.nodes - 1) % 64 == 0 && self.ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        if self.histories.len() >= self.limits.max_receipts {
            return Err(limit_exceeded(
                "the number of history receipts",
                self.limits.max_receipts as u64,
            ));
        }
        let compiled = self.compiled;
        let time = node.history.len();
        let mut due_now = 0.0;
        let mut remaining: Vec<Pending> = Vec::new();
        for pending in node.pending {
            if pending.due == time {
                due_now += dot(node.belief, delayed_utility(compiled, pending.action));
            } else {
                remaining.push(*pending);
            }
        }
        let slot = self.histories.len();
        if time == compiled.horizon {
            if !remaining.is_empty() && compiled.censoring == CensoringRule::Refuse {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.delayed_outcome_beyond_horizon",
                    format!(
                        "{} outcome(s) fall due after the horizon in history {}; the policy's value is not defined without them",
                        remaining.len(),
                        node.history
                    ),
                    "extend the horizon, or declare the censoring rule `forfeit` to count them as zero and report them",
                ));
            }
            let realized = due_now + dot(node.belief, &compiled.terminal);
            self.accounting += node.reach * realized;
            self.histories.push(HistoryReceipt {
                history: node.history.clone(),
                time,
                kind: NodeKind::Horizon,
                reach_probability: node.reach,
                belief: node.belief.to_vec(),
                realized_utility: realized,
                continuation_value: realized,
                forfeited_outcomes: remaining.len(),
            });
            return Ok(realized);
        }
        let forced = compiled.crossed_boundary(node.belief);
        let distribution: Vec<(usize, f64)> = match forced {
            Some(boundary) => vec![(compiled.boundaries[boundary].stop_action, 1.0)],
            None => self
                .policy
                .distribution(node.history)
                .ok_or_else(|| Self::incomplete(node.history))?
                .to_vec(),
        };
        self.histories.push(HistoryReceipt {
            history: node.history.clone(),
            time,
            kind: forced.map_or(NodeKind::Decision, |boundary| NodeKind::ForcedStop { boundary }),
            reach_probability: node.reach,
            belief: node.belief.to_vec(),
            realized_utility: 0.0,
            continuation_value: 0.0,
            forfeited_outcomes: 0,
        });
        self.decisions.push(DecisionReceipt {
            history: node.history.clone(),
            time,
            forced_by: forced,
            actions: distribution.clone(),
            resource_used: node.resource,
        });
        let mut realized = due_now;
        let mut value = due_now;
        for &(action, probability) in &distribution {
            let spec = compiled.actions.get(action).ok_or_else(|| {
                refusal!(
                    "invalid_argument",
                    "sequential_evaluation.action_index",
                    format!("history {} names undeclared action {action}", node.history),
                    "name a declared action",
                )
            })?;
            if forced.is_none() {
                self.check_support(node.history, action, &spec.label)?;
            }
            let used = node.resource + spec.resource_use;
            if compiled.budget.is_some_and(|budget| used > budget + 1e-12) {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.resource_budget_exceeded",
                    format!(
                        "taking `{}` in history {} uses {used} of a budget of {:?}",
                        spec.label, node.history, compiled.budget
                    ),
                    "restrict the policy to the declared resource budget or raise the budget",
                ));
            }
            let step = dot(node.belief, &spec.immediate) - spec.utility_cost;
            realized += probability * step;
            let mut q = step;
            match spec.role {
                ActionRole::Commit => {
                    let pending: Vec<(usize, usize)> =
                        remaining.iter().map(|p| (p.action, p.due)).collect();
                    let tail = commit_tail(compiled, action, node.belief, time, &pending);
                    realized += probability * tail;
                    q += tail;
                }
                ActionRole::Continue => {
                    let mut child_pending = remaining.clone();
                    if let Some((delay, _)) = &spec.delayed {
                        child_pending.push(Pending { action, due: time + delay });
                    }
                    for branch in branches(compiled, node.belief, action) {
                        let child_history = node.history.extended(action, branch.symbol);
                        let child = Node {
                            history: &child_history,
                            belief: &branch.posterior,
                            pending: &child_pending,
                            reach: node.reach * probability * branch.probability,
                            resource: used,
                        };
                        q += branch.probability * self.visit(&child)?;
                    }
                }
            }
            value += probability * q;
        }
        self.accounting += node.reach * realized;
        let receipt = &mut self.histories[slot];
        receipt.realized_utility = realized;
        receipt.continuation_value = value;
        Ok(value)
    }

    fn check_support(
        &self,
        history: &HistoryKey,
        action: usize,
        label: &str,
    ) -> SequentialResult<()> {
        if !self.compiled.support.is_supported(history, action) {
            return Err(refusal!(
                "history_support_violated",
                "sequential_evaluation.action_not_supported",
                format!(
                    "the policy takes `{label}` in the reachable history {history}, where the model declares no support"
                ),
                "restrict the policy to supported actions in this history, or establish the support evidence (a positivity premise) for this action here",
            ));
        }
        if let Some(behaviour) = self.behaviour {
            let supported = behaviour.probability(history, action).is_some_and(|p| p > 0.0);
            if !supported {
                return Err(refusal!(
                    "history_support_violated",
                    "sequential_evaluation.behaviour_support_missing",
                    format!(
                        "the target policy takes `{label}` in the reachable history {history}, where the logging policy has probability zero or no declared decision"
                    ),
                    "evaluate a policy the logged data can identify, or collect data where the logging policy takes this action",
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn walk(
    compiled: &Compiled,
    policy: &ContingentPolicy,
    behaviour: Option<&ContingentPolicy>,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<PolicyEvaluation> {
    policy.validate()?;
    for (history, distribution) in policy.entries() {
        if distribution.iter().any(|(action, _)| *action >= compiled.actions.len()) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_index",
                format!("the policy's decision for history {history} names an undeclared action"),
                "name declared actions only",
            ));
        }
    }
    let mut walker = Walker {
        compiled,
        policy,
        behaviour,
        ctx,
        limits,
        nodes: 0,
        histories: Vec::new(),
        decisions: Vec::new(),
        accounting: 0.0,
    };
    let root = HistoryKey::root();
    let value = walker.visit(&Node {
        history: &root,
        belief: &compiled.initial,
        pending: &[],
        reach: 1.0,
        resource: 0.0,
    })?;
    let residual = (walker.accounting - value).abs();
    if residual.is_nan() || residual > 1e-9 * (1.0 + value.abs()) {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.accounting_residual",
            format!("the realized-utility accounting differs from the value by {residual}"),
            "the enumeration is internally inconsistent; the model probabilities are not normalized",
        ));
    }
    Ok(PolicyEvaluation {
        value,
        policy_version: policy.version().to_owned(),
        policy_identity: policy.identity_hex(),
        model_identity: compiled.identity.clone(),
        histories: walker.histories,
        decisions: walker.decisions,
        nodes_visited: walker.nodes,
        accounting_residual: residual,
        measurement_identities: compiled
            .actions
            .iter()
            .enumerate()
            .filter_map(|(index, action)| {
                action.channel_identity.clone().map(|identity| (index, identity))
            })
            .collect(),
        assumptions: SequentialAssumptions::DECLARED,
        inference: SEQUENTIAL_EVALUATION_INFERENCE,
    })
}

/// Evaluate a fixed policy by exact enumeration of its reachable histories
/// (the sequential g-formula).
///
/// The policy is read, never changed: evaluating collects no observation and
/// learns nothing. Every reachable history must have a declared decision, and
/// every action taken must be supported there.
///
/// # Errors
/// A malformed model or policy; a reachable history without a decision; an
/// action without support (positivity); a resource budget exceeded; an
/// outcome due after the horizon under the `refuse` censoring rule; a bound
/// exceeded; or cancellation.
pub fn evaluate_policy(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<PolicyEvaluation> {
    let compiled = model.compile()?;
    walk(&compiled, policy, None, limits, ctx)
}
