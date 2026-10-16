//! Monte Carlo rollouts of a fixed policy on common random streams.
//!
//! Every uniform a rollout consumes is keyed by its semantic stream identity
//! `(seed, rollout, time, role)` rather than by its position in a sequence, so
//! two policies evaluated with the same seed see the same randomness for the
//! same unit at the same time: the initial state, the transition draw that
//! carries the state to time `t`, the observation draw at time `t` and the
//! policy's own randomization are all coupled across policies. The estimate is
//! a diagnostic of the exact enumeration, with an uncalibrated Monte Carlo
//! standard error; it is never an interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext, RngFactory, StreamDomain};

use super::belief::branch_for;
use super::model::{ActionRole, CensoringRule, Compiled};
use super::policy::{ContingentPolicy, HistoryKey};
use super::{
    FiniteSequentialModel, MAX_ROLLOUTS, SEQUENTIAL_EVALUATION_INFERENCE, SequentialResult,
    cancelled, limit_exceeded,
};

/// The stream scheme every rollout draw is keyed by.
pub const STREAM_SCHEME: &str = "policy_rollout.stream.v1:(seed,rollout,time,role)";

/// Which uniform of a rollout step a stream identity names.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum StreamRole {
    /// The initial latent state.
    InitialState,
    /// The transition that carries the state to the named time.
    Transition,
    /// The observation emitted at the named time.
    Observation,
    /// The policy's own randomization at the named time.
    PolicyChoice,
}

impl StreamRole {
    const fn tag(self) -> u64 {
        match self {
            Self::InitialState => 1,
            Self::Transition => 2,
            Self::Observation => 3,
            Self::PolicyChoice => 4,
        }
    }
}

/// The uniform in `[0, 1)` of one semantic stream identity.
///
/// A pure function of its arguments: it does not depend on how many other
/// uniforms were drawn, in which order, or by which policy.
#[must_use]
pub fn stream_uniform(seed: u64, rollout: u64, time: u64, role: StreamRole) -> f64 {
    let mut writer = CanonicalWriter::new("antecedent.sequential_evaluation.stream");
    writer.word(rollout).word(time).word(role.tag());
    let digest = writer.finish();
    let mut word = [0_u8; 8];
    word.copy_from_slice(&digest[..8]);
    let index = u64::from_le_bytes(word);
    RngFactory::from_seed(seed).stream_for(StreamDomain::Design, index).next_f64()
}

/// How many rollouts to draw and the master seed of their streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RolloutConfig {
    /// Master seed.
    pub seed: u64,
    /// Number of independent rollouts.
    pub rollouts: u64,
}

/// A Monte Carlo estimate of a policy's value, as a diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutEstimate {
    /// Mean total utility over the rollouts.
    pub mean: f64,
    /// Standard error of the mean over rollouts (zero for one rollout).
    pub monte_carlo_standard_error: f64,
    /// Rollouts drawn.
    pub rollouts: u64,
    /// Master seed.
    pub seed: u64,
    /// The stream scheme the draws were keyed by.
    pub stream_scheme: &'static str,
    /// Always `uncalibrated_diagnostic`: the error is not a coverage claim.
    pub error_label: &'static str,
    /// Policy version label.
    pub policy_version: String,
    /// Policy identity (hex).
    pub policy_identity: String,
    /// Model identity (hex).
    pub model_identity: String,
    /// The inferential claim: always `point_only`.
    pub inference: &'static str,
}

/// Index of the interval of cumulative probability containing `u`.
fn pick(probabilities: &[f64], u: f64) -> usize {
    let mut cumulative = 0.0;
    let mut last_positive = 0;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability > 0.0 {
            last_positive = index;
        }
        cumulative += probability;
        if u < cumulative {
            return index;
        }
    }
    last_positive
}

fn delayed_utility(compiled: &Compiled, action: usize) -> &[f64] {
    match &compiled.actions[action].delayed {
        Some((_, utility)) => utility.as_slice(),
        None => &[],
    }
}

struct Rollout<'a> {
    compiled: &'a Compiled,
    policy: &'a ContingentPolicy,
    seed: u64,
    index: u64,
}

impl Rollout<'_> {
    fn uniform(&self, time: usize, role: StreamRole) -> f64 {
        stream_uniform(self.seed, self.index, time as u64, role)
    }

    /// The state after a commit, evolving under the commit action's
    /// transition, with every pending outcome realized at its due time.
    fn run_off(
        &self,
        action: usize,
        time: usize,
        mut state: usize,
        mut pending: Vec<(usize, usize)>,
    ) -> f64 {
        let compiled = self.compiled;
        let n = compiled.states;
        let spec = &compiled.actions[action];
        if let Some((delay, _)) = &spec.delayed {
            pending.push((action, time + delay));
        }
        let last = pending.iter().map(|(_, due)| *due).max().unwrap_or(time);
        let mut total = 0.0;
        for step in (time + 1)..=last {
            let row = &spec.transition[state * n..(state + 1) * n];
            state = pick(row, self.uniform(step, StreamRole::Transition));
            for &(pending_action, due) in &pending {
                if due == step {
                    total += delayed_utility(compiled, pending_action)[state];
                }
            }
        }
        total
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one flat simulation of a unit: credits, boundary, policy, state, observation"
    )]
    fn simulate(&self) -> SequentialResult<f64> {
        let compiled = self.compiled;
        let n = compiled.states;
        let mut state = pick(&compiled.initial, self.uniform(0, StreamRole::InitialState));
        let mut belief = compiled.initial.clone();
        let mut history = HistoryKey::root();
        let mut resource = 0.0;
        let mut total = 0.0;
        let mut pending: Vec<(usize, usize)> = Vec::new();
        for time in 0..compiled.horizon {
            let mut still = Vec::with_capacity(pending.len());
            for &(action, due) in &pending {
                if due == time {
                    total += delayed_utility(compiled, action)[state];
                } else {
                    still.push((action, due));
                }
            }
            pending = still;
            let forced = compiled.crossed_boundary(&belief);
            let action =
                if let Some(boundary) = forced {
                    compiled.boundaries[boundary].stop_action
                } else {
                    let distribution = self.policy.distribution(&history).ok_or_else(|| {
                    refusal!("invalid_argument", "sequential_evaluation.policy_incomplete",
                        format!(
                            "the policy declares no decision for the reachable history {history}"
                        ),
                        "declare an action for every history the policy can reach",
                    )
                })?;
                    let weights: Vec<f64> = distribution.iter().map(|(_, p)| *p).collect();
                    let chosen = pick(&weights, self.uniform(time, StreamRole::PolicyChoice));
                    distribution[chosen].0
                };
            let spec = compiled.actions.get(action).ok_or_else(|| {
                refusal!(
                    "invalid_argument",
                    "sequential_evaluation.action_index",
                    format!("action {action} is not declared"),
                    "name a declared action",
                )
            })?;
            if forced.is_none() && !compiled.support.is_supported(&history, action) {
                return Err(refusal!(
                    "history_support_violated",
                    "sequential_evaluation.action_not_supported",
                    format!(
                        "the policy takes `{}` in the reachable history {history}, where the model declares no support",
                        spec.label
                    ),
                    "restrict the policy to supported actions in this history",
                ));
            }
            resource += spec.resource_use;
            if compiled.budget.is_some_and(|budget| resource > budget + 1e-12) {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.resource_budget_exceeded",
                    format!("`{}` exceeds the declared resource budget", spec.label),
                    "restrict the policy to the declared resource budget",
                ));
            }
            total += spec.immediate[state] - spec.utility_cost;
            match spec.role {
                ActionRole::Commit => {
                    return Ok(total + self.run_off(action, time, state, pending));
                }
                ActionRole::Continue => {
                    if let Some((delay, _)) = &spec.delayed {
                        pending.push((action, time + delay));
                    }
                    let row = &spec.transition[state * n..(state + 1) * n];
                    state = pick(row, self.uniform(time + 1, StreamRole::Transition));
                    let column: Vec<f64> =
                        (0..spec.symbols).map(|s| spec.likelihood[s * n + state]).collect();
                    let symbol = pick(&column, self.uniform(time + 1, StreamRole::Observation));
                    let branch =
                        branch_for(compiled, &belief, action, symbol).ok_or_else(|| {
                            refusal!(
                                "invalid_argument",
                                "sequential_evaluation.rollout_belief",
                                "a simulated observation has zero probability under the belief",
                                "the model laws are inconsistent; revalidate the model",
                            )
                        })?;
                    belief = branch.posterior;
                    history = history.extended(action, symbol);
                }
            }
        }
        for &(action, due) in &pending {
            if due == compiled.horizon {
                total += delayed_utility(compiled, action)[state];
            } else if compiled.censoring == CensoringRule::Refuse {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.delayed_outcome_beyond_horizon",
                    "an outcome falls due after the horizon in a simulated history",
                    "extend the horizon, or declare the censoring rule `forfeit`",
                ));
            }
        }
        Ok(total + compiled.terminal[state])
    }
}

/// Simulate a fixed policy on common random streams.
///
/// Two calls with the same seed and rollout count but different policies
/// share every stream identity, so their estimates are coupled. The standard
/// error is an uncalibrated Monte Carlo diagnostic of the mean over rollouts.
///
/// # Errors
/// A malformed model or policy; a reachable history without a decision; an
/// unsupported action; a resource budget exceeded; a rollout count outside
/// `1..=MAX_ROLLOUTS`; or cancellation.
pub fn rollout_policy(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    config: RolloutConfig,
    ctx: &ExecutionContext,
) -> SequentialResult<RolloutEstimate> {
    let compiled = model.compile()?;
    policy.validate()?;
    if config.rollouts == 0 {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.rollouts",
            "at least one rollout is required",
            "request a positive number of rollouts",
        ));
    }
    if config.rollouts > MAX_ROLLOUTS {
        return Err(limit_exceeded("the number of rollouts", MAX_ROLLOUTS));
    }
    let mut mean = 0.0;
    let mut sum_squares = 0.0;
    for index in 0..config.rollouts {
        if index % 1024 == 0 && ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let total = Rollout { compiled: &compiled, policy, seed: config.seed, index }.simulate()?;
        let delta = total - mean;
        mean += delta / (index + 1) as f64;
        sum_squares += delta * (total - mean);
    }
    let count = config.rollouts as f64;
    let standard_error = if config.rollouts > 1 {
        (sum_squares / (count - 1.0) / count).max(0.0).sqrt()
    } else {
        0.0
    };
    Ok(RolloutEstimate {
        mean,
        monte_carlo_standard_error: standard_error,
        rollouts: config.rollouts,
        seed: config.seed,
        stream_scheme: STREAM_SCHEME,
        error_label: "uncalibrated_diagnostic",
        policy_version: policy.version().to_owned(),
        policy_identity: policy.identity_hex(),
        model_identity: compiled.identity.clone(),
        inference: SEQUENTIAL_EVALUATION_INFERENCE,
    })
}
