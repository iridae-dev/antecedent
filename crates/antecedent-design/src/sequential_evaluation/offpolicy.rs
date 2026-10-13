//! Off-policy evaluation from logged finite trajectories: sequential
//! importance weighting with the logged assignment probabilities.
//!
//! For a trajectory logged under a behaviour policy `mu` with realized return
//! `G`, the weight of a target policy `pi` is the product over the policy
//! steps of `pi(a_t | h_t) / mu(a_t | h_t)`. Steps forced by a declared
//! stopping boundary carry weight one on both sides. The ordinary estimate
//! `mean(w G)` is unbiased for the value of `pi` when every action `pi` can
//! take in a reachable history has `mu > 0` there (positivity), under the
//! declared exchangeability of the logged assignments given the information
//! set and consistency of the logged outcomes.
//!
//! Positivity is checked on every history `pi` can reach in the declared
//! model, not only on the logged ones: a target action the logging policy
//! never takes cannot be evaluated from the log. Every logged action and
//! observation is recorded once in the evidence ledger.
//!
//! The result is a point estimate. Its standard error is an uncalibrated
//! diagnostic of the weighted returns; no interval is produced.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::{
    EvidenceLedger, MAX_OBSERVATIONS_PER_SOURCE, ObservationId, SourceDeclaration, SourceKind,
};
use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::belief::{branch_for, from_ledger, record_id, recorded_in_ledger};
use super::gformula::{EvaluationLimits, walk};
use super::model::{ActionRole, Compiled};
use super::policy::{ContingentPolicy, HistoryKey};
use super::{
    FiniteSequentialModel, MAX_LABEL_BYTES, MAX_LOGGED_TRAJECTORIES,
    SEQUENTIAL_EVALUATION_INFERENCE, SequentialResult, cancelled, limit_exceeded,
};

/// One logged decision cycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoggedStep {
    /// Action taken.
    pub action: usize,
    /// Observation symbol received; `None` for a commit action.
    pub observation: Option<usize>,
    /// Probability the logging policy assigned to the action in this history;
    /// `1.0` for a step forced by a stopping boundary.
    pub assignment_probability: f64,
}

/// One logged trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedTrajectory {
    /// Unique identity of the trajectory.
    pub id: String,
    /// Decision cycles, oldest first.
    pub steps: Vec<LoggedStep>,
    /// Realized state-dependent utility (immediate, delayed and terminal
    /// outcomes), excluding the model's deterministic action costs, which
    /// are added from the model.
    pub outcome_utility: f64,
}

/// A point estimate of a policy's value from logged trajectories.
#[derive(Clone, Debug, PartialEq)]
pub struct OffPolicyEstimate {
    /// Ordinary importance-weighted estimate `mean(w G)`.
    pub ordinary: f64,
    /// Self-normalized estimate `sum(w G) / sum(w)`: consistent, biased in
    /// finite samples.
    pub self_normalized: f64,
    /// Effective sample size `(sum w)^2 / sum w^2`.
    pub effective_sample_size: f64,
    /// Largest weight.
    pub max_weight: f64,
    /// Trajectories used.
    pub trajectories: u64,
    /// Standard error of the ordinary estimate treating the weighted returns
    /// as independent; `None` below two trajectories. An uncalibrated
    /// diagnostic, never an interval.
    pub uncalibrated_standard_error: Option<f64>,
    /// Target policy version.
    pub policy_version: String,
    /// Target policy identity (hex).
    pub policy_identity: String,
    /// Logging policy version.
    pub behaviour_version: String,
    /// Logging policy identity (hex).
    pub behaviour_identity: String,
    /// Model identity (hex).
    pub model_identity: String,
    /// Histories enumerated for the positivity check.
    pub positivity_histories_checked: u64,
    /// Observations and actions recorded once in the ledger.
    pub records_counted: u64,
    /// Whether the complete log was already in the ledger (a re-evaluation
    /// of recorded observations); nothing was recorded again.
    pub reused_recorded_log: bool,
    /// Ledger state digest after recording (hex).
    pub ledger_state: String,
    /// The inferential claim: always `point_only`.
    pub inference: &'static str,
}

fn inconsistent(id: &str, text: &str) -> super::SequentialEvaluationError {
    refusal!(
        "invalid_argument",
        "sequential_evaluation.logged_trajectory_inconsistent",
        format!("trajectory `{id}`: {text}"),
        "log trajectories generated under the declared model, logging policy, boundaries and budget, or correct the model declaration",
    )
}

struct Weighted {
    weight: f64,
    ret: f64,
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat replay of a trajectory: boundary, support, assignment, belief, resources"
)]
fn weigh(
    compiled: &Compiled,
    target: &ContingentPolicy,
    behaviour: &ContingentPolicy,
    trajectory: &LoggedTrajectory,
) -> SequentialResult<Weighted> {
    let id = trajectory.id.as_str();
    if trajectory.steps.is_empty() || trajectory.steps.len() > compiled.horizon {
        return Err(inconsistent(id, "a trajectory has between one and `horizon` decision cycles"));
    }
    if !trajectory.outcome_utility.is_finite() {
        return Err(inconsistent(id, "the realized outcome utility is not finite"));
    }
    let mut belief = compiled.initial.clone();
    let mut history = HistoryKey::root();
    let mut weight = 1.0_f64;
    let mut cost = 0.0;
    let mut resource = 0.0;
    let mut ended = false;
    for step in &trajectory.steps {
        if ended {
            return Err(inconsistent(id, "decision cycles continue after a commit"));
        }
        let spec = compiled
            .actions
            .get(step.action)
            .ok_or_else(|| inconsistent(id, "a logged action is not declared"))?;
        match compiled.crossed_boundary(&belief) {
            Some(boundary) => {
                let stop = compiled.boundaries[boundary].stop_action;
                if step.action != stop || step.assignment_probability.to_bits() != 1.0_f64.to_bits()
                {
                    return Err(inconsistent(
                        id,
                        "a declared stopping boundary was crossed but the logged step is not its forced stop with assignment probability one",
                    ));
                }
            }
            None => {
                let logged = step.assignment_probability;
                if !(logged.is_finite() && logged > 0.0 && logged <= 1.0) {
                    return Err(inconsistent(
                        id,
                        "a logged assignment probability is not in (0, 1]",
                    ));
                }
                let declared = behaviour.probability(&history, step.action);
                if declared.is_none_or(|p| p.to_bits() != logged.to_bits()) {
                    return Err(inconsistent(
                        id,
                        "a logged assignment probability differs from the declared logging policy in that history",
                    ));
                }
                if !compiled.support.is_supported(&history, step.action) {
                    return Err(refusal!(
                        "history_support_violated",
                        "sequential_evaluation.action_not_supported",
                        format!(
                            "trajectory `{id}` took `{}` in history {history}, where the model declares no support",
                            spec.label
                        ),
                        "declare the support evidence for the logged action, or drop the trajectory",
                    ));
                }
                match target.probability(&history, step.action) {
                    Some(pi) => weight *= pi / logged,
                    None if weight > 0.0 => {
                        return Err(refusal!(
                            "invalid_argument",
                            "sequential_evaluation.policy_incomplete",
                            format!(
                                "the target policy declares no decision for the reachable history {history}"
                            ),
                            "declare an action for every history the target policy can reach",
                        ));
                    }
                    None => {}
                }
            }
        }
        resource += spec.resource_use;
        if compiled.budget.is_some_and(|budget| resource > budget + 1e-12) {
            return Err(inconsistent(id, "the trajectory exceeds the declared resource budget"));
        }
        cost += spec.utility_cost;
        match spec.role {
            ActionRole::Commit => {
                if step.observation.is_some() {
                    return Err(inconsistent(id, "a commit action carries an observation"));
                }
                ended = true;
            }
            ActionRole::Continue => {
                let symbol = step.observation.ok_or_else(|| {
                    inconsistent(id, "a continue action has no logged observation")
                })?;
                let branch =
                    branch_for(compiled, &belief, step.action, symbol).ok_or_else(|| {
                        inconsistent(
                            id,
                            "a logged observation has zero probability under the model",
                        )
                    })?;
                belief = branch.posterior;
                history = history.extended(step.action, symbol);
            }
        }
    }
    if !ended && trajectory.steps.len() != compiled.horizon {
        return Err(inconsistent(
            id,
            "the trajectory ends before a commit and before the horizon (censored trajectories are not evaluated)",
        ));
    }
    Ok(Weighted { weight, ret: trajectory.outcome_utility - cost })
}

/// Collect the ledger identities of every logged trajectory, action and
/// observation. Identities bind the content, so a trajectory that changed
/// under the same identity overlaps the earlier record without matching it.
fn ledger_records(scope: &str, trajectories: &[&LoggedTrajectory]) -> Vec<ObservationId> {
    let mut ids = Vec::new();
    for trajectory in trajectories {
        ids.push(record_id(
            scope,
            &trajectory.id,
            &format!("trajectory:{}", trajectory.outcome_utility.to_bits()),
        ));
        for (time, step) in trajectory.steps.iter().enumerate() {
            let record = format!("{}|{time}", trajectory.id);
            ids.push(record_id(scope, &record, &format!("action:{}", step.action)));
            if let Some(symbol) = step.observation {
                ids.push(record_id(scope, &record, &format!("observation:{symbol}")));
            }
        }
    }
    ids
}

/// Evaluate a target policy from trajectories logged under a known policy.
///
/// The ledger is updated only when the whole evaluation succeeds: every
/// logged action and observation is recorded once, and a repeat (within the
/// batch or already in the ledger) is refused as double counting.
///
/// # Errors
/// A malformed model or policy; positivity violated on a reachable history;
/// a logged trajectory that is inconsistent with the model, the logging
/// policy, the boundaries or the budget; no overlap between the target and
/// the log; double counting; a bound exceeded; or cancellation.
#[allow(
    clippy::too_many_lines,
    reason = "positivity, weighting, ledger recording and estimation are one sequential contract"
)]
pub fn evaluate_off_policy(
    model: &FiniteSequentialModel,
    target: &ContingentPolicy,
    behaviour: &ContingentPolicy,
    log: &[LoggedTrajectory],
    ledger: &mut EvidenceLedger,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<OffPolicyEstimate> {
    let compiled = model.compile()?;
    behaviour.validate()?;
    if log.is_empty() {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.empty_log",
            "no logged trajectories were supplied",
            "supply the logged trajectories to evaluate against",
        ));
    }
    if log.len() > MAX_LOGGED_TRAJECTORIES {
        return Err(limit_exceeded(
            "the number of logged trajectories",
            MAX_LOGGED_TRAJECTORIES as u64,
        ));
    }
    if log.iter().any(|t| t.id.trim().is_empty() || t.id.len() > MAX_LABEL_BYTES) {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.trajectory_identity",
            "every logged trajectory needs a non-blank identity of bounded length",
            "identify each trajectory so it can be counted once",
        ));
    }
    // Positivity on every history the target policy can reach in the model.
    let support = walk(&compiled, target, Some(behaviour), limits, ctx)?;
    // Canonical order: the estimate is invariant to the order of the log.
    let mut ordered: Vec<&LoggedTrajectory> = log.iter().collect();
    ordered.sort_by(|a, b| a.id.cmp(&b.id));
    if ordered.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err(refusal!(
            "evidence_double_counted",
            "sequential_evaluation.trajectory_recorded_twice",
            "a trajectory identity appears more than once in the log; its observations would be counted twice",
            "log each trajectory once, or give distinct trajectories distinct identities",
        ));
    }
    let n = ordered.len();
    let mut sum_wg = 0.0;
    let mut sum_w = 0.0;
    let mut sum_w2 = 0.0;
    let mut sum_wg2 = 0.0;
    let mut max_weight = 0.0_f64;
    for (index, trajectory) in ordered.iter().enumerate() {
        if index % 256 == 0 && ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let weighted = weigh(&compiled, target, behaviour, trajectory)?;
        let product = weighted.weight * weighted.ret;
        sum_wg += product;
        sum_w += weighted.weight;
        sum_w2 += weighted.weight * weighted.weight;
        sum_wg2 += product * product;
        max_weight = max_weight.max(weighted.weight);
    }
    if sum_w <= 0.0 {
        return Err(refusal!(
            "history_support_violated",
            "sequential_evaluation.no_overlap",
            "no logged trajectory is consistent with the target policy: every importance weight is zero",
            "log data where the logging policy takes the actions the target policy takes",
        ));
    }
    // Record every action and observation once.
    let scope = format!("logged:{}", behaviour.identity_hex());
    let mut ids = ledger_records(&scope, &ordered);
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(refusal!(
            "evidence_double_counted",
            "sequential_evaluation.observation_recorded_twice",
            "an action or observation is recorded twice inside the log",
            "record each action and observation once",
        ));
    }
    let known = ids.iter().filter(|id| recorded_in_ledger(ledger, **id)).count();
    if known != 0 && known != ids.len() {
        return Err(refusal!(
            "evidence_double_counted",
            "sequential_evaluation.log_partially_counted",
            format!(
                "{known} of {} logged records are already in the evidence ledger and the rest are not; the overlap would be counted twice",
                ids.len()
            ),
            "evaluate the complete earlier log again, or only trajectories the ledger does not hold",
        ));
    }
    let reused = known == ids.len();
    let mut staged = ledger.clone();
    if !reused {
        for (chunk, rows) in ids.chunks(MAX_OBSERVATIONS_PER_SOURCE).enumerate() {
            let mut writer = CanonicalWriter::new("antecedent.sequential_evaluation.logged_source");
            writer.text(&scope).word(chunk as u64).word(rows.len() as u64);
            let digest = writer.finish_hex();
            let declaration = SourceDeclaration::new(
                format!("logged_trajectories:{}", &digest[..32]),
                SourceKind::ObservationSet,
            )
            .with_observations(rows.iter().copied());
            staged.add_source(declaration).map_err(from_ledger)?;
        }
    }
    let records_counted = ids.len() as u64;
    let state = staged.state_digest().to_hex();
    *ledger = staged;
    let nf = n as f64;
    let ordinary = sum_wg / nf;
    let uncalibrated_standard_error = (n > 1).then(|| {
        let variance = ((sum_wg2 - nf * ordinary * ordinary) / (nf - 1.0)).max(0.0);
        (variance / nf).sqrt()
    });
    Ok(OffPolicyEstimate {
        ordinary,
        self_normalized: sum_wg / sum_w,
        effective_sample_size: sum_w * sum_w / sum_w2,
        max_weight,
        trajectories: n as u64,
        uncalibrated_standard_error,
        policy_version: target.version().to_owned(),
        policy_identity: target.identity_hex(),
        behaviour_version: behaviour.version().to_owned(),
        behaviour_identity: behaviour.identity_hex(),
        model_identity: compiled.identity.clone(),
        positivity_histories_checked: support.nodes_visited,
        records_counted,
        reused_recorded_log: reused,
        ledger_state: state,
        inference: SEQUENTIAL_EVALUATION_INFERENCE,
    })
}
