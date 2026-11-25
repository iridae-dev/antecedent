//! Sequential causal decision evaluation of a caller-supplied contingent policy.
//!
//! A fixed policy over a finite-horizon dynamic finite model is evaluated by
//! exact enumeration of the histories it reaches (the sequential g-formula),
//! by sequential importance weighting of logged finite trajectories, and by
//! Monte Carlo rollouts on common random streams. This is evaluation of a
//! supplied policy, not planning or policy learning: a policy changed from the
//! data under evaluation, a reweighted static score table, a different
//! population or period, and any interval are typed refusals.
//!
//! * Information sets, observation and intervention histories, decision times,
//!   delayed outcomes, resource budgets, Bayes belief updates, stopping
//!   boundaries and terminal utility are declared in a
//!   [`FiniteSequentialModel`].
//! * Observations enter a belief update through a typed measurement model
//!   ([`ObservationChannel`]); each recorded observation and action is counted
//!   once in the evidence ledger ([`BeliefTracker`], [`evaluate_off_policy`]).
//! * Act-now, wait, collect-evidence and choose-after-observation comparisons
//!   are evaluated by [`compare_policies`]; the value of collecting evidence is
//!   scored by the design ranker and checked against the enumeration.
//! * [`SequentialEvaluationArtifact`] (`sequential_evaluation_v1`) carries the
//!   policy version, history and decision receipts; a consumer re-evaluates
//!   and compares bit for bit.
//!
//! Every value is point-only. The rollout and importance-weighting errors are
//! uncalibrated diagnostics, never intervals.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_io::IoError;

pub use super::design_exports::design_crate::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, BeliefTracker, CensoringRule, ComparisonKind,
    ComparisonReport, ComparisonRow, ComparisonSpec, ContingentPolicy, DecisionReceipt,
    DelayedOutcome, EvaluationLimits, FiniteSequentialModel, HistoryKey, HistoryReceipt,
    HistoryStep, LoggedStep, LoggedTrajectory, MAX_ACTIONS, MAX_HORIZON, MAX_LOGGED_TRAJECTORIES,
    MAX_POLICY_ENTRIES, MAX_ROLLOUTS, MAX_STATES, NodeKind, OFF_POLICY_INTERVAL_LEVEL,
    ObservationChannel, ObservationReceipt, OffPolicyEstimate, OffPolicyInterval, PolicyEvaluation,
    PolicySpec, RolloutConfig, RolloutEstimate, SEQUENTIAL_EVALUATION_IDENTITY_VERSION,
    SEQUENTIAL_EVALUATION_INFERENCE, STREAM_SCHEME, SequentialAssumptions,
    SequentialEvaluationError, SequentialRequest, SequentialResult, StoppingBoundary, StreamRole,
    VoiCheck, VoiRequest, request_interval, stream_uniform,
};
pub use super::design_exports::design_crate::sequential_evaluation_artifact::{
    EvaluationIdentities, ExpectedEvaluation, IntervalWire,
    MAX_SEQUENTIAL_EVALUATION_ARTIFACT_BYTES, OffPolicyInput,
    SEQUENTIAL_EVALUATION_ARTIFACT_FEATURE, SEQUENTIAL_EVALUATION_ARTIFACT_VERSION,
    SequentialEvaluationArtifact, SequentialEvaluationBody,
};
pub use antecedent_contract::evidence_ledger::EvidenceLedger;

/// Evaluate a fixed policy by exact enumeration of its reachable histories.
///
/// # Errors
///
/// A malformed model or policy, a reachable history without a decision, an
/// action without support, a resource budget exceeded, an outcome due after
/// the horizon under the `refuse` censoring rule, a bound exceeded, or cancellation.
#[deprecated(
    since = "3.1.0",
    note = "use Study::compile_evaluate_policy which calls this kernel function with the same inputs"
)]
pub fn evaluate_policy(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<PolicyEvaluation> {
    antecedent_design::sequential_evaluation::evaluate_policy(model, policy, limits, ctx)
}

/// Weight a declared log by the likelihood ratio of a target policy and the logging policy.
///
/// # Errors
///
/// A malformed model or policy, a positivity failure, an inconsistent trajectory, double
/// counting, a bound exceeded, or cancellation.
#[deprecated(
    since = "3.1.0",
    note = "use Study::compile_evaluate_off_policy which calls this kernel function with the same inputs"
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
    antecedent_design::sequential_evaluation::evaluate_off_policy(
        model, target, behaviour, log, ledger, limits, ctx,
    )
}

/// Roll a fixed policy out on common random streams.
///
/// # Errors
///
/// A malformed model or policy, a reachable history without a decision, an unsupported action,
/// a resource budget exceeded, a rollout count outside the bound, or cancellation.
#[deprecated(
    since = "3.1.0",
    note = "use Study::compile_rollout_policy which calls this kernel function with the same inputs"
)]
pub fn rollout_policy(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    config: RolloutConfig,
    ctx: &ExecutionContext,
) -> SequentialResult<RolloutEstimate> {
    antecedent_design::sequential_evaluation::rollout_policy(model, policy, config, ctx)
}

/// Evaluate caller-supplied comparisons on one model.
///
/// # Errors
///
/// A transported request, a static score table or a policy learned from the data under
/// evaluation, a policy that does not implement its declared kind, any evaluation refusal, or a
/// value-of-information request the ranker and the enumeration disagree on.
#[deprecated(
    since = "3.1.0",
    note = "use Study::compile_compare_policies which calls this kernel function with the same inputs"
)]
pub fn compare_policies(
    model: &FiniteSequentialModel,
    specs: Vec<ComparisonSpec>,
    voi: Option<&VoiRequest>,
    request: &SequentialRequest,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<ComparisonReport> {
    antecedent_design::sequential_evaluation::compare_policies(
        model, specs, voi, request, limits, ctx,
    )
}

/// Score the value of collecting one observation before committing.
///
/// # Errors
///
/// A malformed model, a collect action that is not pure information, a model outside the
/// one-collection scope, a ranker refusal, or a disagreement between the ranker and the
/// enumeration.
#[deprecated(
    since = "3.1.0",
    note = "use Study::compile_value_of_collecting_evidence which calls this kernel function with the same inputs"
)]
pub fn value_of_collecting_evidence(
    model: &FiniteSequentialModel,
    request: &VoiRequest,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<VoiCheck> {
    antecedent_design::sequential_evaluation::value_of_collecting_evidence(
        model, request, limits, ctx,
    )
}

/// Evaluate a fixed policy by exact enumeration with the default bounds.
///
/// Compiles [`Study::compile_evaluate_policy`] with [`EvaluationLimits::default`] and executes
/// that program. Equivalent to [`evaluate_policy`] with those limits.
///
/// # Errors
///
/// A malformed model or policy, a reachable history without a decision, an
/// action without support, a resource budget exceeded, an outcome due after
/// the horizon under the `refuse` censoring rule, cancellation, or compiling
/// from inside a lowered dispatch.
pub fn evaluate(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    ctx: &ExecutionContext,
) -> SequentialResult<PolicyEvaluation> {
    let compiled = super::Study::compile_evaluate_policy(
        model.clone(),
        policy.clone(),
        EvaluationLimits::default(),
        ctx,
    )
    .map_err(evaluation_compile_error)?;
    compiled.evaluate(ctx)
}

fn evaluation_compile_error(error: crate::error::CausalError) -> SequentialEvaluationError {
    match error {
        crate::error::CausalError::Cancelled { .. } => SequentialEvaluationError {
            code: antecedent_core::reason_code!("cancelled_no_claim"),
            detail: antecedent_core::reason_details::sequential_evaluation::CANCELLED.to_owned(),
            message: "the evaluation was cancelled before it produced a value".to_owned(),
            remedy: "re-run the evaluation; a partial enumeration is not a result",
        },
        other => SequentialEvaluationError {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "kernel_program.compile".to_owned(),
            message: other.to_string(),
            remedy: "compile the policy evaluation outside a lowered dispatch",
        },
    }
}

/// Evaluate and seal a policy, with an optional off-policy estimate and an
/// optional rollout diagnostic, as a replayable artifact.
///
/// Compiles [`Study::compile_sequential_evaluation_seal`] with
/// [`EvaluationLimits::default`] and seals through that program.
///
/// # Errors
///
/// Any evaluation refusal, a body over the artifact bounds, cancellation, or
/// compiling from inside a lowered dispatch.
pub fn seal(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    off_policy: Option<&OffPolicyInput>,
    rollout: Option<RolloutConfig>,
    ctx: &ExecutionContext,
) -> Result<SequentialEvaluationArtifact, IoError> {
    let compiled = super::Study::compile_sequential_evaluation_seal(
        model.clone(),
        policy.clone(),
        off_policy.cloned(),
        rollout,
        EvaluationLimits::default(),
        ctx,
    )
    .map_err(seal_compile_error)?;
    compiled.seal(ctx)
}

fn seal_compile_error(error: crate::error::CausalError) -> IoError {
    match error {
        crate::error::CausalError::Cancelled { .. } => IoError::Refused {
            code: antecedent_core::reason_code!("cancelled_no_claim"),
            message: format!(
                "{}: the evaluation was cancelled before it produced a value",
                antecedent_core::reason_details::sequential_evaluation::CANCELLED,
            ),
        },
        other => IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: other.to_string(),
        },
    }
}

/// Serialize an artifact through the bounded sectioned container.
///
/// # Errors
///
/// An empty id, an invalid body or an oversized payload.
pub fn export(
    artifact: &SequentialEvaluationArtifact,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    artifact.to_bytes(artifact_id)
}

/// Consume an artifact by re-evaluating it; `expected` holds identities the
/// consumer retained independently of the bytes.
///
/// # Errors
///
/// Corruption, a changed identity, an evaluation refusal on replay, or any
/// stored number that does not replay bit for bit.
pub fn consume(
    bytes: &[u8],
    expected: Option<&ExpectedEvaluation>,
    ctx: &ExecutionContext,
) -> Result<SequentialEvaluationArtifact, IoError> {
    antecedent_design::sequential_evaluation_artifact::consume(bytes, expected, ctx)
}
