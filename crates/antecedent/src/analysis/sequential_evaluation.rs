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
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_io::IoError;

pub use super::design_exports::design_crate::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, BeliefTracker, CensoringRule, ComparisonKind,
    ComparisonReport, ComparisonRow, ComparisonSpec, ContingentPolicy, DecisionReceipt,
    DelayedOutcome, EvaluationLimits, FiniteSequentialModel, HistoryKey, HistoryReceipt,
    HistoryStep, LoggedStep, LoggedTrajectory, MAX_ACTIONS, MAX_HORIZON, MAX_LOGGED_TRAJECTORIES,
    MAX_POLICY_ENTRIES, MAX_ROLLOUTS, MAX_STATES, NodeKind, ObservationChannel, ObservationReceipt,
    OffPolicyEstimate, PolicyEvaluation, PolicySpec, RolloutConfig, RolloutEstimate,
    SEQUENTIAL_EVALUATION_IDENTITY_VERSION, SEQUENTIAL_EVALUATION_INFERENCE, STREAM_SCHEME,
    SequentialAssumptions, SequentialEvaluationError, SequentialRequest, SequentialResult,
    StoppingBoundary, StreamRole, VoiCheck, VoiRequest, compare_policies, evaluate_off_policy,
    evaluate_policy, request_interval, rollout_policy, stream_uniform,
    value_of_collecting_evidence,
};
pub use super::design_exports::design_crate::sequential_evaluation_artifact::{
    EvaluationIdentities, ExpectedEvaluation, MAX_SEQUENTIAL_EVALUATION_ARTIFACT_BYTES,
    OffPolicyInput, SEQUENTIAL_EVALUATION_ARTIFACT_FEATURE, SEQUENTIAL_EVALUATION_ARTIFACT_VERSION,
    SequentialEvaluationArtifact, SequentialEvaluationBody,
};
pub use antecedent_contract::evidence_ledger::EvidenceLedger;

/// Evaluate a fixed policy by exact enumeration with the default bounds.
///
/// Equivalent to [`evaluate_policy`] with [`EvaluationLimits::default`].
///
/// # Errors
///
/// A malformed model or policy, a reachable history without a decision, an
/// action without support, a resource budget exceeded, an outcome due after
/// the horizon under the `refuse` censoring rule, or cancellation.
pub fn evaluate(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    ctx: &ExecutionContext,
) -> SequentialResult<PolicyEvaluation> {
    evaluate_policy(model, policy, EvaluationLimits::default(), ctx)
}

/// Evaluate and seal a policy, with an optional off-policy estimate and an
/// optional rollout diagnostic, as a replayable artifact.
///
/// # Errors
///
/// Any evaluation refusal, or a body over the artifact bounds.
pub fn seal(
    model: &FiniteSequentialModel,
    policy: &ContingentPolicy,
    off_policy: Option<&OffPolicyInput>,
    rollout: Option<RolloutConfig>,
    ctx: &ExecutionContext,
) -> Result<SequentialEvaluationArtifact, IoError> {
    SequentialEvaluationArtifact::new(
        model,
        policy,
        off_policy,
        rollout,
        EvaluationLimits::default(),
        ctx,
    )
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
