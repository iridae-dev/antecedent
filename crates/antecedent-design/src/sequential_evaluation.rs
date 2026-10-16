//! Sequential causal decision evaluation of a caller-supplied contingent policy.
//!
//! This module evaluates a policy; it does not plan, search or learn one. A
//! finite-horizon problem is a dynamic finite model with discrete latent
//! states and discrete actions ([`FiniteSequentialModel`]). The decision maker
//! repeatedly chooses an action, receives an observation through a declared
//! measurement model, updates a Bayes belief and chooses again, until a commit
//! action, a declared stopping boundary or the horizon ends the episode.
//!
//! # Contract
//!
//! * **Information sets and histories.** A decision may depend only on the
//!   observed history: the ordered `(action, observation)` pairs so far
//!   ([`HistoryKey`]). A [`ContingentPolicy`] is a table from histories to
//!   action distributions, so it cannot condition on the latent state.
//! * **Decision times, delays, resources, stopping.** Decisions happen at
//!   times `0..horizon`. An action's outcome may be delayed by a declared
//!   number of decision times and is credited when it falls due; an outcome
//!   due after the horizon is refused or forfeited as the model declares
//!   ([`CensoringRule`]). Resource use is bounded by a declared budget, and a
//!   declared [`StoppingBoundary`] on the belief forces a commit action.
//! * **Belief updates.** Every observation enters through a typed
//!   measurement model ([`ObservationChannel`]): its binding (construct,
//!   population, regime, arm), non-differential structure and identity are
//!   checked, and the confusion law becomes the likelihood of the Bayes
//!   update. Each observation and action of a realized history is recorded
//!   once in the evidence ledger ([`BeliefTracker`]); a duplicate is refused.
//! * **Support (positivity).** The model declares where each action is
//!   supported ([`ActionSupport`]). A policy that takes an action in a
//!   reachable history with no support is refused, as is a target policy that
//!   puts mass where a logging policy has none.
//!
//! # Routes
//!
//! * [`evaluate_policy`] is the sequential g-formula: exact enumeration of the
//!   histories reachable under the policy, with Bayes beliefs, under the
//!   declared exchangeability, consistency and positivity premises
//!   ([`SequentialAssumptions`]). It returns history and decision receipts.
//! * [`evaluate_off_policy`] is sequential importance weighting from logged
//!   finite trajectories with their logged assignment probabilities. It is a
//!   point estimate; its standard error is an uncalibrated diagnostic.
//! * [`rollout_policy`] is a Monte Carlo cross-check bound to common random
//!   streams keyed by the semantic stream identity `(rollout, time, role)`, so
//!   two policies are compared on the same randomness. Its error is reported as
//!   an uncalibrated diagnostic.
//! * [`compare_policies`] evaluates caller-supplied comparisons (act now, wait,
//!   collect evidence, choose differently after an observation). The value of
//!   collecting evidence is scored by the existing design ranker, not by a
//!   second implementation, and checked against the enumeration.
//!
//! # Refusals
//!
//! A reweighted static score table is not sequential evaluation, learning or
//! changing a policy inside an evaluation is not evaluation, a different
//! population or period needs a transport contract, and intervals or
//! posteriors for any value are not licensed ([`PolicySpec`],
//! [`SequentialRequest`], [`request_interval`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Build a typed refusal with its registered reason code, its
/// `sequential_evaluation.<rule>` detail, the message and the remedy, all
/// written where the refusal is emitted.
macro_rules! refusal {
    ($code:literal, $detail:literal, $message:expr, $remedy:expr $(,)?) => {
        $crate::sequential_evaluation::SequentialEvaluationError::new(
            antecedent_core::reason_code!($code),
            $detail,
            $message,
            $remedy,
        )
    };
}

mod belief;
mod comparison;
mod gformula;
mod model;
mod offpolicy;
mod policy;
mod rollout;

use std::fmt;

use antecedent_contract::ExternalRefusal;

pub use belief::{BeliefTracker, ObservationReceipt};
pub use comparison::{
    ComparisonKind, ComparisonReport, ComparisonRow, ComparisonSpec, VoiCheck, VoiRequest,
    compare_policies, value_of_collecting_evidence,
};
pub use gformula::{
    DecisionReceipt, EvaluationLimits, HistoryReceipt, NodeKind, PolicyEvaluation,
    SequentialAssumptions, evaluate_policy,
};
pub use model::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, DelayedOutcome, FiniteSequentialModel,
    ObservationChannel, StoppingBoundary,
};
pub use offpolicy::{LoggedStep, LoggedTrajectory, OffPolicyEstimate, evaluate_off_policy};
pub use policy::{ContingentPolicy, HistoryKey, HistoryStep, PolicySpec};
pub use rollout::{
    RolloutConfig, RolloutEstimate, STREAM_SCHEME, StreamRole, rollout_policy, stream_uniform,
};

/// Version word written into every sequential-evaluation identity. Bump only
/// when a canonical encoding in this module tree changes.
pub const SEQUENTIAL_EVALUATION_IDENTITY_VERSION: u64 = 1;
/// The only inferential claim any route of this module carries.
pub const SEQUENTIAL_EVALUATION_INFERENCE: &str = "point_only";
/// Most latent states of a model.
pub const MAX_STATES: usize = 32;
/// Most actions of a model.
pub const MAX_ACTIONS: usize = 16;
/// Most decision times of a model.
pub const MAX_HORIZON: usize = 12;
/// Most history entries a policy table may hold.
pub const MAX_POLICY_ENTRIES: usize = 65_536;
/// Most logged trajectories one off-policy evaluation accepts.
pub const MAX_LOGGED_TRAJECTORIES: usize = 150_000;
/// Most Monte Carlo rollouts one request may draw.
pub const MAX_ROLLOUTS: u64 = 10_000_000;
/// Longest label or identifier, in bytes.
pub const MAX_LABEL_BYTES: usize = 256;
/// Tolerance for probability vectors summing to one.
pub(crate) const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// A typed, actionable refusal of a sequential-evaluation request.
///
/// `code` is a registered runtime-refusal reason code and `detail` is a
/// `sequential_evaluation.<rule>` (or pass-through measurement) detail slot.
/// `remedy` names the declaration, evidence or route that would license the
/// request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequentialEvaluationError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: String,
    /// What was refused.
    pub message: String,
    /// What would license the request.
    pub remedy: &'static str,
}

impl SequentialEvaluationError {
    pub(crate) fn new(
        code: &'static str,
        detail: impl Into<String>,
        message: impl Into<String>,
        remedy: &'static str,
    ) -> Self {
        Self { code, detail: detail.into(), message: message.into(), remedy }
    }

    /// The refusal in the shared structured shape.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        ExternalRefusal::new(self.code, "bind", self.detail.clone())
            .with_offending(self.message.clone())
            .with_remedy(self.remedy)
    }
}

impl fmt::Display for SequentialEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for SequentialEvaluationError {}

impl From<ExternalRefusal> for SequentialEvaluationError {
    fn from(refusal: ExternalRefusal) -> Self {
        Self {
            code: refusal.code,
            detail: refusal.detail,
            message: refusal.offending.unwrap_or_default(),
            remedy: refusal.remedy.unwrap_or("declare a measurement model that licenses the use"),
        }
    }
}

/// Result alias of this module.
pub type SequentialResult<T> = Result<T, SequentialEvaluationError>;

pub(crate) fn cancelled() -> SequentialEvaluationError {
    SequentialEvaluationError::new(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "sequential_evaluation.cancelled",
        "the evaluation was cancelled before it produced a value",
        "re-run the evaluation; a partial enumeration is not a result",
    )
}

pub(crate) fn limit_exceeded(what: &str, bound: u64) -> SequentialEvaluationError {
    refusal!(
        "invalid_argument",
        "sequential_evaluation.limit_exceeded",
        format!("{what} exceeds the bound of {bound}"),
        "reduce the horizon, the policy table or the data, or raise the evaluation limit",
    )
}

/// The population and period a policy value is requested for.
///
/// A value is a statement about the model's own population and period. A
/// request for a different population or period is a transport request and is
/// refused with the missing source and selection contract named.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SequentialRequest {
    /// Population the value is requested for; `None` is the model's.
    pub target_population: Option<String>,
    /// Period the value is requested for; `None` is the model's.
    pub target_period: Option<String>,
}

impl SequentialRequest {
    /// Check the request is about the model's own population and period.
    ///
    /// # Errors
    /// A different population or period: the typed transport refusal naming
    /// the contract that is missing.
    pub fn check(&self, model: &FiniteSequentialModel) -> SequentialResult<()> {
        if let Some(population) = &self.target_population {
            if population != &model.population {
                return Err(refusal!(
                    "transport_missing_evidence",
                    "sequential_evaluation.transported_policy_contract_missing",
                    format!(
                        "the policy value was requested for population `{population}` but the model describes `{}`",
                        model.population
                    ),
                    "supply a source-to-target selection diagram with the transition and observation mechanisms marked invariant and a target initial-state law with support overlap, through counterfactual transport; or declare a model native to the target population",
                ));
            }
        }
        if let Some(period) = &self.target_period {
            if period != &model.period {
                return Err(refusal!(
                    "transport_missing_evidence",
                    "sequential_evaluation.temporal_transport_contract_missing",
                    format!(
                        "the policy value was requested for period `{period}` but the model describes `{}`",
                        model.period
                    ),
                    "declare a model established for the target period, or supply a time-invariance contract for the transition and observation mechanisms across the periods; temporal transport is not a bounded route here",
                ));
            }
        }
        Ok(())
    }
}

/// Refuse any interval, posterior or coverage statement for a policy value.
///
/// # Errors
/// Always: no coverage record measures the whole sequential procedure.
pub fn request_interval() -> SequentialResult<std::convert::Infallible> {
    Err(refusal!(
        "cell_not_licensed",
        "sequential_evaluation.interval_not_licensed",
        "no interval, posterior or coverage statement is licensed for a sequential policy value",
        "the whole sequential procedure (model, policy, estimator) needs a measured coverage record; the point value and the uncalibrated diagnostics are reported",
    ))
}
