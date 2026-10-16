//! Bayes belief updates and the once-only record of observations and actions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::{
    EvidenceLedger, LedgerError, ObservationId, SourceDeclaration, SourceKind,
};

use antecedent_core::CanonicalWriter;

use super::model::{Compiled, CompiledAction};
use super::policy::HistoryKey;
use super::{FiniteSequentialModel, SequentialEvaluationError, SequentialResult};

/// One observation outcome of an action: its probability and the posterior.
#[derive(Clone, Debug)]
pub(crate) struct Branch {
    pub(crate) symbol: usize,
    pub(crate) probability: f64,
    pub(crate) posterior: Vec<f64>,
}

/// Predictive law of the state after `action`, before the observation.
pub(crate) fn predict(compiled: &Compiled, belief: &[f64], action: usize) -> Vec<f64> {
    evolve(&compiled.actions[action], belief, compiled.states)
}

/// One application of an action's transition law to a belief.
pub(crate) fn evolve(action: &CompiledAction, belief: &[f64], states: usize) -> Vec<f64> {
    let mut out = vec![0.0; states];
    for (from, &mass) in belief.iter().enumerate() {
        let row = &action.transition[from * states..(from + 1) * states];
        for (slot, &probability) in out.iter_mut().zip(row) {
            *slot += mass * probability;
        }
    }
    out
}

/// The posterior and probability of one observation symbol, if it has
/// positive probability.
pub(crate) fn branch_for(
    compiled: &Compiled,
    belief: &[f64],
    action: usize,
    symbol: usize,
) -> Option<Branch> {
    let predicted = predict(compiled, belief, action);
    branch_from_prediction(compiled, &predicted, action, symbol)
}

fn branch_from_prediction(
    compiled: &Compiled,
    predicted: &[f64],
    action: usize,
    symbol: usize,
) -> Option<Branch> {
    let n = compiled.states;
    let spec = &compiled.actions[action];
    let row = spec.likelihood.get(symbol * n..(symbol + 1) * n)?;
    let joint: Vec<f64> = predicted.iter().zip(row).map(|(p, l)| p * l).collect();
    let probability: f64 = joint.iter().sum();
    if probability.is_nan() || probability <= 0.0 {
        return None;
    }
    let posterior = joint.iter().map(|mass| mass / probability).collect();
    Some(Branch { symbol, probability, posterior })
}

/// Every observation outcome of `action` with positive probability, in symbol
/// order.
pub(crate) fn branches(compiled: &Compiled, belief: &[f64], action: usize) -> Vec<Branch> {
    let predicted = predict(compiled, belief, action);
    let symbols = compiled.actions[action].symbols;
    (0..symbols)
        .filter_map(|symbol| branch_from_prediction(compiled, &predicted, action, symbol))
        .collect()
}

const RECORD_DOMAIN: &str = "antecedent.sequential_evaluation.record";

/// Ledger identity of one action or observation record.
pub(crate) fn record_id(scope: &str, record: &str, role: &str) -> ObservationId {
    let mut writer = CanonicalWriter::new(RECORD_DOMAIN);
    writer.text(scope).text(record).text(role);
    ObservationId::from_key(RECORD_DOMAIN, &writer.finish())
}

/// Map a ledger refusal onto the module's refusal, keeping its code.
pub(crate) fn from_ledger(error: LedgerError) -> SequentialEvaluationError {
    SequentialEvaluationError::new(
        error.code,
        error.detail,
        error.message,
        "declare each observation and action once; do not re-record evidence the ledger already holds",
    )
}

/// What one recorded observation did to the belief.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationReceipt {
    /// Caller identity of the record (unique within the tracker's session).
    pub record_id: String,
    /// Decision time at which the action was taken.
    pub time: usize,
    /// Action taken.
    pub action: usize,
    /// Observation symbol received.
    pub symbol: usize,
    /// Label of the symbol.
    pub symbol_label: String,
    /// Probability of the symbol given the history, under the model.
    pub probability: f64,
    /// Posterior over the latent states.
    pub belief: Vec<f64>,
    /// Hex identity of the measurement model the likelihood came from; absent
    /// for an action with no channel.
    pub measurement_identity: Option<String>,
    /// Ledger identity of the recorded action (hex).
    pub action_record: String,
    /// Ledger identity of the recorded observation (hex).
    pub observation_record: String,
    /// Always `false`: the declared measurement parameters are used as exact.
    pub parameter_uncertainty_propagated: bool,
}

/// Tracks the belief of one realized history, recording each action and each
/// observation once in an evidence ledger.
///
/// This is the surface for collecting *new* observations. Evaluating a fixed
/// policy needs none, and recording an observation never changes the policy.
#[derive(Clone, Debug)]
pub struct BeliefTracker {
    compiled: Compiled,
    session: String,
    belief: Vec<f64>,
    history: HistoryKey,
    resource_used: f64,
    receipts: Vec<ObservationReceipt>,
}

impl BeliefTracker {
    /// Start a session at the model's prior.
    ///
    /// # Errors
    /// A malformed model, or a blank session label.
    pub fn new(
        model: &FiniteSequentialModel,
        session: impl Into<String>,
    ) -> SequentialResult<Self> {
        let session = session.into();
        if session.trim().is_empty() {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.session_label",
                "a belief session needs a non-blank label",
                "label the session so its records are scoped to it",
            ));
        }
        let compiled = model.compile()?;
        let belief = compiled.initial.clone();
        Ok(Self {
            compiled,
            session,
            belief,
            history: HistoryKey::root(),
            resource_used: 0.0,
            receipts: Vec::new(),
        })
    }

    /// The current posterior over the latent states.
    #[must_use]
    pub fn belief(&self) -> &[f64] {
        &self.belief
    }

    /// The realized history so far.
    #[must_use]
    pub fn history(&self) -> &HistoryKey {
        &self.history
    }

    /// The decision time the next action is taken at.
    #[must_use]
    pub fn time(&self) -> usize {
        self.history.len()
    }

    /// Resource consumed so far.
    #[must_use]
    pub fn resource_used(&self) -> f64 {
        self.resource_used
    }

    /// Every recorded observation, in order.
    #[must_use]
    pub fn receipts(&self) -> &[ObservationReceipt] {
        &self.receipts
    }

    /// Identity of the model the session updates under.
    #[must_use]
    pub fn model_identity(&self) -> &str {
        &self.compiled.identity
    }

    /// Record one action and the observation it produced, and update the
    /// belief.
    ///
    /// # Errors
    /// A commit action or one past the horizon; an action the model does not
    /// support in this history; a resource budget exceeded; a symbol outside
    /// the channel or one the model gives zero probability; and a record
    /// already held by the ledger or by this session (double counting).
    #[allow(
        clippy::too_many_lines,
        reason = "a single belief update: normalisation, likelihood weighting and the support checks share one pass"
    )]
    pub fn observe(
        &mut self,
        ledger: &mut EvidenceLedger,
        action: usize,
        symbol: usize,
        record: &str,
    ) -> SequentialResult<&ObservationReceipt> {
        if record.trim().is_empty() {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.record_identity",
                "an observation record needs a non-blank identity",
                "identify each record so it can be counted once",
            ));
        }
        let time = self.history.len();
        if time >= self.compiled.horizon {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.past_horizon",
                format!("the session is at decision time {time}, the model's horizon"),
                "no further decision is taken; evaluate the realized history",
            ));
        }
        let Some(spec) = self.compiled.actions.get(action) else {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_index",
                format!("action {action} is not declared"),
                "name a declared action",
            ));
        };
        if spec.role != super::ActionRole::Continue {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.commit_has_no_observation",
                format!(
                    "`{}` is a commit action; it ends the episode and has no observation",
                    spec.label
                ),
                "record observations of continue actions only",
            ));
        }
        if !self.compiled.support.is_supported(&self.history, action) {
            return Err(refusal!(
                "history_support_violated",
                "sequential_evaluation.action_not_supported",
                format!("action `{}` has no support in history {}", spec.label, self.history),
                "restrict the policy to supported actions, or declare the support evidence for this history",
            ));
        }
        let used = self.resource_used + spec.resource_use;
        if self.compiled.budget.is_some_and(|budget| used > budget + 1e-12) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.resource_budget_exceeded",
                format!(
                    "`{}` would use {used} of a budget of {:?}",
                    spec.label, self.compiled.budget
                ),
                "raise the declared budget or take a cheaper action",
            ));
        }
        let Some(branch) = branch_for(&self.compiled, &self.belief, action, symbol) else {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.observation_impossible",
                format!(
                    "symbol {symbol} of `{}` has zero probability (or is outside the channel) under the model in history {}",
                    spec.label, self.history
                ),
                "the observation contradicts the declared model; revise the model rather than conditioning on it",
            ));
        };
        let action_id = record_id(&self.session, record, "action");
        let observation_id = record_id(&self.session, record, "observation");
        if ledger.holds(action_id) || ledger.holds(observation_id) {
            return Err(refusal!(
                "evidence_double_counted",
                "sequential_evaluation.record_already_counted",
                format!(
                    "record `{record}` of session `{}` is already in the evidence ledger",
                    self.session
                ),
                "record each observation and action once; cite the existing record instead of adding it again",
            ));
        }
        let declaration = SourceDeclaration::new(
            format!("recorded_observation:{}:{record}", self.session),
            SourceKind::ObservationSet,
        )
        .with_observations([action_id, observation_id]);
        ledger.add_source(declaration).map_err(from_ledger)?;
        let receipt = ObservationReceipt {
            record_id: record.to_owned(),
            time,
            action,
            symbol,
            symbol_label: spec.symbol_labels[symbol].clone(),
            probability: branch.probability,
            belief: branch.posterior.clone(),
            measurement_identity: spec.channel_identity.clone(),
            action_record: action_id.to_hex(),
            observation_record: observation_id.to_hex(),
            parameter_uncertainty_propagated: false,
        };
        self.resource_used = used;
        self.history = self.history.extended(action, symbol);
        self.belief = branch.posterior;
        self.receipts.push(receipt);
        Ok(self.receipts.last().expect("a receipt was just pushed"))
    }
}
