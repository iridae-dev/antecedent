//! The dynamic finite model a policy is evaluated in.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::measurement::model_refusal;
use antecedent_core::CanonicalWriter;
use antecedent_core::measurement_model::{
    ArmBinding, ErrorStructure, MeasurementModel, MeasurementParameters, MeasurementRoute,
};

use super::policy::HistoryKey;
use super::{
    MAX_ACTIONS, MAX_HORIZON, MAX_LABEL_BYTES, MAX_MODEL_WORK, MAX_STATES, PROBABILITY_TOLERANCE,
    SEQUENTIAL_EVALUATION_IDENTITY_VERSION, SequentialEvaluationError, SequentialResult,
};
use antecedent_core::named_variants;

/// What taking an action does to the episode.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ActionRole {
    /// The episode continues: the state evolves, an observation is emitted
    /// through the action's channel (when it has one) and the belief updates.
    Continue,
    /// The episode ends: the action's outcome (immediate and delayed) is
    /// realized and no further decision is made.
    Commit,
}

named_variants! {
    ActionRole {
        Continue => "continue",
        Commit => "commit",
    }
}

impl ActionRole {
    const fn tag(self) -> u64 {
        match self {
            Self::Continue => 1,
            Self::Commit => 2,
        }
    }
}

/// What happens to an outcome that falls due after the horizon while the
/// episode is still running.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CensoringRule {
    /// Refuse the evaluation: the policy's value is not defined without the
    /// outcome. This is the safe default.
    Refuse,
    /// The outcome is forfeited (counted as zero) and reported per history.
    Forfeit,
}

named_variants! {
    CensoringRule {
        Refuse => "refuse",
        Forfeit => "forfeit",
    }
}

impl CensoringRule {
    const fn tag(self) -> u64 {
        match self {
            Self::Refuse => 1,
            Self::Forfeit => 2,
        }
    }
}

/// An outcome realized some decision times after the action that causes it.
#[derive(Clone, Debug, PartialEq)]
pub struct DelayedOutcome {
    /// Decision times between the action and the outcome (at least one).
    pub delay: usize,
    /// Utility per latent state, read at the decision time the outcome falls
    /// due.
    pub utility: Vec<f64>,
}

/// How an action's observation relates to the latent state: a typed
/// measurement model whose construct is the model's state variable.
///
/// The binary and categorical confusion routes are accepted; the confusion
/// law is the likelihood of the belief update. Error that differs by arm,
/// population or time is refused, never pooled.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationChannel {
    /// The measurement model; its identity is bound into every receipt.
    pub measurement: MeasurementModel,
}

/// One declared action.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionSpec {
    /// Unique label.
    pub label: String,
    /// Whether the action continues or ends the episode.
    pub role: ActionRole,
    /// Utility deducted when the action is taken (non-negative).
    pub utility_cost: f64,
    /// Resource consumed against the model's budget (non-negative).
    pub resource_use: f64,
    /// Row-major `states x states` transition law `P(next | current)` applied
    /// when the action is taken; for a commit action it governs the state's
    /// evolution after the commit, until the delayed outcomes fall due.
    pub transition: Vec<f64>,
    /// The observation the action produces; `None` produces no signal.
    pub channel: Option<ObservationChannel>,
    /// Utility per latent state realized when the action is taken.
    pub immediate_utility: Vec<f64>,
    /// An outcome realized later.
    pub delayed: Option<DelayedOutcome>,
}

/// A declared stopping boundary on the belief.
///
/// Before the policy is consulted, if the posterior mass on `states` is at
/// least `upper` or at most `lower`, the episode stops by taking
/// `stop_action`, which must be a commit action. The boundary is part of the
/// problem declaration and applies to every policy equally.
#[derive(Clone, Debug, PartialEq)]
pub struct StoppingBoundary {
    /// Unique label.
    pub label: String,
    /// Latent states whose posterior mass is monitored.
    pub states: Vec<usize>,
    /// Stop when the mass is at least this.
    pub upper: Option<f64>,
    /// Stop when the mass is at most this.
    pub lower: Option<f64>,
    /// Commit action taken when the boundary is crossed.
    pub stop_action: usize,
}

/// Where each action is supported by the evidence behind the model: the
/// positivity declaration.
///
/// `default_allowed` is the support in every history without an override. A
/// policy that takes an unsupported action in a history it can reach is
/// refused rather than extrapolated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionSupport {
    default_allowed: Vec<bool>,
    overrides: BTreeMap<HistoryKey, Vec<bool>>,
}

impl ActionSupport {
    /// Every action supported in every history.
    #[must_use]
    pub fn everywhere(actions: usize) -> Self {
        Self { default_allowed: vec![true; actions], overrides: BTreeMap::new() }
    }

    /// Support by action, the same in every history.
    #[must_use]
    pub fn by_action(default_allowed: Vec<bool>) -> Self {
        Self { default_allowed, overrides: BTreeMap::new() }
    }

    /// Replace the support in one history.
    #[must_use]
    pub fn with_history(mut self, history: HistoryKey, allowed: Vec<bool>) -> Self {
        self.overrides.insert(history, allowed);
        self
    }

    /// Whether `action` is supported in `history`.
    #[must_use]
    pub fn is_supported(&self, history: &HistoryKey, action: usize) -> bool {
        self.overrides
            .get(history)
            .unwrap_or(&self.default_allowed)
            .get(action)
            .copied()
            .unwrap_or(false)
    }

    /// The support applied to histories without an override.
    #[must_use]
    pub fn default_allowed(&self) -> &[bool] {
        &self.default_allowed
    }

    /// The per-history overrides, in canonical history order.
    pub fn overrides(&self) -> impl Iterator<Item = (&HistoryKey, &[bool])> {
        self.overrides.iter().map(|(key, allowed)| (key, allowed.as_slice()))
    }
}

/// A finite-horizon dynamic finite model: discrete latent states, discrete
/// actions, Bayes beliefs through declared measurement models.
///
/// The per-action transition and observation laws are the interventional
/// (do-) laws of the dynamic structural model. Supplying them as such is the
/// exchangeability premise of the g-formula: no unmeasured cause of both the
/// action choice and later states beyond the information set.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteSequentialModel {
    /// Display label; not part of the identity.
    pub label: String,
    /// Population the laws were established for.
    pub population: String,
    /// Regime (setting under which the laws apply).
    pub regime: String,
    /// Period the laws were established for.
    pub period: String,
    /// Identity of the latent state variable; the construct every
    /// observation channel measures.
    pub state_variable: String,
    /// Latent state names.
    pub states: Vec<String>,
    /// Initial law over the states.
    pub initial: Vec<f64>,
    /// Number of decision times.
    pub horizon: usize,
    /// Declared actions.
    pub actions: Vec<ActionSpec>,
    /// Utility per state when the horizon is reached without a commit.
    pub terminal_utility: Vec<f64>,
    /// What happens to outcomes due after the horizon.
    pub censoring: CensoringRule,
    /// Resource budget over the whole episode.
    pub resource_budget: Option<f64>,
    /// Declared stopping boundaries, checked in order.
    pub stopping: Vec<StoppingBoundary>,
    /// Where each action is supported.
    pub support: ActionSupport,
}

// ---------------------------------------------------------------------------
// Compiled form
// ---------------------------------------------------------------------------

/// A validated action with its likelihood matrix resolved.
#[derive(Clone, Debug)]
pub(crate) struct CompiledAction {
    pub(crate) label: String,
    pub(crate) role: ActionRole,
    pub(crate) utility_cost: f64,
    pub(crate) resource_use: f64,
    pub(crate) transition: Vec<f64>,
    /// Number of observation symbols (one when the action has no channel).
    pub(crate) symbols: usize,
    /// Row-major `symbols x states`: `P(symbol | next state)`.
    pub(crate) likelihood: Vec<f64>,
    pub(crate) has_channel: bool,
    pub(crate) immediate: Vec<f64>,
    pub(crate) delayed: Option<(usize, Vec<f64>)>,
    /// Hex identity of the bound measurement model.
    pub(crate) channel_identity: Option<String>,
    /// Labels of the symbols (levels of the confusion model), if any.
    pub(crate) symbol_labels: Vec<String>,
}

/// A validated model ready for enumeration.
#[derive(Clone, Debug)]
pub(crate) struct Compiled {
    pub(crate) states: usize,
    pub(crate) horizon: usize,
    pub(crate) initial: Vec<f64>,
    pub(crate) actions: Vec<CompiledAction>,
    pub(crate) terminal: Vec<f64>,
    pub(crate) censoring: CensoringRule,
    pub(crate) budget: Option<f64>,
    pub(crate) boundaries: Vec<StoppingBoundary>,
    pub(crate) support: ActionSupport,
    pub(crate) identity: String,
}

impl Compiled {
    /// Posterior mass the boundary monitors under `belief`.
    pub(crate) fn boundary_mass(boundary: &StoppingBoundary, belief: &[f64]) -> f64 {
        boundary.states.iter().map(|&state| belief[state]).sum()
    }

    /// The first boundary the belief has crossed, if any.
    pub(crate) fn crossed_boundary(&self, belief: &[f64]) -> Option<usize> {
        self.boundaries.iter().position(|boundary| {
            let mass = Self::boundary_mass(boundary, belief);
            boundary.upper.is_some_and(|upper| mass >= upper)
                || boundary.lower.is_some_and(|lower| mass <= lower)
        })
    }
}

pub(crate) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn text_ok(text: &str) -> bool {
    !text.trim().is_empty() && text.len() <= MAX_LABEL_BYTES
}

fn finite_all(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn distribution_ok(values: &[f64], expected: usize) -> bool {
    values.len() == expected
        && values.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0)
        && (values.iter().sum::<f64>() - 1.0).abs() <= PROBABILITY_TOLERANCE
}

fn matrix_ok(values: &[f64], states: usize) -> bool {
    values.len() == states * states
        && values.chunks_exact(states).all(|row| distribution_ok(row, states))
}

impl FiniteSequentialModel {
    /// Check the model is well formed and its channels are licensed.
    ///
    /// # Errors
    /// The first malformed declaration, as a typed refusal naming the field
    /// and what would make it well formed.
    pub fn validate(&self) -> SequentialResult<()> {
        self.compile().map(|_| ())
    }

    /// Canonical identity of the model: laws, bounds, boundaries, support and
    /// the identities of the bound measurement models. The display label is
    /// not part of it.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.sequential_evaluation.model");
        writer
            .word(SEQUENTIAL_EVALUATION_IDENTITY_VERSION)
            .text(&self.population)
            .text(&self.regime)
            .text(&self.period)
            .text(&self.state_variable)
            .texts(&self.states)
            .reals(&self.initial)
            .word(self.horizon as u64)
            .reals(&self.terminal_utility)
            .word(self.censoring.tag())
            .opt_real(self.resource_budget);
        writer.word(self.actions.len() as u64);
        for action in &self.actions {
            writer
                .text(&action.label)
                .word(action.role.tag())
                .real(action.utility_cost)
                .real(action.resource_use)
                .reals(&action.transition)
                .reals(&action.immediate_utility);
            match &action.channel {
                Some(channel) => {
                    writer.word(1).bytes(channel.measurement.identities().model.as_bytes());
                }
                None => {
                    writer.word(0);
                }
            }
            match &action.delayed {
                Some(delayed) => {
                    writer.word(1).word(delayed.delay as u64).reals(&delayed.utility);
                }
                None => {
                    writer.word(0);
                }
            }
        }
        writer.word(self.stopping.len() as u64);
        for boundary in &self.stopping {
            writer.text(&boundary.label).word(boundary.states.len() as u64);
            for state in &boundary.states {
                writer.word(*state as u64);
            }
            writer
                .opt_real(boundary.upper)
                .opt_real(boundary.lower)
                .word(boundary.stop_action as u64);
        }
        writer.word(self.support.default_allowed.len() as u64);
        for allowed in &self.support.default_allowed {
            writer.word(u64::from(*allowed));
        }
        writer.word(self.support.overrides.len() as u64);
        for (history, allowed) in &self.support.overrides {
            history.write(&mut writer);
            writer.word(allowed.len() as u64);
            for flag in allowed {
                writer.word(u64::from(*flag));
            }
        }
        writer.finish_hex()
    }

    pub(crate) fn compile(&self) -> SequentialResult<Compiled> {
        self.check_header()?;
        let states = self.states.len();
        let mut actions = Vec::with_capacity(self.actions.len());
        for action in &self.actions {
            actions.push(self.compile_action(action, states)?);
        }
        self.check_boundaries(&actions, states)?;
        self.check_support()?;
        Ok(Compiled {
            states,
            horizon: self.horizon,
            initial: self.initial.clone(),
            actions,
            terminal: self.terminal_utility.clone(),
            censoring: self.censoring,
            budget: self.resource_budget,
            boundaries: self.stopping.clone(),
            support: self.support.clone(),
            identity: self.identity_hex(),
        })
    }

    fn check_header(&self) -> SequentialResult<()> {
        for (name, value) in [
            ("population", &self.population),
            ("regime", &self.regime),
            ("period", &self.period),
            ("state_variable", &self.state_variable),
        ] {
            if !text_ok(value) {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.model_field",
                    format!(
                        "the model's `{name}` must be non-blank and at most {MAX_LABEL_BYTES} bytes"
                    ),
                    "name the population, regime, period and latent state variable the laws were established for",
                ));
            }
        }
        let states = self.states.len();
        if !(2..=MAX_STATES).contains(&states) || self.states.iter().any(|s| !text_ok(s)) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.model_states",
                format!("a model needs 2 to {MAX_STATES} named latent states (got {states})"),
                "declare the finite latent state space",
            ));
        }
        let mut seen: Vec<&str> = self.states.iter().map(String::as_str).collect();
        seen.sort_unstable();
        if seen.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.model_states",
                "state names repeat",
                "give every latent state a distinct name",
            ));
        }
        if !distribution_ok(&self.initial, states) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.model_initial",
                "the initial law must be one probability per state summing to one",
                "declare the prior over the latent states",
            ));
        }
        if !(1..=MAX_HORIZON).contains(&self.horizon) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.model_horizon",
                format!("the horizon must be between 1 and {MAX_HORIZON} decision times"),
                "freeze a bounded finite horizon",
            ));
        }
        if self.actions.is_empty() || self.actions.len() > MAX_ACTIONS {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.model_actions",
                format!("a model needs 1 to {MAX_ACTIONS} actions"),
                "declare the finite action set",
            ));
        }
        let work = (states as u128)
            .saturating_mul(states as u128)
            .saturating_mul(self.actions.len() as u128)
            .saturating_mul(self.horizon as u128);
        if work > MAX_MODEL_WORK {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.model_work",
                format!(
                    "the model's states x states x actions x horizon is {work}, over the \
                     budget of {MAX_MODEL_WORK}"
                ),
                "reduce the latent states, the actions or the horizon",
            ));
        }
        if self.terminal_utility.len() != states || !finite_all(&self.terminal_utility) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.model_terminal_utility",
                "the terminal utility needs one finite value per state",
                "declare the utility when the horizon is reached without a commit",
            ));
        }
        if let Some(budget) = self.resource_budget {
            if !(budget.is_finite() && budget >= 0.0) {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.model_resource_budget",
                    "the resource budget must be finite and non-negative",
                    "declare a non-negative budget or none",
                ));
            }
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one flat compilation of an action over every declared state and observation, sharing the bound counters"
    )]
    fn compile_action(
        &self,
        action: &ActionSpec,
        states: usize,
    ) -> SequentialResult<CompiledAction> {
        if !text_ok(&action.label) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_label",
                "an action needs a non-blank label of bounded length",
                "label every action",
            ));
        }
        if self.actions.iter().filter(|other| other.label == action.label).count() != 1 {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_label",
                format!("action label `{}` is not unique", action.label),
                "give every action a distinct label",
            ));
        }
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        if !non_negative(action.utility_cost) || !non_negative(action.resource_use) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_cost",
                format!(
                    "action `{}` has a negative or non-finite cost or resource use",
                    action.label
                ),
                "declare costs and resource use as non-negative finite numbers",
            ));
        }
        if !matrix_ok(&action.transition, states) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_transition",
                format!(
                    "the transition of `{}` is not a row-stochastic states x states matrix",
                    action.label
                ),
                "declare one distribution over next states per current state",
            ));
        }
        if action.immediate_utility.len() != states || !finite_all(&action.immediate_utility) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_utility",
                format!("`{}` needs one finite immediate utility per state", action.label),
                "declare the immediate utility by latent state",
            ));
        }
        let delayed = match &action.delayed {
            Some(delayed) => {
                if !(1..=MAX_HORIZON).contains(&delayed.delay)
                    || delayed.utility.len() != states
                    || !finite_all(&delayed.utility)
                {
                    return Err(refusal!(
                        "invalid_argument",
                        "sequential_evaluation.action_delayed_outcome",
                        format!(
                            "the delayed outcome of `{}` needs a delay of 1 to {MAX_HORIZON} decision times and one finite utility per state",
                            action.label
                        ),
                        "declare when the outcome falls due and its utility by state",
                    ));
                }
                Some((delayed.delay, delayed.utility.clone()))
            }
            None => None,
        };
        let (symbols, likelihood, identity, symbol_labels) = match (&action.channel, action.role) {
            (Some(_), ActionRole::Commit) => {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.action_channel",
                    format!(
                        "commit action `{}` ends the episode and cannot emit an observation",
                        action.label
                    ),
                    "attach observation channels to continue actions only",
                ));
            }
            (Some(channel), ActionRole::Continue) => {
                let (symbols, likelihood, labels) = self.channel_matrix(action, channel)?;
                let identity = channel.measurement.identities().model.to_hex();
                (symbols, likelihood, Some(identity), labels)
            }
            (None, _) => (1, vec![1.0; states], None, vec!["no_signal".to_owned()]),
        };
        Ok(CompiledAction {
            label: action.label.clone(),
            role: action.role,
            utility_cost: action.utility_cost,
            resource_use: action.resource_use,
            transition: action.transition.clone(),
            symbols,
            likelihood,
            has_channel: action.channel.is_some(),
            immediate: action.immediate_utility.clone(),
            delayed,
            channel_identity: identity,
            symbol_labels,
        })
    }

    /// Resolve a channel's likelihood matrix after the binding checks.
    ///
    /// The binding is checked here, against the model's construct, population,
    /// regime and arm, with the same refusal codes the measurement-model
    /// layer uses. The route's `condition_on_construct` card refuses a
    /// population-level correction used for per-unit conditioning; the belief
    /// update is a Bayes update on a declared finite model with the confusion
    /// law as its likelihood, and the parameters are used as declared, without
    /// propagating their estimation uncertainty.
    #[allow(
        clippy::too_many_lines,
        reason = "one flat construction of the channel matrix with its normalisation and support refusals"
    )]
    fn channel_matrix(
        &self,
        action: &ActionSpec,
        channel: &ObservationChannel,
    ) -> SequentialResult<(usize, Vec<f64>, Vec<String>)> {
        let model = &channel.measurement;
        model.validate().map_err(|error| SequentialEvaluationError::from(model_refusal(&error)))?;
        if model.indicator.id == self.state_variable {
            return Err(SequentialEvaluationError::new(
                antecedent_core::reason_code!("proxy_offered_as_construct"),
                "sequential_evaluation.indicator_as_state",
                format!(
                    "the channel of `{}` names the measured column `{}` as the latent state variable",
                    action.label, model.indicator.id
                ),
                "name the latent construct as the state variable and relate the measured column to it by the measurement model",
            ));
        }
        if model.construct.id != self.state_variable {
            return Err(refusal!(
                "measurement_binding_mismatch",
                "sequential_evaluation.channel_construct_mismatch",
                format!(
                    "the channel of `{}` measures `{}`, not the model's state variable `{}`",
                    action.label, model.construct.id, self.state_variable
                ),
                "bind the measurement model to the construct the states describe",
            ));
        }
        if model.binding.population_id != self.population
            || model.binding.regime_id != self.regime
            || model.binding.arm != ArmBinding::AllArms
        {
            return Err(refusal!(
                "measurement_binding_mismatch",
                "sequential_evaluation.channel_binding_mismatch",
                format!(
                    "the channel of `{}` was established for population `{}`, regime `{}`; the model is for `{}`, `{}`",
                    action.label,
                    model.binding.population_id,
                    model.binding.regime_id,
                    self.population,
                    self.regime
                ),
                "establish the measurement model for this population and regime, or declare the invariance premise under a transport contract",
            ));
        }
        if model.structure != ErrorStructure::NonDifferential {
            return Err(SequentialEvaluationError::new(
                antecedent_core::reason_code!("differential_error_not_pooled"),
                "sequential_evaluation.differential_channel",
                format!("the channel of `{}` is declared differential", action.label),
                "declare non-differential error; error that differs by arm, population or time has no bounded route",
            ));
        }
        let states = self.states.len();
        match (&model.route, &model.parameters) {
            (
                MeasurementRoute::BinaryMisclassification,
                MeasurementParameters::Binary { rates, .. },
            ) => {
                let [rates] = rates.as_slice() else {
                    return Err(refusal!(
                        "route_not_supported",
                        "sequential_evaluation.channel_rates",
                        "a binary channel needs exactly one shared sensitivity and specificity",
                        "declare shared non-differential rates",
                    ));
                };
                if states != 2 || rates.arm_id.is_some() {
                    return Err(refusal!(
                        "route_not_supported",
                        "sequential_evaluation.channel_rates",
                        "a binary channel needs a two-state model and shared rates",
                        "use a categorical channel for more than two states",
                    ));
                }
                let (sens, spec) = (rates.sensitivity, rates.specificity);
                Ok((
                    2,
                    vec![spec, 1.0 - sens, 1.0 - spec, sens],
                    vec!["negative".to_owned(), "positive".to_owned()],
                ))
            }
            (
                MeasurementRoute::ProbabilisticCategorical,
                MeasurementParameters::Categorical { levels, confusion, .. },
            ) => {
                if levels != &self.states {
                    return Err(refusal!(
                        "measurement_binding_mismatch",
                        "sequential_evaluation.channel_levels_mismatch",
                        format!(
                            "the confusion levels of `{}` are not the model's states in order",
                            action.label
                        ),
                        "declare the confusion model over exactly the model's latent states, in the same order",
                    ));
                }
                Ok((states, confusion.clone(), levels.clone()))
            }
            _ => Err(refusal!(
                "route_not_supported",
                "sequential_evaluation.channel_route",
                format!(
                    "the channel of `{}` uses the `{}` route; only binary misclassification and categorical confusion supply a finite likelihood",
                    action.label,
                    model.route.as_str()
                ),
                "declare a binary or categorical measurement model for the observation",
            )),
        }
    }

    fn check_boundaries(&self, actions: &[CompiledAction], states: usize) -> SequentialResult<()> {
        for boundary in &self.stopping {
            let mut members = boundary.states.clone();
            members.sort_unstable();
            let distinct = members.windows(2).all(|pair| pair[0] != pair[1]);
            let unit = |value: Option<f64>| value.is_none_or(|v| (0.0..=1.0).contains(&v));
            let ordered = match (boundary.lower, boundary.upper) {
                (Some(lower), Some(upper)) => lower < upper,
                _ => true,
            };
            let stop_is_commit =
                actions.get(boundary.stop_action).is_some_and(|a| a.role == ActionRole::Commit);
            if !text_ok(&boundary.label)
                || boundary.states.is_empty()
                || !distinct
                || members.last().is_some_and(|&m| m >= states)
                || (boundary.upper.is_none() && boundary.lower.is_none())
                || !unit(boundary.upper)
                || !unit(boundary.lower)
                || !ordered
                || !stop_is_commit
            {
                return Err(refusal!(
                    "invalid_argument",
                    "sequential_evaluation.stopping_boundary",
                    format!("stopping boundary `{}` is not well formed", boundary.label),
                    "monitor distinct in-range states with thresholds in [0, 1], lower below upper, and stop with a commit action",
                ));
            }
        }
        Ok(())
    }

    fn check_support(&self) -> SequentialResult<()> {
        let actions = self.actions.len();
        let mut lengths = vec![self.support.default_allowed.len()];
        lengths.extend(self.support.overrides.values().map(Vec::len));
        if lengths.iter().any(|&len| len != actions) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.action_support",
                "the action support declaration needs one flag per action",
                "declare, for the default and for each override, whether every action is supported",
            ));
        }
        Ok(())
    }
}
