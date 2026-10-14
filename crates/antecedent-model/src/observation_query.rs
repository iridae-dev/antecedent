//! Observation-conditioned queries on exact finite laws and declared Bayes
//! models.
//!
//! Conditioning on an observed value and intervening on a variable are two
//! different operations. On a composed finite model both are computable and
//! they generally disagree: `P(Y | S = s)` reads the observational law and
//! keeps every association between `S` and `Y`, including the one a common
//! cause creates, while `P(Y | do(S = s))` is the truncated factorization that
//! cuts the arrows into `S`. This module computes each from the same
//! [`ComposedFinite`] model and never lets one stand in for the other.
//!
//! # Queries
//!
//! - [`condition`]: the exact conditional distribution of an outcome given
//!   recorded values, with the probability of the evidence.
//! - [`intervene`] and [`conditioning_and_intervention`]: the interventional
//!   distribution, and both laws for the same assignment side by side.
//! - [`condition_on_measured`]: conditioning on the *recorded value of an
//!   indicator* of a construct through the indicator's declared likelihood
//!   `P(recorded | construct)`. The construct's posterior is the Bayes
//!   inversion of that likelihood against the model's own law; the recorded
//!   value is never treated as the construct. The indicator is assumed
//!   conditionally independent of every other model variable given the
//!   construct (non-differential error), which is the premise the caller
//!   declares.
//! - [`threshold_crossing_observations`]: which observed levels of a signal
//!   move an outcome probability to or past a declared level, and how likely
//!   such an observation is.
//! - [`FiniteBayesModel`] and [`BetaBernoulli`]: the posterior of a declared
//!   finite hypothesis space and of a declared conjugate (Beta-Bernoulli)
//!   model after new evidence.
//! - [`required_bayes_factor`], [`observations_required_binary_signal`] and
//!   [`observations_required_conjugate`]: exact solves for the evidence that
//!   moves a posterior probability to a declared level. A level that no
//!   admissible evidence reaches is [`ObservationQueryError::LevelUnreachable`]
//!   and states the most extreme value that is reachable.
//!
//! # Boundary
//!
//! Every value is exact given the declared model; nothing here is a
//! statement about how often a sample-fitted model is right. A posterior or
//! decision probability read from a model fitted to a sample is a point value,
//! labelled so by the caller; no interval or calibration is produced here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::ExecutionContext;

use crate::compose::{ComposedFinite, CompositionError, FiniteLaw};

/// Largest observation count an evidence-requirement solve may search or return.
pub const MAX_REQUIRED_OBSERVATIONS: u64 = 1 << 24;
/// Most hypotheses a declared finite Bayes model may hold.
pub const MAX_BAYES_HYPOTHESES: usize = 256;
/// Most observation levels a declared finite Bayes model may hold.
pub const MAX_BAYES_OBSERVATION_LEVELS: usize = 256;
/// Most observations one posterior update may consume.
pub const MAX_BAYES_SEQUENCE: usize = 1 << 20;
/// Slack, in log-odds (or probability for the conjugate solve), within which a
/// level counts as reached. The boundary itself is reached.
pub const LEVEL_TOLERANCE: f64 = 1e-12;
/// Slack allowed when a declared distribution is checked to sum to one.
pub const SIMPLEX_TOLERANCE: f64 = 1e-9;

/// Which way a probability is asked to move.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Direction {
    /// Reach the level or exceed it.
    Raise,
    /// Reach the level or fall below it.
    Lower,
}

impl Direction {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Raise => "raise",
            Self::Lower => "lower",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "raise" => Some(Self::Raise),
            "lower" => Some(Self::Lower),
            _ => None,
        }
    }
}

/// Why no admissible evidence reaches a declared level.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum UnreachableBound {
    /// The declared signal does not move the probability in the requested
    /// direction (its likelihood ratio does not favor that direction, or it
    /// carries no information).
    SignalDoesNotMove,
    /// The level is reachable only with more observations than the declared
    /// bound allows.
    ObservationBudget {
        /// The declared bound.
        max_observations: u64,
    },
    /// No observable level of the signal moves the conditional probability to
    /// the level.
    NoObservationCrosses,
}

/// Why an observation query could not be answered.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ObservationQueryError {
    /// A named variable is not in the model.
    #[error("unknown variable `{0}`")]
    UnknownVariable(String),
    /// A named level is not a level of the variable.
    #[error("unknown level `{level}` of `{variable}`")]
    UnknownLevel {
        /// Variable (or `observation` for a Bayes model's observation level).
        variable: String,
        /// The unknown level.
        level: String,
    },
    /// A variable is assigned more than once.
    #[error("variable `{0}` is assigned more than once")]
    RepeatedVariable(String),
    /// The outcome is also one of the observed values.
    #[error("outcome `{0}` is also conditioned on")]
    OutcomeObserved(String),
    /// The construct is also conditioned on directly alongside its indicator.
    #[error("construct `{0}` is also conditioned on directly")]
    ConstructObserved(String),
    /// The observed evidence has probability zero under the model, so the
    /// conditional law is undefined.
    #[error("the observed evidence has probability zero under the declared model")]
    ZeroProbabilityEvidence,
    /// A declaration is malformed.
    #[error("invalid declaration: {0}")]
    Invalid(&'static str),
    /// No admissible evidence reaches the level.
    #[error(
        "level {level} cannot be reached ({direction:?}); the most extreme reachable value is {extremum}"
    )]
    LevelUnreachable {
        /// The requested level.
        level: f64,
        /// The most extreme value that is reachable: a supremum when raising,
        /// an infimum when lowering.
        extremum: f64,
        /// The requested direction.
        direction: Direction,
        /// Why the level is out of reach.
        bound: UnreachableBound,
    },
    /// Cancellation was observed at a work boundary.
    #[error("observation query cancelled")]
    Cancelled,
    /// The underlying composed model refused.
    #[error("composed model: {0}")]
    Composition(CompositionError),
}

fn composition(error: CompositionError) -> ObservationQueryError {
    match error {
        CompositionError::Cancelled => ObservationQueryError::Cancelled,
        other => ObservationQueryError::Composition(other),
    }
}

fn variable_index(law: &FiniteLaw, name: &str) -> Result<usize, ObservationQueryError> {
    law.variables()
        .iter()
        .position(|v| v == name)
        .ok_or_else(|| ObservationQueryError::UnknownVariable(name.to_owned()))
}

fn level_index(
    law: &FiniteLaw,
    variable: usize,
    level: &str,
) -> Result<usize, ObservationQueryError> {
    law.levels()[variable].iter().position(|l| l == level).ok_or_else(|| {
        ObservationQueryError::UnknownLevel {
            variable: law.variables()[variable].clone(),
            level: level.to_owned(),
        }
    })
}

/// `(variable index, level index)` of each assignment; a variable twice is refused.
fn resolve_assignments(
    law: &FiniteLaw,
    assignments: &[(&str, &str)],
) -> Result<Vec<(usize, usize)>, ObservationQueryError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(assignments.len());
    for &(variable, level) in assignments {
        let v = variable_index(law, variable)?;
        if !seen.insert(v) {
            return Err(ObservationQueryError::RepeatedVariable(variable.to_owned()));
        }
        out.push((v, level_index(law, v, level)?));
    }
    Ok(out)
}

/// Sum the law over every state matching `fixed`, bucketed by the levels of
/// `bucket_vars` (row-major in the order given).
fn bucketed(
    law: &FiniteLaw,
    bucket_vars: &[usize],
    fixed: &[(usize, usize)],
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, ObservationQueryError> {
    let n = law.variables().len();
    let widths: Vec<usize> = law.levels().iter().map(Vec::len).collect();
    let mut strides = vec![1_usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * widths[i + 1];
    }
    let mut out = vec![0.0_f64; bucket_vars.iter().map(|&b| widths[b]).product::<usize>()];
    for (state, &p) in law.probabilities().iter().enumerate() {
        if state % 4096 == 4095 && ctx.cancellation.is_cancelled() {
            return Err(ObservationQueryError::Cancelled);
        }
        if p <= 0.0 {
            continue;
        }
        if fixed.iter().any(|&(v, l)| (state / strides[v]) % widths[v] != l) {
            continue;
        }
        let mut slot = 0_usize;
        for &b in bucket_vars {
            slot = slot * widths[b] + (state / strides[b]) % widths[b];
        }
        out[slot] += p;
    }
    Ok(out)
}

fn named_levels(law: &FiniteLaw, variable: usize, values: &[f64]) -> Vec<(String, f64)> {
    law.levels()[variable].iter().cloned().zip(values.iter().copied()).collect()
}

fn check_open_probability(name: &'static str, value: f64) -> Result<(), ObservationQueryError> {
    if value.is_finite() && value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(ObservationQueryError::Invalid(name))
    }
}

fn check_closed_probability(name: &'static str, value: f64) -> Result<(), ObservationQueryError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ObservationQueryError::Invalid(name))
    }
}

fn log_odds(p: f64) -> f64 {
    p.ln() - (-p).ln_1p()
}

fn from_log_odds(lo: f64) -> f64 {
    1.0 / (1.0 + (-lo).exp())
}

// ------------------------------------------------------------- conditioning

/// The exact conditional distribution of an outcome given recorded values.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalDistribution {
    /// The outcome variable.
    pub outcome: String,
    /// Probability of the evidence under the model, `P(observed)`.
    pub evidence_probability: f64,
    /// `(level, P(outcome = level | observed))` in the model's level order.
    pub levels: Vec<(String, f64)>,
}

/// `P(outcome | observed)` on an exact finite law.
///
/// # Errors
/// An unknown variable or level, a variable assigned twice, the outcome among
/// the observed variables, evidence of probability zero, or cancellation.
pub fn condition(
    law: &FiniteLaw,
    outcome: &str,
    observed: &[(&str, &str)],
    ctx: &ExecutionContext,
) -> Result<ConditionalDistribution, ObservationQueryError> {
    let y = variable_index(law, outcome)?;
    let fixed = resolve_assignments(law, observed)?;
    if fixed.iter().any(|&(v, _)| v == y) {
        return Err(ObservationQueryError::OutcomeObserved(outcome.to_owned()));
    }
    let joint = bucketed(law, &[y], &fixed, ctx)?;
    let evidence: f64 = joint.iter().sum();
    if evidence.is_nan() || evidence <= 0.0 {
        return Err(ObservationQueryError::ZeroProbabilityEvidence);
    }
    let conditional: Vec<f64> = joint.iter().map(|p| p / evidence).collect();
    Ok(ConditionalDistribution {
        outcome: outcome.to_owned(),
        evidence_probability: evidence,
        levels: named_levels(law, y, &conditional),
    })
}

/// `P(outcome | do(interventions))` by truncated factorization of the model.
///
/// # Errors
/// An unknown variable or level, a variable intervened on twice, or
/// cancellation.
pub fn intervene(
    model: &ComposedFinite,
    outcome: &str,
    interventions: &[(&str, &str)],
    ctx: &ExecutionContext,
) -> Result<Vec<(String, f64)>, ObservationQueryError> {
    let law = model.law(interventions, ctx).map_err(composition)?;
    let y = variable_index(&law, outcome)?;
    let probabilities = bucketed(&law, &[y], &[], ctx)?;
    Ok(named_levels(&law, y, &probabilities))
}

/// Conditioning and intervention on one assignment, from one model.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditioningInterventionPair {
    /// `P(outcome | assignment)` from the observational law.
    pub conditioning: ConditionalDistribution,
    /// `P(outcome | do(assignment))` from the truncated factorization.
    pub intervention: Vec<(String, f64)>,
    /// Largest absolute difference between the two over the outcome's levels.
    pub max_abs_difference: f64,
}

/// Both `P(outcome | assignment)` and `P(outcome | do(assignment))` for the
/// same assignment on the same model. Where a common cause of the assigned
/// variables and the outcome is in the model the two differ; where it is not
/// they agree; the pair reports which, and never substitutes one for the other.
///
/// # Errors
/// As [`condition`] and [`intervene`].
pub fn conditioning_and_intervention(
    model: &ComposedFinite,
    outcome: &str,
    assignment: &[(&str, &str)],
    ctx: &ExecutionContext,
) -> Result<ConditioningInterventionPair, ObservationQueryError> {
    let observational = model.law(&[], ctx).map_err(composition)?;
    let conditioning = condition(&observational, outcome, assignment, ctx)?;
    let intervention = intervene(model, outcome, assignment, ctx)?;
    let max_abs_difference = conditioning
        .levels
        .iter()
        .zip(&intervention)
        .map(|((_, a), (_, b))| (a - b).abs())
        .fold(0.0_f64, f64::max);
    Ok(ConditioningInterventionPair { conditioning, intervention, max_abs_difference })
}

// ------------------------------------------------------- measured signals

/// The answer to conditioning on the recorded value of a construct's indicator.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredConditional {
    /// `P(recorded value, other observed values)` under the model.
    pub evidence_probability: f64,
    /// `(level, P(outcome = level | recorded value, other observed values))`.
    pub outcome: Vec<(String, f64)>,
    /// `(level, P(construct = level | recorded value, other observed values))`.
    pub construct_posterior: Vec<(String, f64)>,
}

/// Condition an outcome on the recorded value of an indicator of `construct`.
///
/// `likelihood[c]` is `P(recorded value | construct = level c)` for the
/// construct's levels in the model's level order. The indicator is taken to be
/// conditionally independent of the outcome and of every other observed
/// variable given the construct, so
/// `P(y | m, o) = sum_c P(y, c, o) L(m | c) / sum_{y, c} P(y, c, o) L(m | c)`.
/// The outcome may be the construct itself, which returns its posterior.
///
/// # Errors
/// An unknown variable or level, the construct also observed directly, the
/// outcome among the observed variables, a likelihood that is misaligned or
/// outside `[0, 1]`, evidence of probability zero, or cancellation.
pub fn condition_on_measured(
    law: &FiniteLaw,
    outcome: &str,
    construct: &str,
    observed: &[(&str, &str)],
    likelihood: &[f64],
    ctx: &ExecutionContext,
) -> Result<MeasuredConditional, ObservationQueryError> {
    let y = variable_index(law, outcome)?;
    let c = variable_index(law, construct)?;
    let fixed = resolve_assignments(law, observed)?;
    if fixed.iter().any(|&(v, _)| v == c) {
        return Err(ObservationQueryError::ConstructObserved(construct.to_owned()));
    }
    if fixed.iter().any(|&(v, _)| v == y) {
        return Err(ObservationQueryError::OutcomeObserved(outcome.to_owned()));
    }
    let width_c = law.levels()[c].len();
    if likelihood.len() != width_c
        || likelihood.iter().any(|l| !l.is_finite() || !(0.0..=1.0).contains(l))
    {
        return Err(ObservationQueryError::Invalid(
            "the indicator likelihood must hold one probability per construct level",
        ));
    }
    let mut posterior = vec![0.0_f64; width_c];
    let outcome_unnormalized: Vec<f64> = if y == c {
        let joint = bucketed(law, &[c], &fixed, ctx)?;
        for (ci, p) in joint.iter().enumerate() {
            posterior[ci] = p * likelihood[ci];
        }
        posterior.clone()
    } else {
        let width_y = law.levels()[y].len();
        let joint = bucketed(law, &[y, c], &fixed, ctx)?;
        let mut numerator = vec![0.0_f64; width_y];
        for (row, total) in joint.chunks(width_c).zip(numerator.iter_mut()) {
            for ((cell, &l), post) in row.iter().zip(likelihood).zip(posterior.iter_mut()) {
                let w = cell * l;
                *total += w;
                *post += w;
            }
        }
        numerator
    };
    let evidence: f64 = posterior.iter().sum();
    if evidence.is_nan() || evidence <= 0.0 {
        return Err(ObservationQueryError::ZeroProbabilityEvidence);
    }
    let outcome_probabilities: Vec<f64> =
        outcome_unnormalized.iter().map(|w| w / evidence).collect();
    let construct_probabilities: Vec<f64> = posterior.iter().map(|w| w / evidence).collect();
    Ok(MeasuredConditional {
        evidence_probability: evidence,
        outcome: named_levels(law, y, &outcome_probabilities),
        construct_posterior: named_levels(law, c, &construct_probabilities),
    })
}

// ----------------------------------------------------- threshold crossing

/// One observable level of a signal and what it does to the outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct SignalLevelOutcome {
    /// The signal level.
    pub level: String,
    /// `P(signal = level | background)`.
    pub probability: f64,
    /// `P(outcome = outcome_level | signal = level, background)`.
    pub conditional: f64,
    /// Whether the conditional probability reaches the threshold.
    pub crosses: bool,
}

/// Which observations of a signal move an outcome probability across a level.
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdCrossing {
    /// The outcome level whose probability is tracked.
    pub outcome_level: String,
    /// The declared threshold.
    pub threshold: f64,
    /// Whether crossing means reaching it from below or from above.
    pub direction: Direction,
    /// Every signal level of positive probability, in level order.
    pub levels: Vec<SignalLevelOutcome>,
    /// Total probability of observing a crossing level.
    pub crossing_probability: f64,
}

/// The observed levels of `signal` under which `P(outcome = outcome_level)`
/// reaches `threshold` (at or above for [`Direction::Raise`], at or below for
/// [`Direction::Lower`]), and the probability of observing one.
///
/// `background` is recorded values held fixed throughout.
///
/// # Errors
/// An unknown variable or level, a repeated variable, the signal or outcome in
/// the background, a threshold outside `[0, 1]`, background of probability
/// zero, [`ObservationQueryError::LevelUnreachable`] (with the most extreme
/// conditional probability any observation attains) when no level of the
/// signal crosses, or cancellation.
#[allow(
    clippy::too_many_arguments,
    reason = "the query names the outcome, its level, the signal, the background, the threshold and its direction as separate declared scalars"
)]
pub fn threshold_crossing_observations(
    law: &FiniteLaw,
    outcome: &str,
    outcome_level: &str,
    signal: &str,
    background: &[(&str, &str)],
    threshold: f64,
    direction: Direction,
    ctx: &ExecutionContext,
) -> Result<ThresholdCrossing, ObservationQueryError> {
    check_closed_probability("the threshold must lie in [0, 1]", threshold)?;
    let y = variable_index(law, outcome)?;
    let s = variable_index(law, signal)?;
    if y == s {
        return Err(ObservationQueryError::OutcomeObserved(outcome.to_owned()));
    }
    let target = level_index(law, y, outcome_level)?;
    let fixed = resolve_assignments(law, background)?;
    if fixed.iter().any(|&(v, _)| v == y) {
        return Err(ObservationQueryError::OutcomeObserved(outcome.to_owned()));
    }
    if fixed.iter().any(|&(v, _)| v == s) {
        return Err(ObservationQueryError::RepeatedVariable(signal.to_owned()));
    }
    let width_y = law.levels()[y].len();
    let joint = bucketed(law, &[s, y], &fixed, ctx)?;
    let total: f64 = joint.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return Err(ObservationQueryError::ZeroProbabilityEvidence);
    }
    let mut rows = Vec::new();
    let mut crossing_probability = 0.0;
    let mut extremum = match direction {
        Direction::Raise => f64::NEG_INFINITY,
        Direction::Lower => f64::INFINITY,
    };
    for (si, level) in law.levels()[s].iter().enumerate() {
        let mass: f64 = joint[si * width_y..(si + 1) * width_y].iter().sum();
        if mass <= 0.0 {
            continue;
        }
        let conditional = joint[si * width_y + target] / mass;
        let crosses = match direction {
            Direction::Raise => conditional >= threshold - LEVEL_TOLERANCE,
            Direction::Lower => conditional <= threshold + LEVEL_TOLERANCE,
        };
        extremum = match direction {
            Direction::Raise => extremum.max(conditional),
            Direction::Lower => extremum.min(conditional),
        };
        let probability = mass / total;
        if crosses {
            crossing_probability += probability;
        }
        rows.push(SignalLevelOutcome { level: level.clone(), probability, conditional, crosses });
    }
    if !rows.iter().any(|r| r.crosses) {
        return Err(ObservationQueryError::LevelUnreachable {
            level: threshold,
            extremum,
            direction,
            bound: UnreachableBound::NoObservationCrosses,
        });
    }
    Ok(ThresholdCrossing {
        outcome_level: outcome_level.to_owned(),
        threshold,
        direction,
        levels: rows,
        crossing_probability,
    })
}

// ------------------------------------------------------- finite Bayes model

/// A declared finite hypothesis space with a likelihood for each observation
/// level. Observations are conditionally independent given the hypothesis and
/// exchangeable, so the posterior depends on the multiset of observations and
/// not on their order.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteBayesModel {
    hypotheses: Vec<String>,
    prior: Vec<f64>,
    observation_levels: Vec<String>,
    // likelihood[h * n_observation_levels + o] = P(observation = o | hypothesis = h)
    likelihood: Vec<f64>,
}

/// The posterior of a [`FiniteBayesModel`] after a set of observations.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesPosterior {
    /// `(hypothesis, posterior probability)` in declaration order.
    pub hypotheses: Vec<(String, f64)>,
    /// Natural log of the probability of the observations under the model.
    pub log_evidence: f64,
    /// Probability of the observations under the model.
    pub evidence_probability: f64,
    /// `(observation level, posterior predictive probability of the next
    /// observation)`.
    pub predictive: Vec<(String, f64)>,
}

fn distinct_names(names: &[String], what: &'static str) -> Result<(), ObservationQueryError> {
    if names.is_empty() || names.iter().any(|n| n.trim().is_empty()) {
        return Err(ObservationQueryError::Invalid(what));
    }
    let set: BTreeSet<&String> = names.iter().collect();
    if set.len() != names.len() {
        return Err(ObservationQueryError::Invalid(what));
    }
    Ok(())
}

impl FiniteBayesModel {
    /// Declare a model. `likelihood` is row-major `[hypothesis][observation]`
    /// and every row is a distribution over the observation levels.
    ///
    /// # Errors
    /// Empty, blank or duplicate names, a bound exceeded, a prior or a
    /// likelihood row that is not a distribution within
    /// [`SIMPLEX_TOLERANCE`], or a shape mismatch.
    pub fn new(
        hypotheses: Vec<String>,
        prior: Vec<f64>,
        observation_levels: Vec<String>,
        likelihood: Vec<f64>,
    ) -> Result<Self, ObservationQueryError> {
        distinct_names(&hypotheses, "hypotheses must be distinct non-blank names")?;
        distinct_names(&observation_levels, "observation levels must be distinct non-blank names")?;
        if hypotheses.len() > MAX_BAYES_HYPOTHESES
            || observation_levels.len() > MAX_BAYES_OBSERVATION_LEVELS
        {
            return Err(ObservationQueryError::Invalid("the Bayes model exceeds its size bound"));
        }
        let h = hypotheses.len();
        let o = observation_levels.len();
        if prior.len() != h || likelihood.len() != h * o {
            return Err(ObservationQueryError::Invalid("prior or likelihood shape mismatch"));
        }
        let simplex = |values: &[f64]| {
            values.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                && (values.iter().sum::<f64>() - 1.0).abs() <= SIMPLEX_TOLERANCE
        };
        if !simplex(&prior) {
            return Err(ObservationQueryError::Invalid("the prior is not a distribution"));
        }
        if likelihood.chunks(o).any(|row| !simplex(row)) {
            return Err(ObservationQueryError::Invalid(
                "a likelihood row is not a distribution over the observation levels",
            ));
        }
        Ok(Self { hypotheses, prior, observation_levels, likelihood })
    }

    /// Hypothesis names in declaration order.
    #[must_use]
    pub fn hypotheses(&self) -> &[String] {
        &self.hypotheses
    }

    /// Prior probabilities, aligned with [`Self::hypotheses`].
    #[must_use]
    pub fn prior(&self) -> &[f64] {
        &self.prior
    }

    /// Observation level names in declaration order.
    #[must_use]
    pub fn observation_levels(&self) -> &[String] {
        &self.observation_levels
    }

    /// Row-major `[hypothesis][observation]` likelihood.
    #[must_use]
    pub fn likelihood(&self) -> &[f64] {
        &self.likelihood
    }

    /// The posterior after `observations` (level names; any order).
    ///
    /// # Errors
    /// An unknown observation level, more than [`MAX_BAYES_SEQUENCE`]
    /// observations, or observations of probability zero under every
    /// hypothesis.
    pub fn posterior(
        &self,
        observations: &[&str],
    ) -> Result<BayesPosterior, ObservationQueryError> {
        if observations.len() > MAX_BAYES_SEQUENCE {
            return Err(ObservationQueryError::Invalid("too many observations"));
        }
        let o = self.observation_levels.len();
        let mut counts = vec![0_u64; o];
        for level in observations {
            let Some(i) = self.observation_levels.iter().position(|l| l == level) else {
                return Err(ObservationQueryError::UnknownLevel {
                    variable: "observation".to_owned(),
                    level: (*level).to_owned(),
                });
            };
            counts[i] += 1;
        }
        let log_weights: Vec<f64> = (0..self.hypotheses.len())
            .map(|h| {
                let mut w = self.prior[h].ln();
                for (i, &count) in counts.iter().enumerate() {
                    if count > 0 {
                        w += (count as f64) * self.likelihood[h * o + i].ln();
                    }
                }
                w
            })
            .collect();
        let max = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            return Err(ObservationQueryError::ZeroProbabilityEvidence);
        }
        let log_evidence = max + log_weights.iter().map(|w| (w - max).exp()).sum::<f64>().ln();
        let posterior: Vec<f64> = log_weights.iter().map(|w| (w - log_evidence).exp()).collect();
        let predictive = (0..o)
            .map(|i| {
                let p: f64 = (0..self.hypotheses.len())
                    .map(|h| posterior[h] * self.likelihood[h * o + i])
                    .sum();
                (self.observation_levels[i].clone(), p)
            })
            .collect();
        Ok(BayesPosterior {
            hypotheses: self.hypotheses.iter().cloned().zip(posterior).collect(),
            log_evidence,
            evidence_probability: log_evidence.exp(),
            predictive,
        })
    }
}

// ------------------------------------------------------------ conjugate model

/// A Beta prior on a Bernoulli success probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBernoulli {
    alpha: f64,
    beta: f64,
}

impl BetaBernoulli {
    /// Declare a prior `Beta(alpha, beta)`.
    ///
    /// # Errors
    /// A shape that is not finite and positive.
    pub fn new(alpha: f64, beta: f64) -> Result<Self, ObservationQueryError> {
        if alpha.is_finite() && beta.is_finite() && alpha > 0.0 && beta > 0.0 {
            Ok(Self { alpha, beta })
        } else {
            Err(ObservationQueryError::Invalid("Beta shapes must be finite and positive"))
        }
    }

    /// First shape.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Second shape.
    #[must_use]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// The mean, which is also the predictive probability of the next success.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let total = self.alpha + self.beta;
        if total.is_finite() {
            return self.alpha / total;
        }
        // Both shapes are individually finite, while their sum can overflow
        // even though the ratio is well defined.
        let scale = self.alpha.max(self.beta);
        let alpha = self.alpha / scale;
        alpha / (alpha + self.beta / scale)
    }

    /// The posterior after `successes` and `failures` (conjugate update).
    ///
    /// # Errors
    /// A total count above `2^53`, where the update would not be exact.
    pub fn update(&self, successes: u64, failures: u64) -> Result<Self, ObservationQueryError> {
        const EXACT: u64 = 1 << 53;
        if successes.checked_add(failures).is_none_or(|count| count > EXACT) {
            return Err(ObservationQueryError::Invalid("observation counts exceed 2^53"));
        }
        let alpha = self.alpha + successes as f64;
        let beta = self.beta + failures as f64;
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(ObservationQueryError::Invalid("updated Beta shapes must remain finite"));
        }
        Ok(Self { alpha, beta })
    }
}

// ---------------------------------------------------------- evidence needed

/// The likelihood ratio that moves a probability to a level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RequiredBayesFactor {
    /// The requested direction.
    pub direction: Direction,
    /// Smallest Bayes factor (posterior odds over prior odds) in that
    /// direction that reaches the level; `1.0` when no evidence is needed.
    pub factor: f64,
    /// The prior already satisfies the request.
    pub already_satisfied: bool,
}

fn satisfied_log_odds(direction: Direction, value: f64, target: f64) -> bool {
    match direction {
        Direction::Raise => value >= target - LEVEL_TOLERANCE,
        Direction::Lower => value <= target + LEVEL_TOLERANCE,
    }
}

/// The exact Bayes factor that moves a probability from `prior` to `level`:
/// `[level / (1 - level)] / [prior / (1 - prior)]`.
///
/// # Errors
/// A prior or level outside `(0, 1)`.
pub fn required_bayes_factor(
    prior: f64,
    level: f64,
    direction: Direction,
) -> Result<RequiredBayesFactor, ObservationQueryError> {
    check_open_probability("the prior must lie in (0, 1)", prior)?;
    check_open_probability("the level must lie in (0, 1)", level)?;
    let (start, target) = (log_odds(prior), log_odds(level));
    if satisfied_log_odds(direction, start, target) {
        return Ok(RequiredBayesFactor { direction, factor: 1.0, already_satisfied: true });
    }
    Ok(RequiredBayesFactor { direction, factor: (target - start).exp(), already_satisfied: false })
}

/// The smallest number of identical observations that reaches a level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvidenceRequirement {
    /// The requested direction.
    pub direction: Direction,
    /// Smallest observation count that reaches the level.
    pub observations: u64,
    /// The probability after that many observations.
    pub value_at_requirement: f64,
    /// The starting probability already satisfies the request (zero
    /// observations).
    pub already_satisfied: bool,
    /// Bayes factor of one observation in the requested direction, where the
    /// route is a likelihood-ratio route.
    pub bayes_factor_per_observation: Option<f64>,
}

/// Smallest number `n` of consecutive conditionally independent binary-signal
/// observations of one kind that moves `P(H)` from `prior` to `level`.
///
/// Raising uses positive readings (Bayes factor `rate_given_hypothesis /
/// rate_given_complement`); lowering uses negative readings
/// (`(1 - rate_given_hypothesis) / (1 - rate_given_complement)`). A level that
/// the signal cannot reach, or reaches only beyond `max_observations`, is
/// [`ObservationQueryError::LevelUnreachable`] carrying the most extreme
/// probability reachable.
///
/// # Errors
/// A probability outside its range, `max_observations` of zero or above
/// [`MAX_REQUIRED_OBSERVATIONS`], or an unreachable level.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the solve converts a nonnegative integer-valued float already compared against max_observations"
)]
pub fn observations_required_binary_signal(
    prior: f64,
    level: f64,
    direction: Direction,
    rate_given_hypothesis: f64,
    rate_given_complement: f64,
    max_observations: u64,
) -> Result<EvidenceRequirement, ObservationQueryError> {
    check_open_probability("the prior must lie in (0, 1)", prior)?;
    check_open_probability("the level must lie in (0, 1)", level)?;
    check_closed_probability("rate_given_hypothesis must lie in [0, 1]", rate_given_hypothesis)?;
    check_closed_probability("rate_given_complement must lie in [0, 1]", rate_given_complement)?;
    if max_observations == 0 || max_observations > MAX_REQUIRED_OBSERVATIONS {
        return Err(ObservationQueryError::Invalid("max_observations is outside its bound"));
    }
    let (start, target) = (log_odds(prior), log_odds(level));
    if satisfied_log_odds(direction, start, target) {
        return Ok(EvidenceRequirement {
            direction,
            observations: 0,
            value_at_requirement: prior,
            already_satisfied: true,
            bayes_factor_per_observation: None,
        });
    }
    let (numerator, denominator) = match direction {
        Direction::Raise => (rate_given_hypothesis, rate_given_complement),
        Direction::Lower => (1.0 - rate_given_hypothesis, 1.0 - rate_given_complement),
    };
    let unreachable = |extremum: f64, bound: UnreachableBound| {
        Err(ObservationQueryError::LevelUnreachable { level, extremum, direction, bound })
    };
    // Log Bayes factor of one observation of the moving kind.
    let log_bf = if numerator <= 0.0 && denominator <= 0.0 {
        return unreachable(prior, UnreachableBound::SignalDoesNotMove);
    } else if denominator <= 0.0 {
        f64::INFINITY
    } else if numerator <= 0.0 {
        f64::NEG_INFINITY
    } else {
        numerator.ln() - denominator.ln()
    };
    let moves = match direction {
        Direction::Raise => log_bf > 0.0,
        Direction::Lower => log_bf < 0.0,
    };
    if !moves {
        return unreachable(prior, UnreachableBound::SignalDoesNotMove);
    }
    let bayes_factor_per_observation = Some(log_bf.exp());
    if log_bf.is_infinite() {
        // One observation settles the hypothesis: the posterior is 0 or 1.
        let settled = if matches!(direction, Direction::Raise) { 1.0 } else { 0.0 };
        return Ok(EvidenceRequirement {
            direction,
            observations: 1,
            value_at_requirement: settled,
            already_satisfied: false,
            bayes_factor_per_observation,
        });
    }
    let at = |n: u64| start + (n as f64) * log_bf;
    let estimate = ((target - start) / log_bf).ceil();
    let budget = max_observations as f64;
    if !estimate.is_finite() || estimate > budget + 1.0 {
        return unreachable(
            from_log_odds(at(max_observations)),
            UnreachableBound::ObservationBudget { max_observations },
        );
    }
    let mut n = estimate.max(0.0) as u64;
    while n > 0 && satisfied_log_odds(direction, at(n - 1), target) {
        n -= 1;
    }
    while !satisfied_log_odds(direction, at(n), target) {
        n += 1;
        if n > max_observations {
            return unreachable(
                from_log_odds(at(max_observations)),
                UnreachableBound::ObservationBudget { max_observations },
            );
        }
    }
    if n > max_observations {
        return unreachable(
            from_log_odds(at(max_observations)),
            UnreachableBound::ObservationBudget { max_observations },
        );
    }
    Ok(EvidenceRequirement {
        direction,
        observations: n,
        value_at_requirement: from_log_odds(at(n)),
        already_satisfied: false,
        bayes_factor_per_observation,
    })
}

/// Smallest number of consecutive successes (raising) or failures (lowering)
/// that moves a Beta-Bernoulli predictive probability to `level`.
///
/// After `n` successes the predictive probability is `(a + n) / (a + b + n)`;
/// after `n` failures it is `a / (a + b + n)`.
///
/// # Errors
/// A level outside `(0, 1)`, `max_observations` outside its bound, or a level
/// reachable only beyond `max_observations` (with the predictive probability
/// at the bound as the most extreme reachable value).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the solve converts a nonnegative integer-valued float already compared against max_observations"
)]
pub fn observations_required_conjugate(
    prior: &BetaBernoulli,
    level: f64,
    direction: Direction,
    max_observations: u64,
) -> Result<EvidenceRequirement, ObservationQueryError> {
    check_open_probability("the level must lie in (0, 1)", level)?;
    if max_observations == 0 || max_observations > MAX_REQUIRED_OBSERVATIONS {
        return Err(ObservationQueryError::Invalid("max_observations is outside its bound"));
    }
    let (a, b) = (prior.alpha, prior.beta);
    let value_after = |n: u64| -> f64 {
        let n = n as f64;
        let total = a + b + n;
        if total.is_finite() {
            return match direction {
                Direction::Raise => (a + n) / total,
                Direction::Lower => a / total,
            };
        }
        // Scaling avoids overflow for valid finite shapes near f64::MAX.
        let scale = a.max(b).max(n);
        let (a, b, n) = (a / scale, b / scale, n / scale);
        match direction {
            Direction::Raise => (a + n) / (a + b + n),
            Direction::Lower => a / (a + b + n),
        }
    };
    let satisfied = |p: f64| match direction {
        Direction::Raise => p >= level - LEVEL_TOLERANCE,
        Direction::Lower => p <= level + LEVEL_TOLERANCE,
    };
    if satisfied(prior.mean()) {
        return Ok(EvidenceRequirement {
            direction,
            observations: 0,
            value_at_requirement: prior.mean(),
            already_satisfied: true,
            bayes_factor_per_observation: None,
        });
    }
    let budget_exceeded = || {
        Err(ObservationQueryError::LevelUnreachable {
            level,
            extremum: value_after(max_observations),
            direction,
            bound: UnreachableBound::ObservationBudget { max_observations },
        })
    };
    if !satisfied(value_after(max_observations)) {
        return budget_exceeded();
    }
    // The predictive probability is monotone in n. Binary search the bounded
    // integer domain instead of using a closed form whose intermediate sums
    // can overflow for otherwise valid Beta shapes.
    let mut low = 0_u64;
    let mut high = max_observations;
    while low < high {
        let mid = low + (high - low) / 2;
        if satisfied(value_after(mid)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let n = low;
    Ok(EvidenceRequirement {
        direction,
        observations: n,
        value_at_requirement: value_after(n),
        already_satisfied: false,
        bayes_factor_per_observation: None,
    })
}
