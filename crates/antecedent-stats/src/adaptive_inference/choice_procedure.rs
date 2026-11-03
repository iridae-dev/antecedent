//! The exact law of a sequential categorical-choice procedure.
//!
//! The categorical sibling of [`super::procedure_law::enumerate_law`]. That enumerator's state is
//! the success and failure counts of Bernoulli arm-by-context cells; a unit that makes one choice
//! among `K` options has a different state (the `K` choice counts) and a different posterior (the
//! Dirichlet of [`super::dirichlet_choice`]), so it cannot be written as a [`ProcedureSpec`]. The
//! method is the same: dynamic programming over the *observed* state, units of a batch expanded one
//! at a time and merged by state, the stopping rule evaluated at every look on the observed counts,
//! the terminal state being the state at the stopping time, and the same state and work bounds
//! ([`MAX_LAW_STATES`], [`MAX_LAW_WORK`]) refused with the same capacity details.
//!
//! Two kernels are supported: a fixed truth `theta` (the sampling law of the procedure under that
//! truth, from which error rates and expected sample size follow) and a Dirichlet prior predictive
//! (a Polya urn: the law under which the posterior is checked).
//!
//! The stopping rule [`ChoiceStoppingRule`] is the single definition of the rule: the
//! design-layer sequential look evaluates the same function, so the procedure whose law is
//! computed here is the procedure that is run. The comparison band is
//! [`THRESHOLD_TOLERANCE`], as in the Bernoulli rules.
//!
//! Exact finite-law arithmetic supplies no interval or calibration support: the law is of the
//! declared procedure under a hypothesized truth, nothing more.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::AdaptiveInferenceError as Error;
use super::capacity::CapacityError;
use super::dirichlet_choice::DirichletPosterior;
use super::procedure_law::{
    CANCEL_STRIDE, MAX_LAW_STATES, MAX_LAW_WORK, THRESHOLD_TOLERANCE, check_states, remember,
};

/// Most options an enumerated choice procedure may have.
pub const MAX_CHOICE_PROCEDURE_OPTIONS: usize = 4;
/// Most looks an enumerated choice procedure may have.
pub const MAX_CHOICE_PROCEDURE_LOOKS: usize = 24;
/// Most units an enumerated choice procedure may enrol.
pub const MAX_CHOICE_PROCEDURE_UNITS: u32 = 65_535;

/// The Bayesian stopping rule of a choice procedure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChoiceStoppingRule {
    /// Posterior probability that an option has the largest share, at least which the procedure
    /// may stop, in `(0.5, 1)`; `None` leaves the criterion out.
    pub min_probability_best: Option<f64>,
    /// Largest posterior expected loss (in share) of the selected option, finite and positive;
    /// `None` leaves the criterion out.
    pub max_expected_loss: Option<f64>,
    /// Units that must have been observed before any stop.
    pub min_observations: u32,
}

impl ChoiceStoppingRule {
    /// Whether the rule is inside its domain.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (self.min_probability_best.is_some() || self.max_expected_loss.is_some())
            && self.min_probability_best.is_none_or(|p| p > 0.5 && p < 1.0)
            && self.max_expected_loss.is_none_or(|l| l.is_finite() && l > 0.0)
    }

    /// The option the rule selects at these posterior quantities, or `None` to continue.
    ///
    /// A quantity within [`THRESHOLD_TOLERANCE`] of its threshold is read as equal to it: "at
    /// least" and "at most" hold at equality.
    #[must_use]
    pub fn assess(
        &self,
        observed: f64,
        probability_best: &[f64],
        expected_loss: &[f64],
    ) -> Option<usize> {
        let (best, p) = probability_best
            .iter()
            .copied()
            .enumerate()
            .fold((0, f64::MIN), |a, (i, p)| if p > a.1 { (i, p) } else { a });
        (observed >= f64::from(self.min_observations)
            && self.min_probability_best.is_none_or(|t| p >= t - THRESHOLD_TOLERANCE)
            && self
                .max_expected_loss
                .is_none_or(|t| expected_loss[best] <= t + THRESHOLD_TOLERANCE))
        .then_some(best)
    }
}

/// A sequential choice procedure: batches of choices, a look after each, a stopping rule.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceProcedureSpec {
    /// Dirichlet prior shapes (each at least one) of the analysis posterior.
    pub prior: Vec<f64>,
    /// Choices enrolled before each look.
    pub batches: Vec<u32>,
    /// The stopping rule evaluated at every look.
    pub rule: ChoiceStoppingRule,
}

/// The outcome mechanism of a hypothesized truth.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ChoiceKernel {
    /// A fixed vector of choice shares (nonnegative, summing to one).
    Fixed {
        /// The shares.
        theta: Vec<f64>,
    },
    /// A Dirichlet prior predictive (a Polya urn) with these shapes.
    PriorPredictive {
        /// The prior shapes.
        alpha: Vec<f64>,
    },
}

/// One terminal state of the procedure and its probability.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceTerminalAtom {
    /// Look (zero-based) at which the procedure ended.
    pub look: usize,
    /// Whether the stopping rule, rather than the planned horizon, ended it.
    pub rule_stopped: bool,
    /// Observed choice counts.
    pub counts: Vec<u16>,
    /// The option selected, if the rule stopped.
    pub decision: Option<usize>,
    /// Choices enrolled when the procedure ended.
    pub enrolled: u32,
    /// Probability of this terminal state.
    pub probability: f64,
}

/// The law of the terminal state of a choice procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceLaw {
    /// Terminal atoms, in a deterministic order.
    pub atoms: Vec<ChoiceTerminalAtom>,
    /// Cumulative choices enrolled through each look.
    pub cumulative_enrolment: Vec<u32>,
    /// Unit expansions performed (observed work).
    pub work_units: u64,
    /// Largest number of live states at any look (observed work).
    pub peak_states: usize,
}

impl ChoiceLaw {
    /// Probability that each option is selected, and the mass that ends undecided.
    #[must_use]
    pub fn selection_probabilities(&self, options: usize) -> (Vec<f64>, f64) {
        let mut chosen = vec![0.0; options];
        let mut undecided = 0.0;
        for atom in &self.atoms {
            match atom.decision {
                Some(option) => chosen[option] += atom.probability,
                None => undecided += atom.probability,
            }
        }
        (chosen, undecided)
    }

    /// Expected choices enrolled.
    #[must_use]
    pub fn expected_enrolled(&self) -> f64 {
        self.atoms.iter().map(|a| f64::from(a.enrolled) * a.probability).sum()
    }

    /// Probability that the rule stopped at each look.
    #[must_use]
    pub fn rule_stop_probabilities(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.cumulative_enrolment.len()];
        for atom in self.atoms.iter().filter(|a| a.rule_stopped) {
            out[atom.look] += atom.probability;
        }
        out
    }

    /// Probability that the planned horizon ended the procedure without a stop.
    #[must_use]
    pub fn horizon_probability(&self) -> f64 {
        self.atoms.iter().filter(|a| !a.rule_stopped).map(|a| a.probability).sum()
    }
}

type StateMap = BTreeMap<Vec<u16>, f64>;

struct Rules<'a> {
    spec: &'a ChoiceProcedureSpec,
    cancelled: &'a dyn Fn() -> bool,
    cache: BTreeMap<Vec<u16>, (Vec<f64>, Vec<f64>)>,
}

impl Rules<'_> {
    fn assess(&mut self, counts: &[u16], observed: u32) -> Result<Option<usize>, Error> {
        if !self.cache.contains_key(counts) {
            let as_f64: Vec<f64> = counts.iter().map(|c| f64::from(*c)).collect();
            let posterior = DirichletPosterior::from_counts(&self.spec.prior, &as_f64)?;
            let best = posterior.best_option_probabilities(self.cancelled)?;
            let loss = posterior.expected_losses(self.cancelled)?;
            remember(&mut self.cache, MAX_LAW_STATES, counts, (best, loss));
        }
        let (best, loss) = &self.cache[counts];
        Ok(self.spec.rule.assess(f64::from(observed), best, loss))
    }
}

fn validate(spec: &ChoiceProcedureSpec, kernel: &ChoiceKernel) -> Result<(), Error> {
    let k = spec.prior.len();
    if !(2..=MAX_CHOICE_PROCEDURE_OPTIONS).contains(&k) {
        return Err(Error::Shape("an enumerated choice procedure has two to four options"));
    }
    DirichletPosterior::new(spec.prior.clone())?;
    if !spec.rule.is_valid() {
        return Err(Error::InvalidParameter("the choice stopping rule is outside its domain"));
    }
    if spec.batches.is_empty()
        || spec.batches.len() > MAX_CHOICE_PROCEDURE_LOOKS
        || spec.batches.contains(&0)
        || spec.batches.iter().map(|b| u64::from(*b)).sum::<u64>()
            > u64::from(MAX_CHOICE_PROCEDURE_UNITS)
    {
        return Err(Error::Shape(
            "a choice procedure has 1 to 24 nonempty batches of at most 65535 units",
        ));
    }
    match kernel {
        ChoiceKernel::Fixed { theta } => {
            if theta.len() != k
                || theta.iter().any(|t| !(t.is_finite() && *t >= 0.0))
                || (theta.iter().sum::<f64>() - 1.0).abs() > 1e-9
            {
                return Err(Error::InvalidParameter(
                    "the fixed shares must be a distribution over the options",
                ));
            }
        }
        ChoiceKernel::PriorPredictive { alpha } => {
            if alpha.len() != k || alpha.iter().any(|a| !(a.is_finite() && *a > 0.0)) {
                return Err(Error::InvalidParameter(
                    "the predictive shapes must be positive, one per option",
                ));
            }
        }
    }
    Ok(())
}

/// The exact law of the terminal state of `spec` under `kernel`.
///
/// # Errors
/// A spec or kernel outside its domain, a state or work bound exceeded, a numerical failure of
/// a posterior, or cancellation.
pub fn enumerate_choice_law(
    spec: &ChoiceProcedureSpec,
    kernel: &ChoiceKernel,
    cancelled: &dyn Fn() -> bool,
) -> Result<ChoiceLaw, Error> {
    validate(spec, kernel)?;
    let k = spec.prior.len();
    let mut cumulative = Vec::with_capacity(spec.batches.len());
    let mut total = 0_u32;
    for b in &spec.batches {
        total += b;
        cumulative.push(total);
    }
    let mut rules = Rules { spec, cancelled, cache: BTreeMap::new() };
    let mut work = 0_u64;
    let mut live: StateMap = BTreeMap::new();
    live.insert(vec![0_u16; k], 1.0);
    let mut terminal: BTreeMap<(usize, bool, Vec<u16>, Option<usize>), f64> = BTreeMap::new();
    let mut peak = 1;
    for (look, &batch) in spec.batches.iter().enumerate() {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        for _ in 0..batch {
            let mut next: StateMap = BTreeMap::new();
            for (key, probability) in &live {
                let seen: f64 = key.iter().map(|c| f64::from(*c)).sum();
                for option in 0..k {
                    let q = match kernel {
                        ChoiceKernel::Fixed { theta } => theta[option],
                        ChoiceKernel::PriorPredictive { alpha } => {
                            let a0: f64 = alpha.iter().sum();
                            (alpha[option] + f64::from(key[option])) / (a0 + seen)
                        }
                    };
                    if q > 0.0 {
                        let mut grown = key.clone();
                        grown[option] += 1;
                        *next.entry(grown).or_insert(0.0) += probability * q;
                    }
                }
            }
            check_states(next.len())?;
            let before = work / CANCEL_STRIDE;
            work = work.saturating_add(next.len() as u64 + 1);
            if work > MAX_LAW_WORK {
                return Err(Error::Capacity(CapacityError::LawWork {
                    limit: MAX_LAW_WORK,
                    requested: work,
                }));
            }
            if work / CANCEL_STRIDE != before && cancelled() {
                return Err(Error::Cancelled);
            }
            live = next;
        }
        peak = peak.max(live.len());
        let last = look + 1 == spec.batches.len();
        let mut continuing: StateMap = BTreeMap::new();
        for (key, probability) in live {
            let decision = rules.assess(&key, cumulative[look])?;
            if decision.is_some() || last {
                *terminal
                    .entry((look, decision.is_some(), key.clone(), decision))
                    .or_insert(0.0) += probability;
            } else {
                continuing.insert(key, probability);
            }
        }
        live = continuing;
    }
    let atoms: Vec<ChoiceTerminalAtom> = terminal
        .into_iter()
        .map(|((look, rule_stopped, counts, decision), probability)| ChoiceTerminalAtom {
            look,
            rule_stopped,
            counts,
            decision,
            enrolled: cumulative[look],
            probability,
        })
        .collect();
    let mass: f64 = atoms.iter().map(|a| a.probability).sum();
    if mass.is_nan() || (mass - 1.0).abs() >= 1e-9 {
        return Err(Error::Numerical("the choice procedure law does not sum to one"));
    }
    Ok(ChoiceLaw { atoms, cumulative_enrolment: cumulative, work_units: work, peak_states: peak })
}
