//! Uncertainty derived for the complete adaptive procedure.
//!
//! The estimators of this module tree are analysed together with the process
//! that produced their data: the assignment policy, the delay and loss of
//! outcomes, exposure failures, the stopping rule and the final selection of
//! an arm. No static interval or posterior guarantee is inherited from the
//! analysis of a final sample.
//!
//! Paradigms stay separate. Each [`UncertaintyParadigm`] has its own premise
//! gate ([`admit`]) and its own report; a report of one paradigm is never a
//! statement of another:
//!
//! * **Bayesian posterior validity** ([`bayesian_report`],
//!   [`posterior_mean_gaps`]): the posterior of a declared prior is the Beta
//!   of the observed counts whatever the policy and stopping rule, when both
//!   depend only on the observed history and the loss of outcomes does not
//!   depend on their values. Whether the posterior mean is a martingale of
//!   the whole procedure is computed exactly under the prior predictive; a
//!   nonzero gap is the witness that a premise failed.
//! * **Decision-theoretic stopping** ([`decision_report`]): the stopping time,
//!   the decision and the regret of the declared rule, under a hypothesized
//!   truth.
//! * **Frequentist fixed sample** and **frequentist group sequential**: the
//!   sampling law of the design as declared ([`ProcedureLaw::estimator_law`],
//!   [`ordering_p_values`]); admitted only for a fixed-randomization design.
//! * **Adaptively weighted scores**: the estimator's law under the stopping
//!   time and policy selection, by whole-procedure simulation
//!   ([`super::procedure_simulation`]); its plug-in variance is a diagnostic.
//! * **Anytime-valid**: a time-uniform confidence sequence at the
//!   (data-dependent) stopping time ([`anytime_valid_report`]), with the exact
//!   whole-procedure probability that it excludes the truth
//!   ([`anytime_exclusion_probability`]).
//! * **Whole-procedure test inversion** ([`inversion_interval`]): the
//!   confidence set of a contrast obtained by inverting a stagewise-ordered
//!   test whose null law is the exact law of the declared procedure.
//!
//! Every result carries [`DERIVED_UNMEASURED`]: the method is derived for the
//! procedure and its law is exact under the stated model, but no coverage of
//! the method has been measured, so none is claimed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::AdaptiveInferenceError as Error;
use super::beta_bernoulli::{BetaPosterior, best_arm_probabilities, expected_losses};
use super::confidence_sequence::mixture_hoeffding_radius;
use super::procedure_contract::{
    AssignmentRule, Attribution, BetaPrior, DelayMechanism, ProcedureSpec, StoppingRule,
};
use super::procedure_law::{OutcomeKernel, ProcedureLaw, TerminalAtom, enumerate_law};

/// Calibration status carried by every result of this module.
pub const DERIVED_UNMEASURED: &str = "derived_unmeasured";
/// Most laws one inversion may evaluate.
pub const MAX_INVERSION_LAWS: usize = 4096;

const TIE_TOLERANCE: f64 = 1e-12;

/// Which inferential paradigm a derived statement belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UncertaintyParadigm {
    /// A posterior of a declared prior.
    BayesianPosterior,
    /// A stop-or-continue decision from posterior expected loss.
    DecisionTheoreticStopping,
    /// The sampling law of a design fixed in advance.
    FrequentistFixedSample,
    /// Efficacy boundaries over pre-specified looks of a fixed design.
    FrequentistGroupSequential,
    /// An adaptively weighted estimator under the whole procedure.
    FrequentistAdaptiveWeighting,
    /// A time-uniform confidence sequence.
    AnytimeValidFrequentist,
    /// A confidence set from inverting the whole-procedure law.
    WholeProcedureInversion,
}

impl UncertaintyParadigm {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BayesianPosterior => "bayesian_posterior",
            Self::DecisionTheoreticStopping => "decision_theoretic_stopping",
            Self::FrequentistFixedSample => "frequentist_fixed_sample",
            Self::FrequentistGroupSequential => "frequentist_group_sequential",
            Self::FrequentistAdaptiveWeighting => "frequentist_adaptive_weighting",
            Self::AnytimeValidFrequentist => "anytime_valid_frequentist",
            Self::WholeProcedureInversion => "whole_procedure_inversion",
        }
    }
}

/// What the arm-level quantity of a result is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimandStatement {
    /// The mean outcome of units that received the arm.
    DeliveredArmMean,
    /// The mean outcome of units assigned the arm, whether or not it was
    /// delivered (intention to treat): a mixture of arm means when exposure
    /// can fail.
    AssignedArmIntentToTreat,
}

/// Registered reason code of every premise failure raised by [`admit`].
#[must_use]
pub fn premise_reason_code() -> &'static str {
    antecedent_core::reason_code!("cell_not_licensed")
}

fn premise(detail: &'static str, message: &'static str) -> Error {
    Error::PremiseFailed { detail, message }
}

/// The validity premises of one paradigm against the declared procedure.
fn check_paradigm(spec: &ProcedureSpec, paradigm: UncertaintyParadigm) -> Result<(), Error> {
    let outcome_dependent = spec.observation().mechanism() == DelayMechanism::OutcomeDependent;
    let fixed = matches!(spec.assignment(), AssignmentRule::Fixed { .. });
    match paradigm {
        UncertaintyParadigm::BayesianPosterior => {
            if outcome_dependent {
                return Err(premise(
                    "adaptive_inference.posterior_premise_outcome_dependent_delay",
                    "the observed counts alone are not the likelihood when whether an outcome is observed depends on its value",
                ));
            }
        }
        UncertaintyParadigm::DecisionTheoreticStopping => {
            if outcome_dependent {
                return Err(premise(
                    "adaptive_inference.posterior_premise_outcome_dependent_delay",
                    "a posterior expected loss from observed counts is not valid when loss of outcomes depends on their values",
                ));
            }
            if !matches!(spec.stopping(), StoppingRule::PosteriorExpectedLoss { .. }) {
                return Err(premise(
                    "adaptive_inference.rule_mismatch",
                    "decision-theoretic stopping needs a posterior expected-loss rule",
                ));
            }
        }
        UncertaintyParadigm::FrequentistFixedSample => {
            if !fixed {
                return Err(premise(
                    "adaptive_inference.fixed_sample_design_adaptive",
                    "a fixed-sample analysis presumes assignment probabilities fixed in advance",
                ));
            }
            if !matches!(spec.stopping(), StoppingRule::NoStopping) {
                return Err(premise(
                    "adaptive_inference.fixed_sample_data_dependent_stopping",
                    "a fixed-sample analysis presumes a sample size that does not depend on the data",
                ));
            }
            if outcome_dependent {
                return Err(premise(
                    "adaptive_inference.outcome_dependent_delay",
                    "outcomes lost depending on their values are not a fixed-sample design",
                ));
            }
        }
        UncertaintyParadigm::FrequentistGroupSequential => {
            if !fixed {
                return Err(premise(
                    "adaptive_inference.group_sequential_assignment_adaptive",
                    "alpha-spending boundaries presume independent increments, which an outcome-adaptive assignment does not have",
                ));
            }
            if !matches!(spec.stopping(), StoppingRule::WaldBoundaries { .. }) {
                return Err(premise(
                    "adaptive_inference.rule_mismatch",
                    "group-sequential inference needs declared boundaries over the looks",
                ));
            }
            if outcome_dependent {
                return Err(premise(
                    "adaptive_inference.outcome_dependent_delay",
                    "outcomes lost depending on their values break the independent-increment model",
                ));
            }
        }
        UncertaintyParadigm::FrequentistAdaptiveWeighting => {
            if outcome_dependent {
                return Err(premise(
                    "adaptive_inference.outcome_dependent_delay",
                    "the weighted scores are not martingale differences when loss of outcomes depends on their values",
                ));
            }
        }
        UncertaintyParadigm::AnytimeValidFrequentist => {
            if outcome_dependent {
                return Err(premise(
                    "adaptive_inference.outcome_dependent_delay",
                    "the observed outcomes of an arm are not a prefix of an independent sequence when loss depends on their values",
                ));
            }
            if spec.contexts().len() > 1 {
                return Err(premise(
                    "adaptive_inference.anytime_premise_contexts",
                    "an arm's outcomes are not identically distributed when units of different contexts are assigned adaptively",
                ));
            }
        }
        UncertaintyParadigm::WholeProcedureInversion => {
            if spec.contexts().len() > 1 {
                return Err(premise(
                    "adaptive_inference.inversion_premise_contexts",
                    "inversion is defined for a contrast of arms within one context",
                ));
            }
        }
    }
    Ok(())
}

/// Check that the declared procedure meets the validity premises of a
/// paradigm, and say what the arm-level quantity is.
///
/// # Errors
/// [`Error::PremiseFailed`] naming the premise, with a stable detail slug.
pub fn admit(
    spec: &ProcedureSpec,
    paradigm: UncertaintyParadigm,
) -> Result<EstimandStatement, Error> {
    check_paradigm(spec, paradigm)?;
    let intention =
        spec.exposure().attribution == Attribution::Assigned && spec.exposure().has_failures();
    Ok(if intention {
        EstimandStatement::AssignedArmIntentToTreat
    } else {
        EstimandStatement::DeliveredArmMean
    })
}

// -- Bayesian ------------------------------------------------------------------------------------

/// The posterior of each cell at an observed terminal state.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianReport {
    /// Posterior of each arm-by-context cell.
    pub posteriors: Vec<BetaPosterior>,
    /// Posterior probability that each arm is best (single context; empty
    /// otherwise).
    pub probability_best: Vec<f64>,
    /// Posterior expected loss of choosing each arm (single context; empty
    /// otherwise).
    pub expected_loss: Vec<f64>,
    /// What the posterior is about.
    pub estimand: EstimandStatement,
    /// Always [`DERIVED_UNMEASURED`].
    pub calibration: &'static str,
}

/// The posterior at `observed` counts (`2 * cells` entries, success and
/// failure per cell) under `prior`, if the procedure's premises hold.
///
/// # Errors
/// A premise failure, counts of the wrong shape, or a numerical failure.
pub fn bayesian_report(
    spec: &ProcedureSpec,
    prior: BetaPrior,
    observed: &[u16],
    cancelled: &dyn Fn() -> bool,
) -> Result<BayesianReport, Error> {
    let estimand = admit(spec, UncertaintyParadigm::BayesianPosterior)?;
    if observed.len() != 2 * spec.cells() {
        return Err(Error::Shape("observed counts must hold a success and a failure per cell"));
    }
    if !prior.is_valid() {
        return Err(Error::InvalidParameter("a Beta prior shape must be finite and at least one"));
    }
    let posteriors: Vec<BetaPosterior> = observed
        .chunks_exact(2)
        .map(|c| {
            BetaPosterior::from_counts(prior.alpha, prior.beta, u64::from(c[0]), u64::from(c[1]))
        })
        .collect::<Result<_, _>>()?;
    let (probability_best, expected_loss) = if spec.contexts().len() == 1 {
        (best_arm_probabilities(&posteriors, cancelled)?, expected_losses(&posteriors, cancelled)?)
    } else {
        (Vec::new(), Vec::new())
    };
    Ok(BayesianReport {
        posteriors,
        probability_best,
        expected_loss,
        estimand,
        calibration: DERIVED_UNMEASURED,
    })
}

/// How far the posterior mean of one cell is from a martingale of the whole
/// procedure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorMeanGap {
    /// The cell.
    pub cell: usize,
    /// Prior mean.
    pub prior_mean: f64,
    /// Expectation of the posterior mean at the terminal state under the
    /// prior predictive of the declared procedure.
    pub expected_posterior_mean: f64,
    /// `expected_posterior_mean - prior_mean`.
    pub gap: f64,
}

/// Expected terminal posterior mean of every cell under the prior
/// predictive of the declared procedure, against the prior mean.
///
/// When policy and stopping depend on the observed history only and outcomes
/// are lost independently of their values, the terminal posterior mean is a
/// bounded-stopping-time martingale and every gap is zero up to rounding. A
/// nonzero gap shows that the posterior computed from observed counts is not
/// the posterior of the procedure that produced them.
///
/// # Errors
/// As [`enumerate_law`].
pub fn posterior_mean_gaps(
    spec: &ProcedureSpec,
    priors: &[BetaPrior],
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<PosteriorMeanGap>, Error> {
    let law = enumerate_law(
        spec,
        &OutcomeKernel::PriorPredictive { priors: priors.to_vec() },
        cancelled,
    )?;
    Ok(priors
        .iter()
        .enumerate()
        .map(|(cell, prior)| {
            let expected: f64 = law
                .atoms
                .iter()
                .map(|atom| {
                    let (s, f) = atom.cell_counts(cell);
                    atom.probability * (prior.alpha + f64::from(s))
                        / (prior.alpha + prior.beta + f64::from(s) + f64::from(f))
                })
                .sum();
            let prior_mean = prior.alpha / (prior.alpha + prior.beta);
            PosteriorMeanGap {
                cell,
                prior_mean,
                expected_posterior_mean: expected,
                gap: expected - prior_mean,
            }
        })
        .collect())
}

// -- Laws and their functionals ------------------------------------------------------------------

/// A statistic of the terminal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Functional {
    /// The observed mean of a cell.
    CellMean(usize),
    /// The observed mean of `plus` minus that of `minus`.
    Contrast {
        /// Cell with a plus sign.
        plus: usize,
        /// Cell with a minus sign.
        minus: usize,
    },
    /// The observed mean of the arm the procedure decided for or selected
    /// (first context).
    DecidedArmMean,
}

/// The law of a statistic of the terminal state.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimatorLaw {
    /// `(value, probability)` sorted by value, equal values merged.
    pub support: Vec<(f64, f64)>,
    /// Probability that the statistic is undefined (a needed cell has no
    /// observation, or no arm was decided).
    pub undefined_mass: f64,
}

impl EstimatorLaw {
    /// Probability that the statistic is defined.
    #[must_use]
    pub fn defined_mass(&self) -> f64 {
        self.support.iter().map(|s| s.1).sum()
    }

    /// Mean given that the statistic is defined.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        let mass = self.defined_mass();
        (mass > 0.0).then(|| self.support.iter().map(|(v, p)| v * p).sum::<f64>() / mass)
    }

    /// Variance given that the statistic is defined.
    #[must_use]
    pub fn variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        let mass = self.defined_mass();
        Some(self.support.iter().map(|(v, p)| (v - mean).powi(2) * p).sum::<f64>() / mass)
    }

    /// Smallest value whose conditional distribution function reaches `q`.
    #[must_use]
    pub fn quantile(&self, q: f64) -> Option<f64> {
        let mass = self.defined_mass();
        if !(mass > 0.0 && (0.0..=1.0).contains(&q)) {
            return None;
        }
        let mut cumulative = 0.0;
        for &(value, probability) in &self.support {
            cumulative += probability / mass;
            if cumulative + 1e-12 >= q {
                return Some(value);
            }
        }
        self.support.last().map(|s| s.0)
    }

    /// Unconditional probability that the statistic is defined and at most
    /// `x`.
    #[must_use]
    pub fn probability_at_most(&self, x: f64) -> f64 {
        self.support.iter().filter(|(v, _)| *v <= x + TIE_TOLERANCE).map(|s| s.1).sum()
    }
}

fn functional_value(atom: &TerminalAtom, functional: Functional) -> Option<f64> {
    match functional {
        Functional::CellMean(cell) => atom.cell_mean(cell),
        Functional::Contrast { plus, minus } => {
            Some(atom.cell_mean(plus)? - atom.cell_mean(minus)?)
        }
        Functional::DecidedArmMean => atom.cell_mean(atom.decision?),
    }
}

impl ProcedureLaw {
    /// The law of a statistic of the terminal state.
    ///
    /// # Errors
    /// A functional that names a cell outside the procedure.
    pub fn estimator_law(&self, functional: Functional) -> Result<EstimatorLaw, Error> {
        let in_range = |cell: usize| cell < self.cells;
        let valid = match functional {
            Functional::CellMean(cell) => in_range(cell),
            Functional::Contrast { plus, minus } => in_range(plus) && in_range(minus),
            Functional::DecidedArmMean => true,
        };
        if !valid {
            return Err(Error::Shape("a functional names a cell outside the procedure"));
        }
        let mut points: Vec<(f64, f64)> = Vec::new();
        let mut undefined = 0.0;
        for atom in &self.atoms {
            match functional_value(atom, functional) {
                Some(value) => points.push((value, atom.probability)),
                None => undefined += atom.probability,
            }
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut support: Vec<(f64, f64)> = Vec::new();
        for (value, probability) in points {
            match support.last_mut() {
                Some(last) if (value - last.0).abs() <= TIE_TOLERANCE => last.1 += probability,
                _ => support.push((value, probability)),
            }
        }
        Ok(EstimatorLaw { support, undefined_mass: undefined })
    }

    /// Probability that the stopping rule ends the procedure at each look.
    #[must_use]
    pub fn rule_stop_probabilities(&self) -> Vec<f64> {
        let looks = self.cumulative_enrolment.len();
        let mut out = vec![0.0; looks];
        for atom in self.atoms.iter().filter(|a| a.rule_stopped) {
            out[atom.look] += atom.probability;
        }
        out
    }

    /// Probability that the planned horizon is run without a rule stop.
    #[must_use]
    pub fn horizon_probability(&self) -> f64 {
        self.atoms.iter().filter(|a| !a.rule_stopped).map(|a| a.probability).sum()
    }

    /// Expected units enrolled when the procedure ends.
    #[must_use]
    pub fn expected_enrolled(&self) -> f64 {
        self.atoms.iter().map(|a| a.probability * f64::from(a.enrolled)).sum()
    }

    /// Probability of each decided arm, and of no decision.
    #[must_use]
    pub fn decision_probabilities(&self) -> (Vec<f64>, f64) {
        let mut arms = vec![0.0; self.arms];
        let mut none = 0.0;
        for atom in &self.atoms {
            match atom.decision {
                Some(arm) => arms[arm] += atom.probability,
                None => none += atom.probability,
            }
        }
        (arms, none)
    }
}

// -- Decision-theoretic ------------------------------------------------------------------------------

/// The stopping time and decision quality of a declared rule under a truth.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionReport {
    /// Probability of a rule stop at each look.
    pub stop_probabilities: Vec<f64>,
    /// Probability that no rule stop occurs.
    pub horizon_probability: f64,
    /// Expected units enrolled.
    pub expected_enrolled: f64,
    /// Probability of deciding each arm.
    pub decision_probabilities: Vec<f64>,
    /// Probability of no decision.
    pub undecided_probability: f64,
    /// Probability that a decided arm is not a best arm.
    pub wrong_decision_probability: f64,
    /// Expected regret `E[(max_a theta_a - theta_decided) 1{decided}]`; an
    /// undecided run contributes zero and is reported separately.
    pub expected_regret: f64,
    /// Always [`DERIVED_UNMEASURED`].
    pub calibration: &'static str,
}

/// The decision report of a decision-theoretic procedure under the success
/// probabilities `truth` (one per arm; single context).
///
/// # Errors
/// A premise failure, a truth of the wrong shape, or an enumeration failure.
pub fn decision_report(
    spec: &ProcedureSpec,
    truth: &[f64],
    cancelled: &dyn Fn() -> bool,
) -> Result<DecisionReport, Error> {
    admit(spec, UncertaintyParadigm::DecisionTheoreticStopping)?;
    if spec.contexts().len() != 1 || truth.len() != spec.arms() {
        return Err(Error::Shape("a decision report needs one truth per arm in one context"));
    }
    let law = enumerate_law(
        spec,
        &OutcomeKernel::Fixed { success_probabilities: truth.to_vec() },
        cancelled,
    )?;
    let best = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (decisions, undecided) = law.decision_probabilities();
    let mut wrong = 0.0;
    let mut regret = 0.0;
    for (arm, &probability) in decisions.iter().enumerate() {
        if (truth[arm] - best).abs() > TIE_TOLERANCE {
            wrong += probability;
        }
        regret += probability * (best - truth[arm]);
    }
    Ok(DecisionReport {
        stop_probabilities: law.rule_stop_probabilities(),
        horizon_probability: law.horizon_probability(),
        expected_enrolled: law.expected_enrolled(),
        decision_probabilities: decisions,
        undecided_probability: undecided,
        wrong_decision_probability: wrong,
        expected_regret: regret,
        calibration: DERIVED_UNMEASURED,
    })
}

// -- Anytime-valid -----------------------------------------------------------------------------------

/// The time-uniform bound of one cell at an observed terminal state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnytimeCellBound {
    /// The cell.
    pub cell: usize,
    /// Observed units.
    pub units: u32,
    /// Observed mean, `None` without an observation.
    pub mean: Option<f64>,
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
}

/// Confidence-sequence bounds of every cell at the terminal state.
#[derive(Clone, Debug, PartialEq)]
pub struct AnytimeReport {
    /// One bound per cell, each at miscoverage `alpha / cells`.
    pub bounds: Vec<AnytimeCellBound>,
    /// Miscoverage of each cell's bound.
    pub alpha_per_cell: f64,
    /// What the bounds are about.
    pub estimand: EstimandStatement,
    /// Always [`DERIVED_UNMEASURED`].
    pub calibration: &'static str,
}

/// The simultaneous time-uniform bounds of every cell's mean at `observed`
/// counts, valid at the data-dependent stopping time when the procedure
/// meets the paradigm's premises.
///
/// # Errors
/// A premise failure, counts of the wrong shape, or a bad level.
pub fn anytime_valid_report(
    spec: &ProcedureSpec,
    observed: &[u16],
    alpha: f64,
    tuning_samples: f64,
) -> Result<AnytimeReport, Error> {
    let estimand = admit(spec, UncertaintyParadigm::AnytimeValidFrequentist)?;
    if observed.len() != 2 * spec.cells() {
        return Err(Error::Shape("observed counts must hold a success and a failure per cell"));
    }
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter("the miscoverage level must lie in (0, 1)"));
    }
    let alpha_per_cell = alpha / spec.cells() as f64;
    let mut bounds = Vec::with_capacity(spec.cells());
    for (cell, pair) in observed.chunks_exact(2).enumerate() {
        let units = u32::from(pair[0]) + u32::from(pair[1]);
        if units == 0 {
            bounds.push(AnytimeCellBound { cell, units, mean: None, lower: 0.0, upper: 1.0 });
            continue;
        }
        let mean = f64::from(pair[0]) / f64::from(units);
        let radius = mixture_hoeffding_radius(u64::from(units), alpha_per_cell, tuning_samples)?;
        bounds.push(AnytimeCellBound {
            cell,
            units,
            mean: Some(mean),
            lower: (mean - radius).max(0.0),
            upper: (mean + radius).min(1.0),
        });
    }
    Ok(AnytimeReport { bounds, alpha_per_cell, estimand, calibration: DERIVED_UNMEASURED })
}

/// Exact probability, under the success probabilities `truth` (one per
/// cell), that the bound of [`anytime_valid_report`] at the terminal state
/// excludes the truth of at least one cell.
///
/// This is the whole-procedure error of reporting the bounds at the
/// stopping time. The confidence-sequence theorem bounds it by `alpha`
/// whenever the paradigm's premises hold; the value is computed, not assumed.
///
/// # Errors
/// A truth of the wrong shape, a bad level, or an enumeration failure.
pub fn anytime_exclusion_probability(
    spec: &ProcedureSpec,
    truth: &[f64],
    alpha: f64,
    tuning_samples: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<f64, Error> {
    admit(spec, UncertaintyParadigm::AnytimeValidFrequentist)?;
    let law = enumerate_law(
        spec,
        &OutcomeKernel::Fixed { success_probabilities: truth.to_vec() },
        cancelled,
    )?;
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter("the miscoverage level must lie in (0, 1)"));
    }
    let alpha_per_cell = alpha / spec.cells() as f64;
    let mut excluded = 0.0;
    for atom in &law.atoms {
        let mut misses = false;
        for (cell, &truth_cell) in truth.iter().enumerate().take(spec.cells()) {
            let units = atom.cell_units(cell);
            if units == 0 {
                continue;
            }
            let mean = f64::from(atom.cell_counts(cell).0) / f64::from(units);
            let radius =
                mixture_hoeffding_radius(u64::from(units), alpha_per_cell, tuning_samples)?;
            if (mean - radius).max(0.0) > truth_cell || (mean + radius).min(1.0) < truth_cell {
                misses = true;
            }
        }
        if misses {
            excluded += atom.probability;
        }
    }
    Ok(excluded)
}

// -- Stagewise ordering and inversion --------------------------------------------------------------

/// The terminal state a real procedure ended in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedTerminal {
    /// Look (zero-based) at which it ended.
    pub look: usize,
    /// Whether the stopping rule ended it.
    pub rule_stopped: bool,
    /// The arm decided, if any.
    pub decision: Option<usize>,
    /// The observed contrast estimate.
    pub estimate: f64,
}

/// Tail probabilities of the whole-procedure law at an observed terminal
/// state under the stagewise ordering.
///
/// The ordering ranks terminal states by evidence for the contrast: states
/// the rule stopped on in favour of the plus arm rank above everything else
/// (earlier stops higher), states the run reached the horizon on or stopped
/// for another reason rank by the estimate, and stops in favour of the minus
/// arm rank lowest (later stops higher). Terminal states with an undefined
/// estimate are counted in both tails.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderingPValues {
    /// Probability of a terminal state ranked at least as high.
    pub upper: f64,
    /// Probability of a terminal state ranked at least as low.
    pub lower: f64,
    /// Mass of terminal states with an undefined estimate.
    pub undefined_mass: f64,
}

struct Rank {
    class: u8,
    key: i64,
    estimate: f64,
}

fn rank_of(
    look: usize,
    rule_stopped: bool,
    decision: Option<usize>,
    estimate: f64,
    plus: usize,
    minus: usize,
) -> Rank {
    let look = i64::try_from(look).unwrap_or(i64::MAX);
    if rule_stopped && decision == Some(plus) {
        Rank { class: 2, key: -look, estimate }
    } else if rule_stopped && decision == Some(minus) {
        Rank { class: 0, key: look, estimate }
    } else {
        Rank { class: 1, key: 0, estimate }
    }
}

/// `Less`, `Equal` or `Greater` of `a` against `b` under the ordering.
fn compare(a: &Rank, b: &Rank) -> std::cmp::Ordering {
    a.class.cmp(&b.class).then(a.key.cmp(&b.key)).then_with(|| {
        if (a.estimate - b.estimate).abs() <= TIE_TOLERANCE {
            std::cmp::Ordering::Equal
        } else {
            a.estimate.total_cmp(&b.estimate)
        }
    })
}

/// Upper and lower tail probabilities of `observed` under `law`, for the
/// contrast of cell `plus` against cell `minus`.
///
/// # Errors
/// A cell outside the procedure or a non-finite observed estimate.
pub fn ordering_p_values(
    law: &ProcedureLaw,
    plus: usize,
    minus: usize,
    observed: &ObservedTerminal,
) -> Result<OrderingPValues, Error> {
    if plus >= law.cells || minus >= law.cells || plus == minus {
        return Err(Error::Shape("a contrast needs two different cells of the procedure"));
    }
    if !observed.estimate.is_finite() {
        return Err(Error::InvalidParameter("the observed estimate must be finite"));
    }
    let target = rank_of(
        observed.look,
        observed.rule_stopped,
        observed.decision,
        observed.estimate,
        plus,
        minus,
    );
    let (mut upper, mut lower, mut undefined) = (0.0, 0.0, 0.0);
    for atom in &law.atoms {
        let (Some(p), Some(m)) = (atom.cell_mean(plus), atom.cell_mean(minus)) else {
            undefined += atom.probability;
            continue;
        };
        let value = p - m;
        let rank = rank_of(atom.look, atom.rule_stopped, atom.decision, value, plus, minus);
        match compare(&rank, &target) {
            std::cmp::Ordering::Greater => upper += atom.probability,
            std::cmp::Ordering::Less => lower += atom.probability,
            std::cmp::Ordering::Equal => {
                upper += atom.probability;
                lower += atom.probability;
            }
        }
    }
    Ok(OrderingPValues {
        upper: upper + undefined,
        lower: lower + undefined,
        undefined_mass: undefined,
    })
}

/// Candidate contrasts and nuisance baselines of an inversion.
#[derive(Clone, Debug, PartialEq)]
pub struct InversionGrid {
    /// Contrast values `theta_plus - theta_minus` tested.
    pub contrasts: Vec<f64>,
    /// Baseline success probabilities of the minus arm (and of every other
    /// arm) that the supremum is taken over.
    pub baselines: Vec<f64>,
}

/// The confidence set of an inversion.
#[derive(Clone, Debug, PartialEq)]
pub struct InversionInterval {
    /// The tested contrasts that were not rejected, ascending.
    pub accepted: Vec<f64>,
    /// Smallest accepted contrast.
    pub lower: Option<f64>,
    /// Largest accepted contrast.
    pub upper: Option<f64>,
    /// Laws evaluated (observed work).
    pub laws_evaluated: usize,
    /// Always [`DERIVED_UNMEASURED`]: exact for the declared grid and
    /// procedure, with no measured coverage.
    pub calibration: &'static str,
}

/// Invert the stagewise-ordered test of every grid contrast, taking the
/// supremum of each tail probability over the declared baselines, and accept
/// a contrast when neither tail probability falls below `alpha / 2`.
///
/// The accepted set is exact for the declared procedure and grid: each test
/// uses the exact law of the whole procedure (policy, delay, exposure,
/// stopping and selection) under the tested truth. Contrasts and baselines
/// between the grid points, and baselines outside the grid, are not tested.
///
/// # Errors
/// A premise failure, a contrast of arm 1 against arm 0 on a procedure with
/// other than two arms in one context, an empty or oversized grid, or an
/// enumeration failure.
pub fn inversion_interval(
    spec: &ProcedureSpec,
    observed: &ObservedTerminal,
    grid: &InversionGrid,
    alpha: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<InversionInterval, Error> {
    admit(spec, UncertaintyParadigm::WholeProcedureInversion)?;
    if spec.arms() != 2 {
        return Err(Error::Shape(
            "an inversion contrasts arm 1 against arm 0 of a two-arm procedure",
        ));
    }
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter("the miscoverage level must lie in (0, 1)"));
    }
    if grid.contrasts.is_empty() || grid.baselines.is_empty() {
        return Err(Error::Shape("an inversion needs at least one contrast and one baseline"));
    }
    if grid.contrasts.len() * grid.baselines.len() > MAX_INVERSION_LAWS {
        return Err(Error::Limit("the inversion grid exceeds its law bound"));
    }
    if grid
        .contrasts
        .iter()
        .chain(&grid.baselines)
        .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
    {
        return Err(Error::InvalidParameter("grid values must be finite and within [-1, 1]"));
    }
    let mut accepted = Vec::new();
    let mut evaluated = 0;
    for &delta in &grid.contrasts {
        let (mut upper_p, mut lower_p) = (0.0_f64, 0.0_f64);
        for &baseline in &grid.baselines {
            let other = baseline + delta;
            if !(0.0..=1.0).contains(&baseline) || !(0.0..=1.0).contains(&other) {
                continue;
            }
            let law = enumerate_law(
                spec,
                &OutcomeKernel::Fixed { success_probabilities: vec![baseline, other] },
                cancelled,
            )?;
            evaluated += 1;
            let p = ordering_p_values(&law, 1, 0, observed)?;
            upper_p = upper_p.max(p.upper);
            lower_p = lower_p.max(p.lower);
        }
        if upper_p >= alpha / 2.0 && lower_p >= alpha / 2.0 {
            accepted.push(delta);
        }
    }
    accepted.sort_by(f64::total_cmp);
    Ok(InversionInterval {
        lower: accepted.first().copied(),
        upper: accepted.last().copied(),
        accepted,
        laws_evaluated: evaluated,
        calibration: DERIVED_UNMEASURED,
    })
}
