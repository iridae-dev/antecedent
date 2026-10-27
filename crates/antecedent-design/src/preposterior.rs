//! Preposterior decision analysis: the expected value of sample information.
//!
//! A candidate design collects `n` observations whose sampling distribution depends
//! on the decision state `θ` ([`DecisionSignal`]). Before the data arrive, the value
//! of running the design is the expected value of sample information
//!
//! ```text
//! EVSI(n) = E_y[ max_{a∈F} E[U(a,θ) | y] ] − max_{a∈F} E[U(a,θ)]
//!         = E[regret of the current Bayes action] − E_y[regret of the Bayes action after y]
//! ```
//!
//! i.e. the expected reduction in decision regret (terminal opportunity loss) relative
//! to perfect information. `F` is the admissible action set fixed by the problem's
//! constraints under the current belief, so the terminal-act space is the same
//! before and after sampling; constraints exclude actions and never enter as cost.
//! EVSI is non-negative, bounded by the expected value of perfect information (EVPI),
//! non-decreasing in `n`, unchanged when a constant is added to the utility, and
//! scales with a positive rescaling of the utility.
//!
//! Evaluation is exact where the model allows it and a labelled Monte Carlo
//! estimate otherwise:
//!
//! * [`DecisionPrior::Draws`] with a signal of finite support (for example
//!   [`BinomialSignal`]): the posterior over the draws is exact and the expectation
//!   over the data is a finite sum — [`ScoreEvaluation::Exact`].
//! * [`DecisionPrior::Normal`] with a Gaussian sample-mean signal and a utility
//!   affine in the state: the conjugate normal model. The posterior mean is
//!   preposteriorly `N(μ₀, s²)` with `s² = τ²·nτ²/(nτ² + σ²)`, and the expectation of
//!   the upper envelope of the affine expected utilities is a sum of normal
//!   linear-loss integrals — [`ScoreEvaluation::Exact`].
//! * [`DecisionPrior::Draws`] with a continuous signal: each draw samples a state,
//!   simulates the statistic, updates the draws exactly, and scores the regret
//!   reduction; the mean over draws is unbiased for EVSI —
//!   [`ScoreEvaluation::MonteCarlo`].
//!
//! References: H. Raiffa and R. Schlaifer, *Applied Statistical Decision Theory*
//! (Harvard, 1961), ch. 1 (preposterior analysis), ch. 4 (opportunity loss, EVPI and
//! EVSI) and ch. 5 (linear terminal analysis and the normal linear-loss integral);
//! A. E. Ades, G. Lu and K. Claxton, "Expected value of sample information
//! calculations in medical decision modeling", *Medical Decision Making* 24(2),
//! 2004, pp. 207–227 (Monte Carlo EVSI).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Uniform index draws truncate a non-negative float; envelope slopes are compared
// exactly to merge identical lines.
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::{
    CanonicalWriter, CausalRng, ExecutionContext, RngFactory, StreamDomain, mix64,
};
use antecedent_kernels::{norm_cdf, norm_pdf, norm_sf, standard_normal};
use antecedent_stats::{Welford, ln_gamma};

use crate::candidate::CandidateDesign;
use crate::decision::{DecisionProblem, DecisionTable};
use crate::error::DesignError;
use crate::result::ScoreEvaluation;

/// Current belief about the decision state.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionPrior<O> {
    /// Equally weighted draws of the state (for example posterior draws).
    Draws(Vec<O>),
    /// Normal belief `θ ~ N(mean, variance)` on a scalar state. Conjugate with a
    /// Gaussian sample-mean signal; requires a utility that declares affine state
    /// coefficients ([`crate::Utility::affine_in_state`]) and no constraints.
    Normal {
        /// Prior mean `μ₀`.
        mean: f64,
        /// Prior variance `τ²` (> 0).
        variance: f64,
    },
}

/// Sampling model of the data a candidate design collects about the decision state.
pub trait DecisionSignal<O>: Send + Sync {
    /// Signal name for diagnostics.
    fn name(&self) -> &str;

    /// Number of observations `candidate` collects about the decision state, or
    /// `None` when the signal has no data model for this kind of candidate (the
    /// ranker then records the candidate as unlicensed instead of scoring it).
    ///
    /// The default reads the declared row counts of sampling and environment plans;
    /// override it when environment rows are not draws of this signal.
    fn sample_size(&self, candidate: &CandidateDesign) -> Option<u64> {
        match candidate {
            CandidateDesign::IncreaseSamplingRate(plan) => Some(plan.additional_samples),
            CandidateDesign::ObserveEnvironment(plan) => Some(plan.additional_rows),
            CandidateDesign::Measure(_) | CandidateDesign::Intervene(_) => None,
        }
    }

    /// Every value the sufficient statistic of `n ≥ 1` observations can take, when
    /// that set is finite. Finite signals are evaluated by exact enumeration.
    fn finite_support(&self, n: u64) -> Option<Vec<f64>> {
        let _ = n;
        None
    }

    /// Normalized log density (or log mass) of `statistic` from `n ≥ 1` observations
    /// under each state, written into `out` (length `states.len()`). `-inf` marks an
    /// impossible statistic.
    ///
    /// # Errors
    ///
    /// States outside the signal's domain.
    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[O],
        out: &mut [f64],
    ) -> Result<(), DesignError>;

    /// Draw the sufficient statistic of `n ≥ 1` observations under `state`.
    ///
    /// # Errors
    ///
    /// A state outside the signal's domain.
    fn sample_statistic(&self, state: &O, n: u64, rng: &mut CausalRng) -> Result<f64, DesignError>;

    /// Known variance `σ²` of one observation when the statistic is a Gaussian sample
    /// mean `ȳ ~ N(θ, σ²/n)`. Enables the conjugate-normal closed form.
    fn gaussian_noise_variance(&self) -> Option<f64> {
        None
    }
}

/// Sample mean of `n` Gaussian observations with known variance: `ȳ ~ N(θ, σ²/n)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianMeanSignal {
    noise_variance: f64,
}

impl GaussianMeanSignal {
    /// Construct from the variance `σ²` of a single observation.
    ///
    /// # Errors
    ///
    /// Non-positive or non-finite variance.
    pub fn new(noise_variance: f64) -> Result<Self, DesignError> {
        if !(noise_variance.is_finite() && noise_variance > 0.0) {
            return Err(DesignError::Config(format!(
                "Gaussian signal noise variance must be finite and > 0 (got {noise_variance})"
            )));
        }
        Ok(Self { noise_variance })
    }

    /// Variance `σ²` of one observation.
    #[must_use]
    pub const fn noise_variance(&self) -> f64 {
        self.noise_variance
    }
}

/// A sample mean needs at least one observation: a zero-size study carries no signal (its value
/// of information is exactly zero, which the analysis returns itself), and is not the signal of
/// a one-observation study.
fn require_observations(n: u64) -> Result<(), DesignError> {
    if n == 0 {
        Err(DesignError::Config("a Gaussian sample mean needs at least one observation".into()))
    } else {
        Ok(())
    }
}

fn finite_state(state: f64) -> Result<f64, DesignError> {
    if state.is_finite() {
        Ok(state)
    } else {
        Err(DesignError::Shape(format!("decision state {state} is not finite")))
    }
}

impl DecisionSignal<f64> for GaussianMeanSignal {
    fn name(&self) -> &'static str {
        "gaussian_mean"
    }

    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        require_observations(n)?;
        let var_n = self.noise_variance / n as f64;
        if !statistic.is_finite() || !var_n.is_finite() || var_n <= 0.0 {
            return Err(DesignError::Numerical(
                "invalid Gaussian sample-mean statistic or variance".into(),
            ));
        }
        let log_norm = -0.5 * (std::f64::consts::TAU.ln() + var_n.ln());
        let sd = var_n.sqrt();
        for (slot, &theta) in out.iter_mut().zip(states) {
            let theta = finite_state(theta)?;
            let dev = statistic - theta;
            let standardized = if dev.is_finite() { dev / sd } else { statistic / sd - theta / sd };
            *slot = log_norm - 0.5 * standardized.powi(2);
        }
        Ok(())
    }

    fn sample_statistic(
        &self,
        state: &f64,
        n: u64,
        rng: &mut CausalRng,
    ) -> Result<f64, DesignError> {
        require_observations(n)?;
        let sd_n = (self.noise_variance / n as f64).sqrt();
        Ok(finite_state(*state)? + sd_n * standard_normal(rng))
    }

    fn gaussian_noise_variance(&self) -> Option<f64> {
        Some(self.noise_variance)
    }
}

/// Number of successes in `n` Bernoulli trials whose success probability is the
/// decision state `θ ∈ [0, 1]`. Finite support `{0, …, n}`, so preposterior values
/// are exact; enumeration costs `O(n · draws · actions)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BinomialSignal;

fn probability_state(state: f64) -> Result<f64, DesignError> {
    if (0.0..=1.0).contains(&state) {
        Ok(state)
    } else {
        Err(DesignError::Shape(format!(
            "binomial signal state {state} is not a probability in [0, 1]"
        )))
    }
}

/// Trials below which a binomial count is the sum of Bernoulli draws, the exact stream the
/// signal always used; larger studies use a constant-time sampler.
const BERNOULLI_SUM_LIMIT: u64 = 64;

/// A draw from `Binomial(n, p)`.
///
/// Up to [`BERNOULLI_SUM_LIMIT`] trials this sums Bernoulli draws. Beyond it the draw is made for
/// `q = min(p, 1 − p)` (and reflected): by waiting-time inversion when the mean `n q` is below
/// ten, whose cost is the mean, and otherwise by Hörmann's transformed rejection (BTRS), whose
/// cost does not grow with `n`.
fn sample_binomial(n: u64, p: f64, rng: &mut CausalRng) -> u64 {
    if n <= BERNOULLI_SUM_LIMIT {
        return (0..n).filter(|_| rng.next_f64() < p).count() as u64;
    }
    if p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    let reflect = p > 0.5;
    let q = if reflect { 1.0 - p } else { p };
    let nf = n as f64;
    let successes =
        if nf * q < 10.0 { binomial_inversion(n, q, rng) } else { binomial_btrs(n, q, rng) };
    if reflect { n - successes } else { successes }
}

/// Binomial draw by waiting times: the geometric gaps between successes are summed until they
/// pass `n` trials. Expected cost `n q + 1` draws.
fn binomial_inversion(n: u64, q: f64, rng: &mut CausalRng) -> u64 {
    let log_fail = (-q).ln_1p();
    let mut position = 0.0_f64;
    let mut successes = 0_u64;
    loop {
        // Trials until the next success is geometric: ⌈ln(U) / ln(1 − q)⌉, U in (0, 1].
        let u = 1.0 - rng.next_f64();
        position += (u.ln() / log_fail).floor() + 1.0;
        if position > n as f64 {
            return successes;
        }
        successes += 1;
    }
}

/// Binomial draw by Hörmann's BTRS transformed rejection; valid for `n q ≥ 10`, `q ≤ 1/2`.
fn binomial_btrs(n: u64, q: f64, rng: &mut CausalRng) -> u64 {
    let nf = n as f64;
    let spq = (nf * q * (1.0 - q)).sqrt();
    let b = 1.15 + 2.53 * spq;
    let a = -0.0873 + 0.0248 * b + 0.01 * q;
    let c = nf * q + 0.5;
    let v_r = 0.92 - 4.2 / b;
    let alpha = (2.83 + 5.1 / b) * spq;
    let lpq = (q / (1.0 - q)).ln();
    let m = ((nf + 1.0) * q).floor();
    let h = ln_gamma(m + 1.0) + ln_gamma(nf - m + 1.0);
    loop {
        let u = rng.next_f64() - 0.5;
        let v = rng.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + c).floor();
        if k < 0.0 || k > nf {
            continue;
        }
        if us >= 0.07 && v <= v_r {
            return k as u64;
        }
        let v = (v * alpha / (a / (us * us) + b)).ln();
        if v <= h - ln_gamma(k + 1.0) - ln_gamma(nf - k + 1.0) + (k - m) * lpq {
            return k as u64;
        }
    }
}

impl DecisionSignal<f64> for BinomialSignal {
    fn name(&self) -> &'static str {
        "binomial"
    }

    fn finite_support(&self, n: u64) -> Option<Vec<f64>> {
        Some((0..=n).map(|y| y as f64).collect())
    }

    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        let nf = n as f64;
        if !(statistic >= 0.0 && statistic <= nf && statistic.fract() == 0.0) {
            out[..states.len()].fill(f64::NEG_INFINITY);
            return Ok(());
        }
        let ln_choose =
            ln_gamma(nf + 1.0) - ln_gamma(statistic + 1.0) - ln_gamma(nf - statistic + 1.0);
        let failures = nf - statistic;
        for (slot, &theta) in out.iter_mut().zip(states) {
            let theta = probability_state(theta)?;
            let success = if statistic > 0.0 { statistic * theta.ln() } else { 0.0 };
            let failure = if failures > 0.0 { failures * (1.0 - theta).ln() } else { 0.0 };
            *slot = ln_choose + success + failure;
        }
        Ok(())
    }

    fn sample_statistic(
        &self,
        state: &f64,
        n: u64,
        rng: &mut CausalRng,
    ) -> Result<f64, DesignError> {
        let theta = probability_state(*state)?;
        Ok(sample_binomial(n, theta, rng) as f64)
    }
}

/// Belief-specific part of a prepared preposterior analysis.
enum Model<'a, O> {
    /// Equally weighted draws; `gaps[i*K + k] = U(adm_i, θ_k) − U(a*, θ_k)`.
    Draws { states: &'a [O], gaps: Vec<f64>, n_admissible: usize, prior_weights: Option<Vec<f64>> },
    /// Conjugate normal; `lines[a] = (Δc_a, Δβ_a)` with expected utility relative to
    /// the Bayes action equal to `Δc_a + Δβ_a·(m − μ₀)` at posterior mean `m`.
    Normal { variance: f64, noise_variance: f64, lines: Vec<(f64, f64)> },
}

/// A decision problem prepared for preposterior analysis under one belief and one
/// signal. Utilities and admissibility are evaluated once.
pub struct PreposteriorAnalysis<'a, O> {
    signal: &'a dyn DecisionSignal<O>,
    model: Model<'a, O>,
    bayes_action: usize,
    prior_expected_utility: f64,
    evpi: f64,
}

impl<'a, O> PreposteriorAnalysis<'a, O> {
    /// Prepare `problem` under `prior` and `signal`.
    ///
    /// # Errors
    ///
    /// [`DesignError::NoAdmissibleAction`] when no action meets the chance threshold;
    /// empty draws or actions; utility callback failures; and, for a normal prior,
    /// constraints, a utility without declared affine coefficients, a non-Gaussian
    /// signal, or an invalid prior (these are the conditions under which the
    /// conjugate closed form holds).
    pub fn new<A>(
        problem: &DecisionProblem<A, O>,
        prior: &'a DecisionPrior<O>,
        signal: &'a dyn DecisionSignal<O>,
    ) -> Result<Self, DesignError> {
        match prior {
            DecisionPrior::Draws(states) => Self::from_draws(problem, states, signal),
            DecisionPrior::Normal { mean, variance } => {
                Self::from_normal(problem, *mean, *variance, signal)
            }
        }
    }

    fn from_draws<A>(
        problem: &DecisionProblem<A, O>,
        states: &'a [O],
        signal: &'a dyn DecisionSignal<O>,
    ) -> Result<Self, DesignError> {
        let table = DecisionTable::build(problem, states)?;
        let (bayes_action, prior_expected_utility) = table.bayes_action();
        let k = table.n_outcomes;
        let mut gaps = Vec::with_capacity(table.admissible.len() * k);
        for &a in &table.admissible {
            for draw in 0..k {
                gaps.push(table.utility(a, draw) - table.utility(bayes_action, draw));
            }
        }
        let n_admissible = table.admissible.len();
        let evpi = (0..k)
            .map(|draw| (0..n_admissible).map(|i| gaps[i * k + draw]).fold(0.0, f64::max))
            .sum::<f64>()
            / k as f64;
        Ok(Self {
            signal,
            model: Model::Draws { states, gaps, n_admissible, prior_weights: None },
            bayes_action,
            prior_expected_utility,
            evpi,
        })
    }

    /// Prepare the same finite-state experiment/update engine under a weighted belief.
    ///
    /// This preserves an earlier Bayesian update instead of resampling or treating the
    /// weighted states as equally likely. Utilities may be arbitrary; constraints currently
    /// refuse because their callback reports only an unweighted satisfaction aggregate.
    ///
    /// # Errors
    /// Invalid weights, empty states/actions, constraints, utility or signal failures.
    pub fn new_weighted<A>(
        problem: &DecisionProblem<A, O>,
        states: &'a [O],
        weights: &[f64],
        signal: &'a dyn DecisionSignal<O>,
    ) -> Result<Self, DesignError> {
        if !problem.constraints.is_empty() {
            return Err(DesignError::Config("weighted preposterior constraints need statewise satisfaction; aggregate callbacks cannot be reweighted".into()));
        }
        let weights = update_finite_belief(weights, &vec![0.0; states.len()])?;
        let table = DecisionTable::build(problem, states)?;
        let (bayes_action, prior_expected_utility) = table
            .admissible
            .iter()
            .map(|&action| {
                (
                    action,
                    weights
                        .iter()
                        .enumerate()
                        .map(|(slot, weight)| weight * table.utility(action, slot))
                        .sum::<f64>(),
                )
            })
            .max_by(|left, right| left.1.total_cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
            .ok_or_else(|| DesignError::NoAdmissibleAction("empty action set".into()))?;
        let gaps = table
            .admissible
            .iter()
            .flat_map(|&action| {
                (0..states.len())
                    .map(|slot| table.utility(action, slot) - table.utility(bayes_action, slot))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let n_admissible = table.admissible.len();
        let evpi = weights
            .iter()
            .enumerate()
            .map(|(slot, weight)| {
                weight
                    * (0..n_admissible)
                        .map(|action| gaps[action * states.len() + slot])
                        .fold(0.0, f64::max)
            })
            .sum();
        Ok(Self {
            signal,
            model: Model::Draws { states, gaps, n_admissible, prior_weights: Some(weights) },
            bayes_action,
            prior_expected_utility,
            evpi,
        })
    }

    fn from_normal<A>(
        problem: &DecisionProblem<A, O>,
        mean: f64,
        variance: f64,
        signal: &'a dyn DecisionSignal<O>,
    ) -> Result<Self, DesignError> {
        if problem.actions.is_empty() {
            return Err(DesignError::Config("decision problem has no actions".into()));
        }
        if !(mean.is_finite() && variance.is_finite() && variance > 0.0) {
            return Err(DesignError::Config(format!(
                "normal decision prior needs a finite mean and variance > 0 (got {mean}, {variance})"
            )));
        }
        if !problem.constraints.is_empty() {
            return Err(DesignError::Config(
                "chance constraints are evaluated on draws of the decision state; use \
                 DecisionPrior::Draws for a constrained problem"
                    .into(),
            ));
        }
        let Some(noise_variance) = signal.gaussian_noise_variance() else {
            return Err(DesignError::Config(format!(
                "a normal decision prior is conjugate only with a Gaussian sample-mean signal \
                 (signal `{}` is not)",
                signal.name()
            )));
        };
        if !(noise_variance.is_finite() && noise_variance > 0.0) {
            return Err(DesignError::Config(format!(
                "Gaussian signal noise variance must be finite and > 0 (got {noise_variance})"
            )));
        }
        let coefficients = problem
            .utility
            .affine_in_state(&problem.actions)
            .filter(|c| c.len() == problem.actions.len())
            .ok_or_else(|| {
                DesignError::Config(
                    "a normal decision prior needs a utility that declares affine state \
                     coefficients for every action (Utility::affine_in_state); use \
                     DecisionPrior::Draws for any other utility"
                        .into(),
                )
            })?;
        if coefficients.iter().any(|(alpha, beta)| !(alpha.is_finite() && beta.is_finite())) {
            return Err(DesignError::Numerical(
                "affine utility coefficients must be finite".into(),
            ));
        }
        let mut bayes_action = 0;
        let mut prior_expected_utility = f64::NEG_INFINITY;
        for (a, (alpha, beta)) in coefficients.iter().enumerate() {
            let eu = alpha + beta * mean;
            if eu > prior_expected_utility {
                prior_expected_utility = eu;
                bayes_action = a;
            }
        }
        let (alpha_star, beta_star) = coefficients[bayes_action];
        let lines: Vec<(f64, f64)> = coefficients
            .iter()
            .map(|&(alpha, beta)| {
                let slope = beta - beta_star;
                ((alpha - alpha_star + slope * mean).min(0.0), slope)
            })
            .collect();
        let evpi = expected_max_affine(&lines, variance.sqrt());
        Ok(Self {
            signal,
            model: Model::Normal { variance, noise_variance, lines },
            bayes_action,
            prior_expected_utility,
            evpi,
        })
    }

    /// Index of the Bayes action under the current belief.
    #[must_use]
    pub const fn bayes_action(&self) -> usize {
        self.bayes_action
    }

    /// Expected utility of the Bayes action under the current belief.
    #[must_use]
    pub const fn prior_expected_utility(&self) -> f64 {
        self.prior_expected_utility
    }

    /// Expected value of perfect information (the current expected regret).
    #[must_use]
    pub const fn expected_value_of_perfect_information(&self) -> f64 {
        self.evpi
    }

    /// Signal the analysis was prepared with.
    #[must_use]
    pub fn signal(&self) -> &'a dyn DecisionSignal<O> {
        self.signal
    }

    /// How [`Self::sample_evsi`] evaluates EVSI for `n` observations.
    #[must_use]
    pub fn evaluation(&self, n: u64) -> ScoreEvaluation {
        match &self.model {
            Model::Normal { .. } => ScoreEvaluation::Exact,
            Model::Draws { .. } if n == 0 => ScoreEvaluation::Exact,
            Model::Draws { .. } if self.signal.finite_support(n).is_some() => {
                ScoreEvaluation::Exact
            }
            Model::Draws { .. } => ScoreEvaluation::MonteCarlo,
        }
    }

    /// Exact EVSI for `n` observations, or `None` when only a Monte Carlo estimate
    /// is available ([`ScoreEvaluation::MonteCarlo`]).
    ///
    /// # Errors
    ///
    /// Signal likelihood failures, or a finite-support likelihood that does not sum
    /// to one.
    pub fn exact_evsi(&self, n: u64) -> Result<Option<f64>, DesignError> {
        if n == 0 {
            return Ok(Some(0.0));
        }
        match &self.model {
            Model::Normal { variance, noise_variance, lines } => {
                let nf = n as f64;
                // Var(E[θ | ȳ]) = τ² − (1/τ² + n/σ²)⁻¹ = τ²·nτ²/(nτ² + σ²).
                let preposterior_var =
                    variance * (nf * variance) / (nf * variance + noise_variance);
                Ok(Some(expected_max_affine(lines, preposterior_var.sqrt())))
            }
            Model::Draws { states, gaps, n_admissible, prior_weights } => {
                let Some(support) = self.signal.finite_support(n) else {
                    return Ok(None);
                };
                let k = states.len();
                let mut log_lik = vec![0.0; k];
                let mut weights = vec![0.0; k];
                let mut total = 0.0;
                let mut mass = 0.0;
                for y in support {
                    self.signal.log_likelihood(y, n, states, &mut log_lik)?;
                    let Some(max_ll) = finite_max(&log_lik)? else {
                        continue;
                    };
                    let scale = max_ll.exp();
                    let mut sum_w = 0.0;
                    for (sum_index, (w, &ll)) in weights.iter_mut().zip(&log_lik).enumerate() {
                        *w = (ll - max_ll).exp()
                            * prior_weights.as_ref().map_or(1.0, |prior| prior[sum_index]);
                        sum_w += *w;
                    }
                    mass += scale * sum_w;
                    total += scale * best_gap(gaps, *n_admissible, &weights);
                }
                let kf = if prior_weights.is_some() { 1.0 } else { k as f64 };
                mass /= kf;
                if (mass - 1.0).abs() > 1e-6 {
                    return Err(DesignError::Numerical(format!(
                        "signal `{}` likelihood sums to {mass} over its finite support for n = {n}, \
                         not 1",
                        self.signal.name()
                    )));
                }
                Ok(Some(total / kf))
            }
        }
    }

    /// One draw whose expectation is EVSI for `n` observations: the exact value when
    /// [`Self::evaluation`] is [`ScoreEvaluation::Exact`], otherwise one Monte Carlo
    /// replicate (sample a state from the draws, simulate the statistic, update the
    /// draws exactly, and return the regret reduction of the updated Bayes action).
    ///
    /// The exactness probe is [`Self::evaluation`], which does not allocate the
    /// likelihood tables. Monte Carlo replicates reuse `log_lik` and `weights`
    /// through [`Self::draw_evsi_replicate`].
    ///
    /// # Errors
    ///
    /// Signal failures.
    pub fn sample_evsi(&self, n: u64, rng: &mut CausalRng) -> Result<f64, DesignError> {
        if self.evaluation(n) == ScoreEvaluation::Exact {
            return self.exact_evsi(n)?.ok_or_else(|| {
                DesignError::Numerical("exact EVSI evaluation produced no value".into())
            });
        }
        let mut log_lik = Vec::new();
        let mut weights = Vec::new();
        self.draw_evsi_replicate(n, rng, &mut log_lik, &mut weights)
    }

    /// Monte Carlo EVSI for `n` observations under an execution context.
    ///
    /// Replicate `r` draws from the stream [`evsi_replicate_rng`] keyed by
    /// `(run seed, purpose, r)`: a function of the semantic coordinates only, shared by
    /// every candidate (common random numbers, so tied candidates agree exactly) and
    /// independent of how many replicates or candidates ran before it. Cancellation is checked before every
    /// replicate, so the work between checks is one replicate.
    ///
    /// # Errors
    ///
    /// Cancellation, signal failures, or a call on a model whose EVSI is exact.
    pub(crate) fn monte_carlo_evsi(
        &self,
        n: u64,
        batches: u32,
        batch_size: u32,
        ctx: &ExecutionContext,
    ) -> Result<EvsiMonteCarlo, DesignError> {
        let mut acc = Welford::new();
        let mut log_lik = Vec::new();
        let mut weights = Vec::new();
        let replicates = u64::from(batches) * u64::from(batch_size);
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(DesignError::Cancelled);
            }
            let mut rng = evsi_replicate_rng(&ctx.rng, replicate);
            acc.push(self.draw_evsi_replicate(n, &mut rng, &mut log_lik, &mut weights)?);
        }
        if ctx.cancellation.is_cancelled() {
            return Err(DesignError::Cancelled);
        }
        Ok(EvsiMonteCarlo {
            mean: acc.mean(),
            stderr: acc.stderr_of_mean(),
            replicates: acc.count(),
            stream_digest: evsi_stream_digest(ctx.rng.master_seed()),
        })
    }

    /// One Monte Carlo EVSI replicate into caller-owned scratch. `log_lik` and
    /// `weights` are resized to the prior and overwritten. Call only when
    /// [`Self::evaluation`] is [`ScoreEvaluation::MonteCarlo`].
    ///
    /// # Errors
    ///
    /// Signal failures, or a call on a model whose EVSI is exact.
    pub(crate) fn draw_evsi_replicate(
        &self,
        n: u64,
        rng: &mut CausalRng,
        log_lik: &mut Vec<f64>,
        weights: &mut Vec<f64>,
    ) -> Result<f64, DesignError> {
        let Model::Draws { states, gaps, n_admissible, prior_weights } = &self.model else {
            return Err(DesignError::Config(
                "draw_evsi_replicate is the Monte Carlo path; the normal model is exact".into(),
            ));
        };
        let k = states.len();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "next_f64 is in [0, 1), so the product is a non-negative value below k"
        )]
        let truth = if let Some(prior) = prior_weights {
            let uniform = rng.next_f64();
            let mut cumulative = 0.0;
            prior
                .iter()
                .position(|weight| {
                    cumulative += *weight;
                    cumulative > uniform
                })
                .unwrap_or(k - 1)
        } else {
            ((rng.next_f64() * k as f64) as usize).min(k - 1)
        };
        let statistic = self.signal.sample_statistic(&states[truth], n, rng)?;
        log_lik.clear();
        log_lik.resize(k, 0.0);
        self.signal.log_likelihood(statistic, n, states, log_lik)?;
        if let Some(prior) = prior_weights {
            // Normalize log joint masses, not likelihoods alone: an impossible
            // prior state must never determine the numerical scaling.
            *weights = update_finite_belief(prior, log_lik)?;
            return Ok(best_gap(gaps, *n_admissible, weights));
        }
        let Some(max_ll) = finite_max(log_lik)? else {
            return Err(DesignError::Numerical(format!(
                "signal `{}` assigns zero likelihood to its own draw",
                self.signal.name()
            )));
        };
        weights.clear();
        weights.resize(k, 0.0);
        let mut sum_w = 0.0;
        for (w, &ll) in weights.iter_mut().zip(log_lik.iter()) {
            *w = (ll - max_ll).exp();
            sum_w += *w;
        }
        for w in weights.iter_mut() {
            *w /= sum_w;
        }
        Ok(best_gap(gaps, *n_admissible, weights))
    }
}

/// Purpose word of the Monte Carlo EVSI replicate streams.
const EVSI_REPLICATE_PURPOSE: u64 = 0x4556_5349_5250;

/// Outcome of [`PreposteriorAnalysis::monte_carlo_evsi`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EvsiMonteCarlo {
    /// Mean of the replicates.
    pub mean: f64,
    /// Standard error of the mean.
    pub stderr: f64,
    /// Replicates drawn.
    pub replicates: u64,
    /// Digest of the stream coordinates `(run seed, purpose)`.
    pub stream_digest: String,
}

/// Generator of one Monte Carlo EVSI replicate, keyed by the semantic coordinates
/// `(purpose, replicate)` under the run seed; shared across candidates.
#[must_use]
pub fn evsi_replicate_rng(rng: &RngFactory, replicate: u64) -> CausalRng {
    rng.stream_for(
        StreamDomain::Design,
        mix64(antecedent_core::derive_stream(EVSI_REPLICATE_PURPOSE, replicate)),
    )
}

/// Typed length-delimited digest of the replicate-stream coordinates.
fn evsi_stream_digest(run_seed: u64) -> String {
    CanonicalWriter::new("antecedent.design.evsi_replicate_stream.v1")
        .word(run_seed)
        .word(EVSI_REPLICATE_PURPOSE)
        .finish_hex()
}

/// Largest finite log-likelihood, `None` when every entry is `-inf`.
fn finite_max(log_lik: &[f64]) -> Result<Option<f64>, DesignError> {
    let mut max = f64::NEG_INFINITY;
    for &ll in log_lik {
        if ll.is_nan() || ll == f64::INFINITY {
            return Err(DesignError::Numerical(format!("signal log-likelihood {ll}")));
        }
        max = max.max(ll);
    }
    Ok((max > f64::NEG_INFINITY).then_some(max))
}

/// `max_i Σ_k w_k·gaps[i, k]` over admissible rows; non-negative because the Bayes
/// action's row is identically zero.
fn best_gap(gaps: &[f64], n_admissible: usize, weights: &[f64]) -> f64 {
    let k = weights.len();
    (0..n_admissible)
        .map(|i| gaps[i * k..(i + 1) * k].iter().zip(weights).map(|(g, w)| g * w).sum::<f64>())
        .fold(0.0, f64::max)
}

/// `E[max_a (c_a + d_a·s·Z)]` for `Z ~ N(0, 1)`, computed exactly as a sum of normal
/// linear-loss integrals over the segments of the upper envelope of the lines.
///
/// With `c_a ≤ 0` and a zero line present (the Bayes action), this is the expected
/// regret reduction of learning the state's preposterior mean with spread `s`.
#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
pub(crate) fn expected_max_affine(lines: &[(f64, f64)], s: f64) -> f64 {
    if !(s > 0.0) {
        return lines.iter().map(|&(c, _)| c).fold(f64::NEG_INFINITY, f64::max);
    }
    let mut sorted: Vec<(f64, f64)> = lines.iter().map(|&(c, d)| (c, d * s)).collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.total_cmp(&b.0)));
    // Keep the highest intercept among equal slopes (last after the sort).
    let mut distinct: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for line in sorted {
        #[allow(
            clippy::float_cmp,
            reason = "lines were sorted by slope, so bit-identical slopes are the duplicates to collapse"
        )]
        let same_slope = distinct.last().is_some_and(|last| last.1 == line.1);
        if same_slope {
            distinct.pop();
        }
        distinct.push(line);
    }
    let cross = |l: (f64, f64), r: (f64, f64)| (l.0 - r.0) / (r.1 - l.1);
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(distinct.len());
    for line in distinct {
        while hull.len() >= 2 {
            let (l1, l2) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            if cross(l1, line) <= cross(l1, l2) {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(line);
    }
    let mut total = 0.0;
    let mut left = f64::NEG_INFINITY;
    for (i, &(c, d)) in hull.iter().enumerate() {
        let right = hull.get(i + 1).map_or(f64::INFINITY, |&next| cross((c, d), next));
        let prob = if left >= 0.0 {
            norm_sf(left) - norm_sf(right)
        } else {
            norm_cdf(right) - norm_cdf(left)
        };
        let density = |z: f64| if z.is_finite() { norm_pdf(z) } else { 0.0 };
        total += c * prob + d * (density(left) - density(right));
        left = right;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_max_of_two_lines_is_the_normal_linear_loss_integral() {
        // E[max(0, c + dZ)] = d·φ(c/d) + c·Φ(c/d).
        for (c, d) in [(-0.3_f64, 1.2_f64), (-2.0, 0.5), (0.0, 3.0)] {
            let got = expected_max_affine(&[(0.0, 0.0), (c, d)], 1.0);
            let expected = d * norm_pdf(c / d) + c * norm_cdf(c / d);
            assert!((got - expected).abs() < 1e-14, "{c} {d}: {got} vs {expected}");
        }
        assert_eq!(expected_max_affine(&[(0.0, 0.0), (-1.0, 2.0)], 0.0), 0.0);
    }

    #[test]
    fn expected_max_envelope_drops_dominated_lines() {
        // The middle line is never on top; adding it must not change the value.
        let two = expected_max_affine(&[(0.0, -1.0), (0.0, 1.0)], 1.0);
        let three = expected_max_affine(&[(0.0, -1.0), (-5.0, 0.0), (0.0, 1.0)], 1.0);
        assert!((two - three).abs() < 1e-15);
        // E|Z| = sqrt(2/π).
        assert!((two - (2.0 / std::f64::consts::PI).sqrt()).abs() < 1e-14);
    }
}

/// Bayesian update of a finite belief, retaining every state slot and using log-sum-exp.
///
/// # Errors
/// Ragged/empty inputs, negative or nonfinite weights, invalid log likelihood, or zero
/// posterior mass. A zero prior weight remains zero and failed states cannot be hidden.
pub fn update_finite_belief(
    prior: &[f64],
    log_likelihood: &[f64],
) -> Result<Vec<f64>, DesignError> {
    antecedent_prob::finite_belief::condition(prior, log_likelihood)
        .map_err(|error| DesignError::Numerical(error.to_string()))
}

#[cfg(test)]
mod weighted_belief_tests {
    use super::*;
    use crate::decision::AffineUtility;
    use std::sync::Arc;
    #[test]
    fn gaussian_signal_keeps_large_finite_variance_and_scaled_deviations() {
        let signal = GaussianMeanSignal::new(f64::MAX).unwrap();
        let mut likelihood = [0.0; 2];
        signal.log_likelihood(1e200, 1, &[0.0, 1.0], &mut likelihood).unwrap();
        assert!(likelihood.iter().all(|value| value.is_finite()));
        assert!(signal.log_likelihood(f64::INFINITY, 1, &[0.0], &mut [0.0]).is_err());
        let tiny = GaussianMeanSignal::new(f64::from_bits(1)).unwrap();
        assert!(tiny.log_likelihood(0.0, 100, &[0.0], &mut [0.0]).is_err());
    }

    #[test]
    fn a_zero_size_gaussian_study_is_refused() {
        let signal = GaussianMeanSignal::new(1.0).unwrap();
        assert!(signal.log_likelihood(0.0, 0, &[0.0], &mut [0.0]).is_err());
        let mut rng = evsi_replicate_rng(&RngFactory::from_seed(1), 0);
        assert!(signal.sample_statistic(&0.0, 0, &mut rng).is_err());
    }

    #[test]
    fn large_binomial_draws_have_the_binomial_moments() {
        let mut rng = evsi_replicate_rng(&RngFactory::from_seed(7), 0);
        for (n, p) in
            [(1_000_u64, 0.3), (50_000, 0.5), (200, 0.01), (1_000_000, 0.9999), (400, 0.75)]
        {
            let draws = 4_000;
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for _ in 0..draws {
                let x = sample_binomial(n, p, &mut rng);
                assert!(x <= n);
                let x = x as f64;
                sum += x;
                sum_sq += x * x;
            }
            let mean = sum / f64::from(draws);
            let variance = sum_sq / f64::from(draws) - mean * mean;
            let (true_mean, true_variance) = (n as f64 * p, n as f64 * p * (1.0 - p));
            let mean_se = (true_variance / f64::from(draws)).sqrt();
            assert!((mean - true_mean).abs() < 5.0 * mean_se.max(1e-9), "n={n} p={p} mean={mean}");
            assert!(
                (variance - true_variance).abs() < 0.2 * true_variance.max(1e-9) + 1e-6,
                "n={n} p={p} variance={variance} vs {true_variance}"
            );
        }
    }

    #[test]
    fn weighted_preposterior_preserves_updated_state_mass_and_exact_finite_signal() {
        let weights = update_finite_belief(&[0.5, 0.5], &[0.1_f64.ln(), 0.9_f64.ln()]).unwrap();
        assert!((weights[1] - 0.9).abs() < 1e-12);
        let states = [0.0, 1.0];
        let problem = DecisionProblem::new(
            vec![0, 1],
            Arc::new(AffineUtility::new(vec![0.0, -0.6], vec![0.0, 1.0]).unwrap()),
            vec![],
        );
        let signal = BinomialSignal;
        let analysis =
            PreposteriorAnalysis::new_weighted(&problem, &states, &weights, &signal).unwrap();
        assert_eq!(analysis.bayes_action(), 1);
        assert!((analysis.prior_expected_utility() - 0.3).abs() < 1e-12);
        // One Bernoulli observation fully identifies either endpoint state.
        assert!((analysis.exact_evsi(1).unwrap().unwrap() - 0.06).abs() < 1e-12);
        assert!(update_finite_belief(&weights, &[f64::NEG_INFINITY, f64::NEG_INFINITY]).is_err());
        assert_eq!(
            update_finite_belief(&[1.0, 0.0], &[-1000.0, 0.0]).unwrap(),
            vec![1.0, 0.0],
            "a zero-prior high-likelihood state cannot underflow every supported state"
        );
    }
}
