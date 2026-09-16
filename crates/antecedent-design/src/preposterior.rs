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
#![allow(clippy::cast_sign_loss, clippy::float_cmp)]

use antecedent_core::CausalRng;
use antecedent_kernels::{norm_cdf, norm_pdf, norm_sf, standard_normal};
use antecedent_stats::ln_gamma;

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

fn finite_state(state: f64) -> Result<f64, DesignError> {
    if state.is_finite() {
        Ok(state)
    } else {
        Err(DesignError::Shape(format!("decision state {state} is not finite")))
    }
}

impl DecisionSignal<f64> for GaussianMeanSignal {
    fn name(&self) -> &str {
        "gaussian_mean"
    }

    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        let var_n = self.noise_variance / n.max(1) as f64;
        let log_norm = -0.5 * (std::f64::consts::TAU * var_n).ln();
        for (slot, &theta) in out.iter_mut().zip(states) {
            let dev = statistic - finite_state(theta)?;
            *slot = log_norm - dev * dev / (2.0 * var_n);
        }
        Ok(())
    }

    fn sample_statistic(
        &self,
        state: &f64,
        n: u64,
        rng: &mut CausalRng,
    ) -> Result<f64, DesignError> {
        let sd_n = (self.noise_variance / n.max(1) as f64).sqrt();
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

impl DecisionSignal<f64> for BinomialSignal {
    fn name(&self) -> &str {
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
        Ok((0..n).filter(|_| rng.next_f64() < theta).count() as f64)
    }
}

/// Belief-specific part of a prepared preposterior analysis.
enum Model<'a, O> {
    /// Equally weighted draws; `gaps[i*K + k] = U(adm_i, θ_k) − U(a*, θ_k)`.
    Draws { states: &'a [O], gaps: Vec<f64>, n_admissible: usize },
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
            model: Model::Draws { states, gaps, n_admissible },
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
            Model::Draws { states, gaps, n_admissible } => {
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
                    for (w, &ll) in weights.iter_mut().zip(&log_lik) {
                        *w = (ll - max_ll).exp();
                        sum_w += *w;
                    }
                    mass += scale * sum_w;
                    total += scale * best_gap(gaps, *n_admissible, &weights);
                }
                let kf = k as f64;
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
    /// # Errors
    ///
    /// Signal failures.
    pub fn sample_evsi(&self, n: u64, rng: &mut CausalRng) -> Result<f64, DesignError> {
        if let Some(exact) = self.exact_evsi(n)? {
            return Ok(exact);
        }
        let Model::Draws { states, gaps, n_admissible } = &self.model else {
            unreachable!("the normal model is always exact");
        };
        let k = states.len();
        let truth = ((rng.next_f64() * k as f64) as usize).min(k - 1);
        let statistic = self.signal.sample_statistic(&states[truth], n, rng)?;
        let mut log_lik = vec![0.0; k];
        self.signal.log_likelihood(statistic, n, states, &mut log_lik)?;
        let Some(max_ll) = finite_max(&log_lik)? else {
            return Err(DesignError::Numerical(format!(
                "signal `{}` assigns zero likelihood to its own draw",
                self.signal.name()
            )));
        };
        let mut weights: Vec<f64> = log_lik.iter().map(|ll| (ll - max_ll).exp()).collect();
        let sum_w: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum_w;
        }
        Ok(best_gap(gaps, *n_admissible, &weights))
    }
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
pub(crate) fn expected_max_affine(lines: &[(f64, f64)], s: f64) -> f64 {
    if !(s > 0.0) {
        return lines.iter().map(|&(c, _)| c).fold(f64::NEG_INFINITY, f64::max);
    }
    let mut sorted: Vec<(f64, f64)> = lines.iter().map(|&(c, d)| (c, d * s)).collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.total_cmp(&b.0)));
    // Keep the highest intercept among equal slopes (last after the sort).
    let mut distinct: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for line in sorted {
        if distinct.last().is_some_and(|last| last.1 == line.1) {
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
