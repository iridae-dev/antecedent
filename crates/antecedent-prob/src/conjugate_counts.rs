//! Conjugate count backends: Beta-Binomial and Gamma-Poisson.
//!
//! Two closed-form posteriors for queries about a single cell:
//!
//! * [`BetaBinomialPosterior`]: a proportion `θ` with a `Beta(α, β)` prior
//!   ([`PriorSpec::Beta`](crate::prior::PriorSpec::Beta)) and `k` successes in `n` trials has the posterior
//!   `Beta(α + k, β + n − k)`. The exceedance query `P(Y > threshold)` is such a
//!   proportion: a trial is an outcome in the cell and a success is an outcome
//!   strictly above the threshold ([`BetaBinomialPosterior::from_exceedance`]).
//! * [`GammaPoissonPosterior`]: a Poisson rate `λ` with a `Gamma(a, b)` prior
//!   ([`PriorSpec::Gamma`](crate::prior::PriorSpec::Gamma); shape `a`, rate `b`) and counts `y₁…yₙ` over unit
//!   exposures has the posterior `Gamma(a + Σyᵢ, b + n)`. With exposures `tᵢ` the
//!   rate is per unit exposure and the posterior is `Gamma(a + Σyᵢ, b + Σtᵢ)`.
//!
//! Each posterior reports its mean, variance, CDF, exact quantiles, equal-tailed
//! credible interval, posterior predictive distribution (beta-binomial and
//! negative-binomial) and the log marginal likelihood of the observed data. A
//! credible interval is a posterior statement under the declared prior and
//! model; it is not a frequentist confidence interval. A prior is an assumption
//! and never upgrades identification: the posterior describes the cell's
//! observed-data proportion or rate, nothing causal beyond what the cell's own
//! identification supports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "counts convert to f64 for the closed-form special functions"
)]

use antecedent_stats::{gamma_q, ln_beta, ln_gamma, regularized_incomplete_beta};

use crate::conjugate_moment_match::{BetaHyperparameters, GammaHyperparameters};
use crate::error::ProbError;
use crate::prior::PriorSet;

/// Bisection iteration cap; bisection stops earlier once the bracket is two adjacent floats.
const MAX_BISECTION: usize = 2_000;

fn check_level(level: f64) -> Result<(), ProbError> {
    if level.is_finite() && level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(ProbError::InvalidPrior { message: "credible level must be finite and in (0, 1)" })
    }
}

fn check_probability(p: f64) -> Result<(), ProbError> {
    if p.is_finite() && p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(ProbError::InvalidPrior {
            message: "quantile probability must be finite and in (0, 1)",
        })
    }
}

/// `ln C(n, k)`.
fn ln_choose(n: u64, k: u64) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Smallest `x` in `[lo, hi]` with `cdf(x) >= p` for a non-decreasing `cdf`.
fn invert_cdf(cdf: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64, p: f64) -> f64 {
    for _ in 0..MAX_BISECTION {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// `ln` of the beta-binomial pmf of `j` successes in `m` trials under `base`.
fn log_beta_binomial_pmf(base: BetaHyperparameters, m: u64, j: u64) -> f64 {
    ln_choose(m, j) + ln_beta(base.alpha + j as f64, base.beta + (m - j) as f64)
        - ln_beta(base.alpha, base.beta)
}

/// Beta-Binomial posterior of a proportion in one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBinomialPosterior {
    prior: BetaHyperparameters,
    successes: u64,
    trials: u64,
}

impl BetaBinomialPosterior {
    /// Posterior of `θ` after `successes` in `trials` Bernoulli trials under `prior`.
    ///
    /// # Errors
    ///
    /// An invalid prior or `successes > trials`.
    pub fn new(prior: BetaHyperparameters, successes: u64, trials: u64) -> Result<Self, ProbError> {
        prior.validate()?;
        if successes > trials {
            return Err(ProbError::Shape { message: "successes must not exceed trials" });
        }
        Ok(Self { prior, successes, trials })
    }

    /// Posterior under the [`PriorSpec::Beta`](crate::prior::PriorSpec::Beta) of `prior`.
    ///
    /// # Errors
    ///
    /// The set has no Beta prior, is invalid, or `successes > trials`.
    pub fn from_prior_set(
        prior: &PriorSet,
        successes: u64,
        trials: u64,
    ) -> Result<Self, ProbError> {
        prior.validate()?;
        let beta = prior
            .beta_prior()
            .ok_or(ProbError::InvalidPrior { message: "the prior set carries no Beta prior" })?;
        Self::new(beta, successes, trials)
    }

    /// Exceedance posterior: `θ = P(Y > threshold)` within the cell whose outcomes are
    /// `outcomes`. A success is an outcome strictly greater than `threshold`.
    ///
    /// # Errors
    ///
    /// A non-finite threshold or outcome, an invalid prior, or no outcomes.
    pub fn from_exceedance(
        prior: &PriorSet,
        outcomes: &[f64],
        threshold: f64,
    ) -> Result<Self, ProbError> {
        if !threshold.is_finite() {
            return Err(ProbError::InvalidPrior { message: "exceedance threshold must be finite" });
        }
        if outcomes.is_empty() {
            return Err(ProbError::Shape { message: "exceedance needs at least one outcome" });
        }
        let mut successes = 0u64;
        for &y in outcomes {
            if !y.is_finite() {
                return Err(ProbError::InvalidPrior {
                    message: "exceedance outcomes must be finite",
                });
            }
            successes += u64::from(y > threshold);
        }
        Self::from_prior_set(prior, successes, outcomes.len() as u64)
    }

    /// Prior hyperparameters.
    #[must_use]
    pub const fn prior(&self) -> BetaHyperparameters {
        self.prior
    }

    /// Observed successes.
    #[must_use]
    pub const fn successes(&self) -> u64 {
        self.successes
    }

    /// Observed trials.
    #[must_use]
    pub const fn trials(&self) -> u64 {
        self.trials
    }

    /// Posterior `Beta(α + k, β + n − k)`.
    #[must_use]
    pub fn posterior(&self) -> BetaHyperparameters {
        BetaHyperparameters {
            alpha: self.prior.alpha + self.successes as f64,
            beta: self.prior.beta + (self.trials - self.successes) as f64,
        }
    }

    /// Posterior mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.posterior().mean()
    }

    /// Posterior variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.posterior().variance()
    }

    /// Posterior CDF `P(θ ≤ x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        let post = self.posterior();
        if x.is_nan() {
            return f64::NAN;
        }
        regularized_incomplete_beta(x.clamp(0.0, 1.0), post.alpha, post.beta)
    }

    /// Posterior quantile at probability `p`.
    ///
    /// # Errors
    ///
    /// `p` outside `(0, 1)`.
    pub fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        check_probability(p)?;
        let post = self.posterior();
        Ok(invert_cdf(|x| regularized_incomplete_beta(x, post.alpha, post.beta), 0.0, 1.0, p))
    }

    /// Equal-tailed credible interval at `level`.
    ///
    /// # Errors
    ///
    /// `level` outside `(0, 1)`.
    pub fn credible_interval(&self, level: f64) -> Result<(f64, f64), ProbError> {
        check_level(level)?;
        let tail = 0.5 * (1.0 - level);
        Ok((self.quantile(tail)?, self.quantile(1.0 - tail)?))
    }

    /// `ln p(k | n, prior)`: the log marginal likelihood of the observed binomial count.
    #[must_use]
    pub fn log_marginal_likelihood(&self) -> f64 {
        log_beta_binomial_pmf(self.prior, self.trials, self.successes)
    }

    /// `ln P(J = j)` for `j` successes in `m` future trials (posterior predictive,
    /// beta-binomial).
    ///
    /// # Errors
    ///
    /// `j > m`.
    pub fn predictive_log_pmf(&self, m: u64, j: u64) -> Result<f64, ProbError> {
        if j > m {
            return Err(ProbError::Shape {
                message: "predictive successes must not exceed trials",
            });
        }
        Ok(log_beta_binomial_pmf(self.posterior(), m, j))
    }

    /// `P(J = j)` for `j` successes in `m` future trials.
    ///
    /// # Errors
    ///
    /// `j > m`.
    pub fn predictive_pmf(&self, m: u64, j: u64) -> Result<f64, ProbError> {
        self.predictive_log_pmf(m, j).map(f64::exp)
    }

    /// Posterior predictive mean of the number of successes in `m` future trials.
    #[must_use]
    pub fn predictive_mean(&self, m: u64) -> f64 {
        m as f64 * self.mean()
    }

    /// Posterior predictive variance of the number of successes in `m` future trials.
    #[must_use]
    pub fn predictive_variance(&self, m: u64) -> f64 {
        let post = self.posterior();
        let total = post.alpha + post.beta;
        let m = m as f64;
        m * post.alpha * post.beta * (total + m) / (total * total * (total + 1.0))
    }

    /// `P(J ≤ j)` for `m` future trials.
    ///
    /// # Errors
    ///
    /// `j > m`.
    pub fn predictive_cdf(&self, m: u64, j: u64) -> Result<f64, ProbError> {
        if j > m {
            return Err(ProbError::Shape {
                message: "predictive successes must not exceed trials",
            });
        }
        let mut total = 0.0;
        for i in 0..=j {
            total += self.predictive_pmf(m, i)?;
        }
        Ok(total.min(1.0))
    }

    /// Smallest `j` with `P(J ≤ j) ≥ p` among `m` future trials.
    ///
    /// # Errors
    ///
    /// `p` outside `(0, 1)`.
    pub fn predictive_quantile(&self, m: u64, p: f64) -> Result<u64, ProbError> {
        check_probability(p)?;
        let mut total = 0.0;
        for j in 0..=m {
            total += self.predictive_pmf(m, j)?;
            if total >= p {
                return Ok(j);
            }
        }
        Ok(m)
    }
}

/// Gamma-Poisson posterior of a rate in one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaPoissonPosterior {
    prior: GammaHyperparameters,
    total_count: u64,
    exposure: f64,
    /// `Σ (yᵢ ln tᵢ − ln yᵢ!)`: the part of the Poisson log likelihood that does not
    /// involve the rate.
    log_kernel: f64,
}

impl GammaPoissonPosterior {
    /// Posterior of the rate after `counts` over unit exposures (`exposures = None`) or
    /// over the given positive `exposures`.
    ///
    /// # Errors
    ///
    /// An invalid prior, no counts, an exposure length mismatch, or a non-positive or
    /// non-finite exposure.
    pub fn from_counts(
        prior: GammaHyperparameters,
        counts: &[u64],
        exposures: Option<&[f64]>,
    ) -> Result<Self, ProbError> {
        prior.validate()?;
        if counts.is_empty() {
            return Err(ProbError::Shape { message: "a Poisson rate needs at least one count" });
        }
        if let Some(t) = exposures
            && t.len() != counts.len()
        {
            return Err(ProbError::Shape { message: "exposures length must match counts" });
        }
        let mut total_count = 0u64;
        let mut exposure = 0.0;
        let mut log_kernel = 0.0;
        for (i, &y) in counts.iter().enumerate() {
            let t = exposures.map_or(1.0, |t| t[i]);
            if !(t.is_finite() && t > 0.0) {
                return Err(ProbError::InvalidPrior {
                    message: "Poisson exposures must be finite and > 0",
                });
            }
            total_count = total_count
                .checked_add(y)
                .ok_or(ProbError::Shape { message: "total count overflows u64" })?;
            exposure += t;
            log_kernel += y as f64 * t.ln() - ln_gamma(y as f64 + 1.0);
        }
        Ok(Self { prior, total_count, exposure, log_kernel })
    }

    /// Posterior from the sufficient statistics `Σ y` and `Σ t` alone.
    ///
    /// The data-only constant of the Poisson likelihood is not recoverable from the
    /// totals, so [`Self::log_marginal_likelihood`] of this constructor omits it: it
    /// compares priors on the same data but is not the density of the counts.
    ///
    /// # Errors
    ///
    /// An invalid prior or a negative or non-finite exposure.
    pub fn from_totals(
        prior: GammaHyperparameters,
        total_count: u64,
        exposure: f64,
    ) -> Result<Self, ProbError> {
        prior.validate()?;
        if !(exposure.is_finite() && exposure >= 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "Poisson total exposure must be finite and >= 0",
            });
        }
        Ok(Self { prior, total_count, exposure, log_kernel: 0.0 })
    }

    /// Posterior under the [`PriorSpec::Gamma`](crate::prior::PriorSpec::Gamma) of `prior`.
    ///
    /// # Errors
    ///
    /// The set has no Gamma prior, is invalid, or the counts are invalid.
    pub fn from_prior_set(
        prior: &PriorSet,
        counts: &[u64],
        exposures: Option<&[f64]>,
    ) -> Result<Self, ProbError> {
        prior.validate()?;
        let gamma = prior
            .gamma_prior()
            .ok_or(ProbError::InvalidPrior { message: "the prior set carries no Gamma prior" })?;
        Self::from_counts(gamma, counts, exposures)
    }

    /// Prior hyperparameters.
    #[must_use]
    pub const fn prior(&self) -> GammaHyperparameters {
        self.prior
    }

    /// `Σ yᵢ`.
    #[must_use]
    pub const fn total_count(&self) -> u64 {
        self.total_count
    }

    /// `Σ tᵢ` (the number of observations for unit exposures).
    #[must_use]
    pub const fn exposure(&self) -> f64 {
        self.exposure
    }

    /// Posterior `Gamma(a + Σy, b + Σt)`.
    #[must_use]
    pub fn posterior(&self) -> GammaHyperparameters {
        GammaHyperparameters {
            shape: self.prior.shape + self.total_count as f64,
            rate: self.prior.rate + self.exposure,
        }
    }

    /// Posterior mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.posterior().mean()
    }

    /// Posterior variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.posterior().variance()
    }

    /// Posterior CDF `P(λ ≤ x)`.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        let post = self.posterior();
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 {
            return 0.0;
        }
        1.0 - gamma_q(post.shape, post.rate * x)
    }

    /// Posterior quantile at probability `p`.
    ///
    /// # Errors
    ///
    /// `p` outside `(0, 1)`.
    pub fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        check_probability(p)?;
        let post = self.posterior();
        let mut hi = post.mean() + 10.0 * post.variance().sqrt() + 1.0 / post.rate;
        for _ in 0..200 {
            if self.cdf(hi) >= p {
                break;
            }
            hi *= 2.0;
        }
        Ok(invert_cdf(|x| self.cdf(x), 0.0, hi, p))
    }

    /// Equal-tailed credible interval at `level`.
    ///
    /// # Errors
    ///
    /// `level` outside `(0, 1)`.
    pub fn credible_interval(&self, level: f64) -> Result<(f64, f64), ProbError> {
        check_level(level)?;
        let tail = 0.5 * (1.0 - level);
        Ok((self.quantile(tail)?, self.quantile(1.0 - tail)?))
    }

    /// `ln p(y | prior)`: the log marginal likelihood of the observed counts.
    #[must_use]
    pub fn log_marginal_likelihood(&self) -> f64 {
        let a = self.prior.shape;
        let b = self.prior.rate;
        let s = self.total_count as f64;
        a * b.ln() - ln_gamma(a) + ln_gamma(a + s) - (a + s) * (b + self.exposure).ln()
            + self.log_kernel
    }

    /// `ln P(Y = y)` for a new count over `exposure` (posterior predictive, negative
    /// binomial).
    ///
    /// # Errors
    ///
    /// A non-positive or non-finite `exposure`.
    pub fn predictive_log_pmf(&self, y: u64, exposure: f64) -> Result<f64, ProbError> {
        if !(exposure.is_finite() && exposure > 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "predictive exposure must be finite and > 0",
            });
        }
        let post = self.posterior();
        let y = y as f64;
        let denom = post.rate + exposure;
        Ok(ln_gamma(y + post.shape) - ln_gamma(post.shape) - ln_gamma(y + 1.0)
            + post.shape * (post.rate / denom).ln()
            + y * (exposure / denom).ln())
    }

    /// `P(Y = y)` for a new count over `exposure`.
    ///
    /// # Errors
    ///
    /// A non-positive or non-finite `exposure`.
    pub fn predictive_pmf(&self, y: u64, exposure: f64) -> Result<f64, ProbError> {
        self.predictive_log_pmf(y, exposure).map(f64::exp)
    }

    /// Posterior predictive mean count over `exposure`.
    #[must_use]
    pub fn predictive_mean(&self, exposure: f64) -> f64 {
        exposure * self.mean()
    }

    /// Posterior predictive variance of the count over `exposure`.
    #[must_use]
    pub fn predictive_variance(&self, exposure: f64) -> f64 {
        let post = self.posterior();
        exposure * post.shape / post.rate * (1.0 + exposure / post.rate)
    }

    /// `P(Y ≤ y)` for a new count over `exposure`.
    ///
    /// # Errors
    ///
    /// A non-positive or non-finite `exposure`.
    pub fn predictive_cdf(&self, y: u64, exposure: f64) -> Result<f64, ProbError> {
        let mut total = 0.0;
        for i in 0..=y {
            total += self.predictive_pmf(i, exposure)?;
        }
        Ok(total.min(1.0))
    }

    /// Smallest `y` with `P(Y ≤ y) ≥ p` for a new count over `exposure`.
    ///
    /// # Errors
    ///
    /// `p` outside `(0, 1)` or a non-positive `exposure`.
    pub fn predictive_quantile(&self, p: f64, exposure: f64) -> Result<u64, ProbError> {
        check_probability(p)?;
        let mut total = 0.0;
        let mut y = 0u64;
        loop {
            total += self.predictive_pmf(y, exposure)?;
            if total >= p || y == u64::from(u32::MAX) {
                return Ok(y);
            }
            y += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::PriorSpec;

    fn beta_set(alpha: f64, beta: f64) -> PriorSet {
        let mut set = PriorSet::new();
        set.push(PriorSpec::Beta(BetaHyperparameters { alpha, beta }));
        set
    }

    #[test]
    fn beta_posterior_is_the_closed_form() {
        let post = BetaBinomialPosterior::new(BetaHyperparameters { alpha: 2.0, beta: 3.0 }, 7, 20)
            .unwrap();
        let p = post.posterior();
        assert_eq!((p.alpha, p.beta), (9.0, 16.0));
        assert!((post.mean() - 9.0 / 25.0).abs() < 1e-15);
        assert!((post.cdf(post.quantile(0.3).unwrap()) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn exceedance_counts_strictly_above_the_threshold() {
        let set = beta_set(1.0, 1.0);
        let post =
            BetaBinomialPosterior::from_exceedance(&set, &[1.0, 2.0, 2.0, 3.0, 5.0], 2.0).unwrap();
        assert_eq!((post.successes(), post.trials()), (2, 5));
    }

    #[test]
    fn marginal_likelihood_sums_to_one_over_counts() {
        let prior = BetaHyperparameters { alpha: 1.5, beta: 0.7 };
        let total: f64 = (0..=6)
            .map(|k| {
                BetaBinomialPosterior::new(prior, k, 6).unwrap().log_marginal_likelihood().exp()
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gamma_posterior_is_the_closed_form() {
        let post = GammaPoissonPosterior::from_counts(
            GammaHyperparameters { shape: 2.0, rate: 1.0 },
            &[3, 0, 4, 2],
            None,
        )
        .unwrap();
        let p = post.posterior();
        assert_eq!((p.shape, p.rate), (11.0, 5.0));
        assert!((post.cdf(post.quantile(0.9).unwrap()) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn gamma_marginal_likelihood_matches_the_prior_predictive_chain() {
        let prior = GammaHyperparameters { shape: 2.0, rate: 1.5 };
        let all = GammaPoissonPosterior::from_counts(prior, &[3, 1], None).unwrap();
        let first = GammaPoissonPosterior::from_counts(prior, &[3], None).unwrap();
        let chain = first.log_marginal_likelihood() + first.predictive_log_pmf(1, 1.0).unwrap();
        assert!((all.log_marginal_likelihood() - chain).abs() < 1e-12);
    }

    #[test]
    fn gaussian_fit_refuses_a_count_prior() {
        assert!(beta_set(1.0, 1.0).reject_count_priors().is_err());
    }
}
