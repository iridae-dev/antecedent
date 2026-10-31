//! Conjugate Dirichlet posterior of repeated categorical choices, with the posterior
//! probability that an option is the most chosen and the posterior expected loss of selecting it.
//!
//! Each observation is one choice of exactly one of `K` options from a fixed choice set. The
//! likelihood of an ordered sequence of `n` choices with counts `n_k` is `prod_k theta_k^{n_k}`,
//! where `theta` is the vector of choice shares; with a `Dirichlet(alpha)` prior the posterior is
//! `Dirichlet(alpha + n)` whatever process selected the sample sizes, provided that process
//! depended only on the observed choices and independent randomization. This module does not
//! check that premise; the caller states it.
//!
//! The functionals are exact under the declared prior and likelihood and are evaluated by
//! quadrature, not simulation. With `G_k ~ Gamma(alpha_k, 1)` independent and `S = sum_k G_k`,
//! `theta = G / S` is `Dirichlet(alpha)` and `S` is independent of `theta` (Lukacs 1955), so
//!
//! ```text
//! P(option k has the largest share) = P(G_k = max_j G_j) = integral g_k(x) prod_{j != k} F_j(x) dx
//! E[max_j theta_j]                  = E[max_j G_j] / sum_j alpha_j,
//! E[max_j G_j]                      = integral (1 - prod_j F_j(t)) dt,
//! ```
//!
//! with `g`, `F` the Gamma density and distribution function. The log marginal likelihood of an
//! ordered sequence is `lnG(A) - lnG(A + n) + sum_k (lnG(alpha_k + n_k) - lnG(alpha_k))` with
//! `A = sum_k alpha_k`.
//!
//! These are statements about the choices of the sampled population in the declared choice set.
//! They are not statements about behaviour outside it, and stated choices are not an identified
//! causal or behavioural effect. No coverage or calibration claim is made about any interval or
//! decision built from them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{AdaptiveInferenceError as Error, graded_rule, panel_rule};
use crate::special::{gamma_q, ln_gamma};

/// Most options one choice set may hold.
pub const MAX_CHOICE_OPTIONS: usize = 64;
/// Most choices one update may add.
pub const MAX_CHOICE_COUNT: u64 = 1_000_000_000;

const MAX_PANELS: usize = 8192;
const TAIL_STANDARD_DEVIATIONS: f64 = 14.0;
const TAIL_MASS: f64 = 1e-18;
const CANCEL_STRIDE: usize = 64;

/// A `Dirichlet(alpha)` posterior of the choice shares of `K` options.
#[derive(Clone, Debug, PartialEq)]
pub struct DirichletPosterior {
    alpha: Vec<f64>,
}

impl DirichletPosterior {
    /// A posterior with the given shapes.
    ///
    /// # Errors
    /// Fewer than two or more than [`MAX_CHOICE_OPTIONS`] options, or a shape that is not finite
    /// and at least one (a shape below one makes the density unbounded for the quadrature).
    pub fn new(alpha: Vec<f64>) -> Result<Self, Error> {
        if alpha.len() < 2 || alpha.len() > MAX_CHOICE_OPTIONS {
            return Err(Error::Shape("a choice set needs between 2 and 64 options"));
        }
        if alpha.iter().any(|a| !(a.is_finite() && *a >= 1.0))
            || !alpha.iter().sum::<f64>().is_finite()
        {
            return Err(Error::InvalidParameter(
                "a Dirichlet shape must be finite and at least one",
            ));
        }
        Ok(Self { alpha })
    }

    /// The posterior of a Dirichlet prior after the given choice counts.
    ///
    /// # Errors
    /// As [`Self::new`], a count vector that does not match the prior, or a total beyond
    /// [`MAX_CHOICE_COUNT`].
    pub fn from_counts(prior: &[f64], counts: &[f64]) -> Result<Self, Error> {
        if prior.len() != counts.len() {
            return Err(Error::Shape("the counts must match the options of the prior"));
        }
        if counts.iter().any(|c| !(c.is_finite() && *c >= 0.0))
            || counts.iter().sum::<f64>() > MAX_CHOICE_COUNT as f64
        {
            return Err(Error::InvalidParameter("choice counts must be finite and nonnegative"));
        }
        Self::new(prior.iter().zip(counts).map(|(a, c)| a + c).collect())
    }

    /// The shapes.
    #[must_use]
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Number of options.
    #[must_use]
    pub fn options(&self) -> usize {
        self.alpha.len()
    }

    /// Total shape, the number of pseudo-choices the posterior is worth.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Posterior mean shares, which are also the posterior predictive probabilities of the next
    /// single choice.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        let total = self.total();
        self.alpha.iter().map(|a| a / total).collect()
    }

    /// Posterior variance of each share, `m (1 - m) / (A + 1)` with `m` the mean.
    #[must_use]
    pub fn variance(&self) -> Vec<f64> {
        let total = self.total();
        self.alpha.iter().map(|a| (a / total) * (1.0 - a / total) / (total + 1.0)).collect()
    }

    /// Log marginal likelihood of an ordered sequence of choices with the given counts under the
    /// Dirichlet prior `prior` (no multinomial coefficient: the order is the observed one).
    ///
    /// # Errors
    /// A prior with a non-positive or non-finite shape, or counts that do not match it.
    pub fn log_marginal_likelihood(prior: &[f64], counts: &[f64]) -> Result<f64, Error> {
        if prior.len() != counts.len() || prior.len() < 2 {
            return Err(Error::Shape("the counts must match the options of the prior"));
        }
        if prior.iter().any(|a| !(a.is_finite() && *a > 0.0))
            || counts.iter().any(|c| !(c.is_finite() && *c >= 0.0))
        {
            return Err(Error::InvalidParameter("shapes must be positive and counts nonnegative"));
        }
        let (a0, n): (f64, f64) = (prior.iter().sum(), counts.iter().sum());
        Ok(ln_gamma(a0) - ln_gamma(a0 + n)
            + prior.iter().zip(counts).map(|(a, c)| ln_gamma(a + c) - ln_gamma(*a)).sum::<f64>())
    }

    fn windows(&self) -> Vec<(f64, f64)> {
        self.alpha
            .iter()
            .map(|&a| {
                let sd = a.sqrt();
                let mut lo = (a - TAIL_STANDARD_DEVIATIONS * sd).max(0.0);
                let mut hi = a + TAIL_STANDARD_DEVIATIONS * sd;
                let mut step = TAIL_STANDARD_DEVIATIONS * sd;
                while lo > 0.0 && 1.0 - gamma_q(a, lo) > TAIL_MASS {
                    lo = (lo - step).max(0.0);
                    step *= 2.0;
                }
                let mut step = TAIL_STANDARD_DEVIATIONS * sd;
                while gamma_q(a, hi) > TAIL_MASS {
                    hi += step;
                    step *= 2.0;
                }
                (lo, hi)
            })
            .collect()
    }

    fn quadrature(&self) -> Result<(Vec<f64>, Vec<f64>), Error> {
        let windows = self.windows();
        let lo = windows.iter().map(|w| w.0).fold(f64::INFINITY, f64::min);
        let hi = windows.iter().map(|w| w.1).fold(f64::NEG_INFINITY, f64::max);
        let narrowest = self.alpha.iter().map(|a| a.sqrt()).fold(f64::INFINITY, f64::min);
        let raw = ((hi - lo) / (2.0 * narrowest)).ceil();
        if !raw.is_finite() || raw > MAX_PANELS as f64 {
            return Err(Error::Limit("the posterior is too concentrated for the quadrature"));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "bounded above")]
        let panels = raw.max(4.0) as usize;
        Ok(graded_rule(&panel_rule(), lo, hi, panels, lo <= 0.0, false))
    }

    fn ln_gamma_density(a: f64, x: f64) -> f64 {
        (a - 1.0) * x.ln() - x - ln_gamma(a)
    }

    /// Posterior probability that each option has the largest choice share.
    ///
    /// The probabilities sum to one up to the quadrature error (about `1e-10`); they are not
    /// renormalized.
    ///
    /// # Errors
    /// A posterior too concentrated for the quadrature, or cancellation.
    pub fn best_option_probabilities(
        &self,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f64>, Error> {
        let (nodes, weights) = self.quadrature()?;
        let count = self.alpha.len();
        let mut result = vec![0.0; count];
        let mut cdf = vec![0.0; count];
        let mut suffix = vec![1.0; count + 1];
        for (index, (&x, &weight)) in nodes.iter().zip(&weights).enumerate() {
            if index % CANCEL_STRIDE == 0 && cancelled() {
                return Err(Error::Cancelled);
            }
            for (slot, &a) in cdf.iter_mut().zip(&self.alpha) {
                *slot = 1.0 - gamma_q(a, x);
            }
            for position in (0..count).rev() {
                suffix[position] = suffix[position + 1] * cdf[position];
            }
            let mut prefix = 1.0;
            for (position, &a) in self.alpha.iter().enumerate() {
                result[position] +=
                    weight * Self::ln_gamma_density(a, x).exp() * prefix * suffix[position + 1];
                prefix *= cdf[position];
            }
        }
        Ok(result)
    }

    /// Posterior expected loss `E[max_j theta_j] - E[theta_k]` of selecting each option, in units
    /// of choice share.
    ///
    /// # Errors
    /// As [`Self::best_option_probabilities`].
    pub fn expected_losses(&self, cancelled: &dyn Fn() -> bool) -> Result<Vec<f64>, Error> {
        let (nodes, weights) = self.quadrature()?;
        let mut expected_max_gamma = 0.0;
        for (index, (&t, &weight)) in nodes.iter().zip(&weights).enumerate() {
            if index % CANCEL_STRIDE == 0 && cancelled() {
                return Err(Error::Cancelled);
            }
            let product: f64 = self.alpha.iter().map(|&a| 1.0 - gamma_q(a, t)).product();
            expected_max_gamma += weight * (1.0 - product);
        }
        // The window starts at the smallest lower edge; below it the integrand is one.
        let lower = self.windows().iter().map(|w| w.0).fold(f64::INFINITY, f64::min);
        expected_max_gamma += lower;
        let expected_max_share = expected_max_gamma / self.total();
        Ok(self.mean().iter().map(|m| expected_max_share - m).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never() -> bool {
        false
    }

    #[test]
    fn two_options_reduce_to_the_beta_posterior() {
        let d = DirichletPosterior::new(vec![7.0, 4.0]).unwrap();
        let p = d.best_option_probabilities(&never).unwrap();
        // Shares of two options are theta and 1 - theta: option 1 is best iff theta > 1/2.
        let exact =
            1.0 - super::super::beta_bernoulli::regularized_incomplete_beta(0.5, 7.0, 4.0).unwrap();
        assert!((p[0] - exact).abs() < 1e-9, "{} vs {exact}", p[0]);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn symmetric_posterior_is_uniform_and_has_the_known_expected_max() {
        // Dirichlet(1,1,1): P(best) = 1/3; E[max share] = (1 + 1/2 + 1/3) / 3 = 11/18.
        let d = DirichletPosterior::new(vec![1.0, 1.0, 1.0]).unwrap();
        for p in d.best_option_probabilities(&never).unwrap() {
            assert!((p - 1.0 / 3.0).abs() < 1e-9);
        }
        let losses = d.expected_losses(&never).unwrap();
        assert!((losses[0] - (11.0 / 18.0 - 1.0 / 3.0)).abs() < 1e-9, "{losses:?}");
    }

    #[test]
    fn marginal_likelihood_matches_a_hand_computation() {
        // Uniform Dirichlet(1,1) prior: ordered sequence with k of n in option 0 has
        // marginal 1 / ((n + 1) C(n, k)).
        let value = DirichletPosterior::log_marginal_likelihood(&[1.0, 1.0], &[3.0, 2.0]).unwrap();
        assert!((value.exp() - 1.0 / (6.0 * 10.0)).abs() < 1e-14);
    }

    #[test]
    fn refuses_bad_shapes() {
        assert!(DirichletPosterior::new(vec![1.0]).is_err());
        assert!(DirichletPosterior::new(vec![0.5, 2.0]).is_err());
        assert!(DirichletPosterior::from_counts(&[1.0, 1.0], &[1.0]).is_err());
    }
}
