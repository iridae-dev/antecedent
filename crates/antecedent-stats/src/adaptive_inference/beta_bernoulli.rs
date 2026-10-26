//! Conjugate Beta-Bernoulli posterior for bandit arms, with the posterior
//! probability that an arm is best and the posterior expected loss of
//! choosing it.
//!
//! For arm `k` with a `Beta(alpha, beta)` prior and `s` successes and `f`
//! failures among the *observed* outcomes of that arm, the posterior is
//! `Beta(alpha + s, beta + f)` whatever assignment rule produced the data,
//! provided the rule depended only on the observed history and on
//! randomization independent of the unobserved outcomes and parameters (the
//! likelihood principle: the assignment probabilities factor out of the
//! likelihood). This module does not check that premise; the caller states it.
//!
//! The posterior probability that arm `k` is best is
//! `P_k = integral f_k(x) prod_{j != k} F_j(x) dx` and the posterior expected
//! loss of choosing arm `k` is `E[max_j theta_j] - E[theta_k]` with
//! `E[max_j theta_j] = integral (1 - prod_j F_j(t)) dt`. Both are evaluated by
//! composite Gauss-Legendre quadrature of the Beta density and of the
//! regularized incomplete beta function (continued fraction, Press et al.,
//! *Numerical Recipes*, section 6.4). Prior shapes of at least one keep the
//! density bounded, which is what the quadrature accuracy relies on.
//!
//! These are exact posterior functionals under the declared prior and model.
//! They are not frequentist statements: no coverage or error-rate claim is
//! made about intervals or decisions built from them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{AdaptiveInferenceError as Error, composite_rule, panel_rule};
use crate::special::ln_gamma;

/// Total prior-plus-observed trials up to which an arm's posterior is evaluated with the exact
/// Beta functions. Above it the log-gamma cancellation in the Beta density and incomplete beta
/// function exceeds the quadrature accuracy, and the posterior is evaluated with the normal law
/// of the same mean and variance, which differs from the Beta by `O(1 / sqrt(trials))` in
/// distribution at that size (skewness `~ 2 (b - a) / (a + b) / sqrt(a + b)`).
pub const EXACT_BETA_TRIALS: u64 = 16_000_000;
/// Most arms one posterior comparison accepts.
pub const MAX_BANDIT_ARMS: usize = 64;

const MAX_PANELS: usize = 8192;
const TAIL_STANDARD_DEVIATIONS: f64 = 14.0;
const MAX_FRACTION_ITERATIONS: usize = 200_000;
const FRACTION_EPSILON: f64 = 1e-15;
const TINY: f64 = 1e-300;
const CANCEL_STRIDE: usize = 64;

/// A `Beta(alpha, beta)` posterior of one Bernoulli arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaPosterior {
    alpha: f64,
    beta: f64,
}

impl BetaPosterior {
    /// A posterior with the given shapes.
    ///
    /// # Errors
    /// A shape that is not finite or is below one, or whose total is not finite. Totals above
    /// [`EXACT_BETA_TRIALS`] are accepted and evaluated through the normal approximation
    /// ([`Self::is_normal_approximated`]).
    pub fn new(alpha: f64, beta: f64) -> Result<Self, Error> {
        if !(alpha.is_finite() && beta.is_finite() && alpha >= 1.0 && beta >= 1.0) {
            return Err(Error::InvalidParameter("a Beta shape must be finite and at least one"));
        }
        if !(alpha + beta).is_finite() {
            return Err(Error::InvalidParameter("a Beta posterior's total shape must be finite"));
        }
        Ok(Self { alpha, beta })
    }

    /// Whether this posterior carries more than [`EXACT_BETA_TRIALS`] trials, so that its
    /// density and distribution function are those of the normal law with the same mean and
    /// variance instead of the exact Beta functions.
    #[must_use]
    pub fn is_normal_approximated(self) -> bool {
        self.alpha + self.beta > EXACT_BETA_TRIALS as f64
    }

    /// Distribution function at `x`: the exact incomplete beta function, or the matching
    /// normal law above [`EXACT_BETA_TRIALS`].
    fn cdf(self, x: f64) -> Result<f64, Error> {
        if self.is_normal_approximated() {
            Ok(antecedent_kernels::norm_cdf((x - self.mean()) / self.standard_deviation()))
        } else {
            regularized_incomplete_beta(x, self.alpha, self.beta)
        }
    }

    /// The posterior of a `Beta(prior_alpha, prior_beta)` prior after
    /// `successes` and `failures`.
    ///
    /// # Errors
    /// As [`Self::new`] for the resulting shapes.
    pub fn from_counts(
        prior_alpha: f64,
        prior_beta: f64,
        successes: u64,
        failures: u64,
    ) -> Result<Self, Error> {
        Self::new(prior_alpha + successes as f64, prior_beta + failures as f64)
    }

    /// First shape.
    #[must_use]
    pub const fn alpha(self) -> f64 {
        self.alpha
    }

    /// Second shape.
    #[must_use]
    pub const fn beta(self) -> f64 {
        self.beta
    }

    /// Posterior mean `alpha / (alpha + beta)`.
    #[must_use]
    pub fn mean(self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance `alpha beta / ((alpha + beta)^2 (alpha + beta + 1))`.
    #[must_use]
    pub fn variance(self) -> f64 {
        let total = self.alpha + self.beta;
        self.alpha * self.beta / (total * total * (total + 1.0))
    }

    fn standard_deviation(self) -> f64 {
        self.variance().sqrt()
    }

    fn ln_density(self, x: f64) -> f64 {
        if self.is_normal_approximated() {
            let sd = self.standard_deviation();
            let z = (x - self.mean()) / sd;
            return -0.5 * z * z - sd.ln() - 0.5 * std::f64::consts::TAU.ln();
        }
        let ln_beta = ln_gamma(self.alpha) + ln_gamma(self.beta) - ln_gamma(self.alpha + self.beta);
        (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (-x).ln_1p() - ln_beta
    }

    /// Mean plus or minus the tail window, clipped to the unit interval.
    fn window(self) -> (f64, f64) {
        let sd = self.standard_deviation();
        (
            (self.mean() - TAIL_STANDARD_DEVIATIONS * sd).max(0.0),
            (self.mean() + TAIL_STANDARD_DEVIATIONS * sd).min(1.0),
        )
    }
}

/// Continued fraction of the incomplete beta function (modified Lentz).
fn beta_fraction(x: f64, a: f64, b: f64) -> Result<f64, Error> {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_FRACTION_ITERATIONS {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let even = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + even * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + even / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let odd = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + odd * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + odd / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < FRACTION_EPSILON {
            return Ok(h);
        }
    }
    Err(Error::Numerical("the incomplete beta continued fraction did not converge"))
}

/// Regularized incomplete beta function `I_x(a, b)`, the CDF of
/// `Beta(a, b)` at `x`.
///
/// Accurate to roughly `1e-10` for shapes up to about `1e5` and to roughly
/// `1e-8` near sixteen million (the log-gamma cancellation limits it); [`BetaPosterior`] switches
/// to the normal law beyond that.
///
/// # Errors
/// `x` outside `[0, 1]`, a shape that is not finite and positive, or a
/// continued fraction that does not converge.
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> Result<f64, Error> {
    if !(a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0) {
        return Err(Error::InvalidParameter("incomplete beta shapes must be finite and positive"));
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(Error::InvalidParameter("incomplete beta argument must lie in [0, 1]"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x >= 1.0 {
        return Ok(1.0);
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok((front * beta_fraction(x, a, b)? / a).min(1.0))
    } else {
        Ok((1.0 - front * beta_fraction(1.0 - x, b, a)? / b).clamp(0.0, 1.0))
    }
}

fn check_arms(arms: &[BetaPosterior]) -> Result<(), Error> {
    if arms.len() < 2 || arms.len() > MAX_BANDIT_ARMS {
        return Err(Error::Shape("a posterior comparison needs between 2 and 64 arms"));
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a nonnegative ceiling bounded by MAX_PANELS before the cast"
)]
fn panel_count(lo: f64, hi: f64, scale: f64) -> Result<usize, Error> {
    let raw = ((hi - lo) / (2.0 * scale)).ceil();
    if !raw.is_finite() {
        return Err(Error::Numerical("the quadrature window is not finite"));
    }
    if raw > MAX_PANELS as f64 {
        return Err(Error::Limit("the posteriors are too concentrated for the quadrature"));
    }
    Ok(raw.max(4.0) as usize)
}

fn narrowest(arms: &[BetaPosterior]) -> f64 {
    arms.iter().map(|arm| arm.standard_deviation()).fold(f64::INFINITY, f64::min)
}

/// Posterior probability that each arm has the largest success probability.
///
/// Arms are independent a posteriori. The probabilities sum to one up to the
/// quadrature error (about `1e-10`); they are not renormalized.
///
/// # Errors
/// Fewer than two or more than [`MAX_BANDIT_ARMS`] arms, a posterior too
/// concentrated for the quadrature, a continued fraction that does not
/// converge, or cancellation.
pub fn best_arm_probabilities(
    arms: &[BetaPosterior],
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<f64>, Error> {
    check_arms(arms)?;
    let lo = arms.iter().map(|arm| arm.window().0).fold(f64::INFINITY, f64::min);
    let hi = arms.iter().map(|arm| arm.window().1).fold(f64::NEG_INFINITY, f64::max);
    let panels = panel_count(lo, hi, narrowest(arms))?;
    let (nodes, weights) = composite_rule(&panel_rule(), lo, hi, panels);
    let count = arms.len();
    let mut result = vec![0.0; count];
    let mut cdf = vec![0.0; count];
    let mut suffix = vec![1.0; count + 1];
    for (index, (&x, &weight)) in nodes.iter().zip(&weights).enumerate() {
        if index % CANCEL_STRIDE == 0 && cancelled() {
            return Err(Error::Cancelled);
        }
        for (slot, arm) in cdf.iter_mut().zip(arms) {
            *slot = arm.cdf(x)?;
        }
        for position in (0..count).rev() {
            suffix[position] = suffix[position + 1] * cdf[position];
        }
        let mut prefix = 1.0;
        for (position, arm) in arms.iter().enumerate() {
            result[position] += weight * arm.ln_density(x).exp() * prefix * suffix[position + 1];
            prefix *= cdf[position];
        }
    }
    Ok(result)
}

/// Posterior probability that arm `a` has a larger success probability than
/// arm `b`.
///
/// # Errors
/// As [`best_arm_probabilities`].
pub fn probability_superior(
    a: BetaPosterior,
    b: BetaPosterior,
    cancelled: &dyn Fn() -> bool,
) -> Result<f64, Error> {
    Ok(best_arm_probabilities(&[a, b], cancelled)?[0])
}

/// Posterior probability that arm `a` exceeds arm `b` by more than `margin`,
/// `P(theta_a - theta_b > margin)`, by quadrature of
/// `integral f_a(x) F_b(x - margin) dx` over the window of arm `a`. With a
/// zero margin this is [`probability_superior`].
///
/// # Errors
/// A margin outside `[0, 1)`, a posterior too concentrated for the
/// quadrature, a continued fraction that does not converge, or cancellation.
pub fn probability_exceeds(
    a: BetaPosterior,
    b: BetaPosterior,
    margin: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<f64, Error> {
    if !(margin.is_finite() && (0.0..1.0).contains(&margin)) {
        return Err(Error::InvalidParameter("a margin must lie in [0, 1)"));
    }
    // Below the margin the integrand is zero and its first kink is at the margin, so the
    // quadrature starts there and the integrand is smooth over every panel.
    let (window_lo, hi) = a.window();
    let lo = window_lo.max(margin);
    if lo >= hi {
        return Ok(0.0);
    }
    let panels = panel_count(lo, hi, a.standard_deviation().min(b.standard_deviation()))?;
    let (nodes, weights) = composite_rule(&panel_rule(), lo, hi, panels);
    let mut total = 0.0;
    for (index, (&x, &weight)) in nodes.iter().zip(&weights).enumerate() {
        if index % CANCEL_STRIDE == 0 && cancelled() {
            return Err(Error::Cancelled);
        }
        let cdf = b.cdf(x - margin)?;
        total += weight * a.ln_density(x).exp() * cdf;
    }
    Ok(total.clamp(0.0, 1.0))
}

/// Posterior expected loss `E[max_j theta_j] - E[theta_k]` of choosing each
/// arm `k`, in units of the success probability.
///
/// # Errors
/// As [`best_arm_probabilities`].
pub fn expected_losses(
    arms: &[BetaPosterior],
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<f64>, Error> {
    check_arms(arms)?;
    // Below the largest lower window edge some cdf is zero, so the integrand
    // is one; above the largest upper edge every cdf is one, so it is zero.
    let lower = arms.iter().map(|arm| arm.window().0).fold(f64::NEG_INFINITY, f64::max);
    let upper = arms.iter().map(|arm| arm.window().1).fold(f64::NEG_INFINITY, f64::max);
    let mut expected_max = lower;
    if upper > lower {
        let panels = panel_count(lower, upper, narrowest(arms))?;
        let (nodes, weights) = composite_rule(&panel_rule(), lower, upper, panels);
        for (index, (&t, &weight)) in nodes.iter().zip(&weights).enumerate() {
            if index % CANCEL_STRIDE == 0 && cancelled() {
                return Err(Error::Cancelled);
            }
            let mut product = 1.0;
            for arm in arms {
                product *= arm.cdf(t)?;
            }
            expected_max += weight * (1.0 - product);
        }
    }
    Ok(arms.iter().map(|arm| expected_max - arm.mean()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never() -> bool {
        false
    }

    #[test]
    fn incomplete_beta_matches_closed_forms() {
        let x: f64 = 0.37;
        let cases = [(1.0, 1.0, x), (2.0, 1.0, x * x), (1.0, 3.0, 1.0 - (1.0 - x).powi(3))];
        for (a, b, expected) in cases {
            let value = regularized_incomplete_beta(x, a, b).unwrap();
            assert!((value - expected).abs() < 1e-12, "I_x({a},{b}) = {value}, wanted {expected}");
        }
        // Symmetry: I_{1/2}(a, a) = 1/2.
        let half = regularized_incomplete_beta(0.5, 7.5, 7.5).unwrap();
        assert!((half - 0.5).abs() < 1e-12);
    }

    #[test]
    fn posteriors_beyond_the_exact_limit_use_the_normal_law() {
        let big = EXACT_BETA_TRIALS as f64;
        let a = BetaPosterior::new(0.5 * big + 1.0e6, 0.5 * big - 1.0e6 + 10.0).unwrap();
        assert!(a.is_normal_approximated());
        let b = BetaPosterior::new(0.5 * big - 1.0e6 + 10.0, 0.5 * big + 1.0e6).unwrap();
        let p = probability_superior(a, b, &never).unwrap();
        assert!(p > 0.9999, "{p}");
        let exact = BetaPosterior::new(100.0, 100.0).unwrap();
        assert!(!exact.is_normal_approximated());
    }

    #[test]
    fn uniform_pair_has_half_superiority_and_one_sixth_loss() {
        let u = BetaPosterior::new(1.0, 1.0).unwrap();
        let p = probability_superior(u, u, &never).unwrap();
        assert!((p - 0.5).abs() < 1e-10);
        let losses = expected_losses(&[u, u], &never).unwrap();
        assert!((losses[0] - 1.0 / 6.0).abs() < 1e-10);
    }
}
