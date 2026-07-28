//! Conjugate moment-matching converters: Gaussian summary → Beta / Gamma.
//!
//! [`crate::prior::PriorSpec`] speaks Gaussian coefficients only. A Gaussian
//! *summary* — a mean and a variance, however it was obtained (a composed
//! prior's moments, a posterior artifact's moments, a domain expert's
//! elicited belief) — cannot itself express a prior over a **bounded
//! proportion** or a **non-negative rate**: no Beta has variance greater
//! than `mean * (1 - mean)`, and no Gaussian respects either boundary. This
//! module converts such a summary into [`BetaHyperparameters`] or
//! [`GammaHyperparameters`] via two constructors per family, each with a
//! distinct, honest contract:
//!
//! * [`BetaHyperparameters::from_moments`] /
//!   [`GammaHyperparameters::from_moments`] match **both** `(mean,
//!   variance)` exactly. Prior strength ([`BetaHyperparameters::ess`] /
//!   [`GammaHyperparameters::ess`]) is whatever those moments imply — a
//!   derived consequence, read back after the fact, never requested.
//! * [`BetaHyperparameters::from_mean_and_ess`] /
//!   [`GammaHyperparameters::from_mean_and_ess`] match the mean and a
//!   caller-declared prior-strength `ess` exactly. There is no `variance`
//!   parameter: for either family, `mean` and `ess` alone determine every
//!   other moment, so a `variance` argument would have nothing to do.
//!
//! An earlier version of this module offered one three-argument
//! `from_moments(mean, variance, target_ess)` per family: moment-match to
//! `(mean, variance)`, then throw that match away and rescale to
//! `target_ess` instead. `variance` never affected the output under that
//! signature — the rescale replaced the moment-matched concentration
//! outright — which misnamed the function (it built from `(mean,
//! target_ess)`, not from moments) and made the Beta variant reject
//! satisfiable requests: `(mean=0.5, variance=0.3, target_ess=10.0)`
//! errored on the variance support check even though the value actually
//! *returned*, `Beta(6, 6)`, has no relationship to `0.3`. The two-
//! constructor split below replaces that signature.
//!
//! ## Scope: hyperparameters, not a new inference path
//!
//! No backend in this crate ([`crate::conjugate`], [`crate::laplace`],
//! [`crate::hmc`], [`crate::gaussian_target`]) accepts anything but a
//! [`crate::prior::GaussianCoefficientPrior`] design-matrix prior plus a
//! residual-variance model ([`crate::prior::GaussianVarianceModel`]). A Beta
//! prior on a bounded proportion does not fit that shape, so this module is
//! deliberately **not** wired into [`crate::prior::PriorSpec`]: adding a
//! variant there would create a type the library accepts syntactically but
//! no backend can consume. Callers get plain hyperparameter structs to hand
//! to their own conjugate update (Beta-Binomial, Gamma-Poisson) or to record
//! as a [`antecedent_core::PriorAssumption`].
//!
//! ## No silent clamping
//!
//! [`BetaHyperparameters::from_moments`] is only defined when `variance <
//! mean * (1 - mean)`; beyond that bound no Beta distribution has those
//! moments. Rather than clamp `mean` into some interior interval or cap
//! `variance` — which would silently hand back a prior whose moments differ
//! from what the caller asked for — out-of-support input is rejected via
//! [`ProbError`]. The comparison against the variance bound is exact (no
//! epsilon): a caller who needs to tolerate floating-point noise at the
//! boundary should pad their own `variance` before calling, so the padding
//! amount is visible at the call site instead of buried in this module.
//! [`BetaHyperparameters::from_mean_and_ess`] has no equivalent gate to
//! violate: every `(mean, ess >= 0)` request is satisfiable, because `alpha`
//! / `beta` are solved directly from `mean` and `ess` rather than inferred
//! from a separately supplied `variance`.
//!
//! ## ESS convention
//!
//! Both conversions report **prior-strength ESS** — the same notion
//! [`crate::external_prior`] reports for composed Gaussian priors (how much
//! evidence a prior's concentration is worth, in sample-size terms) — now
//! for two conjugate families:
//!
//! * **Beta**: `ess = α + β − 2`. This convention sets the flat reference
//!   prior `Beta(1, 1)` to `ess = 0` (zero pseudo-observations beyond
//!   uniform), which is exactly what makes `from_mean_and_ess(mean, 0.0)`
//!   degrade to a `Beta(1,1)`-equivalent-strength prior at the requested
//!   mean rather than to something vanishing or improper. Some references
//!   instead report `α + β` (total pseudo-count, under which `Beta(1,1)` is
//!   `ess = 2`); this module never uses that convention.
//! * **Gamma**: `ess = shape − 1`. Symmetric reasoning: the reference
//!   exponential prior `Gamma(shape=1, ·)` maps to `ess = 0`. Some
//!   references instead report `shape` itself (under which `Gamma(1, ·)` is
//!   `ess = 1`); this module never uses that convention.
//!
//! Neither convention is interchangeable with MCMC/autocorrelation ESS or
//! with Kish importance-weighting ESS — see the module docs on
//! [`crate::external_prior`] for that three-way distinction. This is a
//! fourth, family-specific accounting of the same prior-strength notion,
//! applied to conjugate families instead of a Gaussian coefficient's
//! precision.
//!
//! [`BetaHyperparameters::from_moments`] and
//! [`GammaHyperparameters::from_moments`] can report a **negative** `.ess()`.
//! Any `(mean, variance)` moment match weaker than the flat/reference prior
//! (Beta: total concentration `κ < 2`, i.e. `variance > mean * (1 - mean) /
//! 3`; Gamma: `shape < 1`, i.e. `variance > mean²`) has `α + β − 2 < 0` or
//! `shape − 1 < 0` respectively. This is deliberately **not** treated as an
//! error: `α > 0` and `β > 0` (or `shape > 0`, `rate > 0`) still hold, the
//! distribution is proper, and the negative value is a truthful report that
//! the supplied moments describe a prior weaker than the reference — useful
//! information in its own right (e.g. flagging a caller-supplied `variance`
//! as suspiciously large). [`BetaHyperparameters::from_mean_and_ess`] /
//! [`GammaHyperparameters::from_mean_and_ess`] reject a negative `ess`
//! *input* (a caller cannot request negative prior strength), which is a
//! distinct check from the *output* `.ess()` a moment match may report.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::ProbError;

/// `Beta(α, β)` conjugate hyperparameters.
///
/// See the module docs for the `ess = α + β − 2` convention and for why
/// [`Self::from_moments`] may report a negative [`Self::ess`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaHyperparameters {
    /// Shape `α > 0`.
    pub alpha: f64,
    /// Shape `β > 0`.
    pub beta: f64,
}

impl BetaHyperparameters {
    /// Moment-match a Beta prior to `(mean, variance)` exactly.
    ///
    /// Prior strength ([`Self::ess`]) is whatever the moments imply — a
    /// derived consequence, not a caller input. See
    /// [`Self::from_mean_and_ess`] to request a specific prior strength
    /// instead.
    ///
    /// # Errors
    ///
    /// - `mean` non-finite or not strictly inside `(0, 1)`.
    /// - `variance` non-finite, non-positive, or `>= mean * (1 - mean)` (no
    ///   Beta distribution has those moments — the comparison is exact, no
    ///   epsilon slack at the boundary).
    /// - The resulting `α` / `β` are non-finite (extreme inputs).
    pub fn from_moments(mean: f64, variance: f64) -> Result<Self, ProbError> {
        if !(mean.is_finite() && mean > 0.0 && mean < 1.0) {
            return Err(ProbError::InvalidPrior {
                message: "beta_from_moments: mean must be finite and strictly inside (0, 1)",
            });
        }
        if !variance.is_finite() || !(variance > 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "beta_from_moments: variance must be finite and > 0",
            });
        }
        // Total concentration implied by the moments: kappa > 0 iff
        // variance < mean * (1 - mean), the Beta support bound.
        let kappa = mean * (1.0 - mean) / variance - 1.0;
        if !(kappa > 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "beta_from_moments: variance must be < mean * (1 - mean) for a Beta \
                          distribution to have these moments",
            });
        }
        let alpha = mean * kappa;
        let beta = (1.0 - mean) * kappa;
        if !alpha.is_finite() || !beta.is_finite() || !(alpha > 0.0) || !(beta > 0.0) {
            return Err(ProbError::Numerical {
                message: "beta_from_moments: alpha/beta non-finite or non-positive".into(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Build a Beta prior from `mean` and a caller-declared prior-strength
    /// `ess` exactly. No `variance` parameter — `mean` and `ess` alone
    /// determine `α` and `β`.
    ///
    /// `ess = 0` returns a prior with the same strength as the flat
    /// reference `Beta(1, 1)` at the requested mean (never a vanishing or
    /// improper one).
    ///
    /// # Errors
    ///
    /// - `mean` non-finite or not strictly inside `(0, 1)`.
    /// - `ess` non-finite or negative.
    /// - The resulting `α` / `β` are non-finite or non-positive (`ess` large
    ///   enough to overflow).
    pub fn from_mean_and_ess(mean: f64, ess: f64) -> Result<Self, ProbError> {
        if !(mean.is_finite() && mean > 0.0 && mean < 1.0) {
            return Err(ProbError::InvalidPrior {
                message: "beta_from_mean_and_ess: mean must be finite and strictly inside (0, 1)",
            });
        }
        if !ess.is_finite() || ess < 0.0 {
            return Err(ProbError::InvalidPrior {
                message: "beta_from_mean_and_ess: ess must be finite and >= 0",
            });
        }
        let total = ess + 2.0;
        let alpha = mean * total;
        let beta = (1.0 - mean) * total;
        if !alpha.is_finite() || !beta.is_finite() || !(alpha > 0.0) || !(beta > 0.0) {
            return Err(ProbError::Numerical {
                message: "beta_from_mean_and_ess: alpha/beta non-finite or non-positive".into(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Mean `α / (α + β)`.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance `αβ / ((α + β)² (α + β + 1))`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let total = self.alpha + self.beta;
        (self.alpha * self.beta) / (total * total * (total + 1.0))
    }

    /// Prior-strength ESS under this module's convention: `α + β − 2` (see
    /// module docs, including when this may be negative for
    /// [`Self::from_moments`]).
    #[must_use]
    pub fn ess(&self) -> f64 {
        self.alpha + self.beta - 2.0
    }
}

/// `Gamma(shape, rate)` conjugate hyperparameters.
///
/// See the module docs for the `ess = shape − 1` convention and for why
/// [`Self::from_moments`] may report a negative [`Self::ess`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaHyperparameters {
    /// Shape `> 0`.
    pub shape: f64,
    /// Rate `> 0` (`mean = shape / rate`).
    pub rate: f64,
}

impl GammaHyperparameters {
    /// Moment-match a Gamma prior to `(mean, variance)` exactly.
    ///
    /// Prior strength ([`Self::ess`]) is whatever the moments imply — a
    /// derived consequence, not a caller input. See
    /// [`Self::from_mean_and_ess`] to request a specific prior strength
    /// instead.
    ///
    /// # Errors
    ///
    /// - `mean` non-finite or non-positive.
    /// - `variance` non-finite or non-positive.
    /// - The resulting `shape` / `rate` are non-finite or non-positive
    ///   (extreme inputs).
    pub fn from_moments(mean: f64, variance: f64) -> Result<Self, ProbError> {
        if !mean.is_finite() || !(mean > 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "gamma_from_moments: mean must be finite and > 0",
            });
        }
        if !variance.is_finite() || !(variance > 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "gamma_from_moments: variance must be finite and > 0",
            });
        }
        // No support bound exists for Gamma beyond mean > 0, variance > 0 —
        // unlike Beta, any positive variance is achievable at a given
        // positive mean via some shape.
        let shape = mean * mean / variance;
        let rate = mean / variance;
        if !shape.is_finite() || !rate.is_finite() || !(shape > 0.0) || !(rate > 0.0) {
            return Err(ProbError::Numerical {
                message: "gamma_from_moments: shape/rate non-finite or non-positive".into(),
            });
        }
        Ok(Self { shape, rate })
    }

    /// Build a Gamma prior from `mean` and a caller-declared prior-strength
    /// `ess` exactly. No `variance` parameter — `mean` and `ess` alone
    /// determine `shape` and `rate`.
    ///
    /// `ess = 0` returns `Gamma(shape=1, ·)` — the reference exponential
    /// prior — at the requested mean (never a vanishing or improper one).
    ///
    /// # Errors
    ///
    /// - `mean` non-finite or non-positive.
    /// - `ess` non-finite or negative.
    /// - The resulting `shape` / `rate` are non-finite or non-positive
    ///   (`ess` large enough to overflow).
    pub fn from_mean_and_ess(mean: f64, ess: f64) -> Result<Self, ProbError> {
        if !mean.is_finite() || !(mean > 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "gamma_from_mean_and_ess: mean must be finite and > 0",
            });
        }
        if !ess.is_finite() || ess < 0.0 {
            return Err(ProbError::InvalidPrior {
                message: "gamma_from_mean_and_ess: ess must be finite and >= 0",
            });
        }
        let shape = ess + 1.0;
        let rate = shape / mean;
        if !shape.is_finite() || !rate.is_finite() || !(rate > 0.0) {
            return Err(ProbError::Numerical {
                message: "gamma_from_mean_and_ess: shape/rate non-finite or non-positive".into(),
            });
        }
        Ok(Self { shape, rate })
    }

    /// Mean `shape / rate`.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Variance `shape / rate²`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Prior-strength ESS under this module's convention: `shape − 1` (see
    /// module docs, including when this may be negative for
    /// [`Self::from_moments`]).
    #[must_use]
    pub fn ess(&self) -> f64 {
        self.shape - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn beta_from_moments_round_trips_input_moments() {
        // mean=0.3, var=0.02: kappa = 0.21/0.02 - 1 = 9.5, so alpha=2.85,
        // beta=6.65, ess=kappa-2=7.5. from_moments matches both moments
        // exactly; there is no rescale to undo.
        let h = BetaHyperparameters::from_moments(0.3, 0.02).unwrap();
        assert!((h.alpha - 2.85).abs() < TOL);
        assert!((h.beta - 6.65).abs() < TOL);
        assert!((h.mean() - 0.3).abs() < TOL);
        assert!((h.variance() - 0.02).abs() < TOL);
        assert!((h.ess() - 7.5).abs() < TOL);
    }

    #[test]
    fn beta_from_moments_can_report_negative_ess() {
        // mean=0.5, var=0.24: support bound is 0.25, so this is a valid
        // (barely) moment match, but kappa = 0.25/0.24 - 1 ~= 0.041667 < 2,
        // so ess = kappa - 2 < 0. alpha/beta stay positive and proper; the
        // negative ess is a truthful report of a prior weaker than the
        // flat reference, not an error.
        let h = BetaHyperparameters::from_moments(0.5, 0.24).unwrap();
        assert!(h.alpha > 0.0 && h.beta > 0.0);
        assert!(h.ess() < 0.0);
        assert!((h.mean() - 0.5).abs() < TOL);
        assert!((h.variance() - 0.24).abs() < 1e-6);
    }

    #[test]
    fn beta_from_moments_rejects_variance_at_or_above_support_bound() {
        // mean*(1-mean) = 0.25; variance == bound (kappa == 0) and variance
        // > bound must both be rejected, with no epsilon slack.
        assert!(BetaHyperparameters::from_moments(0.5, 0.25).is_err());
        assert!(BetaHyperparameters::from_moments(0.5, 0.3).is_err());
    }

    #[test]
    fn beta_from_moments_rejects_mean_outside_open_interval() {
        assert!(BetaHyperparameters::from_moments(0.0, 0.01).is_err());
        assert!(BetaHyperparameters::from_moments(1.0, 0.01).is_err());
        assert!(BetaHyperparameters::from_moments(-0.1, 0.01).is_err());
        assert!(BetaHyperparameters::from_moments(1.1, 0.01).is_err());
    }

    #[test]
    fn beta_from_moments_rejects_nonfinite_inputs() {
        assert!(BetaHyperparameters::from_moments(f64::NAN, 0.02).is_err());
        assert!(BetaHyperparameters::from_moments(0.3, f64::NAN).is_err());
        assert!(BetaHyperparameters::from_moments(0.3, f64::INFINITY).is_err());
        assert!(BetaHyperparameters::from_moments(f64::INFINITY, 0.02).is_err());
    }

    #[test]
    fn beta_from_moments_rejects_nonpositive_variance() {
        assert!(BetaHyperparameters::from_moments(0.3, 0.0).is_err());
        assert!(BetaHyperparameters::from_moments(0.3, -0.01).is_err());
    }

    #[test]
    fn beta_from_mean_and_ess_zero_is_beta_1_1_strength_at_requested_mean() {
        // No variance argument to satisfy here; ess=0 must give ess()==0
        // (Beta(1,1) strength) at mean=0.3, not a vanishing or improper
        // prior. total = 0 + 2 = 2; alpha=0.6, beta=1.4.
        let h = BetaHyperparameters::from_mean_and_ess(0.3, 0.0).unwrap();
        assert!((h.alpha - 0.6).abs() < TOL);
        assert!((h.beta - 1.4).abs() < TOL);
        assert!((h.mean() - 0.3).abs() < TOL);
        assert!(h.ess().abs() < TOL);
        assert!(h.alpha > 0.0 && h.beta > 0.0);
    }

    #[test]
    fn beta_from_mean_and_ess_zero_at_mean_half_is_exactly_beta_1_1() {
        let h = BetaHyperparameters::from_mean_and_ess(0.5, 0.0).unwrap();
        assert!((h.alpha - 1.0).abs() < TOL);
        assert!((h.beta - 1.0).abs() < TOL);
    }

    #[test]
    fn beta_from_mean_and_ess_matches_any_nonnegative_request() {
        // Unlike from_moments, every (mean, ess >= 0) pair is satisfiable —
        // no support gate to violate, including the value that from_moments
        // rejects as out-of-support when read as a variance.
        let h = BetaHyperparameters::from_mean_and_ess(0.5, 10.0).unwrap();
        assert!((h.alpha - 6.0).abs() < TOL);
        assert!((h.beta - 6.0).abs() < TOL);
        assert!((h.mean() - 0.5).abs() < TOL);
        assert!((h.ess() - 10.0).abs() < TOL);
    }

    #[test]
    fn beta_from_mean_and_ess_rejects_mean_outside_open_interval() {
        assert!(BetaHyperparameters::from_mean_and_ess(0.0, 1.0).is_err());
        assert!(BetaHyperparameters::from_mean_and_ess(1.0, 1.0).is_err());
    }

    #[test]
    fn beta_from_mean_and_ess_rejects_negative_ess() {
        assert!(BetaHyperparameters::from_mean_and_ess(0.3, -1.0).is_err());
    }

    #[test]
    fn beta_from_mean_and_ess_rejects_nonfinite_inputs() {
        assert!(BetaHyperparameters::from_mean_and_ess(f64::NAN, 1.0).is_err());
        assert!(BetaHyperparameters::from_mean_and_ess(0.3, f64::NAN).is_err());
        assert!(BetaHyperparameters::from_mean_and_ess(0.3, f64::INFINITY).is_err());
    }

    #[test]
    fn gamma_from_moments_round_trips_input_moments() {
        // mean=4.0, var=2.0: shape = 16/2 = 8, rate = 4/2 = 2, ess = 7.
        let h = GammaHyperparameters::from_moments(4.0, 2.0).unwrap();
        assert!((h.shape - 8.0).abs() < TOL);
        assert!((h.rate - 2.0).abs() < TOL);
        assert!((h.mean() - 4.0).abs() < TOL);
        assert!((h.variance() - 2.0).abs() < TOL);
        assert!((h.ess() - 7.0).abs() < TOL);
    }

    #[test]
    fn gamma_from_moments_can_report_negative_ess() {
        // mean=4.0, var=32.0: shape = 16/32 = 0.5 < 1, so ess = shape - 1 <
        // 0. shape/rate stay positive and proper; the negative ess is a
        // truthful report of a prior weaker than the reference exponential.
        let h = GammaHyperparameters::from_moments(4.0, 32.0).unwrap();
        assert!(h.shape > 0.0 && h.rate > 0.0);
        assert!(h.ess() < 0.0);
        assert!((h.mean() - 4.0).abs() < TOL);
        assert!((h.variance() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn gamma_from_moments_rejects_nonpositive_mean() {
        assert!(GammaHyperparameters::from_moments(0.0, 1.0).is_err());
        assert!(GammaHyperparameters::from_moments(-1.0, 1.0).is_err());
    }

    #[test]
    fn gamma_from_moments_rejects_nonpositive_variance() {
        assert!(GammaHyperparameters::from_moments(4.0, 0.0).is_err());
        assert!(GammaHyperparameters::from_moments(4.0, -1.0).is_err());
    }

    #[test]
    fn gamma_from_moments_rejects_nonfinite_inputs() {
        assert!(GammaHyperparameters::from_moments(f64::NAN, 2.0).is_err());
        assert!(GammaHyperparameters::from_moments(4.0, f64::NAN).is_err());
        assert!(GammaHyperparameters::from_moments(4.0, f64::INFINITY).is_err());
    }

    #[test]
    fn gamma_from_mean_and_ess_zero_is_reference_exponential_at_requested_mean() {
        // No variance argument to satisfy here; ess=0 must give shape=1 (the
        // reference exponential), not a vanishing or improper prior.
        let h = GammaHyperparameters::from_mean_and_ess(4.0, 0.0).unwrap();
        assert!((h.shape - 1.0).abs() < TOL);
        assert!((h.rate - 0.25).abs() < TOL);
        assert!((h.mean() - 4.0).abs() < TOL);
        assert!(h.ess().abs() < TOL);
        assert!(h.shape > 0.0 && h.rate > 0.0);
    }

    #[test]
    fn gamma_from_mean_and_ess_matches_any_nonnegative_request() {
        let h = GammaHyperparameters::from_mean_and_ess(4.0, 7.0).unwrap();
        assert!((h.shape - 8.0).abs() < TOL);
        assert!((h.rate - 2.0).abs() < TOL);
        assert!((h.mean() - 4.0).abs() < TOL);
        assert!((h.ess() - 7.0).abs() < TOL);
    }

    #[test]
    fn gamma_from_mean_and_ess_rejects_nonpositive_mean() {
        assert!(GammaHyperparameters::from_mean_and_ess(0.0, 1.0).is_err());
        assert!(GammaHyperparameters::from_mean_and_ess(-1.0, 1.0).is_err());
    }

    #[test]
    fn gamma_from_mean_and_ess_rejects_negative_ess() {
        assert!(GammaHyperparameters::from_mean_and_ess(4.0, -1.0).is_err());
    }

    #[test]
    fn gamma_from_mean_and_ess_rejects_nonfinite_inputs() {
        assert!(GammaHyperparameters::from_mean_and_ess(f64::NAN, 2.0).is_err());
        assert!(GammaHyperparameters::from_mean_and_ess(4.0, f64::NAN).is_err());
        assert!(GammaHyperparameters::from_mean_and_ess(4.0, f64::INFINITY).is_err());
    }
}
