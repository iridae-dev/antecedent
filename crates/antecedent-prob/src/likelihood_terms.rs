//! Per-observation GLM likelihood value / score / observed curvature.
//!
//! One primitive per family so Laplace and HMC share exact derivatives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::norm_cdf;

use crate::backend::{BayesDesignRef, BayesLikelihood};
use crate::error::ProbError;
use crate::prior::GaussianCoefficientPrior;

/// Per-observation log-likelihood contribution and derivatives w.r.t. linear predictor `η`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LikelihoodTerms {
    /// Weighted log-likelihood contribution (constants in β may be omitted).
    pub log_value: f64,
    /// Weighted score `∂ℓ/∂η`.
    pub score_eta: f64,
    /// Weighted observed negative curvature `−∂²ℓ/∂η²` (≥ 0 for proper GLMs).
    pub neg_hessian_eta: f64,
}

/// Poisson log-link terms. Uncapped `μ = exp(η)`; overflows return [`ProbError::Numerical`].
///
/// # Errors
///
/// Non-finite Poisson rate.
pub fn poisson_terms(y: f64, eta: f64, weight: f64) -> Result<LikelihoodTerms, ProbError> {
    let mu = eta.exp();
    if !mu.is_finite() {
        return Err(ProbError::Numerical { message: "Poisson rate overflow".into() });
    }
    Ok(LikelihoodTerms {
        log_value: weight * (y * eta - mu),
        score_eta: weight * (y - mu),
        neg_hessian_eta: weight * mu,
    })
}

/// Bernoulli logit terms.
#[must_use]
pub fn logit_terms(y: f64, eta: f64, weight: f64) -> LikelihoodTerms {
    let tail = (-eta.abs()).exp();
    let denominator = 1.0 + tail;
    let (mu, complement) = if eta >= 0.0 {
        (1.0 / denominator, tail / denominator)
    } else {
        (tail / denominator, 1.0 / denominator)
    };
    LikelihoodTerms {
        log_value: -weight * (y * softplus(-eta) + (1.0 - y) * softplus(eta)),
        score_eta: weight * (y * complement - (1.0 - y) * mu),
        neg_hessian_eta: weight * (tail / denominator / denominator),
    }
}

/// Gaussian identity-link terms with residual precision `inv_sigma2`.
#[must_use]
pub fn gaussian_terms(y: f64, eta: f64, weight: f64, inv_sigma2: f64) -> LikelihoodTerms {
    let resid = y - eta;
    LikelihoodTerms {
        log_value: weight * (-0.5 * inv_sigma2 * resid * resid),
        score_eta: weight * resid * inv_sigma2,
        neg_hessian_eta: weight * inv_sigma2,
    }
}

/// Log standard-normal density.
#[must_use]
pub fn log_phi(x: f64) -> f64 {
    const LOG_INV_SQRT_2PI: f64 = -0.918_938_533_204_672_8; // -0.5 ln(2π)
    LOG_INV_SQRT_2PI - 0.5 * x * x
}

// For t >= 8, the inverse Mills ratio is t + delta, where
// delta = 1 / (t + 2 / (t + 3 / (...))). Keeping delta separately avoids
// subtracting nearly equal values when computing the observed curvature.
fn left_tail_mills(t: f64) -> (f64, f64) {
    let mut delta = 0.0;
    for k in (1..=64).rev() {
        delta = f64::from(k) / (t + delta);
    }
    (t + delta, delta)
}

/// Numerically stable `log Φ(x)`, using a continued fraction in the left tail.
#[must_use]
pub fn log_normal_cdf(x: f64) -> f64 {
    if x <= -8.0 {
        return log_phi(x) - left_tail_mills(-x).0.ln();
    }
    if x > 0.0 {
        return (-norm_cdf(-x)).ln_1p();
    }
    norm_cdf(x).ln()
}

/// Bernoulli probit terms using observed Hessian (Mills-ratio form).
///
/// # Errors
///
/// Non-finite Mills ratio / curvature in pathological inputs.
pub fn probit_terms(y: f64, eta: f64, weight: f64) -> Result<LikelihoodTerms, ProbError> {
    let sign = if y > 0.5 { 1.0 } else { -1.0 };
    let x = sign * eta;
    let log_cdf = log_normal_cdf(x);
    let (ratio, curvature) = if x <= -8.0 {
        let (ratio, delta) = left_tail_mills(-x);
        (ratio, ratio * delta)
    } else {
        let ratio = (log_phi(x) - log_cdf).exp();
        (ratio, ratio * (ratio + x))
    };
    let terms = LikelihoodTerms {
        log_value: weight * log_cdf,
        score_eta: weight * sign * ratio,
        neg_hessian_eta: weight * curvature,
    };
    if !terms.log_value.is_finite()
        || !terms.score_eta.is_finite()
        || !terms.neg_hessian_eta.is_finite()
    {
        return Err(ProbError::Numerical { message: "non-finite probit likelihood terms".into() });
    }
    Ok(terms)
}

fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

pub(crate) fn validate_design(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
) -> Result<(), ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    if design.y.len() != nrows {
        return Err(ProbError::Shape { message: "y length != nrows" });
    }
    if design.x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(ProbError::Shape { message: "X buffer too short" });
    }
    if nrows == 0 || ncols == 0 {
        return Err(ProbError::Shape { message: "empty design" });
    }
    let x_len = nrows.saturating_mul(ncols);
    for &v in &design.x_colmajor[..x_len] {
        if !v.is_finite() {
            return Err(ProbError::Shape { message: "X must be finite" });
        }
    }
    for &yi in design.y {
        match likelihood {
            BayesLikelihood::GaussianIdentity => {
                if !yi.is_finite() {
                    return Err(ProbError::Shape { message: "y must be finite" });
                }
            }
            BayesLikelihood::BernoulliLogit | BayesLikelihood::BernoulliProbit => {
                #[allow(
                    clippy::float_cmp,
                    reason = "a Bernoulli outcome must be exactly the coded value 0 or 1"
                )]
                let is_binary = yi == 0.0 || yi == 1.0;
                if !is_binary {
                    return Err(ProbError::Shape { message: "Bernoulli outcomes must be 0 or 1" });
                }
            }
            BayesLikelihood::PoissonLog => {
                if !(yi.is_finite() && yi >= 0.0) {
                    return Err(ProbError::Shape {
                        message: "Poisson outcomes must be finite and non-negative",
                    });
                }
            }
        }
    }
    if let Some(w) = design.weights {
        if w.len() != nrows {
            return Err(ProbError::Shape { message: "weights length != nrows" });
        }
        let mut mass = 0.0;
        for &wr in w {
            if !(wr.is_finite() && wr >= 0.0) {
                return Err(ProbError::Shape {
                    message: "weights must be finite and non-negative",
                });
            }
            mass += wr;
        }
        if !(mass > 0.0) || !mass.is_finite() {
            return Err(ProbError::Shape { message: "weights must have positive total mass" });
        }
    }
    if let Some(o) = design.offsets {
        if o.len() != nrows {
            return Err(ProbError::Shape { message: "offsets length != nrows" });
        }
        for &oi in o {
            if !oi.is_finite() {
                return Err(ProbError::Shape { message: "offsets must be finite" });
            }
        }
    }
    Ok(())
}

/// Share of the (weighted) rows that must be fitted with certainty on their observed side
/// before a Bernoulli mode is flagged as (quasi-)separated.
///
/// Separation is a property of the data as a whole: the likelihood keeps improving as the
/// coefficients grow because a large share of rows is already classified with certainty.
/// One extreme row in an otherwise informative sample is a high-leverage point, not
/// separation, and must not refuse the posterior.
const SEPARATION_ROW_SHARE: f64 = 0.10;

/// Fitted probability of the observed side beyond which a Bernoulli row is counted as fitted
/// with certainty.
const SEPARATION_CERTAINTY: f64 = 1e-8;

/// Accumulate likelihood gradient and −Hessian at `beta`. Returns (grad_inf, separation);
/// the separation flag (at least [`SEPARATION_ROW_SHARE`] of the weighted rows fitted with
/// certainty on their observed side) is evaluated only when `want_hessian` is set.
///
/// `gaussian_sigma2` scales the GaussianIdentity working weights / scores (`1/σ²`). Other
/// likelihoods ignore it.
///
/// `want_hessian: false` skips the O(n·p²) curvature accumulation entirely
/// (leaving `neg_hess` zeroed); the gradient is bit-identical either way. HMC
/// leapfrog steps read only the gradient, so they use the cheap form.
#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_likelihood(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    beta: &[f64],
    grad: &mut [f64],
    neg_hess: &mut [f64],
    eta: &mut [f64],
    work_w: &mut [f64],
    gaussian_sigma2: f64,
    want_hessian: bool,
) -> Result<(f64, bool), ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    grad.fill(0.0);
    neg_hess.fill(0.0);
    let inv_sigma2 = gaussian_precision(likelihood, gaussian_sigma2)?;

    let mut certain_weight = 0.0;
    let mut total_weight = 0.0;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut e = offset;
        for c in 0..ncols {
            e += design.x_colmajor[c * nrows + r] * beta[c];
        }
        eta[r] = e;
        let w_obs = design.weights.map_or(1.0, |w| w[r]);
        if w_obs == 0.0 {
            work_w[r] = 0.0;
            continue;
        }
        let y = design.y[r];

        let terms = glm_observation_terms(likelihood, y, e, w_obs, inv_sigma2)?;
        // The separation flag is read only where curvature is (the mode fit);
        // gradient-only leapfrog evaluations discard it, so skip its exp/erfc.
        if want_hessian
            && matches!(
                likelihood,
                BayesLikelihood::BernoulliLogit | BayesLikelihood::BernoulliProbit
            )
        {
            let mu = if matches!(likelihood, BayesLikelihood::BernoulliLogit) {
                1.0 / (1.0 + (-e).exp())
            } else {
                antecedent_kernels::norm_cdf(e)
            };
            // Certain on the *observed* side: a misclassified extreme row is an outlier, not
            // evidence that the labels are separable.
            let observed_one = y > 0.5;
            let miss = if observed_one { 1.0 - mu } else { mu };
            total_weight += w_obs;
            if miss < SEPARATION_CERTAINTY {
                certain_weight += w_obs;
            }
        }
        work_w[r] = terms.neg_hessian_eta;
        let score_scale = terms.score_eta;

        for c in 0..ncols {
            let x = design.x_colmajor[c * nrows + r];
            grad[c] += x * score_scale;
        }
        // −Hessian = X' diag(−ℓ'') X
        if want_hessian {
            for c1 in 0..ncols {
                let x1 = design.x_colmajor[c1 * nrows + r];
                for c2 in c1..ncols {
                    let x2 = design.x_colmajor[c2 * nrows + r];
                    let add = work_w[r] * x1 * x2;
                    neg_hess[c1 * ncols + c2] += add;
                    if c1 != c2 {
                        neg_hess[c2 * ncols + c1] += add;
                    }
                }
            }
        }
    }

    let mut ginf: f64 = 0.0;
    for g in grad.iter() {
        ginf = ginf.max(g.abs());
    }
    let separation = total_weight > 0.0 && certain_weight >= SEPARATION_ROW_SHARE * total_weight;
    Ok((ginf, separation))
}

fn glm_observation_terms(
    likelihood: BayesLikelihood,
    y: f64,
    eta: f64,
    weight: f64,
    inv_sigma2: f64,
) -> Result<LikelihoodTerms, ProbError> {
    match likelihood {
        BayesLikelihood::GaussianIdentity => Ok(gaussian_terms(y, eta, weight, inv_sigma2)),
        BayesLikelihood::BernoulliLogit => Ok(logit_terms(y, eta, weight)),
        BayesLikelihood::BernoulliProbit => probit_terms(y, eta, weight),
        BayesLikelihood::PoissonLog => poisson_terms(y, eta, weight),
    }
}

pub(crate) fn log_posterior_value(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    beta: &[f64],
    prior: &GaussianCoefficientPrior,
    prec: &[f64],
    eta: &mut [f64],
    gaussian_sigma2: f64,
) -> Result<f64, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    let inv_sigma2 = gaussian_precision(likelihood, gaussian_sigma2)?;
    let mut ll = 0.0;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut e = offset;
        for c in 0..ncols {
            e += design.x_colmajor[c * nrows + r] * beta[c];
        }
        eta[r] = e;
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        if w == 0.0 {
            continue;
        }
        let y = design.y[r];
        ll += glm_observation_terms(likelihood, y, e, w, inv_sigma2)?.log_value;
    }
    let mut lp = 0.0;
    for i in 0..ncols {
        let d = beta[i] - prior.mean[i];
        lp -= 0.5 * prec[i] * d * d;
    }
    Ok(ll + lp)
}

fn gaussian_precision(likelihood: BayesLikelihood, variance: f64) -> Result<f64, ProbError> {
    if likelihood != BayesLikelihood::GaussianIdentity {
        return Ok(1.0);
    }
    let precision = 1.0 / variance;
    if !variance.is_finite() || variance <= 0.0 || !precision.is_finite() {
        return Err(ProbError::Numerical {
            message: "Gaussian variance must be positive and have finite precision".into(),
        });
    }
    Ok(precision)
}

#[cfg(test)]
mod tests {
    #[test]
    fn review_zero_weight_rows_equal_deleted_rows() {
        for family in [
            BayesLikelihood::GaussianIdentity,
            BayesLikelihood::PoissonLog,
            BayesLikelihood::BernoulliProbit,
            BayesLikelihood::BernoulliLogit,
        ] {
            let y =
                if family == BayesLikelihood::GaussianIdentity { [1.0, 1e200] } else { [1.0, 0.0] };
            let full = BayesDesignRef {
                x_colmajor: &[1.0, 1.0],
                nrows: 2,
                ncols: 1,
                y: &y,
                weights: Some(&[1.0, 0.0]),
                offsets: Some(&[0.0, 1000.0]),
            };
            let dropped = BayesDesignRef {
                x_colmajor: &[1.0],
                nrows: 1,
                ncols: 1,
                y: &[1.0],
                weights: None,
                offsets: None,
            };
            let mut results = Vec::new();
            for design in [full, dropped] {
                validate_design(family, design).unwrap();
                let (mut grad, mut hess, mut eta, mut work) = ([0.0], [0.0], [0.0; 2], [0.0; 2]);
                let diagnostic = accumulate_likelihood(
                    family,
                    design,
                    &[0.0],
                    &mut grad,
                    &mut hess,
                    &mut eta,
                    &mut work,
                    1.0,
                    true,
                )
                .unwrap();
                let prior = GaussianCoefficientPrior::isotropic(1, 1.0);
                let value =
                    log_posterior_value(family, design, &[0.0], &prior, &[1.0], &mut eta, 1.0)
                        .unwrap();
                results.push((grad, hess, diagnostic, value));
            }
            assert_eq!(results[0], results[1], "{family:?}");
        }
    }

    #[test]
    fn review_probit_deep_tail_score_and_curvature() {
        for (y, eta, sign) in [(1.0, -1e8, 1.0), (0.0, 1e8, -1.0)] {
            let terms = probit_terms(y, eta, 0.25).unwrap();
            assert!((terms.score_eta / (sign * 0.25e8) - 1.0).abs() < 1e-14);
            assert!((terms.neg_hessian_eta - 0.25).abs() < 1e-14);
        }
        // Independent erfc-based reference at a tail where erfc remains representable.
        let terms = probit_terms(1.0, -10.0, 1.0).unwrap();
        assert!((terms.log_value + 53.231_285_150_512_46).abs() < 1e-12);
        assert!((terms.score_eta - 10.098_093_233_962_42).abs() < 1e-12);
        assert!((terms.neg_hessian_eta - 0.990_554_622_173_402_5).abs() < 1e-11);
    }

    #[test]
    fn review_probit_retains_upper_tail_log_probability() {
        let lp = log_normal_cdf(8.0);
        assert!((lp / -6.220_960_574_271_784e-16 - 1.0).abs() < 1e-12);
        for x in [-10.0, -8.0001, -7.9999, 8.0] {
            let h = 1e-5;
            let terms = probit_terms(1.0, x, 1.0).unwrap();
            let score = (log_normal_cdf(x + h) - log_normal_cdf(x - h)) / (2.0 * h);
            let curvature = -(probit_terms(1.0, x + h, 1.0).unwrap().score_eta
                - probit_terms(1.0, x - h, 1.0).unwrap().score_eta)
                / (2.0 * h);
            assert!((score / terms.score_eta - 1.0).abs() < 1e-7);
            assert!((curvature / terms.neg_hessian_eta - 1.0).abs() < 1e-7);
        }
    }

    #[test]
    fn review_gaussian_likelihood_uses_the_requested_precision() {
        let design = BayesDesignRef {
            x_colmajor: &[1.0],
            nrows: 1,
            ncols: 1,
            y: &[1e-10],
            weights: None,
            offsets: None,
        };
        let (mut grad, mut hessian, mut eta, mut work) = ([0.0], [0.0], [0.0], [0.0]);
        accumulate_likelihood(
            BayesLikelihood::GaussianIdentity,
            design,
            &[0.0],
            &mut grad,
            &mut hessian,
            &mut eta,
            &mut work,
            1e-20,
            true,
        )
        .unwrap();
        assert!((grad[0] / 1e10 - 1.0).abs() < 1e-12);
        assert!((hessian[0] / 1e20 - 1.0).abs() < 1e-12);
        let prior = GaussianCoefficientPrior::isotropic(1, 1.0);
        let value = log_posterior_value(
            BayesLikelihood::GaussianIdentity,
            design,
            &[0.0],
            &prior,
            &[1.0],
            &mut eta,
            1e-20,
        )
        .unwrap();
        assert!((value + 0.5).abs() < 1e-12);
    }

    #[test]
    fn review_logit_terms_retain_representable_tails() {
        let tail = (-40.0_f64).exp();
        for (outcome, eta, score_sign) in [(1.0, 40.0, 1.0), (0.0, -40.0, -1.0)] {
            let terms = logit_terms(outcome, eta, 1.0);
            assert!((terms.log_value / -tail - 1.0).abs() < 1e-12);
            assert!((terms.score_eta / (score_sign * tail) - 1.0).abs() < 1e-12);
            assert!((terms.neg_hessian_eta / tail - 1.0).abs() < 1e-12);
        }
    }

    use super::*;

    fn fd_score(f: impl Fn(f64) -> f64, eta: f64) -> f64 {
        let eps = 1e-5;
        (f(eta + eps) - f(eta - eps)) / (2.0 * eps)
    }

    fn fd_curvature(f: impl Fn(f64) -> f64, eta: f64) -> f64 {
        let eps = 1e-5;
        -(f(eta + eps) - 2.0 * f(eta) + f(eta - eps)) / (eps * eps)
    }

    #[test]
    fn poisson_fd_score_and_curvature_grid() {
        let etas = [-3.0, -1.0, 0.0, 1.0, (1e6_f64).ln(), 10.0, 20.0];
        for &eta in &etas {
            for &y in &[0.0, 1.0, 5.0] {
                let t = poisson_terms(y, eta, 1.0).expect("finite");
                let f = |e: f64| poisson_terms(y, e, 1.0).unwrap().log_value;
                let s_fd = fd_score(f, eta);
                let h_fd = fd_curvature(f, eta);
                assert!(
                    (t.score_eta - s_fd).abs() < 1e-5 * (1.0 + t.score_eta.abs()),
                    "score y={y} eta={eta}: got={} fd={s_fd}",
                    t.score_eta
                );
                assert!(
                    (t.neg_hessian_eta - h_fd).abs() < 1e-4 * (1.0 + t.neg_hessian_eta.abs()),
                    "hess y={y} eta={eta}: got={} fd={h_fd}",
                    t.neg_hessian_eta
                );
            }
        }
    }

    #[test]
    fn poisson_near_overflow_consistent_and_overflow_errors() {
        let eta_ok = 800.0_f64; // beyond ~709 where exp overflows
        assert!(poisson_terms(1.0, eta_ok, 1.0).is_err());
        let eta_safe = 20.0;
        let t = poisson_terms(2.0, eta_safe, 1.5).unwrap();
        let mu = eta_safe.exp();
        assert!((t.log_value - 1.5 * (2.0 * eta_safe - mu)).abs() < 1e-9);
        assert!((t.score_eta - 1.5 * (2.0 - mu)).abs() < 1e-9);
        assert!((t.neg_hessian_eta - 1.5 * mu).abs() < 1e-9);
    }

    #[test]
    fn probit_fd_grid_both_outcomes() {
        let mut eta = -12.0;
        while eta <= 12.0 {
            for &y in &[0.0, 1.0] {
                let t = probit_terms(y, eta, 1.0).unwrap();
                assert!(t.neg_hessian_eta >= -1e-12, "curvature neg at eta={eta} y={y}");
                assert!(t.log_value.is_finite());
                // Closed-form observed curvature identity.
                let lp = log_phi(eta);
                if y > 0.5 {
                    let lambda1 = (lp - log_normal_cdf(eta)).exp();
                    let curv = (lambda1 * (lambda1 + eta)).max(0.0);
                    assert!((t.neg_hessian_eta - curv).abs() < 1e-12);
                    assert!((t.score_eta - lambda1).abs() < 1e-12);
                } else {
                    let lambda0 = (lp - log_normal_cdf(-eta)).exp();
                    let curv = (lambda0 * (lambda0 - eta)).max(0.0);
                    assert!((t.neg_hessian_eta - curv).abs() < 1e-12);
                    assert!((t.score_eta + lambda0).abs() < 1e-12);
                }
                // FD score check on the interior (Hastings Φ limits Hessian FD accuracy).
                if eta.abs() <= 2.0 {
                    let f = |e: f64| probit_terms(y, e, 1.0).unwrap().log_value;
                    let s_fd = fd_score(f, eta);
                    assert!(
                        (t.score_eta - s_fd).abs() < 2e-3 * (1.0 + t.score_eta.abs()),
                        "score y={y} eta={eta}: got={} fd={s_fd}",
                        t.score_eta
                    );
                }
            }
            eta += 0.5;
        }
    }

    #[test]
    fn probit_observed_differs_from_fisher_fixture() {
        // At η=1.5, y=1: observed −ℓ″ = λ(λ+η) ≠ φ²/(Φ(1−Φ)).
        let eta = 1.5_f64;
        let t = probit_terms(1.0, eta, 1.0).unwrap();
        let mu = norm_cdf(eta);
        let dens = (log_phi(eta)).exp();
        let fisher = (dens * dens) / (mu * (1.0 - mu));
        assert!(
            (t.neg_hessian_eta - fisher).abs() > 0.01,
            "fixture must differ: obs={} fisher={fisher}",
            t.neg_hessian_eta
        );
        let lambda1 = (log_phi(eta) - log_normal_cdf(eta)).exp();
        let obs_ref = lambda1 * (lambda1 + eta);
        assert!((t.neg_hessian_eta - obs_ref).abs() < 1e-10);
    }

    #[test]
    fn separation_flag_is_only_evaluated_with_curvature() {
        // One row at eta = 30: fitted P(y=1) = 1 - 9e-14, inside the 1e-8 boundary.
        let design = BayesDesignRef {
            x_colmajor: &[1.0],
            nrows: 1,
            ncols: 1,
            y: &[1.0],
            weights: None,
            offsets: None,
        };
        let flag = |want_hessian: bool| {
            let (mut grad, mut hess, mut eta, mut work) = ([0.0], [0.0], [0.0], [0.0]);
            accumulate_likelihood(
                BayesLikelihood::BernoulliLogit,
                design,
                &[30.0],
                &mut grad,
                &mut hess,
                &mut eta,
                &mut work,
                1.0,
                want_hessian,
            )
            .unwrap()
            .1
        };
        assert!(flag(true));
        assert!(!flag(false));
    }

    /// One row fitted with certainty among 100 is a leverage point, not separation; the same
    /// row among ten (10% of the weight) is, and a misclassified extreme row never counts.
    #[test]
    fn separation_needs_a_share_of_rows_fitted_with_certainty_on_their_observed_side() {
        let flag = |eta_values: &[f64], y: &[f64]| {
            let n = y.len();
            let design = BayesDesignRef {
                x_colmajor: &vec![1.0; n],
                nrows: n,
                ncols: 1,
                y,
                weights: None,
                offsets: None,
            };
            // The design column is the constant 1, so eta = beta; use offsets to place each
            // row's linear predictor.
            let design = BayesDesignRef { offsets: Some(eta_values), ..design };
            let (mut grad, mut hess) = ([0.0], [0.0]);
            let (mut eta, mut work) = (vec![0.0; n], vec![0.0; n]);
            accumulate_likelihood(
                BayesLikelihood::BernoulliLogit,
                design,
                &[0.0],
                &mut grad,
                &mut hess,
                &mut eta,
                &mut work,
                1.0,
                true,
            )
            .unwrap()
            .1
        };
        // 100 rows, y = 1, one at eta = 30 (certain), the rest at eta = 0.
        let mut etas = vec![0.0; 100];
        etas[0] = 30.0;
        assert!(!flag(&etas, &vec![1.0; 100]));
        // Ten rows, one certain: 10% of the weight.
        let mut etas = vec![0.0; 10];
        etas[0] = 30.0;
        assert!(flag(&etas, &[1.0; 10]));
        // The same extreme predictor on the wrong side (y = 0 at eta = +30) is a misfit.
        let mut y = vec![1.0; 10];
        y[0] = 0.0;
        assert!(!flag(&etas, &y));
    }
}
