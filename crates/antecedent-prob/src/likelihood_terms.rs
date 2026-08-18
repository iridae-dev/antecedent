//! Per-observation GLM likelihood value / score / observed curvature.
//!
//! One primitive per family so Laplace and HMC share exact derivatives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

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
    let mu = 1.0 / (1.0 + (-eta).exp());
    let v = (mu * (1.0 - mu)).max(1e-12);
    LikelihoodTerms {
        log_value: weight * (y * eta - softplus(eta)),
        score_eta: weight * (y - mu),
        neg_hessian_eta: weight * v,
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

/// Numerically stable `log Φ(x)`, with asymptotic left-tail expansion.
#[must_use]
pub fn log_normal_cdf(x: f64) -> f64 {
    if x >= 8.0 {
        return 0.0;
    }
    if x > -30.0 {
        let p = norm_cdf(x);
        if p > 0.0 {
            return p.ln();
        }
    }
    // Deep left tail: log Φ(x) ≈ log φ(x) − log(−x) + log(1 − 1/x² + 3/x⁴ − …)
    let inv_x2 = 1.0 / (x * x);
    let expansion = 1.0 - inv_x2 * (1.0 - 3.0 * inv_x2 * (1.0 - 5.0 * inv_x2));
    log_phi(x) - (-x).ln() + expansion.max(f64::MIN_POSITIVE).ln()
}

/// Bernoulli probit terms using observed Hessian (Mills-ratio form).
///
/// # Errors
///
/// Non-finite Mills ratio / curvature in pathological inputs.
pub fn probit_terms(y: f64, eta: f64, weight: f64) -> Result<LikelihoodTerms, ProbError> {
    let lp = log_phi(eta);
    let terms = if y > 0.5 {
        let log_cdf = log_normal_cdf(eta);
        let lambda1 = (lp - log_cdf).exp();
        // λ₁(λ₁+η) can underflow negative from cancellation in deep left tails; clamp.
        let curv = (lambda1 * (lambda1 + eta)).max(0.0);
        LikelihoodTerms {
            log_value: weight * log_cdf,
            score_eta: weight * lambda1,
            neg_hessian_eta: weight * curv,
        }
    } else {
        let log_sf = log_normal_cdf(-eta);
        let lambda0 = (lp - log_sf).exp();
        let curv = (lambda0 * (lambda0 - eta)).max(0.0);
        LikelihoodTerms {
            log_value: weight * log_sf,
            score_eta: weight * (-lambda0),
            neg_hessian_eta: weight * curv,
        }
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
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
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
                if !(yi == 0.0 || yi == 1.0) {
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

/// Accumulate likelihood gradient and −Hessian at `beta`. Returns (grad_inf, separation).
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
    let inv_sigma2 = 1.0 / gaussian_sigma2.max(1e-12);

    let mut separation = false;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut e = offset;
        for c in 0..ncols {
            e += design.x_colmajor[c * nrows + r] * beta[c];
        }
        eta[r] = e;
        let w_obs = design.weights.map_or(1.0, |w| w[r]);
        let y = design.y[r];

        let terms = glm_observation_terms(likelihood, y, e, w_obs, inv_sigma2)?;
        if matches!(likelihood, BayesLikelihood::BernoulliLogit | BayesLikelihood::BernoulliProbit)
        {
            let mu = if matches!(likelihood, BayesLikelihood::BernoulliLogit) {
                1.0 / (1.0 + (-e).exp())
            } else {
                antecedent_kernels::norm_cdf(e)
            };
            if mu < 1e-8 || mu > 1.0 - 1e-8 {
                separation = true;
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
    let inv_sigma2 = 1.0 / gaussian_sigma2.max(1e-12);
    let mut ll = 0.0;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut e = offset;
        for c in 0..ncols {
            e += design.x_colmajor[c * nrows + r] * beta[c];
        }
        eta[r] = e;
        let w = design.weights.map_or(1.0, |ww| ww[r]);
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

#[cfg(test)]
mod tests {
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
}
