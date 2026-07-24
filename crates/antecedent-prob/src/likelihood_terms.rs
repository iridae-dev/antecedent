//! Per-observation GLM likelihood value / score / observed curvature.
//!
//! One primitive per family so Laplace and HMC share exact derivatives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use antecedent_kernels::norm_cdf;

use crate::error::ProbError;

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
        return Err(ProbError::Numerical {
            message: "Poisson rate overflow".into(),
        });
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
        return Err(ProbError::Numerical {
            message: "non-finite probit likelihood terms".into(),
        });
    }
    Ok(terms)
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
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
                    let s_fd = fd_score(&f, eta);
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
