//! Inference diagnostics for Laplace / conjugate / MCMC backends
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// Factorization used for the Laplace covariance / MCMC marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HessianFactorization {
    /// Cholesky of the negative Hessian.
    Cholesky,
    /// Structured LDLT fallback.
    Ldlt,
    /// Analytic conjugate (exact posterior; no Hessian).
    Analytic,
    /// Multi-chain MCMC (HMC / SMC); curvature from sampling, not Hessian.
    Mcmc,
}

/// Convergence / curvature / chain diagnostics required before reporting a posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceDiagnostics {
    /// Whether the optimizer / sampler reported convergence.
    pub converged: bool,
    /// Iterations used (Newton steps or post-warmup length).
    pub iterations: u32,
    /// Final gradient infinity-norm (MAP); unused for MCMC (`0.0`).
    pub grad_inf_norm: f64,
    /// Estimated condition number of −Hessian (or NaN if unavailable).
    pub hessian_condition: f64,
    /// Factorization path used.
    pub factorization: HessianFactorization,
    /// Separation / complete-separation warning for Bernoulli models.
    pub separation_warning: bool,
    /// Human-readable notes.
    pub notes: Vec<Arc<str>>,
    /// Backend identifier (e.g. "laplace", "conjugate_gaussian", "hmc").
    pub backend_id: Arc<str>,
    /// MCMC: number of chains (None for Laplace / conjugate).
    pub n_chains: Option<u32>,
    /// MCMC: warmup iterations per chain.
    pub n_warmup: Option<u32>,
    /// MCMC: minimum bulk ESS across parameters.
    pub ess_bulk_min: Option<f64>,
    /// MCMC: maximum split-Ř across parameters.
    pub rhat_max: Option<f64>,
    /// MCMC: leapfrog / trajectory divergence count.
    pub n_divergences: Option<u32>,
}

impl InferenceDiagnostics {
    /// Analytic conjugate path (always "converged").
    #[must_use]
    pub fn analytic(backend_id: impl Into<Arc<str>>) -> Self {
        Self {
            converged: true,
            iterations: 0,
            grad_inf_norm: 0.0,
            hessian_condition: 1.0,
            factorization: HessianFactorization::Analytic,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: backend_id.into(),
            n_chains: None,
            n_warmup: None,
            ess_bulk_min: None,
            rhat_max: None,
            n_divergences: None,
        }
    }

    /// Whether this diagnostic set is sufficient to publish a posterior.
    ///
    /// Narrow Laplace posteriors without convergence + curvature are refused.
    /// MCMC requires finite Ř < 1.05, ESS > 10, and no hard divergence flood
    /// (> 10% of total leapfrog proposals is refused via `converged` flag).
    #[must_use]
    pub fn allows_posterior(&self) -> bool {
        match self.factorization {
            HessianFactorization::Analytic => true,
            HessianFactorization::Mcmc => {
                let rhat_ok = self.rhat_max.is_some_and(|r| r.is_finite() && r < 1.05);
                let ess_ok = self.ess_bulk_min.is_some_and(|e| e.is_finite() && e > 10.0);
                let div_ok = self.n_divergences.is_some();
                self.converged && rhat_ok && ess_ok && div_ok
            }
            HessianFactorization::Cholesky | HessianFactorization::Ldlt => {
                self.converged
                    && self.grad_inf_norm.is_finite()
                    && self.hessian_condition.is_finite()
                    && self.hessian_condition > 0.0
            }
        }
    }
}

/// Optional prior-sensitivity summary attached to a causal posterior.
///
/// Mode-select: isotropic scale grid (`prior_scales` non-empty, `alphas` empty)
/// or external power-prior α-multiplier grid (`alphas` non-empty, `prior_scales`
/// empty). `effect_means` / `effect_sds` always align with the active grid.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PriorSensitivitySummary {
    /// Prior scale grid evaluated (isotropic mode).
    pub prior_scales: Arc<[f64]>,
    /// External α multipliers applied to post-conflict `alphas_applied` (bank mode).
    pub alphas: Arc<[f64]>,
    /// Posterior mean of the primary effect at each grid point.
    pub effect_means: Arc<[f64]>,
    /// Posterior SD of the primary effect at each grid point.
    pub effect_sds: Arc<[f64]>,
}

/// External-prior conflict shrink summary (attached beside prior sensitivity).
///
/// Records requested vs applied power-prior alphas after conflict policy shrink
/// (orchestration lives in `antecedent-validate`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConflictSummary {
    /// Source artifact / catalog ids.
    pub source_ids: Arc<[Arc<str>]>,
    /// Caller-requested alphas.
    pub alphas_requested: Arc<[f64]>,
    /// Alphas after conflict shrink (`≤` requested).
    pub alphas_applied: Arc<[f64]>,
    /// Prior-PPC p-values used per source (`None` if unused).
    pub p_values: Arc<[Option<f64>]>,
    /// Gaussian KL (nats) used per source (`None` if unused).
    pub kl_values: Arc<[Option<f64>]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplace_requires_convergence() {
        let mut d = InferenceDiagnostics {
            converged: false,
            iterations: 10,
            grad_inf_norm: 1.0,
            hessian_condition: 10.0,
            factorization: HessianFactorization::Cholesky,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: Arc::from("laplace"),
            n_chains: None,
            n_warmup: None,
            ess_bulk_min: None,
            rhat_max: None,
            n_divergences: None,
        };
        assert!(!d.allows_posterior());
        d.converged = true;
        assert!(d.allows_posterior());
    }

    #[test]
    fn mcmc_requires_rhat_and_ess() {
        let mut d = InferenceDiagnostics {
            converged: true,
            iterations: 100,
            grad_inf_norm: 0.0,
            hessian_condition: f64::NAN,
            factorization: HessianFactorization::Mcmc,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: Arc::from("hmc"),
            n_chains: Some(4),
            n_warmup: Some(50),
            ess_bulk_min: Some(5.0),
            rhat_max: Some(1.2),
            n_divergences: Some(0),
        };
        assert!(!d.allows_posterior());
        d.ess_bulk_min = Some(50.0);
        d.rhat_max = Some(1.01);
        assert!(d.allows_posterior());
    }
}
