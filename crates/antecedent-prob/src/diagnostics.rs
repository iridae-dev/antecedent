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
    /// MCMC: minimum tail ESS across parameters.
    pub ess_tail_min: Option<f64>,
    /// MCMC: maximum rank∪folded split-Ř across parameters.
    pub rhat_max: Option<f64>,
    /// MCMC: leapfrog / trajectory divergence count (post-warmup; legacy alias).
    pub n_divergences: Option<u32>,
    /// MCMC: mean Metropolis acceptance probability over all transitions.
    pub mean_accept_prob: Option<f64>,
    /// MCMC: divergences during warmup only.
    pub n_warmup_divergences: Option<u32>,
    /// MCMC: divergences after warmup (publication uses this).
    pub n_postwarmup_divergences: Option<u32>,
    /// MCMC: maximum absolute Hamiltonian energy error observed.
    pub max_abs_delta_h: Option<f64>,
    /// MCMC: every chain moved on at least one unconstrained parameter.
    pub all_chains_moved: Option<bool>,
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
            ess_tail_min: None,
            rhat_max: None,
            n_divergences: None,
            mean_accept_prob: None,
            n_warmup_divergences: None,
            n_postwarmup_divergences: None,
            max_abs_delta_h: None,
            all_chains_moved: None,
        }
    }

    /// Whether this diagnostic set is sufficient to publish a posterior.
    ///
    /// Narrow Laplace posteriors without convergence + curvature are refused.
    /// MCMC requires finite Ř ≤ 1.01, bulk and tail ESS ≥ 100, zero post-warmup
    /// divergences, and movement on every chain.
    #[must_use]
    pub fn allows_posterior(&self) -> bool {
        match self.factorization {
            HessianFactorization::Analytic => true,
            HessianFactorization::Mcmc => self.converged && self.mcmc_publication_ok(),
            HessianFactorization::Cholesky | HessianFactorization::Ldlt => {
                self.converged
                    && self.grad_inf_norm.is_finite()
                    && self.hessian_condition.is_finite()
                    && self.hessian_condition > 0.0
            }
        }
    }

    /// Full MATH-002 MCMC publication predicate (independent of `converged`).
    #[must_use]
    pub fn mcmc_publication_ok(&self) -> bool {
        if self.factorization != HessianFactorization::Mcmc {
            return false;
        }
        let rhat_ok = self.rhat_max.is_some_and(|r| r.is_finite() && r <= 1.01);
        let ess_bulk_ok = self.ess_bulk_min.is_some_and(|e| e.is_finite() && e >= 100.0);
        let ess_tail_ok = self.ess_tail_min.is_some_and(|e| e.is_finite() && e >= 100.0);
        let div_ok = self.n_postwarmup_divergences.is_some_and(|n| n == 0);
        let moved_ok = self.all_chains_moved == Some(true);
        let accept_ok = self.mean_accept_prob.is_some_and(f64::is_finite);
        let delta_ok = self.max_abs_delta_h.is_some_and(f64::is_finite);
        let chains_ok = self.n_chains.is_some_and(|c| c >= 2);
        let warmup_ok = self.n_warmup.is_some();
        rhat_ok
            && ess_bulk_ok
            && ess_tail_ok
            && div_ok
            && moved_ok
            && accept_ok
            && delta_ok
            && chains_ok
            && warmup_ok
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

    fn mcmc_ok_base() -> InferenceDiagnostics {
        InferenceDiagnostics {
            converged: true,
            iterations: 400,
            grad_inf_norm: 0.0,
            hessian_condition: f64::NAN,
            factorization: HessianFactorization::Mcmc,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: Arc::from("hmc"),
            n_chains: Some(4),
            n_warmup: Some(100),
            ess_bulk_min: Some(200.0),
            ess_tail_min: Some(150.0),
            rhat_max: Some(1.005),
            n_divergences: Some(0),
            mean_accept_prob: Some(0.8),
            n_warmup_divergences: Some(0),
            n_postwarmup_divergences: Some(0),
            max_abs_delta_h: Some(0.1),
            all_chains_moved: Some(true),
        }
    }

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
            ess_tail_min: None,
            rhat_max: None,
            n_divergences: None,
            mean_accept_prob: None,
            n_warmup_divergences: None,
            n_postwarmup_divergences: None,
            max_abs_delta_h: None,
            all_chains_moved: None,
        };
        assert!(!d.allows_posterior());
        d.converged = true;
        assert!(d.allows_posterior());
    }

    #[test]
    fn mcmc_requires_full_gate() {
        let mut d = mcmc_ok_base();
        assert!(d.allows_posterior());
        d.ess_bulk_min = Some(50.0);
        assert!(!d.allows_posterior());
        d.ess_bulk_min = Some(200.0);
        d.n_postwarmup_divergences = Some(1);
        assert!(!d.allows_posterior());
        d.n_postwarmup_divergences = Some(0);
        d.all_chains_moved = Some(false);
        assert!(!d.allows_posterior());
        d.all_chains_moved = Some(true);
        d.rhat_max = Some(1.02);
        assert!(!d.allows_posterior());
        d.rhat_max = Some(1.01);
        assert!(d.allows_posterior());
    }

    #[test]
    fn mcmc_divergence_presence_is_not_enough() {
        let mut d = mcmc_ok_base();
        d.n_postwarmup_divergences = Some(3);
        d.n_divergences = Some(3);
        assert!(!d.allows_posterior());
    }
}
