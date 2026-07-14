//! Inference diagnostics for Laplace / conjugate backends (DESIGN.md §14.5, §18.4 subset).
//!
//! Chain ESS / divergence counts are deferred to MCMC adapters.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// Factorization used for the Laplace covariance.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HessianFactorization {
    /// Cholesky of the negative Hessian.
    Cholesky,
    /// Structured LDLT fallback.
    Ldlt,
    /// Analytic conjugate (exact posterior; no Hessian).
    Analytic,
}

/// Convergence / curvature diagnostics required before reporting a Laplace posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceDiagnostics {
    /// Whether the optimizer reported convergence.
    pub converged: bool,
    /// Iterations used.
    pub iterations: u32,
    /// Final gradient infinity-norm (MAP).
    pub grad_inf_norm: f64,
    /// Estimated condition number of −Hessian (or NaN if unavailable).
    pub hessian_condition: f64,
    /// Factorization path used.
    pub factorization: HessianFactorization,
    /// Separation / complete-separation warning for Bernoulli models.
    pub separation_warning: bool,
    /// Human-readable notes.
    pub notes: Vec<Arc<str>>,
    /// Backend identifier (e.g. "laplace", "conjugate_gaussian").
    pub backend_id: Arc<str>,
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
        }
    }

    /// Whether this diagnostic set is sufficient to publish a posterior.
    ///
    /// Narrow Laplace posteriors without convergence + curvature are refused.
    #[must_use]
    pub fn allows_posterior(&self) -> bool {
        if self.factorization == HessianFactorization::Analytic {
            return true;
        }
        self.converged
            && self.grad_inf_norm.is_finite()
            && self.hessian_condition.is_finite()
            && self.hessian_condition > 0.0
    }
}

/// Optional prior-sensitivity summary attached to a causal posterior.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PriorSensitivitySummary {
    /// Prior scale grid evaluated.
    pub prior_scales: Arc<[f64]>,
    /// Posterior mean of the primary effect at each scale.
    pub effect_means: Arc<[f64]>,
    /// Posterior SD of the primary effect at each scale.
    pub effect_sds: Arc<[f64]>,
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
        };
        assert!(!d.allows_posterior());
        d.converged = true;
        assert!(d.allows_posterior());
    }
}
