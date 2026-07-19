//! Bayesian inference configuration for the facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_estimate::BayesianBackendKind;
use causal_prob::BayesLikelihood;

/// Frequentist vs Bayesian inference mode.
#[derive(Clone, Debug, PartialEq)]
pub enum InferenceMode {
    /// Classical point-estimate path (default).
    Frequentist,
    /// Bayesian g-computation / posterior path.
    Bayesian(BayesianConfig),
}

impl Default for InferenceMode {
    fn default() -> Self {
        Self::Frequentist
    }
}

/// Bayesian analysis configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianConfig {
    /// Backend kind.
    pub backend: BayesianBackendKind,
    /// Likelihood (Laplace path).
    pub likelihood: BayesLikelihood,
    /// Posterior draws.
    pub n_draws: usize,
    /// Isotropic prior scale.
    pub prior_scale: f64,
}

impl BayesianConfig {
    /// Laplace Gaussian defaults.
    #[must_use]
    pub fn laplace() -> Self {
        Self {
            backend: BayesianBackendKind::Laplace,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            prior_scale: 10.0,
        }
    }

    /// Conjugate Gaussian defaults.
    #[must_use]
    pub fn conjugate() -> Self {
        Self {
            backend: BayesianBackendKind::ConjugateGaussian,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            prior_scale: 10.0,
        }
    }

    /// Native HMC defaults.
    #[must_use]
    pub fn hmc() -> Self {
        Self {
            backend: BayesianBackendKind::Hmc,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 200,
            prior_scale: 10.0,
        }
    }

    /// Weakly informative prior scale.
    #[must_use]
    pub fn prior_scale(mut self, scale: f64) -> Self {
        self.prior_scale = scale;
        self
    }

    /// Draw count.
    #[must_use]
    pub fn n_draws(mut self, n: usize) -> Self {
        self.n_draws = n;
        self
    }
}
