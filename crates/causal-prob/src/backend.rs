//! Inference backend interface and reusable workspaces (DESIGN.md §14.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::ExecutionContext;

use crate::diagnostics::InferenceDiagnostics;
use crate::error::ProbError;
use crate::posterior::{PosteriorDraws, PosteriorSchema};
use crate::prior::PriorSet;

/// Likelihood / link for Bayesian GLM.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BayesLikelihood {
    /// Gaussian with identity link.
    GaussianIdentity,
    /// Bernoulli with logit link.
    BernoulliLogit,
    /// Bernoulli with probit link.
    BernoulliProbit,
    /// Poisson with log link.
    PoissonLog,
}

/// Borrowed design for Bayesian fitting (column-major).
#[derive(Clone, Copy, Debug)]
pub struct BayesDesignRef<'a> {
    /// Column-major design.
    pub x_colmajor: &'a [f64],
    /// Rows.
    pub nrows: usize,
    /// Columns.
    pub ncols: usize,
    /// Outcome.
    pub y: &'a [f64],
    /// Optional observation weights (length `nrows`).
    pub weights: Option<&'a [f64]>,
    /// Optional offsets (length `nrows`).
    pub offsets: Option<&'a [f64]>,
}

/// Fitting / draw options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesFitOptions {
    /// Number of posterior draws to materialize.
    pub n_draws: usize,
    /// Maximum Newton iterations (Laplace).
    pub max_iter: u32,
    /// Gradient infinity-norm convergence tolerance.
    pub grad_tol: f64,
    /// RNG seed for draws (deterministic given seed).
    pub seed: u64,
}

impl Default for BayesFitOptions {
    fn default() -> Self {
        Self { n_draws: 1000, max_iter: 50, grad_tol: 1e-8, seed: 0 }
    }
}

/// Result of a Bayesian coefficient fit.
#[derive(Clone, Debug)]
pub struct BayesFitResult {
    /// Columnar coefficient (and dispersion) draws.
    pub draws: PosteriorDraws,
    /// MAP / posterior mean coefficients (length = ncols).
    pub map: Vec<f64>,
    /// Diagnostics (required for Laplace).
    pub diagnostics: InferenceDiagnostics,
}

/// Backend-neutral Bayesian inference interface.
pub trait InferenceBackend: Send + Sync {
    /// Fit coefficients under `likelihood` / `prior` and return columnar draws.
    ///
    /// # Errors
    ///
    /// Shape, prior, numerical, or missing-diagnostics failures.
    fn fit(
        &self,
        likelihood: BayesLikelihood,
        design: BayesDesignRef<'_>,
        prior: &PriorSet,
        options: &BayesFitOptions,
        workspace: &mut LaplaceWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BayesFitResult, ProbError>;
}

/// Reusable gradient / Hessian / factorization workspace for Laplace.
///
/// Buffers grow on demand and are reused across iterations and fits.
#[derive(Clone, Debug, Default)]
pub struct LaplaceWorkspace {
    /// Gradient (length = ncols).
    pub grad: Vec<f64>,
    /// Dense negative Hessian, column-major (ncols²).
    pub neg_hessian: Vec<f64>,
    /// Cholesky / LDLT factor buffer (ncols²).
    pub factor: Vec<f64>,
    /// Newton step / scratch (ncols).
    pub step: Vec<f64>,
    /// Working coefficients (ncols).
    pub beta: Vec<f64>,
    /// Linear predictor / mu / working residual (nrows).
    pub eta: Vec<f64>,
    /// Working weights / variance terms (nrows).
    pub work_w: Vec<f64>,
    /// Scratch for draws / MVN sampling.
    pub draw_scratch: Vec<f64>,
    /// Times [`Self::prepare`] grew any buffer.
    pub grow_count: u32,
}

impl LaplaceWorkspace {
    /// Ensure capacity for a design of the given shape (grows, does not shrink).
    pub fn prepare(&mut self, nrows: usize, ncols: usize, n_draws: usize) {
        let mut grew = false;
        grew |= resize_min(&mut self.grad, ncols);
        grew |= resize_min(&mut self.neg_hessian, ncols.saturating_mul(ncols));
        grew |= resize_min(&mut self.factor, ncols.saturating_mul(ncols));
        grew |= resize_min(&mut self.step, ncols);
        grew |= resize_min(&mut self.beta, ncols);
        grew |= resize_min(&mut self.eta, nrows);
        grew |= resize_min(&mut self.work_w, nrows);
        let draw_need = n_draws.saturating_mul(ncols).max(ncols);
        grew |= resize_min(&mut self.draw_scratch, draw_need);
        if grew {
            self.grow_count = self.grow_count.saturating_add(1);
        }
    }

    /// Clear numeric contents without freeing capacity (for tests / reuse checks).
    pub fn zero_numeric(&mut self) {
        for v in [
            &mut self.grad,
            &mut self.neg_hessian,
            &mut self.factor,
            &mut self.step,
            &mut self.beta,
            &mut self.eta,
            &mut self.work_w,
            &mut self.draw_scratch,
        ] {
            for x in v.iter_mut() {
                *x = 0.0;
            }
        }
    }
}

fn resize_min(buf: &mut Vec<f64>, need: usize) -> bool {
    if buf.len() < need {
        buf.resize(need, 0.0);
        true
    } else {
        false
    }
}

/// Helper: schema for coefficient draws only.
#[must_use]
pub fn coefficient_schema(ncols: usize) -> PosteriorSchema {
    PosteriorSchema::coefficients(ncols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_reuses_buffers() {
        let mut ws = LaplaceWorkspace::default();
        ws.prepare(100, 5, 200);
        let g1 = ws.grow_count;
        assert!(g1 >= 1);
        ws.prepare(100, 5, 200);
        assert_eq!(ws.grow_count, g1, "second prepare must not grow");
        ws.prepare(200, 5, 200);
        assert!(ws.grow_count > g1);
    }
}
