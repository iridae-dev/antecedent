//! Posterior target adapters for HMC trajectories.

use crate::backend::{BayesDesignRef, BayesLikelihood, LaplaceWorkspace};
use crate::error::ProbError;
use crate::gaussian_target::PosteriorTarget;
use crate::likelihood_terms::{LikelihoodScratch, accumulate_likelihood, log_posterior_value};
use crate::prior::{CoefficientPrecision, GaussianCoefficientPrior};

/// A posterior the leapfrog integrator can traverse.
///
/// The two implementors differ only in how they evaluate the gradient and the
/// log density, and in which sweep errors mean "this trajectory diverged":
///
/// * the GLM score path sweeps the data once per leapfrog step for `∇U` and once
///   more per transition for the final log posterior (`L + 1` sweeps), and
///   propagates every [`ProbError`] except [`ProbError::Numerical`];
/// * the Gaussian target returns the log density with its gradient (`L` sweeps)
///   and treats any error as a divergent trajectory.
///
/// Both leave the gradient in a [`LaplaceWorkspace`] slot (`step` for GLM, which
/// holds `∇U`; `grad` for Gaussian, which holds the log-posterior score), and the
/// slot is also the convention of the carried gradient between transitions.
pub(super) trait HmcTarget {
    /// Log density at `q` and, in `carried`, the gradient in this target's slot
    /// convention. Uses `ws` as scratch.
    fn init(
        &mut self,
        q: &[f64],
        ws: &mut LaplaceWorkspace,
        carried: &mut [f64],
    ) -> Result<f64, ProbError>;

    /// Gradient at `ws.q[..dim]`, written to [`Self::slot`]. `Some(logp)` when the
    /// same sweep also yields the log density.
    fn eval_grad(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
    ) -> Result<Option<f64>, ProbError>;

    /// Log density at the trajectory's final state `ws.q[..dim]`; `lp_last` is the
    /// last value [`Self::eval_grad`] returned (or the old density if none).
    fn final_logp(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
        lp_last: f64,
    ) -> Result<f64, ProbError>;

    /// Whether a sweep error means a divergent trajectory (otherwise it aborts).
    fn is_divergent(error: &ProbError) -> bool;

    /// Workspace slice holding this target's gradient after [`Self::eval_grad`].
    fn slot(ws: &LaplaceWorkspace) -> &[f64];

    /// `∇U` component from a value stored in the gradient slot.
    fn grad_u(slot_value: f64) -> f64;
}

/// GLM score-path target (every likelihood but Gaussian identity).
pub(super) struct GlmTarget<'a> {
    pub(super) likelihood: BayesLikelihood,
    pub(super) design: BayesDesignRef<'a>,
    pub(super) coef_prior: &'a GaussianCoefficientPrior,
    pub(super) prec: &'a CoefficientPrecision,
}

impl HmcTarget for GlmTarget<'_> {
    fn init(
        &mut self,
        q: &[f64],
        ws: &mut LaplaceWorkspace,
        carried: &mut [f64],
    ) -> Result<f64, ProbError> {
        let nrows = self.design.nrows;
        let ncols = self.design.ncols;
        // The current state's log-posterior and ∇U are carried across transitions:
        // after an accept they are exactly the trajectory's final-state values, after
        // a reject they are unchanged, so no transition re-sweeps the data for them.
        let lp = log_posterior_value(
            self.likelihood,
            self.design,
            q,
            self.coef_prior,
            self.prec,
            &mut ws.eta[..nrows],
            1.0,
        )?;
        self.neg_log_posterior_grad(
            q,
            LikelihoodScratch {
                grad: &mut ws.grad[..ncols],
                neg_hess: &mut ws.neg_hessian[..ncols * ncols],
                eta: &mut ws.eta[..nrows],
                work_w: &mut ws.work_w[..nrows],
            },
            carried,
        )?;
        Ok(lp)
    }

    fn eval_grad(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
    ) -> Result<Option<f64>, ProbError> {
        let nrows = self.design.nrows;
        self.neg_log_posterior_grad(
            &ws.q[..dim],
            LikelihoodScratch {
                grad: &mut ws.grad[..dim],
                neg_hess: &mut ws.neg_hessian[..dim * dim],
                eta: &mut ws.eta[..nrows],
                work_w: &mut ws.work_w[..nrows],
            },
            &mut ws.step[..dim],
        )?;
        Ok(None)
    }

    fn final_logp(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
        _lp_last: f64,
    ) -> Result<f64, ProbError> {
        let nrows = self.design.nrows;
        log_posterior_value(
            self.likelihood,
            self.design,
            &ws.q[..dim],
            self.coef_prior,
            self.prec,
            &mut ws.eta[..nrows],
            1.0,
        )
    }

    fn is_divergent(error: &ProbError) -> bool {
        matches!(error, ProbError::Numerical { .. })
    }

    fn slot(ws: &LaplaceWorkspace) -> &[f64] {
        // `step` holds ∇U afterwards, disjoint from `grad` (log-posterior score).
        &ws.step
    }

    fn grad_u(slot_value: f64) -> f64 {
        slot_value
    }
}

/// Gaussian-identity target behind [`PosteriorTarget`].
pub(super) struct GaussianRef<'a>(pub(super) &'a mut dyn PosteriorTarget);

impl HmcTarget for GaussianRef<'_> {
    fn init(
        &mut self,
        q: &[f64],
        ws: &mut LaplaceWorkspace,
        carried: &mut [f64],
    ) -> Result<f64, ProbError> {
        let dim = q.len();
        // Log-posterior and its gradient at the current state, carried across transitions.
        let lp = self.0.logp_and_grad(q, &mut ws.grad[..dim])?;
        carried.copy_from_slice(&ws.grad[..dim]);
        Ok(lp)
    }

    fn eval_grad(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
    ) -> Result<Option<f64>, ProbError> {
        self.0.logp_and_grad(&ws.q[..dim], &mut ws.grad[..dim]).map(Some)
    }

    fn final_logp(
        &mut self,
        _dim: usize,
        _ws: &mut LaplaceWorkspace,
        lp_last: f64,
    ) -> Result<f64, ProbError> {
        // The last leapfrog evaluation already produced the final density.
        Ok(lp_last)
    }

    fn is_divergent(_error: &ProbError) -> bool {
        true
    }

    fn slot(ws: &LaplaceWorkspace) -> &[f64] {
        &ws.grad
    }

    fn grad_u(slot_value: f64) -> f64 {
        -slot_value
    }
}

impl GlmTarget<'_> {
    /// `∇U(beta)` into `grad_out`, using `scratch` for the likelihood sweep.
    fn neg_log_posterior_grad(
        &self,
        beta: &[f64],
        scratch: LikelihoodScratch<'_>,
        grad_out: &mut [f64],
    ) -> Result<(), ProbError> {
        let LikelihoodScratch { grad: grad_lp, neg_hess, eta, work_w } = scratch;
        accumulate_likelihood(
            self.likelihood,
            self.design,
            beta,
            LikelihoodScratch { grad: &mut *grad_lp, neg_hess, eta, work_w },
            1.0,
            false,
        )?;
        self.prec.sub_prior_gradient(beta, &self.coef_prior.mean, grad_lp);
        for i in 0..beta.len() {
            grad_out[i] = -grad_lp[i];
        }
        Ok(())
    }
}
