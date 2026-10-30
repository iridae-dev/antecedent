//! Inference backend interface and reusable workspaces.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use std::sync::Arc;

use crate::diagnostics::InferenceDiagnostics;
use crate::error::ProbError;
use crate::posterior::PosteriorDraws;
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
    ///
    /// Weights must be finite and ≥ 0. Zero is allowed and drops that row from
    /// the likelihood contribution; the total weight mass must still be > 0.
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
    /// Point estimate of the coefficients (length = ncols): the posterior mode
    /// for Laplace, the posterior mean for conjugate and HMC (the two coincide
    /// for the Gaussian linear model). HMC never reports a best-visited draw as
    /// a mode.
    pub map: Vec<f64>,
    /// Diagnostics (required for Laplace).
    pub diagnostics: InferenceDiagnostics,
    /// Dense posterior covariance (`ncols²`, column-major) when the approximation
    /// is Gaussian (Laplace / conjugate known-σ²). Enables adaptive redraws
    /// without re-fitting MAP.
    pub cov: Option<Vec<f64>>,
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
    /// Position / Newton `beta_old` (ncols or unconstrained HMC dim).
    pub q: Vec<f64>,
    /// Momentum / triangular-solve scratch (same length as `q`).
    pub p: Vec<f64>,
    /// Linear predictor / mu / working residual (nrows).
    pub eta: Vec<f64>,
    /// Working weights / variance terms (nrows).
    pub work_w: Vec<f64>,
    /// Scratch for draws / MVN sampling.
    pub draw_scratch: Vec<f64>,
    /// Times [`Self::prepare`] grew any buffer.
    pub grow_count: u32,
    /// Cached unweighted/weighted `XᵀX` for conjugate refits on the same design matrix.
    pub(crate) conjugate_xtx: Vec<f64>,
    /// Cached effective sample size (sum of weights) paired with [`Self::conjugate_xtx`].
    pub(crate) conjugate_n_eff: f64,
    /// Design-matrix identity last used to fill the conjugate `XᵀX` cache.
    pub(crate) conjugate_key: Option<ConjugateGramKey>,
    /// Owned immutable buffers that certify the unweighted Gaussian validation
    /// and content key. Keeping owners prevents allocator-address reuse.
    immutable_gaussian: Option<ImmutableGaussianDesign>,
}

/// One ephemeral nuisance-fit workspace. Its private owner cannot be cloned,
/// resized by callers, or reused under another admission pool. Field drop order
/// releases the buffers before returning their admission.
pub(crate) struct AdmittedLaplaceWorkspace {
    pub(crate) workspace: LaplaceWorkspace,
    _admission: Option<antecedent_core::execution::AdmissionLease>,
}

impl AdmittedLaplaceWorkspace {
    pub(crate) fn new(nrows: usize, ncols: usize, draws: usize) -> Result<Self, ProbError> {
        let overflow = || ProbError::Shape { message: "Laplace workspace size overflow" };
        let square = ncols.checked_mul(ncols).ok_or_else(overflow)?;
        let draw_cells = draws.checked_mul(ncols).ok_or_else(overflow)?.max(ncols);
        let cells = square
            .checked_mul(2)
            .and_then(|cells| ncols.checked_mul(5).and_then(|cols| cells.checked_add(cols)))
            .and_then(|cells| nrows.checked_mul(2).and_then(|rows| cells.checked_add(rows)))
            .and_then(|cells| cells.checked_add(draw_cells))
            .ok_or_else(overflow)?;
        // Admit before allocating. Fresh boxed slices ensure each reported Vec
        // capacity equals its declared buffer length, with no growth slack.
        let admission = crate::admission::admit_cells(None, cells)?;
        let buffer = |len| vec![0.0; len].into_boxed_slice().into_vec();
        let workspace = LaplaceWorkspace {
            grad: buffer(ncols),
            neg_hessian: buffer(square),
            factor: buffer(square),
            step: buffer(ncols),
            beta: buffer(ncols),
            q: buffer(ncols),
            p: buffer(ncols),
            eta: buffer(nrows),
            work_w: buffer(nrows),
            draw_scratch: buffer(draw_cells),
            grow_count: u32::from(cells > 0),
            ..LaplaceWorkspace::default()
        };
        Ok(Self { workspace, _admission: admission })
    }
}

#[derive(Clone, Debug)]
struct ImmutableGaussianDesign {
    x: Arc<[f64]>,
    y: Arc<[f64]>,
    key: ConjugateGramKey,
    moments: Option<GaussianMoments>,
}

#[derive(Clone, Debug)]
struct GaussianMoments {
    xtx: Vec<f64>,
    xty: Vec<f64>,
    yty: f64,
    n_eff: f64,
}

/// Content/shape key for [`LaplaceWorkspace`]'s conjugate **`XᵀX`** cache.
///
/// Identity is the contents of `X` and weights, not their addresses: a new
/// allocation can reuse a released design pointer. `y` and offsets stay out
/// of the key so equal-length outcome reallocations still recompute `Xᵀy`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ConjugateGramKey {
    content: [u8; 32],
    nrows: usize,
    ncols: usize,
}

impl ConjugateGramKey {
    pub(crate) fn from_design(design: &BayesDesignRef<'_>) -> Self {
        Self { content: hash_design_contents(design), nrows: design.nrows, ncols: design.ncols }
    }
}

fn hash_design_contents(design: &BayesDesignRef<'_>) -> [u8; 32] {
    let mut h = CanonicalWriter::new("antecedent.prob.conjugate_gram.v1");
    h.word(design.nrows as u64).word(design.ncols as u64).reals(design.x_colmajor);
    match design.weights {
        None => {
            h.word(0);
        }
        Some(w) => {
            h.word(1).reals(w);
        }
    }
    h.finish()
}

impl LaplaceWorkspace {
    /// Bind owned immutable, unweighted Gaussian columns for repeated fits.
    ///
    /// Validates once per buffer binding. Later calls with these exact buffers
    /// can reuse validation and the design digest; priors, outcomes under another
    /// binding, and draws are still evaluated for each fit.
    ///
    /// # Errors
    ///
    /// Invalid shape or non-finite Gaussian data.
    pub fn bind_immutable_gaussian_design(
        &mut self,
        x: Arc<[f64]>,
        y: Arc<[f64]>,
        nrows: usize,
        ncols: usize,
    ) -> Result<(), ProbError> {
        let design =
            BayesDesignRef { x_colmajor: &x, y: &y, nrows, ncols, weights: None, offsets: None };
        if self.immutable_gaussian_key(design).is_some() {
            return Ok(());
        }
        crate::likelihood_terms::validate_design(BayesLikelihood::GaussianIdentity, design)?;
        let key = ConjugateGramKey::from_design(&design);
        self.immutable_gaussian = Some(ImmutableGaussianDesign { x, y, key, moments: None });
        Ok(())
    }

    pub(crate) fn immutable_gaussian_key(
        &self,
        design: BayesDesignRef<'_>,
    ) -> Option<ConjugateGramKey> {
        let bound = self.immutable_gaussian.as_ref()?;
        (design.weights.is_none()
            && design.offsets.is_none()
            && design.nrows == bound.key.nrows
            && design.ncols == bound.key.ncols
            && std::ptr::eq(design.x_colmajor, bound.x.as_ref())
            && std::ptr::eq(design.y, bound.y.as_ref()))
        .then_some(bound.key)
    }

    pub(crate) fn restore_immutable_gaussian_moments(
        &mut self,
        design: BayesDesignRef<'_>,
    ) -> Option<(f64, f64)> {
        self.immutable_gaussian_key(design)?;
        let moments = self.immutable_gaussian.as_ref()?.moments.as_ref()?;
        self.neg_hessian[..moments.xtx.len()].copy_from_slice(&moments.xtx);
        self.grad[..moments.xty.len()].copy_from_slice(&moments.xty);
        Some((moments.yty, moments.n_eff))
    }

    pub(crate) fn cache_immutable_gaussian_moments(
        &mut self,
        design: BayesDesignRef<'_>,
        yty: f64,
        n_eff: f64,
    ) {
        if self.immutable_gaussian_key(design).is_none() {
            return;
        }
        if let Some(bound) = self.immutable_gaussian.as_mut() {
            bound.moments = Some(GaussianMoments {
                xtx: self.neg_hessian[..design.ncols * design.ncols].to_vec(),
                xty: self.grad[..design.ncols].to_vec(),
                yty,
                n_eff,
            });
        }
    }

    /// Logical retained buffer bytes (shared immutable buffers counted in full).
    #[must_use]
    pub fn retained_buffer_bytes(&self) -> usize {
        let vectors = [
            &self.grad,
            &self.neg_hessian,
            &self.factor,
            &self.step,
            &self.beta,
            &self.q,
            &self.p,
            &self.eta,
            &self.work_w,
            &self.draw_scratch,
            &self.conjugate_xtx,
        ];
        let mut cells = vectors.iter().fold(0usize, |n, v| n.saturating_add(v.capacity()));
        if let Some(bound) = &self.immutable_gaussian {
            cells = cells.saturating_add(bound.x.len()).saturating_add(bound.y.len());
            if let Some(moments) = &bound.moments {
                cells = cells
                    .saturating_add(moments.xtx.capacity())
                    .saturating_add(moments.xty.capacity());
            }
        }
        cells.saturating_mul(size_of::<f64>())
    }

    /// Invalidate cached sufficient statistics after changing a design in place.
    /// Retains allocated buffers for data-augmentation and other refit loops.
    pub fn invalidate_design(&mut self) {
        self.conjugate_key = None;
        self.immutable_gaussian = None;
    }
    /// Ensure capacity for a design of the given shape (grows, does not shrink).
    pub fn prepare(&mut self, nrows: usize, ncols: usize, n_draws: usize) {
        let mut grew = false;
        grew |= resize_min(&mut self.grad, ncols);
        grew |= resize_min(&mut self.neg_hessian, ncols.saturating_mul(ncols));
        grew |= resize_min(&mut self.factor, ncols.saturating_mul(ncols));
        grew |= resize_min(&mut self.step, ncols);
        grew |= resize_min(&mut self.beta, ncols);
        grew |= resize_min(&mut self.q, ncols);
        grew |= resize_min(&mut self.p, ncols);
        grew |= resize_min(&mut self.eta, nrows);
        grew |= resize_min(&mut self.work_w, nrows);
        let draw_need = n_draws.saturating_mul(ncols).max(ncols);
        grew |= resize_min(&mut self.draw_scratch, draw_need);
        if grew {
            self.grow_count = self.grow_count.saturating_add(1);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_cache_digest_tracks_contents_shape_and_weights_not_allocation() {
        let x = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 3,
            ncols: 2,
            y: &[1.0, 2.0, 3.0],
            weights: None,
            offsets: None,
        };
        let key = ConjugateGramKey::from_design(&design);
        let copy = x.to_vec();
        assert_eq!(
            key,
            ConjugateGramKey::from_design(&BayesDesignRef { x_colmajor: &copy, ..design })
        );
        let mut changed = x;
        changed[4] = -changed[4]; // Same column sums of squares, different cross-product.
        assert_ne!(
            key,
            ConjugateGramKey::from_design(&BayesDesignRef { x_colmajor: &changed, ..design })
        );
        assert_ne!(
            key,
            ConjugateGramKey::from_design(&BayesDesignRef { nrows: 2, ncols: 3, ..design })
        );
        let weighted = BayesDesignRef { weights: Some(&[1.0, 2.0, 1.0]), ..design };
        assert_ne!(key, ConjugateGramKey::from_design(&weighted));
        assert_ne!(
            ConjugateGramKey::from_design(&weighted),
            ConjugateGramKey::from_design(&BayesDesignRef {
                weights: Some(&[2.0, 1.0, 1.0]),
                ..design
            })
        );
    }

    #[test]
    fn nuisance_workspace_owner_charges_exact_capacities_until_buffers_drop() {
        let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(4096));
        let owner = antecedent_core::numerical_admission::with_context(&ctx, || {
            AdmittedLaplaceWorkspace::new(10, 2, 3).unwrap()
        });
        let ws = &owner.workspace;
        let buffers = [
            &ws.grad,
            &ws.neg_hessian,
            &ws.factor,
            &ws.step,
            &ws.beta,
            &ws.q,
            &ws.p,
            &ws.eta,
            &ws.work_w,
            &ws.draw_scratch,
        ];
        assert!(buffers.iter().all(|buffer| buffer.len() == buffer.capacity()));
        let bytes = buffers.iter().map(|buffer| buffer.capacity() as u64 * 8).sum::<u64>();
        assert_eq!(bytes, 352);
        assert_eq!(ctx.admission().available().bytes(), 4096 - bytes);
        let other = ExecutionContext::for_tests(2).with_memory_limit(Some(bytes - 1));
        let refused = antecedent_core::numerical_admission::with_context(&other, || {
            AdmittedLaplaceWorkspace::new(10, 2, 3)
        });
        assert!(refused.is_err(), "another pool must fund its own new workspace");
        assert_eq!(ctx.admission().available().bytes(), 4096 - bytes);
        assert_eq!(other.admission().available().bytes(), bytes - 1);
        drop(owner);
        assert_eq!(ctx.admission().available().bytes(), 4096);
        let unbound = AdmittedLaplaceWorkspace::new(10, 2, 3).unwrap();
        assert!(unbound._admission.is_none(), "direct unbound Rust allocations belong to caller");
    }

    #[test]
    fn immutable_gaussian_binding_never_certifies_other_buffers_or_weights() {
        let x: Arc<[f64]> = Arc::from([1.0, 1.0, 0.0, 1.0]);
        let y: Arc<[f64]> = Arc::from([2.0, 4.0]);
        let mut ws = LaplaceWorkspace::default();
        ws.bind_immutable_gaussian_design(Arc::clone(&x), Arc::clone(&y), 2, 2).unwrap();
        let design = BayesDesignRef {
            x_colmajor: &x,
            y: &y,
            nrows: 2,
            ncols: 2,
            weights: None,
            offsets: None,
        };
        assert!(ws.immutable_gaussian_key(design).is_some());
        // The cache keeps ownership: an allocator cannot recycle these addresses.
        assert_eq!(Arc::strong_count(&x), 2);
        let changed_y = [2.0, f64::NAN];
        assert!(ws.immutable_gaussian_key(BayesDesignRef { y: &changed_y, ..design }).is_none());
        assert!(
            ws.bind_immutable_gaussian_design(Arc::clone(&x), Arc::from(changed_y), 2, 2).is_err()
        );
        assert!(
            ws.immutable_gaussian_key(BayesDesignRef { weights: Some(&[1.0, 2.0]), ..design })
                .is_none()
        );
        assert!(
            ws.immutable_gaussian_key(BayesDesignRef { offsets: Some(&[0.0, 1.0]), ..design })
                .is_none()
        );
        assert!(
            ws.immutable_gaussian_key(BayesDesignRef { nrows: 1, ncols: 4, ..design }).is_none()
        );
        ws.invalidate_design();
        assert!(ws.immutable_gaussian_key(design).is_none());
        assert_eq!(Arc::strong_count(&x), 1);
    }

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
