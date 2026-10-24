//! Split out of the mechanism registry. Behavior is unchanged.

use std::sync::Arc;

use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use super::super::cv::FitFrame;
use super::{data_rss, residual_sigma};
use crate::compile::MechanismSlot;
use crate::error::ModelError;

/// Ordinary least squares of a linear Gaussian mechanism.
pub(super) fn fit_linear_gaussian(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    ridge: f64,
) -> Result<MechanismSlot, ModelError> {
    fit_linear_gaussian_weighted(frame, y, backend, ls_ws, ridge, None)
}

#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
/// Weighted least squares; `weights` of `None` is ordinary least squares.
pub(super) fn fit_linear_gaussian_weighted(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    ridge: f64,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = frame.n_parents();
    let ncols = 1 + p;
    // One assembly of the (weighted, ridge-augmented) design: the data rows are written with
    // their square-root weights and the ridge pseudo-rows below them in the same buffer, so
    // neither a separate unweighted copy nor an augmented second copy exists.
    let extra = if ridge > 0.0 { p } else { 0 };
    let rows = n + extra;
    let root: Option<Vec<f64>> = weights.map(|w| w.iter().map(|v| v.sqrt()).collect());
    let scale = |r: usize| root.as_ref().map_or(1.0, |w| w[r]);
    let mut x = vec![0.0; rows * ncols];
    for r in 0..n {
        x[r] = scale(r);
    }
    for (pi, col) in frame.parents.iter().enumerate() {
        let base = (1 + pi) * rows;
        for r in 0..n {
            x[base + r] = col[r] * scale(r);
        }
    }
    let mut target = vec![0.0; rows];
    for r in 0..n {
        target[r] = y[r] * scale(r);
    }
    if ridge > 0.0 {
        // Ridge rows for the coefficients (not the intercept).
        let sqrt_r = ridge.sqrt();
        for j in 0..p {
            x[(1 + j) * rows + (n + j)] = sqrt_r;
        }
    }
    let fit = backend.least_squares(&x, rows, ncols, &target, ls_ws).map_err(ModelError::from)?;
    let intercept = fit.coefficients[0];
    let coeffs: Arc<[f64]> = Arc::from(fit.coefficients[1..].to_vec());
    // σ from the data rows only: a ridge fit's `fit.rss` also sums the penalty
    // pseudo-rows (`λ Σ β²`), which would inflate σ with the prior.
    let rss = if ridge > 0.0 { data_rss(frame, y, intercept, &coeffs, weights) } else { fit.rss };
    let sigma = residual_sigma(rss, n, ncols, y);
    Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma })
}
