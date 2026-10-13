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
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, col) in frame.parents.iter().enumerate() {
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    let weighted_y;
    let y_fit = if let Some(weights) = weights {
        weighted_y = y.iter().zip(weights).map(|(v, w)| v * w.sqrt()).collect::<Vec<_>>();
        for c in 0..ncols {
            for r in 0..n {
                x[c * n + r] *= weights[r].sqrt();
            }
        }
        weighted_y.as_slice()
    } else {
        y
    };
    let fit = if ridge > 0.0 {
        // Augment with ridge rows for coefficients (not intercept).
        let extra = p;
        let mut x2 = vec![0.0; (n + extra) * ncols];
        let mut y2 = vec![0.0; n + extra];
        for c in 0..ncols {
            for r in 0..n {
                x2[c * (n + extra) + r] = x[c * n + r];
            }
        }
        y2[..n].copy_from_slice(y_fit);
        let sqrt_r = ridge.sqrt();
        for j in 0..p {
            x2[(1 + j) * (n + extra) + (n + j)] = sqrt_r;
        }
        backend.least_squares(&x2, n + extra, ncols, &y2, ls_ws).map_err(ModelError::from)?
    } else {
        backend.least_squares(&x, n, ncols, y_fit, ls_ws).map_err(ModelError::from)?
    };
    let intercept = fit.coefficients[0];
    let coeffs: Arc<[f64]> = Arc::from(fit.coefficients[1..].to_vec());
    // σ from the data rows only: a ridge fit's `fit.rss` also sums the penalty
    // pseudo-rows (`λ Σ β²`), which would inflate σ with the prior.
    let rss = if ridge > 0.0 { data_rss(frame, y, intercept, &coeffs, weights) } else { fit.rss };
    let sigma = residual_sigma(rss, n, ncols, y);
    Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma })
}
