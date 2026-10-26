//! Least-squares, logistic and inverse helpers over column-major designs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, fit_glm_ridge,
    fit_glm_ridge_weighted, fit_glm_weighted, form_xtx, invert_square,
};

use crate::error::EstimationError;

/// Solve `A x = b` for symmetric positive-definite row-major `A` (`p × p`).
#[must_use]
pub(crate) fn solve_spd(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    crate::linalg::solve_spd(a, b, p)
}
/// `X'X` (row-major `p × p`) of a column-major `n × p` design.
#[must_use]
pub(crate) fn gram(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut xtx = vec![0.0; p * p];
    form_xtx(x, n, p, &mut xtx);
    xtx
}

/// Coefficients and rank consumed from one `FaerBackend` least-squares solve.
pub(crate) struct LsFit {
    coefficients: Vec<f64>,
    rank: usize,
}

impl LsFit {
    /// Coefficient vector, length `ncols`.
    pub(crate) fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Numerical rank of the design.
    pub(crate) const fn rank(&self) -> usize {
        self.rank
    }
}

/// Solve `min ||X β − y||` on a column-major design.
///
/// The coefficients and rank are the backend fit's fields.
///
/// # Errors
///
/// A stats-layer shape or numerical failure.
pub(crate) fn ls_fit(
    backend: FaerBackend,
    design_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LsFit, EstimationError> {
    let fit =
        backend.least_squares(design_colmajor, nrows, ncols, y, workspace)?;
    Ok(LsFit { coefficients: fit.coefficients, rank: fit.rank })
}

/// OLS residual variance `σ² = RSS / (n − p)` for a fitted coefficient vector.
///
/// The residual sum of squares is compensated. `NaN` when the fit has no positive residual
/// degrees of freedom (`nrows <= ncols`): a saturated fit carries no information about `σ²`,
/// and a caller that needs a number must say so explicitly rather than read a divide-by-one.
pub(crate) fn ols_sigma2(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    beta: &[f64],
) -> f64 {
    if nrows <= ncols {
        return f64::NAN;
    }
    // Column-outer accumulation walks each contiguous column once; every row still sums its
    // terms in ascending column order from zero, so the fitted values are those of the
    // row-outer loop bit for bit.
    let mut fitted = vec![0.0; nrows];
    for (column, &coefficient) in x_colmajor.chunks_exact(nrows).zip(beta).take(ncols) {
        for (acc, &value) in fitted.iter_mut().zip(column) {
            *acc += value * coefficient;
        }
    }
    let rss = antecedent_kernels::numeric::CompensatedSum::of(
        y.iter().zip(&fitted).map(|(&observed, &pred)| {
            let e = observed - pred;
            e * e
        }),
    );
    rss / (nrows - ncols) as f64
}

/// `(XᵀX)⁻¹` for a column-major `nrows × ncols` design, or `None` if it is rank deficient.
///
/// Derived from the triangular factor of a QR of the design itself (`R⁻¹R⁻ᵀ`, see
/// [`crate::linalg::xtx_inverse_qr`]): the Gram matrix is never formed, and the rank rule is the
/// one the least-squares point estimate applies. The unscaled inverse-information matrix of every
/// analytic-SE helper.
pub(crate) fn xtx_inverse(x_colmajor: &[f64], nrows: usize, ncols: usize) -> Option<Vec<f64>> {
    crate::linalg::xtx_inverse_qr(x_colmajor, nrows, ncols)
}

/// Ridge-penalised logistic regression through the stats GLM engine: the maximiser of
/// `Σ wᵢ ℓᵢ(β) − ½ λ |β₁..|²` for a column-major `nrows × ncols` design whose constant first
/// column, when present, is the unpenalised intercept. `weights` are nonnegative case weights
/// (`None` is all ones). The one logistic fit for estimators that need coefficients only;
/// floors, damping and the convergence rule are the engine's.
///
/// # Errors
///
/// A shape, outcome, weight or penalty the engine refuses (`lambda` must be positive and
/// finite), a numerical failure, or a fit that did not converge.
pub(crate) fn fit_logistic_ridge(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    lambda: f64,
) -> Result<Vec<f64>, EstimationError> {
    let design = GlmDesignRef { x_colmajor, nrows, ncols, y };
    let options = GlmOptions::default();
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = match weights {
        Some(weights) => fit_glm_ridge_weighted(
            GlmFamily::BinomialLogit,
            design,
            weights,
            &FaerBackend,
            &mut workspace,
            &options,
            lambda,
        ),
        None => fit_glm_ridge(
            GlmFamily::BinomialLogit,
            design,
            &FaerBackend,
            &mut workspace,
            &options,
            lambda,
        ),
    }?;
    if !fit.converged || fit.coefficients.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::stats_msg("the ridge logistic fit did not converge"));
    }
    Ok(fit.coefficients)
}

/// Unpenalised weighted logistic regression through the stats GLM engine, for a column-major
/// `nrows × ncols` design and nonnegative case weights. Coefficients only: a fit that did not
/// converge or is (quasi-)completely separated has no finite maximum-likelihood estimate and is
/// an error.
///
/// # Errors
///
/// A shape, outcome or weight the engine refuses, a numerical failure, non-convergence or
/// separation.
pub(crate) fn fit_logistic_weighted(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let design = GlmDesignRef { x_colmajor, nrows, ncols, y };
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_glm_weighted(
        GlmFamily::BinomialLogit,
        design,
        weights,
        &FaerBackend,
        &mut workspace,
        &GlmOptions::default().without_separation_ridge(),
    )?;
    if !fit.converged || fit.separated || fit.coefficients.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::stats_msg(
            "the weighted logistic fit did not converge or is separated",
        ));
    }
    Ok(fit.coefficients)
}

/// Invert a small fixed-size matrix with [`invert_square`]'s scale-free rule, or `None` when it
/// is singular or non-finite.
///
/// The one inverse for the Bayesian IV / RD and local-polynomial families, which carry their
/// moment matrices as `[[f64; N]; N]`.
#[must_use]
pub(crate) fn invert_fixed<const N: usize>(a: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let flat: Vec<f64> = a.iter().flatten().copied().collect();
    let inv = invert_square(&flat, N)?;
    Some(std::array::from_fn(|i| std::array::from_fn(|j| inv[i * N + j])))
}

/// [`invert_fixed`] for a runtime-sized square matrix held as rows; `None` when the rows are not
/// square, or the matrix is singular or non-finite.
#[must_use]
pub(crate) fn invert_rows(rows: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = rows.len();
    if rows.iter().any(|r| r.len() != n) {
        return None;
    }
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let inv = invert_square(&flat, n)?;
    Some(inv.chunks(n).map(<[f64]>::to_vec).collect())
}

/// Delta-method SE for a linear contrast `gᵀ β`: `sqrt(σ² · gᵀ (XᵀX)⁻¹ g)`.
///
/// Returns `NaN` if the Gram matrix is singular or the quadratic form is non-finite.
pub(crate) fn delta_method_se(inv_xtx: &[f64], ncols: usize, g: &[f64], sigma2: f64) -> f64 {
    if g.len() != ncols || inv_xtx.len() != ncols * ncols {
        return f64::NAN;
    }
    // v = inv · g
    let mut v = vec![0.0; ncols];
    for i in 0..ncols {
        let mut s = 0.0;
        for j in 0..ncols {
            s += inv_xtx[i * ncols + j] * g[j];
        }
        v[i] = s;
    }
    let mut q = 0.0;
    for i in 0..ncols {
        q += g[i] * v[i];
    }
    let var = sigma2 * q;
    if !var.is_finite() {
        return f64::NAN;
    }
    var.max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_inverse_judges_singularity_scale_free() {
        // A design in units of 1e-8 has pivots far below the old absolute 1e-12 cutoff but is
        // perfectly conditioned once equilibrated.
        let tiny = [[1e-16, 0.0, 0.0], [0.0, 4e-16, 0.0], [0.0, 0.0, 9e-16]];
        let inv = invert_fixed(&tiny).expect("a scaled diagonal is not singular");
        assert!((inv[1][1] * 4e-16 - 1.0).abs() < 1e-12);
        // A genuinely collinear pair is still refused, at any scale.
        let collinear = [[1.0, 2.0], [2.0, 4.0]];
        assert!(invert_fixed(&collinear).is_none());
        let scaled = [[1e12, 2e12], [2e12, 4e12]];
        assert!(invert_fixed(&scaled).is_none());
        // Non-finite entries are refused rather than propagated.
        assert!(invert_fixed(&[[f64::NAN, 0.0], [0.0, 1.0]]).is_none());
    }

    #[test]
    fn row_inverse_matches_the_fixed_inverse_and_refuses_ragged_input() {
        let rows = vec![vec![1e-16, 0.0], vec![0.0, 4e-16]];
        let inv = invert_rows(&rows).expect("a scaled diagonal is not singular");
        assert!((inv[0][0] * 1e-16 - 1.0).abs() < 1e-12 && inv[0][1] == 0.0);
        assert!(invert_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none());
        assert!(invert_rows(&[vec![1.0, 0.0], vec![0.0]]).is_none());
    }
}
