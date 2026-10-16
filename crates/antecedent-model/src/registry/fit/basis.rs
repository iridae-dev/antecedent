//! Split out of the mechanism registry. Behavior is unchanged.
use super::residual_sigma;

use std::sync::Arc;

use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GlmOptions, LeastSquaresWorkspace, MultinomialDesignRef,
};

use super::super::MechanismFamily;
use super::super::cv::FitFrame;
use crate::basis::{ParentBasis, column_moments, spline_knots};
use crate::compile::MechanismSlot;
use crate::error::ModelError;

// ---------------------------------------------------------------- basis families

/// Rows required per fitted column before a basis family is admissible.
///
/// A two-way interaction or spline expansion adds columns quadratically in the
/// parent count. Below this ratio the expansion is fitting noise, and the
/// cross-validated score would say so only after paying for an unstable solve;
/// refusing is cheaper and the refusal is recorded in
/// [`MechanismAssignment::failed_families`].
pub(crate) const BASIS_MIN_ROWS_PER_COLUMN: usize = 10;

/// Relative rank-deficiency floor for the spline solve, on column-scaled design
/// columns. Not a smoothing penalty: smoothing comes from the fixed three-knot
/// cubic *regression* spline, and this only keeps a near-collinear truncated
/// power basis from producing an exploding solve.
pub(super) const SPLINE_CONDITIONING_RIDGE: f64 = 1e-6;

/// Build the expansion a basis family fits on.
pub(super) fn build_basis(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    n: usize,
) -> Result<ParentBasis, ModelError> {
    let p = frame.n_parents();
    if p < 2 {
        return Err(ModelError::Unsupported {
            message: format!(
                "{} needs at least two parents to form a cross-parent product (got {p}); with one \
                 parent the contrast is the same for every unit under any additive-noise \
                 mechanism",
                family.id()
            ),
        });
    }
    let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
    let (centers, scales) = column_moments(&refs, n);
    let centers: Arc<[f64]> = Arc::from(centers);
    let scales: Arc<[f64]> = Arc::from(scales);
    let basis = match family {
        MechanismFamily::LinearInteractions | MechanismFamily::DiscreteInteractions => {
            ParentBasis::interactions(p, centers, scales)?
        }
        MechanismFamily::LinearSpline | MechanismFamily::DiscreteSpline => {
            let knots = spline_knots(&refs, n, &centers, &scales);
            if knots.iter().all(|k| k.is_empty()) {
                return Err(ModelError::Unsupported {
                    message: format!(
                        "{} found no parent with enough distinct values for a spline; the \
                         expansion would duplicate the interaction family",
                        family.id()
                    ),
                });
            }
            ParentBasis::spline_interactions(p, centers, scales, Arc::from(knots))?
        }
        other => {
            return Err(ModelError::Unsupported {
                message: format!("{} is not a basis family", other.id()),
            });
        }
    };
    if !basis.has_cross_parent_product() {
        return Err(ModelError::Unsupported {
            message: format!("{} produced no cross-parent product", family.id()),
        });
    }
    let ncols = 1 + basis.n_terms();
    if n < ncols.saturating_mul(BASIS_MIN_ROWS_PER_COLUMN) {
        return Err(ModelError::Unsupported {
            message: format!(
                "{} expands {p} parents to {ncols} columns and needs at least \
                 {BASIS_MIN_ROWS_PER_COLUMN} rows per column (got {n})",
                family.id()
            ),
        });
    }
    Ok(basis)
}

/// Column-major design `[1 | φ(pa)]` for a basis.
pub(super) fn basis_design_matrix(
    basis: &ParentBasis,
    parent_cols: &[std::borrow::Cow<'_, [f64]>],
    n: usize,
) -> Result<Vec<f64>, ModelError> {
    let t = basis.n_terms();
    let mut x = vec![0.0; n * (1 + t)];
    x[..n].fill(1.0);
    let mut row = vec![0.0; basis.n_parents()];
    let mut expanded = vec![0.0; t];
    for r in 0..n {
        for (p, slot) in row.iter_mut().enumerate() {
            *slot = basis.standardize(p, parent_cols[p][r]);
        }
        basis.expand_standardized(&row, &mut expanded)?;
        for (c, value) in expanded.iter().enumerate() {
            x[(1 + c) * n + r] = *value;
        }
    }
    Ok(x)
}

/// Column-major design, response, weights and ridge strength of one scaled ridge solve.
#[derive(Clone, Copy)]
pub(super) struct RidgeProblem<'a> {
    x: &'a [f64],
    n: usize,
    ncols: usize,
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    ridge: f64,
}

/// Weighted ridge least squares on a column-major design, solved on
/// RMS-normalized columns and un-normalized afterwards.
///
/// Column normalization is an exact change of variables — the fitted surface is
/// identical — and it is what makes a truncated power basis solvable at all: the
/// raw columns of `z³` and `z·z³` differ in scale by orders of magnitude.
/// Returns the coefficients on the original column scale and the weighted RSS.
pub(super) fn solve_scaled_ridge(
    problem: &RidgeProblem<'_>,
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<(Vec<f64>, f64), ModelError> {
    let RidgeProblem { x, n, ncols, y, weights, ridge } = *problem;
    let mut scale = vec![1.0_f64; ncols];
    for (c, s) in scale.iter_mut().enumerate().skip(1) {
        let rms = (x[c * n..c * n + n].iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64).sqrt();
        *s = if rms.is_finite() && rms > 1e-12 { rms } else { 1.0 };
    }
    let extra = if ridge > 0.0 { ncols - 1 } else { 0 };
    let rows = n + extra;
    let mut design = vec![0.0; rows * ncols];
    let mut target = vec![0.0; rows];
    for c in 0..ncols {
        for r in 0..n {
            let w = weights.map_or(1.0, |w| w[r].sqrt());
            design[c * rows + r] = x[c * n + r] / scale[c] * w;
        }
    }
    for r in 0..n {
        target[r] = y[r] * weights.map_or(1.0, |w| w[r].sqrt());
    }
    if ridge > 0.0 {
        let sqrt_r = ridge.sqrt();
        for j in 1..ncols {
            design[j * rows + (n + j - 1)] = sqrt_r;
        }
    }
    let fit =
        backend.least_squares(&design, rows, ncols, &target, ls_ws).map_err(ModelError::from)?;
    let coeffs: Vec<f64> = fit.coefficients.iter().zip(&scale).map(|(b, s)| b / s).collect();
    if coeffs.iter().any(|b| !b.is_finite()) {
        return Err(ModelError::Numerical {
            message: "basis mechanism solve produced a non-finite coefficient".into(),
        });
    }
    // Residual sum of squares over the data rows only: the solver's own RSS also
    // sums the penalty pseudo-rows.
    let mut rss = 0.0;
    for r in 0..n {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += coeffs[c] * x[c * n + r];
        }
        rss += weights.map_or(1.0, |w| w[r]) * (y[r] - pred).powi(2);
    }
    Ok((coeffs, rss))
}

/// Per-row ridge weight for a basis solve: the conditioning floor for a spline
/// expansion, and nothing at all for a pure interaction expansion (which is
/// ordinary least squares on the same columns `LinearGaussian` fits, plus the
/// products).
pub(super) fn basis_ridge_rel(basis: &ParentBasis) -> f64 {
    if basis.knots().iter().any(|k| !k.is_empty()) { SPLINE_CONDITIONING_RIDGE } else { 0.0 }
}

/// Linear Gaussian fit in a basis expansion of the parents.
pub(super) fn fit_linear_basis(
    basis: &ParentBasis,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let x = basis_design_matrix(basis, &frame.parents, n)?;
    let ncols = 1 + basis.n_terms();
    let ridge = basis_ridge_rel(basis) * n as f64;
    let (coeffs, rss) =
        solve_scaled_ridge(&RidgeProblem { x: &x, n, ncols, y, weights, ridge }, backend, ls_ws)?;
    let sigma = residual_sigma(rss, n, ncols, y);
    Ok(MechanismSlot::LinearBasis {
        intercept: coeffs[0],
        basis: basis.clone(),
        coeffs: Arc::from(coeffs[1..].to_vec()),
        sigma,
    })
}

/// Parent-conditional softmax in a basis expansion of the parents.
pub(super) fn fit_discrete_basis(
    basis: &ParentBasis,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let (support, probs) = discrete_support(y, weights)?;
    let k = support.len();
    if k < 2 {
        return Err(ModelError::Unsupported {
            message: "discrete basis families need at least two categories".into(),
        });
    }
    let x = basis_design_matrix(basis, &frame.parents, n)?;
    let ncols = 1 + basis.n_terms();
    let y_cat = discrete_categories(y, &support)?;
    let fit = antecedent_stats::fit_multinomial_logit_weighted(
        MultinomialDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols,
            y_category: &y_cat,
            n_categories: k,
        },
        weights,
        &backend,
        ls_ws,
        &GlmOptions::default(),
    )?;
    if !fit.converged {
        return Err(ModelError::NotConverged {
            message: format!(
                "multinomial logit on a parent basis did not converge (iters={}, deviance={})",
                fit.iterations, fit.deviance
            ),
        });
    }
    Ok(MechanismSlot::DiscreteBasis {
        support: Arc::from(support),
        probs: Arc::from(probs),
        basis: basis.clone(),
        logit_coeffs: Arc::from(fit.coefficients),
    })
}

/// Weighted support and marginal probabilities of a discrete column.
pub(super) fn discrete_support(
    y: &[f64],
    weights: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>), ModelError> {
    let mut pairs: Vec<(i64, f64, f64)> = Vec::new();
    for (r, &yi) in y.iter().enumerate() {
        if !yi.is_finite() {
            continue;
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the key quantises a rounded finite outcome to 1e-6, and the saturating cast only merges outcomes beyond 9e12 in magnitude"
        )]
        let key = (yi * 1e6).round() as i64;
        if let Some(e) = pairs.iter_mut().find(|(k, _, _)| *k == key) {
            e.2 += weights.map_or(1.0, |w| w[r]);
        } else {
            pairs.push((key, yi, weights.map_or(1.0, |w| w[r])));
        }
    }
    if pairs.is_empty() {
        return Err(ModelError::Shape { message: "no finite values for discrete fit".into() });
    }
    pairs.sort_by(|a, b| a.1.total_cmp(&b.1));
    let total = pairs.iter().map(|(_, _, c)| *c).sum::<f64>();
    Ok((
        pairs.iter().map(|(_, v, _)| *v).collect(),
        pairs.iter().map(|(_, _, c)| *c / total).collect(),
    ))
}

/// Category index per row against a fitted support.
pub(super) fn discrete_categories(y: &[f64], support: &[f64]) -> Result<Vec<u32>, ModelError> {
    let mut out = vec![0u32; y.len()];
    for (r, &yi) in y.iter().enumerate() {
        let Some(idx) = support.iter().position(|&s| (s - yi).abs() < 1e-12) else {
            return Err(ModelError::Shape {
                message: "discrete outcome not in fitted support".into(),
            });
        };
        out[r] = u32::try_from(idx)
            .map_err(|_| ModelError::Shape { message: "too many discrete categories".into() })?;
    }
    Ok(out)
}
