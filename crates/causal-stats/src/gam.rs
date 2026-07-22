//! Generalized additive models — cubic B-splines + backfitting.
//!
//! Gaussian identity additive model `Y = β₀ + Σ fⱼ(Xⱼ) + ε`. Analytic standard
//! errors are not returned; use resampling / bootstrap for uncertainty.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::sync::Arc;

use causal_core::VariableId;

use crate::design::{BasisKind, DesignColumn, DesignColumnMap, DesignColumnRole, RecordedSmooth};
use crate::error::StatsError;
use crate::gram::{form_xtx, invert_square};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresWorkspace};

/// Cubic B-spline degree (order = 4).
const CUBIC_DEGREE: usize = 3;
const CUBIC_ORDER: usize = CUBIC_DEGREE + 1;

/// One smooth term specification for [`fit_gam`].
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothSpec {
    /// Column index into the raw predictor matrix (`x_colmajor`).
    pub raw_col: usize,
    /// Number of cubic B-spline basis columns.
    pub n_basis: usize,
    /// Ridge penalty λ on basis coefficients (must be ≥ 0).
    pub lambda: f64,
    /// Optional full knot vector (length `n_basis + 4` for cubic). When `None`,
    /// interior knots are placed at sample quantiles of the column.
    pub knots: Option<Arc<[f64]>>,
    /// Optional variable id for design provenance.
    pub variable: Option<VariableId>,
}

impl SmoothSpec {
    /// Smooth on `raw_col` with `n_basis` bases and penalty `lambda`.
    #[must_use]
    pub fn new(raw_col: usize, n_basis: usize, lambda: f64) -> Self {
        Self { raw_col, n_basis, lambda, knots: None, variable: None }
    }

    /// Attach a variable id for [`RecordedSmooth`] provenance.
    #[must_use]
    pub fn with_variable(mut self, id: VariableId) -> Self {
        self.variable = Some(id);
        self
    }

    /// Supply a precomputed knot vector.
    #[must_use]
    pub fn with_knots(mut self, knots: impl Into<Arc<[f64]>>) -> Self {
        self.knots = Some(knots.into());
        self
    }
}

/// Options for [`fit_gam`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GamOptions {
    /// Maximum backfitting iterations.
    pub max_iter: u32,
    /// Max absolute change in any fitted smooth value for convergence.
    pub tol: f64,
}

impl Default for GamOptions {
    fn default() -> Self {
        Self { max_iter: 100, tol: 1e-6 }
    }
}

/// Reusable buffers for GAM backfitting.
#[derive(Clone, Debug, Default)]
pub struct GamWorkspace {
    /// Partial residual vector (`nrows`).
    pub partial: Vec<f64>,
    /// Current fitted values (`nrows`).
    pub fitted: Vec<f64>,
    /// Scratch for one smooth's contribution (`nrows`).
    pub smooth_fit: Vec<f64>,
    /// Scratch Gram / solve buffers.
    pub gram: Vec<f64>,
    /// Scratch right-hand side / coefficients.
    pub rhs: Vec<f64>,
    /// Nested least-squares workspace (unused by ridge path; reserved for callers).
    pub ls: LeastSquaresWorkspace,
    grow_count: u32,
}

impl GamWorkspace {
    /// Ensure capacity for `nrows` and max basis width `max_basis`.
    pub fn prepare(&mut self, nrows: usize, max_basis: usize) {
        let grow = |v: &mut Vec<f64>, n: usize, count: &mut u32| {
            if v.capacity() < n {
                *count = count.saturating_add(1);
            }
            if v.len() < n {
                v.resize(n, 0.0);
            } else {
                v.truncate(n);
            }
        };
        grow(&mut self.partial, nrows, &mut self.grow_count);
        grow(&mut self.fitted, nrows, &mut self.grow_count);
        grow(&mut self.smooth_fit, nrows, &mut self.grow_count);
        grow(&mut self.gram, max_basis * max_basis, &mut self.grow_count);
        grow(&mut self.rhs, max_basis, &mut self.grow_count);
    }
}

/// Result of a Gaussian identity GAM fit.
///
/// Analytic standard errors are intentionally omitted; pair with bootstrap.
#[derive(Clone, Debug)]
pub struct GamFit {
    /// Intercept β₀.
    pub intercept: f64,
    /// Concatenated basis coefficients in smooth order (length = Σ `n_basis`).
    pub coefficients: Vec<f64>,
    /// Provenance for each smooth term (knots, λ, column ranges into an expanded design).
    pub smooths: Vec<RecordedSmooth>,
    /// In-sample fitted values.
    pub fitted: Vec<f64>,
    /// Residuals `y − fitted`.
    pub residuals: Vec<f64>,
    /// Approximate effective degrees of freedom (ridge trace formula + intercept).
    pub edf_approx: f64,
    /// Backfitting iterations used.
    pub iterations: u32,
    /// Whether the outer loop converged.
    pub converged: bool,
    /// Rank / condition / backend / allocation diagnostics.
    pub diagnostics: FitDiagnostics,
    /// Raw predictor column indexes matching `smooths` / `coefficients` order.
    raw_cols: Vec<usize>,
    /// Training mean of each uncentered smooth `Bβ` (identifiability centers).
    /// Prediction subtracts these fixed centers — not the prediction-batch mean.
    centers: Vec<f64>,
}

/// Expand one numeric column into a cubic B-spline basis (column-major).
///
/// When `knots` is `None`, builds an open uniform-style knot vector with interior
/// knots at sample quantiles so that there are exactly `n_basis` basis functions.
///
/// # Errors
///
/// Empty `x`, `n_basis < 4`, non-finite values, or invalid supplied knot vector.
pub fn expand_bspline(
    x: &[f64],
    n_basis: usize,
    knots: Option<&[f64]>,
) -> Result<(Vec<f64>, Arc<[f64]>), StatsError> {
    if x.is_empty() {
        return Err(StatsError::Shape { message: "empty x for B-spline expansion" });
    }
    if n_basis < CUBIC_ORDER {
        return Err(StatsError::Shape { message: "n_basis must be ≥ 4 for cubic B-splines" });
    }
    for &v in x {
        if !v.is_finite() {
            return Err(StatsError::Shape {
                message: "non-finite predictor in B-spline expansion",
            });
        }
    }
    let knot_vec: Arc<[f64]> = if let Some(k) = knots {
        validate_knots(k, n_basis)?;
        Arc::from(k.to_vec())
    } else {
        Arc::from(quantile_knots(x, n_basis)?)
    };
    let nrows = x.len();
    let mut basis = vec![0.0; nrows * n_basis];
    for r in 0..nrows {
        eval_cubic_bspline(x[r], &knot_vec, n_basis, &mut basis, r, nrows);
    }
    Ok((basis, knot_vec))
}

/// Build an expanded additive design matrix `[1 | B₁ | B₂ | …]` with column metadata.
///
/// Column ranges on returned [`RecordedSmooth`] values are relative to this expanded matrix
/// (intercept at column 0; smooth bases follow in `specs` order).
///
/// # Errors
///
/// Shape mismatch, bad specs, or B-spline expansion failure.
pub fn compile_additive_design(
    x_colmajor: &[f64],
    nrows: usize,
    n_raw_cols: usize,
    specs: &[SmoothSpec],
) -> Result<(Vec<f64>, DesignColumnMap, Vec<RecordedSmooth>), StatsError> {
    validate_raw_layout(x_colmajor, nrows, n_raw_cols, specs)?;
    let mut ncols = 1usize;
    for s in specs {
        ncols = ncols.saturating_add(s.n_basis);
    }
    let mut matrix = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        matrix[r] = 1.0;
    }
    let mut columns = vec![DesignColumn::from_role(DesignColumnRole::Intercept)];
    let mut smooths = Vec::with_capacity(specs.len());
    let mut col = 1usize;
    for (si, spec) in specs.iter().enumerate() {
        let xcol = raw_column(x_colmajor, nrows, spec.raw_col);
        let (basis, knots) = expand_bspline(xcol, spec.n_basis, spec.knots.as_deref())?;
        let start = col;
        let end = col + spec.n_basis;
        for b in 0..spec.n_basis {
            let src = b * nrows;
            let dst = (start + b) * nrows;
            matrix[dst..dst + nrows].copy_from_slice(&basis[src..src + nrows]);
            let role = match spec.variable {
                Some(id) => DesignColumnRole::Covariate(id),
                None => DesignColumnRole::Covariate(VariableId::from_raw(spec.raw_col as u32)),
            };
            columns.push(DesignColumn {
                role,
                contrast_idx: None,
                standardization_idx: None,
                smooth_idx: Some(si),
            });
        }
        smooths.push(RecordedSmooth {
            variable: spec.variable.or(Some(VariableId::from_raw(spec.raw_col as u32))),
            basis: BasisKind::CubicBSpline,
            knots,
            lambda: spec.lambda,
            column_range: (start, end),
            n_basis: spec.n_basis,
        });
        col = end;
    }
    let map = DesignColumnMap::from_columns(columns).with_smooth_links(&smooths);
    Ok((matrix, map, smooths))
}

/// Fit a Gaussian identity GAM by backfitting penalized cubic B-spline smooths.
///
/// # Errors
///
/// Shape mismatch, invalid λ / basis sizes, singular penalized Gram, or empty specs.
pub fn fit_gam(
    x_colmajor: &[f64],
    nrows: usize,
    n_raw_cols: usize,
    y: &[f64],
    specs: &[SmoothSpec],
    options: &GamOptions,
    _backend: &impl DenseLinearAlgebra,
    workspace: &mut GamWorkspace,
) -> Result<GamFit, StatsError> {
    if specs.is_empty() {
        return Err(StatsError::Shape { message: "GAM requires at least one smooth term" });
    }
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    validate_raw_layout(x_colmajor, nrows, n_raw_cols, specs)?;
    for s in specs {
        if !(s.lambda.is_finite() && s.lambda >= 0.0) {
            return Err(StatsError::Shape { message: "smooth lambda must be finite and ≥ 0" });
        }
        if s.n_basis < CUBIC_ORDER {
            return Err(StatsError::Shape { message: "n_basis must be ≥ 4 for cubic B-splines" });
        }
    }

    let max_basis = specs.iter().map(|s| s.n_basis).max().unwrap_or(0);
    workspace.prepare(nrows, max_basis);

    // Expand bases once.
    let mut bases: Vec<Arc<[f64]>> = Vec::with_capacity(specs.len());
    let mut smooth_meta: Vec<RecordedSmooth> = Vec::with_capacity(specs.len());
    let mut coef_offsets = Vec::with_capacity(specs.len());
    let mut total_coefs = 0usize;
    let mut col_cursor = 1usize; // expanded-design column after intercept
    for spec in specs {
        let xcol = raw_column(x_colmajor, nrows, spec.raw_col);
        let (basis, knots) = expand_bspline(xcol, spec.n_basis, spec.knots.as_deref())?;
        coef_offsets.push(total_coefs);
        total_coefs += spec.n_basis;
        let start = col_cursor;
        let end = col_cursor + spec.n_basis;
        smooth_meta.push(RecordedSmooth {
            variable: spec.variable.or(Some(VariableId::from_raw(spec.raw_col as u32))),
            basis: BasisKind::CubicBSpline,
            knots,
            lambda: spec.lambda,
            column_range: (start, end),
            n_basis: spec.n_basis,
        });
        bases.push(Arc::from(basis));
        col_cursor = end;
    }

    let mut coefficients = vec![0.0; total_coefs];
    // Per-smooth fitted contributions.
    let mut smooth_fits: Vec<Vec<f64>> = (0..specs.len()).map(|_| vec![0.0; nrows]).collect();

    let y_mean = mean(y);
    let mut intercept = y_mean;
    workspace.fitted.fill(intercept);
    let mut converged = false;
    let mut iterations = 0u32;
    let mut edf_approx = 1.0; // intercept
    let mut prev_rss = f64::INFINITY;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        for (j, spec) in specs.iter().enumerate() {
            // partial = y - intercept - sum_{k≠j} f_k
            for r in 0..nrows {
                let mut other = intercept;
                for (k, sf) in smooth_fits.iter().enumerate() {
                    if k != j {
                        other += sf[r];
                    }
                }
                workspace.partial[r] = y[r] - other;
            }
            let basis = bases[j].as_ref();
            let beta = ridge_basis_solve(
                basis,
                nrows,
                spec.n_basis,
                &workspace.partial[..nrows],
                spec.lambda,
                &mut workspace.gram,
                &mut workspace.rhs,
            )?;
            let off = coef_offsets[j];
            coefficients[off..off + spec.n_basis].copy_from_slice(&beta);

            // f_j = B β, then center.
            for r in 0..nrows {
                let mut pred = 0.0;
                for b in 0..spec.n_basis {
                    pred += basis[b * nrows + r] * beta[b];
                }
                workspace.smooth_fit[r] = pred;
            }
            let f_mean = mean(&workspace.smooth_fit[..nrows]);
            for r in 0..nrows {
                workspace.smooth_fit[r] -= f_mean;
                max_delta = max_delta.max((workspace.smooth_fit[r] - smooth_fits[j][r]).abs());
                smooth_fits[j][r] = workspace.smooth_fit[r];
            }

            if iter == 1 {
                edf_approx +=
                    ridge_edf(basis, nrows, spec.n_basis, spec.lambda, &mut workspace.gram)?;
            }
        }
        // Refresh intercept: mean(y - Σ f_j)
        let mut sum = 0.0;
        for r in 0..nrows {
            let mut s = 0.0;
            for sf in &smooth_fits {
                s += sf[r];
            }
            sum += y[r] - s;
        }
        intercept = sum / nrows as f64;

        let mut rss = 0.0;
        for r in 0..nrows {
            let mut pred = intercept;
            for sf in &smooth_fits {
                pred += sf[r];
            }
            workspace.fitted[r] = pred;
            let e = y[r] - pred;
            rss += e * e;
        }

        let fit_scale =
            workspace.fitted[..nrows].iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
        let rss_delta = (prev_rss - rss).abs();
        prev_rss = rss;
        if max_delta < options.tol * fit_scale || rss_delta < options.tol * (1.0 + rss) {
            converged = true;
            break;
        }
    }

    let mut residuals = vec![0.0; nrows];
    for r in 0..nrows {
        residuals[r] = y[r] - workspace.fitted[r];
    }

    // Centers used at fit time: mean(Bβ) before centering each smooth.
    // Recover from final coefficients so predict subtracts the same constants.
    let mut centers = vec![0.0; specs.len()];
    for (j, spec) in specs.iter().enumerate() {
        let basis = bases[j].as_ref();
        let off = coef_offsets[j];
        let mut sum = 0.0;
        for r in 0..nrows {
            let mut pred = 0.0;
            for b in 0..spec.n_basis {
                pred += basis[b * nrows + r] * coefficients[off + b];
            }
            sum += pred;
        }
        centers[j] = sum / nrows as f64;
    }

    let rank = 1 + specs.iter().map(|s| s.n_basis).sum::<usize>();
    let raw_cols: Vec<usize> = specs.iter().map(|s| s.raw_col).collect();
    Ok(GamFit {
        intercept,
        coefficients,
        smooths: smooth_meta,
        fitted: workspace.fitted[..nrows].to_vec(),
        residuals,
        edf_approx,
        iterations,
        converged,
        diagnostics: FitDiagnostics::new(rank, None, "gam", workspace.grow_count),
        raw_cols,
        centers,
    })
}

/// Predict additive fitted values from a [`GamFit`] on new raw predictors.
///
/// `x_colmajor` must have the same raw column layout as the training matrix.
/// Analytic SEs are not returned.
///
/// # Errors
///
/// Shape mismatch or B-spline evaluation failure.
pub fn predict_gam(
    fit: &GamFit,
    x_colmajor: &[f64],
    nrows: usize,
    n_raw_cols: usize,
) -> Result<Vec<f64>, StatsError> {
    if x_colmajor.len() < nrows.saturating_mul(n_raw_cols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if fit.smooths.len() != fit.raw_cols.len() || fit.smooths.len() != fit.centers.len() {
        return Err(StatsError::Backend("GAM fit smooth/raw_col/center length mismatch".into()));
    }
    let mut pred = vec![fit.intercept; nrows];
    let mut coef_off = 0usize;
    for (j, smooth) in fit.smooths.iter().enumerate() {
        let raw_col = fit.raw_cols[j];
        if raw_col >= n_raw_cols {
            return Err(StatsError::Shape { message: "predict raw column out of range" });
        }
        let xcol = raw_column(x_colmajor, nrows, raw_col);
        let (basis, _) = expand_bspline(xcol, smooth.n_basis, Some(smooth.knots.as_ref()))?;
        let center = fit.centers[j];
        for r in 0..nrows {
            let mut s = 0.0;
            for b in 0..smooth.n_basis {
                s += basis[b * nrows + r] * fit.coefficients[coef_off + b];
            }
            pred[r] += s - center;
        }
        coef_off += smooth.n_basis;
    }
    Ok(pred)
}

/// Predict using training-row basis matrices cached on the fit (exact train fitted values).
///
/// Prefer this for in-sample checks; [`predict_gam`] re-expands bases for new `x`.
#[must_use]
pub fn fitted_from_gam(fit: &GamFit) -> &[f64] {
    &fit.fitted
}

fn validate_raw_layout(
    x_colmajor: &[f64],
    nrows: usize,
    n_raw_cols: usize,
    specs: &[SmoothSpec],
) -> Result<(), StatsError> {
    if nrows == 0 {
        return Err(StatsError::Shape { message: "empty design" });
    }
    if x_colmajor.len() < nrows.saturating_mul(n_raw_cols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    for s in specs {
        if s.raw_col >= n_raw_cols {
            return Err(StatsError::Shape { message: "smooth raw_col out of range" });
        }
    }
    Ok(())
}

fn raw_column(x_colmajor: &[f64], nrows: usize, col: usize) -> &[f64] {
    &x_colmajor[col * nrows..(col + 1) * nrows]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn validate_knots(knots: &[f64], n_basis: usize) -> Result<(), StatsError> {
    let need = n_basis + CUBIC_ORDER;
    if knots.len() != need {
        return Err(StatsError::Shape {
            message: "knot vector length must equal n_basis + 4 for cubic B-splines",
        });
    }
    for w in knots.windows(2) {
        if !(w[0].is_finite() && w[1].is_finite()) || w[1] < w[0] {
            return Err(StatsError::Shape { message: "knots must be finite and non-decreasing" });
        }
    }
    Ok(())
}

fn quantile_knots(x: &[f64], n_basis: usize) -> Result<Vec<f64>, StatsError> {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let xmin = sorted[0];
    let xmax = sorted[sorted.len() - 1];
    if !(xmax - xmin).is_finite() {
        return Err(StatsError::Shape { message: "non-finite predictor range" });
    }
    // Degenerate constant column: spread slightly so basis is defined.
    let (xmin, xmax) =
        if (xmax - xmin).abs() < 1e-15 { (xmin - 1.0, xmax + 1.0) } else { (xmin, xmax) };
    let n_interior = n_basis.saturating_sub(CUBIC_ORDER);
    let mut knots = Vec::with_capacity(n_basis + CUBIC_ORDER);
    for _ in 0..CUBIC_ORDER {
        knots.push(xmin);
    }
    if n_interior > 0 {
        let n = sorted.len();
        for i in 1..=n_interior {
            let q = i as f64 / (n_interior + 1) as f64;
            let pos = q * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let t = pos - lo as f64;
            let v = sorted[lo] * (1.0 - t) + sorted[hi.min(n - 1)] * t;
            knots.push(v);
        }
    }
    for _ in 0..CUBIC_ORDER {
        knots.push(xmax);
    }
    Ok(knots)
}

/// Cox–de Boor evaluation of all cubic basis functions at `x` into column-major `out`.
fn eval_cubic_bspline(
    x: f64,
    knots: &[f64],
    n_basis: usize,
    out: &mut [f64],
    row: usize,
    nrows: usize,
) {
    // Clamp to open interval of the interior so the last basis is hit at xmax.
    let eps = 1e-14;
    let left = knots[CUBIC_DEGREE];
    let right = knots[knots.len() - CUBIC_ORDER];
    let xx = if x >= right {
        right - eps
    } else if x < left {
        left
    } else {
        x
    };

    // Find knot span.
    let mut span = CUBIC_DEGREE;
    for i in CUBIC_DEGREE..(knots.len() - CUBIC_ORDER) {
        if xx >= knots[i] && xx < knots[i + 1] {
            span = i;
            break;
        }
        if i == knots.len() - CUBIC_ORDER - 1 {
            span = i;
        }
    }

    // Basis of degree 0..3 on the local span (Piegl/Tiller style).
    let mut ndu = [[0.0_f64; CUBIC_ORDER]; CUBIC_ORDER];
    ndu[0][0] = 1.0;
    let mut left = [0.0_f64; CUBIC_ORDER];
    let mut right = [0.0_f64; CUBIC_ORDER];
    for j in 1..CUBIC_ORDER {
        left[j] = xx - knots[span + 1 - j];
        right[j] = knots[span + j] - xx;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = ndu[r][j - 1] / (right[r + 1] + left[j - r]);
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    // Zero all bases for this row then write the order nonzeros.
    for b in 0..n_basis {
        out[b * nrows + row] = 0.0;
    }
    let first = span.saturating_sub(CUBIC_DEGREE);
    for i in 0..CUBIC_ORDER {
        let b = first + i;
        if b < n_basis {
            out[b * nrows + row] = ndu[i][CUBIC_DEGREE];
        }
    }
}

fn ridge_basis_solve(
    basis: &[f64],
    nrows: usize,
    n_basis: usize,
    y: &[f64],
    lambda: f64,
    gram: &mut [f64],
    rhs: &mut [f64],
) -> Result<Vec<f64>, StatsError> {
    if gram.len() < n_basis * n_basis || rhs.len() < n_basis {
        return Err(StatsError::Backend("GAM workspace too small".into()));
    }
    form_xtx(basis, nrows, n_basis, gram);
    for c in 0..n_basis {
        gram[c * n_basis + c] += lambda;
    }
    for c in 0..n_basis {
        let mut s = 0.0;
        let col = &basis[c * nrows..(c + 1) * nrows];
        for r in 0..nrows {
            s += col[r] * y[r];
        }
        rhs[c] = s;
    }
    let Some(inv) = invert_square(&gram[..n_basis * n_basis], n_basis) else {
        return Err(StatsError::Backend("GAM: singular B'B+λI".into()));
    };
    let mut beta = vec![0.0; n_basis];
    for i in 0..n_basis {
        let mut s = 0.0;
        for j in 0..n_basis {
            s += inv[i * n_basis + j] * rhs[j];
        }
        beta[i] = s;
    }
    Ok(beta)
}

fn ridge_edf(
    basis: &[f64],
    nrows: usize,
    n_basis: usize,
    lambda: f64,
    gram: &mut [f64],
) -> Result<f64, StatsError> {
    form_xtx(basis, nrows, n_basis, gram);
    let mut penalized = gram[..n_basis * n_basis].to_vec();
    for c in 0..n_basis {
        penalized[c * n_basis + c] += lambda;
    }
    let Some(inv) = invert_square(&penalized, n_basis) else {
        return Err(StatsError::Backend("GAM: singular B'B+λI for EDF".into()));
    };
    // edf = tr((B'B+λI)^{-1} B'B) = n_basis - λ tr((B'B+λI)^{-1})
    let mut tr_inv = 0.0;
    for i in 0..n_basis {
        tr_inv += inv[i * n_basis + i];
    }
    Ok(n_basis as f64 - lambda * tr_inv)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;

    fn linspace(n: usize, a: f64, b: f64) -> Vec<f64> {
        (0..n).map(|i| a + (b - a) * (i as f64) / (n - 1) as f64).collect()
    }

    fn colmajor_from_cols(cols: &[Vec<f64>]) -> (Vec<f64>, usize, usize) {
        let nrows = cols[0].len();
        let ncols = cols.len();
        let mut x = vec![0.0; nrows * ncols];
        for (c, col) in cols.iter().enumerate() {
            x[c * nrows..(c + 1) * nrows].copy_from_slice(col);
        }
        (x, nrows, ncols)
    }

    #[test]
    fn expand_bspline_partition_of_unity() {
        let x = linspace(50, -1.0, 1.0);
        let (basis, knots) = expand_bspline(&x, 8, None).unwrap();
        assert_eq!(knots.len(), 8 + CUBIC_ORDER);
        for r in 0..x.len() {
            let mut s = 0.0;
            for b in 0..8 {
                s += basis[b * x.len() + r];
            }
            assert!((s - 1.0).abs() < 1e-10, "row {r} sum={s}");
        }
    }

    #[test]
    fn fit_gam_recovers_additive_signal() {
        let n = 300usize;
        let x1 = linspace(n, 0.0, 1.0);
        let x2: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 2.0 - 1.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 + (2.0 * std::f64::consts::PI * x1[i]).sin() + 0.5 * x2[i] * x2[i])
            .collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1, x2]);
        let specs = [
            SmoothSpec::new(0, 10, 0.1).with_variable(VariableId::from_raw(0)),
            SmoothSpec::new(1, 10, 0.1).with_variable(VariableId::from_raw(1)),
        ];
        let backend = FaerBackend;
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(&x, nrows, ncols, &y, &specs, &GamOptions::default(), &backend, &mut ws)
            .unwrap();
        assert!(fit.converged, "iterations={}", fit.iterations);
        let ss_res: f64 = fit.residuals.iter().map(|e| e * e).sum();
        let y_bar = mean(&y);
        let ss_tot: f64 = y
            .iter()
            .map(|yi| {
                let d = yi - y_bar;
                d * d
            })
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.95, "R²={r2}");
        assert!(fit.edf_approx > 1.0);
        assert_eq!(fit.diagnostics.backend, "gam");
        assert_eq!(fit.smooths.len(), 2);
    }

    #[test]
    fn high_lambda_smooth_approaches_constant_plus_intercept() {
        // Very large λ → nearly constant smooth (after centering ≈ 0) so fit ≈ mean(y).
        let n = 80usize;
        let x1 = linspace(n, -1.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| 3.0 + 0.01 * v).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let specs = [SmoothSpec::new(0, 6, 1e6)];
        let backend = FaerBackend;
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(&x, nrows, ncols, &y, &specs, &GamOptions::default(), &backend, &mut ws)
            .unwrap();
        assert!((fit.intercept - 3.0).abs() < 0.05);
        let max_abs_smooth: f64 =
            fit.fitted.iter().map(|&f| (f - fit.intercept).abs()).fold(0.0, f64::max);
        assert!(max_abs_smooth < 0.05, "max_abs_smooth={max_abs_smooth}");
    }

    #[test]
    fn predict_matches_fitted_on_training() {
        let n = 100usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| (std::f64::consts::PI * v).sin()).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let specs = [SmoothSpec::new(0, 8, 0.01).with_variable(VariableId::from_raw(0))];
        let backend = FaerBackend;
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(&x, nrows, ncols, &y, &specs, &GamOptions::default(), &backend, &mut ws)
            .unwrap();
        let pred = predict_gam(&fit, &x, nrows, ncols).unwrap();
        for r in 0..nrows {
            assert!(
                (pred[r] - fit.fitted[r]).abs() < 1e-6,
                "row {r}: pred={} fit={}",
                pred[r],
                fit.fitted[r]
            );
        }
        assert_eq!(fitted_from_gam(&fit).len(), nrows);
    }

    #[test]
    fn predict_single_row_is_not_just_intercept() {
        // Batch-mean centering would zero the only smooth contribution when nrows=1.
        let n = 80usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| (2.0 * std::f64::consts::PI * v).sin()).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1.clone()]);
        let specs = [SmoothSpec::new(0, 8, 0.01)];
        let backend = FaerBackend;
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(&x, nrows, ncols, &y, &specs, &GamOptions::default(), &backend, &mut ws)
            .unwrap();
        // Quarter-period peak: sin(π/2)=1, away from the mean-zero smooth.
        let idx = n / 4;
        let x_one = vec![x1[idx]];
        let pred = predict_gam(&fit, &x_one, 1, 1).unwrap();
        assert!(
            (pred[0] - fit.fitted[idx]).abs() < 1e-5,
            "single-row pred={} train_fit={} intercept={}",
            pred[0],
            fit.fitted[idx],
            fit.intercept
        );
        assert!((pred[0] - fit.intercept).abs() > 0.5);
    }

    #[test]
    fn compile_additive_design_sets_smooth_links() {
        let n = 20usize;
        let x1 = linspace(n, 0.0, 1.0);
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let specs = [SmoothSpec::new(0, 6, 0.5).with_variable(VariableId::from_raw(7))];
        let (matrix, map, smooths) = compile_additive_design(&x, nrows, ncols, &specs).unwrap();
        assert_eq!(matrix.len(), nrows * (1 + 6));
        assert_eq!(smooths.len(), 1);
        assert_eq!(smooths[0].column_range, (1, 7));
        assert_eq!(smooths[0].n_basis, 6);
        assert_eq!(map.get(0).unwrap().smooth_idx, None);
        assert_eq!(map.get(1).unwrap().smooth_idx, Some(0));
        assert_eq!(map.get(6).unwrap().smooth_idx, Some(0));
        assert_eq!(map.get(1).unwrap().role, DesignColumnRole::Covariate(VariableId::from_raw(7)));
    }

    #[test]
    fn shape_errors() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(expand_bspline(&x, 3, None).is_err());
        assert!(expand_bspline(&[], 6, None).is_err());
        let specs = [SmoothSpec::new(0, 6, -1.0)];
        let backend = FaerBackend;
        let mut ws = GamWorkspace::default();
        let err =
            fit_gam(&x, 3, 1, &[1.0, 2.0, 3.0], &specs, &GamOptions::default(), &backend, &mut ws);
        assert!(err.is_err());
        let specs = [SmoothSpec::new(1, 6, 0.1)];
        let err =
            fit_gam(&x, 3, 1, &[1.0, 2.0, 3.0], &specs, &GamOptions::default(), &backend, &mut ws);
        assert!(err.is_err());
    }

    #[test]
    fn with_smooth_provenance_on_compiled_design() {
        use crate::design::CompiledDesign;
        let t = vec![0.0_f64, 1.0];
        let y = vec![1.0_f64, 2.0];
        let design = CompiledDesign::linear_adjustment(&t, &[], &y, &[]).unwrap();
        assert!(design.smooths.is_empty());
        let smooth = RecordedSmooth {
            variable: Some(VariableId::from_raw(0)),
            basis: BasisKind::CubicBSpline,
            knots: Arc::from(vec![0.0; 10]),
            lambda: 0.1,
            column_range: (1, 2),
            n_basis: 1,
        };
        let design = design.with_smooth_provenance(vec![smooth]);
        assert_eq!(design.smooths.len(), 1);
        assert_eq!(design.columns.get(1).and_then(|c| c.smooth_idx), Some(0));
    }
}
