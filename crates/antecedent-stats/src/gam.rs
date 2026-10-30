//! Generalized additive models — cubic B-splines + backfitting.
//!
//! Gaussian identity additive model `Y = β₀ + Σ fⱼ(Xⱼ) + ε`. Each smooth is
//! fit with a roughness penalty (the second-difference `P = D₂'D₂` by default, or the knot-aware
//! integrated squared second derivative, see [`RoughnessPenalty`]), not coefficient ridge. Analytic standard errors are not
//! returned; use resampling / bootstrap for uncertainty.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

#[cfg(test)]
use crate::design::DesignColumn;
#[cfg(test)]
use crate::design::DesignColumnMap;
#[cfg(test)]
use crate::design::DesignColumnRole;
use crate::design::{BasisKind, RecordedSmooth};
use crate::error::StatsError;
use crate::gram::form_xtx;
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresWorkspace};

/// Cubic B-spline degree (order = 4).
const CUBIC_DEGREE: usize = 3;
const CUBIC_ORDER: usize = CUBIC_DEGREE + 1;

/// A column-major raw predictor matrix of `nrows × n_raw_cols`, the input every
/// [`SmoothSpec::raw_col`] indexes into.
#[derive(Clone, Copy, Debug)]
pub struct PredictorMatrix<'a> {
    /// Column-major values, `nrows * n_raw_cols` long.
    pub x_colmajor: &'a [f64],
    /// Rows of the matrix.
    pub nrows: usize,
    /// Raw predictor columns.
    pub n_raw_cols: usize,
}

impl<'a> PredictorMatrix<'a> {
    /// View `x_colmajor` as `nrows × n_raw_cols`.
    #[must_use]
    pub fn new(x_colmajor: &'a [f64], nrows: usize, n_raw_cols: usize) -> Self {
        Self { x_colmajor, nrows, n_raw_cols }
    }
}

/// Which quadratic roughness penalty a smooth carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoughnessPenalty {
    /// Second differences of adjacent coefficients, `D₂'D₂`. It ignores the knot spacing, so a
    /// straight line is penalised whenever the knots are not equally spaced.
    #[default]
    CoefficientSecondDifference,
    /// The integrated squared second derivative `∫ s''(x)² dx` of the fitted smooth, scaled by
    /// the cube of the mean interior knot spacing so λ keeps the size it has for equally spaced
    /// knots and is unchanged by rescaling the predictor. It is exactly zero on every straight
    /// line, whatever the knots, so a linear signal is not shrunk.
    IntegratedSecondDerivative,
}

/// One smooth term specification for [`fit_gam`].
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothSpec {
    /// Column index into the raw predictor matrix (`x_colmajor`).
    pub raw_col: usize,
    /// Number of cubic B-spline basis columns.
    pub n_basis: usize,
    /// Second-difference roughness penalty λ (≥ 0). Ignored when [`Self::auto_lambda`].
    pub lambda: f64,
    /// When true, choose λ on a log-spaced grid by minimizing GCV for this smooth.
    pub auto_lambda: bool,
    /// Optional full knot vector (length `n_basis + 4` for cubic). When `None`,
    /// interior knots are placed at sample quantiles of the column.
    pub knots: Option<Arc<[f64]>>,
    /// Optional variable id for design provenance.
    pub variable: Option<VariableId>,
    /// The roughness penalty the smooth carries.
    pub penalty: RoughnessPenalty,
}

impl SmoothSpec {
    /// Smooth on `raw_col` with `n_basis` bases and roughness penalty `lambda`.
    #[must_use]
    pub fn new(raw_col: usize, n_basis: usize, lambda: f64) -> Self {
        Self {
            raw_col,
            n_basis,
            lambda,
            auto_lambda: false,
            knots: None,
            variable: None,
            penalty: RoughnessPenalty::default(),
        }
    }

    /// Smooth whose λ is selected by GCV on a log-spaced grid.
    #[must_use]
    pub fn auto(raw_col: usize, n_basis: usize) -> Self {
        Self {
            raw_col,
            n_basis,
            lambda: 0.0,
            auto_lambda: true,
            knots: None,
            variable: None,
            penalty: RoughnessPenalty::default(),
        }
    }

    /// Use `penalty` as the roughness penalty.
    #[must_use]
    pub fn with_penalty(mut self, penalty: RoughnessPenalty) -> Self {
        self.penalty = penalty;
        self
    }

    /// Attach a variable id for [`RecordedSmooth`] provenance.
    #[must_use]
    #[cfg(test)]
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
    /// Nested least-squares workspace (unused by the roughness path; reserved for callers).
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
    /// Approximate effective degrees of freedom (roughness-penalty trace + intercept) at the
    /// final smoothing parameters.
    pub edf_approx: f64,
    /// Indices (into `smooths`) of auto-λ smooths whose selected λ sits at the edge of the
    /// GCV grid (`1e-6` or `1e6`): the score was still improving there, so the true
    /// optimum may lie outside the searched range.
    pub boundary_lambda_smooths: Vec<usize>,
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
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
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
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
#[cfg(test)]
pub(crate) fn compile_additive_design(
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
                None => DesignColumnRole::Covariate(column_variable_id(spec.raw_col)),
            };
            columns.push(DesignColumn {
                role,
                contrast_idx: None,
                standardization_idx: None,
                smooth_idx: Some(si),
            });
        }
        smooths.push(RecordedSmooth {
            variable: spec.variable.or(Some(column_variable_id(spec.raw_col))),
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

/// Fit a Gaussian identity GAM by backfitting roughness-penalized cubic B-spline smooths.
///
/// Each smooth uses the second-difference penalty `P = D₂'D₂` in
/// `(B'B + λP)β = B'y`. The intercept (and any future parametric columns) are
/// unpenalized. When [`SmoothSpec::auto_lambda`] is set, λ is chosen by GCV on a
/// log-spaced grid for that smooth's partial residuals.
///
/// # Errors
///
/// Shape mismatch, invalid λ / basis sizes, singular penalized Gram, or empty specs.
pub fn fit_gam(
    predictors: &PredictorMatrix<'_>,
    y: &[f64],
    specs: &[SmoothSpec],
    options: &GamOptions,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut GamWorkspace,
) -> Result<GamFit, StatsError> {
    fit_gam_weighted(predictors, y, specs, options, None, backend, workspace)
}

/// Fit the same additive GAM with nonnegative observation weights.
/// Weights are normalized to sum to nrows so the roughness penalty retains its scale.
/// Knots remain selected on the original predictor support.
///
/// # Errors
/// Invalid weights or any error from the unweighted GAM fit.
pub fn fit_gam_weighted(
    predictors: &PredictorMatrix<'_>,
    y: &[f64],
    specs: &[SmoothSpec],
    options: &GamOptions,
    weights: Option<&[f64]>,
    _backend: &impl DenseLinearAlgebra,
    workspace: &mut GamWorkspace,
) -> Result<GamFit, StatsError> {
    let PredictorMatrix { x_colmajor, nrows, n_raw_cols } = *predictors;
    validate_weights(weights, nrows)?;
    if specs.is_empty() {
        return Err(StatsError::Shape { message: "GAM requires at least one smooth term" });
    }
    validate_response(y, nrows)?;
    let design = AdditiveDesign::expand(x_colmajor, nrows, n_raw_cols, specs)?;
    fit_gam_weighted_design(&design, y, options, weights, workspace)
}

fn validate_weights(weights: Option<&[f64]>, nrows: usize) -> Result<(), StatsError> {
    if let Some(w) = weights {
        let sum: f64 = w.iter().sum();
        if w.len() != nrows
            || w.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !sum.is_finite()
            || sum <= 0.0
        {
            return Err(StatsError::Shape { message: "invalid GAM observation weights" });
        }
    }
    Ok(())
}

fn validate_response(y: &[f64], nrows: usize) -> Result<(), StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape { message: "GAM response must be finite" });
    }
    Ok(())
}

/// An additive GAM design expanded once over a fixed row set: every smooth's
/// knot vector and cubic B-spline basis, ready for any number of
/// [`fit_gam_weighted_design`] refits on those rows.
///
/// [`fit_gam_weighted`] expands the design on every call, which re-places the
/// quantile knots (a sort of each column) and re-evaluates every basis. A
/// caller that refits the same rows under different responses or observation
/// weights — a Bayesian-bootstrap draw loop over fixed cross-fitting folds —
/// expands once and reuses the design; the fit is bit-identical to the
/// per-call expansion, because the knots and bases are the same values.
#[derive(Clone, Debug)]
pub struct AdditiveDesign {
    nrows: usize,
    /// Specs as given, each with its knots resolved.
    specs: Vec<SmoothSpec>,
    /// Column-major `[nrows × n_basis]` basis of each smooth.
    bases: Vec<Arc<[f64]>>,
    /// Provenance template of each smooth (λ is the spec's; auto-λ fits overwrite it).
    smooths: Vec<RecordedSmooth>,
    coef_offsets: Vec<usize>,
    total_coefs: usize,
}

impl AdditiveDesign {
    /// Expand `specs` over the `nrows × n_raw_cols` column-major predictor matrix.
    ///
    /// Knots left `None` on a spec are placed at the column's sample quantiles,
    /// exactly as [`fit_gam_weighted`] places them.
    ///
    /// # Errors
    ///
    /// Empty specs, shape mismatch, invalid λ / basis sizes, or B-spline
    /// expansion failure.
    pub fn expand(
        x_colmajor: &[f64],
        nrows: usize,
        n_raw_cols: usize,
        specs: &[SmoothSpec],
    ) -> Result<Self, StatsError> {
        if specs.is_empty() {
            return Err(StatsError::Shape { message: "GAM requires at least one smooth term" });
        }
        validate_raw_layout(x_colmajor, nrows, n_raw_cols, specs)?;
        for s in specs {
            if !(s.auto_lambda || (s.lambda.is_finite() && s.lambda >= 0.0)) {
                return Err(StatsError::Shape {
                    message: "smooth lambda must be finite and ≥ 0"
                });
            }
            if s.n_basis < CUBIC_ORDER {
                return Err(StatsError::Shape {
                    message: "n_basis must be ≥ 4 for cubic B-splines",
                });
            }
        }
        let mut resolved: Vec<SmoothSpec> = Vec::with_capacity(specs.len());
        let mut bases: Vec<Arc<[f64]>> = Vec::with_capacity(specs.len());
        let mut smooths: Vec<RecordedSmooth> = Vec::with_capacity(specs.len());
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
            smooths.push(RecordedSmooth {
                variable: spec.variable.or(Some(column_variable_id(spec.raw_col))),
                basis: BasisKind::CubicBSpline,
                knots: Arc::clone(&knots),
                lambda: spec.lambda,
                column_range: (start, end),
                n_basis: spec.n_basis,
            });
            resolved.push(SmoothSpec { knots: Some(knots), ..spec.clone() });
            bases.push(Arc::from(basis));
            col_cursor = end;
        }
        Ok(Self { nrows, specs: resolved, bases, smooths, coef_offsets, total_coefs })
    }

    /// Rows the design was expanded over.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// The smooths with their resolved knots.
    #[must_use]
    pub fn specs(&self) -> &[SmoothSpec] {
        &self.specs
    }
}

/// Fit the additive GAM of an expanded [`AdditiveDesign`] to `y` under optional
/// nonnegative observation weights: [`fit_gam_weighted`] without the expansion.
///
/// # Errors
///
/// Invalid weights or response, or a singular penalized Gram.
#[allow(
    clippy::too_many_lines,
    reason = "backfitting loop with per-smooth GCV; the sweep state is read by every step"
)]
pub fn fit_gam_weighted_design(
    design: &AdditiveDesign,
    y: &[f64],
    options: &GamOptions,
    weights: Option<&[f64]>,
    workspace: &mut GamWorkspace,
) -> Result<GamFit, StatsError> {
    let nrows = design.nrows;
    let specs = design.specs.as_slice();
    let bases = design.bases.as_slice();
    validate_weights(weights, nrows)?;
    let weights = match weights {
        Some(w) => {
            let sum: f64 = w.iter().sum();
            w.iter().map(|v| v * nrows as f64 / sum).collect::<Vec<_>>()
        }
        None => vec![1.0; nrows],
    };
    let weighted_mean =
        |v: &[f64]| v.iter().zip(&weights).map(|(v, w)| v * w).sum::<f64>() / nrows as f64;
    validate_response(y, nrows)?;

    let max_basis = specs.iter().map(|s| s.n_basis).max().unwrap_or(0);
    workspace.prepare(nrows, max_basis);

    let mut smooth_meta: Vec<RecordedSmooth> = design.smooths.clone();
    let coef_offsets = design.coef_offsets.as_slice();
    let total_coefs = design.total_coefs;
    let mut chosen_lambda: Vec<f64> = specs.iter().map(|s| s.lambda).collect();

    let weighted_bases: Vec<Vec<f64>> = bases
        .iter()
        .map(|basis| basis.iter().enumerate().map(|(i, v)| v * weights[i % nrows].sqrt()).collect())
        .collect();
    let sqrt_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    // `B'B` and the penalty depend on neither λ nor the response: form them once per
    // smooth. The Cholesky factor of `B'B + λP` is cached per smooth and only recomputed when
    // λ changes, so a backfit sweep costs `O(n K)` per smooth (the right-hand side `B'r`)
    // plus a triangular solve rather than an `O(n K²)` Gram plus a dense inverse.
    let smoothers: Vec<PenalizedSmoother> = weighted_bases
        .iter()
        .zip(specs)
        .map(|(basis, spec)| {
            let penalty = roughness_penalty(spec.penalty, spec.knots.as_deref(), spec.n_basis)?;
            PenalizedSmoother::with_penalty(basis, nrows, spec.n_basis, penalty)
        })
        .collect::<Result<_, _>>()?;
    let mut factors: Vec<Option<PenalizedFactor>> = (0..specs.len()).map(|_| None).collect();
    let mut boundary_lambda = vec![false; specs.len()];
    // Auto-λ is re-selected against the current partial residuals every sweep until a full
    // sweep leaves every λ unchanged (or the sweep cap freezes it), because the partial
    // residual of smooth `j` only stops containing the other smooths' unfitted signal once
    // they have been fitted. Convergence is not declared while λ is still moving.
    let mut lambda_settled = !specs.iter().any(|s| s.auto_lambda);
    let mut weighted_partial = vec![0.0; nrows];
    let mut coefficients = vec![0.0; total_coefs];
    // Per-smooth fitted contributions.
    let mut smooth_fits: Vec<Vec<f64>> = (0..specs.len()).map(|_| vec![0.0; nrows]).collect();

    let y_mean = weighted_mean(y);
    let mut intercept = y_mean;
    workspace.fitted.fill(intercept);
    let mut converged = false;
    let mut iterations = 0u32;
    let mut prev_rss = f64::INFINITY;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        let mut lambda_changed = false;
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
            let solve_basis = weighted_bases[j].as_slice();
            for r in 0..nrows {
                weighted_partial[r] = workspace.partial[r] * sqrt_weights[r];
            }
            if spec.auto_lambda && !lambda_settled {
                let (lambda, at_boundary) = select_lambda_gcv(
                    &smoothers[j],
                    solve_basis,
                    nrows,
                    &weighted_partial,
                    &sqrt_weights,
                )?;
                boundary_lambda[j] = at_boundary;
                if factors[j].is_none() || lambda.to_bits() != chosen_lambda[j].to_bits() {
                    chosen_lambda[j] = lambda;
                    smooth_meta[j].lambda = lambda;
                    factors[j] = None;
                    lambda_changed = true;
                }
            }
            if factors[j].is_none() {
                factors[j] = Some(smoothers[j].factor(chosen_lambda[j])?);
            }
            let inverse = factors[j].as_ref().expect("factor cached above");
            let beta = smoothers[j].coefficients(inverse, solve_basis, nrows, &weighted_partial);
            let off = coef_offsets[j];
            coefficients[off..off + spec.n_basis].copy_from_slice(&beta);

            // f_j = B β, then center (identifiability; intercept absorbs the mean).
            for r in 0..nrows {
                let mut pred = 0.0;
                for b in 0..spec.n_basis {
                    pred += basis[b * nrows + r] * beta[b];
                }
                workspace.smooth_fit[r] = pred;
            }
            let f_mean = weighted_mean(&workspace.smooth_fit[..nrows]);
            for r in 0..nrows {
                workspace.smooth_fit[r] -= f_mean;
                max_delta = max_delta.max((workspace.smooth_fit[r] - smooth_fits[j][r]).abs());
                smooth_fits[j][r] = workspace.smooth_fit[r];
            }
        }
        if !lambda_settled && (!lambda_changed || iter >= MAX_LAMBDA_RESELECT_SWEEPS) {
            lambda_settled = true;
        }
        // Refresh intercept: mean(y - Σ f_j)
        let mut sum = 0.0;
        for r in 0..nrows {
            let mut s = 0.0;
            for sf in &smooth_fits {
                s += sf[r];
            }
            sum += weights[r] * (y[r] - s);
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
            rss += weights[r] * e * e;
        }

        // The scale of the smooth part alone: the intercept carries the outcome's offset, and a
        // tolerance scaled by the offset would admit an absolute error in the centred smooths
        // that grows with it.
        let fit_scale = (0..nrows)
            .map(|r| smooth_fits.iter().map(|sf| sf[r]).sum::<f64>().abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let rss_delta = (prev_rss - rss).abs();
        prev_rss = rss;
        if lambda_settled
            && (max_delta < options.tol * fit_scale || rss_delta < options.tol * (1.0 + rss))
        {
            converged = true;
            break;
        }
    }

    // `smoothers[j].edf` is tr(S₁) for the *uncentered* smoother S₁ = B(BᵀB+λP)⁻¹Bᵀ. The
    // B-spline basis is a partition of unity and P annihilates constant coefficient vectors,
    // so S₁·1 = 1 exactly for every λ — the constant is an eigenvector with eigenvalue 1.
    // The centering step projects that direction out, so the smoother actually applied has
    // tr(S₁) − 1 degrees of freedom. Without the −1 the constant is counted twice (once
    // here, once in the intercept) and edf_approx runs high by exactly one per smooth term.
    // Evaluated at the final λ, not the first sweep's.
    let mut edf_approx = 1.0; // intercept (unpenalized)
    for (smoother, inverse) in smoothers.iter().zip(&factors) {
        if let Some(inverse) = inverse {
            edf_approx += smoother.edf(inverse) - 1.0;
        }
    }
    let boundary_lambda_smooths: Vec<usize> =
        (0..specs.len()).filter(|&j| specs[j].auto_lambda && boundary_lambda[j]).collect();

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
            sum += weights[r] * pred;
        }
        centers[j] = sum / nrows as f64;
    }

    // Every B-spline block spans the constants (partition of unity), so the additive design
    // has one shared constant direction: identifiable rank is `1 + Σ (n_basis − 1)`.
    let rank = 1 + specs.iter().map(|s| s.n_basis - 1).sum::<usize>();
    let raw_cols: Vec<usize> = specs.iter().map(|s| s.raw_col).collect();
    Ok(GamFit {
        intercept,
        coefficients,
        smooths: smooth_meta,
        fitted: workspace.fitted[..nrows].to_vec(),
        residuals,
        edf_approx,
        boundary_lambda_smooths,
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
#[cfg(test)]
pub(crate) fn fitted_from_gam(fit: &GamFit) -> &[f64] {
    &fit.fitted
}

impl GamFit {
    /// Predict one row without heap allocation (hot-path form of [`predict_gam`]).
    ///
    /// Evaluates only the `CUBIC_ORDER` nonzero basis functions per smooth via the
    /// span-local Cox–de Boor recursion; the result matches a one-row
    /// [`predict_gam`] call bit for bit.
    ///
    /// # Errors
    ///
    /// Row shorter than the fit's raw column layout, or internal length mismatch.
    pub fn predict_row(&self, raw_row: &[f64]) -> Result<f64, StatsError> {
        if self.smooths.len() != self.raw_cols.len() || self.smooths.len() != self.centers.len() {
            return Err(StatsError::Backend(
                "GAM fit smooth/raw_col/center length mismatch".into(),
            ));
        }
        let mut pred = self.intercept;
        let mut coef_off = 0usize;
        for (j, smooth) in self.smooths.iter().enumerate() {
            let raw_col = self.raw_cols[j];
            let Some(&x) = raw_row.get(raw_col) else {
                return Err(StatsError::Shape { message: "predict raw column out of range" });
            };
            pred += self.smooth_dot(smooth, coef_off, x) - self.centers[j];
            coef_off += smooth.n_basis;
        }
        Ok(pred)
    }

    /// Centered contribution `f_j(x) − c_j` of one smooth term at a scalar `x`.
    ///
    /// For an additive fit, a prediction is
    /// `intercept + Σ_j smooth_partial(j, x_j)`; callers exploiting additivity
    /// (e.g. averaging over covariates at a fixed treatment) can evaluate a single
    /// term instead of a full row.
    ///
    /// # Errors
    ///
    /// `smooth_index` out of range.
    pub fn smooth_partial(&self, smooth_index: usize, x: f64) -> Result<f64, StatsError> {
        if smooth_index >= self.smooths.len() || smooth_index >= self.centers.len() {
            return Err(StatsError::Shape { message: "smooth index out of range" });
        }
        let coef_off: usize = self.smooths[..smooth_index].iter().map(|s| s.n_basis).sum();
        let smooth = &self.smooths[smooth_index];
        Ok(self.smooth_dot(smooth, coef_off, x) - self.centers[smooth_index])
    }

    /// Index of the smooth attached to raw predictor column `raw_col`, if any.
    #[must_use]
    pub fn smooth_for_raw_col(&self, raw_col: usize) -> Option<usize> {
        self.raw_cols.iter().position(|&c| c == raw_col)
    }

    /// Uncentered `f_j(x) = Σ_b B_b(x) β_b` over the span-local nonzero bases.
    fn smooth_dot(&self, smooth: &RecordedSmooth, coef_off: usize, x: f64) -> f64 {
        let (span, values) = cubic_bspline_nonzeros(x, smooth.knots.as_ref());
        let first = span.saturating_sub(CUBIC_DEGREE);
        let mut s = 0.0;
        for (i, &v) in values.iter().enumerate() {
            let b = first + i;
            if b < smooth.n_basis {
                s += v * self.coefficients[coef_off + b];
            }
        }
        s
    }

    /// Derivative `f_j'(x)` of one uncentered smooth.
    ///
    /// The evaluation map clamps `x` outside the knot domain to a constant,
    /// so the derivative is zero there. At either endpoint, return the
    /// one-sided derivative from inside the domain.
    ///
    /// # Errors
    ///
    /// `smooth_index` out of range.
    pub fn smooth_derivative(&self, smooth_index: usize, x: f64) -> Result<f64, StatsError> {
        if smooth_index >= self.smooths.len() {
            return Err(StatsError::Shape { message: "smooth index out of range" });
        }
        let smooth = &self.smooths[smooth_index];
        let knots = smooth.knots.as_ref();
        if knots.len() < CUBIC_ORDER + CUBIC_DEGREE {
            return Err(StatsError::Shape { message: "smooth knot vector is too short" });
        }
        let left = knots[CUBIC_DEGREE];
        let right = knots[knots.len() - CUBIC_ORDER];
        if x < left || x > right {
            return Ok(0.0);
        }
        let coef_off: usize = self.smooths[..smooth_index].iter().map(|s| s.n_basis).sum();
        let (span, ders) = cubic_bspline_deriv_nonzeros(x, knots);
        let first = span.saturating_sub(CUBIC_DEGREE);
        let mut s = 0.0;
        for (i, &d) in ders.iter().enumerate() {
            let b = first + i;
            if b < smooth.n_basis {
                s += d * self.coefficients[coef_off + b];
            }
        }
        Ok(s)
    }

    /// Derivative of every basis function of one smooth at `x`: the vector
    /// `d` with `smooth_derivative(j, x) = Σ_b d_b β_b`.
    ///
    /// A plug-in derivative is this linear functional of the fitted
    /// coefficients, so a coefficient sandwich gives its variance. The same
    /// clamping as [`Self::smooth_derivative`] applies: outside the knot
    /// domain every entry is zero; endpoints use the interior one-sided derivative.
    ///
    /// # Errors
    ///
    /// `smooth_index` out of range, or a knot vector too short for a cubic basis.
    pub fn smooth_basis_derivative(
        &self,
        smooth_index: usize,
        x: f64,
    ) -> Result<Vec<f64>, StatsError> {
        let smooth = self
            .smooths
            .get(smooth_index)
            .ok_or(StatsError::Shape { message: "smooth index out of range" })?;
        let knots = smooth.knots.as_ref();
        if knots.len() < CUBIC_ORDER + CUBIC_DEGREE {
            return Err(StatsError::Shape { message: "smooth knot vector is too short" });
        }
        let mut out = vec![0.0; smooth.n_basis];
        let left = knots[CUBIC_DEGREE];
        let right = knots[knots.len() - CUBIC_ORDER];
        if x < left || x > right {
            return Ok(out);
        }
        let (span, ders) = cubic_bspline_deriv_nonzeros(x, knots);
        let first = span.saturating_sub(CUBIC_DEGREE);
        for (i, &d) in ders.iter().enumerate() {
            if let Some(slot) = out.get_mut(first + i) {
                *slot = d;
            }
        }
        Ok(out)
    }
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

/// Dataset column index as a [`VariableId`].
#[allow(
    clippy::cast_possible_truncation,
    reason = "VariableId raw indices are u32 by construction, so a dataset column index always fits"
)]
fn column_variable_id(raw_col: usize) -> VariableId {
    VariableId::from_raw(raw_col as u32)
}

fn raw_column(x_colmajor: &[f64], nrows: usize, col: usize) -> &[f64] {
    &x_colmajor[col * nrows..(col + 1) * nrows]
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

#[allow(clippy::float_cmp, reason = "Only exactly constant data need an artificial knot domain")]
fn quantile_knots(x: &[f64], n_basis: usize) -> Result<Vec<f64>, StatsError> {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let xmin = sorted[0];
    let xmax = sorted[sorted.len() - 1];
    if !(xmax - xmin).is_finite() {
        return Err(StatsError::Shape { message: "non-finite predictor range" });
    }
    // Degenerate constant column: spread slightly so basis is defined.
    let (xmin, xmax) = if xmax == xmin { (xmin - 1.0, xmax + 1.0) } else { (xmin, xmax) };
    let n_interior = n_basis.saturating_sub(CUBIC_ORDER);
    let mut knots = Vec::with_capacity(n_basis + CUBIC_ORDER);
    for _ in 0..CUBIC_ORDER {
        knots.push(xmin);
    }
    if n_interior > 0 {
        for i in 1..=n_interior {
            let q = i as f64 / (n_interior + 1) as f64;
            knots.push(antecedent_kernels::quantile_type7_sorted(&sorted, q));
        }
    }
    for _ in 0..CUBIC_ORDER {
        knots.push(xmax);
    }
    Ok(knots)
}

/// Span-local Cox–de Boor evaluation: the knot span and the `CUBIC_ORDER`
/// nonzero cubic basis values at `x` (all other basis functions are exactly 0).
fn cubic_bspline_nonzeros(x: f64, knots: &[f64]) -> (usize, [f64; CUBIC_ORDER]) {
    // Evaluate the boundary on the last span, including caller-supplied knot
    // vectors that are not clamped. An absolute epsilon moves small-unit
    // inputs outside their knot domain.
    let left = knots[CUBIC_DEGREE];
    let right = knots[knots.len() - CUBIC_ORDER];
    let xx = x.max(left).min(right);

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
    if xx >= right {
        // Quantile knots can repeat the boundary more than degree + 1 times
        // for discrete covariates. Use the last nonempty interval; evaluating
        // a zero-width trailing span would divide by zero.
        span = (CUBIC_DEGREE..(knots.len() - CUBIC_ORDER))
            .rev()
            .find(|&i| knots[i] < right)
            .unwrap_or(CUBIC_DEGREE);
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

    let mut values = [0.0_f64; CUBIC_ORDER];
    for (i, v) in values.iter_mut().enumerate() {
        *v = ndu[i][CUBIC_DEGREE];
    }
    (span, values)
}

/// First derivatives of the span-local cubic bases.
///
/// Outside the knot domain the clamped evaluation is constant, so this
/// returns zeros. Endpoint derivatives are one-sided from inside the domain.
/// Derivatives use the standard degree-drop recurrence
/// on the same span as [`cubic_bspline_nonzeros`]; coincident end knots contribute
/// a zero term rather than a division by zero.
fn cubic_bspline_deriv_nonzeros(x: f64, knots: &[f64]) -> (usize, [f64; CUBIC_ORDER]) {
    let left = knots[CUBIC_DEGREE];
    let right = knots[knots.len() - CUBIC_ORDER];
    let (span, _) = cubic_bspline_nonzeros(x, knots);
    if x < left || x > right {
        return (span, [0.0; CUBIC_ORDER]);
    }
    let quadratic = quadratic_nonzeros(x, knots, span);
    let mut ders = [0.0_f64; CUBIC_ORDER];
    for k in 0..CUBIC_ORDER {
        let i = span - CUBIC_DEGREE + k;
        let dt0 = knots[i + CUBIC_DEGREE] - knots[i];
        let dt1 = knots[i + CUBIC_ORDER] - knots[i + 1];
        let n2_i = if k == 0 { 0.0 } else { quadratic[k - 1] };
        let n2_ip1 = if k == CUBIC_DEGREE { 0.0 } else { quadratic[k] };
        let t0 = if dt0 == 0.0 { 0.0 } else { n2_i / dt0 };
        let t1 = if dt1 == 0.0 { 0.0 } else { n2_ip1 / dt1 };
        ders[k] = (CUBIC_DEGREE as f64) * (t0 - t1);
    }
    (span, ders)
}

/// Degree-2 Cox–de Boor values on the same span as a cubic evaluation.
///
/// `values[k] = N_{span-2+k,2}(x)` for `k = 0,1,2`; `values[3]` is unused.
fn quadratic_nonzeros(x: f64, knots: &[f64], span: usize) -> [f64; CUBIC_ORDER] {
    const QUAD_DEGREE: usize = 2;
    const QUAD_ORDER: usize = 3;
    let mut ndu = [[0.0_f64; QUAD_ORDER]; QUAD_ORDER];
    ndu[0][0] = 1.0;
    let mut left = [0.0_f64; QUAD_ORDER];
    let mut right = [0.0_f64; QUAD_ORDER];
    for j in 1..QUAD_ORDER {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom == 0.0 { 0.0 } else { ndu[r][j - 1] / denom };
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    let mut values = [0.0_f64; CUBIC_ORDER];
    for i in 0..QUAD_ORDER {
        values[i] = ndu[i][QUAD_DEGREE];
    }
    values
}

/// Cox–de Boor evaluation of all cubic basis functions at `x` into column-major `out`.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn eval_cubic_bspline(
    x: f64,
    knots: &[f64],
    n_basis: usize,
    out: &mut [f64],
    row: usize,
    nrows: usize,
) {
    let (span, values) = cubic_bspline_nonzeros(x, knots);

    // Zero all bases for this row then write the order nonzeros.
    for b in 0..n_basis {
        out[b * nrows + row] = 0.0;
    }
    let first = span.saturating_sub(CUBIC_DEGREE);
    for i in 0..CUBIC_ORDER {
        let b = first + i;
        if b < n_basis {
            out[b * nrows + row] = values[i];
        }
    }
}

/// Second-difference matrix `D₂` of size `(K-2)×K` with rows `[1, -2, 1]`.
fn second_difference_matrix(n_basis: usize) -> Result<Vec<f64>, StatsError> {
    if n_basis < 3 {
        return Err(StatsError::Shape { message: "second-difference penalty needs n_basis ≥ 3" });
    }
    let rows = n_basis - 2;
    let mut d2 = vec![0.0; rows * n_basis];
    for i in 0..rows {
        d2[i * n_basis + i] = 1.0;
        d2[i * n_basis + i + 1] = -2.0;
        d2[i * n_basis + i + 2] = 1.0;
    }
    Ok(d2)
}

/// Roughness penalty `P = D₂'D₂` (`K×K` row-major).
fn second_difference_penalty(n_basis: usize) -> Result<Vec<f64>, StatsError> {
    let d2 = second_difference_matrix(n_basis)?;
    let rows = n_basis - 2;
    let mut p = vec![0.0; n_basis * n_basis];
    for i in 0..n_basis {
        for j in i..n_basis {
            let mut acc = 0.0;
            for r in 0..rows {
                acc += d2[r * n_basis + i] * d2[r * n_basis + j];
            }
            p[i * n_basis + j] = acc;
            if i != j {
                p[j * n_basis + i] = acc;
            }
        }
    }
    Ok(p)
}

/// The penalty matrix of one smooth (`K×K` row-major) for its knot vector.
fn roughness_penalty(
    kind: RoughnessPenalty,
    knots: Option<&[f64]>,
    n_basis: usize,
) -> Result<Vec<f64>, StatsError> {
    match kind {
        RoughnessPenalty::CoefficientSecondDifference => second_difference_penalty(n_basis),
        RoughnessPenalty::IntegratedSecondDerivative => {
            let knots = knots.ok_or(StatsError::Shape {
                message: "the integrated second-derivative penalty needs resolved knots",
            })?;
            integrated_second_derivative_penalty(knots, n_basis)
        }
    }
}

/// `h̄³ ∫ B_i''(x) B_j''(x) dx` for the cubic B-splines on `knots`, `K×K` row-major, with `h̄` the
/// mean interior knot spacing.
///
/// The second derivative of `Σ cᵢBᵢ` is the linear spline `Σ dⱼNⱼ` on the knots `t₂ … t_{K+1}`
/// with `dⱼ` the repeated divided differences of `c` (de Boor's derivative formula; a repeated
/// knot contributes a zero coefficient). The integral is then `d'Gd` with `G` the exact Gram
/// matrix of the linear B-splines, so the penalty is a straight-line null space for any knots.
fn integrated_second_derivative_penalty(
    knots: &[f64],
    n_basis: usize,
) -> Result<Vec<f64>, StatsError> {
    let k = n_basis;
    if k < CUBIC_ORDER || knots.len() != k + CUBIC_ORDER {
        return Err(StatsError::Shape { message: "integrated penalty needs n_basis + 4 knots" });
    }
    let t = knots;
    let ratio = |numerator: f64, gap: f64| if gap > 0.0 { numerator / gap } else { 0.0 };
    let rows = k - 2;
    // D maps the cubic coefficients to the coefficients of the second derivative.
    let mut d = vec![0.0; rows * k];
    for j in 0..rows {
        let outer = ratio(2.0, t[j + 4] - t[j + 2]);
        let upper = ratio(3.0, t[j + 5] - t[j + 2]);
        let lower = ratio(3.0, t[j + 4] - t[j + 1]);
        d[j * k + j] = outer * lower;
        d[j * k + j + 1] = -outer * (upper + lower);
        d[j * k + j + 2] = outer * upper;
    }
    // G: Gram matrix of the linear B-splines on t₂ … t_{K+1}.
    let mut gram = vec![0.0; rows * rows];
    for j in 0..rows {
        gram[j * rows + j] = (t[j + 4] - t[j + 2]) / 3.0;
        if j + 1 < rows {
            let off = (t[j + 4] - t[j + 3]) / 6.0;
            gram[j * rows + j + 1] = off;
            gram[(j + 1) * rows + j] = off;
        }
    }
    let spacing = (t[k] - t[CUBIC_DEGREE]) / (k - CUBIC_DEGREE) as f64;
    let scale = spacing.powi(3);
    // P = scale · D'GD.
    let mut gd = vec![0.0; rows * k];
    for r in 0..rows {
        for c in 0..k {
            gd[r * k + c] = (0..rows).map(|m| gram[r * rows + m] * d[m * k + c]).sum();
        }
    }
    let mut p = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let value = scale * (0..rows).map(|r| d[r * k + i] * gd[r * k + j]).sum::<f64>();
            p[i * k + j] = value;
            p[j * k + i] = value;
        }
    }
    Ok(p)
}

/// Sweeps after which auto-λ selection is frozen even if λ is still changing.
const MAX_LAMBDA_RESELECT_SWEEPS: u32 = 20;

/// The λ-independent pieces of one smooth's penalized normal equations
/// `(B'B + λP)β = B'y`, for a (possibly `√w`-scaled) basis `B`.
struct PenalizedSmoother {
    n_basis: usize,
    /// `B'B`, `K×K` row-major.
    xtx: Vec<f64>,
    /// Roughness penalty `P = D₂'D₂`, `K×K` row-major.
    penalty: Vec<f64>,
    /// Demmler–Reinsch form of the pencil `(B'B, P)`, built on first use by GCV selection.
    spectrum: std::cell::OnceCell<Option<PencilSpectrum>>,
}

/// Lower Cholesky factor of `B'B + λP`.
struct PenalizedFactor {
    chol: Vec<f64>,
}

/// Spectral form of the pencil `(B'B, P)` for scoring many λ from one factorization.
///
/// With `A₀ = B'B + P = L₀L₀'` and `M = L₀⁻¹ P L₀⁻ᵀ = U diag(m) U'` (`m ∈ [0, 1]`), the
/// whitened `B'B + λP` is `U diag((1 − mᵢ) + λ mᵢ) U'`, so every λ-dependent quantity of the
/// penalized fit is a diagonal operation in the `U` basis.
struct PencilSpectrum {
    /// `L₀`, row-major lower.
    chol0: Vec<f64>,
    /// Eigenvectors of `M` as columns (row-major `K×K`).
    vectors: Vec<f64>,
    /// Eigenvalues `mᵢ` of `M`, clamped to `[0, 1]`.
    m: Vec<f64>,
}

impl PenalizedSmoother {
    #[cfg(test)]
    fn new(basis: &[f64], nrows: usize, n_basis: usize) -> Result<Self, StatsError> {
        Self::with_penalty(basis, nrows, n_basis, second_difference_penalty(n_basis)?)
    }

    fn with_penalty(
        basis: &[f64],
        nrows: usize,
        n_basis: usize,
        penalty: Vec<f64>,
    ) -> Result<Self, StatsError> {
        let mut xtx = vec![0.0; n_basis * n_basis];
        form_xtx(basis, nrows, n_basis, &mut xtx);
        Ok(Self { n_basis, xtx, penalty, spectrum: std::cell::OnceCell::new() })
    }

    /// Cholesky factor of `B'B + λP`.
    fn factor(&self, lambda: f64) -> Result<PenalizedFactor, StatsError> {
        let mut penalized = self.xtx.clone();
        if lambda != 0.0 {
            for (g, p) in penalized.iter_mut().zip(&self.penalty) {
                *g += lambda * p;
            }
        }
        crate::gram::cholesky_spd(&penalized, self.n_basis)
            .map(|chol| PenalizedFactor { chol })
            .ok_or_else(|| StatsError::Backend("GAM: singular B'B+λP".into()))
    }

    /// `B'y` for a (scaled) basis.
    fn rhs(&self, basis: &[f64], nrows: usize, y: &[f64]) -> Vec<f64> {
        (0..self.n_basis)
            .map(|c| basis[c * nrows..(c + 1) * nrows].iter().zip(y).map(|(b, y)| b * y).sum())
            .collect()
    }

    /// `β = (B'B + λP)⁻¹ B'y` by a Cholesky solve against the factor (no explicit inverse).
    fn coefficients(
        &self,
        factor: &PenalizedFactor,
        basis: &[f64],
        nrows: usize,
        y: &[f64],
    ) -> Vec<f64> {
        let k = self.n_basis;
        let rhs = self.rhs(basis, nrows, y);
        let mut y_scratch = vec![0.0; k];
        let mut beta = vec![0.0; k];
        antecedent_kernels::chol_solve_into(&factor.chol, k, &rhs, &mut y_scratch, &mut beta);
        beta
    }

    /// `tr((B'B + λP)⁻¹ B'B)` — the trace of the uncentered smoother — from the factor,
    /// one triangular solve per column of `B'B`.
    fn edf(&self, factor: &PenalizedFactor) -> f64 {
        let k = self.n_basis;
        let mut column = vec![0.0; k];
        let mut y_scratch = vec![0.0; k];
        let mut solved = vec![0.0; k];
        let mut edf = 0.0;
        for j in 0..k {
            for i in 0..k {
                column[i] = self.xtx[i * k + j];
            }
            antecedent_kernels::chol_solve_into(
                &factor.chol,
                k,
                &column,
                &mut y_scratch,
                &mut solved,
            );
            edf += solved[j];
        }
        edf
    }

    /// The spectral form of the pencil, or `None` when `B'B + P` or its eigendecomposition
    /// is not usable (selection then scores each λ from its own factorization).
    fn spectrum(&self) -> Option<&PencilSpectrum> {
        self.spectrum.get_or_init(|| self.build_spectrum()).as_ref()
    }

    fn build_spectrum(&self) -> Option<PencilSpectrum> {
        let k = self.n_basis;
        let mut a0 = self.xtx.clone();
        for (g, p) in a0.iter_mut().zip(&self.penalty) {
            *g += p;
        }
        let chol0 = crate::gram::cholesky_spd(&a0, k)?;
        // W = L₀⁻¹ P (column by column), then M = L₀⁻¹ Wᵀ.
        let mut w = vec![0.0; k * k];
        let mut column = vec![0.0; k];
        let mut solved = vec![0.0; k];
        for j in 0..k {
            for i in 0..k {
                column[i] = self.penalty[i * k + j];
            }
            antecedent_kernels::forward_solve_into(&chol0, k, &column, &mut solved);
            for i in 0..k {
                w[i * k + j] = solved[i];
            }
        }
        let mut m = vec![0.0; k * k];
        for j in 0..k {
            for i in 0..k {
                column[i] = w[j * k + i];
            }
            antecedent_kernels::forward_solve_into(&chol0, k, &column, &mut solved);
            for i in 0..k {
                m[i * k + j] = solved[i];
            }
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let mean = 0.5 * (m[i * k + j] + m[j * k + i]);
                m[i * k + j] = mean;
                m[j * k + i] = mean;
            }
        }
        let (values, vectors) = crate::covariance::symmetric_eigen(&m, k).ok()?;
        Some(PencilSpectrum {
            chol0,
            vectors,
            m: values.into_iter().map(|v| v.clamp(0.0, 1.0)).collect(),
        })
    }
}

impl PencilSpectrum {
    /// `Uᵀ L₀⁻¹ v` for a `K`-vector `v`.
    fn to_eigenbasis(&self, v: &[f64]) -> Vec<f64> {
        let k = self.m.len();
        let mut whitened = vec![0.0; k];
        antecedent_kernels::forward_solve_into(&self.chol0, k, v, &mut whitened);
        (0..k).map(|c| (0..k).map(|r| self.vectors[r * k + c] * whitened[r]).sum()).collect()
    }

    /// `β = L₀⁻ᵀ U t` for eigenbasis coefficients `t`.
    fn coefficients(&self, t: &[f64]) -> Vec<f64> {
        let k = self.m.len();
        let ut: Vec<f64> =
            (0..k).map(|r| (0..k).map(|c| self.vectors[r * k + c] * t[c]).sum()).collect();
        let mut beta = vec![0.0; k];
        antecedent_kernels::back_solve_into(&self.chol0, k, &ut, &mut beta);
        beta
    }
}

const GCV_LAMBDA_GRID: [f64; 25] = [
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0,
    100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6,
];

/// Residual sums of squares below this fraction of `y'y` are recomputed from the residual
/// vector: the moment form `y'y − 2β'B'y + β'B'Bβ` has cancelled too far to be trusted.
const GCV_MOMENT_RSS_FLOOR: f64 = 1e-6;

/// Grid λ minimizing the centered-smoother GCV, and whether the winner is an edge of the
/// grid (the optimum may then lie outside the searched range).
///
/// `basis` and `y` are the `√w`-scaled design and response; `sqrt_w` the `√w` vector.
/// One factorization of the pencil `(B'B, P)` scores every grid λ in `O(K²)` (`O(K)` for
/// the fit and trace, `O(K²)` for the back-transform when the residual has to be rebuilt).
fn select_lambda_gcv(
    smoother: &PenalizedSmoother,
    basis: &[f64],
    nrows: usize,
    y: &[f64],
    sqrt_w: &[f64],
) -> Result<(f64, bool), StatsError> {
    let mut best_index = 0usize;
    let mut best_gcv = f64::INFINITY;
    if let Some(spectrum) = smoother.spectrum() {
        let scores = spectral_gcv_scores(smoother, spectrum, basis, nrows, y, sqrt_w);
        for (index, gcv) in scores.into_iter().enumerate() {
            if gcv < best_gcv {
                best_gcv = gcv;
                best_index = index;
            }
        }
    } else {
        for (index, &lambda) in GCV_LAMBDA_GRID.iter().enumerate() {
            let factor = smoother.factor(lambda)?;
            let gcv = centered_smooth_gcv(smoother, &factor, basis, nrows, y, sqrt_w);
            if gcv < best_gcv {
                best_gcv = gcv;
                best_index = index;
            }
        }
    }
    Ok((GCV_LAMBDA_GRID[best_index], best_index == 0 || best_index == GCV_LAMBDA_GRID.len() - 1))
}

/// Centered GCV at every grid λ from the spectral form of the pencil.
fn spectral_gcv_scores(
    smoother: &PenalizedSmoother,
    spectrum: &PencilSpectrum,
    basis: &[f64],
    nrows: usize,
    y: &[f64],
    sqrt_w: &[f64],
) -> Vec<f64> {
    let k = smoother.n_basis;
    let n = nrows as f64;
    let rhs = smoother.rhs(basis, nrows, y);
    let centering = smoother.rhs(basis, nrows, sqrt_w);
    let s = spectrum.to_eigenbasis(&rhs);
    let g = spectrum.to_eigenbasis(&centering);
    let yty: f64 = y.iter().map(|v| v * v).sum();
    let sqrt_w_y: f64 = sqrt_w.iter().zip(y).map(|(w, v)| w * v).sum();
    let weight_total: f64 = sqrt_w.iter().map(|w| w * w).sum();
    let mut t = vec![0.0; k];
    GCV_LAMBDA_GRID
        .iter()
        .map(|&lambda| {
            let mut fit_cross = 0.0; // β'B'y
            let mut fit_quad = 0.0; // β'B'Bβ
            let mut centering_dot = 0.0; // (B'√w)'β
            let mut trace = 0.0; // tr(S₁)
            for i in 0..k {
                let gram = 1.0 - spectrum.m[i];
                let denominator = gram + lambda * spectrum.m[i];
                let ti = s[i] / denominator;
                t[i] = ti;
                fit_cross += ti * s[i];
                fit_quad += gram * ti * ti;
                centering_dot += g[i] * ti;
                trace += gram / denominator;
            }
            let mean = centering_dot / n;
            let moment_rss = yty - 2.0 * fit_cross
                + fit_quad
                + 2.0 * mean * (sqrt_w_y - centering_dot)
                + mean * mean * weight_total;
            let edf = (trace - 1.0).max(0.0);
            let denom = (n - edf).max(1e-8);
            let rss = if moment_rss.is_finite() && moment_rss > GCV_MOMENT_RSS_FLOOR * yty {
                moment_rss
            } else {
                let beta = spectrum.coefficients(&t);
                centered_rss(&beta, k, basis, nrows, y, sqrt_w)
            };
            n * rss / (denom * denom)
        })
        .collect()
}

/// Centered residual sum of squares of `Bβ` against `y` (both `√w`-scaled).
fn centered_rss(
    beta: &[f64],
    n_basis: usize,
    basis: &[f64],
    nrows: usize,
    y: &[f64],
    sqrt_w: &[f64],
) -> f64 {
    let mut scaled_pred = vec![0.0; nrows];
    for r in 0..nrows {
        let mut pred = 0.0;
        for b in 0..n_basis {
            pred += basis[b * nrows + r] * beta[b];
        }
        scaled_pred[r] = pred;
    }
    let weighted_mean: f64 =
        scaled_pred.iter().zip(sqrt_w).map(|(p, w)| p * w).sum::<f64>() / nrows as f64;
    let mut rss = 0.0;
    for r in 0..nrows {
        let e = y[r] - scaled_pred[r] + weighted_mean * sqrt_w[r];
        rss += e * e;
    }
    rss
}

/// GCV of the smoother actually applied in backfitting: centered `Bβ` with
/// `edf = tr(S₁) − 1`. Uncentered RSS / `tr(S₁)` scores a different operator.
///
/// Inputs are in `√w`-scaled coordinates (`basis = √w B`, `y = √w y₀`), so with
/// `fᵣ = (Bβ)ᵣ` the weighted-centered residual `√wᵣ (y₀ᵣ − fᵣ + f̄_w)` is
/// `yᵣ − (√w B β)ᵣ + f̄_w √wᵣ`, where `f̄_w = Σ wᵣ fᵣ / n = Σ √wᵣ (√w B β)ᵣ / n`. The
/// centering constant multiplies `√w`, it is not subtracted from the scaled prediction —
/// the two coincide only when every weight is equal.
fn centered_smooth_gcv(
    smoother: &PenalizedSmoother,
    factor: &PenalizedFactor,
    basis: &[f64],
    nrows: usize,
    y: &[f64],
    sqrt_w: &[f64],
) -> f64 {
    let beta = smoother.coefficients(factor, basis, nrows, y);
    let rss = centered_rss(&beta, smoother.n_basis, basis, nrows, y, sqrt_w);
    let edf = (smoother.edf(factor) - 1.0).max(0.0);
    let denom = (nrows as f64 - edf).max(1e-8);
    (nrows as f64) * rss / (denom * denom)
}

/// `β = (B'B + λP)⁻¹ B'y` (test helper over [`PenalizedSmoother`]).
#[cfg(test)]
fn roughness_basis_solve(
    basis: &[f64],
    nrows: usize,
    n_basis: usize,
    y: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, StatsError> {
    let smoother = PenalizedSmoother::new(basis, nrows, n_basis)?;
    let factor = smoother.factor(lambda)?;
    Ok(smoother.coefficients(&factor, basis, nrows, y))
}

/// `tr((B'B + λP)⁻¹ B'B)` (test helper over [`PenalizedSmoother`]).
#[cfg(test)]
fn roughness_edf(
    basis: &[f64],
    nrows: usize,
    n_basis: usize,
    lambda: f64,
) -> Result<f64, StatsError> {
    let smoother = PenalizedSmoother::new(basis, nrows, n_basis)?;
    Ok(smoother.edf(&smoother.factor(lambda)?))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert exactly representable basis values and copied inputs"
)]
mod tests {
    #[test]
    fn review_custom_unclamped_knots_preserve_boundary_basis() {
        let knots: Vec<_> = (0..12).map(f64::from).collect();
        let (_, actual) = cubic_bspline_nonzeros(8.0, &knots);
        for (value, expected) in actual.iter().zip([0.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]) {
            assert!((value - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn review_spline_basis_and_derivatives_respect_units() {
        let x = linspace(30, -1.0, 1.0);
        let (_, knots) = expand_bspline(&x, 8, None).unwrap();
        for scale in [1e-20, 1.0, 1e20] {
            let scaled_x: Vec<_> = x.iter().map(|v| v * scale).collect();
            let (_, scaled_knots) = expand_bspline(&scaled_x, 8, None).unwrap();
            for point in [-1.0, -0.3, 0.4, 1.0, 2.0] {
                let (_, expected) = cubic_bspline_nonzeros(point, &knots);
                let (_, actual) = cubic_bspline_nonzeros(point * scale, &scaled_knots);
                let (_, expected_deriv) = cubic_bspline_deriv_nonzeros(point, &knots);
                let (_, actual_deriv) = cubic_bspline_deriv_nonzeros(point * scale, &scaled_knots);
                for i in 0..4 {
                    assert!(
                        (actual[i] - expected[i]).abs() < 1e-12,
                        "basis scale={scale} point={point}"
                    );
                    assert!(
                        (actual_deriv[i] * scale - expected_deriv[i]).abs() < 1e-11,
                        "derivative scale={scale} point={point}"
                    );
                }
            }
        }
    }

    use super::*;
    use crate::faer_backend::FaerBackend;

    /// A design expanded once and refit under several weight vectors is the
    /// per-call expansion bit for bit: same quantile knots, same bases, same
    /// coefficients, fitted values and edf.
    #[test]
    fn design_refits_are_bit_identical_to_per_call_expansion() {
        let nrows = 80usize;
        let x0: Vec<f64> = linspace(nrows, -1.0, 2.0);
        let x1: Vec<f64> = (0..nrows).map(|i| ((i * 37) % 11) as f64 / 3.0).collect();
        let x: Vec<f64> = x0.iter().chain(&x1).copied().collect();
        let y: Vec<f64> = x0
            .iter()
            .zip(&x1)
            .enumerate()
            .map(|(i, (&a, &b))| (2.0 * a).sin() + 0.3 * b * b + 0.05 * (i % 5) as f64)
            .collect();
        let specs = [SmoothSpec::new(0, 6, 0.5), SmoothSpec::new(1, 5, 2.0)];
        let options = GamOptions { max_iter: 500, tol: 1e-6 };
        let design = AdditiveDesign::expand(&x, nrows, 2, &specs).unwrap();
        assert_eq!(design.nrows(), nrows);
        assert!(design.specs().iter().all(|s| s.knots.is_some()));
        let weight_sets: [Option<Vec<f64>>; 3] = [
            None,
            Some((0..nrows).map(|i| 0.2 + (i % 4) as f64).collect()),
            Some((0..nrows).map(|i| if i % 3 == 0 { 2.5 } else { 0.4 }).collect()),
        ];
        for weights in &weight_sets {
            let mut ws_a = GamWorkspace::default();
            let mut ws_b = GamWorkspace::default();
            let per_call = fit_gam_weighted(
                &PredictorMatrix::new(&x, nrows, 2),
                &y,
                &specs,
                &options,
                weights.as_deref(),
                &FaerBackend,
                &mut ws_a,
            )
            .unwrap();
            let on_design =
                fit_gam_weighted_design(&design, &y, &options, weights.as_deref(), &mut ws_b)
                    .unwrap();
            assert_eq!(per_call.intercept.to_bits(), on_design.intercept.to_bits());
            assert_eq!(per_call.coefficients, on_design.coefficients);
            assert_eq!(per_call.fitted, on_design.fitted);
            assert_eq!(per_call.residuals, on_design.residuals);
            assert_eq!(per_call.edf_approx.to_bits(), on_design.edf_approx.to_bits());
            assert_eq!(per_call.iterations, on_design.iterations);
            assert_eq!(per_call.smooths, on_design.smooths);
            assert_eq!(per_call.centers, on_design.centers);
        }
    }

    /// `edf_approx` must equal the trace of the smoother actually applied.
    ///
    /// A GAM fit is a linear operator `fitted = H·y`, and effective degrees of freedom *is*
    /// `tr(H)`. That makes this measurable without reimplementing anything: perturb `y[i]`,
    /// refit, and read `∂fitted[i]/∂y[i]` off the difference. This is an independent check —
    /// the conformance fixture's `edf` field is regenerated from this same code path, so it
    /// cannot arbitrate its own correctness.
    ///
    /// The smooths are mean-centered (the intercept absorbs the level), and because the
    /// B-spline basis is a partition of unity with constants in the penalty's null space,
    /// each uncentered `tr(S₁)` already includes that constant. Counting the intercept
    /// separately on top of it inflates the total by one per smooth term.
    #[test]
    fn edf_approx_matches_finite_difference_operator_trace() {
        fn fit_for(y: &[f64], x: &[f64], nrows: usize, specs: &[SmoothSpec]) -> GamFit {
            let mut ws = GamWorkspace::default();
            fit_gam(
                &PredictorMatrix::new(x, nrows, 1),
                y,
                specs,
                &GamOptions { max_iter: 5000, tol: 1e-12 },
                &FaerBackend,
                &mut ws,
            )
            .unwrap()
        }

        let nrows = 60usize;
        let x: Vec<f64> = linspace(nrows, 0.0, 1.0);
        let y: Vec<f64> =
            x.iter().enumerate().map(|(i, &v)| (3.0 * v).sin() + 0.05 * (i % 7) as f64).collect();

        for n_basis in [6usize, 10] {
            for lambda in [0.01f64, 1.0, 25.0] {
                let specs = [SmoothSpec::new(0, n_basis, lambda)];
                let base = fit_for(&y, &x, nrows, &specs);

                // tr(H) = Σ_i ∂fitted[i]/∂y[i], by central difference.
                let h = 1e-6;
                let mut trace = 0.0;
                for i in 0..nrows {
                    let mut up = y.clone();
                    up[i] += h;
                    let mut down = y.clone();
                    down[i] -= h;
                    let fu = fit_for(&up, &x, nrows, &specs);
                    let fd = fit_for(&down, &x, nrows, &specs);
                    trace += (fu.fitted[i] - fd.fitted[i]) / (2.0 * h);
                }

                assert!(
                    (base.edf_approx - trace).abs() < 1e-4,
                    "n_basis={n_basis} lambda={lambda}: edf_approx={} but measured tr(H)={trace}",
                    base.edf_approx
                );
            }
        }
    }

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
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &backend,
            &mut ws,
        )
        .unwrap();
        assert!(fit.converged, "iterations={}", fit.iterations);
        let ss_res: f64 = fit.residuals.iter().map(|e| e * e).sum();
        let y_bar = y.iter().sum::<f64>() / y.len() as f64;
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
    fn predict_row_matches_predict_gam_bit_for_bit() {
        let n = 200usize;
        let x1 = linspace(n, 0.0, 1.0);
        let x2: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % n) as f64 / n as f64 - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + (3.0 * x1[i]).sin() + x2[i] * x2[i] + 0.01 * (i % 5) as f64)
            .collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1.clone(), x2.clone()]);
        let specs = [SmoothSpec::new(0, 8, 0.5), SmoothSpec::new(1, 6, 1.0)];
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        // Probe on and off the training support, including the clamped edges.
        for &(a, b) in
            &[(0.0, -0.5), (0.31, 0.12), (1.0, 0.49), (-0.2, 0.0), (1.3, -0.7), (0.777, 0.123)]
        {
            let row = [a, b];
            let batch = predict_gam(&fit, &row, 1, 2).unwrap()[0];
            let single = fit.predict_row(&row).unwrap();
            assert!(
                batch.to_bits() == single.to_bits(),
                "predict_row diverged at ({a},{b}): batch={batch:?} single={single:?}"
            );
            // Additivity: intercept + Σ_j smooth_partial(j, x_j) is the same prediction.
            let additive = fit.intercept
                + fit.smooth_partial(0, a).unwrap()
                + fit.smooth_partial(1, b).unwrap();
            assert!(
                (additive - single).abs() <= 1e-12 * single.abs().max(1.0),
                "smooth_partial decomposition diverged at ({a},{b})"
            );
        }
        assert_eq!(fit.smooth_for_raw_col(0), Some(0));
        assert_eq!(fit.smooth_for_raw_col(1), Some(1));
        assert_eq!(fit.smooth_for_raw_col(2), None);
    }

    #[test]
    fn smooth_derivative_is_zero_outside_the_clamped_range() {
        let n = 80usize;
        let x1 = linspace(n, -1.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| 2.0 * v + 0.1 * v * v).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &[SmoothSpec::new(0, 8, 0.2)],
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        let knots = fit.smooths[0].knots.as_ref();
        let left = knots[CUBIC_DEGREE];
        let right = knots[knots.len() - CUBIC_ORDER];
        assert_eq!(fit.smooth_derivative(0, left - 1.0).unwrap(), 0.0);
        assert_eq!(fit.smooth_derivative(0, right + 2.0).unwrap(), 0.0);
        let h = (right - left) * 1e-6;
        for (endpoint, interior) in [(left, left + h), (right, right - h)] {
            let difference = (fit.smooth_partial(0, endpoint).unwrap()
                - fit.smooth_partial(0, interior).unwrap())
                / (endpoint - interior);
            let derivative = fit.smooth_derivative(0, endpoint).unwrap();
            assert!((derivative - difference).abs() < 1e-4);
            let basis = fit.smooth_basis_derivative(0, endpoint).unwrap();
            let basis_derivative: f64 =
                basis.iter().zip(&fit.coefficients).map(|(a, b)| a * b).sum();
            assert!((basis_derivative - derivative).abs() < 1e-12);
        }
    }

    #[test]
    fn smooth_derivative_matches_finite_difference_of_smooth_partial() {
        let n = 120usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> =
            x1.iter().enumerate().map(|(i, &v)| (4.0 * v).sin() + 0.02 * (i % 5) as f64).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &[SmoothSpec::new(0, 10, 0.3)],
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        let knots = fit.smooths[0].knots.as_ref();
        let left = knots[CUBIC_DEGREE];
        let right = knots[knots.len() - CUBIC_ORDER];
        let h = 1e-6;
        for x0 in [0.12, 0.37, 0.5, 0.64, 0.81] {
            assert!(x0 > left + 10.0 * h && x0 < right - 10.0 * h);
            let analytic = fit.smooth_derivative(0, x0).unwrap();
            let fd = (fit.smooth_partial(0, x0 + h).unwrap()
                - fit.smooth_partial(0, x0 - h).unwrap())
                / (2.0 * h);
            assert!((analytic - fd).abs() < 1e-5, "x={x0}: analytic={analytic} fd={fd}");
        }
    }

    #[test]
    fn smooth_basis_derivative_is_the_linear_map_behind_smooth_derivative() {
        let n = 120usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| (3.0 * v).cos()).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &[SmoothSpec::new(0, 9, 0.1)],
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        let right = fit.smooths[0].knots[fit.smooths[0].knots.len() - CUBIC_ORDER];
        for x0 in [0.05, 0.3, 0.5, 0.77, right, right + 1.0, -1.0] {
            let basis = fit.smooth_basis_derivative(0, x0).unwrap();
            assert_eq!(basis.len(), 9);
            let dot: f64 = basis.iter().zip(&fit.coefficients).map(|(b, c)| b * c).sum();
            let direct = fit.smooth_derivative(0, x0).unwrap();
            assert!((dot - direct).abs() < 1e-12, "x={x0}: {dot} vs {direct}");
            // Partition of unity: the basis derivatives sum to zero.
            assert!(basis.iter().sum::<f64>().abs() < 1e-9, "x={x0}");
        }
        assert!(fit.smooth_basis_derivative(1, 0.5).is_err());
    }

    #[test]
    fn predict_row_matches_training_fitted_values() {
        // A training-row prediction must reproduce the cached in-sample fitted
        // value exactly; the pseudo-outcome hoist in antecedent-estimate relies
        // on `fitted[position]` standing in for a full re-prediction.
        let n = 150usize;
        let x1 = linspace(n, -2.0, 2.0);
        let x2: Vec<f64> = (0..n).map(|i| ((i * 13 + 1) % n) as f64 / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 0.3 * x1[i] - 0.8 * x2[i] + (x1[i] * 1.7).cos()).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1.clone(), x2.clone()]);
        let specs = [SmoothSpec::new(0, 7, 0.2), SmoothSpec::new(1, 5, 0.7)];
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        for r in 0..n {
            let row = [x1[r], x2[r]];
            let pred = fit.predict_row(&row).unwrap();
            assert!(
                pred.to_bits() == fit.fitted[r].to_bits(),
                "row {r}: predict_row={pred:?} fitted={:?}",
                fit.fitted[r]
            );
        }
    }

    #[test]
    fn high_lambda_smooth_approaches_linear_null_space() {
        // Large second-difference λ → null space of D₂ (degree ≤ 1). For a nearly
        // constant signal the centered smooth stays near zero.
        let n = 80usize;
        let x1 = linspace(n, -1.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| 3.0 + 0.01 * v).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let specs = [SmoothSpec::new(0, 6, 1e6)];
        let backend = FaerBackend;
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &backend,
            &mut ws,
        )
        .unwrap();
        assert!((fit.intercept - 3.0).abs() < 0.05);
        let max_abs_smooth: f64 =
            fit.fitted.iter().map(|&f| (f - fit.intercept).abs()).fold(0.0, f64::max);
        assert!(max_abs_smooth < 0.05, "max_abs_smooth={max_abs_smooth}");
    }

    #[test]
    fn linear_signal_has_near_zero_second_difference_penalty() {
        // Null space of D₂: coefficient sequences that are linear in the basis index.
        for k in [4usize, 6, 10] {
            let p = second_difference_penalty(k).unwrap();
            let beta: Vec<f64> = (0..k).map(|i| 2.0 + 0.75 * i as f64).collect();
            let mut quad = 0.0;
            for i in 0..k {
                for j in 0..k {
                    quad += beta[i] * p[i * k + j] * beta[j];
                }
            }
            assert!(quad.abs() < 1e-12, "K={k} β'Pβ={quad}");
        }
        // Fitted values on an exact line also have tiny discrete curvature.
        let n = 120usize;
        let x1 = linspace(n, -1.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| 1.5 + 2.0 * v).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1.clone()]);
        let specs = [SmoothSpec::new(0, 8, 1e-4)];
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        let mut max_d2 = 0.0_f64;
        for r in 1..n - 1 {
            let d2 = fit.fitted[r - 1] - 2.0 * fit.fitted[r] + fit.fitted[r + 1];
            max_d2 = max_d2.max(d2.abs());
        }
        assert!(max_d2 < 1e-4, "max discrete curvature={max_d2}");
    }

    #[test]
    fn increasing_lambda_monotonically_reduces_edf() {
        let n = 100usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> =
            x1.iter().map(|&v| (2.0 * std::f64::consts::PI * v).sin() + 0.05 * v).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let mut ws = GamWorkspace::default();
        let mut prev = f64::INFINITY;
        for &lambda in &[0.01, 0.1, 1.0, 10.0, 100.0, 1e4] {
            let specs = [SmoothSpec::new(0, 10, lambda)];
            let fit = fit_gam(
                &PredictorMatrix::new(&x, nrows, ncols),
                &y,
                &specs,
                &GamOptions::default(),
                &FaerBackend,
                &mut ws,
            )
            .unwrap();
            assert!(
                fit.edf_approx <= prev + 1e-9,
                "edf rose with λ={lambda}: {} > {prev}",
                fit.edf_approx
            );
            prev = fit.edf_approx;
        }
    }

    #[test]
    fn roughness_penalty_beats_identity_ridge_on_curved_signal() {
        // Compare RSS under equal λ: D₂'D₂ should recover a smooth curve better than λI
        // when both are applied to the same B-spline expansion of a noisy sinusoid.
        let n = 200usize;
        let x1 = linspace(n, 0.0, 1.0);
        let mut y = Vec::with_capacity(n);
        for (i, &v) in x1.iter().enumerate() {
            let noise = 0.15 * (((i * 17) % 10) as f64 / 10.0 - 0.5);
            y.push((2.0 * std::f64::consts::PI * v).sin() + noise);
        }
        let (basis, _) = expand_bspline(&x1, 12, None).unwrap();
        let mut gram = vec![0.0; 12 * 12];
        let mut rhs = [0.0; 12];
        let beta_r = roughness_basis_solve(&basis, n, 12, &y, 1.0).unwrap();
        // Identity ridge baseline (local to this test).
        form_xtx(&basis, n, 12, &mut gram);
        for c in 0..12 {
            gram[c * 12 + c] += 1.0;
        }
        for c in 0..12 {
            let mut s = 0.0;
            for r in 0..n {
                s += basis[c * n + r] * y[r];
            }
            rhs[c] = s;
        }
        let inv = crate::gram::invert_square(&gram[..144], 12).unwrap();
        let mut beta_i = [0.0; 12];
        for i in 0..12 {
            let mut s = 0.0;
            for j in 0..12 {
                s += inv[i * 12 + j] * rhs[j];
            }
            beta_i[i] = s;
        }
        let mut rss_r = 0.0;
        let mut rss_i = 0.0;
        let mut curv_err_r = 0.0;
        let mut curv_err_i = 0.0;
        for r in 0..n {
            let truth = (2.0 * std::f64::consts::PI * x1[r]).sin();
            let mut pr = 0.0;
            let mut pi = 0.0;
            for b in 0..12 {
                pr += basis[b * n + r] * beta_r[b];
                pi += basis[b * n + r] * beta_i[b];
            }
            // Center both for fair comparison with mean-zero sinusoid.
            // (absolute level absorbed by intercept in GAM; here compare shape RSS to truth.)
            rss_r += (pr - truth) * (pr - truth);
            rss_i += (pi - truth) * (pi - truth);
            curv_err_r += (pr - truth).abs();
            curv_err_i += (pi - truth).abs();
        }
        assert!(rss_r < rss_i, "roughness RSS={rss_r} should beat identity ridge RSS={rss_i}");
        assert!(curv_err_r < curv_err_i);
    }

    /// Marsden's identity: the coefficients `c_i = (t_{i+1}+t_{i+2}+t_{i+3})/3` (the Greville
    /// abscissae) reproduce `x`, and `e2(t_{i+1..i+3})` reproduces `x²`.
    fn marsden_coefficients(knots: &[f64], n_basis: usize) -> (Vec<f64>, Vec<f64>) {
        let linear =
            (0..n_basis).map(|i| (knots[i + 1] + knots[i + 2] + knots[i + 3]) / 3.0).collect();
        let quadratic = (0..n_basis)
            .map(|i| {
                let (a, b, c) = (knots[i + 1], knots[i + 2], knots[i + 3]);
                (a * b + a * c + b * c) / 3.0
            })
            .collect();
        (linear, quadratic)
    }

    fn quadratic_form(p: &[f64], c: &[f64]) -> f64 {
        let k = c.len();
        (0..k).map(|i| (0..k).map(|j| c[i] * p[i * k + j] * c[j]).sum::<f64>()).sum()
    }

    /// A straight line has no integrated curvature on knots that are not equally spaced, where
    /// the coefficient second difference of the same line is not zero.
    #[test]
    fn integrated_penalty_is_zero_on_a_line_where_the_second_difference_is_not() {
        let x: Vec<f64> = (0..60).map(|i| (f64::from(i) / 59.0).powi(2) * 3.0 - 1.0).collect();
        let k = 6;
        let (_, knots) = expand_bspline(&x, k, None).unwrap();
        let (line, _) = marsden_coefficients(&knots, k);
        let integrated = integrated_second_derivative_penalty(&knots, k).unwrap();
        let difference = second_difference_penalty(k).unwrap();
        let scale = line.iter().map(|v| v * v).sum::<f64>();
        assert!(quadratic_form(&integrated, &line).abs() < 1e-12 * scale);
        assert!(quadratic_form(&difference, &line) > 1e-4 * scale);
        // The constant is in the null space too.
        let ones = vec![1.0; k];
        assert!(quadratic_form(&integrated, &ones).abs() < 1e-12 * scale);
    }

    /// `s = x²` has `s'' = 2`, so the roughness is `4 (right - left)` times the spacing cube.
    #[test]
    fn integrated_penalty_equals_the_closed_form_curvature_of_a_parabola() {
        let x: Vec<f64> = (0..80).map(|i| (f64::from(i) / 79.0).powi(3) * 5.0 + 2.0).collect();
        for k in [5, 6, 9] {
            let (_, knots) = expand_bspline(&x, k, None).unwrap();
            let (_, parabola) = marsden_coefficients(&knots, k);
            let p = integrated_second_derivative_penalty(&knots, k).unwrap();
            let (left, right) = (knots[CUBIC_DEGREE], knots[k]);
            let spacing = (right - left) / (k - CUBIC_DEGREE) as f64;
            let expected = 4.0 * (right - left) * spacing.powi(3);
            let actual = quadratic_form(&p, &parabola);
            assert!((actual - expected).abs() < 1e-10 * expected, "{k}: {actual} vs {expected}");
        }
    }

    /// Rescaling the predictor leaves the penalty matrix unchanged.
    #[test]
    fn integrated_penalty_is_invariant_to_the_predictor_scale() {
        let x: Vec<f64> = (0..50).map(|i| (f64::from(i) / 49.0).powi(2)).collect();
        let k = 7;
        let (_, knots) = expand_bspline(&x, k, None).unwrap();
        let scaled: Vec<f64> = knots.iter().map(|t| 250.0 * t + 1000.0).collect();
        let a = integrated_second_derivative_penalty(&knots, k).unwrap();
        let b = integrated_second_derivative_penalty(&scaled, k).unwrap();
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-9 * (1.0 + u.abs()), "{u} vs {v}");
        }
    }

    /// A penalised fit of an exactly linear signal on uneven quantile knots keeps its slope under
    /// the integrated penalty and is shrunk under the coefficient second difference.
    #[test]
    fn a_penalised_fit_of_a_line_on_uneven_knots_keeps_its_slope_under_the_integrated_penalty() {
        let n = 120;
        let x: Vec<f64> = (0..n).map(|i| (f64::from(i) / f64::from(n - 1)).powi(3) * 4.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v).collect();
        let slope = |penalty: RoughnessPenalty| {
            let spec = [SmoothSpec::new(0, 6, 1.0).with_penalty(penalty)];
            let fit = fit_gam(
                &PredictorMatrix::new(&x, n as usize, 1),
                &y,
                &spec,
                &GamOptions { max_iter: 500, tol: 1e-12 },
                &FaerBackend,
                &mut GamWorkspace::default(),
            )
            .unwrap();
            let mid = 2.0;
            fit.smooth_derivative(0, mid).unwrap()
        };
        let integrated = slope(RoughnessPenalty::IntegratedSecondDerivative);
        let difference = slope(RoughnessPenalty::CoefficientSecondDifference);
        assert!((integrated - 2.0).abs() < 1e-8, "{integrated}");
        assert!((difference - 2.0).abs() > 1e-3, "{difference}");
    }

    #[test]
    fn second_difference_penalty_matches_direct_d2t_d2() {
        for k in [4usize, 6, 8, 12] {
            let p = second_difference_penalty(k).unwrap();
            let d2 = second_difference_matrix(k).unwrap();
            let rows = k - 2;
            for i in 0..k {
                for j in 0..k {
                    let mut acc = 0.0;
                    for r in 0..rows {
                        acc += d2[r * k + i] * d2[r * k + j];
                    }
                    assert!(
                        (p[i * k + j] - acc).abs() < 1e-14,
                        "P mismatch at ({i},{j}) for K={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn intercept_remains_unpenalized_under_large_lambda() {
        let n = 60usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> =
            x1.iter().map(|&v| 5.0 + 0.2 * (2.0 * std::f64::consts::PI * v).sin()).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let specs = [SmoothSpec::new(0, 8, 1e8)];
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        // Mean level lives in the intercept; large roughness λ must not shrink it to 0.
        assert!((fit.intercept - 5.0).abs() < 0.15, "intercept={}", fit.intercept);
    }

    #[test]
    fn auto_lambda_gcv_selects_finite_penalty() {
        let n = 100usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, &v)| (2.0 * std::f64::consts::PI * v).sin() + 0.05 * (i as f64).sin())
            .collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let specs = [SmoothSpec::auto(0, 10)];
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        assert!(fit.smooths[0].lambda.is_finite() && fit.smooths[0].lambda > 0.0);
        assert!(fit.converged);
    }

    #[test]
    fn review_gcv_minimizes_centered_smoother_score() {
        let n = 80usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, &v)| (2.0 * std::f64::consts::PI * v).sin() + 0.2 * (i as f64).sin())
            .collect();
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = y.iter().map(|v| v - y_mean).collect();
        let (basis, _) = expand_bspline(&x1, 10, None).unwrap();
        let ones = vec![1.0; n];
        let smoother = PenalizedSmoother::new(&basis, n, 10).unwrap();
        let (chosen, _) = select_lambda_gcv(&smoother, &basis, n, &centered, &ones).unwrap();
        let mut best_lambda = GCV_LAMBDA_GRID[0];
        let mut best_gcv = f64::INFINITY;
        let mut uncentered_winner = GCV_LAMBDA_GRID[0];
        let mut best_uncentered_gcv = f64::INFINITY;
        for &lambda in &GCV_LAMBDA_GRID {
            let inverse = smoother.factor(lambda).unwrap();
            let centered_gcv =
                centered_smooth_gcv(&smoother, &inverse, &basis, n, &centered, &ones);
            if centered_gcv < best_gcv {
                best_gcv = centered_gcv;
                best_lambda = lambda;
            }
            let beta = roughness_basis_solve(&basis, n, 10, &centered, lambda).unwrap();
            let mut rss = 0.0;
            for r in 0..n {
                let mut pred = 0.0;
                for b in 0..10 {
                    pred += basis[b * n + r] * beta[b];
                }
                let e = centered[r] - pred;
                rss += e * e;
            }
            let edf = roughness_edf(&basis, n, 10, lambda).unwrap();
            let denom = (n as f64 - edf).max(1e-8);
            let uncentered_gcv = (n as f64) * rss / (denom * denom);
            if uncentered_gcv < best_uncentered_gcv {
                best_uncentered_gcv = uncentered_gcv;
                uncentered_winner = lambda;
            }
        }
        assert_eq!(chosen, best_lambda);
        if uncentered_winner != best_lambda {
            assert_ne!(
                chosen, uncentered_winner,
                "centered GCV must not inherit the uncentered argmin"
            );
        }
    }

    #[test]
    fn weighted_gcv_centers_by_the_weighted_mean_of_the_unscaled_smooth() {
        // Independent truth in original coordinates: with weights w (Σw = n),
        // f = Bβ from the weighted penalized fit, f̄ = Σ w f / n, RSS_w = Σ w (y − f + f̄)²,
        // GCV = n RSS_w / (n − (tr S₁ − 1))². The scaled-coordinate implementation used to
        // subtract the *unweighted* mean of √w·f from the √w-scaled vector, which is the same
        // number only when all weights are equal.
        let n = 12usize;
        let x1 = linspace(n, 0.0, 1.0);
        let raw_w = [0.2, 0.5, 3.0, 0.1, 1.7, 2.2, 0.4, 0.9, 1.3, 0.05, 1.6, 0.05];
        let total: f64 = raw_w.iter().sum();
        let w: Vec<f64> = raw_w.iter().map(|v| v * n as f64 / total).collect();
        let y: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, &v)| (3.0 * v).sin() + 0.3 * ((i * 7) % 5) as f64)
            .collect();
        let (basis, _) = expand_bspline(&x1, 6, None).unwrap();
        let sqrt_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
        let scaled_basis: Vec<f64> =
            basis.iter().enumerate().map(|(i, v)| v * sqrt_w[i % n]).collect();
        let scaled_y: Vec<f64> = y.iter().zip(&sqrt_w).map(|(v, s)| v * s).collect();
        let smoother = PenalizedSmoother::new(&scaled_basis, n, 6).unwrap();
        for lambda in [1e-3, 0.1, 10.0] {
            let inverse = smoother.factor(lambda).unwrap();
            let got =
                centered_smooth_gcv(&smoother, &inverse, &scaled_basis, n, &scaled_y, &sqrt_w);

            let beta = smoother.coefficients(&inverse, &scaled_basis, n, &scaled_y);
            let f: Vec<f64> =
                (0..n).map(|r| (0..6).map(|b| basis[b * n + r] * beta[b]).sum()).collect();
            let f_bar: f64 = f.iter().zip(&w).map(|(f, w)| f * w).sum::<f64>() / n as f64;
            let rss: f64 =
                (0..n).map(|r| w[r] * (y[r] - f[r] + f_bar) * (y[r] - f[r] + f_bar)).sum();
            let edf = smoother.edf(&inverse) - 1.0;
            let want = n as f64 * rss / ((n as f64 - edf) * (n as f64 - edf));
            assert!((got - want).abs() <= 1e-10 * want.abs(), "λ={lambda}: {got} vs {want}");
        }
    }

    #[test]
    fn auto_lambda_is_a_fixed_point_of_gcv_on_the_final_partial_residuals() {
        // Two auto smooths: smooth 0's partial residual initially contains all of smooth 1's
        // signal. λ must be re-selected until each smooth's λ minimizes GCV on its final
        // partial residual (compared by score, robust to near-ties on the coarse grid).
        let n = 160usize;
        let x1 = linspace(n, 0.0, 1.0);
        let x2: Vec<f64> = (0..n).map(|i| ((i * 37 + 11) % n) as f64 / n as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * x1[i]).sin()
                    + 2.0 * (3.0 * std::f64::consts::PI * x2[i]).cos()
                    + 0.05 * (((i * 13) % 7) as f64 - 3.0)
            })
            .collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1.clone(), x2.clone()]);
        let specs = [SmoothSpec::auto(0, 8), SmoothSpec::auto(1, 8)];
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        assert!(fit.converged);
        let cols = [&x1, &x2];
        let ones = vec![1.0; n];
        for j in 0..2 {
            let other = 1 - j;
            let partial: Vec<f64> = (0..n)
                .map(|r| y[r] - fit.intercept - fit.smooth_partial(other, cols[other][r]).unwrap())
                .collect();
            let (basis, _) =
                expand_bspline(cols[j], 8, Some(fit.smooths[j].knots.as_ref())).unwrap();
            let smoother = PenalizedSmoother::new(&basis, n, 8).unwrap();
            let score = |lambda: f64| {
                let inverse = smoother.factor(lambda).unwrap();
                centered_smooth_gcv(&smoother, &inverse, &basis, n, &partial, &ones)
            };
            let best = GCV_LAMBDA_GRID.iter().map(|&l| score(l)).fold(f64::INFINITY, f64::min);
            let chosen = score(fit.smooths[j].lambda);
            assert!(
                chosen <= best * (1.0 + 1e-3),
                "smooth {j}: chosen GCV {chosen} vs best {best}"
            );
        }
    }

    #[test]
    fn gcv_reports_when_the_winner_is_a_grid_edge() {
        // An all-zero response has GCV 0 at every λ; the first grid point wins and is an edge.
        let n = 40usize;
        let x1 = linspace(n, 0.0, 1.0);
        let (basis, _) = expand_bspline(&x1, 6, None).unwrap();
        let smoother = PenalizedSmoother::new(&basis, n, 6).unwrap();
        let ones = vec![1.0; n];
        let (lambda, boundary) =
            select_lambda_gcv(&smoother, &basis, n, &vec![0.0; n], &ones).unwrap();
        assert_eq!(lambda, GCV_LAMBDA_GRID[0]);
        assert!(boundary);
    }

    #[test]
    fn additive_rank_excludes_the_shared_constant_and_response_must_be_finite() {
        let n = 60usize;
        let x1 = linspace(n, 0.0, 1.0);
        let x2: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % n) as f64 / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| x1[i] * x1[i] + x2[i]).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1, x2]);
        let specs = [SmoothSpec::new(0, 6, 0.1), SmoothSpec::new(1, 5, 0.1)];
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        // Both partitions of unity span the constant: 1 + (6 − 1) + (5 − 1) = 10, not 12.
        assert_eq!(fit.diagnostics.rank, 10);
        assert!(fit.boundary_lambda_smooths.is_empty());
        let mut bad = y.clone();
        bad[3] = f64::NAN;
        assert!(
            fit_gam(
                &PredictorMatrix::new(&x, nrows, ncols),
                &bad,
                &specs,
                &GamOptions::default(),
                &FaerBackend,
                &mut ws
            )
            .is_err()
        );
    }

    #[test]
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    fn predict_matches_fitted_on_training() {
        let n = 100usize;
        let x1 = linspace(n, 0.0, 1.0);
        let y: Vec<f64> = x1.iter().map(|&v| (std::f64::consts::PI * v).sin()).collect();
        let (x, nrows, ncols) = colmajor_from_cols(&[x1]);
        let specs = [SmoothSpec::new(0, 8, 0.01).with_variable(VariableId::from_raw(0))];
        let backend = FaerBackend;
        let mut ws = GamWorkspace::default();
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &backend,
            &mut ws,
        )
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
        let fit = fit_gam(
            &PredictorMatrix::new(&x, nrows, ncols),
            &y,
            &specs,
            &GamOptions::default(),
            &backend,
            &mut ws,
        )
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
        let err = fit_gam(
            &PredictorMatrix::new(&x, 3, 1),
            &[1.0, 2.0, 3.0],
            &specs,
            &GamOptions::default(),
            &backend,
            &mut ws,
        );
        assert!(err.is_err());
        let specs = [SmoothSpec::new(1, 6, 0.1)];
        let err = fit_gam(
            &PredictorMatrix::new(&x, 3, 1),
            &[1.0, 2.0, 3.0],
            &specs,
            &GamOptions::default(),
            &backend,
            &mut ws,
        );
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
    #[test]
    fn weighted_gam_matches_repeated_rows_with_fixed_knots() {
        let n = 30;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin() + 0.2 * v * v).collect();
        let (_, knots) = expand_bspline(&x, 5, None).unwrap();
        let specs = [SmoothSpec::new(0, 5, 0.3).with_knots(knots)];
        // Counts sum to n: penalty scale is identical in both fits.
        let weights: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let mut repeated_x = Vec::new();
        let mut repeated_y = Vec::new();
        for i in 0..n {
            for _ in 0..i % 3 {
                repeated_x.push(x[i]);
                repeated_y.push(y[i]);
            }
        }
        let options = GamOptions { max_iter: 500, tol: 1e-9 };
        let weighted = fit_gam_weighted(
            &PredictorMatrix::new(&x, n, 1),
            &y,
            &specs,
            &options,
            Some(&weights),
            &FaerBackend,
            &mut GamWorkspace::default(),
        )
        .unwrap();
        let repeated = fit_gam(
            &PredictorMatrix::new(&repeated_x, n, 1),
            &repeated_y,
            &specs,
            &options,
            &FaerBackend,
            &mut GamWorkspace::default(),
        )
        .unwrap();
        assert!(weighted.converged && repeated.converged);
        for at in [0.2, 1.1, 2.4] {
            assert!(
                (weighted.predict_row(&[at]).unwrap() - repeated.predict_row(&[at]).unwrap()).abs()
                    < 1e-7
            );
            assert!(
                (weighted.smooth_derivative(0, at).unwrap()
                    - repeated.smooth_derivative(0, at).unwrap())
                .abs()
                    < 1e-7
            );
        }
    }

    #[test]
    fn spectral_gcv_scores_match_the_per_lambda_factorization() {
        let n = 70usize;
        let x1 = linspace(n, 0.0, 1.0);
        let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 4) as f64 * 0.4).collect();
        let y: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, &v)| (4.0 * v).sin() + 0.2 * ((i * 5) % 7) as f64)
            .collect();
        let (basis, _) = expand_bspline(&x1, 9, None).unwrap();
        let sqrt_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
        let scaled_basis: Vec<f64> =
            basis.iter().enumerate().map(|(i, v)| v * sqrt_w[i % n]).collect();
        let scaled_y: Vec<f64> = y.iter().zip(&sqrt_w).map(|(v, s)| v * s).collect();
        let smoother = PenalizedSmoother::new(&scaled_basis, n, 9).unwrap();
        let spectrum = smoother.spectrum().expect("pencil is factorizable");
        let scores = spectral_gcv_scores(&smoother, spectrum, &scaled_basis, n, &scaled_y, &sqrt_w);
        for (&lambda, &got) in GCV_LAMBDA_GRID.iter().zip(&scores) {
            let factor = smoother.factor(lambda).unwrap();
            let want =
                centered_smooth_gcv(&smoother, &factor, &scaled_basis, n, &scaled_y, &sqrt_w);
            assert!((got - want).abs() <= 1e-6 * want.abs(), "λ={lambda}: {got} vs {want}");
        }
    }
}
