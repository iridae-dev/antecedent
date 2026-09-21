//! Weighted least squares and two-stage least squares.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};
use crate::special::{gamma_q, normal_ppf, student_t_ppf};

/// Fit weighted least squares by row-scaling with `sqrt(weight)`.
///
/// `weights` length = `nrows`. Weights must be finite and non-negative.
/// Zero is allowed and drops that row (scale factor 0); negatives and non-finite
/// values are rejected as upstream errors rather than coerced.
///
/// # Errors
///
/// Shape mismatch, invalid weights, or backend failure.
pub fn fit_wls(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LeastSquaresFit, StatsError> {
    if y.len() != nrows || weights.len() != nrows {
        return Err(StatsError::Shape { message: "y/weights length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    let mut x_w = vec![0.0; nrows * ncols];
    let mut y_w = vec![0.0; nrows];
    for r in 0..nrows {
        let wr = weights[r];
        if !(wr.is_finite() && wr >= 0.0) {
            return Err(StatsError::Shape {
                message: "WLS weights must be finite and non-negative",
            });
        }
        let w = wr.sqrt();
        y_w[r] = y[r] * w;
        for c in 0..ncols {
            x_w[c * nrows + r] = x_colmajor[c * nrows + r] * w;
        }
    }
    backend.least_squares(&x_w, nrows, ncols, &y_w, workspace)
}

/// Weak-instrument diagnostic: joint significance of the excluded instruments in the
/// first-stage regression of the endogenous variable on `[instruments | exogenous]`.
///
/// [`fit_2sls`] never hard-fails on a weak instrument (only on an exactly
/// degenerate first stage). Licensed IV uncertainty is the Anderson–Rubin set in
/// [`Self::anderson_rubin`], not a Wald / pretest SE.
#[derive(Clone, Debug, PartialEq)]
pub struct FirstStageDiagnostics {
    /// F-statistic for the joint null that all excluded-instrument coefficients are zero,
    /// controlling for the included exogenous regressors.
    pub f_statistic: f64,
    /// Numerator degrees of freedom (number of excluded instruments).
    pub df1: usize,
    /// Denominator degrees of freedom (first-stage residual df, `n − (df1 + x_ncols)`).
    pub df2: usize,
    /// Partial R² attributable to the excluded instruments:
    /// `(RSS_restricted − RSS_full) / RSS_restricted`.
    pub partial_r2: f64,
    /// Homoskedastic Anderson–Rubin confidence set for the structural treatment
    /// coefficient at [`Self::anderson_rubin`]'s third component (nominal level).
    ///
    /// `(lower, upper, level)`. Endpoints may be infinite when the non-rejection set
    /// is an unbounded ray or the whole line. `None` when the set is empty, a union of
    /// disjoint components, or when AR was withheld (see [`Self::uncertainty_withheld`]).
    pub anderson_rubin: Option<(f64, f64, f64)>,
    /// Why a licensed IV uncertainty product was withheld (AR union / empty set /
    /// non-homoskedastic SE kind / numerical failure).
    pub uncertainty_withheld: Option<&'static str>,
}

/// Result of two-stage least squares.
#[derive(Clone, Debug)]
pub struct TwoSlsFit {
    /// First-stage coefficients (full instrument set `[Z | X]` → endogenous).
    pub first_stage: LeastSquaresFit,
    /// Second-stage coefficients (fitted endogenous + covariates → outcome).
    pub second_stage: LeastSquaresFit,
    /// Fitted endogenous values used in stage 2.
    pub fitted_endogenous: Vec<f64>,
    /// Structural residual sum of squares `‖y − Tβ̂ − Xγ̂‖²` using the *actual*
    /// endogenous values (not the fitted ones). This is the σ̂² numerator for the
    /// conventional 2SLS analytic standard error.
    pub structural_rss: f64,
    /// Structural residuals `e_i = y_i − T_i β̂ − X_i γ̂` (same as RSS terms).
    pub structural_residuals: Vec<f64>,
    /// Weak-instrument diagnostic for the excluded instrument set (see
    /// [`FirstStageDiagnostics`]).
    pub first_stage_diagnostics: FirstStageDiagnostics,
}

/// Two-stage least squares.
///
/// Stage 1: `endogenous ~ [instruments | exogenous]` — the full instrument set. Included
/// exogenous regressors instrument themselves, so `exogenous` (column-major, may be
/// empty / intercept-only) is appended to the excluded instruments in the first-stage
/// design. Pass the intercept in exactly one of the two blocks.
/// Stage 2: `y ~ [fitted_endogenous | exogenous]`.
///
/// Convention: stage-2 design is `[fitted_T | X]` with `1 + x_ncols` columns; the treatment
/// coefficient is `second_stage.coefficients[0]`.
///
/// # Errors
///
/// Shape mismatch or backend failure.
#[allow(clippy::too_many_arguments)]
pub fn fit_2sls(
    instruments_colmajor: &[f64],
    z_nrows: usize,
    z_ncols: usize,
    endogenous: &[f64],
    exogenous_colmajor: &[f64],
    x_ncols: usize,
    y: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<TwoSlsFit, StatsError> {
    if endogenous.len() != z_nrows || y.len() != z_nrows {
        return Err(StatsError::Shape { message: "endogenous/y length != nrows" });
    }
    if exogenous_colmajor.len() < z_nrows.saturating_mul(x_ncols) {
        return Err(StatsError::Shape { message: "exogenous buffer too short" });
    }
    // Full instrument set: excluded instruments plus included exogenous regressors.
    let stage1_ncols = z_ncols + x_ncols;
    let mut x1 = vec![0.0; z_nrows * stage1_ncols];
    x1[..z_nrows * z_ncols].copy_from_slice(&instruments_colmajor[..z_nrows * z_ncols]);
    x1[z_nrows * z_ncols..].copy_from_slice(&exogenous_colmajor[..z_nrows * x_ncols]);
    let first_stage = backend.least_squares(&x1, z_nrows, stage1_ncols, endogenous, workspace)?;
    let first_stage_diagnostics = first_stage_f_test(
        z_nrows,
        z_ncols,
        x_ncols,
        exogenous_colmajor,
        endogenous,
        first_stage.rss,
        backend,
        workspace,
    )?;
    let mut fitted = vec![0.0; z_nrows];
    for r in 0..z_nrows {
        let mut pred = 0.0;
        for c in 0..stage1_ncols {
            pred += x1[c * z_nrows + r] * first_stage.coefficients[c];
        }
        fitted[r] = pred;
    }
    let stage2_ncols = 1 + x_ncols;
    let mut x2 = vec![0.0; z_nrows * stage2_ncols];
    for r in 0..z_nrows {
        x2[r] = fitted[r];
        for c in 0..x_ncols {
            x2[(1 + c) * z_nrows + r] = exogenous_colmajor[c * z_nrows + r];
        }
    }
    let second_stage = backend.least_squares(&x2, z_nrows, stage2_ncols, y, workspace)?;
    // Structural residuals evaluate the second-stage coefficients at the ACTUAL
    // endogenous values; `second_stage.rss` uses fitted T and is not σ̂².
    let mut structural_residuals = vec![0.0; z_nrows];
    let mut structural_rss = 0.0;
    for r in 0..z_nrows {
        let mut pred = second_stage.coefficients[0] * endogenous[r];
        for c in 0..x_ncols {
            pred += exogenous_colmajor[c * z_nrows + r] * second_stage.coefficients[1 + c];
        }
        let e = y[r] - pred;
        structural_residuals[r] = e;
        structural_rss += e * e;
    }
    Ok(TwoSlsFit {
        first_stage,
        second_stage,
        fitted_endogenous: fitted,
        structural_rss,
        structural_residuals,
        first_stage_diagnostics,
    })
}

/// Partial F-test for the excluded-instrument block of a first-stage regression:
/// compares the unrestricted fit `[instruments | exogenous]` (`full_rss`, already
/// computed by the caller) against a restricted fit on `exogenous` alone (or the
/// zero model when `x_ncols == 0`, i.e. `RSS = Σ endogenous²`).
#[allow(clippy::too_many_arguments)]
fn first_stage_f_test(
    nrows: usize,
    z_ncols: usize,
    x_ncols: usize,
    exogenous_colmajor: &[f64],
    endogenous: &[f64],
    full_rss: f64,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<FirstStageDiagnostics, StatsError> {
    let restricted_rss = if x_ncols == 0 {
        endogenous.iter().map(|v| v * v).sum::<f64>()
    } else {
        let restricted =
            backend.least_squares(exogenous_colmajor, nrows, x_ncols, endogenous, workspace)?;
        restricted.rss
    };
    let df1 = z_ncols;
    let df2 = nrows.saturating_sub(z_ncols + x_ncols);
    let f_statistic = if df1 == 0 || df2 == 0 {
        f64::NAN
    } else if full_rss > 0.0 {
        ((restricted_rss - full_rss) / df1 as f64) / (full_rss / df2 as f64)
    } else {
        f64::INFINITY
    };
    let partial_r2 = if restricted_rss > 0.0 {
        ((restricted_rss - full_rss) / restricted_rss).max(0.0)
    } else {
        0.0
    };
    Ok(FirstStageDiagnostics {
        f_statistic,
        df1,
        df2,
        partial_r2,
        anderson_rubin: None,
        uncertainty_withheld: None,
    })
}

/// χ² critical value `c` with `P(χ²_df > c) = 1 - level` via bisection on [`gamma_q`].
///
/// Prefer [`f_critical`] / [`anderson_rubin_kf_critical`] for AR inversion; this remains
/// for asymptotic χ² comparisons.
#[must_use]
pub fn chi2_critical(level: f64, df: usize) -> f64 {
    if df == 0 || !(level.is_finite() && (0.0..1.0).contains(&level)) {
        return f64::NAN;
    }
    if df == 1 {
        // Exact: χ²_1 = Z².
        let z = normal_ppf(0.5 + 0.5 * level);
        return z * z;
    }
    let alpha = 1.0 - level;
    let a = df as f64 * 0.5;
    let mut lo = 0.0;
    let mut hi = (df as f64) + 40.0;
    while gamma_q(a, hi * 0.5) > alpha {
        hi *= 2.0;
        if !hi.is_finite() || hi > 1e14 {
            return f64::INFINITY;
        }
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if gamma_q(a, mid * 0.5) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Upper `level` critical value of `F(d1, d2)`: `P(F ≤ f) = level`.
///
/// For `d1 == 1` this is exactly `t_{d2, (1+level)/2}²`. Otherwise it bisects the
/// regularized incomplete-beta F CDF.
#[must_use]
pub fn f_critical(level: f64, d1: usize, d2: usize) -> f64 {
    if d1 == 0 || d2 == 0 || !(level.is_finite() && (0.0..1.0).contains(&level)) {
        return f64::NAN;
    }
    if d1 == 1 {
        let t = student_t_ppf(0.5 + 0.5 * level, d2 as f64);
        return t * t;
    }
    let a = 0.5 * d1 as f64;
    let b = 0.5 * d2 as f64;
    let mut lo = 0.0;
    let mut hi = 1.0;
    while f_cdf(hi, a, b) < level {
        hi *= 2.0;
        if !hi.is_finite() || hi > 1e14 {
            return f64::INFINITY;
        }
    }
    for _ in 0..120 {
        let mid = 0.5 * (lo + hi);
        if f_cdf(mid, a, b) < level {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn f_cdf(f: f64, a: f64, b: f64) -> f64 {
    if !(f.is_finite() && f >= 0.0) {
        return f64::NAN;
    }
    // P(F ≤ f) = I_{d1 f / (d1 f + d2)}(d1/2, d2/2) with a=d1/2, b=d2/2.
    let x = (2.0 * a * f) / (2.0 * a * f + 2.0 * b);
    crate::special::regularized_incomplete_beta(x, a, b)
}

/// Critical value for [`anderson_rubin_statistic`] (`k · F`) at finite-sample level
/// `F_{k, df, level}`: accept when `anderson_rubin_statistic ≤ anderson_rubin_kf_critical`.
#[must_use]
pub fn anderson_rubin_kf_critical(level: f64, k: usize, df: usize) -> f64 {
    k as f64 * f_critical(level, k, df)
}

/// Homoskedastic Anderson–Rubin statistic for `H₀: β = beta0`.
///
/// Forms `ỹ = y − beta0 · t` and returns `k · F` from the partial F-test that the
/// excluded instruments are jointly insignificant in `ỹ ~ [Z | X]`. Compare against
/// [`anderson_rubin_kf_critical`] (finite-sample `k · F_{k, df, level}`), not a bare χ².
///
/// # Errors
///
/// Shape mismatch or backend failure.
#[allow(clippy::too_many_arguments)]
pub fn anderson_rubin_statistic(
    y: &[f64],
    t: &[f64],
    instruments_colmajor: &[f64],
    z_nrows: usize,
    z_ncols: usize,
    exogenous_colmajor: &[f64],
    x_ncols: usize,
    beta0: f64,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<f64, StatsError> {
    if y.len() != z_nrows || t.len() != z_nrows {
        return Err(StatsError::Shape { message: "y/t length != nrows" });
    }
    if instruments_colmajor.len() < z_nrows.saturating_mul(z_ncols) {
        return Err(StatsError::Shape { message: "instruments buffer too short" });
    }
    if exogenous_colmajor.len() < z_nrows.saturating_mul(x_ncols) {
        return Err(StatsError::Shape { message: "exogenous buffer too short" });
    }
    let mut y_star = vec![0.0; z_nrows];
    for i in 0..z_nrows {
        y_star[i] = y[i] - beta0 * t[i];
    }
    let stage_ncols = z_ncols + x_ncols;
    let mut xz = vec![0.0; z_nrows * stage_ncols];
    xz[..z_nrows * z_ncols].copy_from_slice(&instruments_colmajor[..z_nrows * z_ncols]);
    if x_ncols > 0 {
        xz[z_nrows * z_ncols..].copy_from_slice(&exogenous_colmajor[..z_nrows * x_ncols]);
    }
    let full = backend.least_squares(&xz, z_nrows, stage_ncols, &y_star, workspace)?;
    let restricted_rss = if x_ncols == 0 {
        y_star.iter().map(|v| v * v).sum::<f64>()
    } else {
        backend.least_squares(exogenous_colmajor, z_nrows, x_ncols, &y_star, workspace)?.rss
    };
    let k = z_ncols;
    let df = z_nrows.saturating_sub(stage_ncols);
    if k == 0 || df == 0 {
        return Ok(f64::NAN);
    }
    if full.rss <= 0.0 {
        return Ok(if restricted_rss > full.rss { f64::INFINITY } else { 0.0 });
    }
    let f = ((restricted_rss - full.rss) / k as f64) / (full.rss / df as f64);
    Ok(k as f64 * f.max(0.0))
}

fn residualize_on_exogenous(
    v: &[f64],
    exogenous_colmajor: &[f64],
    nrows: usize,
    x_ncols: usize,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<Vec<f64>, StatsError> {
    if x_ncols == 0 {
        return Ok(v.to_vec());
    }
    let fit = backend.least_squares(exogenous_colmajor, nrows, x_ncols, v, workspace)?;
    Ok(fit.residuals)
}

/// Invert `A β² + B β + C ≤ 0` into an Anderson–Rubin acceptance set.
fn classify_quadratic_le_zero(a: f64, b: f64, c: f64) -> QuadraticAcceptance {
    let scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
    let eps = 1e-12 * scale;
    if a.abs() <= eps {
        // Linear (or constant): B β + C ≤ 0.
        if b.abs() <= eps {
            return if c <= eps {
                QuadraticAcceptance::WholeLine
            } else {
                QuadraticAcceptance::Empty
            };
        }
        let root = -c / b;
        return if b > 0.0 {
            QuadraticAcceptance::Ray { lo: f64::NEG_INFINITY, hi: root }
        } else {
            QuadraticAcceptance::Ray { lo: root, hi: f64::INFINITY }
        };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < -eps * eps {
        // No real roots: sign follows A.
        return if a < 0.0 { QuadraticAcceptance::WholeLine } else { QuadraticAcceptance::Empty };
    }
    let sqrt_disc = disc.max(0.0).sqrt();
    let r1 = (-b - sqrt_disc) / (2.0 * a);
    let r2 = (-b + sqrt_disc) / (2.0 * a);
    let (lo, hi) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
    if disc <= eps * eps {
        // Tangent: a single point when A > 0; whole line when A < 0.
        return if a > 0.0 {
            QuadraticAcceptance::Interval { lo, hi: lo }
        } else {
            QuadraticAcceptance::WholeLine
        };
    }
    if a > 0.0 {
        QuadraticAcceptance::Interval { lo, hi }
    } else {
        QuadraticAcceptance::Union { lo, hi }
    }
}

#[derive(Clone, Debug)]
enum QuadraticAcceptance {
    Empty,
    WholeLine,
    Interval {
        lo: f64,
        hi: f64,
    },
    Ray {
        lo: f64,
        hi: f64,
    },
    /// Accepts (−∞, lo] ∪ [hi, ∞).
    Union {
        #[allow(dead_code)]
        lo: f64,
        #[allow(dead_code)]
        hi: f64,
    },
}

/// Invert the homoskedastic Anderson–Rubin test at `level` by solving the exact
/// quadratic inequality in β after residualizing on the exogenous block (including
/// the intercept).
///
/// After Frisch–Waugh, `N(β) = ‖P_Z ỹ(β)‖²` and `D(β) = ‖M_Z ỹ(β)‖²` are quadratic in
/// β. Acceptance `F(β) ≤ F_{k, df, level}` rearranges to `A β² + B β + C ≤ 0`. The
/// critical value is the finite-sample F quantile (not a probe-grid χ² screen).
///
/// # Errors
///
/// Shape mismatch or backend failure while residualizing / projecting.
#[allow(clippy::too_many_arguments)]
pub fn anderson_rubin_confidence_set(
    y: &[f64],
    t: &[f64],
    instruments_colmajor: &[f64],
    z_nrows: usize,
    z_ncols: usize,
    exogenous_colmajor: &[f64],
    x_ncols: usize,
    level: f64,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<(Option<(f64, f64, f64)>, Option<&'static str>), StatsError> {
    if z_ncols == 0 {
        return Ok((None, Some("anderson_rubin_requires_excluded_instruments")));
    }
    if !(level.is_finite() && (0.0..1.0).contains(&level)) {
        return Ok((None, Some("anderson_rubin_invalid_level")));
    }
    if y.len() != z_nrows || t.len() != z_nrows {
        return Err(StatsError::Shape { message: "y/t length != nrows" });
    }
    if instruments_colmajor.len() < z_nrows.saturating_mul(z_ncols) {
        return Err(StatsError::Shape { message: "instruments buffer too short" });
    }
    if exogenous_colmajor.len() < z_nrows.saturating_mul(x_ncols) {
        return Err(StatsError::Shape { message: "exogenous buffer too short" });
    }
    let df = z_nrows.saturating_sub(z_ncols + x_ncols);
    if df == 0 {
        return Ok((None, Some("anderson_rubin_critical_value_failed")));
    }
    let f_crit = f_critical(level, z_ncols, df);
    if !f_crit.is_finite() {
        return Ok((None, Some("anderson_rubin_critical_value_failed")));
    }
    // Compare k·F to k·F_crit ⇔ df·N ≤ (k·F_crit)·D.
    let c = z_ncols as f64 * f_crit;

    let y_star =
        residualize_on_exogenous(y, exogenous_colmajor, z_nrows, x_ncols, backend, workspace)?;
    let t_star =
        residualize_on_exogenous(t, exogenous_colmajor, z_nrows, x_ncols, backend, workspace)?;
    let mut z_star = vec![0.0; z_nrows * z_ncols];
    for j in 0..z_ncols {
        let col = &instruments_colmajor[j * z_nrows..(j + 1) * z_nrows];
        let resid = residualize_on_exogenous(
            col,
            exogenous_colmajor,
            z_nrows,
            x_ncols,
            backend,
            workspace,
        )?;
        z_star[j * z_nrows..(j + 1) * z_nrows].copy_from_slice(&resid);
    }

    let fit_y = backend.least_squares(&z_star, z_nrows, z_ncols, &y_star, workspace)?;
    let fit_t = backend.least_squares(&z_star, z_nrows, z_ncols, &t_star, workspace)?;
    let yy: f64 = y_star.iter().map(|v| v * v).sum();
    let tt: f64 = t_star.iter().map(|v| v * v).sum();
    let yt: f64 = y_star.iter().zip(t_star.iter()).map(|(a, b)| a * b).sum();
    let m_yy = fit_y.rss;
    let m_tt = fit_t.rss;
    let m_yt: f64 = fit_y.residuals.iter().zip(fit_t.residuals.iter()).map(|(a, b)| a * b).sum();
    let n_yy = (yy - m_yy).max(0.0);
    let n_tt = (tt - m_tt).max(0.0);
    let n_yt = yt - m_yt;

    let df_f = df as f64;
    let a = df_f * n_tt - c * m_tt;
    let b = -2.0 * df_f * n_yt + 2.0 * c * m_yt;
    let c0 = df_f * n_yy - c * m_yy;

    match classify_quadratic_le_zero(a, b, c0) {
        QuadraticAcceptance::Empty => Ok((None, Some("anderson_rubin_set_empty"))),
        QuadraticAcceptance::WholeLine => {
            Ok((Some((f64::NEG_INFINITY, f64::INFINITY, level)), None))
        }
        QuadraticAcceptance::Interval { lo, hi } | QuadraticAcceptance::Ray { lo, hi } => {
            Ok((Some((lo, hi, level)), None))
        }
        QuadraticAcceptance::Union { .. } => Ok((None, Some("anderson_rubin_set_is_union"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;

    #[test]
    fn wls_matches_ols_with_unit_weights() {
        let n = 20usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 1.0 + 2.0 * (i as f64);
        }
        let w = vec![1.0; n];
        let mut ws = LeastSquaresWorkspace::default();
        let ols = FaerBackend.least_squares(&x, n, 2, &y, &mut ws).unwrap();
        let wls = fit_wls(&x, n, 2, &y, &w, &FaerBackend, &mut ws).unwrap();
        assert!((ols.coefficients[0] - wls.coefficients[0]).abs() < 1e-10);
        assert!((ols.coefficients[1] - wls.coefficients[1]).abs() < 1e-10);
    }

    #[test]
    fn wls_rejects_negative_and_nonfinite_weights() {
        let n = 4usize;
        let x = vec![1.0; n * 2];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let mut ws = LeastSquaresWorkspace::default();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut w = vec![1.0; n];
            w[1] = bad;
            let err = fit_wls(&x, n, 2, &y, &w, &FaerBackend, &mut ws).unwrap_err();
            assert_eq!(
                err,
                StatsError::Shape { message: "WLS weights must be finite and non-negative" }
            );
        }
    }

    #[test]
    fn wls_zero_weight_drops_row() {
        // Zero weight is an explicit drop-row policy: omitting the zero-weight
        // observation must match fitting on the complementary subset with unit weights.
        let n = 4usize;
        let mut x_full = vec![0.0; n * 2];
        let y_full = [1.0, 10.0, 3.0, 4.0];
        for i in 0..n {
            x_full[i] = 1.0;
            x_full[n + i] = i as f64;
        }
        let w = [1.0, 0.0, 1.0, 1.0];
        let mut ws = LeastSquaresWorkspace::default();
        let wls = fit_wls(&x_full, n, 2, &y_full, &w, &FaerBackend, &mut ws).unwrap();

        let n_sub = 3usize;
        let mut x_sub = vec![0.0; n_sub * 2];
        let y_sub = [1.0, 3.0, 4.0];
        for (j, i) in [0usize, 2, 3].into_iter().enumerate() {
            x_sub[j] = 1.0;
            x_sub[n_sub + j] = i as f64;
        }
        let ols = FaerBackend.least_squares(&x_sub, n_sub, 2, &y_sub, &mut ws).unwrap();
        assert!((ols.coefficients[0] - wls.coefficients[0]).abs() < 1e-10);
        assert!((ols.coefficients[1] - wls.coefficients[1]).abs() < 1e-10);
    }

    #[test]
    fn twosls_recovers_just_identified() {
        // Z → T → Y with no confounding on Z→Y; T = Z + e, Y = 2T + u.
        // Instruments carry only the excluded Z column; the intercept lives in the
        // exogenous block (stage 1 uses [Z | 1], stage 2 uses [fitted_T | 1]).
        let n = 200usize;
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n]; // intercept only exogenous
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            z[i] = zi;
            t[i] = zi + 0.01 * ((i % 7) as f64 - 3.0);
            y[i] = 2.0 * t[i] + 0.01 * ((i % 5) as f64 - 2.0);
            x[i] = 1.0;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_2sls(&z, n, 1, &t, &x, 1, &y, &FaerBackend, &mut ws).unwrap();
        assert!((fit.second_stage.coefficients[0] - 2.0).abs() < 0.05);
        assert!(fit.structural_rss <= fit.second_stage.rss);
    }

    #[test]
    fn first_stage_diagnostics_strong_vs_weak_instrument() {
        // Same DGP shape (Z -> T -> Y, intercept-only exogenous block), varied only by how
        // strongly Z drives T. Strong: T = Z + small noise (first stage explains most of
        // T's variance). Weak: T = 0.001*Z + large noise (first stage explains ~none of
        // it). The F-statistic must separate these cases sharply.
        let n = 200usize;
        let mut z = vec![0.0; n];
        let mut x = vec![0.0; n];
        for i in 0..n {
            z[i] = (i as f64) / n as f64 - 0.5;
            x[i] = 1.0;
        }

        let mut t_strong = vec![0.0; n];
        let mut y_strong = vec![0.0; n];
        let mut t_weak = vec![0.0; n];
        let mut y_weak = vec![0.0; n];
        for i in 0..n {
            let small_noise = 0.01 * ((i % 7) as f64 - 3.0);
            let large_noise = 1.0 * ((i % 7) as f64 - 3.0);
            t_strong[i] = z[i] + small_noise;
            y_strong[i] = 2.0 * t_strong[i] + 0.01 * ((i % 5) as f64 - 2.0);
            t_weak[i] = 0.001 * z[i] + large_noise;
            y_weak[i] = 2.0 * t_weak[i] + 0.01 * ((i % 5) as f64 - 2.0);
        }

        let mut ws = LeastSquaresWorkspace::default();
        let strong =
            fit_2sls(&z, n, 1, &t_strong, &x, 1, &y_strong, &FaerBackend, &mut ws).unwrap();
        let weak = fit_2sls(&z, n, 1, &t_weak, &x, 1, &y_weak, &FaerBackend, &mut ws).unwrap();

        assert_eq!(strong.first_stage_diagnostics.df1, 1);
        assert_eq!(strong.first_stage_diagnostics.df2, n - 2);
        assert_eq!(weak.first_stage_diagnostics.df1, 1);
        assert_eq!(weak.first_stage_diagnostics.df2, n - 2);

        assert!(
            strong.first_stage_diagnostics.f_statistic > 1000.0,
            "strong F={}",
            strong.first_stage_diagnostics.f_statistic
        );
        assert!(
            weak.first_stage_diagnostics.f_statistic < 5.0,
            "weak F={}",
            weak.first_stage_diagnostics.f_statistic
        );
        assert!(strong.first_stage_diagnostics.partial_r2 > 0.9);
        assert!(weak.first_stage_diagnostics.partial_r2 < 0.1);
    }

    #[test]
    fn anderson_rubin_at_true_beta_matches_reduced_form_test() {
        // Tiny just-identified fixture: Z drives T, Y = 2T + noise. At the true
        // β = 2, ỹ = Y − 2T is pure noise orthogonal to Z (up to the same noise),
        // so the AR statistic equals k·F from the reduced-form regression of ỹ on Z.
        let n = 8usize;
        let z = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let t = [0.1, 0.2, 0.0, 0.3, 1.1, 0.9, 1.2, 0.8];
        let y: Vec<f64> =
            t.iter().enumerate().map(|(i, &ti)| 2.0 * ti + 0.05 * (i as f64 - 3.5)).collect();
        let x = [1.0; 8];
        let mut ws = LeastSquaresWorkspace::default();
        let ar =
            anderson_rubin_statistic(&y, &t, &z, n, 1, &x, 1, 2.0, &FaerBackend, &mut ws).unwrap();

        let y_star: Vec<f64> = y.iter().zip(t).map(|(yi, ti)| yi - 2.0 * ti).collect();
        let mut xz = vec![0.0; n * 2];
        xz[..n].copy_from_slice(&z);
        xz[n..].copy_from_slice(&x);
        let full = FaerBackend.least_squares(&xz, n, 2, &y_star, &mut ws).unwrap();
        let restricted = FaerBackend.least_squares(&x, n, 1, &y_star, &mut ws).unwrap();
        let df = n - 2;
        let f = ((restricted.rss - full.rss) / 1.0) / (full.rss / df as f64);
        let expected = f;
        assert!((ar - expected).abs() < 1e-10, "AR={ar} reduced-form k·F={expected}");
    }

    #[test]
    fn anderson_rubin_confidence_set_covers_true_beta_on_strong_fixture() {
        let n = 200usize;
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n];
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            z[i] = zi;
            t[i] = zi + 0.01 * ((i % 7) as f64 - 3.0);
            y[i] = 2.0 * t[i] + 0.01 * ((i % 5) as f64 - 2.0);
            x[i] = 1.0;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let (ar, reason) =
            anderson_rubin_confidence_set(&y, &t, &z, n, 1, &x, 1, 0.95, &FaerBackend, &mut ws)
                .unwrap();
        assert!(reason.is_none(), "unexpected withhold: {reason:?}");
        let (lo, hi, level) = ar.expect("connected AR set");
        assert_eq!(level, 0.95);
        assert!(lo <= 2.0 && 2.0 <= hi, "AR [{lo}, {hi}] should cover 2");
        assert!(lo.is_finite() && hi.is_finite());
    }

    #[test]
    fn anderson_rubin_quadratic_recovers_narrow_interval_off_mesh() {
        // True β = 2.05 with a strong first stage and tiny residual noise so the 95%
        // AR set is narrower than the old 0.1 probe mesh and contains no mesh point.
        // A grid inversion would report empty; the quadratic must publish a finite
        // interval covering 2.05.
        let n = 400usize;
        let beta = 2.05;
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let x = vec![1.0; n];
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            z[i] = zi;
            t[i] = 3.0 * zi + 1e-6 * ((i % 7) as f64 - 3.0);
            y[i] = beta * t[i] + 1e-6 * ((i % 5) as f64 - 2.0);
        }
        let mut ws = LeastSquaresWorkspace::default();
        let (ar, reason) =
            anderson_rubin_confidence_set(&y, &t, &z, n, 1, &x, 1, 0.95, &FaerBackend, &mut ws)
                .unwrap();
        assert!(reason.is_none(), "unexpected withhold: {reason:?}");
        let (lo, hi, level) = ar.expect("narrow AR set must exist");
        assert_eq!(level, 0.95);
        assert!(lo.is_finite() && hi.is_finite(), "narrow set must be finite [{lo}, {hi}]");
        assert!(
            hi - lo < 0.1,
            "fixture must be narrower than the old 0.1 mesh, got width {}",
            hi - lo
        );
        // No 0.1-mesh point lies in (lo, hi) when the interval sits between 2.0 and 2.1.
        let mesh_hit = (0..=400).any(|i| {
            let p = -20.0 + 0.1 * f64::from(i);
            lo <= p && p <= hi
        });
        assert!(
            !mesh_hit,
            "fixture must miss the 0.1 mesh so a grid inversion would fail; got [{lo}, {hi}]"
        );
        assert!(lo <= beta && beta <= hi, "AR [{lo}, {hi}] must cover {beta}");
    }

    #[test]
    fn anderson_rubin_large_finite_interval_is_not_reported_infinite() {
        // Half-width well past 20 must stay finite endpoints — the old grid promoted
        // any outermost accepting probe to ±∞.
        let n = 200usize;
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let x = vec![1.0; n];
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            z[i] = zi;
            // Strong first stage (keeps A > 0 → connected finite set) with huge
            // outcome noise so the roots sit far past ±20.
            t[i] = zi + 0.02 * ((i % 7) as f64 - 3.0);
            y[i] = 2.0 * t[i] + 40.0 * ((i % 11) as f64 - 5.0);
        }
        let mut ws = LeastSquaresWorkspace::default();
        let (ar, reason) =
            anderson_rubin_confidence_set(&y, &t, &z, n, 1, &x, 1, 0.95, &FaerBackend, &mut ws)
                .unwrap();
        assert!(reason.is_none(), "unexpected withhold: {reason:?}");
        let (lo, hi, _) = ar.expect("connected AR set");
        assert!(lo.is_finite() && hi.is_finite(), "expected finite endpoints, got [{lo}, {hi}]");
        assert!(
            hi - lo > 40.0,
            "fixture must have half-width ≫ 20 (width {}), got [{lo}, {hi}]",
            hi - lo
        );
        assert!(lo <= 2.0 && 2.0 <= hi, "AR [{lo}, {hi}] should still cover 2");
    }

    #[test]
    fn twosls_first_stage_includes_exogenous_regressors() {
        // Y = 2T + 1.5X + u with X correlated with T beyond Z; projecting T on [1, Z]
        // only (the old first stage) is inconsistent here.
        let n = 400usize;
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n * 2];
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            let xi = ((i % 13) as f64 - 6.0) / 6.0;
            let e = 0.01 * ((i % 7) as f64 - 3.0);
            z[i] = zi;
            t[i] = zi + 0.8 * xi + e;
            y[i] = 2.0 * t[i] + 1.5 * xi + 0.01 * ((i % 5) as f64 - 2.0);
            x[i] = 1.0;
            x[n + i] = xi;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_2sls(&z, n, 1, &t, &x, 2, &y, &FaerBackend, &mut ws).unwrap();
        assert!(
            (fit.second_stage.coefficients[0] - 2.0).abs() < 0.05,
            "beta_T={}",
            fit.second_stage.coefficients[0]
        );
        assert!((fit.second_stage.coefficients[2] - 1.5).abs() < 0.05);
    }
}
