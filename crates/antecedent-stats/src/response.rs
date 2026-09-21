//! Numerical kernels for continuous causal-response estimation.
//!
//! These routines deliberately contain no causal semantics.  They provide the
//! local-polynomial and Gaussian-density calculations used by response estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use crate::StatsError;

/// Result of a Gaussian-kernel local quadratic regression at one coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalPolynomialPoint {
    /// Fitted level.
    pub value: f64,
    /// First derivative with respect to the coordinate.
    pub first_derivative: f64,
    /// Second derivative with respect to the coordinate.
    pub second_derivative: f64,
    /// Pointwise standard error for the fitted level.
    pub standard_error: f64,
    /// Pointwise plug-in standard error for the first derivative.
    pub first_derivative_standard_error: f64,
    /// Pointwise plug-in standard error for the second derivative.
    pub second_derivative_standard_error: f64,
    /// Kish effective sample size of the local kernel weights.
    pub local_ess: f64,
    /// Sum of unnormalized kernel weights.
    pub weight_sum: f64,
}

/// Local-polynomial fit together with observation-level linearized influences.
///
/// The influences are for the fitted level (the intercept in the centered local
/// polynomial). Their sum is zero up to numerical precision. They are suitable
/// for a fixed-grid multiplier bootstrap conditional on the supplied response.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalPolynomialInfluence {
    /// Local-polynomial point estimate and diagnostics.
    pub point: LocalPolynomialPoint,
    /// Observation-level linearized residual contributions.
    pub influences: Vec<f64>,
    /// Heteroskedasticity-robust standard error from the linearized contributions.
    pub robust_standard_error: f64,
    /// Heteroskedasticity-robust standard error for the first derivative.
    pub robust_first_derivative_standard_error: f64,
    /// Heteroskedasticity-robust standard error for the second derivative.
    pub robust_second_derivative_standard_error: f64,
}

/// Scratch for a local-quadratic fit. Reuse across a treatment grid so kernel
/// rows are not reallocated at every evaluation point.
#[derive(Clone, Debug, Default)]
pub struct LocalQuadraticWorkspace {
    weights: Vec<(f64, [f64; 3], f64)>,
}

/// Silverman's normal-reference bandwidth, with a range-based lower bound.
///
/// # Errors
///
/// Fewer than two finite observations or a degenerate sample.
pub fn silverman_bandwidth(x: &[f64]) -> Result<f64, StatsError> {
    if x.len() < 2 || x.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape { message: "bandwidth requires finite x with n >= 2" });
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let (minimum, maximum) =
        x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = maximum - minimum;
    if variance <= 0.0 || range <= 0.0 {
        return Err(StatsError::Shape { message: "bandwidth requires non-degenerate x" });
    }
    Ok((1.06 * variance.sqrt() * n.powf(-0.2)).max(range * 1e-6))
}

/// Gaussian probability density with mean and strictly positive standard deviation.
#[must_use]
pub fn gaussian_density(value: f64, mean: f64, standard_deviation: f64) -> f64 {
    if !value.is_finite()
        || !mean.is_finite()
        || !standard_deviation.is_finite()
        || standard_deviation <= 0.0
    {
        return f64::NAN;
    }
    let z = (value - mean) / standard_deviation;
    (-0.5 * z * z).exp() / (standard_deviation * (2.0 * std::f64::consts::PI).sqrt())
}

/// Fit a Gaussian-kernel local quadratic regression at `at`.
///
/// Coefficients use centered powers `[1, x-at, (x-at)^2]`, so the second
/// derivative is twice the quadratic coefficient. The uncertainty is a local
/// homoskedastic plug-in estimate and has pointwise, not simultaneous, semantics.
/// Kernel weights are not inverse-variance weights, so the plug-in uses the
/// sandwich `sigma^2 (X'WX)^-1 (X'W^2X) (X'WX)^-1`; the naive `(X'WX)^-1` form
/// overstates the variance for every kernel weight sequence.
///
/// # Errors
///
/// Shape mismatch, non-finite input, invalid bandwidth, or a singular local design.
pub fn gaussian_local_quadratic(
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
) -> Result<LocalPolynomialPoint, StatsError> {
    Ok(gaussian_local_quadratic_influence(x, y, at, bandwidth)?.point)
}

/// Fit a Gaussian-kernel local quadratic and return its linearized influences.
///
/// This uses the weighted-least-squares influence
/// `(X'WX)^(-1) x_i w_i residual_i` for the fitted level. It does not attach
/// causal meaning to `y`; callers remain responsible for the statistical
/// contract of any resampling procedure.
///
/// # Errors
///
/// Shape mismatch, non-finite input, invalid bandwidth, or a singular local design.
pub fn gaussian_local_quadratic_influence(
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
) -> Result<LocalPolynomialInfluence, StatsError> {
    gaussian_local_quadratic_influence_with(
        &mut LocalQuadraticWorkspace::default(),
        x,
        y,
        at,
        bandwidth,
    )
}

/// Same as [`gaussian_local_quadratic_influence`], reusing `workspace` storage.
///
/// `x`/`y` are checked for finiteness once here. For a shared grid, call
/// [`gaussian_local_quadratic_influence_prechecked`] after a single scan.
pub fn gaussian_local_quadratic_influence_with(
    workspace: &mut LocalQuadraticWorkspace,
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
) -> Result<LocalPolynomialInfluence, StatsError> {
    fit_local_quadratic(workspace, x, y, at, bandwidth, true, None)
}

/// Local quadratic after the caller has already refused non-finite `x`/`y`.
pub fn gaussian_local_quadratic_influence_prechecked(
    workspace: &mut LocalQuadraticWorkspace,
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
) -> Result<LocalPolynomialInfluence, StatsError> {
    fit_local_quadratic(workspace, x, y, at, bandwidth, false, None)
}

/// Local quadratic with observation weights in addition to the Gaussian kernel.
/// Weights must sum to the number of observations (up to floating-point error).
///
/// # Errors
/// Invalid weights or an unsupported local design.
pub fn gaussian_local_quadratic_weighted(
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
    weights: &[f64],
) -> Result<LocalPolynomialPoint, StatsError> {
    if weights.len() != x.len()
        || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
        || (weights.iter().sum::<f64>() - x.len() as f64).abs() > 1e-8 * x.len() as f64
    {
        return Err(StatsError::Shape { message: "invalid local quadratic weights" });
    }
    Ok(fit_local_quadratic(
        &mut LocalQuadraticWorkspace::default(),
        x,
        y,
        at,
        bandwidth,
        true,
        Some(weights),
    )?
    .point)
}

fn fit_local_quadratic(
    workspace: &mut LocalQuadraticWorkspace,
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
    check_finite: bool,
    observation_weights: Option<&[f64]>,
) -> Result<LocalPolynomialInfluence, StatsError> {
    if x.len() != y.len() || x.len() < 3 {
        return Err(StatsError::Shape {
            message: "local quadratic requires aligned x/y with n >= 3",
        });
    }
    if !at.is_finite() || !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(StatsError::Shape { message: "local quadratic inputs must be finite" });
    }
    if check_finite && x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(StatsError::Shape { message: "local quadratic inputs must be finite" });
    }
    let mut gram = [[0.0; 3]; 3];
    let mut squared_gram = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    workspace.weights.clear();
    workspace.weights.reserve(x.len());
    let mut weight_sum = 0.0;
    let mut weight_sq_sum = 0.0;
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        let dx = xi - at;
        let z = dx / bandwidth;
        let w = (-0.5 * z * z).exp() * observation_weights.map_or(1.0, |w| w[i]);
        let row = [1.0, dx, dx * dx];
        for j in 0..3 {
            rhs[j] += w * row[j] * yi;
            for k in 0..3 {
                gram[j][k] += w * row[j] * row[k];
                squared_gram[j][k] += w * w * row[j] * row[k];
            }
        }
        workspace.weights.push((w, row, yi));
        weight_sum += w;
        weight_sq_sum += w * w;
    }
    // A local quadratic spends three degrees of freedom; below that the residual
    // scale is not estimable and any interval would be invented rather than fitted.
    if weight_sum <= 3.0 || weight_sq_sum <= f64::EPSILON {
        return Err(StatsError::Backend(
            "local response window has too little effective weight for a local quadratic".into(),
        ));
    }
    let inverse = inverse_3x3(gram).ok_or(StatsError::SingularLocalDesign { order: 2 })?;
    let beta = matvec(inverse, rhs);
    let mut influences = Vec::with_capacity(workspace.weights.len());
    let mut first_derivative_influence_ss = 0.0;
    let mut second_derivative_influence_ss = 0.0;
    let mut weighted_rss = 0.0;
    for &(w, row, yi) in &workspace.weights {
        let residual = yi - row.iter().zip(beta).map(|(a, b)| a * b).sum::<f64>();
        weighted_rss += w * residual * residual;
        let hat = inverse[0].iter().zip(row).map(|(c, v)| c * v).sum::<f64>();
        influences.push(w * hat * residual);
        let first_hat = inverse[1].iter().zip(row).map(|(c, v)| c * v).sum::<f64>();
        first_derivative_influence_ss += (w * first_hat * residual).powi(2);
        let second_hat = inverse[2].iter().zip(row).map(|(c, v)| c * v).sum::<f64>();
        // The fitted quadratic coefficient is half the second derivative.
        second_derivative_influence_ss += (2.0 * w * second_hat * residual).powi(2);
    }
    let local_ess = weight_sum * weight_sum / weight_sq_sum;
    let sigma2 = weighted_rss / (weight_sum - 3.0);
    // sigma^2 (X'WX)^-1 (X'W^2X) (X'WX)^-1 — the kernel weights are not
    // inverse-variance weights, so the bread-only form is not the variance.
    let sandwich = matmul3(matmul3(inverse, squared_gram), inverse);
    let coefficient_se = |index: usize| (sigma2 * sandwich[index][index]).max(0.0).sqrt();
    let point = LocalPolynomialPoint {
        value: beta[0],
        first_derivative: beta[1],
        second_derivative: 2.0 * beta[2],
        standard_error: coefficient_se(0),
        first_derivative_standard_error: coefficient_se(1),
        second_derivative_standard_error: 2.0 * coefficient_se(2),
        local_ess,
        weight_sum,
    };
    let robust_standard_error = influences.iter().map(|value| value * value).sum::<f64>().sqrt();
    Ok(LocalPolynomialInfluence {
        point,
        influences,
        robust_standard_error,
        robust_first_derivative_standard_error: first_derivative_influence_ss.sqrt(),
        robust_second_derivative_standard_error: second_derivative_influence_ss.sqrt(),
    })
}

/// Robust bias-corrected (RBC) companion of a Gaussian-kernel local quadratic.
///
/// Calonico, Cattaneo & Titiunik (2014) correct the leading smoothing bias of an
/// order-`p` local-polynomial estimate of the `ν`-th derivative with a
/// higher-order fit at a pilot bandwidth `b`, and studentize by the variance of
/// the bias-corrected statistic rather than of the uncorrected one. With
/// `b = h` (`ρ = 1`) the bias-corrected estimator is numerically the matching
/// coefficient of the higher-order fit at `h`: the order-`p` regression of `Y`
/// minus the fitted higher powers returns the higher-order fit's lower
/// coefficients, because its weighted residuals are orthogonal to every lower
/// power. Its robust variance is therefore that fit's Eicker–White sandwich.
///
/// Which higher order removes the leading bias depends on the parity of
/// `p − ν`. The order-`p` bias is `h^{p+1−ν}·B_{p+1}·m^{(p+1)} +
/// h^{p+2−ν}·B_{p+2}·m^{(p+2)} + …`, and for a symmetric kernel in the interior
/// `B_{p+1}` vanishes when `p − ν` is even. For the local quadratic (`p = 2`):
///
/// - `ν = 1` (`p − ν` odd): the leading term involves `m'''`, so the correction
///   is a local cubic.
/// - `ν = 0` and `ν = 2` (`p − ν` even): the leading bias is `h^{4−ν}·m^{(4)}`,
///   which a local cubic does not remove (its level and second-derivative
///   biases are of the same order as the quadratic's). The correction is a
///   local quartic.
///
/// The resulting interval is centered at the bias-corrected coordinate, not at
/// the local-quadratic point estimate, and is wider than the conventional
/// interval: it pays for the bias estimate's variance instead of ignoring the
/// bias.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalPolynomialBiasCorrected {
    /// Bias-corrected level (local-quartic intercept).
    pub value: f64,
    /// Bias-corrected first derivative (local-cubic slope).
    pub first_derivative: f64,
    /// Bias-corrected second derivative (local-quartic curvature).
    pub second_derivative: f64,
    /// Heteroskedasticity-robust standard error of the bias-corrected level.
    pub robust_standard_error: f64,
    /// Heteroskedasticity-robust standard error of the bias-corrected first derivative.
    pub robust_first_derivative_standard_error: f64,
    /// Heteroskedasticity-robust standard error of the bias-corrected second derivative.
    pub robust_second_derivative_standard_error: f64,
}

/// Robust bias-corrected local-quadratic coordinates at `at` (see
/// [`LocalPolynomialBiasCorrected`]). `observation_weights`, when present,
/// multiply the kernel weights and must be finite and nonnegative.
///
/// # Errors
///
/// Shape mismatch, non-finite input, invalid bandwidth or weights, too little
/// effective kernel weight for five coefficients, or a singular local design.
pub fn gaussian_local_quadratic_bias_corrected(
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
    observation_weights: Option<&[f64]>,
) -> Result<LocalPolynomialBiasCorrected, StatsError> {
    if x.len() != y.len() || x.len() < 5 {
        return Err(StatsError::Shape {
            message: "local bias correction requires aligned x/y with n >= 5",
        });
    }
    if !at.is_finite()
        || !bandwidth.is_finite()
        || bandwidth <= 0.0
        || x.iter().chain(y).any(|v| !v.is_finite())
    {
        return Err(StatsError::Shape { message: "local bias-correction inputs must be finite" });
    }
    if let Some(weights) = observation_weights {
        if weights.len() != x.len() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(StatsError::Shape { message: "invalid local bias-correction weights" });
        }
    }
    let (cubic, cubic_se) = robust_local_polynomial::<4>(x, y, at, bandwidth, observation_weights)?;
    let (quartic, quartic_se) =
        robust_local_polynomial::<5>(x, y, at, bandwidth, observation_weights)?;
    // Derivative ν of a fitted polynomial in u = (x − at)/h is ν!·γ_ν / h^ν.
    let first_scale = 1.0 / bandwidth;
    let second_scale = 2.0 / (bandwidth * bandwidth);
    Ok(LocalPolynomialBiasCorrected {
        value: quartic[0],
        first_derivative: cubic[1] * first_scale,
        second_derivative: quartic[2] * second_scale,
        robust_standard_error: quartic_se[0],
        robust_first_derivative_standard_error: cubic_se[1] * first_scale,
        robust_second_derivative_standard_error: quartic_se[2] * second_scale,
    })
}

/// Gaussian-kernel local polynomial with `N` coefficients in the scaled power
/// basis `u = (x − at)/h`, returning the coefficients and their Eicker–White
/// sandwich standard errors. Inputs are validated by the caller.
fn robust_local_polynomial<const N: usize>(
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
    observation_weights: Option<&[f64]>,
) -> Result<([f64; N], [f64; N]), StatsError> {
    // Scaled powers keep the Gram well conditioned; the coefficient on u^k is
    // h^k times the coefficient on (x − at)^k.
    let mut gram = [[0.0; N]; N];
    let mut rhs = [0.0; N];
    let mut rows = Vec::with_capacity(x.len());
    let mut weight_sum = 0.0;
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        let u = (xi - at) / bandwidth;
        let w = (-0.5 * u * u).exp() * observation_weights.map_or(1.0, |w| w[i]);
        let mut row = [1.0; N];
        for k in 1..N {
            row[k] = row[k - 1] * u;
        }
        for j in 0..N {
            rhs[j] += w * row[j] * yi;
            for k in 0..N {
                gram[j][k] += w * row[j] * row[k];
            }
        }
        weight_sum += w;
        rows.push((w, row, yi));
    }
    if weight_sum <= N as f64 {
        return Err(StatsError::Backend(format!(
            "local response window has too little effective weight for a local polynomial of order {}",
            N - 1
        )));
    }
    let inverse = inverse_small(gram).ok_or(StatsError::SingularLocalDesign { order: N - 1 })?;
    let gamma: [f64; N] = std::array::from_fn(|j| (0..N).map(|k| inverse[j][k] * rhs[k]).sum());
    let mut influence_ss = [0.0; N];
    for &(w, row, yi) in &rows {
        let residual = yi - row.iter().zip(gamma).map(|(a, b)| a * b).sum::<f64>();
        for (k, ss) in influence_ss.iter_mut().enumerate() {
            let hat = inverse[k].iter().zip(row).map(|(c, v)| c * v).sum::<f64>();
            *ss += (w * hat * residual).powi(2);
        }
    }
    Ok((gamma, influence_ss.map(f64::sqrt)))
}

/// Gauss–Jordan inverse with partial pivoting for a small dense matrix.
fn inverse_small<const N: usize>(a: [[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut m = a;
    let mut out = [[0.0; N]; N];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let scale = a.iter().flatten().fold(0.0_f64, |acc, v| acc.max(v.abs())).max(f64::MIN_POSITIVE);
    for col in 0..N {
        let pivot = (col..N).max_by(|&r, &s| m[r][col].abs().total_cmp(&m[s][col].abs()))?;
        if !m[pivot][col].is_finite() || m[pivot][col].abs() <= f64::EPSILON * scale * 64.0 {
            return None;
        }
        m.swap(col, pivot);
        out.swap(col, pivot);
        let p = m[col][col];
        for k in 0..N {
            m[col][k] /= p;
            out[col][k] /= p;
        }
        for r in 0..N {
            if r != col {
                let factor = m[r][col];
                if factor != 0.0 {
                    for k in 0..N {
                        m[r][k] -= factor * m[col][k];
                        out[r][k] -= factor * out[col][k];
                    }
                }
            }
        }
    }
    Some(out)
}

fn matvec(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * b[j]).sum())
}

fn matmul3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn inverse_3x3(a: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    // Scale-free singularity: |det(G)| / ∏_j G_jj = |det(R)| for the column
    // correlation of the weighted local design. G_jj is the squared magnitude of
    // design column j, so the ratio is invariant under affine dose rescaling
    // (G → DGD with D = diag(1, α, α²)). The old threshold
    // `EPS * 64 * max(|a_ij|, 1)^3` is unit-dependent: continuous doses with
    // sd ≈ 1000 look singular and are refused as treatment_support_too_discrete.
    let scale = a[0][0].abs() * a[1][1].abs() * a[2][2].abs();
    if !det.is_finite() || scale <= 0.0 || det.abs() <= f64::EPSILON * scale * 64.0 {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    out[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    out[0][1] = a[0][2] * a[2][1] - a[0][1] * a[2][2];
    out[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    out[1][0] = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    out[1][1] = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    out[1][2] = a[0][2] * a[1][0] - a[0][0] * a[1][2];
    out[2][0] = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    out[2][1] = a[0][1] * a[2][0] - a[0][0] * a[2][1];
    out[2][2] = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    for row in &mut out {
        for value in row {
            *value /= det;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_quadratic_recovers_level_and_derivatives() {
        let x: Vec<f64> = (0..101).map(|i| -2.0 + 4.0 * f64::from(i) / 100.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v + 3.0 * v * v).collect();
        let fit = gaussian_local_quadratic(&x, &y, 0.4, 0.5).unwrap();
        assert!((fit.value - 2.28).abs() < 1e-10);
        assert!((fit.first_derivative - 4.4).abs() < 1e-10);
        assert!((fit.second_derivative - 6.0).abs() < 1e-10);
        assert!(fit.local_ess > 20.0);
    }

    #[test]
    fn two_point_regressor_is_a_typed_singular_local_design() {
        // A binary regressor gives the local quadratic two support points for
        // three coefficients, at every evaluation point and every bandwidth.
        let x: Vec<f64> = (0..200).map(|i| f64::from(i % 2)).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 1.5 * v).collect();
        for at in [0.0, 0.5, 1.0] {
            assert_eq!(
                gaussian_local_quadratic(&x, &y, at, 0.4).unwrap_err(),
                StatsError::SingularLocalDesign { order: 2 }
            );
            assert_eq!(
                gaussian_local_quadratic_bias_corrected(&x, &y, at, 0.4, None).unwrap_err(),
                StatsError::SingularLocalDesign { order: 3 }
            );
        }
    }

    #[test]
    fn local_quadratic_accepts_an_affine_rescaling_of_a_well_conditioned_dose() {
        // stats-a-1: a continuous dose with sd ≈ 1000 is an affine rescaling of a
        // unit-scale design. The singularity test must not refuse it.
        let x0: Vec<f64> = (0..101).map(|i| -2.0 + 4.0 * f64::from(i) / 100.0).collect();
        let y: Vec<f64> = x0.iter().map(|v| 1.0 + 2.0 * v + 3.0 * v * v).collect();
        let base = gaussian_local_quadratic(&x0, &y, 0.4, 0.5).unwrap();
        let (alpha, shift) = (1000.0, 5000.0);
        let x: Vec<f64> = x0.iter().map(|v| alpha * v + shift).collect();
        let fit = gaussian_local_quadratic(&x, &y, alpha * 0.4 + shift, alpha * 0.5).unwrap();
        assert!((fit.value - base.value).abs() < 1e-8);
        assert!((fit.first_derivative - base.first_derivative / alpha).abs() < 1e-10);
        assert!((fit.second_derivative - base.second_derivative / (alpha * alpha)).abs() < 1e-12);
    }

    #[test]
    fn local_quadratic_refuses_a_constant_dose_as_singular() {
        let x = vec![42.0; 100];
        let y: Vec<f64> = (0..100).map(|i| f64::from(i)).collect();
        assert_eq!(
            gaussian_local_quadratic(&x, &y, 42.0, 1.0).unwrap_err(),
            StatsError::SingularLocalDesign { order: 2 }
        );
    }

    #[test]
    fn bias_corrected_fit_is_exact_on_a_cubic_where_the_quadratic_is_biased() {
        let x: Vec<f64> = (0..201).map(|i| -2.0 + 4.0 * f64::from(i) / 200.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v + 0.5 * v * v + v.powi(3)).collect();
        let at = 0.3;
        let truth_first = 2.0 + at + 3.0 * at * at;
        let quadratic = gaussian_local_quadratic(&x, &y, at, 0.5).unwrap();
        let corrected = gaussian_local_quadratic_bias_corrected(&x, &y, at, 0.5, None).unwrap();
        assert!((quadratic.first_derivative - truth_first).abs() > 0.1);
        assert!((corrected.first_derivative - truth_first).abs() < 1e-9);
        assert!((corrected.second_derivative - (1.0 + 6.0 * at)).abs() < 1e-9);
        assert!((corrected.value - (1.0 + 2.0 * at + 0.5 * at * at + at.powi(3))).abs() < 1e-10);
    }

    #[test]
    fn second_derivative_correction_is_exact_on_a_quartic_where_the_cubic_is_biased() {
        // A local cubic leaves the second-derivative bias of the quadratic at the
        // same order (the m'''' term); the quartic correction removes it.
        let x: Vec<f64> = (0..201).map(|i| -2.0 + 4.0 * f64::from(i) / 200.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + v + v.powi(4)).collect();
        let (at, h) = (0.3, 0.5);
        let truth_second = 12.0 * at * at;
        let quadratic = gaussian_local_quadratic(&x, &y, at, h).unwrap();
        let (cubic, _) = robust_local_polynomial::<4>(&x, &y, at, h, None).unwrap();
        let cubic_second = 2.0 * cubic[2] / (h * h);
        let corrected = gaussian_local_quadratic_bias_corrected(&x, &y, at, h, None).unwrap();
        assert!((quadratic.second_derivative - truth_second).abs() > 0.5);
        assert!((cubic_second - truth_second).abs() > 0.5);
        assert!((corrected.second_derivative - truth_second).abs() < 1e-8);
        assert!((corrected.value - (1.0 + at + at.powi(4))).abs() < 1e-10);
    }

    /// Unscaled Gaussian-kernel local polynomial coefficients on (x − at)^k.
    fn unscaled_fit<const N: usize>(x: &[f64], y: &[f64], at: f64, h: f64) -> [f64; N] {
        let mut gram = [[0.0; N]; N];
        let mut rhs = [0.0; N];
        for (&xi, &yi) in x.iter().zip(y) {
            let dx = xi - at;
            let w = (-0.5 * (dx / h).powi(2)).exp();
            let row: [f64; N] = std::array::from_fn(|k| dx.powi(i32::try_from(k).unwrap()));
            for j in 0..N {
                rhs[j] += w * row[j] * yi;
                for k in 0..N {
                    gram[j][k] += w * row[j] * row[k];
                }
            }
        }
        let inv = inverse_small(gram).unwrap();
        std::array::from_fn(|j| (0..N).map(|k| inv[j][k] * rhs[k]).sum())
    }

    #[test]
    fn bias_corrected_fit_equals_the_two_step_correction_at_rho_one() {
        // CCT two-step: fit the higher-order coefficients at pilot bandwidth
        // b = h, remove their contribution, refit the quadratic. At ρ = 1 this is
        // the higher-order fit: cubic for the first derivative, quartic for the
        // level and the second derivative.
        let x: Vec<f64> = (0..301).map(|i| -3.0 + 6.0 * f64::from(i) / 300.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() * 2.0 + 0.3 * (1.9 * i as f64).cos())
            .collect();
        let (at, h) = (0.4, 0.6);
        let corrected = gaussian_local_quadratic_bias_corrected(&x, &y, at, h, None).unwrap();
        let cubic = unscaled_fit::<4>(&x, &y, at, h);
        let adjusted: Vec<f64> =
            x.iter().zip(&y).map(|(xi, yi)| yi - cubic[3] * (xi - at).powi(3)).collect();
        let two_step = gaussian_local_quadratic(&x, &adjusted, at, h).unwrap();
        assert!((two_step.first_derivative - corrected.first_derivative).abs() < 1e-8);
        let quartic = unscaled_fit::<5>(&x, &y, at, h);
        let adjusted: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| yi - quartic[3] * (xi - at).powi(3) - quartic[4] * (xi - at).powi(4))
            .collect();
        let two_step = gaussian_local_quadratic(&x, &adjusted, at, h).unwrap();
        assert!((two_step.second_derivative - corrected.second_derivative).abs() < 1e-7);
        assert!((two_step.value - corrected.value).abs() < 1e-9);
        assert!(corrected.robust_first_derivative_standard_error > 0.0);
        assert!(corrected.robust_second_derivative_standard_error > 0.0);
    }

    #[test]
    fn density_is_normalized_at_mean() {
        let got = gaussian_density(2.0, 2.0, 0.5);
        let expected = 1.0 / (0.5 * (2.0 * std::f64::consts::PI).sqrt());
        assert!((got - expected).abs() < 1e-14);
    }

    #[test]
    fn plugin_level_standard_error_matches_the_sandwich_not_the_bread() {
        // Homoskedastic noise: the plug-in sandwich and the influence-based robust
        // standard error estimate the same quantity and must agree closely. The
        // bread-only form `sigma^2 (X'WX)^-1` is systematically wider.
        let x: Vec<f64> = (0..500).map(|i| -2.0 + 4.0 * f64::from(i) / 499.0).collect();
        let y: Vec<f64> =
            x.iter().enumerate().map(|(i, v)| 1.0 + 2.0 * v + 0.2 * (i as f64).sin()).collect();
        let fit = gaussian_local_quadratic_influence(&x, &y, 0.0, 0.5).unwrap();
        let plugin = fit.point.standard_error;
        let robust = fit.robust_standard_error;
        assert!(plugin > 0.0 && robust > 0.0);
        assert!(
            (plugin - robust).abs() / robust < 0.05,
            "plugin={plugin} robust={robust} — sandwich and robust SEs should agree"
        );
    }

    #[test]
    fn derivative_robust_standard_errors_use_rowwise_squared_residuals() {
        // Deliberately heteroskedastic residuals. A common sigma² multiplied by
        // X'W²X is not the Eicker-White meat for a derivative; every row must
        // carry its own residual².
        let x: Vec<f64> = (0..401).map(|i| -2.0 + 4.0 * f64::from(i) / 400.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let scale = 0.02 + value.abs();
                1.0 + 2.0 * value + scale * (1.7 * i as f64).sin()
            })
            .collect();
        let at = 0.1;
        let bandwidth = 0.65;
        let fit = gaussian_local_quadratic_influence(&x, &y, at, bandwidth).unwrap();

        let mut gram = [[0.0; 3]; 3];
        for &xi in &x {
            let dx = xi - at;
            let w = (-0.5 * (dx / bandwidth).powi(2)).exp();
            let row = [1.0, dx, dx * dx];
            for j in 0..3 {
                for k in 0..3 {
                    gram[j][k] += w * row[j] * row[k];
                }
            }
        }
        let inverse = inverse_3x3(gram).unwrap();
        let beta = [fit.point.value, fit.point.first_derivative, fit.point.second_derivative / 2.0];
        let mut first_ss = 0.0;
        let mut second_ss = 0.0;
        for (&xi, &yi) in x.iter().zip(&y) {
            let dx = xi - at;
            let w = (-0.5 * (dx / bandwidth).powi(2)).exp();
            let row = [1.0, dx, dx * dx];
            let residual = yi - row.iter().zip(beta).map(|(a, b)| a * b).sum::<f64>();
            let first_hat = inverse[1].iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
            let second_hat = inverse[2].iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
            first_ss += (w * first_hat * residual).powi(2);
            second_ss += (2.0 * w * second_hat * residual).powi(2);
        }
        assert!((fit.robust_first_derivative_standard_error - first_ss.sqrt()).abs() < 1e-12);
        assert!((fit.robust_second_derivative_standard_error - second_ss.sqrt()).abs() < 1e-12);
        assert!(
            (fit.robust_first_derivative_standard_error
                - fit.point.first_derivative_standard_error)
                .abs()
                > 1e-5,
            "heteroskedastic robust SE must not collapse to the common-sigma plug-in"
        );
    }

    #[test]
    fn local_quadratic_refuses_a_window_without_enough_effective_weight() {
        let x = vec![-10.0, 0.0, 10.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(gaussian_local_quadratic_influence(&x, &y, 0.0, 0.05).is_err());
    }

    #[test]
    fn local_quadratic_influences_are_centered_and_finite() {
        let x: Vec<f64> = (0..101).map(|i| -2.0 + 4.0 * f64::from(i) / 100.0).collect();
        let y: Vec<f64> =
            x.iter().enumerate().map(|(i, value)| 1.0 + value + 0.1 * (i as f64).sin()).collect();
        let fit = gaussian_local_quadratic_influence(&x, &y, 0.0, 0.5).unwrap();
        assert_eq!(fit.influences.len(), x.len());
        assert!(fit.robust_standard_error.is_finite() && fit.robust_standard_error > 0.0);
        assert!(fit.influences.iter().sum::<f64>().abs() < 1e-10);
    }
    #[test]
    fn weighted_local_quadratic_matches_repeated_rows() {
        let x: Vec<f64> = (0..30).map(|i| f64::from(i) / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let weights: Vec<f64> = (0..30).map(|i| f64::from(i % 3)).collect();
        let mut xr = Vec::new();
        let mut yr = Vec::new();
        for i in 0..30 {
            for _ in 0..i % 3 {
                xr.push(x[i]);
                yr.push(y[i]);
            }
        }
        let a = gaussian_local_quadratic_weighted(&x, &y, 1.2, 0.6, &weights).unwrap();
        let b = gaussian_local_quadratic(&xr, &yr, 1.2, 0.6).unwrap();
        assert!((a.first_derivative - b.first_derivative).abs() < 1e-12);
        assert!((a.second_derivative - b.second_derivative).abs() < 1e-12);
    }
}
