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
    if x.len() != y.len() || x.len() < 3 {
        return Err(StatsError::Shape {
            message: "local quadratic requires aligned x/y with n >= 3",
        });
    }
    if !at.is_finite()
        || !bandwidth.is_finite()
        || bandwidth <= 0.0
        || x.iter().chain(y).any(|v| !v.is_finite())
    {
        return Err(StatsError::Shape { message: "local quadratic inputs must be finite" });
    }
    let mut gram = [[0.0; 3]; 3];
    let mut squared_gram = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    let mut weights = Vec::with_capacity(x.len());
    let mut weight_sum = 0.0;
    let mut weight_sq_sum = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - at;
        let z = dx / bandwidth;
        let w = (-0.5 * z * z).exp();
        let row = [1.0, dx, dx * dx];
        for j in 0..3 {
            rhs[j] += w * row[j] * yi;
            for k in 0..3 {
                gram[j][k] += w * row[j] * row[k];
                squared_gram[j][k] += w * w * row[j] * row[k];
            }
        }
        weights.push((w, row, yi));
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
    let inverse = inverse_3x3(gram)
        .ok_or_else(|| StatsError::Backend("singular local response design".into()))?;
    let beta = matvec(inverse, rhs);
    let mut weighted_rss = 0.0;
    for (w, row, yi) in weights {
        let residual = yi - row.iter().zip(beta).map(|(a, b)| a * b).sum::<f64>();
        weighted_rss += w * residual * residual;
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
    let influences: Vec<f64> = weights_for_influence(x, y, at, bandwidth, beta, inverse);
    let robust_standard_error = influences.iter().map(|value| value * value).sum::<f64>().sqrt();
    Ok(LocalPolynomialInfluence { point, influences, robust_standard_error })
}

fn weights_for_influence(
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
    beta: [f64; 3],
    inverse: [[f64; 3]; 3],
) -> Vec<f64> {
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| {
            let dx = xi - at;
            let row = [1.0, dx, dx * dx];
            let weight = (-0.5 * (dx / bandwidth).powi(2)).exp();
            let residual = yi - row.iter().zip(beta).map(|(a, b)| a * b).sum::<f64>();
            weight
                * inverse[0]
                    .iter()
                    .zip(row)
                    .map(|(coefficient, value)| coefficient * value)
                    .sum::<f64>()
                * residual
        })
        .collect()
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
    let scale = a.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    if !det.is_finite() || det.abs() <= f64::EPSILON * scale.powi(3) * 64.0 {
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
}
