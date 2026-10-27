//! Gaussian-kernel local quadratic regression against independent dense algebra.
//!
//! * The fitted level, slope and curvature solve the kernel-weighted normal equations built and
//!   solved here by elimination, and recover an exact quadratic exactly.
//! * The level influences sum to zero, and their sum of squares is the robust variance.
//! * For homoskedastic uncorrelated noise the plug-in level variance is unbiased. The expectation
//!   of the weighted residual sum of squares is a quadratic form in the noise, so it is the sum
//!   over unit-vector responses; the summed plug-in variances must then equal the exact sandwich
//!   `e0'(X'WX)^-1 X'W^2 X (X'WX)^-1 e0`. Fractional Gaussian weights make `sum(w) - 3` the wrong
//!   residual divisor; `sum(w) - tr((X'WX)^-1 X'W^2 X)` is the right one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{LocalPolynomialPoint, StatsError, gaussian_local_quadratic_influence};

fn gaussian_local_quadratic(
    x: &[f64],
    y: &[f64],
    at: f64,
    bandwidth: f64,
) -> Result<LocalPolynomialPoint, StatsError> {
    Ok(gaussian_local_quadratic_influence(x, y, at, bandwidth)?.point)
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let tail: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - tail) / a[r][r];
    }
    x
}

fn grid(n: u32) -> Vec<f64> {
    (0..n).map(|i| -2.0 + 4.0 * (f64::from(i) + 0.37) / f64::from(n)).collect()
}

fn weighted_normal_equations(x: &[f64], y: &[f64], at: f64, h: f64) -> Vec<f64> {
    let mut gram = vec![vec![0.0; 3]; 3];
    let mut rhs = vec![0.0; 3];
    for (&xi, &yi) in x.iter().zip(y) {
        let d = xi - at;
        let w = (-0.5 * (d / h).powi(2)).exp();
        let row = [1.0, d, d * d];
        for j in 0..3 {
            rhs[j] += w * row[j] * yi;
            for k in 0..3 {
                gram[j][k] += w * row[j] * row[k];
            }
        }
    }
    solve_dense(gram, rhs)
}

#[test]
fn point_estimate_solves_the_weighted_normal_equations_and_recovers_a_quadratic() {
    let x = grid(60);
    // Exact quadratic: level, slope and curvature are recovered at any bandwidth.
    let exact: Vec<f64> = x.iter().map(|v| 0.7 - 1.1 * v + 0.45 * v * v).collect();
    let at = 0.4;
    let fit = gaussian_local_quadratic(&x, &exact, at, 0.6).unwrap();
    assert!((fit.value - (0.7 - 1.1 * at + 0.45 * at * at)).abs() < 1e-10);
    assert!((fit.first_derivative - (-1.1 + 0.9 * at)).abs() < 1e-10);
    assert!((fit.second_derivative - 0.9).abs() < 1e-9);
    // Non-polynomial response: compare with the independent solve.
    let y: Vec<f64> = x.iter().map(|v| (2.0 * v).sin() + 0.3 * (7.0 * v).cos()).collect();
    for (at, h) in [(0.0, 0.5), (1.1, 0.4), (-0.7, 0.8)] {
        let beta = weighted_normal_equations(&x, &y, at, h);
        let fit = gaussian_local_quadratic(&x, &y, at, h).unwrap();
        assert!((fit.value - beta[0]).abs() < 1e-10, "level at {at}");
        assert!((fit.first_derivative - beta[1]).abs() < 1e-9, "slope at {at}");
        assert!((fit.second_derivative - 2.0 * beta[2]).abs() < 1e-8, "curvature at {at}");
    }
}

#[test]
fn level_influences_sum_to_zero_and_square_to_the_robust_variance() {
    let x = grid(50);
    let y: Vec<f64> = x.iter().map(|v| (3.0 * v).sin() + 0.1 * v).collect();
    let fit = gaussian_local_quadratic_influence(&x, &y, 0.2, 0.5).unwrap();
    let total: f64 = fit.influences.iter().sum();
    assert!(total.abs() < 1e-12, "{total}");
    let ss: f64 = fit.influences.iter().map(|v| v * v).sum();
    assert!((ss.sqrt() - fit.robust_standard_error).abs() < 1e-13);
    assert!((fit.coefficient_covariance[0][0] - ss).abs() < 1e-13);
}

#[test]
fn plug_in_level_variance_is_unbiased_under_homoskedastic_noise() {
    let n = 40_u32;
    let x = grid(n);
    for (at, h) in [(0.0, 0.45), (0.8, 0.7), (-1.2, 0.35)] {
        let w: Vec<f64> = x.iter().map(|&v| (-0.5 * ((v - at) / h).powi(2)).exp()).collect();
        let mut gram = vec![vec![0.0; 3]; 3];
        let mut gram2 = [[0.0; 3]; 3];
        for (&v, wi) in x.iter().zip(&w) {
            let d = v - at;
            let r = [1.0, d, d * d];
            for j in 0..3 {
                for k in 0..3 {
                    gram[j][k] += wi * r[j] * r[k];
                    gram2[j][k] += wi * wi * r[j] * r[k];
                }
            }
        }
        // c = (X'WX)^-1 e0; the exact level sandwich is c' (X'W^2 X) c.
        let c = solve_dense(gram, vec![1.0, 0.0, 0.0]);
        let sandwich: f64 =
            (0..3).map(|j| (0..3).map(|k| c[j] * gram2[j][k] * c[k]).sum::<f64>()).sum();
        let mut summed = 0.0;
        for j in 0..n as usize {
            let mut y = vec![0.0; n as usize];
            y[j] = 1.0;
            let se = gaussian_local_quadratic(&x, &y, at, h).unwrap().standard_error;
            summed += se * se;
        }
        assert!(
            (summed - sandwich).abs() <= 1e-9 * sandwich,
            "at {at}, h {h}: expected plug-in variance {summed} to equal the sandwich {sandwich}"
        );
    }
}
