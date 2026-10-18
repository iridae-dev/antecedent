//! Shared numeric assertions for the artifact integration tests.
//!
//! Tolerances stay with the caller. These checks do not read or write an artifact.
//! Each integration test is its own crate and uses one of the two checks.

#![allow(dead_code)]

/// Absolute difference at most `tolerance`.
pub fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

/// Absolute difference strictly below `tolerance`.
pub fn close_below(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance, "{actual} != {expected}");
}

/// Gauss-Jordan inverse of a `p x p` row-major matrix (the oracle's own solver).
pub fn invert(matrix: &[f64], p: usize) -> Vec<f64> {
    let mut a: Vec<f64> = matrix.to_vec();
    let mut inverse: Vec<f64> = (0..p * p).map(|k| f64::from(k / p == k % p)).collect();
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&i, &j| a[i * p + col].abs().total_cmp(&a[j * p + col].abs()))
            .unwrap();
        for k in 0..p {
            a.swap(col * p + k, pivot * p + k);
            inverse.swap(col * p + k, pivot * p + k);
        }
        let scale = a[col * p + col];
        for k in 0..p {
            a[col * p + k] /= scale;
            inverse[col * p + k] /= scale;
        }
        for row in (0..p).filter(|&row| row != col) {
            let factor = a[row * p + col];
            for k in 0..p {
                a[row * p + k] -= factor * a[col * p + k];
                inverse[row * p + k] -= factor * inverse[col * p + k];
            }
        }
    }
    inverse
}

/// Independent HC0 to HC3 sandwich covariance (`kind` is `hc0` .. `hc3`) of the full coefficient
/// vector of the least-squares fit of `y` on `columns` (the design columns, intercept included),
/// straight from the textbook definitions with a dense inverse of `X'X`. Row-major `p x p`.
pub fn dense_sandwich(columns: &[Vec<f64>], y: &[f64], kind: &str) -> Vec<f64> {
    let n = y.len();
    let p = columns.len();
    let row = |i: usize| -> Vec<f64> { columns.iter().map(|c| c[i]).collect() };
    let mut gram = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let x = row(i);
        for a in 0..p {
            xty[a] += x[a] * y[i];
            for b in 0..p {
                gram[a * p + b] += x[a] * x[b];
            }
        }
    }
    let inverse = invert(&gram, p);
    let beta: Vec<f64> =
        (0..p).map(|a| (0..p).map(|b| inverse[a * p + b] * xty[b]).sum()).collect();
    let mut meat = vec![0.0; p * p];
    for i in 0..n {
        let x = row(i);
        let e = y[i] - x.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
        let h: f64 =
            (0..p).map(|a| x[a] * (0..p).map(|b| inverse[a * p + b] * x[b]).sum::<f64>()).sum();
        let weight = match kind {
            "hc0" => 1.0,
            "hc1" => n as f64 / (n - p) as f64,
            "hc2" => 1.0 / (1.0 - h),
            _ => 1.0 / ((1.0 - h) * (1.0 - h)),
        };
        for a in 0..p {
            for b in 0..p {
                meat[a * p + b] += weight * e * e * x[a] * x[b];
            }
        }
    }
    let mut left = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            left[a * p + b] = (0..p).map(|c| inverse[a * p + c] * meat[c * p + b]).sum();
        }
    }
    let mut full = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            full[a * p + b] = (0..p).map(|c| left[a * p + c] * inverse[c * p + b]).sum();
        }
    }
    full
}

/// The block of a dense `p x p` covariance at `offset..offset + k`, symmetrized, compared with
/// `actual` (row-major `k x k`) to a relative `1e-9`.
pub fn assert_block_matches(actual: &[f64], full: &[f64], p: usize, offset: usize, label: &str) {
    let k = actual.len().isqrt();
    assert_eq!(k * k, actual.len(), "{label}: a square block");
    for i in 0..k {
        for j in 0..k {
            let expected =
                0.5 * (full[(offset + i) * p + offset + j] + full[(offset + j) * p + offset + i]);
            assert!(
                (actual[i * k + j] - expected).abs() <= 1e-9 * expected.abs().max(1e-12),
                "{label} V[{i}][{j}]: {} vs {expected}",
                actual[i * k + j]
            );
        }
    }
}
