//! Independent lag-aligned OLS g-computation oracle for temporal response pins.
//!
//! Recomputes a completion's point response from raw columns, a named adjustment
//! set, and plain normal equations — none of the estimator's design, lag-alignment
//! or linear-algebra code is reused — so a pin against it is a numeric comparison,
//! not a shape check.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, clippy::cast_precision_loss, clippy::cast_sign_loss)]

/// What one response cell evaluates.
#[derive(Clone, Copy, Debug)]
pub enum Eval {
    /// Treatment fixed at a dose.
    Dose(f64),
    /// Treatment at its lag-aligned sample mean plus a shift.
    Shift(f64),
}

/// `E_n[Y_{o+h} | do(T_o = ·)]` by OLS of the lag-aligned outcome on the treatment and
/// `adjustment` (`(column, offset)` pairs, offsets relative to the policy origin),
/// standardized over the sample average of the adjustment columns.
///
/// `columns[v]` is variable `v`'s series. The outcome sits at offset
/// `treatment_offset + horizon`; each column's lag is measured from that outcome anchor,
/// and rows run over every anchor time at which all lags exist.
#[must_use]
pub fn lagged_ols_level(
    columns: &[&[f64]],
    outcome: usize,
    treatment: usize,
    treatment_offset: i32,
    horizon: u32,
    adjustment: &[(usize, i32)],
    eval: Eval,
) -> f64 {
    let anchor = treatment_offset + i32::try_from(horizon).unwrap();
    let lag_of = |offset: i32| -> usize {
        usize::try_from(anchor - offset).expect("adjustment after the outcome anchor")
    };
    let mut regressors = vec![(treatment, lag_of(treatment_offset))];
    regressors.extend(adjustment.iter().map(|&(column, offset)| (column, lag_of(offset))));
    let max_lag = regressors.iter().map(|&(_, lag)| lag).max().unwrap_or(0);
    let n = columns[outcome].len();
    let p = regressors.len() + 1;
    let rows: Vec<usize> = (max_lag..n).collect();
    let design = |s: usize, j: usize| -> f64 {
        if j == 0 {
            1.0
        } else {
            let (column, lag) = regressors[j - 1];
            columns[column][s - lag]
        }
    };
    let mut gram = vec![vec![0.0; p]; p];
    let mut cross = vec![0.0; p];
    let mut means = vec![0.0; p];
    for &s in &rows {
        for a in 0..p {
            let xa = design(s, a);
            means[a] += xa / rows.len() as f64;
            cross[a] += xa * columns[outcome][s];
            for b in 0..p {
                gram[a][b] += xa * design(s, b);
            }
        }
    }
    let beta = solve(gram, cross);
    let treatment_value = match eval {
        Eval::Dose(dose) => dose,
        Eval::Shift(shift) => means[1] + shift,
    };
    (0..p).map(|j| beta[j] * if j == 1 { treatment_value } else { means[j] }).sum()
}

/// Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let p = b.len();
    for col in 0..p {
        let pivot = (col..p).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        assert!(a[col][col].abs() > 1e-12, "singular lagged OLS design");
        for row in col + 1..p {
            let factor = a[row][col] / a[col][col];
            for k in col..p {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; p];
    for row in (0..p).rev() {
        let tail: f64 = (row + 1..p).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}
