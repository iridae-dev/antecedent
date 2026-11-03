//! Independent known-truth checks for the local-polynomial ratio rows (fuzzy jump and
//! regression kink).
//!
//! Two kinds of evidence, neither calling the production solver to produce its own expected
//! value:
//!
//! * Exact truth. Outcome and treatment are exact polynomials of the fitted order on each side
//!   plus a known jump (or slope change), so the bias-corrected one-sided coefficients, the
//!   ratio and its zero residual variance are analytic.
//! * Minimal reimplementation. The robust-bias-corrected coefficient of a local order-`p`
//!   polynomial with the order-`p + 1` pilot equals the corresponding coefficient of the
//!   order-`p + 1` triangular-kernel weighted least squares fit on that side (Frisch-Waugh).
//!   This test fits that polynomial with its own Gauss-Jordan solver, forms the HC0 sandwich
//!   `A^-1 (sum w^2 r^2 x x') A^-1` explicitly, and combines the two sides and the two
//!   equations with the delta method for the ratio, then compares estimate, first stage,
//!   reduced form, standard error and the normal interval to the library output.
//!
//! The repeated-sampling coverage tests are calibration evidence and are a separate claim.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::needless_range_loop, reason = "small dense linear algebra written by index")]

use antecedent_estimate::local_polynomial_ratio::fit_local_polynomial_ratio;

const Z975: f64 = 1.959_963_984_540_054;

fn spread(key: u64) -> f64 {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn rel_close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * (1.0 + want.abs()), "{what}: got {got:e}, want {want:e}");
}

/// Inverse by Gauss-Jordan elimination with partial pivoting.
fn invert(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|&x, &y| m[x][col].abs().total_cmp(&m[y][col].abs())).unwrap();
        m.swap(col, pivot);
        let d = m[col][col];
        assert!(d.abs() > 1e-300, "singular design");
        for j in 0..2 * n {
            m[col][j] /= d;
        }
        for i in 0..n {
            if i != col {
                let f = m[i][col];
                for j in 0..2 * n {
                    m[i][j] -= f * m[col][j];
                }
            }
        }
    }
    m.into_iter().map(|row| row[n..].to_vec()).collect()
}

struct Side {
    coef_y: Vec<f64>,
    coef_t: Vec<f64>,
    // HC0 sandwich entries of coefficient `j`, filled for the requested j.
    var_y: f64,
    var_t: f64,
    cov: f64,
}

/// Triangular-kernel weighted polynomial fit of `degree` on one side, HC0 sandwich of
/// coefficient `j`. Scaled distance `d = (x - cutoff) / h`; the window is `|d| < 1` with the
/// cutoff on the right side.
fn side_fit(
    x: &[f64],
    y: &[f64],
    t: &[f64],
    cutoff: f64,
    h: f64,
    right: bool,
    degree: usize,
    j: usize,
) -> Side {
    let k = degree + 1;
    let rows: Vec<(Vec<f64>, f64, f64, f64)> = (0..x.len())
        .filter_map(|i| {
            let d = (x[i] - cutoff) / h;
            let inside = d.abs() < 1.0 && if right { d >= 0.0 } else { d < 0.0 };
            inside.then(|| {
                let basis: Vec<f64> = (0..k).map(|p| d.powi(p as i32)).collect();
                (basis, 1.0 - d.abs(), y[i], t[i])
            })
        })
        .collect();
    let mut gram = vec![vec![0.0; k]; k];
    let (mut ry, mut rt) = (vec![0.0; k], vec![0.0; k]);
    for (b, w, yy, tt) in &rows {
        for r in 0..k {
            ry[r] += w * b[r] * yy;
            rt[r] += w * b[r] * tt;
            for c in 0..k {
                gram[r][c] += w * b[r] * b[c];
            }
        }
    }
    let inv = invert(&gram);
    let solve = |rhs: &[f64]| -> Vec<f64> {
        (0..k).map(|r| (0..k).map(|c| inv[r][c] * rhs[c]).sum()).collect()
    };
    let (coef_y, coef_t) = (solve(&ry), solve(&rt));
    let mut meat = [vec![vec![0.0; k]; k], vec![vec![0.0; k]; k], vec![vec![0.0; k]; k]];
    for (b, w, yy, tt) in &rows {
        let e_y = yy - (0..k).map(|p| coef_y[p] * b[p]).sum::<f64>();
        let e_t = tt - (0..k).map(|p| coef_t[p] * b[p]).sum::<f64>();
        for r in 0..k {
            for c in 0..k {
                let s = w * w * b[r] * b[c];
                meat[0][r][c] += s * e_y * e_y;
                meat[1][r][c] += s * e_t * e_t;
                meat[2][r][c] += s * e_y * e_t;
            }
        }
    }
    let entry = |m: &Vec<Vec<f64>>| -> f64 {
        let mut total = 0.0;
        for a in 0..k {
            for b in 0..k {
                total += inv[j][a] * m[a][b] * inv[b][j];
            }
        }
        total
    };
    Side { coef_y, coef_t, var_y: entry(&meat[0]), var_t: entry(&meat[1]), cov: entry(&meat[2]) }
}

struct Reference {
    estimate: f64,
    reduced: f64,
    first: f64,
    se: f64,
}

fn reference(x: &[f64], y: &[f64], t: &[f64], cutoff: f64, h: f64, kink: bool) -> Reference {
    // Jump: the intercept of the cubic fit. Kink: the slope of the quartic fit, per unit of x.
    let (degree, j, scale) = if kink { (4, 1, 1.0 / h) } else { (3, 0, 1.0) };
    let left = side_fit(x, y, t, cutoff, h, false, degree, j);
    let right = side_fit(x, y, t, cutoff, h, true, degree, j);
    let reduced = (right.coef_y[j] - left.coef_y[j]) * scale;
    let first = (right.coef_t[j] - left.coef_t[j]) * scale;
    let (vy, vt, vyt) = (
        (left.var_y + right.var_y) * scale * scale,
        (left.var_t + right.var_t) * scale * scale,
        (left.cov + right.cov) * scale * scale,
    );
    let estimate = reduced / first;
    let variance = (vy + estimate * estimate * vt - 2.0 * estimate * vyt) / (first * first);
    Reference { estimate, reduced, first, se: variance.max(0.0).sqrt() }
}

fn grid() -> Vec<f64> {
    // Dyadic grid on (-1, 1) with the cutoff itself on the treated side.
    (-63..64).map(|k| f64::from(k) / 64.0).collect()
}

fn exact_poly(x: f64, c: &[f64]) -> f64 {
    c.iter().enumerate().map(|(p, a)| a * x.powi(p as i32)).sum()
}

#[test]
fn fuzzy_jump_exact_cubic_truth_recovers_the_ratio_with_zero_variance() {
    let x = grid();
    let y_base = [1.0, 2.0, 0.5, -0.7];
    let t_base = [0.25, 0.1, -0.2, 0.05];
    let jump_t = 0.5; // first-stage discontinuity
    let effect = 2.0; // local effect: outcome jump is effect * jump_t
    let t: Vec<f64> =
        x.iter().map(|&v| exact_poly(v, &t_base) + if v >= 0.0 { jump_t } else { 0.0 }).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&t)
        .map(|(&v, _)| exact_poly(v, &y_base) + if v >= 0.0 { effect * jump_t } else { 0.0 })
        .collect();
    let fit = fit_local_polynomial_ratio(&x, &y, &t, 0.0, 1.0, false).unwrap();
    rel_close(fit.first_stage, jump_t, 1e-9, "first stage equals the planted treatment jump");
    rel_close(
        fit.reduced_form,
        effect * jump_t,
        1e-9,
        "reduced form equals the planted outcome jump",
    );
    rel_close(fit.estimate, effect, 1e-9, "ratio equals the planted local effect");
    assert!(fit.standard_error < 1e-6, "exact polynomial has no residual variance");
}

#[test]
fn regression_kink_exact_quartic_truth_recovers_the_slope_ratio_with_zero_variance() {
    let x = grid();
    let h = 1.0;
    let y_base = [0.5, -1.0, 0.8, 0.3, -0.6];
    let t_base = [0.2, 0.4, -0.3, 0.1, 0.2];
    let kink_t = 0.4; // first-stage slope change
    let effect = 3.0; // outcome slope change is effect * kink_t
    let t: Vec<f64> = x.iter().map(|&v| exact_poly(v, &t_base) + kink_t * v.max(0.0)).collect();
    let y: Vec<f64> =
        x.iter().map(|&v| exact_poly(v, &y_base) + effect * kink_t * v.max(0.0)).collect();
    let fit = fit_local_polynomial_ratio(&x, &y, &t, 0.0, h, true).unwrap();
    rel_close(fit.first_stage, kink_t, 1e-8, "first-stage slope change");
    rel_close(fit.reduced_form, effect * kink_t, 1e-8, "outcome slope change");
    rel_close(fit.estimate, effect, 1e-8, "ratio equals the planted kink effect");
    assert!(fit.standard_error < 1e-5, "exact polynomial has no residual variance");
}

#[test]
fn fuzzy_jump_estimate_standard_error_and_interval_match_an_independent_fit() {
    let x = grid();
    let t: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| 0.3 + 0.4 * v + if v >= 0.0 { 0.5 } else { 0.0 } + 0.08 * spread(i as u64))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            1.0 + 2.0 * v
                + 0.5 * v * v
                + 2.0 * (if v >= 0.0 { 0.5 } else { 0.0 })
                + 0.1 * spread(1_000 + i as u64)
        })
        .collect();
    for (cutoff_shift, h) in [(0.0, 1.0), (0.0, 0.8)] {
        let fit = fit_local_polynomial_ratio(&x, &y, &t, cutoff_shift, h, false).unwrap();
        let want = reference(&x, &y, &t, cutoff_shift, h, false);
        rel_close(fit.first_stage, want.first, 1e-8, "first stage");
        rel_close(fit.reduced_form, want.reduced, 1e-8, "reduced form");
        rel_close(fit.estimate, want.estimate, 1e-8, "ratio");
        rel_close(fit.standard_error, want.se, 1e-7, "delta-method HC0 standard error");
        rel_close(fit.ci_lower, want.estimate - Z975 * want.se, 1e-7, "interval lower");
        rel_close(fit.ci_upper, want.estimate + Z975 * want.se, 1e-7, "interval upper");
        assert!(want.se > 0.0);
    }
}

#[test]
fn regression_kink_estimate_standard_error_and_interval_match_an_independent_fit() {
    let x = grid();
    let t: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| 0.2 + 0.3 * v + 0.4 * v.max(0.0) + 0.004 * spread(7_000 + i as u64))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            0.5 - v + 0.8 * v * v + 3.0 * 0.4 * v.max(0.0) + 0.008 * spread(9_000 + i as u64)
        })
        .collect();
    for h in [1.0, 0.75] {
        let fit = fit_local_polynomial_ratio(&x, &y, &t, 0.0, h, true).unwrap();
        let want = reference(&x, &y, &t, 0.0, h, true);
        rel_close(fit.first_stage, want.first, 1e-7, "first-stage slope change");
        rel_close(fit.reduced_form, want.reduced, 1e-7, "outcome slope change");
        rel_close(fit.estimate, want.estimate, 1e-7, "ratio");
        rel_close(fit.standard_error, want.se, 1e-6, "delta-method HC0 standard error");
        rel_close(fit.ci_lower, want.estimate - Z975 * want.se, 1e-6, "interval lower");
        rel_close(fit.ci_upper, want.estimate + Z975 * want.se, 1e-6, "interval upper");
        assert!(want.se > 0.0);
    }
}
