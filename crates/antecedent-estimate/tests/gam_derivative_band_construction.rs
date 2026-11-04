//! Independent construction check of the pointwise band of the additive-GAM plug-in Jacobian.
//!
//! The identity-scale Jacobian of a continuous response is published with a pointwise band
//! `g +/- z se`, where `g = f'(a)` is the derivative of the treatment smooth of the penalized
//! additive outcome fit, and `se` is the coefficient-sandwich standard error with
//! leverage-corrected (HC2) residuals. The band conditions on the fixed basis and the adjustment
//! penalty. This test recomputes the whole construction by hand on a small fixed design.
//!
//! With four basis functions per smooth the clamped cubic B-spline basis has no interior knot, so
//! it is the cubic Bernstein basis on `[min, max]` of its column: `B_k(u) = C(3,k) u^k (1-u)^(3-k)`
//! with `u = (x - min) / (max - min)`, and `B_k'(x) = 3 (b_{k-1}(u) - b_k(u)) / (max - min)` with
//! `b` the quadratic Bernstein basis. The reference therefore needs no knot or recursion code from
//! the product. The design is `[1 | T0 T1 T2 | X0 X1 X2]` (the last basis of each smooth is dropped:
//! it is aliased with the intercept by the partition of unity), the treatment smooth is
//! unpenalized and the adjustment smooth carries `lambda D2'D2` restricted to the kept
//! coefficients, with `D2` the second-difference matrix of the four coefficients. The reference then:
//!
//! 1. solves `(X'X + S) beta = X'y` by Gaussian elimination with partial pivoting,
//! 2. takes the residuals `e = y - X beta` and the leverages `h_i = x_i'(X'X + S)^-1 x_i`, and
//!    divides each residual by `sqrt(1 - h_i)`,
//! 3. forms the influence `psi_i = n e_i x_i'(X'X + S)^-1 c` of `c'beta`, with `c` the treatment
//!    basis derivatives at the evaluation point (the dropped basis contributes nothing: the
//!    derivatives sum to zero), centres it and sets `se = sqrt(sum psi^2) / n`,
//! 4. forms `g +/- z se` with `z = 1.959963984540054`, the standard normal 97.5% quantile
//!    (tabulated, not computed by the product's quantile routine).
//!
//! The design has 24 rows (the estimator requires at least 20) and seven coefficients, so the mean
//! leverage is 7/24 and the HC2 inflation of a typical residual is about 1/sqrt(1 - 7/24) = 1.19;
//! the half-width ratio is asserted above 1.05, well under that, and far above the 10^-5 tolerance.
//!
//! The published point is also compared, which ties the reference coefficients to the product's
//! fit. Tolerance: the product solves the same penalized least-squares problem by a conditioned
//! direct solve or a backfit to a relative 1e-10; the Gram matrix here has condition well below
//! 10^4, so coefficients agree to about 10^-8 and the point to 10^-6 relative. The half-width is a
//! smooth function of the residuals and leverages and is compared to 10^-5 relative. The
//! construction is discriminated well above that: the same design with uncorrected (HC0)
//! residuals moves the half-width by more than a factor of 1.05, which the test asserts, so a
//! different leverage correction cannot pass.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "integration test builds a deterministic fixture from small counts"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, DerivativeScale, IdentificationStatus, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::response::ContinuousResponseEstimator;

const N: usize = 24;
const LAMBDA: f64 = 0.5;
/// Standard normal 97.5% quantile.
const Z_975: f64 = 1.959_963_984_540_054;

type Basis3 = [f64; 3];
type Basis4 = [f64; 4];

/// Quadratic Bernstein basis `b_0..b_2` at `u`.
fn quadratic(u: f64) -> Basis3 {
    [(1.0 - u) * (1.0 - u), 2.0 * u * (1.0 - u), u * u]
}

/// Cubic Bernstein basis at `u`.
fn cubic(u: f64) -> Basis4 {
    let v = 1.0 - u;
    [v * v * v, 3.0 * u * v * v, 3.0 * u * u * v, u * u * u]
}

/// Cubic Bernstein basis derivative with respect to `x` on `[lo, hi]`.
fn cubic_derivative(x: f64, lo: f64, hi: f64) -> Basis4 {
    let q = quadratic((x - lo) / (hi - lo));
    let scale = 3.0 / (hi - lo);
    [-scale * q[0], scale * (q[0] - q[1]), scale * (q[1] - q[2]), scale * q[2]]
}

/// Solve `a z = b` (row-major `a`, `m x m`) by Gaussian elimination with partial pivoting.
fn solve(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut aug: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = a[i * m..(i + 1) * m].to_vec();
            row.push(b[i]);
            row
        })
        .collect();
    for col in 0..m {
        let pivot =
            (col..m).max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs())).unwrap();
        aug.swap(col, pivot);
        let p = aug[col][col];
        assert!(p.abs() > 1e-12, "the reference Gram matrix is singular");
        for r in 0..m {
            if r != col {
                let f = aug[r][col] / p;
                for c in col..=m {
                    let v = aug[col][c];
                    aug[r][c] -= f * v;
                }
            }
        }
    }
    (0..m).map(|i| aug[i][m] / aug[i][i]).collect()
}

struct Fixture {
    a: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn fixture() -> Fixture {
    let a: Vec<f64> = (0..N).map(|i| ((i * 5) % N) as f64 / 12.0).collect();
    let x: Vec<f64> = (0..N).map(|i| ((i * 7 + 2) % N) as f64 / 12.0 - 1.0).collect();
    let y: Vec<f64> = (0..N)
        .map(|i| {
            let t = a[i];
            1.0 + 0.8 * t - 0.5 * t * t
                + 0.2 * t * t * t
                + 0.7 * x[i]
                + 0.3 * ((7 * i + 1) as f64).sin()
        })
        .collect();
    Fixture { a, x, y }
}

struct Reference {
    point: f64,
    half_width: f64,
    half_width_uncorrected: f64,
}

fn reference(f: &Fixture, at: f64) -> Reference {
    let range = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let (alo, ahi) = range(&f.a);
    let (xlo, xhi) = range(&f.x);
    let p = 7;
    // Row-major design rows [1, T0, T1, T2, X0, X1, X2].
    let rows: Vec<[f64; 7]> = (0..N)
        .map(|i| {
            let t = cubic((f.a[i] - alo) / (ahi - alo));
            let c = cubic((f.x[i] - xlo) / (xhi - xlo));
            [1.0, t[0], t[1], t[2], c[0], c[1], c[2]]
        })
        .collect();
    // Gram = X'X + S with S = lambda D2'D2 restricted to the kept adjustment coefficients.
    let mut gram = vec![0.0; p * p];
    for row in &rows {
        for j in 0..p {
            for k in 0..p {
                gram[j * p + k] += row[j] * row[k];
            }
        }
    }
    for start in 0..2 {
        // Row `start` of D2 is [1, -2, 1] on coefficients start..start + 3 of the adjustment smooth.
        for (da, ia) in [1.0, -2.0, 1.0].iter().zip(start..start + 3) {
            for (db, ib) in [1.0, -2.0, 1.0].iter().zip(start..start + 3) {
                if ia < 3 && ib < 3 {
                    gram[(4 + ia) * p + 4 + ib] += LAMBDA * da * db;
                }
            }
        }
    }
    let xty: Vec<f64> =
        (0..p).map(|j| rows.iter().zip(&f.y).map(|(row, y)| row[j] * y).sum()).collect();
    let beta = solve(&gram, &xty, p);
    // Coefficient functional: derivatives of the three kept treatment bases at `at`.
    let d = cubic_derivative(at, alo, ahi);
    assert!(d.iter().sum::<f64>().abs() < 1e-12, "basis derivatives sum to zero");
    let mut c = vec![0.0; p];
    c[1..4].copy_from_slice(&d[..3]);
    let point: f64 = (1..4).map(|j| c[j] * beta[j]).sum();
    let v = solve(&gram, &c, p); // (X'X + S)^-1 c; the Gram matrix is symmetric.
    let mut half = [0.0; 2];
    for (slot, corrected) in half.iter_mut().zip([true, false]) {
        let psi: Vec<f64> = (0..N)
            .map(|i| {
                let fitted: f64 = (0..p).map(|j| rows[i][j] * beta[j]).sum();
                let mut e = f.y[i] - fitted;
                if corrected {
                    let h_vec = solve(&gram, &rows[i], p);
                    let h: f64 = (0..p).map(|j| rows[i][j] * h_vec[j]).sum();
                    assert!((0.0..1.0).contains(&h), "leverage {h}");
                    e /= (1.0 - h).sqrt();
                }
                N as f64 * e * (0..p).map(|j| rows[i][j] * v[j]).sum::<f64>()
            })
            .collect();
        let centre = psi.iter().sum::<f64>() / N as f64;
        let ss: f64 = psi.iter().map(|q| (q - centre) * (q - centre)).sum();
        *slot = Z_975 * ss.sqrt() / N as f64;
    }
    Reference { point, half_width: half[0], half_width_uncorrected: half[1] }
}

#[test]
fn the_identity_jacobian_band_equals_the_hand_computed_hc2_sandwich() {
    let f = fixture();
    let data = TabularData::from_f64_columns([
        ("y", f.y.as_slice()),
        ("a", f.a.as_slice()),
        ("x", f.x.as_slice()),
    ])
    .unwrap();
    let mut estimator = ContinuousResponseEstimator::new([VariableId::from_raw(2)]);
    estimator.options.nuisance_basis = 4;
    estimator.options.nuisance_lambda = LAMBDA;
    estimator.options.confidence_level = 0.95;
    for at in [0.3, 0.9, 1.5] {
        let want = reference(&f, at);
        let response = estimator
            .estimate_identified(
                &data,
                &ResponseQuery::new(ResponseFunctional::Jacobian {
                    outcomes: Arc::from([VariableId::from_raw(0)]),
                    treatments: Arc::from([VariableId::from_raw(1)]),
                    at: Arc::from([at]),
                    scale: DerivativeScale::Identity,
                }),
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_or_else(|e| panic!("at {at}: {e}"));
        let ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) =
            &response.estimate
        else {
            panic!("at {at}: a Jacobian");
        };
        let ResponseUncertainty::PointwiseBand { level, lower, upper, .. } = &response.uncertainty
        else {
            panic!("at {at}: the identity-scale Jacobian reports a band");
        };
        assert!((level - 0.95).abs() < 1e-15);
        assert_eq!((lower.len(), upper.len(), values.len()), (1, 1, 1));
        let point = values[0];
        assert!(
            (point - want.point).abs() <= 1e-6 * want.point.abs() + 1e-9,
            "at {at}: point {point} vs {}",
            want.point
        );
        let half = (upper[0] - lower[0]) / 2.0;
        assert!(
            (half - want.half_width).abs() <= 1e-5 * want.half_width,
            "at {at}: half-width {half} vs {}",
            want.half_width
        );
        assert!(((upper[0] + lower[0]) / 2.0 - point).abs() <= 1e-9 * (1.0 + point.abs()));
        // The tolerance discriminates the leverage correction.
        assert!(
            want.half_width > 1.05 * want.half_width_uncorrected,
            "at {at}: HC2 {} vs HC0 {}",
            want.half_width,
            want.half_width_uncorrected
        );
    }
}
