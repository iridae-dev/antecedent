//! The Laplace-noise BIC local score against an exhaustive least-absolute-deviations fit.
//!
//! A least-absolute-deviations line (or hyperplane) with an intercept is attained at a fit that
//! interpolates `k = |Pa| + 1` data points, so the minimum of the L1 objective is the minimum over
//! every `k`-subset of the rows of the L1 norm of the residuals of the interpolating fit. That
//! enumeration shares nothing with the iteratively reweighted least squares behind
//! `laplace_bic_local`, and the score follows from the optimum as
//! `-n (ln(2 b) + 1) - (|Pa| + 2) / 2 ln n` with `b` the mean absolute residual.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::{GraphScoreData, laplace_bic_local};

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Solve the square system `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let k = b.len();
    for c in 0..k {
        let piv = (c..k).max_by(|&p, &q| a[p][c].abs().total_cmp(&a[q][c].abs()))?;
        if a[piv][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, piv);
        b.swap(c, piv);
        for r in (c + 1)..k {
            let f = a[r][c] / a[c][c];
            for j in c..k {
                a[r][j] -= f * a[c][j];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; k];
    for c in (0..k).rev() {
        x[c] = (b[c] - (c + 1..k).map(|j| a[c][j] * x[j]).sum::<f64>()) / a[c][c];
    }
    Some(x)
}

/// Minimum L1 objective of `y` on an intercept and `xs`, by enumerating interpolating fits.
fn lad_minimum(xs: &[Vec<f64>], y: &[f64]) -> f64 {
    let n = y.len();
    let k = xs.len() + 1;
    let row = |r: usize| -> Vec<f64> { std::iter::once(1.0).chain(xs.iter().map(|c| c[r])).collect() };
    let mut best = f64::INFINITY;
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        if let Some(beta) = solve(idx.iter().map(|&r| row(r)).collect(), idx.iter().map(|&r| y[r]).collect()) {
            let l1: f64 = (0..n)
                .map(|r| (y[r] - row(r).iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>()).abs())
                .sum();
            best = best.min(l1);
        }
        // next k-subset in lexicographic order
        let mut i = k;
        loop {
            if i == 0 {
                return best;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return best;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

fn reference_score(xs: &[Vec<f64>], y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let b = lad_minimum(xs, y) / n;
    -n * ((2.0 * b).ln() + 1.0) - 0.5 * (xs.len() as f64 + 2.0) * n.ln()
}

fn score(cols: &[Vec<f64>], node: usize, parents: &[usize]) -> f64 {
    let n = cols[0].len();
    let flat: Vec<f64> = cols.iter().flatten().copied().collect();
    let data = GraphScoreData::new(n, cols.len(), Arc::from(flat)).unwrap();
    laplace_bic_local(&data, u32::try_from(node).unwrap(), &parents.iter().map(|&p| u32::try_from(p).unwrap()).collect::<Vec<_>>()).unwrap()
}

fn laplace(rng: &mut Lcg) -> f64 {
    let u = rng.unit() - 0.5;
    -u.signum() * (1.0 - 2.0 * u.abs()).max(1e-12).ln()
}

#[test]
fn reference_enumeration_solves_known_least_absolute_deviation_problems() {
    // Median of an odd sample: intercept only, objective is the sum of absolute deviations.
    let y = [1.0, 2.0, 4.0, 8.0, 16.0];
    assert!((lad_minimum(&[], &y) - (3.0 + 2.0 + 0.0 + 4.0 + 12.0)).abs() < 1e-12);
    // Exact line plus one gross outlier: the fit ignores the outlier.
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut yy: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    yy[3] = 100.0;
    assert!((lad_minimum(&[x], &yy) - (100.0 - 7.0)).abs() < 1e-9);
}

#[test]
fn laplace_score_equals_the_exhaustive_lad_optimum() {
    let mut rng = Lcg(0xFEED_5EED);
    let mut worst = 0.0_f64;
    for trial in 0..60 {
        let n = 9 + trial % 6;
        let p = trial % 3;
        let xs: Vec<Vec<f64>> = (0..p).map(|_| (0..n).map(|_| laplace(&mut rng)).collect()).collect();
        let mut y: Vec<f64> = (0..n)
            .map(|r| 0.5 + xs.iter().enumerate().map(|(j, c)| (1.0 + j as f64) * c[r]).sum::<f64>() + laplace(&mut rng))
            .collect();
        if trial % 4 == 3 {
            y[0] += 50.0; // a gross outlier
        }
        let mut cols = vec![y.clone()];
        cols.extend(xs.iter().cloned());
        let parents: Vec<usize> = (1..=p).collect();
        let got = score(&cols, 0, &parents);
        let want = reference_score(&xs, &y);
        worst = worst.max((got - want).abs());
        assert!((got - want).abs() < 1e-8 * want.abs().max(1.0), "trial {trial} (n={n}, p={p}): {got} vs {want}");
    }
    assert!(worst < 1e-6, "largest score gap {worst}");
}

#[test]
fn laplace_score_is_unchanged_by_rescaling_and_shifting_the_outcome() {
    let mut rng = Lcg(77);
    let x: Vec<f64> = (0..12).map(|_| laplace(&mut rng)).collect();
    let y: Vec<f64> = x.iter().map(|v| 0.7 * v + laplace(&mut rng)).collect();
    let base = score(&[y.clone(), x.clone()], 0, &[1]);
    // Scaling the outcome by c shifts the log-likelihood by -n ln c.
    let c = 1.0e-6;
    let scaled: Vec<f64> = y.iter().map(|v| c * v + 3.0).collect();
    let moved = score(&[scaled, x], 0, &[1]);
    let n = 12.0;
    assert!((moved - (base - n * c.ln())).abs() < 1e-5 * base.abs(), "{moved} vs {}", base - n * c.ln());
}
