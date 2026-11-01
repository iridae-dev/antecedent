//! Independent samplers for the choice-study calibration designs: the product's gamma sampler
//! and categorical draw are not reused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code, reason = "shared by several test binaries, each of which uses a subset")]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]

use super::calibration::{self, uniform};

/// Independent Marsaglia-Tsang gamma draw (shape at least one); the product sampler is not
/// reused.
pub fn gamma(shape: f64, normal: &mut impl FnMut() -> f64, unit: &mut impl FnMut() -> f64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 && unit().ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// A Dirichlet draw from independent gammas.
pub fn dirichlet(alpha: &[f64], seed: u64) -> Vec<f64> {
    let mut normal = calibration::gaussian(seed);
    let mut unit = uniform(seed, 11);
    let g: Vec<f64> = alpha.iter().map(|a| gamma(*a, &mut normal, &mut unit)).collect();
    let total: f64 = g.iter().sum();
    g.iter().map(|x| x / total).collect()
}

/// `n` categorical choices from `theta`, as counts.
pub fn choices(theta: &[f64], n: usize, next: &mut impl FnMut() -> f64) -> Vec<f64> {
    let mut counts = vec![0.0; theta.len()];
    for _ in 0..n {
        let mut u = next();
        let mut pick = theta.len() - 1;
        for (j, t) in theta.iter().enumerate() {
            if u < *t {
                pick = j;
                break;
            }
            u -= t;
        }
        counts[pick] += 1.0;
    }
    counts
}

/// A gamma draw of any positive shape.
pub fn gamma_any(
    shape: f64,
    normal: &mut impl FnMut() -> f64,
    unit: &mut impl FnMut() -> f64,
) -> f64 {
    if shape >= 1.0 {
        gamma(shape, normal, unit)
    } else {
        gamma(shape + 1.0, normal, unit) * unit().powf(1.0 / shape)
    }
}
