//! Shared special functions (Abramowitz–Stegun erf family).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Complementary error function (Hastings / A–S 7.1.26, max abs error ~1.5e-7).
#[must_use]
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let erf_c = (-z * z).exp() * (((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t);
    if x >= 0.0 { erf_c } else { 2.0 - erf_c }
}

/// Error function via [`erfc`].
#[must_use]
pub fn erf(x: f64) -> f64 {
    1.0 - erfc(x)
}

/// Standard normal CDF Φ(x) via erf.
#[must_use]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}
