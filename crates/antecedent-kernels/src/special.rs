//! Shared special functions (Abramowitz–Stegun erf family).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

const ERFC_A1: f64 = 0.254_829_592;
const ERFC_A2: f64 = -0.284_496_736;
const ERFC_A3: f64 = 1.421_413_741;
const ERFC_A4: f64 = -1.453_152_027;
const ERFC_A5: f64 = 1.061_405_429;

/// Complementary error function (Hastings / A–S 7.1.26, max abs error ~1.5e-7).
#[must_use]
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let erf_c = (-z * z).exp()
        * (((((ERFC_A5 * t + ERFC_A4) * t + ERFC_A3) * t + ERFC_A2) * t + ERFC_A1) * t);
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

/// Standard normal PDF ϕ(x) = (1/√(2π)) exp(−x²/2).
#[must_use]
pub fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal quantile Φ⁻¹(p) (Acklam rational approximation).
///
/// Absolute error is typically below ~1e-9 on `(0, 1)`. Returns `±∞` at the
/// endpoints and `NaN` outside `[0, 1]`.
#[must_use]
#[allow(
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::manual_range_contains
)]
pub fn norm_inv(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    // Coefficients from Peter J. Acklam's approximation.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_inv_known_quantiles() {
        // Reference Φ⁻¹ values (high precision).
        assert!((norm_inv(0.5) - 0.0).abs() < 1e-12);
        assert!((norm_inv(0.975) - 1.959_963_984_540_054).abs() < 1e-8);
        assert!((norm_inv(0.025) + 1.959_963_984_540_054).abs() < 1e-8);
        assert!((norm_inv(0.841_344_746_068_542_9) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn norm_inv_endpoints() {
        assert!(norm_inv(0.0).is_infinite() && norm_inv(0.0).is_sign_negative());
        assert!(norm_inv(1.0).is_infinite() && norm_inv(1.0).is_sign_positive());
        assert!(norm_inv(-0.1).is_nan());
        assert!(norm_inv(1.1).is_nan());
    }
}
