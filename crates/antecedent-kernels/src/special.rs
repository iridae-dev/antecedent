//! Shared special functions (Cody rational-Chebyshev erf family).
//!
//! These underpin reported p-values across the workspace (refutation tests in
//! `antecedent-validate`, two-sample tests in `antecedent-stats`, the probit link
//! in `antecedent-stats::glm` and `antecedent-prob`), so they are held to
//! *relative* rather than absolute accuracy.
//!
//! The previous implementation (Hastings / A&S 7.1.26) was accurate to ~1.5e-7
//! absolute, but its relative error grows steadily into the tail — measured
//! against a reference: 6.6e-7 at x=1, 6.6e-4 at x=3, 6.7e-3 at x=5, 2.7e-2 at
//! x=8. A reported p-value wrong by a few percent is not acceptable, and the
//! error compounds through probit IRLS. Cody's approximation below holds
//! relative error near machine epsilon across the whole range.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Coefficients for erf on |x| <= 0.46875 (Cody, SPECFUN `CALERF`).
#[allow(clippy::excessive_precision)]
const CODY_A: [f64; 5] = [
    3.161_123_743_870_565_60e00,
    1.138_641_541_510_501_56e02,
    3.774_852_376_853_020_21e02,
    3.209_377_589_138_469_47e03,
    1.857_777_061_846_031_53e-1,
];
#[allow(clippy::excessive_precision)]
const CODY_B: [f64; 4] = [
    2.360_129_095_234_412_09e01,
    2.440_246_379_344_441_73e02,
    1.282_616_526_077_372_28e03,
    2.844_236_833_439_170_62e03,
];

/// Coefficients for erfc on 0.46875 < |x| <= 4.
#[allow(clippy::excessive_precision)]
const CODY_C: [f64; 9] = [
    5.641_884_969_886_700_89e-1,
    8.883_149_794_388_375_94e0,
    6.611_919_063_714_162_95e01,
    2.986_351_381_974_001_31e02,
    8.819_522_212_417_690_90e02,
    1.712_047_612_634_070_58e03,
    2.051_078_377_826_071_47e03,
    1.230_339_354_797_997_25e03,
    2.153_115_354_744_038_46e-8,
];
#[allow(clippy::excessive_precision)]
const CODY_D: [f64; 8] = [
    1.574_492_611_070_983_47e01,
    1.176_939_508_913_124_99e02,
    5.371_811_018_620_098_58e02,
    1.621_389_574_566_690_19e03,
    3.290_799_235_733_459_63e03,
    4.362_619_090_143_247_16e03,
    3.439_367_674_143_721_64e03,
    1.230_339_354_803_749_42e03,
];

/// Coefficients for erfc on |x| > 4.
#[allow(clippy::excessive_precision)]
const CODY_P: [f64; 6] = [
    3.053_266_349_612_323_44e-1,
    3.603_448_999_498_044_39e-1,
    1.257_817_261_112_292_46e-1,
    1.608_378_514_874_227_66e-2,
    6.587_491_615_298_378_03e-4,
    1.631_538_713_730_209_78e-2,
];
#[allow(clippy::excessive_precision)]
const CODY_Q: [f64; 5] = [
    2.568_520_192_289_822_42e00,
    1.872_952_849_923_460_47e00,
    5.279_051_029_514_284_12e-1,
    6.051_834_131_244_131_91e-2,
    2.335_204_976_268_691_85e-3,
];

/// 1/sqrt(pi).
#[allow(clippy::excessive_precision)]
const SQRT_PI_INV: f64 = 5.641_895_835_477_562_869_5e-1;
/// Below this |x|, erfc is evaluated as `1 - erf`.
const CODY_THRESH: f64 = 0.46875;
/// Above this |x|, `exp(-x*x)` underflows f64 and erfc(x) is 0.
const CODY_XBIG: f64 = 26.543;

/// `exp(-y*y)` split as `exp(-t*t) * exp(-(y-t)(y+t))` with `t = trunc(16y)/16`.
///
/// Squaring `y` directly loses low-order bits of the exponent; splitting on a
/// 1/16 grid keeps the dominant factor exactly representable and confines the
/// rounding to the small correction term. This is what preserves relative
/// accuracy far out in the tail.
fn exp_neg_square_split(y: f64) -> f64 {
    let t = (y * 16.0).trunc() / 16.0;
    let del = (y - t) * (y + t);
    (-t * t).exp() * (-del).exp()
}

/// Complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Cody's rational-Chebyshev approximation; relative error near machine epsilon
/// across the whole range, including the far tail where `erfc` underflows to 0
/// beyond |x| ~ 26.5.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    let y = x.abs();
    if y <= CODY_THRESH {
        // erfc is O(1) here, so 1 - erf carries no cancellation.
        return 1.0 - erf(x);
    }

    let mag = if y <= 4.0 {
        let mut num = CODY_C[8] * y;
        let mut den = y;
        for i in 0..7 {
            num = (num + CODY_C[i]) * y;
            den = (den + CODY_D[i]) * y;
        }
        exp_neg_square_split(y) * ((num + CODY_C[7]) / (den + CODY_D[7]))
    } else if y >= CODY_XBIG {
        0.0
    } else {
        let ysq = 1.0 / (y * y);
        let mut num = CODY_P[5] * ysq;
        let mut den = ysq;
        for i in 0..4 {
            num = (num + CODY_P[i]) * ysq;
            den = (den + CODY_Q[i]) * ysq;
        }
        let r = ysq * (num + CODY_P[4]) / (den + CODY_Q[4]);
        exp_neg_square_split(y) * ((SQRT_PI_INV - r) / y)
    };

    if x < 0.0 { 2.0 - mag } else { mag }
}

/// Error function `erf(x)`.
///
/// Evaluated directly on the central interval rather than as `1 - erfc(x)`, which
/// would lose all relative precision as `x -> 0`.
#[must_use]
pub fn erf(x: f64) -> f64 {
    let y = x.abs();
    if y > CODY_THRESH {
        let c = erfc(x);
        return if x < 0.0 { c - 1.0 } else { 1.0 - c };
    }
    let ysq = y * y;
    let mut num = CODY_A[4] * ysq;
    let mut den = ysq;
    for i in 0..3 {
        num = (num + CODY_A[i]) * ysq;
        den = (den + CODY_B[i]) * ysq;
    }
    x * (num + CODY_A[3]) / (den + CODY_B[3])
}

/// Standard normal CDF Φ(x).
///
/// Uses `0.5 * erfc(-x / sqrt(2))` rather than `0.5 * (1 + erf(x / sqrt(2)))`, so
/// the left tail keeps full relative accuracy instead of cancelling against 1.
#[must_use]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal survival function `1 - Φ(x)`, computed directly.
///
/// Prefer this to `1.0 - norm_cdf(x)` for upper-tail probabilities: the
/// subtraction form saturates to 0 once Φ(x) rounds to 1 (around x ~ 8), while
/// this stays accurate until the true value underflows near x ~ 37.
#[must_use]
pub fn norm_sf(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
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

    /// Reference values from `mpmath.erfc` at 50 digits, rounded to f64.
    ///
    /// These are *relative* tolerances, which is the point: the previous A&S
    /// 7.1.26 implementation passes an absolute-error check but fails every row
    /// from x = 1 outward here, by 6.6e-7 at x = 1 rising to 2.7e-2 at x = 8.
    #[test]
    fn erfc_matches_reference_into_the_tail() {
        #[allow(clippy::excessive_precision)]
        const CASES: &[(f64, f64)] = &[
            (0.0, 1.0),
            (0.25, 7.236_736_098_317_630_7e-1),
            (0.5, 4.795_001_221_869_534_6e-1),
            (1.0, 1.572_992_070_502_851_3e-1),
            (2.0, 4.677_734_981_047_265_8e-3),
            (3.0, 2.209_049_699_858_544_1e-5),
            (4.0, 1.541_725_790_028_001_9e-8),
            (5.0, 1.537_459_794_428_034_9e-12),
            (6.0, 2.151_973_671_249_891_3e-17),
            (8.0, 1.122_429_717_298_292_7e-29),
            (10.0, 2.088_487_583_762_544_8e-45),
            (20.0, 5.395_865_611_607_900_9e-176),
        ];
        for &(x, expected) in CASES {
            let got = erfc(x);
            let rel = (got - expected).abs() / expected;
            assert!(rel < 1e-13, "erfc({x}) = {got:e}, expected {expected:e} (rel {rel:e})");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact underflow to zero is the property under test
    fn erfc_reflection_and_saturation() {
        // erfc(-x) = 2 - erfc(x).
        for &x in &[0.3, 1.0, 2.5, 5.0] {
            assert!((erfc(-x) - (2.0 - erfc(x))).abs() < 1e-15);
        }
        // Never leaves [0, 2] — the old approximation could return a negative
        // "probability" in the tail, which reached callers unclamped.
        for i in 0..400 {
            let x = -20.0 + f64::from(i) * 0.1;
            let v = erfc(x);
            assert!((0.0..=2.0).contains(&v), "erfc({x}) = {v} out of range");
        }
        // Underflows to exactly zero rather than to a small wrong number.
        assert_eq!(erfc(30.0), 0.0);
    }

    #[test]
    fn erf_keeps_relative_accuracy_near_zero() {
        #[allow(clippy::excessive_precision)]
        const CASES: &[(f64, f64)] = &[
            (0.5, 5.204_998_778_130_465_3e-1),
            (1.0, 8.427_007_929_497_148_7e-1),
            (2.0, 9.953_222_650_189_527_3e-1),
        ];
        // erf(x) -> 2x/sqrt(pi) as x -> 0, with relative defect x^2/3, so the
        // leading term is only a valid reference well below x = 1e-8. Computing
        // erf as 1 - erfc(x) destroys every significant digit in this range.
        for &x in &[1e-14, 1e-12, 1e-10] {
            let expected = 2.0 * x / std::f64::consts::PI.sqrt();
            let rel = (erf(x) - expected).abs() / expected;
            assert!(rel < 1e-15, "erf({x}) rel error {rel:e}");
        }
        for &(x, expected) in CASES {
            assert!((erf(x) - expected).abs() / expected < 1e-14, "erf({x})");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // Phi(0) = 1/2 is exact in binary floating point
    fn norm_cdf_and_sf_agree_with_reference() {
        #[allow(clippy::excessive_precision)]
        const CASES: &[(f64, f64)] = &[
            (0.0, 0.5),
            (-1.0, 1.586_552_539_314_570_5e-1),
            (-1.959_963_984_540_054, 2.500_000_000_000_000_0e-2),
            (-3.0, 1.349_898_031_630_094_6e-3),
            (-6.0, 9.865_876_450_376_981_4e-10),
            (-10.0, 7.619_853_024_160_594_5e-24),
        ];
        for &(x, expected) in CASES {
            let rel = (norm_cdf(x) - expected).abs() / expected;
            assert!(rel < 1e-13, "norm_cdf({x}) rel error {rel:e}");
            // Symmetry: the survival function is the mirror of the CDF.
            let rel_sf = (norm_sf(-x) - expected).abs() / expected;
            assert!(rel_sf < 1e-13, "norm_sf({}) rel error {rel_sf:e}", -x);
        }
        assert!(norm_cdf(0.0) == 0.5 && norm_sf(0.0) == 0.5);
    }

    #[test]
    #[allow(clippy::float_cmp)] // the saturation of 1 - Phi(x) to exactly 0 is the point
    fn norm_sf_survives_where_one_minus_cdf_saturates() {
        // 1 - norm_cdf(x) hits exactly 0 around x = 9; norm_sf keeps going.
        assert_eq!(1.0 - norm_cdf(9.0), 0.0);
        let sf = norm_sf(9.0);
        assert!(sf > 0.0 && (sf - 1.128_588_405_953_842_2e-19).abs() / sf < 1e-13);
    }

    #[test]
    fn norm_cdf_round_trips_through_norm_inv() {
        // Accuracy here is bounded by `norm_inv` (Acklam, ~1e-9 relative in x),
        // not by `norm_cdf`. An error of dx in x becomes a relative error of
        // (phi(x)/Phi(x)) * dx in p, which grows with |x| — hence the looser
        // tolerance out at p = 1e-8 (x ~ -5.6).
        for &p in &[1e-8, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - 1e-4] {
            let x = norm_inv(p);
            let rel = (norm_cdf(x) - p).abs() / p;
            assert!(rel < 1e-7, "round trip at p={p} gave rel error {rel:e}");
        }
    }
}
