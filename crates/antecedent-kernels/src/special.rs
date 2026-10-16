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
//! relative error near machine epsilon for normal results; subnormal tails
//! necessarily lose relative precision.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Coefficients for erf on |x| <= 0.46875 (Cody, SPECFUN `CALERF`).
#[allow(
    clippy::excessive_precision,
    reason = "reference coefficients are kept digit for digit as published"
)]
const CODY_A: [f64; 5] = [
    3.161_123_743_870_565_60e00,
    1.138_641_541_510_501_56e02,
    3.774_852_376_853_020_21e02,
    3.209_377_589_138_469_47e03,
    1.857_777_061_846_031_53e-1,
];
#[allow(
    clippy::excessive_precision,
    reason = "reference coefficients are kept digit for digit as published"
)]
const CODY_B: [f64; 4] = [
    2.360_129_095_234_412_09e01,
    2.440_246_379_344_441_73e02,
    1.282_616_526_077_372_28e03,
    2.844_236_833_439_170_62e03,
];

/// Coefficients for erfc on 0.46875 < |x| <= 4.
#[allow(
    clippy::excessive_precision,
    reason = "reference coefficients are kept digit for digit as published"
)]
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
#[allow(
    clippy::excessive_precision,
    reason = "reference coefficients are kept digit for digit as published"
)]
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
#[allow(
    clippy::excessive_precision,
    reason = "reference coefficients are kept digit for digit as published"
)]
const CODY_P: [f64; 6] = [
    3.053_266_349_612_323_44e-1,
    3.603_448_999_498_044_39e-1,
    1.257_817_261_112_292_46e-1,
    1.608_378_514_874_227_66e-2,
    6.587_491_615_298_378_03e-4,
    1.631_538_713_730_209_78e-2,
];
#[allow(
    clippy::excessive_precision,
    reason = "reference coefficients are kept digit for digit as published"
)]
const CODY_Q: [f64; 5] = [
    2.568_520_192_289_822_42e00,
    1.872_952_849_923_460_47e00,
    5.279_051_029_514_284_12e-1,
    6.051_834_131_244_131_91e-2,
    2.335_204_976_268_691_85e-3,
];

/// 1/sqrt(pi).
#[allow(
    clippy::excessive_precision,
    reason = "reference coefficients are kept digit for digit as published"
)]
const SQRT_PI_INV: f64 = 5.641_895_835_477_562_869_5e-1;
/// Below this |x|, erfc is evaluated as `1 - erf`.
const CODY_THRESH: f64 = 0.46875;
/// Above this |x|, erfc(x) rounds to zero even with subnormal f64 support.
const CODY_XBIG: f64 = 27.3;

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
/// for normal results. Subnormal tail values retain absolute floating-point
/// accuracy until rounding to zero near x = 27.3.
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
        return 1.0 - c;
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
/// this stays accurate until the true value underflows near x ~ 38.5.
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

/// `ln √(2π) = ½ ln 2π`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Log standard-normal density `ln φ(x) = −x²/2 − ½ ln 2π`, finite where `φ` underflows.
#[must_use]
pub fn log_norm_pdf(x: f64) -> f64 {
    -LN_SQRT_2PI - 0.5 * x * x
}

/// Left-tail Mills ratio pieces for `t ≥ 8`: returns `(ratio, delta)` with
/// `ratio = φ(t) / Φ(−t) = t + delta` and `delta = 1 / (t + 1 / (t + 2 / (t + 3 / …)))`.
///
/// The continued fraction (64 levels) is evaluated bottom-up. Keeping `delta` separate avoids
/// subtracting nearly equal values when the observed curvature `ratio · delta` is needed.
/// Accurate to a few ulps for `t ≥ 3`; not intended for smaller `t`.
#[must_use]
pub fn mills_left_tail(t: f64) -> (f64, f64) {
    let mut delta = 0.0;
    for k in (1..=64).rev() {
        delta = f64::from(k) / (t + delta);
    }
    (t + delta, delta)
}

/// Below this `x`, [`log_norm_cdf`] switches from `ln Φ(x)` to the Mills continued fraction.
const LOG_CDF_TAIL_FROM: f64 = -8.0;

/// Numerically stable `ln Φ(x)`, the single owner of the log normal tail.
///
/// Branch policy (each branch is the accurate one for its range):
///
/// * `x ≤ −8`: `ln φ(x) − ln(ratio)` from the Mills continued fraction
///   ([`mills_left_tail`]). It never underflows (`x = −40` gives `−804.6`) and keeps relative
///   accuracy at a few ulps of the result; `norm_cdf(x).ln()` is accurate to `~1e-16`
///   absolute here but hits subnormals near `x ≈ −37.5` and `0` at `−38.6`.
/// * `−8 < x ≤ 0`: `ln Φ(x)` through the Cody CDF.
/// * `x > 0`: `ln_1p(−Φ(−x))`, which keeps the tiny negative values (`−6.2e−16` at `x = 8`)
///   that `ln Φ(x)` would round to `0`.
///
/// `ln Φ̄(z) = log_norm_cdf(−z)`; checked against 60-digit `mpmath` over `[−40, 10]`.
#[must_use]
pub fn log_norm_cdf(x: f64) -> f64 {
    if x <= LOG_CDF_TAIL_FROM {
        return log_norm_pdf(x) - mills_left_tail(-x).0.ln();
    }
    if x > 0.0 {
        return (-norm_cdf(-x)).ln_1p();
    }
    norm_cdf(x).ln()
}

/// Numerically stable log survival function `ln Φ̄(z) = ln Φ(−z)`, without underflow in the
/// upper tail. See [`log_norm_cdf`] for the branch policy.
#[must_use]
pub fn log_norm_sf(z: f64) -> f64 {
    log_norm_cdf(-z)
}

/// The 97.5% standard-normal quantile `z_{0.975}`, the two-sided 95% critical value.
///
/// The single workspace owner of the constant a Wald 95% interval multiplies its standard
/// error by. The nearest `f64` to `z_{0.975} = 1.95996398454005423552…` (verified against 40-digit `mpmath`; asserted within one ulp of `norm_inv(0.975)`), but a
/// `const` so it carries no evaluation cost and reads identically at every interval site.
pub const NORMAL_Q975: f64 = 1.959_963_984_540_054_3;

/// Standard normal quantile Φ⁻¹(p): Acklam's rational approximation refined by one Halley step.
///
/// This is the one high-accuracy quantile owner; `antecedent_stats::normal_ppf` is an alias.
/// Acklam's approximation alone has relative error ~1.2e-9; one Halley (second-order Newton)
/// correction against the Cody-rational CDF/survival function brings the result to a few
/// ulps for every normal-range probability, including the far tails. Subnormal
/// probabilities keep reduced relative precision. Returns `±∞` at `0` / `1` and `NaN`
/// outside `[0, 1]` (and for `NaN`).
#[must_use]
pub fn norm_inv(p: f64) -> f64 {
    let x0 = norm_inv_acklam(p);
    if !x0.is_finite() {
        return x0;
    }
    // Halley step for f(x) = Φ(x) − p with f' = φ and f'' = −xφ: the update
    // 2 f f' / (2 f'² − f f'') simplifies to 2f / (2φ + f·x).
    //
    // For p > 0.5 the residual is taken through the survival function: `norm_cdf` saturates to
    // exactly 1.0 beyond x ≈ 8, which would erase the sub-ulp information the step needs in
    // the upper tail, while `Φ(x) − p == −(Sf(x) − (1 − p))` stays accurate there.
    let f = if p <= 0.5 { norm_cdf(x0) - p } else { -(norm_sf(x0) - (1.0 - p)) };
    let phi = norm_pdf(x0);
    let denom = 2.0 * phi + f * x0;
    if phi > 0.0 && denom.is_finite() && denom != 0.0 {
        let refined = x0 - 2.0 * f / denom;
        if refined.is_finite() {
            return refined;
        }
    }
    x0
}

/// Acklam's rational approximation of Φ⁻¹(p), relative error ~1.2e-9; the Halley seed of
/// [`norm_inv`].
#[allow(
    clippy::float_cmp,
    clippy::items_after_statements,
    reason = "the exact comparisons test the endpoints p == 0.0 and p == 1.0, which are special-cased"
)]
fn norm_inv_acklam(p: f64) -> f64 {
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
    fn erf_is_odd_across_approximation_regions() {
        // NIST DLMF 7.4.1: erf(-x) = -erf(x).
        for x in [1e-12, 0.25, 0.46875, 0.5, 1.0, 4.0, 10.0, f64::INFINITY] {
            assert!((erf(-x) + erf(x)).abs() < 2e-16, "x={x}");
            assert!((erf(-x) + erfc(-x) - 1.0).abs() < 2e-16);
        }
    }

    #[test]
    fn erfc_preserves_representable_subnormal_tails() {
        // References rounded from high-precision erfc; a normal-only cutoff at
        // 26.543 discarded these representable probabilities altogether.
        for (x, expected) in [(26.6, 1.088_512_588_544_227e-309), (27.0, 5.237_05e-319)] {
            let actual = erfc(x);
            assert!(actual > 0.0);
            assert!((actual - expected).abs() <= 4.0 * f64::from_bits(1), "{x}: {actual:e}");
        }
        assert!(norm_sf(38.0) > 0.0);
    }

    #[test]
    fn norm_inv_known_quantiles() {
        // Reference Φ⁻¹ values (high precision).
        assert!((norm_inv(0.5) - 0.0).abs() < 1e-15);
        assert!((norm_inv(0.975) - NORMAL_Q975).abs() < 2.3e-16);
        assert!((norm_inv(0.025) + NORMAL_Q975).abs() < 2.3e-16);
        assert!((norm_inv(0.841_344_746_068_542_9) - 1.0).abs() < 4e-16);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    #[allow(
        clippy::excessive_precision,
        reason = "reference coefficients are kept digit for digit as published"
    )]
    fn norm_inv_matches_mpmath_across_the_tails() {
        // Φ⁻¹ of the exact f64 argument, solved by `mpmath.findroot` on `ncdf` at 60 digits.
        // The Acklam seed alone is off by up to ~1e-9 relative; the Halley step must reach a
        // few ulps everywhere, including p = 1 − 1e-10 where the argument itself is rounded.
        let cases: [(f64, f64); 22] = [
            (1e-300, -37.047_096_299_361_2),
            (1e-100, -21.273_453_560_965_324),
            (1e-50, -14.933_337_534_788_49),
            (1e-15, -7.941_345_326_170_997),
            (1e-12, -7.034_483_825_301_132),
            (1e-10, -6.361_340_902_404_056),
            (1e-8, -5.612_001_244_174_789),
            (1e-5, -4.264_890_793_922_825),
            (0.001, -3.090_232_306_167_813_5),
            (0.01, -2.326_347_874_040_841),
            (0.025, -1.959_963_984_540_054_2),
            (0.1, -1.281_551_565_544_600_4),
            (0.3, -0.524_400_512_708_040_8),
            (0.7, 0.524_400_512_708_040_66),
            (0.9, 1.281_551_565_544_600_6),
            (0.975, 1.959_963_984_540_053_9),
            (0.99, 2.326_347_874_040_840_8),
            (0.999, 3.090_232_306_167_813_3),
            (0.99999, 4.264_890_793_923_840_8),
            (0.999_999_99, 5.612_001_243_305_505),
            (0.999_999_999_9, 6.361_340_889_697_422),
            (0.999_999_999_999, 7.034_486_910_047_835),
        ];
        for (p, expected) in cases {
            let got = norm_inv(p);
            let rel = ((got - expected) / expected).abs();
            assert!(rel < 5e-15, "p={p:e}: got {got:.17e} want {expected:.17e} rel {rel:e}");
        }
        assert_eq!(norm_inv(0.5), 0.0);
        // Symmetry holds to rounding.
        for p in [1e-10, 1e-4, 0.2, 0.4] {
            assert!(
                (norm_inv(p) + norm_inv(1.0 - p)).abs() < 1e-8 * norm_inv(1.0 - p).abs().max(1.0)
            );
        }
    }

    #[test]
    #[allow(clippy::approx_constant, reason = "tabulated reference value, kept digit for digit")]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn log_norm_cdf_matches_mpmath_across_branches() {
        // `mpmath.log(mpmath.ncdf(x))` at 60 digits; covers the 8-40 sigma band that the
        // previous prob (cf below -8) and stats (series from 30) owners treated differently,
        // both branch boundaries, and the upper side where ln Φ is a tiny negative.
        let cases: [(f64, f64); 17] = [
            (-40.0, -804.608_442_013_753_8),
            (-38.0, -726.557_216_018_820_1),
            (-30.0, -454.321_243_956_343_2),
            (-20.0, -203.917_155_371_097_26),
            (-12.0, -75.410_673_001_568_8),
            (-9.0, -43.628_149_113_332_115),
            (-8.0, -35.013_437_159_914_55),
            (-7.5, -31.075_890_902_890_0),
            (-5.0, -15.064_998_393_988_726),
            (-3.0, -6.607_726_221_510_35),
            (-1.0, -1.841_021_645_009_263_5),
            (0.0, -0.693_147_180_559_945_3),
            (1.0, -0.172_753_779_023_449_9),
            (3.0, -0.001_350_809_964_748_193_8),
            (5.0, -2.866_516_129_637_636e-7),
            (8.0, -6.220_960_574_271_786e-16),
            (10.0, -7.619_853_024_160_526e-24),
        ];
        for (x, expected) in cases {
            let got = log_norm_cdf(x);
            let tol = 1e-14 * expected.abs();
            assert!((got - expected).abs() <= tol, "x={x}: got {got:.17e} want {expected:.17e}");
            // The survival-function form is the same function reflected.
            assert_eq!(log_norm_sf(-x), got);
        }
        assert_eq!(log_norm_sf(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(log_norm_cdf(f64::INFINITY), 0.0);
        assert!(log_norm_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn log_norm_cdf_branches_agree_where_they_meet() {
        // Left tail: the continued fraction against ln of the Cody CDF at the switch point.
        let cf = log_norm_pdf(-8.0) - mills_left_tail(8.0).0.ln();
        assert!((cf - norm_cdf(-8.0).ln()).abs() < 1e-14 * cf.abs());
        // Centre: ln Φ(x) against ln_1p(−Φ(−x)) at x = 0 and a point beyond it.
        for x in [0.0, 0.5, 2.0] {
            let direct = norm_cdf(x).ln();
            let stable = (-norm_cdf(-x)).ln_1p();
            assert!((direct - stable).abs() < 1e-15 * direct.abs().max(1e-300), "x={x}");
        }
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
        #[allow(
            clippy::excessive_precision,
            reason = "reference coefficients are kept digit for digit as published"
        )]
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
    #[allow(clippy::float_cmp, reason = "exact underflow to zero is the property under test")]
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
        #[allow(
            clippy::excessive_precision,
            reason = "reference coefficients are kept digit for digit as published"
        )]
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
    #[allow(clippy::float_cmp, reason = "Phi(0) = 1/2 is exact in binary floating point")]
    fn norm_cdf_and_sf_agree_with_reference() {
        #[allow(
            clippy::excessive_precision,
            reason = "reference coefficients are kept digit for digit as published"
        )]
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
    #[allow(clippy::float_cmp, reason = "the saturation of 1 - Phi(x) to exactly 0 is the point")]
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
