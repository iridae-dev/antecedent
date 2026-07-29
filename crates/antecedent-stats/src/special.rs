//! Shared special functions for stats (PPF, gamma, incomplete beta).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

/// Standard-normal PPF: Acklam's rational approximation refined by one Halley step.
///
/// Acklam's approximation alone has relative error ~1.6e-9 at p=0.975, rising to ~7.6e-9
/// in the far tails (p=1e-12, 1e-15) — inconsistent with the ~1e-13 accuracy of the
/// Cody-rational `erf`/`erfc` in `antecedent-kernels`, which back the CDF/PDF used here.
/// One Halley (second-order Newton) correction against the exact CDF/PDF brings this to
/// near machine epsilon across the whole domain.
#[must_use]
pub fn normal_ppf(p: f64) -> f64 {
    // Acklam central-region coefficients (|p - 0.5| region).
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
    // Acklam tail coefficients.
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
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    let x0 = if p < P_LOW {
        // Lower tail.
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > 1.0 - P_LOW {
        // Upper tail.
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        // Central region.
        let q = p - 0.5;
        let r = q * q;
        q * (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5])
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    // One Halley refinement step against the exact (Cody-rational) normal CDF/PDF:
    // for f(x) = Phi(x) - p, phi = f'(x), and f''(x) = -x * phi(x), the Halley update
    // 2 f f' / (2 f'^2 - f f'') simplifies to 2f / (2*phi + f*x). This converts Acklam's
    // ~1e-9 starting error into a near-machine-epsilon result in a single step.
    //
    // For p > 0.5 (x0 > 0), compute the residual via the survival function instead of the
    // CDF: `norm_cdf` saturates to exactly 1.0 once x exceeds ~8, which would silently
    // erase the sub-ulp information the refinement needs in the upper tail. `norm_sf`
    // stays accurate there by construction, and Phi(x) - p == -(Sf(x) - (1 - p)).
    let f = if p <= 0.5 {
        antecedent_kernels::norm_cdf(x0) - p
    } else {
        -(antecedent_kernels::norm_sf(x0) - (1.0 - p))
    };
    let phi = antecedent_kernels::norm_pdf(x0);
    let denom = 2.0 * phi + f * x0;
    if phi > 0.0 && denom.is_finite() && denom != 0.0 {
        let refined = x0 - 2.0 * f / denom;
        if refined.is_finite() {
            return refined;
        }
    }
    x0
}

/// Digamma `ψ(z) = d/dz ln Γ(z)` (reflection + asymptotic series).
#[must_use]
pub fn digamma(mut z: f64) -> f64 {
    if !(z.is_finite() && z > 0.0) {
        return f64::NAN;
    }
    let mut result = 0.0;
    // Reflection for z < 0.5: ψ(1−z) − ψ(z) = π cot(πz).
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        result -= pi / (pi * z).tan();
        z = 1.0 - z;
    }
    // Recurrence to z ≥ 8.
    while z < 8.0 {
        result -= 1.0 / z;
        z += 1.0;
    }
    // Asymptotic: ψ(z) ≈ ln z − 1/(2z) − Σ B_{2k}/(2k z^{2k})
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    result += z.ln() - 0.5 * iz;
    // Bernoulli terms: 1/12, −1/120, 1/252, −1/240, 1/132, −691/32760, …
    result -= iz2
        * (1.0 / 12.0
            - iz2
                * (1.0 / 120.0
                    - iz2
                        * (1.0 / 252.0
                            - iz2
                                * (1.0 / 240.0 - iz2 * (1.0 / 132.0 - iz2 * (691.0 / 32760.0))))));
    result
}

/// Trigamma `ψ₁(z) = d²/dz² ln Γ(z)` (reflection + asymptotic series).
#[must_use]
pub fn trigamma(mut z: f64) -> f64 {
    if !(z.is_finite() && z > 0.0) {
        return f64::NAN;
    }
    // Reflection: ψ₁(z) + ψ₁(1−z) = π² / sin²(πz) ⇒ ψ₁(z) = π²csc²(πz) − ψ₁(1−z).
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        let s = (pi * z).sin();
        return (pi * pi) / (s * s) - trigamma(1.0 - z);
    }
    let mut result = 0.0;
    while z < 8.0 {
        result += 1.0 / (z * z);
        z += 1.0;
    }
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    // ψ₁(z) ≈ 1/z + 1/(2z²) + Σ B_{2k}/z^{2k+1}
    result += iz + 0.5 * iz2;
    result += iz2
        * iz
        * (1.0 / 6.0
            - iz2 * (1.0 / 30.0 - iz2 * (1.0 / 42.0 - iz2 * (1.0 / 30.0 - iz2 * (5.0 / 66.0)))));
    result
}

/// Lanczos approximation of `ln Γ(z)`.
#[must_use]
pub fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_654_078_675e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        // Gamma has simple poles at z = 0, -1, -2, ... In floating point, sin(pi * z) at
        // these poles is a tiny nonzero residual (e.g. sin(-pi) ~= -1.2246e-16 rather than
        // exactly 0), and it can land on either side of zero depending on z's exact bit
        // pattern. When it lands negative, `.ln()` of it is NaN, which then propagates
        // through every caller. `scipy.special.gammaln` returns +Inf at these poles by
        // convention, so detect them directly rather than trusting the reflection formula's
        // floating-point residual.
        #[allow(clippy::float_cmp)] // exact-integer test, not a computed-value comparison
        if z <= 0.0 && z == z.trunc() {
            return f64::INFINITY;
        }
        // Between the poles `sin(pi * z)` is genuinely negative on half the intervals
        // (z = -0.5, -2.5, …), so `.ln()` of it is NaN there too — not just at the poles.
        // Γ itself is negative on those intervals, so the real-valued quantity this function
        // can return is `ln|Γ(z)|`, which is what `scipy.special.gammaln` returns
        // (`gammaln(-0.5) = 1.2655…`). Take the magnitude.
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().abs().ln()
            - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = C[0];
    for (i, &c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Regularized incomplete beta `I_x(a, b)`.
#[must_use]
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) where the continued fraction
    // converges fastest (Numerical Recipes criterion).
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp() / a;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;
    // A fixed cap silently returns a partially-converged value once `a`/`b` grow: at
    // x=0.5, a=b=1e6 the continued fraction needs ~420 terms, but a 200-term cap stops
    // early and returns 0.4996679895 against a true value of 0.5 (rel. err 6.6e-4). Not
    // reachable from `student_t_sf` today (it pins b=0.5, which the symmetry switch above
    // routes to the fast side), but this is public API with no documented restriction on
    // a/b. Scale the cap with sqrt(max(a,b)) the same way `gamma_p_series` does for an
    // identical convergence-rate issue, so the tolerance break — not the cap — ends the
    // loop across the practically reachable range.
    let max_iter = 500 + 10 * (a.max(b).max(1.0).sqrt().min(1.0e9) as usize);
    for m in 1..max_iter {
        let m_f = m as f64;
        let num = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= d * c;
        let num = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }
    (front * f).clamp(0.0, 1.0)
}

/// Survival function P(T > t) for Student-t with `df` degrees of freedom.
#[must_use]
pub fn student_t_sf(t: f64, df: f64) -> f64 {
    if !(df.is_finite() && df > 0.0) || t.is_nan() {
        return f64::NAN;
    }
    if t == f64::INFINITY {
        return 0.0;
    }
    if t == f64::NEG_INFINITY {
        return 1.0;
    }
    let x = df / (df + t * t);
    let half_tail = 0.5 * regularized_incomplete_beta(x, 0.5 * df, 0.5);
    if t >= 0.0 { half_tail } else { 1.0 - half_tail }
}

/// Inverse Student-t CDF (quantile function) via bisection on [`student_t_sf`].
///
/// Returns `t` such that `P(T <= t) = p` for `T ~ Student-t(df)`. `p` must lie strictly
/// inside `(0, 1)`; returns `NaN` otherwise.
///
/// `df <= 0` (or non-finite `df`) returns `f64::INFINITY`: a zero- or negative-degrees-of-
/// freedom Student-t has no meaningful critical value, and callers that combine this with
/// an already-infinite standard error (e.g. a single Monte Carlo draw, which has no sample
/// variance) want an unbounded — not NaN — critical multiplier out of `INFINITY * INFINITY`.
#[must_use]
pub fn student_t_ppf(p: f64, df: f64) -> f64 {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    if !(df.is_finite() && df > 0.0) {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }
    // student_t_sf(t, df) is strictly decreasing in t, from 1 (t -> -inf) to 0 (t -> +inf).
    // Solve for the non-negative root and mirror by symmetry for p < 0.5.
    let target_sf = if p > 0.5 { 1.0 - p } else { p };
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while student_t_sf(hi, df) > target_sf && hi < 1e15 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_sf(mid, df) > target_sf {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    if p > 0.5 { t } else { -t }
}

/// Regularized upper incomplete gamma `Q(a, x)`.
#[must_use]
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        (1.0 - gamma_p_series(a, x)).clamp(0.0, 1.0)
    } else {
        gamma_q_cf(a, x).clamp(0.0, 1.0)
    }
}

/// Lower regularized incomplete gamma `P(a, x)` by series expansion.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    // Terms decay like exp(-n^2/2a), so convergence needs O(sqrt(a)) of them -- a fixed cap
    // silently returns a partial sum once `a` grows. Scale the cap with sqrt(a) so the
    // tolerance break, not the cap, ends the loop (mirrored in
    // `regularized_incomplete_beta`'s continued fraction below, an identical issue).
    let max_iter = 500 + 10 * (a.max(1.0).sqrt().min(1.0e9) as usize);
    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Upper regularized incomplete gamma `Q(a, x)` by Lentz continued fraction.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -f64::from(i) * (f64::from(i) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_ppf_pins_common_quantiles() {
        assert!((normal_ppf(0.975) - 1.959_964).abs() < 1e-4);
        assert!((normal_ppf(0.995) - 2.575_829).abs() < 1e-4);
        assert!((normal_ppf(0.95) - 1.644_854).abs() < 1e-4);
        assert!((normal_ppf(0.99) - 2.326_348).abs() < 1e-4);
        assert!(normal_ppf(0.5).abs() < 1e-12);
    }

    #[test]
    fn normal_ppf_monotone_over_grid() {
        let mut prev = f64::NEG_INFINITY;
        for i in 1..1000 {
            let p = f64::from(i) / 1000.0;
            let x = normal_ppf(p);
            assert!(x > prev, "not monotone at p={p}: {x} <= {prev}");
            prev = x;
        }
    }

    #[test]
    fn normal_ppf_symmetric() {
        for &p in &[0.001, 0.01, 0.05, 0.1, 0.25, 0.4, 0.49] {
            let lo = normal_ppf(p);
            let hi = normal_ppf(1.0 - p);
            assert!((lo + hi).abs() < 1e-9, "asymmetry at p={p}: {lo} vs {hi}");
        }
    }

    /// `normal_ppf` must reach near-machine-epsilon accuracy, matching the ~1e-13
    /// precision of the Cody-rational `erf`/`erfc` it is now refined against —
    /// not just the ~1e-9 Acklam approximation on its own.
    ///
    /// References: `scipy.stats.norm.ppf(p)`, scipy 1.14.
    #[test]
    fn normal_ppf_matches_scipy_to_near_machine_epsilon() {
        let cases = [
            (0.975, 1.959_963_984_540_054),
            (0.995, 2.575_829_303_548_900_4),
            (0.9, 1.281_551_565_544_600_4),
            (0.99, 2.326_347_874_040_840_8),
            (1e-6, -4.753_424_308_822_899),
            (1e-10, -6.361_340_902_404_056),
            (1e-12, -7.034_483_825_301_131),
            (1e-15, -7.941_345_326_170_998),
            (1.0 - 1e-6, 4.753_424_308_817_087),
            (1.0 - 1e-12, 7.034_486_910_047_835_6),
            (1.0 - 1e-15, 7.941_444_487_415_979),
        ];
        for (p, expected) in cases {
            let got = normal_ppf(p);
            let rel = (got - expected).abs() / expected.abs().max(1e-8);
            assert!(rel < 5e-10, "normal_ppf({p}): got={got} expected={expected} rel={rel:e}");
        }
    }

    /// `ln_gamma` at non-positive integers (Gamma's poles) must return +Inf, matching
    /// `scipy.special.gammaln`'s convention -- not NaN from a negative-going floating-point
    /// residual in `sin(pi * z)` at the pole.
    #[test]
    fn ln_gamma_returns_infinity_at_non_positive_integer_poles() {
        for z in [0.0, -1.0, -2.0, -3.0, -10.0, -100.0] {
            let got = ln_gamma(z);
            assert!(got.is_infinite() && got > 0.0, "ln_gamma({z}) = {got}, expected +inf");
        }
    }

    #[test]
    fn digamma_trigamma_pin_known_values() {
        const EULER: f64 = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + EULER).abs() < 1e-10);
        assert!((digamma(0.5) + EULER + 2.0 * std::f64::consts::LN_2).abs() < 1e-10);
        assert!((trigamma(1.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-10);
        // Recurrence: ψ(z+1) = ψ(z) + 1/z
        let z = 2.3;
        assert!((digamma(z + 1.0) - digamma(z) - 1.0 / z).abs() < 1e-12);
        assert!((trigamma(z + 1.0) - trigamma(z) + 1.0 / (z * z)).abs() < 1e-12);
    }

    #[test]
    fn trigamma_golden_values() {
        // Reference values from scipy.special.polygamma(1, z).
        let cases = [
            (0.01, 10_001.621_213_528_313),
            (0.1, 101.433_299_150_792_76),
            (0.25, 17.197_329_154_507_113),
            (0.49, 5.108_092_483_881_403),
            (0.5, 4.934_802_200_544_68),
            (1.0, std::f64::consts::PI.powi(2) / 6.0),
            (5.0, 0.221_322_955_737_115_3),
        ];
        for &(z, expected) in &cases {
            let got = trigamma(z);
            let rel = (got - expected).abs() / expected.abs().max(1.0);
            assert!(rel < 1e-10, "trigamma({z}): got={got} expected={expected} rel={rel}");
        }
    }

    #[test]
    fn trigamma_reflection_identity_grid() {
        let pi = std::f64::consts::PI;
        let mut z = 0.01;
        while z < 0.5 {
            let lhs = trigamma(z) + trigamma(1.0 - z);
            let s = (pi * z).sin();
            let rhs = (pi * pi) / (s * s);
            let rel = (lhs - rhs).abs() / rhs.max(1.0);
            assert!(rel < 1e-10, "identity fail at z={z}: lhs={lhs} rhs={rhs}");
            z += 0.01;
        }
    }

    #[test]
    fn student_t_sf_symmetry_and_guards() {
        for &df in &[1.0, 2.0, 5.0, 30.0] {
            assert!((student_t_sf(0.0, df) - 0.5).abs() < 1e-12, "sf(0) df={df}");
            for &t in &[0.5, 1.0, 2.0, 3.5] {
                let pos = student_t_sf(t, df);
                let neg = student_t_sf(-t, df);
                assert!((neg - (1.0 - pos)).abs() < 1e-12, "symmetry t={t} df={df}");
            }
            let mut prev = 1.0;
            for i in -40..=40 {
                let t = f64::from(i) * 0.25;
                let s = student_t_sf(t, df);
                assert!(s <= prev + 1e-12, "not monotone at t={t} df={df}: {s} > {prev}");
                prev = s;
            }
        }
        assert!((student_t_sf(f64::INFINITY, 5.0) - 0.0).abs() < 1e-15);
        assert!((student_t_sf(f64::NEG_INFINITY, 5.0) - 1.0).abs() < 1e-15);
        assert!(student_t_sf(1.0, f64::NAN).is_nan());
        assert!(student_t_sf(1.0, 0.0).is_nan());
        assert!(student_t_sf(1.0, -1.0).is_nan());
        assert!(student_t_sf(f64::NAN, 5.0).is_nan());
    }

    /// `gamma_q` must stay accurate at the large `a` reached by G² on many strata.
    ///
    /// `gamma_q` routes to the series branch whenever `x < a + 1`, which covers the bulk of
    /// the null p-value distribution (a true non-edge puts the statistic near its df). The
    /// series needs O(√a) terms, so a fixed iteration cap silently returns a partial sum as
    /// `a` grows -- inflating p-values, which then propagates into any p-value-ordered step
    /// (BH/BY, edge ranking).
    ///
    /// References are `scipy.special.gammaincc(df/2, df/2)`, scipy 1.14.
    #[test]
    fn gamma_q_series_branch_stays_accurate_at_large_a() {
        // (df, gammaincc(df/2, df/2)) -- the median case, deepest in the series branch.
        let cases = [
            (1.0e3_f64, 0.494_052_853_829_239_64_f64),
            (1.0e4, 0.498_119_365_966_182_67),
            (1.0e5, 0.499_405_291_895_206_7),
            (1.0e6, 0.499_811_936_803_394_5),
        ];
        for (df, expected) in cases {
            let got = gamma_q(df * 0.5, df * 0.5);
            assert!(
                (got - expected).abs() < 1e-6,
                "gamma_q at df={df}: got {got}, scipy says {expected}"
            );
        }
    }

    /// `regularized_incomplete_beta`'s continued fraction must stay accurate for large,
    /// comparable shape parameters. At x=0.5, a=b=1e6 the true value is exactly 0.5 (up to
    /// float rounding), but a fixed 200-term cap converges only partially and returns
    /// 0.4996679895 -- a 6.6e-4 relative error. Convergence there needs ~420 terms; a
    /// 200-term cap silently returns a partial sum instead. Not currently reachable via
    /// `student_t_sf` (which pins b=0.5, routed to the fast symmetry side), but this is
    /// public API with no documented restriction on a/b.
    ///
    /// References are `scipy.special.betainc(a, b, x)`, scipy 1.14.
    #[test]
    fn regularized_incomplete_beta_stays_accurate_for_large_comparable_shapes() {
        let got = regularized_incomplete_beta(0.5, 1.0e6, 1.0e6);
        assert!((got - 0.5).abs() < 1e-6, "I_0.5(1e6, 1e6): got={got}, scipy says ~0.5");

        // Smaller-shape sanity checks against exact scipy references stay unaffected.
        let got2 = regularized_incomplete_beta(0.4, 2.0, 3.0);
        assert!((got2 - 0.5248).abs() < 1e-6, "I_0.4(2,3): got={got2}");
        let got3 = regularized_incomplete_beta(0.3, 0.5, 0.5);
        assert!((got3 - 0.369_010_119_565_545_36).abs() < 1e-6, "I_0.3(0.5,0.5): got={got3}");
    }

    #[test]
    fn student_t_sf_golden_values() {
        // scipy.stats.t.sf(t, df)
        let cases = [
            (1.0, 1.0, 0.25),
            (2.0, 5.0, 0.050_969_739_414_929_14),
            (0.0, 10.0, 0.5),
            (-1.0, 1.0, 0.75),
        ];
        for &(t, df, expected) in &cases {
            let got = student_t_sf(t, df);
            assert!((got - expected).abs() < 1e-8, "t={t} df={df}: got={got} expected={expected}");
        }
    }

    #[test]
    fn student_t_ppf_golden_values() {
        // Well-known two-sided-95%/90% critical-value table entries (t_{alpha,df}), to the
        // precision commonly tabulated in textbooks. A generous 1e-3 tolerance guards
        // against transcription imprecision in the reference digits while still catching a
        // grossly wrong implementation.
        let cases =
            [(0.975, 1.0, 12.706), (0.975, 5.0, 2.571), (0.975, 30.0, 2.042), (0.95, 10.0, 1.812)];
        for &(p, df, expected) in &cases {
            let got = student_t_ppf(p, df);
            assert!((got - expected).abs() < 1e-3, "ppf({p}, {df}): got={got} expected={expected}");
        }
        // Exact closed form at df == 1: Student-t(1) is the standard Cauchy distribution,
        // whose CDF inverts to t = tan(pi * (p - 0.5)).
        let exact_df1 = (std::f64::consts::PI * (0.975 - 0.5)).tan();
        assert!(
            (student_t_ppf(0.975, 1.0) - exact_df1).abs() < 1e-9,
            "df=1 closed form: got={} exact={exact_df1}",
            student_t_ppf(0.975, 1.0)
        );
    }

    #[test]
    fn student_t_ppf_symmetric_and_matches_normal_at_large_df() {
        for &df in &[1.0, 5.0, 30.0] {
            for &p in &[0.6, 0.75, 0.9, 0.99] {
                let hi = student_t_ppf(p, df);
                let lo = student_t_ppf(1.0 - p, df);
                assert!((hi + lo).abs() < 1e-6, "asymmetry p={p} df={df}: hi={hi} lo={lo}");
            }
        }
        // Student-t converges to the standard normal as df -> infinity.
        let t_large_df = student_t_ppf(0.975, 1e7);
        let z = normal_ppf(0.975);
        assert!((t_large_df - z).abs() < 1e-3, "t={t_large_df} z={z}");
    }

    #[test]
    fn student_t_ppf_degenerate_df_is_infinite_not_nan() {
        // df <= 0 (e.g. a single-sample Monte Carlo estimate, n_samples - 1 == 0) must
        // return +inf, not NaN, so combining with an already-infinite stderr in a
        // downstream CI computation stays inf * inf = inf rather than inf * NaN = NaN.
        assert!(student_t_ppf(0.975, 0.0).is_infinite() && student_t_ppf(0.975, 0.0) > 0.0);
        assert!(student_t_ppf(0.975, -1.0).is_infinite() && student_t_ppf(0.975, -1.0) > 0.0);
        assert!(student_t_ppf(1.5, 5.0).is_nan());
        assert!(student_t_ppf(0.0, 5.0).is_nan());
    }
}
