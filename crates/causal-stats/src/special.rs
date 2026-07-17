//! Shared special functions for stats (PPF, gamma, incomplete beta).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_lossless, clippy::many_single_char_names)]

/// Approximate standard-normal PPF (Acklam’s rational approximation).
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
    if p < P_LOW {
        // Lower tail.
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > 1.0 - P_LOW {
        // Upper tail.
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    // Central region.
    let q = p - 0.5;
    let r = q * q;
    q * (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5])
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
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
                                * (1.0 / 240.0
                                    - iz2 * (1.0 / 132.0 - iz2 * (691.0 / 32760.0))))));
    result
}

/// Trigamma `ψ₁(z) = d²/dz² ln Γ(z)` (reflection + asymptotic series).
#[must_use]
pub fn trigamma(mut z: f64) -> f64 {
    if !(z.is_finite() && z > 0.0) {
        return f64::NAN;
    }
    let mut result = 0.0;
    // Reflection: ψ₁(z) + ψ₁(1−z) = π² / sin²(πz).
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        let s = (pi * z).sin();
        result += (pi * pi) / (s * s);
        z = 1.0 - z;
    }
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
            - iz2
                * (1.0 / 30.0
                    - iz2 * (1.0 / 42.0 - iz2 * (1.0 / 30.0 - iz2 * (5.0 / 66.0)))));
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
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
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
    for m in 1..200 {
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
    let x = df / (df + t * t);
    0.5 * regularized_incomplete_beta(x, df * 0.5, 0.5)
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
    for _ in 0..500 {
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
}
