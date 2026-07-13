//! Analytic `ParCorr` significance (Student-t / incomplete beta).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_lossless, clippy::many_single_char_names)]

/// Two-sided analytic p-value for a partial correlation with residual `df`.
pub(crate) fn analytic_parcorr_pvalue(r: f64, df: f64) -> f64 {
    let r = r.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    let t = r * (df / (1.0 - r * r)).sqrt();
    2.0 * student_t_sf(t.abs(), df)
}

/// Fisher-z confidence interval for a partial correlation.
///
/// Returns `(lower, upper)` at the given two-sided `level` (e.g. 0.95).
#[must_use]
pub fn analytic_parcorr_ci(r: f64, df: f64, level: f64) -> (f64, f64) {
    if df <= 0.0 || !(0.0..1.0).contains(&level) {
        return (f64::NAN, f64::NAN);
    }
    let r = r.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let se = 1.0 / (df - 1.0).sqrt();
    // Approximate normal critical value via inverse erf for common levels.
    let alpha = 1.0 - level;
    let zcrit = normal_ppf(1.0 - alpha * 0.5);
    let lo_z = z - zcrit * se;
    let hi_z = z + zcrit * se;
    (z_to_r(lo_z), z_to_r(hi_z))
}

fn z_to_r(z: f64) -> f64 {
    let e = (2.0 * z).exp();
    ((e - 1.0) / (e + 1.0)).clamp(-1.0, 1.0)
}

/// Approximate standard-normal PPF (Acklam’s rational approximation).
pub(crate) fn normal_ppf(p: f64) -> f64 {
    // Coefficients for central region.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_459_740_757e2,
        -3.066_479_806_614_736e1,
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
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5])
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }
    let r = if q > 0.0 { 1.0 - p } else { p };
    let r = (-r.ln()).sqrt();
    let x = if r <= 5.0 {
        let r = r - 1.6;
        (((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5])
            / ((((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0)
    } else {
        // Extreme tail: asymptotic.
        r
    };
    if q < 0.0 { -x } else { x }
}

/// Survival function P(T > t) for Student-t with `df` degrees of freedom.
fn student_t_sf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 * regularized_incomplete_beta(x, df * 0.5, 0.5)
}

fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
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

fn ln_gamma(z: f64) -> f64 {
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
