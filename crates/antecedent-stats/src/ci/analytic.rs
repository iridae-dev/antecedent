//! Analytic `ParCorr` significance (Student-t / incomplete beta).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_lossless, clippy::many_single_char_names)]

use crate::special::{normal_ppf, student_t_sf};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::normal_ppf;

    #[test]
    fn fisher_z_ci_covers_r_and_uses_correct_zcrit() {
        // r = 0.5, df = 100 → z = 0.5493, se = 1/sqrt(99), zcrit(95%) = 1.959964.
        let (lo, hi) = analytic_parcorr_ci(0.5, 100.0, 0.95);
        let z = 0.5 * (1.5_f64 / 0.5).ln();
        let se = 1.0 / 99.0_f64.sqrt();
        let expect_lo =
            ((2.0 * (z - 1.959_964 * se)).exp() - 1.0) / ((2.0 * (z - 1.959_964 * se)).exp() + 1.0);
        let expect_hi =
            ((2.0 * (z + 1.959_964 * se)).exp() - 1.0) / ((2.0 * (z + 1.959_964 * se)).exp() + 1.0);
        assert!((lo - expect_lo).abs() < 1e-4);
        assert!((hi - expect_hi).abs() < 1e-4);
        assert!(lo < 0.5 && 0.5 < hi);
        let _ = normal_ppf(0.975);
    }
}
