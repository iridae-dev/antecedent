//! Multiple-testing / false-discovery-rate helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

/// Multiple-testing adjustment procedure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum MultipleTestingMethod {
    /// Benjamini–Hochberg (1995) FDR (independent / positive regression dependence).
    #[default]
    BenjaminiHochberg,
    /// Benjamini–Yekutieli (2001) FDR (arbitrary dependence; multiplies by harmonic sum).
    BenjaminiYekutieli,
    /// Bonferroni family-wise error control: `min(1, m · p)`.
    Bonferroni,
    /// Holm–Bonferroni step-down FWER control.
    Holm,
}

/// Configuration for adjusting a family of p-values (pinned baseline-style options).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FdrAdjustment {
    /// Correction procedure.
    pub method: MultipleTestingMethod,
    /// When true (pinned baseline default), contemporaneous (lag-0) tests are left
    /// unadjusted — only lagged p-values enter the correction family.
    pub exclude_contemporaneous: bool,
}

impl Default for FdrAdjustment {
    fn default() -> Self {
        Self {
            method: MultipleTestingMethod::BenjaminiHochberg,
            // Matches pinned baseline `get_corrected_pvalues(..., exclude_contemporaneous=True)`.
            exclude_contemporaneous: true,
        }
    }
}

impl FdrAdjustment {
    /// BH with pinned baseline's default contemporaneous exclusion.
    #[must_use]
    pub const fn bh() -> Self {
        Self { method: MultipleTestingMethod::BenjaminiHochberg, exclude_contemporaneous: true }
    }

    /// BY with contemporaneous exclusion.
    #[must_use]
    pub const fn by() -> Self {
        Self { method: MultipleTestingMethod::BenjaminiYekutieli, exclude_contemporaneous: true }
    }

    /// Override contemporaneous exclusion.
    #[must_use]
    pub const fn with_exclude_contemporaneous(mut self, exclude: bool) -> Self {
        self.exclude_contemporaneous = exclude;
        self
    }
}

/// Adjust p-values with the selected procedure (input order preserved).
///
/// Nonfinite values and values outside [0, 1] produce NaN at their positions.
/// They still count toward family size, conservatively acting as p = 1 for
/// adjustment of valid tests; a failed test must not shrink the testing family.
#[must_use]
pub fn adjust_pvalues(p_values: &[f64], method: MultipleTestingMethod) -> Vec<f64> {
    match method {
        MultipleTestingMethod::BenjaminiHochberg => benjamini_hochberg(p_values),
        MultipleTestingMethod::BenjaminiYekutieli => benjamini_yekutieli(p_values),
        MultipleTestingMethod::Bonferroni => bonferroni(p_values),
        MultipleTestingMethod::Holm => holm(p_values),
    }
}

/// Benjamini–Hochberg adjusted p-values (input order preserved).
#[must_use]
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    bh_family(p_values, 1.0)
}

/// Benjamini–Yekutieli adjusted p-values (input order preserved).
///
/// Same step-up form as BH with an extra harmonic factor `H_m = Σ_{i=1}^m 1/i`.
#[must_use]
pub fn benjamini_yekutieli(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut h = 0.0;
    for i in 1..=m {
        h += 1.0 / i as f64;
    }
    bh_family(p_values, h)
}

/// Bonferroni adjusted p-values: `min(1, m · p_i)`.
#[must_use]
pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len() as f64;
    p_values.iter().map(|&p| if valid_pvalue(p) { (p * m).min(1.0) } else { f64::NAN }).collect()
}

/// Holm–Bonferroni adjusted p-values (input order preserved).
#[must_use]
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| adjustment_pvalue(p_values[a]).total_cmp(&adjustment_pvalue(p_values[b])));
    let mut adj = vec![0.0; m];
    let mut running = 0.0_f64;
    for (rank0, &i) in idx.iter().enumerate() {
        let remaining = m - rank0; // m, m-1, ..., 1
        let candidate = (adjustment_pvalue(p_values[i]) * remaining as f64).min(1.0);
        running = running.max(candidate);
        adj[i] = if valid_pvalue(p_values[i]) { running } else { f64::NAN };
    }
    adj
}

fn bh_family(p_values: &[f64], scale: f64) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| adjustment_pvalue(p_values[a]).total_cmp(&adjustment_pvalue(p_values[b])));
    let mut adj = vec![0.0; m];
    let mut running: f64 = 1.0;
    for (rank_rev, &i) in idx.iter().rev().enumerate() {
        let rank = m - rank_rev; // 1..=m from largest p
        let candidate = (adjustment_pvalue(p_values[i]) * scale * m as f64 / rank as f64).min(1.0);
        running = running.min(candidate);
        adj[i] = if valid_pvalue(p_values[i]) { running } else { f64::NAN };
    }
    adj
}

fn valid_pvalue(p: f64) -> bool {
    (0.0..=1.0).contains(&p)
}

fn adjustment_pvalue(p: f64) -> f64 {
    if valid_pvalue(p) { p } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact conservative adjustment / infinite endpoints
    fn invalid_tests_remain_missing_without_reducing_family_size() {
        for method in [
            MultipleTestingMethod::BenjaminiHochberg,
            MultipleTestingMethod::BenjaminiYekutieli,
            MultipleTestingMethod::Bonferroni,
            MultipleTestingMethod::Holm,
        ] {
            for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
                let result = adjust_pvalues(&[0.01, invalid, 0.04], method);
                let conservative = adjust_pvalues(&[0.01, 1.0, 0.04], method);
                assert!(result[1].is_nan());
                assert_eq!(result[0], conservative[0]);
                assert_eq!(result[2], conservative[2]);
            }
        }
    }

    #[test]
    fn bh_preserves_length_and_bounds() {
        let p = [0.001, 0.04, 0.5, 0.02];
        let a = benjamini_hochberg(&p);
        assert_eq!(a.len(), 4);
        assert!(a.iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert!(a[0] <= a[2]);
    }

    #[test]
    fn by_is_at_least_as_conservative_as_bh() {
        let p = [0.001, 0.01, 0.02, 0.04, 0.2];
        let bh = benjamini_hochberg(&p);
        let by = benjamini_yekutieli(&p);
        for (a, b) in bh.iter().zip(by.iter()) {
            assert!(b + 1e-12 >= *a, "bh={a} by={b}");
        }
    }

    #[test]
    fn bonferroni_scales_by_m() {
        let p = [0.01, 0.02];
        let a = bonferroni(&p);
        assert!((a[0] - 0.02).abs() < 1e-12);
        assert!((a[1] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn holm_matches_known_two_test_case() {
        // p = (0.01, 0.04), m=2 → sorted adj: max(0.01*2, ...) = 0.02 then max(0.02, 0.04*1)=0.04
        let p = [0.01, 0.04];
        let a = holm(&p);
        assert!((a[0] - 0.02).abs() < 1e-12);
        assert!((a[1] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn adjust_pvalues_dispatches() {
        let p = [0.01, 0.02, 0.03];
        assert_eq!(
            adjust_pvalues(&p, MultipleTestingMethod::BenjaminiHochberg),
            benjamini_hochberg(&p)
        );
        assert_eq!(adjust_pvalues(&p, MultipleTestingMethod::Holm), holm(&p));
    }
}
