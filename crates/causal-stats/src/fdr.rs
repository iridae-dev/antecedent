//! Multiple-testing / false-discovery-rate helpers (DESIGN.md §11.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

/// Multiple-testing adjustment procedure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum MultipleTestingMethod {
    /// Benjamini–Hochberg FDR (independent / positive regression dependence).
    #[default]
    BenjaminiHochberg,
    /// Benjamini–Yekutieli FDR (arbitrary dependence; multiplies by harmonic sum).
    BenjaminiYekutieli,
    /// Bonferroni family-wise error control: `min(1, m · p)`.
    Bonferroni,
    /// Holm–Bonferroni step-down FWER control.
    Holm,
}

/// Configuration for adjusting a family of p-values (tigramite-style options).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FdrAdjustment {
    /// Correction procedure.
    pub method: MultipleTestingMethod,
    /// When true (tigramite default), contemporaneous (lag-0) tests are left
    /// unadjusted — only lagged p-values enter the correction family.
    pub exclude_contemporaneous: bool,
}

impl Default for FdrAdjustment {
    fn default() -> Self {
        Self {
            method: MultipleTestingMethod::BenjaminiHochberg,
            // Matches tigramite `get_corrected_pvalues(..., exclude_contemporaneous=True)`.
            exclude_contemporaneous: true,
        }
    }
}

impl FdrAdjustment {
    /// BH with tigramite's default contemporaneous exclusion.
    #[must_use]
    pub const fn bh() -> Self {
        Self {
            method: MultipleTestingMethod::BenjaminiHochberg,
            exclude_contemporaneous: true,
        }
    }

    /// BY with contemporaneous exclusion.
    #[must_use]
    pub const fn by() -> Self {
        Self {
            method: MultipleTestingMethod::BenjaminiYekutieli,
            exclude_contemporaneous: true,
        }
    }

    /// Override contemporaneous exclusion.
    #[must_use]
    pub const fn with_exclude_contemporaneous(mut self, exclude: bool) -> Self {
        self.exclude_contemporaneous = exclude;
        self
    }
}

/// Adjust p-values with the selected procedure (input order preserved).
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
    p_values.iter().map(|&p| (p * m).min(1.0).max(0.0)).collect()
}

/// Holm–Bonferroni adjusted p-values (input order preserved).
#[must_use]
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| {
        p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut adj = vec![0.0; m];
    let mut running = 0.0_f64;
    for (rank0, &i) in idx.iter().enumerate() {
        let remaining = m - rank0; // m, m-1, ..., 1
        let candidate = (p_values[i] * remaining as f64).min(1.0);
        running = running.max(candidate);
        adj[i] = running;
    }
    adj
}

fn bh_family(p_values: &[f64], scale: f64) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| {
        p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut adj = vec![0.0; m];
    let mut running: f64 = 1.0;
    for (rank_rev, &i) in idx.iter().rev().enumerate() {
        let rank = m - rank_rev; // 1..=m from largest p
        let candidate = (p_values[i] * scale * m as f64 / rank as f64).min(1.0);
        running = running.min(candidate);
        adj[i] = running;
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            adjust_pvalues(&p, MultipleTestingMethod::Holm),
            holm(&p)
        );
    }
}
