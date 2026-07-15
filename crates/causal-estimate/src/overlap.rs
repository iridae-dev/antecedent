//! Overlap / positivity policy and reports (DESIGN.md §14.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

/// Overlap / positivity handling (DESIGN §14.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlapPolicy {
    /// Explicitly skip propensity-based overlap (linear adjustment path).
    ExplicitOverride,
    /// Require propensity diagnostics; optional clip/trim thresholds in `(0, 0.5)`.
    RequireDiagnostics {
        /// Clip propensities into `[clip, 1 - clip]` when `Some`.
        clip: Option<f64>,
        /// Drop units outside `[trim, 1 - trim]` when `Some`.
        trim: Option<f64>,
    },
}

impl OverlapPolicy {
    /// Require diagnostics with no clipping or trimming.
    #[must_use]
    pub const fn require_diagnostics() -> Self {
        Self::RequireDiagnostics { clip: None, trim: None }
    }
}

/// Closed propensity interval excluded from the target population (DESIGN §14.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropensityInterval {
    /// Inclusive lower bound in `[0, 1]`.
    pub low: f64,
    /// Inclusive upper bound in `[0, 1]`.
    pub high: f64,
}

/// Sensitivity of ESS / extreme weights to neighboring clip thresholds (DESIGN §14.3).
#[derive(Clone, Debug, PartialEq)]
pub struct ClipSensitivity {
    /// Neighboring clip thresholds evaluated (typically `{clip/2, clip, 2·clip}` capped).
    pub thresholds: Arc<[f64]>,
    /// Kish ESS at each threshold (same order as [`Self::thresholds`]).
    pub ess: Arc<[f64]>,
    /// Extreme-weight counts (`w > 10`) at each threshold.
    pub extreme_weight_counts: Arc<[u32]>,
}

/// Propensity overlap / positivity report retained on estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlapReport {
    /// Minimum fitted propensity (before clipping).
    pub propensity_min: f64,
    /// Maximum fitted propensity (before clipping).
    pub propensity_max: f64,
    /// Kish effective sample size of the applied weights (`None` when weights were not supplied).
    pub ess: Option<f64>,
    /// Count of weights above the extreme-weight threshold (default 10).
    pub extreme_weight_count: u32,
    /// Fraction of rows excluded by trimming (0 if no trim).
    pub excluded_fraction: f64,
    /// Fraction of units whose propensity lies in the retained common-support band.
    ///
    /// Band is `[clip, 1 - clip]` when clip is set, else `[trim, 1 - trim]` when trim is set,
    /// else the full unit interval (support = 1).
    pub target_population_support: f64,
    /// Propensity intervals excluded by trimming (empty when no trim).
    pub excluded_regions: Arc<[PropensityInterval]>,
    /// Clip threshold applied, if any.
    pub clip: Option<f64>,
    /// Trim threshold applied, if any.
    pub trim: Option<f64>,
    /// ESS / extreme-weight sensitivity across neighboring clip thresholds.
    pub clip_sensitivity: Option<ClipSensitivity>,
}

impl OverlapReport {
    /// Build a report from fitted propensities and optional IPW weights.
    #[must_use]
    pub fn from_propensities(
        propensities: &[f64],
        weights: Option<&[f64]>,
        policy: OverlapPolicy,
    ) -> Self {
        let (clip, trim) = match policy {
            OverlapPolicy::ExplicitOverride => (None, None),
            OverlapPolicy::RequireDiagnostics { clip, trim } => (clip, trim),
        };
        let mut min_p = f64::INFINITY;
        let mut max_p = f64::NEG_INFINITY;
        let mut excluded = 0u32;
        let mut in_support = 0u32;
        let support_lo = clip.or(trim).unwrap_or(0.0);
        let support_hi = 1.0 - support_lo;
        for &p in propensities {
            min_p = min_p.min(p);
            max_p = max_p.max(p);
            if let Some(t) = trim {
                if p < t || p > 1.0 - t {
                    excluded = excluded.saturating_add(1);
                }
            }
            if p >= support_lo && p <= support_hi {
                in_support = in_support.saturating_add(1);
            }
        }
        if propensities.is_empty() {
            min_p = f64::NAN;
            max_p = f64::NAN;
        }
        let n = propensities.len().max(1) as f64;
        let excluded_fraction = f64::from(excluded) / n;
        let target_population_support =
            if propensities.is_empty() { f64::NAN } else { f64::from(in_support) / n };
        let excluded_regions: Arc<[PropensityInterval]> = match trim {
            Some(t) if t > 0.0 => Arc::from([
                PropensityInterval { low: 0.0, high: t },
                PropensityInterval { low: 1.0 - t, high: 1.0 },
            ]),
            _ => Arc::from([]),
        };
        let (ess, extreme_weight_count) = match weights {
            Some(w) if !w.is_empty() => {
                let (e, c) = weight_summary(w);
                (Some(e), c)
            }
            _ => (None, 0),
        };
        let clip_sensitivity = clip.map(|c| clip_sensitivity_grid(propensities, c));
        Self {
            propensity_min: min_p,
            propensity_max: max_p,
            ess,
            extreme_weight_count,
            excluded_fraction,
            target_population_support,
            excluded_regions,
            clip,
            trim,
            clip_sensitivity,
        }
    }
}

fn weight_summary(weights: &[f64]) -> (f64, u32) {
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|x| x * x).sum();
    let ess = if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 };
    let extreme = weights.iter().filter(|&&x| x > 10.0).count();
    (ess, u32::try_from(extreme).unwrap_or(u32::MAX))
}

/// ATE-style IPW weights from propensities at a clip threshold (for sensitivity grids).
fn ate_ipw_weights_from_propensity(propensities: &[f64], clip: f64) -> Vec<f64> {
    let lo = clip.clamp(1e-6, 0.49);
    let hi = 1.0 - lo;
    propensities
        .iter()
        .map(|&p_raw| {
            let p = p_raw.clamp(lo, hi);
            // Two-arm contribution: treat as treated weight 1/p for sensitivity shape.
            1.0 / p + 1.0 / (1.0 - p)
        })
        .collect()
}

fn clip_sensitivity_grid(propensities: &[f64], clip: f64) -> ClipSensitivity {
    let c = clip.clamp(1e-6, 0.49);
    let candidates = [c * 0.5, c, (c * 2.0).min(0.49)];
    let mut thresholds = Vec::with_capacity(3);
    let mut ess_vals = Vec::with_capacity(3);
    let mut extreme_counts = Vec::with_capacity(3);
    for &thr in &candidates {
        if thresholds.last().is_some_and(|&prev: &f64| (prev - thr).abs() < 1e-15) {
            continue;
        }
        let rebuilt = ate_ipw_weights_from_propensity(propensities, thr);
        let (ess, extreme) = weight_summary(&rebuilt);
        thresholds.push(thr);
        ess_vals.push(ess);
        extreme_counts.push(extreme);
    }
    ClipSensitivity {
        thresholds: Arc::from(thresholds),
        ess: Arc::from(ess_vals),
        extreme_weight_counts: Arc::from(extreme_counts),
    }
}
