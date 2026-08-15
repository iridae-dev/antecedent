//! Overlap / positivity policy and reports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_core::TargetPopulation;

use crate::error::EstimationError;

/// Overlap / positivity handling.
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

/// Closed propensity interval excluded from the target population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropensityInterval {
    /// Inclusive lower bound in `[0, 1]`.
    pub low: f64,
    /// Inclusive upper bound in `[0, 1]`.
    pub high: f64,
}

/// Sensitivity of ESS / extreme weights to neighboring clip thresholds.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipSensitivity {
    /// Neighboring clip thresholds evaluated (typically `{clip/2, clip, 2·clip}` capped).
    pub thresholds: Arc<[f64]>,
    /// Kish ESS at each threshold (same order as [`Self::thresholds`]).
    pub ess: Arc<[f64]>,
    /// Kish ESS among treated units (`T > 0.5`) at each threshold.
    pub treated_ess: Arc<[f64]>,
    /// Kish ESS among control units (`T ≤ 0.5`) at each threshold.
    pub control_ess: Arc<[f64]>,
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
    /// Fraction of units retained after trim / caliper / empty-stratum attrition
    /// (1.0 when nothing was dropped beyond propensity trim already in `excluded_fraction`).
    pub retained_fraction: f64,
    /// ESS / extreme-weight sensitivity across neighboring clip thresholds.
    pub clip_sensitivity: Option<ClipSensitivity>,
}

/// Production IPW estimand for observed-arm weights (shared by estimators and overlap diagnostics).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IpwTarget {
    /// Average treatment effect on the observed population.
    Ate,
    /// Average treatment effect on the treated.
    Att,
    /// Average treatment effect on the controls.
    Atc,
    /// ATE-style IPW reweighted by `CustomDistribution` observation weights.
    Custom,
}

impl IpwTarget {
    /// Map a [`TargetPopulation`] to the corresponding IPW weight formula.
    ///
    /// # Errors
    ///
    /// Unsupported target population variants.
    pub fn from_population(pop: &TargetPopulation) -> Result<Self, EstimationError> {
        match pop {
            TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => Ok(Self::Ate),
            TargetPopulation::CustomDistribution(_) => Ok(Self::Custom),
            TargetPopulation::Treated => Ok(Self::Att),
            TargetPopulation::Untreated => Ok(Self::Atc),
            _ => Err(EstimationError::unsupported(
                "propensity weighting supports AllObserved, Treated, Untreated, Predicate, or CustomDistribution",
            )),
        }
    }

    /// Observed-arm IPW weight for binary treatment `t` and propensity `e`.
    #[must_use]
    pub fn weight(self, t: f64, e: f64) -> f64 {
        match self {
            Self::Ate | Self::Custom => {
                if t > 0.5 {
                    1.0 / e
                } else {
                    1.0 / (1.0 - e)
                }
            }
            Self::Att => {
                if t > 0.5 {
                    1.0
                } else {
                    e / (1.0 - e)
                }
            }
            Self::Atc => {
                if t > 0.5 {
                    (1.0 - e) / e
                } else {
                    1.0
                }
            }
        }
    }
}

impl OverlapReport {
    /// Build a report from fitted propensities and optional IPW weights.
    ///
    /// Clip-sensitivity ESS is computed only when `clip` is set **and** both `treatment` and
    /// `target` are provided, so diagnostics use the same estimand-specific observed weights as
    /// production estimators. When `observation_weights` is set (`CustomDistribution`), those
    /// weights are multiplied into the sensitivity grid the same way as production IPW.
    #[must_use]
    pub fn from_propensities(
        propensities: &[f64],
        weights: Option<&[f64]>,
        policy: OverlapPolicy,
        treatment: Option<&[f64]>,
        target: Option<IpwTarget>,
        observation_weights: Option<&[f64]>,
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
        let clip_sensitivity = match (clip, treatment, target) {
            (Some(c), Some(treat), Some(ipw_target)) if treat.len() == propensities.len() => {
                Some(clip_sensitivity_grid(propensities, treat, ipw_target, c, observation_weights))
            }
            _ => None,
        };
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
            retained_fraction: 1.0 - excluded_fraction,
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

fn arm_ess(weights: &[f64], treatment: &[f64], treated: bool) -> f64 {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for (&w, &t) in weights.iter().zip(treatment) {
        let is_treated = t > 0.5;
        if is_treated != treated {
            continue;
        }
        sum += w;
        sum_sq += w * w;
    }
    if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 }
}

/// Rebuild estimand-specific observed-arm weights at a clip threshold.
///
/// Uses the same clip bounds as production [`crate::propensity::prepare::clamp_scores`]
/// (no artificial floor/ceiling on `clip`). When `observation_weights` is set, multiplies
/// each arm weight the same way as `CustomDistribution` IPW.
pub(crate) fn observed_ipw_weights(
    treatment: &[f64],
    propensities: &[f64],
    target: IpwTarget,
    clip: f64,
    observation_weights: Option<&[f64]>,
) -> Vec<f64> {
    let lo = clip;
    let hi = 1.0 - clip;
    treatment
        .iter()
        .zip(propensities)
        .enumerate()
        .map(|(i, (&t, &p_raw))| {
            let e = p_raw.clamp(lo, hi);
            let mut w = target.weight(t, e);
            if let Some(ow) = observation_weights {
                w *= ow.get(i).copied().unwrap_or(1.0);
            }
            w
        })
        .collect()
}

fn clip_sensitivity_grid(
    propensities: &[f64],
    treatment: &[f64],
    target: IpwTarget,
    clip: f64,
    observation_weights: Option<&[f64]>,
) -> ClipSensitivity {
    // Neighbor grid stays in (0, 0.5); the applied clip itself is not remapped.
    let candidates = [clip * 0.5, clip, (clip * 2.0).min(0.49)];
    let mut thresholds = Vec::with_capacity(3);
    let mut ess_vals = Vec::with_capacity(3);
    let mut treated_ess = Vec::with_capacity(3);
    let mut control_ess = Vec::with_capacity(3);
    let mut extreme_counts = Vec::with_capacity(3);
    for &thr in &candidates {
        if thresholds.last().is_some_and(|&prev: &f64| (prev - thr).abs() < 1e-15) {
            continue;
        }
        let rebuilt =
            observed_ipw_weights(treatment, propensities, target, thr, observation_weights);
        let (ess, extreme) = weight_summary(&rebuilt);
        thresholds.push(thr);
        ess_vals.push(ess);
        treated_ess.push(arm_ess(&rebuilt, treatment, true));
        control_ess.push(arm_ess(&rebuilt, treatment, false));
        extreme_counts.push(extreme);
    }
    ClipSensitivity {
        thresholds: Arc::from(thresholds),
        ess: Arc::from(ess_vals),
        treated_ess: Arc::from(treated_ess),
        control_ess: Arc::from(control_ess),
        extreme_weight_counts: Arc::from(extreme_counts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propensity::weighting::compute_ipw_weights;

    #[test]
    fn guide_ate_clip_sensitivity_ess() {
        let e = [0.01, 0.20, 0.50, 0.80, 0.99];
        let t = [0.0, 0.0, 0.0, 1.0, 1.0];
        let report = OverlapReport::from_propensities(
            &e,
            None,
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
            Some(&t),
            Some(IpwTarget::Ate),
            None,
        );
        let sens = report.clip_sensitivity.as_ref().expect("clip sensitivity");
        // Threshold list includes the applied clip; find ESS at clip = 0.01.
        let idx = sens.thresholds.iter().position(|&c| (c - 0.01).abs() < 1e-12).expect("0.01");
        assert!((sens.ess[idx] - 4.6383).abs() < 5e-4, "ess={}", sens.ess[idx]);
        assert_eq!(sens.extreme_weight_counts[idx], 0);
        let w = observed_ipw_weights(&t, &e, IpwTarget::Ate, 0.01, None);
        let expected = [1.010_101_010_101_01, 1.25, 2.0, 1.25, 1.010_101_010_101_01];
        for (got, exp) in w.iter().zip(expected) {
            assert!((got - exp).abs() < 1e-9, "w={got} expected={exp}");
        }
    }

    #[test]
    fn diagnostic_weights_match_estimator_for_ate_att_atc() {
        let e = [0.05_f64, 0.2, 0.4, 0.7, 0.95];
        let t = [0.0, 1.0, 0.0, 1.0, 1.0];
        let clip = 0.05_f64;
        for target in [IpwTarget::Ate, IpwTarget::Att, IpwTarget::Atc] {
            let lo = clip;
            let hi = 1.0 - clip;
            let clipped: Vec<f64> = e.iter().map(|&p| p.clamp(lo, hi)).collect();
            let est = compute_ipw_weights(&t, &clipped, &e, target, None);
            let diag = observed_ipw_weights(&t, &e, target, clip, None);
            assert_eq!(est.len(), diag.len());
            for (a, b) in est.iter().zip(&diag) {
                assert!((a - b).abs() < 1e-12, "target={target:?} est={a} diag={b}");
            }
        }
    }

    #[test]
    fn att_swaps_to_atc_under_label_flip() {
        let e = [0.1, 0.3, 0.6, 0.85];
        let t = [0.0, 0.0, 1.0, 1.0];
        let clip = 0.05;
        let weights_treated = observed_ipw_weights(&t, &e, IpwTarget::Att, clip, None);
        let t_flip: Vec<f64> = t.iter().map(|&x| 1.0 - x).collect();
        let e_flip: Vec<f64> = e.iter().map(|&x| 1.0 - x).collect();
        let weights_control_flip =
            observed_ipw_weights(&t_flip, &e_flip, IpwTarget::Atc, clip, None);
        for (a, b) in weights_treated.iter().zip(&weights_control_flip) {
            assert!((a - b).abs() < 1e-12, "att={a} atc_flipped={b}");
        }
        let overlap_treated = OverlapReport::from_propensities(
            &e,
            None,
            OverlapPolicy::RequireDiagnostics { clip: Some(clip), trim: None },
            Some(&t),
            Some(IpwTarget::Att),
            None,
        );
        let overlap_control_flip = OverlapReport::from_propensities(
            &e_flip,
            None,
            OverlapPolicy::RequireDiagnostics { clip: Some(clip), trim: None },
            Some(&t_flip),
            Some(IpwTarget::Atc),
            None,
        );
        let sa = overlap_treated.clip_sensitivity.as_ref().unwrap();
        let sc = overlap_control_flip.clip_sensitivity.as_ref().unwrap();
        for (a, b) in sa.ess.iter().zip(sc.ess.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn clip_sensitivity_absent_without_treatment() {
        let ps = [0.1, 0.5, 0.9];
        let report = OverlapReport::from_propensities(
            &ps,
            None,
            OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: None },
            None,
            None,
            None,
        );
        assert!(report.clip_sensitivity.is_none());
    }

    #[test]
    fn tiny_clip_matches_production_clamp_scores() {
        // Production clamp_scores uses clip as-is; diagnostics must not floor to 1e-6.
        let e = [1e-8_f64, 0.2, 0.5, 0.8, 1.0 - 1e-8];
        let t = [1.0, 0.0, 0.0, 1.0, 1.0];
        let clip = 1e-8_f64;
        let mut clipped = e.to_vec();
        for s in &mut clipped {
            *s = s.clamp(clip, 1.0 - clip);
        }
        let est = compute_ipw_weights(&t, &clipped, &e, IpwTarget::Ate, None);
        let diag = observed_ipw_weights(&t, &e, IpwTarget::Ate, clip, None);
        for (a, b) in est.iter().zip(&diag) {
            assert!((a - b).abs() < 1e-12, "est={a} diag={b}");
        }
        let report = OverlapReport::from_propensities(
            &e,
            None,
            OverlapPolicy::RequireDiagnostics { clip: Some(clip), trim: None },
            Some(&t),
            Some(IpwTarget::Ate),
            None,
        );
        let sens = report.clip_sensitivity.as_ref().unwrap();
        assert!(
            sens.thresholds.iter().any(|&c| (c - clip).abs() < 1e-20),
            "thresholds={:?}",
            sens.thresholds
        );
    }

    #[test]
    fn custom_clip_grid_includes_observation_weights() {
        let e = [0.05_f64, 0.2, 0.4, 0.7, 0.95];
        let t = [0.0, 1.0, 0.0, 1.0, 1.0];
        let ow = [2.0_f64, 0.5, 1.0, 3.0, 0.25];
        let clip = 0.05_f64;
        let mut clipped = e.to_vec();
        for s in &mut clipped {
            *s = s.clamp(clip, 1.0 - clip);
        }
        let mut est = compute_ipw_weights(&t, &clipped, &e, IpwTarget::Custom, None);
        for (w, &o) in est.iter_mut().zip(&ow) {
            *w *= o;
        }
        let (ess_prod, extreme_prod) = weight_summary(&est);
        let report = OverlapReport::from_propensities(
            &e,
            Some(&est),
            OverlapPolicy::RequireDiagnostics { clip: Some(clip), trim: None },
            Some(&t),
            Some(IpwTarget::Custom),
            Some(&ow),
        );
        let sens = report.clip_sensitivity.as_ref().unwrap();
        let idx = sens.thresholds.iter().position(|&c| (c - clip).abs() < 1e-12).unwrap();
        assert!(
            (sens.ess[idx] - ess_prod).abs() < 1e-10,
            "grid ess={} production ess={ess_prod}",
            sens.ess[idx]
        );
        assert_eq!(sens.extreme_weight_counts[idx], extreme_prod);
        // Without observation weights the Custom grid would disagree.
        let bare = observed_ipw_weights(&t, &e, IpwTarget::Custom, clip, None);
        let (ess_bare, _) = weight_summary(&bare);
        assert!((ess_bare - ess_prod).abs() > 1e-6);
    }
}
