//! Cohort-specific event-time contrasts against never-treated subjects.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
/// One cohort-specific event-time contrast and pointwise cluster uncertainty.
pub struct EventTimeEffect {
    /// First treated period for this adoption cohort.
    pub cohort: i64,
    /// Calendar period of the contrast.
    pub period: i64,
    /// Calendar period minus adoption cohort; the reference `-1` is omitted.
    pub event_time: i64,
    /// Difference in outcome changes from the cohort-specific reference period.
    pub effect: f64,
    /// Subjects in the adoption cohort.
    pub treated_subjects: usize,
    /// Never-treated comparison subjects.
    pub comparison_subjects: usize,
    /// Pointwise cluster sandwich standard error, without an interval claim.
    pub standard_error: f64,
    /// Distinct inference clusters in the comparison.
    pub clusters: usize,
}

/// Pointwise 95% interval for an event-time contrast only when each side has
/// enough independent clusters for the calibrated normal approximation.
/// This does not establish simultaneous coverage of the event-study curve.
pub fn pointwise_interval_95(
    effect: &EventTimeEffect, treated_clusters: usize, control_clusters: usize,
) -> Option<[f64; 2]> {
    if treated_clusters < 24 || control_clusters < 24
        || effect.clusters < 48 || !effect.effect.is_finite()
        || !effect.standard_error.is_finite() || effect.standard_error <= 0.0
    { return None; }
    let span = antecedent_stats::normal_ppf(0.975) * effect.standard_error;
    let bounds = [effect.effect - span, effect.effect + span];
    bounds.iter().all(|bound| bound.is_finite()).then_some(bounds)
}

/// Balanced-panel event-time comparisons relative to cohort-specific period `g - 1`.
/// Pre-adoption contrasts are descriptive diagnostics, not a parallel-trends test.
pub fn estimate(
    outcomes: &[f64],
    subjects: &[impl AsRef<str>],
    periods: &[i64],
    cohorts: &[i64],
    clusters: &[impl AsRef<str>],
) -> Result<Vec<EventTimeEffect>, String> {
    let n = outcomes.len();
    if n == 0 || subjects.len() != n || periods.len() != n || cohorts.len() != n || clusters.len() != n {
        return Err("outcome, subject, period, cohort, and cluster vectors must have equal non-zero length".into());
    }
    let mut units: BTreeMap<&str, (i64, &str, BTreeMap<i64, f64>)> = BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        let subject = subjects[i].as_ref();
        let cluster = clusters[i].as_ref();
        if subject.trim().is_empty() || cluster.trim().is_empty() || !outcomes[i].is_finite() || periods[i] <= 0 || cohorts[i] < 0 {
            return Err("event study requires finite outcomes, non-empty IDs, positive periods, and nonnegative cohorts".into());
        }
        all_periods.insert(periods[i]);
        let unit = units.entry(subject).or_insert_with(|| (cohorts[i], cluster, BTreeMap::new()));
        if unit.0 != cohorts[i] || unit.1 != cluster {
            return Err("adoption cohort and cluster must remain constant within subject".into());
        }
        if unit.2.insert(periods[i], outcomes[i]).is_some() {
            return Err("each subject must have exactly one observation per period".into());
        }
    }
    if all_periods.len() < 2 || units.values().any(|unit| unit.2.len() != all_periods.len() || all_periods.iter().any(|period| !unit.2.contains_key(period))) {
        return Err("event study requires a balanced panel with at least two common periods".into());
    }
    let cohorts_present: BTreeSet<i64> = units.values().map(|unit| unit.0).collect();
    if !cohorts_present.contains(&0) {
        return Err("event study requires never-treated controls (cohort 0)".into());
    }
    let controls: Vec<_> = units.values().filter(|unit| unit.0 == 0).collect();
    let mut effects = Vec::new();
    for cohort in cohorts_present.into_iter().filter(|cohort| *cohort > 0) {
        let baseline = cohort - 1;
        if !all_periods.contains(&baseline) || !all_periods.contains(&cohort) {
            return Err("each adoption cohort needs an observed immediately pre-treatment baseline and adoption period".into());
        }
        let treated: Vec<_> = units.values().filter(|unit| unit.0 == cohort).collect();
        for period in all_periods.iter().copied().filter(|period| *period != baseline) {
            let means: [f64; 2] = [
                controls.iter().map(|unit| unit.2[&period] - unit.2[&baseline]).sum::<f64>() / controls.len() as f64,
                treated.iter().map(|unit| unit.2[&period] - unit.2[&baseline]).sum::<f64>() / treated.len() as f64,
            ];
            let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
            let mut group_clusters: [BTreeSet<&str>; 2] = Default::default();
            for (group, selected) in [(0usize, &controls), (1usize, &treated)] {
                for unit in selected {
                    group_clusters[group].insert(unit.1);
                    let delta = unit.2[&period] - unit.2[&baseline];
                    *scores.entry(unit.1).or_default() += (if group == 1 { 1.0 } else { -1.0 })
                        * (delta - means[group]) / selected.len() as f64;
                }
            }
            if group_clusters.iter().any(|set| set.len() < 2) {
                return Err("cluster-robust standard error requires at least two clusters in each cohort/control group".into());
            }
            let g = scores.len();
            let variance = g as f64 / (g - 1) as f64 * scores.values().map(|score| score * score).sum::<f64>();
            effects.push(EventTimeEffect {
                cohort, period, event_time: period - cohort,
                effect: means[1] - means[0],
                treated_subjects: treated.len(), comparison_subjects: controls.len(),
                standard_error: variance.max(0.0).sqrt(), clusters: g,
            });
        }
    }
    if effects.is_empty() {
        return Err("no supported event-time comparisons are available".into());
    }
    Ok(effects)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_event_time_cluster_coverage_and_thin_cluster_boundary() {
        let simulations = 2_000;
        let mut state = 0x39AD_5C72_0F81_EB46_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / ((1_u64 << 53) as f64)
        };
        for clusters_per_group in [4, 8, 16, 24, 32] {
            let mut subjects = Vec::new();
            let mut clusters = Vec::new();
            let mut cohorts = Vec::new();
            let mut periods = Vec::new();
            for (group, cohort) in [("control", 0), ("treated", 3)] {
                for cluster in 0..clusters_per_group {
                    let id = format!("{group}-{cluster}");
                    for period in 1..=4 {
                        subjects.push(id.clone());
                        clusters.push(id.clone());
                        cohorts.push(cohort);
                        periods.push(period);
                    }
                }
            }
            let mut covered_95 = [0_usize; 2];
            for _ in 0..simulations {
                let outcomes = cohorts.iter().zip(&periods).map(|(&cohort, &period)| {
                    0.5 * period as f64 + if cohort == 3 && period >= 3 { 2.0 } else { 0.0 }
                        + (uniform() - 0.5) * 12.0_f64.sqrt()
                }).collect::<Vec<_>>();
                let effects = estimate(&outcomes, &subjects, &periods, &cohorts, &clusters).unwrap();
                for (target, event_time) in [0_i64, 1].into_iter().enumerate() {
                    let fit = effects.iter().find(|effect| effect.event_time == event_time).unwrap();
                    assert!(fit.standard_error > 0.0);
                    let supported = pointwise_interval_95(fit, clusters_per_group, clusters_per_group);
                    assert_eq!(supported.is_some(), clusters_per_group >= 24);
                    covered_95[target] += usize::from((fit.effect - 2.0).abs()
                        <= antecedent_stats::normal_ppf(0.975) * fit.standard_error);
                    if let Some(bounds) = supported {
                        assert_eq!(bounds[0] <= 2.0 && 2.0 <= bounds[1],
                            (fit.effect - 2.0).abs() <= antecedent_stats::normal_ppf(0.975) * fit.standard_error);
                    }
                }
            }
            let coverage = covered_95.map(|hits| hits as f64 / simulations as f64);
            println!("staggered event 95% coverage, clusters/group={clusters_per_group}: {coverage:?}");
            if clusters_per_group >= 24 {
                let mcse = (0.95_f64 * 0.05 / simulations as f64).sqrt();
                assert!(coverage.iter().all(|rate| (rate - 0.95).abs() <= 3.0 * mcse));
            }
        }
    }
}
