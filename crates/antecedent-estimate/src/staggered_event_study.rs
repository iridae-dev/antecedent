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
