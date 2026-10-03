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

/// Descriptive maximum across cluster-studentized, cohort-specific pre-period
/// contrasts. This is not a calibrated joint hypothesis test or evidence that
/// parallel trends holds when the value is small.
#[derive(Clone, Debug, PartialEq)]
pub struct PretrendFalsificationStatistic {
    /// Maximum absolute effect divided by its cluster sandwich standard error.
    pub max_absolute_cluster_studentized_lead: f64,
    /// Number of non-reference pre-period contrasts jointly summarized.
    pub leads: usize,
    /// Smallest number of independent clusters among the summarized contrasts.
    pub min_clusters: usize,
}

/// Summarize at least two pre-adoption leads; refuse a statistic when any
/// constituent cluster SE is degenerate. No p-value or calibrated cutoff is
/// attached to this descriptive statistic.
pub fn pretrend_falsification_statistic(
    effects: &[EventTimeEffect],
) -> Result<PretrendFalsificationStatistic, &'static str> {
    let leads = effects.iter().filter(|effect| effect.event_time < -1).collect::<Vec<_>>();
    if leads.len() < 2 {
        return Err("at least two non-reference pre-period contrasts are required");
    }
    if leads.iter().any(|effect| {
        !effect.effect.is_finite()
            || !effect.standard_error.is_finite()
            || effect.standard_error <= 0.0
            || effect.clusters < 4
    }) {
        return Err(
            "every pre-period contrast needs finite effect, positive cluster SE, and at least four clusters",
        );
    }
    let maximum = leads
        .iter()
        .map(|effect| (effect.effect / effect.standard_error).abs())
        .fold(0.0_f64, f64::max);
    if !maximum.is_finite() {
        return Err("pre-period cluster-studentized statistic overflowed finite precision");
    }
    Ok(PretrendFalsificationStatistic {
        max_absolute_cluster_studentized_lead: maximum,
        leads: leads.len(),
        min_clusters: leads.iter().map(|effect| effect.clusters).min().unwrap_or(0),
    })
}

/// Pointwise 95% interval for an event-time contrast only when each side has
/// enough independent clusters for the calibrated normal approximation.
/// This does not establish simultaneous coverage of the event-study curve.
#[must_use]
pub fn pointwise_interval_95(
    effect: &EventTimeEffect,
    treated_clusters: usize,
    control_clusters: usize,
) -> Option<[f64; 2]> {
    if treated_clusters < 24
        || control_clusters < 24
        || effect.clusters < 48
        || !effect.effect.is_finite()
        || !effect.standard_error.is_finite()
        || effect.standard_error <= 0.0
    {
        return None;
    }
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
    if n == 0
        || subjects.len() != n
        || periods.len() != n
        || cohorts.len() != n
        || clusters.len() != n
    {
        return Err(
            "outcome, subject, period, cohort, and cluster vectors must have equal non-zero length"
                .into(),
        );
    }
    let mut units: BTreeMap<&str, (i64, &str, BTreeMap<i64, f64>)> = BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        let subject = subjects[i].as_ref();
        let cluster = clusters[i].as_ref();
        if subject.trim().is_empty()
            || cluster.trim().is_empty()
            || !outcomes[i].is_finite()
            || periods[i] <= 0
            || cohorts[i] < 0
        {
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
    if all_periods.len() < 2
        || units.values().any(|unit| {
            unit.2.len() != all_periods.len()
                || all_periods.iter().any(|period| !unit.2.contains_key(period))
        })
    {
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
            // Each unit's period-vs-baseline delta, computed once in the means
            // pass (same iteration order) and reused in the score/cluster pass
            // below rather than repeating the two BTreeMap lookups.
            let deltas: [Vec<f64>; 2] = [
                controls.iter().map(|unit| unit.2[&period] - unit.2[&baseline]).collect(),
                treated.iter().map(|unit| unit.2[&period] - unit.2[&baseline]).collect(),
            ];
            let means: [f64; 2] = [
                deltas[0].iter().sum::<f64>() / controls.len() as f64,
                deltas[1].iter().sum::<f64>() / treated.len() as f64,
            ];
            let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
            let mut group_clusters: [BTreeSet<&str>; 2] = Default::default();
            for (group, selected) in [(0usize, &controls), (1usize, &treated)] {
                for (unit, &delta) in selected.iter().zip(deltas[group].iter()) {
                    group_clusters[group].insert(unit.1);
                    *scores.entry(unit.1).or_default() += (if group == 1 { 1.0 } else { -1.0 })
                        * (delta - means[group])
                        / selected.len() as f64;
                }
            }
            if group_clusters.iter().any(|set| set.len() < 2) {
                return Err("cluster-robust standard error requires at least two clusters in each cohort/control group".into());
            }
            let g = scores.len();
            let variance =
                g as f64 / (g - 1) as f64 * scores.values().map(|score| score * score).sum::<f64>();
            let effect = means[1] - means[0];
            if !effect.is_finite() || !variance.is_finite() || variance < 0.0 {
                return Err("event-time effect or cluster variance overflowed".into());
            }
            effects.push(EventTimeEffect {
                cohort,
                period,
                event_time: period - cohort,
                effect,
                treated_subjects: treated.len(),
                comparison_subjects: controls.len(),
                standard_error: variance.sqrt(),
                clusters: g,
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
    #![cfg_attr(
        test,
        allow(clippy::float_cmp, reason = "tests assert exact deterministic estimates")
    )]
    use super::*;

    #[test]
    fn finite_panel_values_cannot_publish_infinite_event_effects() {
        let outcomes = [-1e308, 1e308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let subjects = ["t1", "t1", "t2", "t2", "c1", "c1", "c2", "c2"];
        let periods = [1, 2, 1, 2, 1, 2, 1, 2];
        let cohorts = [2, 2, 2, 2, 0, 0, 0, 0];
        let clusters = ["t1", "t1", "t2", "t2", "c1", "c1", "c2", "c2"];
        assert!(estimate(&outcomes, &subjects, &periods, &cohorts, &clusters).is_err());
    }

    #[test]
    fn joint_preperiod_cluster_statistic_has_no_inferential_cutoff() {
        let lead = |event_time, effect, standard_error| EventTimeEffect {
            cohort: 4,
            period: 4 + event_time,
            event_time,
            effect,
            treated_subjects: 8,
            comparison_subjects: 8,
            standard_error,
            clusters: 16,
        };
        let effects = [lead(-3, 1.0, 0.5), lead(-2, -3.0, 1.0), lead(0, 4.0, 0.1)];
        let summary = pretrend_falsification_statistic(&effects).unwrap();
        assert_eq!(summary.leads, 2);
        assert_eq!(summary.min_clusters, 16);
        assert_eq!(summary.max_absolute_cluster_studentized_lead, 3.0);
        assert!(pretrend_falsification_statistic(&effects[..1]).is_err());
        assert!(
            pretrend_falsification_statistic(&[lead(-3, 1.0, 0.0), effects[1].clone()]).is_err()
        );
    }

    #[test]
    fn calibrated_event_time_cluster_coverage_and_thin_cluster_boundary() {
        let simulations = 2_000;
        let mut state = 0x39AD_5C72_0F81_EB46_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
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
            let mut covered_90 = [0_usize; 2];
            let mut covered_95 = [0_usize; 2];
            for _ in 0..simulations {
                let outcomes = cohorts
                    .iter()
                    .zip(&periods)
                    .map(|(&cohort, &period)| {
                        0.5 * period as f64
                            + if cohort == 3 && period >= 3 { 2.0 } else { 0.0 }
                            + (uniform() - 0.5) * 12.0_f64.sqrt()
                    })
                    .collect::<Vec<_>>();
                let effects =
                    estimate(&outcomes, &subjects, &periods, &cohorts, &clusters).unwrap();
                for (target, event_time) in [0_i64, 1].into_iter().enumerate() {
                    let fit =
                        effects.iter().find(|effect| effect.event_time == event_time).unwrap();
                    assert!(fit.standard_error > 0.0);
                    let supported =
                        pointwise_interval_95(fit, clusters_per_group, clusters_per_group);
                    assert_eq!(supported.is_some(), clusters_per_group >= 24);
                    covered_95[target] += usize::from(
                        (fit.effect - 2.0).abs()
                            <= antecedent_stats::normal_ppf(0.975) * fit.standard_error,
                    );
                    if let Some(bounds) = supported {
                        assert_eq!(
                            bounds[0] <= 2.0 && 2.0 <= bounds[1],
                            (fit.effect - 2.0).abs()
                                <= antecedent_stats::normal_ppf(0.975) * fit.standard_error
                        );
                    }
                    covered_90[target] += usize::from(
                        (fit.effect - 2.0).abs()
                            <= antecedent_stats::normal_ppf(0.95) * fit.standard_error,
                    );
                }
            }
            let coverage_90 = covered_90.map(|hits| hits as f64 / f64::from(simulations));
            let coverage_95 = covered_95.map(|hits| hits as f64 / f64::from(simulations));
            println!(
                "staggered event coverage, clusters/group={clusters_per_group}: 90%={coverage_90:?}, 95%={coverage_95:?}"
            );
            if clusters_per_group >= 24 {
                for (nominal, coverage) in [(0.90_f64, coverage_90), (0.95, coverage_95)] {
                    let mcse = (nominal * (1.0 - nominal) / f64::from(simulations)).sqrt();
                    assert!(coverage.iter().all(|rate| (rate - nominal).abs() <= 3.0 * mcse));
                }
            }
        }
    }

    #[test]
    fn shared_cluster_shocks_keep_supported_post_event_coverage() {
        let clusters_per_group = 24;
        let subjects_per_cluster = 2;
        let simulations = 2_000;
        let mut state = 0x7EC4_2A91_5D0F_B836_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        let mut subjects = Vec::new();
        let mut clusters = Vec::new();
        let mut cohorts = Vec::new();
        let mut periods = Vec::new();
        for (group, cohort) in [("control", 0), ("treated", 3)] {
            for cluster in 0..clusters_per_group {
                let cluster_id = format!("{group}-{cluster}");
                for subject in 0..subjects_per_cluster {
                    let id = format!("{cluster_id}-{subject}");
                    for period in 1..=4 {
                        subjects.push(id.clone());
                        clusters.push(cluster_id.clone());
                        cohorts.push(cohort);
                        periods.push(period);
                    }
                }
            }
        }
        let mut covered = [0_usize; 2];
        for _ in 0..simulations {
            let mut outcomes = Vec::with_capacity(subjects.len());
            for cohort in [0, 3] {
                for _ in 0..clusters_per_group {
                    let cluster_shock =
                        (0..4).map(|_| (uniform() - 0.5) * 12.0_f64.sqrt()).collect::<Vec<_>>();
                    for _ in 0..subjects_per_cluster {
                        for period in 1..=4 {
                            let noise = 0.5 * (uniform() - 0.5) * 12.0_f64.sqrt();
                            outcomes.push(
                                0.5 * period as f64
                                    + cluster_shock[period - 1]
                                    + noise
                                    + if cohort == 3 && period >= 3 { 2.0 } else { 0.0 },
                            );
                        }
                    }
                }
            }
            let effects = estimate(&outcomes, &subjects, &periods, &cohorts, &clusters).unwrap();
            for (target, event_time) in [0_i64, 1].into_iter().enumerate() {
                let fit = effects.iter().find(|effect| effect.event_time == event_time).unwrap();
                let bounds =
                    pointwise_interval_95(fit, clusters_per_group, clusters_per_group).unwrap();
                covered[target] += usize::from(bounds[0] <= 2.0 && 2.0 <= bounds[1]);
            }
        }
        let coverage = covered.map(|hits| hits as f64 / f64::from(simulations));
        println!("staggered shared-cluster post-event 95% coverage: {coverage:?}");
        let mcse = (0.95_f64 * 0.05 / f64::from(simulations)).sqrt();
        assert!(coverage.iter().all(|rate| (rate - 0.95).abs() <= 3.0 * mcse));
    }
}
