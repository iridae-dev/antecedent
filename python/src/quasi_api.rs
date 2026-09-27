//! Small, native estimators for quasi-experimental designs.
// SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Retained panel DiD result; cluster-robust SE is exposed without an interval claim.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PanelDidSection {
    /// Difference in mean subject changes.
    pub effect: f64,
    /// Cluster-robust standard error.
    pub standard_error: f64,
    /// Treated subject count.
    pub treated_subjects: usize,
    /// Comparison subject count.
    pub comparison_subjects: usize,
    /// Number of distinct clusters.
    pub clusters: usize,
    /// Explicit uncertainty semantics.
    pub uncertainty: String,
}

impl From<&antecedent::PanelDidEstimate> for PanelDidSection {
    fn from(value: &antecedent::PanelDidEstimate) -> Self {
        Self {
            effect: value.effect,
            standard_error: value.standard_error,
            treated_subjects: value.treated_subjects,
            comparison_subjects: value.comparison_subjects,
            clusters: value.clusters,
            uncertainty: value.uncertainty.to_string(),
        }
    }
}

/// Repeated-cross-section 2x2 difference in differences, without interval claims.
#[pyfunction]
fn difference_in_differences(
    outcome: PyReadonlyArray1<'_, f64>,
    treated: Vec<bool>,
    post: Vec<bool>,
) -> PyResult<f64> {
    let values = outcome.as_array();
    let n = values.len();
    if n == 0 || treated.len() != n || post.len() != n {
        return Err(PyValueError::new_err(
            "outcome, treated, and post must have equal non-zero length",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    let mut sum = [0.0; 4];
    let mut count = [0usize; 4];
    for i in 0..n {
        // control-pre, control-post, treated-pre, treated-post
        let cell = usize::from(treated[i]) * 2 + usize::from(post[i]);
        sum[cell] += values[i];
        count[cell] += 1;
    }
    if count.contains(&0) {
        return Err(PyValueError::new_err(
            "2x2 difference in differences requires observations in all four treatment-by-time cells",
        ));
    }
    let mean = |cell: usize| sum[cell] / count[cell] as f64;
    Ok((mean(3) - mean(2)) - (mean(1) - mean(0)))
}

/// Two-period panel DiD from subject-level outcome changes; no interval claim.
#[pyfunction]
fn panel_difference_in_differences(
    outcome: PyReadonlyArray1<'_, f64>,
    subjects: Vec<String>,
    treated: Vec<bool>,
    post: Vec<bool>,
    clusters: Vec<String>,
) -> PyResult<(f64, f64, usize, usize, usize)> {
    use std::collections::BTreeMap;

    let values = outcome.as_array();
    let n = values.len();
    if n == 0 || subjects.len() != n || treated.len() != n || post.len() != n || clusters.len() != n
    {
        return Err(PyValueError::new_err(
            "outcome, subject, treated, and post vectors must have equal non-zero length",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    let mut rows: BTreeMap<&str, (Option<f64>, Option<f64>, Option<bool>, Option<&str>)> =
        BTreeMap::new();
    for i in 0..n {
        let id = subjects[i].as_str();
        if id.is_empty() {
            return Err(PyValueError::new_err("subject IDs must be non-empty strings"));
        }
        let entry = rows.entry(id).or_insert((None, None, None, None));
        if entry.2.is_some_and(|group| group != treated[i]) {
            return Err(PyValueError::new_err(
                "treatment assignment must be stable within each subject",
            ));
        }
        entry.2 = Some(treated[i]);
        if clusters[i].is_empty() {
            return Err(PyValueError::new_err("cluster IDs must be non-empty strings"));
        }
        if entry.3.is_some_and(|cluster| cluster != clusters[i]) {
            return Err(PyValueError::new_err("cluster ID must be constant within each subject"));
        }
        entry.3 = Some(clusters[i].as_str());
        let slot = if post[i] { &mut entry.1 } else { &mut entry.0 };
        if slot.replace(values[i]).is_some() {
            return Err(PyValueError::new_err(
                "each subject must have exactly one pre and one post observation",
            ));
        }
    }
    let mut change_sum = [0.0; 2];
    let mut subject_count = [0usize; 2];
    let mut changes = Vec::new();
    for (pre, after, group, cluster) in rows.values() {
        let (Some(pre), Some(after), Some(group), Some(cluster)) = (pre, after, group, cluster)
        else {
            return Err(PyValueError::new_err(
                "each subject must have exactly one pre and one post observation",
            ));
        };
        let cell = usize::from(*group);
        change_sum[cell] += after - pre;
        subject_count[cell] += 1;
        changes.push((*group, *cluster, after - pre));
    }
    if subject_count.contains(&0) {
        return Err(PyValueError::new_err(
            "panel DiD requires at least one treated and one control subject",
        ));
    }
    let mean_change = |group: usize| change_sum[group] / subject_count[group] as f64;
    let estimate = mean_change(1) - mean_change(0);
    let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
    let mut group_clusters: [std::collections::BTreeSet<&str>; 2] = Default::default();
    for (group, cluster, change) in &changes {
        let cell = usize::from(*group);
        group_clusters[cell].insert(cluster);
        *scores.entry(cluster).or_default() += (if *group { 1.0 } else { -1.0 })
            * (*change - mean_change(cell))
            / subject_count[cell] as f64;
    }
    if group_clusters.iter().any(|set| set.len() < 2) {
        return Err(PyValueError::new_err(
            "cluster-robust standard error requires at least two clusters in each treatment group",
        ));
    }
    let g = scores.len();
    if g < 2 {
        return Err(PyValueError::new_err(
            "cluster-robust standard error requires at least two clusters",
        ));
    }
    let variance =
        (g as f64 / (g - 1) as f64) * scores.values().map(|score| score * score).sum::<f64>();
    Ok((estimate, variance.max(0.0).sqrt(), subject_count[1], subject_count[0], g))
}

/// Staggered-adoption group-time ATT against never-treated units. Requires a
/// balanced panel; reports points and group sizes only.
#[pyfunction]
fn group_time_att(
    outcome: PyReadonlyArray1<'_, f64>,
    subjects: Vec<String>,
    periods: Vec<i64>,
    cohorts: Vec<i64>,
    clusters: Vec<String>,
) -> PyResult<Vec<(i64, i64, f64, usize, usize, f64, usize)>> {
    use std::collections::{BTreeMap, BTreeSet};

    let values = outcome.as_array();
    let n = values.len();
    if n == 0
        || subjects.len() != n
        || periods.len() != n
        || cohorts.len() != n
        || clusters.len() != n
    {
        return Err(PyValueError::new_err(
            "outcome, subject, period, and cohort vectors must have equal non-zero length",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }

    // Per unit: its adoption cohort and exactly one outcome at every common period.
    let mut units: BTreeMap<&str, (Option<i64>, BTreeMap<i64, f64>, Option<&str>)> =
        BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        if subjects[i].is_empty() {
            return Err(PyValueError::new_err("subject IDs must be non-empty strings"));
        }
        if periods[i] <= 0 || cohorts[i] < 0 {
            return Err(PyValueError::new_err(
                "periods must be positive integers and cohort 0 denotes never treated",
            ));
        }
        all_periods.insert(periods[i]);
        if clusters[i].is_empty() {
            return Err(PyValueError::new_err("cluster IDs must be non-empty strings"));
        }
        let unit = units.entry(subjects[i].as_str()).or_insert((None, BTreeMap::new(), None));
        if unit.0.is_some_and(|cohort| cohort != cohorts[i]) {
            return Err(PyValueError::new_err(
                "adoption cohort must be constant within each subject",
            ));
        }
        unit.0 = Some(cohorts[i]);
        if unit.2.is_some_and(|cluster| cluster != clusters[i]) {
            return Err(PyValueError::new_err("cluster ID must be constant within each subject"));
        }
        unit.2 = Some(clusters[i].as_str());
        if unit.1.insert(periods[i], values[i]).is_some() {
            return Err(PyValueError::new_err(
                "each subject must have exactly one observation per period",
            ));
        }
    }
    let period_list: Vec<i64> = all_periods.into_iter().collect();
    if period_list.len() < 2 {
        return Err(PyValueError::new_err(
            "staggered group-time ATT requires at least two periods",
        ));
    }
    for (_, outcomes, cluster) in units.values() {
        if outcomes.len() != period_list.len()
            || period_list.iter().any(|period| !outcomes.contains_key(period))
            || cluster.is_none()
        {
            return Err(PyValueError::new_err(
                "staggered group-time ATT requires a balanced panel with the same periods for every subject",
            ));
        }
    }

    let cohorts_present: BTreeSet<i64> = units.values().filter_map(|unit| unit.0).collect();
    if !cohorts_present.contains(&0) {
        return Err(PyValueError::new_err(
            "group-time ATT requires never-treated controls (cohort 0)",
        ));
    }
    let mut output = Vec::new();
    for cohort in cohorts_present.iter().copied().filter(|cohort| *cohort > 0) {
        let baseline = cohort - 1;
        if !period_list.contains(&baseline) || !period_list.contains(&cohort) {
            return Err(PyValueError::new_err(
                "each adoption cohort needs an observed immediately pre-treatment baseline and adoption period",
            ));
        }
        let treated: Vec<_> = units.values().filter(|unit| unit.0 == Some(cohort)).collect();
        let controls: Vec<_> = units.values().filter(|unit| unit.0 == Some(0)).collect();
        for period in period_list.iter().copied().filter(|period| *period >= cohort) {
            let treated_change: f64 =
                treated.iter().map(|unit| unit.1[&period] - unit.1[&baseline]).sum::<f64>()
                    / treated.len() as f64;
            let control_change: f64 =
                controls.iter().map(|unit| unit.1[&period] - unit.1[&baseline]).sum::<f64>()
                    / controls.len() as f64;
            let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
            let mut group_clusters: [BTreeSet<&str>; 2] = Default::default();
            for (group, selected, mean, denominator) in [
                (1usize, &treated, treated_change, treated.len()),
                (0usize, &controls, control_change, controls.len()),
            ] {
                for unit in selected {
                    let change = unit.1[&period] - unit.1[&baseline];
                    let cluster = unit.2.expect("cluster validated above");
                    group_clusters[group].insert(cluster);
                    *scores.entry(cluster).or_default() += (if group == 1 { 1.0 } else { -1.0 })
                        * (change - mean)
                        / denominator as f64;
                }
            }
            if group_clusters.iter().any(|set| set.len() < 2) {
                return Err(PyValueError::new_err(
                    "cluster-robust standard error requires at least two clusters in each cohort/control group",
                ));
            }
            let cluster_count = scores.len();
            let variance = cluster_count as f64 / (cluster_count - 1) as f64
                * scores.values().map(|score| score * score).sum::<f64>();
            output.push((
                cohort,
                period,
                treated_change - control_change,
                treated.len(),
                controls.len(),
                variance.max(0.0).sqrt(),
                cluster_count,
            ));
        }
    }
    if output.is_empty() {
        return Err(PyValueError::new_err(
            "no supported post-adoption group-time comparisons are available",
        ));
    }
    Ok(output)
}

/// Cohort-specific event-time contrasts against never-treated units. Point-only.
#[pyfunction]
fn staggered_event_study(
    outcome: PyReadonlyArray1<'_, f64>,
    subjects: Vec<String>,
    periods: Vec<i64>,
    cohorts: Vec<i64>,
    clusters: Vec<String>,
) -> PyResult<Vec<(i64, i64, i64, f64, usize, usize, f64, usize)>> {
    use std::collections::{BTreeMap, BTreeSet};
    let values = outcome.as_array();
    let n = values.len();
    if n == 0
        || subjects.len() != n
        || periods.len() != n
        || cohorts.len() != n
        || clusters.len() != n
    {
        return Err(PyValueError::new_err(
            "outcome, subject, period, and cohort vectors must have equal non-zero length",
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    let mut units: BTreeMap<&str, (Option<i64>, BTreeMap<i64, f64>, Option<&str>)> =
        BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        if subjects[i].is_empty() {
            return Err(PyValueError::new_err("subject IDs must be non-empty strings"));
        }
        if periods[i] <= 0 || cohorts[i] < 0 {
            return Err(PyValueError::new_err(
                "periods must be positive integers and cohort 0 denotes never treated",
            ));
        }
        all_periods.insert(periods[i]);
        if clusters[i].is_empty() {
            return Err(PyValueError::new_err("cluster IDs must be non-empty strings"));
        }
        let unit = units.entry(subjects[i].as_str()).or_insert((None, BTreeMap::new(), None));
        if unit.0.is_some_and(|g| g != cohorts[i]) {
            return Err(PyValueError::new_err(
                "adoption cohort must be constant within each subject",
            ));
        }
        unit.0 = Some(cohorts[i]);
        if unit.2.is_some_and(|cluster| cluster != clusters[i]) {
            return Err(PyValueError::new_err("cluster ID must be constant within each subject"));
        }
        unit.2 = Some(clusters[i].as_str());
        if unit.1.insert(periods[i], values[i]).is_some() {
            return Err(PyValueError::new_err(
                "each subject must have exactly one observation per period",
            ));
        }
    }
    let ps: Vec<i64> = all_periods.into_iter().collect();
    if ps.len() < 2 {
        return Err(PyValueError::new_err("event study requires at least two periods"));
    }
    if units
        .values()
        .any(|u| u.1.len() != ps.len() || ps.iter().any(|p| !u.1.contains_key(p)) || u.2.is_none())
    {
        return Err(PyValueError::new_err(
            "event study requires a balanced panel with the same periods for every subject",
        ));
    }
    let gs: BTreeSet<i64> = units.values().filter_map(|u| u.0).collect();
    if !gs.contains(&0) {
        return Err(PyValueError::new_err(
            "event study requires never-treated controls (cohort 0)",
        ));
    }
    let controls: Vec<_> = units.values().filter(|u| u.0 == Some(0)).collect();
    let mut output = Vec::new();
    for g in gs.into_iter().filter(|g| *g > 0) {
        let baseline = g - 1;
        if !ps.contains(&baseline) || !ps.contains(&g) {
            return Err(PyValueError::new_err(
                "each adoption cohort needs an observed immediately pre-treatment baseline and adoption period",
            ));
        }
        let treated: Vec<_> = units.values().filter(|u| u.0 == Some(g)).collect();
        for t in ps.iter().copied().filter(|t| *t != baseline) {
            let dt: f64 = treated.iter().map(|u| u.1[&t] - u.1[&baseline]).sum::<f64>()
                / treated.len() as f64;
            let dc: f64 = controls.iter().map(|u| u.1[&t] - u.1[&baseline]).sum::<f64>()
                / controls.len() as f64;
            let estimate = dt - dc;
            let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
            let mut group_clusters: [BTreeSet<&str>; 2] = Default::default();
            for (group, cohort_units, mean, denominator) in
                [(1usize, &treated, dt, treated.len()), (0usize, &controls, dc, controls.len())]
            {
                for unit in cohort_units {
                    let delta = unit.1[&t] - unit.1[&baseline];
                    let cluster = unit.2.expect("cluster validated above");
                    group_clusters[group].insert(cluster);
                    *scores.entry(cluster).or_default() +=
                        (if group == 1 { 1.0 } else { -1.0 }) * (delta - mean) / denominator as f64;
                }
            }
            if group_clusters.iter().any(|set| set.len() < 2) {
                return Err(PyValueError::new_err(
                    "cluster-robust standard error requires at least two clusters in each cohort/control group",
                ));
            }
            let cluster_count = scores.len();
            let variance = cluster_count as f64 / (cluster_count - 1) as f64
                * scores.values().map(|score| score * score).sum::<f64>();
            output.push((
                g,
                t,
                t - g,
                estimate,
                treated.len(),
                controls.len(),
                variance.max(0.0).sqrt(),
                cluster_count,
            ));
        }
    }
    if output.is_empty() {
        return Err(PyValueError::new_err("no supported event-time comparisons are available"));
    }
    Ok(output)
}

fn project_simplex(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += value;
        let candidate = (cumulative - 1.0) / (index + 1) as f64;
        if index + 1 == sorted.len() || sorted[index + 1] <= candidate {
            theta = candidate;
            break;
        }
    }
    values.iter().map(|value| (*value - theta).max(0.0)).collect()
}

fn fit_synthetic_weights(target: &[f64], donors: &[Vec<f64>]) -> (Vec<f64>, f64) {
    let n_donors = donors.len();
    let n_pre = target.len();
    let frobenius_sq: f64 = donors.iter().flatten().map(|value| *value * *value).sum();
    let lipschitz = 2.0 * frobenius_sq / n_pre as f64;
    let step = if lipschitz > 1e-15 { 1.0 / lipschitz } else { 1.0 };
    let mut weights = vec![1.0 / n_donors as f64; n_donors];
    for _ in 0..10_000 {
        let mut gradient = vec![0.0; n_donors];
        for period in 0..n_pre {
            let residual: f64 = donors
                .iter()
                .zip(&weights)
                .map(|(donor, weight)| donor[period] * *weight)
                .sum::<f64>()
                - target[period];
            for (index, donor) in donors.iter().enumerate() {
                gradient[index] += 2.0 * donor[period] * residual / n_pre as f64;
            }
        }
        let next = project_simplex(
            &weights
                .iter()
                .zip(gradient)
                .map(|(weight, gradient)| *weight - step * gradient)
                .collect::<Vec<_>>(),
        );
        let movement = next
            .iter()
            .zip(&weights)
            .map(|(next, current)| (*next - *current).abs())
            .fold(0.0_f64, f64::max);
        weights = next;
        if movement < 1e-12 {
            break;
        }
    }
    let rmse = ((0..n_pre)
        .map(|period| {
            let residual = donors
                .iter()
                .zip(&weights)
                .map(|(donor, weight)| donor[period] * *weight)
                .sum::<f64>()
                - target[period];
            residual * residual
        })
        .sum::<f64>()
        / n_pre as f64)
        .sqrt();
    (weights, rmse)
}

/// Balanced-panel synthetic control using the shared native estimator.
#[pyfunction]
fn synthetic_control(
    outcome: PyReadonlyArray1<'_, f64>,
    units: Vec<String>,
    periods: Vec<i64>,
    treated_unit: String,
    intervention_period: i64,
) -> PyResult<(f64, f64, Vec<(String, f64)>, Vec<f64>, f64, usize, usize)> {
    let values: Vec<f64> = outcome.as_array().iter().copied().collect();
    let fit = antecedent_estimate::synthetic_control::fit_synthetic_control(
        &values, &units, &periods, &treated_unit, intervention_period,
    ).map_err(PyValueError::new_err)?;
    Ok((fit.effect, fit.pre_treatment_rmse, fit.donor_weights,
        fit.placebo_effects, fit.placebo_rank, fit.n_pre_periods, fit.n_post_periods))
}

/// Synthetic difference-in-differences with simplex unit and pre-period weights.
/// Point estimate only; no interval or inference claim is made.
#[pyfunction]
fn synthetic_difference_in_differences(
    outcome: PyReadonlyArray1<'_, f64>,
    units: Vec<String>,
    periods: Vec<i64>,
    treated_unit: String,
    intervention_period: i64,
) -> PyResult<(f64, f64, Vec<(String, f64)>, Vec<(i64, f64)>, usize, usize, usize)> {
    use std::collections::{BTreeMap, BTreeSet};

    let values = outcome.as_array();
    let n = values.len();
    if n == 0 || units.len() != n || periods.len() != n {
        return Err(PyValueError::new_err(
            "outcome, unit, and period vectors must have equal non-zero length",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    if treated_unit.is_empty() || intervention_period <= 0 {
        return Err(PyValueError::new_err("treated unit and intervention period are required"));
    }
    let mut panel: BTreeMap<String, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        if units[i].is_empty() || periods[i] <= 0 {
            return Err(PyValueError::new_err("unit IDs must be non-empty and periods positive"));
        }
        all_periods.insert(periods[i]);
        if panel.entry(units[i].clone()).or_default().insert(periods[i], values[i]).is_some() {
            return Err(PyValueError::new_err(
                "each unit must have exactly one outcome per period",
            ));
        }
    }
    let period_list: Vec<i64> = all_periods.into_iter().collect();
    let pre: Vec<i64> =
        period_list.iter().copied().filter(|period| *period < intervention_period).collect();
    let post: Vec<i64> =
        period_list.iter().copied().filter(|period| *period >= intervention_period).collect();
    if pre.len() < 2 || post.is_empty() {
        return Err(PyValueError::new_err(
            "synthetic DiD requires at least two pre-periods and one post-period",
        ));
    }
    if panel.len() < 3 || !panel.contains_key(&treated_unit) {
        return Err(PyValueError::new_err(
            "synthetic DiD requires one treated unit and at least two donor units",
        ));
    }
    let n_pre = pre.len();
    let n_post = post.len();
    if panel.values().any(|observed| {
        observed.len() != period_list.len()
            || period_list.iter().any(|period| !observed.contains_key(period))
    }) {
        return Err(PyValueError::new_err(
            "synthetic DiD requires a balanced panel with identical periods for every unit",
        ));
    }
    let treated = &panel[&treated_unit];
    let donors: Vec<(&String, &BTreeMap<i64, f64>)> =
        panel.iter().filter(|(unit, _)| unit.as_str() != treated_unit).collect();
    let target_pre: Vec<f64> = pre.iter().map(|period| treated[period]).collect();
    let donor_pre: Vec<Vec<f64>> = donors
        .iter()
        .map(|(_, observed)| pre.iter().map(|period| observed[period]).collect())
        .collect();
    let (unit_weights, pre_rmse) = fit_synthetic_weights(&target_pre, &donor_pre);

    // SDID time weights solve the dual simplex problem: represent the average
    // pre-period donor profile as a convex combination of individual periods.
    let target_donor_mean: Vec<f64> = donors
        .iter()
        .map(|(_, observed)| {
            pre.iter().map(|period| observed[period]).sum::<f64>() / pre.len() as f64
        })
        .collect();
    let period_profiles: Vec<Vec<f64>> = pre
        .iter()
        .map(|period| donors.iter().map(|(_, observed)| observed[period]).collect())
        .collect();
    let (time_weights, _) = fit_synthetic_weights(&target_donor_mean, &period_profiles);

    let treated_post_mean =
        post.iter().map(|period| treated[period]).sum::<f64>() / post.len() as f64;
    let treated_pre_weighted: f64 =
        pre.iter().zip(&time_weights).map(|(period, weight)| treated[period] * weight).sum();
    let donor_post_mean: f64 = donors
        .iter()
        .zip(&unit_weights)
        .map(|((_, observed), weight)| {
            weight * post.iter().map(|period| observed[period]).sum::<f64>() / post.len() as f64
        })
        .sum();
    let donor_joint_pre: f64 = donors
        .iter()
        .zip(&unit_weights)
        .map(|((_, observed), unit_weight)| {
            pre.iter()
                .zip(&time_weights)
                .map(|(period, time_weight)| unit_weight * time_weight * observed[period])
                .sum::<f64>()
        })
        .sum();
    let estimate = (treated_post_mean - treated_pre_weighted) - (donor_post_mean - donor_joint_pre);
    if !estimate.is_finite() || !pre_rmse.is_finite() {
        return Err(PyValueError::new_err("synthetic DiD estimate overflowed finite precision"));
    }
    let weights = donors
        .iter()
        .zip(unit_weights)
        .map(|((unit, _), weight)| ((*unit).clone(), weight))
        .collect();
    let time_weights = pre.into_iter().zip(time_weights).collect();
    Ok((estimate, pre_rmse, weights, time_weights, donors.len(), n_pre, n_post))
}

/// Augmented panel DiD from caller-supplied untreated-change predictions and
/// propensity scores. The caller is responsible for out-of-fold nuisance fits.
#[pyfunction]
fn augmented_panel_difference_in_differences(
    outcome_pre: PyReadonlyArray1<'_, f64>,
    outcome_post: PyReadonlyArray1<'_, f64>,
    treated: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
    untreated_change_prediction: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, usize, usize, f64, f64, f64)> {
    let pre = outcome_pre.as_array();
    let post = outcome_post.as_array();
    let propensity = propensity.as_array();
    let prediction = untreated_change_prediction.as_array();
    let n = pre.len();
    if n == 0
        || post.len() != n
        || treated.len() != n
        || propensity.len() != n
        || prediction.len() != n
    {
        return Err(PyValueError::new_err(
            "pre/post outcomes, treatment, propensity, and outcome predictions must have equal non-zero length",
        ));
    }
    if pre.iter().chain(post.iter()).chain(prediction.iter()).any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes and nuisance predictions must be finite"));
    }
    if propensity.iter().any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0) {
        return Err(PyValueError::new_err(
            "augmented DiD overlap failure: propensity scores must be strictly between zero and one",
        ));
    }
    let treated_count = treated.iter().filter(|&&value| value).count();
    let control_count = n - treated_count;
    if treated_count == 0 || control_count == 0 {
        return Err(PyValueError::new_err("augmented DiD requires treated and control subjects"));
    }
    let mut treated_change = 0.0;
    let mut predicted_counterfactual = 0.0;
    let mut control_residual_correction = 0.0;
    let mut control_weight = 0.0;
    let mut propensity_min = f64::INFINITY;
    let mut propensity_max: f64 = 0.0;
    let mut squared_weights = 0.0;
    for i in 0..n {
        let p = propensity[i];
        propensity_min = propensity_min.min(p);
        propensity_max = propensity_max.max(p);
        let change = post[i] - pre[i];
        if treated[i] {
            treated_change += change;
            predicted_counterfactual += prediction[i];
        } else {
            let weight = p / (1.0 - p);
            control_weight += weight;
            squared_weights += weight * weight;
            control_residual_correction += weight * (change - prediction[i]);
        }
    }
    treated_change /= treated_count as f64;
    predicted_counterfactual /= treated_count as f64;
    let counterfactual_change =
        predicted_counterfactual + control_residual_correction / treated_count as f64;
    let effect = treated_change - counterfactual_change;
    let effective_controls =
        if squared_weights > 0.0 { control_weight.powi(2) / squared_weights } else { 0.0 };
    if !effect.is_finite() || !effective_controls.is_finite() {
        return Err(PyValueError::new_err("augmented DiD estimate overflowed finite precision"));
    }
    Ok((effect, treated_count, control_count, propensity_min, propensity_max, effective_controls))
}

fn invert_three(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut augmented = [[0.0; 6]; 3];
    for row in 0..3 {
        augmented[row][..3].copy_from_slice(&matrix[row]);
        augmented[row][row + 3] = 1.0;
    }
    for column in 0..3 {
        let pivot = (column..3).max_by(|left, right| {
            augmented[*left][column].abs().total_cmp(&augmented[*right][column].abs())
        })?;
        if augmented[pivot][column].abs() < 1e-12 {
            return None;
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for value in &mut augmented[column] {
            *value /= scale;
        }
        for row in 0..3 {
            if row == column {
                continue;
            }
            let scale = augmented[row][column];
            for index in 0..6 {
                augmented[row][index] -= scale * augmented[column][index];
            }
        }
    }
    let mut inverse = [[0.0; 3]; 3];
    for row in 0..3 {
        inverse[row].copy_from_slice(&augmented[row][3..]);
    }
    Some(inverse)
}

fn local_quadratic_side(
    running: &[f64],
    outcome: &[f64],
    treatment: &[f64],
    cutoff: f64,
    bandwidth: f64,
    right_side: bool,
) -> Option<([f64; 3], [f64; 3], usize)> {
    let mut gram = [[0.0; 3]; 3];
    let mut rhs_y = [0.0; 3];
    let mut rhs_t = [0.0; 3];
    let mut count = 0usize;
    for index in 0..running.len() {
        let distance = (running[index] - cutoff) / bandwidth;
        if distance == 0.0
            || distance.abs() >= 1.0
            || (right_side && distance < 0.0)
            || (!right_side && distance > 0.0)
        {
            continue;
        }
        let basis = [1.0, distance, distance * distance];
        let weight = 1.0 - distance.abs();
        count += 1;
        for row in 0..3 {
            rhs_y[row] += weight * basis[row] * outcome[index];
            rhs_t[row] += weight * basis[row] * treatment[index];
            for column in 0..3 {
                gram[row][column] += weight * basis[row] * basis[column];
            }
        }
    }
    let inverse = invert_three(gram)?;
    let mut beta_y = [0.0; 3];
    let mut beta_t = [0.0; 3];
    for row in 0..3 {
        for column in 0..3 {
            beta_y[row] += inverse[row][column] * rhs_y[column];
            beta_t[row] += inverse[row][column] * rhs_t[column];
        }
    }
    Some((beta_y, beta_t, count))
}

#[derive(Clone, Copy)]
struct LocalCubicFit {
    beta_y: [f64; 4],
    beta_t: [f64; 4],
    var_y: [[f64; 4]; 4],
    var_t: [[f64; 4]; 4],
    cov_yt: [[f64; 4]; 4],
    count: usize,
}

fn invert_four(matrix: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut augmented = [[0.0; 8]; 4];
    for row in 0..4 {
        augmented[row][..4].copy_from_slice(&matrix[row]);
        augmented[row][row + 4] = 1.0;
    }
    for column in 0..4 {
        let pivot = (column..4).max_by(|left, right| {
            augmented[*left][column].abs().total_cmp(&augmented[*right][column].abs())
        })?;
        if augmented[pivot][column].abs() < 1e-12 {
            return None;
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for value in &mut augmented[column] {
            *value /= scale;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let scale = augmented[row][column];
            for index in 0..8 {
                augmented[row][index] -= scale * augmented[column][index];
            }
        }
    }
    let mut inverse = [[0.0; 4]; 4];
    for row in 0..4 {
        inverse[row].copy_from_slice(&augmented[row][4..]);
    }
    Some(inverse)
}

fn sandwich_four(inverse: [[f64; 4]; 4], meat: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut covariance = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            for left in 0..4 {
                for right in 0..4 {
                    covariance[row][column] +=
                        inverse[row][left] * meat[left][right] * inverse[right][column];
                }
            }
        }
    }
    covariance
}

fn local_cubic_side(
    running: &[f64],
    outcome: &[f64],
    treatment: &[f64],
    cutoff: f64,
    bandwidth: f64,
    right_side: bool,
) -> Option<LocalCubicFit> {
    let mut gram = [[0.0; 4]; 4];
    let mut rhs_y = [0.0; 4];
    let mut rhs_t = [0.0; 4];
    let mut rows = Vec::new();
    for index in 0..running.len() {
        let distance = (running[index] - cutoff) / bandwidth;
        if distance == 0.0
            || distance.abs() >= 1.0
            || (right_side && distance < 0.0)
            || (!right_side && distance > 0.0)
        {
            continue;
        }
        let basis = [1.0, distance, distance.powi(2), distance.powi(3)];
        let weight = 1.0 - distance.abs();
        rows.push((basis, weight, outcome[index], treatment[index]));
        for row in 0..4 {
            rhs_y[row] += weight * basis[row] * outcome[index];
            rhs_t[row] += weight * basis[row] * treatment[index];
            for column in 0..4 {
                gram[row][column] += weight * basis[row] * basis[column];
            }
        }
    }
    let inverse = invert_four(gram)?;
    let mut beta_y = [0.0; 4];
    let mut beta_t = [0.0; 4];
    for row in 0..4 {
        for column in 0..4 {
            beta_y[row] += inverse[row][column] * rhs_y[column];
            beta_t[row] += inverse[row][column] * rhs_t[column];
        }
    }
    let mut meat_y = [[0.0; 4]; 4];
    let mut meat_t = [[0.0; 4]; 4];
    let mut meat_yt = [[0.0; 4]; 4];
    for (basis, weight, value_y, value_t) in &rows {
        let residual_y = *value_y - (0..4).map(|i| beta_y[i] * basis[i]).sum::<f64>();
        let residual_t = *value_t - (0..4).map(|i| beta_t[i] * basis[i]).sum::<f64>();
        for row in 0..4 {
            for column in 0..4 {
                let scale = *weight * *weight * basis[row] * basis[column];
                meat_y[row][column] += scale * residual_y * residual_y;
                meat_t[row][column] += scale * residual_t * residual_t;
                meat_yt[row][column] += scale * residual_y * residual_t;
            }
        }
    }
    Some(LocalCubicFit {
        beta_y,
        beta_t,
        var_y: sandwich_four(inverse, meat_y),
        var_t: sandwich_four(inverse, meat_t),
        cov_yt: sandwich_four(inverse, meat_yt),
        count: rows.len(),
    })
}

fn cubic_bias_projection(
    running: &[f64],
    cutoff: f64,
    bandwidth: f64,
    right_side: bool,
) -> Option<[f64; 3]> {
    let mut gram = [[0.0; 3]; 3];
    let mut cross = [0.0; 3];
    for value in running {
        let distance = (*value - cutoff) / bandwidth;
        if distance == 0.0
            || distance.abs() >= 1.0
            || (right_side && distance < 0.0)
            || (!right_side && distance > 0.0)
        {
            continue;
        }
        let basis = [1.0, distance, distance * distance];
        let weight = 1.0 - distance.abs();
        for row in 0..3 {
            cross[row] += weight * basis[row] * distance.powi(3);
            for column in 0..3 {
                gram[row][column] += weight * basis[row] * basis[column];
            }
        }
    }
    let inverse = invert_three(gram)?;
    let mut projection = [0.0; 3];
    for row in 0..3 {
        for column in 0..3 {
            projection[row] += inverse[row][column] * cross[column];
        }
    }
    Some(projection)
}

/// Local quadratic fuzzy RD or kink with cubic-pilot bias correction and HC0 inference.
#[pyfunction]
fn local_polynomial_fuzzy_discontinuity(
    running: PyReadonlyArray1<'_, f64>,
    outcome: PyReadonlyArray1<'_, f64>,
    treatment: PyReadonlyArray1<'_, f64>,
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
) -> PyResult<(f64, f64, f64, usize, usize, f64, f64, f64, f64, f64)> {
    let running = running.as_array();
    let outcome = outcome.as_array();
    let treatment = treatment.as_array();
    let n = running.len();
    if n == 0 || outcome.len() != n || treatment.len() != n {
        return Err(PyValueError::new_err(
            "running, outcome, and treatment vectors must have equal non-zero length",
        ));
    }
    if !cutoff.is_finite() || !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(PyValueError::new_err("cutoff must be finite and bandwidth positive"));
    }
    if running.iter().chain(outcome.iter()).chain(treatment.iter()).any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("running, outcome, and treatment values must be finite"));
    }
    let running: Vec<f64> = running.iter().copied().collect();
    let outcome: Vec<f64> = outcome.iter().copied().collect();
    let treatment: Vec<f64> = treatment.iter().copied().collect();
    let (left_y_p, left_t_p, n_left) =
        local_quadratic_side(&running, &outcome, &treatment, cutoff, bandwidth, false)
            .filter(|(_, _, count)| *count >= 4)
            .ok_or_else(|| {
                PyValueError::new_err("local polynomial lacks full-rank support on the left side")
            })?;
    let (right_y_p, right_t_p, n_right) =
        local_quadratic_side(&running, &outcome, &treatment, cutoff, bandwidth, true)
            .filter(|(_, _, count)| *count >= 4)
            .ok_or_else(|| {
                PyValueError::new_err("local polynomial lacks full-rank support on the right side")
            })?;
    let left_q = local_cubic_side(&running, &outcome, &treatment, cutoff, bandwidth, false)
        .filter(|fit| fit.count >= 5)
        .ok_or_else(|| {
            PyValueError::new_err("robust bias correction lacks cubic support on the left side")
        })?;
    let right_q = local_cubic_side(&running, &outcome, &treatment, cutoff, bandwidth, true)
        .filter(|fit| fit.count >= 5)
        .ok_or_else(|| {
            PyValueError::new_err("robust bias correction lacks cubic support on the right side")
        })?;
    let left_projection = cubic_bias_projection(&running, cutoff, bandwidth, false)
        .ok_or_else(|| PyValueError::new_err("bias correction lacks left-side support"))?;
    let right_projection = cubic_bias_projection(&running, cutoff, bandwidth, true)
        .ok_or_else(|| PyValueError::new_err("bias correction lacks right-side support"))?;
    let coefficient = if kink { 1 } else { 0 };
    let scale = if kink { 1.0 / bandwidth } else { 1.0 };
    // The p=2 local-polynomial coefficient's leading omitted-cubic bias is
    // projection[j] * beta_3. Subtracting it gives the RBC coefficient; the
    // same identity makes the q=3 HC0 covariance the corrected-coefficient
    // covariance (including the covariance of the estimated bias term).
    let corrected_y_left = left_y_p[coefficient] - left_projection[coefficient] * left_q.beta_y[3];
    let corrected_y_right =
        right_y_p[coefficient] - right_projection[coefficient] * right_q.beta_y[3];
    let corrected_t_left = left_t_p[coefficient] - left_projection[coefficient] * left_q.beta_t[3];
    let corrected_t_right =
        right_t_p[coefficient] - right_projection[coefficient] * right_q.beta_t[3];
    let reduced_form = (corrected_y_right - corrected_y_left) * scale;
    let first_stage = (corrected_t_right - corrected_t_left) * scale;
    if first_stage.abs() < 1e-10 {
        return Err(PyValueError::new_err(
            "local treatment discontinuity is too small to form a fuzzy design estimate",
        ));
    }
    let variance_y = (left_q.var_y[coefficient][coefficient]
        + right_q.var_y[coefficient][coefficient])
        * scale
        * scale;
    let variance_t = (left_q.var_t[coefficient][coefficient]
        + right_q.var_t[coefficient][coefficient])
        * scale
        * scale;
    let covariance_yt = (left_q.cov_yt[coefficient][coefficient]
        + right_q.cov_yt[coefficient][coefficient])
        * scale
        * scale;
    let se_first = variance_t.max(0.0).sqrt();
    if first_stage.abs() <= 1.96 * se_first {
        return Err(PyValueError::new_err(
            "weak first stage: local treatment change is not separated from zero by its HC0 95% interval",
        ));
    }
    let estimate = reduced_form / first_stage;
    let variance_estimate = variance_y / first_stage.powi(2)
        + reduced_form.powi(2) * variance_t / first_stage.powi(4)
        - 2.0 * reduced_form * covariance_yt / first_stage.powi(3);
    if variance_estimate < -1e-10 || !variance_estimate.is_finite() {
        return Err(PyValueError::new_err("robust ratio variance is not finite and non-negative"));
    }
    let standard_error = variance_estimate.max(0.0).sqrt();
    let critical = 1.959_963_984_540_054;
    Ok((
        estimate,
        reduced_form,
        first_stage,
        n_left,
        n_right,
        standard_error,
        estimate - critical * standard_error,
        estimate + critical * standard_error,
        variance_y.max(0.0).sqrt(),
        se_first,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PanelDidSection>()?;
    module.add_function(wrap_pyfunction!(difference_in_differences, module)?)?;
    module.add_function(wrap_pyfunction!(panel_difference_in_differences, module)?)?;
    module.add_function(wrap_pyfunction!(group_time_att, module)?)?;
    module.add_function(wrap_pyfunction!(staggered_event_study, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_control, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_difference_in_differences, module)?)?;
    module.add_function(wrap_pyfunction!(augmented_panel_difference_in_differences, module)?)?;
    module.add_function(wrap_pyfunction!(local_polynomial_fuzzy_discontinuity, module)?)?;
    Ok(())
}
