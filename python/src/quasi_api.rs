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

/// Retained synthetic-control point result; placebo rank is descriptive only.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct SyntheticControlSection {
    pub effect: f64,
    pub pre_treatment_rmse: f64,
    pub donor_weights: Vec<(String, f64)>,
    pub placebo_effects: Vec<f64>,
    pub placebo_rank: f64,
    pub effective_donors: f64,
    pub n_pre_periods: usize,
    pub n_post_periods: usize,
    pub uncertainty: String,
}

impl From<&antecedent::SyntheticControlEstimate> for SyntheticControlSection {
    fn from(value: &antecedent::SyntheticControlEstimate) -> Self {
        Self {
            effect: value.effect,
            pre_treatment_rmse: value.pre_treatment_rmse,
            donor_weights: value.donor_weights.iter().map(|(unit, weight)| (unit.to_string(), *weight)).collect(),
            placebo_effects: value.placebo_effects.to_vec(),
            placebo_rank: value.placebo_rank,
            effective_donors: value.effective_donors,
            n_pre_periods: value.n_pre_periods,
            n_post_periods: value.n_post_periods,
            uncertainty: value.uncertainty.to_string(),
        }
    }
}

/// Retained point-only synthetic difference-in-differences result.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct SyntheticDidSection {
    pub effect: f64,
    pub pre_treatment_rmse: f64,
    pub donor_weights: Vec<(String, f64)>,
    pub time_weights: Vec<(i64, f64)>,
    pub n_donors: usize,
    pub n_pre_periods: usize,
    pub n_post_periods: usize,
    pub uncertainty: String,
}

impl From<&antecedent::SyntheticDidEstimate> for SyntheticDidSection {
    fn from(value: &antecedent::SyntheticDidEstimate) -> Self {
        Self {
            effect: value.effect,
            pre_treatment_rmse: value.pre_treatment_rmse,
            donor_weights: value.donor_weights.iter().map(|(unit, weight)| (unit.to_string(), *weight)).collect(),
            time_weights: value.time_weights.to_vec(),
            n_donors: value.n_donors,
            n_pre_periods: value.n_pre_periods,
            n_post_periods: value.n_post_periods,
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

/// Balanced-panel synthetic difference in differences using the shared kernel.
#[pyfunction]
fn synthetic_difference_in_differences(
    outcome: PyReadonlyArray1<'_, f64>,
    units: Vec<String>,
    periods: Vec<i64>,
    treated_unit: String,
    intervention_period: i64,
) -> PyResult<(f64, f64, Vec<(String, f64)>, Vec<(i64, f64)>, usize, usize, usize)> {
    let values: Vec<f64> = outcome.as_array().iter().copied().collect();
    let fit = antecedent_estimate::synthetic_control::fit_synthetic_did(
        &values, &units, &periods, &treated_unit, intervention_period,
    ).map_err(PyValueError::new_err)?;
    Ok((fit.effect, fit.pre_treatment_rmse, fit.donor_weights, fit.time_weights,
        fit.n_donors, fit.n_pre_periods, fit.n_post_periods))
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
    let running: Vec<f64> = running.as_array().iter().copied().collect();
    let outcome: Vec<f64> = outcome.as_array().iter().copied().collect();
    let treatment: Vec<f64> = treatment.as_array().iter().copied().collect();
    let fit = antecedent_estimate::local_polynomial_ratio::fit_local_polynomial_ratio(
        &running, &outcome, &treatment, cutoff, bandwidth, kink,
    ).map_err(PyValueError::new_err)?;
    Ok((fit.estimate, fit.reduced_form, fit.first_stage, fit.n_left, fit.n_right,
        fit.standard_error, fit.ci_lower, fit.ci_upper,
        fit.reduced_form_standard_error, fit.first_stage_standard_error))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PanelDidSection>()?;
    module.add_class::<SyntheticControlSection>()?;
    module.add_class::<SyntheticDidSection>()?;
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
