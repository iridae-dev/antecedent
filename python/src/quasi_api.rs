//! Small, native estimators for quasi-experimental designs.
// SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// One staggered event-study effect: cohort, period, event time, estimate,
/// treated and comparison counts, standard error, clusters.
type EventTimeEffect = (i64, i64, i64, f64, usize, usize, f64, usize);
/// One group-time ATT: cohort, period, estimate, treated and comparison
/// counts, standard error, clusters.
type GroupTimeAtt = (i64, i64, f64, usize, usize, f64, usize);
/// Per-unit staggered panel row: adoption cohort, outcome by period, cluster.
type UnitPanel<'a> = (Option<i64>, std::collections::BTreeMap<i64, f64>, Option<&'a str>);

/// Retained panel DiD result with support-gated pointwise uncertainty.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PanelDidSection {
    /// Difference in mean subject changes.
    pub effect: f64,
    /// Cluster-robust standard error.
    pub standard_error: Option<f64>,
    /// Supported pointwise 95% interval for a scalar two-period design.
    pub interval_95: Option<(f64, f64)>,
    /// Treated subject count.
    pub treated_subjects: usize,
    /// Comparison subject count.
    pub comparison_subjects: usize,
    /// Number of distinct clusters.
    pub clusters: usize,
    /// Explicit uncertainty semantics.
    pub uncertainty: String,
    /// Exact graphless two-period DiD support license, when matched.
    pub graphless_support_status: Option<String>,
    /// Cohort, period, event time, estimate, treated, controls, SE, clusters.
    pub event_time_effects: Vec<EventTimeEffect>,
    pub event_time_intervals_95: Vec<Option<(f64, f64)>>,
    /// Propensity range, effective control count, and caller cross-fit declaration.
    pub augmented: Option<(f64, f64, f64, bool)>,
}

impl From<&antecedent::PanelDidEstimate> for PanelDidSection {
    fn from(value: &antecedent::PanelDidEstimate) -> Self {
        Self {
            effect: value.effect,
            standard_error: if value.augmented.is_some() {
                None
            } else {
                Some(value.standard_error)
            },
            interval_95: value.interval_95.map(|bounds| (bounds[0], bounds[1])),
            treated_subjects: value.treated_subjects,
            comparison_subjects: value.comparison_subjects,
            clusters: value.clusters,
            uncertainty: value.uncertainty.to_string(),
            graphless_support_status: None,
            event_time_effects: value
                .event_time_effects
                .iter()
                .map(|effect| {
                    (
                        effect.cohort,
                        effect.period,
                        effect.event_time,
                        effect.effect,
                        effect.treated_subjects,
                        effect.comparison_subjects,
                        effect.standard_error,
                        effect.clusters,
                    )
                })
                .collect(),
            event_time_intervals_95: value
                .event_time_intervals_95
                .iter()
                .map(|interval| interval.map(|bounds| (bounds[0], bounds[1])))
                .collect(),
            augmented: value.augmented,
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
    pub randomization_p_value: Option<f64>,
    pub randomization_null_effect: Option<f64>,
    pub randomization_statistics: Vec<(String, f64)>,
    pub unadjusted_effect: Option<f64>,
    pub outcome_model_correction: Option<f64>,
    pub augmentation_ridge: Option<f64>,
}

impl From<&antecedent::SyntheticControlEstimate> for SyntheticControlSection {
    fn from(value: &antecedent::SyntheticControlEstimate) -> Self {
        Self {
            effect: value.effect,
            pre_treatment_rmse: value.pre_treatment_rmse,
            donor_weights: value
                .donor_weights
                .iter()
                .map(|(unit, weight)| (unit.to_string(), *weight))
                .collect(),
            placebo_effects: value.placebo_effects.to_vec(),
            placebo_rank: value.placebo_rank,
            effective_donors: value.effective_donors,
            n_pre_periods: value.n_pre_periods,
            n_post_periods: value.n_post_periods,
            uncertainty: value.uncertainty.to_string(),
            randomization_p_value: value.randomization_p_value,
            randomization_null_effect: value.randomization_null_effect,
            randomization_statistics: value
                .randomization_statistics
                .iter()
                .map(|(unit, statistic)| (unit.to_string(), *statistic))
                .collect(),
            unadjusted_effect: value.unadjusted_effect,
            outcome_model_correction: value.outcome_model_correction,
            augmentation_ridge: value.augmentation_ridge,
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
    pub randomization_p_value: Option<f64>,
    pub randomization_null_effect: Option<f64>,
    pub randomization_statistics: Vec<(String, f64)>,
}

impl From<&antecedent::SyntheticDidEstimate> for SyntheticDidSection {
    fn from(value: &antecedent::SyntheticDidEstimate) -> Self {
        Self {
            effect: value.effect,
            pre_treatment_rmse: value.pre_treatment_rmse,
            donor_weights: value
                .donor_weights
                .iter()
                .map(|(unit, weight)| (unit.to_string(), *weight))
                .collect(),
            time_weights: value.time_weights.to_vec(),
            n_donors: value.n_donors,
            n_pre_periods: value.n_pre_periods,
            n_post_periods: value.n_post_periods,
            uncertainty: value.uncertainty.to_string(),
            randomization_p_value: value.randomization_p_value,
            randomization_null_effect: value.randomization_null_effect,
            randomization_statistics: value
                .randomization_statistics
                .iter()
                .map(|(unit, statistic)| (unit.to_string(), *statistic))
                .collect(),
        }
    }
}

/// Retained local ratio result with a fixed-bandwidth normal interval.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct LocalPolynomialRatioSection {
    pub effect: f64,
    pub reduced_form: f64,
    pub first_stage: f64,
    pub cutoff: f64,
    pub bandwidth: f64,
    pub kink: bool,
    pub n_left: usize,
    pub n_right: usize,
    pub standard_error: f64,
    pub ci_lower: Option<f64>,
    pub ci_upper: Option<f64>,
    pub reduced_form_standard_error: f64,
    pub first_stage_standard_error: f64,
    pub uncertainty: String,
    pub graphless_support_status: Option<String>,
}

impl From<&antecedent::LocalPolynomialRatioEstimate> for LocalPolynomialRatioSection {
    fn from(value: &antecedent::LocalPolynomialRatioEstimate) -> Self {
        Self {
            effect: value.effect,
            reduced_form: value.reduced_form,
            first_stage: value.first_stage,
            cutoff: value.cutoff,
            bandwidth: value.bandwidth,
            kink: value.kink,
            n_left: value.n_left,
            n_right: value.n_right,
            standard_error: value.standard_error,
            ci_lower: value.ci_lower,
            ci_upper: value.ci_upper,
            reduced_form_standard_error: value.reduced_form_standard_error,
            first_stage_standard_error: value.first_stage_standard_error,
            uncertainty: value.uncertainty.to_string(),
            graphless_support_status: None,
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

/// Staggered-adoption group-time ATT against never-treated units. Requires a
/// balanced panel; reports points and group sizes only.
#[pyfunction]
fn group_time_att(
    outcome: PyReadonlyArray1<'_, f64>,
    subjects: Vec<String>,
    periods: Vec<i64>,
    cohorts: Vec<i64>,
    clusters: Vec<String>,
) -> PyResult<Vec<GroupTimeAtt>> {
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
    let mut units: BTreeMap<&str, UnitPanel<'_>> = BTreeMap::new();
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PanelDidSection>()?;
    module.add_class::<SyntheticControlSection>()?;
    module.add_class::<LocalPolynomialRatioSection>()?;
    module.add_class::<SyntheticDidSection>()?;
    module.add_function(wrap_pyfunction!(difference_in_differences, module)?)?;
    module.add_function(wrap_pyfunction!(group_time_att, module)?)?;
    Ok(())
}
