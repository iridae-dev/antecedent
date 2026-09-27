//! Held-out binary treatment-policy value under known randomized assignment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Typed held-out policy answer. It is deliberately separate from an ATE.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PolicyValueSection {
    /// Net policy value per subject.
    pub policy_value: f64,
    /// Net reference value per subject.
    pub reference_value: f64,
    /// Paired incremental net value.
    pub incremental_value: f64,
    /// Reference minus policy value.
    pub relative_value_gap: f64,
    /// Fraction recommended treatment.
    pub treatment_rate: f64,
    /// Total policy cost.
    pub total_cost: f64,
    /// Policy and reference standard errors from independent-subject row scores.
    pub policy_standard_error: f64,
    /// Reference standard error.
    pub reference_standard_error: f64,
    /// Paired incremental standard error.
    pub incremental_standard_error: f64,
    /// Pointwise 95% policy value interval, if supported.
    pub policy_interval_95: Option<(f64, f64)>,
    /// Paired pointwise 95% incremental-value interval, if supported.
    pub incremental_interval_95: Option<(f64, f64)>,
    /// Caller-declared prediction ownership.
    pub prediction_ownership: String,
    /// Propensity support range.
    pub propensity_min: f64,
    /// Propensity support maximum.
    pub propensity_max: f64,
    /// Native estimator uncertainty semantics.
    pub uncertainty: String,
    /// Exact graphless support license, when its policy row matches.
    pub graphless_support_status: Option<String>,
    /// Held-out randomized uplift by descending frozen score bin.
    pub uplift_bins: Vec<(usize, f64, f64, usize, Option<(f64, f64)>)>,
    /// Conditional contrasts: group, action, effect, total and observed rows, SE, interval.
    pub multi_action_cate: Vec<(String, String, f64, usize, usize, usize, f64, Option<(f64, f64)>)>,
    /// Finite-class regret: gap, simultaneous bounds, values, paired SEs, selected index.
    pub regret: Option<(f64, (f64, f64), Vec<f64>, Vec<f64>, usize)>,
}

/// Fixed group-dose policy value under a kernel-smoothed intervention.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct DosePolicyValueSection {
    pub policy_doses: Vec<(String, f64)>,
    pub reference_doses: Vec<(String, f64)>,
    pub policy_value: f64,
    pub reference_value: f64,
    pub incremental_value: f64,
    pub policy_variance: f64,
    pub reference_variance: f64,
    pub incremental_variance: f64,
    pub policy_interval_95: Option<(f64, f64)>,
    pub reference_interval_95: Option<(f64, f64)>,
    pub incremental_interval_95: Option<(f64, f64)>,
    pub minimum_local_rows: usize,
    pub minimum_effective_sample_size: f64,
    pub maximum_normalized_weight: f64,
    pub minimum_dose_density: f64,
}

/// Retained conditional continuous-dose grid and optional fixed policy value.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct ContinuousDoseResponseSection {
    /// Group, target, response, local rows, effective N, minimum density,
    /// maximum normalized weight, and descriptive local outcome SD.
    pub points: Vec<(String, f64, f64, usize, f64, f64, f64, f64)>,
    /// Prespecified kernel bandwidth.
    pub bandwidth: f64,
    /// Caller-declared density provenance.
    pub density_provenance: String,
    /// Point-only uncertainty semantics.
    pub uncertainty: String,
    /// Kernel-smoothed policy and reference value when group-dose rules were supplied.
    pub fixed_policy: Option<DosePolicyValueSection>,
}

impl From<&antecedent::ContinuousDoseResponseEstimate> for ContinuousDoseResponseSection {
    fn from(value: &antecedent::ContinuousDoseResponseEstimate) -> Self {
        Self {
            points: value.points.iter().map(|point| (
                point.baseline_group.clone(), point.target_dose, point.response,
                point.local_rows, point.effective_sample_size, point.minimum_dose_density,
                point.maximum_normalized_weight, point.local_outcome_sd,
            )).collect(),
            bandwidth: value.bandwidth,
            density_provenance: value.density_provenance.to_string(),
            uncertainty: value.uncertainty.to_string(),
            fixed_policy: value.fixed_policy.as_ref().map(|policy| DosePolicyValueSection {
                policy_doses: policy.policy_doses.clone(),
                reference_doses: policy.reference_doses.clone(),
                policy_value: policy.policy_value,
                reference_value: policy.reference_value,
                incremental_value: policy.incremental_value,
                policy_variance: policy.policy_variance,
                reference_variance: policy.reference_variance,
                incremental_variance: policy.incremental_variance,
                policy_interval_95: policy.policy_interval_95.map(|[a, b]| (a, b)),
                reference_interval_95: policy.reference_interval_95.map(|[a, b]| (a, b)),
                incremental_interval_95: policy.incremental_interval_95.map(|[a, b]| (a, b)),
                minimum_local_rows: policy.minimum_local_rows,
                minimum_effective_sample_size: policy.minimum_effective_sample_size,
                maximum_normalized_weight: policy.maximum_normalized_weight,
                minimum_dose_density: policy.minimum_dose_density,
            }),
        }
    }
}

impl From<&antecedent::PolicyValueEstimate> for PolicyValueSection {
    fn from(value: &antecedent::PolicyValueEstimate) -> Self {
        Self {
            policy_value: value.policy_value,
            reference_value: value.reference_value,
            incremental_value: value.incremental_value,
            relative_value_gap: value.relative_value_gap,
            treatment_rate: value.treatment_rate,
            total_cost: value.total_cost,
            policy_standard_error: value.policy_standard_error,
            reference_standard_error: value.reference_standard_error,
            incremental_standard_error: value.incremental_standard_error,
            policy_interval_95: value.policy_interval_95.map(|bounds| (bounds[0], bounds[1])),
            incremental_interval_95: value.incremental_interval_95.map(|bounds| (bounds[0], bounds[1])),
            prediction_ownership: value.prediction_ownership.to_string(),
            propensity_min: value.propensity_min,
            propensity_max: value.propensity_max,
            uncertainty: value.uncertainty.to_string(),
            graphless_support_status: None,
            uplift_bins: value.uplift_bins.iter().map(|bin| (
                bin.rank, bin.effect, bin.standard_error, bin.evaluation_rows,
                bin.interval_95.map(|bounds| (bounds[0], bounds[1])),
            )).collect(),
            multi_action_cate: value.multi_action_cate.iter().map(|point| (
                point.group.clone(), point.action.clone(), point.effect,
                point.evaluation_rows, point.observed_action_rows, point.observed_control_rows,
                point.standard_error, point.interval_95.map(|bounds| (bounds[0], bounds[1])),
            )).collect(),
            regret: value.regret.as_ref().map(|regret| (
                regret.regret, (regret.interval_95[0], regret.interval_95[1]),
                regret.candidate_values.clone(), regret.contrast_standard_errors.clone(),
                regret.selected_index,
            )),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    outcome, assignment, actions, propensity, *, reference=None, costs=vec![0.0], reference_costs=None,
    available=None, capacity=None, budget=None, reference_capacity=None,
    reference_budget=None
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_binary_policy(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    actions: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
    reference: Option<Vec<bool>>,
    costs: Vec<f64>,
    reference_costs: Option<Vec<f64>>,
    available: Option<Vec<bool>>,
    capacity: Option<usize>,
    budget: Option<f64>,
    reference_capacity: Option<usize>,
    reference_budget: Option<f64>,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let y = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let p = propensity.as_array();
    let n = y.len();
    if n == 0 || assignment.len() != n || actions.len() != n {
        return Err(PyValueError::new_err(
            "outcome, assignment, and policy actions must have equal non-zero length",
        ));
    }
    if p.len() != 1 && p.len() != n {
        return Err(PyValueError::new_err(
            "propensity must have one value or one value per evaluation row",
        ));
    }
    let reference = reference.unwrap_or_else(|| vec![false; n]);
    if reference.len() != n {
        return Err(PyValueError::new_err("reference policy length must match evaluation rows"));
    }
    let available = available.unwrap_or_else(|| vec![true; n]);
    if available.len() != n {
        return Err(PyValueError::new_err("availability length must match evaluation rows"));
    }
    if costs.len() != 1 && costs.len() != n {
        return Err(PyValueError::new_err("costs must be scalar or one value per row"));
    }
    let reference_costs = reference_costs.unwrap_or_else(|| costs.clone());
    if reference_costs.len() != 1 && reference_costs.len() != n {
        return Err(PyValueError::new_err("reference_costs must be scalar or one value per row"));
    }
    if reference_costs.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err(PyValueError::new_err("reference costs must be finite and non-negative"));
    }
    if costs.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err(PyValueError::new_err("treatment costs must be finite and non-negative"));
    }
    if [budget, reference_budget]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return Err(PyValueError::new_err("budgets must be finite and non-negative"));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    let cost = |i: usize, values: &[f64]| values[if values.len() == 1 { 0 } else { i }];
    let probability = |i: usize| p[if p.len() == 1 { 0 } else { i }];
    let policy_cost = |policy: &[bool], costs: &[f64]| -> f64 {
        policy.iter().enumerate().map(|(i, treat)| f64::from(*treat) * cost(i, costs)).sum()
    };
    let policy_actions = actions.iter().filter(|&&treat| treat).count();
    if actions.iter().enumerate().any(|(i, treat)| *treat && !available[i]) {
        return Err(PyValueError::new_err("policy treats a unit whose action is unavailable"));
    }
    if reference.iter().enumerate().any(|(i, treat)| *treat && !available[i]) {
        return Err(PyValueError::new_err(
            "reference policy treats a unit whose action is unavailable",
        ));
    }
    if capacity.is_some_and(|limit| policy_actions > limit) {
        return Err(PyValueError::new_err("policy exceeds the declared treatment capacity"));
    }
    let reference_actions = reference.iter().filter(|&&treat| treat).count();
    if reference_capacity.is_some_and(|limit| reference_actions > limit) {
        return Err(PyValueError::new_err(
            "reference policy exceeds the declared treatment capacity",
        ));
    }
    let total_cost = policy_cost(&actions, &costs);
    if budget.is_some_and(|limit| total_cost > limit) {
        return Err(PyValueError::new_err("policy exceeds the declared treatment budget"));
    }
    let reference_total_cost = policy_cost(&reference, &reference_costs);
    if reference_budget.is_some_and(|limit| reference_total_cost > limit) {
        return Err(PyValueError::new_err(
            "reference policy exceeds the declared treatment budget",
        ));
    }
    let policy_value = ipw_value(&y, &assignment, &actions, &probability)?;
    let reference_value = ipw_value(&y, &assignment, &reference, &probability)?;
    let policy_net = policy_value - total_cost / n as f64;
    let reference_net = reference_value - reference_total_cost / n as f64;
    Ok((
        policy_net,
        reference_net,
        policy_net - reference_net,
        reference_net - policy_net,
        policy_actions as f64 / n as f64,
        total_cost,
    ))
}

/// Doubly robust held-out binary policy value with a paired pointwise standard error.
#[pyfunction]
#[pyo3(signature = (
    outcome, assignment, actions, propensity, mu0, mu1, *, reference=None,
    costs=vec![0.0], reference_costs=None
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_binary_policy_doubly_robust(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    actions: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
    mu0: PyReadonlyArray1<'_, f64>,
    mu1: PyReadonlyArray1<'_, f64>,
    reference: Option<Vec<bool>>,
    costs: Vec<f64>,
    reference_costs: Option<Vec<f64>>,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
    let y = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let p = propensity.as_array();
    let mu0 = mu0.as_array();
    let mu1 = mu1.as_array();
    let n = y.len();
    if n < 2 || assignment.len() != n || actions.len() != n || mu0.len() != n || mu1.len() != n {
        return Err(PyValueError::new_err(
            "outcome, assignment, policy, and both nuisance predictions must align on at least two rows",
        ));
    }
    if p.len() != 1 && p.len() != n {
        return Err(PyValueError::new_err("propensity must be scalar or one value per row"));
    }
    let reference = reference.unwrap_or_else(|| vec![false; n]);
    if reference.len() != n {
        return Err(PyValueError::new_err("reference policy must match evaluation rows"));
    }
    if y.iter().chain(mu0.iter()).chain(mu1.iter()).any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes and nuisance predictions must be finite"));
    }
    let probability = |i: usize| p[if p.len() == 1 { 0 } else { i }];
    if (0..n).any(|i| !probability(i).is_finite() || probability(i) <= 0.0 || probability(i) >= 1.0)
    {
        return Err(PyValueError::new_err(
            "randomized treatment propensities must be strictly between zero and one",
        ));
    }
    if costs.len() != 1 && costs.len() != n {
        return Err(PyValueError::new_err("policy costs must be scalar or one per row"));
    }
    let reference_costs = reference_costs.unwrap_or_else(|| costs.clone());
    if reference_costs.len() != 1 && reference_costs.len() != n {
        return Err(PyValueError::new_err("reference costs must be scalar or one per row"));
    }
    if costs.iter().chain(&reference_costs).any(|cost| !cost.is_finite() || *cost < 0.0) {
        return Err(PyValueError::new_err("policy costs must be finite and non-negative"));
    }
    let scores = antecedent_estimate::policy_value::evaluate_policy_value_scores(
        &y,
        &assignment,
        &actions,
        p.as_slice().unwrap_or(&[]),
        &mu0.to_vec(),
        &mu1.to_vec(),
        &reference,
        &costs,
        &reference_costs,
    )
    .map_err(|message| PyValueError::new_err(message))?;
    let policy_value = scores.policy_value;
    let reference_value = scores.reference_value;
    let incremental = scores.incremental_value;
    let policy_se = scores.policy_standard_error;
    let reference_se = scores.reference_standard_error;
    let incremental_se = scores.incremental_standard_error;
    let propensity_min = scores.propensity_min;
    let propensity_max = scores.propensity_max;
    if [
        policy_value,
        reference_value,
        incremental,
        policy_se,
        reference_se,
        incremental_se,
        propensity_min,
        propensity_max,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "doubly robust policy score overflowed finite precision",
        ));
    }
    Ok((
        policy_value,
        reference_value,
        incremental,
        -incremental,
        scores.treatment_rate,
        scores.total_cost,
        policy_se,
        reference_se,
        incremental_se,
        propensity_min,
        propensity_max,
    ))
}

fn ipw_value(
    outcome: &[f64],
    assignment: &[bool],
    actions: &[bool],
    propensity: &impl Fn(usize) -> f64,
) -> PyResult<f64> {
    let n = outcome.len();
    let mut value = 0.0;
    for i in 0..n {
        let e = propensity(i);
        if !e.is_finite() || e <= 0.0 || e >= 1.0 {
            return Err(PyValueError::new_err(
                "assignment propensities must be finite and strictly between zero and one",
            ));
        }
        if assignment[i] == actions[i] {
            let action_probability = if actions[i] { e } else { 1.0 - e };
            value += outcome[i] / action_probability;
        }
    }
    Ok(value / n as f64)
}

/// Held-out multi-action policy value under known randomized action probabilities.
#[pyfunction]
#[pyo3(signature = (
    outcome, assigned, actions, probabilities, *, reference=None, costs=None, reference_costs=None,
    available=None, capacities=None, reference_capacities=None, budget=None,
    reference_budget=None
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_multi_action_policy(
    outcome: PyReadonlyArray1<'_, f64>,
    assigned: Vec<usize>,
    actions: Vec<usize>,
    probabilities: PyReadonlyArray2<'_, f64>,
    reference: Option<Vec<usize>>,
    costs: Option<Vec<f64>>,
    reference_costs: Option<Vec<f64>>,
    available: Option<Vec<Vec<bool>>>,
    capacities: Option<Vec<usize>>,
    reference_capacities: Option<Vec<usize>>,
    budget: Option<f64>,
    reference_budget: Option<f64>,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let y = outcome.as_array();
    let p = probabilities.as_array();
    let n = y.len();
    let action_count = p.ncols();
    if n == 0 || p.nrows() != n || assigned.len() != n || actions.len() != n {
        return Err(PyValueError::new_err(
            "outcome, assignment, policy, and propensity rows must have equal non-zero length",
        ));
    }
    if action_count < 2 || assigned.iter().chain(&actions).any(|&a| a >= action_count) {
        return Err(PyValueError::new_err("action index is outside the declared action set"));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    for row in p.rows() {
        let total = row.iter().sum::<f64>();
        if row.iter().any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0) || (total - 1.0).abs() > 1e-8
        {
            return Err(PyValueError::new_err(
                "each action propensity must be positive and rows must sum to one",
            ));
        }
    }
    let reference = reference.unwrap_or_else(|| vec![0; n]);
    if reference.len() != n || reference.iter().any(|&a| a >= action_count) {
        return Err(PyValueError::new_err("reference action indexes are invalid"));
    }
    let costs = costs.unwrap_or_else(|| vec![0.0; action_count]);
    if costs.len() != action_count || costs.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err(PyValueError::new_err(
            "costs must contain one finite non-negative value per action",
        ));
    }
    let reference_costs = reference_costs.unwrap_or_else(|| costs.clone());
    if reference_costs.len() != action_count
        || reference_costs.iter().any(|c| !c.is_finite() || *c < 0.0)
    {
        return Err(PyValueError::new_err(
            "reference_costs must contain one finite non-negative value per action",
        ));
    }
    if [budget, reference_budget]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return Err(PyValueError::new_err("budgets must be finite and non-negative"));
    }
    let available = available.unwrap_or_else(|| vec![vec![true; action_count]; n]);
    if available.len() != n || available.iter().any(|row| row.len() != action_count) {
        return Err(PyValueError::new_err("availability must have one bool per row and action"));
    }
    let capacities = capacities.unwrap_or_else(|| vec![n; action_count]);
    if capacities.len() != action_count {
        return Err(PyValueError::new_err("capacities must contain one limit per action"));
    }
    let reference_capacities = reference_capacities.unwrap_or_else(|| vec![n; action_count]);
    if reference_capacities.len() != action_count {
        return Err(PyValueError::new_err(
            "reference_capacities must contain one limit per action",
        ));
    }
    let validate_policy = |policy: &[usize],
                           limits: &[usize],
                           policy_costs: &[f64],
                           policy_budget: Option<f64>|
     -> PyResult<(f64, usize)> {
        let mut total_cost = 0.0;
        let mut counts = vec![0usize; action_count];
        for (i, &action) in policy.iter().enumerate() {
            if !available[i][action] {
                return Err(PyValueError::new_err("policy selects an unavailable action"));
            }
            counts[action] += 1;
            total_cost += policy_costs[action];
        }
        if counts.iter().zip(limits).any(|(count, limit)| count > limit) {
            return Err(PyValueError::new_err("policy exceeds an action capacity"));
        }
        if policy_budget.is_some_and(|limit| total_cost > limit) {
            return Err(PyValueError::new_err("policy exceeds the declared budget"));
        }
        Ok((total_cost, counts.iter().skip(1).sum()))
    };
    let (policy_cost, non_control) = validate_policy(&actions, &capacities, &costs, budget)?;
    let (reference_cost, _) =
        validate_policy(&reference, &reference_capacities, &reference_costs, reference_budget)?;
    let value = |policy: &[usize]| -> PyResult<f64> {
        let mut sum = 0.0;
        for i in 0..n {
            if assigned[i] == policy[i] {
                sum += y[i] / p[[i, assigned[i]]];
            }
        }
        Ok(sum / n as f64)
    };
    let policy_value = value(&actions)? - policy_cost / n as f64;
    let reference_value = value(&reference)? - reference_cost / n as f64;
    Ok((
        policy_value,
        reference_value,
        policy_value - reference_value,
        non_control as f64 / n as f64,
        policy_cost,
    ))
}

/// Held-out uplift by precomputed score rank bins under known randomization.
#[pyfunction]
fn uplift_by_score(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    score_bin: Vec<usize>,
    propensity: PyReadonlyArray1<'_, f64>,
    bin_count: usize,
) -> PyResult<Vec<(usize, f64, f64, usize)>> {
    let y = outcome.as_array();
    let p = propensity.as_array();
    let n = y.len();
    if n == 0
        || bin_count == 0
        || assignment.len() != n
        || score_bin.len() != n
        || (p.len() != 1 && p.len() != n)
    {
        return Err(PyValueError::new_err("uplift arrays and non-zero bin count must align"));
    }
    let mut values = vec![Vec::<f64>::new(); bin_count];
    for i in 0..n {
        let pi = p[if p.len() == 1 { 0 } else { i }];
        if !y[i].is_finite() || !pi.is_finite() || pi <= 0.0 || pi >= 1.0 {
            return Err(PyValueError::new_err(
                "outcomes must be finite and propensities in (0, 1)",
            ));
        }
        if score_bin[i] >= bin_count {
            return Err(PyValueError::new_err("score bin index is outside bin_count"));
        }
        let score = if assignment[i] { y[i] / pi } else { -y[i] / (1.0 - pi) };
        values[score_bin[i]].push(score);
    }
    values
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            if group.is_empty() {
                return Err(PyValueError::new_err("every score bin must contain evaluation rows"));
            }
            let n_group = group.len();
            let effect = group.iter().sum::<f64>() / n_group as f64;
            let variance = group.iter().map(|v| (v - effect).powi(2)).sum::<f64>()
                / ((n_group.saturating_sub(1).max(1)) as f64 * n_group as f64);
            Ok((index, effect, variance.sqrt(), n_group))
        })
        .collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PolicyValueSection>()?;
    module.add_class::<ContinuousDoseResponseSection>()?;
    module.add_class::<DosePolicyValueSection>()?;
    module.add_function(wrap_pyfunction!(evaluate_binary_policy, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_binary_policy_doubly_robust, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_multi_action_policy, module)?)?;
    module.add_function(wrap_pyfunction!(uplift_by_score, module)?)?;
    Ok(())
}
