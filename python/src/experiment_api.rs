//! Randomized encouragement and noncompliance estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Randomized ITT result plus its frozen experiment design identity.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct RandomizedEffectSection {
    /// ITT contrast under the declared assignment mechanism.
    pub effect: f64,
    /// `itt` or `cace_late`.
    pub estimand: String,
    /// Outcome assignment effect for CACE/LATE.
    pub intention_to_treat_effect: Option<f64>,
    /// Treatment-receipt first stage for CACE/LATE.
    pub first_stage_effect: Option<f64>,
    /// Row-aligned observed receipt for CACE/LATE.
    pub received_treatment: Option<Vec<bool>>,
    /// Exact two-sided Fisher sharp-null p-value.
    pub randomization_p_value: Option<f64>,
    /// Number of complete-design assignments enumerated.
    pub randomization_allocations: Option<u64>,
    /// Second-factor marginal effect in a fixed four-cell design.
    pub second_factor_effect: Option<f64>,
    /// Difference of primary-factor effects between second-factor levels.
    pub factorial_interaction: Option<f64>,
    /// Conservative variance of the second-factor main effect.
    pub second_factor_variance: Option<f64>,
    /// Conservative variance of the factorial interaction.
    pub factorial_interaction_variance: Option<f64>,
    /// Ordered multi-arm arm means, variance contributions, and observed support.
    pub multi_arm_values: Vec<(String, f64, f64, usize)>,
    /// Design variance or conservative bound as labeled by uncertainty;
    /// switchback uses a sequence sandwich estimate. No interval is implied.
    pub variance_upper_bound: f64,
    /// Primary pointwise 95% interval and its reported standard error.
    pub standard_error: Option<f64>,
    pub interval_95: Option<(f64, f64)>,
    /// Additional pointwise factorial intervals.
    pub second_factor_interval_95: Option<(f64, f64)>,
    pub factorial_interaction_interval_95: Option<(f64, f64)>,
    /// Pointwise multi-arm intervals aligned with declared action labels.
    pub multi_arm_intervals_95: Vec<Option<(f64, f64)>>,
    /// Minimum assignment probability.
    pub minimum_assignment_probability: f64,
    /// Assignment design (`bernoulli`, `complete`, or `stratified`).
    pub assignment_design: String,
    /// Row-aligned block labels when stratified.
    pub blocks: Vec<String>,
    /// Row-aligned periods for switchback designs.
    pub periods: Vec<String>,
    /// Number of analyzed control units.
    pub control_units: usize,
    /// Number of analyzed treatment units.
    pub treatment_units: usize,
    /// Explicit uncertainty semantics; never implies a confidence interval.
    pub uncertainty: String,
    /// Assignment units in analyzed row order.
    pub assignment_units: Vec<String>,
    /// Outcome units in analyzed row order.
    pub outcome_units: Vec<String>,
    /// Control arm label.
    pub control_arm: String,
    /// Treatment arm label.
    pub treatment_arm: String,
}

impl From<&antecedent::RandomizedEffectEstimate> for RandomizedEffectSection {
    fn from(value: &antecedent::RandomizedEffectEstimate) -> Self {
        Self {
            effect: value.effect,
            estimand: value.estimand.to_string(),
            intention_to_treat_effect: value.intention_to_treat_effect,
            first_stage_effect: value.first_stage_effect,
            received_treatment: value.received_treatment.as_ref().map(|receipt| receipt.to_vec()),
            randomization_p_value: value.randomization_p_value,
            randomization_allocations: value.randomization_allocations,
            second_factor_effect: value.second_factor_effect,
            factorial_interaction: value.factorial_interaction,
            second_factor_variance: value.second_factor_variance,
            factorial_interaction_variance: value.factorial_interaction_variance,
            multi_arm_values: value.multi_arm_values.iter().map(|(label, mean, variance, support)|
                (label.to_string(), *mean, *variance, *support)).collect(),
            variance_upper_bound: value.variance_upper_bound,
            standard_error: value.standard_error,
            interval_95: value.interval_95.map(|interval| (interval[0], interval[1])),
            second_factor_interval_95: value.second_factor_interval_95.map(|interval| (interval[0], interval[1])),
            factorial_interaction_interval_95: value.factorial_interaction_interval_95.map(|interval| (interval[0], interval[1])),
            multi_arm_intervals_95: value.multi_arm_intervals_95.iter().map(|interval|
                interval.map(|value| (value[0], value[1]))).collect(),
            minimum_assignment_probability: value.minimum_assignment_probability,
            assignment_design: value.assignment_design.to_string(),
            blocks: value.blocks.iter().map(ToString::to_string).collect(),
            periods: value.periods.iter().map(ToString::to_string).collect(),
            control_units: value.control_units,
            treatment_units: value.treatment_units,
            uncertainty: value.uncertainty.to_string(),
            assignment_units: value.assignment_units.iter().map(ToString::to_string).collect(),
            outcome_units: value.outcome_units.iter().map(ToString::to_string).collect(),
            control_arm: value.treatment_arms.0.to_string(),
            treatment_arm: value.treatment_arms.1.to_string(),
        }
    }
}

/// Horvitz--Thompson ITT and Wald complier effect with an influence-function SE.
#[pyfunction]
fn estimate_complier_effect(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    received: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    let y = outcome.as_array();
    let p = propensity.as_array();
    let n = y.len();
    if n < 2 || assignment.len() != n || received.len() != n || (p.len() != 1 && p.len() != n) {
        return Err(PyValueError::new_err(
            "outcome, assignment, receipt, and propensity must have compatible non-zero rows",
        ));
    }
    let probability = |i: usize| p[if p.len() == 1 { 0 } else { i }];
    let mut outcome_itt = 0.0;
    let mut receipt_itt = 0.0;
    let mut outcome_scores = Vec::with_capacity(n);
    let mut receipt_scores = Vec::with_capacity(n);
    for i in 0..n {
        let pi = probability(i);
        if !y[i].is_finite() {
            return Err(PyValueError::new_err("outcomes must be finite"));
        }
        if !pi.is_finite() || pi <= 0.0 || pi >= 1.0 {
            return Err(PyValueError::new_err(
                "assignment propensities must be strictly between zero and one",
            ));
        }
        let sign = if assignment[i] { 1.0 / pi } else { -1.0 / (1.0 - pi) };
        let sy = sign * y[i];
        let sd = sign * f64::from(received[i]);
        outcome_scores.push(sy);
        receipt_scores.push(sd);
        outcome_itt += sy / n as f64;
        receipt_itt += sd / n as f64;
    }
    if receipt_itt <= f64::EPSILON {
        return Err(PyValueError::new_err(
            "first stage must be positive under the declared monotonicity assumption",
        ));
    }
    let effect = outcome_itt / receipt_itt;
    let influence = outcome_scores
        .iter()
        .zip(&receipt_scores)
        .map(|(y_score, d_score)| (y_score - effect * d_score) / receipt_itt)
        .collect::<Vec<_>>();
    let mean = influence.iter().sum::<f64>() / n as f64;
    let variance = influence.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / ((n - 1) as f64 * n as f64);
    Ok((outcome_itt, receipt_itt, effect, variance.sqrt()))
}

/// CUPED covariate residualization with a known-propensity HT contrast.
#[pyfunction]
fn estimate_cuped_effect(
    outcome: PyReadonlyArray1<'_, f64>,
    covariate: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64)> {
    let y = outcome.as_array();
    let x = covariate.as_array();
    let p = propensity.as_array();
    let n = y.len();
    if n < 2 || x.len() != n || assignment.len() != n || (p.len() != 1 && p.len() != n) {
        return Err(PyValueError::new_err(
            "outcome, covariate, assignment, and propensity must have compatible rows",
        ));
    }
    if y.iter().chain(x.iter()).any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes and covariates must be finite"));
    }
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let x_mean = x.iter().sum::<f64>() / n as f64;
    let covariance = y.iter().zip(x.iter()).map(|(a, b)| (a - y_mean) * (b - x_mean)).sum::<f64>();
    let x_variance = x.iter().map(|value| (value - x_mean).powi(2)).sum::<f64>();
    if x_variance <= f64::EPSILON {
        return Err(PyValueError::new_err("CUPED covariate must have non-zero variance"));
    }
    let beta = covariance / x_variance;
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let pi = p[if p.len() == 1 { 0 } else { i }];
        if !pi.is_finite() || pi <= 0.0 || pi >= 1.0 {
            return Err(PyValueError::new_err(
                "assignment propensities must be strictly between zero and one",
            ));
        }
        let adjusted = y[i] - beta * (x[i] - x_mean);
        let sign = if assignment[i] { 1.0 / pi } else { -1.0 / (1.0 - pi) };
        scores.push(sign * adjusted);
    }
    let estimate = scores.iter().sum::<f64>() / n as f64;
    let variance = scores.iter().map(|score| (score - estimate).powi(2)).sum::<f64>()
        / ((n - 1) as f64 * n as f64);
    Ok((estimate, beta, variance.sqrt()))
}

/// Stratified complete-randomization effect with a conservative variance bound.
#[pyfunction]
fn estimate_stratified_effect(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    blocks: Vec<String>,
) -> PyResult<(f64, f64, f64, usize)> {
    let y = outcome.as_array();
    let n = y.len();
    if n == 0 || assignment.len() != n || blocks.len() != n {
        return Err(PyValueError::new_err(
            "outcome, assignment, and block labels must have equal non-zero length",
        ));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    let mut labels = blocks.clone();
    labels.sort();
    labels.dedup();
    let mut effect = 0.0;
    let mut variance = 0.0;
    let mut minimum_probability: f64 = 1.0;
    for label in &labels {
        let rows = blocks
            .iter()
            .enumerate()
            .filter_map(|(i, value)| (value == label).then_some(i))
            .collect::<Vec<_>>();
        let treated = rows.iter().filter(|&&i| assignment[i]).copied().collect::<Vec<_>>();
        let control = rows.iter().filter(|&&i| !assignment[i]).copied().collect::<Vec<_>>();
        if treated.len() < 2 || control.len() < 2 {
            return Err(PyValueError::new_err(
                "each stratum needs at least two treated and two control units for variance estimation",
            ));
        }
        let mean =
            |indices: &[usize]| indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
        let treated_mean = mean(&treated);
        let control_mean = mean(&control);
        let treated_var = treated.iter().map(|&i| (y[i] - treated_mean).powi(2)).sum::<f64>()
            / (treated.len() - 1) as f64;
        let control_var = control.iter().map(|&i| (y[i] - control_mean).powi(2)).sum::<f64>()
            / (control.len() - 1) as f64;
        let block_weight = rows.len() as f64 / n as f64;
        effect += block_weight * (treated_mean - control_mean);
        variance += block_weight.powi(2)
            * (treated_var / treated.len() as f64 + control_var / control.len() as f64);
        let p = treated.len() as f64 / rows.len() as f64;
        minimum_probability = minimum_probability.min(p.min(1.0 - p));
    }
    Ok((effect, variance, minimum_probability, labels.len()))
}

/// Exact Bernoulli randomization test of the sharp no-effect null.
#[pyfunction]
fn exact_randomization_test(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, usize)> {
    let y = outcome.as_array();
    let p = propensity.as_array();
    let n = y.len();
    if n == 0 || n > 20 || assignment.len() != n || (p.len() != 1 && p.len() != n) {
        return Err(PyValueError::new_err(
            "exact randomization inference requires 1–20 aligned rows and propensities",
        ));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    let probabilities = (0..n).map(|i| p[if p.len() == 1 { 0 } else { i }]).collect::<Vec<_>>();
    if probabilities.iter().any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0) {
        return Err(PyValueError::new_err(
            "assignment propensities must be strictly between zero and one",
        ));
    }
    let observed =
        (0..n)
            .map(|i| {
                if assignment[i] {
                    y[i] / probabilities[i]
                } else {
                    -y[i] / (1.0 - probabilities[i])
                }
            })
            .sum::<f64>()
            / n as f64;
    let mut p_value = 0.0;
    let allocations = 1usize << n;
    for mask in 0..allocations {
        let mut probability = 1.0;
        let mut statistic = 0.0;
        for i in 0..n {
            let treated = mask & (1 << i) != 0;
            let pi = probabilities[i];
            probability *= if treated { pi } else { 1.0 - pi };
            statistic += if treated { y[i] / pi } else { -y[i] / (1.0 - pi) };
        }
        statistic /= n as f64;
        if statistic.abs() + 1e-12 >= observed.abs() {
            p_value += probability;
        }
    }
    Ok((observed, p_value.min(1.0), allocations))
}

/// Horvitz--Thompson arm means for a known randomized multi-arm design.
#[pyfunction]
fn estimate_multi_arm_effects(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<usize>,
    probabilities: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<(f64, f64, usize)>> {
    let y = outcome.as_array();
    let p = probabilities.as_array();
    let rows = p.rows().into_iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    antecedent_estimate::multi_arm::estimate_multi_arm(&y.to_vec(), &assignment, &rows)
        .map(|arms| arms.into_iter().map(|arm| (arm.value, arm.variance_bound, arm.observed_support)).collect())
        .map_err(PyValueError::new_err)
}

/// Horvitz--Thompson switchback ITT with independent-sequence sandwich variance.
#[pyfunction]
fn estimate_switchback_effect(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    sequences: Vec<String>,
    propensity: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, usize)> {
    let y = outcome.as_array();
    let p = propensity.as_array();
    let n = y.len();
    if n < 2 || assignment.len() != n || sequences.len() != n || p.len() != n {
        return Err(PyValueError::new_err(
            "switchback outcome, assignment, sequence, and propensity arrays must align",
        ));
    }
    let mut labels = sequences.clone();
    if labels.iter().any(|label| label.trim().is_empty()) {
        return Err(PyValueError::new_err("sequence labels must be non-empty"));
    }
    labels.sort();
    labels.dedup();
    if labels.len() < 2 {
        return Err(PyValueError::new_err(
            "switchback variance requires at least two independent sequences",
        ));
    }
    let mut cluster_scores = vec![0.0; labels.len()];
    let mut observed_arms = vec![[false; 2]; labels.len()];
    let mut minimum_probability: f64 = 1.0;
    for i in 0..n {
        if !y[i].is_finite() {
            return Err(PyValueError::new_err("outcomes must be finite"));
        }
        let pi = p[i];
        if !pi.is_finite() || pi <= 0.0 || pi >= 1.0 {
            return Err(PyValueError::new_err(
                "switchback assignment probabilities must be strictly between zero and one",
            ));
        }
        let cluster = labels.binary_search(&sequences[i]).expect("label was collected");
        let treated = assignment[i];
        observed_arms[cluster][usize::from(treated)] = true;
        let score = if treated { y[i] / pi } else { -y[i] / (1.0 - pi) };
        cluster_scores[cluster] += score;
        minimum_probability = minimum_probability.min(pi.min(1.0 - pi));
    }
    if observed_arms.iter().any(|arms| !arms[0] || !arms[1]) {
        return Err(PyValueError::new_err(
            "each sequence must contain observed treated and control periods",
        ));
    }
    let effect = cluster_scores.iter().sum::<f64>() / n as f64;
    let mean = cluster_scores.iter().sum::<f64>() / labels.len() as f64;
    let variance = labels.len() as f64 / (labels.len() - 1) as f64
        * cluster_scores.iter().map(|score| (score - mean).powi(2)).sum::<f64>()
        / (n as f64).powi(2);
    Ok((effect, variance.sqrt(), minimum_probability, labels.len()))
}

/// OLS ANCOVA treatment coefficient with an HC0 independent-row sandwich SE.
#[pyfunction]
fn estimate_ancova_effect(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    covariates: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, f64, Vec<f64>, usize, usize)> {
    let y = outcome.as_array();
    let cov = covariates.as_array();
    let outcomes = y.iter().copied().collect::<Vec<_>>();
    let columns = (0..cov.ncols())
        .map(|j| cov.column(j).iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let fit = antecedent_estimate::ancova::fit_ancova(
        &outcomes,
        &assignment,
        &refs,
    ).map_err(PyValueError::new_err)?;
    Ok((fit.effect, fit.hc0_variance.sqrt(), fit.covariate_coefficients, fit.treated, fit.control))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RandomizedEffectSection>()?;
    module.add_function(wrap_pyfunction!(estimate_complier_effect, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_cuped_effect, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_stratified_effect, module)?)?;
    module.add_function(wrap_pyfunction!(exact_randomization_test, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_multi_arm_effects, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_switchback_effect, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_ancova_effect, module)?)?;
    Ok(())
}
