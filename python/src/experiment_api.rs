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
    /// Design-based conservative variance estimate; no interval is implied.
    pub variance_upper_bound: f64,
    /// Minimum assignment probability.
    pub minimum_assignment_probability: f64,
    /// Assignment design (`bernoulli`, `complete`, or `stratified`).
    pub assignment_design: String,
    /// Row-aligned block labels when stratified.
    pub blocks: Vec<String>,
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
            variance_upper_bound: value.variance_upper_bound,
            minimum_assignment_probability: value.minimum_assignment_probability,
            assignment_design: value.assignment_design.to_string(),
            blocks: value.blocks.iter().map(ToString::to_string).collect(),
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
    let n = y.len();
    let arms = p.ncols();
    if n == 0 || arms < 2 || assignment.len() != n || p.nrows() != n {
        return Err(PyValueError::new_err(
            "outcome, assignment, and multi-arm probability rows must align",
        ));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    let mut means = vec![0.0; arms];
    let mut variance_bounds = vec![0.0; arms];
    let mut support = vec![0usize; arms];
    for i in 0..n {
        if assignment[i] >= arms {
            return Err(PyValueError::new_err("assigned action index is outside the action set"));
        }
        let row = p.row(i);
        let total = row.iter().sum::<f64>();
        if row.iter().any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
            || (total - 1.0).abs() > 1e-8
        {
            return Err(PyValueError::new_err(
                "each action propensity must be positive and rows must sum to one",
            ));
        }
        let arm = assignment[i];
        let probability = p[[i, arm]];
        means[arm] += y[i] / probability / n as f64;
        variance_bounds[arm] +=
            (1.0 - probability) * y[i].powi(2) / probability.powi(2) / (n as f64).powi(2);
        support[arm] += 1;
    }
    if support.contains(&0) {
        return Err(PyValueError::new_err(
            "multi-arm positivity failure: every declared arm needs observed support",
        ));
    }
    Ok((0..arms).map(|arm| (means[arm], variance_bounds[arm], support[arm])).collect())
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
    let n = y.len();
    let k = cov.ncols();
    let q = k + 2;
    if n <= q || assignment.len() != n || cov.nrows() != n || k == 0 {
        return Err(PyValueError::new_err(
            "ANCOVA requires aligned rows, at least one covariate, and residual degrees of freedom",
        ));
    }
    if y.iter().any(|value| !value.is_finite()) || cov.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes and covariates must be finite"));
    }
    let treated = assignment.iter().filter(|&&value| value).count();
    let control = n - treated;
    if treated == 0 || control == 0 {
        return Err(PyValueError::new_err("ANCOVA requires observed treated and control units"));
    }
    let rows = (0..n)
        .map(|i| {
            let mut row = Vec::with_capacity(q);
            row.push(1.0);
            row.push(if assignment[i] { 1.0 } else { 0.0 });
            row.extend(cov.row(i).iter().copied());
            row
        })
        .collect::<Vec<_>>();
    let mut gram = vec![vec![0.0; q]; q];
    let mut rhs = vec![0.0; q];
    for (row, &target) in rows.iter().zip(y.iter()) {
        for j in 0..q {
            rhs[j] += row[j] * target;
            for l in 0..q {
                gram[j][l] += row[j] * row[l];
            }
        }
    }
    let inverse = invert_gram(gram)?;
    let beta =
        (0..q).map(|j| (0..q).map(|l| inverse[j][l] * rhs[l]).sum::<f64>()).collect::<Vec<_>>();
    let mut meat = vec![vec![0.0; q]; q];
    for (row, &target) in rows.iter().zip(y.iter()) {
        let residual = target - row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
        for j in 0..q {
            for l in 0..q {
                meat[j][l] += row[j] * row[l] * residual.powi(2);
            }
        }
    }
    // The sandwich is (X'X)^-1 (X' diag(e^2) X) (X'X)^-1.
    let mut effect_variance = 0.0;
    for j in 0..q {
        for l in 0..q {
            effect_variance += inverse[1][j] * meat[j][l] * inverse[l][1];
        }
    }
    Ok((beta[1], effect_variance.max(0.0).sqrt(), beta[2..].to_vec(), treated, control))
}

fn invert_gram(mut matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let n = matrix.len();
    let max_diagonal = (0..n).map(|i| matrix[i][i].abs()).fold(0.0_f64, f64::max);
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        inverse[i][i] = 1.0;
    }
    for column in 0..n {
        let pivot_row = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .expect("non-empty pivot range");
        let pivot_scale = matrix[pivot_row][column].abs();
        if !pivot_scale.is_finite()
            || pivot_scale <= f64::EPSILON * 16.0
            || pivot_scale <= max_diagonal.max(1.0) * 1e-12
        {
            return Err(PyValueError::new_err(
                "ANCOVA design is rank deficient or covariates are collinear",
            ));
        }
        matrix.swap(column, pivot_row);
        inverse.swap(column, pivot_row);
        let pivot = matrix[column][column];
        for j in 0..n {
            matrix[column][j] /= pivot;
            inverse[column][j] /= pivot;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let multiplier = matrix[row][column];
            for j in 0..n {
                matrix[row][j] -= multiplier * matrix[column][j];
                inverse[row][j] -= multiplier * inverse[column][j];
            }
        }
    }
    Ok(inverse)
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
