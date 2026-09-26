//! Sequential inverse-probability value for prespecified longitudinal regimes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;

/// Point-only Horvitz--Thompson value of a static or dynamic regime.
#[pyfunction]
fn evaluate_longitudinal_regime_value(
    outcome: PyReadonlyArray1<'_, f64>,
    treatment: PyReadonlyArray2<'_, bool>,
    regime: PyReadonlyArray2<'_, bool>,
    treatment_probability: PyReadonlyArray2<'_, f64>,
    outcome_observed: PyReadonlyArray1<'_, bool>,
    censoring_survival: PyReadonlyArray2<'_, f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    let y = outcome.as_array();
    let a = treatment.as_array();
    let d = regime.as_array();
    let p = treatment_probability.as_array();
    let r = outcome_observed.as_array();
    let g = censoring_survival.as_array();
    let (n, periods) = a.dim();
    if n == 0
        || periods == 0
        || y.len() != n
        || r.len() != n
        || d.dim() != (n, periods)
        || p.dim() != (n, periods)
        || g.dim() != (n, periods)
    {
        return Err(PyValueError::new_err(
            "regime arrays must have matching subject and period dimensions",
        ));
    }
    let mut weighted_total = 0.0;
    let mut weight_sum = 0.0;
    let mut weight_square_sum = 0.0;
    let mut max_weight: f64 = 0.0;
    let mut supported = 0usize;
    for i in 0..n {
        if !y[i].is_finite() {
            return Err(PyValueError::new_err("outcomes must be finite"));
        }
        let mut matches = true;
        let mut weight = 1.0;
        for t in 0..periods {
            let probability_treated = p[[i, t]];
            let censor_probability = g[[i, t]];
            if !probability_treated.is_finite()
                || probability_treated <= 0.0
                || probability_treated >= 1.0
            {
                return Err(PyValueError::new_err(
                    "treatment probabilities must be strictly between zero and one",
                ));
            }
            if !censor_probability.is_finite()
                || censor_probability <= 0.0
                || censor_probability > 1.0
            {
                return Err(PyValueError::new_err(
                    "censoring survival probabilities must be in (0, 1]",
                ));
            }
            if a[[i, t]] != d[[i, t]] {
                matches = false;
                continue;
            }
            if !matches {
                continue;
            }
            let action_probability =
                if a[[i, t]] { probability_treated } else { 1.0 - probability_treated };
            weight /= action_probability * censor_probability;
            if !weight.is_finite() {
                return Err(PyValueError::new_err(
                    "sequential inverse-probability weight overflowed; raise the positivity floor or shorten the horizon",
                ));
            }
        }
        if matches && r[i] {
            supported += 1;
            weighted_total += weight * y[i];
            weight_sum += weight;
            weight_square_sum += weight * weight;
            max_weight = max_weight.max(weight);
        }
    }
    if supported == 0 {
        return Err(PyValueError::new_err(
            "no observed trajectories followed the requested regime",
        ));
    }
    let ess =
        if weight_square_sum > 0.0 { weight_sum * weight_sum / weight_square_sum } else { 0.0 };
    Ok((weighted_total / n as f64, ess, supported as f64 / n as f64, max_weight))
}

/// Additive marginal structural mean model with sequential stabilized weights.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_binary_msm(
    outcome: PyReadonlyArray1<'_, f64>,
    treatment: PyReadonlyArray2<'_, bool>,
    treatment_probability: PyReadonlyArray2<'_, f64>,
    numerator_probability: PyReadonlyArray1<'_, f64>,
    outcome_observed: PyReadonlyArray1<'_, bool>,
    censoring_survival: PyReadonlyArray2<'_, f64>,
    minimum_probability: f64,
) -> PyResult<(f64, Vec<f64>, Vec<f64>, f64, f64, usize)> {
    let y = outcome.as_array();
    let a = treatment.as_array();
    let p = treatment_probability.as_array();
    let numerator = numerator_probability.as_array();
    let observed = outcome_observed.as_array();
    let censor = censoring_survival.as_array();
    let (n, periods) = a.dim();
    if n == 0
        || periods == 0
        || y.len() != n
        || observed.len() != n
        || p.dim() != (n, periods)
        || censor.dim() != (n, periods)
        || numerator.len() != periods
    {
        return Err(PyValueError::new_err(
            "MSM arrays must have matching subject and period dimensions",
        ));
    }
    if !minimum_probability.is_finite()
        || !(0.0..=0.5).contains(&minimum_probability)
        || minimum_probability == 0.0
    {
        return Err(PyValueError::new_err("minimum_probability must be finite and in (0, 0.5]"));
    }
    for (t, &probability) in numerator.iter().enumerate() {
        if !probability.is_finite()
            || probability < minimum_probability
            || probability > 1.0 - minimum_probability
        {
            return Err(PyValueError::new_err(format!(
                "stabilizing numerator probability at period {t} violates the declared positivity floor"
            )));
        }
    }
    let columns = periods + 1;
    let included = observed.iter().filter(|&&included| included).count();
    if included <= columns {
        return Err(PyValueError::new_err(
            "MSM requires more observed subjects than intercept-plus-treatment coefficients",
        ));
    }
    let mut design = vec![vec![0.0; columns]; n];
    let mut weights = vec![0.0; n];
    let mut max_weight: f64 = 0.0;
    let mut weight_sum = 0.0;
    let mut weight_square_sum = 0.0;
    for i in 0..n {
        design[i][0] = 1.0;
        if observed[i] && !y[i].is_finite() {
            return Err(PyValueError::new_err("observed terminal outcomes must be finite"));
        }
        let mut weight = 1.0;
        for t in 0..periods {
            let probability = p[[i, t]];
            let censor_probability = censor[[i, t]];
            if !probability.is_finite()
                || probability < minimum_probability
                || probability > 1.0 - minimum_probability
            {
                return Err(PyValueError::new_err(format!(
                    "sequential treatment positivity is violated at subject {i}, period {t}"
                )));
            }
            if !censor_probability.is_finite()
                || censor_probability < minimum_probability
                || censor_probability > 1.0
            {
                return Err(PyValueError::new_err(format!(
                    "sequential censoring positivity is violated at subject {i}, period {t}"
                )));
            }
            design[i][t + 1] = if a[[i, t]] { 1.0 } else { 0.0 };
            if observed[i] {
                let numerator_action = if a[[i, t]] { numerator[t] } else { 1.0 - numerator[t] };
                let denominator_action = if a[[i, t]] { probability } else { 1.0 - probability };
                weight *= numerator_action / (denominator_action * censor_probability);
                if !weight.is_finite() {
                    return Err(PyValueError::new_err(
                        "stabilized sequential weight overflowed; raise positivity floors or shorten the horizon",
                    ));
                }
            }
        }
        if observed[i] {
            weights[i] = weight;
            max_weight = max_weight.max(weight);
            weight_sum += weight;
            weight_square_sum += weight * weight;
            if !weight_sum.is_finite() || !weight_square_sum.is_finite() {
                return Err(PyValueError::new_err(
                    "stabilized weight diagnostics overflowed; raise positivity floors or shorten the horizon",
                ));
            }
        }
    }
    let mut bread_input = vec![vec![0.0; columns]; columns];
    let mut rhs = vec![0.0; columns];
    for i in 0..n {
        if !observed[i] {
            continue;
        }
        for j in 0..columns {
            rhs[j] += weights[i] * design[i][j] * y[i];
            for k in 0..columns {
                bread_input[j][k] += weights[i] * design[i][j] * design[i][k];
            }
        }
    }
    if rhs.iter().chain(bread_input.iter().flatten()).any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "weighted marginal structural model normal equations overflowed",
        ));
    }
    let bread = invert_matrix(bread_input)?;
    let beta = multiply_vector(&bread, &rhs);
    if beta.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("marginal structural model coefficients are non-finite"));
    }
    let mut meat = vec![vec![0.0; columns]; columns];
    for i in 0..n {
        if !observed[i] {
            continue;
        }
        let residual = y[i] - dot(&design[i], &beta);
        let score = design[i].iter().map(|value| weights[i] * value * residual).collect::<Vec<_>>();
        for j in 0..columns {
            for k in 0..columns {
                meat[j][k] += score[j] * score[k];
            }
        }
    }
    let mut covariance = multiply_matrices(&multiply_matrices(&bread, &meat), &bread);
    let correction = included as f64 / (included - columns) as f64;
    for row in &mut covariance {
        for element in row {
            *element *= correction;
        }
    }
    let standard_errors =
        (1..columns).map(|index| covariance[index][index].max(0.0).sqrt()).collect::<Vec<_>>();
    let effective_sample_size = weight_sum * weight_sum / weight_square_sum;
    if covariance.iter().flatten().any(|value| !value.is_finite())
        || standard_errors.iter().any(|value| !value.is_finite())
        || !effective_sample_size.is_finite()
    {
        return Err(PyValueError::new_err("marginal structural model covariance is non-finite"));
    }
    Ok((
        beta[0],
        beta.into_iter().skip(1).collect(),
        standard_errors,
        effective_sample_size,
        max_weight,
        included,
    ))
}

fn invert_matrix(mut matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let size = matrix.len();
    let mut inverse = vec![vec![0.0; size]; size];
    for index in 0..size {
        inverse[index][index] = 1.0;
    }
    for column in 0..size {
        let pivot = (column..size)
            .max_by(|&left, &right| {
                matrix[left][column].abs().total_cmp(&matrix[right][column].abs())
            })
            .expect("non-empty pivot candidates");
        let scale = matrix[pivot].iter().map(|value| value.abs()).fold(0.0, f64::max);
        if matrix[pivot][column].abs() <= 1e-12 * scale.max(1.0) {
            return Err(PyValueError::new_err(
                "weighted marginal structural model design is rank-deficient",
            ));
        }
        matrix.swap(column, pivot);
        inverse.swap(column, pivot);
        let divisor = matrix[column][column];
        for index in 0..size {
            matrix[column][index] /= divisor;
            inverse[column][index] /= divisor;
        }
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for index in 0..size {
                matrix[row][index] -= factor * matrix[column][index];
                inverse[row][index] -= factor * inverse[column][index];
            }
        }
    }
    Ok(inverse)
}

fn multiply_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn multiply_matrices(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let size = left.len();
    let mut result = vec![vec![0.0; size]; size];
    for row in 0..size {
        for middle in 0..size {
            for column in 0..size {
                result[row][column] += left[row][middle] * right[middle][column];
            }
        }
    }
    result
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Plug-in sequential g-formula value for caller-supplied conditional rewards.
#[pyfunction]
fn evaluate_sequential_gformula(
    period_outcome_predictions: PyReadonlyArray2<'_, f64>,
    regime_actions: PyReadonlyArray2<'_, bool>,
    treatment_probability: PyReadonlyArray2<'_, f64>,
    censoring_survival: PyReadonlyArray2<'_, f64>,
    subject_ids: Vec<String>,
    fold_ids: Vec<i64>,
    minimum_probability: f64,
) -> PyResult<(f64, usize, f64, f64)> {
    let q = period_outcome_predictions.as_array();
    let actions = regime_actions.as_array();
    let p = treatment_probability.as_array();
    let g = censoring_survival.as_array();
    let (n, periods) = q.dim();
    if n == 0
        || periods == 0
        || actions.dim() != (n, periods)
        || p.dim() != (n, periods)
        || g.dim() != (n, periods)
        || subject_ids.len() != n
        || fold_ids.len() != n
    {
        return Err(PyValueError::new_err(
            "prediction, regime, probability, subject, and fold inputs must have matching non-zero subject-period dimensions",
        ));
    }
    if !minimum_probability.is_finite() || minimum_probability <= 0.0 || minimum_probability > 0.5 {
        return Err(PyValueError::new_err("minimum_probability must be finite and in (0, 0.5]"));
    }
    if subject_ids.iter().any(|id| id.trim().is_empty()) {
        return Err(PyValueError::new_err("subject_ids must be non-empty strings"));
    }
    let unique_subjects: HashSet<&str> = subject_ids.iter().map(String::as_str).collect();
    if unique_subjects.len() != n {
        return Err(PyValueError::new_err(
            "g-formula inputs require one row per unique subject to preserve fold ownership",
        ));
    }
    if fold_ids.iter().any(|&fold| fold < 0) {
        return Err(PyValueError::new_err(
            "fold_ids must be non-negative to preserve subject-level fold ownership",
        ));
    }

    let mut reward_total = 0.0;
    let mut minimum_action_probability: f64 = 1.0;
    let mut minimum_censoring_probability: f64 = 1.0;
    for i in 0..n {
        let mut subject_reward = 0.0;
        for t in 0..periods {
            let prediction = q[[i, t]];
            let probability_treated = p[[i, t]];
            let censor_probability = g[[i, t]];
            if !prediction.is_finite() {
                return Err(PyValueError::new_err("period_outcome_predictions must be finite"));
            }
            if !probability_treated.is_finite()
                || probability_treated < minimum_probability
                || probability_treated > 1.0 - minimum_probability
            {
                return Err(PyValueError::new_err(
                    "sequential treatment positivity is violated at the declared probability floor",
                ));
            }
            if !censor_probability.is_finite()
                || censor_probability < minimum_probability
                || censor_probability > 1.0
            {
                return Err(PyValueError::new_err(
                    "sequential censoring positivity is violated at the declared probability floor",
                ));
            }
            let action_probability =
                if actions[[i, t]] { probability_treated } else { 1.0 - probability_treated };
            minimum_action_probability = minimum_action_probability.min(action_probability);
            minimum_censoring_probability = minimum_censoring_probability.min(censor_probability);
            subject_reward += prediction;
        }
        if !subject_reward.is_finite() {
            return Err(PyValueError::new_err("summed subject outcome prediction overflowed"));
        }
        reward_total += subject_reward;
        if !reward_total.is_finite() {
            return Err(PyValueError::new_err(
                "summed outcome predictions overflowed across subjects",
            ));
        }
    }
    Ok((reward_total / n as f64, n, minimum_action_probability, minimum_censoring_probability))
}

/// Sequentially augmented regime value from caller-supplied cross-fitted Q predictions.
#[pyfunction]
fn evaluate_sequential_doubly_robust(
    outcome: PyReadonlyArray1<'_, f64>,
    outcome_observed: PyReadonlyArray1<'_, bool>,
    observation_history: PyReadonlyArray2<'_, bool>,
    treatment: PyReadonlyArray2<'_, bool>,
    regime: PyReadonlyArray2<'_, bool>,
    q_prediction: PyReadonlyArray2<'_, f64>,
    treatment_probability: PyReadonlyArray2<'_, f64>,
    censoring_probability: PyReadonlyArray2<'_, f64>,
    subject_ids: Vec<String>,
    fold_ids: Vec<i64>,
    prediction_fold_ids: Vec<i64>,
    minimum_probability: f64,
) -> PyResult<(f64, usize, f64, f64)> {
    let y = outcome.as_array();
    let observed_y = outcome_observed.as_array();
    let observed = observation_history.as_array();
    let a = treatment.as_array();
    let d = regime.as_array();
    let q = q_prediction.as_array();
    let p = treatment_probability.as_array();
    let g = censoring_probability.as_array();
    let (n, periods) = a.dim();
    if n == 0
        || periods == 0
        || y.len() != n
        || observed_y.len() != n
        || observed.dim() != (n, periods)
        || d.dim() != (n, periods)
        || q.dim() != (n, periods)
        || p.dim() != (n, periods)
        || g.dim() != (n, periods)
        || subject_ids.len() != n
        || fold_ids.len() != n
        || prediction_fold_ids.len() != n
    {
        return Err(PyValueError::new_err(
            "outcome, history, regime, Q, probability, subject, and fold inputs must have matching non-zero dimensions",
        ));
    }
    if !minimum_probability.is_finite() || minimum_probability <= 0.0 || minimum_probability > 0.5 {
        return Err(PyValueError::new_err("minimum_probability must be finite and in (0, 0.5]"));
    }
    if subject_ids.iter().any(|id| id.trim().is_empty()) {
        return Err(PyValueError::new_err("subject_ids must be non-empty strings"));
    }
    let unique_subjects: HashSet<&str> = subject_ids.iter().map(String::as_str).collect();
    if unique_subjects.len() != n {
        return Err(PyValueError::new_err(
            "doubly robust inputs require one row per unique subject to preserve fold ownership",
        ));
    }
    if fold_ids.iter().any(|&fold| fold < 0)
        || fold_ids.iter().zip(&prediction_fold_ids).any(|(fold, owner)| fold != owner)
    {
        return Err(PyValueError::new_err(
            "Q prediction fold ownership must match the non-negative subject fold IDs",
        ));
    }
    if !observed_y.iter().any(|&is_observed| is_observed) {
        return Err(PyValueError::new_err(
            "at least one terminal outcome must be observed for sequential augmentation",
        ));
    }

    let mut total = 0.0;
    let mut minimum_action_probability: f64 = 1.0;
    let mut minimum_censoring_probability: f64 = 1.0;
    for i in 0..n {
        if observed_y[i] && !y[i].is_finite() {
            return Err(PyValueError::new_err("observed terminal outcomes must be finite"));
        }
        if observed[[i, periods - 1]] != observed_y[i] {
            return Err(PyValueError::new_err(
                "terminal outcome observation must agree with the final observation-history period",
            ));
        }
        for t in 1..periods {
            if !observed[[i, t - 1]] && observed[[i, t]] {
                return Err(PyValueError::new_err(
                    "observation_history must be monotone after censoring or dropout",
                ));
            }
        }
        let mut next_value = if observed_y[i] { y[i] } else { 0.0 };
        for t in (0..periods).rev() {
            let q_value = q[[i, t]];
            let probability_treated = p[[i, t]];
            let censor_probability = g[[i, t]];
            if !q_value.is_finite() {
                return Err(PyValueError::new_err("q_prediction values must be finite"));
            }
            if !probability_treated.is_finite()
                || probability_treated < minimum_probability
                || probability_treated > 1.0 - minimum_probability
            {
                return Err(PyValueError::new_err(
                    "sequential treatment positivity is violated at the declared probability floor",
                ));
            }
            if !censor_probability.is_finite()
                || censor_probability < minimum_probability
                || censor_probability > 1.0
            {
                return Err(PyValueError::new_err(
                    "sequential censoring positivity is violated at the declared probability floor",
                ));
            }
            let action_probability =
                if d[[i, t]] { probability_treated } else { 1.0 - probability_treated };
            minimum_action_probability = minimum_action_probability.min(action_probability);
            minimum_censoring_probability = minimum_censoring_probability.min(censor_probability);
            if observed[[i, t]] && a[[i, t]] == d[[i, t]] {
                next_value =
                    q_value + (next_value - q_value) / (action_probability * censor_probability);
            } else {
                next_value = q_value;
            }
            if !next_value.is_finite() {
                return Err(PyValueError::new_err(
                    "sequential augmentation overflowed; raise the positivity floor or shorten the horizon",
                ));
            }
        }
        total += next_value;
        if !total.is_finite() {
            return Err(PyValueError::new_err(
                "sequential augmented values overflowed across subjects",
            ));
        }
    }
    Ok((total / n as f64, n, minimum_action_probability, minimum_censoring_probability))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(evaluate_longitudinal_regime_value, module)?)?;
    module.add_function(wrap_pyfunction!(fit_binary_msm, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_sequential_gformula, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_sequential_doubly_robust, module)?)?;
    Ok(())
}
