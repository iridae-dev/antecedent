//! Sequential inverse-probability value for prespecified longitudinal regimes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;

/// Retained sequential randomized regime value with bounded IPW inference.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct LongitudinalRegimeSection {
    pub method: String,
    pub rule_id: Option<String>,
    pub rule_version: Option<String>,
    pub rule_provenance: Option<String>,
    pub value: f64,
    pub value_standard_error: Option<f64>,
    pub value_interval_95: Option<(f64, f64)>,
    pub interval_reason: Option<String>,
    pub effective_sample_size: f64,
    pub matched_observed_fraction: f64,
    pub maximum_weight: f64,
    pub minimum_action_probability: f64,
    pub minimum_censoring_probability: f64,
    pub uncertainty: String,
    pub probability_ownership: String,
    pub period_effects: Option<Vec<f64>>,
    pub standard_errors: Option<Vec<f64>>,
    pub stabilizing_numerator_probabilities: Option<Vec<f64>>,
    pub observed_subjects: Option<usize>,
}

impl From<&antecedent::LongitudinalRegimeEstimate> for LongitudinalRegimeSection {
    fn from(value: &antecedent::LongitudinalRegimeEstimate) -> Self {
        Self {
            method: value.method.to_string(),
            rule_id: value.rule_id.as_ref().map(ToString::to_string),
            rule_version: value.rule_version.as_ref().map(ToString::to_string),
            rule_provenance: value.rule_provenance.as_ref().map(ToString::to_string),
            value: value.value,
            value_standard_error: value.value_standard_error,
            value_interval_95: value.value_interval_95.map(|bounds| (bounds[0], bounds[1])),
            interval_reason: value.interval_reason.as_ref().map(ToString::to_string),
            effective_sample_size: value.effective_sample_size,
            matched_observed_fraction: value.matched_observed_fraction,
            maximum_weight: value.maximum_weight,
            minimum_action_probability: value.minimum_action_probability,
            minimum_censoring_probability: value.minimum_censoring_probability,
            uncertainty: value.uncertainty.to_string(),
            probability_ownership: value.probability_ownership.to_string(),
            period_effects: value.period_effects.as_ref().map(|v| v.to_vec()),
            standard_errors: value.standard_errors.as_ref().map(|v| v.to_vec()),
            stabilizing_numerator_probabilities: value.stabilizing_numerator_probabilities.as_ref().map(|v| v.to_vec()),
            observed_subjects: value.observed_subjects,
        }
    }
}

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
    let summary = antecedent_estimate::longitudinal_regime::evaluate_regime_value(
        &y.iter().copied().collect::<Vec<_>>(),
        &a.iter().copied().collect::<Vec<_>>(),
        &d.iter().copied().collect::<Vec<_>>(),
        &p.iter().copied().collect::<Vec<_>>(),
        &r.iter().copied().collect::<Vec<_>>(),
        &g.iter().copied().collect::<Vec<_>>(),
        periods,
        f64::EPSILON,
    )
    .map_err(|message| {
        PyValueError::new_err(if message.contains("no observed subjects") {
            "no observed trajectories followed the requested regime"
        } else {
            message
        })
    })?;
    Ok((
        summary.value,
        summary.effective_sample_size,
        summary.matched_observed_fraction,
        summary.maximum_weight,
    ))
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
    let y = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let a = treatment.as_array();
    let (n, periods) = a.dim();
    let p = treatment_probability.as_array();
    let censor = censoring_survival.as_array();
    if p.dim() != (n, periods) || censor.dim() != (n, periods) {
        return Err(PyValueError::new_err("MSM arrays must have matching subject and period dimensions"));
    }
    let a = a.iter().copied().collect::<Vec<_>>();
    let p = p.iter().copied().collect::<Vec<_>>();
    let censor = censor.iter().copied().collect::<Vec<_>>();
    let numerator = numerator_probability.as_array().iter().copied().collect::<Vec<_>>();
    let observed = outcome_observed.as_array().iter().copied().collect::<Vec<_>>();
    let summary = antecedent_estimate::marginal_structural_model::fit_binary_msm(
        &y, &a, &p, &numerator, &observed,
        &censor, periods, minimum_probability,
    ).map_err(PyValueError::new_err)?;
    Ok((summary.intercept, summary.period_effects, summary.standard_errors,
        summary.effective_sample_size, summary.maximum_weight, summary.observed_subjects))
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

    let summary = antecedent_estimate::longitudinal_regime::evaluate_g_formula_value(
        &q.iter().copied().collect::<Vec<_>>(),
        &actions.iter().copied().collect::<Vec<_>>(),
        &p.iter().copied().collect::<Vec<_>>(),
        &g.iter().copied().collect::<Vec<_>>(),
        n,
        periods,
        minimum_probability,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        summary.value,
        n,
        summary.minimum_action_probability,
        summary.minimum_censoring_probability,
    ))
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

    let summary = antecedent_estimate::longitudinal_regime::evaluate_sequential_dr_value(
        &y.iter().copied().collect::<Vec<_>>(),
        &observed_y.iter().copied().collect::<Vec<_>>(),
        &observed.iter().copied().collect::<Vec<_>>(),
        &a.iter().copied().collect::<Vec<_>>(),
        &d.iter().copied().collect::<Vec<_>>(),
        &q.iter().copied().collect::<Vec<_>>(),
        &p.iter().copied().collect::<Vec<_>>(),
        &g.iter().copied().collect::<Vec<_>>(),
        periods, minimum_probability,
    ).map_err(PyValueError::new_err)?;
    Ok((summary.value, n, summary.minimum_action_probability, summary.minimum_censoring_probability))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LongitudinalRegimeSection>()?;
    module.add_function(wrap_pyfunction!(evaluate_longitudinal_regime_value, module)?)?;
    module.add_function(wrap_pyfunction!(fit_binary_msm, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_sequential_gformula, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_sequential_doubly_robust, module)?)?;
    Ok(())
}
