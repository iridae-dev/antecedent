//! Randomized encouragement and noncompliance estimators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::detach_catch;

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
    /// Exact graphless matrix status for the retained result, when licensed.
    pub graphless_support_status: Option<String>,
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
            multi_arm_values: value
                .multi_arm_values
                .iter()
                .map(|(label, mean, variance, support)| {
                    (label.to_string(), *mean, *variance, *support)
                })
                .collect(),
            variance_upper_bound: value.variance_upper_bound,
            standard_error: value.standard_error,
            interval_95: value.interval_95.map(|interval| (interval[0], interval[1])),
            second_factor_interval_95: value
                .second_factor_interval_95
                .map(|interval| (interval[0], interval[1])),
            factorial_interaction_interval_95: value
                .factorial_interaction_interval_95
                .map(|interval| (interval[0], interval[1])),
            multi_arm_intervals_95: value
                .multi_arm_intervals_95
                .iter()
                .map(|interval| interval.map(|value| (value[0], value[1])))
                .collect(),
            minimum_assignment_probability: value.minimum_assignment_probability,
            assignment_design: value.assignment_design.to_string(),
            blocks: value.blocks.iter().map(ToString::to_string).collect(),
            periods: value.periods.iter().map(ToString::to_string).collect(),
            control_units: value.control_units,
            treatment_units: value.treatment_units,
            uncertainty: value.uncertainty.to_string(),
            graphless_support_status: None,
            assignment_units: value.assignment_units.iter().map(ToString::to_string).collect(),
            outcome_units: value.outcome_units.iter().map(ToString::to_string).collect(),
            control_arm: value.treatment_arms.0.to_string(),
            treatment_arm: value.treatment_arms.1.to_string(),
        }
    }
}

/// CUPED covariate residualization with a known-propensity HT contrast.
#[pyfunction]
fn estimate_cuped_effect(
    py: Python<'_>,
    outcome: PyReadonlyArray1<'_, f64>,
    covariate: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64)> {
    let outcome = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let covariate = covariate.as_array().iter().copied().collect::<Vec<_>>();
    let propensity = propensity.as_array().iter().copied().collect::<Vec<_>>();
    let fit = detach_catch(py, move || {
        antecedent_estimate::randomized_scores::cuped_ht_effect(
            &outcome,
            &covariate,
            &assignment,
            &propensity,
        )
        .map_err(crate::value_err)
    })?;
    Ok((fit.effect, fit.beta, fit.standard_error))
}

/// Stratified complete-randomization effect with a conservative variance bound.
#[pyfunction]
fn estimate_stratified_effect(
    py: Python<'_>,
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    blocks: Vec<String>,
) -> PyResult<(f64, f64, f64, usize)> {
    let outcome = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let fit = detach_catch(py, move || {
        antecedent_estimate::randomized_neyman::stratified_neyman_effect(
            &outcome,
            &assignment,
            &blocks,
        )
        .map_err(crate::value_err)
    })?;
    Ok((fit.effect, fit.variance, fit.minimum_probability, fit.strata))
}

/// Exact Bernoulli randomization test of the sharp no-effect null.
#[pyfunction]
fn exact_randomization_test(
    py: Python<'_>,
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    propensity: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, usize)> {
    let outcome = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let propensity = propensity.as_array().iter().copied().collect::<Vec<_>>();
    let fit = detach_catch(py, move || {
        antecedent_estimate::randomized_scores::bernoulli_exact_randomization_test(
            &outcome,
            &assignment,
            &propensity,
        )
        .map_err(crate::value_err)
    })?;
    Ok((fit.observed, fit.p_value, fit.allocations))
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
        return Err(crate::value_err(
            "switchback outcome, assignment, sequence, and propensity arrays must align",
        ));
    }
    if sequences.iter().any(|label| label.trim().is_empty()) {
        return Err(crate::value_err("sequence labels must be non-empty"));
    }
    let ids = sequences.iter().map(String::as_str).collect::<Vec<_>>();
    let fit = antecedent_estimate::switchback::switchback_itt(
        &y.to_vec(), &assignment, &p.to_vec(), &ids,
    ).ok_or_else(|| crate::value_err(
        "switchback requires finite outcomes, valid probabilities, at least two independent sequences, and global support for both arms",
    ))?;
    let minimum_probability =
        p.iter().copied().map(|pi| pi.min(1.0 - pi)).fold(f64::INFINITY, f64::min);
    Ok((fit.effect, fit.variance.sqrt(), minimum_probability, fit.sequences))
}

/// OLS ANCOVA treatment coefficient with an HC0 independent-row sandwich SE.
#[pyfunction]
fn estimate_ancova_effect(
    py: Python<'_>,
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
    let fit = detach_catch(py, move || {
        let refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        antecedent_estimate::ancova::fit_ancova(&outcomes, &assignment, &refs)
            .map_err(crate::value_err)
    })?;
    Ok((fit.effect, fit.hc0_variance.sqrt(), fit.covariate_coefficients, fit.treated, fit.control))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RandomizedEffectSection>()?;
    module.add_function(wrap_pyfunction!(estimate_cuped_effect, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_stratified_effect, module)?)?;
    module.add_function(wrap_pyfunction!(exact_randomization_test, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_switchback_effect, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_ancova_effect, module)?)?;
    Ok(())
}
