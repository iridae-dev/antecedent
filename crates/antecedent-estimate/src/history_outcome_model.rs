//! Bayesian outcome-model effect on the treated for units that carry a history and a calendar
//! period.
//!
//! A history-matched contrast has no posterior (see [`crate::matching_posterior_boundary`]).
//! This module builds the estimand that does: the effect of treatment on the treated units in
//! the declared Gaussian outcome model
//!
//! ```text
//! y = a_period + g * score + b * t + d * t * (score - score_treated) + noise
//! ```
//!
//! where `score` is the mean of the unit's history and `score_treated` is its mean over the
//! treated units. Centring the interaction at the treated mean makes `b` the model-based
//! contrast averaged over the treated units, for every posterior draw. It is not a matching
//! estimand: no unit is paired with a donor, the model's linearity in the score and its period
//! effects carry the confounding adjustment, and a period without both arms contributes no
//! information about the contrast, so its units are left out and reported. Rows are modelled
//! as independent given the period and the score. The same fit goes through
//! [`crate::BayesianGComputationAte`], so the prior, the draws and the diagnostics are those of
//! Bayesian g-computation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{VariableId, reason_code};
use antecedent_stats::{CompiledDesign, DesignColumnMap, DesignColumnRole, StandardizationRecord};

use crate::EstimationError;
use crate::bayesian::PreparedBayesianProblem;
use crate::history_match::HistoryUnit;
use crate::overlap::OverlapPolicy;
use crate::serial_dependence::SerialDependence;

/// The prepared outcome-model problem and what was left out of it.
#[derive(Clone, Debug)]
pub struct HistoryOutcomeProblem {
    /// The design for [`crate::BayesianGComputationAte::fit`]; its treatment coefficient is the
    /// effect on the treated.
    pub prepared: PreparedBayesianProblem,
    /// Rows in the fitted design.
    pub rows: usize,
    /// Treated rows in the fitted design.
    pub treated_rows: usize,
    /// Calendar periods in the fitted design.
    pub periods: usize,
    /// Input indices of treated units in a period without a control, left out of the fit.
    pub unused_treated: Vec<usize>,
}

/// Build the outcome-model design for the effect on the treated.
///
/// # Errors
///
/// The histories differ in length or there are no units; the sampling weights are not all equal
/// (a precision-weighted fit is not a sampling-weighted posterior); no period has both arms; or
/// the design has too few rows for its coefficients.
pub fn prepare_history_outcome_model(
    units: &[HistoryUnit],
) -> Result<HistoryOutcomeProblem, EstimationError> {
    let Some(first) = units.first() else {
        return Err(EstimationError::data_msg(
            "the history outcome model requires treated and control units",
        ));
    };
    if units.iter().any(|unit| unit.history_len() != first.history_len()) {
        return Err(EstimationError::data_msg(
            "the history outcome model requires every history to have the same length",
        ));
    }
    if units.iter().any(|unit| unit.weight().to_bits() != first.weight().to_bits()) {
        return Err(EstimationError::refused(
            reason_code!("route_not_supported"),
            "the Bayesian history outcome model reads equal sampling weights only: a weighted \
             Gaussian fit would treat sampling weights as precisions and misstate the posterior. \
             Next attempt: resample or replicate the units to the sampling weights, or use the \
             frequentist history-matched contrast, which reads them",
        ));
    }
    let mut arms: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
    for unit in units {
        let entry = arms.entry(unit.period()).or_default();
        if unit.is_treated() {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }
    let kept: Vec<i64> =
        arms.iter().filter(|(_, (t, c))| *t > 0 && *c > 0).map(|(period, _)| *period).collect();
    if kept.is_empty() {
        return Err(EstimationError::data_msg(
            "no calendar period has both treated and control units",
        ));
    }
    let period_index: BTreeMap<i64, usize> =
        kept.iter().enumerate().map(|(index, period)| (*period, index)).collect();
    let mut rows: Vec<usize> = Vec::new();
    let mut unused_treated: Vec<usize> = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        if period_index.contains_key(&unit.period()) {
            rows.push(index);
        } else if unit.is_treated() {
            unused_treated.push(index);
        }
    }
    let n = rows.len();
    let periods = kept.len();
    let ncols = periods + 3;
    if n < ncols + 2 {
        return Err(EstimationError::data_msg(
            "the history outcome model needs more units than coefficients",
        ));
    }
    let treated_rows = rows.iter().filter(|&&row| units[row].is_treated()).count();
    let treated_mean_score = rows
        .iter()
        .filter(|&&row| units[row].is_treated())
        .map(|&row| units[row].score())
        .sum::<f64>()
        / treated_rows as f64;

    let mut matrix = vec![0.0_f64; n * ncols];
    matrix[..n].fill(1.0);
    let mut outcome = Vec::with_capacity(n);
    for (slot, &row) in rows.iter().enumerate() {
        let unit = &units[row];
        let score = unit.score();
        let treated = if unit.is_treated() { 1.0 } else { 0.0 };
        matrix[n + slot] = treated;
        matrix[2 * n + slot] = score;
        let index = period_index[&unit.period()];
        if index > 0 {
            matrix[(2 + index) * n + slot] = 1.0;
        }
        matrix[(ncols - 1) * n + slot] = treated * (score - treated_mean_score);
        outcome.push(unit.outcome());
    }

    let mut roles = vec![
        DesignColumnRole::Intercept,
        DesignColumnRole::Treatment,
        DesignColumnRole::Covariate(VariableId::from_raw(0)),
    ];
    let mut names: Vec<Arc<str>> =
        vec![Arc::from("intercept"), Arc::from("coef_treatment"), Arc::from("coef_history_mean")];
    for (index, period) in kept.iter().enumerate().skip(1) {
        roles.push(DesignColumnRole::Covariate(VariableId::from_raw(index as u32)));
        names.push(Arc::from(format!("coef_period_{period}")));
    }
    roles.push(DesignColumnRole::Covariate(VariableId::from_raw(0)));
    names.push(Arc::from("treatment_x_centered_history_mean"));

    let design = CompiledDesign {
        nrows: n,
        ncols,
        matrix: Arc::from(matrix),
        columns: DesignColumnMap::from_roles(roles),
        outcome: Arc::from(outcome),
        row_selection: Arc::from(rows),
        contrasts: Vec::new(),
        smooths: Vec::new(),
        standardization: StandardizationRecord::default(),
    };
    let prepared = PreparedBayesianProblem {
        design,
        method: Arc::from("bayesian.period_history_outcome_model"),
        adjustment_set: Arc::from(Vec::<VariableId>::new()),
        active: 1.0,
        control: 0.0,
        overlap: OverlapPolicy::ExplicitOverride,
        coef_names: Some(Arc::from(names)),
        unit_ids: None,
        serial_dependence: SerialDependence::Iid,
        preparation: None,
    };
    Ok(HistoryOutcomeProblem { prepared, rows: n, treated_rows, periods, unused_treated })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte};
    use antecedent_core::{ExecutionContext, IdentificationStatus};

    /// Three calendar periods; the effect `1 + 2 * score` grows with the history mean, treatment
    /// is more likely at a high score, and period 9 holds treated units only.
    fn units() -> (Vec<HistoryUnit>, f64) {
        let mut units = Vec::new();
        let mut effects = Vec::new();
        for period in 0..3_i64 {
            for k in 0..60_usize {
                let base = ((k as f64) * 0.37 + period as f64).sin();
                let history = [base, base + 0.1 * ((k % 5) as f64 - 2.0)];
                let score = (history[0] + history[1]) / 2.0;
                let treated = score + ((k as f64) * 0.11).cos() * 0.8 > 0.0;
                let effect = 1.0 + 2.0 * score;
                let wiggle = ((k as f64) * 1.3).sin() * 0.05;
                let outcome =
                    1.5 * period as f64 + 0.5 * score + if treated { effect } else { 0.0 } + wiggle;
                if treated {
                    effects.push(effect);
                }
                units.push(
                    HistoryUnit::try_new(&history, outcome, treated, 1.0)
                        .unwrap()
                        .in_period(period),
                );
            }
        }
        for k in 0..3_usize {
            units.push(
                HistoryUnit::try_new(&[0.0, 0.0], 10.0 + k as f64, true, 1.0).unwrap().in_period(9),
            );
        }
        let truth = effects.iter().sum::<f64>() / effects.len() as f64;
        (units, truth)
    }

    #[test]
    fn the_treatment_coefficient_recovers_the_modelled_effect_on_the_treated() {
        let (units, truth) = units();
        let problem = prepare_history_outcome_model(&units).unwrap();
        assert_eq!(problem.periods, 3);
        assert_eq!(problem.rows, 180);
        assert_eq!(problem.unused_treated, vec![180, 181, 182]);
        assert_eq!(problem.prepared.design.ncols, 6);
        let names = problem.prepared.coef_names.as_ref().unwrap();
        assert_eq!(names[5].as_ref(), "treatment_x_centered_history_mean");
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 2000,
            seed: 4,
            prior_scale: 100.0,
            ..BayesianGComputationAte::new()
        };
        let mut workspace = BayesianGCompWorkspace::default();
        let posterior = bayes
            .fit(
                &problem.prepared,
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                &mut workspace,
                &ExecutionContext::for_tests(2),
            )
            .unwrap();
        let effect = posterior.effect_column().unwrap();
        let mean = posterior.summaries.mean[effect];
        assert!((mean - truth).abs() < 0.05, "posterior mean {mean} vs effect on treated {truth}");
    }

    #[test]
    fn unequal_sampling_weights_and_missing_arms_are_refused() {
        let unit = |treated: bool, weight: f64, period: i64| {
            HistoryUnit::try_new(&[0.0, 1.0], 1.0, treated, weight).unwrap().in_period(period)
        };
        let weighted = [unit(true, 1.0, 0), unit(false, 2.0, 0)];
        let error = prepare_history_outcome_model(&weighted).unwrap_err().to_string();
        assert!(error.contains("equal sampling weights"), "{error}");
        assert!(error.contains("Next attempt"), "{error}");
        let one_arm = [unit(true, 1.0, 0), unit(true, 1.0, 1)];
        let error = prepare_history_outcome_model(&one_arm).unwrap_err().to_string();
        assert!(error.contains("both treated and control"), "{error}");
    }
}
