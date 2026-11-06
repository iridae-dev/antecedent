//! Synthetic difference-in-differences fitting and assignment tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::{SyntheticDidFit, SyntheticUnitRandomizationTest, demean, fit_weights};

/// Forward rounding bound for a contrast summing `terms` floating-point products whose
/// absolute magnitudes total `magnitude`.
///
/// Each operation adds at most one relative rounding of `f64::EPSILON / 2`; the factor
/// below is `f64::EPSILON` per operation, twice that, with a fixed allowance for the
/// divisions, weight products and the subtraction of a sharp-null effect.
fn accumulation_bound(terms: usize, magnitude: f64) -> f64 {
    (terms as f64 + 8.0) * f64::EPSILON * magnitude
}

/// Fit unit and time simplex weights without claiming interval calibration.
// length reflects the estimator's fixed statistical contract; refactor would change behavior
#[allow(
    clippy::too_many_lines,
    reason = "build the panel, fit unit and time weights, then the effect and placebo permutation from the same donor set"
)]
pub fn fit_synthetic_did(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
) -> Result<SyntheticDidFit, String> {
    let values = outcome;
    let n = values.len();
    if n == 0 || units.len() != n || periods.len() != n {
        return Err(String::from(
            "outcome, unit, and period vectors must have equal non-zero length",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(String::from("outcomes must be finite"));
    }
    if treated_unit.is_empty() || intervention_period <= 0 {
        return Err(String::from("treated unit and intervention period are required"));
    }
    let mut panel: BTreeMap<String, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        if units[i].is_empty() || periods[i] <= 0 {
            return Err(String::from("unit IDs must be non-empty and periods positive"));
        }
        all_periods.insert(periods[i]);
        if panel.entry(units[i].clone()).or_default().insert(periods[i], values[i]).is_some() {
            return Err(String::from("each unit must have exactly one outcome per period"));
        }
    }
    let period_list: Vec<i64> = all_periods.into_iter().collect();
    let pre: Vec<i64> =
        period_list.iter().copied().filter(|period| *period < intervention_period).collect();
    let post: Vec<i64> =
        period_list.iter().copied().filter(|period| *period >= intervention_period).collect();
    if pre.len() < 2 || post.is_empty() {
        return Err(String::from(
            "synthetic DiD requires at least two pre-periods and one post-period",
        ));
    }
    if panel.len() < 3 || !panel.contains_key(treated_unit) {
        return Err(String::from(
            "synthetic DiD requires one treated unit and at least two donor units",
        ));
    }
    let n_pre = pre.len();
    let n_post = post.len();
    if panel.values().any(|observed| {
        observed.len() != period_list.len()
            || period_list.iter().any(|period| !observed.contains_key(period))
    }) {
        return Err(String::from(
            "synthetic DiD requires a balanced panel with identical periods for every unit",
        ));
    }
    let treated = &panel[treated_unit];
    let donors: Vec<(&String, &BTreeMap<i64, f64>)> =
        panel.iter().filter(|(unit, _)| unit.as_str() != treated_unit).collect();
    let target_pre: Vec<f64> = pre.iter().map(|period| treated[period]).collect();
    let donor_pre: Vec<Vec<f64>> = donors
        .iter()
        .map(|(_, observed)| pre.iter().map(|period| observed[period]).collect())
        .collect();
    let (unit_weights, pre_rmse) = fit_weights(&target_pre, &donor_pre)?;

    // SDID time weights (Arkhangelsky et al.) represent each donor's post-period
    // mean as a convex combination of its pre-period values, up to a common
    // intercept. Demeaning across donors absorbs that intercept.
    let target_donor_mean = demean(
        &donors
            .iter()
            .map(|(_, observed)| {
                post.iter().map(|period| observed[period]).sum::<f64>() / post.len() as f64
            })
            .collect::<Vec<_>>(),
    );
    let period_profiles: Vec<Vec<f64>> = pre
        .iter()
        .map(|period| {
            demean(&donors.iter().map(|(_, observed)| observed[period]).collect::<Vec<_>>())
        })
        .collect();
    let (time_weights, _) = fit_weights(&target_donor_mean, &period_profiles)?;

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
    let outcome_scale = values.iter().fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let weight_residual = (unit_weights.iter().sum::<f64>() - 1.0).abs()
        + (time_weights.iter().sum::<f64>() - 1.0).abs();
    let magnitude = treated_post_mean.abs()
        + pre.iter().zip(&time_weights).map(|(p, w)| (treated[p] * w).abs()).sum::<f64>()
        + donors
            .iter()
            .zip(&unit_weights)
            .map(|((_, observed), weight)| {
                let post_mean = post.iter().map(|p| observed[p]).sum::<f64>() / post.len() as f64;
                weight.abs() * post_mean.abs()
                    + pre
                        .iter()
                        .zip(&time_weights)
                        .map(|(p, time_weight)| (weight * time_weight * observed[p]).abs())
                        .sum::<f64>()
            })
            .sum::<f64>();
    let rounding_bound = accumulation_bound(
        donors.len() * (pre.len() + post.len()) + pre.len() + post.len(),
        magnitude,
    ) + weight_residual * outcome_scale;
    if !estimate.is_finite() || !pre_rmse.is_finite() || !rounding_bound.is_finite() {
        return Err(String::from("synthetic DiD estimate overflowed finite precision"));
    }
    let weights = donors
        .iter()
        .zip(unit_weights)
        .map(|((unit, _), weight)| ((*unit).clone(), weight))
        .collect();
    let time_weights = pre.into_iter().zip(time_weights).collect();
    Ok(SyntheticDidFit {
        effect: estimate,
        pre_treatment_rmse: pre_rmse,
        donor_weights: weights,
        time_weights,
        n_donors: donors.len(),
        n_pre_periods: n_pre,
        n_post_periods: n_post,
        rounding_bound,
    })
}

/// Exact sharp-null assignment test for a uniformly selected synthetic-DiD treated unit.
///
/// Refit both unit and time weights for every possible assignment on the
/// unchanged outcome panel. This is valid for a declared uniform one-unit
/// randomization under the sharp null; it does not calibrate an interval or
/// turn observational donor placebos into a randomization test.
pub fn exact_synthetic_did_unit_randomization_test(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
) -> Result<SyntheticUnitRandomizationTest, String> {
    fit_synthetic_did(outcome, units, periods, treated_unit, intervention_period)?;
    let candidates: BTreeSet<&str> = units.iter().map(String::as_str).collect();
    let mut statistics = Vec::with_capacity(candidates.len());
    let mut ranking_tolerance = 0.0_f64;
    for candidate in candidates {
        let fit = fit_synthetic_did(outcome, units, periods, candidate, intervention_period)?;
        ranking_tolerance = ranking_tolerance.max(fit.rounding_bound);
        statistics.push((candidate.to_string(), fit.effect.abs()));
    }
    let observed = statistics
        .iter()
        .find(|(unit, _)| unit == treated_unit)
        .ok_or_else(|| {
            "treated unit is absent from exact synthetic DiD randomization distribution".to_string()
        })?
        .1;
    // Mathematically tied assignments differ only by arithmetic noise, so every statistic
    // within the accumulated rounding bound of the observed one counts as at least as
    // extreme. The tail fraction is therefore never below the strict one.
    let extreme = statistics
        .iter()
        .filter(|(_, statistic)| *statistic >= observed - ranking_tolerance)
        .count();
    Ok(SyntheticUnitRandomizationTest {
        p_value: extreme as f64 / statistics.len() as f64,
        statistics,
        ranking_tolerance,
    })
}
