//! Deterministic simplex synthetic control shared by direct and retained routes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

/// Point estimate and support diagnostics. The placebo rank is descriptive and uncalibrated.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticControlFit {
    /// Average treated-minus-synthetic outcome over post-intervention periods.
    pub effect: f64,
    /// Root mean squared treated-versus-synthetic gap before intervention.
    pub pre_treatment_rmse: f64,
    /// Simplex donor weights in stable unit-name order.
    pub donor_weights: Vec<(String, f64)>,
    /// Leave-one-donor-out placebo effects in the same donor order.
    pub placebo_effects: Vec<f64>,
    /// Descriptive finite-pool rank, without a calibrated p-value claim.
    pub placebo_rank: f64,
    /// Number of observed pre-intervention periods.
    pub n_pre_periods: usize,
    /// Number of observed post-intervention periods.
    pub n_post_periods: usize,
}

/// Exhaustive sharp-null test over a declared uniform choice of one treated unit.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticUnitRandomizationTest {
    /// Absolute post-intervention gaps for every possible treated unit, in unit order.
    pub statistics: Vec<(String, f64)>,
    /// Exact two-sided finite-assignment tail fraction under the sharp null.
    pub p_value: f64,
}

fn project_simplex(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += value;
        let candidate = (cumulative - 1.0) / (index + 1) as f64;
        if index + 1 == sorted.len() || sorted[index + 1] <= candidate {
            theta = candidate;
            break;
        }
    }
    values.iter().map(|value| (*value - theta).max(0.0)).collect()
}

/// Fit donor weights by projected gradient descent on the simplex.
/// The fixed iteration cap and sorted donor ordering make the result deterministic.
fn fit_weights(target: &[f64], donors: &[Vec<f64>]) -> (Vec<f64>, f64) {
    let n_donors = donors.len();
    let n_pre = target.len();
    let frobenius_sq: f64 = donors.iter().flatten().map(|value| *value * *value).sum();
    let lipschitz = 2.0 * frobenius_sq / n_pre as f64;
    let step = if lipschitz > 1e-15 { 1.0 / lipschitz } else { 1.0 };
    let mut weights = vec![1.0 / n_donors as f64; n_donors];
    for _ in 0..10_000 {
        let mut gradient = vec![0.0; n_donors];
        for period in 0..n_pre {
            let residual: f64 = donors
                .iter()
                .zip(&weights)
                .map(|(donor, weight)| donor[period] * *weight)
                .sum::<f64>()
                - target[period];
            for (index, donor) in donors.iter().enumerate() {
                gradient[index] += 2.0 * donor[period] * residual / n_pre as f64;
            }
        }
        let next = project_simplex(
            &weights
                .iter()
                .zip(gradient)
                .map(|(weight, gradient)| *weight - step * gradient)
                .collect::<Vec<_>>(),
        );
        let movement = next
            .iter()
            .zip(&weights)
            .map(|(next, current)| (*next - *current).abs())
            .fold(0.0_f64, f64::max);
        weights = next;
        if movement < 1e-12 {
            break;
        }
    }
    let rmse = ((0..n_pre)
        .map(|period| {
            let residual = donors
                .iter()
                .zip(&weights)
                .map(|(donor, weight)| donor[period] * *weight)
                .sum::<f64>()
                - target[period];
            residual * residual
        })
        .sum::<f64>()
        / n_pre as f64)
        .sqrt();
    (weights, rmse)
}

fn placebo_ratio(gap: f64, pre_rmse: f64) -> f64 {
    if pre_rmse > 1e-12 {
        gap.abs() / pre_rmse
    } else if gap.abs() <= 1e-12 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// Fit a balanced-panel synthetic control and leave-one-donor-out placebos.
/// No p-value or interval calibration is attached to the returned placebo rank.
pub fn fit_synthetic_control(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
) -> Result<SyntheticControlFit, String> {
    let n = outcome.len();
    if n == 0 || units.len() != n || periods.len() != n {
        return Err("outcome, unit, and period vectors must have equal non-zero length".into());
    }
    if outcome.iter().any(|value| !value.is_finite()) {
        return Err("outcomes must be finite".into());
    }
    if treated_unit.is_empty() || intervention_period <= 0 {
        return Err("treated unit and intervention period are required".into());
    }
    let mut panel: BTreeMap<&str, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        if units[i].is_empty() || periods[i] <= 0 {
            return Err("unit IDs must be non-empty and periods positive".into());
        }
        all_periods.insert(periods[i]);
        if panel.entry(units[i].as_str()).or_default().insert(periods[i], outcome[i]).is_some() {
            return Err("each unit must have exactly one outcome per period".into());
        }
    }
    let period_list: Vec<i64> = all_periods.into_iter().collect();
    let pre: Vec<i64> = period_list.iter().copied().filter(|p| *p < intervention_period).collect();
    let post: Vec<i64> =
        period_list.iter().copied().filter(|p| *p >= intervention_period).collect();
    if pre.len() < 2 || post.is_empty() {
        return Err(
            "synthetic control requires at least two pre-periods and one post-period".into()
        );
    }
    if panel.len() < 4 || !panel.contains_key(treated_unit) {
        return Err(
            "synthetic control requires one observed treated unit and at least three donor units"
                .into(),
        );
    }
    if panel.values().any(|observed| {
        observed.len() != period_list.len()
            || period_list.iter().any(|period| !observed.contains_key(period))
    }) {
        return Err(
            "synthetic control requires a balanced panel with identical periods for every unit"
                .into(),
        );
    }
    let treated = &panel[treated_unit];
    let donors: Vec<(&&str, &BTreeMap<i64, f64>)> =
        panel.iter().filter(|(unit, _)| **unit != treated_unit).collect();
    let target_pre: Vec<f64> = pre.iter().map(|period| treated[period]).collect();
    let donor_pre: Vec<Vec<f64>> = donors
        .iter()
        .map(|(_, values)| pre.iter().map(|period| values[period]).collect())
        .collect();
    let (weights, pre_rmse) = fit_weights(&target_pre, &donor_pre);
    let treated_post = post.iter().map(|period| treated[period]).sum::<f64>() / post.len() as f64;
    let synthetic_post: f64 = donors
        .iter()
        .zip(&weights)
        .map(|((_, values), weight)| {
            *weight * post.iter().map(|period| values[period]).sum::<f64>() / post.len() as f64
        })
        .sum();
    let effect = treated_post - synthetic_post;
    let treated_ratio = placebo_ratio(effect, pre_rmse);
    let mut placebo_effects = Vec::with_capacity(donors.len());
    let mut extreme = 0usize;
    for placebo_index in 0..donors.len() {
        let reference_pre: Vec<Vec<f64>> = donor_pre
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != placebo_index)
            .map(|(_, donor)| donor.clone())
            .collect();
        let (placebo_weights, placebo_rmse) =
            fit_weights(&donor_pre[placebo_index], &reference_pre);
        let pseudo_post = post.iter().map(|period| donors[placebo_index].1[period]).sum::<f64>()
            / post.len() as f64;
        let reference_post: f64 = donors
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != placebo_index)
            .zip(&placebo_weights)
            .map(|((_, (_, values)), weight)| {
                *weight * post.iter().map(|period| values[period]).sum::<f64>() / post.len() as f64
            })
            .sum();
        let placebo_effect = pseudo_post - reference_post;
        placebo_effects.push(placebo_effect);
        if placebo_ratio(placebo_effect, placebo_rmse) >= treated_ratio {
            extreme += 1;
        }
    }
    let placebo_rank = (1 + extreme) as f64 / (1 + donors.len()) as f64;
    if !effect.is_finite()
        || !pre_rmse.is_finite()
        || weights.iter().any(|w| !w.is_finite())
        || placebo_effects.iter().any(|value| !value.is_finite())
    {
        return Err("synthetic control fit overflowed finite precision".into());
    }
    Ok(SyntheticControlFit {
        effect,
        pre_treatment_rmse: pre_rmse,
        donor_weights: donors
            .iter()
            .zip(weights)
            .map(|((unit, _), weight)| ((*unit).to_string(), weight))
            .collect(),
        placebo_effects,
        placebo_rank,
        n_pre_periods: pre.len(),
        n_post_periods: post.len(),
    })
}

/// Refit every possible treated unit against all other units under the sharp null.
///
/// This is an exact Fisher test only when the one treated unit was selected
/// uniformly from the observed balanced panel before outcomes were seen.
/// The test statistic is the absolute post-treatment synthetic gap. No
/// confidence interval or observational placebo interpretation follows.
pub fn exact_synthetic_unit_randomization_test(
    outcome: &[f64], units: &[String], periods: &[i64], treated_unit: &str,
    intervention_period: i64,
) -> Result<SyntheticUnitRandomizationTest, String> {
    fit_synthetic_control(outcome, units, periods, treated_unit, intervention_period)?;
    let mut panel: BTreeMap<&str, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut period_set = BTreeSet::new();
    for ((unit, period), value) in units.iter().zip(periods).zip(outcome) {
        panel.entry(unit).or_default().insert(*period, *value);
        period_set.insert(*period);
    }
    if panel.len() > 32 {
        return Err("exact synthetic unit randomization currently supports at most 32 candidate units".into());
    }
    let pre: Vec<i64> = period_set.iter().copied().filter(|period| *period < intervention_period).collect();
    let post: Vec<i64> = period_set.iter().copied().filter(|period| *period >= intervention_period).collect();
    let mut statistics = Vec::with_capacity(panel.len());
    for (candidate, values) in &panel {
        let donors: Vec<_> = panel.iter().filter(|(unit, _)| *unit != candidate).collect();
        let target_pre: Vec<f64> = pre.iter().map(|period| values[period]).collect();
        let donor_pre: Vec<Vec<f64>> = donors.iter().map(|(_, values)| pre.iter().map(|period| values[period]).collect()).collect();
        let (weights, _) = fit_weights(&target_pre, &donor_pre);
        let treated_post = post.iter().map(|period| values[period]).sum::<f64>() / post.len() as f64;
        let donor_post: f64 = donors.iter().zip(&weights).map(|((_, values), weight)| {
            *weight * post.iter().map(|period| values[period]).sum::<f64>() / post.len() as f64
        }).sum();
        statistics.push(((*candidate).to_string(), (treated_post - donor_post).abs()));
    }
    let observed = statistics.iter().find(|(unit, _)| unit == treated_unit)
        .ok_or_else(|| "treated unit is absent from exact randomization distribution".to_string())?.1;
    let extreme = statistics.iter().filter(|(_, statistic)| *statistic >= observed).count();
    Ok(SyntheticUnitRandomizationTest {
        p_value: extreme as f64 / statistics.len() as f64,
        statistics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_convex_donor_truth_and_support_refusal() {
        let donors = [
            ("a", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            ("b", [0.0, 1.0, 0.0, 1.0, 2.0, 2.0]),
            ("c", [1.0, 0.0, 0.0, 1.0, 0.0, 2.0]),
        ];
        let mut y = Vec::new();
        let mut units = Vec::new();
        let mut periods = Vec::new();
        for (unit, values) in donors {
            for (i, value) in values.into_iter().enumerate() {
                units.push(unit.to_string());
                periods.push(i as i64 + 1);
                y.push(value);
            }
        }
        for i in 0..6 {
            units.push("treated".into());
            periods.push(i as i64 + 1);
            let value = 0.2 * donors[0].1[i] + 0.3 * donors[1].1[i] + 0.5 * donors[2].1[i];
            y.push(value + if i >= 4 { 5.0 } else { 0.0 });
        }
        let fit = fit_synthetic_control(&y, &units, &periods, "treated", 5).unwrap();
        assert!((fit.effect - 5.0).abs() < 1e-5);
        assert!(fit.pre_treatment_rmse < 1e-5);
        assert_eq!(fit.n_pre_periods, 4);
        assert_eq!(fit.n_post_periods, 2);
        assert_eq!(fit.donor_weights.len(), 3);
        assert_eq!(fit.placebo_effects.len(), 3);
        let too_few = fit_synthetic_control(&y[..18], &units[..18], &periods[..18], "a", 5);
        assert!(too_few.unwrap_err().contains("at least three donor"));
    }
}

/// Point-only balanced-panel synthetic difference-in-differences fit.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticDidFit {
    /// Post-intervention difference-in-differences contrast.
    pub effect: f64,
    /// Pre-period treated-versus-synthetic root mean square gap.
    pub pre_treatment_rmse: f64,
    /// Convex donor weights.
    pub donor_weights: Vec<(String, f64)>,
    /// Convex pre-period weights.
    pub time_weights: Vec<(i64, f64)>,
    /// Number of donor units.
    pub n_donors: usize,
    /// Number of pre-periods.
    pub n_pre_periods: usize,
    /// Number of post-periods.
    pub n_post_periods: usize,
}

/// Fit unit and time simplex weights without claiming interval calibration.
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
            return Err(String::from(
                "each unit must have exactly one outcome per period",
            ));
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
    let (unit_weights, pre_rmse) = fit_weights(&target_pre, &donor_pre);

    // SDID time weights solve the dual simplex problem: represent the average
    // pre-period donor profile as a convex combination of individual periods.
    let target_donor_mean: Vec<f64> = donors
        .iter()
        .map(|(_, observed)| {
            pre.iter().map(|period| observed[period]).sum::<f64>() / pre.len() as f64
        })
        .collect();
    let period_profiles: Vec<Vec<f64>> = pre
        .iter()
        .map(|period| donors.iter().map(|(_, observed)| observed[period]).collect())
        .collect();
    let (time_weights, _) = fit_weights(&target_donor_mean, &period_profiles);

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
    if !estimate.is_finite() || !pre_rmse.is_finite() {
        return Err(String::from("synthetic DiD estimate overflowed finite precision"));
    }
    let weights = donors
        .iter()
        .zip(unit_weights)
        .map(|((unit, _), weight)| ((*unit).clone(), weight))
        .collect();
    let time_weights = pre.into_iter().zip(time_weights).collect();
    Ok(SyntheticDidFit { effect: estimate, pre_treatment_rmse: pre_rmse, donor_weights: weights, time_weights, n_donors: donors.len(), n_pre_periods: n_pre, n_post_periods: n_post })
}

#[cfg(test)]
mod synthetic_did_tests {
    use super::fit_synthetic_did;

    #[test]
    fn additive_unit_and_time_effects_recover_known_treatment_and_refuse_sparse_pre_support() {
        let units: Vec<String> = ["treated", "d0", "d1", "d2"]
            .into_iter().flat_map(|unit| std::iter::repeat_n(unit.to_string(), 4)).collect();
        let periods: Vec<i64> = (0..4).flat_map(|_| 1..=4).collect();
        let outcome: Vec<f64> = units.iter().zip(&periods).map(|(unit, period)| {
            let baseline = match unit.as_str() { "treated" | "d1" => 10.0, "d0" => 2.0, _ => 18.0 };
            let common = match period { 1 => 1.0, 2 => 3.0, 3 => -2.0, _ => 5.0 };
            baseline + common + if unit == "treated" && *period == 4 { 7.0 } else { 0.0 }
        }).collect();
        let fit = fit_synthetic_did(&outcome, &units, &periods, "treated", 4).unwrap();
        assert!((fit.effect - 7.0).abs() < 1e-8);
        assert_eq!((fit.n_donors, fit.n_pre_periods, fit.n_post_periods), (3, 3, 1));
        assert!((fit.donor_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() < 1e-8);
        assert!((fit.time_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() < 1e-8);
        assert!(fit_synthetic_did(&outcome, &units, &periods, "treated", 2)
            .unwrap_err().contains("at least two pre-periods"));
    }
}
