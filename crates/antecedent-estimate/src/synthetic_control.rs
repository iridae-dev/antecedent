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
