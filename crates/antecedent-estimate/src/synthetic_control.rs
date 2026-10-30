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

/// Ridge outcome-model correction to the simplex synthetic-control contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct AugmentedSyntheticControlFit {
    /// Unadjusted simplex fit, including donor support diagnostics.
    pub control: SyntheticControlFit,
    /// Post-period treated-minus-synthetic gap after the outcome-model correction.
    pub effect: f64,
    /// Donor-trained prediction difference subtracted from the simplex gap.
    pub outcome_model_correction: f64,
    /// Positive ridge penalty used for the donor outcome model.
    pub ridge_penalty: f64,
}

/// Solve the donor ridge system `(G + λI) c = r` (row-major flat `G`, symmetric positive
/// definite by construction) with the shared Cholesky rule.
fn solve_positive_ridge(gram: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let solution = crate::util::solve_spd(gram, rhs, rhs.len())
        .ok_or("augmented synthetic-control ridge system is numerically singular")?;
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("augmented synthetic-control ridge prediction overflowed".into());
    }
    Ok(solution)
}

/// Correct a simplex synthetic-control gap using a donor-trained ridge outcome model.
///
/// Each donor's pre-period trajectory predicts its post-period mean. The centered
/// linear ridge model is fitted only on donors; its treated-minus-weighted-donor
/// prediction difference is subtracted from the original synthetic-control gap.
/// No sampling interval or placebo calibration is implied by this point fit.
pub fn fit_augmented_synthetic_control(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
    ridge_penalty: f64,
) -> Result<AugmentedSyntheticControlFit, String> {
    if !ridge_penalty.is_finite() || ridge_penalty <= 0.0 {
        return Err("augmented synthetic control requires a finite positive ridge penalty".into());
    }
    let control =
        fit_synthetic_control(outcome, units, periods, treated_unit, intervention_period)?;
    let mut panel: BTreeMap<&str, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut period_set = BTreeSet::new();
    for ((unit, period), value) in units.iter().zip(periods).zip(outcome) {
        panel.entry(unit).or_default().insert(*period, *value);
        period_set.insert(*period);
    }
    let pre: Vec<i64> =
        period_set.iter().copied().filter(|period| *period < intervention_period).collect();
    let post: Vec<i64> =
        period_set.iter().copied().filter(|period| *period >= intervention_period).collect();
    let donors: Vec<&str> = control.donor_weights.iter().map(|(unit, _)| unit.as_str()).collect();
    let donor_pre: Vec<Vec<f64>> =
        donors.iter().map(|unit| pre.iter().map(|period| panel[unit][period]).collect()).collect();
    let donor_post: Vec<f64> = donors
        .iter()
        .map(|unit| post.iter().map(|period| panel[unit][period]).sum::<f64>() / post.len() as f64)
        .collect();
    let treated_pre: Vec<f64> = pre.iter().map(|period| panel[treated_unit][period]).collect();
    let n = donors.len();
    let p = pre.len();
    let pre_mean: Vec<f64> = (0..p)
        .map(|period| donor_pre.iter().map(|values| values[period]).sum::<f64>() / n as f64)
        .collect();
    let post_mean = donor_post.iter().sum::<f64>() / n as f64;
    let centered: Vec<Vec<f64>> = donor_pre
        .iter()
        .map(|values| values.iter().zip(&pre_mean).map(|(value, mean)| value - mean).collect())
        .collect();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] =
                centered[i].iter().zip(&centered[j]).map(|(left, right)| left * right).sum::<f64>()
                    / p as f64;
        }
        gram[i * n + i] += ridge_penalty;
    }
    let target: Vec<f64> = donor_post.iter().map(|value| value - post_mean).collect();
    let coefficients = solve_positive_ridge(&gram, &target)?;
    let prediction = |features: &[f64]| -> f64 {
        post_mean
            + coefficients
                .iter()
                .zip(&centered)
                .map(|(coefficient, donor)| {
                    coefficient
                        * donor
                            .iter()
                            .zip(features.iter().zip(&pre_mean))
                            .map(|(donor_value, (value, mean))| donor_value * (value - mean))
                            .sum::<f64>()
                        / p as f64
                })
                .sum::<f64>()
    };
    let donor_prediction: f64 = donor_pre
        .iter()
        .zip(&control.donor_weights)
        .map(|(features, (_, weight))| weight * prediction(features))
        .sum();
    let correction = prediction(&treated_pre) - donor_prediction;
    let effect = control.effect - correction;
    if !effect.is_finite() || !correction.is_finite() {
        return Err("augmented synthetic-control correction overflowed finite precision".into());
    }
    Ok(AugmentedSyntheticControlFit {
        control,
        effect,
        outcome_model_correction: correction,
        ridge_penalty,
    })
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

fn demean(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| value - mean).collect()
}

/// Fit donor weights by projected gradient descent on the simplex.
/// The fixed iteration cap and sorted donor ordering make the result deterministic.
fn fit_weights(target: &[f64], donors: &[Vec<f64>]) -> Result<(Vec<f64>, f64), String> {
    fit_weights_with_limit(target, donors, 10_000)
}

fn fit_weights_with_limit(
    target: &[f64],
    donors: &[Vec<f64>],
    iterations: usize,
) -> Result<(Vec<f64>, f64), String> {
    let n_donors = donors.len();
    let n_pre = target.len();
    if n_donors == 0 || n_pre == 0 || donors.iter().any(|d| d.len() != n_pre) {
        return Err("synthetic control weight inputs have invalid shape".into());
    }
    // Translation by a donor is exact for simplex weights and removes common
    // levels that otherwise dominate the gradient and step size.
    if target.iter().chain(donors.iter().flatten()).any(|value| !value.is_finite()) {
        return Err("synthetic control weight objective is not finite".into());
    }
    // Subtract first to retain representable differences on large common
    // offsets; if a subtraction overflows, halve every operand consistently.
    let half = (0..n_pre).any(|t| {
        !(target[t] - donors[0][t]).is_finite()
            || donors.iter().any(|donor| !(donor[t] - donors[0][t]).is_finite())
    });
    let difference = |value: f64, t: usize| {
        if half { value * 0.5 - donors[0][t] * 0.5 } else { value - donors[0][t] }
    };
    let mut spread = 0.0_f64;
    for t in 0..n_pre {
        spread = spread.max(difference(target[t], t).abs());
        for donor in donors {
            spread = spread.max(difference(donor[t], t).abs());
        }
    }
    let divisor = if spread == 0.0 { 1.0 } else { spread };
    let value = |x: f64, t: usize| difference(x, t) / divisor;
    let frobenius_sq: f64 = donors
        .iter()
        .flat_map(|donor| donor.iter().enumerate())
        .map(|(t, x)| value(*x, t).powi(2))
        .sum();
    let lipschitz = 2.0 * frobenius_sq / n_pre as f64;
    if !lipschitz.is_finite() {
        return Err("synthetic control weight objective is not finite".into());
    }
    let step = if lipschitz > 0.0 { 1.0 / lipschitz } else { 1.0 };
    if !step.is_finite() {
        return Err("synthetic control weight step is not finite".into());
    }
    let mut weights = vec![1.0 / n_donors as f64; n_donors];
    let mut converged = false;
    for _ in 0..iterations {
        let mut gradient = vec![0.0; n_donors];
        for period in 0..n_pre {
            let residual: f64 = donors
                .iter()
                .zip(&weights)
                .map(|(donor, weight)| value(donor[period], period) * *weight)
                .sum::<f64>()
                - value(target[period], period);
            for (index, donor) in donors.iter().enumerate() {
                gradient[index] += 2.0 * value(donor[period], period) * residual / n_pre as f64;
            }
        }
        if gradient.iter().any(|v| !v.is_finite()) {
            return Err("synthetic control weight gradient is not finite".into());
        }
        // A projected-gradient mapping is zero exactly at simplex KKT
        // stationarity. It is independent of the optimization step size.
        let projected = project_simplex(
            &weights
                .iter()
                .zip(&gradient)
                .map(|(weight, gradient)| *weight - *gradient)
                .collect::<Vec<_>>(),
        );
        let stationarity = projected
            .iter()
            .zip(&weights)
            .map(|(next, current)| (*next - *current).abs())
            .fold(0.0_f64, f64::max);
        if stationarity < 1e-12 {
            converged = true;
            break;
        }
        weights = project_simplex(
            &weights
                .iter()
                .zip(&gradient)
                .map(|(weight, gradient)| *weight - step * *gradient)
                .collect::<Vec<_>>(),
        );
    }
    if !converged {
        return Err("synthetic control weight iteration did not meet its stopping criterion".into());
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
    Ok((weights, rmse))
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

/// What a synthetic control reads of an unbalanced panel.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservedWindowSupport {
    /// Pre-intervention periods the treated unit is observed in; the donor weights are fitted
    /// on these periods only.
    pub pre_periods: Vec<i64>,
    /// Post-intervention periods the treated unit is observed in; the effect averages these.
    pub post_periods: Vec<i64>,
    /// Donors observed in every period of the window.
    pub donors_used: usize,
    /// Units other than the treated unit missing at least one period of the window, each with
    /// the number of window periods it lacks. They are not donors.
    pub donors_dropped: Vec<(String, usize)>,
    /// Observed unit-periods no estimate reads: periods outside the window and the rows of
    /// dropped donors.
    pub unread_rows: usize,
}

/// Restrict an unbalanced panel to the treated unit's observed window.
///
/// The window is the set of periods in which the treated unit is observed: its observed
/// pre-intervention periods fit the donor weights and its observed post-intervention periods
/// define the effect. A donor is eligible only if it is observed in every period of the window,
/// so the restricted panel is balanced by construction and nothing is imputed: a donor's
/// missing period is never filled, periods the treated unit lacks are never read, and no
/// unrelated period of one unit is joined to another's. The returned vectors are the rows of
/// the treated unit and eligible donors inside the window, in input order.
///
/// # Errors
///
/// Misaligned or non-finite inputs, a repeated unit-period, an unobserved treated unit, fewer
/// than two observed pre-intervention periods or no observed post-intervention period of the
/// treated unit, or fewer than three donors observed through the whole window. Each support
/// message names the next attempt.
#[allow(
    clippy::type_complexity,
    reason = "the restricted rows and the support report are returned together"
)]
pub fn restrict_to_observed_window(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
) -> Result<(Vec<f64>, Vec<String>, Vec<i64>, ObservedWindowSupport), String> {
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
    for i in 0..n {
        if units[i].is_empty() || periods[i] <= 0 {
            return Err("unit IDs must be non-empty and periods positive".into());
        }
        if panel.entry(units[i].as_str()).or_default().insert(periods[i], outcome[i]).is_some() {
            return Err("each unit must have at most one outcome per period".into());
        }
    }
    let treated = panel.get(treated_unit).ok_or_else(|| {
        "the treated unit is not observed in any period; next attempt: name a unit of the panel"
            .to_string()
    })?;
    let pre: Vec<i64> =
        treated.keys().copied().filter(|period| *period < intervention_period).collect();
    let post: Vec<i64> =
        treated.keys().copied().filter(|period| *period >= intervention_period).collect();
    if pre.len() < 2 || post.is_empty() {
        return Err(format!(
            "the treated unit is observed in {} pre-intervention and {} post-intervention \
             periods; the weights need at least two observed pre-intervention periods and the \
             effect at least one observed post-intervention period. Next attempt: observe the \
             treated unit in more periods around the intervention",
            pre.len(),
            post.len()
        ));
    }
    let window: Vec<i64> = pre.iter().chain(&post).copied().collect();
    let mut kept: BTreeSet<&str> = BTreeSet::new();
    kept.insert(treated_unit);
    let mut dropped = Vec::new();
    for (unit, observed) in &panel {
        if *unit == treated_unit {
            continue;
        }
        let missing = window.iter().filter(|period| !observed.contains_key(*period)).count();
        if missing == 0 {
            kept.insert(unit);
        } else {
            dropped.push(((*unit).to_string(), missing));
        }
    }
    let donors_used = kept.len() - 1;
    if donors_used < 3 {
        return Err(format!(
            "only {donors_used} donors are observed in every one of the treated unit's {} \
             observed periods (at least three are required); {} units miss part of the window. \
             Next attempt: add donors observed throughout the window, or drop the periods only \
             the treated unit has",
            window.len(),
            dropped.len()
        ));
    }
    let in_window: BTreeSet<i64> = window.iter().copied().collect();
    let (mut y, mut u, mut p) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        if kept.contains(units[i].as_str()) && in_window.contains(&periods[i]) {
            y.push(outcome[i]);
            u.push(units[i].clone());
            p.push(periods[i]);
        }
    }
    let unread_rows = n - y.len();
    Ok((
        y,
        u,
        p,
        ObservedWindowSupport {
            pre_periods: pre,
            post_periods: post,
            donors_used,
            donors_dropped: dropped,
            unread_rows,
        },
    ))
}

/// Fit a balanced-panel synthetic control and leave-one-donor-out placebos.
/// No p-value or interval calibration is attached to the returned placebo rank.
// length reflects the estimator's fixed statistical contract; refactor would change behavior
#[allow(
    clippy::too_many_lines,
    reason = "build the panel, fit the donor weights, then the effect and placebo permutation from the same donor set"
)]
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
    let (weights, pre_rmse) = fit_weights(&target_pre, &donor_pre)?;
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
            fit_weights(&donor_pre[placebo_index], &reference_pre)?;
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
            .map(|((unit, _), weight)| ((*(*unit)).to_string(), weight))
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
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
) -> Result<SyntheticUnitRandomizationTest, String> {
    fit_synthetic_control(outcome, units, periods, treated_unit, intervention_period)?;
    let mut panel: BTreeMap<&str, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut period_set = BTreeSet::new();
    for ((unit, period), value) in units.iter().zip(periods).zip(outcome) {
        panel.entry(unit).or_default().insert(*period, *value);
        period_set.insert(*period);
    }
    let pre: Vec<i64> =
        period_set.iter().copied().filter(|period| *period < intervention_period).collect();
    let post: Vec<i64> =
        period_set.iter().copied().filter(|period| *period >= intervention_period).collect();
    let mut statistics = Vec::with_capacity(panel.len());
    for (candidate, values) in &panel {
        let donors: Vec<_> = panel.iter().filter(|(unit, _)| *unit != candidate).collect();
        let target_pre: Vec<f64> = pre.iter().map(|period| values[period]).collect();
        let donor_pre: Vec<Vec<f64>> = donors
            .iter()
            .map(|(_, values)| pre.iter().map(|period| values[period]).collect())
            .collect();
        let (weights, _) = fit_weights(&target_pre, &donor_pre)?;
        let treated_post =
            post.iter().map(|period| values[period]).sum::<f64>() / post.len() as f64;
        let donor_post: f64 = donors
            .iter()
            .zip(&weights)
            .map(|((_, values), weight)| {
                *weight * post.iter().map(|period| values[period]).sum::<f64>() / post.len() as f64
            })
            .sum();
        statistics.push(((*candidate).to_string(), (treated_post - donor_post).abs()));
    }
    let observed = statistics
        .iter()
        .find(|(unit, _)| unit == treated_unit)
        .ok_or_else(|| "treated unit is absent from exact randomization distribution".to_string())?
        .1;
    let extreme = statistics.iter().filter(|(_, statistic)| *statistic >= observed).count();
    Ok(SyntheticUnitRandomizationTest {
        p_value: extreme as f64 / statistics.len() as f64,
        statistics,
    })
}

/// Exact sharp-null Fisher test for the augmented synthetic-control statistic.
///
/// The ridge outcome model and simplex donor fit are recomputed for every
/// candidate treated unit. Reusing the unaugmented randomization distribution
/// would test a different statistic from the reported augmented effect.
pub fn exact_augmented_synthetic_unit_randomization_test(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
    ridge_penalty: f64,
) -> Result<SyntheticUnitRandomizationTest, String> {
    let mut candidates: Vec<&str> = units.iter().map(String::as_str).collect();
    candidates.sort_unstable();
    candidates.dedup();
    let mut statistics = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let fit = fit_augmented_synthetic_control(
            outcome,
            units,
            periods,
            candidate,
            intervention_period,
            ridge_penalty,
        )?;
        statistics.push((candidate.to_string(), fit.effect.abs()));
    }
    let observed = statistics
        .iter()
        .find(|(unit, _)| unit == treated_unit)
        .ok_or_else(|| {
            "treated unit is absent from exact augmented randomization distribution".to_string()
        })?
        .1;
    let extreme = statistics.iter().filter(|(_, statistic)| *statistic >= observed).count();
    Ok(SyntheticUnitRandomizationTest {
        p_value: extreme as f64 / statistics.len() as f64,
        statistics,
    })
}

/// Exact Fisher test of a prespecified constant additive post-treatment effect.
///
/// Under the null, removing `null_effect` from the observed treated unit's
/// post-period outcomes recovers every unit's untreated outcome panel. The
/// selected method is then refitted for every possible uniform one-unit
/// assignment. This tests the *sharp* constant-effect null; it does not give
/// an observational placebo p-value or an effect confidence interval.
pub fn exact_synthetic_constant_effect_test(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    treated_unit: &str,
    intervention_period: i64,
    null_effect: f64,
    method: SyntheticConstantEffectMethod,
) -> Result<SyntheticUnitRandomizationTest, String> {
    if !null_effect.is_finite() {
        return Err("synthetic sharp-null effect must be finite".into());
    }
    if outcome.len() != units.len() || outcome.len() != periods.len() {
        return Err("outcome, unit, and period vectors must have equal length".into());
    }
    let mut untreated = outcome.to_vec();
    let mut treated_post = 0usize;
    for ((value, unit), period) in untreated.iter_mut().zip(units).zip(periods) {
        if unit == treated_unit && *period >= intervention_period {
            *value -= null_effect;
            treated_post += 1;
        }
    }
    if treated_post == 0 {
        return Err("treated unit has no observed post-intervention outcome".into());
    }
    match method {
        SyntheticConstantEffectMethod::Control => exact_synthetic_unit_randomization_test(
            &untreated,
            units,
            periods,
            treated_unit,
            intervention_period,
        ),
        SyntheticConstantEffectMethod::DifferenceInDifferences => {
            exact_synthetic_did_unit_randomization_test(
                &untreated,
                units,
                periods,
                treated_unit,
                intervention_period,
            )
        }
        SyntheticConstantEffectMethod::AugmentedControl { ridge_penalty } => {
            exact_augmented_synthetic_unit_randomization_test(
                &untreated,
                units,
                periods,
                treated_unit,
                intervention_period,
                ridge_penalty,
            )
        }
    }
}

/// Statistic refitted for each candidate assignment under a sharp null.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SyntheticConstantEffectMethod {
    /// Simplex synthetic control.
    Control,
    /// Unit and time weighted synthetic `DiD`.
    DifferenceInDifferences,
    /// Ridge-augmented simplex synthetic control.
    AugmentedControl {
        /// Positive donor outcome-model penalty.
        ridge_penalty: f64,
    },
}

#[cfg(test)]
mod tests {
    #![cfg_attr(
        test,
        allow(
            clippy::cast_possible_wrap,
            reason = "fixtures build small nonnegative period labels as i64"
        )
    )]
    use super::*;
    #[test]
    fn synthetic_weights_refuse_iteration_exhaustion_without_losing_valid_solutions() {
        use super::fit_weights;
        for scale in [1e-10, 1.0, 1e10] {
            let fit = fit_weights(&[scale], &[vec![0.0], vec![scale]]).unwrap();
            assert!(fit.0[0] < 1e-10 && (fit.0[1] - 1.0).abs() < 1e-10);
        }
        let ordinary = fit_weights(&[1.0, 0.5], &[vec![1.0, 0.0], vec![1.0, 1.0]]).unwrap();
        assert!((ordinary.0[0] - 0.5).abs() < 1e-12);
        assert!((ordinary.0[1] - 0.5).abs() < 1e-12);
        assert!(ordinary.1 < 1e-12);
        let donors = vec![vec![1.0, 0.0], vec![1.0, 0.01], vec![1.0, 0.0]];
        let solved = fit_weights(&[1.0, 0.008], &donors).unwrap();
        assert!((solved.0[1] - 0.8).abs() < 1e-9);
        let error = super::fit_weights_with_limit(&[1.0, 0.008], &donors, 0).unwrap_err();
        assert!(error.contains("stopping criterion"));
        let base = 1e12;
        let shifted_target = base + 0.8;
        let shifted = fit_weights(&[shifted_target], &[vec![base], vec![base + 1.0]]).unwrap();
        assert!((shifted.0[1] - (shifted_target - base)).abs() < 1e-9);
        let tiny = f64::from_bits(1);
        let subnormal = fit_weights(&[tiny], &[vec![0.0], vec![2.0 * tiny]]).unwrap();
        assert!((subnormal.0[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn augmented_synthetic_control_corrects_outside_convex_hull_bias() {
        let mut outcome = Vec::new();
        let mut units = Vec::new();
        let mut periods = Vec::new();
        for (unit, position) in [("a", 0.0), ("b", 1.0), ("c", 2.0), ("treated", 3.0)] {
            for period in 1..=4 {
                units.push(unit.to_string());
                periods.push(period);
                outcome.push(if period < 4 {
                    position * period as f64
                } else {
                    4.0 * position + if unit == "treated" { 5.0 } else { 0.0 }
                });
            }
        }
        let fit = fit_augmented_synthetic_control(&outcome, &units, &periods, "treated", 4, 1e-8)
            .unwrap();
        assert!((fit.control.effect - 9.0).abs() < 1e-5);
        assert!((fit.outcome_model_correction - 4.0).abs() < 1e-4);
        assert!((fit.effect - 5.0).abs() < 1e-4);
        assert!(
            fit_augmented_synthetic_control(&outcome, &units, &periods, "treated", 4, 0.0)
                .unwrap_err()
                .contains("positive ridge penalty")
        );
    }

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

    /// Donors a, b, c are exact convex pieces of the treated unit (weights 0.2, 0.3, 0.5) over
    /// periods 1..=6 with a post effect of 5 at periods 5 and 6; donor d is unrelated.
    fn exact_panel() -> (Vec<f64>, Vec<String>, Vec<i64>) {
        let donors = [
            ("a", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            ("b", [0.0, 1.0, 0.0, 1.0, 2.0, 2.0]),
            ("c", [1.0, 0.0, 0.0, 1.0, 0.0, 2.0]),
            ("d", [4.0, 1.0, 3.0, 2.0, 0.0, 5.0]),
        ];
        let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
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
        (y, units, periods)
    }

    #[test]
    fn observed_window_support_fits_only_donors_observed_through_the_treated_window() {
        let (y, units, periods) = exact_panel();
        // The treated unit misses period 2, donor d misses period 3 and donor a misses
        // period 2 (a period the treated unit does not have, so it is never read).
        let gone = |unit: &str, period: i64| (unit.to_string(), period);
        let removed = [gone("treated", 2), gone("d", 3), gone("a", 2)];
        let keep: Vec<usize> =
            (0..y.len()).filter(|i| !removed.contains(&(units[*i].clone(), periods[*i]))).collect();
        let (y, units, periods): (Vec<f64>, Vec<String>, Vec<i64>) = (
            keep.iter().map(|i| y[*i]).collect(),
            keep.iter().map(|i| units[*i].clone()).collect(),
            keep.iter().map(|i| periods[*i]).collect(),
        );
        let (ry, ru, rp, support) =
            restrict_to_observed_window(&y, &units, &periods, "treated", 5).unwrap();
        assert_eq!(support.pre_periods, vec![1, 3, 4]);
        assert_eq!(support.post_periods, vec![5, 6]);
        assert_eq!(support.donors_used, 3);
        assert_eq!(support.donors_dropped, vec![("d".to_string(), 1)]);
        // Donor d's rows and every row at period 2 are unread; nothing was invented.
        assert_eq!(support.unread_rows, y.len() - ry.len());
        assert!(rp.iter().all(|period| *period != 2) && ru.iter().all(|unit| unit != "d"));
        let fit = fit_synthetic_control(&ry, &ru, &rp, "treated", 5).unwrap();
        assert!((fit.effect - 5.0).abs() < 1e-5, "{}", fit.effect);
        assert_eq!((fit.n_pre_periods, fit.n_post_periods), (3, 2));
        let weight =
            |name: &str| fit.donor_weights.iter().find(|(u, _)| u.as_str() == name).unwrap().1;
        assert!((weight("a") - 0.2).abs() < 1e-4);
        assert!((weight("b") - 0.3).abs() < 1e-4);
        assert!((weight("c") - 0.5).abs() < 1e-4);
        let augmented = fit_augmented_synthetic_control(&ry, &ru, &rp, "treated", 5, 1e-8).unwrap();
        assert!((augmented.effect - 5.0).abs() < 1e-4, "{}", augmented.effect);
        // The balanced entry point still refuses the same unbalanced panel.
        assert!(
            fit_synthetic_control(&y, &units, &periods, "treated", 5)
                .unwrap_err()
                .contains("balanced panel")
        );
    }

    #[test]
    fn observed_window_support_refuses_designs_without_enough_support() {
        let (y, units, periods) = exact_panel();
        let without = |units_gone: &[(&str, i64)]| {
            let keep: Vec<usize> = (0..y.len())
                .filter(|i| !units_gone.iter().any(|(u, p)| units[*i] == *u && periods[*i] == *p))
                .collect();
            (
                keep.iter().map(|i| y[*i]).collect::<Vec<f64>>(),
                keep.iter().map(|i| units[*i].clone()).collect::<Vec<String>>(),
                keep.iter().map(|i| periods[*i]).collect::<Vec<i64>>(),
            )
        };
        // One observed pre-intervention period of the treated unit.
        let (a, b, c) = without(&[("treated", 1), ("treated", 2), ("treated", 3)]);
        let error = restrict_to_observed_window(&a, &b, &c, "treated", 5).unwrap_err();
        assert!(error.contains("at least two observed pre-intervention"), "{error}");
        // No observed post-intervention period of the treated unit.
        let (a, b, c) = without(&[("treated", 5), ("treated", 6)]);
        let error = restrict_to_observed_window(&a, &b, &c, "treated", 5).unwrap_err();
        assert!(error.contains("post-intervention"), "{error}");
        // Two of the four donors miss part of the window, leaving two donors.
        let (a, b, c) = without(&[("a", 4), ("d", 6)]);
        let error = restrict_to_observed_window(&a, &b, &c, "treated", 5).unwrap_err();
        assert!(error.contains("only 2 donors") && error.contains("Next attempt"), "{error}");
        // A repeated unit-period is malformed, not unbalanced.
        let mut repeated = (y.clone(), units.clone(), periods.clone());
        repeated.0.push(0.0);
        repeated.1.push("a".into());
        repeated.2.push(1);
        assert!(
            restrict_to_observed_window(&repeated.0, &repeated.1, &repeated.2, "treated", 5)
                .unwrap_err()
                .contains("at most one outcome")
        );
    }

    #[test]
    fn uniform_unit_sharp_null_tail_is_finite_sample_superuniform() {
        let mut outcome = Vec::new();
        let mut units = Vec::new();
        let mut periods = Vec::new();
        for (unit, baseline, trend) in
            [("a", 0.0, 0.1), ("b", 1.0, -0.2), ("c", 2.0, 0.3), ("d", 3.0, -0.1)]
        {
            for period in 1..=6 {
                units.push(unit.to_string());
                periods.push(period);
                outcome.push(baseline + trend * period as f64 + (period % 3) as f64 * 0.25);
            }
        }
        // Under the sharp null this one observed panel is the outcome table for
        // every possible uniformly selected treated unit.
        let assignments = ["a", "b", "c", "d"];
        let tests: Vec<_> = assignments
            .iter()
            .map(|unit| {
                exact_synthetic_unit_randomization_test(&outcome, &units, &periods, unit, 5)
                    .unwrap()
            })
            .collect();
        for test in &tests[1..] {
            assert_eq!(test.statistics, tests[0].statistics);
        }
        for numerator in 1..=assignments.len() {
            let alpha = numerator as f64 / assignments.len() as f64;
            let rejections = tests.iter().filter(|test| test.p_value <= alpha).count();
            assert!(rejections <= numerator);
        }
    }

    #[test]
    fn augmented_uniform_unit_test_refits_the_reported_statistic() {
        let mut outcome = Vec::new();
        let mut units = Vec::new();
        let mut periods = Vec::new();
        for (unit, level, trend) in
            [("a", 0.0, 0.2), ("b", 1.0, -0.1), ("c", 2.0, 0.3), ("d", 3.0, 0.0)]
        {
            for period in 1..=6 {
                units.push(unit.to_string());
                periods.push(period);
                outcome.push(
                    level
                        + trend * period as f64
                        + if period >= 5 && unit == "d" { 2.0 } else { 0.0 },
                );
            }
        }
        let candidates = ["a", "b", "c", "d"];
        let tests: Vec<_> = candidates
            .iter()
            .map(|unit| {
                exact_augmented_synthetic_unit_randomization_test(
                    &outcome, &units, &periods, unit, 5, 0.1,
                )
                .unwrap()
            })
            .collect();
        for test in &tests[1..] {
            assert_eq!(test.statistics, tests[0].statistics);
        }
        for (unit, statistic) in &tests[0].statistics {
            let fit =
                fit_augmented_synthetic_control(&outcome, &units, &periods, unit, 5, 0.1).unwrap();
            assert!((statistic - fit.effect.abs()).abs() < 1e-12);
        }
        for numerator in 1..=candidates.len() {
            let alpha = numerator as f64 / candidates.len() as f64;
            let rejections = tests.iter().filter(|test| test.p_value <= alpha).count();
            assert!(rejections <= numerator);
        }
        assert!(
            exact_augmented_synthetic_unit_randomization_test(
                &outcome, &units, &periods, "absent", 5, 0.1,
            )
            .is_err()
        );
    }

    #[test]
    fn ridge_system_singularity_is_scale_free() {
        // Pivots of 1e-16 sat under the former absolute 1e-14 cutoff of a pre-period panel in
        // small units; the system is a scaled identity and must solve.
        let solution = solve_positive_ridge(&[1e-16, 0.0, 0.0, 4e-16], &[1e-16, 8e-16]).unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-9 && (solution[1] - 2.0).abs() < 1e-9);
        // A rank-one system (ridge underflowed against the Gram) is still refused.
        let e = solve_positive_ridge(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0]).unwrap_err();
        assert!(e.contains("numerically singular"), "{e}");
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
    if !estimate.is_finite() || !pre_rmse.is_finite() {
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
    for candidate in candidates {
        let fit = fit_synthetic_did(outcome, units, periods, candidate, intervention_period)?;
        statistics.push((candidate.to_string(), fit.effect.abs()));
    }
    let observed = statistics
        .iter()
        .find(|(unit, _)| unit == treated_unit)
        .ok_or_else(|| {
            "treated unit is absent from exact synthetic DiD randomization distribution".to_string()
        })?
        .1;
    let extreme = statistics.iter().filter(|(_, statistic)| *statistic >= observed).count();
    Ok(SyntheticUnitRandomizationTest {
        p_value: extreme as f64 / statistics.len() as f64,
        statistics,
    })
}

#[cfg(test)]
mod synthetic_did_tests {
    #![cfg_attr(
        test,
        allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "fixtures map small nonnegative i64 period labels to indices"
        )
    )]
    use super::{exact_synthetic_did_unit_randomization_test, fit_synthetic_did};

    #[test]
    fn time_weights_match_donor_post_periods_rather_than_staying_uniform() {
        // Donors load on a factor that is active only in pre-period 3 and the post period,
        // so the post-period donor means are reproduced by period 3 alone.
        let loadings = [("treated", 1.5), ("d0", 0.0), ("d1", 1.0), ("d2", 2.0), ("d3", 3.0)];
        let factor = [0.0, 0.0, 1.0, 1.0];
        let mut units = Vec::new();
        let mut periods = Vec::new();
        let mut outcome = Vec::new();
        for (index, (unit, loading)) in loadings.into_iter().enumerate() {
            for (period, f) in (1..=4).zip(factor) {
                units.push(unit.to_string());
                periods.push(period);
                let effect = if unit == "treated" && period == 4 { 7.0 } else { 0.0 };
                outcome.push(index as f64 + loading * f + effect);
            }
        }
        let fit = fit_synthetic_did(&outcome, &units, &periods, "treated", 4).unwrap();
        let period_three = fit.time_weights.iter().find(|(period, _)| *period == 3).unwrap().1;
        assert!(period_three > 0.99, "time weights {:?}", fit.time_weights);
        assert!((fit.effect - 7.0).abs() < 1e-6, "effect {}", fit.effect);
    }

    #[test]
    fn additive_unit_and_time_effects_recover_known_treatment_and_refuse_sparse_pre_support() {
        let units: Vec<String> = ["treated", "d0", "d1", "d2"]
            .into_iter()
            .flat_map(|unit| std::iter::repeat_n(unit.to_string(), 4))
            .collect();
        let periods: Vec<i64> = (0..4).flat_map(|_| 1..=4).collect();
        let outcome: Vec<f64> = units
            .iter()
            .zip(&periods)
            .map(|(unit, period)| {
                let baseline = match unit.as_str() {
                    "treated" | "d1" => 10.0,
                    "d0" => 2.0,
                    _ => 18.0,
                };
                let common = match period {
                    1 => 1.0,
                    2 => 3.0,
                    3 => -2.0,
                    _ => 5.0,
                };
                baseline + common + if unit == "treated" && *period == 4 { 7.0 } else { 0.0 }
            })
            .collect();
        let fit = fit_synthetic_did(&outcome, &units, &periods, "treated", 4).unwrap();
        assert!((fit.effect - 7.0).abs() < 1e-8);
        assert_eq!((fit.n_donors, fit.n_pre_periods, fit.n_post_periods), (3, 3, 1));
        assert!(
            (fit.donor_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() < 1e-8
        );
        assert!(
            (fit.time_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() < 1e-8
        );
        assert!(
            fit_synthetic_did(&outcome, &units, &periods, "treated", 2)
                .unwrap_err()
                .contains("at least two pre-periods")
        );
    }

    #[test]
    fn exact_assignment_p_values_are_superuniform_under_sharp_null() {
        let mut units = Vec::new();
        let mut periods = Vec::new();
        let mut outcome = Vec::new();
        for unit in 0..8 {
            for period in 1..=5 {
                units.push(format!("unit-{unit}"));
                periods.push(period);
                outcome.push(
                    unit as f64 * 0.31
                        + period as f64 * 0.17
                        + ((unit * 7 + period as usize * 3) % 11) as f64 * 0.13,
                );
            }
        }
        let fits: Vec<_> = (0..8)
            .map(|unit| {
                exact_synthetic_did_unit_randomization_test(
                    &outcome,
                    &units,
                    &periods,
                    &format!("unit-{unit}"),
                    5,
                )
                .unwrap()
            })
            .collect();
        for fit in &fits[1..] {
            assert_eq!(fit.statistics, fits[0].statistics);
        }
        for alpha in [0.125, 0.25, 0.5] {
            let rejects = fits.iter().filter(|fit| fit.p_value <= alpha).count();
            assert!(rejects as f64 / 8.0 <= alpha);
        }
        assert!(
            exact_synthetic_did_unit_randomization_test(&outcome, &units, &periods, "missing", 5,)
                .is_err()
        );
    }
}

#[cfg(test)]
mod constant_effect_null_tests {
    #![cfg_attr(
        test,
        allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "fixtures map small nonnegative i64 period labels to indices"
        )
    )]
    use super::{
        SyntheticConstantEffectMethod, exact_augmented_synthetic_unit_randomization_test,
        exact_synthetic_constant_effect_test, exact_synthetic_did_unit_randomization_test,
    };

    #[test]
    fn prespecified_nonzero_sharp_null_is_exact_over_2000_assignments() {
        let candidates = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let mut units = Vec::new();
        let mut periods = Vec::new();
        let mut untreated = Vec::new();
        for (index, unit) in candidates.iter().enumerate() {
            for period in 1..=5 {
                units.push((*unit).to_string());
                periods.push(period);
                untreated.push(
                    index as f64 * 0.27
                        + period as f64 * 0.13
                        + ((index * 11 + period as usize * 7) % 13) as f64 * 0.09,
                );
            }
        }
        let null_effect = 2.75;
        let mut p_values = Vec::new();
        for candidate in candidates {
            let observed: Vec<f64> = untreated
                .iter()
                .zip(&units)
                .zip(&periods)
                .map(|((value, unit), period)| {
                    value + if unit == candidate && *period >= 5 { null_effect } else { 0.0 }
                })
                .collect();
            let test = exact_synthetic_constant_effect_test(
                &observed,
                &units,
                &periods,
                candidate,
                5,
                null_effect,
                SyntheticConstantEffectMethod::Control,
            )
            .unwrap();
            p_values.push(test.p_value);
        }
        // Each candidate is selected exactly 250 times, giving 2,000 draws
        // from the declared uniform assignment mechanism on a fixed Y(0) panel.
        for alpha in [0.05, 0.125, 0.25, 0.5] {
            let rejections: usize =
                (0..2_000).filter(|draw| p_values[draw % candidates.len()] <= alpha).count();
            assert!(rejections as f64 / 2_000.0 <= alpha);
        }
        assert!(
            exact_synthetic_constant_effect_test(
                &untreated,
                &units,
                &periods,
                "a",
                5,
                f64::NAN,
                SyntheticConstantEffectMethod::Control,
            )
            .unwrap_err()
            .contains("finite")
        );
    }

    #[test]
    fn nonzero_null_refits_sdid_and_augmented_statistics() {
        let candidates = ["a", "b", "c", "treated"];
        let mut units = Vec::new();
        let mut periods = Vec::new();
        let mut observed = Vec::new();
        for (index, unit) in candidates.iter().enumerate() {
            for period in 1..=4 {
                units.push((*unit).to_string());
                periods.push(period);
                observed.push(
                    index as f64 * 0.25
                        + period as f64 * 0.3
                        + if *unit == "treated" && period == 4 { 3.0 } else { 0.0 },
                );
            }
        }
        for method in [
            SyntheticConstantEffectMethod::DifferenceInDifferences,
            SyntheticConstantEffectMethod::AugmentedControl { ridge_penalty: 0.1 },
        ] {
            let test = exact_synthetic_constant_effect_test(
                &observed, &units, &periods, "treated", 4, 3.0, method,
            )
            .unwrap();
            let untreated: Vec<f64> = observed
                .iter()
                .zip(&units)
                .zip(&periods)
                .map(|((value, unit), period)| {
                    value - if unit == "treated" && *period == 4 { 3.0 } else { 0.0 }
                })
                .collect();
            let expected = match method {
                SyntheticConstantEffectMethod::DifferenceInDifferences => {
                    exact_synthetic_did_unit_randomization_test(
                        &untreated, &units, &periods, "treated", 4,
                    )
                    .unwrap()
                }
                SyntheticConstantEffectMethod::AugmentedControl { ridge_penalty } => {
                    exact_augmented_synthetic_unit_randomization_test(
                        &untreated,
                        &units,
                        &periods,
                        "treated",
                        4,
                        ridge_penalty,
                    )
                    .unwrap()
                }
                SyntheticConstantEffectMethod::Control => unreachable!(),
            };
            assert_eq!(test.statistics.len(), 4);
            assert_eq!(test, expected);
            assert!(test.statistics.iter().all(|(_, statistic)| statistic.is_finite()));
        }
    }
}
