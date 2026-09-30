//! Conservative score inference for independently randomized action contrasts.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Minimum independently assigned rows for a pointwise action-contrast interval.
pub const MIN_ROWS_FOR_INTERVAL: usize = 400;
/// Minimum observed rows in each contrasted action for an interval.
pub const MIN_ACTION_ROWS_FOR_INTERVAL: usize = 30;
/// Minimum declared probability for either contrasted action at every row.
pub const MIN_ACTION_PROBABILITY_FOR_INTERVAL: f64 = 0.2;
use antecedent_stats::NORMAL_Q975 as NORMAL_95;

/// Horvitz–Thompson effect and its conservative independent-row score variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndependentActionContrast {
    /// Average potential-outcome contrast, action minus reference.
    pub effect: f64,
    /// Sample-score sandwich variance for the effect.
    pub variance_upper_bound: f64,
    /// Number of observed rows assigned to the reference action.
    pub reference_support: usize,
    /// Number of observed rows assigned to the contrasted action.
    pub action_support: usize,
    /// Pointwise normal interval when the calibrated support conditions hold.
    pub interval_95: Option<[f64; 2]>,
}

/// Estimate an action contrast under independent assignment with known probabilities.
///
/// A row score is `I(A=action)Y/p_action − I(A=reference)Y/p_reference`.
/// Its sample variance divided by the row count conservatively estimates the
/// fixed-population assignment variance: between-row effect variation enters
/// the sandwich but not the true randomization variance. The normal interval
/// is withheld for small or weakly supported actions.
#[must_use]
pub fn independent_action_contrast(
    outcomes: &[f64],
    assignments: &[usize],
    probabilities: &[Vec<f64>],
    reference: usize,
    action: usize,
) -> Option<IndependentActionContrast> {
    let n = outcomes.len();
    if n < 2 || assignments.len() != n || probabilities.len() != n || reference == action {
        return None;
    }
    let arms = probabilities.first()?.len();
    if arms < 2 || reference >= arms || action >= arms {
        return None;
    }
    let mut scores = Vec::with_capacity(n);
    let mut reference_support = 0;
    let mut action_support = 0;
    let mut probability_supported = true;
    for i in 0..n {
        let row = &probabilities[i];
        if !outcomes[i].is_finite()
            || row.len() != arms
            || assignments[i] >= arms
            || row.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
            || (row.iter().sum::<f64>() - 1.0).abs() > 1e-8
        {
            return None;
        }
        probability_supported &= row[reference] + 1e-12 >= MIN_ACTION_PROBABILITY_FOR_INTERVAL
            && row[action] + 1e-12 >= MIN_ACTION_PROBABILITY_FOR_INTERVAL;
        let score = if assignments[i] == action {
            action_support += 1;
            outcomes[i] / row[action]
        } else if assignments[i] == reference {
            reference_support += 1;
            -outcomes[i] / row[reference]
        } else {
            0.0
        };
        scores.push(score);
    }
    if reference_support == 0 || action_support == 0 {
        return None;
    }
    let effect = scores.iter().sum::<f64>() / n as f64;
    let variance_upper_bound =
        scores.iter().map(|score| (score - effect).powi(2)).sum::<f64>() / ((n - 1) * n) as f64;
    if !effect.is_finite() || !variance_upper_bound.is_finite() || variance_upper_bound < 0.0 {
        return None;
    }
    let interval_95 = (n >= MIN_ROWS_FOR_INTERVAL
        && reference_support >= MIN_ACTION_ROWS_FOR_INTERVAL
        && action_support >= MIN_ACTION_ROWS_FOR_INTERVAL
        && probability_supported
        && variance_upper_bound > 0.0)
        .then(|| {
            let radius = NORMAL_95 * variance_upper_bound.sqrt();
            [effect - radius, effect + radius]
        });
    Some(IndependentActionContrast {
        effect,
        variance_upper_bound,
        reference_support,
        action_support,
        interval_95,
    })
}

/// Wald complier effect with its independent-unit influence-function variance.
///
/// Under exclusion, monotonicity, and randomized encouragement the ratio of the
/// Horvitz–Thompson intention-to-treat outcome contrast to the receipt first
/// stage estimates the complier average causal effect (CACE/LATE). Under
/// declared one-sided noncompliance every recipient is a complier, so the same
/// ratio estimates the treatment-on-treated effect. The ratio influence
/// function `(outcome_score − effect · receipt_score) / first_stage` gives the
/// independent-unit asymptotic variance; the normal interval is withheld for
/// small or weakly supported designs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplierWaldEffect {
    /// Wald complier contrast (CACE/LATE, or one-sided treatment-on-treated).
    pub effect: f64,
    /// Horvitz–Thompson intention-to-treat outcome contrast (the ratio numerator).
    pub intention_to_treat_effect: f64,
    /// Receipt first stage (the ratio denominator).
    pub first_stage: f64,
    /// Independent-unit influence-function variance of the Wald contrast.
    pub variance: f64,
    /// Pointwise normal interval when the calibrated support conditions hold.
    pub interval_95: Option<[f64; 2]>,
}

/// Estimate the Wald complier effect from randomized encouragement and receipt.
///
/// `probabilities` are the known per-row treatment-assignment probabilities.
/// Returns `None` when the rows are misaligned, an outcome or probability is
/// out of range, or the receipt first stage is not finite and positive. The
/// interval matches the retained analyze route's published interval exactly;
/// the support license itself is granted only on that route.
#[must_use]
pub fn complier_wald_effect(
    outcomes: &[f64],
    assignment: &[bool],
    received: &[bool],
    probabilities: &[f64],
) -> Option<ComplierWaldEffect> {
    let n = outcomes.len();
    if n < 2 || assignment.len() != n || received.len() != n || probabilities.len() != n {
        return None;
    }
    let mut outcome_scores = Vec::with_capacity(n);
    let mut receipt_scores = Vec::with_capacity(n);
    let mut treated = 0;
    let mut control = 0;
    let mut min_probability = f64::INFINITY;
    for i in 0..n {
        let p = probabilities[i];
        if !outcomes[i].is_finite() || !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return None;
        }
        min_probability = min_probability.min(p.min(1.0 - p));
        let sign = if assignment[i] {
            treated += 1;
            1.0 / p
        } else {
            control += 1;
            -1.0 / (1.0 - p)
        };
        outcome_scores.push(sign * outcomes[i]);
        receipt_scores.push(sign * f64::from(received[i]));
    }
    let intention_to_treat_effect = outcome_scores.iter().sum::<f64>() / n as f64;
    let first_stage = receipt_scores.iter().sum::<f64>() / n as f64;
    if !first_stage.is_finite() || first_stage <= f64::EPSILON {
        return None;
    }
    let effect = intention_to_treat_effect / first_stage;
    let influence = outcome_scores
        .iter()
        .zip(&receipt_scores)
        .map(|(outcome, receipt)| (outcome - effect * receipt) / first_stage)
        .collect::<Vec<_>>();
    let mean = influence.iter().sum::<f64>() / n as f64;
    let variance =
        influence.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / ((n - 1) * n) as f64;
    if !effect.is_finite() || !variance.is_finite() || variance < 0.0 {
        return None;
    }
    let interval_95 = (n >= MIN_ROWS_FOR_INTERVAL
        && control >= MIN_ACTION_ROWS_FOR_INTERVAL
        && treated >= MIN_ACTION_ROWS_FOR_INTERVAL
        && min_probability + 1e-12 >= MIN_ACTION_PROBABILITY_FOR_INTERVAL
        && variance > 0.0)
        .then(|| {
            let radius = NORMAL_95 * variance.sqrt();
            [effect - radius, effect + radius]
        });
    Some(ComplierWaldEffect {
        effect,
        intention_to_treat_effect,
        first_stage,
        variance,
        interval_95,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(state: u64) -> f64 {
        crate::splitmix::unit_f64(crate::splitmix::mix64(state))
    }

    fn study(
        arms: usize,
        replicate: usize,
        n: usize,
    ) -> (Vec<f64>, Vec<usize>, Vec<Vec<f64>>, f64) {
        let mut outcomes = Vec::with_capacity(n);
        let mut assignments = Vec::with_capacity(n);
        let mut probabilities = Vec::with_capacity(n);
        let mut truth = 0.0;
        for i in 0..n {
            let x = i as f64;
            let baseline = 1.0 + 0.55 * (0.31 * x).sin();
            let effect = 2.0 + 0.25 * (0.23 * x).cos();
            truth += effect / n as f64;
            let row = if arms == 2 {
                let p = 0.2 + 0.6 * (i % 7) as f64 / 6.0;
                vec![1.0 - p, p]
            } else if arms == 3 {
                vec![0.3, 0.3, 0.4]
            } else {
                vec![1.0 / arms as f64; arms]
            };
            let draw = uniform(
                (replicate as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03),
            );
            let mut cumulative = 0.0;
            let action = row
                .iter()
                .position(|p| {
                    cumulative += *p;
                    draw < cumulative
                })
                .unwrap_or(arms - 1);
            outcomes.push(
                baseline
                    + if action == 1 {
                        effect
                    } else if action >= 2 {
                        1.1 + 0.4 * (action - 2) as f64
                    } else {
                        0.0
                    },
            );
            assignments.push(action);
            probabilities.push(row);
        }
        (outcomes, assignments, probabilities, truth)
    }

    #[test]
    fn independent_bernoulli_and_multi_arm_coverage() {
        const REPLICATES: usize = 2_000;
        for arms in [2, 3, 4, 5] {
            let mut covered = vec![0_usize; arms - 1];
            for rep in 0..REPLICATES {
                let (outcomes, assignments, probabilities, truth) = study(arms, rep, 400);
                for action in 1..arms {
                    let fit = independent_action_contrast(
                        &outcomes,
                        &assignments,
                        &probabilities,
                        0,
                        action,
                    )
                    .unwrap();
                    let [lower, upper] = fit.interval_95.expect("action is supported");
                    let target = if action == 1 { truth } else { 1.1 + 0.4 * (action - 2) as f64 };
                    covered[action - 1] += usize::from(lower <= target && target <= upper);
                }
            }
            for (index, count) in covered.into_iter().enumerate() {
                let rate = count as f64 / REPLICATES as f64;
                eprintln!(
                    "{arms}-arm action {} contrast: {count}/{REPLICATES} = {rate:.4}",
                    index + 1
                );
                assert!((0.93..=0.985).contains(&rate));
            }
        }
    }

    #[test]
    fn sparse_or_low_probability_action_withholds_interval() {
        let (outcomes, assignments, mut probabilities, _) = study(3, 2, 100);
        assert!(
            independent_action_contrast(&outcomes, &assignments, &probabilities, 0, 1)
                .unwrap()
                .interval_95
                .is_none()
        );
        let (outcomes, assignments, _, _) = study(3, 2, 400);
        probabilities = vec![vec![0.89, 0.01, 0.10]; 400];
        assert!(
            independent_action_contrast(&outcomes, &assignments, &probabilities, 0, 1)
                .unwrap()
                .interval_95
                .is_none()
        );
        assert!(
            independent_action_contrast(&outcomes, &assignments, &probabilities, 1, 1).is_none()
        );
    }

    #[test]
    fn fixed_preassignment_cuped_score_covers_original_itt() {
        const REPLICATES: usize = 2_000;
        const N: usize = 400;
        let coefficient = 1.25;
        for probability_case in ["low_boundary", "middle", "high_boundary", "heterogeneous"] {
            let mut covered = 0;
            for rep in 0..REPLICATES {
                let mut adjusted = Vec::with_capacity(N);
                let mut assignment = Vec::with_capacity(N);
                let mut probabilities = Vec::with_capacity(N);
                let mut truth = 0.0;
                for i in 0..N {
                    let x = (0.19 * i as f64).sin();
                    let effect = 1.8 + 0.25 * (0.13 * i as f64).cos();
                    truth += effect / N as f64;
                    let p = match probability_case {
                        "low_boundary" => 0.2,
                        "middle" => 0.5,
                        "high_boundary" => 0.8,
                        _ => 0.2 + 0.6 * (i % 7) as f64 / 6.0,
                    };
                    let treated = uniform(
                        (rep as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03),
                    ) < p;
                    let outcome = 2.0
                        + coefficient * x
                        + 0.4 * (0.29 * i as f64).sin()
                        + f64::from(treated) * effect;
                    adjusted.push(outcome - coefficient * x);
                    assignment.push(usize::from(treated));
                    probabilities.push(vec![1.0 - p, p]);
                }
                let fit = independent_action_contrast(&adjusted, &assignment, &probabilities, 0, 1)
                    .unwrap();
                let [lower, upper] = fit.interval_95.expect("calibrated Bernoulli support");
                covered += usize::from(lower <= truth && truth <= upper);
            }
            let rate = covered as f64 / REPLICATES as f64;
            eprintln!(
                "fixed CUPED Bernoulli {probability_case} coverage: {covered}/{REPLICATES} = {rate:.4}"
            );
            assert!((0.93..=0.985).contains(&rate));
        }
    }

    fn standard_normal(u1: f64, u2: f64) -> f64 {
        let radius = (-2.0 * u1.max(1e-12).ln()).sqrt();
        radius * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// One randomized-encouragement study with known complier effect.
    ///
    /// Types are drawn under monotonicity: compliers take treatment iff
    /// encouraged, always-takers always, never-takers never (and no
    /// always-takers under one-sidedness). Outcomes satisfy exclusion — the
    /// encouragement enters only through receipt — so the Wald ratio targets
    /// the constant complier effect `tau_c`.
    fn complier_study(
        rep: usize,
        n: usize,
        complier_share: f64,
        always_taker_share: f64,
        tau_c: f64,
        one_sided: bool,
    ) -> (Vec<f64>, Vec<bool>, Vec<bool>, Vec<f64>) {
        let draw = |i: usize, salt: u64| {
            uniform(
                (rep as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)
                    ^ salt.wrapping_mul(0xA076_1D64_78BD_642F),
            )
        };
        let mut outcomes = Vec::with_capacity(n);
        let mut assignment = Vec::with_capacity(n);
        let mut received = Vec::with_capacity(n);
        let mut probabilities = Vec::with_capacity(n);
        for i in 0..n {
            let p = 0.3 + 0.3 * (i % 7) as f64 / 6.0;
            let assigned = draw(i, 1) < p;
            let type_draw = draw(i, 2);
            let is_complier = type_draw < complier_share;
            let is_always = !one_sided
                && type_draw >= complier_share
                && type_draw < complier_share + always_taker_share;
            let receipt = if is_always {
                true
            } else if is_complier {
                assigned
            } else {
                false
            };
            let base = 1.0 + 0.5 * (0.3 * i as f64).sin();
            let noise = standard_normal(draw(i, 3), draw(i, 4));
            let level = if is_always {
                0.7
            } else if is_complier {
                0.0
            } else {
                -0.3
            };
            let outcome =
                base + level + noise + if is_complier { tau_c * f64::from(receipt) } else { 0.0 };
            outcomes.push(outcome);
            assignment.push(assigned);
            received.push(receipt);
            probabilities.push(p);
        }
        (outcomes, assignment, received, probabilities)
    }

    #[test]
    fn complier_wald_influence_interval_covers_known_truth() {
        const REPLICATES: usize = 2_000;
        const N: usize = 1_200;
        let truth = 1.5;
        let mut covered = 0;
        for rep in 0..REPLICATES {
            let (outcomes, assignment, received, probabilities) =
                complier_study(rep, N, 0.5, 0.25, truth, false);
            let fit = complier_wald_effect(&outcomes, &assignment, &received, &probabilities)
                .expect("finite positive first stage under strong compliance");
            let [lower, upper] = fit.interval_95.expect("calibrated complier support");
            covered += usize::from(lower <= truth && truth <= upper);
        }
        let rate = covered as f64 / REPLICATES as f64;
        eprintln!("CACE Wald influence interval coverage: {covered}/{REPLICATES} = {rate:.4}");
        assert!((0.93..=0.985).contains(&rate));
    }

    #[test]
    fn one_sided_treatment_on_treated_influence_interval_covers_known_truth() {
        const REPLICATES: usize = 2_000;
        const N: usize = 1_200;
        let truth = 1.5;
        let mut covered = 0;
        for rep in 0..REPLICATES {
            let (outcomes, assignment, received, probabilities) =
                complier_study(rep, N, 0.6, 0.0, truth, true);
            assert!(
                received.iter().zip(&assignment).all(|(got, assigned)| !*got || *assigned),
                "one-sided noncompliance: no control-arm receipt"
            );
            let fit = complier_wald_effect(&outcomes, &assignment, &received, &probabilities)
                .expect("finite positive first stage under one-sided encouragement");
            let [lower, upper] = fit.interval_95.expect("calibrated treatment-on-treated support");
            covered += usize::from(lower <= truth && truth <= upper);
        }
        let rate = covered as f64 / REPLICATES as f64;
        eprintln!(
            "one-sided ToT Wald influence interval coverage: {covered}/{REPLICATES} = {rate:.4}"
        );
        assert!((0.93..=0.985).contains(&rate));
    }

    #[test]
    fn complier_effect_withholds_interval_without_support() {
        let (outcomes, assignment, received, probabilities) =
            complier_study(7, 200, 0.5, 0.25, 1.5, false);
        assert!(
            complier_wald_effect(&outcomes, &assignment, &received, &probabilities)
                .unwrap()
                .interval_95
                .is_none()
        );
    }
}
