//! Conservative score inference for independently randomized action contrasts.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Minimum independently assigned rows for a pointwise action-contrast interval.
pub const MIN_ROWS_FOR_INTERVAL: usize = 400;
/// Minimum observed rows in each contrasted action for an interval.
pub const MIN_ACTION_ROWS_FOR_INTERVAL: usize = 30;
/// Minimum declared probability for either contrasted action at every row.
pub const MIN_ACTION_PROBABILITY_FOR_INTERVAL: f64 = 0.2;
const NORMAL_95: f64 = 1.959_963_984_540_054;

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
        if !outcomes[i].is_finite() || row.len() != arms || assignments[i] >= arms
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
    if reference_support == 0 || action_support == 0 { return None; }
    let effect = scores.iter().sum::<f64>() / n as f64;
    let variance_upper_bound = scores.iter().map(|score| (score - effect).powi(2)).sum::<f64>()
        / ((n - 1) * n) as f64;
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
        effect, variance_upper_bound, reference_support, action_support, interval_95,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(mut state: u64) -> f64 {
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
        ((state ^ (state >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn study(arms: usize, replicate: usize, n: usize) -> (Vec<f64>, Vec<usize>, Vec<Vec<f64>>, f64) {
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
            let draw = uniform((replicate as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let mut cumulative = 0.0;
            let action = row.iter().position(|p| { cumulative += *p; draw < cumulative })
                .unwrap_or(arms - 1);
            outcomes.push(baseline + if action == 1 { effect }
                else if action >= 2 { 1.1 + 0.4 * (action - 2) as f64 }
                else { 0.0 });
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
                    let fit = independent_action_contrast(&outcomes, &assignments, &probabilities, 0, action)
                        .unwrap();
                    let [lower, upper] = fit.interval_95.expect("action is supported");
                    let target = if action == 1 { truth } else { 1.1 + 0.4 * (action - 2) as f64 };
                    covered[action - 1] += usize::from(lower <= target && target <= upper);
                }
            }
            for (index, count) in covered.into_iter().enumerate() {
                let rate = count as f64 / REPLICATES as f64;
                eprintln!("{arms}-arm action {} contrast: {count}/{REPLICATES} = {rate:.4}", index + 1);
                assert!((0.93..=0.985).contains(&rate));
            }
        }
    }

    #[test]
    fn sparse_or_low_probability_action_withholds_interval() {
        let (outcomes, assignments, mut probabilities, _) = study(3, 2, 100);
        assert!(independent_action_contrast(&outcomes, &assignments, &probabilities, 0, 1)
            .unwrap().interval_95.is_none());
        let (outcomes, assignments, _, _) = study(3, 2, 400);
        probabilities = vec![vec![0.89, 0.01, 0.10]; 400];
        assert!(independent_action_contrast(&outcomes, &assignments, &probabilities, 0, 1)
            .unwrap().interval_95.is_none());
        assert!(independent_action_contrast(&outcomes, &assignments, &probabilities, 1, 1).is_none());
    }
}
