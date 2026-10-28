//! Horvitz-Thompson contrasts and the Wald complier ratio against exact enumeration of
//! independent assignment.
//!
//! Rows have fixed potential outcomes and known, unequal assignment probabilities. Every
//! assignment vector is enumerated (2^10 for two actions, 3^7 for three) and weighted by its
//! probability. The reference computes each estimator from its definition directly, so the
//! product is compared assignment by assignment (its `None` for an empty arm is the only
//! allowed difference), and the design-based identities are then checked on the reference law:
//!
//! * the Horvitz-Thompson contrast is exactly unbiased for the average contrast;
//! * the mean of the sandwich variance equals the exact variance plus
//!   `sum_i (tau_i - mean tau)^2 / (n (n - 1))`, the between-row effect variation the record
//!   describes as the source of conservatism;
//! * the receipt first stage and the intention-to-treat contrast are unbiased, and the Wald
//!   ratio and its influence-function variance equal the delta-method expression written from
//!   the sample covariance of the two score columns.

use antecedent_estimate::randomized_scores::{complier_wald_effect, independent_action_contrast};

const Y: [[f64; 3]; 7] = [
    [0.0, 2.0, 5.0],
    [1.0, 0.5, 4.0],
    [3.0, 3.0, 2.0],
    [2.5, 6.0, 1.0],
    [0.0, -1.0, 0.5],
    [4.0, 4.5, 9.0],
    [1.5, 0.0, 2.0],
];

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-11 * (1.0 + a.abs().max(b.abs())), "{what}: {a} vs {b}");
}

#[test]
fn two_action_contrast_is_unbiased_and_its_sandwich_slack_is_the_effect_variation() {
    let y0 = [0.0, 1.0, 3.0, 2.5, 0.0, 4.0, 1.5, 2.0, 7.0, 0.5];
    let tau = [2.0, -0.5, 0.0, 3.5, -1.0, 0.5, -1.5, 2.5, 4.0, -3.0];
    let p1 = [0.5, 0.3, 0.7, 0.4, 0.5, 0.6, 0.25, 0.5, 0.35, 0.8];
    let n = 10;
    let (mut mean, mut mean_sq, mut mean_v, mut matched) = (0.0, 0.0, 0.0, 0.0);
    for mask in 0..(1_u32 << n) {
        let mut weight = 1.0;
        let mut scores = Vec::new();
        let mut outcomes = Vec::new();
        let mut assignments = Vec::new();
        for i in 0..n {
            let treated = mask >> i & 1 == 1;
            weight *= if treated { p1[i] } else { 1.0 - p1[i] };
            let y1 = y0[i] + tau[i];
            scores.push(if treated { y1 / p1[i] } else { -y0[i] / (1.0 - p1[i]) });
            outcomes.push(if treated { y1 } else { y0[i] });
            assignments.push(usize::from(treated));
        }
        let nf = n as f64;
        let effect = scores.iter().sum::<f64>() / nf;
        let variance =
            scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / (nf * (nf - 1.0));
        mean += weight * effect;
        mean_sq += weight * effect * effect;
        mean_v += weight * variance;
        let table: Vec<Vec<f64>> = p1.iter().map(|&p| vec![1.0 - p, p]).collect();
        match independent_action_contrast(&outcomes, &assignments, table.as_slice(), 0, 1) {
            Some(got) => {
                close(got.effect, effect, "effect");
                close(got.variance_upper_bound, variance, "variance");
                matched += weight;
            }
            None => assert!(mask == 0 || mask == (1 << n) - 1, "only an empty arm is refused"),
        }
    }
    assert!(matched > 0.99);
    close(mean, tau.iter().sum::<f64>() / n as f64, "unbiased");
    let exact_variance = mean_sq - mean * mean;
    let tau_mean = tau.iter().sum::<f64>() / n as f64;
    let slack = tau.iter().map(|t| (t - tau_mean).powi(2)).sum::<f64>() / (n * (n - 1)) as f64;
    close(mean_v, exact_variance + slack, "E[variance] = variance + effect variation");
}

#[test]
fn three_action_contrast_selects_the_right_columns_and_stays_unbiased() {
    let probs: [[f64; 3]; 7] = [
        [0.2, 0.3, 0.5],
        [0.5, 0.25, 0.25],
        [0.1, 0.6, 0.3],
        [0.4, 0.4, 0.2],
        [0.3, 0.3, 0.4],
        [0.25, 0.5, 0.25],
        [0.6, 0.1, 0.3],
    ];
    let table: Vec<Vec<f64>> = probs.iter().map(|p| p.to_vec()).collect();
    for (reference, action) in [(0_usize, 2_usize), (2, 1), (1, 0)] {
        let (mut mean, mut mean_sq, mut mean_v) = (0.0, 0.0, 0.0);
        for code in 0..3_usize.pow(7) {
            let assignments: Vec<usize> = (0..7).map(|i| code / 3_usize.pow(i as u32) % 3).collect();
            let weight: f64 = (0..7).map(|i| probs[i][assignments[i]]).product();
            let outcomes: Vec<f64> = (0..7).map(|i| Y[i][assignments[i]]).collect();
            let scores: Vec<f64> = (0..7)
                .map(|i| {
                    if assignments[i] == action {
                        outcomes[i] / probs[i][action]
                    } else if assignments[i] == reference {
                        -outcomes[i] / probs[i][reference]
                    } else {
                        0.0
                    }
                })
                .collect();
            let effect = scores.iter().sum::<f64>() / 7.0;
            let variance = scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / 42.0;
            mean += weight * effect;
            mean_sq += weight * effect * effect;
            mean_v += weight * variance;
            if let Some(got) =
                independent_action_contrast(&outcomes, &assignments, table.as_slice(), reference, action)
            {
                close(got.effect, effect, "effect");
                close(got.variance_upper_bound, variance, "variance");
                assert_eq!(got.action_support, assignments.iter().filter(|&&a| a == action).count());
                assert_eq!(
                    got.reference_support,
                    assignments.iter().filter(|&&a| a == reference).count()
                );
            }
        }
        let tau: Vec<f64> = (0..7).map(|i| Y[i][action] - Y[i][reference]).collect();
        let tau_mean = tau.iter().sum::<f64>() / 7.0;
        close(mean, tau_mean, "unbiased");
        let slack = tau.iter().map(|t| (t - tau_mean).powi(2)).sum::<f64>() / 42.0;
        close(mean_v, mean_sq - mean * mean + slack, "E[variance] = variance + effect variation");
    }
}

#[test]
fn wald_ratio_components_are_unbiased_and_the_variance_is_the_delta_method() {
    // Potential outcomes under encouragement 0/1 and potential receipts (one-sided
    // noncompliance: the control group never receives treatment).
    let y0 = [1.0, 0.0, 2.0, 3.0, 1.5, 0.5, 2.5, 4.0];
    let y1 = [3.0, 0.0, 2.5, 6.0, 1.5, 2.5, 2.0, 7.0];
    let d1 = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let p = [0.5, 0.4, 0.6, 0.5, 0.3, 0.5, 0.45, 0.55];
    let n = 8;
    let nf = n as f64;
    let (mut itt, mut stage) = (0.0, 0.0);
    for mask in 0..(1_u32 << n) {
        let mut weight = 1.0;
        let (mut outcomes, mut assignment, mut received) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let z = mask >> i & 1 == 1;
            weight *= if z { p[i] } else { 1.0 - p[i] };
            outcomes.push(if z { y1[i] } else { y0[i] });
            assignment.push(z);
            received.push(z && d1[i] == 1.0);
        }
        let ys: Vec<f64> = (0..n)
            .map(|i| if assignment[i] { outcomes[i] / p[i] } else { -outcomes[i] / (1.0 - p[i]) })
            .collect();
        let ds: Vec<f64> = (0..n)
            .map(|i| {
                let r = f64::from(received[i]);
                if assignment[i] { r / p[i] } else { -r / (1.0 - p[i]) }
            })
            .collect();
        let (itt_hat, stage_hat) = (ys.iter().sum::<f64>() / nf, ds.iter().sum::<f64>() / nf);
        itt += weight * itt_hat;
        stage += weight * stage_hat;
        let Some(got) = complier_wald_effect(&outcomes, &assignment, &received, &p) else {
            assert!(stage_hat <= f64::EPSILON, "refusal only for a non-positive first stage");
            continue;
        };
        close(got.intention_to_treat_effect, itt_hat, "itt");
        close(got.first_stage, stage_hat, "first stage");
        let effect = itt_hat / stage_hat;
        close(got.effect, effect, "wald");
        // Delta method from the sample moments of the two score columns.
        let moment = |a: &[f64], b: &[f64]| {
            let (ma, mb) = (a.iter().sum::<f64>() / nf, b.iter().sum::<f64>() / nf);
            a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / (nf - 1.0)
        };
        let variance = (moment(&ys, &ys) - 2.0 * effect * moment(&ys, &ds)
            + effect * effect * moment(&ds, &ds))
            / (nf * stage_hat * stage_hat);
        close(got.variance, variance, "delta-method variance");
    }
    let itt_truth = (0..n).map(|i| y1[i] - y0[i]).sum::<f64>() / nf;
    let stage_truth = d1.iter().sum::<f64>() / nf;
    close(itt, itt_truth, "ITT unbiased");
    close(stage, stage_truth, "first stage unbiased");
}
