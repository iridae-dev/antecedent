//! Independent hand computation of the fixed-known-Q two-period g-formula value and its
//! published independent-subject normal interval.
//!
//! Each subject contributes the sum of its two supplied period rewards. The rewards are built in
//! mirrored pairs so the regime value is exactly 5; the standard error is
//! `sqrt(sum (score - value)^2 / (n (n - 1)))` and the interval is `value +/- 1.959963984540054
//! se`, all written out here from the predictions alone.
//!
//! The repeated-sampling coverage test in `longitudinal_regime.rs` is calibration evidence.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::longitudinal_regime::{
    evaluate_g_formula_value, g_formula_fixed_q_pointwise_interval_95,
};

const Z975: f64 = 1.959_963_984_540_054;

#[test]
fn fixed_known_q_g_formula_value_standard_error_and_interval_match_the_hand_computation() {
    let subjects = 320_usize;
    let mut predictions = Vec::with_capacity(subjects * 2);
    for pair in 0..subjects / 2 {
        let d = 0.25 + 1.75 * ((pair * 37) % 101) as f64 / 101.0;
        let split = 0.3 * ((pair * 11) % 7) as f64;
        // Mirrored subjects: scores 5 + d and 5 - d, so the regime value is exactly 5.
        predictions.extend([2.0 + split, 3.0 + d - split]);
        predictions.extend([2.0 - split, 3.0 - d + split]);
    }
    let actions = vec![true; subjects * 2];
    let summary = evaluate_g_formula_value(
        &predictions,
        &actions,
        &vec![0.5; subjects * 2],
        &vec![0.9; subjects * 2],
        subjects,
        2,
        0.01,
    )
    .unwrap();
    let scores: Vec<f64> = (0..subjects).map(|s| predictions[2 * s] + predictions[2 * s + 1]).collect();
    let n = subjects as f64;
    let value = scores.iter().sum::<f64>() / n;
    assert!((value - 5.0).abs() < 1e-12, "mirrored rewards give the exact regime value");
    assert!((summary.value - value).abs() < 1e-12);
    let se = (scores.iter().map(|s| (s - value).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();
    let (got_se, bounds) =
        g_formula_fixed_q_pointwise_interval_95(&summary, &predictions, subjects, 2)
            .expect("320 subjects with known probabilities publish the interval");
    assert!((got_se - se).abs() < 1e-12 * (1.0 + se));
    assert!((bounds[0] - (value - Z975 * se)).abs() < 1e-11);
    assert!((bounds[1] - (value + Z975 * se)).abs() < 1e-11);
}
