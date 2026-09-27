//! Repeated-sampling evidence for g-formula inference conditional on a fixed known Q law.

use antecedent_estimate::longitudinal_regime::{
    evaluate_g_formula_value, g_formula_fixed_q_pointwise_interval_95,
};

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn uniform(state: &mut u64) -> f64 {
    (next_u64(state) >> 11) as f64 / (1_u64 << 53) as f64
}

#[test]
fn fixed_known_q_subject_intervals_cover_two_period_regime_truth() {
    const SUBJECTS: usize = 400;
    const TRIALS: usize = 2_000;
    let mut state = 0xC3A5_7842_192F_81DD;
    let mut covered = 0;
    for trial in 0..TRIALS {
        let mut predictions = Vec::with_capacity(SUBJECTS * 2);
        for _ in 0..SUBJECTS {
            let baseline = if uniform(&mut state) < 0.5 { -1.0 } else { 1.0 };
            let centered = (0..12).map(|_| uniform(&mut state)).sum::<f64>() - 6.0;
            // The Q law is prespecified and known: no outcome-model fit enters.
            predictions.push(1.0 + 0.2 * baseline + 0.1 * centered);
            predictions.push(2.0 + 0.3 * baseline + 0.15 * centered);
        }
        let actions = vec![true; SUBJECTS * 2];
        let treatment_probability = vec![0.5; SUBJECTS * 2];
        let censoring_probability = vec![1.0; SUBJECTS * 2];
        let summary = evaluate_g_formula_value(
            &predictions, &actions, &treatment_probability, &censoring_probability,
            SUBJECTS, 2, 0.1,
        ).unwrap();
        let (se, bounds) = g_formula_fixed_q_pointwise_interval_95(
            &summary, &predictions, SUBJECTS, 2,
        ).expect("fixed Q, independent subject histories, and support");
        assert!(se > 0.0 && bounds[0] <= summary.value && summary.value <= bounds[1]);
        covered += usize::from(bounds[0] <= 3.0 && 3.0 <= bounds[1]);
        if trial == 0 {
            let mut mismatched = summary;
            mismatched.value += 1.0;
            assert!(g_formula_fixed_q_pointwise_interval_95(
                &mismatched, &predictions, SUBJECTS, 2,
            ).is_none());
            assert!(g_formula_fixed_q_pointwise_interval_95(
                &summary, &predictions[..100], 50, 2,
            ).is_none());
            assert!(g_formula_fixed_q_pointwise_interval_95(
                &summary, &predictions, SUBJECTS, 3,
            ).is_none());
            let constant = vec![1.5; SUBJECTS * 2];
            let constant_summary = evaluate_g_formula_value(
                &constant, &actions, &treatment_probability, &censoring_probability,
                SUBJECTS, 2, 0.1,
            ).unwrap();
            assert!(g_formula_fixed_q_pointwise_interval_95(
                &constant_summary, &constant, SUBJECTS, 2,
            ).is_none());
        }
    }
    eprintln!("fixed-known-Q g-formula pointwise coverage: {covered}/{TRIALS}");
    assert!(covered >= 1_800, "fixed-known-Q g-formula coverage {covered}/{TRIALS}");
}
