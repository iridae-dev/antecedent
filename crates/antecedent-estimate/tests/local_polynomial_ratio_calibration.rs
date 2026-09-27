//! Repeated-sampling evidence for the native fuzzy RD and regression-kink intervals.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;
use antecedent_estimate::local_polynomial_ratio::fit_local_polynomial_ratio;
use antecedent_kernels::standard_normal;

const REPLICATIONS: usize = 2_000;
const TRUTH: f64 = 2.0;

fn coverage(kink: bool) -> (usize, usize) {
    let mut rng = CausalRng::from_seed(if kink { 0x21_09_27_02 } else { 0x21_09_27_01 });
    let running: Vec<f64> = (1..=1_200).map(|index| -1.0 + f64::from(index) / 600.5).collect();
    let mut treatment = vec![0.0; running.len()];
    let mut outcome = vec![0.0; running.len()];
    let mut accepted = 0;
    let mut covered = 0;
    for _ in 0..REPLICATIONS {
        for (index, &x) in running.iter().enumerate() {
            treatment[index] = if kink {
                1.0 + 0.2 * x + 1.5 * x.max(0.0) + 0.1 * standard_normal(&mut rng)
            } else {
                let probability = 0.1 + 0.02 * x + if x > 0.0 { 0.8 } else { 0.0 };
                f64::from(rng.next_f64() < probability)
            };
            let noise_scale = if kink { 0.5 } else { 1.0 };
            outcome[index] = 0.7 + 0.5 * x + 0.3 * x * x + 0.1 * x.powi(3)
                + TRUTH * treatment[index] + noise_scale * standard_normal(&mut rng);
        }
        let bandwidth = if kink { 1.0 } else { 0.5 };
        let fit = match fit_local_polynomial_ratio(
            &running, &outcome, &treatment, 0.0, bandwidth, kink,
        ) {
            Ok(fit) => fit,
            Err(message) if message.contains("weak first stage") => continue,
            Err(message) => panic!("unexpected local-ratio refusal: {message}"),
        };
        accepted += 1;
        covered += usize::from(fit.ci_lower <= TRUTH && TRUTH <= fit.ci_upper);
    }
    (accepted, covered)
}

#[test]
fn fuzzy_rd_bias_corrected_normal_interval_has_nominal_95_coverage() {
    let (accepted, covered) = coverage(false);
    let rate = covered as f64 / accepted as f64;
    eprintln!("fuzzy RD: {covered}/{accepted} = {rate:.4} coverage at nominal 0.95");
    assert!(accepted >= 1_900, "weak-stage refusal rate exceeded the fixture boundary");
    assert!((0.925..=0.975).contains(&rate), "fuzzy RD coverage {rate:.4}");
}

#[test]
fn regression_kink_bias_corrected_normal_interval_has_nominal_95_coverage() {
    let (accepted, covered) = coverage(true);
    let rate = covered as f64 / accepted as f64;
    eprintln!("regression kink: {covered}/{accepted} = {rate:.4} coverage at nominal 0.95");
    assert!(accepted >= 1_900, "weak-stage refusal rate exceeded the fixture boundary");
    assert!((0.925..=0.975).contains(&rate), "regression-kink coverage {rate:.4}");
}
