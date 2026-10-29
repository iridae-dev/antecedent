use super::*;
use crate::bayesian_rd::{RdSample, fit_bayesian_sharp_rd};

const DRAWS: usize = 40_000;

fn treatment_noise(i: usize) -> f64 {
    0.2 * (((i * 13) % 17) as f64 / 8.0 - 1.0)
}

fn outcome_noise(i: usize) -> f64 {
    0.2 * (((i * 7) % 11) as f64 / 5.0 - 1.0)
}

/// Four hundred running-variable values on [-2, 2). The treatment follows the named law and the
/// outcome is `1 + 0.8 R + 1.5 T + noise`, so the true ratio is 1.5 for the jump and the kink.
fn fixture(law: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let running: Vec<f64> = (0..400).map(|i| (f64::from(i) - 200.0) / 100.0).collect();
    let treatment: Vec<f64> = running
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let a = f64::from(r >= 0.0);
            let base = match law {
                "fuzzy" => 0.2 + 0.6 * a + 0.1 * r,
                "kink" => 1.0 + 0.5 * r + 0.7 * a * r,
                _ => 0.5 + 0.1 * r,
            };
            base + treatment_noise(i)
        })
        .collect();
    let outcome: Vec<f64> = running
        .iter()
        .zip(&treatment)
        .enumerate()
        .map(|(i, (&r, &t))| 1.0 + 0.8 * r + 1.5 * t + outcome_noise(i))
        .collect();
    (running, treatment, outcome)
}

fn spec(kind: LocalRatioKind) -> LocalRatioSpec {
    LocalRatioSpec { kind, cutoff: 0.0, bandwidth: 1.0, prior_sd: 20.0, first_stage_floor: 0.1 }
}

#[test]
fn fuzzy_jump_posterior_matches_the_independent_bivariate_normal_ratio() {
    let (running, treatment, outcome) = fixture("fuzzy");
    let sample = LocalRatioSample { running: &running, treatment: &treatment, outcome: &outcome };
    let fit = fit_bayesian_local_ratio(
        &sample,
        &spec(LocalRatioKind::FuzzyJump),
        DrawSpec::new(DRAWS, 11),
    )
    .unwrap();
    // Reference: the closed-form coefficient posterior of the 201-row window, with the ratio's
    // quantiles from four million bivariate Normal draws computed outside the implementation.
    assert_eq!(fit.n_window, 201);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    assert!((mean(&fit.reduced_form_draws) - 0.913_768_53).abs() < 0.005);
    assert!((fit.first_stage_mean - 0.603_260_52).abs() < 0.004);
    assert!((fit.first_stage_sd - 0.034_774_46).abs() < 0.002);
    assert!((fit.ratio_median - 1.514_672_64).abs() < 0.02, "{}", fit.ratio_median);
    let [lower, upper] = fit.ratio_interval.expect("a strong first stage keeps its interval");
    assert!((lower - 1.397_139).abs() < 0.03, "{lower}");
    assert!((upper - 1.632_842_03).abs() < 0.03, "{upper}");
    assert!(fit.first_stage_near_zero_mass < 1e-9 && !fit.weak_first_stage);
    assert!(fit.first_stage_positive_probability > 0.999);
    // The numerator is the sharp-RD posterior of the outcome jump on the same design and prior.
    let assignment: Vec<f64> = running.iter().map(|r| f64::from(*r >= 0.0)).collect();
    let sharp = fit_bayesian_sharp_rd(
        &RdSample { running: &running, treatment: &assignment, outcome: &outcome },
        0.0,
        1.0,
        20.0,
        DrawSpec::new(DRAWS, 12),
    )
    .unwrap();
    assert!((sharp.mean - mean(&fit.reduced_form_draws)).abs() < 0.006);
    // Paired draws: the ratio draw is exactly the paired quotient.
    for d in [0, 17, DRAWS - 1] {
        let quotient = fit.reduced_form_draws[d] / fit.first_stage_draws[d];
        assert!((fit.ratio_draws[d] - quotient).abs() < 1e-12);
    }
}

#[test]
fn kink_posterior_uses_the_slope_change_coefficients() {
    let (running, treatment, outcome) = fixture("kink");
    let sample = LocalRatioSample { running: &running, treatment: &treatment, outcome: &outcome };
    let fit =
        fit_bayesian_local_ratio(&sample, &spec(LocalRatioKind::Kink), DrawSpec::new(DRAWS, 13))
            .unwrap();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    assert!((mean(&fit.reduced_form_draws) - 1.046_382_5).abs() < 0.008);
    assert!((fit.first_stage_mean - 0.689_259_76).abs() < 0.006);
    assert!((fit.first_stage_sd - 0.059_916_79).abs() < 0.003);
    assert!((fit.ratio_median - 1.518_062_16).abs() < 0.03, "{}", fit.ratio_median);
    let [lower, upper] = fit.ratio_interval.expect("a strong kink keeps its interval");
    assert!((lower - 1.339_655_67).abs() < 0.04, "{lower}");
    assert!((upper - 1.698_099_49).abs() < 0.04, "{upper}");
    // The jump coefficient of the same data is a different, near-zero quantity; reading it
    // instead of the slope change would not reproduce these values.
    let jump = fit_bayesian_local_ratio(
        &sample,
        &spec(LocalRatioKind::FuzzyJump),
        DrawSpec::new(DRAWS, 13),
    )
    .unwrap();
    assert!((jump.first_stage_mean - fit.first_stage_mean).abs() > 0.1);
}

#[test]
fn a_denominator_with_mass_near_zero_withholds_the_interval_and_reports_the_mass() {
    let (running, treatment, outcome) = fixture("weak");
    let sample = LocalRatioSample { running: &running, treatment: &treatment, outcome: &outcome };
    let fit = fit_bayesian_local_ratio(
        &sample,
        &spec(LocalRatioKind::FuzzyJump),
        DrawSpec::new(DRAWS, 14),
    )
    .unwrap();
    assert!(fit.weak_first_stage);
    assert!(fit.ratio_interval.is_none());
    assert!(
        (fit.first_stage_near_zero_mass - 0.995_763).abs() < 0.01,
        "{}",
        fit.first_stage_near_zero_mass
    );
    assert!((fit.first_stage_positive_probability - 0.538_859).abs() < 0.02);
    // The ratio draws still exist and are visibly unbounded rather than hidden.
    assert!(fit.ratio_draws.iter().any(|r| r.abs() > 50.0));
}

#[test]
fn thin_windows_and_invalid_inputs_are_refused() {
    let (running, treatment, outcome) = fixture("fuzzy");
    let sample = LocalRatioSample { running: &running, treatment: &treatment, outcome: &outcome };
    let narrow = LocalRatioSpec { bandwidth: 0.01, ..spec(LocalRatioKind::FuzzyJump) };
    let error =
        fit_bayesian_local_ratio(&sample, &narrow, DrawSpec::new(100, 1)).unwrap_err().to_string();
    assert!(error.contains("local_ratio_insufficient_window_support"), "{error}");
    let mut broken = outcome.clone();
    broken[5] = f64::INFINITY;
    let bad = LocalRatioSample { running: &running, treatment: &treatment, outcome: &broken };
    let non_finite =
        fit_bayesian_local_ratio(&bad, &spec(LocalRatioKind::FuzzyJump), DrawSpec::new(100, 1))
            .unwrap_err()
            .to_string();
    let no_floor = LocalRatioSpec { first_stage_floor: 0.0, ..spec(LocalRatioKind::FuzzyJump) };
    let floorless = fit_bayesian_local_ratio(&sample, &no_floor, DrawSpec::new(100, 1))
        .unwrap_err()
        .to_string();
    // Both are the one typed input refusal naming the finite-vector and positive-floor rules.
    for text in [&non_finite, &floorless] {
        assert!(text.contains("Bayesian local ratio needs equal finite vectors"), "{text}");
        assert!(text.contains("first_stage_floor > 0"), "{text}");
    }
}

#[test]
fn a_constant_treatment_is_flagged_weak_not_divided_through() {
    let (running, _, outcome) = fixture("fuzzy");
    let constant = vec![1.0; running.len()];
    let sample = LocalRatioSample { running: &running, treatment: &constant, outcome: &outcome };
    let fit = fit_bayesian_local_ratio(
        &sample,
        &spec(LocalRatioKind::FuzzyJump),
        DrawSpec::new(2_000, 2),
    )
    .unwrap();
    assert!(fit.weak_first_stage && fit.ratio_interval.is_none());
    assert!(fit.first_stage_near_zero_mass > 0.99);
}
