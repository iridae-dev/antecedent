//! Actual independent source/target OLS discrepancy diagnostics, not invariance certificates.
//! Balanced fixed X±1; source Y=1+X+N(0,1), target same baseline with independent
//! N(0,2.25). Difference covariance is(3.25/n)I. Full coefficient null, slope-only
//! null, partial-null strong slope shift.5, and slope shift equal to the independent
//! normal MDD(1.959964+.841621)*sqrt(3.25/n) exercise actual Wald/Holm and MDD.
//! n256/512/1024 per population,R>=2000. Every failed fit fails. Null Wald size
//! .05±3MCSE, Holm FWER<=.05+3MCSE; strong power>=.9; MDD power.8±3MCSE+.02
//! (explicit finite-n normal approximation margin). Mean SE² and empirical variance
//! relative error<=.15, bias<=5MCSE, mean MDD relative error<=.15. No coverage IDs.
#![allow(clippy::cast_precision_loss, reason = "bounded measurement dimensions")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
use antecedent_estimate::mechanism_discrepancy::{
    DiscrepancyOptions, MechanismDiscrepancyResult, MechanismMeasurement, ParentSpec,
    PopulationSample, SampleDependence, test_mechanism_discrepancy,
};
use serde_json::json;
const DETECTABILITY_FACTOR: f64 = 1.959_963_984_540_054 + 0.841_621_233_572_914_3;
fn fit(n: usize, seed: u64, intercept: bool, delta: f64) -> Option<MechanismDiscrepancyResult> {
    let mut rng = candidate::Generator::new(seed);
    let x: Vec<_> = (0..n).map(|i| if i % 2 == 0 { -1. } else { 1. }).collect();
    let measurement = MechanismMeasurement {
        node: "y".into(),
        node_unit: "unit".into(),
        parents: vec![ParentSpec { name: "x".into(), unit: "unit".into() }],
        protocol_id: "gaussian-fixed-design-v1".into(),
    };
    let source = PopulationSample {
        label: "source".into(),
        measurement: measurement.clone(),
        outcome: x.iter().map(|v| 1. + v + rng.normal()).collect(),
        parent_values: vec![x.clone()],
        unit_ids: vec![],
    };
    let target = PopulationSample {
        label: "target".into(),
        measurement,
        outcome: x.iter().map(|v| 1. + (1. + delta) * v + 1.5 * rng.normal()).collect(),
        parent_values: vec![x],
        unit_ids: vec![],
    };
    test_mechanism_discrepancy(
        &source,
        &target,
        &DiscrepancyOptions {
            compare_intercept: intercept,
            alpha: 0.05,
            power: 0.8,
            dependence: SampleDependence::Independent,
        },
    )
    .ok()
}
fn measure(test: &str, intercept: bool) {
    let n = calibration::grid_n(512);
    let seed = calibration::grid_seed(0xB3DD_0001);
    let r = calibration::n_sim().max(2000);
    let variance = 3.25 / n as f64;
    let factor = DETECTABILITY_FACTOR;
    for (label, delta) in
        [("null", 0.), ("strong_partial_null", 0.5), ("mdd_sized_slope", factor * variance.sqrt())]
    {
        if intercept && label == "mdd_sized_slope" {
            continue;
        }
        let results: Vec<_> =
            (0..r).map(|i| fit(n, seed.wrapping_add(u64::from(i)), intercept, delta)).collect();
        assess(test, label, n, seed, variance, delta, &results);
    }
}
fn assess(
    test: &str,
    label: &str,
    n: usize,
    seed: u64,
    variance: f64,
    delta: f64,
    results: &[Option<MechanismDiscrepancyResult>],
) {
    let factor = DETECTABILITY_FACTOR;
    let failed = results.iter().filter(|f| f.is_none()).count();
    let fits: Vec<_> = results.iter().flatten().collect();
    let invalid = fits
        .iter()
        .filter(|f| {
            f.non_rejection_certifies_invariance
                || f.coefficients.iter().any(|c| {
                    !c.standard_error.is_finite()
                        || c.standard_error <= 0.
                        || !c.p_holm.is_finite()
                        || !(0. ..=1.).contains(&c.p_holm)
                })
        })
        .count();
    if failed + invalid > 0 {
        println!(
            "INFERENCE_DIAGNOSTIC_RECORD {}",
            json!({"test":test,"design":label,"rows_per_population":n,"seed":seed,"attempts":results.len(),"failed":failed,"invalid":invalid,"pass":false})
        );
        panic!("all attempted fits must produce valid original diagnostics");
    }
    let r = results.len() as f64;
    let slack = 3. * (0.05 * 0.95 / r).sqrt();
    let cols = fits[0].coefficients.len();
    let truth: Vec<_> = if cols == 2 { vec![0., delta] } else { vec![delta] };
    let means: Vec<_> = (0..cols)
        .map(|j| fits.iter().map(|f| f.coefficients[j].difference).sum::<f64>() / r)
        .collect();
    let empirical: Vec<_> = (0..cols)
        .map(|j| {
            fits.iter().map(|f| (f.coefficients[j].difference - means[j]).powi(2)).sum::<f64>()
                / (r - 1.)
        })
        .collect();
    let estimated: Vec<_> = (0..cols)
        .map(|j| fits.iter().map(|f| f.coefficients[j].standard_error.powi(2)).sum::<f64>() / r)
        .collect();
    let mdd: Vec<_> = (0..cols)
        .map(|j| {
            fits.iter().map(|f| f.coefficients[j].minimal_detectable_difference).sum::<f64>() / r
        })
        .collect();
    let covariance_error =
        empirical.iter().chain(&estimated).map(|v| (v / variance - 1.).abs()).fold(0., f64::max);
    let mdd_error =
        mdd.iter().map(|v| (v / (factor * variance.sqrt()) - 1.).abs()).fold(0., f64::max);
    let bias = means
        .iter()
        .zip(&truth)
        .map(|(m, t)| (m - t).abs() / (variance / r).sqrt())
        .fold(0., f64::max);
    let wald = fits.iter().filter(|f| f.test.p_value <= 0.05).count() as f64 / r;
    let holm = fits
        .iter()
        .filter(|f| {
            f.coefficients
                .iter()
                .zip(&truth)
                .any(|(c, t)| t.abs() < f64::EPSILON && c.p_holm <= 0.05)
        })
        .count() as f64
        / r;
    let rate_ok = match label {
        "null" => (wald - 0.05).abs() <= slack,
        "strong_partial_null" => wald >= 0.9,
        _ => (wald - 0.8).abs() <= 3. * (0.8 * 0.2 / r).sqrt() + 0.02,
    };
    let pass = covariance_error <= 0.15
        && mdd_error <= 0.15
        && bias <= 5.
        && holm <= 0.05 + slack
        && rate_ok;
    println!(
        "INFERENCE_DIAGNOSTIC_RECORD {}",
        json!({"test":test,"design":label,"rows_per_population":n,"seed":seed,"attempts":results.len(),"failed":failed,"truth_difference":truth,"truth_covariance_diagonal":variance,"estimated_variance":estimated,"empirical_variance":empirical,"wald_rejection_rate":wald,"false_holm_rate":holm,"mean_mdd":mdd,"true_mdd":factor*variance.sqrt(),"variance_relative_error":covariance_error,"mdd_relative_error":mdd_error,"bias_mcse":bias,"null_alpha":0.05,"mdd_power":0.8,"mdd_power_approximation_margin":0.02,"pass":pass,"activation":"none; non-rejection never certifies invariance"})
    );
    assert!(pass, "{test} {label} n{n}: diagnostic frozen rules failed");
}
#[test]
#[ignore = "calibration final only"]
fn independent_mechanism_full_coefficient_diagnostic() {
    measure("independent_mechanism_full_coefficient_diagnostic", true);
}
#[test]
#[ignore = "calibration final only"]
fn independent_mechanism_slope_mdd_diagnostic() {
    measure("independent_mechanism_slope_mdd_diagnostic", false);
}

/// Ordinary one-snapshot numerical plumbing; no repeated sampling or error-rate measurement.
#[test]
fn mechanism_one_snapshot_independent_balanced_ols_oracle() {
    let n = 512;
    let seed = 97;
    let delta = 0.5;
    let result = fit(n, seed, true, delta).unwrap();
    let mut rng = candidate::Generator::new(seed);
    let x: Vec<_> = (0..n).map(|i| if i % 2 == 0 { -1. } else { 1. }).collect();
    let source: Vec<_> = x.iter().map(|v| 1. + v + rng.normal()).collect();
    let target: Vec<_> = x.iter().map(|v| 1. + (1. + delta) * v + 1.5 * rng.normal()).collect();
    let plain = |y: &[f64]| {
        let intercept = y.iter().sum::<f64>() / n as f64;
        let slope = y.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>() / n as f64;
        let residual_variance =
            y.iter().zip(&x).map(|(a, b)| (a - intercept - slope * b).powi(2)).sum::<f64>()
                / (n - 2) as f64;
        ([intercept, slope], residual_variance)
    };
    let (s, sv) = plain(&source);
    let (t, tv) = plain(&target);
    let se = ((sv + tv) / n as f64).sqrt();
    for (j, c) in result.coefficients.iter().enumerate() {
        assert!((c.difference - (t[j] - s[j])).abs() < 1e-9);
        assert!((c.standard_error - se).abs() < 1e-9);
        assert!((c.minimal_detectable_difference - DETECTABILITY_FACTOR * se).abs() < 1e-9);
    }
    assert!(!result.non_rejection_certifies_invariance);
}
