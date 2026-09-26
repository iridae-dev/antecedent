//! Scoped repeated-sampling check for the modular Bayesian robust ATE route.
//!
//! This fixture checks only finite-sample behavior for the declared linear
//! nuisance fits. It does not establish causal identification or general
//! double-robust interval coverage.

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_estimate::bayesian_robust_ate::{
    BayesianRobustAteInput, BayesianRobustAteOptions, estimate_bayesian_robust_ate,
};

#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // independent fixture switches, each a distinct data-generating choice
struct Scenario {
    name: &'static str,
    nonlinear_propensity: bool,
    nonlinear_outcome: bool,
    weak_overlap: bool,
    heterogeneous: bool,
}

struct Summary {
    bias: f64,
    coverage: f64,
    width: f64,
    failures: usize,
    reps: usize,
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

fn sample(seed: u64, n: usize, scenario: Scenario) -> (BayesianRobustAteInput, f64) {
    let ctx = ExecutionContext::for_tests(seed);
    let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, 0xCA1B_0000 + seed);
    let mut x = Vec::with_capacity(n);
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut tau_sum = 0.0;
    for _ in 0..n {
        let v = 2.0 * rng.next_f64() - 1.0;
        let prop_eta = if scenario.weak_overlap {
            4.0 * v
        } else if scenario.nonlinear_propensity {
            -0.15 + 0.75 * v + 1.0 * v * v
        } else {
            -0.15 + 0.75 * v
        };
        let t = rng.next_f64() < logistic(prop_eta);
        let base = if scenario.nonlinear_outcome { 1.1 * v * v } else { 0.6 * v };
        let tau = if scenario.heterogeneous { 1.0 + 0.5 * v } else { 1.0 };
        // Box-Muller normal noise, independent of treatment conditional on X.
        let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
        let u2 = rng.next_f64();
        let noise = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos() * 0.8;
        x.push(v);
        treatment.push(t);
        outcome.push(base + if t { tau } else { 0.0 } + noise);
        tau_sum += tau;
    }
    let truth = tau_sum / n as f64;
    (
        BayesianRobustAteInput {
            row_ids: (0..n).map(|i| (i as u64 + 1) * 31 + seed * 1_000_003).collect(),
            treatment,
            outcome,
            covariates: vec![x],
        },
        truth,
    )
}

fn run(scenario: Scenario, reps: usize) -> Summary {
    let mut errors = Vec::new();
    let mut widths = Vec::new();
    let mut covered = 0;
    let mut failures = 0;
    for rep in 0..reps {
        let seed = 1_900 + rep as u64;
        let (input, truth) = sample(seed, 360, scenario);
        let ctx = ExecutionContext::for_tests(seed + 50_000);
        match estimate_bayesian_robust_ate(
            &input,
            BayesianRobustAteOptions {
                draws: 64,
                folds: 4,
                ridge_penalty: 0.05,
                propensity_clip: 0.01,
                coverage: 0.90,
            },
            &ctx,
        ) {
            Ok(result) => {
                errors.push(result.estimate - truth);
                widths.push(result.interval.1 - result.interval.0);
                covered += usize::from(result.interval.0 <= truth && truth <= result.interval.1);
            }
            Err(_) => failures += 1,
        }
    }
    let successful = errors.len();
    Summary {
        bias: if successful == 0 {
            f64::NAN
        } else {
            errors.iter().sum::<f64>() / successful as f64
        },
        coverage: if successful == 0 { f64::NAN } else { covered as f64 / successful as f64 },
        width: if successful == 0 {
            f64::NAN
        } else {
            widths.iter().sum::<f64>() / successful as f64
        },
        failures,
        reps,
    }
}

#[test]
fn repeated_sampling_is_scoped_to_one_correct_linear_nuisance() {
    // Forty replications keep this executable test bounded. The broad lower
    // coverage threshold is a smoke check, not a calibration certificate.
    let scenarios = [
        Scenario {
            name: "outcome correct; propensity nonlinear",
            nonlinear_propensity: true,
            nonlinear_outcome: false,
            weak_overlap: false,
            heterogeneous: true,
        },
        Scenario {
            name: "propensity correct; outcome nonlinear",
            nonlinear_propensity: false,
            nonlinear_outcome: true,
            weak_overlap: false,
            heterogeneous: true,
        },
        Scenario {
            name: "both nuisances misspecified",
            nonlinear_propensity: true,
            nonlinear_outcome: true,
            weak_overlap: false,
            heterogeneous: true,
        },
        Scenario {
            name: "weak overlap",
            nonlinear_propensity: false,
            nonlinear_outcome: false,
            weak_overlap: true,
            heterogeneous: true,
        },
    ];
    for scenario in scenarios {
        let summary = run(scenario, 40);
        eprintln!(
            "{}: bias={:.3}, 90% interval coverage={:.3}, mean width={:.3}, failed fits={}/{}",
            scenario.name,
            summary.bias,
            summary.coverage,
            summary.width,
            summary.failures,
            summary.reps
        );
        assert_eq!(summary.failures, 0, "{}", scenario.name);
        assert!(summary.bias.abs() < 0.25, "{} bias={}", scenario.name, summary.bias);
        if scenario.name.starts_with("outcome correct")
            || scenario.name.starts_with("propensity correct")
        {
            // This permissive floor catches gross undercoverage while avoiding
            // an unsupported high-precision claim at only 40 replications.
            assert!(summary.coverage >= 0.70, "{} coverage={}", scenario.name, summary.coverage);
        }
    }
}
