//! The simulated sampling law of the adaptively weighted estimator at the terminal look of a
//! declared procedure against its exact law (`oracle_data/procedure_weighting_simulation_reference.py`).
//!
//! The reference enumerates every path of five small procedures in exact rational arithmetic
//! (Thompson with a posterior stop, variance-stabilizing weights, lagged epsilon-greedy with random
//! loss, batches with exposure failure under both attributions, a resource cap) and evaluates the
//! weighted estimator and its plug-in variance at 60 digits on each path, giving the exact
//! probability of stopping, the expected enrolment, the distribution of analysed units and of the
//! decision, and the exact mean and standard deviation of the estimate and of the plug-in variance
//! given an estimate exists. The seeded simulation (60,000 replicates) must agree at five Monte
//! Carlo standard errors; the replay of a seed is bit-identical and another seed differs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::{
    Estimand, OutcomeModelPrior, SimulationRequest, WeightScheme, simulate_procedure,
};
use serde_json::Value;
use support::{frac, fracs, never, spec};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/procedure_weighting_simulation_reference.json"))
        .expect("frozen reference")
}

fn request(case: &Value, seed: u64) -> SimulationRequest {
    let e = &case["estimand"];
    let estimand = if e[0] == "arm" {
        Estimand::ArmMean(e[1].as_u64().unwrap() as usize)
    } else {
        Estimand::Contrast {
            plus: e[1].as_u64().unwrap() as usize,
            minus: e[2].as_u64().unwrap() as usize,
        }
    };
    SimulationRequest {
        replications: case["replications"].as_u64().unwrap() as u32,
        seed,
        estimand,
        scheme: if case["scheme"] == "uniform" {
            WeightScheme::Uniform
        } else {
            WeightScheme::VarianceStabilizing
        },
        outcome_prior: OutcomeModelPrior {
            mean: frac(&case["outcome_prior"]["mean"]),
            strength: case["outcome_prior"]["strength"].as_f64().unwrap(),
        },
        probability_floor: 1e-6,
    }
}

fn within_sigmas(got: f64, want: f64, standard_error: f64, what: &str) {
    assert!(
        (got - want).abs() <= 5.0 * standard_error + 1e-12,
        "{what}: simulated {got} vs exact {want} (standard error {standard_error})"
    );
}

fn proportion(got: f64, want: f64, n: f64, what: &str) {
    within_sigmas(got, want, (want * (1.0 - want) / n).sqrt(), what);
}

#[test]
fn the_simulated_law_agrees_with_the_exact_law_of_the_procedure() {
    let reference = reference();
    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let spec = spec(&case["declaration"]);
        let truth = fracs(&case["truth"]);
        let request = request(case, 20_241_028);
        let law = simulate_procedure(&spec, &truth, &request, &never).expect("a simulation");
        let n = f64::from(request.replications);
        assert_eq!(law.replicates.len(), request.replications as usize);
        proportion(law.rule_stop_fraction(), case["stop_probability"].as_f64().unwrap(), n, name);
        // Enrolment ranges over the declared cumulative sizes, whose spread bounds the deviation.
        let enrolled: u32 = case["declaration"]["enrolment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as u32)
            .sum();
        within_sigmas(
            law.mean_enrolled(),
            case["expected_enrolled"].as_f64().unwrap(),
            f64::from(enrolled) / 2.0 / n.sqrt(),
            name,
        );
        let defined = law.replicates.iter().filter(|r| r.estimate.is_some()).count() as f64;
        proportion(defined / n, case["defined_probability"].as_f64().unwrap(), n, name);
        for (units, want) in case["analysed_units"].as_object().unwrap() {
            let units: usize = units.parse().unwrap();
            let frequency =
                law.replicates.iter().filter(|r| r.analysed_units == units).count() as f64 / n;
            proportion(frequency, want.as_f64().unwrap(), n, &format!("{name}: {units} analysed"));
        }
        for (label, want) in case["decided"].as_object().unwrap() {
            let frequency = law
                .replicates
                .iter()
                .filter(|r| match (label.as_str(), r.decision) {
                    ("none", None) => true,
                    (l, Some(d)) => l.parse::<usize>().ok() == Some(d),
                    _ => false,
                })
                .count() as f64
                / n;
            proportion(frequency, want.as_f64().unwrap(), n, &format!("{name}: decision {label}"));
        }
        let summary = law.summarize(0.0, 1.96).expect("a summary");
        let sd = case["estimate_sd"].as_f64().unwrap();
        within_sigmas(
            summary.mean,
            case["estimate_mean"].as_f64().unwrap(),
            sd / defined.sqrt(),
            name,
        );
        within_sigmas(
            summary.mean_reported_variance,
            case["variance_mean"].as_f64().unwrap(),
            case["variance_sd"].as_f64().unwrap() / defined.sqrt(),
            name,
        );
        assert!(
            (summary.standard_deviation / sd - 1.0).abs() < 0.05,
            "{name}: sd {} vs {sd}",
            summary.standard_deviation
        );
        // The stopped estimator is biased for the arm contrast exactly as the exact law says;
        // nothing here is a coverage statement.
        assert_eq!(summary.defined as f64, defined);
    }
}

#[test]
fn a_seed_replays_exactly_and_another_seed_does_not() {
    let reference = reference();
    let case = &reference["cases"][0];
    let spec = spec(&case["declaration"]);
    let truth = fracs(&case["truth"]);
    let mut small = request(case, 7);
    small.replications = 500;
    let first = simulate_procedure(&spec, &truth, &small, &never).unwrap();
    let again = simulate_procedure(&spec, &truth, &small, &never).unwrap();
    assert_eq!(first.replicates, again.replicates);
    small.seed = 8;
    let other = simulate_procedure(&spec, &truth, &small, &never).unwrap();
    assert_ne!(first.replicates, other.replicates);
    assert!(simulate_procedure(&spec, &[0.5], &small, &never).is_err());
    small.replications = 0;
    assert!(simulate_procedure(&spec, &truth, &small, &never).is_err());
}
