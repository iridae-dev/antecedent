//! The subject-level adaptively weighted estimate with a subject-size multiplier against a 60-digit
//! evaluation of its definition from raw measurements
//! (`oracle_data/repeated_measures_weighting_reference.py`).
//!
//! The reference reduces every subject to the mean of its measurements, builds the predictable
//! prior-shrunk running mean of earlier subject means, the augmented inverse-probability score of
//! each subject, the weight `m_i * h_i` (measurement count times uniform or variance-stabilizing
//! weight) and the self-normalized estimate, plug-in variance, effective units and largest weight
//! share. Subjects have 1 to 400 measurements, assignment probabilities down to 1/1024, two and
//! three arms, arm-mean and contrast estimands and a prior mean far from the data. The test also
//! checks the two identities the multiplier must satisfy: a constant multiplier changes nothing and
//! a multiplier of ones is the unweighted route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::adaptive_inference::{
    Estimand, OutcomeModelPrior, WeightScheme, WeightedHistory, WeightedOptions,
    adaptively_weighted_estimate,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/repeated_measures_weighting_reference.json"))
        .expect("frozen reference")
}

fn frac(value: &Value) -> f64 {
    let text = value.as_str().expect("text");
    let (n, d) = text.split_once('/').expect("a fraction");
    n.parse::<f64>().unwrap() / d.parse::<f64>().unwrap()
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-10 * want.abs().max(1.0), "{what}: {got} vs {want}");
}

#[test]
fn subject_scores_weighted_by_measurement_count_match_the_definition() {
    let reference = reference();
    let ctx = ExecutionContext::for_tests(5);
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let arms = case["arms"].as_u64().unwrap() as usize;
        let assigned: Vec<usize> = case["assigned"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_u64().unwrap() as usize)
            .collect();
        let probabilities: Vec<f64> = case["probabilities"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(frac).collect::<Vec<_>>())
            .collect();
        // The route reduces each subject to the mean of its measurements.
        let (outcomes, multiplier): (Vec<f64>, Vec<f64>) = case["measurements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|subject| {
                let ys: Vec<f64> = subject.as_array().unwrap().iter().map(frac).collect();
                (ys.iter().sum::<f64>() / ys.len() as f64, ys.len() as f64)
            })
            .unzip();
        let history = WeightedHistory {
            arms,
            assigned: &assigned,
            probabilities: &probabilities,
            outcomes: &outcomes,
            multiplier: Some(&multiplier),
        };
        let e = &case["estimand"];
        let estimand = if e[0] == "arm" {
            Estimand::ArmMean(e[1].as_u64().unwrap() as usize)
        } else {
            Estimand::Contrast {
                plus: e[1].as_u64().unwrap() as usize,
                minus: e[2].as_u64().unwrap() as usize,
            }
        };
        let options = WeightedOptions {
            scheme: if case["scheme"] == "uniform" {
                WeightScheme::Uniform
            } else {
                WeightScheme::VarianceStabilizing
            },
            prior: OutcomeModelPrior {
                mean: frac(&case["mean"]),
                strength: case["strength"].as_f64().unwrap(),
            },
            probability_floor: frac(&case["floor"]),
        };
        let got = adaptively_weighted_estimate(&history, estimand, &options, &ctx).expect(name);
        let want = &case["expected"];
        close(got.estimate, want["estimate"].as_f64().unwrap(), name);
        close(got.uncalibrated_variance, want["variance"].as_f64().unwrap(), name);
        close(got.effective_units, want["effective_units"].as_f64().unwrap(), name);
        close(got.max_weight_share, want["max_weight_share"].as_f64().unwrap(), name);
        assert_eq!(got.units, assigned.len());

        // A constant multiplier changes nothing; a multiplier of ones is the unweighted route.
        let scaled: Vec<f64> = vec![7.0; assigned.len()];
        let ones: Vec<f64> = vec![1.0; assigned.len()];
        let plain = adaptively_weighted_estimate(
            &WeightedHistory { multiplier: None, ..history },
            estimand,
            &options,
            &ctx,
        )
        .unwrap();
        let by_seven = adaptively_weighted_estimate(
            &WeightedHistory { multiplier: Some(&scaled), ..history },
            estimand,
            &options,
            &ctx,
        )
        .unwrap();
        let by_one = adaptively_weighted_estimate(
            &WeightedHistory { multiplier: Some(&ones), ..history },
            estimand,
            &options,
            &ctx,
        )
        .unwrap();
        close(by_seven.estimate, plain.estimate, name);
        close(by_seven.uncalibrated_variance, plain.uncalibrated_variance, name);
        assert_eq!(by_one.estimate, plain.estimate, "{name}");
        assert_eq!(by_one.uncalibrated_variance, plain.uncalibrated_variance, "{name}");
    }
}
