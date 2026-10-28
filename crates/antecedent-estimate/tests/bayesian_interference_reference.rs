//! The Gaussian fixed-network interference posterior against an exact conjugate reference.
//!
//! `oracle_data/bayesian_interference_reference.py` (frozen in the JSON beside it) builds the
//! neighbour-count exposures of a 48-unit weighted network under a realised assignment, forms the
//! known-variance Normal posterior of `y = a + b own + g neighbours` under independent
//! `N(0, tau^2)` priors in 40-digit arithmetic, and the posterior mean and standard deviation of
//! the contrast between two exposure levels. Edge weights must not enter the neighbour-count
//! mapping, and the residual variance is fixed at one, so the interval does not depend on the
//! outcome noise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery,
    VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::estimate_interference_bayesian;
use serde_json::Value;

const DRAWS: usize = 400_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/bayesian_interference_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn level(v: &Value) -> ExposureLevel {
    let pair = floats(v);
    ExposureLevel { own: pair[0], neighbors: pair[1] }
}

fn run(case: &Value, draws: usize) -> antecedent_estimate::BayesianInterferenceEstimate {
    run_with(case, draws, 0.4, None)
}

/// `weight` replaces every edge weight and `probability` is the declared Bernoulli probability.
fn run_with(
    case: &Value,
    draws: usize,
    probability: f64,
    weight: Option<f64>,
) -> antecedent_estimate::BayesianInterferenceEstimate {
    let outcome = floats(&case["outcome"]);
    let units = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let edges: Vec<NetworkEdge> = case["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            NetworkEdge::new(
                e[0].as_u64().unwrap(),
                e[1].as_u64().unwrap(),
                weight.unwrap_or_else(|| e[2].as_f64().unwrap()),
            )
        })
        .collect();
    let data = NetworkData::try_new(units, edges).unwrap();
    let assignment: Vec<bool> =
        case["assignment"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let query = InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([probability]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: level(&case["from_level"]),
            to: level(&case["to_level"]),
        },
    );
    estimate_interference_bayesian(
        &query,
        &data,
        &assignment,
        draws,
        7,
        case["prior_sd"].as_f64().unwrap(),
    )
    .unwrap()
}

#[test]
fn the_exposure_contrast_posterior_matches_the_exact_gaussian_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let fit = run(&case, DRAWS);
        let (mean, sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
        let root = (DRAWS as f64).sqrt();
        assert!((fit.mean - mean).abs() < 5.0 * sd / root, "{name} mean {} vs {mean}", fit.mean);
        let draws = &fit.contrast_draws;
        let m = draws.iter().sum::<f64>() / draws.len() as f64;
        let measured =
            (draws.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (draws.len() - 1) as f64).sqrt();
        assert!((measured / sd - 1.0).abs() < 8.0 / root, "{name} sd {measured} vs {sd}");
    }
}

#[test]
fn rescaling_the_outcome_moves_the_mean_but_not_the_fixed_variance_spread() {
    let case = cases().into_iter().find(|c| c["name"] == "wide_prior").unwrap();
    let mut shifted = case.clone();
    shifted["outcome"] =
        Value::Array(floats(&case["outcome"]).iter().map(|v| Value::from(v * 10.0)).collect());
    let (a, b) = (run(&case, 20_000), run(&shifted, 20_000));
    let spread = |d: &[f64]| {
        let m = d.iter().sum::<f64>() / d.len() as f64;
        (d.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (d.len() - 1) as f64).sqrt()
    };
    // The same seed gives the same standard normals, so the spread is identical.
    assert!((spread(&a.contrast_draws) / spread(&b.contrast_draws) - 1.0).abs() < 1e-9);
    assert!(b.mean > 5.0 * a.mean);
}

#[test]
fn edge_weights_and_the_declared_assignment_probability_do_not_enter_the_posterior() {
    // The neighbour-count mapping counts treated in-neighbours whatever their weight, and the
    // model-based posterior conditions on the realised assignment, so neither the weights nor the
    // declared Bernoulli probability may move it (the same seed gives identical draws).
    let case = cases().into_iter().find(|c| c["name"] == "tight_prior").unwrap();
    let base = run_with(&case, 5_000, 0.4, None);
    assert_eq!(run_with(&case, 5_000, 0.4, Some(1.0)).contrast_draws, base.contrast_draws);
    assert_eq!(run_with(&case, 5_000, 0.9, None).contrast_draws, base.contrast_draws);
}
