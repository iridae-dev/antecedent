//! Bayesian unit counterfactual downstream of a categorical node: the noise posterior.
//!
//! A binary outcome has a many-to-one disturbance. Given the parameters, the unit's disturbance
//! is uniform within the CDF bin of its factual category, and the factual and counterfactual
//! worlds replay that one disturbance (rank-preserving coupling). A Bayesian counterfactual draws
//! the parameters AND that disturbance together, once per posterior draw, so each unit's
//! posterior mean is an average over both. Every expected value below is the closed form of that
//! construction from the empirical conditional probabilities `p0 = P(y=1 | x=0)` and
//! `p1 = P(y=1 | x=1)`; the library is never consulted for them.
//!
//! With `p1 > p0`, categories ordered 0 < 1 and `do(x=1)` against `do(x=0)`:
//!
//! - factual `(x=0, y=0)`: `u ~ U(0, 1-p0]`, `Y(1)=1` iff `u > 1-p1`, `Y(0)=0`, so the unit effect
//!   is `Bernoulli((p1 - p0) / (1 - p0))`;
//! - factual `(x=1, y=1)`: `u ~ U(1-p1, 1]`, `Y(1)=1`, `Y(0)=1` iff `u > 1-p0`, so the unit effect
//!   is `Bernoulli(1 - p0/p1)`;
//! - factual `(x=0, y=1)` and `(x=1, y=0)`: the effect is 0 for every `u`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "test scaffolding divides small constant counts")]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

/// `(x, y, count)`: `p0 = 30/120 = 0.25`, `p1 = 72/120 = 0.6`.
const CELLS: [(f64, f64, usize); 4] =
    [(0.0, 0.0, 90), (0.0, 1.0, 30), (1.0, 0.0, 48), (1.0, 1.0, 72)];

fn fixture() -> (TabularData, Vec<f64>, Vec<f64>, Dag) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (xv, yv, count) in CELLS {
        x.extend(std::iter::repeat_n(xv, count));
        y.extend(std::iter::repeat_n(yv, count));
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, x, y, graph)
}

fn run(inference: InferenceMode) -> antecedent::StudyResult {
    let (data, _, _, graph) = fixture();
    let query = CounterfactualQuery::new(
        VariableId::from_raw(1),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    )
    .with_control_level(0.0);
    Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Counterfactual(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(71))
        .unwrap()
}

fn group(x: &[f64], y: &[f64], want_x: f64, want_y: f64) -> Vec<usize> {
    (0..x.len())
        .filter(|&i| x[i].to_bits() == want_x.to_bits() && y[i].to_bits() == want_y.to_bits())
        .collect()
}

#[test]
fn bayesian_unit_effects_integrate_the_categorical_noise_posterior() {
    let (_, x, y, _) = fixture();
    let (p0, p1) = (30.0 / 120.0, 72.0 / 120.0);
    let result = run(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1_500)));
    let effects = &result.counterfactual().unwrap().unit_effects;
    assert_eq!(effects.len(), x.len());
    let engine = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "gcm.counterfactual")
        .expect("the abduction disclosure");
    assert!(engine.message.contains("noise_inference=PosteriorNoise"), "{}", engine.message);

    let cases = [
        (0.0, 0.0, (p1 - p0) / (1.0 - p0)),
        (1.0, 1.0, 1.0 - p0 / p1),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ];
    for (xv, yv, expected) in cases {
        let units = group(&x, &y, xv, yv);
        assert!(!units.is_empty());
        let mean = units.iter().map(|&i| effects[i]).sum::<f64>() / units.len() as f64;
        assert!((mean - expected).abs() < 0.03, "(x={xv}, y={yv}) mean {mean} vs {expected}");
        // A frozen disturbance would put every unit's posterior mean near 0 or 1. Integrating
        // the noise leaves every unit of the group at the same interior value, up to the Monte
        // Carlo error of 1500 draws (about 0.013).
        for &i in &units {
            assert!(
                (effects[i] - expected).abs() < 0.09,
                "(x={xv}, y={yv}) unit {i}: posterior mean {} vs {expected}",
                effects[i]
            );
        }
    }
}

#[test]
fn the_frequentist_twin_replays_one_seeded_disturbance() {
    let (_, x, y, _) = fixture();
    let result = run(InferenceMode::Frequentist);
    let effects = &result.counterfactual().unwrap().unit_effects;
    // One draw of the rank-preserving noise: each unit's effect is exactly 0 or 1.
    assert!(effects.iter().all(|e| e.abs() < 1e-9 || (e - 1.0).abs() < 1e-9));
    // Across units the draws are independent uniforms, so the group frequencies are the
    // closed-form probabilities up to binomial error; the interval stays unavailable.
    let units = group(&x, &y, 0.0, 0.0);
    let frequency = units.iter().map(|&i| effects[i]).sum::<f64>() / units.len() as f64;
    let (p0, p1) = (30.0 / 120.0, 72.0 / 120.0);
    let expected = (p1 - p0) / (1.0 - p0);
    assert!((frequency - expected).abs() < 0.2, "{frequency} vs {expected}");
    assert!(result.counterfactual().unwrap().unit_effect_intervals.is_none());
}
