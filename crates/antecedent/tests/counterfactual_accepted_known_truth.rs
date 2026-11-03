//! Known-truth check for `Counterfactual x Dag x accepted x {Frequentist, Bayesian}`.
//!
//! The data-generating law of `conformance/estimate/staged_static_kinds` is
//! `A = sin(.71 i)`, `M = 2A + cos(1.13 i)`, `Y = 3A + 4M + .1 sin(.31 i)`, so the mean
//! individual effect of moving `A` from 0.2 to 0.8 is `(3 + 4*2) * 0.6 = 6.6` (the fixture's
//! `counterfactual_mean`). The accepted-structure coordinate is executed and compared with
//! that closed form, not with the explicit-structure result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn accepted_counterfactual_matches_the_closed_form_mean_effect() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_static_kinds/expected.json"
    ))
    .unwrap();
    let control = pin["control"].as_f64().unwrap();
    let active = pin["active"].as_f64().unwrap();
    let truth = pin["counterfactual_mean"].as_f64().unwrap();
    assert!((truth - (3.0 + 4.0 * 2.0) * (active - control)).abs() < 1e-12);
    let a: Vec<f64> = (0..500).map(|i| (f64::from(i) * 0.71).sin()).collect();
    let m: Vec<f64> =
        a.iter().enumerate().map(|(i, a)| 2.0 * a + (i as f64 * 1.13).cos()).collect();
    let y: Vec<f64> = a
        .iter()
        .zip(&m)
        .enumerate()
        .map(|(i, (a, m))| 3.0 * a + 4.0 * m + 0.1 * (i as f64 * 0.31).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(active))]),
    )
    .with_control_level(control);
    let ctx = ExecutionContext::for_tests(13);
    for (mode, inference, tolerance) in [
        ("frequentist", InferenceMode::Frequentist, pin["tolerance"].as_f64().unwrap()),
        (
            "bayesian",
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128).prior_scale(1_000.0)),
            0.08,
        ),
    ] {
        let result = Study::tabular(data.clone())
            .graph(AcceptedGraph::from(dag.clone()))
            .query(CausalQuery::Counterfactual(query.clone()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap();
        let cf = result.counterfactual().unwrap();
        assert!(
            (cf.mean_ite - truth).abs() < tolerance,
            "accepted {mode}: mean effect {} vs closed form {truth}",
            cf.mean_ite
        );
        assert_eq!(cf.unit_effects.len(), 500);
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit"));
    }
}
