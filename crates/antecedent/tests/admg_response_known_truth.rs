//! Known-truth evidence for the static response estimators on an explicit and an accepted
//! ADMG (no bidirected edge, so the ADMG is the back-door DAG) under every validation suite.
//!
//! The design is a deterministic noise-free table:
//! `z` a centred grid, `t` a coprime-stride permutation of it, `y = 1 + 2 t`. The intervention level
//! `E[y | do(t = L)] = 1 + 2 L`. The frequentist
//! additive g-computation is exact on the linear law; the Bayesian conjugate response is held
//! to six Monte-Carlo standard errors of at least 2 000 draws of a posterior whose sd is at
//! most 1e-1 (tolerance 6 * 0.1 / sqrt(2000) = 0.0134).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "known_truth_cells/mod.rs"]
mod support;

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId};
use support::{Coordinate, GraphKind, Structure, for_each_matching, on_large_stack, run_retained};

const N: usize = 400;
const LEVEL: f64 = 0.5;

fn table() -> (TabularData, f64) {
    // The plug-in outcome GAM leaves the treatment smooth unpenalized (exact on a linear dose
    // effect) but roughness-penalizes the adjustment smooth, which shrinks even a linear `z`
    // effect; the outcome law therefore has no `z` term (a fitted smooth of a zero residual is
    // exactly zero). `t` is a coprime-stride permutation of the `z` grid so they are not concurved.
    let n1 = (N - 1) as f64;
    let z: Vec<f64> = (0..N).map(|i| -1.0 + 2.0 * i as f64 / n1).collect();
    let t: Vec<f64> = (0..N).map(|i| -1.0 + 2.0 * ((i * 173 + 37) % N) as f64 / n1).collect();
    let y: Vec<f64> = t.iter().map(|t| 1.0 + 2.0 * t).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    (data, 1.0 + 2.0 * LEVEL)
}

fn admg() -> Admg {
    let mut graph = Admg::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn level() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(LEVEL))]),
    })
}

fn explicit_or_accepted(c: &Coordinate) -> bool {
    c.graph == GraphKind::Admg && c.structure != Structure::GraphPosterior
}

fn check(family: &'static str, estimator: EstimatorId, id: &'static str, tolerance: f64) {
    let (data, truth) = table();
    on_large_stack(move || {
        for_each_matching(family, explicit_or_accepted, |c| {
            let inference = if c.bayesian {
                InferenceMode::Bayesian(
                    BayesianConfig::conjugate().n_draws(4_000).prior_scale(1_000.0),
                )
            } else {
                InferenceMode::Frequentist
            };
            let builder = Study::tabular(data.clone());
            let builder = match c.structure {
                Structure::Explicit => builder.graph(admg()),
                _ => builder.graph(AcceptedGraph::from(admg())),
            };
            let result = run_retained(
                builder
                    .query(CausalQuery::Response(level()))
                    .estimator(estimator)
                    .inference(inference)
                    .refute(c.suite)
                    .bootstrap_replicates(0),
                &c.text,
                &ExecutionContext::for_tests(52),
            );
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(id), "{}", c.text);
            let value =
                result.response().map_or(result.estimate().as_effect().unwrap().ate, |r| match &r
                    .estimate
                {
                    ResponseIdentification::PointIdentified(ResponseValue::Scalar(v)) => *v,
                    _ => result.estimate().as_effect().unwrap().ate,
                });
            assert!((value - truth).abs() < tolerance, "{}: level {value} vs {truth}", c.text);
            match c.suite {
                antecedent::RefuteSuite::None => {
                    assert!(result.refutations.is_empty(), "{}", c.text);
                }
                _ if c.bayesian => assert!(
                    !result.refutations.is_empty() || !result.predictive_checks.is_empty(),
                    "{}: validation",
                    c.text
                ),
                _ => assert!(!result.refutations.is_empty(), "{}: refuters", c.text),
            }
        });
    });
}

#[test]
fn intervention_gcomp_level_matches_the_closed_form_on_admg_coordinates() {
    check(
        "response.intervention_gcomp|InterventionResponse",
        EstimatorId::ResponseInterventionGcomp,
        "response.intervention_gcomp",
        1e-6,
    );
}

#[test]
fn bayesian_response_level_matches_the_closed_form_on_admg_coordinates() {
    check(
        "response.bayesian|InterventionResponse",
        EstimatorId::ResponseBayesian,
        "response.bayesian",
        0.0134,
    );
}
