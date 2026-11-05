//! Known-truth evidence for the front-door plug-in functional on an explicit and an accepted
//! DAG under every validation suite.
//!
//! The table is the exact 4000-row population of the binary structural model
//! `U ~ Bern(.5)`, `P(T=1|U) = .1 + .5U`, `P(M=1|T) = .1 + .7T`, `P(Y=1|M,U) = .05 + .9MU` with
//! `U` dropped (every cell probability is a multiple of 1/4000, so the cell counts are exact).
//! The latent modifies the mediator's effect, so the product of coefficients misses the
//! truth while the front-door functional
//! `E[Y | do(t)] = sum_m P(m | t) sum_t' P(t') E[Y | m, t']`
//! recovers it. The test evaluates that functional from the cell counts alone (no estimator),
//! and checks it equals the structural enumeration `0.7 * 0.9 * E[U] = 0.315` that
//! `conformance/estimate/frontdoor_functional` records.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "known_truth_cells/mod.rs"]
mod support;

use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, InferenceMode, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use support::{Coordinate, GraphKind, Structure, for_each_matching, on_large_stack, run_retained};

/// Exact equality of two indicator values (the design only holds literal 0/1 levels).
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Columns `t, y, m` (ids 0, 1, 2).
fn frontdoor_table() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for u in [0.0_f64, 1.0] {
        for ti in [0.0_f64, 1.0] {
            let pt = if same(ti, 1.0) { 0.1 + 0.5 * u } else { 0.9 - 0.5 * u };
            for mi in [0.0_f64, 1.0] {
                let pm = if same(mi, 1.0) { 0.1 + 0.7 * ti } else { 0.9 - 0.7 * ti };
                for yi in [0.0_f64, 1.0] {
                    let py1 = 0.05 + 0.9 * mi * u;
                    let py = if same(yi, 1.0) { py1 } else { 1.0 - py1 };
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "the rounded expected count is a small non-negative integer"
                    )]
                    let count = (0.5 * pt * pm * py * 4000.0).round() as usize;
                    t.extend(std::iter::repeat_n(ti, count));
                    m.extend(std::iter::repeat_n(mi, count));
                    y.extend(std::iter::repeat_n(yi, count));
                }
            }
        }
    }
    assert_eq!(t.len(), 4000);
    (t, y, m)
}

fn frontdoor_data() -> TabularData {
    let (t, y, m) = frontdoor_table();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("m", m.as_slice())])
        .unwrap()
}

/// `E[Y | do(t = level)]` from the counts: `sum_m P(m | level) sum_t' P(t') P(Y = 1 | t', m)`.
fn frontdoor_mean(level: f64) -> f64 {
    let (t, y, m) = frontdoor_table();
    let n = t.len() as f64;
    let count = |pred: &dyn Fn(usize) -> bool| (0..t.len()).filter(|&i| pred(i)).count() as f64;
    let mut total = 0.0;
    for mi in [0.0, 1.0] {
        let p_m = count(&|i| same(t[i], level) && same(m[i], mi)) / count(&|i| same(t[i], level));
        let mut inner = 0.0;
        for ti in [0.0, 1.0] {
            let p_t = count(&|i| same(t[i], ti)) / n;
            let p_y = count(&|i| same(t[i], ti) && same(m[i], mi) && same(y[i], 1.0))
                / count(&|i| same(t[i], ti) && same(m[i], mi));
            inner += p_t * p_y;
        }
        total += p_m * inner;
    }
    total
}

fn frontdoor_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap(); // t -> m
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // m -> y
    dag
}

#[test]
fn frontdoor_functional_matches_the_enumerated_effect_on_every_dag_coordinate() {
    let truth = frontdoor_mean(1.0) - frontdoor_mean(0.0);
    assert!((truth - 0.315).abs() < 1e-12, "functional from counts {truth}");
    on_large_stack(move || {
        let claimed = |c: &Coordinate| {
            c.graph == GraphKind::Dag
                && matches!(c.structure, Structure::Explicit | Structure::Accepted)
        };
        for_each_matching("frontdoor.functional|AverageEffect", claimed, |coordinate| {
            let builder = Study::tabular(frontdoor_data());
            let builder = match coordinate.structure {
                Structure::Explicit => builder.graph(frontdoor_dag()),
                _ => builder.graph(AcceptedGraph::from(frontdoor_dag())),
            };
            let result = run_retained(
                builder
                    .query(AverageEffectQuery::with_levels(
                        VariableId::from_raw(0),
                        VariableId::from_raw(1),
                        0.0,
                        1.0,
                    ))
                    .identifier(IdentifierId::Frontdoor)
                    .estimator(EstimatorId::FrontDoorFunctional)
                    .inference(InferenceMode::Frequentist)
                    .refute(coordinate.suite)
                    .bootstrap_replicates(0),
                &coordinate.text,
                &ExecutionContext::for_tests(43),
            );
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some("frontdoor.functional"),
                "{}",
                coordinate.text
            );
            let ate = result.estimate().as_effect().unwrap().ate;
            assert!(
                (ate - truth).abs() < 1e-9,
                "{}: ate {ate} vs functional {truth}",
                coordinate.text
            );
            match coordinate.suite {
                antecedent::RefuteSuite::None => {
                    assert!(result.refutations.is_empty(), "{}", coordinate.text);
                }
                _ => assert!(!result.refutations.is_empty(), "{}: refuters", coordinate.text),
            }
        });
    });
}
