//! The direct graph falsification `antecedent::gcm::evaluate_graph` and the compiled validation
//! operation `PreparedStudy::falsify_graph` give the same report.
//!
//! The compiled operation tests the prepared study's own DAG against the table the study
//! retained, through a lowered program of its own (program family `GraphFalsification`), and
//! calls the same kernel (`antecedent_model::falsify_graph`) with the same graph, rows,
//! configuration and execution seed; its context differs only in the lowered-dispatch scope.
//! Every number of the two reports is compared on `to_bits()`, the verdict and counts exactly.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    reason = "the test calls the direct entry the compiled operation is compared with"
)]
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "bit-identical equality is the property under test; columns come from small indices"
)]

use std::sync::Arc;

use antecedent::gcm::{GraphFalsificationConfig, GraphFalsificationReport, evaluate_graph};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, TargetPopulation, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const ROWS: usize = 400;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// `a -> b -> c -> d` with bounded wiggles. Variables `0 = a`, `1 = b`, `2 = c`, `3 = d`.
fn chain() -> (TabularData, Dag) {
    let a: Vec<f64> =
        (0..ROWS).map(|i| (i % 19) as f64 * 0.2 - 1.8 + 0.3 * (i as f64 * 1.7).sin()).collect();
    let b: Vec<f64> = (0..ROWS).map(|i| 0.9 * a[i] + 0.6 * (i as f64 * 0.37).sin()).collect();
    let c: Vec<f64> = (0..ROWS).map(|i| -0.7 * b[i] + 0.6 * (i as f64 * 0.91).cos()).collect();
    let d: Vec<f64> = (0..ROWS).map(|i| 1.2 * c[i] + 0.6 * (i as f64 * 0.53).sin()).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
        ("d", d.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn config() -> GraphFalsificationConfig {
    GraphFalsificationConfig {
        n_permutations: 40,
        min_permutations: 10,
        ci_replicates: 49,
        ..GraphFalsificationConfig::default()
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn assert_reports_equal(compiled: &GraphFalsificationReport, direct: &GraphFalsificationReport) {
    assert_eq!(compiled.outcome, direct.outcome);
    assert_eq!(compiled.uninformative_reason, direct.uninformative_reason);
    assert_eq!(compiled.n_implications, direct.n_implications);
    assert!(direct.n_implications > 0, "the chain has testable implications");
    assert_eq!(compiled.n_violations, direct.n_violations);
    assert_eq!(compiled.evidence.len(), direct.evidence.len());
    for (route, kernel) in compiled.evidence.iter().zip(&direct.evidence) {
        assert_eq!((route.node, route.other), (kernel.node, kernel.other));
        assert_eq!(route.conditioning, kernel.conditioning);
        assert_eq!(route.statistic.to_bits(), kernel.statistic.to_bits());
        assert_eq!(route.p_value.to_bits(), kernel.p_value.to_bits());
        assert_eq!(route.violated, kernel.violated);
    }
    assert_eq!(compiled.permutation_violations, direct.permutation_violations);
    assert_eq!(compiled.p_value.map(f64::to_bits), direct.p_value.map(f64::to_bits));
    assert_eq!(compiled.permutations_requested, direct.permutations_requested);
    assert_eq!(compiled.permutations_completed, direct.permutations_completed);
    assert_eq!(compiled.exhaustive, direct.exhaustive);
    assert_eq!(compiled.equivalent_permutations, direct.equivalent_permutations);
    assert_eq!(bits(&[compiled.equivalent_share]), bits(&[direct.equivalent_share]));
    assert_eq!(bits(&[compiled.max_equivalent_share]), bits(&[direct.max_equivalent_share]));
    assert_eq!(compiled.ci_test, direct.ci_test);
    assert_eq!(compiled.alpha.to_bits(), direct.alpha.to_bits());
    assert_eq!(compiled.significance.to_bits(), direct.significance.to_bits());
    assert_eq!(compiled.permutation_seed, direct.permutation_seed);
    assert_eq!(compiled.execution_seed, direct.execution_seed);
    assert_eq!(compiled.tests_run, direct.tests_run);
    assert_eq!(compiled.tests_skipped, direct.tests_skipped);
    assert_eq!(compiled.max_ci_tests, direct.max_ci_tests);
    assert_eq!(compiled.limitations, direct.limitations);
}

#[test]
fn evaluate_graph_equals_the_compiled_falsification_of_the_study_graph() {
    let (data, graph) = chain();
    let ctx = ExecutionContext::for_tests(0x5EED_6F15);
    let direct = evaluate_graph(&graph, &data, &config(), &ctx).unwrap();

    let query = AverageEffectQuery::new(
        v(0),
        v(3),
        Arc::<[VariableId]>::from([]),
        Intervention::set(v(0), Value::f64(0.0)),
        Intervention::set(v(0), Value::f64(1.0)),
        TargetPopulation::AllObserved,
    );
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::AverageEffect(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let compiled = prepared.falsify_graph(&config(), &ctx).unwrap();
    assert_eq!(compiled.lowered().route(), "GraphFalsification");
    assert_reports_equal(compiled.report(), &direct);
}
