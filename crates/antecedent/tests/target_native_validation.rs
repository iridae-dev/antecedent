//! Target-native validation through the compiler: a derivative target refuted by refits of
//! the prepared program, and a claim that refuses a scalar average-effect refutation for it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::target_validation::{
    PreparedTargetData, RefutationTarget, TargetCertificationError, TargetFamily,
    TargetRefitOptions, TargetRefutationConfig, TargetRefuterKind, TargetSuite,
    certify_result_refutations, refute_prepared_target, result_refutation_target,
    result_target_family, result_target_values,
};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, DerivativeScale, ExecutionContext, ResponseFunctional, ResponseQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_validate::RefutationReport;

/// `y = 5 + 2a − 0.5b + e`, `a` confounded by `b`; the derivative of `E[y | do(a)]` is 2.
fn fixture(n: u32) -> (TabularData, Dag) {
    let b: Vec<f64> = (0..n).map(|i| (f64::from(i) * 1.13).cos()).collect();
    let a: Vec<f64> = (0..n)
        .zip(&b)
        .map(|(i, b)| {
            2.0 + 0.5 * b + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos()
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .zip(a.iter().zip(&b))
        .map(|(i, (a, b))| 5.0 + 2.0 * a - 0.5 * b + 0.3 * (f64::from(i) * 2.39).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (source, target) in [(1, 0), (0, 2), (1, 2)] {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    (data, graph)
}

fn query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at: 2.0,
        order: 1,
        scale: DerivativeScale::Identity,
    })
}

fn prepared(data: &TabularData, graph: &Dag, ctx: &ExecutionContext) -> antecedent::PreparedStudy {
    Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(query()))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.35), ..Default::default() })
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}

#[test]
fn a_derivative_target_is_refuted_by_refits_of_its_own_program() {
    let (data, graph) = fixture(500);
    let ctx = ExecutionContext::for_tests(61);
    let prepared = prepared(&data, &graph, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result_target_family(&result), TargetFamily::ResponseDerivative);
    let (_, labels, values) = result_target_values(&result).unwrap();
    assert_eq!(labels.len(), 1);
    // The true derivative is the coefficient 2 of `a`; the local fit lands within 0.007 of it.
    assert!((values[0] - 2.0).abs() < 0.03, "derivative {values:?}");
    let target = result_refutation_target(&result).unwrap();
    assert_eq!(
        target.source(),
        result.lowered_program_identity().unwrap().as_bytes(),
        "the target is defined by the lowered program"
    );

    let config = TargetRefutationConfig { replicates: 12, ..TargetRefutationConfig::default() };
    let outcome = refute_prepared_target(
        &prepared,
        &result,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Cheap,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(outcome.plan.target(), &target);
    let placebo = outcome
        .run
        .reports
        .iter()
        .find(|r| r.kind == TargetRefuterKind::PlaceboTreatment)
        .expect("placebo report");
    assert!(placebo.passed, "{:?}", placebo.failure);
    // A permuted treatment has no effect on the outcome: the placebo derivative is near 0
    // (observed -0.06), far from the real derivative 2.
    assert!(placebo.refuted[0].abs() < 0.25, "placebo derivative {:?}", placebo.refuted);
    // The derivative diagnostics read the study's own fitted response curve.
    let refused: Vec<_> = outcome.run.refused.iter().map(|r| r.kind).collect();
    assert!(
        outcome
            .run
            .reports
            .iter()
            .any(|r| r.kind == TargetRefuterKind::FiniteDifferenceConsistency),
        "{:?}",
        outcome.run.refused
    );
    assert!(outcome.run.refused.iter().all(|r| !r.next_attempt.is_empty()));
    // Subset stability either ran on the whole vector or was refused with its reason.
    assert!(
        outcome.run.reports.iter().any(|r| r.kind == TargetRefuterKind::SubsetStability)
            || refused.contains(&TargetRefuterKind::SubsetStability)
    );
    assert!(outcome.run.refused.iter().all(|r| !r.next_attempt.is_empty()));
    assert_eq!(outcome.certificate.target, target);
    // Deterministic under the same seed.
    let again = refute_prepared_target(
        &prepared,
        &result,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Cheap,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(again.run, outcome.run);
}

#[test]
fn the_claim_refuses_a_scalar_average_effect_refutation_of_a_derivative() {
    let (data, graph) = fixture(300);
    let ctx = ExecutionContext::for_tests(62);
    let prepared = prepared(&data, &graph, &ctx);
    let clean = prepared.estimate(&data, &ctx).unwrap();
    let contract = prepared.contract().unwrap();
    if let Err(error) = clean.claim(&contract, &ctx) {
        assert!(!error.to_string().contains("cannot certify"), "{error}");
    }
    let mut certified_by_ate = clean.clone();
    certified_by_ate.refutations.push(RefutationReport::new(
        "placebo_treatment",
        2.0,
        0.0,
        0.9,
        true,
        true,
        None,
        50,
    ));
    let error = certified_by_ate.claim(&contract, &ctx).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("cannot certify a response_derivative target"), "{message}");
    assert!(message.contains("next attempt"), "{message}");
    assert!(matches!(
        certify_result_refutations(&certified_by_ate, &[]),
        Err(TargetCertificationError::ScalarRefuterForOtherTarget { .. })
    ));
}

#[test]
fn a_report_the_route_produced_for_its_own_target_is_accepted() {
    let (data, graph) = fixture(300);
    let ctx = ExecutionContext::for_tests(64);
    let prepared = prepared(&data, &graph, &ctx);
    let clean = prepared.estimate(&data, &ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let program = *clean.lowered_program_identity().unwrap().as_bytes();
    let report = || RefutationReport::new("sensitivity.evalue", 2.0, 2.0, 3.1, true, true, None, 0);

    // Stamped by the result's own program: a route-native report, whatever its name.
    let mut native = clean.clone();
    let mut own = report();
    own.stamp_producer(program);
    native.refutations.push(own);
    if let Err(error) = native.claim(&contract, &ctx) {
        assert!(!error.to_string().contains("cannot certify"), "{error}");
    }

    // Stamped by another program: refused, like an unstamped scalar report.
    let mut foreign = clean.clone();
    let mut theirs = report();
    theirs.stamp_producer([7; 32]);
    foreign.refutations.push(theirs);
    assert!(matches!(
        certify_result_refutations(&foreign, &[]),
        Err(TargetCertificationError::ScalarRefuterForOtherTarget { .. })
    ));
}

#[test]
fn a_report_on_another_program_does_not_certify_the_result() {
    let (data, graph) = fixture(300);
    let ctx = ExecutionContext::for_tests(63);
    let prepared = prepared(&data, &graph, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let config = TargetRefutationConfig { replicates: 8, ..TargetRefutationConfig::default() };
    let outcome = refute_prepared_target(
        &prepared,
        &result,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Placebo,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    )
    .unwrap();
    assert!(certify_result_refutations(&result, &outcome.run.reports).is_ok());
    // The same reports relabelled to a target defined by another program.
    let elsewhere = RefutationTarget::new(
        TargetFamily::ResponseDerivative,
        result_refutation_target(&result).unwrap().quadrant(),
        [Arc::<str>::from("value")],
        [9; 32],
    )
    .unwrap();
    let moved: Vec<_> = outcome
        .run
        .reports
        .iter()
        .cloned()
        .map(|mut report| {
            report.target = elsewhere.clone();
            report
        })
        .collect();
    assert!(matches!(
        certify_result_refutations(&result, &moved),
        Err(TargetCertificationError::TargetMismatch { .. })
    ));
}

/// `z -> t -> y`, `z -> y` over `(z, t, y)` with a deterministic, well-conditioned table.
fn attribution_fixture(n: u32) -> (TabularData, Dag) {
    let z: Vec<f64> = (0..n).map(|i| (f64::from(i) * 1.37).sin()).collect();
    let t: Vec<f64> = (0..n).zip(&z).map(|(i, z)| 0.6 * z + (f64::from(i) * 0.91).cos()).collect();
    let y: Vec<f64> = (0..n)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 1.0 + 2.0 * t + z + 0.4 * (f64::from(i) * 2.17).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (source, target) in [(0, 1), (0, 2), (1, 2)] {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    (data, graph)
}

#[test]
fn an_attribution_vector_is_refuted_by_refits_of_its_own_attribution_program() {
    let (data, graph) = attribution_fixture(96);
    let ctx = ExecutionContext::for_tests(67);
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::AnomalyAttribution(antecedent_core::AnomalyAttributionQuery::new(
            [VariableId::from_raw(2)],
            100,
        )))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result_target_family(&result), TargetFamily::Attribution);
    let (_, labels, values) = result_target_values(&result).unwrap();
    // The mean score and one mean Shapley contribution per ancestor noise component.
    assert!(labels.len() >= 2 && labels.len() == values.len(), "{labels:?}");
    let config = TargetRefutationConfig { replicates: 8, ..TargetRefutationConfig::default() };
    let outcome = refute_prepared_target(
        &prepared,
        &result,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Full,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(outcome.plan.target().family(), TargetFamily::Attribution);
    // No treatment-keyed refuter denotes for an attribution; only stability refuters run, on
    // the whole vector.
    let kinds: Vec<_> = outcome.run.reports.iter().map(|report| report.kind).collect();
    assert!(
        kinds.iter().all(|kind| matches!(
            kind,
            TargetRefuterKind::SubsetStability | TargetRefuterKind::BootstrapStability
        )),
        "{kinds:?} {:?}",
        outcome.run.refused
    );
    assert!(!kinds.is_empty(), "{:?}", outcome.run.refused);
    assert!(outcome.run.reports.iter().all(|report| report.original.len() == values.len()));
}
