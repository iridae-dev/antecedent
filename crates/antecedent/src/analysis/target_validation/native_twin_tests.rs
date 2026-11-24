//! Twin equality of the target-native check kernels and the compiled refuters that run them.
//!
//! The deprecated direct entries (`finite_difference_consistency`, `local_linearity`,
//! `posterior_predictive_target_check`) evaluate a caller-supplied response or caller-supplied
//! draws. The compiled refuter (`refute_prepared_target`) calls the same kernel on the study's
//! own fitted response and reference-model draws. Each test reads that response, or those draws,
//! through the refit the compiled path builds and hands them to the kernel directly, then
//! asserts that the compiled report equals the kernel's: the original and refuted vectors by
//! `to_bits()`, the joint statistic, and the verdict.
//!
//! Seeds: both paths run under one context; the reference-model draws come from the refuter's
//! own keyed streams (`target_refuter_stream`), so the replicates are the same draws.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::RefuteSuite;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::Dag;

/// `y = 5 + 2a - 0.5b + e`, `a` confounded by `b`; the derivative of `E[y | do(a)]` is 2.
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

fn derivative_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at: 2.0,
        order: 1,
        scale: DerivativeScale::Identity,
    })
}

fn prepared(data: &TabularData, graph: &Dag, ctx: &ExecutionContext) -> PreparedStudy {
    crate::Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(derivative_query()))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.35), ..Default::default() })
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}

/// The refit `refute_prepared_target` builds for `result`.
fn refit<'a>(
    prepared: &'a PreparedStudy,
    result: &'a StudyResult,
    data: &'a TabularData,
    config: &TargetRefutationConfig,
    ctx: &ExecutionContext,
) -> PreparedRefit<'a> {
    let target = result_refutation_target(result).unwrap();
    let (_, _, original) = result_target_values(result).unwrap();
    PreparedRefit {
        prepared,
        result,
        data: PreparedTargetData::Tabular(data),
        target,
        member: None,
        temporal_budget: prepared.study().temporal_validation_budget.clone().unwrap_or_else(|| {
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
        }),
        original,
        treatment: result.treatment.unwrap(),
        outcome: result.outcome,
        options: TargetRefitOptions::default(),
        replicates: config.replicates,
        alpha: config.alpha,
        curve: std::cell::OnceCell::new(),
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn same_report(compiled: &TargetRefutationReport, direct: &TargetRefutationReport) {
    assert_eq!(compiled.kind, direct.kind);
    assert_eq!(bits(&compiled.original), bits(&direct.original), "original");
    assert_eq!(bits(&compiled.refuted), bits(&direct.refuted), "refuted");
    assert_eq!(compiled.statistic, direct.statistic, "statistic");
    assert_eq!(compiled.passed, direct.passed, "passed");
    assert_eq!(compiled.informative, direct.informative, "informative");
}

#[test]
fn the_compiled_finite_difference_check_is_the_kernel_on_the_studys_own_response() {
    let (data, graph) = fixture(500);
    let ctx = ExecutionContext::for_tests(61);
    let prepared = prepared(&data, &graph, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
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
    let compiled = outcome
        .run
        .reports
        .iter()
        .find(|r| r.kind == TargetRefuterKind::FiniteDifferenceConsistency)
        .expect("the cheap suite runs the finite-difference check on a derivative");

    let direct_refit = refit(&prepared, &result, &data, &config, &ctx);
    let (nodes, reported, step) = direct_refit.response_curve(&ctx).unwrap();
    assert_eq!(reported.len(), 1, "a point derivative has one horizon");
    let direct = finite_difference_consistency(
        |offset| direct_refit.node_mean(&nodes, offset, 0),
        &[DerivativeCheckPoint { x: 0.0, reported: reported[0] }],
        step,
        direct_refit.options.derivative_tolerance,
    )
    .unwrap();
    let direct = direct_refit
        .merge_coordinates(
            TargetRefuterKind::FiniteDifferenceConsistency,
            vec![DerivativeCoordinate::FiniteDifference(direct)],
        )
        .unwrap();
    same_report(compiled, &direct);
}

#[test]
fn the_compiled_local_linearity_check_is_the_kernel_on_the_studys_own_response() {
    let (data, graph) = fixture(500);
    let ctx = ExecutionContext::for_tests(67);
    let prepared = prepared(&data, &graph, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let config = TargetRefutationConfig::default();

    let compiled = refit(&prepared, &result, &data, &config, &ctx)
        .derivative_check(TargetRefuterKind::LocalLinearity, &ctx)
        .unwrap();

    let direct_refit = refit(&prepared, &result, &data, &config, &ctx);
    let (nodes, _, step) = direct_refit.response_curve(&ctx).unwrap();
    let direct = local_linearity(
        |offset| direct_refit.node_mean(&nodes, offset, 0),
        &[0.0],
        step,
        direct_refit.options.linearity_tolerance,
        1e-9,
    )
    .unwrap();
    let direct = direct_refit
        .merge_coordinates(
            TargetRefuterKind::LocalLinearity,
            vec![DerivativeCoordinate::Linearity(direct)],
        )
        .unwrap();
    same_report(&compiled, &direct);
}

#[test]
fn the_compiled_posterior_predictive_check_is_the_kernel_on_the_studys_own_draws() {
    let (data, graph) = fixture(300);
    let ctx = ExecutionContext::for_tests(71);
    let prepared = prepared(&data, &graph, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let config = TargetRefutationConfig { replicates: 6, ..TargetRefutationConfig::default() };

    let compiled =
        refit(&prepared, &result, &data, &config, &ctx).posterior_predictive(&ctx).unwrap();

    let direct_refit = refit(&prepared, &result, &data, &config, &ctx);
    let (draws, replicated) = direct_refit.tabular_predictive_replicates(&data, &ctx).unwrap();
    let direct = posterior_predictive_target_check(&PosteriorPredictiveTarget {
        target: &direct_refit.target,
        observed: &direct_refit.original,
        draws: &draws,
        replicated: &replicated,
        alpha: direct_refit.alpha,
    })
    .unwrap();
    same_report(&compiled, &direct);
}
