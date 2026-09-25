//! Builder-independent execution evidence for the checked DAG conditional-effect family.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, OutcomeFunctional,
    StreamDomain, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

fn data(outcome_shift: f64) -> TabularData {
    data_n(400, outcome_shift)
}

fn data_n(n: usize, outcome_shift: f64) -> TabularData {
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let y: Vec<f64> =
        t.iter().zip(&w).map(|(t, w)| 1.0 + outcome_shift + 2.0 * t + 0.5 * t * w).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())])
        .unwrap()
}

fn graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn quantile_data() -> TabularData {
    let mut rng = ExecutionContext::for_tests(615).rng.stream_for(StreamDomain::Test, 0x51);
    let mut t = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for _ in 0..2400 {
        let modifier = standard_normal(&mut rng);
        let treatment = f64::from(rng.next_f64() < 0.5);
        t.push(treatment);
        w.push(modifier);
        y.push(2.0 * treatment + 0.3 * modifier + 0.5 * standard_normal(&mut rng));
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())])
        .unwrap()
}

fn query(functional: OutcomeFunctional) -> ConditionalEffectQuery {
    ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)])
            .with_outcome_functional(functional),
    )
    .unwrap()
}

#[test]
fn frequentist_conditional_effect_keeps_six_dag_choices_sealed_through_refresh() {
    let original = data(0.0);
    let graph = graph();
    let ctx = ExecutionContext::for_tests(817);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let route_builder = Study::tabular(original.clone());
            let route_builder = if accepted {
                route_builder.graph(AcceptedGraph::from(graph.clone()))
            } else {
                route_builder.graph(graph.clone())
            };
            let builder = route_builder
                .query(CausalQuery::ConditionalEffect(query(OutcomeFunctional::Mean)))
                .identifier(IdentifierId::BackdoorAdjustment)
                .estimator(EstimatorId::ConditionalLinearAdjustment)
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let one_shot = builder.run(&ctx).unwrap();
            let mut prepared = builder.prepare(&ctx).unwrap();
            drop(builder);
            let plan = prepared
                .checked_conditional_effect_info()
                .expect("retained conditional target and lowering");
            assert_eq!(plan.identifier, IdentifierId::BackdoorAdjustment);
            assert_eq!(plan.estimator, EstimatorId::ConditionalLinearAdjustment);
            assert_eq!(plan.procedure.as_ref(), "linear_interaction_plugin");
            assert_eq!(plan.validation, suite);
            assert_eq!(
                plan.design_roles.as_ref(),
                &[VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2),]
            );
            assert_eq!(plan.source_rows.len(), original.row_count());
            let result = prepared.estimate(&original, &ctx).unwrap();
            // Independent structural truth: E[2 + 0.5 W] with balanced W=0..4.
            assert!((result.effect() - 3.0).abs() < 1e-9);
            assert!((one_shot.effect() - result.effect()).abs() < 1e-10);
            assert_eq!(result.refutations.is_empty(), suite == RefuteSuite::None);
            let refreshed = prepared.refresh(data(0.4), &ctx).unwrap();
            assert!((refreshed.effect() - 3.0).abs() < 1e-9);
            let rebound = prepared.checked_conditional_effect_info().unwrap();
            assert_eq!(rebound.query, plan.query);
            assert_eq!(rebound.procedure, plan.procedure);
            let smaller = prepared.refresh(data_n(200, 0.4), &ctx).unwrap();
            assert!((smaller.effect() - 3.0).abs() < 1e-9);
            assert_eq!(prepared.checked_conditional_effect_info().unwrap().source_rows.len(), 200);
            let artifact =
                prepared.encode_contracted_result(&smaller, "checked-conditional", &ctx).unwrap();
            let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
            assert!(consumed.contract.is_some());
            assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
                dependency.as_ref() == "dependencies.checked_conditional_effect_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}

#[test]
fn conditional_refresh_refuses_column_reordering_that_changes_semantic_ids() {
    let original = data(0.0);
    let ctx = ExecutionContext::for_tests(819);
    let builder = Study::tabular(original)
        .graph(graph())
        .query(CausalQuery::ConditionalEffect(query(OutcomeFunctional::Mean)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut prepared = builder.prepare(&ctx).unwrap();
    drop(builder);
    let n = 400usize;
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let y: Vec<f64> = t.iter().zip(&w).map(|(t, w)| 1.0 + 2.0 * t + 0.5 * t * w).collect();
    let reordered = TabularData::from_f64_columns([
        ("w", w.as_slice()),
        ("y", y.as_slice()),
        ("t", t.as_slice()),
    ])
    .unwrap();
    assert!(prepared.refresh(reordered, &ctx).is_err());
}

#[test]
fn conditional_grid_and_quantile_retain_distribution_score_procedure() {
    let original = data(0.0);
    let ctx = ExecutionContext::for_tests(818);
    for functional in
        [OutcomeFunctional::exceedance_grid([1.5, 2.5, 3.5]), OutcomeFunctional::quantile(0.5)]
    {
        let route_data = if matches!(functional, OutcomeFunctional::Quantile(_)) {
            quantile_data()
        } else {
            original.clone()
        };
        let builder = Study::tabular(route_data.clone())
            .graph(graph())
            .query(CausalQuery::ConditionalEffect(query(functional.clone())))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);
        let plan = prepared.checked_conditional_effect_info().unwrap();
        assert_eq!(plan.procedure.as_ref(), "crossfit_aipw_distribution_scores");
        assert_eq!(plan.query.inner.outcome_functional, functional);
        let result = prepared.estimate(&route_data, &ctx).unwrap();
        assert!(result.estimate.exceedance_cdf.is_some());
        assert!(result.estimate.joint_covariance.is_some());
        let refreshed_data = if matches!(functional, OutcomeFunctional::Quantile(_)) {
            quantile_data()
        } else {
            data(0.2)
        };
        let refreshed = prepared.refresh(refreshed_data, &ctx).unwrap();
        assert_eq!(prepared.checked_conditional_effect_info().unwrap().query, plan.query);
        assert!(refreshed.estimate.exceedance_cdf.is_some());
    }
}
