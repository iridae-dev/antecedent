//! Builder-independent evidence for Bayesian conditional effects on static CPDAG and PAG
//! classes whose every completion identifies the same adjustment target.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, GraphClass, IdentifierId, InferenceMode,
    RefuteSuite, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, DenseNodeId, Pag};
use antecedent_validate::PredictiveCheckKind;

enum ClassGraph {
    Cpdag(Cpdag),
    Pag(Pag),
}

/// Six-variable classes: `t -> y`, `w -> y`, `z -> t`, `z -> y`, plus completions
/// that never touch the adjustment target. The CPDAG leaves `u - v` undirected; the
/// PAG leaves `z o-> t` and `u o-> t` circle-marked. Every completion identifies the
/// conditional effect by adjusting for `z`.
fn class_graphs() -> [(GraphClass, ClassGraph); 2] {
    let mut cpdag = Cpdag::with_variables(6);
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_undirected(DenseNodeId::from_raw(4), DenseNodeId::from_raw(5)).unwrap();
    let mut pag = Pag::with_variables(6);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    for source in [3, 4] {
        pag.insert_marked(antecedent_graph::MarkedEdge {
            a: DenseNodeId::from_raw(source),
            b: DenseNodeId::from_raw(0),
            at_a: antecedent_graph::Endpoint::Circle,
            at_b: antecedent_graph::Endpoint::Arrow,
            middle: antecedent_graph::MiddleMark::Empty,
        })
        .unwrap();
    }
    [(GraphClass::Cpdag, ClassGraph::Cpdag(cpdag)), (GraphClass::Pag, ClassGraph::Pag(pag))]
}

/// Deterministic linear SCM: `y = 1 + shift + 0.25 z + 2 t + 0.5 t w` with balanced
/// `w` in 0..5, so the conditional contrast averages to exactly 3 over the sample.
fn class_data(outcome_shift: f64) -> TabularData {
    let n = 600usize;
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let u: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let v: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
    let y: Vec<f64> = t
        .iter()
        .zip(&w)
        .zip(&z)
        .map(|((t, w), z)| 1.0 + outcome_shift + 0.25 * z + 2.0 * t + 0.5 * t * w)
        .collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("w", w.as_slice()),
        ("z", z.as_slice()),
        ("u", u.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap()
}

/// The same rows with `t` and `w` swapped: variable IDs change, so a retained plan
/// must refuse to bind it.
fn reordered_class_data() -> TabularData {
    let n = 600usize;
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let u: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let v: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
    let y: Vec<f64> = t
        .iter()
        .zip(&w)
        .zip(&z)
        .map(|((t, w), z)| 1.0 + 0.25 * z + 2.0 * t + 0.5 * t * w)
        .collect();
    TabularData::from_f64_columns([
        ("w", w.as_slice()),
        ("y", y.as_slice()),
        ("t", t.as_slice()),
        ("z", z.as_slice()),
        ("u", u.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap()
}

fn conditional_query() -> ConditionalEffectQuery {
    ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap()
}

fn assert_validation(result: &antecedent::StudyResult, suite: RefuteSuite, label: &str) {
    if suite == RefuteSuite::None {
        assert!(result.refutations.is_empty(), "{label}: validation none must emit no reports");
        assert!(result.predictive_checks.is_empty(), "{label}: validation none must not run PPC");
        return;
    }
    assert!(!result.refutations.is_empty(), "{label}: {suite:?} must execute a refuter");
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior),
        "{label}: {suite:?} must attach prior PPC"
    );
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Posterior),
        "{label}: {suite:?} must attach posterior PPC"
    );
    let posterior = result.posterior.as_ref().expect("Bayesian class posterior");
    assert_eq!(
        posterior.prior_sensitivity.is_some(),
        suite == RefuteSuite::Full,
        "{label}: prior sensitivity is attached exactly under full validation"
    );
}

#[test]
fn bayesian_class_conditional_effect_is_sealed_across_lifecycle() {
    run_on_large_stack(|| {
        let original = class_data(0.0);
        let ctx = ExecutionContext::for_tests(913);
        let conditional = conditional_query();
        let config = BayesianConfig::conjugate().n_draws(96).prior_scale(30.0);
        for (expected_class, graph) in class_graphs() {
            for accepted in [false, true] {
                for (suite_label, suite) in [
                    ("none", RefuteSuite::None),
                    ("cheap", RefuteSuite::Cheap),
                    ("full", RefuteSuite::Full),
                ] {
                    let coordinate = format!(
                        "ConditionalEffect:{expected_class:?}:{}:Bayesian:{suite_label}",
                        if accepted { "accepted" } else { "explicit" }
                    );
                    let base = Study::tabular(original.clone());
                    let base = match (&graph, accepted) {
                        (ClassGraph::Cpdag(graph), true) => {
                            base.graph(AcceptedGraph::from(graph.clone()))
                        }
                        (ClassGraph::Cpdag(graph), false) => base.graph(graph.clone()),
                        (ClassGraph::Pag(graph), true) => {
                            base.graph(AcceptedGraph::from(graph.clone()))
                        }
                        (ClassGraph::Pag(graph), false) => base.graph(graph.clone()),
                    };
                    let builder = base
                        .query(CausalQuery::ConditionalEffect(conditional.clone()))
                        .identifier(IdentifierId::GeneralizedAdjustment)
                        .inference(InferenceMode::Bayesian(config.clone()))
                        .refute(suite)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap();
                    let one_shot = builder.clone().run(&ctx).unwrap();
                    let mut prepared = builder.prepare(&ctx).unwrap();
                    drop(builder);

                    let plan = prepared
                        .checked_bayesian_class_conditional_info()
                        .expect("fully identified Bayesian class conditional operation is sealed");
                    assert!(prepared.checked_conditional_effect_info().is_none(), "{coordinate}");
                    assert!(
                        prepared.checked_bayesian_conditional_operation().is_none(),
                        "{coordinate}"
                    );
                    assert_eq!(plan.graph_class, expected_class, "{coordinate}");
                    assert_eq!(plan.query, conditional, "{coordinate}");
                    assert_eq!(
                        plan.identifier,
                        IdentifierId::GeneralizedAdjustment,
                        "{coordinate}"
                    );
                    assert_eq!(plan.estimator, EstimatorId::BayesianConditional, "{coordinate}");
                    assert_eq!(
                        plan.inference,
                        InferenceMode::Bayesian(config.clone()),
                        "{coordinate}"
                    );
                    assert_eq!(plan.validation, suite, "{coordinate}");
                    assert_eq!(
                        plan.adjustment_set.as_ref(),
                        &[VariableId::from_raw(3)],
                        "{coordinate}"
                    );
                    assert!(plan.completion_count > 1, "{coordinate}: {plan:?}");
                    assert!(
                        (plan.identified_mass - plan.completion_count as f64).abs() < 1e-12,
                        "{coordinate}: every completion must carry identified mass: {plan:?}"
                    );
                    assert!(plan.unresolved_mass.abs() < 1e-12, "{coordinate}: {plan:?}");
                    assert_eq!(
                        prepared.plan().logical.record.estimator.as_deref(),
                        Some("conditional.bayesian"),
                        "{coordinate}"
                    );

                    let result = prepared.estimate(&original, &ctx).unwrap();
                    // Independent linear SCM truth: E[2 + 0.5 W] for balanced W = 0..4.
                    assert!(
                        (result.effect() - 3.0).abs() < 0.2,
                        "{coordinate}: {}",
                        result.effect()
                    );
                    assert_eq!(
                        result.logical_plan.estimator.as_deref(),
                        Some("conditional.bayesian")
                    );
                    assert_eq!(
                        result.posterior.as_ref().unwrap().draws.n_draws,
                        96,
                        "{coordinate}"
                    );
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "exec.identify.cached"),
                        "{coordinate}: click must execute the retained proof"
                    );
                    assert_validation(&result, suite, &coordinate);
                    assert_eq!(
                        one_shot.effect().to_bits(),
                        result.effect().to_bits(),
                        "{coordinate}: one-shot and prepared clicks must agree bitwise"
                    );

                    let refreshed = prepared.refresh(class_data(0.35), &ctx).unwrap();
                    assert!((refreshed.effect() - 3.0).abs() < 0.2, "{coordinate}");
                    let rebound = prepared.checked_bayesian_class_conditional_info().unwrap();
                    assert_eq!(rebound.graph_class, plan.graph_class);
                    assert_eq!(rebound.query, plan.query);
                    assert_eq!(rebound.inference, plan.inference);
                    assert_eq!(rebound.adjustment_set, plan.adjustment_set);
                    assert_eq!(rebound.completion_count, plan.completion_count);
                    assert!(
                        prepared.refresh(reordered_class_data(), &ctx).is_err(),
                        "{coordinate}: schema-changing refresh must be refused"
                    );

                    let artifact = prepared
                        .encode_contracted_result(
                            &refreshed,
                            &format!("checked-bayesian-class-conditional-{suite_label}"),
                            &ctx,
                        )
                        .unwrap();
                    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                    assert!(consumed.contract.is_some());
                    assert!(
                        consumed.acceptance.unresolved.iter().any(|dependency| {
                            dependency.as_ref()
                                == "dependencies.checked_bayesian_class_conditional_operation"
                        }),
                        "{coordinate}: {:?}",
                        consumed.acceptance.unresolved
                    );
                    assert!(!consumed.acceptance.accepts_as_verified_program());
                }
            }
        }
    });
}

/// A CPDAG whose completions disagree on the adjustment target (`z - t` may point
/// either way) is partially identified and stays outside the point-estimate plan.
#[test]
fn partially_identified_bayesian_class_conditional_effect_is_not_sealed() {
    let data = class_data(0.0);
    let mut cpdag = Cpdag::with_variables(6);
    cpdag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_undirected(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    let ctx = ExecutionContext::for_tests(914);
    let builder = Study::tabular(data.clone())
        .graph(cpdag)
        .query(CausalQuery::ConditionalEffect(conditional_query()))
        .identifier(IdentifierId::GeneralizedAdjustment)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    drop(builder);
    assert!(prepared.checked_bayesian_class_conditional_info().is_none());
    assert!(prepared.checked_conditional_effect_info().is_none());
    assert!(prepared.checked_bayesian_conditional_operation().is_none());
}

/// Class envelopes execute through deep prepare/estimate frames in debug builds.
fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-bayesian-class-evidence".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
