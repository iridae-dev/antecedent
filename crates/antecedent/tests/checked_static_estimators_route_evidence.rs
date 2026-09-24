//! Checked GLM and sharp-RD high-level lifecycles against analytic truth.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, ExecutionContext, PopulationRegistry, PredicateExpr, TargetPopulation,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

#[test]
fn checked_glm_accepted_and_explicit_retain_design_and_dependency_refusal() {
    let z: Vec<f64> = (0..200).map(|i| (f64::from(i) * 0.19).sin()).collect();
    let t: Vec<f64> = (0..200).map(|i| f64::from(i % 2)).collect();
    let y: Vec<f64> = t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + 0.4 * z).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let graph = graph();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(71);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let mut registry = PopulationRegistry::new();
            registry.insert_predicate("cohort", (0..200).step_by(3).collect::<Vec<_>>());
            for population in [
                TargetPopulation::AllObserved,
                TargetPopulation::Treated,
                TargetPopulation::Untreated,
                TargetPopulation::Predicate(PredicateExpr::rows(
                    (0..200).step_by(3).collect::<Vec<_>>(),
                )),
                TargetPopulation::Predicate(PredicateExpr::named("cohort")),
            ] {
                let mut fitter = antecedent_estimate::GlmAdjustmentAte::new()
                    .with_family(antecedent_stats::GlmFamily::GaussianIdentity);
                fitter.bootstrap_replicates = 0;
                let builder = if accepted {
                    Study::tabular(data.clone()).graph(AcceptedGraph::from(graph.clone()))
                } else {
                    Study::tabular(data.clone()).graph(graph.clone())
                }
                .query(query.clone().with_target_population(population.clone()))
                .population_registry(registry.clone())
                .estimator(fitter)
                .refute(suite)
                .build()
                .unwrap();
                let one_shot = builder.run(&ctx).unwrap();
                let mut prepared = builder.prepare(&ctx).unwrap();
                drop(builder);
                let plan = prepared.checked_glm_adjustment().expect("retained checked GLM plan");
                assert_eq!(plan.active, 1.0);
                assert_eq!(plan.control, 0.0);
                assert_eq!(plan.adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
                assert_eq!(plan.target_population, population);
                let result = prepared.estimate(&data, &ctx).unwrap();
                assert!((result.effect() - 2.0).abs() < 0.05, "effect={}", result.effect());
                if suite != RefuteSuite::None {
                    assert!(!result.refutations.is_empty());
                }
                assert!((one_shot.effect() - result.effect()).abs() < 1e-10);
                let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
                assert!((refreshed.effect() - result.effect()).abs() < 1e-10);
                let bytes =
                    prepared.encode_contracted_result(&refreshed, "checked-glm", &ctx).unwrap();
                let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
                assert!(
                    consumed
                        .acceptance
                        .unresolved
                        .iter()
                        .any(|reason| reason.as_ref() == "dependencies.checked_glm_operation")
                );
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

#[test]
fn checked_rd_accepted_and_explicit_retain_cutoff_and_dependency_refusal() {
    let running: Vec<f64> = (0..200).map(|i| (f64::from(i) - 99.5) / 100.0).collect();
    let treatment: Vec<f64> = running.iter().map(|r| f64::from(*r >= 0.0)).collect();
    let outcome: Vec<f64> =
        running.iter().zip(&treatment).map(|(r, t)| 1.0 + 0.4 * r + 2.0 * t).collect();
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", running.as_slice()),
    ])
    .unwrap();
    let graph = graph();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(72);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let builder = if accepted {
                Study::tabular(data.clone()).graph(AcceptedGraph::from(graph.clone()))
            } else {
                Study::tabular(data.clone()).graph(graph.clone())
            }
            .query(query.clone())
            .identifier(IdentifierId::RdSharp)
            .estimator(EstimatorId::RdSharp)
            .rd_config(VariableId::from_raw(2), 0.0, 0.8)
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
            let one_shot = builder.run(&ctx).unwrap();
            let mut prepared = builder.prepare(&ctx).unwrap();
            drop(builder);
            let plan = prepared.checked_rd_preparation().expect("retained checked RD plan");
            assert_eq!(plan.lowering().bandwidth, 0.8);
            assert_eq!(plan.lowering().cutoff, 0.0);
            let result = prepared.estimate(&data, &ctx).unwrap();
            assert!((result.effect() - 2.0).abs() < 1e-8, "effect={}", result.effect());
            if suite != RefuteSuite::None {
                assert!(!result.refutations.is_empty());
            }
            assert!((one_shot.effect() - result.effect()).abs() < 1e-10);
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            assert!((refreshed.effect() - result.effect()).abs() < 1e-10);
            let bytes = prepared.encode_contracted_result(&refreshed, "checked-rd", &ctx).unwrap();
            let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
            assert!(
                consumed
                    .acceptance
                    .unresolved
                    .iter()
                    .any(|reason| reason.as_ref() == "dependencies.checked_rd_operation")
            );
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}
