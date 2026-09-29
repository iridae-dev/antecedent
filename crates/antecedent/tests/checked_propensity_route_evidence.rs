//! Checked propensity lifecycle across licensed DAG estimator choices.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, ExecutionContext, PopulationRegistry, PredicateExpr, TargetPopulation,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{PropensityMatching, PropensityWeighting};
use antecedent_graph::{Dag, DenseNodeId};

fn make_data(outcome_shift: f64) -> TabularData {
    let n = 400usize;
    let z: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) / 30.0).collect();
    let t: Vec<f64> = (0..n)
        .map(|i| {
            let propensity = 1.0 / (1.0 + (-0.25 - 0.8 * z[i]).exp());
            let u = ((i * 53 % 401) as f64 + 0.5) / 401.0;
            f64::from(u < propensity)
        })
        .collect();
    let y: Vec<f64> = t
        .iter()
        .zip(&z)
        .enumerate()
        .map(|(i, (t, z))| {
            1.5 + outcome_shift + 2.0 * t + 0.7 * z + ((i * 13 % 47) as f64 - 23.0) / 100.0
        })
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

#[test]
fn propensity_estimators_retain_procedure_through_public_lifecycle() {
    let base_data = make_data(0.0);
    let graph = dag();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(711);
    let mut registry = PopulationRegistry::new();
    registry.insert_predicate(
        "named_cohort",
        (0..base_data.row_count()).step_by(2).collect::<Vec<_>>(),
    );

    for estimator in [EstimatorId::PropensityWeighting, EstimatorId::PropensityMatching] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let population = if suite == RefuteSuite::None && accepted {
                    TargetPopulation::Predicate(PredicateExpr::named("named_cohort"))
                } else {
                    TargetPopulation::AllObserved
                };
                let query = query.clone().with_target_population(population.clone());
                let mut builder = if accepted {
                    Study::tabular(base_data.clone()).graph(AcceptedGraph::from(graph.clone()))
                } else {
                    Study::tabular(base_data.clone()).graph(graph.clone())
                };
                builder = builder
                    .query(query.clone())
                    .identifier(IdentifierId::BackdoorAdjustment)
                    .population_registry(registry.clone())
                    .refute(suite);
                let builder =
                    match estimator {
                        EstimatorId::PropensityWeighting => builder
                            .estimator(PropensityWeighting::new().with_bootstrap_replicates(20)),
                        EstimatorId::PropensityMatching => builder
                            .estimator(PropensityMatching::new().with_bootstrap_replicates(20)),
                        _ => unreachable!(),
                    }
                    .build()
                    .unwrap();

                let one_shot = builder.run(&ctx).unwrap();
                let mut prepared = builder.prepare(&ctx).unwrap();
                drop(builder);
                let plan =
                    prepared.checked_propensity_info().expect("retained checked propensity plan");
                let bound_source_rows = plan.source_rows.clone();
                assert_eq!(plan.estimator, estimator);
                assert_eq!(plan.adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
                assert_eq!(plan.population, population);
                assert!(!plan.source_rows.is_empty());
                match estimator {
                    EstimatorId::PropensityWeighting => {
                        assert_eq!(
                            plan.uncertainty.as_ref(),
                            "hajek_analytic_and_optional_bootstrap"
                        );
                        assert_eq!(plan.bootstrap_replicates, Some(20));
                    }
                    EstimatorId::PropensityMatching => {
                        assert_eq!(plan.uncertainty.as_ref(), "abadie_imbens_analytic");
                        assert_eq!(plan.bootstrap_replicates, None);
                    }
                    _ => unreachable!(),
                }

                let result = prepared.estimate(&base_data, &ctx).unwrap();
                assert!(
                    (result.effect() - 2.0).abs() < 0.45,
                    "{estimator:?} effect={}",
                    result.effect()
                );
                assert!((one_shot.effect() - result.effect()).abs() < 1e-10);
                if suite != RefuteSuite::None {
                    assert!(!result.refutations.is_empty());
                }
                if estimator == EstimatorId::PropensityWeighting {
                    assert!(result.estimate.se_bootstrap.is_some());
                } else {
                    assert!(result.estimate.se_analytic.is_finite());
                    assert!(result.estimate.se_bootstrap.is_none());
                }

                let refreshed_data = make_data(4.0);
                let refreshed = prepared.refresh(refreshed_data, &ctx).unwrap();
                assert!((refreshed.effect() - 2.0).abs() < 0.45);
                assert!((refreshed.effect() - result.effect()).abs() < 0.2);
                let rebound_plan =
                    prepared.checked_propensity_info().expect("checked plan remains after refresh");
                assert_eq!(rebound_plan.source_rows, bound_source_rows);
                let bytes = prepared
                    .encode_contracted_result(&refreshed, "checked-propensity", &ctx)
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
                assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_propensity_operation"
                }));
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}
