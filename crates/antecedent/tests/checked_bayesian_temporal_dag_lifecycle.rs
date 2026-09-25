//! Builder-independent evidence for Bayesian fixed-DAG temporal effects.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn series(scale: f64) -> TimeSeriesData {
    let n = 2_400usize;
    let mut rng = antecedent_core::CausalRng::from_seed(77_814);
    let treatment = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect::<Vec<_>>();
    let outcome = (0..n).map(|i| {
        scale * (0.3 + 1.75 * i.checked_sub(1).map_or(0.0, |j| treatment[j])
            + 0.15 * antecedent_kernels::standard_normal(&mut rng))
    }).collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())], 1).unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(1);
    let treatment_lag = ensure_lagged(&mut graph, treatment, Lag::from_raw(1)).unwrap();
    let outcome_now = ensure_lagged(&mut graph, outcome, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(treatment_lag, outcome_now).unwrap();
    graph
}

fn query(sustained: bool) -> TemporalEffectQuery {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    if sustained {
        TemporalEffectQuery::sustained(t, y, 0, 1.0).with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::pulse(-1))
    }
}

#[test]
fn bayesian_temporal_dag_route_retains_proof_refreshes_and_exports_dependency() {
    let data = series(1.0);
    let ctx = ExecutionContext::for_tests(88_401);
    for sustained in [false, true] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let target = query(sustained);
                let structure = if accepted { AcceptedGraph::temporal_dag(graph()) } else { graph().into() };
                let builder = Study::series(data.clone())
                    .graph(structure)
                    .query(CausalQuery::TemporalEffect(target.clone()))
                    .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(384).prior_scale(100.0)))
                    .refute(suite)
                    .build().unwrap();
                let mut prepared = builder.prepare(&ctx).unwrap();
                drop(builder);

                let plan = prepared.checked_bayesian_temporal_dag_effect_info().expect("sealed Bayesian temporal DAG plan");
                assert_eq!(plan.query, target);
                assert_eq!(plan.identifier.as_str(), "temporal.backdoor.unfolded");
                assert_eq!(plan.estimator.as_str(), "bayesian.temporal.gcomp");
                assert_eq!(plan.validation, suite);
                assert_eq!(plan.posterior_draws, 384);

                let result = prepared.estimate_series(&data, &ctx).unwrap();
                assert!((result.estimate.ate - 1.75).abs() < 0.15, "sustained={sustained}, accepted={accepted}, suite={suite:?}, estimate={}", result.estimate.ate);
                assert!(result.posterior.is_some(), "posterior must be retained");

                let updated = series(1.2);
                let refreshed = prepared.refresh_series(updated, &ctx).unwrap();
                assert!((refreshed.estimate.ate - 2.1).abs() < 0.18, "refresh estimate={}", refreshed.estimate.ate);
                assert_eq!(prepared.checked_bayesian_temporal_dag_effect_info().unwrap().query, target);

                let bytes = prepared.encode_contracted_result(&refreshed, "checked-bayesian-temporal-dag", &ctx).unwrap();
                let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
                assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_bayesian_temporal_dag_effect_operation"
                }), "unresolved={:?}", consumed.acceptance.unresolved);
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

#[test]
fn checked_bayesian_temporal_dag_route_does_not_claim_multi_step_or_class_plans() {
    let data = series(1.0);
    let ctx = ExecutionContext::for_tests(88_402);
    let multi = TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1));
    let builder = Study::series(data.clone()).graph(graph()).query(CausalQuery::TemporalEffect(multi))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))).build().unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    assert!(prepared.checked_bayesian_temporal_dag_effect_info().is_none());

    let class = antecedent_graph::TemporalCpdag::empty();
    let accepted = AcceptedGraph::temporal_cpdag(class).unwrap();
    let pulse = query(false);
    let class_builder = Study::series(data).graph(accepted).query(CausalQuery::TemporalEffect(pulse))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))).build().unwrap();
    let class_prepared = class_builder.prepare(&ctx).unwrap();
    assert!(class_prepared.checked_bayesian_temporal_dag_effect_info().is_none());
}
