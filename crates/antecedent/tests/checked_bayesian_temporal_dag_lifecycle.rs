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

fn lagged_pulse_series(scale: f64) -> TimeSeriesData {
    let n = 2_800usize;
    let mut rng = antecedent_core::CausalRng::from_seed(88_403);
    let treatment = (0..n)
        .map(|_| antecedent_kernels::standard_normal(&mut rng))
        .collect::<Vec<_>>();
    let outcome = (0..n)
        .map(|i| {
            scale
                * (2.0 * i.checked_sub(1).map_or(0.0, |j| treatment[j])
                    + 4.0 * i.checked_sub(2).map_or(0.0, |j| treatment[j])
                    + 0.12 * antecedent_kernels::standard_normal(&mut rng))
        })
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns(
        [("t", treatment.as_slice()), ("y", outcome.as_slice())],
        1,
    )
    .unwrap()
}

fn two_lag_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(1);
    let treatment_lag1 = ensure_lagged(&mut graph, treatment, Lag::from_raw(1)).unwrap();
    let treatment_lag2 = ensure_lagged(&mut graph, treatment, Lag::from_raw(2)).unwrap();
    let outcome_now = ensure_lagged(&mut graph, outcome, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(treatment_lag1, outcome_now).unwrap();
    graph.insert_directed(treatment_lag2, outcome_now).unwrap();
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

fn multi_step_sustained_query() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        -2,
        1.0,
    )
    .with_policy(TemporalPolicy::sustained(-2, -1))
}

#[test]
fn bayesian_temporal_dag_route_retains_proof_refreshes_and_exports_dependency() {
    let ctx = ExecutionContext::for_tests(88_401);
    let routes = [
        (series(1.0), graph(), query(false), 1.75, series(1.2), 2.1),
        (series(1.0), graph(), query(true), 1.75, series(1.2), 2.1),
        (
            lagged_pulse_series(1.0),
            two_lag_graph(),
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1),
            2.0,
            lagged_pulse_series(1.2),
            2.4,
        ),
        (
            lagged_pulse_series(1.0),
            two_lag_graph(),
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(2),
            4.0,
            lagged_pulse_series(1.2),
            4.8,
        ),
        (
            series(1.0),
            graph(),
            multi_step_sustained_query(),
            1.75,
            series(1.2),
            2.1,
        ),
    ];
    for (data, graph, target, truth, updated_data, refreshed_truth) in routes {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let structure = if accepted {
                    AcceptedGraph::temporal_dag(graph.clone())
                } else {
                    graph.clone().into()
                };
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
                assert_eq!(
                    plan.estimator.as_str(),
                    if target.is_multi_step_sustained() {
                        "temporal.sequential.gcomp"
                    } else {
                        "bayesian.temporal.gcomp"
                    }
                );
                assert_eq!(plan.validation, suite);
                assert_eq!(plan.posterior_draws, 384);

                let result = prepared.estimate_series(&data, &ctx).unwrap();
                assert!(
                    (result.estimate.ate - truth).abs() < 0.16,
                    "query={target:?}, accepted={accepted}, suite={suite:?}, estimate={}",
                    result.estimate.ate
                );
                assert!(result.posterior.is_some(), "posterior must be retained");

                let refreshed = prepared.refresh_series(updated_data.clone(), &ctx).unwrap();
                assert!(
                    (refreshed.estimate.ate - refreshed_truth).abs() < 0.2,
                    "refresh estimate={}",
                    refreshed.estimate.ate
                );
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
fn checked_bayesian_temporal_dag_route_does_not_claim_class_plans() {
    let data = series(1.0);
    let ctx = ExecutionContext::for_tests(88_402);
    let class = antecedent_graph::TemporalCpdag::empty();
    let accepted = AcceptedGraph::temporal_cpdag(class).unwrap();
    let pulse = query(false);
    let class_builder = Study::series(data).graph(accepted).query(CausalQuery::TemporalEffect(pulse))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))).build().unwrap();
    let class_prepared = class_builder.prepare(&ctx).unwrap();
    assert!(class_prepared.checked_bayesian_temporal_dag_effect_info().is_none());
}
