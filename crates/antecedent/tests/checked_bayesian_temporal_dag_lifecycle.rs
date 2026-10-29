//! Builder-independent evidence for Bayesian fixed-DAG temporal effects.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn series(scale: f64) -> TimeSeriesData {
    let n = 2_400usize;
    let mut rng = antecedent_core::CausalRng::from_seed(77_814);
    let treatment =
        (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect::<Vec<_>>();
    let outcome = (0..n)
        .map(|i| {
            scale
                * (0.3
                    + 1.75 * i.checked_sub(1).map_or(0.0, |j| treatment[j])
                    + 0.15 * antecedent_kernels::standard_normal(&mut rng))
        })
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())], 1)
        .unwrap()
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
    let treatment =
        (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect::<Vec<_>>();
    let outcome = (0..n)
        .map(|i| {
            scale
                * (2.0 * i.checked_sub(1).map_or(0.0, |j| treatment[j])
                    + 4.0 * i.checked_sub(2).map_or(0.0, |j| treatment[j])
                    + 0.12 * antecedent_kernels::standard_normal(&mut rng))
        })
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())], 1)
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
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
}

// The two-lag fixtures y_i = s (2 t_{i-1} + 4 t_{i-2} + noise) are fitted at each horizon by a
// regression on one lag alone (the other lag is an independent omitted regressor, which is
// unbiased but adds its variance to the residual). The estimate is therefore the marginal slope
// of the exact rows, not the analytic coefficient: it sits 0.94 and 0.89 of its own standard
// error above 2 and 4 (0.071 and 0.034; posterior sd 0.073 and 0.037; scaled 1.2: 0.085 and
// 0.041), a finite-sample draw proportional to the data scale and not a bias. The route returns
// that slope less a conjugate-prior and tempering shrinkage of at most 0.002.
const TWO_LAG_PRIOR_BAND: f64 = 0.003;

#[test]
fn two_lag_references_lie_within_one_standard_error_of_the_analytic_coefficients() {
    for (reference, truth, standard_error) in [
        // (slope, coefficient, standard error)
        (2.070_930_485_854_101_4_f64, 2.0, 0.075_650_511_117),
        (4.033_811_131_024_400_2, 4.0, 0.037_950_373_259),
        (2.485_116_583_024_921_6, 2.4, 0.090_780_613_340),
        (4.840_573_357_229_280_1, 4.8, 0.045_540_447_911),
    ] {
        let gap: f64 = reference - truth;
        assert!(gap.abs() < standard_error, "{reference} vs {truth}");
    }
}

#[test]
fn bayesian_temporal_dag_route_retains_proof_refreshes_and_exports_dependency() {
    let ctx = ExecutionContext::for_tests(88_401);
    let routes = [
        (series(1.0), graph(), query(false), 1.75, series(1.2), 2.1, 0.01),
        (series(1.0), graph(), query(true), 1.75, series(1.2), 2.1, 0.01),
        (
            lagged_pulse_series(1.0),
            two_lag_graph(),
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1),
            // Marginal slope of the exact rows (bayesian_temporal_two_lag_reference.py); the
            // analytic coefficients are 2.0 and 2.4. See the note on the two-lag fixtures below.
            2.070_930_485_854_101_4,
            lagged_pulse_series(1.2),
            2.485_116_583_024_921_6,
            TWO_LAG_PRIOR_BAND,
        ),
        (
            lagged_pulse_series(1.0),
            two_lag_graph(),
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(2),
            4.033_811_131_024_400_2,
            lagged_pulse_series(1.2),
            4.840_573_357_229_280_1,
            TWO_LAG_PRIOR_BAND,
        ),
        (series(1.0), graph(), multi_step_sustained_query(), 1.75, series(1.2), 2.1, 0.01),
    ];
    for (data, graph, target, truth, updated_data, refreshed_truth, tolerance) in routes {
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
                    .inference(InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(384).prior_scale(100.0),
                    ))
                    .refute(suite)
                    .build()
                    .unwrap();
                let mut prepared = builder.prepare(&ctx).unwrap();
                drop(builder);

                let RetainedOperation::BayesianTemporalDagEffect(plan) =
                    prepared.retained_operation()
                else {
                    panic!("sealed Bayesian temporal DAG plan")
                };
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
                    (result.estimate().as_effect().unwrap().ate - truth).abs() < tolerance,
                    "query={target:?}, accepted={accepted}, suite={suite:?}, estimate={}",
                    result.estimate().as_effect().unwrap().ate
                );
                assert!(result.posterior.is_some(), "posterior must be retained");

                let refreshed = prepared.refresh_series(updated_data.clone(), &ctx).unwrap();
                assert!(
                    (refreshed.estimate().as_effect().unwrap().ate - refreshed_truth).abs() < tolerance,
                    "refresh estimate={}",
                    refreshed.estimate().as_effect().unwrap().ate
                );
                let RetainedOperation::BayesianTemporalDagEffect(retained) =
                    prepared.retained_operation()
                else {
                    panic!("sealed Bayesian temporal DAG plan after refresh")
                };
                assert_eq!(retained.query, target);

                let bytes = prepared
                    .encode_contracted_result(&refreshed, "checked-bayesian-temporal-dag", &ctx)
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
                assert!(
                    consumed.acceptance.unresolved.iter().any(|reason| {
                        reason.as_ref()
                            == "dependencies.checked_bayesian_temporal_dag_effect_operation"
                    }),
                    "unresolved={:?}",
                    consumed.acceptance.unresolved
                );
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

#[test]
fn checked_bayesian_temporal_dag_route_does_not_claim_class_plans() {
    let data = series(1.0);
    let ctx = ExecutionContext::for_tests(88_402);
    let mut class = antecedent_graph::TemporalCpdag::empty();
    let treatment_lag = class.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let outcome_now = class.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    class.insert_directed(treatment_lag, outcome_now).unwrap();
    let accepted = AcceptedGraph::temporal_cpdag(class).unwrap();
    let pulse = query(false);
    let class_builder = Study::series(data)
        .graph(accepted)
        .query(CausalQuery::TemporalEffect(pulse.clone()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let class_prepared = class_builder.prepare(&ctx).unwrap();
    // The class plan is sealed by its own route, never by the fixed-DAG kernel.
    assert!(!matches!(
        class_prepared.retained_operation(),
        RetainedOperation::BayesianTemporalDagEffect(..)
    ));
    let RetainedOperation::BayesianTemporalClassEffect(class_plan) =
        class_prepared.retained_operation()
    else {
        panic!("class route retains the Bayesian class plan")
    };
    assert_eq!(class_plan.query, pulse);
    assert_eq!(class_plan.graph_class, antecedent::GraphClass::TemporalCpdag);
    assert_eq!(class_plan.estimator.as_str(), "bayesian.temporal.gcomp");
}
