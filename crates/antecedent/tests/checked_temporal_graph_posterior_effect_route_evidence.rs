//! Builder independent evidence for temporal graph-posterior pulse and sustained effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one loop covers every licensed coordinate")]

mod common;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, set_edge};
use antecedent_prob::InferenceDiagnostics;
use common::fixtures::{self, B2, DBN_UNIDENTIFIED_MASS, confounded_series, heterogeneous_dbn};

const DRAWS: usize = 96;

fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (*state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
}

/// The `known_truth_mixtures` `TemporalCpdag` posterior on `(pressure, defect)`:
/// one identified atom `pressure@1 -> defect` (mass 0.7) and one autoregressive
/// atom that stays `NotCertified` (mass 0.3).
fn cpdag_posterior() -> GraphPosterior {
    let kind = GraphPosteriorAtomKind::Cpdag;
    GraphPosterior::new(
        2,
        vec![0.7, 0.3],
        vec![0, 0],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0,
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.3, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![2, 3])
    .unwrap()
}

/// A `TemporalPag` posterior on `(t, y, z)` whose contemporaneous `z -> t`,
/// `z -> y` make `t@1 -> y` visible (mass 0.7); the autoregressive atom stays
/// `NotCertified` (mass 0.3).
fn pag_posterior() -> GraphPosterior {
    let visible = set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true);
    let lag = 1u64 << 1;
    GraphPosterior::new(
        3,
        vec![0.7, 0.3],
        vec![visible, visible],
        vec![0.0; 9],
        vec![0.0; 9],
        1.0,
        InferenceDiagnostics::analytic("visible_pag_known_truth"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_lagged_marginals(1, vec![0.0; 9])
    .unwrap()
    .with_lag_masks(vec![lag, lag | 1])
    .unwrap()
}

/// `t = 0.5 z + e`, `y_t = 0.4 z_t + 0.9 t_{t-1}` on the pinned generator.
fn pag_series(seed: u64) -> TimeSeriesData {
    let n = 400usize;
    let mut state = seed;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = lcg_uniform(&mut state);
        t[i] = 0.5 * z[i] + lcg_uniform(&mut state);
        y[i] = 0.4 * z[i] + if i > 0 { 0.9 * t[i - 1] } else { 0.0 };
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

/// `defect_t = 0.9 * pressure_{t-1}` on the pinned generator.
fn class_series(seed: u64) -> TimeSeriesData {
    let n = 400usize;
    let mut state = seed;
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 0..n {
        pressure[t] = lcg_uniform(&mut state);
        if t > 0 {
            defect[t] = 0.9 * pressure[t - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("pressure", pressure.as_slice()), ("defect", defect.as_slice())],
        1,
    )
    .unwrap()
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
}

fn multi_step_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
}

fn inferences() -> [InferenceMode; 2] {
    [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(100.0)),
    ]
}

fn inference_label(inference: &InferenceMode) -> &'static str {
    match inference {
        InferenceMode::Frequentist => "frequentist",
        InferenceMode::Bayesian(_) => "bayesian",
    }
}

fn point(result: &antecedent::StudyResult, inference: &InferenceMode) -> f64 {
    match inference {
        InferenceMode::Frequentist => result.estimate.ate,
        InferenceMode::Bayesian(_) => result
            .posterior
            .as_ref()
            .and_then(|posterior| {
                let column = posterior.effect_column()?;
                posterior.summaries.mean.get(column).copied()
            })
            .unwrap_or(result.estimate.ate),
    }
}

struct Route {
    posterior: GraphPosterior,
    data: TimeSeriesData,
    refreshed: TimeSeriesData,
    query: TemporalEffectQuery,
    truth: f64,
    tolerance: f64,
    unidentified_mass: f64,
}

fn routes() -> Vec<Route> {
    let mut routes = Vec::new();
    for multi_step in [false, true] {
        let truth = fixtures::dbn_mixture_truth(if multi_step {
            fixtures::dbn_multistep_atom_truths(0.0)
        } else {
            fixtures::dbn_pulse_atom_truths(0.0)
        });
        routes.push(Route {
            posterior: heterogeneous_dbn(),
            data: confounded_series(1_200, B2, 0.0, 4_101),
            refreshed: confounded_series(1_200, B2, 0.0, 4_102),
            query: if multi_step { multi_step_sustained() } else { pulse() },
            truth,
            tolerance: 0.12,
            unidentified_mass: DBN_UNIDENTIFIED_MASS,
        });
    }
    routes.push(Route {
        posterior: cpdag_posterior(),
        data: class_series(42),
        refreshed: class_series(43),
        query: pulse().with_horizon_steps(1).with_max_history_lag(Some(1)),
        truth: 0.9,
        tolerance: 0.05,
        unidentified_mass: 0.3,
    });
    routes.push(Route {
        posterior: pag_posterior(),
        data: pag_series(42),
        refreshed: pag_series(43),
        query: pulse().with_horizon_steps(1).with_max_history_lag(Some(1)),
        truth: 0.9,
        tolerance: 0.1,
        unidentified_mass: 0.3,
    });
    routes
}

#[test]
fn all_temporal_graph_posterior_effect_coordinates_execute_from_retained_atom_proofs() {
    let ctx = ExecutionContext::for_tests(4_100);
    let suites = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];
    for route in routes() {
        let kind = route.posterior.atom_kind;
        for inference in inferences() {
            for suite in suites {
                let builder = Study::series(route.data.clone())
                    .graph_posterior(route.posterior.clone())
                    .query(CausalQuery::TemporalEffect(route.query.clone()))
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(8)
                    .build()
                    .unwrap();
                let one_shot = builder.run(&ctx).unwrap();
                let mut prepared = builder.prepare(&ctx).unwrap();
                drop(builder);

                let plan = prepared
                    .checked_temporal_graph_posterior_effect_info()
                    .expect("temporal graph-posterior effect must retain its execution plan");
                assert_eq!(plan.query, route.query);
                assert_eq!(plan.atom_kind, kind);
                assert_eq!(plan.identifier.as_str(), "temporal.backdoor.unfolded");
                assert_eq!(
                    plan.estimator.as_str(),
                    match (&inference, route.query.is_multi_step_sustained()) {
                        (_, true) => "temporal.sequential.gcomp",
                        (InferenceMode::Bayesian(_), false) => "bayesian.temporal.gcomp",
                        (InferenceMode::Frequentist, false) => "temporal.linear.adjustment",
                    }
                );
                assert_eq!(plan.validation, suite);
                assert!(plan.inference.starts_with(inference_label(&inference)));
                assert_eq!(plan.bootstrap_replicates, 8);
                assert_eq!(plan.graph_keys.len(), route.posterior.n_graphs);
                assert_eq!(plan.weights.as_ref(), route.posterior.weights.as_ref());
                assert!(plan.identified_atom_count > 0);
                assert!(plan.identified_atom_count < route.posterior.n_graphs);

                let result = prepared.estimate_series(&route.data, &ctx).unwrap();
                let estimate = point(&result, &inference);
                assert!(
                    (estimate - route.truth).abs() < route.tolerance,
                    "kind={kind:?}, inference={}, suite={suite:?}, query={:?}: {estimate} vs {}",
                    inference_label(&inference),
                    route.query.policy,
                    route.truth
                );
                let unidentified_mass = result
                    .structural_response
                    .as_ref()
                    .map(|mixture| mixture.unidentified_mass)
                    .or_else(|| {
                        result.posterior.as_ref().map(|posterior| posterior.unidentified_mass)
                    })
                    .expect("posterior atoms remain in the structural mixture or effect posterior");
                assert!(
                    (unidentified_mass - route.unidentified_mass).abs() < 1e-9,
                    "unidentified posterior mass must be retained: {unidentified_mass}"
                );
                // The one-shot facade executes the same sealed program.
                let one_shot_estimate = point(&one_shot, &inference);
                assert!(
                    (one_shot_estimate - estimate).abs() < 1e-9,
                    "one-shot {one_shot_estimate} differs from the prepared plan {estimate}"
                );
                assert!(
                    one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );

                let refreshed = prepared.refresh_series(route.refreshed.clone(), &ctx).unwrap();
                let refreshed_estimate = point(&refreshed, &inference);
                assert!(
                    (refreshed_estimate - route.truth).abs() < route.tolerance,
                    "refreshed {refreshed_estimate} vs {}",
                    route.truth
                );
                assert_eq!(
                    prepared.checked_temporal_graph_posterior_effect_info().unwrap().query,
                    route.query
                );
                let narrowed = TimeSeriesData::from_f64_columns(
                    [("t", &[0.0, 1.0, 0.0, 1.0][..]), ("y", &[0.0, 1.0, 1.0, 0.0][..])],
                    1,
                )
                .unwrap();
                assert!(
                    prepared.refresh_series(narrowed, &ctx).is_err(),
                    "schema-changed data must be refused"
                );
                assert_eq!(
                    prepared.checked_temporal_graph_posterior_effect_info().unwrap().query,
                    route.query
                );

                let artifact = prepared
                    .encode_contracted_result(
                        &refreshed,
                        "checked-temporal-graph-posterior-effect",
                        &ctx,
                    )
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(
                    consumed.acceptance.unresolved.iter().any(|dependency| {
                        dependency.as_ref()
                            == "dependencies.checked_temporal_graph_posterior_effect_operation"
                    }),
                    "unresolved={:?} contract={:?}",
                    consumed.acceptance.unresolved,
                    consumed.contract.as_ref().map(|c| (
                        c.graph_class.clone(),
                        c.structure_source.clone(),
                        c.estimator.clone()
                    ))
                );
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}
