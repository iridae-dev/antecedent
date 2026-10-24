//! An `AverageEffect` on one series under a temporal graph is the lag-0 contrast
//! `E[Y_t | do(X_t = 1)] - E[Y_t | do(X_t = 0)]`.
//!
//! The build compiles it as the pulse at offset 0 read at horizon 1, so it runs the temporal
//! class envelope (frequentist and Bayesian) on a `TemporalCpdag` or `TemporalPag` and the
//! temporal backdoor on a `TemporalDag`. These tests pin that the answer is exactly the
//! explicit lag-0 pulse's, that the support status stays `refused` (the `AverageEffect` cell is
//! executing and unlicensed) on the direct and prepared paths, that the result discloses the
//! reading, and that modifiers and another functional are refused. Over a graph posterior the
//! same pulse is compiled, so the alias equals the explicit pulse over the posterior in both
//! inference modes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Lag, OrderedFloatBits, OutcomeFunctional,
    TemporalEffectQuery, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag};

#[path = "fixtures_common/mod.rs"]
mod common;
use common::fixtures::{B2, confounded_series, heterogeneous_dbn};

const N: usize = 1500;
const BETA: f64 = 2.0;
const DISCLOSURE: &str = "estimate.temporal_class.lag_zero_average_effect";

fn x() -> VariableId {
    VariableId::from_raw(0)
}

fn y() -> VariableId {
    VariableId::from_raw(1)
}

fn z() -> VariableId {
    VariableId::from_raw(2)
}

/// A deterministic standard-normal stream (xorshift and Box-Muller).
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        })
        .collect()
}

/// `x_t = 0.6 z_{t-1} + e_t`, `y_t = 2 x_t + 0.8 z_{t-1} + u_t`: the law of [`dag`].
fn series() -> TimeSeriesData {
    series_with(0.8)
}

/// [`series`] with `z_{t-1} -> y_t` of strength `z_to_y`. Zero is the law of [`pag`], whose
/// `z[t-1]` is not adjacent to `y[t]`; on [`series`] that PAG is misspecified and omits the
/// confounder (the lag-0 regression then converges to `2 + 0.8 * 0.6 / 1.36 = 2.353`).
fn series_with(z_to_y: f64) -> TimeSeriesData {
    let zs = normals(N, 1);
    let e = normals(N, 2);
    let u = normals(N, 3);
    let mut xs = vec![0.0; N];
    let mut ys = vec![0.0; N];
    for t in 1..N {
        xs[t] = 0.6 * zs[t - 1] + e[t];
        ys[t] = BETA * xs[t] + z_to_y * zs[t - 1] + 0.5 * u[t];
    }
    TimeSeriesData::from_f64_columns(
        [("x", xs.as_slice()), ("y", ys.as_slice()), ("z", zs.as_slice())],
        1,
    )
    .unwrap()
}

fn dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let z1 = graph.add_lagged(z(), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(x(), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(y(), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x0).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(x0, y0).unwrap();
    graph
}

/// `z[t-1] -> x[t] -> y[t]` with `z[t-1]` not adjacent to `y[t]`: the edge out of `x[t]` is
/// visible in every MAG completion.
fn pag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let z1 = graph.add_lagged(z(), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(x(), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(y(), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x0).unwrap();
    graph.insert_directed(x0, y0).unwrap();
    graph
}

fn average() -> AverageEffectQuery {
    AverageEffectQuery::with_levels(x(), y(), 0.0, 1.0)
}

fn lag_zero_pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(x(), y(), 1.0)
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200).prior_scale(100.0))
}

fn build(graph: AcceptedGraph, query: CausalQuery, inference: InferenceMode) -> Study {
    build_on(series(), graph, query, inference)
}

fn build_on(
    data: TimeSeriesData,
    graph: AcceptedGraph,
    query: CausalQuery,
    inference: InferenceMode,
) -> Study {
    Study::series(data)
        .graph(graph)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn summary(result: &StudyResult) -> String {
    let effect = result.estimate.as_effect().map(|effect| effect.ate.to_bits());
    let envelope = result.structural_response.as_ref().map(|mixture| {
        format!("{:?} {:?}", mixture.identified_set, mixture.identified_set_interval)
    });
    format!("{effect:?} {envelope:?} {:?}", result.identification.status)
}

fn discloses(result: &StudyResult) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == DISCLOSURE)
}

fn graphs() -> [(&'static str, AcceptedGraph); 3] {
    [
        ("TemporalDag", dag().into()),
        ("TemporalCpdag", TemporalCpdag::from_temporal_dag(&dag()).into()),
        ("TemporalPag", AcceptedGraph::temporal_pag(pag())),
    ]
}

#[test]
fn the_lag_zero_average_effect_is_the_explicit_pulse_and_stays_unlicensed() {
    let ctx = ExecutionContext::for_tests(11);
    for (name, graph) in graphs() {
        for inference in [InferenceMode::Frequentist, bayesian()] {
            let label = format!("{name} {inference:?}");
            let alias =
                build(graph.clone(), CausalQuery::AverageEffect(average()), inference.clone());
            assert_eq!(alias.support_status(), Some(CellStatus::Refused), "{label}");
            let explicit =
                build(graph.clone(), CausalQuery::TemporalEffect(lag_zero_pulse()), inference);
            let aliased = alias.run(&ctx).unwrap();
            let direct = explicit.run(&ctx).unwrap();
            assert_eq!(summary(&aliased), summary(&direct), "{label}");
            assert_eq!(aliased.support_status, Some(CellStatus::Refused), "{label}");
            assert!(discloses(&aliased), "{label}");
            assert!(!discloses(&direct), "{label}");
        }
    }
}

#[test]
fn the_point_identified_graphs_recover_the_contemporaneous_effect() {
    let ctx = ExecutionContext::for_tests(12);
    // Each graph is fitted on the law it encodes: the PAG has no z[t-1] -> y[t] edge.
    let cases: [(&str, AcceptedGraph, f64); 2] = [
        ("TemporalDag", dag().into(), 0.8),
        ("TemporalPag", AcceptedGraph::temporal_pag(pag()), 0.0),
    ];
    for (name, graph, z_to_y) in cases {
        let result = build_on(
            series_with(z_to_y),
            graph,
            CausalQuery::AverageEffect(average()),
            InferenceMode::Frequentist,
        )
        .run(&ctx)
        .unwrap();
        let ate = result.estimate.as_effect().unwrap().ate;
        assert!((ate - BETA).abs() < 0.1, "{name}: ate={ate}");
    }
}

#[test]
fn the_prepared_path_keeps_the_refused_status_and_the_disclosure() {
    let ctx = ExecutionContext::for_tests(14);
    let graph: AcceptedGraph = TemporalCpdag::from_temporal_dag(&dag()).into();
    let prepared = build(graph, CausalQuery::AverageEffect(average()), InferenceMode::Frequentist)
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate_series(&series(), &ctx).unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    assert!(discloses(&result));
}

#[test]
fn modifiers_other_functionals_and_populations_are_refused_with_the_next_attempt() {
    let graph: AcceptedGraph = TemporalCpdag::from_temporal_dag(&dag()).into();
    let mut modified = average();
    modified.effect_modifiers = std::sync::Arc::from([z()]);
    let mut exceedance = average();
    exceedance.outcome_functional = OutcomeFunctional::Exceedance(OrderedFloatBits::from_f64(1.0));
    for query in [modified, exceedance] {
        let error = Study::series(series())
            .graph(graph.clone())
            .query(CausalQuery::AverageEffect(query))
            .refute(RefuteSuite::None)
            .build()
            .unwrap_err()
            .to_string();
        assert!(error.contains("Next attempt"), "{error}");
    }
}

#[test]
fn over_a_graph_posterior_the_alias_is_the_explicit_pulse_over_the_posterior() {
    let ctx = ExecutionContext::for_tests(15);
    let data = confounded_series(1_200, B2, 0.0, 4_101);
    for inference in [InferenceMode::Frequentist, bayesian()] {
        let label = format!("{inference:?}");
        let run = |query: CausalQuery| {
            Study::series(data.clone())
                .graph_posterior(heterogeneous_dbn())
                .query(query)
                .inference(inference.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap()
        };
        let aliased = run(CausalQuery::AverageEffect(average()));
        let direct = run(CausalQuery::TemporalEffect(lag_zero_pulse()));
        // The compiled program is the pulse's, so the answer and the mixture over the
        // posterior's atoms are identical; only the support status and the disclosure differ.
        assert_eq!(summary(&aliased), summary(&direct), "{label}");
        assert_eq!(aliased.support_status, Some(CellStatus::Refused), "{label}");
        assert!(discloses(&aliased), "{label}");
        assert!(!discloses(&direct), "{label}");
        assert_eq!(
            aliased.structural_response.as_ref().map(|mixture| mixture.unidentified_mass),
            direct.structural_response.as_ref().map(|mixture| mixture.unidentified_mass),
            "{label}"
        );
    }
}
