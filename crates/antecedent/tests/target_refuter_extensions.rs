//! Target-native refuters that previously refused: an average derivative checked by its
//! weighting law's nodes, the posterior-predictive check on a series under a lagged reference
//! model, and the random common cause declared in every atom of a graph posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::target_validation::{
    PreparedTargetData, TargetRefitOptions, TargetRefutationConfig, TargetRefutationOutcome,
    TargetRefuterKind, TargetSuite, WithinClassDistribution, refute_prepared_target,
};
use antecedent::{BayesianConfig, InferenceMode, LatencyMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalRng, DerivativeScale, DerivativeWeighting,
    ExecutionContext, Lag, ResponseFunctional, ResponseQuery, TemporalEffectQuery, TemporalPolicy,
    VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_kernels::standard_normal;
use antecedent_prob::InferenceDiagnostics;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn report(
    outcome: &TargetRefutationOutcome,
    kind: TargetRefuterKind,
) -> &antecedent::analysis::target_validation::TargetRefutationReport {
    outcome
        .run
        .reports
        .iter()
        .find(|r| r.kind == kind)
        .unwrap_or_else(|| panic!("{kind:?} did not run: {:?}", outcome.run.refused))
}

fn run(
    prepared: &PreparedStudy,
    data: PreparedTargetData<'_>,
    suite: TargetSuite,
    config: &TargetRefutationConfig,
    options: &TargetRefitOptions,
    ctx: &ExecutionContext,
) -> TargetRefutationOutcome {
    let result = match data {
        PreparedTargetData::Tabular(d) => prepared.estimate(d, ctx),
        PreparedTargetData::Series(d) => prepared.estimate_series(d, ctx),
        PreparedTargetData::Panel(d) => prepared.estimate_panel(d, ctx),
        PreparedTargetData::MultiEnv(d) => prepared.estimate_multi_env(d, ctx),
    }
    .unwrap();
    refute_prepared_target(prepared, &result, data, suite, config, options, ctx).unwrap()
}

/// `y = 2 a + 0.5 b + e` with `a` confounded by `b`.
fn dose_data(n: u32) -> TabularData {
    let mut rng = CausalRng::from_seed(9);
    let b: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let a: Vec<f64> = b.iter().map(|b| 2.0 + 0.3 * b + 0.5 * standard_normal(&mut rng)).collect();
    let y: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(a, b)| 2.0 * a + 0.5 * b + 0.05 * standard_normal(&mut rng))
        .collect();
    TabularData::from_f64_columns([("a", a.as_slice()), ("b", b.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

#[test]
fn an_average_derivative_is_checked_at_the_nodes_of_its_weighting_law() {
    let ctx = ExecutionContext::for_tests(61);
    let data = dose_data(800);
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(1, 0), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let config = TargetRefutationConfig { replicates: 4, ..TargetRefutationConfig::default() };
    let options =
        TargetRefitOptions { derivative_step: Some(0.5), ..TargetRefitOptions::default() };
    // The Gaussian-score Riesz estimator admits the observed-law weighting only, so that is the
    // law whose nodes are checked.
    {
        let weighting = DerivativeWeighting::Observed;
        let prepared = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(ResponseQuery::new(
                ResponseFunctional::AverageDerivative {
                    outcome: v(2),
                    treatment: v(0),
                    weighting: weighting.clone(),
                },
            )))
            .response_options(ContinuousResponseOptions {
                bandwidth: Some(0.35),
                ..Default::default()
            })
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let outcome = run(
            &prepared,
            PreparedTargetData::Tabular(&data),
            TargetSuite::Cheap,
            &config,
            &options,
            &ctx,
        );
        let fd = report(&outcome, TargetRefuterKind::FiniteDifferenceConsistency);
        assert!((fd.refuted[0] - 2.0).abs() < 0.05, "{weighting:?}: curve slope {:?}", fd.refuted);
        assert!(report(&outcome, TargetRefuterKind::LocalLinearity).passed);
    }
}

/// `y_t = 0.5 x_{t-1} + 0.4 y_{t-1} + e_t` with independent `x`.
fn series(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let x: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let mut y = vec![0.0; n];
    for t in 1..n {
        y[t] = 0.5 * x[t - 1] + 0.4 * y[t - 1] + 0.5 * standard_normal(&mut rng);
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

#[test]
fn the_posterior_predictive_check_replicates_a_series_under_a_lagged_reference_model() {
    let ctx = ExecutionContext::for_tests(62);
    let data = series(500, 12);
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let prepared = Study::series(data.clone())
        .graph(graph)
        .temporal_query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let config = TargetRefutationConfig { replicates: 6, ..TargetRefutationConfig::default() };
    let outcome = run(
        &prepared,
        PreparedTargetData::Series(&data),
        TargetSuite::Full,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    );
    // The check ran on the series: it reports on the whole target, with all replicates refit
    // (it may fail this graph, which omits the series' own lag, but it must speak to it).
    let ppc = report(&outcome, TargetRefuterKind::PosteriorPredictive);
    assert!(ppc.informative && ppc.replicates == 6, "{ppc:?}");
    // The check no longer refuses a series for replicating independent rows.
    assert!(
        !outcome.run.refused.iter().any(|r| r.kind == TargetRefuterKind::PosteriorPredictive
            && r.reason.contains("independent rows")),
        "{:?}",
        outcome.run.refused
    );
}

/// Columns `t, y, z, x`: `z` confounds `t` and `y`; `x` is a child of `z`.
fn confounded(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z, mut x) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let ti = f64::from(u8::from(zi + standard_normal(&mut rng) > 0.0));
        let yi = 1.0 + 2.0 * ti + zi + standard_normal(&mut rng);
        t.push(ti);
        y.push(yi);
        z.push(zi);
        x.push(zi + standard_normal(&mut rng));
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

#[test]
fn a_common_cause_is_declared_in_every_atom_of_a_graph_posterior() {
    let ctx = ExecutionContext::for_tests(63);
    let data = confounded(800, 4);
    // Both atoms give the treatment the parent z and the outcome the parents z and t, so they
    // name one estimand (a graph posterior whose atoms name different estimands has no scalar
    // mixture); they differ in how the extra node x hangs off the graph.
    let base = [(2, 0), (2, 1), (0, 1), (2, 3)];
    let first =
        base.iter().fold(AdjacencyMask::EMPTY, |mask, &(a, b)| set_edge(mask, 4, a, b, true));
    let second = set_edge(&first, 4, 0, 3, true);
    let posterior = GraphPosterior::from_atoms(
        4,
        vec![0.7, 0.3],
        [first, second],
        vec![0.0; 16],
        vec![0.0; 16],
        SamplingSummary {
            ess: 2.0,
            diagnostics: InferenceDiagnostics::analytic("refuter_extensions"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag);
    let prepared = Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let config = TargetRefutationConfig { replicates: 6, ..TargetRefutationConfig::default() };
    let options =
        TargetRefitOptions { common_cause_strength: 0.5, ..TargetRefitOptions::default() };
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Full,
        &config,
        &options,
        &ctx,
    );
    let cause = report(&outcome, TargetRefuterKind::RandomCommonCause);
    assert!(cause.informative);
    // A random common cause carries no information about t or y, so it leaves the effect where
    // it was (observed shift 1e-3 on the ate near 2).
    assert!((cause.refuted[0] - cause.original[0]).abs() < 0.02, "{cause:?}");
    assert!(cause.passed, "{:?}", cause.failure);
}

#[test]
fn gathered_resamples_serve_a_route_that_reads_no_row_weights() {
    let ctx = ExecutionContext::for_tests(64);
    let data = series(600, 14);
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let prepared = Study::series(data.clone())
        .graph(graph)
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let config = TargetRefutationConfig {
        replicates: 12,
        pre_trend_lead: 2,
        ..TargetRefutationConfig::default()
    };
    let options = TargetRefitOptions { gathered_resamples: true, ..TargetRefitOptions::default() };
    let outcome = run(
        &prepared,
        PreparedTargetData::Series(&data),
        TargetSuite::Full,
        &config,
        &options,
        &ctx,
    );
    // The bootstrap, pre-trend, anticipation and lagged-placebo refits run as explicit gathered
    // series and cut windows, and a sound design passes them.
    for kind in [
        TargetRefuterKind::BootstrapStability,
        TargetRefuterKind::PreTrend,
        TargetRefuterKind::Anticipation,
        TargetRefuterKind::NegativeControlOutcome,
    ] {
        let r = report(&outcome, kind);
        assert!(r.passed, "{kind:?}: {:?}", r.failure);
    }
}

#[test]
fn a_jacobian_is_checked_along_each_treatment_axis() {
    let ctx = ExecutionContext::for_tests(65);
    // Two roots and two outcomes that are linear in them: the shape of the checked static
    // derivative route (the Jacobian of `(y, v)` in `(a, b)` is `[[2, -0.5], [0.25, 1.5]]`).
    let a: Vec<f64> = (0..800)
        .map(|i| 2.0 + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos())
        .collect();
    let b: Vec<f64> = (0..800).map(|i| (f64::from(i) * 1.13).cos()).collect();
    let y: Vec<f64> = a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b).collect();
    let w: Vec<f64> = a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 1.5 * b).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
        ("v", w.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: std::sync::Arc::from([v(2), v(3)]),
            treatments: std::sync::Arc::from([v(0), v(1)]),
            at: std::sync::Arc::from([2.0, 0.0]),
            scale: DerivativeScale::Identity,
        })))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.35), ..Default::default() })
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let config = TargetRefutationConfig { replicates: 6, ..TargetRefutationConfig::default() };
    let options =
        TargetRefitOptions { derivative_step: Some(0.5), ..TargetRefitOptions::default() };
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Cheap,
        &config,
        &options,
        &ctx,
    );
    let fd = report(&outcome, TargetRefuterKind::FiniteDifferenceConsistency);
    assert_eq!(fd.refuted.len(), 4);
    for (got, want) in fd.refuted.iter().zip([2.0, -0.5, 0.25, 1.5]) {
        assert!((got - want).abs() < 1e-6, "finite differences {:?}", fd.refuted);
    }
}

/// Columns `t, y, z, w, x`: `z` and `w` cause `t`, `t` causes `y`, `x` is a child of `w`.
fn collider_data(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z, mut w, mut x) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let wi = standard_normal(&mut rng);
        let ti = f64::from(u8::from(zi + wi + 0.5 * standard_normal(&mut rng) > 0.0));
        t.push(ti);
        y.push(1.0 + 2.0 * ti + standard_normal(&mut rng));
        z.push(zi);
        w.push(wi);
        x.push(wi + standard_normal(&mut rng));
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("w", w.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

#[test]
fn a_common_cause_on_a_class_posterior_needs_the_declared_within_class_distribution() {
    let ctx = ExecutionContext::for_tests(66);
    let data = collider_data(900, 6);
    // z -> t <- w, t -> y, and w - x in either direction: one class with two DAG members.
    let dag = |edges: &[(usize, usize)]| {
        edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(a, b)| set_edge(mask, 5, a, b, true))
    };
    let first = dag(&[(2, 0), (3, 0), (0, 1), (3, 4)]);
    let second = dag(&[(2, 0), (3, 0), (0, 1), (4, 3)]);
    let dags = GraphPosterior::from_atoms(
        5,
        vec![0.5, 0.5],
        [first, second],
        vec![0.0; 25],
        vec![0.0; 25],
        SamplingSummary {
            ess: 2.0,
            diagnostics: InferenceDiagnostics::analytic("within_class"),
            rejected_invalid: 0,
        },
    )
    .unwrap();
    let classes = dags.to_equivalence_classes().unwrap();
    assert_eq!(classes.atom_kind(), GraphPosteriorAtomKind::Cpdag);
    assert_eq!(classes.n_graphs(), 1);
    let prepared = Study::tabular(data.clone())
        .graph_posterior(classes)
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let config = TargetRefutationConfig { replicates: 6, ..TargetRefutationConfig::default() };
    let undeclared =
        TargetRefitOptions { common_cause_strength: 0.5, ..TargetRefitOptions::default() };
    let refused = run(
        &prepared,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Full,
        &config,
        &undeclared,
        &ctx,
    );
    let refusal = refused
        .run
        .refused
        .iter()
        .find(|r| r.kind == TargetRefuterKind::RandomCommonCause)
        .expect("a class posterior refuses the common cause until the division is declared");
    assert!(refusal.reason.contains("WithinClassDistribution"), "{}", refusal.reason);
    let declared = TargetRefitOptions {
        common_cause_strength: 0.5,
        within_class: Some(WithinClassDistribution::UniformOverMembers),
        ..TargetRefitOptions::default()
    };
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Full,
        &config,
        &declared,
        &ctx,
    );
    let cause = report(&outcome, TargetRefuterKind::RandomCommonCause);
    assert!(cause.informative);
    // A random common cause carries no information about t or y, so it leaves the effect where
    // it was (observed shift 1e-3 on the ate near 2).
    assert!((cause.refuted[0] - cause.original[0]).abs() < 0.02, "{cause:?}");
    assert!(cause.passed, "{:?}", cause.failure);
}
