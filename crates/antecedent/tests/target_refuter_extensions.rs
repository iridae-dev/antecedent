//! Target-native refuters that previously refused: an average derivative checked by its
//! weighting law's nodes, the posterior-predictive check on a series under a lagged reference
//! model, and the random common cause declared in every atom of a graph posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::target_validation::{
    PreparedTargetData, TargetRefitOptions, TargetRefutationConfig, TargetRefutationOutcome,
    TargetRefuterKind, TargetSuite, refute_prepared_target,
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
    for weighting in
        [DerivativeWeighting::Observed, DerivativeWeighting::Uniform { lower: 1.5, upper: 2.5 }]
    {
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
        assert!((fd.refuted[0] - 2.0).abs() < 0.4, "{weighting:?}: curve slope {:?}", fd.refuted);
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
    // The check no longer refuses a series for replicating independent rows.
    assert!(
        !outcome.run.refused.iter().any(|r| r.kind == TargetRefuterKind::PosteriorPredictive
            && r.reason.contains("independent rows")),
        "{:?}",
        outcome.run.refused
    );
}

/// Columns `t, y, z`: `z` confounds `t` and `y`.
fn confounded(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let ti = f64::from(u8::from(zi + standard_normal(&mut rng) > 0.0));
        let yi = 1.0 + 2.0 * ti + zi + standard_normal(&mut rng);
        t.push(ti);
        y.push(yi);
        z.push(zi);
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

#[test]
fn a_common_cause_is_declared_in_every_atom_of_a_graph_posterior() {
    let ctx = ExecutionContext::for_tests(63);
    let data = confounded(800, 4);
    // z -> t, z -> y, t -> y, and the same graph without the z -> t edge.
    let full = [(2, 0), (2, 1), (0, 1)]
        .iter()
        .fold(AdjacencyMask::EMPTY, |mask, &(a, b)| set_edge(mask, 3, a, b, true));
    let thin = [(2, 1), (0, 1)]
        .iter()
        .fold(AdjacencyMask::EMPTY, |mask, &(a, b)| set_edge(mask, 3, a, b, true));
    let posterior = GraphPosterior::from_atoms(
        3,
        vec![0.7, 0.3],
        [full, thin],
        vec![0.0; 9],
        vec![0.0; 9],
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
}
