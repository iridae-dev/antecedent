//! Target-native refuters constructed through the compiler, on simulated data with a known
//! truth: each passes on a sound design and flags the violation planted for it.
//!
//! * random common cause: a simulated cause declared in the graph and adjusted for;
//! * negative controls read from the graph (static) and the lagged placebo (series);
//! * series bootstrap stability by circular-block row multiplicities;
//! * placebo timing on one series (a future treatment) and the pre-period outcome;
//! * Bayesian prior sensitivity under a prior-scale grid compiled from the study;
//! * derivative diagnostics on the study's own fitted response curve.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::target_validation::{
    PreparedTargetData, TargetRefitOptions, TargetRefutationConfig, TargetRefutationOutcome,
    TargetRefuterKind, TargetSuite, refute_prepared_target,
};
use antecedent::{BayesianConfig, InferenceMode, LatencyMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalRng, DerivativeScale, ExecutionContext, Lag,
    ResponseFunctional, ResponseQuery, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_kernels::standard_normal;

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

/// Columns `t, y, z, x, w`: `z` confounds `t` and `y`; `x` and `w` are children of `z` only.
/// With `planted`, an unobserved `u` also drives `t`, `y` and `x`.
fn static_data(n: usize, seed: u64, planted: bool) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z, mut x, mut w) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let zi = standard_normal(&mut rng);
        let u = if planted { standard_normal(&mut rng) } else { 0.0 };
        let ti = f64::from(u8::from(zi + 1.5 * u + standard_normal(&mut rng) > 0.0));
        let xi = f64::from(u8::from(zi + 1.5 * u + standard_normal(&mut rng) > 0.0));
        let wi = zi + standard_normal(&mut rng);
        let yi = 1.0 + 2.0 * ti + zi + 1.5 * u + standard_normal(&mut rng);
        t.push(ti);
        y.push(yi);
        z.push(zi);
        x.push(xi);
        w.push(wi);
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("x", x.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap()
}

fn static_graph() -> Dag {
    let mut dag = Dag::with_variables(5);
    for (from, to) in [(2, 0), (2, 1), (0, 1), (2, 3), (2, 4)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    dag
}

fn static_study(
    data: &TabularData,
    inference: InferenceMode,
    ctx: &ExecutionContext,
) -> PreparedStudy {
    Study::tabular(data.clone())
        .graph(static_graph())
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .inference(inference)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
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

#[test]
fn a_declared_common_cause_and_graph_negative_controls_pass_on_a_sound_design() {
    let data = static_data(800, 3, false);
    let ctx = ExecutionContext::for_tests(31);
    let prepared = static_study(&data, InferenceMode::Frequentist, &ctx);
    let config = TargetRefutationConfig { replicates: 12, ..TargetRefutationConfig::default() };
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
    for kind in [
        TargetRefuterKind::PlaceboTreatment,
        TargetRefuterKind::RandomCommonCause,
        TargetRefuterKind::BootstrapStability,
        TargetRefuterKind::NegativeControlExposure,
        TargetRefuterKind::NegativeControlOutcome,
    ] {
        let r = report(&outcome, kind);
        assert!(r.passed, "{kind:?}: {:?}", r.failure);
    }
    // The planted-and-declared common cause is adjusted away: the effect stays at 2.
    let rcc = report(&outcome, TargetRefuterKind::RandomCommonCause);
    assert!((rcc.refuted[0] - 2.0).abs() < 0.3, "{:?}", rcc.refuted);
}

#[test]
fn the_graph_negative_control_exposure_flags_a_planted_unobserved_confounder() {
    let data = static_data(1500, 4, true);
    let ctx = ExecutionContext::for_tests(32);
    let prepared = static_study(&data, InferenceMode::Frequentist, &ctx);
    let config = TargetRefutationConfig { replicates: 12, ..TargetRefutationConfig::default() };
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Full,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    );
    let nce = report(&outcome, TargetRefuterKind::NegativeControlExposure);
    assert!(!nce.passed, "the negative-control exposure shares the planted confounder");
}

/// `y_t = 0.5 x_{t-1} + anticipation · x_{t+1} + e_t` with independent `x`.
fn series(n: usize, seed: u64, anticipation: f64) -> TimeSeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let x: Vec<f64> = (0..=n).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<f64> = (0..n)
        .map(|t| {
            let past = if t > 0 { x[t - 1] } else { 0.0 };
            0.5 * past + anticipation * x[t + 1] + 0.5 * standard_normal(&mut rng)
        })
        .collect();
    TimeSeriesData::from_f64_columns([("x", &x[..n]), ("y", y.as_slice())], 1).unwrap()
}

fn pulse_study(data: &TimeSeriesData, ctx: &ExecutionContext) -> PreparedStudy {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    Study::series(data.clone())
        .graph(graph)
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}

#[test]
fn a_single_series_gets_block_bootstrap_placebo_timing_and_a_lagged_placebo() {
    let data = series(600, 5, 0.0);
    let ctx = ExecutionContext::for_tests(33);
    let prepared = pulse_study(&data, &ctx);
    let config = TargetRefutationConfig {
        replicates: 12,
        pre_trend_lead: 2,
        ..TargetRefutationConfig::default()
    };
    let outcome = run(
        &prepared,
        PreparedTargetData::Series(&data),
        TargetSuite::Full,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    );
    for kind in [
        TargetRefuterKind::PlaceboTreatment,
        TargetRefuterKind::BootstrapStability,
        TargetRefuterKind::PreTrend,
        TargetRefuterKind::Anticipation,
        TargetRefuterKind::NegativeControlOutcome,
        TargetRefuterKind::RandomCommonCause,
    ] {
        let r = report(&outcome, kind);
        assert!(r.passed, "{kind:?}: {:?}", r.failure);
    }
    // Only x and y are in the graph: no variable can be a negative-control exposure.
    assert!(
        outcome.run.refused.iter().any(|r| r.kind == TargetRefuterKind::NegativeControlExposure)
    );
}

#[test]
fn placebo_timing_flags_an_outcome_that_anticipates_the_treatment() {
    let data = series(600, 6, 0.8);
    let ctx = ExecutionContext::for_tests(34);
    let prepared = pulse_study(&data, &ctx);
    let config = TargetRefutationConfig {
        replicates: 12,
        pre_trend_lead: 2,
        ..TargetRefutationConfig::default()
    };
    let outcome = run(
        &prepared,
        PreparedTargetData::Series(&data),
        TargetSuite::Placebo,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    );
    let pre_trend = report(&outcome, TargetRefuterKind::PreTrend);
    assert!(!pre_trend.passed, "the outcome responds to the treatment two periods ahead");
}

#[test]
fn prior_sensitivity_runs_under_a_compiled_prior_grid() {
    let ctx = ExecutionContext::for_tests(35);
    let config = TargetRefutationConfig { replicates: 8, ..TargetRefutationConfig::default() };
    let bayes = || InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200));
    // Many rows: the likelihood dominates every prior on the grid.
    let data = static_data(800, 7, false);
    let prepared = static_study(&data, bayes(), &ctx);
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&data),
        TargetSuite::Cheap,
        &config,
        &TargetRefitOptions::default(),
        &ctx,
    );
    let sensitivity = report(&outcome, TargetRefuterKind::PriorSensitivity);
    assert!(sensitivity.passed, "{:?}", sensitivity.failure);
    // Few rows and a prior a thousand times tighter: the prior decides the target.
    let small = static_data(16, 8, false);
    let prepared = static_study(&small, bayes(), &ctx);
    let options =
        TargetRefitOptions { prior_scale_factors: vec![1e-3], ..TargetRefitOptions::default() };
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&small),
        TargetSuite::Cheap,
        &config,
        &options,
        &ctx,
    );
    let sensitivity = report(&outcome, TargetRefuterKind::PriorSensitivity);
    assert!(!sensitivity.passed, "a dominating prior moves the target");
}

/// `y = f(a) + 0.5 b + e` with `a` confounded by `b`.
fn dose_data(n: u32, curved: bool) -> TabularData {
    let mut rng = CausalRng::from_seed(9);
    let b: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let a: Vec<f64> = b.iter().map(|b| 2.0 + 0.3 * b + 0.5 * standard_normal(&mut rng)).collect();
    let y: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(a, b)| {
            let f = if curved { (2.0 * a).exp() / 10.0 } else { 2.0 * a };
            f + 0.5 * b + 0.05 * standard_normal(&mut rng)
        })
        .collect();
    TabularData::from_f64_columns([("a", a.as_slice()), ("b", b.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn derivative_study(data: &TabularData, ctx: &ExecutionContext) -> PreparedStudy {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(1, 0), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: v(2),
            treatment: v(0),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        })))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.35), ..Default::default() })
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}

#[test]
fn derivative_diagnostics_read_the_studys_fitted_response() {
    let ctx = ExecutionContext::for_tests(36);
    let config = TargetRefutationConfig { replicates: 4, ..TargetRefutationConfig::default() };
    let options =
        TargetRefitOptions { derivative_step: Some(0.5), ..TargetRefitOptions::default() };
    let linear = dose_data(800, false);
    let prepared = derivative_study(&linear, &ctx);
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&linear),
        TargetSuite::Cheap,
        &config,
        &options,
        &ctx,
    );
    let fd = report(&outcome, TargetRefuterKind::FiniteDifferenceConsistency);
    assert!((fd.refuted[0] - 2.0).abs() < 0.3, "curve slope {:?}", fd.refuted);
    assert!(report(&outcome, TargetRefuterKind::LocalLinearity).passed);
    // exp(2a): over a step of 0.5 the curve bends by tanh(0.5) of its slope, above the bar.
    let curved = dose_data(800, true);
    let prepared = derivative_study(&curved, &ctx);
    let outcome = run(
        &prepared,
        PreparedTargetData::Tabular(&curved),
        TargetSuite::Cheap,
        &config,
        &options,
        &ctx,
    );
    assert!(!report(&outcome, TargetRefuterKind::LocalLinearity).passed);
}
