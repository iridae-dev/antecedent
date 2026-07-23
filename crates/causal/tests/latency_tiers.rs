//! Latency tiers + cancel mid-bootstrap conformance (backlog A).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use std::sync::{Arc, Mutex};

use causal::{AnalysisStageEvent, CausalAnalysis, LatencyMode, RefuteSuite, StageResultSink};
use causal_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    ProgressSink, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId};

/// Confounded linear SCM with structural ATE = 2.
fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        // Box-Muller-ish unit noise from two uniforms.
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let zi = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let e = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos()
            * 0.4;
        let yi = 2.0 * ti + zi + e;
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), dag, query)
}

#[test]
fn interactive_vs_standard_records_mode_and_effort() {
    let (data, dag, query) = confounded_scm(600, 7);

    let interactive = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .latency_mode(LatencyMode::Interactive)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();

    assert!(interactive.estimate.ate.is_finite());
    assert!((interactive.estimate.ate - 2.0).abs() < 0.5, "ate={}", interactive.estimate.ate);
    assert_eq!(
        interactive.performance.latency_mode.as_deref(),
        Some("interactive")
    );
    assert_eq!(interactive.performance.bootstrap_replicates_requested, Some(0));
    assert!(
        interactive.estimate.bootstrap_replicates_ok.is_none()
            || interactive.estimate.bootstrap_replicates_ok == Some(0)
    );
    assert!(
        interactive.performance.stage_timings_ns.iter().any(|(s, _)| s.as_ref() == "identify")
    );

    let standard = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .latency_mode(LatencyMode::Standard)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();

    assert!(standard.estimate.ate.is_finite());
    assert!((standard.estimate.ate - 2.0).abs() < 0.5);
    assert_eq!(standard.performance.latency_mode.as_deref(), Some("standard"));
    assert_eq!(standard.performance.bootstrap_replicates_requested, Some(50));
    let ok = standard.estimate.bootstrap_replicates_ok.expect("bootstrap ok count");
    assert!(ok >= 2, "expected bootstrap survivors, got {ok}");
    assert!(standard.estimate.se_bootstrap.is_some());
    assert_eq!(
        format!("{:?}", interactive.identification.status),
        format!("{:?}", standard.identification.status)
    );
}

struct CancelOnBootstrap {
    token: causal_core::CancellationToken,
}

impl ProgressSink for CancelOnBootstrap {
    fn report(&self, _fraction: f64, stage: &str) {
        if stage == "bootstrap" {
            self.token.cancel();
        }
    }
}

#[test]
fn cancel_mid_bootstrap_yields_partial_not_silent_full() {
    let (data, dag, query) = confounded_scm(400, 11);
    let requested = 80u32;

    let full = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .bootstrap_replicates(requested)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let full_ok = full.estimate.bootstrap_replicates_ok.unwrap_or(0);
    assert_eq!(full_ok, requested);

    let mut ctx = ExecutionContext::for_tests(3);
    let token = ctx.cancellation.clone();
    ctx.progress = Some(Arc::new(CancelOnBootstrap { token }));

    let partial = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(requested)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();

    assert!(partial.estimate.ate.is_finite());
    assert!(partial.performance.cancelled || partial.estimate.bootstrap_cancelled);
    let ok = partial.estimate.bootstrap_replicates_ok.unwrap_or(0);
    assert!(
        ok < requested,
        "cancelled run must not report full replicates (ok={ok}, requested={requested})"
    );
    assert_ne!(ok, full_ok);
}

#[test]
fn interactive_refuses_inline_discovery() {
    use causal::{DiscoveryAccept, FdrControl};

    let (data, dag, query) = confounded_scm(200, 23);
    let err = CausalAnalysis::builder()
        .data(data.clone())
        .discover_pc(0.05, 3, FdrControl::Off, DiscoveryAccept::AutoAccept)
        .query(query.clone())
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Interactive") || msg.contains("discovery"),
        "unexpected: {msg}"
    );

    // Standard one-shot discovery remains a valid script path (may review/fail later).
    let standard = CausalAnalysis::builder()
        .data(data.clone())
        .discover_pc(0.05, 3, FdrControl::Off, DiscoveryAccept::AutoAccept)
        .query(query.clone())
        .latency_mode(LatencyMode::Standard)
        .refute(RefuteSuite::None)
        .build();
    assert!(standard.is_ok(), "{standard:?}");

    let supplied = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .build();
    assert!(supplied.is_ok(), "{supplied:?}");
}

#[test]
fn adaptive_bootstrap_pin_stable_count_and_se() {
    use causal_core::AdaptiveBootstrapBudget;

    let (data, dag, query) = confounded_scm(500, 19);
    let max_reps = 80u32;

    let mut ctx_full = ExecutionContext::for_tests(5);
    ctx_full.adaptive_bootstrap = AdaptiveBootstrapBudget::disabled();
    let full = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .bootstrap_replicates(max_reps)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_full)
        .unwrap();
    let full_se = full.estimate.se_bootstrap.expect("full SE");
    assert_eq!(full.estimate.bootstrap_replicates_ok, Some(max_reps));
    assert!(!full.performance.early_stopped);

    let mut ctx_adapt = ExecutionContext::for_tests(5);
    ctx_adapt.adaptive_bootstrap = AdaptiveBootstrapBudget {
        enabled: true,
        min_replicates: 12,
        se_rel_epsilon: 0.05,
    };
    let a1 = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .bootstrap_replicates(max_reps)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();
    let a2 = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(max_reps)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();

    assert!(a1.performance.early_stopped, "expected adaptive early-stop");
    assert_eq!(
        a1.estimate.bootstrap_replicates_ok, a2.estimate.bootstrap_replicates_ok,
        "fixed seed must pin early-stop replicate count"
    );
    let ok = a1.estimate.bootstrap_replicates_ok.expect("ok count");
    assert!(ok >= 12 && ok < max_reps, "ok={ok}");
    let adapt_se = a1.estimate.se_bootstrap.expect("adaptive SE");
    let rel = (adapt_se - full_se).abs() / full_se.abs().max(1e-12);
    assert!(
        rel < 0.35,
        "adaptive SE={adapt_se} vs full SE={full_se} rel={rel}"
    );
}

#[test]
fn adaptive_draws_pin_stable_count_and_width() {
    use causal::inference::{BayesianConfig, InferenceMode};
    use causal_core::AdaptiveDrawBudget;

    let (data, dag, query) = confounded_scm(400, 23);
    let max_draws = 256usize;

    let mut ctx_full = ExecutionContext::for_tests(9);
    ctx_full.adaptive_draws = AdaptiveDrawBudget::disabled();
    let full = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::laplace().n_draws(max_draws),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_full)
        .unwrap();
    let full_post = full.posterior.as_ref().expect("posterior");
    assert_eq!(full_post.draws.n_draws, max_draws);
    assert!(!full.performance.early_stopped);
    assert!(!full_post.early_stopped);
    let full_width = effect_quantile_width_95(full_post);

    let mut ctx_adapt = ExecutionContext::for_tests(9);
    ctx_adapt.adaptive_draws = AdaptiveDrawBudget {
        enabled: true,
        min_draws: 32,
        quantile_width_rel_epsilon: 0.05,
        ess_target: 10_000.0,
    };
    let a1 = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::laplace().n_draws(max_draws),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();
    let a2 = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::laplace().n_draws(max_draws),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();

    let p1 = a1.posterior.as_ref().expect("adaptive posterior");
    let p2 = a2.posterior.as_ref().expect("adaptive posterior 2");
    assert!(a1.performance.early_stopped, "expected adaptive early-stop");
    assert_eq!(p1.draws.n_draws, p2.draws.n_draws, "fixed seed must pin n_draws");
    assert_eq!(p1.early_stopped, p2.early_stopped);
    assert_eq!(a1.performance.n_draws, a2.performance.n_draws);
    assert!(p1.draws.n_draws >= 32 && p1.draws.n_draws < max_draws, "n={}", p1.draws.n_draws);
    let adapt_width = effect_quantile_width_95(p1);
    let rel = (adapt_width - full_width).abs() / full_width.abs().max(1e-12);
    assert!(
        rel < 0.35,
        "adaptive width={adapt_width} vs full={full_width} rel={rel}"
    );
}

fn effect_quantile_width_95(post: &causal_estimate::CausalPosterior) -> f64 {
    let col = post.effect_column().expect("effect column");
    let vals_src = post.draws.column(col).expect("effect draws");
    let mut vals = vals_src.to_vec();
    let n = vals.len();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = vals[((n as f64) * 0.025) as usize];
    let hi = vals[(((n as f64) * 0.975) as usize).min(n - 1)];
    hi - lo
}

struct RecordingStageSink {
    stages: Mutex<Vec<&'static str>>,
    point_ate: Mutex<Option<f64>>,
    uncertainty_has_boot: Mutex<Option<bool>>,
}

impl StageResultSink for RecordingStageSink {
    fn on_stage(&self, event: &AnalysisStageEvent) {
        self.stages.lock().unwrap().push(event.stage_id());
        match event {
            AnalysisStageEvent::Point { estimate } => {
                *self.point_ate.lock().unwrap() = Some(estimate.ate);
                assert!(
                    estimate.se_bootstrap.is_none(),
                    "point stage must not carry bootstrap SE"
                );
            }
            AnalysisStageEvent::Uncertainty { estimate } => {
                *self.uncertainty_has_boot.lock().unwrap() = Some(estimate.se_bootstrap.is_some());
            }
            _ => {}
        }
    }
}

#[test]
fn progressive_stages_stream_payloads_in_order() {
    let (data, dag, query) = confounded_scm(400, 11);
    let sink = Arc::new(RecordingStageSink {
        stages: Mutex::new(Vec::new()),
        point_ate: Mutex::new(None),
        uncertainty_has_boot: Mutex::new(None),
    });
    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(40)
        .refute(RefuteSuite::None)
        .stage_sink(sink.clone())
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap();

    let stages = sink.stages.lock().unwrap().clone();
    assert_eq!(
        stages,
        vec!["identify", "estimate_point", "uncertainty", "validate"],
        "stages={stages:?}"
    );
    let point_ate = sink.point_ate.lock().unwrap().expect("point ate");
    assert!(point_ate.is_finite());
    assert!((point_ate - result.estimate.ate).abs() < 1e-12);
    assert_eq!(
        *sink.uncertainty_has_boot.lock().unwrap(),
        Some(true),
        "uncertainty stage must fill bootstrap SE"
    );
    assert!(result.estimate.se_bootstrap.is_some());
    let timing_ids: Vec<&str> =
        result.performance.stage_timings_ns.iter().map(|(s, _)| s.as_ref()).collect();
    assert!(
        timing_ids.contains(&"identify")
            && timing_ids.contains(&"estimate_point")
            && timing_ids.contains(&"uncertainty"),
        "timings={timing_ids:?}"
    );
}

