//! Latency tiers + cancel mid-bootstrap conformance (backlog A).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

use std::sync::{Arc, Mutex};

use antecedent::{LatencyMode, RefuteSuite, StageEvent, StageResultSink, Study};
use antecedent_core::{ExecutionContext, ProgressSink, VariableId};

mod common;

// The confounded static ATE study five suites run, in one owner.
use common::fixtures::confounded_scm;

#[test]
fn interactive_vs_standard_records_mode_and_effort() {
    let (data, dag, query) = confounded_scm(600, 7);

    let interactive = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .latency_mode(LatencyMode::Interactive)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();

    assert!(interactive.estimate.ate.is_finite());
    assert!((interactive.estimate.ate - 2.0).abs() < 0.5, "ate={}", interactive.estimate.ate);
    assert_eq!(interactive.performance.latency_mode.as_deref(), Some("interactive"));
    assert_eq!(interactive.performance.bootstrap_replicates_requested, Some(0));
    assert!(
        interactive.estimate.bootstrap_replicates_ok.is_none()
            || interactive.estimate.bootstrap_replicates_ok == Some(0)
    );
    assert!(interactive.performance.stage_timings_ns.iter().any(|(s, _)| s.as_ref() == "identify"));

    let standard = Study::tabular(data)
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
    assert_eq!(standard.performance.bootstrap_replicates_requested, Some(199));
    let ok = standard.estimate.bootstrap_replicates_ok.expect("bootstrap ok count");
    assert!(ok >= 2, "expected bootstrap survivors, got {ok}");
    assert!(standard.estimate.se_bootstrap.is_some());
    assert_eq!(
        format!("{:?}", interactive.identification.status),
        format!("{:?}", standard.identification.status)
    );
}

struct CancelOnBootstrap {
    token: antecedent_core::CancellationToken,
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

    let full = Study::tabular(data.clone())
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

    let partial = Study::tabular(data)
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
fn discovered_graph_builds_under_every_latency_tier() {
    // MIGRATION NOTE: the behavior this test asserted — `.build()` refusing to combine
    // `LatencyMode::Interactive` with inline discovery — has no equivalent left in the
    // current API, and this is a genuine behavior retirement, not a relocation.
    // `.discover_pc(..)` and `DiscoveryAccept` were deleted; discovery is now always a
    // standalone call (`antecedent::discovery::discover_pc`) whose result must be
    // explicitly turned into an `AcceptedGraph` *before* a `Study` exists at all, so
    // "inline discovery under a latency-tagged builder" is no longer an expressible
    // call shape — there is nothing left for `.build()` to refuse. A crate-wide check
    // confirms no `LatencyMode`-vs-discovery-provenance refusal exists anywhere in
    // `analysis/{builder,latency,execute/*}.rs`: `AcceptedGraph::algorithm_id()` (the
    // only signal that would let such a check exist) is set by discovery but has zero
    // production readers. The type system now makes the old failure mode
    // unconstructible rather than refusing it at runtime — arguably a strictly
    // stronger form of the same guarantee, but not the same runtime assertion, so this
    // is flagged rather than silently preserved. See the migration report for detail.
    use antecedent::discovery::{StaticDiscoverParams, discover_pc};
    use antecedent::discovery_defaults::resolve_ci;
    use antecedent::{AcceptedGraph, FdrControl};

    let (data, dag, query) = confounded_scm(200, 23);
    let ctx = ExecutionContext::for_tests(23);
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let params = StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 3,
        fdr: FdrControl::Off.adjustment(),
        ci: resolve_ci("parcorr", None).unwrap(),
        screen_pc: false,
        max_subset: None,
    };
    let discovered = discover_pc(&data, &vars, &params, &ctx).unwrap();
    let mut review = discovered.review;
    // Orient undirected marks FIRST: `orient_edge` pushes the newly-oriented edge onto
    // `pending_edges`, so draining directed edges first would leave the review
    // incomplete again. Snapshot each list before looping, since accept/orient consume
    // `review` and return a new one.
    let pending_undirected: Vec<_> = review.pending_undirected.iter().copied().collect();
    for (a, b) in pending_undirected {
        review = review.orient_edge(a, b).unwrap();
    }
    let pending_directed: Vec<_> = review.pending_edges.iter().copied().collect();
    for (from, to) in pending_directed {
        review = review.accept_edge(from, to);
    }
    assert!(review.is_complete(), "review must be complete before accept");
    let accepted = AcceptedGraph::accept(review).unwrap();

    // Standalone discovery + explicit accept, then built under every latency tier:
    // there is no discovery-provenance check left to distinguish this from a
    // directly-asserted graph.
    let standard = Study::tabular(data.clone())
        .graph(accepted.clone())
        .query(query.clone())
        .latency_mode(LatencyMode::Standard)
        .refute(RefuteSuite::None)
        .build();
    assert!(standard.is_ok(), "{standard:?}");

    let interactive = Study::tabular(data.clone())
        .graph(accepted)
        .query(query.clone())
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .build();
    assert!(
        interactive.is_ok(),
        "a discovered-then-accepted graph is an ordinary AcceptedGraph now; Interactive \
         tier has no discovery-provenance check to refuse it: {interactive:?}"
    );

    let supplied = Study::tabular(data)
        .graph(dag)
        .query(query)
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .build();
    assert!(supplied.is_ok(), "{supplied:?}");
}

#[test]
fn adaptive_bootstrap_pin_stable_count_and_se() {
    use antecedent_core::AdaptiveBootstrapBudget;

    let (data, dag, query) = confounded_scm(500, 19);
    let max_reps = 80u32;

    let mut ctx_full = ExecutionContext::for_tests(5);
    ctx_full.adaptive_bootstrap = AdaptiveBootstrapBudget::disabled();
    let full = Study::tabular(data.clone())
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

    // Opt-in budget: stop once the relative Monte Carlo SE of the bootstrap SE,
    // ≈ 1/√(2(B−1)), is ≤ 10%, i.e. after ⌈1 + 1/(2·0.1²)⌉ = 51 successes.
    let budget = AdaptiveBootstrapBudget { enabled: true, min_replicates: 12, se_rel_epsilon: 0.1 };
    assert_eq!(budget.required_replicates(), 51);
    let mut ctx_adapt = ExecutionContext::for_tests(5);
    ctx_adapt.adaptive_bootstrap = budget;
    let a1 = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .bootstrap_replicates(max_reps)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();
    let a2 = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .bootstrap_replicates(max_reps)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();

    assert!(a1.performance.early_stopped, "expected MC-error-bound early-stop");
    assert_eq!(
        a1.estimate.bootstrap_replicates_ok, a2.estimate.bootstrap_replicates_ok,
        "fixed seed must pin early-stop replicate count"
    );
    // The stop is a function of the success count alone, so it is pinned exactly.
    assert_eq!(a1.estimate.bootstrap_replicates_ok, Some(budget.required_replicates()));
    let adapt_se = a1.estimate.se_bootstrap.expect("adaptive SE");
    // Two SEs with ≈10% and ≈8% relative MC error each; a 3σ band on their ratio.
    let rel = (adapt_se - full_se).abs() / full_se.abs().max(1e-12);
    assert!(rel < 0.4, "adaptive SE={adapt_se} vs full SE={full_se} rel={rel}");

    // A budget whose bound exceeds the request never stops.
    let mut ctx_loose = ExecutionContext::for_tests(5);
    ctx_loose.adaptive_bootstrap = AdaptiveBootstrapBudget::enabled_default();
    assert!(ctx_loose.adaptive_bootstrap.required_replicates() > max_reps);
    let loose = Study::tabular(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(max_reps)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_loose)
        .unwrap();
    assert!(!loose.performance.early_stopped);
    assert_eq!(loose.estimate.bootstrap_replicates_ok, Some(max_reps));
    assert_eq!(loose.estimate.se_bootstrap, Some(full_se));
}

#[test]
fn adaptive_draws_preserve_exact_nig_count_and_width() {
    use antecedent::inference::{BayesianConfig, InferenceMode};
    use antecedent_core::AdaptiveDrawBudget;

    let (data, dag, query) = confounded_scm(400, 23);
    let max_draws = 256usize;

    let mut ctx_full = ExecutionContext::for_tests(9);
    ctx_full.adaptive_draws = AdaptiveDrawBudget::disabled();
    let full = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(max_draws)))
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
    let a1 = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(max_draws)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();
    let a2 = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(max_draws)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx_adapt)
        .unwrap();

    let p1 = a1.posterior.as_ref().expect("adaptive posterior");
    let p2 = a2.posterior.as_ref().expect("adaptive posterior 2");
    assert!(!a1.performance.early_stopped, "exact NIG draws must not use Gaussian early-stop");
    assert_eq!(p1.draws.n_draws, p2.draws.n_draws, "fixed seed must pin n_draws");
    assert_eq!(p1.early_stopped, p2.early_stopped);
    assert_eq!(a1.performance.n_draws, a2.performance.n_draws);
    assert_eq!(p1.draws.n_draws, max_draws);
    let adapt_width = effect_quantile_width_95(p1);
    let rel = (adapt_width - full_width).abs() / full_width.abs().max(1e-12);
    assert!(rel < 1e-12, "NIG width={adapt_width} vs full={full_width} rel={rel}");
}

fn effect_quantile_width_95(post: &antecedent_estimate::CausalPosterior) -> f64 {
    let col = post.effect_column().expect("effect column");
    let vals_src = post.draws.column(col).expect("effect draws");
    let mut vals = vals_src.to_vec();
    let n = vals.len();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lo_idx = ((n as f64) * 0.025) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let hi_idx = (((n as f64) * 0.975) as usize).min(n.saturating_sub(1));
    let lo = vals[lo_idx];
    let hi = vals[hi_idx];
    hi - lo
}

struct RecordingStageSink {
    stages: Mutex<Vec<&'static str>>,
    point_ate: Mutex<Option<f64>>,
    uncertainty_has_boot: Mutex<Option<bool>>,
}

impl StageResultSink for RecordingStageSink {
    fn on_stage(&self, event: &StageEvent) {
        self.stages.lock().unwrap().push(event.stage_id());
        match event {
            StageEvent::Point { estimate } => {
                *self.point_ate.lock().unwrap() = Some(estimate.ate);
                assert!(estimate.se_bootstrap.is_none(), "point stage must not carry bootstrap SE");
            }
            StageEvent::Uncertainty { estimate } => {
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
    let result = Study::tabular(data)
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
