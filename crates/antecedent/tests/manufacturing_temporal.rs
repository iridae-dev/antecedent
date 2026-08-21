//! Manufacturing-style temporal effect example.
//!
//! Run: `cargo +1.85 test -p antecedent --test manufacturing_temporal -- --nocapture`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::AcceptedGraph;
use antecedent::io::{decode_causal_posterior_bytes, encode_causal_posterior_bytes};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};

fn manufacturing_series(n: usize) -> (TimeSeriesData, TemporalDag, TemporalEffectQuery) {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 1..n {
        pressure[t] = ((t as f64) * 0.04).sin();
        defect[t] = 0.9 * pressure[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(pressure),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(defect),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 3_600_000_000_000 },
            length: n,
        },
    )
    .unwrap();

    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p1, d0).unwrap();

    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    (series, g, q)
}

fn white_noise_pulse_series(
    n: usize,
    seed: u64,
) -> (TimeSeriesData, TemporalDag, TemporalEffectQuery) {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    let mut state = seed;
    for t in 0..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u = (state >> 33) as f64 / (1u64 << 31) as f64;
        pressure[t] = u * 2.0 - 1.0;
        if t > 0 {
            defect[t] = 0.9 * pressure[t - 1];
        }
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(pressure),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(defect),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p1, d0).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    (series, g, q)
}

#[test]
fn manufacturing_dbn_posterior_bayesian_envelope() {
    use antecedent::discovery::{
        BayesianDiscoverParams, GraphMcmcSchedule, discover_dbn_posterior,
    };

    // White-noise treatment so BIC mass lands on the lag edge (not AR loops
    // that hit temporal backdoor history caps).
    let (series, _g, q) = white_noise_pulse_series(400, 42);
    let ctx = ExecutionContext::for_tests(11);
    // Discovery is now standalone (no `.discover_dbn_posterior(..)` builder method):
    // run it explicitly and feed the resulting `GraphPosterior` in via `.graph_posterior(..)`.
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let schedule = GraphMcmcSchedule {
        n_chains: 2,
        n_warmup: 40,
        n_draws: 60,
        ..GraphMcmcSchedule::default()
    };
    let gp = discover_dbn_posterior(
        &series,
        &vars,
        &BayesianDiscoverParams::default(),
        1,
        false,
        &schedule,
        &ctx,
    )
    .unwrap();
    let analysis = Study::series(series.clone())
        .graph_posterior(gp)
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(64).prior_scale(100.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ctx).unwrap();
    let prepared = analysis.prepare(&ctx).unwrap();
    let click = prepared.estimate_series(&series, &ctx).unwrap();
    assert_eq!(click.support_status.unwrap().as_str(), "licensed");
    let post = result.posterior.expect("DBN mixture posterior");
    assert!((0.0..=1.0).contains(&post.unidentified_mass));
    let eq = post.effect_column().unwrap();
    assert!(post.summaries.mean[eq].is_finite());
    assert!((post.summaries.mean[eq] - 0.9).abs() < 0.35, "mean={}", post.summaries.mean[eq]);
}

#[test]
fn manufacturing_dbn_envelope_composed_prior_conflict() {
    use antecedent::discovery::{
        BayesianDiscoverParams, GraphMcmcSchedule, discover_dbn_posterior,
    };
    use antecedent_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        compose_external_priors,
    };
    use antecedent_validate::ConflictPolicy;

    let (series, _g, q) = white_noise_pulse_series(400, 42);
    let ctx = ExecutionContext::for_tests(11);
    // Temporal pulse design is typically intercept + treatment (2 coefs).
    let ncols = 2;
    let mut mean = vec![0.0; ncols];
    mean[1] = 0.9;
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(vec![0.25; ncols]),
    }));
    let sources = Arc::<[ExternalPriorSource]>::from(vec![ExternalPriorSource {
        id: Arc::from("dbn_bank"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }]);
    let baseline = PriorSet::weakly_informative(ncols);
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    let policy = ConflictPolicy::try_new(0.05, 1.0).unwrap();

    // Discovery is now standalone: run it explicitly and feed the resulting
    // `GraphPosterior` in via `.graph_posterior(..)` instead of the removed
    // `.discover_dbn_posterior(..)` builder chain method.
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let schedule = GraphMcmcSchedule {
        n_chains: 2,
        n_warmup: 40,
        n_draws: 60,
        ..GraphMcmcSchedule::default()
    };
    let gp = discover_dbn_posterior(
        &series,
        &vars,
        &BayesianDiscoverParams::default(),
        1,
        false,
        &schedule,
        &ctx,
    )
    .unwrap();

    let analysis = Study::series(series)
        .graph_posterior(gp)
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(48).prior_from_composed(
                Arc::clone(&sources),
                composed,
                Some(policy),
            ),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ctx).unwrap();
    let post = result.posterior.expect("DBN mixture posterior");
    assert!(post.summaries.mean[post.effect_column().unwrap()].is_finite());
    assert!(
        post.conflict_summary.is_some()
            || result.diagnostics.iter().any(|d| d.code.as_ref() == "bayes.prior_bank.conflict"),
        "envelope should surface conflict when policy is set"
    );
}

#[test]
fn supplied_complete_temporal_pag_estimates() {
    let (series, _g, q) = manufacturing_series(200);
    let mut pag = antecedent_graph::TemporalPag::empty();
    let p1 = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_directed(p1, d0).unwrap();
    let analysis = Study::series(series)
        .graph(AcceptedGraph::temporal_pag(pag).unwrap())
        .temporal_query(q)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(7)).unwrap();
    assert!((result.estimate.ate - 0.9).abs() < 0.05, "ate={}", result.estimate.ate);
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "temporal.pag.completed_to_dag"),
        "expected completion diagnostic"
    );
}

#[test]
fn incomplete_temporal_pag_review_required_structured() {
    // MIGRATION NOTE: a directly-supplied `TemporalPag` now converts into an
    // `AcceptedGraph` unconditionally (`AcceptedGraph::temporal_pag` /
    // `impl From<TemporalPag> for AcceptedGraph`, crates/antecedent/src/accepted.rs) —
    // "circles are information, not incompleteness" for a *supplied* structure, unlike
    // a discovery *review artifact* (`TemporalPagReview`), which is the only path that
    // can still produce `CausalError::ReviewRequired`. So this scenario (a supplied
    // `TemporalPag` with an unresolved circle-arrow mark) can no longer raise
    // `ReviewRequired` at all — that variant is structurally unreachable here now.
    // Tracing `Study::compile()` (crates/antecedent/src/analysis/execute/compile.rs)
    // shows the temporal-backdoor path only knows how to proceed when the supplied
    // `TemporalPag` happens to be fully directed (`TemporalPag::try_into_temporal_dag`,
    // crates/antecedent-graph/src/temporal_pag.rs); no class-aware temporal PAG
    // identifier exists yet, so an unresolved circle mark now surfaces as
    // `CausalError::Compile` with a message that says exactly that. The underlying
    // An unresolved circle mark on a *temporal* PAG blocks estimation: no class-aware
    // temporal PAG identifier exists to consume it. The refusal must stay structured —
    // Python drives a review UI off `kind` / `pending_edge_count` / `hint`, so
    // degrading this to a flat message is a user-visible regression, not a shape change.
    let (series, _g, q) = manufacturing_series(80);
    let mut pag = antecedent_graph::TemporalPag::empty();
    let p1 = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_circle_arrow(p1, d0).unwrap();
    let err = AcceptedGraph::temporal_pag(pag).unwrap_err();
    match err {
        antecedent::CausalError::ReviewRequired { kind, pending_edge_count, hint, .. } => {
            assert_eq!(kind, "temporal_pag");
            assert!(pending_edge_count >= 1, "expected pending circle marks");
            // Pin the hint content, as the pre-refactor test did: a non-empty
            // string is not an actionable hint.
            assert!(
                hint.contains("TemporalDag") || hint.contains("PAG"),
                "hint must name the resolution: {hint}"
            );
        }
        other => panic!("expected CausalError::ReviewRequired, got {other:?}"),
    }
    let _ = (series, q);
}

#[test]
fn manufacturing_pressure_defect_bayesian() {
    let (series, g, q) = manufacturing_series(400);
    let analysis = Study::series(series)
        .graph(g)
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(256).prior_scale(100.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(42);
    let result = analysis.run(&ctx).unwrap();

    let post = result.posterior.as_ref().expect("Bayesian temporal should attach posterior");
    let eq = post.effect_column().unwrap();
    let mean = post.summaries.mean[eq];
    assert!((mean - 0.9).abs() < 0.05, "posterior mean={mean} expected ~0.9");
    assert!((result.estimate.ate - mean).abs() < 1e-12);
    let p_below = post.probability_below(0.0).unwrap();
    assert!(p_below.is_finite(), "p_below_zero={p_below}");
    let bytes = encode_causal_posterior_bytes(post, "temporal-pulse").unwrap();
    let (meta, _) = decode_causal_posterior_bytes(&bytes).unwrap();
    assert_eq!(meta.n_draws as usize, post.draws.n_draws);
}

/// A temporal PAG that reached the study via discovery must record the discovering
/// algorithm, not the generic `supplied.` prefix.
///
/// Regression gate. The plan record's `discovery_algorithm` is surfaced to Python and
/// serialized into artifacts, and `temporal_path` reads it to decide whether to emit the
/// `temporal.pag.completed_to_dag` scientific diagnostic — the disclosure that
/// identification went through PAG completion rather than class-aware temporal PAG ID.
/// Recording a discovered graph as "supplied" makes a stored analysis misstate how its
/// structure was obtained.
#[test]
fn discovered_temporal_pag_records_its_algorithm() {
    let (series, _g, q) = manufacturing_series(80);
    let mut pag = antecedent_graph::TemporalPag::empty();
    let p1 = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_directed(p1, d0).unwrap();

    // Arrives through the discovery-accept path, carrying its algorithm id.
    let review = antecedent_graph::TemporalPagReview::from_pag(pag.clone(), "lpcmci");
    let accepted = AcceptedGraph::accept(review).unwrap();
    assert_eq!(accepted.algorithm_id(), Some("lpcmci"));

    let plan = Study::series(series.clone())
        .graph(accepted)
        .temporal_query(q.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .plan(&ExecutionContext::for_tests(7))
        .unwrap();
    assert_eq!(
        plan.logical.record.discovery_algorithm.as_deref(),
        Some("lpcmci.pag_completed_to_dag"),
        "a discovered PAG must not be recorded as `supplied.`"
    );

    // An asserted PAG genuinely is supplied, and keeps that prefix.
    let plan = Study::series(series)
        .graph(AcceptedGraph::temporal_pag(pag).unwrap())
        .temporal_query(q)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .plan(&ExecutionContext::for_tests(7))
        .unwrap();
    assert_eq!(
        plan.logical.record.discovery_algorithm.as_deref(),
        Some("supplied.temporal_pag.completed_to_dag")
    );
}
