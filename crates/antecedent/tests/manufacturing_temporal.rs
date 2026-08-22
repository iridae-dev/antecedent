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
fn manufacturing_dbn_posterior_bayesian_sustained_envelope() {
    use antecedent::discovery::{
        BayesianDiscoverParams, GraphMcmcSchedule, discover_dbn_posterior,
    };
    use antecedent_core::TemporalPolicy;

    let (series, _g, q) = white_noise_pulse_series(400, 42);
    let q = q.with_policy(TemporalPolicy::sustained(-1, -1));
    let ctx = ExecutionContext::for_tests(11);
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
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/manufacturing/temporal_pressure_defect/expected.json"
    ))
    .unwrap();
    let n = usize::try_from(expected["n"].as_u64().unwrap()).unwrap();
    let expected_ate = expected["expected_ate"].as_f64().unwrap();
    let tolerance = expected["ate_abs_tolerance"].as_f64().unwrap();
    let interval_ns = expected["sampling_interval_ns"].as_u64().unwrap();
    let treatment_lag = u32::try_from(expected["treatment_lag"].as_u64().unwrap()).unwrap();
    let horizon_steps = u32::try_from(expected["horizon_steps"].as_u64().unwrap()).unwrap();
    assert_eq!(expected["treatment"], "pressure");
    assert_eq!(expected["outcome"], "defect");

    let (series, g, q) = manufacturing_series(n);
    assert_eq!(series.time_index().regularity, SamplingRegularity::Regular { interval_ns });
    assert_eq!(q.policy, TemporalPolicy::pulse(-i32::try_from(treatment_lag).unwrap()));
    assert_eq!(q.horizon_steps, horizon_steps);
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
    assert!(
        (mean - expected_ate).abs() < tolerance,
        "posterior mean={mean} expected ~{expected_ate}"
    );
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

// ---------------------------------------------------------------------------
// PulseEffect / SustainedEffect × TemporalDag × Frequentist × {cheap, full}
// refuter coverage.
//
// `parity/support_licensed.toml` licenses these 8 cells with
// `staged = true`, but before this addition no test anywhere ever passed
// `RefuteSuite::Cheap` or `RefuteSuite::Full` through the temporal Frequentist
// path — every existing test used `RefuteSuite::None`. `execute_temporal`
// (crates/antecedent/src/analysis/execute/temporal_path.rs) does wire
// `run_refuters` with a `TemporalRefitContext`, so this closes a real gap
// rather than a paper one: it exercises the suite and pins which refuters
// the temporal-unfolded design actually runs (several of the static-DAG
// refuters — Overlap, `OverlapRule`, Riesz, the drop-covariate Graph
// refuter — are `NotApplicable` for a temporal unfolded design and are
// silently dropped by `ValidationSuite::reports_only`, not surfaced as a
// "skipped" report).
//
// The series is the same deterministic `t`/`y` design
// `conformance/response/temporal_dose_horizon` pins
// (`y_i = 1 + 2*t_(i-1) + 3*t_(i-2)`, n chosen so both lag-aligned OLS
// windows are exactly orthogonal): a `Pulse` at treatment lag 1 recovers the
// structural coefficient 2.0 exactly, and the single-step `Sustained` window
// at the same offset matches it (`parity/support_licensed.toml`'s own
// "Projection matches Pulse at the same offset" language for the Sustained
// cells).
// ---------------------------------------------------------------------------

fn dose_horizon_series() -> (TimeSeriesData, TemporalDag) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_dose_horizon/expected.json"
    ))
    .unwrap();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let t: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            _ => -1.0,
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect();

    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();

    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    (series, graph)
}

fn dose_horizon_pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

/// Single-step Sustained window at `-treatment_lag`, the only licensed Sustained
/// form (multi-step remains estimator-refused per `parity/support_licensed.toml`).
fn dose_horizon_sustained_query() -> TemporalEffectQuery {
    let mut q =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0);
    q.policy = TemporalPolicy::sustained(-1, -1);
    q.horizon_steps = 1;
    q
}

/// Build + `prepare()` + `estimate_series()` (never the one-shot `.run()`) for a
/// Frequentist temporal query, on either the explicit `TemporalDag` or the
/// `AcceptedGraph::temporal_dag` wrapper for the `accepted` structure axis.
///
/// Returns the `estimate_series` `Result` directly (does not `unwrap()`): see
/// `dose_horizon_full_suite_data_subset_masking_error` below — `RefuteSuite::Full`
/// genuinely errors at every one of these coordinates, so callers must be able to
/// assert the failure rather than have it hidden behind a panic message.
fn run_temporal_frequentist_case(
    accepted: bool,
    query: TemporalEffectQuery,
    suite: RefuteSuite,
) -> Result<antecedent::result::StudyResult, antecedent::CausalError> {
    let (series, graph) = dose_horizon_series();
    let builder = Study::series(series.clone());
    let builder = if accepted {
        builder.graph(AcceptedGraph::temporal_dag(graph))
    } else {
        builder.graph(graph)
    };
    let analysis =
        builder.temporal_query(query).refute(suite).bootstrap_replicates(0).build().unwrap_or_else(
            |e| panic!("build failed for accepted={accepted}, suite={suite:?}: {e}"),
        );
    let ctx = ExecutionContext::for_tests(7);
    let prepared = analysis
        .prepare(&ctx)
        .unwrap_or_else(|e| panic!("prepare failed for accepted={accepted}, suite={suite:?}: {e}"));
    prepared.estimate_series(&series, &ctx)
}

fn assert_dose_horizon_ate(result: &antecedent::result::StudyResult) {
    assert_eq!(result.support_status.unwrap().as_str(), "licensed");
    assert!(
        (result.estimate.ate - 2.0).abs() < 1e-6,
        "structural contrast is exactly 2.0 on the orthogonal-design fixture; got ate={}",
        result.estimate.ate
    );
}

/// `RefuteSuite::Cheap` on a temporal-unfolded design actually runs exactly one
/// refuter: `OverlapRefuter` is `NotApplicable` here (`ValidatorId::Overlap` in
/// `crates/antecedent-validate/src/suite.rs` refuses when `problem.temporal.is_some()`),
/// so only E-value is left in `result.refutations`. `parity/support_licensed.toml`'s
/// "Cheap suite is overlap + E-value" limitations text for these cells describes the
/// suite's *configuration*, not what actually executes at this coordinate — that's the
/// finding this test pins. The Overlap skip itself is not lost: `result.diagnostics`
/// carries a `refute.validator.not_applicable` entry for it (see
/// `assert_cheap_temporal_overlap_skip_diagnostic`) — a per-run skip, not the support
/// matrix's permanent `not_applicable`.
fn assert_cheap_temporal_refuters(result: &antecedent::result::StudyResult) {
    let names: Vec<&str> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
    assert_eq!(
        names,
        vec!["sensitivity.evalue"],
        "cheap suite on TemporalDag Frequentist runs only E-value (Overlap is \
         NotApplicable for temporal unfolded designs, but that skip is now a diagnostic, \
         not a silent drop)"
    );
    assert!(result.refutations[0].informative, "e-value must be informative here");
}

/// The `OverlapRefuter` skip `assert_cheap_temporal_refuters` documents must be visible
/// to a caller who only reads `result.diagnostics`, and must say plainly that it is a
/// per-run skip rather than the support matrix's permanent `SupportRefusal::NotApplicable`
/// / wire `not_applicable` (`crates/antecedent/tests/manufacturing_temporal.rs` and
/// `parity/support_licensed.toml`'s `staged = true` on this exact cell prove the two are
/// not the same claim: this cell is licensed and runs, it just skips one validator).
fn assert_cheap_temporal_overlap_skip_diagnostic(result: &antecedent::result::StudyResult) {
    let skip = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "refute.validator.not_applicable")
        .unwrap_or_else(|| {
            panic!(
                "expected a refute.validator.not_applicable diagnostic; got codes {:?}",
                result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
            )
        });
    assert!(
        skip.fields.iter().any(|(k, v)| k.as_ref() == "validator" && v.as_ref() == "overlap"),
        "expected the skip diagnostic to name the overlap validator; got fields {:?}",
        skip.fields
    );
    assert!(
        skip.message.contains("per-run") && skip.message.contains("not a permanent"),
        "diagnostic message must distinguish this per-run skip from the support matrix's \
         permanent not_applicable state; got: {}",
        skip.message
    );
}

/// `RefuteSuite::Full` on `TemporalDag` Frequentist Pulse/Sustained now runs the full
/// ATE-shaped falsification + stability suite to completion (it previously errored
/// before producing any result — see the fixed root cause below).
///
/// Root cause (fixed): `DataSubsetRefuter` (`crates/antecedent-validate/src/data_subset.rs`)
/// used to keep a random ~80% of rows by calling `TabularData::with_analysis_mask` — it
/// *masked* excluded rows rather than physically dropping them. `refit_effect`'s temporal
/// branch (`crates/antecedent-validate/src/common.rs`) rebuilt a `TimeSeriesData` from that
/// masked table and called `TemporalLinearAdjustment::prepare_with_extras`, which
/// lag-gathers through `LaggedSample::prepare` (`crates/antecedent-data/src/sample.rs`).
/// That gather calls `ensure_unmasked`, which unconditionally rejects any non-fully-valid
/// analysis mask (the lag map indexes raw rows by position, so a masked-out row would
/// silently corrupt the lag alignment if allowed through). Physically dropping the
/// *interior* holes instead would have dodged that error but not the underlying problem:
/// rows on either side of a dropped interior row are no longer adjacent in time, so a lag
/// gather across the seam would quietly average together samples separated by more than
/// the intended step — a confidently wrong refuter number, not a fix.
///
/// The actual fix: `DataSubsetRefuter` now keeps one random *contiguous* window of rows for
/// temporal (non-panel) designs (`with_contiguous_row_window` in
/// `crates/antecedent-validate/src/common.rs`) instead of a scattered row mask. Every
/// retained row's immediate predecessor in the window is still its immediate predecessor in
/// the original series, so lag-1/lag-2/etc. mean exactly what they meant on the full data,
/// and `refit_effect` rebuilds the series `TimeIndex` at the window's (shorter) length
/// instead of reusing the original.
fn assert_full_suite_data_subset_refuter_ran(result: &antecedent::result::StudyResult) {
    let names: Vec<&str> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
    assert_eq!(
        names,
        vec![
            "placebo.treatment",
            "random.common_cause",
            "unobserved.common_cause",
            "dummy.outcome",
            "sensitivity.evalue",
            "sensitivity.linear",
            "sensitivity.partial_linear",
            "sensitivity.nonparametric",
            "bootstrap.ci_coverage",
            "data.subset",
        ],
        "RefuteSuite::Full on TemporalDag Frequentist runs this fixed refuter set"
    );
    let subset = names
        .iter()
        .zip(result.refutations.iter())
        .find(|(n, _)| **n == "data.subset")
        .map(|(_, r)| r)
        .expect("data.subset refuter must have run");
    // The structural fixture (y_i = 1 + 2*t_(i-1) + 3*t_(i-2), no noise) is exactly
    // recovered by OLS on *any* complete contiguous window with enough variation, so the
    // contiguous-window subset refit reproduces the original ATE (2.0) to floating point:
    // a genuine, non-degenerate data-subset check that keeps lag semantics valid, not a
    // trivially-passing no-op.
    assert!(
        (subset.original_ate - 2.0).abs() < 1e-6,
        "original ATE should be exactly 2.0 on the orthogonal-design fixture; got {}",
        subset.original_ate
    );
    assert!(
        (subset.refuted_ate - 2.0).abs() < 1e-6,
        "contiguous-window subset ATE should match the structural coefficient 2.0 \
         (lag semantics preserved); got {}",
        subset.refuted_ate
    );
    assert!(
        (subset.comparison - 1.0).abs() < 1e-6,
        "subset ATE distribution should be fully consistent with the original estimate \
         (p≈1.0 on the noiseless fixture); got {}",
        subset.comparison
    );
    assert!(subset.passed, "data.subset refuter should pass on this design");
    assert_eq!(subset.replicates, 20);
}

#[test]
fn dose_horizon_pulse_explicit_cheap_runs_evalue_only() {
    let result =
        run_temporal_frequentist_case(false, dose_horizon_pulse_query(), RefuteSuite::Cheap)
            .unwrap();
    assert_dose_horizon_ate(&result);
    assert_cheap_temporal_refuters(&result);
    assert_cheap_temporal_overlap_skip_diagnostic(&result);
}

/// See `assert_full_suite_data_subset_refuter_ran`: Full now completes here.
#[test]
fn dose_horizon_pulse_explicit_full_completes_with_data_subset_refuter() {
    let result =
        run_temporal_frequentist_case(false, dose_horizon_pulse_query(), RefuteSuite::Full)
            .expect("RefuteSuite::Full completes for TemporalDag Frequentist");
    assert_dose_horizon_ate(&result);
    assert_full_suite_data_subset_refuter_ran(&result);
}

#[test]
fn dose_horizon_pulse_accepted_cheap_runs_evalue_only() {
    let result =
        run_temporal_frequentist_case(true, dose_horizon_pulse_query(), RefuteSuite::Cheap)
            .unwrap();
    assert_dose_horizon_ate(&result);
    assert_cheap_temporal_refuters(&result);
}

/// See `assert_full_suite_data_subset_refuter_ran`: Full now completes here.
#[test]
fn dose_horizon_pulse_accepted_full_completes_with_data_subset_refuter() {
    let result = run_temporal_frequentist_case(true, dose_horizon_pulse_query(), RefuteSuite::Full)
        .expect("RefuteSuite::Full completes for TemporalDag Frequentist");
    assert_dose_horizon_ate(&result);
    assert_full_suite_data_subset_refuter_ran(&result);
}

#[test]
fn dose_horizon_sustained_explicit_cheap_runs_evalue_only() {
    let result =
        run_temporal_frequentist_case(false, dose_horizon_sustained_query(), RefuteSuite::Cheap)
            .unwrap();
    assert_dose_horizon_ate(&result);
    assert_cheap_temporal_refuters(&result);
}

/// See `assert_full_suite_data_subset_refuter_ran`: Full now completes here.
#[test]
fn dose_horizon_sustained_explicit_full_completes_with_data_subset_refuter() {
    let result =
        run_temporal_frequentist_case(false, dose_horizon_sustained_query(), RefuteSuite::Full)
            .expect("RefuteSuite::Full completes for TemporalDag Frequentist");
    assert_dose_horizon_ate(&result);
    assert_full_suite_data_subset_refuter_ran(&result);
}

#[test]
fn dose_horizon_sustained_accepted_cheap_runs_evalue_only() {
    let result =
        run_temporal_frequentist_case(true, dose_horizon_sustained_query(), RefuteSuite::Cheap)
            .unwrap();
    assert_dose_horizon_ate(&result);
    assert_cheap_temporal_refuters(&result);
}

/// See `assert_full_suite_data_subset_refuter_ran`: Full now completes here.
#[test]
fn dose_horizon_sustained_accepted_full_completes_with_data_subset_refuter() {
    let result =
        run_temporal_frequentist_case(true, dose_horizon_sustained_query(), RefuteSuite::Full)
            .expect("RefuteSuite::Full completes for TemporalDag Frequentist");
    assert_dose_horizon_ate(&result);
    assert_full_suite_data_subset_refuter_ran(&result);
}

/// `PlaceboAndRcc` (Placebo + `RandomCommonCause`, no `DataSubsetRefuter`) runs exactly
/// those two refuters on this design. Historically this isolation was what pinned the
/// since-fixed `DataSubsetRefuter` masking bug to that one refuter rather than the
/// temporal refit path in general (see `assert_full_suite_data_subset_refuter_ran`);
/// `Full` now also succeeds on the identical design, so this test just confirms the
/// suite-restriction behavior on its own.
#[test]
fn dose_horizon_pulse_explicit_placebo_and_rcc_succeed() {
    let result = run_temporal_frequentist_case(
        false,
        dose_horizon_pulse_query(),
        RefuteSuite::PlaceboAndRcc,
    )
    .unwrap();
    assert_dose_horizon_ate(&result);
    let names: Vec<&str> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
    assert_eq!(names, vec!["placebo.treatment", "random.common_cause"]);
}
