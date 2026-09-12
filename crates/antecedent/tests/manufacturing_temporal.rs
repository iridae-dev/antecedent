//! Manufacturing-style temporal effect example.
//!
//! Run: `cargo +1.85 test -p antecedent --test manufacturing_temporal -- --nocapture`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::AcceptedGraph;
use antecedent::discovery::GraphPosterior;
use antecedent::io::{decode_causal_posterior_bytes, encode_causal_posterior_bytes};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, IdentificationStatus, Lag, MeasurementSpec,
    MediationContrast, MediationQuery, RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy,
    ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{mask_is_dag, temporal_dag_from_dbn_masks};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::{
    IdentificationError, TemporalBackdoorIdentifier, TemporalMediationIdentifier,
};
use antecedent_prob::InferenceDiagnostics;

/// Records every progress label so a test can prove identification was
/// computed exactly where the contract says (fresh run, prepare) and never on
/// a prepared estimate or refresh click.
#[derive(Default)]
struct RecordingProgress(std::sync::Mutex<Vec<String>>);

impl antecedent_core::ProgressSink for RecordingProgress {
    fn report(&self, _fraction: f64, stage: &str) {
        self.0.lock().unwrap().push(stage.to_owned());
    }
}

fn recording_ctx(seed: u64) -> (ExecutionContext, Arc<RecordingProgress>) {
    let sink = Arc::new(RecordingProgress::default());
    let mut ctx = ExecutionContext::for_tests(seed);
    ctx.progress = Some(Arc::clone(&sink) as Arc<dyn antecedent_core::ProgressSink>);
    (ctx, sink)
}

fn identify_computations(sink: &RecordingProgress) -> usize {
    sink.0.lock().unwrap().iter().filter(|stage| stage.as_str() == "identify.compute").count()
}

fn cached_count(result: &antecedent::StudyResult) -> usize {
    result.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count()
}

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

fn known_truth_dbn_posterior(
    pin: &serde_json::Value,
    weights: &[f64],
    query: &TemporalEffectQuery,
) -> GraphPosterior {
    // Atom 0 is the valid lag-one effect graph. Atom 1 is also a valid DBN
    // and adds pressure_{t-1} -> pressure_t. Stationarity extends that AR
    // ancestry through every finite history boundary; TemporalBackdoor must
    // return NotCertified at its cap and retain the atom's posterior weight
    // as unidentified rather than letting the parameter prior upgrade it.
    let valid_contemporaneous = pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let noncertified_contemporaneous =
        pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let pressure_lag1_to_defect = pin["identified_atom"]["lag_mask"].as_u64().unwrap();
    let pressure_ar_and_lag1_effect = pin["unidentified_atom"]["lag_mask"].as_u64().unwrap();
    assert!(mask_is_dag(valid_contemporaneous, 2));
    assert!(mask_is_dag(noncertified_contemporaneous, 2));
    assert_eq!(pin["unidentified_atom"]["identification_error"], "NotCertified");
    let variables = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let noncertified_graph = temporal_dag_from_dbn_masks(
        noncertified_contemporaneous,
        pressure_ar_and_lag1_effect,
        2,
        1,
        &variables,
    )
    .unwrap();
    let error = TemporalBackdoorIdentifier::new()
        .identify_temporal(&noncertified_graph, query)
        .unwrap_err();
    assert!(
        matches!(error, IdentificationError::NotCertified { .. }),
        "autoregressive DBN atom must remain unidentified, got {error:?}"
    );

    let contemporaneous_marginals = vec![0.0; 4];
    let lagged_marginals: Vec<f64> = pin["lagged_edge_marginals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    assert_eq!(lagged_marginals, [weights[1], 1.0, 0.0, 0.0]);
    GraphPosterior::new(
        2,
        weights.to_vec(),
        vec![valid_contemporaneous, noncertified_contemporaneous],
        contemporaneous_marginals.clone(),
        contemporaneous_marginals,
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, lagged_marginals)
    .unwrap()
    .with_lag_masks(vec![pressure_lag1_to_defect, pressure_ar_and_lag1_effect])
    .unwrap()
    .with_algorithm("known_truth_fixture")
}

#[allow(clippy::too_many_lines)]
fn assert_manufacturing_dbn_known_truth_mixture(policy: TemporalPolicy, suite: RefuteSuite) {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_effect"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let identified_mass = pin["identified_mass"].as_f64().unwrap();
    let effect_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    assert!((weights[0] - identified_mass).abs() < 1e-12);
    assert!((weights[1] - unidentified_truth).abs() < 1e-12);
    assert!((pin["identified_atom"]["weight"].as_f64().unwrap() - weights[0]).abs() < 1e-12);
    assert!((pin["unidentified_atom"]["weight"].as_f64().unwrap() - weights[1]).abs() < 1e-12);
    assert!((pin["identified_atom"]["effect"].as_f64().unwrap() - effect_truth).abs() < 1e-12);
    match policy {
        TemporalPolicy::Pulse { at } => {
            assert_eq!(at, i32::try_from(pin["pulse_at"].as_i64().unwrap()).unwrap());
        }
        TemporalPolicy::Sustained { from, until } => {
            let pin_from = i32::try_from(pin["sustained_from"].as_i64().unwrap()).unwrap();
            let pin_until = i32::try_from(pin["sustained_until"].as_i64().unwrap()).unwrap();
            if from == until {
                assert_eq!(from, pin_from);
                assert_eq!(until, pin_until);
            } else {
                let multi = &expected["temporal_sustained_multistep"];
                assert_eq!(from, i32::try_from(multi["sustained_from"].as_i64().unwrap()).unwrap());
                assert_eq!(
                    until,
                    i32::try_from(multi["sustained_until"].as_i64().unwrap()).unwrap()
                );
            }
        }
        _ => panic!("known-truth DBN fixture covers Pulse and Sustained only"),
    }

    let (series, _g, q) = white_noise_pulse_series(n, seed);
    let q = q.with_policy(policy);
    let gp = known_truth_dbn_posterior(pin, &weights, &q);

    let (ctx, sink) = recording_ctx(11);
    let analysis = Study::series(series.clone())
        .graph_posterior(gp)
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(256).prior_scale(1_000_000.0),
        ))
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = analysis.clone().run(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1, "fresh DBN-posterior execution identifies");
    let mut prepared = analysis.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2, "prepare identifies the atoms once");
    let click = prepared.estimate_series(&series, &ctx).unwrap();
    let refreshed = prepared.refresh_series(series.clone(), &ctx).unwrap();
    assert_eq!(
        identify_computations(&sink),
        2,
        "prepared estimate and refresh clicks must reuse the frozen atom identification"
    );
    assert_eq!(click.support_status.unwrap().as_str(), "licensed");
    assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    assert!((refreshed.estimate.ate - click.estimate.ate).abs() < 1e-12);
    assert_eq!(cached_count(&fresh), 0, "fresh DBN-posterior execution must identify its atoms");
    assert_eq!(cached_count(&click), 1, "prepared execution must consume its cache exactly once");
    assert_eq!(cached_count(&refreshed), 1, "same-schema refresh must reuse identification");
    let post = fresh.posterior.as_ref().expect("DBN mixture posterior");
    let click_post = click.posterior.as_ref().expect("prepared DBN mixture posterior");
    let refreshed_post = refreshed.posterior.as_ref().expect("refreshed DBN mixture posterior");
    assert_eq!(post.identification, IdentificationStatus::GraphDependent);
    assert_eq!(click_post.identification, IdentificationStatus::GraphDependent);
    assert!((post.unidentified_mass - unidentified_truth).abs() < 1e-12);
    assert!((click_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
    assert!((refreshed_post.unidentified_mass - unidentified_truth).abs() < 1e-12);
    let eq = post.effect_column().unwrap();
    let click_eq = click_post.effect_column().unwrap();
    let refreshed_eq = refreshed_post.effect_column().unwrap();
    assert!((click_post.summaries.mean[click_eq] - post.summaries.mean[eq]).abs() < 1e-12);
    assert!((refreshed_post.summaries.mean[refreshed_eq] - post.summaries.mean[eq]).abs() < 1e-12);
    assert!(
        (post.summaries.mean[eq] - effect_truth).abs() < tolerance,
        "posterior mean={} truth={effect_truth}",
        post.summaries.mean[eq]
    );
    if suite != RefuteSuite::None {
        assert!(!click.refutations.is_empty());
        assert!(!click.predictive_checks.is_empty());
        assert!(
            click.diagnostics.iter().any(|d| d.code.as_ref() == "refute.envelope.effect_mixture"),
            "DBN cheap/full must mix effect refuters across contributing atoms"
        );
        assert!(
            click
                .diagnostics
                .iter()
                .all(|d| d.code.as_ref() != "refute.dbn_posterior.effect_reference")
        );
    }
    if suite == RefuteSuite::Full {
        assert!(click_post.prior_sensitivity.is_some());
    }
}

#[test]
fn manufacturing_dbn_posterior_bayesian_envelope() {
    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        assert_manufacturing_dbn_known_truth_mixture(TemporalPolicy::pulse(-1), suite);
    }
}

#[test]
fn manufacturing_dbn_posterior_frequentist_shared_block_bootstrap() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_effect"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let (series, _, query) = white_noise_pulse_series(n, seed);
    let posterior = known_truth_dbn_posterior(pin, &weights, &query);
    let study = Study::series(series.clone())
        .graph_posterior(posterior)
        .temporal_query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(12)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(41);
    let result = study.run(&ctx).unwrap();
    assert!(
        (result.estimate.ate - pin["expected_effect_given_identified"].as_f64().unwrap()).abs()
            < pin["effect_abs_tolerance"].as_f64().unwrap()
    );
    assert!(result.estimate.se_bootstrap.is_some());
    assert_eq!(result.estimate.bootstrap_replicates_ok, Some(12));
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.code.as_ref() == "estimate.dbn_posterior.frequentist"
            && diagnostic.message.contains("shared circular-block")
    }));
}

#[test]
fn manufacturing_dbn_posterior_frequentist_cheap_and_full_mix_atom_refuters() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_effect"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (policy, suite) in [
        (TemporalPolicy::pulse(-1), RefuteSuite::Cheap),
        (TemporalPolicy::sustained(-1, -1), RefuteSuite::Full),
    ] {
        let (series, _, query) = white_noise_pulse_series(n, seed);
        let query = query.with_policy(policy);
        let posterior = known_truth_dbn_posterior(pin, &weights, &query);
        let result = Study::series(series)
            .graph_posterior(posterior)
            .temporal_query(query)
            .inference(InferenceMode::Frequentist)
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(43))
            .unwrap();
        assert!(!result.refutations.is_empty(), "{suite:?} must emit mixed atom refuters");
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.code.as_ref() == "refute.envelope.effect_mixture" })
        );
        assert!(
            (result.estimate.ate - pin["expected_effect_given_identified"].as_f64().unwrap()).abs()
                < pin["effect_abs_tolerance"].as_f64().unwrap()
        );
    }
}

#[test]
fn manufacturing_dbn_posterior_bayesian_sustained_envelope() {
    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        assert_manufacturing_dbn_known_truth_mixture(TemporalPolicy::sustained(-1, -1), suite);
    }
}

#[test]
fn manufacturing_dbn_posterior_bayesian_sustained_multistep_envelope() {
    assert_manufacturing_dbn_known_truth_mixture(
        TemporalPolicy::sustained(-2, -1),
        RefuteSuite::None,
    );
}

#[test]
fn manufacturing_dbn_posterior_multistep_sustained_supports_cheap_without_collapse() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_effect"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let (series, _g, q) = white_noise_pulse_series(n, seed);
    let q = q.with_policy(TemporalPolicy::sustained(-2, -1));
    let gp = known_truth_dbn_posterior(pin, &weights, &q);
    let ctx = ExecutionContext::for_tests(11);
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64).prior_scale(1_000_000.0)),
    ] {
        let result = Study::series(series.clone())
            .graph_posterior(gp.clone())
            .temporal_query(q.clone())
            .inference(inference)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!(result.estimate.ate.is_finite());
        assert!(!result.refutations.is_empty());
    }
}

fn mediation_series(n: usize) -> (TimeSeriesData, MediationQuery) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
    }
    for i in 1..n {
        m[i] = 0.8 * t[i - 1] + 0.12 * (0.43 * i as f64).sin();
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * (0.29 * i as f64).cos();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(m), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let q = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    );
    (series, q)
}

fn known_truth_dbn_mediation_posterior(pin: &serde_json::Value, weights: &[f64]) -> GraphPosterior {
    let identified_c = pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let unidentified_c = pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let identified_l = pin["identified_atom"]["lag_mask"].as_u64().unwrap();
    let unidentified_l = pin["unidentified_atom"]["lag_mask"].as_u64().unwrap();
    assert!(mask_is_dag(identified_c, 3));
    assert!(mask_is_dag(unidentified_c, 3));
    assert_eq!(pin["unidentified_atom"]["identification_error"], "NotCertified");
    let variables = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let unidentified_graph =
        temporal_dag_from_dbn_masks(unidentified_c, unidentified_l, 3, 1, &variables).unwrap();
    let q = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    );
    let error = TemporalMediationIdentifier {
        allow_natural_controlled_alias: true,
        ..TemporalMediationIdentifier::new()
    }
    .identify_with_horizon(&unidentified_graph, &q, 1)
    .unwrap_err();
    assert!(
        matches!(error, IdentificationError::NotCertified { .. }),
        "autoregressive mediation atom must remain unidentified, got {error:?}"
    );
    let contemporaneous_marginals = vec![0.0; 9];
    let lagged_marginals: Vec<f64> = pin["lagged_edge_marginals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![identified_c, unidentified_c],
        contemporaneous_marginals.clone(),
        contemporaneous_marginals,
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, lagged_marginals)
    .unwrap()
    .with_lag_masks(vec![identified_l, unidentified_l])
    .unwrap()
    .with_algorithm("known_truth_fixture")
}

fn assert_dbn_mediation_known_truth_mixture(suite: RefuteSuite) {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_mediation"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let effect_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    let (series, q) = mediation_series(n);
    let gp = known_truth_dbn_mediation_posterior(pin, &weights);

    let (ctx, sink) = recording_ctx(pin["seed"].as_u64().unwrap());
    let analysis = Study::series(series.clone())
        .graph_posterior(gp)
        .query(CausalQuery::Mediation(q))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(2048).prior_scale(10.0),
        ))
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = analysis.clone().run(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1);
    let mut prepared = analysis.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2);
    let click = prepared.estimate_series(&series, &ctx).unwrap();
    let refreshed = prepared.refresh_series(series.clone(), &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2);
    assert_eq!(click.support_status.unwrap().as_str(), "licensed");
    assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    assert!((refreshed.estimate.ate - click.estimate.ate).abs() < 1e-12);
    assert_eq!(cached_count(&fresh), 0);
    assert_eq!(cached_count(&click), 1);
    assert_eq!(cached_count(&refreshed), 1);
    let post = fresh.posterior.as_ref().expect("DBN mediation mixture posterior");
    assert_eq!(post.identification, IdentificationStatus::GraphDependent);
    assert!((post.unidentified_mass - unidentified_truth).abs() < 1e-12);
    let eq = post.effect_column().unwrap();
    assert!(
        (post.summaries.mean[eq] - effect_truth).abs() < tolerance,
        "posterior mean={} truth={effect_truth}",
        post.summaries.mean[eq]
    );
    assert!(
        fresh
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identify.dbn_posterior.per_atom_horizon")
    );
    let mediation = fresh.mediation.as_ref().expect("mediation envelope");
    assert!((mediation.effect.ate - fresh.estimate.ate).abs() < 1e-12);
    if suite != RefuteSuite::None {
        assert!(!click.refutations.is_empty());
        assert!(!click.predictive_checks.is_empty());
        assert!(
            click.diagnostics.iter().any(|d| d.code.as_ref() == "refute.envelope.effect_mixture")
        );
    }
    if suite == RefuteSuite::Full {
        assert!(click.posterior.as_ref().unwrap().prior_sensitivity.is_some());
    }
}

#[test]
fn manufacturing_dbn_posterior_bayesian_mediation_envelope() {
    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        assert_dbn_mediation_known_truth_mixture(suite);
    }
}

#[test]
fn manufacturing_dbn_posterior_mediation_retains_multiple_horizons() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_mediation"];
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let (series, query) = mediation_series(usize::try_from(pin["n"].as_u64().unwrap()).unwrap());
    let mut query = query;
    query.horizons = Arc::from([1u32, 2]);
    let posterior = known_truth_dbn_mediation_posterior(pin, &weights);
    let result = Study::series(series)
        .graph_posterior(posterior)
        .query(CausalQuery::Mediation(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(43))
        .unwrap();
    assert_eq!(result.mediation_grid.as_ref().unwrap().slices.len(), 2);
    assert!(result.posterior.is_none());
    assert!(result.mediation.is_none());
}

#[test]
fn dbn_posterior_response_curve_stays_refused() {
    let (series, _g, q) = white_noise_pulse_series(64, 1);
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let weights = vec![0.7, 0.3];
    let gp = known_truth_dbn_posterior(&pin["temporal_effect"], &weights, &q);
    let response =
        antecedent_core::ResponseQuery::new(antecedent_core::ResponseFunctional::MeanCurve {
            outcome: q.outcome,
            treatment: antecedent_core::ContinuousDomain::new(
                q.treatment,
                antecedent_core::GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            antecedent_core::TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None)
                .unwrap(),
        );
    let err = Study::series(series)
        .graph_posterior(gp)
        .query(CausalQuery::Response(response))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert!(
        matches!(err, antecedent::CausalError::Support { .. })
            || err.to_string().contains("response"),
        "{err}"
    );
}

#[test]
fn manufacturing_dbn_posterior_discovered_prepare_reuses_identification() {
    use antecedent::discovery::{
        BayesianDiscoverParams, GraphMcmcSchedule, discover_dbn_posterior,
    };

    // A discovered posterior carries many atoms whose contemporaneous masks
    // repeat across lag structures; the prepared path must key them without
    // collision and must not re-identify on estimate or refresh clicks.
    let (series, _g, q) = white_noise_pulse_series(400, 42);
    let (ctx, sink) = recording_ctx(11);
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
    assert_eq!(identify_computations(&sink), 0, "structure discovery is not identification");
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
    let fresh = analysis.clone().run(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1);
    let mut prepared = analysis.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2);
    let click = prepared.estimate_series(&series, &ctx).unwrap();
    let refreshed = prepared.refresh_series(series.clone(), &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2, "clicks must not re-identify discovered atoms");
    assert_eq!(click.support_status.unwrap().as_str(), "licensed");
    assert_eq!(cached_count(&fresh), 0);
    assert_eq!(cached_count(&click), 1);
    assert_eq!(cached_count(&refreshed), 1);
    let post = fresh.posterior.as_ref().expect("DBN mixture posterior");
    let click_post = click.posterior.as_ref().expect("prepared DBN mixture posterior");
    let refreshed_post = refreshed.posterior.as_ref().expect("refreshed DBN mixture posterior");
    assert!((0.0..=1.0).contains(&post.unidentified_mass));
    assert!((click_post.unidentified_mass - post.unidentified_mass).abs() < 1e-12);
    assert!((refreshed_post.unidentified_mass - post.unidentified_mass).abs() < 1e-12);
    let eq = post.effect_column().unwrap();
    let click_eq = click_post.effect_column().unwrap();
    let refreshed_eq = refreshed_post.effect_column().unwrap();
    assert!((click_post.summaries.mean[click_eq] - post.summaries.mean[eq]).abs() < 1e-12);
    assert!((refreshed_post.summaries.mean[refreshed_eq] - post.summaries.mean[eq]).abs() < 1e-12);
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
fn supplied_complete_temporal_pag_does_not_bypass_visibility() {
    let (series, _g, q) = manufacturing_series(200);
    let mut pag = antecedent_graph::TemporalPag::empty();
    let p1 = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_directed(p1, d0).unwrap();
    let analysis = Study::series(series)
        .graph(AcceptedGraph::temporal_pag(pag))
        .temporal_query(q)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let error = analysis.run(&ExecutionContext::for_tests(7)).unwrap_err();
    assert!(matches!(error, antecedent::CausalError::Compile { .. }));
    assert!(error.to_string().contains("no identified mass"));
}

#[test]
fn incomplete_temporal_pag_does_not_certify_invisible_effect() {
    let (series, _g, q) = manufacturing_series(200);
    let mut pag = antecedent_graph::TemporalPag::empty();
    let p1 = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_circle_arrow(p1, d0).unwrap();
    let analysis = Study::series(series)
        .graph(AcceptedGraph::temporal_pag(pag))
        .temporal_query(q)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let error = analysis.run(&ExecutionContext::for_tests(7)).unwrap_err();
    assert!(matches!(error, antecedent::CausalError::Compile { .. }));
    assert!(error.to_string().contains("no identified mass"));
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
        Some("lpcmci"),
        "a discovered PAG must not be recorded as `supplied.`"
    );

    // An asserted PAG has no discovery algorithm.
    let plan = Study::series(series)
        .graph(AcceptedGraph::temporal_pag(pag))
        .temporal_query(q)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .plan(&ExecutionContext::for_tests(7))
        .unwrap();
    assert_eq!(plan.logical.record.discovery_algorithm.as_deref(), None);
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

/// Single-step Sustained window at `-treatment_lag`. Multi-step windows are a
/// licensed form on validation=none (`temporal.sequential.gcomp`); this helper
/// keeps the single-step coordinate used by the dose-horizon pins.
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
/// so only E-value is left in `result.refutations`. Pulse/Sustained cheap limitations
/// in `parity/support_licensed.toml` record that executed set, not the static ATE
/// cheap-suite configuration. The Overlap skip itself is not lost: `result.diagnostics`
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
