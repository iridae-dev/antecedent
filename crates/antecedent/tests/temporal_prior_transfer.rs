//! Fail-closed prior transfer onto licensed Bayesian temporal cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, InterventionSequence,
    ResponseFunctional, ResponseQuery, SequencedIntervention, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_io::{
    DesignVariableRole, DesignVariableSummary, EstimandFingerprint, PriorCatalog, PriorMapping,
    PriorSourceMeta, PriorSourceRef, TargetDesign,
};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/bayesian/temporal_prior_transfer/expected.json"
    ))
    .unwrap()
}

fn series_xy(n: usize, noise: f64, seed: u64) -> TimeSeriesData {
    let ctx = ExecutionContext::for_tests(seed);
    let mut rng = ctx.rng.stream(1);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.04).sin();
        if t > 0 {
            y[t] = 0.9 * x[t - 1] + noise * (2.0 * rng.next_f64() - 1.0);
        }
    }
    TimeSeriesData::from_f64_columns([("pressure", x.as_slice()), ("defect", y.as_slice())], 1)
        .unwrap()
}

fn series_xyw(n: usize, seed: u64) -> TimeSeriesData {
    let ctx = ExecutionContext::for_tests(seed);
    let mut rng = ctx.rng.stream(2);
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        w[t] = 2.0 * rng.next_f64() - 1.0;
        x[t] = ((t as f64) * 0.04).sin() + 0.4 * w[t];
        if t > 0 {
            y[t] = 0.2 * x[t - 1] + 0.35 * w[t] + 0.25 * (2.0 * rng.next_f64() - 1.0);
        }
    }
    TimeSeriesData::from_f64_columns(
        [("pressure", x.as_slice()), ("defect", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap()
}

fn graph_xy() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), antecedent_core::Lag::from_raw(1))
        .unwrap();
    let y0 =
        ensure_lagged(&mut graph, VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

fn graph_xyw() -> TemporalDag {
    let mut graph = graph_xy();
    let w0 =
        ensure_lagged(&mut graph, VariableId::from_raw(2), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
    let y0 =
        ensure_lagged(&mut graph, VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn sustained_query() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
}

fn curve_query(pin: &serde_json::Value) -> ResponseQuery {
    let grid: Vec<f64> =
        pin["grid"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let horizons: Vec<u32> = pin["horizons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(grid)),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(-1), None).unwrap())
}

fn bayes(n_draws: usize) -> BayesianConfig {
    BayesianConfig::conjugate().n_draws(n_draws).prior_scale(10.0)
}

fn design_xy() -> Vec<DesignVariableSummary> {
    vec![
        DesignVariableSummary::new("pressure", DesignVariableRole::Treatment),
        DesignVariableSummary::new("defect", DesignVariableRole::Outcome),
    ]
}

fn fit_pulse(
    series: TimeSeriesData,
    n_draws: usize,
    seed: u64,
) -> (antecedent::result::StudyResult, Vec<u8>) {
    let result = Study::series(series)
        .graph(graph_xy())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(bayes(n_draws)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap();
    let bytes =
        antecedent::io::encode_causal_posterior_bytes(result.posterior.as_ref().unwrap(), "source")
            .unwrap();
    (result, bytes)
}

fn catalog_for(
    artifact_id: &str,
    query_kind: &str,
    bytes: &[u8],
    mapping: Option<PriorMapping>,
    outcome: &str,
) -> PriorCatalog {
    let mut meta = PriorSourceMeta::new(
        artifact_id,
        EstimandFingerprint::new(query_kind, "pressure", outcome),
        "NonparametricallyIdentified",
    )
    .with_design(design_xy());
    if let Some(mapping) = mapping {
        meta = meta.with_mapping(mapping);
    }
    PriorCatalog::from_sources(vec![PriorSourceRef::with_bytes(meta, bytes.to_vec())])
}

#[test]
fn same_design_pulse_consumes_named_source_target_and_filter() {
    let pin = fixture();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let n_draws = usize::try_from(pin["n_draws"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let truth = pin["true_effect"].as_f64().unwrap();
    let atol = pin["atol"].as_f64().unwrap();
    assert_eq!(
        pin["source_cells"]["same_design_pulse"].as_str().unwrap(),
        "PulseEffect × TemporalDag × explicit × Bayesian × none"
    );
    assert_eq!(
        pin["target_cells"]["same_design_pulse"].as_str().unwrap(),
        "PulseEffect × TemporalDag × explicit × Bayesian × none"
    );
    assert_eq!(pin["compatibility_filter"].as_str().unwrap(), "PriorCatalog.filter_compatible");

    let source_series = series_xy(n, 0.05, seed);
    let (source, bytes) = fit_pulse(source_series, n_draws, seed);
    let source_mean = source.posterior.as_ref().unwrap().summaries.mean
        [source.posterior.as_ref().unwrap().effect_column().unwrap()];
    let catalog = catalog_for("match", "pulse", &bytes, None, "defect");
    let target = TargetDesign::new(
        EstimandFingerprint::new("pulse", "pressure", "defect"),
        ["pressure", "defect"],
    );
    let reports = catalog.filter_compatible(&target);
    assert!(reports[0].is_usable(), "same-design pulse must not be rejected: {:?}", reports[0]);
    let chosen =
        catalog.require_usable(&target).expect("same-design pulse must pass filter_compatible");
    assert_eq!(chosen.meta.artifact_id, "match");

    let target_series = series_xy(n, 0.05, seed.wrapping_add(3));
    let analysis = Study::series(target_series.clone())
        .graph(graph_xy())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(bayes(n_draws).prior_from_artifact(bytes, None)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = analysis.prepare(&ctx).unwrap();
    let clicked = prepared.estimate_series(&target_series, &ctx).unwrap();
    let mean = clicked.posterior.as_ref().unwrap().summaries.mean
        [clicked.posterior.as_ref().unwrap().effect_column().unwrap()];
    assert!(
        (mean - truth).abs() < atol,
        "same-design pulse mean={mean} truth={truth} source={source_mean}"
    );
}

#[test]
fn same_design_sustained_and_response_curve_ride_staged_path() {
    let pin = fixture();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let n_draws = usize::try_from(pin["n_draws"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let truth = pin["true_effect"].as_f64().unwrap();
    let atol = pin["atol"].as_f64().unwrap();
    let (_, bytes) = fit_pulse(series_xy(n, 0.05, seed), n_draws, seed);

    let sustained_catalog = catalog_for(
        "match",
        "sustained",
        &bytes,
        Some(PriorMapping::IdenticalCoefficientSubspace),
        "defect",
    );
    let sustained_target = TargetDesign::new(
        EstimandFingerprint::new("sustained", "pressure", "defect"),
        ["pressure", "defect"],
    );
    assert!(sustained_catalog.require_usable(&sustained_target).is_ok());
    assert_eq!(
        pin["target_cells"]["same_design_sustained"].as_str().unwrap(),
        "SustainedEffect × TemporalDag × explicit × Bayesian × none"
    );

    let sustained_series = series_xy(n, 0.05, seed.wrapping_add(5));
    let analysis =
        Study::series(sustained_series.clone())
            .graph(graph_xy())
            .temporal_query(sustained_query())
            .inference(InferenceMode::Bayesian(bayes(n_draws).prior_from_artifact(
                bytes.clone(),
                Some(PriorMapping::IdenticalCoefficientSubspace),
            )))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = analysis.prepare(&ctx).unwrap();
    let clicked = prepared.estimate_series(&sustained_series, &ctx).unwrap();
    let mean = clicked.posterior.as_ref().unwrap().summaries.mean
        [clicked.posterior.as_ref().unwrap().effect_column().unwrap()];
    assert!((mean - truth).abs() < atol, "same-design sustained mean={mean}");

    let curve_catalog = catalog_for(
        "match",
        "pulse",
        &bytes,
        Some(PriorMapping::IdenticalCoefficientSubspace),
        "defect",
    );
    let curve_target = TargetDesign::new(
        EstimandFingerprint::new("response_curve", "pressure", "defect"),
        ["pressure", "defect"],
    );
    assert!(curve_catalog.require_usable(&curve_target).is_ok());
    assert_eq!(
        pin["target_cells"]["same_design_response_curve"].as_str().unwrap(),
        "ResponseCurve × TemporalDag × explicit × Bayesian × none"
    );

    let curve_series = series_xy(n, 0.05, seed.wrapping_add(7));
    let analysis = Study::series(curve_series.clone())
        .graph(graph_xy())
        .query(CausalQuery::Response(curve_query(&pin)))
        .inference(InferenceMode::Bayesian(
            bayes(n_draws)
                .prior_from_artifact(bytes, Some(PriorMapping::IdenticalCoefficientSubspace)),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = analysis.prepare(&ctx).unwrap();
    let clicked = prepared.estimate_series(&curve_series, &ctx).unwrap();
    assert!(clicked.response.is_some(), "temporal ResponseCurve must estimate under transfer");
}

#[test]
fn mapped_effect_transfer_and_incompatible_catalog_fail_closed() {
    let pin = fixture();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let n_draws = usize::try_from(pin["n_draws"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let (_, bytes) = fit_pulse(series_xy(n, 0.05, seed), n_draws, seed);

    let mapped = catalog_for(
        "mapped",
        "pulse",
        &bytes,
        Some(PriorMapping::EffectFunctional { source_quantity: "ate".into() }),
        "defect",
    );
    let target = TargetDesign::new(
        EstimandFingerprint::new("pulse", "pressure", "defect"),
        ["pressure", "defect", "w"],
    );
    assert!(mapped.require_usable(&target).is_ok());

    let baseline = Study::series(series_xyw(n, seed))
        .graph(graph_xyw())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(bayes(n_draws)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap();
    let mapped_series = series_xyw(n, seed);
    let transferred = Study::series(mapped_series.clone())
        .graph(graph_xyw())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(bayes(n_draws).prior_from_artifact(
            bytes.clone(),
            Some(PriorMapping::EffectFunctional { source_quantity: "ate".into() }),
        )))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = transferred.prepare(&ctx).unwrap();
    let clicked = prepared.estimate_series(&mapped_series, &ctx).unwrap();
    let source_truth = pin["true_effect"].as_f64().unwrap();
    let base = baseline.posterior.as_ref().unwrap().summaries.mean
        [baseline.posterior.as_ref().unwrap().effect_column().unwrap()];
    let mapped_mean = clicked.posterior.as_ref().unwrap().summaries.mean
        [clicked.posterior.as_ref().unwrap().effect_column().unwrap()];
    assert!(
        (mapped_mean - source_truth).abs() < (base - source_truth).abs() + 1e-9,
        "mapped pulse should not sit farther from the source effect than the weak baseline \
         mapped={mapped_mean} baseline={base}"
    );

    let wrong = catalog_for("wrong_outcome", "pulse", &bytes, None, "other");
    let pulse_target = TargetDesign::new(
        EstimandFingerprint::new("pulse", "pressure", "defect"),
        ["pressure", "defect"],
    );
    let err = wrong.require_compatible(&pulse_target).unwrap_err();
    assert!(err.to_string().contains("incompatible"), "{err}");
    assert_eq!(pin["incompatible"]["reason_code"].as_str().unwrap(), "estimand_mismatch");
    let reports = wrong.filter_compatible(&pulse_target);
    assert!(matches!(
        &reports[0],
        antecedent_io::CompatibilityReport::Rejected {
            reason: antecedent_io::CompatibilityRejectReason::EstimandMismatch { .. },
            ..
        }
    ));
}

#[test]
fn sequence_refuses_prior_transfer_without_a_new_filter() {
    let pin = fixture();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let n_draws = usize::try_from(pin["n_draws"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let (_, bytes) = fit_pulse(series_xy(n, 0.05, seed), n_draws, seed);
    let needle = pin["sequence_refuses"]["message_contains"].as_str().unwrap();
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(-1),
        },
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::Sequence(seq)]),
    })
    .with_temporal(
        TemporalResponseSpec::new(vec![1u32, 2], TemporalPolicy::pulse(-1), None).unwrap(),
    );
    let err = Study::series(series_xy(n, 0.05, seed))
        .graph(graph_xy())
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Bayesian(bayes(n_draws).prior_from_artifact(bytes, None)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap_err();
    assert!(err.to_string().contains(needle), "{err}");
}
