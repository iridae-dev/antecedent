//! Temporal observation pairs on `ResponseCurve` / `InterventionResponse`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, InterventionSequence,
    ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SequencedIntervention,
    TEMPORAL_OBSERVATION_UNLICENSED, TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_observation/expected.json"
    ))
    .unwrap()
}

fn temporal_spec(pin: &serde_json::Value) -> TemporalResponseSpec {
    let horizons: Vec<u32> = pin["horizons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let lag = u32::try_from(pin["treatment_lag"].as_u64().unwrap()).unwrap();
    let at = -i32::try_from(lag).unwrap();
    TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(at), None).unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), antecedent_core::Lag::from_raw(1))
        .unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), antecedent_core::Lag::from_raw(2))
        .unwrap();
    let y0 =
        ensure_lagged(&mut graph, VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

type GeneratedObservationData = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn generate(pin: &serde_json::Value) -> GeneratedObservationData {
    let n = usize::try_from(pin["rows"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let stream = pin["stream"].as_u64().unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let mut rng = ctx.rng.stream(stream);
    let mut t = Vec::with_capacity(n);
    let mut latent = Vec::with_capacity(n);
    let mut selected = Vec::with_capacity(n);
    let mut c_ind = Vec::with_capacity(n);
    let mut c_cox = Vec::with_capacity(n);
    for i in 0..n {
        let tv = 2.0 * rng.next_f64() - 1.0;
        t.push(tv);
        let t1 = if i >= 1 { t[i - 1] } else { 0.0 };
        let t2 = if i >= 2 { t[i - 2] } else { 0.0 };
        let y = 5.0 + 2.0 * t1 + 3.0 * t2 + 0.5 * (2.0 * rng.next_f64() - 1.0);
        latent.push(y);
        let p = 1.0 / (1.0 + (-0.4 * t1).exp());
        selected.push(if rng.next_f64() < p { 1.0 } else { 0.0 });
        c_ind.push(-rng.next_f64().max(1e-12).ln() / 0.07);
        c_cox.push(-rng.next_f64().max(1e-12).ln() / (0.07 * (0.6 * t1).exp()));
    }
    let event_ind: Vec<f64> =
        latent.iter().zip(&c_ind).map(|(&y, &c)| if y <= c { 1.0 } else { 0.0 }).collect();
    (t, latent, selected, c_ind, c_cox, event_ind)
}

fn surface_of(result: &antecedent::result::StudyResult) -> Vec<f64> {
    let response = result.response.as_ref().expect("response");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected surface");
    };
    mean.to_vec()
}

fn max_abs(got: &[f64], truth: &[f64]) -> f64 {
    got.iter().zip(truth).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
}

fn run_curve(
    series: TimeSeriesData,
    query: ResponseQuery,
    ctx: &ExecutionContext,
) -> antecedent::result::StudyResult {
    Study::series(series)
        .graph(graph())
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(ctx)
        .unwrap()
}

fn curve_query(
    pin: &serde_json::Value,
    observation: ObservationSpec,
    assumption: ObservationAssumption,
) -> ResponseQuery {
    let grid: Vec<f64> =
        pin["grid"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(grid)),
        ),
    })
    .with_temporal(temporal_spec(pin))
    .with_observation(observation, [assumption])
}

#[test]
fn temporal_observation_pairs_match_fixture_and_beat_complete_proxy() {
    let pin = fixture();
    let ctx = ExecutionContext::for_tests(pin["seed"].as_u64().unwrap());
    let (t, latent, selected, c_ind, c_cox, event_ind) = generate(&pin);
    let n = t.len();
    let mut event_cox = vec![0.0; n];
    for i in 0..n {
        event_cox[i] = if latent[i] <= c_cox[i] { 1.0 } else { 0.0 };
    }
    let truth: Vec<f64> =
        pin["surface"]["mean"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let atol = pin["atol"].as_f64().unwrap();
    let naive_gap = pin["naive_gap_min"].as_f64().unwrap();
    let id = VariableId::from_raw;

    let cases: Vec<(&str, Vec<f64>, ObservationSpec, ObservationAssumption)> = vec![
        (
            "selected",
            (0..n).map(|i| if selected[i] > 0.5 { latent[i] } else { 0.0 }).collect(),
            ObservationSpec::Selected { latent: id(1), observed: id(1), indicator: id(2) },
            ObservationAssumption::OutcomeIndependentGiven(Arc::from([id(0)])),
        ),
        (
            "right_km",
            (0..n).map(|i| latent[i].min(c_ind[i])).collect(),
            ObservationSpec::RightCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([])),
        ),
        (
            "left_km",
            (0..n).map(|i| -latent[i].min(c_ind[i])).collect(),
            ObservationSpec::LeftCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([])),
        ),
        (
            "right_cox",
            (0..n).map(|i| latent[i].min(c_cox[i])).collect(),
            ObservationSpec::RightCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([id(0)])),
        ),
        (
            "left_cox",
            (0..n).map(|i| -latent[i].min(c_cox[i])).collect(),
            ObservationSpec::LeftCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([id(0)])),
        ),
    ];

    let mut right_cox_error = None;
    for (name, y_obs, spec, assumption) in cases {
        let sign = if name.starts_with("left") { -1.0 } else { 1.0 };
        let expected: Vec<f64> = truth.iter().map(|v| v * sign).collect();
        let series = if name == "selected" {
            TimeSeriesData::from_f64_columns(
                [("t", t.as_slice()), ("y", y_obs.as_slice()), ("r", selected.as_slice())],
                1,
            )
            .unwrap()
        } else {
            let (c, event) = if name.ends_with("cox") {
                let c: Vec<f64> = c_cox.iter().map(|v| v * sign).collect();
                (c, event_cox.clone())
            } else {
                let c: Vec<f64> = c_ind.iter().map(|v| v * sign).collect();
                (c, event_ind.clone())
            };
            TimeSeriesData::from_f64_columns(
                [
                    ("t", t.as_slice()),
                    ("y", y_obs.as_slice()),
                    ("c", c.as_slice()),
                    ("event", event.as_slice()),
                ],
                1,
            )
            .unwrap()
        };
        let query = curve_query(&pin, spec, assumption);
        let result = run_curve(series.clone(), query, &ctx);
        let response = result.response.as_ref().unwrap();
        assert_eq!(
            response.provenance_id.as_ref(),
            "estimate.temporal_response.observation_adjusted"
        );
        assert!(matches!(response.uncertainty, ResponseUncertainty::None));
        let got = surface_of(&result);
        let error = max_abs(&got, &expected);
        assert!(error <= atol, "{name}: {got:?} vs {expected:?} error={error}");
        let naive = run_curve(
            series,
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: id(1),
                treatment: ContinuousDomain::new(
                    id(0),
                    GridSpec::Values(Arc::from(
                        pin["grid"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect::<Vec<_>>(),
                    )),
                ),
            })
            .with_temporal(temporal_spec(&pin)),
            &ctx,
        );
        let naive_error = max_abs(&surface_of(&naive), &expected);
        assert!(
            naive_error > error + naive_gap,
            "{name}: complete-on-proxy must miss; corrected={error} naive={naive_error}"
        );
        if name == "right_cox" {
            right_cox_error = Some(error);
        }
        if name == "right_km" {
            // used below for cox vs km on the informative DGP
            let _ = naive_error;
        }
    }

    let y_cox: Vec<f64> = (0..n).map(|i| latent[i].min(c_cox[i])).collect();
    let km_on_cox = run_curve(
        TimeSeriesData::from_f64_columns(
            [
                ("t", t.as_slice()),
                ("y", y_cox.as_slice()),
                ("c", c_cox.as_slice()),
                ("event", event_cox.as_slice()),
            ],
            1,
        )
        .unwrap(),
        curve_query(
            &pin,
            ObservationSpec::RightCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([])),
        ),
        &ctx,
    );
    let km_error = max_abs(&surface_of(&km_on_cox), &truth);
    let cox_error = right_cox_error.unwrap();
    assert!(
        km_error > cox_error + 0.05,
        "temporal Cox must beat marginal KM: cox={cox_error} km={km_error}"
    );
}

#[test]
fn temporal_right_censor_intervention_matches_fixture_path() {
    let pin = fixture();
    let ctx = ExecutionContext::for_tests(pin["seed"].as_u64().unwrap());
    let (t, latent, _, _, c_cox, _) = generate(&pin);
    let n = t.len();
    let y: Vec<f64> = (0..n).map(|i| latent[i].min(c_cox[i])).collect();
    let event: Vec<f64> = (0..n).map(|i| if latent[i] <= c_cox[i] { 1.0 } else { 0.0 }).collect();
    let series = TimeSeriesData::from_f64_columns(
        [
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("c", c_cox.as_slice()),
            ("event", event.as_slice()),
        ],
        1,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.5))]),
    })
    .with_temporal(temporal_spec(&pin))
    .with_observation(
        ObservationSpec::RightCensored {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            censoring: VariableId::from_raw(2),
            event: VariableId::from_raw(3),
        },
        [ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(0)]))],
    );
    let result = run_curve(series, query, &ctx);
    let got = surface_of(&result);
    let truth: Vec<f64> = pin["intervention_set_0_5"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let error = max_abs(&got, &truth);
    assert!(error <= pin["atol"].as_f64().unwrap(), "Set(0.5) {got:?} vs {truth:?}");
    assert_eq!(
        result.response.as_ref().unwrap().provenance_id.as_ref(),
        "estimate.temporal_response.observation_adjusted"
    );
}

#[test]
fn temporal_observation_outer_block_bootstrap_refits_and_returns_pointwise_bands() {
    let pin = fixture();
    let ctx = ExecutionContext::for_tests(37);
    let (t, latent, selected, _, _, _) = generate(&pin);
    let observed =
        latent.iter().zip(&selected).map(|(&value, &keep)| value * keep).collect::<Vec<_>>();
    let series = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", observed.as_slice()), ("r", selected.as_slice())],
        1,
    )
    .unwrap();
    let query = curve_query(
        &pin,
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(2),
        },
        ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)])),
    );
    let study = Study::series(series)
        .graph(graph())
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(12)
        .build()
        .unwrap();
    let result = study.run(&ctx).unwrap();
    let response = result.response.as_ref().unwrap();
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
        panic!("outer observation bootstrap must return pointwise bands");
    };
    assert_eq!(lower.len(), surface_of(&result).len());
    assert_eq!(upper.len(), lower.len());
    assert!(response.support.diagnostics.iter().any(|diagnostic| {
        diagnostic.id.as_ref() == "response.observation_block_bootstrap"
            && diagnostic.values[1] >= 2.0
            && (diagnostic.values[3] - 12.0).abs() < f64::EPSILON
    }));
    assert!(!response.support.warnings.iter().any(|warning| {
        warning.code.as_ref() == "response.observation_joint_uncertainty_unavailable"
    }));
    let mut limited = ExecutionContext::for_tests(37);
    // Enough for the output alone, but not twelve retained bootstrap surfaces.
    limited.memory.hard_limit_bytes = Some((lower.len() * 40) as u64);
    let error = study.run(&limited).unwrap_err();
    assert!(error.to_string().contains("output bytes"), "{error}");
}

#[test]
fn temporal_sequence_observation_outer_block_bootstrap_returns_pointwise_bands() {
    let pin = fixture();
    let ctx = ExecutionContext::for_tests(39);
    let (t, latent, selected, _, _, _) = generate(&pin);
    let observed =
        latent.iter().zip(&selected).map(|(&value, &keep)| value * keep).collect::<Vec<_>>();
    let series = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", observed.as_slice()), ("r", selected.as_slice())],
        1,
    )
    .unwrap();
    let lag = u32::try_from(pin["treatment_lag"].as_u64().unwrap()).unwrap();
    let at = -i32::try_from(lag).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::Sequence(InterventionSequence::new(vec![
            SequencedIntervention {
                intervention: Intervention::set(VariableId::from_raw(0), Value::f64(0.5)),
                temporal: TemporalPolicy::pulse(at),
            },
            SequencedIntervention {
                intervention: Intervention::set(VariableId::from_raw(0), Value::f64(0.5)),
                temporal: TemporalPolicy::pulse(at + 1),
            },
        ]))]),
    })
    .with_temporal(temporal_spec(&pin))
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(2),
        },
        [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)]))],
    );
    let result = Study::series(series)
        .graph(graph())
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(12)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let response = result.response.as_ref().unwrap();
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
        panic!("Sequence observation bootstrap must return pointwise bands");
    };
    assert_eq!(lower.len(), surface_of(&result).len());
    assert_eq!(upper.len(), lower.len());
    assert!(response.support.diagnostics.iter().any(|diagnostic| {
        diagnostic.id.as_ref() == "response.observation_block_bootstrap"
            && diagnostic.values[1] >= 2.0
    }));
    assert!(!response.support.warnings.iter().any(|warning| {
        warning.code.as_ref() == "response.observation_joint_uncertainty_unavailable"
    }));
}

#[test]
fn compile_refuses_unlicensed_temporal_observation() {
    let pin = fixture();
    let (t, y, _, _, _, _) = generate(&pin);
    let lo = vec![-1.0; t.len()];
    let hi = vec![1.0; t.len()];
    let series = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("lo", lo.as_slice()), ("hi", hi.as_slice())],
        1,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.5])),
        ),
    })
    .with_temporal(temporal_spec(&pin))
    .with_observation(
        ObservationSpec::IntervalCensored {
            latent: VariableId::from_raw(1),
            lower: VariableId::from_raw(2),
            upper: VariableId::from_raw(3),
        },
        [ObservationAssumption::IndependentGiven(Arc::from([]))],
    );
    let ctx = ExecutionContext::for_tests(pin["seed"].as_u64().unwrap());
    let err = Study::series(series)
        .graph(graph())
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err();
    assert!(err.to_string().contains(TEMPORAL_OBSERVATION_UNLICENSED), "stable refuse, got {err}");
}
