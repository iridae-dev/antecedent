//! 1.7 observation pins on incomplete temporal classes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, IntoGraphInput, RefuteSuite, Study};
use antecedent_core::{
    Assumption, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    InterventionSequence, Lag, ObservationAssumption, ObservationSpec, ResponseFunctional,
    ResponseQuery, ResponseUncertainty, SequencedIntervention, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalPag};

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_observation/expected.json"
    ))
    .unwrap()
}

fn class_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_class_observation/expected.json"
    ))
    .unwrap()
}

fn generate(pin: &serde_json::Value) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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
    (t, latent, selected, c_ind, c_cox, vec![0.0; n])
}

fn dummy_z(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.11).sin()).collect()
}

fn dummy_w(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.17).cos()).collect()
}

fn temporal_spec(pin: &serde_json::Value) -> TemporalResponseSpec {
    let horizons: Vec<u32> = pin["horizons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let lag = u32::try_from(pin["treatment_lag"].as_u64().unwrap()).unwrap();
    TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(-i32::try_from(lag).unwrap()), None)
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

fn oriented_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g
}

fn incomplete_cpdag() -> TemporalCpdag {
    let mut g = oriented_cpdag();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(4), Lag::from_raw(1)).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

fn incomplete_pag() -> TemporalPag {
    // Z@-1 → T keeps T→Y visible; W@-1 o-o Z@-1 is an off-path class mark.
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(4), Lag::from_raw(1)).unwrap();
    let w1 = g.add_lagged(VariableId::from_raw(5), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g.insert_directed(z1, t1).unwrap();
    g.insert_circle_circle_with_middle(w1, z1, antecedent_graph::MiddleMark::Empty).unwrap();
    g
}

fn selected_series(t: &[f64], latent: &[f64], selected: &[f64]) -> TimeSeriesData {
    let n = t.len();
    let y: Vec<f64> = (0..n).map(|i| if selected[i] > 0.5 { latent[i] } else { 0.0 }).collect();
    let z = dummy_z(n);
    let w = dummy_w(n);
    TimeSeriesData::from_f64_columns(
        [
            ("t", t),
            ("y", y.as_slice()),
            ("r", selected),
            ("event", selected),
            ("z", z.as_slice()),
            ("w", w.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn censored_series(
    t: &[f64],
    latent: &[f64],
    censor: &[f64],
    event: &[f64],
    sign: f64,
) -> TimeSeriesData {
    let n = t.len();
    let y: Vec<f64> = (0..n).map(|i| sign * latent[i].min(censor[i])).collect();
    let c: Vec<f64> = censor.iter().map(|v| v * sign).collect();
    let z = dummy_z(n);
    let w = dummy_w(n);
    TimeSeriesData::from_f64_columns(
        [
            ("t", t),
            ("y", y.as_slice()),
            ("c", c.as_slice()),
            ("event", event),
            ("z", z.as_slice()),
            ("w", w.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn licensed_pairs(
    pin: &serde_json::Value,
    t: &[f64],
    latent: &[f64],
    selected: &[f64],
    c_ind: &[f64],
    c_cox: &[f64],
) -> Vec<(&'static str, TimeSeriesData, ObservationSpec, ObservationAssumption)> {
    let id = VariableId::from_raw;
    let event_ind: Vec<f64> =
        latent.iter().zip(c_ind).map(|(&y, &c)| if y <= c { 1.0 } else { 0.0 }).collect();
    let event_cox: Vec<f64> =
        latent.iter().zip(c_cox).map(|(&y, &c)| if y <= c { 1.0 } else { 0.0 }).collect();
    let _ = pin;
    vec![
        (
            "Selected x OutcomeIndependentGiven",
            selected_series(t, latent, selected),
            ObservationSpec::Selected { latent: id(1), observed: id(1), indicator: id(2) },
            ObservationAssumption::OutcomeIndependentGiven(Arc::from([id(0)])),
        ),
        (
            "RightCensored x IndependentGiven([])",
            censored_series(t, latent, c_ind, &event_ind, 1.0),
            ObservationSpec::RightCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([])),
        ),
        (
            "LeftCensored x IndependentGiven([])",
            censored_series(t, latent, c_ind, &event_ind, -1.0),
            ObservationSpec::LeftCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([])),
        ),
        (
            "RightCensored x IndependentGiven([T])",
            censored_series(t, latent, c_cox, &event_cox, 1.0),
            ObservationSpec::RightCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([id(0)])),
        ),
        (
            "LeftCensored x IndependentGiven([T])",
            censored_series(t, latent, c_cox, &event_cox, -1.0),
            ObservationSpec::LeftCensored {
                latent: id(1),
                observed: id(1),
                censoring: id(2),
                event: id(3),
            },
            ObservationAssumption::IndependentGiven(Arc::from([id(0)])),
        ),
    ]
}

fn run_pair(
    series: TimeSeriesData,
    graph: impl antecedent::IntoGraphInput,
    query: ResponseQuery,
    inference: InferenceMode,
    bootstrap: u32,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let mut builder = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap);
    if bootstrap == 0 {
        builder = builder.observation_options(antecedent_estimate::ObservationEstimatorOptions {
            selected_correction: antecedent_estimate::SelectedOutcomeCorrection::Ipw,
            ..antecedent_estimate::ObservationEstimatorOptions::default()
        });
    }
    builder.build().unwrap().run(&ExecutionContext::for_tests(13))
}

fn carries_observation_claim(result: &antecedent::StudyResult) -> bool {
    let response = result.response.as_ref().expect("class observation response");
    response.assumptions.entries.iter().any(|record| {
        matches!(
            &record.assumption,
            Assumption::Custom { id, .. }
                if id.as_ref() == "observation.outcome_independent_given"
                    || id.as_ref().starts_with("observation.")
        )
    }) || result.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, Assumption::Custom { id, .. } if id.as_ref().starts_with("observation."))
    })
}

#[test]
fn licensed_observation_pairs_run_on_temporal_cpdag_and_pag() {
    let class = class_pin();
    let pairs: Vec<String> = class["licensed_pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert!(pairs.len() >= 5, "licensed_pairs must name every TemporalDag pair: {pairs:?}");
    let pin = pin();
    let (t_full, latent_full, selected_full, c_ind_full, c_cox_full, _) = generate(&pin);
    let n_cov = 400.min(t_full.len());
    let (t, latent, selected, c_ind, c_cox) = (
        &t_full[..n_cov],
        &latent_full[..n_cov],
        &selected_full[..n_cov],
        &c_ind_full[..n_cov],
        &c_cox_full[..n_cov],
    );
    for (name, series, spec, assumption) in
        licensed_pairs(&pin, t, latent, selected, c_ind, c_cox)
    {
        assert!(pairs.iter().any(|listed| listed == name), "{name} missing from fixture");
        let query = curve_query(&pin, spec, assumption);
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
        ] {
            for (graph_class, graph) in [
                (
                    antecedent::GraphClass::TemporalCpdag,
                    incomplete_cpdag().into_graph_input().0,
                ),
                (antecedent::GraphClass::TemporalPag, antecedent::AcceptedGraph::from(incomplete_pag())),
            ] {
                let result = run_pair(series.clone(), graph, query.clone(), inference.clone(), 0)
                    .unwrap_or_else(|error| {
                        panic!("{name} {graph_class:?} {inference:?}: {error}")
                    });
                assert_eq!(
                    result.certificate.as_ref().expect("certificate").graph_class,
                    graph_class
                );
                assert!(
                    result.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.temporal_class.observation_no_complete_band"
                    }),
                    "{name} missing observation diagnostic"
                );
                assert!(carries_observation_claim(&result), "{name} dropped observation claims");
                assert!(!matches!(
                    result.response.as_ref().unwrap().uncertainty,
                    ResponseUncertainty::PointwiseBand { .. }
                ) || result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.temporal_class.observation_no_complete_band"
                }));
            }
        }
    }
}

#[test]
fn selected_incomplete_cpdag_matches_temporal_observation_surface() {
    let pin = pin();
    let (t, latent, selected, _, _, _) = generate(&pin);
    let truth: Vec<f64> =
        pin["surface"]["mean"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let atol = pin["atol"].as_f64().unwrap();
    let series = selected_series(&t, &latent, &selected);
    let query = curve_query(
        &pin,
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(2),
        },
        ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)])),
    );
    let result = run_pair(series, incomplete_cpdag(), query, InferenceMode::Frequentist, 0).unwrap();
    let structural = result.structural_response.as_ref().unwrap();
    let cells = pin["grid"].as_array().unwrap().len();
    let horizons = pin["horizons"].as_array().unwrap().len();
    let slices: Vec<Vec<f64>> = (0..horizons)
        .map(|horizon| (0..cells).map(|cell| truth[cell * horizons + horizon]).collect())
        .collect();
    let mut checked = 0usize;
    for atom in &structural.atoms {
        let Some(antecedent_core::ResponseValue::Surface { mean, .. }) = atom.value.as_ref() else {
            continue;
        };
        if mean.len() != cells {
            continue;
        }
        let err = slices
            .iter()
            .map(|slice| mean.iter().zip(slice).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max))
            .fold(f64::INFINITY, f64::min);
        assert!(
            err <= atol + 0.15,
            "selected class atom {mean:?} vs slices {slices:?} err={err}"
        );
        checked += 1;
    }
    assert!(checked >= 1, "selected pair missing per-horizon atom surfaces");
}

#[test]
fn frequentist_incomplete_class_uses_outer_block_or_withholds() {
    let pin = pin();
    let (t, latent, selected, _, _, _) = generate(&pin);
    let series = selected_series(&t, &latent, &selected);
    let query = curve_query(
        &pin,
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(2),
        },
        ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)])),
    );
    let incomplete = run_pair(
        series.clone(),
        incomplete_cpdag(),
        query.clone(),
        InferenceMode::Frequentist,
        8,
    )
    .unwrap();
    assert_eq!(
        incomplete.certificate.as_ref().expect("certificate").graph_class,
        antecedent::GraphClass::TemporalCpdag
    );
    assert!(
        matches!(incomplete.response.as_ref().unwrap().uncertainty, ResponseUncertainty::None),
        "two-completion identified set withholds the class observation band"
    );
    assert!(incomplete.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.temporal_class.observation_class_band_withheld"
            || d.code.as_ref() == "estimate.temporal_class.observation_no_complete_band"
    }));
    let structural = incomplete.structural_response.as_ref().unwrap();
    assert!(structural.atoms.len() >= 2);
    assert!(
        structural.atoms.iter().any(|atom| {
            atom.response.as_ref().is_some_and(|response| {
                matches!(response.uncertainty, ResponseUncertainty::PointwiseBand { .. })
                    || response.support.diagnostics.iter().any(|d| {
                        d.id.as_ref() == "response.observation_block_bootstrap"
                    })
            })
        }),
        "requested replicates must produce an outer circular-block diagnostic on atoms"
    );

    let oriented = run_pair(series, oriented_cpdag(), query, InferenceMode::Frequentist, 8).unwrap();
    let response = oriented.response.as_ref().unwrap();
    assert!(
        matches!(response.uncertainty, ResponseUncertainty::PointwiseBand { .. })
            || response.support.diagnostics.iter().any(|d| {
                d.id.as_ref() == "response.observation_block_bootstrap"
            }),
        "a one-completion class must publish the outer circular-block band"
    );
    assert!(!matches!(
        response.uncertainty,
        ResponseUncertainty::PointwiseBand { .. }
            if response.provenance_id.as_ref() == "estimate.temporal_response.gcomp"
    ));
}

#[test]
fn sequence_overlay_rides_selected_observation_on_incomplete_class() {
    let pin = pin();
    let (t, latent, selected, _, _, _) = generate(&pin);
    let series = selected_series(&t, &latent, &selected);
    let seq = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(0),
    }]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::Sequence(seq)]),
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
    let result = run_pair(series, incomplete_cpdag(), query, InferenceMode::Frequentist, 0).unwrap();
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.temporal.sequence_overlay")
    );
    assert!(result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.temporal_class.observation_no_complete_band"
    }));
    assert_eq!(
        result.certificate.as_ref().expect("certificate").graph_class,
        antecedent::GraphClass::TemporalCpdag
    );
}
