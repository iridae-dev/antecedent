//! 1.7 Sequence / Soft pins on two-completion temporal classes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionSequence, Lag, MechanismOverride,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, SequencedIntervention,
    TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalPag};

fn dose_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_dose_horizon/expected.json"
    ))
    .unwrap()
}

fn soft_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_soft_mean_mechanisms/expected.json"
    ))
    .unwrap()
}

fn dose_series() -> TimeSeriesData {
    let pin = dose_pin();
    let n = usize::try_from(pin["generation"]["n"].as_u64().unwrap()).unwrap();
    let t: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            3 => -1.0,
            _ => unreachable!(),
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect();
    let z: Vec<f64> = (0..n).map(|i| ((i % 3) as f64) - 1.0).collect();
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn dose_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

fn dose_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g
}

fn soft_series() -> TimeSeriesData {
    let pin = soft_pin();
    let n = usize::try_from(pin["generation"]["n"].as_u64().unwrap()).unwrap();
    let t = (0..n).map(|i| 1.0 + (i as f64 * 1.719).sin()).collect::<Vec<_>>();
    let z = (0..n).map(|i| 2.0 + (i as f64 * 0.813).cos()).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * t[i.saturating_sub(1)]
                + 3.0 * t[i.saturating_sub(2)]
                + 4.0 * z[i.saturating_sub(1)]
                + 0.05 * (i as f64 * 0.419).sin()
        })
        .collect::<Vec<_>>();
    let w: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) / 2.0 - 1.0).collect();
    TimeSeriesData::from_f64_columns(
        [
            ("t", t.as_slice()),
            ("z", z.as_slice()),
            ("y", y.as_slice()),
            ("w", w.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn soft_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let w1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_undirected(w1, t1).unwrap();
    g
}

fn soft_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g
}

fn two_step_sequence() -> CausalQuery {
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::Sequence(seq)]),
        })
        .with_temporal(TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap()),
    )
}

fn joint_series() -> TimeSeriesData {
    let n = 200usize;
    let a: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) / 4.0).collect();
    let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) / 6.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| a[j])
                + 4.0 * i.checked_sub(1).map_or(0.0, |j| b[j])
        })
        .collect();
    let w: Vec<f64> = (0..n).map(|i| ((i % 4) as f64) - 1.5).collect();
    TimeSeriesData::from_f64_columns(
        [
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y", y.as_slice()),
            ("w", w.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn joint_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let a1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b1 = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let w1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    g.insert_directed(a1, y0).unwrap();
    g.insert_directed(b1, y0).unwrap();
    g.insert_undirected(w1, a1).unwrap();
    g
}

fn joint_sequence_query() -> CausalQuery {
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(2),
            interventions: Arc::from([Intervention::Sequence(seq)]),
        })
        .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap()),
    )
}

fn inferences() -> [InferenceMode; 2] {
    [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64).prior_scale(1000.0)),
    ]
}

fn class_surface(result: &antecedent::StudyResult) -> Vec<f64> {
    let response = result.response.as_ref().expect("class sequence response");
    match &response.estimate {
        ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(env)) => {
            env.lower.to_vec()
        }
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => mean.to_vec(),
        other => panic!("unexpected class sequence estimate: {other:?}"),
    }
}

fn run_class(
    data: TimeSeriesData,
    graph: impl antecedent::IntoGraphInput,
    query: CausalQuery,
    inference: InferenceMode,
) -> antecedent::StudyResult {
    Study::series(data)
        .graph(graph)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap()
}

#[test]
fn two_completion_dose_horizon_sequence_matches_fixture() {
    let pin = dose_pin();
    let expected: Vec<f64> = pin["contract"]["intervention_paths"]["sequence_two_step_set_1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let freq_atol = pin["tolerance"]["atol"].as_f64().unwrap();
    let data = dose_series();
    for inference in inferences() {
        let atol = if matches!(inference, InferenceMode::Frequentist) {
            freq_atol
        } else {
            0.12
        };
        let cpdag = run_class(data.clone(), dose_cpdag(), two_step_sequence(), inference.clone());
        assert_eq!(
            cpdag.certificate.as_ref().expect("certificate").graph_class,
            antecedent::GraphClass::TemporalCpdag
        );
        let pag = run_class(data.clone(), dose_pag(), two_step_sequence(), inference.clone());
        assert_eq!(
            pag.certificate.as_ref().expect("certificate").graph_class,
            antecedent::GraphClass::TemporalPag
        );
        for result in [cpdag, pag] {
            let got = class_surface(&result);
            assert_eq!(got.len(), expected.len());
            for (actual, truth) in got.iter().zip(&expected) {
                assert!((actual - truth).abs() < atol, "{got:?} vs {expected:?}");
            }
            assert!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code.as_ref() == "estimate.temporal.sequence_overlay")
            );
        }
    }
}

#[test]
fn two_completion_joint_sequence_matches_structural_level() {
    let data = joint_series();
    for inference in inferences() {
        let result = run_class(data.clone(), joint_cpdag(), joint_sequence_query(), inference.clone());
        let got = class_surface(&result);
        let atol = if matches!(inference, InferenceMode::Frequentist) {
            1e-10
        } else {
            0.12
        };
        assert!((got[0] - 7.0).abs() < atol, "{got:?}");
        assert_eq!(
            result.certificate.as_ref().expect("certificate").graph_class,
            antecedent::GraphClass::TemporalCpdag
        );
    }
}

#[test]
fn two_completion_soft_mean_mechanisms_match_fixture() {
    let pin = soft_pin();
    let atol = pin["atol"].as_f64().unwrap();
    let cases = [
        Intervention::soft(VariableId::from_raw(0), MechanismOverride::multiplicative(2.0)),
        Intervention::soft(
            VariableId::from_raw(0),
            MechanismOverride::truncated_shift(3.0, 0.0, 2.0),
        ),
    ];
    let data = soft_series();
    for (index, intervention) in cases.into_iter().enumerate() {
        let expected = pin["cases"][index]["mean"].as_f64().unwrap();
        let query = CausalQuery::Response(
            ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(2),
                interventions: Arc::from([intervention]),
            })
            .with_temporal(
                TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap(),
            ),
        );
        for inference in inferences() {
            let cpdag = run_class(data.clone(), soft_cpdag(), query.clone(), inference.clone());
            let pag = run_class(data.clone(), soft_pag(), query.clone(), inference);
            for result in [cpdag, pag] {
                let got = class_surface(&result);
                assert!(
                    (got[0] - expected).abs() < atol,
                    "case {index}: {} vs {expected}",
                    got[0]
                );
            }
        }
    }
}

#[test]
fn bidirected_completion_misses_sequence_coordinate_and_keeps_the_class() {
    let mut pag = TemporalPag::empty();
    let t1 = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_circle_arrow(t1, y0).unwrap();
    let seq = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(-1),
    }]);
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::Sequence(seq)]),
        })
        .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap()),
    );
    let n = 400;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut u = vec![0.0; n];
    for i in 0..n {
        u[i] = ((i as f64) * 0.29).sin();
        t[i] = 0.3 + 0.2 * u[i];
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1];
        }
    }
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("u", u.as_slice())],
        1,
    )
    .unwrap();
    let result = Study::series(data)
        .graph(pag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22));
    match result {
        Ok(result) => {
            let structural = result.structural_response.as_ref().expect("class sequence");
            assert!(
                structural.unidentified_mass > 0.0 || structural.unevaluable_mass > 0.0,
                "the latent completion must stay on the class"
            );
            assert_eq!(
                result.certificate.as_ref().expect("certificate").graph_class,
                antecedent::GraphClass::TemporalPag
            );
        }
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("unevaluable")
                    || message.contains("unidentified")
                    || message.contains("no evaluable")
                    || message.contains("not identified"),
                "{message}"
            );
        }
    }
}

#[test]
fn nested_sequence_stays_refused_on_temporal_cpdag() {
    let inner = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(0),
    }]);
    let outer = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::Sequence(inner),
        temporal: TemporalPolicy::pulse(0),
    }]);
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::Sequence(outer)]),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    );
    let err = Study::series(dose_series())
        .graph(dose_cpdag())
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Sequence") && msg.contains("nested") && msg.contains("not licensed"),
        "{msg}"
    );
}
