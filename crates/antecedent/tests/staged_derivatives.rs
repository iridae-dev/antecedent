//! Known-truth licensing evidence for staged derivatives.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names, clippy::too_many_lines, clippy::cast_precision_loss)]
use antecedent::{AcceptedGraph, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, ResponseFunctional as F,
    ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId};
use std::sync::Arc;

#[test]
fn staged_derivatives_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/staged_derivatives/expected.json"
    ))
    .unwrap();
    let a: Vec<_> = (0..800)
        .map(|i| 2.0 + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos())
        .collect();
    let b: Vec<_> = (0..800).map(|i| (f64::from(i) * 1.13).cos()).collect();
    let y: Vec<_> = a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b).collect();
    let v: Vec<_> = a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 1.5 * b).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (s, t) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    let ids =
        |xs: &[u32]| -> Arc<[VariableId]> { xs.iter().map(|&x| VariableId::from_raw(x)).collect() };
    let point = |scale| F::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at: 2.0,
        order: 1,
        scale,
    };
    let cases = vec![
        ("point", point(DerivativeScale::Identity)),
        ("elasticity", point(DerivativeScale::LogLog)),
        ("semi_treatment", point(DerivativeScale::LogTreatment)),
        ("semi_outcome", point(DerivativeScale::LogOutcome)),
        (
            "average",
            F::AverageDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                weighting: DerivativeWeighting::Observed,
            },
        ),
        (
            "jacobian",
            F::Jacobian {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                scale: DerivativeScale::Identity,
            },
        ),
        (
            "directional",
            F::DirectionalDerivative {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                direction: Arc::from([1.0, 2.0]),
            },
        ),
    ];
    for (key, functional) in cases {
        for accepted in [false, true] {
            let builder = if accepted {
                Study::tabular(data.clone()).graph(AcceptedGraph::from(graph.clone()))
            } else {
                Study::tabular(data.clone()).graph(graph.clone())
            };
            let study = builder
                .query(CausalQuery::Response(ResponseQuery::new(functional.clone())))
                .response_options(ContinuousResponseOptions {
                    bandwidth: Some(0.35),
                    ..Default::default()
                })
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let ctx = ExecutionContext::for_tests(13);
            let prepared = study.prepare(&ctx).unwrap();
            let result = prepared.estimate(&data, &ctx).unwrap();
            let fresh = study.run(&ctx).unwrap();
            let response = result.response.as_ref().unwrap();
            assert_eq!(response, fresh.response.as_ref().unwrap());
            let ResponseIdentification::PointIdentified(value) = &response.estimate else {
                panic!("unidentified {key}")
            };
            let got = match value {
                ResponseValue::Scalar(x) => vec![*x],
                ResponseValue::Jacobian { values, .. } => values.to_vec(),
                ResponseValue::Vector(x) => x.to_vec(),
                _ => panic!("unexpected {value:?}"),
            };
            let expected = if let Some(x) = pin[key].as_f64() {
                vec![x]
            } else {
                pin[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
            };
            for (got, want) in got.iter().zip(expected) {
                assert!(
                    (got - want).abs() < pin["tolerance"].as_f64().unwrap(),
                    "{key}: {got} != {want}"
                );
            }
        }
    }
}

fn naive_ols_slope(a: &[f64], y: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (ai, yi) in a.iter().zip(y) {
        let da = ai - mean_a;
        num += da * (yi - mean_y);
        den += da * da;
    }
    num / den
}

#[test]
fn confounded_derivatives_are_not_the_observational_slope() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/staged_derivatives/confounded.json"
    ))
    .unwrap();
    let z: Vec<_> = (0..800).map(|i| (f64::from(i) * 0.41).cos()).collect();
    let a: Vec<_> = (0..800)
        .map(|i| {
            let z = (f64::from(i) * 0.41).cos();
            2.0 + 0.6 * z + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos()
        })
        .collect();
    let y: Vec<_> = a.iter().zip(&z).map(|(a, z)| 5.0 + 2.0 * a + 3.0 * z).collect();
    let naive = naive_ols_slope(&a, &y);
    let structural = pin["point"].as_f64().unwrap();
    assert!((naive - structural).abs() > pin["naive_gap_min"].as_f64().unwrap());
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("z", z.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (s, t) in [(1, 0), (1, 2), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    let ctx = ExecutionContext::for_tests(13);
    let cases = [
        (
            "point",
            F::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 2.0,
                order: 1,
                scale: DerivativeScale::Identity,
            },
            Some(0.35),
        ),
        (
            "average",
            F::AverageDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                weighting: DerivativeWeighting::Observed,
            },
            None,
        ),
    ];
    for (key, functional, bandwidth) in cases {
        let study = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(ResponseQuery::new(functional)))
            .response_options(ContinuousResponseOptions { bandwidth, ..Default::default() })
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let result = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(got)) =
            &result.response.as_ref().unwrap().estimate
        else {
            panic!("unidentified {key}")
        };
        let want = pin[key].as_f64().unwrap();
        assert!((got - want).abs() < pin["tolerance"].as_f64().unwrap(), "{key}: {got} != {want}");
        assert!(
            (got - structural).abs() * 4.0 < (naive - structural).abs(),
            "{key} {got} is not closer to structural than naive {naive}"
        );
    }
}
