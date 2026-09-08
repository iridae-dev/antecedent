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
