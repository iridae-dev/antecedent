//! Visibility-certified MAG estimation against y = 1 + 2t.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::many_single_char_names)]
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, ExecutionContext,
    GridSpec, Intervention, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Pag};
use std::sync::Arc;

#[test]
fn visible_mag_scalar_and_response_cells_reuse_certified_envelopes() {
    let t: Vec<_> = (0..800).map(|i| 0.05 + 0.9 * f64::from(i) / 799.0).collect();
    let y: Vec<_> = t.iter().map(|t| 1.0 + 2.0 * t).collect();
    let r: Vec<_> = (0..800).map(|i| f64::from(i % 2)).collect();
    let s: Vec<_> = (0..800).map(|i| f64::from(i).sin()).collect();
    let v: Vec<_> = (0..800).map(|i| f64::from(i).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("r", r.as_slice()),
        ("s", s.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let average = AverageEffectQuery::binary_ate(t, y);
    let queries: Vec<(CausalQuery, Vec<f64>)> = vec![
        (CausalQuery::AverageEffect(average.clone()), vec![2.0]),
        (
            CausalQuery::ConditionalEffect(
                ConditionalEffectQuery::try_new(
                    average.with_effect_modifiers([VariableId::from_raw(2)]),
                )
                .unwrap(),
            ),
            vec![2.0],
        ),
        (
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([Intervention::set(t, Value::f64(0.5))]),
            })),
            vec![2.0],
        ),
        (
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: y,
                treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from([0.25, 0.75]))),
            })),
            vec![1.5, 2.5],
        ),
    ];
    for mixed in [false, true] {
        let mut pag = Pag::with_variables(5);
        pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        if mixed {
            pag.insert_circle_circle(DenseNodeId::from_raw(3), DenseNodeId::from_raw(4)).unwrap();
        }
        for accepted in [false, true] {
            for bayesian in [false, true] {
                for (query, expected) in &queries {
                    let builder = Study::tabular(data.clone());
                    let builder = if accepted {
                        builder.graph(AcceptedGraph::from(pag.clone()))
                    } else {
                        builder.graph(pag.clone())
                    };
                    let inference = if bayesian {
                        InferenceMode::Bayesian(BayesianConfig::conjugate().prior_scale(100.0))
                    } else {
                        InferenceMode::Frequentist
                    };
                    let study = builder
                        .query(query.clone())
                        .inference(inference)
                        .refute(RefuteSuite::None)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap();
                    let ctx = ExecutionContext::for_tests(7);
                    let fresh = study.clone().run(&ctx).unwrap();
                    let prepared = study.prepare(&ctx).unwrap();
                    let click = prepared.estimate(&data, &ctx).unwrap();
                    for result in [fresh, click] {
                        let values = if let Some(response) = result.response {
                            match response.estimate {
                                ResponseIdentification::PointIdentified(value)
                                | ResponseIdentification::PartiallyIdentified(value) => match value
                                {
                                    ResponseValue::Scalar(x) => vec![x],
                                    ResponseValue::Surface { mean, .. } => mean.to_vec(),
                                    other => panic!("unexpected {other:?}"),
                                },
                                other => panic!("unexpected {other:?}"),
                            }
                        } else {
                            vec![result.estimate.ate]
                        };
                        assert_eq!(values.len(), expected.len());
                        for (value, target) in values.iter().zip(expected) {
                            assert!(
                                (value - target).abs() < 0.05,
                                "{query:?}: {value} vs {target}"
                            );
                        }
                    }
                }
            }
        }
    }
}
