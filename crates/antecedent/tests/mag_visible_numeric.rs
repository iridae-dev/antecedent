//! Visibility-certified MAG estimation against y = 1 + 2t.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::too_many_lines)]
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
#[allow(clippy::cast_sign_loss, reason = "the row index i runs over 0..800, so it is non-negative")]
fn visible_mag_scalar_and_response_cells_reuse_certified_envelopes() {
    // `r -> t`, a measured confounder `z -> t, z -> y`, and `y = 1 + 2t + 1.5z`. The
    // sample mean of `z` is exactly 0, so every certified functional equals the
    // structural value only when `z` is in the adjustment set; the crude slope of `y`
    // on `t` is far from 2, so an empty or partial set cannot pass.
    //
    // `z` and `r` cycle through 5 symmetric levels (mean exactly 0 for `z`, one full
    // period every 5 rows) rather than 2. The PAG's InterventionResponse cell runs the
    // visibility-aware general-ID path, which fits an additive-GAM plug-in target over
    // every adjustment covariate (not only a minimal backdoor set); a cubic smooth spline
    // needs more than two distinct knot locations, and a 2-valued `z`/`r` left that fit
    // singular ("additive GAM target did not converge"). 5 levels is enough for the GAM's
    // quantile knots to be non-degenerate while keeping the DGP exactly computable.
    let z: Vec<_> = (0..800).map(|i: i32| 0.5 * f64::from(i % 5 - 2)).collect();
    let r: Vec<_> = (0..800).map(|i| f64::from((i / 5) % 5)).collect();
    let t: Vec<_> = (0..800)
        .map(|i| {
            let jitter = f64::from((i * 7919) % 101) / 101.0 * 0.1 - 0.05;
            0.5 + 0.2 * z[i as usize] + 0.0375 * (r[i as usize] - 2.0) + jitter
        })
        .collect();
    let y: Vec<_> = t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + 1.5 * z).collect();
    let v: Vec<_> = (0..800).map(|i| f64::from(i).sin()).collect();
    let w: Vec<_> = (0..800).map(|i| f64::from(i).cos()).collect();
    let crude_slope = {
        let n = t.len() as f64;
        let (mt, my) = (t.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let cov: f64 = t.iter().zip(&y).map(|(t, y)| (t - mt) * (y - my)).sum();
        let var: f64 = t.iter().map(|t| (t - mt).powi(2)).sum();
        cov / var
    };
    assert!(
        crude_slope > 2.5,
        "the confounder must bias the crude slope by more than 0.5: {crude_slope}"
    );
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("r", r.as_slice()),
        ("z", z.as_slice()),
        ("v", v.as_slice()),
        ("w", w.as_slice()),
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
        let mut pag = Pag::with_variables(6);
        pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        if mixed {
            pag.insert_circle_circle(DenseNodeId::from_raw(4), DenseNodeId::from_raw(5)).unwrap();
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
                    for result in vec![fresh, click] {
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
                                (value - target).abs() < 0.02,
                                "{query:?}: {value} vs {target}"
                            );
                        }
                    }
                }
            }
        }
    }
}
