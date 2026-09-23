//! Numerical evidence for temporal latent-confounded MAG adjustment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::too_many_lines)]
use antecedent::{AcceptedGraph, RefuteSuite, Study};
use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{Endpoint, MarkedEdge, MiddleMark, TemporalPag};
use std::{collections::HashMap, sync::Arc};

fn fixture() -> (TimeSeriesData, TemporalPag, serde_json::Value) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_envelope/latent.json"
    ))
    .unwrap();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let names: Vec<_> =
        pin["columns"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let mut values = vec![vec![0.0; n]; 4];
    for i in 0..n {
        values[2][i] = (i % 2) as f64;
        values[0][i] = 0.3 + 0.4 * values[2][i] + 0.05 * (i as f64 * 0.017).sin();
        values[3][i] = ((i / 2) % 2) as f64;
        if i > 0 {
            values[1][i] = 1.0 + 2.0 * values[0][i - 1] + 0.5 * values[2][i - 1];
        }
    }
    let mut schema = CausalSchemaBuilder::new();
    for name in &names {
        schema
            .add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let columns = values
        .into_iter()
        .enumerate()
        .map(|(i, col)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(col),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(schema.build().unwrap(), columns, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalPag::empty();
    let mut nodes = HashMap::new();
    for edge in pin["marked_edges"].as_array().unwrap() {
        let mut endpoints = Vec::new();
        for start in [0, 2] {
            let variable = u32::try_from(
                names.iter().position(|name| *name == edge[start].as_str().unwrap()).unwrap(),
            )
            .unwrap();
            let lag = u32::try_from(edge[start + 1].as_u64().unwrap()).unwrap();
            endpoints.push(*nodes.entry((variable, lag)).or_insert_with(|| {
                graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
            }));
        }
        let mark = |s: &str| if s == "arrow" { Endpoint::Arrow } else { Endpoint::Tail };
        graph
            .insert_marked(MarkedEdge {
                a: endpoints[0],
                b: endpoints[1],
                at_a: mark(edge[4].as_str().unwrap()),
                at_b: mark(edge[5].as_str().unwrap()),
                middle: MiddleMark::Empty,
            })
            .unwrap();
    }
    (data, graph, pin)
}

#[test]
fn latent_temporal_effect_preserves_query_and_prepared_estimation() {
    let (data, graph, pin) = fixture();
    for accepted in [false, true] {
        for sustained in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let mut query = TemporalEffectQuery::pulse(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                    1.0,
                );
                query.policy = if sustained {
                    TemporalPolicy::sustained(-1, -1)
                } else {
                    TemporalPolicy::pulse(-1)
                };
                let builder = Study::series(data.clone());
                let builder = if accepted {
                    builder.graph(AcceptedGraph::from(graph.clone()))
                } else {
                    builder.graph(graph.clone())
                };
                let study =
                    builder.query(query).refute(suite).bootstrap_replicates(0).build().unwrap();
                let ctx = ExecutionContext::for_tests(7);
                let fresh = study.clone().run(&ctx).unwrap_or_else(|err| {
                    panic!("accepted={accepted}, sustained={sustained}, suite={suite:?}: {err:?}")
                });
                let prepared = study.prepare(&ctx).unwrap();
                let click = prepared.estimate_series(&data, &ctx).unwrap();
                for result in vec![fresh, click] {
                    let certificate = result.certificate.as_ref().expect("execution certificate");
                    let antecedent::Identification::TemporalEnvelope { envelope, .. } =
                        &certificate.identification
                    else {
                        panic!("class envelope lost")
                    };
                    assert_eq!(envelope.envelope.cases.len(), 1);
                    let z = envelope.envelope.cases[0].result.estimands[0].adjustment_set[0];
                    assert_eq!(envelope.indexers[0].key_of(z.raw()).unwrap().offset, -1);

                    assert!(
                        (result.estimate.ate - pin["effect"].as_f64().unwrap()).abs()
                            < pin["absolute_tolerance"].as_f64().unwrap()
                    );
                    if suite != RefuteSuite::None {
                        let evalue = result
                            .refutations
                            .iter()
                            .find(|report| report.refuter.as_ref() == "sensitivity.evalue")
                            .expect("E-value ran on aligned rows");
                        // The E-value's risk-ratio conversion uses the *residual* SD of the
                        // outcome regression on treatment and the adjustment set (see
                        // `EValue`'s doc comment), not the marginal SD of Y. This fixture's law
                        // (`y_i = 1 + 2*t_(i-1) + 0.5*z_(i-1)`) has no noise term, and the
                        // correctly identified adjustment set is exactly `{z[-1]}` (asserted
                        // above), so Y is an exact linear function of the regression's own
                        // covariates: the residual SD is 0 and the E-value is infinite.
                        // Previously pinned to a finite value (~31.81) computed from the
                        // *marginal* SD of Y instead, which was never a value this refuter
                        // could produce once identification carried the correct adjustment set.
                        assert!(evalue.comparison.is_infinite());
                    }
                }
            }
        }
    }
}
