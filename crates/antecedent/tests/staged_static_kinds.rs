//! Frozen structural truth through the licensed staged handle.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
use antecedent::{AcceptedGraph, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, MediationContrast,
    MediationQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use std::sync::Arc;

#[test]
fn static_kinds_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_static_kinds/expected.json"
    ))
    .unwrap();
    let a: Vec<_> = (0..500).map(|i| (f64::from(i) * 0.71).sin()).collect();
    let m: Vec<_> = a.iter().enumerate().map(|(i, a)| 2.0 * a + (i as f64 * 1.13).cos()).collect();
    let y: Vec<_> = a
        .iter()
        .zip(&m)
        .enumerate()
        .map(|(i, (a, m))| 3.0 * a + 4.0 * m + 0.1 * (i as f64 * 0.31).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    for (s, t) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    let control = pin["control"].as_f64().unwrap();
    let active = pin["active"].as_f64().unwrap();
    let ctx = ExecutionContext::for_tests(13);
    for (key, contrast) in [
        ("direct", MediationContrast::NaturalDirect),
        ("indirect", MediationContrast::NaturalIndirect),
        ("total", MediationContrast::Total),
    ] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let mut query = MediationQuery::binary(
                    VariableId::from_raw(0),
                    VariableId::from_raw(2),
                    Arc::from([VariableId::from_raw(1)]),
                    contrast,
                );
                query.control = Intervention::set(query.treatment, Value::f64(control));
                query.active = Intervention::set(query.treatment, Value::f64(active));
                let builder = if accepted {
                    Study::tabular(data.clone()).graph(AcceptedGraph::from(dag.clone()))
                } else {
                    Study::tabular(data.clone()).graph(dag.clone())
                };
                let study = builder
                    .query(CausalQuery::Mediation(query))
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                let prepared = study.prepare(&ctx).unwrap();
                let result = prepared.estimate(&data, &ctx).unwrap();
                assert!(
                    (result.estimate.ate - pin[key].as_f64().unwrap()).abs()
                        < pin["tolerance"].as_f64().unwrap(),
                    "{key}: {:?}",
                    result.estimate
                );
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );
                assert_eq!(
                    result.refutations.len(),
                    match suite {
                        RefuteSuite::None => 0,
                        RefuteSuite::Cheap => 2,
                        _ => 3,
                    }
                );
            }
        }
    }
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(active))]),
    );
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Counterfactual(query))
        .counterfactual_control(control)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let cf = result.counterfactual.as_ref().unwrap();
    assert!((cf.mean_ite - pin["counterfactual_mean"].as_f64().unwrap()).abs() < 0.03);
    assert_eq!(cf.unit_effects.len(), 500);
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
}

#[test]
fn nested_counterfactual_refuses_before_execution() {
    let a = [0.0, 1.0, 2.0];
    let data = TabularData::from_f64_columns([("a", a.as_slice()), ("y", a.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let mut query = CounterfactualQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    query.allow_nested = true;
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Counterfactual(query))
        .refute(RefuteSuite::None)
        .build()
        .and_then(|study| study.prepare(&ExecutionContext::for_tests(1)));
    assert!(result.is_err());
}
