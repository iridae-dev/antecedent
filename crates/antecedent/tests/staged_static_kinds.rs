//! Frozen structural truth through the licensed staged handle.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::too_many_lines, clippy::cast_precision_loss, clippy::similar_names)]
use antecedent::estimate::validate_static_pair;
use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, RefuteSuite, Study};
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
    )
    .with_control_level(control);
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Counterfactual(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let cf = result.counterfactual.as_ref().unwrap();
    assert!((cf.mean_ite - pin["counterfactual_mean"].as_f64().unwrap()).abs() < 0.03);
    assert_eq!(cf.unit_effects.len(), 500);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit"));
    assert!(result.physical_plan.kernels.iter().any(|(name, _)| name.as_ref() == "gcm.aap"));
    assert!(validate_static_pair(IdentifierId::BackdoorAdjustment, EstimatorId::GcmFit).is_err());
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
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

fn naive_parent_contrasts(a: &[f64], m: &[f64], y: &[f64], delta: f64) -> (f64, f64) {
    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [0.0; 3];
    for i in 0..a.len() {
        let row = [1.0, a[i], m[i]];
        for r in 0..3 {
            xty[r] += row[r] * y[i];
            for c in 0..3 {
                xtx[r][c] += row[r] * row[c];
            }
        }
    }
    let mut aug = [
        [xtx[0][0], xtx[0][1], xtx[0][2], xty[0]],
        [xtx[1][0], xtx[1][1], xtx[1][2], xty[1]],
        [xtx[2][0], xtx[2][1], xtx[2][2], xty[2]],
    ];
    for col in 0..3 {
        let mut pivot = col;
        for r in (col + 1)..3 {
            if aug[r][col].abs() > aug[pivot][col].abs() {
                pivot = r;
            }
        }
        aug.swap(col, pivot);
        let diag = aug[col][col];
        for c in col..4 {
            aug[col][c] /= diag;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = aug[r][col];
            for c in col..4 {
                aug[r][c] -= f * aug[col][c];
            }
        }
    }
    (aug[1][3] * delta, aug[2][3] * 2.0 * delta)
}

#[test]
fn confounded_static_kinds_are_not_the_observational_association() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_static_kinds/confounded.json"
    ))
    .unwrap();
    let z: Vec<_> = (0..500).map(|i| (f64::from(i) * 0.41).cos()).collect();
    let a: Vec<_> =
        (0..500).map(|i| (f64::from(i) * 0.71).sin() + 0.4 * (f64::from(i) * 0.41).cos()).collect();
    let m: Vec<_> = a
        .iter()
        .zip(&z)
        .enumerate()
        .map(|(i, (a, z))| 2.0 * a + 0.5 * z + (i as f64 * 1.13).cos())
        .collect();
    let y: Vec<_> = a
        .iter()
        .zip(&m)
        .zip(&z)
        .enumerate()
        .map(|(i, ((a, m), z))| 3.0 * a + 4.0 * m + 5.0 * z + 0.1 * (i as f64 * 0.31).sin())
        .collect();
    let control = pin["control"].as_f64().unwrap();
    let active = pin["active"].as_f64().unwrap();
    let delta = active - control;
    let (naive_nde, naive_nie) = naive_parent_contrasts(&a, &m, &y, delta);
    let naive_ite = naive_ols_slope(&a, &y) * delta;
    assert!(
        (naive_nde - pin["direct"].as_f64().unwrap()).abs() > pin["tolerance"].as_f64().unwrap()
    );
    assert!(
        (naive_nie - pin["indirect"].as_f64().unwrap()).abs() > pin["tolerance"].as_f64().unwrap()
    );
    assert!(
        (naive_ite - pin["counterfactual_mean"].as_f64().unwrap()).abs()
            > pin["tolerance"].as_f64().unwrap()
    );
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(4);
    for (s, t) in [(0, 1), (0, 2), (1, 2), (3, 0), (3, 1), (3, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    let ctx = ExecutionContext::for_tests(13);
    for (key, contrast) in [
        ("direct", MediationContrast::NaturalDirect),
        ("indirect", MediationContrast::NaturalIndirect),
        ("total", MediationContrast::Total),
    ] {
        let mut query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            Arc::from([VariableId::from_raw(1)]),
            contrast,
        );
        query.control = Intervention::set(query.treatment, Value::f64(control));
        query.active = Intervention::set(query.treatment, Value::f64(active));
        let result = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(CausalQuery::Mediation(query))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap();
        assert!(
            (result.estimate.ate - pin[key].as_f64().unwrap()).abs()
                < pin["tolerance"].as_f64().unwrap(),
            "{key}: {:?}",
            result.estimate
        );
    }
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(active))]),
    )
    .with_control_level(control);
    let result = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Counterfactual(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let cf = result.counterfactual.as_ref().unwrap();
    assert!((cf.mean_ite - pin["counterfactual_mean"].as_f64().unwrap()).abs() < 0.03);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit"));
    assert!((cf.mean_ite - naive_ite).abs() > pin["tolerance"].as_f64().unwrap());
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
