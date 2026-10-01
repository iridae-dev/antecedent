//! A plug-in derivative over more than two treatments is refused through the
//! prepared workflow with its message unchanged, no reason code, and a
//! structured remedy read off the facade error (the field Python exposes as
//! `remedy`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, DerivativeScale, ExecutionContext, ResponseFunctional, ResponseQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn three_treatment_derivatives_are_refused_with_a_remedy() {
    let col = |k: f64| (0..200).map(|i| (f64::from(i) * k).sin()).collect::<Vec<f64>>();
    let (a, b, c) = (col(0.31), col(0.73), col(1.17));
    let y: Vec<f64> = (0..200).map(|i| a[i] - 0.5 * b[i] + 0.25 * c[i]).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for from in 0..3 {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(3)).unwrap();
    }
    let treatments: Arc<[VariableId]> = (0..3).map(VariableId::from_raw).collect();
    let outcomes: Arc<[VariableId]> = Arc::from([VariableId::from_raw(3)]);
    let ctx = ExecutionContext::for_tests(7);
    let cases = [
        (
            ResponseFunctional::Jacobian {
                outcomes: outcomes.clone(),
                treatments: treatments.clone(),
                at: Arc::from([0.0, 0.0, 0.0]),
                scale: DerivativeScale::Identity,
            },
            "plug-in response Jacobian supports at most two treatments",
            "AverageDerivative",
        ),
        (
            ResponseFunctional::DirectionalDerivative {
                outcomes,
                treatments,
                at: Arc::from([0.0, 0.0, 0.0]),
                direction: Arc::from([1.0, 0.0, 0.0]),
            },
            "plug-in directional derivative supports at most two treatments",
            "ResponseJacobian",
        ),
    ];
    for (functional, message, names) in cases {
        let error = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(ResponseQuery::new(functional)))
            .refute(RefuteSuite::None)
            .build()
            .and_then(|study| study.prepare(&ctx))
            .and_then(|prepared| prepared.estimate(&data, &ctx))
            .expect_err("more than two treatments is refused");
        assert_eq!(error.to_string(), message);
        assert_eq!(error.reason_code(), None, "no reason code is added or changed");
        let remedy = error.remedy().expect("the refusal names a remedy");
        assert!(remedy.contains(names), "{remedy}");
        assert!(!error.to_string().contains(remedy), "the remedy is not in the message");
    }
}
