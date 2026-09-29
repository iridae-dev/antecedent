//! Builder-independent checked GCM counterfactual lifecycle.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn checked_counterfactual_accepted_and_explicit_retain_worlds_and_refuse_portable_replay() {
    let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 20.0 - 2.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + 0.01 * (i as f64).sin()).collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = CounterfactualQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    let ctx = ExecutionContext::for_tests(59);
    for accepted in [false, true] {
        let builder = if accepted {
            Study::tabular(data.clone()).graph(AcceptedGraph::from(graph.clone()))
        } else {
            Study::tabular(data.clone()).graph(graph.clone())
        }
        .query(CausalQuery::Counterfactual(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);
        let plan = prepared.checked_counterfactual_operation().expect("retained checked plan");
        assert_eq!(plan.query(), &query);
        assert_eq!(plan.graph().node_count(), graph.node_count());
        let result = prepared.estimate(&data, &ctx).unwrap();
        let mean = result.counterfactual.as_ref().unwrap().mean_ite;
        assert!((mean - 2.0).abs() < 0.1, "mean_ite={mean}");
        let refreshed_y: Vec<f64> =
            x.iter().enumerate().map(|(i, x)| 1.0 + 2.5 * x + 0.01 * (i as f64).sin()).collect();
        let refreshed_data =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", refreshed_y.as_slice())])
                .unwrap();
        let refreshed = prepared.refresh(refreshed_data, &ctx).unwrap();
        let refreshed_mean = refreshed.counterfactual.as_ref().unwrap().mean_ite;
        assert!((refreshed_mean - 2.5).abs() < 0.1, "mean_ite={refreshed_mean}");
        let bytes =
            prepared.encode_contracted_result(&refreshed, "checked-counterfactual", &ctx).unwrap();
        let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
        assert!(
            !consumed.acceptance.accepts_as_verified_program(),
            "acceptance={:?}; estimator={:?}",
            consumed.acceptance,
            consumed.contract.as_ref().map(|contract| (
                contract.estimator.as_deref(),
                contract
                    .program
                    .as_ref()
                    .and_then(|program| program.commitments.resolved_estimator.as_deref())
            ))
        );
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|reason| reason.as_ref() == "dependencies.fitted_counterfactual_mechanisms")
        );
    }
}
