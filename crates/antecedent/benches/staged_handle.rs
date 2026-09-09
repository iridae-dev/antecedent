#![allow(clippy::cast_precision_loss)]
//! Prepared derivative and counterfactual execution costs (1.3).
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(missing_docs)]
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, DerivativeScale, ExecutionContext, Intervention,
    ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
fn staged_handle(c: &mut Criterion) {
    let a: Vec<_> = (0..400).map(|i| (f64::from(i) * 0.71).sin()).collect();
    let y: Vec<_> =
        a.iter().enumerate().map(|(i, a)| 2.0 * a + 0.1 * (i as f64 * 0.13).cos()).collect();
    let data = TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let ctx = ExecutionContext::for_tests(13);
    for (name, query) in [
        (
            "prepared_point_derivative_n400",
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                at: 0.0,
                order: 1,
                scale: DerivativeScale::Identity,
            })),
        ),
        (
            "prepared_counterfactual_n400",
            CausalQuery::Counterfactual(CounterfactualQuery::new(
                VariableId::from_raw(1),
                Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.8))]),
            )),
        ),
    ] {
        let study = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .response_options(antecedent_estimate::ContinuousResponseOptions {
                bandwidth: Some(0.35),
                ..Default::default()
            })
            .build()
            .unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        c.bench_function(name, |b| b.iter(|| black_box(prepared.estimate(&data, &ctx).unwrap())));
    }
}
criterion_group!(benches, staged_handle);
criterion_main!(benches);
