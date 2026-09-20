//! The practitioner suite depends only on `antecedent`, not its component crates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::discovery::{GraphPosterior, GraphPosteriorAtomKind, InferenceDiagnostics};
use antecedent::estimate::ContinuousResponseOptions;
use antecedent::graph::{Lag, ensure_lagged};
use antecedent::prelude::*;
use antecedent::query::{
    ContinuousDomain, DerivativeScale, DerivativeWeighting, GridSpec, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec,
};

#[test]
fn response_and_temporal_queries_are_constructible_through_facade() {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let values: Vec<f64> = (0..100).map(|i| 0.2 + f64::from(i) / 80.0).collect();
    let outcomes: Vec<f64> = values.iter().map(|x| 3.0 + 2.0 * x).collect();
    let data =
        TabularData::from_f64_columns([("t", values.as_slice()), ("y", outcomes.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from([0.5, 1.0]))),
    });
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.3), ..Default::default() })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(731))
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response.unwrap().estimate
    else {
        panic!("expected identified curve");
    };
    assert!((mean[1] - mean[0] - 1.0).abs() < 1e-8);

    let mut temporal = TemporalDag::empty();
    let source = ensure_lagged(&mut temporal, t, Lag::from_raw(1)).unwrap();
    let target = ensure_lagged(&mut temporal, y, Lag::CONTEMPORANEOUS).unwrap();
    temporal.insert_directed(source, target).unwrap();
    let spec = TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap();
    assert_eq!(spec.treatment_offset().unwrap(), -1);
    ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: y,
        treatment: t,
        at: 1.0,
        order: 1,
        scale: DerivativeScale::LogLog,
    })
    .validate()
    .unwrap();
    ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: y,
        treatment: t,
        weighting: DerivativeWeighting::Observed,
    })
    .validate()
    .unwrap();
    let mut posterior = GraphPosterior::new(
        2,
        vec![1.0],
        vec![2],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        1.0,
        InferenceDiagnostics::analytic("practitioner"),
        0,
    )
    .unwrap();
    posterior.atom_kind = GraphPosteriorAtomKind::Cpdag;
    assert_eq!(posterior.n_graphs, 1);
}
