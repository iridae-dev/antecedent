//! Builder-independent evidence for checked TemporalDag MeanCurve execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, GridSpec, Lag, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn series(shift: f64) -> TimeSeriesData {
    let n = 1200usize;
    let treatment =
        (0..n).map(|i| (f64::from((i * 37 % 997) as u32) - 498.0) / 250.0).collect::<Vec<_>>();
    let outcome = std::iter::once(0.0)
        .chain(
            (1..n).map(|i| shift + 1.2 + 1.8 * treatment[i - 1] + 0.03 * ((i as f64) * 0.17).sin()),
        )
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns(
        [("treatment", treatment.as_slice()), ("outcome", outcome.as_slice())],
        1,
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let treatment = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let outcome = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(treatment, outcome).unwrap();
    graph
}

fn query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.0, 0.5])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap())
}

fn means(result: &antecedent::result::StudyResult) -> Vec<f64> {
    let response = result.response.as_ref().expect("temporal response");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified response surface")
    };
    mean.to_vec()
}

#[test]
fn temporal_mean_curve_is_sealed_refreshable_and_has_precise_artifact_dependency() {
    let initial = series(0.0);
    let graph = graph();
    let query = query();
    let context = antecedent_core::ExecutionContext::for_tests(121);

    for accepted in [false, true] {
        let builder = if accepted {
            Study::series(initial.clone()).graph(AcceptedGraph::temporal_dag(graph.clone()))
        } else {
            Study::series(initial.clone()).graph(graph.clone())
        }
        .query(CausalQuery::Response(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
        let one_shot = builder.run(&context).unwrap();
        let mut prepared = builder.prepare(&context).unwrap();
        let info = prepared.checked_temporal_response_info().expect("sealed temporal plan");
        assert_eq!(info.query, query);
        assert_eq!(info.identifier, IdentifierId::TemporalBackdoorUnfolded);
        assert_eq!(info.estimator, EstimatorId::TemporalResponseGcomp);
        assert_eq!(info.uncertainty.as_ref(), "frequentist.no_interval");
        assert_eq!(info.grid_members.as_ref(), &[-0.5, 0.0, 0.5]);
        assert_eq!(info.horizons.as_ref(), &[1]);
        drop(builder);

        let first = prepared.estimate_series(&initial, &context).unwrap();
        let actual = means(&first);
        let one_shot_mean = means(&one_shot);
        for (a, b) in actual.iter().zip(one_shot_mean) {
            assert!((a - b).abs() < 1e-10);
        }
        for (&dose, &estimate) in [-0.5, 0.0, 0.5].iter().zip(&actual) {
            let truth = 1.2 + 1.8 * dose;
            assert!((estimate - truth).abs() < 0.08, "{dose}: {estimate} != {truth}");
        }

        let refreshed_data = series(0.4);
        let refreshed = prepared.refresh_series(refreshed_data, &context).unwrap();
        for (&dose, &estimate) in [-0.5, 0.0, 0.5].iter().zip(means(&refreshed).iter()) {
            let truth = 1.6 + 1.8 * dose;
            assert!((estimate - truth).abs() < 0.08, "refresh {dose}: {estimate}");
        }
        let artifact = prepared
            .encode_contracted_result(&refreshed, "checked-temporal-response", &context)
            .unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
            dependency.as_ref() == "dependencies.checked_temporal_response_operation"
        }));
        assert!(!consumed.acceptance.accepts_as_verified_program());
    }
}
