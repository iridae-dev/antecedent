//! Builder-independent evidence for checked Bayesian `TemporalDag` responses:
//! the dose × horizon mean curve and the single-step intervention response.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, IntervalInterpretation,
    Intervention, Lag, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

/// `y_t = shift + 1 + 2 t_{t-1} + 3 t_{t-2}` on a period-4 treatment with mean zero,
/// so `E[y | do(t@-1 = x)] = shift + 1 + 2 x` at horizon 1.
fn series(shift: f64) -> TimeSeriesData {
    let n = 240usize;
    let treatment: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            _ => -1.0,
        })
        .collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            shift
                + 1.0
                + 2.0 * i.checked_sub(1).map_or(0.0, |j| treatment[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| treatment[j])
        })
        .collect();
    TimeSeriesData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())], 1)
        .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

fn spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap()
}

fn mean_curve() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(spec())
}

fn intervention_response() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    })
    .with_temporal(spec())
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

fn truth(intervention: bool, shift: f64) -> Vec<f64> {
    if intervention { vec![shift + 3.0] } else { vec![shift + 1.0, shift + 3.0] }
}

#[test]
fn bayesian_temporal_responses_are_sealed_refreshable_and_have_precise_artifact_dependency() {
    let initial = series(0.0);
    let graph = graph();
    let context = ExecutionContext::for_tests(131);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512));
    for intervention in [false, true] {
        let query = if intervention { intervention_response() } else { mean_curve() };
        for accepted in [false, true] {
            let builder = if accepted {
                Study::series(initial.clone()).graph(AcceptedGraph::temporal_dag(graph.clone()))
            } else {
                Study::series(initial.clone()).graph(graph.clone())
            }
            .query(CausalQuery::Response(query.clone()))
            .inference(inference.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
            let one_shot = builder.run(&context).unwrap();
            let mut prepared = builder.prepare(&context).unwrap();
            let info = prepared.checked_temporal_response_info().expect("sealed temporal plan");
            assert_eq!(info.query, query);
            assert_eq!(info.identifier, IdentifierId::TemporalBackdoorUnfolded);
            assert_eq!(info.estimator, EstimatorId::TemporalResponseBayesian);
            assert_eq!(info.uncertainty.as_ref(), "bayesian.posterior_draws");
            assert_eq!(info.horizons.as_ref(), &[1]);
            if intervention {
                assert_eq!(info.grid_members.as_ref(), &[1.0]);
            } else {
                assert_eq!(info.grid_members.as_ref(), &[0.0, 1.0]);
            }
            drop(builder);

            let first = prepared.estimate_series(&initial, &context).unwrap();
            assert_eq!(first.estimand.method.as_ref(), "temporal.backdoor.unfolded");
            let response = first.response.as_ref().expect("response payload");
            assert_eq!(response.provenance_id.as_ref(), "estimate.response.temporal.bayesian");
            assert!(matches!(
                response.uncertainty,
                ResponseUncertainty::PointwiseBand {
                    interpretation: IntervalInterpretation::Credible,
                    ..
                }
            ));
            let actual = means(&first);
            for (a, b) in actual.iter().zip(means(&one_shot)) {
                assert!((a - b).abs() < 1e-9, "one-shot and prepared clicks agree: {a} vs {b}");
            }
            for (estimate, expected) in actual.iter().zip(truth(intervention, 0.0)) {
                assert!(
                    (estimate - expected).abs() < 0.1,
                    "intervention={intervention} accepted={accepted}: {estimate} != {expected}"
                );
            }

            let refreshed = prepared.refresh_series(series(0.5), &context).unwrap();
            for (estimate, expected) in means(&refreshed).iter().zip(truth(intervention, 0.5)) {
                assert!((estimate - expected).abs() < 0.1, "refresh: {estimate} != {expected}");
            }
            let widened = TimeSeriesData::from_f64_columns(
                [("t", &[0.0; 16][..]), ("y", &[0.0; 16][..]), ("extra", &[0.0; 16][..])],
                1,
            )
            .unwrap();
            assert!(
                prepared.refresh_series(widened, &context).is_err(),
                "schema-changing refresh must be refused"
            );
            let artifact = prepared
                .encode_contracted_result(
                    &refreshed,
                    "checked-bayesian-temporal-response",
                    &context,
                )
                .unwrap();
            let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
                dependency.as_ref() == "dependencies.checked_temporal_response_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}
