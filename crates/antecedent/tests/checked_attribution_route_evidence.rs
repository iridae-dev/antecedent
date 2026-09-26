//! Builder independent evidence for frequentist explicit DAG attribution plans.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, ExecutionContext,
    PopulationSelector, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn dag() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn anomaly_data(outlier: f64) -> TabularData {
    let x: Vec<_> = (0..20).map(f64::from).collect();
    let y: Vec<_> = (0..20).map(|i| if i == 19 { outlier } else { 2.0 * f64::from(i) }).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn change_data(second_shift: f64) -> TabularData {
    let x: Vec<_> = (0..80).map(|i| f64::from(i % 40) * 0.1).collect();
    let y: Vec<_> = (0..80)
        .map(|i| if i < 40 { 1.0 + 2.0 * x[i] } else { 6.0 + second_shift + 2.0 * x[i] })
        .collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

#[test]
fn all_frequentist_static_dag_attribution_plans_are_sealed_and_dependency_refused() {
    let ctx = ExecutionContext::for_tests(13);

    let anomaly = anomaly_data(200.0);
    let anomaly_query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
        [VariableId::from_raw(1)],
        100,
    ));
    let anomaly_builder = Study::tabular(anomaly.clone())
        .graph(dag())
        .query(anomaly_query.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut anomaly_prepared = anomaly_builder.prepare(&ctx).unwrap();
    let plan = anomaly_prepared.checked_attribution_info().expect("sealed anomaly operation");
    assert_eq!(plan.query, anomaly_query);
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, EstimatorId::GcmFit);
    assert_eq!(plan.graph_edges.as_ref(), &[(0, 1)]);
    drop(anomaly_builder);
    let anomaly_result = anomaly_prepared.estimate(&anomaly, &ctx).unwrap();
    let scores = anomaly_result.anomaly.as_ref().expect("anomaly result");
    let score = scores.iter().find(|score| score.target == VariableId::from_raw(1)).unwrap();
    let maximum = score.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    assert_eq!(score.rows[maximum.0], 19, "the injected outlier is the top anomaly");
    let refreshed_anomaly = anomaly_data(220.0);
    let refreshed = anomaly_prepared.refresh(refreshed_anomaly, &ctx).unwrap();
    assert!(refreshed.anomaly.as_ref().unwrap()[0].scores.iter().any(|score| score.is_finite()));
    let artifact = anomaly_prepared
        .encode_contracted_result(&refreshed, "checked-anomaly-attribution", &ctx)
        .unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|reason| { reason.as_ref() == "dependencies.checked_attribution_operation" })
    );
    assert!(!consumed.acceptance.accepts_as_verified_program());

    let change = change_data(0.0);
    let change_query = ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
    );
    let change_target = CausalQuery::ChangeAttribution(change_query.clone());
    let change_builder = Study::tabular(change.clone())
        .graph(dag())
        .query(change_target.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut change_prepared = change_builder.prepare(&ctx).unwrap();
    let plan = change_prepared.checked_attribution_info().expect("sealed change operation");
    assert_eq!(plan.query, change_target);
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, EstimatorId::GcmFit);
    assert_eq!(plan.graph_edges.as_ref(), &[(0, 1)]);
    drop(change_builder);
    let initial = change_prepared.estimate(&change, &ctx).unwrap();
    let initial_change = initial.change_attribution.as_ref().unwrap().total_change;
    assert!((initial_change - 5.0).abs() < 1e-8, "known population mean shift: {initial_change}");
    let refreshed_data = change_data(0.5);
    let refreshed = change_prepared.refresh(refreshed_data, &ctx).unwrap();
    let refreshed_change = refreshed.change_attribution.as_ref().unwrap().total_change;
    assert!(
        (refreshed_change - 5.5).abs() < 1e-8,
        "refreshed known population shift: {refreshed_change}"
    );
    let artifact = change_prepared
        .encode_contracted_result(&refreshed, "checked-change-attribution", &ctx)
        .unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|reason| { reason.as_ref() == "dependencies.checked_attribution_operation" })
    );
    assert!(!consumed.acceptance.accepts_as_verified_program());
}

#[test]
fn all_bayesian_static_dag_attribution_plans_are_sealed_and_dependency_refused() {
    let ctx = ExecutionContext::for_tests(14);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8));

    let anomaly = anomaly_data(200.0);
    let anomaly_query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
        [VariableId::from_raw(1)],
        100,
    ));
    let anomaly_builder = Study::tabular(anomaly.clone())
        .graph(dag())
        .query(anomaly_query.clone())
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut anomaly_prepared = anomaly_builder.prepare(&ctx).unwrap();
    let plan = anomaly_prepared.checked_attribution_info().expect("sealed anomaly operation");
    assert_eq!(plan.query, anomaly_query);
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, EstimatorId::GcmFitBayesian);
    drop(anomaly_builder);
    let anomaly_result = anomaly_prepared.estimate(&anomaly, &ctx).unwrap();
    assert!(anomaly_result.posterior.is_some());
    let scores = anomaly_result.anomaly.as_ref().expect("anomaly result");
    let score = scores.iter().find(|score| score.target == VariableId::from_raw(1)).unwrap();
    let maximum = score.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    assert_eq!(score.rows[maximum.0], 19, "the injected outlier remains the top anomaly");
    let refreshed = anomaly_prepared.refresh(anomaly_data(220.0), &ctx).unwrap();
    assert!(refreshed.posterior.is_some());
    let artifact = anomaly_prepared
        .encode_contracted_result(&refreshed, "checked-bayesian-anomaly-attribution", &ctx)
        .unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|reason| { reason.as_ref() == "dependencies.checked_attribution_operation" })
    );
    assert!(!consumed.acceptance.accepts_as_verified_program());

    let change = change_data(0.0);
    let change_query = ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
    );
    let change_target = CausalQuery::ChangeAttribution(change_query.clone());
    let change_builder = Study::tabular(change.clone())
        .graph(dag())
        .query(change_target.clone())
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut change_prepared = change_builder.prepare(&ctx).unwrap();
    let plan = change_prepared.checked_attribution_info().expect("sealed change operation");
    assert_eq!(plan.query, change_target);
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, EstimatorId::GcmAttributionBayesian);
    drop(change_builder);
    let initial = change_prepared.estimate(&change, &ctx).unwrap();
    assert!(initial.posterior.is_some());
    let change_result = initial.change_attribution.as_ref().expect("change result");
    assert!((change_result.total_change - 5.0).abs() < 0.15);
    let refreshed = change_prepared.refresh(change_data(0.5), &ctx).unwrap();
    assert!(refreshed.posterior.is_some());
    let refreshed_change = refreshed.change_attribution.as_ref().unwrap();
    assert!((refreshed_change.total_change - 5.5).abs() < 0.15);
    let artifact = change_prepared
        .encode_contracted_result(&refreshed, "checked-bayesian-change-attribution", &ctx)
        .unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|reason| { reason.as_ref() == "dependencies.checked_attribution_operation" })
    );
    assert!(!consumed.acceptance.accepts_as_verified_program());
}
