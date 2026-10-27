//! Lag-aligned `TemporalDag` attribution: known truth, checked execution, typed refusals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, Study, SupportCell, classify, refused_message,
};
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, ExecutionContext, IntervalMethod,
    Lag, PopulationSelector, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::IdentificationStatus;

fn temporal_chain() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    graph
}

fn series(x: &[f64], y: &[f64]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns([("x", x), ("y", y)], 1).unwrap()
}

fn anomaly_series(outlier: f64) -> TimeSeriesData {
    let x: Vec<f64> = (0..20).map(f64::from).collect();
    let y: Vec<f64> = (0..20).map(|i| if i == 19 { outlier } else { 2.0 * f64::from(i) }).collect();
    series(&x, &y)
}

fn change_series(second_shift: f64) -> TimeSeriesData {
    let x: Vec<f64> = (0..80).map(|i| f64::from(i % 40) * 0.1).collect();
    let y: Vec<f64> = (0..80)
        .map(|i| {
            let x = f64::from(i % 40) * 0.1;
            if i < 40 { 1.0 + 2.0 * x } else { 6.0 + second_shift + 2.0 * x }
        })
        .collect();
    series(&x, &y)
}

fn staged_attribution() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_attribution/expected.json"
    ))
    .unwrap()
}

fn refuse(cell: SupportCell, reason: &str) {
    assert!(matches!(classify(cell), CellStatus::Refused));
    assert!(refused_message(cell).contains(reason), "refusal was {}", refused_message(cell));
}

#[test]
fn temporal_attribution_frequentist_known_truth() {
    std::thread::Builder::new()
        .name("temporal-attribution-frequentist".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(frequentist_body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

#[allow(
    clippy::too_many_lines,
    reason = "the route-evidence body: builds, seals, executes and re-executes each frequentist temporal attribution"
)]
fn frequentist_body() {
    let ctx = ExecutionContext::for_tests(13);
    let truth = staged_attribution();
    let graph = temporal_chain();

    let anomaly = anomaly_series(200.0);
    let anomaly_query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
        [VariableId::from_raw(1)],
        100,
    ));
    let anomaly_builder = Study::series(anomaly.clone())
        .graph(graph.clone())
        .query(anomaly_query.clone())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut anomaly_prepared = anomaly_builder.prepare(&ctx).unwrap();
    let RetainedOperation::Attribution(plan) = anomaly_prepared.retained_operation() else {
        panic!("sealed anomaly operation")
    };
    assert_eq!(plan.query, anomaly_query);
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, EstimatorId::GcmFit);
    assert_eq!(plan.graph_edges.as_ref(), &[(2, 1)]);
    assert_eq!(
        plan.identification_status,
        IdentificationStatus::IdentifiedUnderParametricRestrictions
    );
    drop(anomaly_builder);
    let anomaly_result = anomaly_prepared.estimate_series(&anomaly, &ctx).unwrap();
    assert!(anomaly_result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    assert!(
        anomaly_result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.lag_aligned")
    );
    assert!(anomaly_result.posterior.is_none());
    assert_eq!(
        anomaly_result.interval().as_ref().map(|binding| binding.method),
        Some(IntervalMethod::None)
    );
    let scores = anomaly_result.anomaly().expect("anomaly result");
    let score = scores.iter().find(|score| score.target == VariableId::from_raw(1)).unwrap();
    let top = score.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let top_row = usize::try_from(truth["anomaly"]["top_row"].as_u64().unwrap()).unwrap();
    assert!(score.scores[top.0] >= truth["anomaly"]["score_min"].as_f64().unwrap());
    assert_eq!(score.rows[top.0], top_row, "the injected outlier is the top anomaly");
    let refreshed = anomaly_prepared.refresh_series(anomaly_series(220.0), &ctx).unwrap();
    assert!(refreshed.anomaly().unwrap()[0].scores.iter().any(|score| score.is_finite()));
    let artifact = anomaly_prepared
        .encode_contracted_result(&refreshed, "checked-temporal-anomaly-attribution", &ctx)
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

    let change = change_series(0.0);
    let change_query = ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
    );
    let change_target = CausalQuery::ChangeAttribution(change_query);
    let change_builder = Study::series(change.clone())
        .graph(temporal_chain())
        .query(change_target.clone())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let mut change_prepared = change_builder.prepare(&ctx).unwrap();
    let RetainedOperation::Attribution(plan) = change_prepared.retained_operation() else {
        panic!("sealed change operation")
    };
    assert_eq!(plan.query, change_target);
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, EstimatorId::GcmFit);
    assert_eq!(plan.graph_edges.as_ref(), &[(2, 1)]);
    drop(change_builder);
    let initial = change_prepared.estimate_series(&change, &ctx).unwrap();
    let expected = truth["change"]["total_change"].as_f64().unwrap();
    let tolerance = truth["change"]["tolerance"].as_f64().unwrap();
    let initial_change = initial.change_attribution().unwrap().total_change;
    assert!(
        (initial_change - expected).abs() < tolerance,
        "known population mean shift: {initial_change}"
    );
    assert!(initial.posterior.is_none());
    assert_eq!(
        initial.interval().as_ref().map(|binding| binding.method),
        Some(IntervalMethod::None)
    );
    let refreshed = change_prepared.refresh_series(change_series(0.5), &ctx).unwrap();
    let refreshed_change = refreshed.change_attribution().unwrap().total_change;
    assert!(
        (refreshed_change - 5.5).abs() < tolerance,
        "refreshed known population shift: {refreshed_change}"
    );
    let artifact = change_prepared
        .encode_contracted_result(&refreshed, "checked-temporal-change-attribution", &ctx)
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

    let cheap = Study::series(anomaly.clone())
        .graph(temporal_chain())
        .query(anomaly_query.clone())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::Cheap)
        .build();
    // Cheap and full validation are the functional refuter state: an executing, unlicensed cell.
    assert!(cheap.is_ok(), "{:?}", cheap.err());
    let full = Study::series(anomaly.clone())
        .graph(temporal_chain())
        .query(anomaly_query.clone())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::Full)
        .build();
    assert!(full.is_ok(), "{:?}", full.err());
    let accepted = Study::series(anomaly.clone())
        .graph(AcceptedGraph::from(temporal_chain()))
        .query(anomaly_query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build();
    // An accepted TemporalDag is attributed as a supplied one and stays unlicensed.
    assert!(accepted.is_ok(), "{:?}", accepted.err());
    refuse(
        SupportCell::from_labels(
            "AnomalyAttribution",
            "TemporalCpdag",
            "explicit",
            "Frequentist",
            "none",
        )
        .expect("support axis"),
        "every temporal DAG of its class",
    );
    refuse(
        SupportCell::from_labels(
            "ChangeAttribution",
            "TemporalPag",
            "explicit",
            "Bayesian",
            "none",
        )
        .expect("support axis"),
        "every temporal MAG completion",
    );
    // A posterior over structures attributes every atom as a supplied structure of its kind and
    // mixes them by posterior probability: the cell executes, and stays unlicensed.
    refuse(
        SupportCell::from_labels(
            "AnomalyAttribution",
            "TemporalCpdag",
            "graph_posterior",
            "Frequentist",
            "none",
        )
        .expect("support axis"),
        "attributes every atom as a supplied structure",
    );
}

#[test]
fn temporal_attribution_bayesian_known_truth() {
    std::thread::Builder::new()
        .name("temporal-attribution-bayesian".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(bayesian_body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

#[allow(
    clippy::too_many_lines,
    reason = "the route-evidence body: builds, seals, executes and re-executes each Bayesian temporal attribution"
)]
fn bayesian_body() {
    let ctx = ExecutionContext::for_tests(13);
    let truth = staged_attribution();
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4));

    let anomaly = anomaly_series(200.0);
    let anomaly_query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
        [VariableId::from_raw(1)],
        100,
    ));
    let anomaly_builder = Study::series(anomaly.clone())
        .graph(temporal_chain())
        .query(anomaly_query.clone())
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let anomaly_prepared = anomaly_builder.prepare(&ctx).unwrap();
    let RetainedOperation::Attribution(plan) = anomaly_prepared.retained_operation() else {
        panic!("sealed anomaly operation")
    };
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, EstimatorId::GcmFitBayesian);
    drop(anomaly_builder);
    let anomaly_result = anomaly_prepared.estimate_series(&anomaly, &ctx).unwrap();
    // The block Bayesian bootstrap publishes its posterior and credible interval.
    assert!(anomaly_result.posterior.is_some());
    assert!(anomaly_result.effect().is_finite());
    assert_eq!(
        anomaly_result.interval().as_ref().map(|binding| binding.method),
        Some(IntervalMethod::PosteriorQuantile)
    );
    assert!(
        anomaly_result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "gcm.attribution.block_bayesian_bootstrap")
    );
    let scores = anomaly_result.anomaly().unwrap();
    let score = scores.iter().find(|score| score.target == VariableId::from_raw(1)).unwrap();
    let top = score.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    assert_eq!(
        score.rows[top.0],
        usize::try_from(truth["anomaly"]["top_row"].as_u64().unwrap()).unwrap()
    );
    let artifact = anomaly_prepared
        .encode_contracted_result(&anomaly_result, "checked-temporal-bayesian-anomaly", &ctx)
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

    let change = change_series(0.0);
    let change_target = CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
    ));
    let change_builder = Study::series(change.clone())
        .graph(temporal_chain())
        .query(change_target)
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let change_prepared = change_builder.prepare(&ctx).unwrap();
    let RetainedOperation::Attribution(plan) = change_prepared.retained_operation() else {
        panic!("sealed change operation")
    };
    assert_eq!(plan.estimator, EstimatorId::GcmAttributionBayesian);
    drop(change_builder);
    let result = change_prepared.estimate_series(&change, &ctx).unwrap();
    let expected = truth["change"]["total_change"].as_f64().unwrap();
    let tolerance = truth["change"]["tolerance"].as_f64().unwrap();
    let total = result.change_attribution().unwrap().total_change;
    assert!((total - expected).abs() < tolerance, "known population mean shift: {total}");
    assert!(result.posterior.is_some());
    assert!(result.effect().is_finite());
    assert!((result.effect() - expected).abs() < 2.0, "posterior mean stays near the shift");
    assert_eq!(
        result.interval().as_ref().map(|binding| binding.method),
        Some(IntervalMethod::PosteriorQuantile)
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "gcm.attribution.block_bayesian_bootstrap")
    );
    let artifact = change_prepared
        .encode_contracted_result(&result, "checked-temporal-bayesian-change", &ctx)
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

    let mut prior = BayesianConfig::conjugate().n_draws(4);
    prior.prior_artifact = Some(Arc::<[u8]>::from([1u8]));
    let refused = Study::series(anomaly)
        .graph(temporal_chain())
        .query(anomaly_query)
        .inference(InferenceMode::Bayesian(prior))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx);
    assert!(refused.unwrap_err().to_string().contains("does not accept coefficient priors"));
}
