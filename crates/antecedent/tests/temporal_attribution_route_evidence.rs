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

// The query names original ticks [0, 40) and [40, 80). A lag-one graph
// excludes only tick 0, so the means use 39 and 40 observations respectively.
// The DGP is 1 + 2*x before tick 40 and 6 + second_shift + 2*x afterwards;
// its original x ramp restarts at tick 40. Compute the resulting finite-window
// mean difference directly, independently of the fitted attribution machinery.
fn complete_window_mean_shift(second_shift: f64) -> f64 {
    let x = |tick: u32| f64::from(tick % 40) * 0.1;
    let mean_baseline_x = (1..40).map(x).sum::<f64>() / 39.0;
    let mean_comparison_x = (40..80).map(x).sum::<f64>() / 40.0;
    5.0 + second_shift + 2.0 * (mean_comparison_x - mean_baseline_x)
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
    assert_eq!(truth["change"]["total_change"].as_f64().unwrap(), 5.0);
    let expected = complete_window_mean_shift(0.0);
    assert!((expected - 4.9).abs() < 1e-12);
    // Keep the original 0.1 model-sampling band around the complete-window truth.
    let tolerance = truth["change"]["tolerance"].as_f64().unwrap().min(0.1);
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
        (refreshed_change - complete_window_mean_shift(0.5)).abs() < tolerance,
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
    // An accepted TemporalDag is attributed as a supplied one.
    assert!(accepted.is_ok(), "{:?}", accepted.err());
    // The class and posterior cells are licensed by their own known-truth evidence
    // (class_attribution_known_truth); only a change attribution's validation is refused.
    let licensed = |query, graph, structure, inference, validation| {
        let cell = SupportCell::from_labels(query, graph, structure, inference, validation)
            .expect("support axis");
        assert!(matches!(classify(cell), CellStatus::Licensed), "{cell:?}");
    };
    licensed("AnomalyAttribution", "TemporalCpdag", "explicit", "Frequentist", "none");
    licensed("ChangeAttribution", "TemporalPag", "explicit", "Bayesian", "none");
    licensed("AnomalyAttribution", "TemporalCpdag", "graph_posterior", "Frequentist", "none");
    refuse(
        SupportCell::from_labels(
            "ChangeAttribution",
            "TemporalPag",
            "explicit",
            "Bayesian",
            "cheap",
        )
        .expect("support axis"),
        "Cheap and full validation of a change attribution",
    );
}

/// An accepted `TemporalDag` is attributed as a supplied one: the accepted record is the
/// structure's provenance, not a different graph. Both queries and both inference modes hold
/// the lag-aligned truth of the staged fixture.
#[test]
fn temporal_accepted_attribution_known_truth() {
    std::thread::Builder::new()
        .name("temporal-accepted-attribution".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(accepted_body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn accepted_body() {
    let ctx = ExecutionContext::for_tests(13);
    let truth = staged_attribution();
    let top_row = usize::try_from(truth["anomaly"]["top_row"].as_u64().unwrap()).unwrap();
    assert_eq!(truth["change"]["total_change"].as_f64().unwrap(), 5.0);
    let expected = complete_window_mean_shift(0.0);
    assert!((expected - 4.9).abs() < 1e-12);
    for (inference, anomaly_estimator, change_estimator) in [
        (InferenceMode::Frequentist, EstimatorId::GcmFit, EstimatorId::GcmFit),
        (
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4)),
            EstimatorId::GcmFitBayesian,
            EstimatorId::GcmAttributionBayesian,
        ),
    ] {
        let anomaly = anomaly_series(200.0);
        let anomaly_query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(1)],
            100,
        ));
        let anomaly_builder = Study::series(anomaly.clone())
            .graph(AcceptedGraph::from(temporal_chain()))
            .query(anomaly_query.clone())
            .inference(inference.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let anomaly_prepared = anomaly_builder.prepare(&ctx).unwrap();
        let RetainedOperation::Attribution(plan) = anomaly_prepared.retained_operation() else {
            panic!("sealed anomaly operation")
        };
        assert_eq!(plan.query, anomaly_query);
        assert_eq!(plan.identifier, IdentifierId::GcmParametric);
        assert_eq!(plan.estimator, anomaly_estimator);
        assert_eq!(plan.graph_edges.as_ref(), &[(2, 1)]);
        drop(anomaly_builder);
        let result = anomaly_prepared.estimate_series(&anomaly, &ctx).unwrap();
        assert_eq!(result.posterior.is_some(), matches!(inference, InferenceMode::Bayesian(_)));
        let scores = result.anomaly().expect("anomaly result");
        let score = scores.iter().find(|score| score.target == VariableId::from_raw(1)).unwrap();
        let top = score.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        assert_eq!(score.rows[top.0], top_row, "the injected outlier is the top anomaly");
        assert!(score.scores[top.0] >= truth["anomaly"]["score_min"].as_f64().unwrap());

        let change = change_series(0.0);
        let change_target = CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        ));
        let change_builder = Study::series(change.clone())
            .graph(AcceptedGraph::from(temporal_chain()))
            .query(change_target.clone())
            .inference(inference.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let change_prepared = change_builder.prepare(&ctx).unwrap();
        let RetainedOperation::Attribution(plan) = change_prepared.retained_operation() else {
            panic!("sealed change operation")
        };
        assert_eq!(plan.query, change_target);
        assert_eq!(plan.estimator, change_estimator);
        drop(change_builder);
        let result = change_prepared.estimate_series(&change, &ctx).unwrap();
        let total = result.change_attribution().unwrap().total_change;
        let tolerance = truth["change"]["tolerance"].as_f64().unwrap().min(0.1);
        assert!((total - expected).abs() < tolerance, "known population mean shift: {total}");
    }
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
    assert_eq!(truth["change"]["total_change"].as_f64().unwrap(), 5.0);
    let expected = complete_window_mean_shift(0.0);
    assert!((expected - 4.9).abs() < 1e-12);
    // Keep the original 0.1 model-sampling band around the complete-window truth.
    let tolerance = truth["change"]["tolerance"].as_f64().unwrap().min(0.1);
    let total = result.change_attribution().unwrap().total_change;
    assert!((total - expected).abs() < tolerance, "known population mean shift: {total}");
    assert!(result.posterior.is_some());
    assert!(result.effect().is_finite());
    // Retain the existing 0.5 posterior band for the four-draw estimate; the
    // center is the DGP mean difference on the complete observed windows.
    assert!(
        (result.effect() - expected).abs() < 0.5,
        "posterior mean {} stays near the shift {expected}",
        result.effect()
    );
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

#[test]
fn temporal_anomaly_rows_use_original_ticks_after_a_two_tick_prefix() {
    std::thread::Builder::new()
        .name("temporal-original-anomaly-ticks".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let ctx = ExecutionContext::for_tests(13);
            let data = anomaly_series(200.0);
            let mut graph = TemporalDag::empty();
            let parent = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
            let outcome = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
            graph.insert_directed(parent, outcome).unwrap();
            let study = |rows: Vec<usize>| {
                Study::series(data.clone())
                    .graph(graph.clone())
                    .query(CausalQuery::AnomalyAttribution(
                        AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100)
                            .with_unit_rows(rows),
                    ))
                    .inference(InferenceMode::Frequentist)
                    .refute(RefuteSuite::None)
                    .build()
                    .unwrap()
            };
            // Of source ticks 0..20, only 2..20 have a complete two-lag history.
            let requested = vec![2, 7, 19];
            let prepared = study(requested.clone()).prepare(&ctx).unwrap();
            let result = prepared.estimate_series(&data, &ctx).unwrap();
            let bytes = prepared.encode_contracted_result(&result, "temporal-complete-window", &ctx).unwrap();
            let (artifact, _, _) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
            let contract = antecedent_io::decode_analysis_result_contract(&artifact).unwrap().unwrap();
            let program = contract.program.as_ref().unwrap();
            assert_eq!(program.temporal_attribution_rows, Some(
                antecedent_io::TemporalAttributionRowsWire::CompleteObservedDerivedRunBlocksV2
            ));
            assert_eq!(*antecedent_io::program_digest(program).unwrap().as_bytes(), contract.identities.program);
            let held = result.retained_admitted_bytes().expect("mapped row identity retains admission");
            assert!(held > 0);
            let result_clone = result.clone();
            assert_eq!(result_clone.retained_admitted_bytes(), Some(held));
            let score = result.anomaly().unwrap().iter()
                .find(|score| score.target == VariableId::from_raw(1)).unwrap();
            assert_eq!(score.rows.as_ref(), requested.as_slice());
            assert_eq!(score.scores.len(), requested.len());
            assert!(score.scores.iter().all(|value| value.is_finite()));

            // A non-temporal attribution never advertises the temporal row law.
            let x: Vec<f64> = (0..20).map(f64::from).collect();
            let y: Vec<f64> = (0..20).map(|i| if i == 19 { 200.0 } else { 2.0 * f64::from(i) }).collect();
            let table = antecedent_data::TabularData::from_f64_columns([
                ("x", x.as_slice()), ("y", y.as_slice()),
            ]).unwrap();
            let mut static_graph = antecedent_graph::Dag::with_variables(2);
            static_graph.insert_directed(
                antecedent_graph::DenseNodeId::from_raw(0), antecedent_graph::DenseNodeId::from_raw(1),
            ).unwrap();
            let static_study = Study::tabular(table.clone()).graph(static_graph)
                .query(CausalQuery::AnomalyAttribution(
                    AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100),
                ))
                .inference(InferenceMode::Frequentist).refute(RefuteSuite::None).build().unwrap();
            let static_prepared = static_study.prepare(&ctx).unwrap();
            let static_result = static_prepared.estimate(&table, &ctx).unwrap();
            let bytes = static_prepared.encode_contracted_result(&static_result, "static-anomaly", &ctx).unwrap();
            let (artifact, _, _) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
            let contract = antecedent_io::decode_analysis_result_contract(&artifact).unwrap().unwrap();
            let program = contract.program.as_ref().unwrap();
            assert_eq!(program.temporal_attribution_rows, None);
            assert_eq!(*antecedent_io::program_digest(program).unwrap().as_bytes(), contract.identities.program);

            // Tick 1 exists in the source, but has no complete two-step history.
            let weighted = data.with_weights(vec![1.0; 20].into()).unwrap();
            let weighted_result = Study::series(weighted).graph(graph.clone())
                .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100)))
                .inference(InferenceMode::Frequentist).refute(RefuteSuite::None).build()
                .and_then(|study| study.run(&ctx));
            assert!(weighted_result.unwrap_err().to_string().contains("weights"));
            let error = study(vec![1, 7]).run(&ctx).unwrap_err();
            assert!(matches!(&error, antecedent::CausalError::Compile { message }
                if message.contains("original row 1") && message.contains("without complete observed history")), "{error}");
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn temporal_rollout_reads_original_tail_when_complete_window_is_shorter_than_history() {
    std::thread::Builder::new()
        .name("temporal-rollout-source-tail".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            use antecedent::analysis::structure_route::SimulationOutcome;
            use antecedent_core::{InterventionSequence, SimulateQuery, SimulationTarget};

            // n = 31, history = 16, so the fitting window has only 15 rows. The
            // first rollout step still needs source tick 31 - 16 = 15.
            let x: Vec<f64> = (0..31).map(|i| (f64::from(i) * 0.37).sin()).collect();
            let y: Vec<f64> = (0_usize..31)
                .map(|i| 0.8 * x[i.saturating_sub(16)] + (i as f64 * 1.13).sin() * 0.1)
                .collect();
            let data = series(&x, &y);
            let mut graph = TemporalDag::empty();
            let lagged =
                ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(16)).unwrap();
            let outcome =
                ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
            graph.insert_directed(lagged, outcome).unwrap();
            let query =
                CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::TemporalRollout {
                    schedule: InterventionSequence::new(Vec::new()),
                    n_rows: 1,
                    n_steps: 1,
                    window: 16,
                    history: Arc::from([]),
                    seed: 9,
                }));
            let ctx = ExecutionContext::for_tests(13);
            let study = Study::series(data.clone())
                .graph(graph)
                .query(query)
                .inference(InferenceMode::Frequentist)
                .refute(RefuteSuite::None)
                .build()
                .unwrap();
            let result = study.prepare(&ctx).unwrap().estimate_series(&data, &ctx).unwrap();
            let SimulationOutcome::Temporal(path) = &result.simulation().unwrap().outcome else {
                panic!("temporal rollout expected");
            };
            assert_eq!(path.n_rows, 1);
            assert_eq!(path.steps.len(), 1);
            // The unfolded order has x,y at time 0, then x,y at the oldest lag.
            // Its x(t-16) assignment must read the observed x at original tick 15.
            assert_eq!(path.column(0, 2).unwrap(), &[x[15]]);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn temporal_rollout_keeps_live_callback_bound_to_its_compiled_owner() {
    use antecedent::analysis::structure_route::SimulationOutcome;
    use antecedent_core::{
        BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec,
        Intervention, InterventionSequence, SequencedIntervention, SimulateQuery, SimulationTarget,
        TemporalPolicy, ValueExpr,
    };

    #[derive(Debug)]
    struct ConstantCallback {
        spec: CallbackSpec,
        value: f64,
    }
    impl BatchCallback for ConstantCallback {
        fn spec(&self) -> &CallbackSpec {
            &self.spec
        }
        fn call(&self, _inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
            out.fill(self.value);
            Ok(())
        }
    }

    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let x: Vec<f64> = (0..40).map(|i| (f64::from(i) * 0.37).sin()).collect();
            let y: Vec<f64> = (0_usize..40)
                .map(|i| 0.8 * x[i.saturating_sub(1)] + (i as f64 * 1.13).sin() * 0.1)
                .collect();
            let data = series(&x, &y);
            let mut graph = TemporalDag::empty();
            let lagged =
                ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
            let outcome =
                ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
            graph.insert_directed(lagged, outcome).unwrap();
            let spec = CallbackSpec {
                name: "temporal-policy".into(),
                inputs: Arc::from([]),
                output: CallbackOutput::Real,
                dependencies: Arc::from([]),
                implementation: CallbackImplementation::Runtime,
            };
            let study = |value| {
                let callback: Arc<dyn BatchCallback> =
                    Arc::new(ConstantCallback { spec: spec.clone(), value });
                let action = Intervention::assign(
                    VariableId::from_raw(0),
                    ValueExpr::callback(callback, Arc::<[ValueExpr]>::from([])),
                );
                let schedule = InterventionSequence::new([SequencedIntervention::new(
                    action,
                    TemporalPolicy::pulse(0),
                )]);
                let query =
                    CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::TemporalRollout {
                        schedule,
                        n_rows: 1,
                        n_steps: 1,
                        window: 1,
                        history: Arc::from([]),
                        seed: 9,
                    }));
                Study::series(data.clone())
                    .graph(graph.clone())
                    .query(query)
                    .inference(InferenceMode::Frequentist)
                    .refute(RefuteSuite::None)
                    .build()
                    .unwrap()
            };
            let first = study(2.0);
            let other = study(3.0);
            let ctx = ExecutionContext::for_tests(13);
            let plan = first.compile(&ctx).unwrap();
            assert!(
                other.execute(&plan, &ctx).is_err(),
                "a distinct live callback cannot use this plan"
            );
            let result = first.execute(&plan, &ctx).unwrap();
            let SimulationOutcome::Temporal(path) = &result.simulation().unwrap().outcome else {
                panic!("temporal rollout expected")
            };
            assert_eq!(path.column(0, 0).unwrap(), &[2.0]);
        })
        .unwrap()
        .join()
        .unwrap();
}
