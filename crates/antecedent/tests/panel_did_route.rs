//! Retained native balanced-panel DiD route.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PanelDidQuery, Study};
use antecedent_core::{Assumption, VariableId};
use antecedent_data::TabularData;

fn fixture() -> (TabularData, PanelDidQuery) {
    let mut ids = Vec::new();
    let mut treated = Vec::new();
    let mut post = Vec::new();
    let mut clusters = Vec::new();
    let mut outcome = Vec::new();
    let changes = [2.0, 4.0, 6.0, 8.0, 0.0, 2.0, 4.0, 6.0];
    for (i, change) in changes.iter().enumerate() {
        for after in [false, true] {
            ids.push(format!("s{i}"));
            treated.push(i < 4);
            post.push(after);
            clusters.push(format!("c{i}"));
            outcome.push(i as f64 * 10.0 + if after { *change } else { 0.0 });
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::new(
        VariableId::from_raw(0),
        treated,
        post,
        ids.iter().map(|id| Arc::<str>::from(id.as_str())).collect::<Vec<_>>(),
        clusters.iter().map(|id| Arc::<str>::from(id.as_str())).collect::<Vec<_>>(),
    );
    (data, query)
}

#[test]
fn panel_did_runs_identically_through_study_and_prepared_routes() {
    let (data, query) = fixture();
    let context = ExecutionContext::for_tests(7);
    let result =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&context).unwrap();
    let estimate = result.panel_did.as_ref().unwrap();
    assert_eq!(estimate.effect, 2.0);
    assert!((estimate.standard_error - (20.0_f64 / 7.0).sqrt()).abs() < 1e-12);
    assert_eq!(estimate.treated_subjects, 4);
    assert_eq!(estimate.comparison_subjects, 4);
    assert_eq!(estimate.clusters, 8);
    assert_eq!(estimate.uncertainty.as_ref(), "cluster_robust_standard_error_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, Assumption::Custom { id, .. } if id.as_ref() == "parallel_trends"
    )));
    assert!(!result.diagnostics.is_empty());
    assert!(result.diagnostics.iter().any(|diagnostic|
        diagnostic.code.as_ref() == "identification.quasi.parallel_trends_untestable_two_periods"
    ));

    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    let refreshed = prepared.estimate(&data, &context).unwrap();
    assert_eq!(refreshed.panel_did, result.panel_did);
    assert_eq!(refreshed.support_status, None);
    let bytes = prepared.encode_contracted_result(&refreshed, "thin-panel-did", &context).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    body.panel_did.as_mut().unwrap().graphless_support_status = Some("licensed".into());
    assert!(antecedent_io::encode_analysis_result_artifact(&body, header.variable_names, "forged-thin-did-license").is_err());
}

#[test]
fn supported_panel_did_interval_round_trips_and_rejects_fabrication() {
    let mut ids = Vec::new();
    let mut clusters = Vec::new();
    let mut treated = Vec::new();
    let mut post = Vec::new();
    let mut outcome = Vec::new();
    for group in [false, true] {
        for cluster in 0..30 {
            for after in [false, true] {
                ids.push(Arc::<str>::from(format!("subject-{group}-{cluster}")));
                clusters.push(Arc::<str>::from(format!("cluster-{group}-{cluster}")));
                treated.push(group);
                post.push(after);
                outcome.push(if after { cluster as f64 * 0.1 + if group { 2.0 } else { 0.0 } } else { 0.0 });
            }
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::new(VariableId::from_raw(0), treated, post, ids, clusters);
    let context = ExecutionContext::for_tests(91);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    let result = prepared.estimate(&data, &context).unwrap();
    let did = result.panel_did.as_ref().unwrap();
    assert!((did.effect - 2.0).abs() < 1e-12);
    assert_eq!(did.uncertainty.as_ref(), "cluster_robust_normal_interval_independent_clusters");
    let bounds = did.interval_95.unwrap();
    assert!(bounds[0] <= did.effect && did.effect <= bounds[1]);
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::AnalyticSe);
    assert_eq!(result.interval.as_ref().unwrap().dependence, "cluster");
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let bytes = prepared.encode_contracted_result(&result, "supported-panel-did", &context).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.panel_did.as_ref().unwrap().interval_95, Some(bounds));
    assert_eq!(body.panel_did.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
    let mut fabricated = body.clone();
    fabricated.panel_did.as_mut().unwrap().interval_95 = Some([bounds[0] - 1.0, bounds[1]]);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
}

#[test]
fn supported_repeated_cross_section_interval_round_trips_and_rejects_fabrication() {
    let mut ids = Vec::new();
    let mut clusters = Vec::new();
    let mut treated = Vec::new();
    let mut post = Vec::new();
    let mut outcome = Vec::new();
    for group in [false, true] {
        for after in [false, true] {
            for cell_cluster in 0..30 {
                ids.push(Arc::<str>::from(format!("subject-{group}-{after}-{cell_cluster}")));
                clusters.push(Arc::<str>::from(format!("cluster-{group}-{after}-{cell_cluster}")));
                treated.push(group);
                post.push(after);
                outcome.push(f64::from(group) + 0.5 * f64::from(after)
                    + 2.0 * f64::from(group && after) + cell_cluster as f64 * 0.1);
            }
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::repeated_cross_section(VariableId::from_raw(0), treated, post, ids, clusters);
    let context = ExecutionContext::for_tests(93);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    let result = prepared.estimate(&data, &context).unwrap();
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let did = result.panel_did.as_ref().unwrap();
    assert!((did.effect - 2.0).abs() < 1e-12);
    let interval = did.interval_95.expect("30 independent clusters in every cell");
    let bytes = prepared.encode_contracted_result(&result, "repeated-did-license", &context).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.panel_did.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
    body.panel_did.as_mut().unwrap().interval_95 = Some([interval[0] - 1.0, interval[1]]);
    assert!(antecedent_io::encode_analysis_result_artifact(&body, header.variable_names, "forged-repeated-did").is_err());
}

#[test]
fn panel_did_refuses_missing_wave_and_insufficient_group_clusters() {
    let context = ExecutionContext::for_tests(1);
    let (data, query) = fixture();
    let missing = PanelDidQuery::new(
        query.outcome,
        query.treated[..15].to_vec(),
        query.post[..15].to_vec(),
        query.subjects[..15].to_vec(),
        query.clusters[..15].to_vec(),
    );
    assert!(Study::tabular(data.clone()).query(missing).build().unwrap().run(&context).is_err());

    let collapsed = PanelDidQuery::new(
        query.outcome,
        query.treated.to_vec(),
        query.post.to_vec(),
        query.subjects.to_vec(),
        vec![Arc::<str>::from("one"); query.subjects.len()],
    );
    let prepared =
        Study::tabular(data.clone()).query(collapsed).build().unwrap().prepare(&context).unwrap();
    assert!(prepared.estimate(&data, &context).is_err());
}

#[test]
fn panel_did_refuses_cluster_shared_across_treatment_groups() {
    let (data, mut query) = fixture();
    let mut clusters = query.clusters.to_vec();
    clusters[8] = clusters[0].clone();
    clusters[9] = clusters[0].clone();
    query.clusters = clusters.into();
    let error = Study::tabular(data).query(query).build().unwrap()
        .run(&ExecutionContext::for_tests(1)).unwrap_err();
    assert!(error.to_string().contains("clusters nested within treatment groups"));
}

#[test]
fn repeated_cross_section_did_runs_through_study_and_prepared_routes() {
    let outcome = [0.0, 2.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0];
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::repeated_cross_section(
        VariableId::from_raw(0),
        [false, false, false, false, true, true, true, true],
        [false, false, true, true, false, false, true, true],
        (0..8).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| Arc::<str>::from(format!("cluster-{i}"))).collect::<Vec<_>>(),
    );
    let context = ExecutionContext::for_tests(17);
    let result =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&context).unwrap();
    let estimate = result.panel_did.as_ref().unwrap();
    assert_eq!(estimate.effect, 2.0);
    assert!((estimate.standard_error - (16.0_f64 / 7.0).sqrt()).abs() < 1e-12);
    assert_eq!(estimate.treated_subjects, 4);
    assert_eq!(estimate.comparison_subjects, 4);
    assert_eq!(estimate.clusters, 8);
    assert_eq!(estimate.uncertainty.as_ref(), "cluster_robust_standard_error_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    for required in ["parallel_trends", "repeated_cross_section"] {
        assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
            &record.assumption, Assumption::Custom { id, .. } if id.as_ref() == required
        )));
    }
    assert!(result.diagnostics.iter().any(|diagnostic|
        diagnostic.code.as_ref() == "identification.quasi.parallel_trends_untestable_two_periods"
    ));
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    let refreshed = prepared.estimate(&data, &context).unwrap();
    assert_eq!(refreshed.panel_did, result.panel_did);
}

#[test]
fn repeated_cross_section_refuses_duplicate_subject_or_sparse_cell_clusters() {
    let outcome = [0.0, 2.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0];
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let make = |subjects: Vec<Arc<str>>, clusters: Vec<Arc<str>>| {
        PanelDidQuery::repeated_cross_section(
            VariableId::from_raw(0),
            [false, false, false, false, true, true, true, true],
            [false, false, true, true, false, false, true, true],
            subjects,
            clusters,
        )
    };
    let ids = (0..8).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect::<Vec<_>>();
    let clusters = (0..8).map(|i| Arc::<str>::from(format!("cluster-{i}"))).collect::<Vec<_>>();
    let context = ExecutionContext::for_tests(17);
    let mut repeated_ids = ids.clone();
    repeated_ids[1] = repeated_ids[0].clone();
    assert!(
        Study::tabular(data.clone())
            .query(make(repeated_ids, clusters.clone()))
            .build()
            .unwrap()
            .run(&context)
            .is_err()
    );
    let mut sparse_clusters = clusters;
    sparse_clusters[1] = sparse_clusters[0].clone();
    assert!(
        Study::tabular(data)
            .query(make(ids, sparse_clusters))
            .build()
            .unwrap()
            .run(&context)
            .is_err()
    );
}

#[test]
fn repeated_cross_section_refuses_cluster_shared_across_treatment_groups() {
    let outcome = [0.0, 2.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0];
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let mut clusters = (0..8).map(|i| Arc::<str>::from(format!("cluster-{i}"))).collect::<Vec<_>>();
    clusters[4] = clusters[0].clone();
    let query = PanelDidQuery::repeated_cross_section(
        VariableId::from_raw(0),
        [false, false, false, false, true, true, true, true],
        [false, false, true, true, false, false, true, true],
        (0..8).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect::<Vec<_>>(),
        clusters,
    );
    let error = Study::tabular(data).query(query).build().unwrap()
        .run(&ExecutionContext::for_tests(1)).unwrap_err();
    assert!(error.to_string().contains("clusters nested within treatment groups"));
}

#[test]
fn staggered_group_time_runs_as_retained_native_study() {
    let mut outcome = Vec::new();
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut periods = Vec::new();
    let mut cohorts = Vec::new();
    for subject in 0..8 {
        for period in 1..=4 {
            let cohort = if subject < 4 { 0 } else { 3 };
            subjects.push(Arc::<str>::from(format!("s{subject}")));
            clusters.push(Arc::<str>::from(format!("c{subject}")));
            periods.push(period);
            cohorts.push(cohort);
            outcome.push(
                subject as f64
                    + 2.0 * period as f64
                    + if cohort == 3 && period >= 3 { 4.0 } else { 0.0 },
            );
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::staggered_group_time(
        VariableId::from_raw(0),
        subjects,
        clusters,
        periods,
        cohorts,
        3,
        4,
    );
    let context = ExecutionContext::for_tests(7);
    let result =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&context).unwrap();
    let estimate = result.panel_did.as_ref().unwrap();
    assert_eq!(estimate.effect, 4.0);
    assert_eq!(estimate.standard_error, 0.0);
    assert_eq!(estimate.treated_subjects, 4);
    assert_eq!(estimate.comparison_subjects, 4);
    assert_eq!(estimate.uncertainty.as_ref(), "cluster_robust_standard_error_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    for required in
        ["cohort_specific_parallel_untreated_trends", "never_treated_controls_are_valid"]
    {
        assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
            &record.assumption, Assumption::Custom { id, .. } if id.as_ref() == required
        )));
    }
    let prepared = Study::tabular(data.clone())
        .query(query.clone())
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    assert_eq!(prepared.estimate(&data, &context).unwrap().panel_did, result.panel_did);

    let no_controls = PanelDidQuery::staggered_group_time(
        query.outcome,
        query.subjects.to_vec(),
        query.clusters.to_vec(),
        query.periods.to_vec(),
        vec![3; query.cohorts.len()],
        3,
        4,
    );
    assert!(Study::tabular(data).query(no_controls).build().unwrap().run(&context).is_err());
}

#[test]
fn staggered_event_study_retains_full_curve_and_refuses_missing_controls() {
    let mut outcome = Vec::new();
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut periods = Vec::new();
    let mut cohorts = Vec::new();
    for subject in 0..8 {
        for period in 1..=4 {
            let cohort = if subject < 4 { 0 } else { 3 };
            subjects.push(Arc::<str>::from(format!("s{subject}")));
            clusters.push(Arc::<str>::from(format!("c{subject}")));
            periods.push(period);
            cohorts.push(cohort);
            outcome.push(subject as f64 + 2.0 * period as f64
                + if cohort == 3 && period >= 3 { 4.0 } else { 0.0 });
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0), subjects, clusters, periods, cohorts,
    );
    let context = ExecutionContext::for_tests(7);
    let result = Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&context).unwrap();
    let panel = result.panel_did.as_ref().unwrap();
    assert_eq!(panel.event_time_effects.len(), 3);
    assert_eq!(panel.event_time_effects.iter().map(|effect| effect.event_time).collect::<Vec<_>>(), vec![-2, 0, 1]);
    assert_eq!(panel.event_time_effects.iter().map(|effect| effect.effect).collect::<Vec<_>>(), vec![0.0, 4.0, 4.0]);
    assert_eq!(panel.event_time_effects[1].standard_error, 0.0);
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, Assumption::Custom { id, .. } if id.as_ref() == "cohort_specific_parallel_untreated_trends"
    )));
    assert!(result.diagnostics.iter().any(|diagnostic|
        diagnostic.code.as_ref() == "diagnostic.quasi.event_study.pretrend_joint_unavailable"
        && diagnostic.message.contains("at least two non-reference")));
    let prepared = Study::tabular(data.clone()).query(query.clone()).build().unwrap().prepare(&context).unwrap();
    let retained = prepared.estimate(&data, &context).unwrap();
    assert_eq!(retained.panel_did, result.panel_did);
    let bytes = prepared.encode_contracted_result(&retained, "staggered-event-study", &context).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.panel_did.as_ref().unwrap().event_time_effects.len(), 3);
    assert!(body.interval_lower.is_none() && body.interval_upper.is_none());
    let mut fabricated = body.clone();
    fabricated.panel_did.as_mut().unwrap().event_time_effects[1].6 = f64::INFINITY;
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
    let no_controls = PanelDidQuery::staggered_event_study(
        query.outcome, query.subjects.to_vec(), query.clusters.to_vec(),
        query.periods.to_vec(), vec![3; query.cohorts.len()],
    );
    assert!(Study::tabular(data).query(no_controls).build().unwrap().run(&context).is_err());
}

#[test]
fn staggered_event_joint_pretrend_statistic_is_descriptive_and_artifact_retained() {
    let mut outcome = Vec::new();
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut periods = Vec::new();
    let mut cohorts = Vec::new();
    for (group, cohort) in [("control", 0), ("treated", 4)] {
        for cluster in 0..8 {
            let id = Arc::<str>::from(format!("{group}-{cluster}"));
            for period in 1..=5 {
                subjects.push(id.clone());
                clusters.push(id.clone());
                periods.push(period);
                cohorts.push(cohort);
                let noise = (cluster as f64 - 3.5) * ((period % 3) as f64 - 1.0) / 10.0;
                outcome.push(2.0 * period as f64 + noise
                    + if cohort == 4 && period == 1 { 2.0 } else { 0.0 }
                    + if cohort == 4 && period >= 4 { 3.0 } else { 0.0 });
            }
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0), subjects, clusters, periods, cohorts,
    );
    let context = ExecutionContext::for_tests(47);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    let result = prepared.estimate(&data, &context).unwrap();
    let diagnostic = result.diagnostics.iter().find(|diagnostic|
        diagnostic.code.as_ref() == "diagnostic.quasi.event_study.pretrend_joint_max_cluster_z"
    ).expect("supported joint pretrend diagnostic");
    assert!(diagnostic.message.contains("across 2 non-reference"));
    assert!(diagnostic.message.contains("no calibrated p-value or cutoff"));
    assert!(diagnostic.message.contains("cannot establish parallel untreated trends"));
    let bytes = prepared.encode_contracted_result(&result, "joint-pretrend", &context).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(body.diagnostics.iter().any(|wire| wire.code == diagnostic.code.as_ref()
        && wire.message == diagnostic.message.as_ref()));
}

fn staggered_interval_fixture(clusters_per_group: usize) -> (TabularData, PanelDidQuery) {
    let mut outcome = Vec::new();
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut periods = Vec::new();
    let mut cohorts = Vec::new();
    for (group, cohort) in [("control", 0), ("treated", 3)] {
        for cluster in 0..clusters_per_group {
            let id = Arc::<str>::from(format!("{group}-{cluster}"));
            for period in 1..=4 {
                subjects.push(id.clone());
                clusters.push(id.clone());
                periods.push(period);
                cohorts.push(cohort);
                let noise = ((cluster % 7) as f64 - 3.0) * ((period % 3) as f64 - 1.0) / 10.0;
                outcome.push(5.0 + 2.0 * period as f64 + noise
                    + if cohort == 3 && period >= 3 { 4.0 } else { 0.0 });
            }
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0), subjects, clusters, periods, cohorts,
    );
    (data, query)
}

#[test]
fn staggered_event_known_truth_fixture_spans_thin_and_supported_clusters() {
    for clusters_per_group in [8, 24] {
        let (data, query) = staggered_interval_fixture(clusters_per_group);
        let context = ExecutionContext::for_tests(111);
        let study = Study::tabular(data.clone()).query(query).build().unwrap();
        let prepared = study.prepare(&context).unwrap();
        let result = prepared.estimate(&data, &context).unwrap();
        let panel = result.panel_did.as_ref().unwrap();
        assert_eq!(panel.event_time_effects.len(), 3);
        for (effect, truth) in panel.event_time_effects.iter().zip([0.0, 4.0, 4.0]) {
            assert!((effect.effect - truth).abs() < 1e-10);
            assert!(effect.standard_error > 0.0);
            assert_eq!(effect.clusters, 2 * clusters_per_group);
            assert_eq!(effect.treated_subjects, clusters_per_group);
            assert_eq!(effect.comparison_subjects, clusters_per_group);
        }
        assert_eq!(panel.event_time_intervals_95.len(), 3);
        assert_eq!(panel.event_time_intervals_95[0], None);
        if clusters_per_group == 24 {
            for (bounds, truth) in panel.event_time_intervals_95[1..].iter().zip([4.0, 4.0]) {
                let bounds = bounds.expect("calibrated post-adoption interval");
                assert!(bounds[0] < truth && truth < bounds[1]);
            }
            assert_eq!(panel.interval_95, panel.event_time_intervals_95[1]);
            assert_eq!(panel.uncertainty.as_ref(), "event_time_pointwise_normal_intervals_independent_clusters");
            assert!(result.estimate.as_effect().unwrap().se_analytic > 0.0);
            assert_ne!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
        } else {
            assert!(panel.event_time_intervals_95.iter().all(Option::is_none));
            assert_eq!(panel.interval_95, None);
            assert_eq!(panel.uncertainty.as_ref(), "cluster_robust_standard_error_no_interval");
            assert!(result.estimate.as_effect().unwrap().se_analytic.is_nan());
        }
        let bytes = prepared.encode_contracted_result(&result, "staggered-event-interval", &context).unwrap();
        let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(body.panel_did.as_ref().unwrap().event_time_intervals_95.len(), 3);
        if clusters_per_group == 24 {
            body.panel_did.as_mut().unwrap().event_time_intervals_95[1] = Some([0.0, 0.0]);
            assert!(antecedent_io::encode_analysis_result_artifact(
                &body, header.variable_names, "forged-staggered-interval").is_err());
        }
    }
}
