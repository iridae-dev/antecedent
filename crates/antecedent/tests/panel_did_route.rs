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
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, Assumption::Custom { id, .. } if id.as_ref() == "parallel_trends"
    )));
    assert!(!result.diagnostics.is_empty());

    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    let refreshed = prepared.estimate(&data, &context).unwrap();
    assert_eq!(refreshed.panel_did, result.panel_did);
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
    for required in ["parallel_trends", "repeated_cross_section"] {
        assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
            &record.assumption, Assumption::Custom { id, .. } if id.as_ref() == required
        )));
    }
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
