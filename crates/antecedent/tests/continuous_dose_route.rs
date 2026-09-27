//! Retained conditional continuous-dose response evidence.

use antecedent::Study;
use antecedent::prelude::ExecutionContext;
use antecedent_core::{CausalQuery, ContinuousDoseResponseQuery, VariableId};
use antecedent_data::TabularData;
use std::sync::Arc;

fn fixture() -> (TabularData, ContinuousDoseResponseQuery) {
    let doses = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let mut outcome = Vec::new();
    let mut dose = Vec::new();
    for intercept in [1.0, 10.0] {
        for value in doses {
            outcome.push(intercept + 2.0 * value);
            dose.push(value);
        }
    }
    let density = [0.5; 10];
    let data = TabularData::from_f64_columns([
        ("y", outcome.as_slice()),
        ("dose", dose.as_slice()),
        ("density", &density[..]),
    ])
    .unwrap();
    let groups: Vec<Arc<str>> = ["control", "treated"]
        .into_iter()
        .flat_map(|group| std::iter::repeat_n(Arc::<str>::from(group), 5))
        .collect();
    let query = ContinuousDoseResponseQuery {
        outcome: VariableId::from_raw(0),
        dose: VariableId::from_raw(1),
        dose_density: VariableId::from_raw(2),
        baseline_groups: groups.into(),
        target_doses: Arc::from([0.0, 0.5]),
        bandwidth: 0.6,
        min_local_support: 3,
        density_provenance: Arc::from("known"),
    };
    (data, query)
}

#[test]
fn retained_grid_recovers_linear_truth_and_artifact_identity() {
    let (data, query) = fixture();
    let study = Study::tabular(data.clone())
        .query(CausalQuery::ContinuousDoseResponse(query.clone()))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(613);
    let mut prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let grid = result.continuous_dose_response.as_ref().unwrap();
    assert_eq!(grid.points.len(), 4);
    for (point, expected) in grid.points.iter().zip([1.0, 2.0, 10.0, 11.0]) {
        assert!((point.response - expected).abs() < 1e-12);
    }
    assert!(grid.points.iter().all(|point| point.local_rows == 3));
    assert_eq!(grid.uncertainty.as_ref(), "point_only_no_interval");
    assert!(result.identification.required_assumptions.entries.iter().any(|entry| matches!(&entry.assumption,
        antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "conditional_dose_exchangeability")));
    let bytes = prepared.encode_contracted_result(&result, "continuous-dose", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(
        body.interval_lower.is_none() && body.standard_error.is_none() && body.estimate.is_none()
    );
    assert_eq!(
        body.query,
        antecedent_io::causal_query_to_wire(&CausalQuery::ContinuousDoseResponse(query)).unwrap()
    );
    assert!(
        (body.continuous_dose_response.as_ref().unwrap().points[3].response - 11.0).abs() < 1e-12
    );
    let mut fabricated = body.clone();
    fabricated.interval_lower = Some(0.0);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(result.continuous_dose_response, refreshed.continuous_dose_response);
}

#[test]
fn retained_grid_refuses_missing_support_and_invalid_density() {
    let (data, mut query) = fixture();
    query.target_doses = Arc::from([4.0]);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::ContinuousDoseResponse(query))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(614);
    let prepared = study.prepare(&ctx).unwrap();
    assert!(prepared.estimate(&data, &ctx).is_err());
    let mut bad_density = [0.5; 10];
    bad_density[0] = 0.0;
    let (data, query) = fixture();
    let invalid = TabularData::from_f64_columns([
        ("y", &[-1.0, 0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 12.0][..]),
        ("dose", &[-1.0, -0.5, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5, 1.0][..]),
        ("density", &bad_density[..]),
    ])
    .unwrap();
    let study =
        Study::tabular(data).query(CausalQuery::ContinuousDoseResponse(query)).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    assert!(prepared.estimate(&invalid, &ctx).is_err());
}
