//! Retained fixed continuous-dose policy value and artifact identity.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::Study;
use antecedent_core::{CausalQuery, ContinuousDoseResponseQuery, FixedGroupDosePolicy, VariableId};
use antecedent_data::TabularData;

fn fixture(provenance: &str, rows: usize) -> (TabularData, ContinuousDoseResponseQuery) {
    let dose = (0..rows).map(|i| ((i * 37) % 997) as f64 / 997.0).collect::<Vec<_>>();
    let outcome = (0..rows).map(|i| 1.0 + f64::from(i >= rows / 2)
        + 0.35 * (0.13 * i as f64).sin()
        + if i < rows / 2 { 1.5 * dose[i] } else { 2.0 * dose[i] }).collect::<Vec<_>>();
    let density = vec![1.0; rows];
    let data = TabularData::from_f64_columns([
        ("outcome", outcome.as_slice()), ("dose", dose.as_slice()),
        ("density", density.as_slice()),
    ]).unwrap();
    let query = ContinuousDoseResponseQuery {
        outcome: VariableId::from_raw(0), dose: VariableId::from_raw(1),
        dose_density: VariableId::from_raw(2),
        baseline_groups: (0..rows).map(|i| Arc::<str>::from(if i < rows / 2 { "a" } else { "b" }))
            .collect::<Vec<_>>().into(),
        target_doses: Arc::from([]), bandwidth: 0.2, min_local_support: 80,
        density_provenance: Arc::from(provenance),
        fixed_policy: Some(FixedGroupDosePolicy {
            policy_doses: Arc::from([(Arc::from("a"), 0.7), (Arc::from("b"), 0.8)]),
            reference_doses: Arc::from([(Arc::from("a"), 0.3), (Arc::from("b"), 0.4)]),
        }),
    };
    (data, query)
}

#[test]
fn retained_fixed_dose_policy_intervals_round_trip_and_refuse_tampering() {
    let (data, query) = fixture("known", 800);
    let ctx = ExecutionContext::for_tests(0xD05E);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::ContinuousDoseResponse(query))
        .build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let response = result.continuous_dose_response.as_ref().unwrap();
    assert!(response.points.is_empty());
    let value = response.fixed_policy.as_ref().unwrap();
    assert!((value.incremental_value - 0.7).abs() < 0.15);
    assert!(value.policy_interval_95.is_some());
    assert!(value.reference_interval_95.is_some());
    assert!(value.incremental_interval_95.is_some());
    assert_eq!(response.uncertainty.as_ref(),
        "fixed_group_kernel_smoothed_paired_pointwise_95_normal_intervals");
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "fixed_group_dose_policy"
    )));
    let bytes = prepared.encode_contracted_result(&result, "fixed-dose-policy", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let mut forged = artifact.clone();
    forged.continuous_dose_response.as_mut().unwrap().fixed_policy.as_mut().unwrap()
        .incremental_interval_95.as_mut().unwrap()[1] += 0.5;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-dose-interval").is_err());
    let mut forged = artifact;
    forged.continuous_dose_response.as_mut().unwrap().fixed_policy.as_mut().unwrap()
        .policy_doses[0].1 = 0.1;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names, "forged-dose-rule").is_err());
}

#[test]
fn externally_estimated_density_keeps_fixed_dose_value_point_only() {
    let (data, query) = fixture("externally_estimated", 800);
    let ctx = ExecutionContext::for_tests(0xD05E);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::ContinuousDoseResponse(query))
        .build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.continuous_dose_response.as_ref().unwrap().fixed_policy.as_ref().unwrap();
    assert!(value.incremental_interval_95.is_none());
    assert!(value.policy_interval_95.is_none());
    assert!(value.reference_interval_95.is_none());
    let bytes = prepared.encode_contracted_result(&result, "external-dose-density", &ctx).unwrap();
    let (_, _, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(artifact.continuous_dose_response.unwrap().fixed_policy.unwrap()
        .incremental_interval_95.is_none());
}
