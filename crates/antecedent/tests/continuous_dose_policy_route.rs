//! Retained fixed continuous-dose policy value and artifact identity.

use std::sync::Arc;

use antecedent::Study;
use antecedent::prelude::ExecutionContext;
use antecedent_core::{CausalQuery, ContinuousDoseResponseQuery, FixedGroupDosePolicy, VariableId};
use antecedent_data::TabularData;

fn fixture(provenance: &str, rows: usize) -> (TabularData, ContinuousDoseResponseQuery) {
    let dose = (0..rows).map(|i| ((i * 37) % 997) as f64 / 997.0).collect::<Vec<_>>();
    let outcome = (0..rows)
        .map(|i| {
            1.0 + f64::from(i >= rows / 2)
                + 0.35 * (0.13 * i as f64).sin()
                + if i < rows / 2 { 1.5 * dose[i] } else { 2.0 * dose[i] }
        })
        .collect::<Vec<_>>();
    let density = vec![1.0; rows];
    let data = TabularData::from_f64_columns([
        ("outcome", outcome.as_slice()),
        ("dose", dose.as_slice()),
        ("density", density.as_slice()),
    ])
    .unwrap();
    let query = ContinuousDoseResponseQuery {
        outcome: VariableId::from_raw(0),
        dose: VariableId::from_raw(1),
        dose_density: VariableId::from_raw(2),
        baseline_groups: (0..rows)
            .map(|i| Arc::<str>::from(if i < rows / 2 { "a" } else { "b" }))
            .collect::<Vec<_>>()
            .into(),
        target_doses: Arc::from([]),
        bandwidth: 0.2,
        min_local_support: 80,
        density_provenance: Arc::from(provenance),
        fixed_policy: Some(FixedGroupDosePolicy {
            policy_doses: Arc::from([(Arc::from("a"), 0.7), (Arc::from("b"), 0.8)]),
            reference_doses: Arc::from([(Arc::from("a"), 0.3), (Arc::from("b"), 0.4)]),
        }),
    };
    (data, query)
}

#[test]
#[allow(clippy::too_many_lines)] // one end-to-end route fixture; splitting would fragment the flow
fn retained_fixed_dose_policy_intervals_round_trip_and_refuse_tampering() {
    let (data, query) = fixture("known", 800);
    let ctx = ExecutionContext::for_tests(0xD05E);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::ContinuousDoseResponse(query))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let response = result.continuous_dose_response.as_ref().unwrap();
    assert!(response.points.is_empty());
    let value = response.fixed_policy.as_ref().unwrap();
    assert!((value.incremental_value - 0.7).abs() < 0.15);
    assert!(value.policy_interval_95.is_some());
    assert!(value.reference_interval_95.is_some());
    assert!(value.incremental_interval_95.is_some());
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    assert_eq!(
        response.uncertainty.as_ref(),
        "fixed_group_kernel_smoothed_paired_pointwise_95_normal_intervals"
    );
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "fixed_group_dose_policy"
    )));
    let executed_contract = prepared.contract_for_result(&result).unwrap();
    assert_eq!(executed_contract.support_status, result.support_status);
    let support = executed_contract.reasoning.support.as_ref().unwrap();
    assert_eq!(support.matrix_status.as_ref(), "licensed");
    assert_eq!(
        support.matrix_coordinate.as_deref(),
        Some(
            "graphless:continuous_dose_policy/fixed_group_kernel/inverse_density_kernel_paired_scores/policy_reference_incremental_pointwise_95_normal_intervals"
        )
    );
    let bytes = prepared.encode_contracted_result(&result, "fixed-dose-policy", &ctx).unwrap();
    let (encoded, header, artifact) =
        antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(
        artifact.continuous_dose_response.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("licensed")
    );
    let mut legacy_body = artifact.clone();
    legacy_body.continuous_dose_response.as_mut().unwrap().graphless_support_status = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &legacy_body,
            header.variable_names.clone(),
            "legacy-new-encode"
        )
        .is_err()
    );
    let mut legacy_container = encoded.clone();
    let body_index = legacy_container
        .sections
        .iter()
        .position(|section| section.id == "analysis_result.body")
        .unwrap();
    let body_bytes = antecedent_io::to_cbor(&legacy_body).unwrap();
    let (descriptor, section) = antecedent_io::pack_section_shared(
        "analysis_result.body",
        "application/cbor",
        body_bytes.into(),
        antecedent_io::CompressPolicy::Auto,
    );
    legacy_container.sections[body_index] = section;
    legacy_container.manifest.sections[body_index] = descriptor;
    let mut old_bytes = Vec::new();
    legacy_container.write_to(&mut old_bytes).unwrap();
    let (_, _, old_body) = antecedent_io::decode_analysis_result_artifact(&old_bytes).unwrap();
    assert!(old_body.continuous_dose_response.unwrap().graphless_support_status.is_none());
    let mut forged = artifact.clone();
    forged.continuous_dose_response.as_mut().unwrap().graphless_support_status =
        Some("refused".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-dose-support"
        )
        .is_err()
    );
    let mut forged = artifact.clone();
    forged
        .continuous_dose_response
        .as_mut()
        .unwrap()
        .fixed_policy
        .as_mut()
        .unwrap()
        .reference_interval_95 = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "omitted-dose-interval"
        )
        .is_err()
    );
    let mut forged = artifact.clone();
    forged
        .continuous_dose_response
        .as_mut()
        .unwrap()
        .fixed_policy
        .as_mut()
        .unwrap()
        .incremental_interval_95
        .as_mut()
        .unwrap()[1] += 0.5;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-dose-interval"
        )
        .is_err()
    );
    let mut forged = artifact;
    forged.continuous_dose_response.as_mut().unwrap().fixed_policy.as_mut().unwrap().policy_doses
        [0]
    .1 = 0.1;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-dose-rule"
        )
        .is_err()
    );
}

#[test]
fn externally_estimated_density_keeps_fixed_dose_value_point_only() {
    let (data, query) = fixture("externally_estimated", 800);
    let ctx = ExecutionContext::for_tests(0xD05E);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::ContinuousDoseResponse(query))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.support_status, None);
    let value = result.continuous_dose_response.as_ref().unwrap().fixed_policy.as_ref().unwrap();
    assert!(value.incremental_interval_95.is_none());
    assert!(value.policy_interval_95.is_none());
    assert!(value.reference_interval_95.is_none());
    let bytes = prepared.encode_contracted_result(&result, "external-dose-density", &ctx).unwrap();
    let (_, _, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(artifact.continuous_dose_response.as_ref().unwrap().graphless_support_status.is_none());
    assert!(
        artifact
            .continuous_dose_response
            .unwrap()
            .fixed_policy
            .unwrap()
            .incremental_interval_95
            .is_none()
    );
}
