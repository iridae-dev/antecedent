//! End-to-end retained randomized survival and competing-risk execution.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{
    CausalQuery, KnownCensoringSurvival, ObservationAssumption, SurvivalFunctional, SurvivalQuery, VariableId,
};
use antecedent_data::TabularData;

fn query(functional: SurvivalFunctional) -> SurvivalQuery {
    SurvivalQuery {
        duration: VariableId::from_raw(0),
        event: VariableId::from_raw(1),
        treatment: VariableId::from_raw(2),
        tau: 2.0,
        delayed_entry: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional,
    }
}

#[test]
fn known_censoring_survival_is_retained_and_refuses_positivity_violation() {
    let duration = [1.0, 3.0, 1.0, 3.0];
    let event = [1.0, 0.0, 1.0, 0.0];
    let treatment = [0.0, 0.0, 1.0, 1.0];
    let g0 = [1.0; 4];
    let g1 = [1.0, 0.5, 0.5, 0.5];
    let g3 = g1;
    let data = TabularData::from_f64_columns([
        ("duration", &duration[..]), ("event", &event[..]),
        ("treatment", &treatment[..]), ("g0", &g0[..]),
        ("g1", &g1[..]), ("g3", &g3[..]),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.known_censoring = Some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 3.0]),
        columns: Arc::from([VariableId::from_raw(3), VariableId::from_raw(4), VariableId::from_raw(5)]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(88);
    let study = Study::tabular(data.clone()).query(CausalQuery::Survival(q.clone())).build().unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival.as_ref().unwrap();
    assert!((curve.control[1] - 2.0 / 3.0).abs() < 1e-12);
    assert!((curve.treated[1] - 0.5).abs() < 1e-12);
    assert_eq!(curve.uncertainty.as_ref(), "point_only_no_interval");
    let encoded = prepared.encode_contracted_result(&result, "ipcw-survival", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert!(body.interval_lower.is_none());
    assert_eq!(body.survival.as_ref().unwrap().censoring_survival_provenance.as_deref(),
        Some("caller_supplied_fixed_not_fitted_or_verified"));
    assert!(matches!(body.query, antecedent_io::CausalQueryWire::Survival(ref wire) if wire.censoring_columns.len() == 3));
    let mut tampered = body.clone();
    if let antecedent_io::CausalQueryWire::Survival(ref mut wire) = tampered.query {
        wire.censoring_probability_floor = None;
    }
    assert!(antecedent_io::encode_analysis_result_artifact(
        &tampered, vec!["duration".into(), "event".into(), "treatment".into(), "g0".into(), "g1".into(), "g3".into()], "tampered",
    ).is_err());
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().censoring_survival_provenance = None;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, vec!["duration".into(), "event".into(), "treatment".into(), "g0".into(), "g1".into(), "g3".into()], "forged",
    ).is_err());
    let bad_g1 = [1.0, 0.001, 0.5, 0.5];
    let bad = TabularData::from_f64_columns([
        ("duration", &duration[..]), ("event", &event[..]),
        ("treatment", &treatment[..]), ("g0", &g0[..]),
        ("g1", &bad_g1[..]), ("g3", &g3[..]),
    ]).unwrap();
    assert!(prepared.refresh(bad, &ctx).is_err());
}

#[test]
fn randomized_survival_rmst_is_retained_and_round_trips() {
    let duration = [1.0, 2.0, 2.0, 2.0];
    let event = [1.0, 0.0, 0.0, 0.0];
    let treatment = [0.0, 0.0, 1.0, 1.0];
    let data = TabularData::from_f64_columns([
        ("duration", &duration[..]),
        ("event", &event[..]),
        ("treatment", &treatment[..]),
    ])
    .unwrap();
    let study = Study::tabular(data.clone())
        .query(CausalQuery::Survival(query(SurvivalFunctional::SurvivalAndRmst)))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(86);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let curve = first.survival.as_ref().unwrap();
    assert_eq!(&*curve.times, &[0.0, 1.0, 2.0]);
    assert_eq!(&*curve.control, &[1.0, 0.5, 0.5]);
    assert_eq!(&*curve.treated, &[1.0, 1.0, 1.0]);
    assert_eq!(curve.rmst_control, Some(1.5));
    assert_eq!(curve.rmst_treated, Some(2.0));
    assert_eq!(&*curve.uncertainty, "point_only_no_interval");
    assert!(matches!(first.estimate, PrimaryEstimate::NotAnEffect));
    let bytes = prepared.encode_contracted_result(&first, "survival", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.survival.as_ref().unwrap();
    assert_eq!(wire.rmst_control, Some(1.5));
    assert_eq!(wire.uncertainty, "point_only_no_interval");
    assert!(body.estimate.is_none());
    assert!(body.standard_error.is_none());
    assert!(body.interval_lower.is_none());
    assert!(body.interval_upper.is_none());
    let mut fabricated = body.clone();
    fabricated.standard_error = Some(0.1);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
    assert!(first.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "independent_censoring_and_entry")
    }));
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(first.survival, refreshed.survival);
}

#[test]
fn competing_risk_incidence_and_unsupported_observation_contract() {
    let duration = [1.0, 2.0, 1.0, 2.0];
    let event = [1.0, 2.0, 1.0, 1.0];
    let treatment = [0.0, 0.0, 1.0, 1.0];
    let data = TabularData::from_f64_columns([
        ("duration", &duration[..]),
        ("event", &event[..]),
        ("treatment", &treatment[..]),
    ])
    .unwrap();
    let q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    let study =
        Study::tabular(data.clone()).query(CausalQuery::Survival(q.clone())).build().unwrap();
    let ctx = ExecutionContext::for_tests(87);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival.as_ref().unwrap();
    assert_eq!(&*curve.control, &[0.0, 0.5, 0.5]);
    assert_eq!(&*curve.treated, &[0.0, 0.5, 1.0]);
    assert_eq!(curve.rmst_control, None);
    assert_eq!(curve.target_cause, Some(1));

    let mut invalid = q;
    invalid.observation_assumption =
        ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(0)]));
    let unsupported = Study::tabular(data).query(CausalQuery::Survival(invalid)).build().unwrap();
    assert!(unsupported.prepare(&ctx).is_err());
}

#[test]
fn subject_bootstrap_scalar_intervals_are_retained_and_artifact_checked() {
    let n = 40;
    let duration = (0..n).map(|i| if i % 5 == 0 { 1.0 } else { 3.0 })
        .chain((0..n).map(|i| if i % 10 == 0 { 1.0 } else { 3.0 }))
        .collect::<Vec<_>>();
    let event = duration.iter().map(|&time| if time == 1.0 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let treatment = [vec![0.0; n], vec![1.0; n]].concat();
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let study = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(399).build().unwrap();
    let ctx = ExecutionContext::for_tests(17);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let section = first.survival.as_ref().unwrap();
    assert_eq!(section.uncertainty.as_ref(), "subject_stratified_percentile_bootstrap_pointwise_95");
    assert_eq!(section.bootstrap_replicates_requested, Some(399));
    assert_eq!(section.bootstrap_replicates_ok, Some(399));
    let rmst = section.rmst_difference_interval.unwrap();
    assert!(rmst[0] <= 0.2 && 0.2 <= rmst[1]);
    assert_eq!(section.difference_at_tau_interval.unwrap().len(), 2);
    assert_eq!(prepared.refresh(data, &ctx).unwrap().survival, first.survival);
    let artifact = prepared.encode_contracted_result(&first, "survival-bootstrap", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&artifact).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().rmst_difference_interval, Some(rmst));
    let mut tampered = body;
    tampered.survival.as_mut().unwrap().uncertainty = "point_only_no_interval".into();
    assert!(antecedent_io::encode_analysis_result_artifact(&tampered, header.variable_names, "tampered").is_err());
}

#[test]
fn delayed_entry_subject_bootstrap_retains_interval_and_refuses_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..60 {
            let left = if i % 3 == 0 { 1.0 } else { 0.0 };
            let exit = if left == 0.0 && i % 7 == 0 { 1.0 }
                else if i % (if arm { 6 } else { 5 }) == 0 { 2.0 }
                else { 3.0 };
            duration.push(exit);
            event.push(if exit < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()), ("entry", entry.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let ctx = ExecutionContext::for_tests(279);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(299).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival.as_ref().unwrap();
    assert_eq!(curve.uncertainty.as_ref(), "subject_stratified_percentile_bootstrap_pointwise_95");
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert_eq!(curve.bootstrap_replicates_ok, Some(299));
    let encoded = prepared.encode_contracted_result(&result, "delayed-entry-bootstrap", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().rmst_difference_interval, curve.rmst_difference_interval);
    let mut forged = body;
    forged.survival.as_mut().unwrap().uncertainty = "point_only_no_interval".into();
    assert!(antecedent_io::encode_analysis_result_artifact(&forged, header.variable_names, "forged").is_err());
}
