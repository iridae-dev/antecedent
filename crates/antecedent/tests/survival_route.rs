//! End-to-end retained randomized survival and competing-risk execution.
#![allow(clippy::float_cmp, clippy::redundant_locals, reason = "integration test asserts exact deterministic estimates; a rebinding keeps the fixture readable")]

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
fn licensed_survival_scalar_intervals_round_trip_and_refuse_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..120 {
            let failed = i % (if arm { 7 } else { 3 }) == 0;
            duration.push(if failed { 1.0 } else { 3.0 });
            event.push(f64::from(failed));
            treatment.push(f64::from(arm));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(7_729);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q.clone()))
        .bootstrap_replicates(299).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let curve = result.survival.as_ref().unwrap();
    assert_eq!(curve.assignment_counts, [120, 120]);
    assert_eq!(curve.bootstrap_replicates_ok, Some(299));
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "licensed-survival", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().assignment_counts = Some([119, 121]);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-arm-counts",
    ).is_err());
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().bootstrap_replicates_ok = Some(298);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-draw-count",
    ).is_err());
    let mut q = q;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let entry = vec![0.0; 240];
    let with_entry = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()), ("entry", entry.as_slice()),
    ]).unwrap();
    let prepared = Study::tabular(with_entry.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(299).build().unwrap().prepare(&ctx).unwrap();
    let off_axis = prepared.estimate(&with_entry, &ctx).unwrap();
    assert!(off_axis.support_status.is_none());
    let encoded = prepared.encode_contracted_result(&off_axis, "entry-off-axis", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    body.survival.as_mut().unwrap().graphless_support_status = Some("licensed".into());
    assert!(antecedent_io::encode_analysis_result_artifact(
        &body, header.variable_names, "forged-entry-license",
    ).is_err());
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

#[test]
fn delayed_entry_fixed_censoring_competing_risk_interval_is_retained() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..80 {
            let left = if i % 3 == 0 { 1.0 } else { 0.0 };
            let (exit, cause) = if left == 0.0 && i % 7 == 0 { (1.0, 1.0) }
                else if i % 11 == 0 { (1.5, 0.0) }
                else if i % 5 == 0 { (2.0, 2.0) }
                else if i % (if arm { 4 } else { 3 }) == 0 { (2.0, 1.0) }
                else { (3.0, 0.0) };
            duration.push(exit);
            event.push(cause);
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let g0 = vec![1.0; duration.len()];
    let g2 = vec![0.8; duration.len()];
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()), ("entry", entry.as_slice()),
        ("g0", g0.as_slice()), ("g1", g0.as_slice()),
        ("g15", g0.as_slice()), ("g2", g2.as_slice()), ("g3", g2.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    q.known_censoring = Some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 1.5, 2.0, 3.0]),
        columns: Arc::from([
            VariableId::from_raw(4), VariableId::from_raw(5), VariableId::from_raw(6),
            VariableId::from_raw(7), VariableId::from_raw(8),
        ]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(297);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q.clone()))
        .bootstrap_replicates(299).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival.as_ref().unwrap();
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.rmst_difference_interval.is_none());
    assert_eq!(curve.censoring_survival_provenance.as_deref(),
        Some("caller_supplied_fixed_not_fitted_or_verified"));
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "delayed-entry-fixed-g", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().difference_at_tau_interval,
        curve.difference_at_tau_interval);
    let mut forged = body.clone();
    if let antecedent_io::CausalQueryWire::Survival(ref mut wire) = forged.query {
        wire.independent_given.push(VariableId::from_raw(2).raw());
    }
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-conditional-entry",
    ).is_err());
    q.observation_assumption = ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(2)]));
    assert!(Study::tabular(data).query(CausalQuery::Survival(q)).build().unwrap()
        .prepare(&ctx).is_err());
}

#[test]
fn simultaneous_curve_band_is_retained_and_artifact_checked() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..100 {
            let time = if i % (if arm { 4 } else { 5 }) == 0 { 1.0 }
                else if i % 7 == 0 { 2.0 } else { 3.0 };
            duration.push(time);
            event.push(if time < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(923);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(399).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival.as_ref().unwrap();
    let band = curve.difference_band.as_ref().expect("eligible randomized subjects");
    assert_eq!(band.times.as_ref(), curve.times.as_ref());
    assert_eq!(band.replicates_ok, 399);
    assert!(band.lower.iter().zip(band.difference.iter()).zip(band.upper.iter())
        .all(|((&lower, &difference), &upper)| lower <= difference && difference <= upper));
    assert!(curve.rmst_difference_interval.is_some());
    assert_eq!(result.survival, prepared.estimate(&data, &ctx).unwrap().survival);
    let bytes = prepared.encode_contracted_result(&result, "survival-band", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().difference_band.as_ref().unwrap().lower, band.lower.as_ref());
    let mut forged = body;
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap().difference[1] += 0.1;
    assert!(antecedent_io::encode_analysis_result_artifact(&forged, header.variable_names, "forged-band").is_err());
}

#[test]
fn simultaneous_band_withholds_small_arms_without_erasing_scalar_interval() {
    let n = 60;
    let duration = (0..2 * n).map(|i| if i % 5 == 0 { 1.0 } else { 3.0 }).collect::<Vec<_>>();
    let event = duration.iter().map(|&time| if time < 3.0 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let treatment = [vec![0.0; n], vec![1.0; n]].concat();
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(923);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(399).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival.as_ref().unwrap();
    assert!(curve.difference_band.is_none());
    assert!(curve.band_unavailable_reason.as_deref().unwrap().contains("80 subjects"));
    assert!(curve.rmst_difference_interval.is_some());
}

#[test]
fn licensed_delayed_entry_survival_intervals_round_trip_and_refuse_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..170 {
            let left = if i % 4 == 0 { 1.0 } else { 0.0 };
            let mut exit = if i % (if arm { 6 } else { 4 }) == 0 { 2.0 }
                else if i % 5 == 0 { 1.0 } else { 3.0 };
            if left >= exit { exit = 3.0; }
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
    let ctx = ExecutionContext::for_tests(5_161);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(320).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let curve = result.survival.as_ref().unwrap();
    assert_eq!(curve.assignment_counts, [170, 170]);
    assert!(curve.bootstrap_replicates_ok.unwrap() >= 299);
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "licensed-delayed-entry", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().assignment_counts = Some([120, 120]);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-thin-delayed-entry",
    ).is_err());
    let mut forged = body;
    forged.survival.as_mut().unwrap().uncertainty = "point_only_no_interval".into();
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names, "forged-delayed-entry-point",
    ).is_err());
}

#[test]
fn licensed_fixed_g_ipcw_survival_intervals_round_trip_and_refuse_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..170 {
            let time = if i % (if arm { 6 } else { 3 }) == 0 { 1.0 } else { 3.0 };
            duration.push(time);
            event.push(if time < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
        }
    }
    let g0 = vec![1.0; duration.len()];
    let g1 = vec![0.9; duration.len()];
    let g3 = vec![0.9; duration.len()];
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("g0", g0.as_slice()), ("g1", g1.as_slice()), ("g3", g3.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.known_censoring = Some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 3.0]),
        columns: Arc::from([
            VariableId::from_raw(3), VariableId::from_raw(4), VariableId::from_raw(5),
        ]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(5_162);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(320).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let curve = result.survival.as_ref().unwrap();
    assert_eq!(curve.assignment_counts, [170, 170]);
    assert_eq!(curve.censoring_survival_provenance.as_deref(),
        Some("caller_supplied_fixed_not_fitted_or_verified"));
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "licensed-fixed-g", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().graphless_support_status = None;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-drop-license",
    ).is_err());
    let mut forged = body;
    forged.survival.as_mut().unwrap().censoring_survival_provenance = None;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names, "forged-fixed-g-provenance",
    ).is_err());
}

#[test]
fn licensed_delayed_entry_fixed_g_survival_intervals_round_trip_and_refuse_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..210 {
            let left = if i % 4 == 0 { 1.0 } else { 0.0 };
            let mut exit = if i % (if arm { 6 } else { 3 }) == 0 { 2.0 }
                else if i % 9 == 0 { 1.5 } else { 3.0 };
            if left >= exit { exit = 3.0; }
            duration.push(exit);
            event.push(if exit < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let g0 = vec![1.0; duration.len()];
    let g_late = vec![0.85; duration.len()];
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()), ("entry", entry.as_slice()),
        ("g0", g0.as_slice()), ("g1", g0.as_slice()), ("g15", g0.as_slice()),
        ("g2", g_late.as_slice()), ("g3", g_late.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    q.known_censoring = Some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 1.5, 2.0, 3.0]),
        columns: Arc::from([
            VariableId::from_raw(4), VariableId::from_raw(5), VariableId::from_raw(6),
            VariableId::from_raw(7), VariableId::from_raw(8),
        ]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(5_163);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(320).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let curve = result.survival.as_ref().unwrap();
    assert_eq!(curve.assignment_counts, [210, 210]);
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "licensed-delayed-entry-fixed-g", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status.as_deref(), Some("licensed"));
    let mut forged = body;
    forged.survival.as_mut().unwrap().assignment_counts = Some([160, 160]);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names, "forged-combined-thin",
    ).is_err());
}

#[test]
fn licensed_survival_difference_band_round_trips_and_refuses_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..100 {
            let time = if i % (if arm { 4 } else { 5 }) == 0 { 1.0 }
                else if i % 7 == 0 { 2.0 } else { 3.0 };
            duration.push(time);
            event.push(if time < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(5_164);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(399).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    // Only 100 subjects per arm: the scalar row needs 120, so the scalar route
    // is off-axis while the band's own 80-per-arm license applies.
    assert!(result.support_status.is_none());
    let curve = result.survival.as_ref().unwrap();
    let band = curve.difference_band.as_ref().expect("eligible randomized subjects");
    assert_eq!(band.support_status, Some(antecedent::support::CellStatus::Licensed));
    let encoded = prepared.encode_contracted_result(&result, "licensed-survival-band", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status, None);
    assert_eq!(body.survival.as_ref().unwrap().difference_band.as_ref().unwrap()
        .graphless_support_status.as_deref(), Some("licensed"));
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap()
        .graphless_support_status = None;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names.clone(), "forged-drop-band-license",
    ).is_err());
    let mut forged = body;
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap().difference[1] += 0.1;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names, "forged-band-curve",
    ).is_err());
}

#[test]
fn licensed_competing_incidence_band_round_trips_and_refuses_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..100 {
            let (time, cause) = if i % (if arm { 3 } else { 4 }) == 0 { (1.0, 1.0) }
                else if i % 5 == 0 { (1.0, 2.0) }
                else if i % 7 == 0 { (2.0, 1.0) }
                else { (3.0, 0.0) };
            duration.push(time);
            event.push(cause);
            treatment.push(f64::from(arm));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(5_165);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(399).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    // Cumulative incidence never carries the scalar RMST+horizon license.
    assert!(result.support_status.is_none());
    let curve = result.survival.as_ref().unwrap();
    assert!(curve.rmst_difference_interval.is_none());
    assert!(curve.difference_at_tau_interval.is_some());
    let band = curve.difference_band.as_ref().expect("eligible randomized subjects");
    assert_eq!(band.support_status, Some(antecedent::support::CellStatus::Licensed));
    let encoded = prepared.encode_contracted_result(&result, "licensed-incidence-band", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status, None);
    assert_eq!(body.survival.as_ref().unwrap().difference_band.as_ref().unwrap()
        .graphless_support_status.as_deref(), Some("licensed"));
    let mut forged = body;
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap()
        .graphless_support_status = None;
    assert!(antecedent_io::encode_analysis_result_artifact(
        &forged, header.variable_names, "forged-drop-incidence-band-license",
    ).is_err());
}

#[test]
fn competing_incidence_scalar_interval_is_off_axis_and_refuses_forged_license() {
    // Cause-specific cumulative incidence has a single scalar and no RMST
    // companion, so the RMST-shaped graphless survival evidence gate cannot
    // certify it. The estimator still publishes the calibrated interval, but it
    // stays off the graphless axis. Delayed entry suppresses the band so the
    // scalar decision is isolated.
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..170 {
            let left = if i % 4 == 0 { 1.0 } else { 0.0 };
            let (mut exit, cause) = if i % (if arm { 5 } else { 3 }) == 0 { (2.0, 1.0) }
                else if i % 7 == 0 { (2.0, 2.0) }
                else { (3.0, 0.0) };
            if left >= exit { exit = 3.0; }
            duration.push(exit);
            event.push(if exit < 3.0 { cause } else { 0.0 });
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()), ("event", event.as_slice()),
        ("treatment", treatment.as_slice()), ("entry", entry.as_slice()),
    ]).unwrap();
    let mut q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let ctx = ExecutionContext::for_tests(5_166);
    let prepared = Study::tabular(data.clone()).query(CausalQuery::Survival(q))
        .bootstrap_replicates(320).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!(result.support_status.is_none());
    let curve = result.survival.as_ref().unwrap();
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.rmst_difference_interval.is_none());
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "incidence-off-axis", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status, None);
    body.survival.as_mut().unwrap().graphless_support_status = Some("licensed".into());
    assert!(antecedent_io::encode_analysis_result_artifact(
        &body, header.variable_names, "forged-incidence-license",
    ).is_err());
}
