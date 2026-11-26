//! End-to-end retained randomized survival and competing-risk execution.
#![allow(
    clippy::float_cmp,
    clippy::redundant_locals,
    reason = "integration test asserts exact deterministic estimates; a rebinding keeps the fixture readable"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{
    CausalQuery, KnownCensoringSurvival, ObservationAssumption, SurvivalFunctional, SurvivalQuery,
    VariableId,
};
use antecedent_data::TabularData;

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

fn query(functional: SurvivalFunctional) -> SurvivalQuery {
    SurvivalQuery {
        duration: VariableId::from_raw(0),
        event: VariableId::from_raw(1),
        treatment: VariableId::from_raw(2),
        tau: 2.0,
        delayed_entry: None,
        stratified_censoring: None,
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
        ("duration", &duration[..]),
        ("event", &event[..]),
        ("treatment", &treatment[..]),
        ("g0", &g0[..]),
        ("g1", &g1[..]),
        ("g3", &g3[..]),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.known_censoring = Some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 3.0]),
        columns: Arc::from([
            VariableId::from_raw(3),
            VariableId::from_raw(4),
            VariableId::from_raw(5),
        ]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(88);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::Survival(q.clone())).build().unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival().unwrap();
    assert!((curve.control[1] - 2.0 / 3.0).abs() < 1e-12);
    assert!((curve.treated[1] - 0.5).abs() < 1e-12);
    assert_eq!(curve.uncertainty.as_ref(), "point_only_no_interval");
    let encoded = prepared.encode_contracted_result(&result, "ipcw-survival", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert!(body.interval_lower.is_none());
    assert_eq!(
        body.survival.as_ref().unwrap().censoring_survival_provenance.as_deref(),
        Some("caller_supplied_fixed_not_fitted_or_verified")
    );
    assert!(
        matches!(body.query, antecedent_io::CausalQueryWire::Survival(ref wire) if wire.censoring_columns.len() == 3)
    );
    let mut tampered = body.clone();
    if let antecedent_io::CausalQueryWire::Survival(ref mut wire) = tampered.query {
        wire.censoring_probability_floor = None;
    }
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &tampered,
            vec![
                "duration".into(),
                "event".into(),
                "treatment".into(),
                "g0".into(),
                "g1".into(),
                "g3".into()
            ],
            "tampered",
        )
        .is_err()
    );
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().censoring_survival_provenance = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            vec![
                "duration".into(),
                "event".into(),
                "treatment".into(),
                "g0".into(),
                "g1".into(),
                "g3".into()
            ],
            "forged",
        )
        .is_err()
    );
    let bad_g1 = [1.0, 0.001, 0.5, 0.5];
    let bad = TabularData::from_f64_columns([
        ("duration", &duration[..]),
        ("event", &event[..]),
        ("treatment", &treatment[..]),
        ("g0", &g0[..]),
        ("g1", &bad_g1[..]),
        ("g3", &g3[..]),
    ])
    .unwrap();
    assert!(rebind_estimate(&mut prepared, bad, &ctx).is_err());
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
    let curve = first.survival().unwrap();
    assert_eq!(&*curve.times, &[0.0, 1.0, 2.0]);
    assert_eq!(&*curve.control, &[1.0, 0.5, 0.5]);
    assert_eq!(&*curve.treated, &[1.0, 1.0, 1.0]);
    assert_eq!(curve.rmst_control, Some(1.5));
    assert_eq!(curve.rmst_treated, Some(2.0));
    assert_eq!(&*curve.uncertainty, "point_only_no_interval");
    assert!(matches!(first.estimate(), PrimaryEstimate::NotAnEffect));
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
    let refreshed = rebind_estimate(&mut prepared, data, &ctx).unwrap();
    assert_eq!(first.survival(), refreshed.survival());
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
    let curve = result.survival().unwrap();
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
    let duration = (0..n)
        .map(|i| if i % 5 == 0 { 1.0 } else { 3.0 })
        .chain((0..n).map(|i| if i % 10 == 0 { 1.0 } else { 3.0 }))
        .collect::<Vec<_>>();
    let event =
        duration.iter().map(|&time| if time == 1.0 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let treatment = [vec![0.0; n], vec![1.0; n]].concat();
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let study = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(399)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(17);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let section = first.survival().unwrap();
    assert_eq!(
        section.uncertainty.as_ref(),
        "subject_stratified_percentile_bootstrap_pointwise_95"
    );
    assert_eq!(section.bootstrap_replicates_requested, Some(399));
    assert_eq!(section.bootstrap_replicates_ok, Some(399));
    let rmst = section.rmst_difference_interval.unwrap();
    // Control: 8 of 40 fail at time 1 (S = 0.8); treated 4 of 40 (S = 0.9).
    assert_single_event_time_curve(section, 3.0, 0.8, 0.9, 40.0);
    assert_eq!(section.difference_at_tau_interval.unwrap().len(), 2);
    assert_eq!(rebind_estimate(&mut prepared, data, &ctx).unwrap().survival(), first.survival());
    let artifact = prepared.encode_contracted_result(&first, "survival-bootstrap", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&artifact).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().rmst_difference_interval, Some(rmst));
    let mut tampered = body;
    tampered.survival.as_mut().unwrap().uncertainty = "point_only_no_interval".into();
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &tampered,
            header.variable_names,
            "tampered"
        )
        .is_err()
    );
}

#[test]
fn supported_survival_scalar_intervals_round_trip_and_refuse_forgery() {
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
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(7_729);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q.clone()))
        .bootstrap_replicates(299)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "supported_survival_scalar_intervals_round",
        &prepared,
        &result,
        &ctx,
    );
    assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    let curve = result.survival().unwrap();
    assert_eq!(curve.assignment_counts, [120, 120]);
    // Control: i % 3 == 0 gives 40 of 120 failures at time 1; treated: i % 7 == 0 gives 18 of 120.
    assert_single_event_time_curve(curve, 3.0, 80.0 / 120.0, 102.0 / 120.0, 120.0);
    assert_eq!(curve.bootstrap_replicates_ok, Some(299));
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "supported-survival", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(
        body.survival.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("supported")
    );
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().assignment_counts = Some([119, 121]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-arm-counts",
        )
        .is_err()
    );
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().bootstrap_replicates_ok = Some(298);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-draw-count",
        )
        .is_err()
    );
    let mut q = q;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let entry = vec![0.0; 240];
    let with_entry = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entry.as_slice()),
    ])
    .unwrap();
    let prepared = Study::tabular(with_entry.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(299)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let off_axis = prepared.estimate(&with_entry, &ctx).unwrap();
    assert!(off_axis.support_status().is_none());
    let encoded = prepared.encode_contracted_result(&off_axis, "entry-off-axis", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    body.survival.as_mut().unwrap().graphless_support_status = Some("supported".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-entry-support",
        )
        .is_err()
    );
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
            let exit = if left == 0.0 && i % 7 == 0 {
                1.0
            } else if i % (if arm { 6 } else { 5 }) == 0 {
                2.0
            } else {
                3.0
            };
            duration.push(exit);
            event.push(if exit < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entry.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let ctx = ExecutionContext::for_tests(279);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(299)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival().unwrap();
    assert_eq!(curve.uncertainty.as_ref(), "subject_stratified_percentile_bootstrap_pointwise_95");
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert_eq!(curve.bootstrap_replicates_ok, Some(299));
    let encoded =
        prepared.encode_contracted_result(&result, "delayed-entry-bootstrap", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(
        body.survival.as_ref().unwrap().rmst_difference_interval,
        curve.rmst_difference_interval
    );
    let mut forged = body;
    forged.survival.as_mut().unwrap().uncertainty = "point_only_no_interval".into();
    assert!(
        antecedent_io::encode_analysis_result_artifact(&forged, header.variable_names, "forged")
            .is_err()
    );
}

#[test]
fn delayed_entry_fixed_censoring_competing_risk_interval_is_retained() {
    retained_delayed_entry_incidence(80, true, false);
}

#[test]
fn supported_delayed_entry_incidence_round_trips_and_refuses_forgery() {
    retained_delayed_entry_incidence(160, false, true);
}

#[test]
fn supported_delayed_entry_fixed_known_incidence_round_trips_and_refuses_forgery() {
    retained_delayed_entry_incidence(200, true, true);
}

#[allow(
    clippy::too_many_lines,
    reason = "the fixture checks the same public route through execution and replay"
)]
fn retained_delayed_entry_incidence(arm_size: usize, fixed_known: bool, supported: bool) {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..arm_size {
            let left = if i % 3 == 0 { 1.0 } else { 0.0 };
            let (exit, cause) = if left == 0.0 && i % 7 == 0 {
                (1.0, 1.0)
            } else if i % 11 == 0 {
                (1.5, 0.0)
            } else if i % 5 == 0 {
                (2.0, 2.0)
            } else if i % (if arm { 4 } else { 3 }) == 0 {
                (2.0, 1.0)
            } else {
                (3.0, 0.0)
            };
            duration.push(exit);
            event.push(cause);
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let g0 = vec![1.0; duration.len()];
    let g2 = vec![0.8; duration.len()];
    let table = |duration: &[f64], event: &[f64]| {
        TabularData::from_f64_columns([
            ("duration", duration),
            ("event", event),
            ("treatment", treatment.as_slice()),
            ("entry", entry.as_slice()),
            ("g0", g0.as_slice()),
            ("g1", g0.as_slice()),
            ("g15", g0.as_slice()),
            ("g2", g2.as_slice()),
            ("g3", g2.as_slice()),
        ])
        .unwrap()
    };
    let data = table(&duration, &event);
    let mut q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    q.known_censoring = fixed_known.then_some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 1.5, 2.0, 3.0]),
        columns: Arc::from([
            VariableId::from_raw(4),
            VariableId::from_raw(5),
            VariableId::from_raw(6),
            VariableId::from_raw(7),
            VariableId::from_raw(8),
        ]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(297);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q.clone()))
        .bootstrap_replicates(299)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival().unwrap();
    if supported {
        assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    } else {
        assert_eq!(result.support_status(), None);
    }
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.rmst_difference_interval.is_none());
    assert_eq!(
        curve.censoring_survival_provenance.as_deref(),
        fixed_known.then_some("caller_supplied_fixed_not_fitted_or_verified")
    );
    assert!(curve.difference_band.is_none());
    let encoded =
        prepared.encode_contracted_result(&result, "delayed-entry-fixed-g", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "delayed-entry-incidence",
        &prepared,
        &result,
        &ctx,
    );
    let identity = antecedent_io::RetainedArtifactIdentity::of_bytes(&encoded);
    let replayed = antecedent_io::replay_analysis_result_expecting(&encoded, &identity).unwrap();
    assert!(
        replayed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_survival_incidence_operation"
        })
    );
    let replayed_curve = replayed.body.survival.as_ref().unwrap();
    assert_eq!(replayed_curve.target_cause, Some(1));
    assert_eq!(replayed_curve.tau, 3.0);
    assert_eq!(
        replayed_curve.graphless_support_status.as_deref(),
        supported.then_some("supported")
    );
    let replayed_contract = replayed.contract.as_ref().unwrap();
    let basis = replayed_contract.claim.as_ref().unwrap().calibration.basis.as_ref().unwrap();
    assert_eq!(basis.key.query, "Survival");
    assert_eq!(basis.key.interval_method, "percentile_bootstrap");
    assert_eq!(basis.scope.replicates_ok, Some(299));
    assert!(basis.key.functional.contains(if fixed_known {
        "censoring=fixed_known"
    } else {
        "censoring=unweighted"
    }));
    if supported {
        let with_source = prepared
            .encode_contracted_result_with_source(&result, &data, "incidence-with-subjects", &ctx)
            .unwrap();
        let consumed = antecedent_io::load_analysis_result(&with_source).unwrap();
        assert!(consumed.replay(&antecedent_io::ReplayOptions::default()).replayed());
        use antecedent::analysis::composition::{self, BundleBuilder, SuppliedSources};
        let mut builder = BundleBuilder::new();
        assert!(builder.add_detected_artifact(Some("compact"), &encoded).is_err());
        builder.add_detected_artifact(Some("incidence"), &with_source).unwrap();
        let bundle = builder.build().unwrap();
        let bundle_bytes = composition::export_bundle(&bundle, "delayed-incidence-bundle").unwrap();
        let bundled =
            composition::consume(&bundle_bytes, bundle.identity(), &SuppliedSources::default())
                .unwrap();
        assert_eq!(bundled.value("incidence", "horizon"), Some(3.0));
        assert_eq!(
            bundled.value("incidence", "lower_95"),
            curve.difference_at_tau_interval.map(|limits| limits[0]),
        );
        let mut bounded = antecedent_io::ReplayOptions::default();
        bounded.limits.max_operations = 0;
        assert!(
            consumed
                .replay(&bounded)
                .mismatches
                .iter()
                .any(|reason| { reason.as_ref() == "survival_incidence_replay.resource" })
        );
        duration[1] = 2.0;
        event[1] = 1.0;
        let revised_data = table(&duration, &event);
        assert!(
            prepared
                .encode_contracted_result_with_source(
                    &result,
                    &revised_data,
                    "swapped-subjects",
                    &ctx
                )
                .is_err()
        );
        let revised = prepared.rebind_data(revised_data.clone(), &ctx).unwrap();
        let revised_result = revised.estimate(&revised_data, &ctx).unwrap();
        let revised_bytes = revised
            .encode_contracted_result(&revised_result, "revised-delayed-entry-incidence", &ctx)
            .unwrap();
        assert!(
            antecedent_io::replay_analysis_result_expecting(&revised_bytes, &identity).is_err()
        );
        let revised_replay = antecedent_io::replay_analysis_result(&revised_bytes).unwrap();
        assert_ne!(revised_replay.body.survival.unwrap().control, replayed_curve.control);
        assert_ne!(
            revised_replay.contract.unwrap().identities.data_snapshot,
            replayed_contract.identities.data_snapshot,
        );
    }
    assert_eq!(
        body.survival.as_ref().unwrap().difference_at_tau_interval,
        curve.difference_at_tau_interval
    );
    for (cause, tau) in [(2, 3.0), (1, 2.0)] {
        let mut forged = body.clone();
        let curve = forged.survival.as_mut().unwrap();
        curve.target_cause = Some(cause);
        curve.tau = tau;
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &forged,
                header.variable_names.clone(),
                "forged-incidence-target",
            )
            .is_err()
        );
    }
    if fixed_known {
        let mut forged = body.clone();
        forged.survival.as_mut().unwrap().censoring_survival_provenance =
            Some("fitted_censoring_model".into());
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &forged,
                header.variable_names.clone(),
                "forged-fitted-censoring",
            )
            .is_err()
        );
    }
    let mut forged = body.clone();
    if let antecedent_io::CausalQueryWire::Survival(ref mut wire) = forged.query {
        wire.independent_given.push(VariableId::from_raw(2).raw());
    }
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-conditional-entry",
        )
        .is_err()
    );
    q.observation_assumption =
        ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(2)]));
    assert!(
        Study::tabular(data)
            .query(CausalQuery::Survival(q))
            .build()
            .unwrap()
            .prepare(&ctx)
            .is_err()
    );
}

#[test]
fn simultaneous_curve_band_is_retained_and_artifact_checked() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..100 {
            let time = if i % (if arm { 4 } else { 5 }) == 0 {
                1.0
            } else if i % 7 == 0 {
                2.0
            } else {
                3.0
            };
            duration.push(time);
            event.push(if time < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(923);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(399)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival().unwrap();
    let band = curve.difference_band.as_ref().expect("eligible randomized subjects");
    assert_eq!(band.times.as_ref(), curve.times.as_ref());
    assert_eq!(band.replicates_ok, 399);
    assert!(
        band.lower
            .iter()
            .zip(band.difference.iter())
            .zip(band.upper.iter())
            .all(|((&lower, &difference), &upper)| lower <= difference && difference <= upper)
    );
    assert!(curve.rmst_difference_interval.is_some());
    assert_eq!(result.survival(), prepared.estimate(&data, &ctx).unwrap().survival());
    let bytes = prepared.encode_contracted_result(&result, "survival-band", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(
        body.survival.as_ref().unwrap().difference_band.as_ref().unwrap().lower,
        band.lower.as_ref()
    );
    let mut forged = body;
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap().difference[1] += 0.1;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-band"
        )
        .is_err()
    );
}

#[test]
fn simultaneous_band_withholds_small_arms_without_erasing_scalar_interval() {
    let n = 60;
    let duration = (0..2 * n).map(|i| if i % 5 == 0 { 1.0 } else { 3.0 }).collect::<Vec<_>>();
    let event = duration.iter().map(|&time| if time < 3.0 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let treatment = [vec![0.0; n], vec![1.0; n]].concat();
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(923);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(399)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival().unwrap();
    assert!(curve.difference_band.is_none());
    assert!(curve.band_unavailable_reason.as_deref().unwrap().contains("80 subjects"));
    assert!(curve.rmst_difference_interval.is_some());
}

#[test]
fn delayed_entry_bands_replay_both_endpoints_with_and_without_fixed_g() {
    use antecedent::analysis::composition::{self, BundleBuilder, SuppliedSources};
    for fixed_known in [false, true] {
        for incidence in [false, true] {
            let (prepared, data, ctx) =
                delayed_entry_band_fixture(fixed_known, incidence, false, false);
            let result = prepared.estimate(&data, &ctx).unwrap();
            let curve = result.survival().unwrap();
            let band = curve.difference_band.as_ref().unwrap();
            assert_eq!(band.support_status, Some(antecedent::support::CellStatus::Supported));
            assert!(band.minimum_horizon_risk_set.unwrap() >= 25);
            assert_eq!(band.times, curve.times);
            if incidence {
                assert!(result.support_status().is_none());
            }
            let bytes = prepared
                .encode_contracted_result_with_source(&result, &data, "delayed-entry-band", &ctx)
                .unwrap();
            let loaded = antecedent_io::load_analysis_result(&bytes).unwrap();
            assert!(loaded.replay(&antecedent_io::ReplayOptions::default()).replayed());
            let compact = prepared.encode_contracted_result(&result, "compact-band", &ctx).unwrap();
            let mut builder = BundleBuilder::new();
            assert!(builder.add_detected_artifact(Some("compact"), &compact).is_err());
            builder.add_detected_artifact(Some("band"), &bytes).unwrap();
            let bundle = builder.build().unwrap();
            let bundle_bytes = composition::export_bundle(&bundle, "delayed-entry-band").unwrap();
            let consumed =
                composition::consume(&bundle_bytes, bundle.identity(), &SuppliedSources::default())
                    .unwrap();
            assert_eq!(consumed.value("band", "horizon"), Some(3.0));
            assert_eq!(consumed.value("band", "horizon_lower_95"), band.lower.last().copied());
            assert_eq!(
                consumed.value("band", "whole_grid_upper_95"),
                band.upper.iter().copied().reduce(f64::max)
            );
            if !fixed_known && incidence {
                let (_, replacement, _) = delayed_entry_band_fixture(false, true, false, true);
                assert!(
                    prepared
                        .encode_contracted_result_with_source(
                            &result,
                            &replacement,
                            "swapped-band-subjects",
                            &ctx,
                        )
                        .is_err()
                );
                let mut rebound = prepared.clone();
                let revised = rebind_estimate(&mut rebound, replacement.clone(), &ctx).unwrap();
                let revised_curve = revised.survival().unwrap().difference_band.as_ref().unwrap();
                assert_ne!(revised_curve.difference, band.difference);
                let revised_bytes = rebound
                    .encode_contracted_result_with_source(
                        &revised,
                        &replacement,
                        "revised-band-subjects",
                        &ctx,
                    )
                    .unwrap();
                let mut builder = BundleBuilder::new();
                builder.add_detected_artifact(Some("band"), &revised_bytes).unwrap();
                let replacement_bundle = builder.build().unwrap();
                let bytes =
                    composition::export_bundle(&replacement_bundle, "revised-band").unwrap();
                assert!(
                    composition::consume(&bytes, bundle.identity(), &SuppliedSources::default())
                        .is_err()
                );
                let replacement = composition::consume(
                    &bytes,
                    replacement_bundle.identity(),
                    &SuppliedSources::default(),
                )
                .unwrap();
                assert_eq!(
                    replacement.value("band", "horizon_lower_95"),
                    revised_curve.lower.last().copied()
                );
            }
            let replayed = antecedent_io::replay_analysis_result(&bytes).unwrap();
            let wire = replayed.body.survival.as_ref().unwrap();
            assert_eq!(wire.target_cause, incidence.then_some(1));
            assert_eq!(wire.tau, 3.0);
            assert_eq!(wire.difference_band.as_ref().unwrap().lower, band.lower.as_ref());
            let mut forged = replayed.body;
            forged
                .survival
                .as_mut()
                .unwrap()
                .difference_band
                .as_mut()
                .unwrap()
                .minimum_horizon_risk_set = Some(24);
            assert!(
                antecedent_io::encode_analysis_result_artifact(
                    &forged,
                    replayed.header.variable_names,
                    "forged-thin-tail",
                )
                .is_err()
            );
        }
    }
}

#[test]
fn delayed_entry_band_withholds_a_thin_horizon_risk_set() {
    for fixed_known in [false, true] {
        let (prepared, data, ctx) = delayed_entry_band_fixture(fixed_known, true, true, false);
        let result = prepared.estimate(&data, &ctx).unwrap();
        let curve = result.survival().unwrap();
        assert!(curve.difference_at_tau_interval.is_some());
        assert!(curve.difference_band.is_none());
        assert!(curve.band_unavailable_reason.as_deref().unwrap().contains("25 at-risk subjects"));
    }
}

#[test]
fn delayed_entry_band_withholds_when_the_other_arm_lacks_risk_at_an_event_knot() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for treated in [false, true] {
        for index in 0..200 {
            let exit = if !treated && index < 60 {
                1.0
            } else if index % 4 == 0 {
                2.0
            } else {
                3.0
            };
            duration.push(exit);
            event.push(f64::from(exit < 3.0));
            treatment.push(f64::from(treated));
            entry.push(f64::from(treated && index >= 20));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entry.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let ctx = ExecutionContext::for_tests(5_167);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(399)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let curve = result.survival().unwrap();
    assert!(curve.minimum_event_risk_set.unwrap() > 25);
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    assert!(curve.band_unavailable_reason.as_deref().unwrap().contains("25 at-risk subjects"));
}

fn delayed_entry_band_fixture(
    fixed_known: bool,
    incidence: bool,
    thin_tail: bool,
    revised: bool,
) -> (antecedent::PreparedStudy, TabularData, ExecutionContext) {
    let (mut duration, mut event, mut treatment, mut entry, mut known) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for arm in [false, true] {
        for index in 0..200 {
            let first = if arm { 40 } else { 60 };
            let (exit, cause, left, g) = if index < first {
                (1.0, 1.0, 0.0, 0.5)
            } else if index < first + 40 {
                (2.0, if incidence { 2.0 } else { 1.0 }, 1.0, 0.5)
            } else if thin_tail && index < 180 {
                (2.0, 0.0, 0.0, 0.8)
            } else if revised && !arm && index == 199 {
                (2.0, 1.0, 0.0, 0.8)
            } else {
                (3.0, 0.0, 0.0, 0.8)
            };
            duration.push(exit);
            event.push(cause);
            treatment.push(f64::from(arm));
            entry.push(left);
            known.push(g);
        }
    }
    let g0 = vec![1.0; duration.len()];
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entry.as_slice()),
        ("g0", g0.as_slice()),
        ("g1", known.as_slice()),
        ("g2", known.as_slice()),
        ("g3", known.as_slice()),
    ])
    .unwrap();
    let mut query = query(if incidence {
        SurvivalFunctional::CumulativeIncidence { target_cause: 1 }
    } else {
        SurvivalFunctional::SurvivalAndRmst
    });
    query.tau = 3.0;
    query.delayed_entry = Some(VariableId::from_raw(3));
    query.known_censoring = fixed_known.then(|| KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 2.0, 3.0]),
        columns: (4..8).map(VariableId::from_raw).collect(),
        minimum_probability: 0.05,
    });
    let ctx = ExecutionContext::for_tests(5_167);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(query))
        .bootstrap_replicates(399)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    (prepared, data, ctx)
}

#[test]
fn supported_delayed_entry_survival_intervals_round_trip_and_refuse_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..170 {
            let left = if i % 4 == 0 { 1.0 } else { 0.0 };
            let mut exit = if i % (if arm { 6 } else { 4 }) == 0 {
                2.0
            } else if i % 5 == 0 {
                1.0
            } else {
                3.0
            };
            if left >= exit {
                exit = 3.0;
            }
            duration.push(exit);
            event.push(if exit < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entry.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let ctx = ExecutionContext::for_tests(5_161);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(320)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "supported_delayed_entry_survival_interval",
        &prepared,
        &result,
        &ctx,
    );
    assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    let curve = result.survival().unwrap();
    assert_eq!(curve.assignment_counts, [170, 170]);
    assert!(curve.bootstrap_replicates_ok.unwrap() >= 299);
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded =
        prepared.encode_contracted_result(&result, "supported-delayed-entry", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(
        body.survival.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("supported")
    );
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().assignment_counts = Some([120, 120]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-thin-delayed-entry",
        )
        .is_err()
    );
    let mut forged = body;
    forged.survival.as_mut().unwrap().uncertainty = "point_only_no_interval".into();
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-delayed-entry-point",
        )
        .is_err()
    );
}

#[test]
fn supported_fixed_g_ipcw_survival_intervals_round_trip_and_refuse_forgery() {
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
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("g0", g0.as_slice()),
        ("g1", g1.as_slice()),
        ("g3", g3.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.known_censoring = Some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 3.0]),
        columns: Arc::from([
            VariableId::from_raw(3),
            VariableId::from_raw(4),
            VariableId::from_raw(5),
        ]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(5_162);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(320)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "supported_fixed_g_ipcw_survival_intervals",
        &prepared,
        &result,
        &ctx,
    );
    assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    let curve = result.survival().unwrap();
    assert_eq!(curve.assignment_counts, [170, 170]);
    // A constant censoring survival cancels in the normalised weights, so the curve is the
    // product-limit one: i % 3 == 0 gives 57 of 170 failures (control), i % 6 == 0 gives 29.
    assert_single_event_time_curve(curve, 3.0, 113.0 / 170.0, 141.0 / 170.0, 170.0);
    assert_eq!(
        curve.censoring_survival_provenance.as_deref(),
        Some("caller_supplied_fixed_not_fitted_or_verified")
    );
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded = prepared.encode_contracted_result(&result, "supported-fixed-g", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(
        body.survival.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("supported")
    );
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().graphless_support_status = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-drop-support",
        )
        .is_err()
    );
    let mut forged = body;
    forged.survival.as_mut().unwrap().censoring_survival_provenance = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-fixed-g-provenance",
        )
        .is_err()
    );
}

#[test]
fn supported_delayed_entry_fixed_g_survival_intervals_round_trip_and_refuse_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..210 {
            let left = if i % 4 == 0 { 1.0 } else { 0.0 };
            let mut exit = if i % (if arm { 6 } else { 3 }) == 0 {
                2.0
            } else if i % 9 == 0 {
                1.5
            } else {
                3.0
            };
            if left >= exit {
                exit = 3.0;
            }
            duration.push(exit);
            event.push(if exit < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let g0 = vec![1.0; duration.len()];
    let g_late = vec![0.85; duration.len()];
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entry.as_slice()),
        ("g0", g0.as_slice()),
        ("g1", g0.as_slice()),
        ("g15", g0.as_slice()),
        ("g2", g_late.as_slice()),
        ("g3", g_late.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    q.known_censoring = Some(KnownCensoringSurvival {
        times: Arc::from([0.0, 1.0, 1.5, 2.0, 3.0]),
        columns: Arc::from([
            VariableId::from_raw(4),
            VariableId::from_raw(5),
            VariableId::from_raw(6),
            VariableId::from_raw(7),
            VariableId::from_raw(8),
        ]),
        minimum_probability: 0.01,
    });
    let ctx = ExecutionContext::for_tests(5_163);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(320)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "supported_delayed_entry_fixed_g_survival_",
        &prepared,
        &result,
        &ctx,
    );
    assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    let curve = result.survival().unwrap();
    assert_eq!(curve.assignment_counts, [210, 210]);
    assert!(curve.rmst_difference_interval.is_some());
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.difference_band.is_none());
    let encoded = prepared
        .encode_contracted_result(&result, "supported-delayed-entry-fixed-g", &ctx)
        .unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(
        body.survival.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("supported")
    );
    let mut forged = body;
    forged.survival.as_mut().unwrap().assignment_counts = Some([160, 160]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-combined-thin",
        )
        .is_err()
    );
}

#[test]
fn supported_survival_difference_band_round_trips_and_refuses_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..100 {
            let time = if i % (if arm { 4 } else { 5 }) == 0 {
                1.0
            } else if i % 7 == 0 {
                2.0
            } else {
                3.0
            };
            duration.push(time);
            event.push(if time < 3.0 { 1.0 } else { 0.0 });
            treatment.push(f64::from(arm));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::SurvivalAndRmst);
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(5_164);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(399)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "supported_survival_difference_band_round_",
        &prepared,
        &result,
        &ctx,
    );
    // Only 100 subjects per arm: the scalar row needs 120, so the scalar route
    // is off-axis while the band's own 80-per-arm support applies.
    assert!(result.support_status().is_none());
    let curve = result.survival().unwrap();
    let band = curve.difference_band.as_ref().expect("eligible randomized subjects");
    assert_eq!(band.support_status, Some(antecedent::support::CellStatus::Supported));
    let encoded =
        prepared.encode_contracted_result(&result, "supported-survival-band", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status, None);
    assert_eq!(
        body.survival
            .as_ref()
            .unwrap()
            .difference_band
            .as_ref()
            .unwrap()
            .graphless_support_status
            .as_deref(),
        Some("supported")
    );
    // Product-limit by hand: the control arm has 20 of 100 events at time 1 and 12 of the 80
    // at risk at time 2 (S = 0.80, 0.68); the treated arm 25 of 100 and 11 of 75 (S = 0.75,
    // 0.64). The band is centred on those differences, so its curve holds both.
    let centre = &body.survival.as_ref().unwrap().difference_band.as_ref().unwrap().difference;
    for wanted in [0.05, 0.04] {
        assert!(
            centre.iter().any(|value| (value.abs() - wanted).abs() < 1e-12),
            "the band holds the product-limit difference {wanted}: {centre:?}"
        );
    }
    let mut forged = body.clone();
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap().graphless_support_status =
        None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-drop-band-support",
        )
        .is_err()
    );
    let mut forged = body;
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap().difference[1] += 0.1;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-band-curve",
        )
        .is_err()
    );
}

#[test]
fn supported_competing_incidence_band_round_trips_and_refuses_forgery() {
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    for arm in [false, true] {
        for i in 0..100 {
            let (time, cause) = if i % (if arm { 3 } else { 4 }) == 0 {
                (1.0, 1.0)
            } else if i % 5 == 0 {
                (1.0, 2.0)
            } else if i % 7 == 0 {
                (2.0, 1.0)
            } else {
                (3.0, 0.0)
            };
            duration.push(time);
            event.push(cause);
            treatment.push(f64::from(arm));
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    q.tau = 3.0;
    let ctx = ExecutionContext::for_tests(5_165);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(399)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "supported_competing_incidence_band_round_",
        &prepared,
        &result,
        &ctx,
    );
    // The no-entry design only has a band row for competing incidence.
    assert!(result.support_status().is_none());
    let curve = result.survival().unwrap();
    assert!(curve.rmst_difference_interval.is_none());
    assert!(curve.difference_at_tau_interval.is_some());
    let band = curve.difference_band.as_ref().expect("eligible randomized subjects");
    assert_eq!(band.support_status, Some(antecedent::support::CellStatus::Supported));
    let encoded =
        prepared.encode_contracted_result(&result, "supported-incidence-band", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status, None);
    assert_eq!(
        body.survival
            .as_ref()
            .unwrap()
            .difference_band
            .as_ref()
            .unwrap()
            .graphless_support_status
            .as_deref(),
        Some("supported")
    );
    let mut forged_scalar = body.clone();
    forged_scalar.survival.as_mut().unwrap().graphless_support_status = Some("supported".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged_scalar,
            header.variable_names.clone(),
            "forged-incidence-band-as-pointwise",
        )
        .is_err()
    );
    let mut forged = body;
    forged.survival.as_mut().unwrap().difference_band.as_mut().unwrap().graphless_support_status =
        None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-drop-incidence-band-support",
        )
        .is_err()
    );
}

#[test]
fn competing_incidence_scalar_interval_is_off_axis_and_refuses_forged_support() {
    // Large arms can have a thin risk set at an early failure time after delayed entry.
    let mut duration = Vec::new();
    let mut event = Vec::new();
    let mut treatment = Vec::new();
    let mut entry = Vec::new();
    for arm in [false, true] {
        for i in 0..170 {
            let left = if i < 15 { 0.0 } else { 1.0 };
            let (exit, cause) = if i < 4 {
                (1.0, 1.0)
            } else if i % (if arm { 5 } else { 3 }) == 0 {
                (2.0, 1.0)
            } else if i % 7 == 0 {
                (2.0, 2.0)
            } else {
                (3.0, 0.0)
            };
            duration.push(exit);
            event.push(if exit < 3.0 { cause } else { 0.0 });
            treatment.push(f64::from(arm));
            entry.push(left);
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
        ("entry", entry.as_slice()),
    ])
    .unwrap();
    let mut q = query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 });
    q.tau = 3.0;
    q.delayed_entry = Some(VariableId::from_raw(3));
    let ctx = ExecutionContext::for_tests(5_166);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(q))
        .bootstrap_replicates(320)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!(result.support_status().is_none());
    let curve = result.survival().unwrap();
    assert!(curve.difference_at_tau_interval.is_some());
    assert!(curve.rmst_difference_interval.is_none());
    assert!(curve.difference_band.is_none());
    assert_eq!(curve.minimum_event_risk_set, Some(15));
    let encoded = prepared.encode_contracted_result(&result, "incidence-off-axis", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.survival.as_ref().unwrap().graphless_support_status, None);
    body.survival.as_mut().unwrap().graphless_support_status = Some("supported".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-incidence-support",
        )
        .is_err()
    );
}

/// Hand-derived product-limit values of a design whose only events are at time 1 (survival
/// `s_control`/`s_treated` after it, no later events): `S` is 1 before time 1, so the restricted
/// mean over `[0, tau]` is `1 + (tau - 1) S(1)`, and the difference at `tau` is `S_t - S_c`.
fn assert_single_event_time_curve(
    curve: &antecedent::result::SurvivalEstimate,
    tau: f64,
    s_control: f64,
    s_treated: f64,
    n_per_arm: f64,
) {
    let at_one = curve.times.iter().position(|t| *t == 1.0).expect("time 1 is on the grid");
    let last = curve.times.len() - 1;
    assert_eq!(curve.times[last], tau);
    assert!((curve.control[0] - 1.0).abs() < 1e-12 && (curve.treated[0] - 1.0).abs() < 1e-12);
    for index in at_one..=last {
        assert!((curve.control[index] - s_control).abs() < 1e-12, "{:?}", curve.control);
        assert!((curve.treated[index] - s_treated).abs() < 1e-12, "{:?}", curve.treated);
    }
    let rmst = |s: f64| 1.0 + (tau - 1.0) * s;
    assert!((curve.rmst_control.unwrap() - rmst(s_control)).abs() < 1e-12);
    assert!((curve.rmst_treated.unwrap() - rmst(s_treated)).abs() < 1e-12);
    // A percentile-bootstrap interval brackets the point estimate it resamples around and is not
    // degenerate: the resampling variation of a difference of two proportions is positive.
    let rmst_diff = rmst(s_treated) - rmst(s_control);
    let [lo, hi] = curve.rmst_difference_interval.expect("scalar RMST interval");
    assert!(lo < rmst_diff && rmst_diff < hi, "[{lo}, {hi}] vs {rmst_diff}");
    let diff = s_treated - s_control;
    let [lo_d, hi_d] = curve.difference_at_tau_interval.expect("scalar difference interval");
    assert!(lo_d < diff && diff < hi_d, "[{lo_d}, {hi_d}] vs {diff}");
    // Width: the 95 percent percentile interval of a difference of two binomial proportions is
    // about 2 * 1.96 * sqrt(p_c q_c / n + p_t q_t / n); the RMST difference scales it by tau - 1.
    let se = (s_control * (1.0 - s_control) / n_per_arm
        + s_treated * (1.0 - s_treated) / n_per_arm)
        .sqrt();
    let expected = 2.0 * 1.96 * se;
    assert!((hi_d - lo_d - expected).abs() <= 0.2 * expected, "{} vs {expected}", hi_d - lo_d);
    assert!(
        ((hi - lo) - (tau - 1.0) * (hi_d - lo_d)).abs() <= 0.1 * (hi - lo),
        "rmst width {} vs {} x difference width {}",
        hi - lo,
        tau - 1.0,
        hi_d - lo_d
    );
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}

#[path = "common/conditional_survival_fixture.rs"]
mod conditional_survival_fixture;

#[test]
fn fitted_conditional_survival_identifies_the_pre_entry_population() {
    use antecedent_stats::survival::{SurvivalEndpoint, SurvivalRows, randomized_survival_summary};
    use conditional_survival_fixture as fixture;
    let ctx = ExecutionContext::for_tests(9_732);
    let estimate = |rows: &[[f64; 6]], declare_entry| {
        let data = fixture::table(rows);
        Study::tabular(data.clone())
            .query(CausalQuery::Survival(fixture::query(declare_entry)))
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap()
    };
    for delayed in [false, true] {
        for censor in [false, true] {
            let rows = fixture::exact(delayed, censor);
            let result = estimate(&rows, delayed);
            let curve = result.survival().unwrap();
            for (position, &time) in curve.times.iter().enumerate() {
                assert!((curve.control[position] - fixture::truth(0, time)).abs() < 1e-12);
                assert!((curve.treated[position] - fixture::truth(1, time)).abs() < 1e-12);
            }
        }
    }
    let rows = fixture::exact(true, true);
    let ignore_entry = estimate(&rows, false);
    let curve = ignore_entry.survival().unwrap();
    assert!(
        (curve.treated[1] - curve.control[1] - (fixture::truth(1, 1.0) - fixture::truth(0, 1.0)))
            .abs()
            > 0.01
    );

    let mut ignore_censor = [0.0; 2];
    for baseline in [0.0, 1.0] {
        let (mut duration, mut event, mut treated, mut entry) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for row in rows.iter().filter(|row| row[4] == baseline) {
            duration.push(row[0]);
            event.push(i64::from(row[1] == 1.0));
            treated.push(row[2] == 1.0);
            entry.push(row[3]);
        }
        let unweighted = randomized_survival_summary(
            &SurvivalRows {
                duration: &duration,
                event_code: &event,
                treated: &treated,
                delayed_entry: Some(&entry),
            },
            3.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        ignore_censor[0] += unweighted.control.last().unwrap() / 2.0;
        ignore_censor[1] += unweighted.treated.last().unwrap() / 2.0;
    }
    assert!(
        (ignore_censor[1] - ignore_censor[0] - (fixture::truth(1, 3.0) - fixture::truth(0, 3.0)))
            .abs()
            > 0.001
    );
}

#[test]
fn fitted_conditional_survival_band_replays_training_and_refuses_overlap() {
    use antecedent::analysis::composition::{self, BundleBuilder, SuppliedSources};
    use conditional_survival_fixture as fixture;
    let rows = fixture::exact(true, true);
    let data = fixture::table(&rows);
    let ctx = ExecutionContext::for_tests(9_732);
    let query = fixture::query(true);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::Survival(query.clone()))
        .bootstrap_replicates(fixture::DRAWS)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let expected = fixture::truth(1, 3.0) - fixture::truth(0, 3.0);
    let curve = result.survival().unwrap();
    let band = curve.difference_band.as_ref().unwrap();
    assert_eq!(
        curve.censoring_survival_provenance.as_deref(),
        Some("same_subject_stratified_reverse_product_limit_refit")
    );
    assert_eq!(band.support_status, Some(antecedent::support::CellStatus::Supported));
    assert_eq!(band.replicates_ok, fixture::DRAWS);
    assert!((band.difference.last().unwrap() - expected).abs() < 1e-12);
    assert_eq!(curve.difference_at_tau_interval, None);
    let compact = prepared.encode_contracted_result(&result, "compact-fit", &ctx).unwrap();
    assert!(
        antecedent_io::replay_analysis_result(&compact)
            .unwrap()
            .acceptance
            .unresolved
            .iter()
            .any(|reason| reason.as_ref() == "dependencies.checked_survival_conditional_training")
    );
    let bytes = prepared
        .encode_contracted_result_with_source(&result, &data, "retained-fit", &ctx)
        .unwrap();
    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("standardized"), &bytes).unwrap();
    let bundle = builder.build().unwrap();
    let bytes = composition::export_bundle(&bundle, "standardized-band").unwrap();
    let checked =
        composition::consume(&bytes, bundle.identity(), &SuppliedSources::default()).unwrap();
    checked.require_verified().unwrap();
    assert_eq!(checked.value("standardized", "horizon_lower_95"), band.lower.last().copied());

    let mut revised = rows.clone();
    let subject = revised.iter_mut().find(|row| row[0] == 3.0 && row[2] == 0.0).unwrap();
    subject[0] = 2.0;
    subject[1] = 1.0;
    let revised_data = fixture::table(&revised);
    assert!(
        prepared
            .encode_contracted_result_with_source(
                &result,
                &revised_data,
                "swapped-censoring-training",
                &ctx,
            )
            .is_err()
    );
    let mut rebound = prepared.clone();
    let revised_result = rebind_estimate(&mut rebound, revised_data.clone(), &ctx).unwrap();
    let revised_curve = revised_result.survival().unwrap().difference_band.as_ref().unwrap();
    assert_ne!(revised_curve.difference, band.difference);
    let revised_bytes = rebound
        .encode_contracted_result_with_source(
            &revised_result,
            &revised_data,
            "revised-censoring-training",
            &ctx,
        )
        .unwrap();
    let replacement = antecedent_io::replay_analysis_result(&revised_bytes).unwrap();
    assert!(replacement.acceptance.accepts_as_claim());

    let mut missing = query;
    missing.stratified_censoring = None;
    assert!(missing.validate().is_err());
    let mut bad = rows;
    for row in bad.iter_mut().filter(|row| row[2] == 1.0 && row[4] == 1.0) {
        row[5] = 0.0;
    }
    let bad = fixture::table(&bad);
    let refusal = prepared.estimate(&bad, &ctx).unwrap_err();
    assert!(refusal.to_string().contains("inadequate entry or follow-up overlap"));
}
