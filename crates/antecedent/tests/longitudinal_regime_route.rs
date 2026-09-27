//! End-to-end subject-owned sequential randomized regime value.

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{CausalQuery, LongitudinalRegimeQuery, VariableId};
use antecedent_data::TabularData;
use std::sync::Arc;

fn query() -> LongitudinalRegimeQuery {
    LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method: antecedent_core::LongitudinalRegimeMethod::Ipw,
        period_outcome_predictions: None,
        stabilizing_numerator_probabilities: None,
        q_predictions: None,
        observation_history: None,
        prediction_fold_ids: None,
        periods: 2,
        treatment_history: Arc::from([true, true, true, false, false, true, false, false]),
        regime_actions: Arc::from([true; 8]),
        treatment_probabilities: Arc::from([0.5; 8]),
        censoring_probabilities: Arc::from([1.0; 8]),
        outcome_observed: Arc::from([true; 4]),
        subject_ids: Arc::from(["a", "b", "c", "d"].map(Arc::<str>::from)),
        fold_ids: Arc::from([0; 4]),
        excluded_fold_predictions: false,
        probabilities_known_by_design: true,
        minimum_probability: 0.01,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    }
}

#[test]
fn additive_msm_is_retained_with_pointwise_clustered_uncertainty_and_artifact_identity() {
    let y = [1.0, 4.0, 3.0, 6.0, 1.0, 4.0, 3.0, 6.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel;
    q.treatment_history = Arc::from([false, false, false, true, true, false, true, true,
        false, false, false, true, true, false, true, true]);
    q.regime_actions = Arc::from([false; 16]);
    q.treatment_probabilities = Arc::from([0.5; 16]);
    q.censoring_probabilities = Arc::from([1.0; 16]);
    q.outcome_observed = Arc::from([true; 8]);
    q.subject_ids = Arc::from(["a", "b", "c", "d", "e", "f", "g", "h"].map(Arc::<str>::from));
    q.fold_ids = Arc::from([0, 0, 0, 0, 1, 1, 1, 1]);
    q.stabilizing_numerator_probabilities = Some(Arc::from([0.5; 2]));
    let study = Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q.clone())).build().unwrap();
    let ctx = ExecutionContext::for_tests(91);
    let prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let fit = first.longitudinal_regime.as_ref().unwrap();
    assert_eq!(&*fit.method, "marginal_structural_model");
    assert!((fit.value - 1.0).abs() < 1e-10);
    let effects = fit.period_effects.as_ref().unwrap();
    assert!((effects[0] - 2.0).abs() < 1e-10 && (effects[1] - 3.0).abs() < 1e-10);
    assert_eq!(fit.observed_subjects, Some(8));
    assert_eq!(&*fit.uncertainty, "pointwise_subject_clustered_cr1_no_interval");
    assert!(first.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. }
            if id.as_ref() == "additive_marginal_structural_mean")
    }));
    let bytes = prepared.encode_contracted_result(&first, "longitudinal-msm", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let recorded = &body.longitudinal_regime.as_ref().unwrap().period_effects;
    assert!((recorded[0] - 2.0).abs() < 1e-10 && (recorded[1] - 3.0).abs() < 1e-10);
    let mut fabricated = body.clone();
    fabricated.longitudinal_regime.as_mut().unwrap().standard_errors.clear();
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
    q.stabilizing_numerator_probabilities = Some(Arc::from([0.0; 2]));
    assert!(q.validate().is_err());
}

#[test]
fn retained_additive_msm_intervals_include_intercept_and_period_effects() {
    let n = 300;
    let treatment = (0..n).flat_map(|i| match i % 4 {
        0 => [false, false], 1 => [false, true], 2 => [true, false], _ => [true, true],
    }).collect::<Vec<_>>();
    let y = (0..n).map(|i| {
        1.0 + 2.0 * f64::from(treatment[2 * i]) + 3.0 * f64::from(treatment[2 * i + 1])
            + (i % 7) as f64 / 10.0
    }).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel;
    q.treatment_history = treatment.into();
    q.regime_actions = vec![false; n * 2].into();
    q.treatment_probabilities = vec![0.5; n * 2].into();
    q.censoring_probabilities = vec![1.0; n * 2].into();
    q.outcome_observed = vec![true; n].into();
    q.subject_ids = (0..n).map(|i| Arc::<str>::from(format!("msm-{i}"))).collect::<Vec<_>>().into();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into();
    q.stabilizing_numerator_probabilities = Some(Arc::from([0.5; 2]));
    let ctx = ExecutionContext::for_tests(106);
    let study = Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.longitudinal_regime.as_ref().unwrap();
    assert!(fit.value_standard_error.unwrap() > 0.0);
    let intercept = fit.value_interval_95.unwrap();
    assert!(intercept[0] < fit.value && fit.value < intercept[1]);
    let periods = fit.period_intervals_95.as_ref().unwrap();
    assert_eq!(periods.len(), 2);
    for (interval, coefficient) in periods.iter().zip(fit.period_effects.as_ref().unwrap().iter()) {
        assert!(interval[0] < *coefficient && *coefficient < interval[1]);
    }
    let bytes = prepared.encode_contracted_result(&result, "longitudinal-msm-interval", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, Some(intercept));
    body.longitudinal_regime.as_mut().unwrap().period_intervals_95[0] = [0.0, 0.0];
    assert!(antecedent_io::encode_analysis_result_artifact(&body, header.variable_names, "forged-msm").is_err());
}

#[test]
fn known_truth_value_is_retained_and_point_only_artifact_round_trips() {
    let y = [4.0, 0.0, 0.0, 0.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(query()))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(97);
    let direct = study.run(&ctx).unwrap();
    assert_eq!(direct.longitudinal_regime.as_ref().unwrap().value, 4.0);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let regime = first.longitudinal_regime.as_ref().unwrap();
    assert_eq!(regime.value, 4.0);
    assert_eq!(regime.effective_sample_size, 1.0);
    assert_eq!(regime.matched_observed_fraction, 0.25);
    assert_eq!(&*regime.uncertainty, "point_only_no_interval");
    assert!(matches!(first.estimate, PrimaryEstimate::NotAnEffect));
    assert!(first.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. }
            if id.as_ref() == "sequential_censoring_exchangeability")
    }));
    let bytes = prepared.encode_contracted_result(&first, "longitudinal-regime", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value, 4.0);
    assert_eq!(body.estimate, None);
    assert_eq!(body.standard_error, None);
    assert_eq!(body.interval_lower, None);
    let mut fabricated = body.clone();
    fabricated.interval_lower = Some(3.0);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &fabricated,
        header.variable_names,
        "fabricated"
    )
    .is_err());
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(first.longitudinal_regime, refreshed.longitudinal_regime);
}

#[test]
fn retained_subject_ipw_interval_round_trips_with_dynamic_rule_identity() {
    let n = 500;
    let y = (0..n).map(|i| if i % 4 == 0 { 4.0 } else { 0.0 }).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let mut q = query();
    q.treatment_history = (0..n).flat_map(|i| match i % 4 {
        0 => [true, true], 1 => [true, false], 2 => [false, true], _ => [false, false],
    }).collect::<Vec<_>>().into();
    q.regime_actions = vec![true; n * 2].into();
    q.treatment_probabilities = vec![0.5; n * 2].into();
    q.censoring_probabilities = vec![1.0; n * 2].into();
    q.outcome_observed = vec![true; n].into();
    q.subject_ids = (0..n).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect::<Vec<_>>().into();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into();
    q.rule_id = Some(Arc::from("static-all-treated-materialized"));
    q.rule_version = Some(Arc::from("v1"));
    q.rule_provenance = Some(Arc::from("known-rule-fixture"));
    let ctx = ExecutionContext::for_tests(104);
    let study = Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q.clone())).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let regime = result.longitudinal_regime.as_ref().unwrap();
    assert!((regime.value - 4.0).abs() < 1e-12);
    assert!(regime.value_standard_error.unwrap() > 0.0);
    let interval = regime.value_interval_95.unwrap();
    assert!(interval[0] < 4.0 && interval[1] > 4.0);
    assert_eq!(regime.rule_id.as_deref(), Some("static-all-treated-materialized"));
    assert_eq!(regime.interval_reason, None);
    assert_eq!(regime.uncertainty.as_ref(), "pointwise_subject_score_95");
    let bytes = prepared.encode_contracted_result(&result, "regime-ipw-interval", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, Some(interval));
    body.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([4.0, 4.0]);
    assert!(antecedent_io::encode_analysis_result_artifact(&body, header.variable_names, "forged-regime").is_err());
}

#[test]
fn materialized_dynamic_rule_identity_survives_native_query_and_result_artifacts() {
    let data = TabularData::from_f64_columns([("outcome", &[4.0, 0.0, 0.0, 0.0][..])]).unwrap();
    let mut q = query();
    q.regime_actions = Arc::from([true, true, true, true, false, false, false, false]);
    q.rule_id = Some(Arc::from("adaptive-threshold"));
    q.rule_version = Some(Arc::from("v1"));
    q.rule_provenance = Some(Arc::from("study-protocol-7"));
    let ctx = ExecutionContext::for_tests(99);
    let study = Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q.clone())).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.longitudinal_regime.as_ref().unwrap();
    assert_eq!(fit.value, 4.0);
    assert_eq!(fit.rule_id.as_deref(), Some("adaptive-threshold"));
    assert_eq!(&*fit.uncertainty, "point_only_no_interval");
    let bytes = prepared.encode_contracted_result(&result, "dynamic-regime", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.longitudinal_regime.as_ref().unwrap();
    assert_eq!(wire.rule_version.as_deref(), Some("v1"));
    assert_eq!(wire.rule_provenance.as_deref(), Some("study-protocol-7"));
    let antecedent_io::CausalQueryWire::LongitudinalRegime(query_wire) = &body.query else { panic!("wrong query") };
    assert_eq!(query_wire.rule_id.as_deref(), Some("adaptive-threshold"));
    assert_eq!(query_wire.regime_actions, q.regime_actions.as_ref());
    let mut fabricated = body.clone();
    fabricated.longitudinal_regime.as_mut().unwrap().rule_version = Some("v2".into());
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated").is_err());
    q.rule_version = None;
    assert!(q.validate().is_err());
}

#[test]
fn unsupported_ownership_and_sequential_positivity_refuse() {
    let mut q = query();
    q.probabilities_known_by_design = false;
    q.excluded_fold_predictions = true;
    q.fold_ids = Arc::from([0, 1, 0, 1]);
    let y = [4.0, 0.0, 0.0, 0.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let ctx = ExecutionContext::for_tests(98);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    assert!(study.prepare(&ctx).is_err());
    let mut q = query();
    q.treatment_probabilities = Arc::from([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    assert!(q.validate().is_err());
}

#[test]
fn known_truth_g_formula_is_retained_and_artifact_identifies_method() {
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::GFormula;
    q.period_outcome_predictions = Some(Arc::from([1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]));
    q.excluded_fold_predictions = true;
    q.fold_ids = Arc::from([0, 1, 0, 1]);
    let y = [2.0, 3.0, 4.0, 5.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let ctx = ExecutionContext::for_tests(99);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.longitudinal_regime.as_ref().unwrap();
    assert_eq!(value.value, 3.5);
    assert_eq!(&*value.method, "g_formula");
    assert_eq!(value.effective_sample_size, 4.0);
    assert_eq!(&*value.uncertainty, "point_only_no_interval");
    assert!(result.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. }
            if id.as_ref() == "conditional_period_reward_validity")
    }));
    let bytes = prepared.encode_contracted_result(&result, "g-formula-regime", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().method, "g_formula");
    let mut fabricated = body.clone();
    fabricated.longitudinal_regime.as_mut().unwrap().method = "ipw".into();
    assert!(antecedent_io::encode_analysis_result_artifact(
        &fabricated,
        header.variable_names,
        "fabricated"
    )
    .is_err());
}

#[test]
fn g_formula_refuses_missing_or_nonfinite_predictions() {
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::GFormula;
    assert!(q.validate().is_err());
    q.period_outcome_predictions = Some(Arc::from([f64::NAN; 8]));
    assert!(q.validate().is_err());
}

#[test]
fn sequential_dr_known_truth_dropout_and_artifact_round_trip() {
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust;
    q.treatment_history = Arc::from([false, false, false, true, false, false, true, true]);
    q.regime_actions = Arc::from([false; 8]);
    q.q_predictions = Some(Arc::from([1.0, 3.0, 5.0, 4.0, 2.0, 6.0, 1.0, 1.0]));
    q.observation_history = Some(Arc::from([true, true, true, true, true, false, true, true]));
    q.outcome_observed = Arc::from([true, true, false, true]);
    q.censoring_probabilities = Arc::from([0.8; 8]);
    q.fold_ids = Arc::from([0, 1, 0, 1]);
    q.prediction_fold_ids = Some(Arc::from([0, 1, 0, 1]));
    q.excluded_fold_predictions = true;
    let data = TabularData::from_f64_columns([("outcome", &[7.0, 99.0, 0.0, 8.0][..])]).unwrap();
    let ctx = ExecutionContext::for_tests(117);
    let study = Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q.clone())).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.longitudinal_regime.as_ref().unwrap();
    assert_eq!(&*value.method, "sequential_dr");
    assert!((value.value - 46.5 / 4.0).abs() < 1e-12);
    assert_eq!(value.maximum_weight, 6.25);
    assert_eq!(&*value.uncertainty, "point_only_no_interval");
    assert!(result.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. }
            if id.as_ref() == "subject_excluded_fold_predictions")
    }));
    let bytes = prepared.encode_contracted_result(&result, "sequential-dr-regime", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().method, "sequential_dr");
    assert_eq!(body.interval_lower, None);
}

#[test]
fn sequential_dr_refuses_split_q_fold_and_reappearing_subject() {
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust;
    q.q_predictions = Some(Arc::from([1.0; 8]));
    q.observation_history = Some(Arc::from([true; 8]));
    q.prediction_fold_ids = Some(Arc::from([1, 1, 0, 1]));
    q.fold_ids = Arc::from([0, 1, 0, 1]);
    q.excluded_fold_predictions = true;
    assert!(q.validate().is_err());
    q.prediction_fold_ids = Some(q.fold_ids.clone());
    q.observation_history = Some(Arc::from([false, true, true, true, true, true, true, true]));
    assert!(q.validate().is_err());
}
