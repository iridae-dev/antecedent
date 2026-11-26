//! End-to-end subject-owned sequential randomized regime value.
#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "integration test asserts exact deterministic estimates and builds fixtures from small nonnegative counts"
)]

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{CausalQuery, LongitudinalRegimeQuery, VariableId};
use antecedent_data::TabularData;
use std::sync::Arc;

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

fn query() -> LongitudinalRegimeQuery {
    LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method: antecedent_core::LongitudinalRegimeMethod::Ipw,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
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

/// Two-sided 95% normal quantile.
const Z95: f64 = 1.959_963_984_540_054;

fn assert_close(got: f64, want: f64, relative: f64, what: &str) {
    assert!(
        (got - want).abs() <= relative * want.abs().max(1e-12),
        "{what}: got {got}, want {want}"
    );
}

/// Ordinary least squares of `y` on `(1, t1, t2)` for a design whose four `(t1, t2)` cells have
/// equal counts (so the regressors are orthogonal once centred), written from the normal
/// equations: returns the coefficients and their HC1 standard errors
/// `sqrt(n / (n - 3) * sum_i psi_i^2 e_i^2)`, where `psi` is the coefficient's row of
/// `(X'X)^-1 X'`.
fn orthogonal_two_period_ols(y: &[f64], t1: &[f64], t2: &[f64]) -> ([f64; 3], [f64; 3]) {
    let n = y.len() as f64;
    let quarter = n / 4.0;
    let group_mean = |t: &[f64], on: bool| {
        let (sum, count) = y
            .iter()
            .zip(t)
            .filter(|(_, v)| (**v == 1.0) == on)
            .fold((0.0, 0.0), |(s, c), (value, _)| (s + value, c + 1.0));
        sum / count
    };
    let b1 = group_mean(t1, true) - group_mean(t1, false);
    let b2 = group_mean(t2, true) - group_mean(t2, false);
    let b0 = y.iter().sum::<f64>() / n - 0.5 * b1 - 0.5 * b2;
    let mut sums = [0.0; 3];
    for i in 0..y.len() {
        let e = y[i] - b0 - b1 * t1[i] - b2 * t2[i];
        let psi1 = (t1[i] - 0.5) / quarter;
        let psi2 = (t2[i] - 0.5) / quarter;
        let psi0 = 1.0 / n - 0.5 * psi1 - 0.5 * psi2;
        for (sum, psi) in sums.iter_mut().zip([psi0, psi1, psi2]) {
            *sum += (psi * e).powi(2);
        }
    }
    let scale = n / (n - 3.0);
    ([b0, b1, b2], sums.map(|sum| (scale * sum).sqrt()))
}

/// The graphless row floors the per-period action probability. A forged
/// `minimum_action_probability` below the supported row's floor, still above the
/// query's own declared floor, must void a verified support.
fn assert_low_action_probability_refuses(
    body: &antecedent_io::AnalysisResultWire,
    variable_names: &[String],
    forged_minimum: f64,
    name: &str,
) {
    let mut forged = body.clone();
    forged.longitudinal_regime.as_mut().unwrap().minimum_action_probability = forged_minimum;
    assert!(
        antecedent_io::encode_analysis_result_artifact(&forged, variable_names.to_vec(), name)
            .is_err(),
        "{name}: support survived a minimum action probability below its row floor"
    );
}

#[test]
fn additive_msm_is_retained_with_pointwise_clustered_uncertainty_and_artifact_identity() {
    let y = [1.0, 4.0, 3.0, 6.0, 1.0, 4.0, 3.0, 6.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel;
    q.treatment_history = Arc::from([
        false, false, false, true, true, false, true, true, false, false, false, true, true, false,
        true, true,
    ]);
    q.regime_actions = Arc::from([false; 16]);
    q.treatment_probabilities = Arc::from([0.5; 16]);
    q.censoring_probabilities = Arc::from([1.0; 16]);
    q.outcome_observed = Arc::from([true; 8]);
    q.subject_ids = Arc::from(["a", "b", "c", "d", "e", "f", "g", "h"].map(Arc::<str>::from));
    q.fold_ids = Arc::from([0, 0, 0, 0, 1, 1, 1, 1]);
    q.stabilizing_numerator_probabilities = Some(Arc::from([0.5; 2]));
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q.clone()))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(91);
    let prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let fit = first.longitudinal_regime().unwrap();
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
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
    q.stabilizing_numerator_probabilities = Some(Arc::from([0.0; 2]));
    assert!(q.validate().is_err());
}

#[test]
fn retained_additive_msm_intervals_include_intercept_and_period_effects() {
    let n = 300;
    let treatment = (0..n)
        .flat_map(|i| match i % 4 {
            0 => [false, false],
            1 => [false, true],
            2 => [true, false],
            _ => [true, true],
        })
        .collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * f64::from(treatment[2 * i])
                + 3.0 * f64::from(treatment[2 * i + 1])
                + (i % 7) as f64 / 10.0
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel;
    let (t1, t2): (Vec<f64>, Vec<f64>) =
        (0..n).map(|i| (f64::from(treatment[2 * i]), f64::from(treatment[2 * i + 1]))).unzip();
    q.treatment_history = treatment.into();
    q.regime_actions = vec![false; n * 2].into();
    q.treatment_probabilities = vec![0.5; n * 2].into();
    q.censoring_probabilities = vec![1.0; n * 2].into();
    q.outcome_observed = vec![true; n].into();
    q.subject_ids = (0..n).map(|i| Arc::<str>::from(format!("msm-{i}"))).collect::<Vec<_>>().into();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into();
    q.stabilizing_numerator_probabilities = Some(Arc::from([0.5; 2]));
    let ctx = ExecutionContext::for_tests(106);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.longitudinal_regime().unwrap();
    let intercept = fit.value_interval_95.unwrap();
    assert!(intercept[0] < fit.value && fit.value < intercept[1]);
    let periods = fit.period_intervals_95.as_ref().unwrap();
    assert_eq!(periods.len(), 2);
    for (interval, coefficient) in periods.iter().zip(fit.period_effects.as_ref().unwrap().iter()) {
        assert!(interval[0] < *coefficient && *coefficient < interval[1]);
    }
    // The additive MSM with unit stabilized weights is ordinary least squares on (1, t1, t2); its
    // coefficients and subject-clustered CR1 errors (one subject per cluster, so HC1) are
    // recomputed here from the normal equations.
    let ([b0, b1, b2], [se0, se1, se2]) = orthogonal_two_period_ols(&y, &t1, &t2);
    assert_close(fit.value, b0, 1e-9, "intercept");
    let effects = fit.period_effects.as_ref().unwrap();
    assert_close(effects[0], b1, 1e-9, "period 1 effect");
    assert_close(effects[1], b2, 1e-9, "period 2 effect");
    assert_close(fit.value_standard_error.unwrap(), se0, 1e-6, "intercept se");
    let errors = fit.standard_errors.as_ref().unwrap();
    assert_close(errors[0], se1, 1e-6, "period 1 se");
    assert_close(errors[1], se2, 1e-6, "period 2 se");
    assert_close(intercept[1] - intercept[0], 2.0 * Z95 * se0, 1e-6, "intercept width");
    for ((interval, effect), se) in periods.iter().zip([b1, b2]).zip([se1, se2]) {
        assert_close(interval[0], effect - Z95 * se, 1e-6, "period lower");
        assert_close(interval[1], effect + Z95 * se, 1e-6, "period upper");
    }
    let bytes =
        prepared.encode_contracted_result(&result, "longitudinal-msm-interval", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, Some(intercept));
    body.longitudinal_regime.as_mut().unwrap().period_intervals_95[0] = [0.0, 0.0];
    assert!(
        antecedent_io::encode_analysis_result_artifact(&body, header.variable_names, "forged-msm")
            .is_err()
    );
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
    assert_eq!(direct.longitudinal_regime().unwrap().value, 4.0);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let regime = first.longitudinal_regime().unwrap();
    assert_eq!(regime.value, 4.0);
    assert_eq!(regime.effective_sample_size, 1.0);
    assert_eq!(regime.matched_observed_fraction, 0.25);
    assert_eq!(&*regime.uncertainty, "point_only_no_interval");
    assert!(matches!(first.estimate(), PrimaryEstimate::NotAnEffect));
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
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
    let refreshed = rebind_estimate(&mut prepared, data, &ctx).unwrap();
    assert_eq!(first.longitudinal_regime(), refreshed.longitudinal_regime());
}

#[test]
fn retained_subject_ipw_interval_round_trips_with_dynamic_rule_identity() {
    let n = 500;
    let y = (0..n).map(|i| if i % 4 == 0 { 4.0 } else { 0.0 }).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let mut q = query();
    q.treatment_history = (0..n)
        .flat_map(|i| match i % 4 {
            0 => [true, true],
            1 => [true, false],
            2 => [false, true],
            _ => [false, false],
        })
        .collect::<Vec<_>>()
        .into();
    q.regime_actions = vec![true; n * 2].into();
    q.treatment_probabilities = vec![0.5; n * 2].into();
    q.censoring_probabilities = vec![1.0; n * 2].into();
    q.outcome_observed = vec![true; n].into();
    q.subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect::<Vec<_>>().into();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into();
    q.rule_id = Some(Arc::from("static-all-treated-materialized"));
    q.rule_version = Some(Arc::from("v1"));
    q.rule_provenance = Some(Arc::from("known-rule-fixture"));
    let ctx = ExecutionContext::for_tests(104);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q.clone()))
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let regime = result.longitudinal_regime().unwrap();
    assert!((regime.value - 4.0).abs() < 1e-12);
    let interval = regime.value_interval_95.unwrap();
    // Subject score psi_i = y_i / (0.5 * 0.5) = 16 for the 125 subjects who follow the regime and
    // 0 for the other 375 (mean 4); the standard error of the mean is
    // sqrt(sum (psi - 4)^2 / (n - 1) / n), here sqrt(24000 / 499 / 500).
    let want_se = (24_000.0_f64 / 499.0 / 500.0).sqrt();
    assert_close(regime.value_standard_error.unwrap(), want_se, 1e-9, "ipw se");
    assert_close(interval[0], 4.0 - Z95 * want_se, 1e-9, "ipw lower");
    assert_close(interval[1], 4.0 + Z95 * want_se, 1e-9, "ipw upper");
    assert_eq!(regime.rule_id.as_deref(), Some("static-all-treated-materialized"));
    assert_eq!(regime.interval_reason, None);
    assert_eq!(regime.uncertainty.as_ref(), "pointwise_subject_score_95");
    let bytes = prepared.encode_contracted_result(&result, "regime-ipw-interval", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, Some(interval));
    body.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([4.0, 4.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-regime"
        )
        .is_err()
    );
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
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q.clone()))
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.longitudinal_regime().unwrap();
    assert_eq!(fit.value, 4.0);
    assert_eq!(fit.rule_id.as_deref(), Some("adaptive-threshold"));
    assert_eq!(&*fit.uncertainty, "point_only_no_interval");
    let bytes = prepared.encode_contracted_result(&result, "dynamic-regime", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.longitudinal_regime.as_ref().unwrap();
    assert_eq!(wire.rule_version.as_deref(), Some("v1"));
    assert_eq!(wire.rule_provenance.as_deref(), Some("study-protocol-7"));
    let antecedent_io::CausalQueryWire::LongitudinalRegime(query_wire) = &body.query else {
        panic!("wrong query")
    };
    assert_eq!(query_wire.rule_id.as_deref(), Some("adaptive-threshold"));
    assert_eq!(query_wire.regime_actions, q.regime_actions.as_ref());
    let mut fabricated = body.clone();
    fabricated.longitudinal_regime.as_mut().unwrap().rule_version = Some("v2".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
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
    let value = result.longitudinal_regime().unwrap();
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
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated,
            header.variable_names,
            "fabricated"
        )
        .is_err()
    );
}

#[test]
fn fixed_known_q_g_formula_interval_round_trips_and_refuses_forged_provenance() {
    let subjects = 400;
    let predictions = (0..subjects)
        .flat_map(|subject| {
            let x = if subject % 2 == 0 { -1.0 } else { 1.0 };
            [1.0 + 0.2 * x, 2.0 + 0.3 * x]
        })
        .collect::<Vec<_>>();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::GFormula;
    q.period_outcome_predictions = Some(Arc::from(predictions));
    q.known_fixed_outcome_predictions = true;
    q.treatment_history = Arc::from(vec![false; subjects * 2]);
    q.regime_actions = Arc::from(vec![true; subjects * 2]);
    q.treatment_probabilities = Arc::from(vec![0.5; subjects * 2]);
    q.censoring_probabilities = Arc::from(vec![1.0; subjects * 2]);
    q.outcome_observed = Arc::from(vec![true; subjects]);
    q.subject_ids = Arc::from(
        (0..subjects)
            .map(|subject| Arc::<str>::from(format!("subject-{subject}")))
            .collect::<Vec<_>>(),
    );
    q.fold_ids = Arc::from(vec![0; subjects]);
    q.excluded_fold_predictions = false;
    q.probabilities_known_by_design = true;
    let outcome = vec![0.0; subjects];
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(314);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q.clone()))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.longitudinal_regime().unwrap();
    assert_eq!(value.value, 3.0);
    let interval = value.value_interval_95.unwrap();
    // Each subject's plug-in score is 3 + 0.5 x with x = -1 or +1 in equal halves: sample
    // standard deviation 0.5 * sqrt(400 / 399), so the mean has standard error that over 20.
    let want_se = 0.5 * (400.0_f64 / 399.0).sqrt() / 20.0;
    assert_close(value.value_standard_error.unwrap(), want_se, 1e-9, "g-formula se");
    assert_close(interval[0], 3.0 - Z95 * want_se, 1e-9, "g-formula lower");
    assert_close(interval[1], 3.0 + Z95 * want_se, 1e-9, "g-formula upper");
    assert_eq!(value.uncertainty.as_ref(), "pointwise_subject_score_conditional_fixed_known_q_95");
    let encoded = prepared.encode_contracted_result(&result, "fixed-known-q", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, Some(interval));
    assert!(matches!(&body.query, antecedent_io::CausalQueryWire::LongitudinalRegime(query)
        if query.known_fixed_outcome_predictions));
    let mut forged = body;
    if let antecedent_io::CausalQueryWire::LongitudinalRegime(query) = &mut forged.query {
        query.known_fixed_outcome_predictions = false;
    }
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names,
            "forged-fixed-q"
        )
        .is_err()
    );
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
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q.clone()))
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.longitudinal_regime().unwrap();
    assert_eq!(&*value.method, "sequential_dr");
    assert!((value.value - 46.5 / 4.0).abs() < 1e-12);
    assert_eq!(value.maximum_weight, 6.25);
    assert_eq!(&*value.uncertainty, "point_only_no_interval");
    assert_eq!(value.value_interval_95, None);
    assert_eq!(
        value.interval_reason.as_deref(),
        Some("insufficient_calibrated_horizon_or_trajectory_support_for_sequential_dr_interval")
    );
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
#[allow(
    clippy::too_many_lines,
    reason = "one fixture follows the score through support, replay and revision"
)]
fn four_period_sequential_dr_interval_matches_the_forward_score_and_replays() {
    use antecedent::analysis::composition::{BundleBuilder, consume, export_bundle};
    use antecedent_estimate::longitudinal_regime::{
        SequentialDrInputs, SubjectHistories, evaluate_sequential_dr_value_with_subject_scores,
    };

    let n = 2_400;
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust;
    q.periods = 4;
    q.treatment_history = (0..n).flat_map(|i| (0..4).map(move |t| i & (1 << t) != 0)).collect();
    q.regime_actions = vec![true; n * 4].into();
    q.treatment_probabilities = vec![0.5; n * 4].into();
    q.censoring_probabilities = vec![0.95; n * 4].into();
    let observation: Arc<[bool]> =
        (0..n).flat_map(|i| (0..4).map(move |t| i / 16 % 25 >= 4 || t < i / 16 % 25)).collect();
    q.outcome_observed = observation.chunks_exact(4).map(|row| row[3]).collect();
    q.observation_history = Some(observation.clone());
    let predictions: Arc<[f64]> = (0..n)
        .flat_map(|i| {
            [-3.0 / 64.0, -1.0 / 64.0, 1.0 / 64.0, 3.0 / 64.0]
                .map(|offset| 6.0 + ((i % 5) as f64 - 2.0) / 16.0 + offset)
        })
        .collect();
    q.q_predictions = Some(predictions.clone());
    q.subject_ids = (0..n).map(|i| Arc::<str>::from(format!("four-{i}"))).collect();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect();
    q.prediction_fold_ids = Some(q.fold_ids.clone());
    q.excluded_fold_predictions = true;
    let outcomes: Vec<_> = q
        .treatment_history
        .chunks_exact(4)
        .enumerate()
        .map(|(i, row)| {
            2.0 + row.iter().filter(|&&treated| treated).count() as f64 + (i % 11) as f64 / 8.0
                - 0.625
        })
        .collect();

    // Expand the recursion as Q₁ plus a sum of weighted increments along
    // the matching, still-observed prefix. This uses no backwards recursion.
    let expected_scores: Vec<_> = outcomes
        .iter()
        .enumerate()
        .map(|(i, &outcome)| {
            let start = i * 4;
            let mut score = predictions[start];
            for (t, weight) in [1_i32, 2, 3, 4].map(|t| 0.475_f64.powi(-t)).into_iter().enumerate()
            {
                let j = start + t;
                if !(q.treatment_history[j] && observation[j]) {
                    break;
                }
                let next = if t == 3 { outcome } else { predictions[j + 1] };
                score += weight * (next - predictions[j]);
            }
            score
        })
        .collect();
    let (_, scores) = evaluate_sequential_dr_value_with_subject_scores(
        &outcomes,
        &SubjectHistories {
            treatment: &q.treatment_history,
            treatment_probability: &q.treatment_probabilities,
            outcome_observed: &q.outcome_observed,
            censoring_probability: &q.censoring_probabilities,
            periods: 4,
            minimum_probability: q.minimum_probability,
        },
        &SequentialDrInputs {
            actions: &q.regime_actions,
            observation_history: &observation,
            q_predictions: &predictions,
        },
    )
    .unwrap();
    for (&actual, &expected) in scores.iter().zip(&expected_scores) {
        assert_close(actual, expected, 1e-12, "independent forward score");
    }
    let mean = expected_scores.iter().sum::<f64>() / n as f64;
    let sample_variance =
        expected_scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (sample_variance / n as f64).sqrt();
    let ctx = ExecutionContext::for_tests(194);
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q.clone()))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.longitudinal_regime().unwrap();
    let contract = prepared.contract_for_result(&result).unwrap();
    let bases = result.calibration_bases(&contract).unwrap();
    assert_eq!(bases.len(), 1);
    assert_eq!(
        bases[0].key.functional,
        "fixed_all_treated_h4_fair_coin_censor_survival_095_external_q"
    );
    let unit = antecedent_io::graphless_support::sequential_dr_key(4).unwrap();
    let row_earned = antecedent_io::graphless_support::graphless_admits(
        unit,
        antecedent_io::graphless_support::GraphlessAssignmentSupport {
            assignment_unit: "unit",
            rows: n,
            interval_95_published: true,
            reported_intervals: 1,
            min_probability: 0.5,
            disjoint_nuisance_training: true,
            ..Default::default()
        },
    );
    assert_eq!(
        fit.graphless_support_status,
        row_earned.then_some(antecedent::support::CellStatus::Supported)
    );
    assert_close(fit.value, mean, 1e-12, "value");
    assert_close(fit.value_standard_error.unwrap(), se, 1e-12, "standard error");
    let interval = fit.value_interval_95.unwrap();
    assert_close(interval[0], mean - Z95 * se, 1e-12, "lower endpoint");
    assert_close(interval[1], mean + Z95 * se, 1e-12, "upper endpoint");
    retained_graphless::assert_retained_graphless_program(
        "four-period-dr",
        &prepared,
        &result,
        &ctx,
    );
    let bytes = prepared.encode_contracted_result(&result, "four-period-dr", &ctx).unwrap();
    let compact = antecedent_io::replay_analysis_result(&bytes).unwrap();
    assert!(
        compact.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_longitudinal_regime_operation"
        })
    );
    let sourced = prepared
        .encode_contracted_result_with_source(&result, &data, "four-period-dr-source", &ctx)
        .unwrap();
    let consumed = antecedent_io::replay_analysis_result(&sourced).unwrap();
    assert_eq!(
        consumed.body.longitudinal_regime.as_ref().unwrap().value_interval_95,
        Some(interval)
    );
    assert!(
        !consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_longitudinal_regime_operation"
        })
    );
    assert_eq!(
        consumed.contract.as_ref().unwrap().identities,
        compact.contract.as_ref().unwrap().identities
    );
    let rebuild_bundle = |source: &[u8]| {
        let mut builder = BundleBuilder::new();
        builder.add_detected_artifact(Some("regime"), source).unwrap();
        let bundle = builder.build().unwrap();
        let bytes = export_bundle(&bundle, "longitudinal-bundle").unwrap();
        (bundle.identity().to_owned(), bytes)
    };
    let (bundle_identity, bundle_bytes) = rebuild_bundle(&sourced);
    let bundle_replay = consume(&bundle_bytes, &bundle_identity, &Default::default()).unwrap();
    assert!(bundle_replay.all_verified());
    assert_eq!(bundle_replay.value("regime", "regime_value"), Some(fit.value));
    assert_eq!(bundle_replay.value("regime", "lower_95"), Some(interval[0]));
    let mut compact_bundle = BundleBuilder::new();
    assert!(compact_bundle.add_detected_artifact(Some("regime"), &bytes).is_err());
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, Some(interval));
    body.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-four-period-dr"
        )
        .is_err()
    );

    let revised_placeholders: Vec<_> = outcomes
        .iter()
        .zip(&*q.outcome_observed)
        .map(|(&y, &observed)| if observed { y } else { y + 1_000_000.0 })
        .collect();
    let placeholder_table =
        TabularData::from_f64_columns([("outcome", revised_placeholders.as_slice())]).unwrap();
    assert!(
        prepared
            .encode_contracted_result_with_source(
                &result,
                &placeholder_table,
                "swapped-source",
                &ctx
            )
            .is_err()
    );
    let (mut swapped, _, _) = antecedent_io::decode_analysis_result_artifact(&sourced).unwrap();
    let replacement =
        antecedent_io_core::SourceWitnessWire::from_storage(placeholder_table.storage()).unwrap();
    let source_id = antecedent_io_core::SOURCE_WITNESS_SECTION;
    let (descriptor, section) = antecedent_io_core::pack_section_shared(
        source_id,
        "application/cbor",
        antecedent_io_core::to_cbor(&replacement).unwrap().into(),
        antecedent_io_core::CompressPolicy::Auto,
    );
    let index = swapped.sections.iter().position(|section| section.id == source_id).unwrap();
    swapped.manifest.sections[index] = descriptor;
    swapped.sections[index] = section;
    let mut swapped_bytes = Vec::new();
    swapped.write_to(&mut swapped_bytes).unwrap();
    let refused = antecedent_io::replay_analysis_result(&swapped_bytes).unwrap();
    assert!(
        refused
            .acceptance
            .unresolved
            .iter()
            .any(|reason| reason.as_ref() == "longitudinal_regime_replay.source")
    );
    let mut prepared = prepared;
    let placeholder_fit = rebind_estimate(&mut prepared, placeholder_table, &ctx).unwrap();
    assert_eq!(placeholder_fit.longitudinal_regime().unwrap().value, fit.value);
    assert_eq!(placeholder_fit.longitudinal_regime().unwrap().value_interval_95, Some(interval));

    let revised: Vec<_> = outcomes.iter().map(|y| y + 1.0).collect();
    let data_revised = TabularData::from_f64_columns([("outcome", revised.as_slice())]).unwrap();
    let revised_fit = rebind_estimate(&mut prepared, data_revised.clone(), &ctx).unwrap();
    assert!(revised_fit.longitudinal_regime().unwrap().value > mean);
    assert!(prepared.encode_contracted_result(&result, "old-parent", &ctx).is_err());
    let revised_sourced = prepared
        .encode_contracted_result_with_source(&revised_fit, &data_revised, "revised-source", &ctx)
        .unwrap();
    let (revised_identity, revised_bundle_bytes) = rebuild_bundle(&revised_sourced);
    assert_ne!(revised_identity, bundle_identity);
    assert!(consume(&revised_bundle_bytes, &bundle_identity, &Default::default()).is_err());
    let replay = consume(&revised_bundle_bytes, &revised_identity, &Default::default()).unwrap();
    assert!(replay.all_verified());
    assert_eq!(
        replay.value("regime", "regime_value"),
        Some(revised_fit.longitudinal_regime().unwrap().value)
    );

    // Each case crosses a separate support boundary while retaining a point value.
    let mut low_overlap = q.clone();
    Arc::make_mut(&mut low_overlap.treatment_probabilities)[0] = 0.44;
    let mut heavy_censoring = q.clone();
    Arc::make_mut(&mut heavy_censoring.censoring_probabilities)[0] = 0.94;
    let mut few_matches = q.clone();
    for i in (15..n).step_by(16).take(100) {
        Arc::make_mut(&mut few_matches.treatment_history)[i * 4 + 3] = false;
    }
    Arc::make_mut(&mut q.regime_actions)[0] = false;
    for outside in [low_overlap, heavy_censoring, few_matches, q] {
        let result = Study::tabular(data.clone())
            .query(CausalQuery::LongitudinalRegime(outside))
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap();
        let fit = result.longitudinal_regime().unwrap();
        assert!(fit.value.is_finite());
        assert!(fit.value_interval_95.is_none());
        assert_eq!(fit.graphless_support_status, None);
        assert_eq!(
            fit.interval_reason.as_deref(),
            Some(
                "insufficient_calibrated_horizon_or_trajectory_support_for_sequential_dr_interval"
            )
        );
    }
}

#[test]
fn sequential_dr_refuses_interval_below_fitted_q_subject_floor_and_binds_subject_folds() {
    let n = 300;
    let treatment = (0..n)
        .flat_map(|i| match i % 4 {
            0 => [false, false],
            1 => [false, true],
            2 => [true, false],
            _ => [true, true],
        })
        .collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|i| {
            2.0 + f64::from(treatment[2 * i])
                + f64::from(treatment[2 * i + 1])
                + ((i % 7) as f64 - 3.0) / 10.0
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust;
    q.treatment_history = treatment.into();
    q.regime_actions = vec![true; n * 2].into();
    q.treatment_probabilities = vec![0.5; n * 2].into();
    q.censoring_probabilities = vec![1.0; n * 2].into();
    q.outcome_observed = vec![true; n].into();
    q.observation_history = Some(vec![true; n * 2].into());
    q.q_predictions = Some(vec![4.0; n * 2].into());
    q.subject_ids = (0..n).map(|i| Arc::<str>::from(format!("dr-{i}"))).collect::<Vec<_>>().into();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into();
    q.prediction_fold_ids = Some(q.fold_ids.clone());
    q.excluded_fold_predictions = true;
    let ctx = ExecutionContext::for_tests(121);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.longitudinal_regime().unwrap();
    assert!(fit.value_standard_error.is_none());
    assert!(fit.value_interval_95.is_none());
    assert_eq!(fit.uncertainty.as_ref(), "point_only_no_interval");
    let bytes = prepared.encode_contracted_result(&result, "sequential-dr-interval", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let antecedent_io::CausalQueryWire::LongitudinalRegime(query_wire) = &body.query else {
        panic!("wrong query")
    };
    assert_eq!(query_wire.prediction_fold_ids, query_wire.fold_ids);
    assert!(query_wire.excluded_fold_predictions);
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, None);
    let mut fabricated_ownership = body.clone();
    let antecedent_io::CausalQueryWire::LongitudinalRegime(query_wire) =
        &mut fabricated_ownership.query
    else {
        panic!("wrong query")
    };
    query_wire.prediction_fold_ids[0] = 99;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &fabricated_ownership,
            header.variable_names.clone(),
            "forged-dr-ownership"
        )
        .is_err()
    );
    body.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(&body, header.variable_names, "forged-dr")
            .is_err()
    );
}

#[test]
fn two_period_sequential_dr_graphless_support_round_trips_and_refuses_forgery() {
    let n = 500;
    let treatment = (0..n).flat_map(|i| [i % 5 < 2, (i / 5) % 5 < 2]).collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|i| {
            2.0 + f64::from(treatment[2 * i])
                + f64::from(treatment[2 * i + 1])
                + ((i % 7) as f64 - 3.0) / 10.0
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust;
    q.treatment_history = treatment.into();
    q.regime_actions = vec![true; n * 2].into();
    q.treatment_probabilities = vec![0.4; n * 2].into();
    q.censoring_probabilities = vec![0.85; n * 2].into();
    q.outcome_observed = vec![true; n].into();
    q.observation_history = Some(vec![true; n * 2].into());
    q.q_predictions = Some(vec![4.0; n * 2].into());
    q.subject_ids = (0..n).map(|i| Arc::<str>::from(format!("two-{i}"))).collect::<Vec<_>>().into();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into();
    q.prediction_fold_ids = Some(q.fold_ids.clone());
    q.excluded_fold_predictions = true;
    let ctx = ExecutionContext::for_tests(221);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "two_period_sequential_dr_graphless_licen",
        &prepared,
        &result,
        &ctx,
    );
    let fit = result.longitudinal_regime().unwrap();
    assert!(fit.value_interval_95.is_some());
    assert_eq!(fit.graphless_support_status, Some(antecedent::support::CellStatus::Supported));
    assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    let bytes = prepared.encode_contracted_result(&result, "two-period-dr-support", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(
        body.longitudinal_regime.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("supported")
    );
    assert_low_action_probability_refuses(&body, &header.variable_names, 0.3, "low-prob-two-dr");
    body.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-two-period-dr-interval"
        )
        .is_err()
    );
}

#[test]
fn fixed_known_q_g_formula_graphless_support_round_trips_and_refuses_forgery() {
    let subjects = 400;
    let predictions = (0..subjects)
        .flat_map(|subject| {
            let x = if subject % 2 == 0 { -1.0 } else { 1.0 };
            [2.0 + 0.3 * x, 3.0 + 0.2 * x]
        })
        .collect::<Vec<_>>();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::GFormula;
    q.period_outcome_predictions = Some(Arc::from(predictions));
    q.known_fixed_outcome_predictions = true;
    q.treatment_history = Arc::from(vec![false; subjects * 2]);
    q.regime_actions = Arc::from(vec![true; subjects * 2]);
    q.treatment_probabilities = Arc::from(vec![0.5; subjects * 2]);
    q.censoring_probabilities = Arc::from(vec![1.0; subjects * 2]);
    q.outcome_observed = Arc::from(vec![true; subjects]);
    q.subject_ids = Arc::from(
        (0..subjects).map(|subject| Arc::<str>::from(format!("gf-{subject}"))).collect::<Vec<_>>(),
    );
    q.fold_ids = Arc::from(vec![0; subjects]);
    q.probabilities_known_by_design = true;
    let outcome = vec![0.0; subjects];
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(271);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "fixed_known_q_g_formula_graphless_licens",
        &prepared,
        &result,
        &ctx,
    );
    let fit = result.longitudinal_regime().unwrap();
    assert_eq!(&*fit.method, "g_formula");
    assert!(fit.value_interval_95.is_some());
    assert_eq!(fit.uncertainty.as_ref(), "pointwise_subject_score_conditional_fixed_known_q_95");
    assert_eq!(fit.graphless_support_status, Some(antecedent::support::CellStatus::Supported));
    assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    let bytes = prepared.encode_contracted_result(&result, "g-formula-support", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(
        body.longitudinal_regime.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("supported")
    );
    // Stripping the earned support from a still-published interval is rejected.
    let mut cleared = body.clone();
    cleared.longitudinal_regime.as_mut().unwrap().graphless_support_status = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &cleared,
            header.variable_names.clone(),
            "g-formula-cleared"
        )
        .is_err()
    );
    assert_low_action_probability_refuses(&body, &header.variable_names, 0.1, "low-prob-g-formula");
    body.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-g-formula-interval"
        )
        .is_err()
    );
}

#[test]
fn additive_msm_graphless_support_round_trips_and_refuses_forgery() {
    let n = 300;
    let treatment = (0..n)
        .flat_map(|i| match i % 4 {
            0 => [false, false],
            1 => [false, true],
            2 => [true, false],
            _ => [true, true],
        })
        .collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * f64::from(treatment[2 * i])
                + 3.0 * f64::from(treatment[2 * i + 1])
                + (i % 7) as f64 / 10.0
        })
        .collect::<Vec<_>>();
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
    let ctx = ExecutionContext::for_tests(287);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "additive_msm_graphless_support_round_tri",
        &prepared,
        &result,
        &ctx,
    );
    let fit = result.longitudinal_regime().unwrap();
    assert!(fit.value_interval_95.is_some());
    assert_eq!(fit.period_intervals_95.as_ref().unwrap().len(), 2);
    assert_eq!(fit.uncertainty.as_ref(), "pointwise_subject_clustered_cr1_95");
    assert_eq!(fit.graphless_support_status, Some(antecedent::support::CellStatus::Supported));
    assert_eq!(result.support_status(), Some(antecedent::support::CellStatus::Supported));
    let bytes = prepared.encode_contracted_result(&result, "msm-support", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(
        body.longitudinal_regime.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("supported")
    );
    // The support cannot survive without every published coefficient interval.
    let mut cleared = body.clone();
    cleared.longitudinal_regime.as_mut().unwrap().graphless_support_status = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &cleared,
            header.variable_names.clone(),
            "msm-cleared"
        )
        .is_err()
    );
    assert_low_action_probability_refuses(&body, &header.variable_names, 0.3, "low-prob-msm");
    body.longitudinal_regime.as_mut().unwrap().period_intervals_95[0] = [0.0, 0.0];
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-msm-support"
        )
        .is_err()
    );
}

#[test]
fn three_period_sequential_dr_interval_round_trips_and_refuses_uncalibrated_horizon() {
    let n = 800;
    let treatment = (0..n).flat_map(|i| [i & 1 != 0, i & 2 != 0, i & 4 != 0]).collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|i| {
            2.0 + (0..3).map(|t| f64::from(treatment[3 * i + t])).sum::<f64>()
                + ((i % 11) as f64 - 5.0) / 20.0
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let mut q = query();
    q.method = antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust;
    q.periods = 3;
    q.treatment_history = treatment.into();
    q.regime_actions = vec![true; n * 3].into();
    q.treatment_probabilities = vec![0.5; n * 3].into();
    q.censoring_probabilities = vec![0.95; n * 3].into();
    q.outcome_observed = vec![true; n].into();
    q.observation_history = Some(vec![true; n * 3].into());
    q.q_predictions = Some((0..n).flat_map(|_| [4.4, 4.7, 5.1]).collect::<Vec<_>>().into());
    q.subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("three-{i}"))).collect::<Vec<_>>().into();
    q.fold_ids = (0..n).map(|i| (i % 5) as u32).collect::<Vec<_>>().into();
    q.prediction_fold_ids = Some(q.fold_ids.clone());
    q.excluded_fold_predictions = true;
    let ctx = ExecutionContext::for_tests(193);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q.clone()))
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "three_period_sequential_dr_interval_roun",
        &prepared,
        &result,
        &ctx,
    );
    let fit = result.longitudinal_regime().unwrap();
    assert!(fit.value_interval_95.is_some());
    assert_eq!(fit.graphless_support_status, Some(antecedent::support::CellStatus::Supported));
    assert_eq!(fit.uncertainty.as_ref(), "pointwise_subject_score_conditional_excluded_fold_q_95");
    assert!(result.identification.required_assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::Custom { id, .. }
            if id.as_ref() == "sequential_q_validity")
    }));
    let bytes = prepared.encode_contracted_result(&result, "three-period-dr", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().value_interval_95, fit.value_interval_95);
    body.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-three-period-dr"
        )
        .is_err()
    );

    q.periods = 4;
    q.treatment_history = (0..n)
        .flat_map(|i| [i & 1 != 0, i & 2 != 0, i & 4 != 0, i & 8 != 0])
        .collect::<Vec<_>>()
        .into();
    q.regime_actions = vec![true; n * 4].into();
    q.treatment_probabilities = vec![0.5; n * 4].into();
    q.censoring_probabilities = vec![0.95; n * 4].into();
    q.observation_history = Some(vec![true; n * 4].into());
    q.q_predictions = Some(vec![6.0; n * 4].into());
    let study =
        Study::tabular(data.clone()).query(CausalQuery::LongitudinalRegime(q)).build().unwrap();
    let fit = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
    let value = fit.longitudinal_regime().unwrap();
    assert!(value.value_interval_95.is_none());
    assert_eq!(value.uncertainty.as_ref(), "point_only_no_interval");
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

#[test]
#[allow(clippy::too_many_lines, reason = "one shared fixture") ]
fn fitted_two_period_q_repeats_training_and_evaluation_in_both_subject_bootstraps() {
    use antecedent::analysis::composition::{BundleBuilder, consume, export_bundle};
    use antecedent_core::{FittedRegimeQ, LongitudinalRegimeMethod, StreamDomain};

    let n = 800;
    let replicates = 199;
    let baseline: Vec<_> = (0..n).map(|i| ((i / 4) % 2) as f64).collect();
    let outcomes: Vec<_> = (0..n)
        .map(|i| {
            1.0 + (i & 1) as f64
                + 2.0 * f64::from(i & 2 != 0)
                + 3.0 * baseline[i]
                + 2.0 * ((i / 16 % 11) as f64 - 5.0)
        })
        .collect();
    let data = TabularData::from_f64_columns([
        ("outcome", outcomes.as_slice()),
        ("baseline", baseline.as_slice()),
    ])
    .unwrap();
    let mut q = query();
    q.treatment_history = (0..n).flat_map(|i| [i & 1 != 0, i & 2 != 0]).collect();
    q.regime_actions = vec![true; n * 2].into();
    q.treatment_probabilities = vec![0.5; n * 2].into();
    q.censoring_probabilities = vec![1.0; n * 2].into();
    q.outcome_observed = vec![true; n].into();
    q.subject_ids = (0..n).map(|i| Arc::<str>::from(format!("fitted-{i}"))).collect();
    q.fold_ids = (0..n).map(|i| ((i / 8) % 2) as u32).collect();
    let members: [Vec<_>; 2] =
        std::array::from_fn(|fold| (0..n).filter(|&i| q.fold_ids[i] == fold as u32).collect());
    let ctx = ExecutionContext::for_tests(195);

    let reference = |groups: &[Vec<usize>; 2], external: bool| {
        let mut predictions = Vec::new();
        for eval_fold in if external { 1..2 } else { 0..2 } {
            let training = &groups[1 - eval_fold];
            for &i in &groups[eval_fold] {
                let matching =
                    training.iter().copied().filter(|&j| j % 4 == 3 && baseline[j] == baseline[i]);
                let (sum, count) =
                    matching.fold((0.0, 0), |(sum, count), j| (sum + outcomes[j], count + 1));
                assert!(count > 0);
                predictions.push(sum / f64::from(count));
            }
        }
        predictions.iter().sum::<f64>() / predictions.len() as f64
    };
    let quantile = |values: &[f64], probability: f64| {
        let rank = probability * (values.len() - 1) as f64;
        let lower = values[rank.floor() as usize];
        let upper = values[rank.ceil() as usize];
        lower + rank.fract() * (upper - lower)
    };
    let mut fit_widths = Vec::new();
    for external in [false, true] {
        q.method = LongitudinalRegimeMethod::FittedGFormula(FittedRegimeQ {
            baseline: VariableId::from_raw(1),
            external_training_fold: external.then_some(0),
            bootstrap_replicates: replicates,
        });
        let prepared = Study::tabular(data.clone())
            .query(CausalQuery::LongitudinalRegime(q.clone()))
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let fit = result.longitudinal_regime().unwrap();
        assert_close(fit.value, reference(&members, external), 1e-12, "full-refit point");
        assert_eq!(fit.fitted_q_bootstrap_replicates_ok, Some(replicates as usize));
        let mut draws: Vec<_> = (0..replicates)
            .map(|replicate| {
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::Resample,
                    0x5126_7740_0000_0000 | u64::from(replicate),
                );
                let selected = members.each_ref().map(|original| {
                    (0..original.len())
                        .map(|_| {
                            original[antecedent_kernels::unbiased_index(&mut rng, original.len())]
                        })
                        .collect()
                });
                reference(&selected, external)
            })
            .collect();
        draws.sort_by(f64::total_cmp);
        let interval = fit.value_interval_95.unwrap();
        assert_close(interval[0], quantile(&draws, 0.025), 1e-12, "full-refit lower percentile");
        assert_close(interval[1], quantile(&draws, 0.975), 1e-12, "full-refit upper percentile");
        fit_widths.push(interval[1] - interval[0]);
        let sourced = prepared
            .encode_contracted_result_with_source(&result, &data, "fitted-q", &ctx)
            .unwrap();
        let consumed = antecedent_io::replay_analysis_result(&sourced).unwrap();
        assert!(consumed.acceptance.accepts_as_claim(), "{:?}", consumed.acceptance.unresolved);
        assert_eq!(consumed.body.longitudinal_regime.unwrap().value_interval_95, Some(interval));
        let mut builder = BundleBuilder::new();
        builder.add_detected_artifact(Some("fit"), &sourced).unwrap();
        let bundle = builder.build().unwrap();
        let bytes = export_bundle(&bundle, "fitted-q-bundle").unwrap();
        let replayed = consume(&bytes, bundle.identity(), &Default::default()).unwrap();
        assert_eq!(replayed.value("fit", "upper_95"), Some(interval[1]));
    }

    q.method = LongitudinalRegimeMethod::GFormula;
    q.known_fixed_outcome_predictions = true;
    q.period_outcome_predictions =
        Some(baseline.iter().flat_map(|x| [0.0, 4.0 + 3.0 * x]).collect());
    let result = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(q))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let fixed = result.longitudinal_regime().unwrap().value_interval_95.unwrap();
    assert!(fit_widths.iter().all(|width| *width > 3.0 * (fixed[1] - fixed[0])));
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
