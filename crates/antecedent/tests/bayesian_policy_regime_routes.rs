//! Bayesian policy-value and sequential-regime posteriors through the public study flow.
//!
//! Both routes are the Bayesian counterparts of the retained randomized operations: the same
//! query under `Bayesian` inference. The fixtures carry hand scores, and the retained artifact is
//! decoded by the independent loader and recomputed from the draws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the pinned scores are exact small integers")]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, EstimatorId, InferenceMode, PrimaryEstimate, Study};
use antecedent_core::{CausalQuery, LongitudinalRegimeQuery, PolicyValueQuery, VariableId};
use antecedent_data::TabularData;

const DRAWS: usize = 6000;

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS))
}

fn dirichlet_sd(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    let centre = scores.iter().sum::<f64>() / n;
    let spread = scores.iter().map(|s| s * s).sum::<f64>() / n - centre * centre;
    (spread / (n + 1.0)).sqrt()
}

fn policy_query() -> PolicyValueQuery {
    PolicyValueQuery {
        outcome: VariableId::from_raw(0),
        assignment: Arc::from([true, true, false, false]),
        propensity: Arc::from([0.5]),
        actions: Arc::from([true; 4]),
        reference: Arc::from([false; 4]),
        mu0: Arc::from([]),
        mu1: Arc::from([]),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: Arc::from(["a", "b", "c", "d"].map(Arc::<str>::from)),
        disjoint_training_subjects: true,
        crossfit_fold_ownership_valid: false,
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
        regret: None,
    }
}

fn outcome_table() -> TabularData {
    TabularData::from_f64_columns([("outcome", &[3.0, 5.0, 2.0, 1.0][..])]).unwrap()
}

#[test]
fn binary_policy_posterior_runs_through_the_study_flow_and_matches_hand_scores() {
    let data = outcome_table();
    let ctx = ExecutionContext::for_tests(21);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(policy_query()))
        .inference(bayesian())
        .build()
        .unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.estimand.method.as_ref(), "randomized.policy_value");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.binary_policy_value"));
    assert_eq!(result.logical_plan.plan_id.as_ref(), "bayesian_policy_value");
    assert!(matches!(result.estimate, PrimaryEstimate::NotAnEffect));
    // Unlicensed: no support status and no calibration claim.
    assert_eq!(result.support_status, None);
    let policy = result.policy_value().unwrap();
    assert_eq!(policy.policy_value, 4.0);
    assert_eq!(policy.reference_value, 1.5);
    assert_eq!(policy.incremental_value, 2.5);
    assert_eq!(
        policy.uncertainty.as_ref(),
        "bayesian_bootstrap_posterior_known_randomization_no_interval"
    );
    assert!(policy.policy_interval_95.is_none() && policy.incremental_interval_95.is_none());
    let posterior = result.posterior.as_ref().expect("posterior");
    assert_eq!(posterior.draws.n_draws, DRAWS);
    let incremental = posterior.draws.column(0).unwrap();
    let mean = incremental.iter().sum::<f64>() / DRAWS as f64;
    assert!((mean - 2.5).abs() < 0.1, "{mean}");
    let expected = dirichlet_sd(&[6.0, 10.0, -4.0, -2.0]);
    let sd =
        (incremental.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (DRAWS as f64 - 1.0)).sqrt();
    assert!((sd - expected).abs() < 0.06 * expected, "{sd} {expected}");
    // Paired draws: incremental = policy - reference.
    let policy_draws = posterior.draws.column(1).unwrap();
    let reference_draws = posterior.draws.column(2).unwrap();
    for ((d, p), r) in incremental.iter().zip(policy_draws).zip(reference_draws) {
        assert!((d - (p - r)).abs() < 1e-9);
    }
    // Retained result round trip through the contract artifact and its independent loader.
    let bytes = prepared.encode_contracted_result(&result, "bayesian-policy", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.policy_value.as_ref().unwrap();
    assert_eq!(wire.policy_value, 4.0);
    assert_eq!(wire.uncertainty, "bayesian_bootstrap_posterior_known_randomization_no_interval");
    assert_eq!(wire.graphless_support_status, None);
    let (_, flat) =
        antecedent_io::decode_causal_posterior_bytes(body.posterior_artifact.as_ref().unwrap())
            .unwrap();
    assert_eq!(flat.len(), 3 * DRAWS);
    let reloaded_mean = flat[..DRAWS].iter().sum::<f64>() / DRAWS as f64;
    assert!((reloaded_mean - mean).abs() < 1e-9);
    // A fabricated interval on the posterior route is refused.
    let mut forged = body.clone();
    forged.policy_value.as_mut().unwrap().policy_interval_95 = Some([0.0, 1.0]);
    forged.policy_value.as_mut().unwrap().incremental_interval_95 = Some([0.0, 1.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(&forged, vec!["outcome".into()], "forged")
            .is_err()
    );
}

#[test]
fn binary_policy_posterior_names_its_estimator_and_refuses_unread_shapes() {
    let data = outcome_table();
    let ctx = ExecutionContext::for_tests(22);
    // Selecting the estimator by id is the same route.
    let selected = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(policy_query()))
        .estimator(EstimatorId::BayesianBinaryPolicyValue)
        .inference(bayesian())
        .build()
        .unwrap();
    assert!(selected.prepare(&ctx).is_ok());
    // A Bayesian estimator under Frequentist inference is refused by code.
    let mismatch = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(policy_query()))
        .estimator(EstimatorId::BayesianBinaryPolicyValue)
        .build()
        .err()
        .expect("mismatch");
    assert_eq!(mismatch.reason_code(), Some("estimator_inference_mismatch"));
    // Predictions not declared subject-excluded are refused with the next attempt.
    let mut leaked = policy_query();
    leaked.mu0 = Arc::from([1.0; 4]);
    leaked.mu1 = Arc::from([2.0; 4]);
    leaked.disjoint_training_subjects = false;
    let refused = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(leaked))
        .inference(bayesian())
        .build()
        .unwrap()
        .prepare(&ctx)
        .err()
        .expect("refusal");
    assert_eq!(refused.reason_code(), Some("invalid_argument"));
    assert!(refused.to_string().contains("Next attempt"), "{refused}");
    // Held-out predictions give the augmented scores [4, 8, 2, 2].
    let mut augmented = policy_query();
    augmented.mu0 = Arc::from([1.0; 4]);
    augmented.mu1 = Arc::from([2.0; 4]);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(augmented))
        .inference(bayesian())
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let posterior = result.posterior.as_ref().unwrap();
    let policy_draws = posterior.draws.column(1).unwrap();
    let sd = {
        let mean = policy_draws.iter().sum::<f64>() / DRAWS as f64;
        (policy_draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (DRAWS as f64 - 1.0)).sqrt()
    };
    let expected = dirichlet_sd(&[4.0, 8.0, 2.0, 2.0]);
    assert!((sd - expected).abs() < 0.06 * expected, "{sd} {expected}");
}

fn regime_query() -> LongitudinalRegimeQuery {
    LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method: antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: Some(Arc::from([2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0])),
        observation_history: Some(Arc::from([true; 8])),
        prediction_fold_ids: Some(Arc::from([0, 1, 0, 1])),
        periods: 2,
        treatment_history: Arc::from([true, true, true, false, false, true, false, false]),
        regime_actions: Arc::from([true; 8]),
        treatment_probabilities: Arc::from([0.5; 8]),
        censoring_probabilities: Arc::from([1.0; 8]),
        outcome_observed: Arc::from([true; 4]),
        subject_ids: Arc::from(["a", "b", "c", "d"].map(Arc::<str>::from)),
        fold_ids: Arc::from([0, 1, 0, 1]),
        excluded_fold_predictions: true,
        probabilities_known_by_design: true,
        minimum_probability: 0.1,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    }
}

#[test]
fn sequential_regime_posterior_runs_through_the_study_flow_and_matches_hand_scores() {
    // Hand backward recursion: subject scores 12, 4, 1, 1; value 4.5.
    let data = TabularData::from_f64_columns([("outcome", &[5.0, 4.0, 3.0, 0.0][..])]).unwrap();
    let ctx = ExecutionContext::for_tests(24);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(regime_query()))
        .inference(bayesian())
        .build()
        .unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.sequential_regime_value"));
    assert!(matches!(result.estimate, PrimaryEstimate::NotAnEffect));
    assert_eq!(result.support_status, None);
    let regime = result.longitudinal_regime().unwrap();
    assert_eq!(&*regime.method, "sequential_dr");
    assert!((regime.value - 4.5).abs() < 1e-12);
    assert_eq!(
        &*regime.uncertainty,
        "bayesian_bootstrap_posterior_known_sequential_randomization_no_interval"
    );
    assert!(regime.value_interval_95.is_none());
    let posterior = result.posterior.as_ref().expect("posterior");
    let draws = posterior.draws.column(0).unwrap();
    assert_eq!(draws.len(), DRAWS);
    let mean = draws.iter().sum::<f64>() / DRAWS as f64;
    assert!((mean - 4.5).abs() < 0.15, "{mean}");
    let sd = (draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (DRAWS as f64 - 1.0)).sqrt();
    let expected = dirichlet_sd(&[12.0, 4.0, 1.0, 1.0]);
    assert!((sd - expected).abs() < 0.06 * expected, "{sd} {expected}");
    let bytes = prepared.encode_contracted_result(&result, "bayesian-regime", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().method, "sequential_dr");
    assert_eq!(body.longitudinal_regime.as_ref().unwrap().graphless_support_status, None);
    let (_, flat) =
        antecedent_io::decode_causal_posterior_bytes(body.posterior_artifact.as_ref().unwrap())
            .unwrap();
    assert_eq!(flat.len(), DRAWS);
    let mut forged = body.clone();
    forged.longitudinal_regime.as_mut().unwrap().value_interval_95 = Some([0.0, 9.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(&forged, vec!["outcome".into()], "forged")
            .is_err()
    );
}

#[test]
fn sequential_regime_posterior_refuses_other_methods_and_unowned_predictions() {
    let data = TabularData::from_f64_columns([("outcome", &[5.0, 4.0, 3.0, 0.0][..])]).unwrap();
    let ctx = ExecutionContext::for_tests(25);
    let mut ipw = regime_query();
    ipw.method = antecedent_core::LongitudinalRegimeMethod::Ipw;
    ipw.q_predictions = None;
    ipw.observation_history = None;
    ipw.prediction_fold_ids = None;
    ipw.excluded_fold_predictions = false;
    let refused = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(ipw))
        .inference(bayesian())
        .build()
        .unwrap()
        .prepare(&ctx)
        .err()
        .expect("refusal");
    assert_eq!(refused.reason_code(), Some("route_not_supported"));
    assert!(refused.to_string().contains("Next attempt"), "{refused}");
    let selected = Study::tabular(data)
        .query(CausalQuery::LongitudinalRegime(regime_query()))
        .estimator(EstimatorId::BayesianSequentialRegimeValue)
        .build()
        .err()
        .expect("mismatch");
    assert_eq!(selected.reason_code(), Some("estimator_inference_mismatch"));
}
