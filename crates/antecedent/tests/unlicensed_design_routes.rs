//! The observational sequential contrast and the within-period history match through the public
//! study flow, against independent references.
//!
//! The sequential panel is the fixed linear congruential stream whose numpy reference values are
//! pinned in the estimator's own tests. The history fixture carries a brute-force nearest-donor
//! reference with the Abadie-Imbens variance computed from per-unit coefficients. Both routes are
//! unlicensed: they hold no support status and the matching posterior is a typed refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "a 53-bit integer converts exactly and the pinned scores are exact small numbers"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, EstimatorId, InferenceMode, PrimaryEstimate, Study};
use antecedent_core::{CausalQuery, HistoryMatchQuery, SequentialContrastQuery, VariableId};
use antecedent_data::TabularData;

const SUBJECTS: usize = 80;
const DRAWS: usize = 4000;

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / 9_007_199_254_740_992.0
}

/// Columns l0, a0, l1, a1, y, in that order.
fn panel() -> TabularData {
    let mut state = 12_345_u64;
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..SUBJECTS {
        let u1 = uniform(&mut state);
        let u2 = uniform(&mut state);
        let u3 = uniform(&mut state);
        let u4 = uniform(&mut state);
        let u5 = uniform(&mut state);
        let l0 = 2.0 * u1 - 1.0;
        let a0 = if u2 < 0.5 + 0.2 * l0 { 1.0 } else { 0.0 };
        let l1 = 0.5 * l0 + 0.8 * a0 + (2.0 * u3 - 1.0) * 0.5;
        let a1 = if u4 < 0.5 + 0.1 * (l1 - 0.4) { 1.0 } else { 0.0 };
        ys.push(1.0 + 0.5 * l0 + 1.0 * a0 + 1.5 * a1 + 0.4 * l1 + (2.0 * u5 - 1.0));
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
    }
    TabularData::from_f64_columns([
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
    .unwrap()
}

fn contrast_query() -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: VariableId::from_raw(4),
        first_treatment: VariableId::from_raw(1),
        second_treatment: VariableId::from_raw(3),
        baseline: Arc::from([VariableId::from_raw(0)]),
        intermediate: Arc::from([VariableId::from_raw(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: (0..SUBJECTS).map(|row| u32::from(matches!(row % 4, 1 | 2))).collect(),
        fold_count: 2,
        min_probability: 0.05,
    }
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "actual {actual} expected {expected}");
}

#[test]
fn sequential_contrast_wald_matches_the_pinned_reference_and_round_trips() {
    let data = panel();
    let ctx = ExecutionContext::for_tests(41);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::SequentialContrast(contrast_query()))
        .build()
        .unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("sequential.crossfit_contrast.wald"));
    assert!(matches!(result.estimate, PrimaryEstimate::NotAnEffect));
    assert_eq!(result.support_status, None);
    assert!(result.posterior.is_none());
    let fit = result.sequential_contrast().unwrap();
    close(fit.active_mean, 3.577_003_395_896_987_6, 1.0e-4);
    close(fit.control_mean, 1.346_150_090_756_641_5, 1.0e-4);
    close(fit.contrast, 2.230_853_305_140_346, 1.0e-4);
    let scale = 0.689_242_100_473_076_1;
    close(fit.standard_error.unwrap(), scale, 1.0e-4);
    let half = 1.959_963_984_540_054 * scale;
    let [lower, upper] = fit.interval.unwrap();
    close(lower, 2.230_853_305_140_346 - half, 1.0e-3);
    close(upper, 2.230_853_305_140_346 + half, 1.0e-3);
    assert_eq!(&*fit.uncertainty, "normal_interval_calibration_unmeasured");
    assert_eq!(fit.contrast_scores.len(), SUBJECTS);

    // The retained artifact is decoded by the independent loader and recomputed.
    let bytes = prepared.encode_contracted_result(&result, "sequential-wald", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.sequential_contrast.as_ref().unwrap();
    assert!(!wire.licensed && wire.calibration == "unmeasured");
    close(wire.contrast, fit.contrast, 1.0e-12);
    let mut shifted = body.clone();
    shifted.sequential_contrast.as_mut().unwrap().interval = Some([0.0, 100.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &shifted,
            header.variable_names.clone(),
            "forged-interval"
        )
        .is_err()
    );
    let mut forged_mean = body.clone();
    forged_mean.sequential_contrast.as_mut().unwrap().contrast += 0.5;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged_mean,
            header.variable_names,
            "forged-contrast"
        )
        .is_err()
    );
}

#[test]
fn sequential_contrast_posterior_is_the_modular_dirichlet_posterior() {
    let data = panel();
    let ctx = ExecutionContext::for_tests(42);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::SequentialContrast(contrast_query()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .build()
        .unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("bayesian.sequential_crossfit_contrast")
    );
    let fit = result.sequential_contrast().unwrap();
    assert!(fit.interval.is_none() && fit.standard_error.is_none());
    let posterior = result.posterior.as_ref().unwrap();
    let draws = posterior.draws.column(0).unwrap();
    let n = SUBJECTS as f64;
    let mean = draws.iter().sum::<f64>() / DRAWS as f64;
    close(mean, fit.contrast, 0.05);
    let sd = (draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (DRAWS as f64 - 1.0)).sqrt();
    let expected = 0.689_242_100_473_076_1 * ((n - 1.0) / (n + 1.0)).sqrt();
    close(sd, expected, 0.06 * expected);
    let (active, control) =
        (posterior.draws.column(1).unwrap(), posterior.draws.column(2).unwrap());
    for ((d, a), c) in draws.iter().zip(active).zip(control) {
        close(*d, a - c, 1.0e-9);
    }
    let bytes = prepared.encode_contracted_result(&result, "sequential-bayes", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.sequential_contrast.as_ref().unwrap();
    assert_eq!(wire.uncertainty, "modular_bayesian_bootstrap_posterior_no_interval");
    let mut forged = body.clone();
    forged.sequential_contrast.as_mut().unwrap().interval = Some([0.0, 1.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(&forged, header.variable_names, "forged")
            .is_err()
    );
}

#[test]
fn sequential_contrast_refusals_are_typed() {
    let data = panel();
    let ctx = ExecutionContext::for_tests(43);
    let mut floor = contrast_query();
    floor.min_probability = 0.5;
    let refused = Study::tabular(data.clone())
        .query(CausalQuery::SequentialContrast(floor))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .err()
        .expect("positivity");
    assert_eq!(refused.reason_code(), Some("positivity_violated"));
    let mut repeated = contrast_query();
    repeated.first_treatment = repeated.second_treatment;
    assert!(repeated.validate().is_err());
    let mismatch = Study::tabular(data)
        .query(CausalQuery::SequentialContrast(contrast_query()))
        .estimator(EstimatorId::BayesianSequentialCrossfitContrast)
        .build()
        .err()
        .expect("mismatch");
    assert_eq!(mismatch.reason_code(), Some("estimator_inference_mismatch"));
}

/// Columns h1, h2, y, treated, period, weight over nine units and three calendar periods.
fn history_table() -> TabularData {
    let rows: [(f64, [f64; 2], f64, f64, f64); 9] = [
        (1.0, [1.0, 3.0], 10.0, 1.0, 1.0),
        (1.0, [1.0, 1.0], 4.0, 0.0, 1.0),
        (1.0, [3.0, 3.0], 6.0, 0.0, 1.0),
        (1.0, [2.8, 3.0], 9.0, 1.0, 2.0),
        (2.0, [2.0, 2.0], 5.0, 0.0, 1.0),
        (2.0, [6.0, 6.0], 7.0, 0.0, 1.0),
        (2.0, [5.0, 6.0], 12.0, 1.0, 1.5),
        (2.0, [9.0, 9.0], 20.0, 1.0, 1.0),
        (3.0, [1.0, 1.0], 3.0, 1.0, 1.0),
    ];
    let column = |f: &dyn Fn(&(f64, [f64; 2], f64, f64, f64)) -> f64| -> Vec<f64> {
        rows.iter().map(f).collect()
    };
    let (h1, h2) = (column(&|r| r.1[0]), column(&|r| r.1[1]));
    let (y, treated) = (column(&|r| r.2), column(&|r| r.3));
    let (period, weight) = (column(&|r| r.0), column(&|r| r.4));
    TabularData::from_f64_columns([
        ("h1", h1.as_slice()),
        ("h2", h2.as_slice()),
        ("y", y.as_slice()),
        ("treated", treated.as_slice()),
        ("period", period.as_slice()),
        ("weight", weight.as_slice()),
    ])
    .unwrap()
}

fn history_query(caliper: Option<f64>) -> HistoryMatchQuery {
    HistoryMatchQuery {
        outcome: VariableId::from_raw(2),
        treated: VariableId::from_raw(3),
        history: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        period: Some(VariableId::from_raw(4)),
        weight: Some(VariableId::from_raw(5)),
        caliper,
    }
}

#[test]
fn period_history_match_matches_the_brute_force_reference_and_round_trips() {
    let data = history_table();
    let ctx = ExecutionContext::for_tests(44);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::HistoryMatch(history_query(Some(2.0))))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("matching.period_history_contrast"));
    assert_eq!(result.support_status, None);
    let fit = result.history_match().unwrap();
    assert_eq!(&*fit.matched_queries, &[0, 3, 6]);
    assert_eq!(&*fit.matched_donors, &[1, 2, 5]);
    assert_eq!(&*fit.unmatched_treated, &[7, 8]);
    assert_eq!(&*fit.differences, &[6.0, 3.0, 5.0]);
    close(fit.weight_total, 4.5, 1.0e-12);
    close(fit.contrast, 4.333_333_333_333_333, 1.0e-12);
    close(fit.standard_error.unwrap(), 0.931_408_271_125_701_1, 1.0e-12);
    let bytes = prepared.encode_contracted_result(&result, "history-match", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.history_match.as_ref().unwrap();
    assert_eq!(wire.matched_donors, vec![1, 2, 5]);
    assert!(!wire.licensed);
    assert_eq!(body.interval_lower, None);
    let mut forged_se = body.clone();
    forged_se.history_match.as_mut().unwrap().standard_error = Some(0.01);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged_se,
            header.variable_names.clone(),
            "forged-se"
        )
        .is_err()
    );
    let mut forged_contrast = body.clone();
    forged_contrast.history_match.as_mut().unwrap().contrast = 9.0;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged_contrast,
            header.variable_names,
            "forged-contrast"
        )
        .is_err()
    );

    // Without a caliper a donor is reused.
    let open = Study::tabular(data.clone())
        .query(CausalQuery::HistoryMatch(history_query(None)))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let reuse = open.history_match().unwrap();
    assert_eq!(&*reuse.matched_donors, &[1, 2, 5, 5]);
    close(reuse.contrast, 5.909_090_909_090_908, 1.0e-12);
    close(reuse.standard_error.unwrap(), 2.348_609_256_457_778, 1.0e-12);
}

#[test]
fn matching_posteriors_are_the_typed_boundary_through_the_study_flow() {
    let data = history_table();
    let bayesian = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16));
    let refused = Study::tabular(data)
        .query(CausalQuery::HistoryMatch(history_query(None)))
        .inference(bayesian)
        .build()
        .err()
        .expect("history boundary");
    assert_eq!(refused.reason_code(), Some("route_not_supported"));
    let message = refused.to_string();
    assert!(message.contains("no likelihood"), "{message}");
    assert!(message.contains("matched within calendar periods"), "{message}");
    assert!(message.contains("next attempt: declare a Bayesian outcome model"), "{message}");

    let (data, dag, query) = fixtures::confounded_scm(64, 3);
    for estimator in [EstimatorId::PropensityMatching, EstimatorId::DistanceMatching] {
        let error = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query.clone())
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
            .estimator(estimator)
            .bootstrap_replicates(0)
            .build()
            .err()
            .expect("static boundary");
        assert_eq!(error.reason_code(), Some("route_not_supported"));
        assert!(error.to_string().contains("covariates or a propensity score"), "{error}");
    }
}

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures;
