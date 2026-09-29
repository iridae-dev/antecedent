//! Numerical and refusal evidence for Bayesian quadratic-basis g-computation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "small deterministic numerical fixture"
)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_learn::{
    BasisTargetPopulation, BayesianBasisGComputation, BayesianBasisSpec, LearnError,
    LearnerProvenance, PosteriorPrediction, PosteriorPredictionProvenance, PredictionTask,
};

fn quadratic_fixture(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, Vec<f64>) {
    let mut treatment = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut truth_cate = Vec::with_capacity(n);
    for i in 0..n {
        let level = (i % 20) as f64 / 10.0 - 0.95;
        let t = f64::from(((i / 2) % 2) as u8);
        let noise = 0.08 * (i as f64 * 0.37).sin();
        let y = 1.0 + 2.0 * t + 0.7 * level + 1.4 * t * level + 0.35 * level * level + noise;
        treatment.push(t);
        z.push(level);
        outcome.push(y);
        truth_cate.push(2.0 + 1.4 * level);
    }
    let mean_z = z.iter().sum::<f64>() / n as f64;
    (treatment, outcome, z, 2.0 + 1.4 * mean_z, truth_cate)
}

#[test]
fn basis_gcomp_matches_independent_quadratic_ate_and_cate_reference() {
    let n = 400;
    let (treatment, outcome, z, ate_truth, cate_truth) = quadratic_fixture(n);
    let rows: Vec<u32> = (1000..1000 + n as u32).collect();
    let folds: Vec<u16> = (0..n).map(|i| (i % 5) as u16).collect();
    let result = BayesianBasisGComputation
        .fit(
            &treatment,
            &outcome,
            &z,
            1,
            Some(&rows),
            Some(&folds),
            BasisTargetPopulation::AllObserved,
            BayesianBasisSpec { prior_sd: 8.0, n_draws: 600, seed: 17 },
            &ExecutionContext::for_tests(1),
        )
        .unwrap();

    assert_eq!(result.predictions.row_ids.as_ref(), rows);
    assert_eq!(result.predictions.fold_ids.as_ref().unwrap().as_ref(), folds);
    assert_eq!(
        result.predictions.provenance.model_id.as_ref(),
        "bayesian_basis_gcomp.quadratic_tz_z2"
    );
    assert!(result.predictions.provenance.prior_id.contains("sd_8"));
    assert_eq!(result.predictions.draw_count, 600);
    assert_eq!(result.ate_draws.len(), 600);
    assert!((result.ate_mean - ate_truth).abs() < 0.06, "ATE {} vs {ate_truth}", result.ate_mean);

    let effects: Vec<f64> = (0..n)
        .map(|row| {
            (0..result.predictions.draw_count)
                .map(|draw| result.predictions.draw_target(draw, 2).unwrap()[row])
                .sum::<f64>()
                / result.predictions.draw_count as f64
        })
        .collect();
    let cate_rmse = (effects
        .iter()
        .zip(&cate_truth)
        .map(|(estimate, truth)| (estimate - truth).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert!(cate_rmse < 0.07, "CATE RMSE {cate_rmse}");
    assert_eq!(result.predictions.diagnostics.fold_count, Some(5));
    assert_eq!(BayesianBasisGComputation.task(), PredictionTask::Regression);
    assert_eq!(BayesianBasisGComputation.estimator_id(), "bayesian.basis_gcomp");
}

#[test]
fn basis_gcomp_refuses_unlicensed_population_and_non_binary_or_degenerate_treatment() {
    let (treatment, outcome, z, _, _) = quadratic_fixture(80);
    let ctx = ExecutionContext::for_tests(1);
    let spec = BayesianBasisSpec { n_draws: 8, ..Default::default() };
    let other = BayesianBasisGComputation.fit(
        &treatment,
        &outcome,
        &z,
        1,
        None,
        None,
        BasisTargetPopulation::Other,
        spec,
        &ctx,
    );
    assert!(matches!(other, Err(LearnError::Unsupported { .. })));

    let continuous_treatment = vec![0.5; treatment.len()];
    let invalid = BayesianBasisGComputation.fit(
        &continuous_treatment,
        &outcome,
        &z,
        1,
        None,
        None,
        BasisTargetPopulation::AllObserved,
        spec,
        &ctx,
    );
    assert!(matches!(invalid, Err(LearnError::Unsupported { .. })));

    let one_arm = vec![1.0; treatment.len()];
    let degenerate = BayesianBasisGComputation.fit(
        &one_arm,
        &outcome,
        &z,
        1,
        None,
        None,
        BasisTargetPopulation::AllObserved,
        spec,
        &ctx,
    );
    assert!(matches!(degenerate, Err(LearnError::Unsupported { .. })));
}

#[test]
fn posterior_prediction_preserves_joint_draw_row_and_fold_identity() {
    let provenance = PosteriorPredictionProvenance {
        model_id: Arc::from("joint_fixture"),
        prior_id: Arc::from("prior_fixture"),
        model: LearnerProvenance {
            spec: "fixture".into(),
            implementation: "test".into(),
            version: "1".into(),
        },
    };
    let prediction = PosteriorPrediction::new(
        2,
        vec![Arc::from("outcome"), Arc::from("propensity")],
        vec![10, 20],
        Some(vec![0, 1]),
        vec![1.0, 2.0, 0.2, 0.8, 3.0, 4.0, 0.3, 0.7],
        provenance.clone(),
    )
    .unwrap();
    assert_eq!(prediction.draw_target(0, 0).unwrap(), &[1.0, 2.0]);
    assert_eq!(prediction.draw_target(0, 1).unwrap(), &[0.2, 0.8]);
    assert_eq!(prediction.draw_target(1, 0).unwrap(), &[3.0, 4.0]);
    assert_eq!(prediction.row_ids.as_ref(), &[10, 20]);
    assert_eq!(prediction.fold_ids.as_ref().unwrap().as_ref(), &[0, 1]);
    assert_eq!(prediction.diagnostics.fold_count, Some(2));

    let duplicate_rows = PosteriorPrediction::new(
        1,
        vec![Arc::from("y")],
        vec![7, 7],
        None,
        vec![1.0, 2.0],
        provenance.clone(),
    );
    assert!(matches!(duplicate_rows, Err(LearnError::Shape { .. })));
    let ragged_folds = PosteriorPrediction::new(
        1,
        vec![Arc::from("y")],
        vec![7, 8],
        Some(vec![0]),
        vec![1.0, 2.0],
        provenance,
    );
    assert!(matches!(ragged_folds, Err(LearnError::Shape { .. })));
}

#[test]
fn basis_all_observed_credible_interval_has_scoped_repeated_sampling_coverage() {
    let mut state = 0x52A9_8D31_7E64_20B1_u64;
    let mut uniform = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let n = 96;
    let repetitions = 100_usize;
    let mut covered = 0;
    for rep in 0..repetitions {
        let rep_seed = u64::try_from(rep).expect("repetition index fits u64");
        let z: Vec<f64> = (0..n).map(|row| -0.99 + 1.98 * row as f64 / (n - 1) as f64).collect();
        let treatment: Vec<f64> = (0..n).map(|_| f64::from(uniform() < 0.5)).collect();
        let outcome: Vec<f64> = (0..n)
            .map(|row| {
                let u1 = uniform().max(f64::MIN_POSITIVE);
                let u2 = uniform();
                let noise = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                0.4 + 1.5 * treatment[row]
                    + 0.7 * z[row]
                    + 0.6 * treatment[row] * z[row]
                    + 0.3 * z[row] * z[row]
                    + noise
            })
            .collect();
        let truth = 1.5 + 0.6 * z.iter().sum::<f64>() / n as f64;
        let fit = BayesianBasisGComputation
            .fit(
                &treatment,
                &outcome,
                &z,
                1,
                None,
                None,
                BasisTargetPopulation::AllObserved,
                BayesianBasisSpec { prior_sd: 10.0, n_draws: 299, seed: rep_seed + 7 },
                &ExecutionContext::for_tests(rep_seed + 9),
            )
            .unwrap();
        let mut draws = fit.ate_draws.to_vec();
        draws.sort_by(f64::total_cmp);
        let lo = draws[14];
        let hi = draws[284];
        covered += usize::from(lo <= truth && truth <= hi);
    }
    let covered = u32::try_from(covered).expect("coverage count fits u32");
    let repetitions = u32::try_from(repetitions).expect("repetition count fits u32");
    let rate = f64::from(covered) / f64::from(repetitions);
    println!("Bayesian basis AllObserved ATE: {covered}/{repetitions} 90% intervals covered");
    assert!((0.78..=0.98).contains(&rate), "90% interval coverage was {rate:.3}");
}
