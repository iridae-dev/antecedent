//! Original public five-fold AIPW fit and known-weight retarget SE diagnostics.
//! IID Z±1,T~Bernoulli(.5),Y=1+.3Z+T(2+.5Z), no outcome residual.
//! True scores are2+.5Z: unweighted mean2/IF variance.25; known w(Z)=3/1
//! ratio target2.25 with IF±.375/variance.140625. Original exact arm OLS models,
//! five-fold propensity/outcome fit sets, actual frozen-score consumer, no new CI.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, reason = "bounded diagnostic dimensions")]
#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use antecedent::analysis::recalc_cell::{
    ScoreQuantity, ScoreResumeRequest, ScoreResumeSession, execute_resumed_retarget,
    freeze_crossfit_scores,
};
use antecedent::analysis::recalc_receipt::{
    LawValue, RecalcRequest, RecalcSession, TargetWeights, UtilitySpec, execute_with_receipt,
};
use antecedent_core::recalc::RetargetSupport;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::AipwAte;
use antecedent_io::frozen_scores_artifact::FrozenScoreTable;
use antecedent_stats::fit_counts::count_least_squares_solves;
use serde_json::json;

fn request(n: usize, seed: u64) -> RecalcRequest {
    let mut generator = candidate::Generator::new(seed);
    let (mut t, mut y, mut z) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let zv = 2. * generator.binary(0.5) as f64 - 1.;
        let tv = generator.binary(0.5) as f64;
        t.push(tv);
        z.push(zv);
        y.push(1. + 0.3 * zv + tv * (2. + 0.5 * zv));
    }
    RecalcRequest {
        columns: vec![("t".into(), t), ("y".into(), y), ("z".into(), z)],
        edges: vec![(2, 0), (2, 1), (0, 1)],
        treatment: 0,
        outcome: 1,
        estimator: AipwAte::new().with_bootstrap_replicates(0),
        target: None,
        utility: UtilitySpec { benefit_per_unit: 1., cost: 0. },
    }
}
fn weights(r: &RecalcRequest) -> TargetWeights {
    TargetWeights {
        weights: r.columns[2].1.iter().map(|z| if *z > 0. { 3. } else { 1. }).collect(),
        depends_on: vec![VariableId::from_raw(2)],
    }
}
struct Sample {
    laws: [LawValue; 2],
    fold_fit_sets: u64,
    kernel_solves: u64,
}
fn portable(session: &RecalcSession, r: &RecalcRequest, n: usize, law: LawValue) -> Option<()> {
    let artifact = freeze_crossfit_scores(session, n as u64, RetargetSupport::Licensed).ok()?;
    let bytes = artifact.to_bytes("generic-aipw-known-target-se").ok()?;
    let checked = FrozenScoreTable::from_bytes(&bytes, Some(artifact.identity())).ok()?;
    let row_ids = checked.score_table().ok()?.row_index.to_vec();
    let mut fresh = ScoreResumeSession::resume_from_scores(&checked).ok()?;
    let resume = ScoreResumeRequest {
        n_variables: 3,
        edges: r.edges.clone(),
        target: r.target.clone(),
        target_row_ids: Some(row_ids),
        utility: r.utility,
        quantity: ScoreQuantity::AverageEffect,
        changed_inputs: vec![],
    };
    let (result, solves) =
        count_least_squares_solves(|| execute_resumed_retarget(&mut fresh, &resume));
    let out = result.map_err(|error| eprintln!("generic portable: {error}")).ok()?;
    if solves != 0
        || out.receipt.totals().fold_fits != 0
        || out.receipt.totals().model_fits != 0
        || (out.law.ate - law.ate).abs() > 1e-10
        || (out.law.std_error - law.std_error).abs() > 1e-10
    {
        return None;
    }
    Some(())
}
fn fit(n: usize, seed: u64) -> Option<Sample> {
    let mut r = request(n, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let mut session = RecalcSession::new();
    let (result, kernel_solves) =
        count_least_squares_solves(|| execute_with_receipt(&mut session, &r, &ctx));
    let first = result.map_err(|error| eprintln!("generic first fit: {error}")).ok()?;
    // Ten fold fit sets are five propensity fits plus five two-arm outcome
    // sets. Raw kernel LS solves also include iterative propensity IRLS steps;
    // their independently observed count is reported, never equated to models.
    if first.receipt.totals().fold_fits != 10 || kernel_solves < 10 {
        return None;
    }
    let table = session.score_table()?;
    if table.n_folds != 5
        || table.n_rows != n
        || table.fold_ids.iter().any(|f| *f >= 5)
        || table.adjustment_set.as_ref() != [VariableId::from_raw(2)]
    {
        return None;
    }
    let (unchanged, unchanged_solves) =
        count_least_squares_solves(|| execute_with_receipt(&mut session, &r, &ctx));
    if unchanged_solves != 0 || unchanged.ok()?.receipt.totals().total() != 0 {
        return None;
    }
    r.target = Some(weights(&r));
    let (result, retarget_solves) =
        count_least_squares_solves(|| execute_with_receipt(&mut session, &r, &ctx));
    let weighted = result.map_err(|error| eprintln!("generic weighted: {error}")).ok()?;
    if retarget_solves != 0
        || weighted.receipt.totals().fold_fits != 0
        || weighted.receipt.totals().score_computations != 0
    {
        return None;
    }
    portable(&session, &r, n, weighted.law).or_else(|| {
        eprintln!("generic portable stage failed");
        None
    })?;
    Some(Sample {
        laws: [first.law, weighted.law],
        fold_fit_sets: first.receipt.totals().fold_fits,
        kernel_solves,
    })
}
fn independent_laws(z: &[f64]) -> [LawValue; 2] {
    [false, true].map(|weighted| {
        let w: Vec<_> = z.iter().map(|v| if weighted && *v > 0. { 3. } else { 1. }).collect();
        let total = w.iter().sum::<f64>();
        let mean = z.iter().zip(&w).map(|(z, w)| (2. + 0.5 * z) * w).sum::<f64>() / total;
        let variance = z.len() as f64 / (z.len() - 1) as f64
            * z.iter()
                .zip(&w)
                .map(|(z, w)| (w / total * (2. + 0.5 * z - mean)).powi(2))
                .sum::<f64>();
        LawValue { ate: mean, std_error: variance.sqrt() }
    })
}

#[test]
fn generic_aipw_known_weight_exact_scores_and_frozen_consumer_plumbing() {
    // One engineering frame, no repeated statistical measurement.
    let (n, seed) = (1200, 97);
    let sample =
        fit(n, seed).expect("original five-fold fit and independent frozen retarget must execute");
    let r = request(n, seed);
    let expected = independent_laws(&r.columns[2].1);
    for (actual, oracle) in sample.laws.iter().zip(expected) {
        assert!((actual.ate - oracle.ate).abs() < 1e-9);
        assert!((actual.std_error - oracle.std_error).abs() < 1e-9);
    }
    assert_eq!(sample.fold_fit_sets, 10);
    assert!(sample.kernel_solves >= 10);
}

#[test]
#[ignore = "calibration: final measurement only"]
fn generic_aipw_five_fold_known_target_score_precision() {
    let n = calibration::grid_n(1000);
    let seed = calibration::grid_seed(0xC2A5_0001);
    let attempts: Vec<_> = (0..calibration::n_sim().max(2000))
        .map(|rep| fit(n, seed.wrapping_add(u64::from(rep))))
        .collect();
    let failed = attempts.iter().filter(|a| a.is_none()).count();
    if failed > 0 {
        println!(
            "INFERENCE_DIAGNOSTIC_RECORD {}",
            json!({
                "test":"generic_aipw_five_fold_known_target_score_precision", "rows":n,
                "attempts":attempts.len(), "failed":failed, "pass":false
            })
        );
    }
    assert_eq!(
        failed, 0,
        "all fullfit/retarget/independent-consumer attempts remain in the denominator"
    );
    let samples: Vec<_> = attempts.iter().flatten().collect();
    let count = attempts.len() as f64;
    for (j, truth, if_variance) in [(0, 2., 0.25), (1, 2.25, 0.140_625)] {
        let mean = samples.iter().map(|s| s.laws[j].ate).sum::<f64>() / count;
        let empirical =
            samples.iter().map(|s| (s.laws[j].ate - mean).powi(2)).sum::<f64>() / (count - 1.);
        let reported = samples.iter().map(|s| s.laws[j].std_error.powi(2)).sum::<f64>() / count;
        let expected_variance = if_variance / n as f64;
        let bias = (mean - truth).abs() / (expected_variance / count).sqrt();
        let pass = (empirical / expected_variance - 1.).abs() <= 0.15
            && (reported / expected_variance - 1.).abs() <= 0.15
            && bias <= 5.;
        println!(
            "INFERENCE_DIAGNOSTIC_RECORD {}",
            json!({
            "test":"generic_aipw_five_fold_known_target_score_precision","rows":n,
            "seed":seed,"weighted":j==1,"truth":truth,"truth_if_variance":if_variance,
            "attempts":attempts.len(),"failed":failed,
            "actual_successful_least_squares_kernel_solves":samples.iter().map(|s|s.kernel_solves).sum::<u64>(),
            "receipt_propensity_and_arm_fold_fit_sets":samples.iter().map(|s|s.fold_fit_sets).sum::<u64>(),
            "retarget_and_independent_consumer_fits":0,"reported_mean_variance":reported,
            "independent_sampling_variance":empirical,"bias_mcse":bias,
            "variance_relative_error_bound":0.15,"bias_mcse_bound":5.,"pass":pass,
            "activation":"none; known empirical-target ratio SE diagnostic only; no new CI"})
        );
        assert!(pass, "generic five-fold AIPW target score SE precision failed");
    }
}
