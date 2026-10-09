//! Actual retained cell/DML/DR score SE precision, including independent frozen consumers.
//! IID binary Z±1 and randomized treatment arms; Y(0)=1+0.3Z and
//! Y(1)=Y(0)+2+0.5Z (cell interaction uses the same heterogeneous contrast).
//! Linear arm models are exact, with zero outcome residual. The true score is2+0.5Z.
//! Unweighted effect2 has IF variance.25; known weighting w(Z)=3 for+1,1 for-1
//! targets2.25 with ratio IF values±0.375, variance.140625. These weights are known
//! functions, not learned density ratios. n500/1000/2000,R>=2000, original four-fold
//! fits each replicate; retarget and encoded independent frozen consumer use0fits.
//! Every failed fit fails. Reported SE² and independent empirical variance compared
//! with known IF variance/n within15%; point bias<=5MCSE. No intervals are minted.
#![allow(clippy::cast_precision_loss, reason = "bounded diagnostic dimensions")]
#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use antecedent::analysis::recalc_cell::{
    CellRequest, CellSession, CellSpec, ScoreQuantity, ScoreResumeRequest, ScoreResumeSession,
    execute_cell_with_receipt, execute_resumed_retarget,
};
use antecedent::analysis::recalc_dr::{DrEstimator, DrRequest, DrSession, execute_dr_with_receipt};
use antecedent::analysis::recalc_receipt::{LawValue, TargetWeights, UtilitySpec};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::{CellSaturatedAipw, DmlAte, DrLearner};
use antecedent_io::frozen_scores_artifact::FrozenScoreTable;
use antecedent_learn::{LearnerSpec, LinearSpec};
use serde_json::json;
struct Sample {
    laws: [LawValue; 2],
    fits: u64,
}
fn data(n: usize, seed: u64, cell: bool) -> Vec<(String, Vec<f64>)> {
    let mut rng = candidate::Generator::new(seed);
    let (mut a, mut b, mut z, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let av = rng.binary(0.5) as f64;
        let bv = rng.binary(0.5) as f64;
        let zv = 2. * rng.binary(0.5) as f64 - 1.;
        a.push(av);
        b.push(bv);
        z.push(zv);
        y.push(if cell {
            1. + 0.3 * zv + 0.2 * av + 0.1 * bv + av * bv * (2. + 0.5 * zv)
        } else {
            1. + 0.3 * zv + av * (2. + 0.5 * zv)
        });
    }
    if cell {
        vec![("a".into(), a), ("b".into(), b), ("y".into(), y), ("z".into(), z)]
    } else {
        vec![("a".into(), a), ("y".into(), y), ("z".into(), z)]
    }
}
fn portable(
    artifact: &FrozenScoreTable,
    edges: Vec<(u32, u32)>,
    weights: TargetWeights,
    quantity: ScoreQuantity,
    law: LawValue,
) -> Option<()> {
    let bytes = artifact.to_bytes("retained-score-se-diagnostic").ok()?;
    let verified = FrozenScoreTable::from_bytes(&bytes, Some(artifact.identity())).ok()?;
    let rows = verified.score_table().ok()?.row_index.to_vec();
    let mut session = ScoreResumeSession::resume_from_scores(&verified).ok()?;
    let request = ScoreResumeRequest {
        n_variables: if quantity == ScoreQuantity::Interaction { 4 } else { 3 },
        edges,
        target: Some(weights),
        target_row_ids: Some(rows),
        utility: utility(),
        quantity,
        changed_inputs: vec![],
    };
    let out = execute_resumed_retarget(&mut session, &request).ok()?;
    if out.receipt.totals().fold_fits + out.receipt.totals().model_fits != 0
        || (out.law.ate - law.ate).abs() > 1e-10
        || (out.law.std_error - law.std_error).abs() > 1e-10
    {
        return None;
    }
    Some(())
}
fn utility() -> UtilitySpec {
    UtilitySpec { benefit_per_unit: 1., cost: 0. }
}
fn fit(n: usize, seed: u64, mode: u8) -> Option<Sample> {
    let ((result, cell_fits), learner_fits) =
        antecedent_learn::fit_counts::observe_resolved_fits(|| {
            antecedent_estimate::cell_aipw::count_cell_model_fits(|| fit_inner(n, seed, mode))
        });
    let sample = result?;
    if sample.fits != cell_fits + learner_fits {
        return None;
    }
    Some(sample)
}
fn fit_inner(n: usize, seed: u64, mode: u8) -> Option<Sample> {
    let columns = data(n, seed, mode == 2);
    let z = if mode == 2 { 3 } else { 2 };
    let weights = TargetWeights {
        weights: columns[z].1.iter().map(|v| if *v > 0. { 3. } else { 1. }).collect(),
        depends_on: vec![VariableId::from_raw(u32::try_from(z).ok()?)],
    };
    let ctx = ExecutionContext::for_tests(seed);
    let edges = if mode == 2 {
        vec![(3, 0), (3, 1), (3, 2), (0, 2), (1, 2)]
    } else {
        vec![(2, 0), (2, 1), (0, 1)]
    };
    if mode == 2 {
        let mut estimator = CellSaturatedAipw::new();
        estimator.folds = 4;
        let mut req = CellRequest {
            columns,
            spec: CellSpec {
                edges: edges.clone(),
                treatments: vec![0, 1],
                outcome: 2,
                adjustment: vec![3],
                estimator,
                quantity: ScoreQuantity::Interaction,
                target: None,
                utility: utility(),
            },
        };
        let mut session = CellSession::new();
        let first = execute_cell_with_receipt(&mut session, &req, &ctx).ok()?;
        let artifact = session.export_scores().ok()?;
        req.spec.target = Some(weights.clone());
        let weighted = execute_cell_with_receipt(&mut session, &req, &ctx).ok()?;
        if weighted.receipt.totals().fold_fits != 0 {
            return None;
        }
        portable(&artifact, edges, weights, ScoreQuantity::Interaction, weighted.law)?;
        Some(Sample { laws: [first.law, weighted.law], fits: first.receipt.totals().fold_fits })
    } else {
        let linear = LearnerSpec::Linear(LinearSpec::default());
        let estimator = if mode == 0 {
            DrEstimator::Dml(DmlAte::new().with_outcome(linear).with_folds(4))
        } else {
            DrEstimator::Cate(
                DrLearner::new().with_outcome(linear).with_final_learner(linear).with_folds(4),
            )
        };
        let mut req = DrRequest {
            columns,
            edges: edges.clone(),
            treatment: 0,
            outcome: 1,
            estimator,
            target: None,
            utility: utility(),
        };
        let mut session = DrSession::new();
        let first = execute_dr_with_receipt(&mut session, &req, &ctx).ok()?;
        let artifact = session.export_scores().ok()?;
        req.target = Some(weights.clone());
        let weighted = execute_dr_with_receipt(&mut session, &req, &ctx).ok()?;
        if weighted.receipt.totals().fold_fits + weighted.receipt.totals().model_fits != 0 {
            return None;
        }
        portable(&artifact, edges, weights, ScoreQuantity::AverageEffect, weighted.law)?;
        Some(Sample {
            laws: [first.law, weighted.law],
            fits: first.receipt.totals().fold_fits + first.receipt.totals().model_fits,
        })
    }
}
fn measure(test: &str, mode: u8) {
    let n = calibration::grid_n(1000);
    let seed = calibration::grid_seed(0xC2A1_0001);
    let results: Vec<_> = (0..calibration::n_sim().max(2000))
        .map(|i| fit(n, seed.wrapping_add(u64::from(i)), mode))
        .collect();
    let failed = results.iter().filter(|f| f.is_none()).count();
    if failed > 0 {
        println!(
            "INFERENCE_DIAGNOSTIC_RECORD {}",
            json!({"test":test,"rows":n,"attempts":results.len(),"failed":failed,"pass":false})
        );
        panic!("all full-fit and portable consumer attempts must succeed");
    }
    let fits: Vec<_> = results.iter().flatten().collect();
    let r = results.len() as f64;
    for (j, truth, v) in [(0, 2., 0.25), (1, 2.25, 0.140_625)] {
        let mean = fits.iter().map(|f| f.laws[j].ate).sum::<f64>() / r;
        let empirical = fits.iter().map(|f| (f.laws[j].ate - mean).powi(2)).sum::<f64>() / (r - 1.);
        let reported = fits.iter().map(|f| f.laws[j].std_error.powi(2)).sum::<f64>() / r;
        let variance = v / n as f64;
        let bias = (mean - truth).abs() / (variance / r).sqrt();
        let pass = (empirical / variance - 1.).abs() <= 0.15
            && (reported / variance - 1.).abs() <= 0.15
            && bias <= 5.;
        println!(
            "INFERENCE_DIAGNOSTIC_RECORD {}",
            json!({"test":test,"rows":n,"seed":seed,"weighted":j==1,"truth":truth,"truth_if_variance":v,"attempts":results.len(),"failed":failed,"full_fitted_models_total":fits.iter().map(|f|f.fits).sum::<u64>(),"retarget_fits":0,"independent_frozen_consumer_fits":0,"reported_mean_variance":reported,"independent_sampling_variance":empirical,"bias_mcse":bias,"variance_relative_error_bound":0.15,"bias_mcse_bound":5.,"pass":pass,"activation":"none; weighted score SE remains empirical-target plugin diagnostic"})
        );
        assert!(pass, "{test} n{n} weighted{j}: retained score SE rules failed");
    }
}
#[test]
#[ignore = "calibration final only"]
fn retained_dml_weighted_score_precision() {
    measure("retained_dml_weighted_score_precision", 0);
}
#[test]
#[ignore = "calibration final only"]
fn retained_dr_weighted_score_precision() {
    measure("retained_dr_weighted_score_precision", 1);
}
#[test]
#[ignore = "calibration final only"]
fn retained_cell_weighted_score_precision() {
    measure("retained_cell_weighted_score_precision", 2);
}

/// Numerical plumbing only: one actual snapshot per family, no calibration repetitions.
#[test]
fn retained_score_one_snapshot_exact_truth_and_portable_plumbing() {
    let n = 1200;
    let seed = 97;
    for mode in 0..3 {
        let sample =
            fit(n, seed, mode).expect("one-snapshot original fit/retarget/consumer must execute");
        assert!(sample.fits > 0);
        let columns = data(n, seed, mode == 2);
        let z = &columns.last().unwrap().1;
        for weighted in [false, true] {
            let weights: Vec<_> =
                z.iter().map(|v| if weighted && *v > 0. { 3. } else { 1. }).collect();
            let total: f64 = weights.iter().sum();
            let scores: Vec<_> = z.iter().map(|v| 2. + 0.5 * v).collect();
            let mean = scores.iter().zip(&weights).map(|(s, w)| s * w).sum::<f64>() / total;
            let variance = n as f64 / (n - 1) as f64
                * scores
                    .iter()
                    .zip(&weights)
                    .map(|(s, w)| (w / total * (s - mean)).powi(2))
                    .sum::<f64>();
            let law = sample.laws[usize::from(weighted)];
            assert!((law.ate - mean).abs() < 1e-9);
            assert!((law.std_error - variance.sqrt()).abs() < 1e-9);
        }
    }
}
