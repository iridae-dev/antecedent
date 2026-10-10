//! Shared DML-AIPW / DR-Learner recalculation against finite structural truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::ScoreTable;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(97)
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-8, "actual={actual}, expected={expected}");
}

/// Every (T,Z) cell occurs 40 times. Both potential-outcome regressions are exact:
/// Y(0)=1+Z; Y(1)=3+1.5Z, so tau(Z)=2+.5Z independently of nuisance fitting.
fn columns() -> Vec<(String, Vec<f64>)> {
    let (mut t, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..40 {
        for treatment in [0.0, 1.0] {
            for confounder in [-1.0, 1.0] {
                t.push(treatment);
                z.push(confounder);
                y.push(1.0 + confounder + treatment * (2.0 + 0.5 * confounder));
            }
        }
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z)]
}

/// Plain independent score-mean and IID covariance formula; does not call law helpers.
fn oracle(table: &ScoreTable, weights: &[f64]) -> (f64, f64) {
    let control = table
        .column(table.columns.iter().position(|c| c.arm == 0 && c.threshold.is_none()).unwrap())
        .unwrap();
    let active = table
        .column(table.columns.iter().position(|c| c.arm == 1 && c.threshold.is_none()).unwrap())
        .unwrap();
    let phi: Vec<f64> = active.iter().zip(control).map(|(a, c)| a - c).collect();
    let total: f64 = weights.iter().sum();
    let theta = phi.iter().zip(weights).map(|(p, w)| p * w).sum::<f64>() / total;
    let n = f64::from(u32::try_from(phi.len()).unwrap());
    let variance = n / (n - 1.0)
        * phi.iter().zip(weights).map(|(p, w)| (w / total * (p - theta)).powi(2)).sum::<f64>();
    (theta, variance.sqrt())
}

fn original_engine(
    dr: bool,
    columns: &[(String, Vec<f64>)],
) -> antecedent_estimate::EffectEstimate {
    use antecedent::{RefuteSuite, Study};
    use antecedent_core::AverageEffectQuery;
    use antecedent_data::TabularData;
    use antecedent_estimate::{DmlAte, DrLearner};
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_learn::{LearnerSpec, LinearSpec};
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let builder = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .refute(RefuteSuite::None);
    let learner = LearnerSpec::Linear(LinearSpec::default());
    let study = if dr {
        builder.estimator(DrLearner::new().with_outcome(learner).with_folds(4)).build().unwrap()
    } else {
        builder.estimator(DmlAte::new().with_outcome(learner).with_folds(4)).build().unwrap()
    };
    study.run(&ctx()).unwrap().estimate.as_effect().unwrap().clone()
}

#[test]
fn dr_existing_score_sources_match_heterogeneous_structural_truth() {
    use antecedent_learn::fit_counts::count_resolved_fits;
    let columns = columns();
    for dr in [false, true] {
        let (original, fits) = count_resolved_fits(|| original_engine(dr, &columns));
        assert_eq!(fits, if dr { 13 } else { 12 });
        close(original.ate, 2.0);
        close(original.se_analytic, (0.25_f64 / 159.0).sqrt());
        let table = original.score_table.as_ref().unwrap();
        let known = oracle(table, &vec![1.0; 160]);
        close(known.0, original.ate);
        close(known.1, original.se_analytic);
        for (row, score) in table
            .row_index
            .iter()
            .zip(table.column(1).unwrap().iter().zip(table.column(0).unwrap()).map(|(a, c)| a - c))
        {
            close(score, 2.0 + 0.5 * columns[2].1[usize::try_from(*row).unwrap()]);
        }
        if dr {
            let fitted = original.fitted_effect.as_ref().unwrap();
            let bytes = serde_json::to_vec(fitted.as_ref()).unwrap();
            let portable: antecedent_estimate::FittedEffect =
                serde_json::from_slice(&bytes).unwrap();
            let (predictions, fits) = count_resolved_fits(|| {
                portable.predict(&[VariableId::from_raw(2)], &[&[-1.0, 1.0]], 2, &ctx()).unwrap()
            });
            assert_eq!(fits, 0);
            close(predictions[0], 1.5);
            close(predictions[1], 2.5);
        }
    }
}

fn request(dr: bool) -> antecedent::analysis::recalc_dr::DrRequest {
    use antecedent::analysis::recalc_dr::{DrEstimator, DrRequest};
    use antecedent::analysis::recalc_receipt::UtilitySpec;
    use antecedent_estimate::{DmlAte, DrLearner};
    use antecedent_learn::{LearnerSpec, LinearSpec};
    let outcome = LearnerSpec::Linear(LinearSpec::default());
    DrRequest {
        columns: columns(),
        edges: vec![(2, 0), (2, 1), (0, 1)],
        treatment: 0,
        outcome: 1,
        estimator: if dr {
            DrEstimator::Cate(DrLearner::new().with_outcome(outcome).with_folds(4))
        } else {
            DrEstimator::Dml(DmlAte::new().with_outcome(outcome).with_folds(4))
        },
        target: None,
        utility: UtilitySpec { benefit_per_unit: 2.0, cost: 0.5 },
    }
}

fn run(
    session: &mut antecedent::analysis::recalc_dr::DrSession,
    request: &antecedent::analysis::recalc_dr::DrRequest,
) -> antecedent::analysis::recalc_receipt::RecalcOutcome {
    let ((result, actual_builds), actual_fits) =
        antecedent_learn::fit_counts::observe_resolved_fits(|| {
            antecedent_estimate::dml::count_aipw_score_builds(|| {
                antecedent::analysis::recalc_dr::execute_dr_with_receipt(session, request, &ctx())
            })
        });
    let outcome = result.unwrap();
    let counts = outcome.receipt.totals();
    assert_eq!(
        actual_builds, counts.score_computations,
        "external score builder observer matches receipt"
    );
    assert_eq!(
        actual_fits,
        counts.fold_fits + counts.model_fits,
        "external factory observer matches executed receipt"
    );
    outcome
}

fn fails_without_fits(
    session: &mut antecedent::analysis::recalc_dr::DrSession,
    request: &antecedent::analysis::recalc_dr::DrRequest,
    context: &ExecutionContext,
) -> antecedent::analysis::recalc_receipt::RecalcRunError {
    let (result, fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
        antecedent::analysis::recalc_dr::execute_dr_with_receipt(session, request, context)
    });
    assert_eq!(fits, 0, "this refusal is required before any learner fit");
    result.unwrap_err()
}

#[test]
fn dr_first_unchanged_utility_and_weighted_laws_match_known_truth_and_fresh_fits() {
    use antecedent::analysis::recalc_dr::DrSession;
    use antecedent::analysis::recalc_receipt::TargetWeights;
    use antecedent_core::recalc::Stage;
    for dr in [false, true] {
        let mut request = request(dr);
        let mut session = DrSession::new();
        let first = run(&mut session, &request);
        close(first.law.ate, 2.0);
        close(first.law.std_error, (0.25_f64 / 159.0).sqrt());
        assert_eq!(first.receipt.totals().fold_fits, 12);
        assert_eq!(first.receipt.totals().model_fits, u64::from(dr));
        let original = original_engine(dr, &request.columns);
        close(first.law.ate, original.ate);
        close(first.law.std_error, original.se_analytic);
        let scores = session.score_table().unwrap().scores.as_ptr();
        assert_eq!(run(&mut session, &request).receipt.totals().total(), 0);
        assert_eq!(session.score_table().unwrap().scores.as_ptr(), scores);
        request.utility.cost = 10.0;
        let decision = run(&mut session, &request);
        assert_eq!(decision.receipt.totals().decisions, 1);
        assert_eq!(decision.receipt.totals().fold_fits, 0);
        assert_eq!(decision.receipt.totals().model_fits, 0);
        close(decision.decision.net_benefit, -6.0);
        let weights: Vec<f64> =
            request.columns[2].1.iter().map(|z| if *z > 0.0 { 3.0 } else { 1.0 }).collect();
        request.target = Some(TargetWeights {
            weights: weights.clone(),
            depends_on: vec![VariableId::from_raw(2)],
        });
        let weighted = run(&mut session, &request);
        let known = oracle(session.score_table().unwrap(), &weights);
        close(weighted.law.ate, 2.25);
        close(weighted.law.ate, known.0);
        close(weighted.law.std_error, known.1);
        assert_eq!(weighted.receipt.totals().fold_fits, 0);
        assert_eq!(weighted.receipt.totals().model_fits, 0);
        assert_eq!(weighted.plan.status(Stage::ScoreArtifact).unwrap().tag(), "reused");
        let fresh = run(&mut DrSession::new(), &request);
        close(fresh.law.ate, weighted.law.ate);
        close(fresh.law.std_error, weighted.law.std_error);
        assert_eq!(fresh.receipt.totals().fold_fits, 12);
        assert_eq!(session.score_table().unwrap().scores.as_ptr(), scores);
    }
}

#[test]
fn dr_cate_predictions_reuse_portable_final_model_and_validate_feature_schema() {
    use antecedent::analysis::recalc_dr::DrSession;
    let mut session = DrSession::new();
    run(&mut session, &request(true));
    assert_eq!(session.prediction_columns().unwrap(), vec!["z"]);
    let fitted = session.fitted_effect().unwrap();
    let bytes = serde_json::to_vec(fitted).unwrap();
    let portable: antecedent_estimate::FittedEffect = serde_json::from_slice(&bytes).unwrap();
    let predicted =
        session.predict(&[VariableId::from_raw(2)], &[&[-1.0, 1.0]], 2, &ctx()).unwrap();
    let independent =
        portable.predict(&[VariableId::from_raw(2)], &[&[-1.0, 1.0]], 2, &ctx()).unwrap();
    close(predicted[0], 1.5);
    close(predicted[1], 2.5);
    assert_eq!(predicted, independent);
    let artifact = session.export_predictor_result(&ctx()).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
    let model = consumed.body.fitted_effect.unwrap();
    let (replayed, actual_fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
        model.predict(&[VariableId::from_raw(2)], &[&[-1.0, 1.0]], 2, &ctx()).unwrap()
    });
    assert_eq!(actual_fits, 0);
    assert_eq!(replayed, predicted);
    assert!(session.predict(&[VariableId::from_raw(0)], &[&[-1.0, 1.0]], 2, &ctx()).is_err());
    assert!(session.predict(&[VariableId::from_raw(2)], &[&[f64::NAN]], 1, &ctx()).is_err());
    assert!(matches!(
        session.predict(&[VariableId::from_raw(2)], &[&[2.0]], 1, &ctx()),
        Err(antecedent::analysis::recalc_receipt::RecalcRunError::Request(
            "recalc.dr_prediction_out_of_support"
        ))
    ));
    assert_eq!(run(&mut session, &request(true)).receipt.totals().total(), 0);
    let mut dml = DrSession::new();
    run(&mut dml, &request(false));
    assert!(dml.predict(&[VariableId::from_raw(2)], &[&[0.0]], 1, &ctx()).is_err());
}

fn check_portable_retarget(
    request: &antecedent::analysis::recalc_dr::DrRequest,
    table: &ScoreTable,
    restored: &antecedent_io::frozen_scores_artifact::FrozenScoreTable,
) {
    // Execute the existing data-free consumer, not only its artifact decoder.
    use antecedent::analysis::recalc_cell::{
        ScoreQuantity, ScoreResumeRequest, ScoreResumeSession, execute_resumed_retarget,
    };
    use antecedent::analysis::recalc_receipt::TargetWeights;
    let weights: Vec<f64> = table
        .row_index
        .iter()
        .map(
            |row| {
                if request.columns[2].1[usize::try_from(*row).unwrap()] > 0.0 { 3.0 } else { 1.0 }
            },
        )
        .collect();
    let mut portable = ScoreResumeSession::resume_from_scores(restored).unwrap();
    let mut resume_request = ScoreResumeRequest {
        n_variables: 3,
        edges: request.edges.clone(),
        target: Some(TargetWeights {
            weights: weights.clone(),
            depends_on: vec![VariableId::from_raw(2)],
        }),
        target_row_ids: Some(table.row_index.to_vec()),
        utility: request.utility,
        quantity: ScoreQuantity::AverageEffect,
        changed_inputs: Vec::new(),
    };
    let ((resumed, builds), fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
        antecedent_estimate::dml::count_aipw_score_builds(|| {
            execute_resumed_retarget(&mut portable, &resume_request)
        })
    });
    let resumed = resumed.unwrap();
    assert_eq!((fits, builds), (0, 0));
    assert_eq!(
        resumed.receipt.totals().fold_fits
            + resumed.receipt.totals().model_fits
            + resumed.receipt.totals().score_computations,
        0
    );
    let expected = oracle(table, &weights);
    close(resumed.law.ate, expected.0);
    close(resumed.law.std_error, expected.1);
    resume_request.edges.clear();
    let ((refused, builds), fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
        antecedent_estimate::dml::count_aipw_score_builds(|| {
            execute_resumed_retarget(&mut portable, &resume_request)
        })
    });
    assert!(matches!(
        refused,
        Err(antecedent::analysis::recalc_receipt::RecalcRunError::Refused(_))
    ));
    assert_eq!((fits, builds), (0, 0));
}

#[test]
fn dr_complete_case_rows_frozen_scores_and_fresh_refit_are_bound_to_real_data() {
    use antecedent::analysis::recalc_dr::DrSession;
    use antecedent_core::recalc::ResumeContext;
    use antecedent_io::frozen_scores_artifact::FrozenScoreTable;
    for dr in [false, true] {
        let mut request = request(dr);
        request.columns[1].1[0] = f64::NAN;
        let mut session = DrSession::new();
        let first = run(&mut session, &request);
        let table = session.score_table().unwrap();
        assert_eq!(table.n_rows, 159);
        assert_eq!(table.row_index.as_ref(), &(1..160).collect::<Vec<u32>>());
        close(first.law.ate, 2.0 + 0.5 / 159.0);
        let exported = session.export_scores().unwrap();
        let bytes = exported.to_bytes("dr-complete-cases").unwrap();
        let restored = FrozenScoreTable::from_bytes(&bytes, None).unwrap();
        assert_eq!(restored.score_table().unwrap(), *table);
        assert!(FrozenScoreTable::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
        let known = oracle(&restored.score_table().unwrap(), &vec![1.0; 159]);
        close(known.0, first.law.ate);
        close(known.1, first.law.std_error);
        check_portable_retarget(&request, table, &restored);
        let mut unavailable = DrSession::resume(
            session.identities().clone(),
            ResumeContext {
                portable_fit: true,
                portable_scores: true,
                scores_snapshot_bound: true,
                supplied_provider: true,
                supplied_data: false,
            },
        );
        assert!(!unavailable.is_live());
        assert!(
            antecedent::analysis::recalc_dr::execute_dr_with_receipt(
                &mut unavailable,
                &request,
                &ctx()
            )
            .is_err()
        );
        let mut raw = DrSession::resume(
            session.identities().clone(),
            ResumeContext { supplied_data: true, ..ResumeContext::default() },
        );
        let replay = run(&mut raw, &request);
        assert_eq!(replay.receipt.totals().fold_fits, 12);
        close(replay.law.ate, first.law.ate);
        close(replay.law.std_error, first.law.std_error);
    }
}

#[test]
fn dr_data_fold_and_causal_role_changes_invalidate_the_required_stages() {
    use antecedent::analysis::recalc_dr::{DrEstimator, DrSession};
    use antecedent_core::recalc::Stage;
    for dr in [false, true] {
        let mut request = request(dr);
        let mut session = DrSession::new();
        run(&mut session, &request);
        let treatment = request.columns[0].1.clone();
        for (y, t) in request.columns[1].1.iter_mut().zip(treatment) {
            *y += t;
        }
        let changed = run(&mut session, &request);
        close(changed.law.ate, 3.0);
        assert_eq!(changed.receipt.totals().fold_fits, 12);
        assert_eq!(changed.plan.status(Stage::Identification).unwrap().tag(), "reused");
        match &mut request.estimator {
            DrEstimator::Dml(e) => e.folds = 5,
            DrEstimator::Cate(e) => e.folds = 5,
        }
        let folded = run(&mut session, &request);
        assert_eq!(folded.receipt.totals().fold_fits, 15);
        assert_eq!(folded.receipt.totals().model_fits, u64::from(dr));
        close(folded.law.ate, 3.0);
        let full = run(&mut DrSession::new(), &request);
        close(full.law.ate, folded.law.ate);
        close(full.law.std_error, folded.law.std_error);
        request.edges = vec![(0, 1)]; // Z is no longer the automatically identified adjustment.
        let graph = run(&mut session, &request);
        assert_eq!(graph.plan.status(Stage::Identification).unwrap().tag(), "recomputed");
        assert_eq!(graph.receipt.totals().fold_fits, 15);
    }
}

fn assert_unpopulated_arm(error: &antecedent::analysis::recalc_receipt::RecalcRunError) {
    let antecedent::analysis::recalc_receipt::RecalcRunError::Execution(causal) = error else {
        panic!("expected executing arm refusal: {error:?}");
    };
    assert!(
        matches!(causal.peeled(), antecedent::CausalError::Estimate(antecedent_estimate::EstimationError::Refused { code: "arm_not_populated", message }) if message.contains("dml.arm_not_populated"))
    );
}

#[test]
fn dr_plr_target_cancellation_and_invalid_refits_cannot_reuse_an_aipw_law() {
    use antecedent::analysis::recalc_dr::{DrEstimator, DrSession};
    use antecedent::analysis::recalc_receipt::TargetWeights;
    use antecedent_estimate::DmlScore;
    let original = request(false);
    let mut session = DrSession::new();
    run(&mut session, &original);
    let mut invalid = original.clone();
    if let DrEstimator::Dml(e) = &mut invalid.estimator {
        e.score = DmlScore::PartiallyLinear;
    }
    let error = fails_without_fits(&mut session, &invalid, &ctx());
    assert!(matches!(error, antecedent::analysis::recalc_receipt::RecalcRunError::Refused(plan)
        if matches!(plan.first_refusal(), Some((_, antecedent_core::recalc::RefusalReason::Unsupported { .. })))));
    invalid = original.clone();
    invalid.target =
        Some(TargetWeights { weights: vec![1.0; 160], depends_on: vec![VariableId::from_raw(0)] });
    let error = fails_without_fits(&mut session, &invalid, &ctx());
    assert!(matches!(
        error,
        antecedent::analysis::recalc_receipt::RecalcRunError::Request(
            "recalc.invalid_target_weights"
        )
    ));
    invalid = original.clone();
    if let DrEstimator::Dml(e) = &mut invalid.estimator {
        e.overlap = antecedent_estimate::OverlapPolicy::RequireDiagnostics {
            clip: Some(0.01),
            trim: Some(0.1),
        };
    }
    let error = fails_without_fits(&mut session, &invalid, &ctx());
    assert!(matches!(error, antecedent::analysis::recalc_receipt::RecalcRunError::Refused(plan)
        if matches!(plan.first_refusal(), Some((_, antecedent_core::recalc::RefusalReason::Unsupported { .. })))));
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error = fails_without_fits(&mut session, &original, &cancelled);
    assert!(error.to_string().contains("recalc.cancelled"));
    invalid = original.clone();
    invalid.columns[1].1[0] += 0.1;
    let mut constrained = ctx();
    constrained.memory.hard_limit_bytes = Some(1);
    let _ = fails_without_fits(&mut session, &invalid, &constrained);
    invalid = original.clone();
    invalid.columns[0].1.fill(0.0); // No treated arm: no cross-fitted treatment contrast.
    let error = fails_without_fits(&mut session, &invalid, &ctx());
    assert_unpopulated_arm(&error);
    // The raw frame has both arms, but every control outcome is missing.
    invalid = original.clone();
    for (t, y) in invalid.columns[0].1.clone().iter().zip(&mut invalid.columns[1].1) {
        if *t < 0.5 {
            *y = f64::NAN;
        }
    }
    let error = fails_without_fits(&mut session, &invalid, &ctx());
    assert_unpopulated_arm(&error);
    assert_eq!(run(&mut session, &original).receipt.totals().total(), 0);
}

#[test]
fn dr_complete_case_retarget_uses_retained_row_ids_and_checked_adjustment() {
    use antecedent::analysis::recalc_dr::DrSession;
    use antecedent::analysis::recalc_receipt::TargetWeights;
    for dr in [false, true] {
        let mut request = request(dr);
        request.columns[1].1[0] = f64::NAN;
        let mut session = DrSession::new();
        run(&mut session, &request);
        let weights: Vec<f64> =
            session
                .score_table()
                .unwrap()
                .row_index
                .iter()
                .map(|row| {
                    if request.columns[2].1[usize::try_from(*row).unwrap()] > 0.0 {
                        3.0
                    } else {
                        1.0
                    }
                })
                .collect();
        request.target = Some(TargetWeights {
            weights: weights.clone(),
            depends_on: vec![VariableId::from_raw(2)],
        });
        let weighted = run(&mut session, &request);
        close(weighted.law.ate, 2.0 + 0.5 * 161.0 / 319.0);
        let expected = oracle(session.score_table().unwrap(), &weights);
        close(weighted.law.std_error, expected.1);
        assert_eq!(weighted.receipt.totals().fold_fits, 0);
        assert_eq!(weighted.receipt.totals().model_fits, 0);
        let mut wrong = request.clone();
        wrong.target.as_mut().unwrap().weights = vec![1.0; 160];
        let _ = fails_without_fits(&mut session, &wrong, &ctx());
        wrong = request.clone();
        wrong.target.as_mut().unwrap().depends_on = vec![VariableId::from_raw(1)];
        let _ = fails_without_fits(&mut session, &wrong, &ctx());
        assert_eq!(run(&mut session, &request).receipt.totals().total(), 0);
    }
}

#[test]
fn dr_receipt_independent_reader_preserves_executed_counts_and_identity() {
    use antecedent::analysis::recalc_dr::DrSession;
    use antecedent_core::recalc::{RecalcCapabilities, RetargetSupport, StageIdentities};
    use antecedent_io::recalc_receipt_artifact::{CountsWire, RecalcReceiptArtifact};
    use std::collections::BTreeMap;
    let mut session = DrSession::new();
    let first = run(&mut session, &request(true));
    let counts: BTreeMap<_, _> = first
        .receipt
        .entries()
        .iter()
        .map(|entry| {
            let c = entry.counts;
            (
                entry.stage,
                CountsWire {
                    identifications: c.identifications,
                    fold_fits: c.fold_fits,
                    model_fits: c.model_fits,
                    score_computations: c.score_computations,
                    reweights: c.reweights,
                    decisions: c.decisions,
                    ..CountsWire::default()
                },
            )
        })
        .collect();
    let artifact = RecalcReceiptArtifact::seal(
        &StageIdentities::new(),
        session.identities(),
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
        &counts,
    )
    .unwrap();
    assert_eq!(artifact.receipt_identity(), first.receipt.identity().to_hex());
    let bytes = artifact.to_bytes("dr-actual-learner-fits").unwrap();
    let restored =
        RecalcReceiptArtifact::from_bytes(&bytes, Some(artifact.receipt_identity())).unwrap();
    assert_eq!(restored.receipt_identity(), artifact.receipt_identity());
    assert!(RecalcReceiptArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
}

fn bundle_receipt_bytes(
    session: &antecedent::analysis::recalc_dr::DrSession,
    result: &antecedent::analysis::recalc_receipt::RecalcOutcome,
) -> Vec<u8> {
    use antecedent_core::recalc::{RecalcCapabilities, RetargetSupport, StageIdentities};
    use antecedent_io::recalc_receipt_artifact::{CountsWire, RecalcReceiptArtifact};
    let counts = result
        .receipt
        .entries()
        .iter()
        .map(|entry| {
            let c = entry.counts;
            (
                entry.stage,
                CountsWire {
                    identifications: c.identifications,
                    fold_fits: c.fold_fits,
                    model_fits: c.model_fits,
                    score_computations: c.score_computations,
                    reweights: c.reweights,
                    decisions: c.decisions,
                    ..CountsWire::default()
                },
            )
        })
        .collect();
    RecalcReceiptArtifact::seal(
        &StageIdentities::new(),
        session.identities(),
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
        &counts,
    )
    .unwrap()
    .to_bytes("composed-dr-receipt")
    .unwrap()
}

#[test]
fn dr_bundle_independently_consumes_historical_receipt_and_real_resumable_scores() {
    use antecedent::analysis::composition::{
        BundleBuilder, NodeKind, SuppliedSources, consume, export_bundle,
    };
    use antecedent::analysis::recalc_cell::{
        ScoreResumeRequest, ScoreResumeSession, execute_resumed_retarget,
    };
    use antecedent::analysis::recalc_dr::DrSession;
    use antecedent::analysis::recalc_receipt::UtilitySpec;
    let mut session = DrSession::new();
    let outcome = run(&mut session, &request(true));
    let receipt = bundle_receipt_bytes(&session, &outcome);
    let scores = session.export_scores().unwrap();
    let score_bytes = scores.to_bytes("composed-dr-scores").unwrap();
    let mut builder = BundleBuilder::new();
    builder.add_artifact(Some("receipt"), NodeKind::RecalculationReceipt, &receipt).unwrap();
    builder.add_detected_artifact(Some("scores"), &score_bytes).unwrap();
    builder.connect("receipt", "scores").unwrap();
    let bundle = builder.build().unwrap();
    let bytes = export_bundle(&bundle, "dr-state-bundle").unwrap();
    let consumer = consume(&bytes, bundle.identity(), &SuppliedSources::default()).unwrap();
    consumer.require_verified().unwrap();
    let past = consumer.node("receipt").unwrap();
    assert_eq!(past.facts.get("execution_status").unwrap(), "historical_receipt");
    assert_eq!(past.facts.get("trust").unwrap(), "unverified");
    assert!(!past.facts.contains_key("reusable_state"));
    let retained = consumer.node("scores").unwrap();
    assert_eq!(retained.facts.get("reusable_state").unwrap(), "frozen_same_row_scores");
    assert!(!retained.facts.contains_key("law"));
    // Resume executes a new same-row law operation; past receipt counts are not reused.
    let mut resumed =
        ScoreResumeSession::resume_from_score_bytes(&score_bytes, Some(scores.identity())).unwrap();
    let target = ScoreResumeRequest {
        n_variables: 3,
        edges: request(true).edges,
        target: None,
        target_row_ids: None,
        quantity: antecedent::analysis::recalc_cell::ScoreQuantity::AverageEffect,
        changed_inputs: vec![],
        utility: UtilitySpec { benefit_per_unit: 2.0, cost: 0.5 },
    };
    let updated = execute_resumed_retarget(&mut resumed, &target).unwrap();
    close(updated.law.ate, 2.0);
    assert_eq!(updated.receipt.totals().fold_fits, 0);
    assert_eq!(updated.receipt.totals().reweights, 1);
}

#[test]
fn dr_bundle_refuses_same_snapshot_scores_from_another_executed_fit() {
    use antecedent::analysis::composition::{
        BundleBuilder, BundleStage, NodeKind, NodeStatus, SuppliedSources, consume, export_bundle,
    };
    use antecedent::analysis::recalc_dr::{DrEstimator, DrSession};
    use antecedent_estimate::DrLearner;
    use antecedent_learn::{LearnerSpec, LinearSpec};
    let mut session = DrSession::new();
    let first = run(&mut session, &request(true));
    let receipt = bundle_receipt_bytes(&session, &first);
    let original = session.export_scores().unwrap();
    let mut different = request(true);
    different.estimator = DrEstimator::Cate(
        DrLearner::new().with_outcome(LearnerSpec::Linear(LinearSpec::default())).with_folds(3),
    );
    let mut other = DrSession::new();
    run(&mut other, &different);
    let swapped = other.export_scores().unwrap();
    assert_eq!(original.meta().snapshot_digest, swapped.meta().snapshot_digest);
    assert_ne!(original.meta().fit_identity, swapped.meta().fit_identity);
    let mut builder = BundleBuilder::new();
    builder.add_artifact(Some("receipt"), NodeKind::RecalculationReceipt, &receipt).unwrap();
    builder
        .add_artifact(Some("scores"), NodeKind::FrozenScores, &swapped.to_bytes("swapped").unwrap())
        .unwrap();
    builder.connect("receipt", "scores").unwrap();
    let bundle = builder.build().unwrap();
    let bytes = export_bundle(&bundle, "swapped-dr-fit").unwrap();
    let consumed = consume(&bytes, bundle.identity(), &SuppliedSources::default()).unwrap();
    assert!(matches!(
        &consumed.node("scores").unwrap().status,
        NodeStatus::Failed { stage: BundleStage::SwappedEvidence, .. }
    ));
    assert!(consumed.require_verified().is_err());
}
