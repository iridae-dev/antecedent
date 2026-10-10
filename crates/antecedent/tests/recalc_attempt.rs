//! Original component attempts on native failures, reuse, and cancellation.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::recalc_adjusted::{
    AdjustedContrast, AdjustedModel, AdjustedRequest, AdjustedSession,
    execute_adjusted_with_receipt,
};
use antecedent::analysis::recalc_attempt::{Operation, execute_with_attempt};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent_core::{CancellationToken, ExecutionContext, VariableId};
use antecedent_estimate::vector_treatment::VectorCovariance;
use antecedent_stats::{GlmFamily, GlmOptions};

fn utility() -> UtilitySpec {
    UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 }
}
fn adjusted(logit: bool) -> AdjustedRequest {
    let (mut t, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for (a, b, successes) in [(0.0, 0.0, 15), (1.0, 0.0, 30), (0.0, 1.0, 24), (1.0, 1.0, 40)] {
        for row in 0..60 {
            t.push(a);
            z.push(b);
            y.push(if logit {
                f64::from(row < successes)
            } else {
                1.0 + 2.0 * a + 0.3 * b + if row % 2 == 0 { 0.4 } else { -0.4 }
            });
        }
    }
    AdjustedRequest {
        columns: vec![("t".into(), t), ("y".into(), y), ("z".into(), z)],
        edges: vec![(2, 0), (2, 1), (0, 1)],
        treatments: vec![0],
        outcome: 1,
        adjustment: vec![2],
        model: if logit {
            AdjustedModel::Glm { family: GlmFamily::BinomialLogit, options: GlmOptions::default() }
        } else {
            AdjustedModel::Linear { covariance: VectorCovariance::ModelBased }
        },
        contrast: AdjustedContrast::Numeric { active: vec![1.0], control: vec![0.0] },
        target: None,
        utility: utility(),
    }
}
#[test]
fn original_adjusted_attempts_match_success_counters_and_failed_state_is_not_published() {
    use antecedent_stats::fit_counts::count_least_squares_solves;
    let ctx = ExecutionContext::for_tests(83);
    let request = adjusted(false);
    let mut session = AdjustedSession::new();
    let ((first, models), solves) = count_least_squares_solves(|| {
        antecedent_estimate::adjustment_resume::count_adjusted_model_fits(|| {
            execute_with_attempt(|| execute_adjusted_with_receipt(&mut session, &request, &ctx))
        })
    });
    assert_eq!(models, 1);
    assert_eq!(
        solves, 0,
        "this original OLS engine uses its own normal-equation solve, not FaerBackend"
    );
    let first_result = first.result.unwrap();
    assert!((first_result.law.ate - 2.0).abs() < 1e-10);
    assert_eq!(
        first_result.receipt.totals().model_fits,
        first.report.counts(Operation::AdjustedFit).completed
    );
    assert_eq!(first.report.counts(Operation::LeastSquaresSolve).completed, 1);
    assert_eq!(first.report.counts(Operation::LawSummary).completed, 1);
    let covariance = session.covariance().unwrap().as_ptr();
    let mut singular = request.clone();
    singular.columns[2].1 = singular.columns[0].1.clone();
    let failed =
        execute_with_attempt(|| execute_adjusted_with_receipt(&mut session, &singular, &ctx));
    assert!(failed.result.is_err());
    assert_eq!(failed.report.counts(Operation::AdjustedFit).failed, 1);
    assert_eq!(
        failed.report.counts(Operation::LeastSquaresSolve).attempted,
        0,
        "design rejection did not invoke a numerical solve"
    );
    assert_eq!(session.covariance().unwrap().as_ptr(), covariance);
    let unchanged =
        execute_with_attempt(|| execute_adjusted_with_receipt(&mut session, &request, &ctx));
    assert!(unchanged.result.is_ok());
    assert!(unchanged.report.is_empty());
    let cancelled = ExecutionContext::for_tests(83);
    cancelled.cancellation.cancel();
    let stopped =
        execute_with_attempt(|| execute_adjusted_with_receipt(&mut session, &request, &cancelled));
    assert!(stopped.result.is_err());
    assert!(stopped.report.is_empty());
    assert_eq!(session.covariance().unwrap().as_ptr(), covariance);
}
#[test]
fn original_glm_nonconvergence_retains_completed_solves_and_old_model() {
    use antecedent_stats::fit_counts::count_least_squares_solves;
    let ctx = ExecutionContext::for_tests(83);
    let request = adjusted(true);
    let mut session = AdjustedSession::new();
    execute_adjusted_with_receipt(&mut session, &request, &ctx).unwrap();
    let covariance = session.covariance().unwrap().as_ptr();
    let mut invalid = request.clone();
    let AdjustedModel::Glm { options, .. } = &mut invalid.model else { unreachable!() };
    options.max_iter = 1;
    let (failed, solves) = count_least_squares_solves(|| {
        execute_with_attempt(|| execute_adjusted_with_receipt(&mut session, &invalid, &ctx))
    });
    assert!(failed.result.is_err());
    assert!(solves > 0);
    assert_eq!(failed.report.counts(Operation::LeastSquaresSolve).completed, solves);
    assert_eq!(failed.report.counts(Operation::AdjustedFit).failed, 1);
    assert!(failed.report.is_complete());
    assert_eq!(session.covariance().unwrap().as_ptr(), covariance);
    let recovered =
        execute_with_attempt(|| execute_adjusted_with_receipt(&mut session, &request, &ctx));
    assert!(recovered.result.is_ok());
    assert!(recovered.report.is_empty());
}
fn response() -> antecedent::analysis::recalc_static::StaticResponseRequest {
    use antecedent_graph::{Admg, DenseNodeId};
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let mut columns: Vec<(String, Vec<f64>)> =
        ["t", "m", "y"].iter().map(|name| ((*name).into(), Vec::new())).collect();
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (name, values) in &mut columns {
            values.extend(std::iter::repeat_n(cell[name.as_str()].as_f64().unwrap(), count));
        }
    }
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    antecedent::analysis::recalc_static::StaticResponseRequest {
        columns,
        graph,
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(2),
        support: vec![0.0, 1.0],
        actions: vec![0.0, 1.0],
        baseline: 0,
        active: 1,
        utility: utility(),
    }
}
#[test]
fn original_static_mid_execution_cancellation_preserves_completed_work_and_old_state() {
    use antecedent::analysis::recalc_static::{
        StaticResponseSession, execute_static_response_with_receipt,
    };
    use antecedent_expr::execution_counts::count_static_work;
    let request = response();
    let ctx = ExecutionContext::for_tests(7);
    let mut session = StaticResponseSession::new();
    let first = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    let result = std::ptr::from_ref(session.result().unwrap());
    let mut changed = request.clone();
    changed.columns[2].1[0] = 1.0 - changed.columns[2].1[0];
    let mut observed_mid_work = false;
    for checks in 0..512 {
        let mut stopped = ExecutionContext::for_tests(7);
        stopped.cancellation = CancellationToken::cancel_after_checks(checks);
        let (attempt, work) = count_static_work(|| {
            execute_with_attempt(|| {
                execute_static_response_with_receipt(&mut session, &changed, &stopped)
            })
        });
        if attempt.result.is_err() && attempt.report.operations().any(|(_, c)| c.completed > 0) {
            assert!(stopped.cancellation.is_cancelled());
            assert_eq!(
                attempt.report.counts(Operation::FactorConstruction).completed,
                work.factor_builds
            );
            assert_eq!(
                attempt.report.counts(Operation::ProgramCompilation).completed,
                work.program_compilations
            );
            assert_eq!(std::ptr::from_ref(session.result().unwrap()), result);
            let old = execute_with_attempt(|| {
                execute_static_response_with_receipt(&mut session, &request, &ctx)
            });
            assert_eq!(old.result.unwrap().contrast.to_bits(), first.contrast.to_bits());
            assert!(old.report.is_empty());
            observed_mid_work = true;
            break;
        }
        // The search stops before reaching a successful changed publication.
        assert!(
            attempt.result.is_err(),
            "a cancellation boundary with real completed work should precede success"
        );
    }
    assert!(observed_mid_work, "no actual partial-work cancellation boundary was exercised");
}

#[test]
fn original_mapped_prior_construction_failure_and_reuse_have_separate_attempt_counts() {
    use antecedent::BayesianConfig;
    use antecedent::analysis::recalc_bayesian::{
        BayesianModel, BayesianRequest, BayesianSession, PosteriorSummarySpec,
        execute_bayesian_with_receipt,
    };
    use antecedent_io::PriorMapping;
    use antecedent_prob::{GaussianCoefficientPrior, PriorSet, PriorSpec};
    let run = || {
        let base = adjusted(false);
        let mut prior = PriorSet::new();
        prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(3, 10.0)));
        prior.push(PriorSpec::KnownResidualVariance(1.0));
        let source = BayesianRequest {
            columns: base.columns,
            edges: base.edges,
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            model: BayesianModel::Gaussian,
            inference: BayesianConfig::conjugate().n_draws(128).prior(prior),
            summary: PosteriorSummarySpec::default(),
            utility: utility(),
        };
        let ctx = ExecutionContext::for_tests(53);
        let mut original = BayesianSession::new();
        execute_bayesian_with_receipt(&mut original, &source, &ctx).unwrap();
        let source_bytes = original.export_prior_source().unwrap();
        let mut target = source.clone();
        target.inference.prior = None;
        target.inference.prior_artifact = Some(source_bytes.into());
        for (t, y) in target.columns[0].1.clone().iter().zip(&mut target.columns[1].1) {
            *y += 2.0 * t;
        }
        let mut session = BayesianSession::new();
        let first =
            execute_with_attempt(|| execute_bayesian_with_receipt(&mut session, &target, &ctx));
        let first_result = first.result.unwrap();
        assert_eq!(first.report.counts(Operation::PriorConstruction).completed, 1);
        assert_eq!(first.report.counts(Operation::PriorConstruction).failed, 0);
        assert_eq!(
            first.report.counts(Operation::PosteriorFit).completed,
            2,
            "actual source and target fits"
        );
        assert_eq!(first.report.counts(Operation::PosteriorDraw).completed, 256);
        let draws = session.posterior().unwrap().draws.column(0).unwrap().as_ptr();
        let mut invalid = target.clone();
        invalid.inference.prior_mapping = Some(PriorMapping::NamedParameters {
            pairs: vec![("absent_source_parameter".into(), "absent_target_parameter".into())],
        });
        let failed =
            execute_with_attempt(|| execute_bayesian_with_receipt(&mut session, &invalid, &ctx));
        assert!(failed.result.is_err());
        assert_eq!(failed.report.counts(Operation::PriorConstruction).failed, 1);
        assert_eq!(
            failed.report.counts(Operation::PosteriorFit).attempted,
            0,
            "verified source reused; refused prior precedes target solve"
        );
        assert_eq!(session.posterior().unwrap().draws.column(0).unwrap().as_ptr(), draws);
        let unchanged =
            execute_with_attempt(|| execute_bayesian_with_receipt(&mut session, &target, &ctx));
        assert_eq!(unchanged.result.unwrap().law, first_result.law);
        assert!(unchanged.report.is_empty());
    };
    // The checked builder's preexisting aggregate stack frame is large.
    std::thread::Builder::new().stack_size(16 * 1024 * 1024).spawn(run).unwrap().join().unwrap();
}

#[path = "common/external_callback_fixture.rs"]
mod external_reference;
#[test]
fn original_provider_failure_and_post_callback_cancellation_preserve_exact_attempts() {
    use antecedent::analysis::recalc_external::{
        CallbackPolicy, ExternalCallbackError, ExternalCallbackSession, count_external_invocations,
    };
    let request = external_reference::request(0, CallbackPolicy::Deterministic);
    let mut provider = external_reference::Provider::new(&request);
    let mut session = ExternalCallbackSession::new();
    let ctx = ExecutionContext::for_tests(999);
    session.execute(&request, Some(&mut provider), &ctx).unwrap();
    let claim = std::ptr::from_ref(session.issued_claim().unwrap());
    let mut changed = request.clone();
    changed.columns[0].1 = vec![0.0, 2.0];
    provider.fail = true;
    let (failed, calls) = count_external_invocations(|| {
        execute_with_attempt(|| session.execute(&changed, Some(&mut provider), &ctx))
    });
    assert_eq!(calls, 1);
    assert!(matches!(failed.result, Err(ExternalCallbackError::Attempt { .. })));
    let work = failed.report.counts(Operation::ProviderInvocation);
    assert_eq!((work.attempted, work.completed, work.failed), (1, 0, 1));
    assert_eq!(std::ptr::from_ref(session.issued_claim().unwrap()), claim);
    provider.fail = false;
    provider.cancel = true;
    let (cancelled, calls) = count_external_invocations(|| {
        execute_with_attempt(|| session.execute(&changed, Some(&mut provider), &ctx))
    });
    assert_eq!(calls, 1);
    assert!(ctx.cancellation.is_cancelled());
    assert!(matches!(
        cancelled.result,
        Err(ExternalCallbackError::Attempt { detail: "recalc.cancelled", .. })
    ));
    let work = cancelled.report.counts(Operation::ProviderInvocation);
    assert_eq!(
        (work.attempted, work.completed, work.failed),
        (1, 1, 0),
        "the callback completed; cancellation refused its later publication"
    );
    assert_eq!(std::ptr::from_ref(session.issued_claim().unwrap()), claim);
    let old = execute_with_attempt(|| {
        session.execute(&request, Some(&mut provider), &ExecutionContext::for_tests(999))
    });
    assert!(old.result.is_ok());
    assert!(old.report.is_empty());
}

#[test]
fn original_adjusted_law_summary_failure_does_not_modify_the_fit() {
    use antecedent_estimate::adjustment_resume::AdjustedFit;
    use antecedent_estimate::vector_treatment::{
        NamedColumn, TreatmentColumn, VectorTreatmentInput, VectorTreatmentOptions,
    };
    let r = adjusted(false);
    let snapshot = "attempt.adjusted.summary".to_string();
    let fit = AdjustedFit::linear(
        &VectorTreatmentInput {
            outcome: r.columns[1].1.clone(),
            row_snapshot: snapshot.clone(),
            adjustment: vec![NamedColumn { name: "z".into(), values: r.columns[2].1.clone() }],
            treatments: vec![TreatmentColumn {
                name: "t".into(),
                values: r.columns[0].1.clone(),
                adjustment_set: vec!["z".into()],
                row_snapshot: snapshot,
            }],
        },
        &VectorTreatmentOptions { covariance: VectorCovariance::ModelBased, contrasts: vec![] },
    )
    .unwrap();
    let covariance = fit.covariance().as_ptr();
    let first = execute_with_attempt(|| fit.contrast(&[1.0], &[0.0], None));
    assert!((first.result.unwrap().0 - 2.0).abs() < 1e-10);
    assert_eq!(first.report.counts(Operation::LawSummary).completed, 1);
    let zeros = vec![0.0; r.columns[0].1.len()];
    let failed = execute_with_attempt(|| fit.contrast(&[1.0], &[0.0], Some(&zeros)));
    assert!(failed.result.is_err());
    assert_eq!(failed.report.counts(Operation::LawSummary).attempted, 1);
    assert_eq!(failed.report.counts(Operation::LawSummary).completed, 0);
    assert_eq!(failed.report.counts(Operation::LawSummary).failed, 1);
    assert_eq!(fit.covariance().as_ptr(), covariance);
    let retry = execute_with_attempt(|| fit.contrast(&[1.0], &[0.0], None));
    assert!((retry.result.unwrap().0 - 2.0).abs() < 1e-10);
    assert_eq!(retry.report.counts(Operation::LawSummary).completed, 1);
    assert_eq!(retry.report.counts(Operation::LeastSquaresSolve).attempted, 0);
}

#[test]
fn original_score_retarget_summary_reuses_fits_but_records_actual_law_work() {
    use antecedent::analysis::recalc_receipt::{
        RecalcRequest, RecalcSession, TargetWeights, execute_with_receipt,
    };
    let mut data = adjusted(false);
    // Exact conditional outcome removes finite-fold nuisance error from the law oracle.
    data.columns[1].1 = data.columns[0]
        .1
        .iter()
        .zip(&data.columns[2].1)
        .map(|(t, z)| 1.0 + 2.0 * t + 0.3 * z)
        .collect();
    let mut request = RecalcRequest {
        columns: data.columns,
        edges: data.edges,
        treatment: 0,
        outcome: 1,
        estimator: antecedent_estimate::AipwAte::new().with_bootstrap_replicates(0),
        target: None,
        utility: utility(),
    };
    let ctx = ExecutionContext::for_tests(83);
    let mut session = RecalcSession::new();
    let first = execute_with_attempt(|| execute_with_receipt(&mut session, &request, &ctx));
    assert!((first.result.unwrap().law.ate - 2.0).abs() < 1e-8);
    assert!(first.report.counts(Operation::LawSummary).completed > 0);
    let scores = session.score_table().unwrap().scores.as_ptr();
    let unchanged = execute_with_attempt(|| execute_with_receipt(&mut session, &request, &ctx));
    unchanged.result.unwrap();
    assert!(unchanged.report.is_empty());
    request.target = Some(TargetWeights {
        weights: request.columns[2].1.iter().map(|z| if *z > 0.5 { 3.0 } else { 1.0 }).collect(),
        depends_on: vec![VariableId::from_raw(2)],
    });
    let weighted = execute_with_attempt(|| execute_with_receipt(&mut session, &request, &ctx));
    let output = weighted.result.unwrap();
    assert!((output.law.ate - 2.0).abs() < 1e-8);
    assert_eq!(output.receipt.totals().fold_fits, 0);
    assert_eq!(weighted.report.counts(Operation::LawSummary).completed, 1);
    assert_eq!(weighted.report.counts(Operation::LearnerFit).attempted, 0);
    assert_eq!(weighted.report.counts(Operation::LeastSquaresSolve).attempted, 0);
    assert_eq!(session.score_table().unwrap().scores.as_ptr(), scores);
    let reused = execute_with_attempt(|| execute_with_receipt(&mut session, &request, &ctx));
    reused.result.unwrap();
    assert!(reused.report.is_empty());
}
