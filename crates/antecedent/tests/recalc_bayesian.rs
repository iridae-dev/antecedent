//! Native issued-posterior retention and independent Gaussian algebra.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::BayesianConfig;
use antecedent::analysis::recalc_bayesian::{
    BayesianModel, BayesianRequest, BayesianSession, PosteriorSummarySpec,
    execute_bayesian_with_receipt,
};
use antecedent::analysis::recalc_receipt::{RecalcRunError, UtilitySpec};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_identify::execution_counts::count_checked_identifications;
use antecedent_io::consume_analysis_result;
use antecedent_prob::fit_counts::count_bayesian_work;
use antecedent_prob::{GaussianCoefficientPrior, PriorSet, PriorSpec};

fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/recalculation/bayesian_retention/expected.json"
    )))
    .unwrap()
}

fn request() -> BayesianRequest {
    let mut t = Vec::new();
    let mut y = Vec::new();
    for _ in 0..64 {
        for action in [0.0, 1.0] {
            for noise in [-0.4, 0.4] {
                t.push(action);
                y.push(1.0 + 2.0 * action + noise);
            }
        }
    }
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 10.0)));
    prior.push(PriorSpec::KnownResidualVariance(1.0));
    BayesianRequest {
        columns: vec![("t".into(), t), ("y".into(), y)],
        edges: vec![(0, 1)],
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(1),
        model: BayesianModel::Gaussian,
        inference: BayesianConfig::conjugate().n_draws(16_384).prior(prior),
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    }
}
#[test]
fn checked_gaussian_matches_analytic_posterior_and_retains_issued_rows() {
    run_on_large_stack(checked_gaussian_matches_analytic_posterior_and_retains_issued_rows_body);
}
fn checked_gaussian_matches_analytic_posterior_and_retains_issued_rows_body() {
    let context = ExecutionContext::for_tests(41);
    let mut request = request();
    let mut session = BayesianSession::new();
    let ((out, checks), work) = count_bayesian_work(|| {
        count_checked_identifications(|| {
            execute_bayesian_with_receipt(&mut session, &request, &context).unwrap()
        })
    });
    assert_eq!(out.receipt.totals().identifications, checks);
    assert!(checks > 0);
    assert_eq!(work.model_fits, 1);
    assert_eq!(work.posterior_draws, 16_384);
    assert_eq!(out.receipt.totals().model_fits, work.model_fits);
    assert_eq!(out.receipt.totals().posterior_draws, work.posterior_draws);
    // Inverse of [[256+.01,128],[128,128+.01]], times X'y=[512,384].
    let a: f64 = 256.01;
    let b = 128.0;
    let c = 128.01;
    let determinant = a * c - b * b;
    let expected_mean = (a * 384.0 - b * 512.0) / determinant;
    let expected_sd = (a / determinant).sqrt();
    assert!((expected_mean - oracle()["gaussian"]["mean"].as_f64().unwrap()).abs() < 1e-12);
    assert!(
        (expected_sd.powi(2) - oracle()["gaussian"]["variance"].as_f64().unwrap()).abs() < 1e-12
    );
    assert!((out.law.mean - expected_mean).abs() < expected_sd * 0.06);
    assert!((out.law.standard_deviation / expected_sd - 1.0).abs() < 0.03);
    assert!(out.law.lower_quantile < expected_mean && out.law.upper_quantile > expected_mean);
    let rows = session.posterior().unwrap().draws.column(0).unwrap().as_ptr();
    let ((), unchanged) = count_bayesian_work(|| {
        let out = execute_bayesian_with_receipt(&mut session, &request, &context).unwrap();
        assert_eq!(out.receipt.totals().total(), 0);
    });
    assert_eq!(unchanged.model_fits + unchanged.posterior_draws, 0);
    request.summary.threshold = 2.0;
    let ((summary, checks), work) = count_bayesian_work(|| {
        count_checked_identifications(|| {
            execute_bayesian_with_receipt(&mut session, &request, &context).unwrap()
        })
    });
    assert_eq!(checks, 0);
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    assert_eq!(summary.receipt.totals().law_summaries, 1);
    assert_eq!(session.posterior().unwrap().draws.column(0).unwrap().as_ptr(), rows);
    request.utility.cost = 5.0;
    let utility = execute_bayesian_with_receipt(&mut session, &request, &context).unwrap();
    assert_eq!(utility.receipt.totals().decisions, 1);
    assert_eq!(utility.receipt.totals().law_summaries, 0);
    assert!(!utility.decision.treat);
    consume_analysis_result(&session.export_result(&context).unwrap()).unwrap();
}
#[test]
fn configuration_rebind_preserves_causal_proof_and_errors_are_transactional() {
    run_on_large_stack(
        configuration_rebind_preserves_causal_proof_and_errors_are_transactional_body,
    );
}
fn configuration_rebind_preserves_causal_proof_and_errors_are_transactional_body() {
    let context = ExecutionContext::for_tests(43);
    let mut request = request();
    let mut session = BayesianSession::new();
    execute_bayesian_with_receipt(&mut session, &request, &context).unwrap();
    request.inference.n_draws = 128;
    let ((out, checks), work) = count_bayesian_work(|| {
        count_checked_identifications(|| {
            execute_bayesian_with_receipt(&mut session, &request, &context).unwrap()
        })
    });
    assert_eq!(checks, 0);
    assert_eq!(out.law.draws, 128);
    assert_eq!(work.model_fits, 1);
    assert_eq!(work.posterior_draws, 128);
    assert_eq!(out.receipt.totals().identifications, 0);
    consume_analysis_result(&session.export_result(&context).unwrap()).unwrap();
    let cancelled = ExecutionContext::for_tests(43);
    cancelled.cancellation.cancel();
    request.inference.n_draws = 256;
    let (refusal, work) =
        count_bayesian_work(|| execute_bayesian_with_receipt(&mut session, &request, &cancelled));
    assert!(matches!(refusal, Err(RecalcRunError::Request("recalc.cancelled"))));
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    request.inference.n_draws = 128;
    assert_eq!(
        execute_bayesian_with_receipt(&mut session, &request, &context)
            .unwrap()
            .receipt
            .totals()
            .total(),
        0
    );
}

// Original checked Study preparation also uses this established debug-test stack size.
fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(8 * 1024 * 1024).spawn(run).unwrap().join().unwrap();
}
#[test]
fn quadratic_basis_matches_independent_nig_contrast_and_refuses_external_prior() {
    run_on_large_stack(quadratic_basis_matches_independent_nig_contrast_body);
}
fn quadratic_basis_matches_independent_nig_contrast_body() {
    let mut treatment = Vec::new();
    let mut z = Vec::new();
    let mut outcome = Vec::new();
    for _ in 0..20 {
        for action in [0.0, 1.0] {
            for covariate in [-1.0, 0.0, 1.0] {
                for noise in [-0.2, 0.2] {
                    treatment.push(action);
                    z.push(covariate);
                    outcome.push(
                        1.0 + 2.0 * action
                            + 0.5 * covariate
                            + 0.4 * action * covariate
                            + 0.3 * covariate * covariate
                            + noise,
                    );
                }
            }
        }
    }
    let mut precision = vec![vec![0.0; 5]; 5];
    let mut moment = [0.0; 5];
    let design =
        treatment.iter().zip(&z).map(|(t, z)| [1.0, *t, *z, *t * *z, *z * *z]).collect::<Vec<_>>();
    for (x, y) in design.iter().zip(&outcome) {
        for i in 0..5 {
            moment[i] += x[i] * y;
            for j in 0..5 {
                precision[i][j] += x[i] * x[j];
            }
        }
    }
    for (i, row) in precision.iter_mut().enumerate() {
        row[i] += 0.01;
    }
    let inverse = independent_inverse(precision);
    let mean =
        (0..5).map(|i| (0..5).map(|j| inverse[i][j] * moment[j]).sum::<f64>()).collect::<Vec<_>>();
    let rss = design
        .iter()
        .zip(&outcome)
        .map(|(x, y)| {
            let prediction = x.iter().zip(&mean).map(|(x, b)| x * b).sum::<f64>();
            (y - prediction).powi(2)
        })
        .sum::<f64>();
    let prior_penalty = 0.01 * mean.iter().map(|b| b * b).sum::<f64>();
    let expected_sd =
        ((0.001 + 0.5 * (rss + prior_penalty)) / (0.001 + 120.0 - 1.0) * inverse[1][1]).sqrt();
    assert!(
        (mean[1] - oracle()["quadratic_basis"]["coefficient_mean"][1].as_f64().unwrap()).abs()
            < 1e-12
    );
    assert!(
        (expected_sd.powi(2) - oracle()["quadratic_basis"]["effect_variance"].as_f64().unwrap())
            .abs()
            < 1e-12
    );
    let mut request = BayesianRequest {
        columns: vec![("t".into(), treatment), ("z".into(), z), ("y".into(), outcome)],
        edges: vec![(0, 2), (1, 0), (1, 2)],
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(2),
        model: BayesianModel::QuadraticBasis,
        inference: BayesianConfig::conjugate().n_draws(16_384),
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    };
    let ctx = ExecutionContext::for_tests(47);
    let mut session = BayesianSession::new();
    let (out, work) = count_bayesian_work(|| {
        execute_bayesian_with_receipt(&mut session, &request, &ctx).unwrap()
    });
    assert_eq!(work.model_fits, 1);
    assert_eq!(work.posterior_draws, 16_384);
    assert!((out.law.mean - mean[1]).abs() < expected_sd * 0.06);
    assert!((out.law.standard_deviation / expected_sd - 1.0).abs() < 0.03);
    consume_analysis_result(&session.export_result(&ctx).unwrap()).unwrap();
    request.inference.prior = Some(PriorSet::weakly_informative(5));
    let (failure, work) =
        count_bayesian_work(|| execute_bayesian_with_receipt(&mut session, &request, &ctx));
    assert!(matches!(
        failure,
        Err(RecalcRunError::Request("recalc.bayesian_basis_prior_unsupported"))
    ));
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    request.inference.prior = None;
    assert_eq!(
        execute_bayesian_with_receipt(&mut session, &request, &ctx)
            .unwrap()
            .receipt
            .totals()
            .total(),
        0
    );
}
// Independent fixture algebra; no model or linear-solver code under test is used.
fn independent_inverse(mut matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for (i, row) in inverse.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for pivot in 0..n {
        let scale = matrix[pivot][pivot];
        assert!(scale.abs() > 1e-10);
        for j in 0..n {
            matrix[pivot][j] /= scale;
            inverse[pivot][j] /= scale;
        }
        for row in 0..n {
            if row != pivot {
                let factor = matrix[row][pivot];
                for j in 0..n {
                    matrix[row][j] -= factor * matrix[pivot][j];
                    inverse[row][j] -= factor * inverse[pivot][j];
                }
            }
        }
    }
    inverse
}
#[test]
fn independently_bound_prior_transfer_replays_source_and_refuses_known_double_use() {
    run_on_large_stack(independently_bound_prior_transfer_body);
}
fn independently_bound_prior_transfer_body() {
    let ctx = ExecutionContext::for_tests(53);
    let mut source = request();
    source.inference.n_draws = 512;
    let mut source_session = BayesianSession::new();
    execute_bayesian_with_receipt(&mut source_session, &source, &ctx).unwrap();
    let bytes = source_session.export_prior_source().unwrap();
    let mut same = source.clone();
    same.inference.prior = None;
    same.inference.prior_artifact = Some(bytes.clone().into());
    let mut refused = BayesianSession::new();
    let (failure, work) =
        count_bayesian_work(|| execute_bayesian_with_receipt(&mut refused, &same, &ctx));
    assert!(matches!(
        failure,
        Err(RecalcRunError::Request("recalc.bayesian_prior_likelihood_double_use"))
    ));
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    let mut target = same.clone();
    for (t, y) in target.columns[0].1.clone().iter().zip(&mut target.columns[1].1) {
        *y += 2.0 * t;
    }
    let mut session = BayesianSession::new();
    let ((out, checks), work) = count_bayesian_work(|| {
        count_checked_identifications(|| {
            execute_bayesian_with_receipt(&mut session, &target, &ctx).unwrap()
        })
    });
    assert_eq!(out.receipt.totals().identifications, checks);
    assert!(checks >= 4); // actual independent source proof replay and target proof
    assert_eq!(work.model_fits, 2);
    assert_eq!(work.posterior_draws, 1024);
    let transfer = &oracle()["prior_transfer"];
    assert!(
        out.law.mean > transfer["source_effect"].as_f64().unwrap()
            && out.law.mean < transfer["target_effect"].as_f64().unwrap()
    );
    let consumed = consume_analysis_result(&session.export_result(&ctx).unwrap()).unwrap();
    let source_wire =
        antecedent_io::bayesian_prior_recalc_artifact::BoundBayesianPriorWire::decode(&bytes)
            .unwrap();
    let source_snapshot =
        source_wire.source_snapshot.iter().fold(String::new(), |mut text, byte| {
            use std::fmt::Write;
            write!(text, "{byte:02x}").unwrap();
            text
        });
    assert_eq!(
        consumed
            .contract
            .unwrap()
            .inference_binding
            .unwrap()
            .bayesian
            .unwrap()
            .prior_source_snapshot,
        Some(source_snapshot)
    );
    target.inference.n_draws = 256;
    let ((out, checks), work) = count_bayesian_work(|| {
        count_checked_identifications(|| {
            execute_bayesian_with_receipt(&mut session, &target, &ctx).unwrap()
        })
    });
    assert_eq!(checks, 0);
    assert_eq!(work.model_fits, 1);
    assert_eq!(work.posterior_draws, 256);
    assert_eq!(out.law.draws, 256);
    let mut fresh = BayesianSession::resume(
        session.identities().clone(),
        antecedent_core::recalc::ResumeContext {
            supplied_data: true,
            ..antecedent_core::recalc::ResumeContext::default()
        },
    );
    let (replayed, work) =
        count_bayesian_work(|| execute_bayesian_with_receipt(&mut fresh, &target, &ctx).unwrap());
    assert_eq!(work.model_fits, 2);
    assert_eq!(work.posterior_draws, 768);
    assert_eq!(replayed.law, out.law);
    let fresh_consumed = consume_analysis_result(&fresh.export_result(&ctx).unwrap()).unwrap();
    assert!(
        fresh_consumed
            .contract
            .unwrap()
            .inference_binding
            .unwrap()
            .bayesian
            .unwrap()
            .prior_source_snapshot
            .is_some()
    );
    assert_corrupted_source_refused(&mut session, &target, &bytes, &ctx);
}
fn assert_corrupted_source_refused(
    session: &mut BayesianSession,
    target: &BayesianRequest,
    bytes: &[u8],
    ctx: &ExecutionContext,
) {
    let mut tampered =
        antecedent_io::bayesian_prior_recalc_artifact::BoundBayesianPriorWire::decode(bytes)
            .unwrap();
    tampered.columns[1].1[0] = (123.0_f64).to_bits();
    let mut invalid = target.clone();
    invalid.inference.prior_artifact = Some(tampered.export().unwrap().into());
    let (failure, work) =
        count_bayesian_work(|| execute_bayesian_with_receipt(session, &invalid, ctx));
    assert!(matches!(
        failure,
        Err(RecalcRunError::Request("recalc.bayesian_prior_source_unverified"))
    ));
    assert_eq!(work.model_fits + work.posterior_draws, 0);
    // Updating the raw-data seal does not authenticate stale posterior bytes:
    // the independent producer replay must still execute and reject them.
    let decoded_columns: Vec<(String, Vec<f64>)> = tampered
        .columns
        .iter()
        .map(|(name, bits)| (name.clone(), bits.iter().copied().map(f64::from_bits).collect()))
        .collect();
    let borrowed: Vec<(&str, &[f64])> =
        decoded_columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    let data = antecedent_data::TabularData::from_f64_columns(borrowed).unwrap();
    tampered.source_snapshot = data.storage().content_digest();
    invalid.inference.prior_artifact = Some(tampered.export().unwrap().into());
    let (failure, work) =
        count_bayesian_work(|| execute_bayesian_with_receipt(session, &invalid, ctx));
    assert!(matches!(
        failure,
        Err(RecalcRunError::Request("recalc.bayesian_prior_source_unverified"))
    ));
    assert_eq!(work.model_fits, 1);
    assert_eq!(work.posterior_draws, 512);
    assert_eq!(
        execute_bayesian_with_receipt(session, target, ctx).unwrap().receipt.totals().total(),
        0
    );
}
