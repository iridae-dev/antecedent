use std::sync::Arc;

use super::*;
use crate::conjugate::fit_conjugate_gaussian;
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::gaussian_target::gaussian_target_from_model;
use crate::likelihood_terms::log_posterior_value;
use crate::posterior::PosteriorQuantityKind;
use crate::prior::{InvGammaPrior, PriorSpec};
use antecedent_core::{RngFactory, StreamDomain};

/// Exact known-σ² Gaussian posterior `N(m, C)` under the conjugate prior
/// `β | σ² ~ N(0, σ²·v0·I)` (`v0` is the [`GaussianCoefficientPrior`] scale, whose
/// *absolute* variance at known σ² is `σ²·v0`, per its doc and [`crate::conjugate`]'s
/// `posterior_known_sigma2`): `C = (X'X/σ² + I/(v0·σ²))⁻¹`, `m = C X'y/σ²`
/// (column-major `x`, `p` columns).
fn exact_known_sigma2_posterior(
    x: &[f64],
    y: &[f64],
    p: usize,
    v0: f64,
    sigma2: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0;
            for r in 0..n {
                acc += x[i * n + r] * x[j * n + r];
            }
            a[i * p + j] = acc / sigma2 + if i == j { 1.0 / (v0 * sigma2) } else { 0.0 };
        }
        b[i] = (0..n).map(|r| x[i * n + r] * y[r]).sum::<f64>() / sigma2;
    }
    let c = crate::linalg::invert_spd(&a, p).unwrap();
    let m = (0..p).map(|i| (0..p).map(|j| c[i * p + j] * b[j]).sum::<f64>()).collect();
    (m, c)
}

/// Sample mean and (population) variance of coefficient column `j`.
fn column_moments(fit: &BayesFitResult, j: usize) -> (f64, f64) {
    let n = fit.draws.n_draws;
    let col = &fit.draws.values[j * n..(j + 1) * n];
    let mean = col.iter().sum::<f64>() / n as f64;
    let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    (mean, var)
}

fn hmc_opts(n_warmup: usize, n_chains: usize) -> HmcOptions {
    HmcOptions {
        n_chains,
        n_warmup,
        leapfrog_steps: 12,
        step_size: 0.08,
        target_accept: 0.8,
        mass: 1.0,
        adapt_metric: false,
    }
}

#[test]
fn invalid_hmc_metric_step_and_target_are_refused() {
    let x = [1.0, 1.0];
    let y = [0.0, 1.0];
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: 2, ncols: 1, y: &y, weights: None, offsets: None };
    let prior = PriorSet::weakly_informative(1);
    let fit_opts = BayesFitOptions { n_draws: 2, seed: 1, max_iter: 50, grad_tol: 1e-8 };
    for value in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
        for hmc in [
            HmcOptions { step_size: value, ..hmc_opts(1, 2) },
            HmcOptions { mass: value, ..hmc_opts(1, 2) },
        ] {
            assert!(matches!(
                fit_hmc_glm(
                    BayesLikelihood::BernoulliLogit,
                    design,
                    &prior,
                    &fit_opts,
                    hmc,
                    &mut LaplaceWorkspace::default()
                ),
                Err(ProbError::Inference { .. })
            ));
        }
    }
    let hmc = HmcOptions { mass: f64::from_bits(1), ..hmc_opts(1, 2) };
    assert!(matches!(
        fit_hmc_glm(
            BayesLikelihood::BernoulliLogit,
            design,
            &prior,
            &fit_opts,
            hmc,
            &mut LaplaceWorkspace::default()
        ),
        Err(ProbError::Inference { .. })
    ));
    let hmc = HmcOptions { n_warmup: usize::MAX, ..hmc_opts(1, 2) };
    assert!(matches!(
        fit_hmc_glm(
            BayesLikelihood::BernoulliLogit,
            design,
            &prior,
            &fit_opts,
            hmc,
            &mut LaplaceWorkspace::default()
        ),
        Err(ProbError::Shape { message: "HMC transition count overflow" })
    ));
    for target_accept in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1.0, 2.0] {
        let hmc = HmcOptions { target_accept, ..hmc_opts(1, 2) };
        assert!(matches!(
            fit_hmc_glm(
                BayesLikelihood::BernoulliLogit,
                design,
                &prior,
                &fit_opts,
                hmc,
                &mut LaplaceWorkspace::default()
            ),
            Err(ProbError::Inference { .. })
        ));
    }
}

#[test]
fn absolute_scale_preprocessing_obeys_work_and_memory_before_ols() {
    let n = 128;
    let x = vec![1.0; n];
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y, weights: None, offsets: None };
    let mut prior = PriorSet::weakly_informative(1);
    prior.mark_absolute_coefficient_scale(&[0]);
    let options = BayesFitOptions { n_draws: 2, seed: 1, max_iter: 50, grad_tol: 1e-8 };
    let hmc = HmcOptions { n_warmup: 1, n_chains: 2, leapfrog_steps: 1, ..HmcOptions::default() };
    let mut workspace = LaplaceWorkspace::default();
    let limited = ExecutionContext::for_tests(1).with_hmc_work_limit(1);
    assert!(matches!(
        fit_hmc_glm_with_context(
            BayesLikelihood::GaussianIdentity,
            design,
            &prior,
            &options,
            hmc,
            &mut workspace,
            &limited
        ),
        Err(ProbError::Refused { code, .. }) if code == antecedent_core::reason_code!("hmc_work_budget")
    ));
    let constrained = ExecutionContext::for_tests(1).with_memory_limit(Some(1));
    assert!(matches!(
        fit_hmc_glm_with_context(
            BayesLikelihood::GaussianIdentity,
            design,
            &prior,
            &options,
            hmc,
            &mut workspace,
            &constrained
        ),
        Err(ProbError::Admission(_))
    ));
}

#[test]
fn direct_hmc_admits_workspace_growth_before_allocation() {
    let n = 4096;
    let x = vec![1.0; n];
    let y = vec![0.0; n];
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y, weights: None, offsets: None };
    let options = BayesFitOptions { n_draws: 2, seed: 1, max_iter: 50, grad_tol: 1e-8 };
    let hmc = HmcOptions { n_warmup: 1, n_chains: 2, leapfrog_steps: 1, ..HmcOptions::default() };
    let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(16 * 1024));
    let mut workspace = LaplaceWorkspace::default();
    assert!(matches!(
        fit_hmc_glm_with_context(
            BayesLikelihood::BernoulliLogit,
            design,
            &PriorSet::default(),
            &options,
            hmc,
            &mut workspace,
            &ctx
        ),
        Err(ProbError::Admission(_))
    ));
    assert!(workspace.eta.is_empty(), "workspace did not grow before admission");
}

#[test]
fn hmc_work_ceiling_refuses_without_truncating_schedule() {
    let x = [1.0; 4];
    let y = [0.0; 4];
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: 4, ncols: 1, y: &y, weights: None, offsets: None };
    let prior = PriorSet::default();
    let options = BayesFitOptions { n_draws: 10, seed: 1, max_iter: 50, grad_tol: 1e-8 };
    let hmc = HmcOptions { leapfrog_steps: u32::MAX, ..HmcOptions::default() };
    let ctx = ExecutionContext::for_tests(1).with_hmc_work_limit(100);
    let error = fit_hmc_glm_with_context(
        BayesLikelihood::BernoulliLogit,
        design,
        &prior,
        &options,
        hmc,
        &mut LaplaceWorkspace::default(),
        &ctx,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ProbError::Refused { code, .. } if code == antecedent_core::reason_code!("hmc_work_budget")
    ));
}

#[test]
fn hmc_pre_cancel_and_admission_cancellation_keep_the_typed_stage() {
    let x = [1.0, 1.0];
    let y = [0.0, 1.0];
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: 2, ncols: 1, y: &y, weights: None, offsets: None };
    let prior = PriorSet::weakly_informative(1);
    let options = BayesFitOptions { n_draws: 2, seed: 1, max_iter: 50, grad_tol: 1e-8 };
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    assert!(matches!(
        fit_hmc_impl(
            BayesLikelihood::BernoulliLogit,
            design,
            &prior,
            &options,
            hmc_opts(1, 2),
            &mut LaplaceWorkspace::default(),
            Some(&ctx)
        ),
        Err(ProbError::Cancelled { stage: "hmc.cancelled" })
    ));
    assert!(matches!(
        hmc_cancel_error(ProbError::Admission(
            antecedent_core::execution::AdmissionError::Cancelled
        )),
        ProbError::Cancelled { stage: "hmc.cancelled" }
    ));
    assert!(matches!(
        hmc_cancel_error(ProbError::Admission(
            antecedent_core::execution::AdmissionError::Overflow
        )),
        ProbError::Admission(antecedent_core::execution::AdmissionError::Overflow)
    ));
}

#[test]
fn hmc_worker_inherits_cancellation_and_reports_operational_refusal() {
    let x = [1.0, 1.0];
    let y = [0.0, 1.0];
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: 2, ncols: 1, y: &y, weights: None, offsets: None };
    let prior = PriorSet::weakly_informative(1);
    let fit_opts = BayesFitOptions { n_draws: 100, seed: 1, max_iter: 50, grad_tol: 1e-8 };
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(5);
    let error = fit_hmc_impl(
        BayesLikelihood::BernoulliLogit,
        design,
        &prior,
        &fit_opts,
        hmc_opts(10, 2),
        &mut LaplaceWorkspace::default(),
        Some(&ctx),
    )
    .unwrap_err();
    assert!(matches!(error, ProbError::Cancelled { stage: "hmc.cancelled" }));
}

#[test]
fn transition_acceptance_denominator_does_not_saturate_at_u32_max() {
    let mut stats = TransitionStats {
        n_transitions: u128::from(u32::MAX),
        accept_prob_sum: f64::from(u32::MAX),
        ..TransitionStats::default()
    };
    stats.record(
        &HmcStepResult {
            logp: 0.0,
            accepted: true,
            accept_prob: 1.0,
            delta_h: 0.0,
            divergent: false,
        },
        false,
    );
    assert_eq!(stats.mean_accept_prob(), 1.0);
}

#[test]
fn hmc_context_accounts_for_each_worker_workspace() {
    let x = [1.0, 1.0];
    let y = [0.0, 1.0];
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: 2, ncols: 1, y: &y, weights: None, offsets: None };
    let prior = PriorSet::weakly_informative(1);
    let fit_opts = BayesFitOptions { n_draws: 2, seed: 1, max_iter: 50, grad_tol: 1e-8 };
    // Two chains reserve 928 bytes of sampler/diagnostic state and 96 bytes
    // of draws. The full 1024-byte budget leaves no worker workspace.
    let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(1024));
    let result = fit_hmc_impl(
        BayesLikelihood::BernoulliLogit,
        design,
        &prior,
        &fit_opts,
        hmc_opts(1, 2),
        &mut LaplaceWorkspace::default(),
        Some(&ctx),
    );
    assert!(matches!(
        result,
        Err(ProbError::Admission(antecedent_core::execution::AdmissionError::ExceedsRemainder {
            requested_bytes: 96,
            remaining_bytes: 0,
            ..
        }))
    ));
}

#[test]
fn metric_windows_handle_the_largest_warmup_count() {
    let windows = metric_windows(usize::MAX);
    assert_eq!(windows.first().unwrap().0, 75);
    assert_eq!(windows.last().unwrap().1, usize::MAX - 50);
    assert!(windows.windows(2).all(|pair| pair[0].1 == pair[1].0));
    assert!(windows.iter().all(|&(start, end)| start < end));
}

#[test]
fn energy_error_above_threshold_is_divergent() {
    let (div, ap) = finalize_energy(1_001.0);
    assert!(div);
    assert_eq!(ap, 0.0);
    let (div2, ap2) = finalize_energy(0.5);
    assert!(!div2);
    assert!((ap2 - (-0.5_f64).exp()).abs() < 1e-12);
    let (div3, ap3) = finalize_energy(f64::NAN);
    assert!(div3);
    assert_eq!(ap3, 0.0);
}

#[test]
fn poisson_overflow_trajectory_is_divergent() {
    let n = 4;
    let x = vec![1.0; n];
    let y = vec![1.0; n];
    let prior = PriorSet::weakly_informative(1);
    let coef = prior.gaussian_coefficients().unwrap().clone();
    let prec = CoefficientPrecision::Diagonal(coef.precision());
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y, weights: None, offsets: None };
    let mut ws = LaplaceWorkspace::default();
    ws.prepare(n, 1, 8);
    let mut rng = CausalRng::from_seed(9);
    let beta = [0.0_f64];
    let lp_old = log_posterior_value(
        BayesLikelihood::PoissonLog,
        design,
        &beta,
        &coef,
        &prec,
        &mut ws.eta[..n],
        1.0,
    )
    .unwrap();
    // ∇U at eta = 0: Σ(y − e^η) = 0 and the prior mean is 0.
    let gradu = [0.0_f64];
    let step = hmc_step_glm(
        GlmTarget {
            likelihood: BayesLikelihood::PoissonLog,
            design,
            coef_prior: &coef,
            prec: &prec,
        },
        &beta,
        &gradu,
        Trajectory { step_size: 50.0, leapfrog_steps: 20, inv_mass: &[1.0] },
        lp_old,
        &mut ws,
        &mut rng,
    )
    .unwrap();
    assert!(step.divergent);
    assert!(!step.accepted);
}

#[test]
fn warmup_divergences_do_not_count_as_postwarmup() {
    let mut stats = TransitionStats::default();
    let warm = HmcStepResult {
        logp: 0.0,
        accepted: false,
        accept_prob: 0.0,
        delta_h: 1_001.0,
        divergent: true,
    };
    let ok = HmcStepResult {
        logp: 0.0,
        accepted: true,
        accept_prob: 0.7,
        delta_h: 0.1,
        divergent: false,
    };
    stats.record(&warm, true);
    stats.record(&ok, false);
    assert_eq!(stats.n_warmup_divergences, 1);
    assert_eq!(stats.n_postwarmup_divergences, 0);
    assert!((stats.mean_accept_prob() - 0.35).abs() < 1e-12);
}

#[test]
fn postwarmup_divergence_fails_publication_gate() {
    let mut d = InferenceDiagnostics {
        converged: true,
        iterations: 400,
        grad_inf_norm: 0.0,
        hessian_condition: f64::NAN,
        factorization: HessianFactorization::Mcmc,
        separation_warning: false,
        notes: Vec::new(),
        backend_id: Arc::from("hmc"),
        n_chains: Some(4),
        n_warmup: Some(100),
        // Healthy for the 4 chains below: the gate requires ~100 bulk/tail ESS per chain.
        ess_bulk_min: Some(500.0),
        ess_tail_min: Some(450.0),
        rhat_max: Some(1.0),
        n_divergences: Some(1),
        mean_accept_prob: Some(0.8),
        n_warmup_divergences: Some(0),
        n_postwarmup_divergences: Some(1),
        max_abs_delta_h: Some(0.2),
        all_chains_moved: Some(true),
    };
    assert!(!d.allows_posterior());
    d.n_postwarmup_divergences = Some(0);
    d.n_divergences = Some(0);
    d.converged = d.mcmc_publication_ok();
    assert!(d.allows_posterior());
}

#[test]
fn hmc_gaussian_recovers_slope() {
    let n = 80;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let xi = (r as f64 - 40.0) * 0.05;
        x[r] = 1.0;
        x[n + r] = xi;
        y[r] = 0.5 + 1.5 * xi + ((r % 5) as f64 - 2.0) * 0.2;
    }
    let prior = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(
                GaussianCoefficientPrior::shared(2, 0.0, 25.0).unwrap(),
            ),
            PriorSpec::KnownResidualVariance(1.0),
        ],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let mut ws = LaplaceWorkspace::default();
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y, weights: None, offsets: None };
    let fit_opts = BayesFitOptions { n_draws: 2000, seed: 42, max_iter: 50, grad_tol: 1e-8 };
    // Longer schedule + milder steps: R̂≤1.01 is tight on 2-parameter Gaussian HMC and was
    // flaking on Linux CI around 1.012 at step_size 0.04 (rhat=1.0123, ess_bulk=431 here);
    // 0.03 clears the gate with a wide margin (rhat=1.0018, ess_bulk=1836).
    let hmc = HmcOptions {
        n_chains: 4,
        n_warmup: 1500,
        leapfrog_steps: 16,
        step_size: 0.03,
        target_accept: 0.85,
        mass: 1.0,
        adapt_metric: false,
    };
    let fit =
        fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
            .expect("hmc fit");
    assert!(fit.diagnostics.allows_posterior());
    assert!(fit.diagnostics.rhat_max.unwrap() <= 1.01);
    assert!(fit.diagnostics.ess_bulk_min.unwrap() >= 100.0);
    assert!(fit.diagnostics.ess_tail_min.unwrap() >= 100.0);
    assert_eq!(fit.diagnostics.n_postwarmup_divergences, Some(0));
    assert_eq!(fit.diagnostics.all_chains_moved, Some(true));
    // The reported point estimate is the posterior mean of the retained
    // draws, so it must sit within Monte Carlo error of the exact posterior mean.
    let (m, c) = exact_known_sigma2_posterior(&x, &y, 2, 25.0, 1.0);
    let ess = fit.diagnostics.ess_bulk_min.unwrap();
    for j in 0..2 {
        let tol = 6.0 * (c[j * 2 + j] / ess).sqrt();
        assert!(
            (fit.map[j] - m[j]).abs() < tol,
            "map[{j}] {} vs exact {} tol {tol}",
            fit.map[j],
            m[j]
        );
        let (mean, _) = column_moments(&fit, j);
        assert!((fit.map[j] - mean).abs() < 1e-12, "map is the mean of the retained draws");
    }
    assert_eq!(fit.draws.n_draws, 8000);
    assert_eq!(fit.draws.schema.n_quantities(), 2);
}

#[test]
fn hmc_known_sigma2_one_predictor_matches_conjugate() {
    let n = 30;
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for r in 0..n {
        x[r] = 1.0;
        y[r] = 2.0 + ((r % 4) as f64 - 1.5) * 0.05;
    }
    let prior = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::shared(1, 0.0, 4.0).unwrap()),
            PriorSpec::KnownResidualVariance(0.16),
        ],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y, weights: None, offsets: None };
    let mut ws = LaplaceWorkspace::default();
    let fit_opts = BayesFitOptions { n_draws: 600, seed: 5, max_iter: 50, grad_tol: 1e-8 };
    let hmc = hmc_opts(600, 4);
    let fit =
        fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
            .unwrap();
    // Exact posterior: precision n/σ² + 1/v0, mean (Σy/σ²)/precision.
    let (m, c) = exact_known_sigma2_posterior(&x, &y, 1, 4.0, 0.16);
    let (mean, _) = column_moments(&fit, 0);
    let ess = fit.diagnostics.ess_bulk_min.unwrap();
    let tol = 6.0 * (c[0] / ess).sqrt();
    assert!((mean - m[0]).abs() < tol, "mean hmc={mean} exact={} tol={tol}", m[0]);
}

#[test]
fn hmc_known_sigma2_matches_conjugate_moments() {
    let n = 40;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let xi = (r as f64 - 20.0) * 0.1;
        x[r] = 1.0;
        x[n + r] = xi;
        y[r] = 1.0 + 2.0 * xi + ((r % 7) as f64 - 3.0) * 0.02;
    }
    let sigma2 = 0.25;
    let prior = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::shared(2, 0.0, 4.0).unwrap()),
            PriorSpec::KnownResidualVariance(sigma2),
        ],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y, weights: None, offsets: None };
    let mut ws = LaplaceWorkspace::default();
    let fit_opts = BayesFitOptions { n_draws: 800, seed: 3, max_iter: 50, grad_tol: 1e-8 };
    let hmc = hmc_opts(800, 4);
    let fit =
        fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
            .unwrap();

    // Exact known-σ² posterior; tolerances are Monte Carlo errors from the
    // reported ESS (mean: sd/√ESS; variance: relative √(2/ESS) for a Gaussian).
    let (m, c) = exact_known_sigma2_posterior(&x, &y, 2, 4.0, sigma2);
    let ess = fit.diagnostics.ess_bulk_min.unwrap();
    for j in 0..2 {
        let (mean, var) = column_moments(&fit, j);
        let sd = c[j * 2 + j].sqrt();
        assert!(
            (mean - m[j]).abs() < 6.0 * sd / ess.sqrt(),
            "coef {j} mean hmc={mean} exact={}",
            m[j]
        );
        let c_var = c[j * 2 + j];
        assert!(
            (var - c_var).abs() / c_var < 6.0 * (2.0 / ess).sqrt(),
            "coef {j} var hmc={var} exact={c_var}"
        );
    }
}

#[test]
fn hmc_nig_matches_conjugate_moments() {
    let n = 50;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let xi = (r as f64) * 0.08;
        x[r] = 1.0;
        x[n + r] = xi;
        y[r] = 0.5 + 1.2 * xi + ((r % 5) as f64 - 2.0) * 0.1;
    }
    let prior = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(
                GaussianCoefficientPrior::shared(2, 0.0, 10.0).unwrap(),
            ),
            PriorSpec::ResidualInvGamma(InvGammaPrior { shape: 2.0, scale: 1.0 }),
        ],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y, weights: None, offsets: None };
    let mut ws = LaplaceWorkspace::default();
    let fit_opts = BayesFitOptions { n_draws: 2000, seed: 17, max_iter: 50, grad_tol: 1e-8 };
    let conj = fit_conjugate_gaussian(design, &prior, &fit_opts, &mut ws).unwrap();
    // 1000 warmup / step 0.04 was refused by the R-hat gate for 10 of 500 seeds, so this
    // passed only on its fixed seed and flipped on Linux CI (rhat=1.0107). The schedule of
    // the sibling test above (1500 warmup, step 0.03, 2000 draws) is refused for none of 500.
    let hmc = HmcOptions {
        n_chains: 4,
        n_warmup: 1500,
        leapfrog_steps: 16,
        step_size: 0.03,
        target_accept: 0.85,
        mass: 1.0,
        adapt_metric: false,
    };
    let fit =
        fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
            .unwrap();
    assert_eq!(fit.draws.schema.n_quantities(), 3);
    assert!(matches!(fit.draws.schema.quantities[2], PosteriorQuantityKind::ResidualVariance));

    // HMC mean vs the exact (iid) conjugate draws: the difference has
    // variance sd²/ESS_hmc + sd²/n_conj, so 6 of those SEs bounds it.
    let ess = fit.diagnostics.ess_bulk_min.unwrap();
    let n_draws = fit.draws.n_draws;
    let n_conj = conj.draws.n_draws;
    for j in 0..3 {
        let h = &fit.draws.values[j * n_draws..(j + 1) * n_draws];
        let c = &conj.draws.values[j * n_conj..(j + 1) * n_conj];
        let h_mean = h.iter().sum::<f64>() / n_draws as f64;
        let c_mean = c.iter().sum::<f64>() / n_conj as f64;
        let c_var = c.iter().map(|v| (v - c_mean).powi(2)).sum::<f64>() / (n_conj - 1) as f64;
        let tol = 6.0 * (c_var / ess + c_var / n_conj as f64).sqrt();
        assert!(
            (h_mean - c_mean).abs() < tol,
            "quantity {j} mean hmc={h_mean} conj={c_mean} tol={tol}"
        );
    }
}

#[test]
fn leapfrog_is_reversible() {
    let n = 8;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for r in 0..n {
        x[r] = 1.0;
        x[n + r] = r as f64;
        y[r] = 1.0 + 0.5 * (r as f64);
    }
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y, weights: None, offsets: None };
    let prior = GaussianCoefficientPrior::shared(2, 0.0, 4.0).unwrap();
    let mut target = gaussian_target_from_model(
        design,
        prior,
        GaussianVarianceModel::InvGamma { shape: 2.0, scale: 1.0 },
    )
    .unwrap();
    let q0 = vec![0.5, 0.4, 0.0];
    let p0 = vec![0.3, -0.2, 0.1];
    let step = 0.01;
    let l = 20;
    let mass = 1.0;
    let (q1, p1) = leapfrog_trajectory(&mut target, &q0, &p0, step, l, mass).unwrap();
    let p_neg: Vec<_> = p1.iter().map(|v| -v).collect();
    let (q2, p2) = leapfrog_trajectory(&mut target, &q1, &p_neg, step, l, mass).unwrap();
    for i in 0..3 {
        assert!((q2[i] - q0[i]).abs() < 1e-8, "q[{i}] {} vs {}", q2[i], q0[i]);
        assert!((p2[i] + p0[i]).abs() < 1e-8, "p[{i}] {} vs {}", p2[i], -p0[i]);
    }
}

/// Standard-normal target that counts density/gradient sweeps.
struct CountingTarget {
    calls: usize,
}

impl PosteriorTarget for CountingTarget {
    fn dim(&self) -> usize {
        2
    }

    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        self.calls += 1;
        for i in 0..2 {
            grad[i] = -q[i];
        }
        Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
    }
}

#[test]
fn transition_costs_exactly_one_sweep_per_leapfrog_step() {
    let leapfrog = 7;
    let mut target = CountingTarget { calls: 0 };
    let mut ws = LaplaceWorkspace::default();
    ws.prepare(1, 2, 1);
    let mut rng = CausalRng::from_seed(11);
    let q0 = [0.3, -0.4];
    let lp_old = -0.5 * (q0[0] * q0[0] + q0[1] * q0[1]);
    let grad_old = [-q0[0], -q0[1]];
    let step = hmc_step_target(
        &mut target,
        &q0,
        lp_old,
        &grad_old,
        Trajectory { step_size: 0.05, leapfrog_steps: leapfrog, inv_mass: &[1.0, 1.0] },
        &mut ws,
        &mut rng,
    )
    .unwrap();
    assert_eq!(target.calls, leapfrog as usize, "gradient at q0 and the final density are reused");
    assert!(step.accepted, "a 0.05-step trajectory on N(0, I) is accepted");
    // The reported density is that of the trajectory's final state.
    let q = &ws.q[..2];
    assert!((step.logp - (-0.5 * (q[0] * q[0] + q[1] * q[1]))).abs() < 1e-12);
    // And the workspace gradient is the score at that state, ready to carry forward.
    assert!((ws.grad[0] + q[0]).abs() < 1e-12 && (ws.grad[1] + q[1]).abs() < 1e-12);
}

#[test]
fn dual_averaging_shrinks_toward_ten_times_initial_step() {
    // With acceptance exactly on target, H̄ stays 0 and the iterate is the
    // shrinkage point μ = ln(10 ε₀) = ln 0.2.
    let hmc = HmcOptions { step_size: 0.02, target_accept: 0.8, ..HmcOptions::default() };
    let (mut h_bar, mut log_eps_bar, mut step) = (0.0, hmc.step_size.ln(), hmc.step_size);
    dual_average_update(&mut h_bar, &mut log_eps_bar, &mut step, hmc, 0, 0.8);
    assert!((step - 0.2).abs() < 1e-12, "step {step}");
    assert!((log_eps_bar.exp() - 0.2).abs() < 1e-12);
}

#[test]
fn averaged_step_never_exceeds_a_step_that_was_tried() {
    // Perfect acceptance drives the raw iterate far above the cap; the
    // averaged log-step must be built from the clamped iterates.
    let hmc = HmcOptions::default();
    let (mut h_bar, mut log_eps_bar, mut step) = (0.0, hmc.step_size.ln(), hmc.step_size);
    for t in 0..400 {
        dual_average_update(&mut h_bar, &mut log_eps_bar, &mut step, hmc, t, 1.0);
        assert!(step <= MAX_STEP + 1e-15);
        assert!(log_eps_bar <= MAX_STEP.ln() + 1e-12, "t={t} log_eps_bar={log_eps_bar}");
    }
    assert!(dual_average_finalize(log_eps_bar) <= MAX_STEP);
}

#[test]
fn chain_streams_come_from_the_stream_mixer() {
    let seed = 91;
    for chain in 0..4 {
        let mut ours = chain_rng(seed, chain);
        let mut mixed =
            RngFactory::from_seed(seed).stream_for(StreamDomain::Bayesian, chain as u64);
        assert_eq!(ours.next_u64(), mixed.next_u64());
    }
    let firsts: Vec<u64> = (0..4).map(|c| chain_rng(seed, c).next_u64()).collect();
    for a in 0..4 {
        for b in (a + 1)..4 {
            assert_ne!(firsts[a], firsts[b]);
        }
    }
}

#[test]
fn threaded_chains_reproduce_serial_draws() {
    let n = 30;
    let x = vec![1.0; n];
    let y: Vec<f64> = (0..n).map(|r| 2.0 + ((r % 4) as f64 - 1.5) * 0.05).collect();
    let prior = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::shared(1, 0.0, 4.0).unwrap()),
            PriorSpec::KnownResidualVariance(0.16),
        ],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y, weights: None, offsets: None };
    let fit_opts = BayesFitOptions { n_draws: 600, seed: 5, max_iter: 50, grad_tol: 1e-8 };
    let hmc = hmc_opts(600, 4);
    let mut ws = LaplaceWorkspace::default();
    let serial =
        fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
            .unwrap();
    let ctx = ExecutionContext::production(5, 4);
    let threaded = fit_hmc_impl(
        BayesLikelihood::GaussianIdentity,
        design,
        &prior,
        &fit_opts,
        hmc,
        &mut LaplaceWorkspace::default(),
        Some(&ctx),
    )
    .unwrap();
    assert_eq!(serial.draws.values, threaded.draws.values);
    assert_eq!(serial.map, threaded.map);
}

#[test]
fn metric_windows_follow_the_stan_schedule() {
    // Too short to adapt.
    assert!(metric_windows(99).is_empty());
    // Short warmups: one window between proportional buffers (15% / 10%).
    assert_eq!(metric_windows(100), vec![(15, 90)]);
    // Stan's 75 / 25-doubling / 50 layout; a doubled window that would not fit is absorbed.
    assert_eq!(metric_windows(200), vec![(75, 100), (100, 150)]);
    assert_eq!(
        metric_windows(1000),
        vec![(75, 100), (100, 150), (150, 250), (250, 450), (450, 950)]
    );
    for n in [100, 137, 200, 350, 1000, 1500] {
        let windows = metric_windows(n);
        assert!(windows.windows(2).all(|w| w[0].1 == w[1].0), "n={n}");
        assert!(windows.iter().all(|(s, e)| s < e), "n={n}");
        assert!(windows.last().unwrap().1 < n, "n={n}");
    }
}

#[test]
fn diagonal_metric_is_the_regularised_window_variance() {
    let hmc = HmcOptions { n_warmup: 100, ..HmcOptions::default() };
    let mut metric = DiagonalMetric::new(2, hmc);
    assert_eq!(metric.inv_mass, vec![1.0, 1.0]);
    let mut installed_at = None;
    for t in 0..100 {
        // Coordinate 0 walks 0, 1, 2, ...; coordinate 1 is ten times that.
        let q = [t as f64, 10.0 * t as f64];
        metric.observe(t, &q);
        if metric.install_if_window_closed(t) {
            installed_at = Some(t);
        }
    }
    // The single window is iterations 15..90: n = 75 consecutive integers, whose sample
    // variance is n(n + 1)/12.
    assert_eq!(installed_at, Some(89));
    let n = 75.0_f64;
    let var0 = n * (n + 1.0) / 12.0;
    let weight = n / (n + 5.0);
    let expect = |var: f64| weight * var + (1.0 - weight) * 1e-3;
    assert!((metric.inv_mass[0] - expect(var0)).abs() < 1e-9 * var0);
    assert!((metric.inv_mass[1] - expect(100.0 * var0)).abs() < 1e-9 * 100.0 * var0);

    // Disabled adaptation and too-short warmups leave the initial metric alone.
    for hmc in [
        HmcOptions { n_warmup: 100, adapt_metric: false, mass: 4.0, ..HmcOptions::default() },
        HmcOptions { n_warmup: 60, mass: 4.0, ..HmcOptions::default() },
    ] {
        let mut metric = DiagonalMetric::new(2, hmc);
        for t in 0..hmc.n_warmup {
            metric.observe(t, &[t as f64, 1.0]);
            assert!(!metric.install_if_window_closed(t));
        }
        assert_eq!(metric.inv_mass, vec![0.25, 0.25]);
    }
}

/// A design whose columns differ in scale by 10^2 puts the posterior scales of intercept
/// and slope ~10^2 apart: a unit-mass leapfrog step small enough for the slope crawls along
/// the intercept (asserted below: the unadapted sampler fails the gate on this same
/// design). With the adapted diagonal metric the fit publishes and recovers the exact
/// known-variance posterior.
///
/// Stan's own regularisation of the windowed variance estimate toward a fixed floor
/// (`1e-3 · 5/(n+5)`, mirrored in [`DiagonalMetric::install_if_window_closed`]) assumes
/// coordinates are within a few orders of magnitude of unit scale; a 10^4 column scale
/// pushes the slope's true posterior variance (~2e-10) far enough below that floor that
/// the metric estimate never escapes it within a 1500-iteration warmup (confirmed by
/// running it out to 10000 iterations, where it does converge) — the same bootstrapping
/// limit a real Stan run would hit on this unstandardised parameterisation. 10^2 is
/// comfortably inside the range this warmup budget resolves (bulk ESS in the
/// thousands, R-hat well under the 1.01 gate) while still failing the unadapted,
/// unit-mass sampler outright.
#[test]
fn adapted_metric_samples_a_badly_scaled_design() {
    let n = 60;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let z = (r as f64 - 30.0) * 0.05;
        x[r] = 1.0;
        x[n + r] = 1.0e2 * z;
        y[r] = 0.5 + 1.5 * z + ((r % 5) as f64 - 2.0) * 0.2;
    }
    let prior = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(
                GaussianCoefficientPrior::shared(2, 0.0, 25.0).unwrap(),
            ),
            PriorSpec::KnownResidualVariance(1.0),
        ],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y, weights: None, offsets: None };
    let fit_opts = BayesFitOptions { n_draws: 2000, seed: 42, max_iter: 50, grad_tol: 1e-8 };
    let hmc = HmcOptions {
        n_chains: 4,
        n_warmup: 1500,
        leapfrog_steps: 16,
        step_size: 0.04,
        target_accept: 0.85,
        mass: 1.0,
        adapt_metric: true,
    };
    let mut ws = LaplaceWorkspace::default();
    let unadapted = HmcOptions { adapt_metric: false, ..hmc };
    assert!(
        fit_hmc_glm(
            BayesLikelihood::GaussianIdentity,
            design,
            &prior,
            &fit_opts,
            unadapted,
            &mut ws,
        )
        .is_err(),
        "the unadapted, unit-mass sampler should fail the gate on this design"
    );
    let fit =
        fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
            .expect("the adapted sampler publishes");
    let (m, c) = exact_known_sigma2_posterior(&x, &y, 2, 25.0, 1.0);
    let ess = fit.diagnostics.ess_bulk_min.unwrap();
    for j in 0..2 {
        let (mean, var) = column_moments(&fit, j);
        let sd = c[j * 2 + j].sqrt();
        assert!((mean - m[j]).abs() < 6.0 * sd / ess.sqrt(), "mean[{j}] {mean} vs {}", m[j]);
        assert!((var.sqrt() / sd - 1.0).abs() < 0.15, "sd[{j}] {} vs {sd}", var.sqrt());
    }
}
