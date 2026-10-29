//! Tests for the binomial (logistic / probit) fits and separation handling.
#![allow(unused_imports)]

use super::*;
use crate::faer_backend::FaerBackend;
use antecedent_kernels::norm_cdf;

#[test]
fn logistic_separates_simple_signal() {
    let n = 80usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = if i % 10 == 0 { 1.0 - t } else { t };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-6),
    )
    .unwrap();
    assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
    // The saturated 2x2 table: P(y=1 | t=0) = 4/40 and P(y=1 | t=1) = 36/40 (every tenth
    // row flipped), so the MLE is logit(0.1) and logit(0.9) - logit(0.1) = 2 ln 9.
    assert!((fit.coefficients[0] + 9.0_f64.ln()).abs() < 1e-5, "{:?}", fit.coefficients);
    assert!((fit.coefficients[1] - 2.0 * 9.0_f64.ln()).abs() < 1e-5, "{:?}", fit.coefficients);
    assert!(!fit.separated);
}

#[test]
fn logistic_flags_complete_separation() {
    let n = 60usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = t;
    }
    let mut ws = LeastSquaresWorkspace::default();
    let opts = GlmOptions { ridge_on_separation: None, ..GlmOptions::new(100, 1e-6) };
    let fit = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &opts,
    )
    .unwrap();
    assert!(fit.separated);
    assert!(fit.require_ok().is_err());
}

#[test]
fn probit_recovers_positive_slope() {
    let n = 200usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        // Soft signal: mostly follows t with a few flips.
        y[i] = if i % 8 == 0 { 1.0 - t } else { t };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::BinomialProbit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-6),
    )
    .unwrap();
    assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
    // Rows flipped every eighth: 13 of the 100 rows with t = 0 and 12 of the 100 with t = 1,
    // so the MLE is Phi^-1(0.13) and Phi^-1(0.88) - Phi^-1(0.13).
    assert!((fit.coefficients[0] + 1.126_391_129_04).abs() < 1e-5, "{:?}", fit.coefficients);
    assert!((fit.coefficients[1] - 2.301_377_921_09).abs() < 1e-5, "{:?}", fit.coefficients);
    assert!(!fit.separated);
}

#[test]
fn ridge_on_separation_keeps_separated_flag() {
    let n = 60usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = t;
    }
    let mut ws = LeastSquaresWorkspace::default();
    let opts = GlmOptions { ridge_on_separation: Some(1.0), ..GlmOptions::new(100, 1e-6) };
    let fit = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &opts,
    )
    .unwrap();
    assert!(fit.converged);
    assert!(fit.separated, "ridge fallback must not clear separation");
    assert!(fit.require_ok().is_err());
    assert!(fit.coefficients[1] > 0.0);
}

#[test]
fn estimation_options_skip_the_discarded_separation_ridge_refit() {
    // Perfectly separated data: with the caller's ridge option the returned fit is the
    // ridge refit (`penalized`); without it the unpenalized IRLS iterate comes back.
    // Both are flagged separated, so `require_ok` refuses either, and the second spares
    // the refit whose result an estimation path would throw away.
    let n = 40usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = t;
    }
    let with_ridge = GlmOptions::new(100, 1e-6);
    let without = with_ridge.without_separation_ridge();
    assert_eq!(without.ridge_on_separation, None);
    assert_eq!((without.max_iter, without.tol), (with_ridge.max_iter, with_ridge.tol));
    let mut ws = LeastSquaresWorkspace::default();
    let mut fit = |opts: &GlmOptions| {
        fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            opts,
        )
        .unwrap()
    };
    let ridge_fit = fit(&with_ridge);
    let plain_fit = fit(&without);
    assert!(ridge_fit.penalized && !plain_fit.penalized);
    assert!(ridge_fit.separated && plain_fit.separated);
    assert!(ridge_fit.require_ok().is_err() && plain_fit.require_ok().is_err());
}

#[test]
fn a_refused_fit_carries_the_facts_it_measured() {
    let n = 40usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = t;
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-6).without_separation_ridge(),
    )
    .unwrap();
    // Complete separation drives some fitted probability inside the 1e-8 band.
    let count = fit.boundary_count.unwrap();
    assert!((1..=n as u64).contains(&count), "count={count}");
    assert!(fit.boundary_margin.unwrap() < 1e-8);
    match fit.require_ok().unwrap_err() {
        StatsError::GlmRefused {
            kind,
            message,
            iterations,
            boundary_margin_bits,
            boundary_count,
        } => {
            // A saturated fit that is separated, converged or not, is reported as separation:
            // hitting the iteration limit is the symptom of an absent maximum-likelihood
            // estimate, not a separate cause.
            assert!(fit.separated);
            let (expected_kind, expected_message) = (
                GlmRefusalKind::Separated,
                "GLM indicates (quasi-)complete separation; refuse propensity/outcome scores",
            );
            assert_eq!(kind, expected_kind);
            assert_eq!(message, expected_message);
            assert_eq!(iterations, fit.iterations);
            assert_eq!(boundary_margin_bits, fit.boundary_margin.map(f64::to_bits));
            assert_eq!(boundary_count, Some(count));
        }
        other => panic!("expected a GLM refusal record, got {other:?}"),
    }
    // A least-squares fit measures no probability margin: absent, not zero.
    let gaussian = fit_glm(
        GlmFamily::GaussianIdentity,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::default(),
    )
    .unwrap();
    assert!(gaussian.boundary_margin.is_none() && gaussian.boundary_count.is_none());
}

#[test]
fn separation_verdict_needs_more_than_saturation() {
    let saturated_only = BinomialDiagnostics {
        deviance: 1.0,
        saturated: true,
        min_margin: 0.0,
        n_saturated: 1,
        perfectly_classified: false,
    };
    // A converged fit that saturates but misclassifies a row has a finite MLE.
    assert!(!saturated_only.separated(true));
    // The same saturation without convergence is divergence: separation.
    assert!(saturated_only.separated(false));
    let perfect = BinomialDiagnostics {
        deviance: 0.0,
        saturated: true,
        min_margin: 0.0,
        n_saturated: 1,
        perfectly_classified: true,
    };
    assert!(perfect.separated(true));
    let interior = BinomialDiagnostics {
        deviance: 0.0,
        saturated: false,
        min_margin: 0.5,
        n_saturated: 0,
        perfectly_classified: true,
    };
    assert!(!interior.separated(false));
}

#[test]
fn require_ok_names_the_actual_defect() {
    let base = GlmFit {
        coefficients: vec![0.0],
        iterations: 1,
        converged: true,
        separated: false,
        boundary_saturated: false,
        boundary_margin: Some(0.25),
        boundary_count: Some(0),
        penalized: false,
        deviance: 0.0,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(1, None, "glm-irls", 0),
    };
    assert!(base.require_ok().is_ok());
    let separated = GlmFit { separated: true, boundary_saturated: true, ..base.clone() };
    assert!(separated.require_ok().unwrap_err().to_string().contains("separation"));
    let extreme = GlmFit { boundary_saturated: true, ..base.clone() };
    let msg = extreme.require_ok().unwrap_err().to_string();
    assert!(msg.contains("within 1e-8") && !msg.contains("(quasi-)complete"), "{msg}");
}

#[test]
fn deliberate_ridge_fit_is_penalized_but_not_separated() {
    let n = 80usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = if i % 10 == 0 { 1.0 - t } else { t };
    }
    let fit = fit_glm_ridge(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut LeastSquaresWorkspace::default(),
        &GlmOptions::new(200, 1e-8),
        0.5,
    )
    .unwrap();
    assert!(fit.converged);
    assert!(fit.penalized && !fit.separated && !fit.boundary_saturated);
    assert!(fit.require_ok().is_ok(), "an ordinary ridge fit must pass require_ok");
}

#[test]
fn laplace_fit_penalizes_every_coefficient_and_returns_the_inverse_information() {
    let n = 40usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        // Complete separation: the outcome equals the second column.
        y[i] = t;
    }
    let design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y };
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_logit_laplace(design, None, 1.0, &mut ws, &GlmOptions::new(100, 1e-10)).unwrap();
    assert!(fit.fit.converged);
    assert!(fit.fit.coefficients.iter().all(|b| b.is_finite()));
    // The intercept is penalized too: it is pulled below the unpenalized -infinity, not left free.
    assert!(fit.fit.coefficients[1] > 0.0);
    // Maximiser of loglik - 0.5 |b|^2 by Newton in an independent script, and the inverse of
    // X'WX + I at it.
    let expected = [-1.291_836_85, 3.019_004_51];
    for (got, want) in fit.fit.coefficients.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "{:?}", fit.fit.coefficients);
    }
    let covariance = [0.196_038_69, -0.141_022_41, -0.141_022_41, 0.382_085_80];
    assert_eq!(fit.covariance.len(), 4);
    for (got, want) in fit.covariance.iter().zip(covariance) {
        assert!((got - want).abs() < 1e-6, "{:?}", fit.covariance);
    }
    // Unit case weights are the unweighted fit.
    let ones = vec![1.0; n];
    let weighted =
        fit_logit_laplace(design, Some(&ones), 1.0, &mut ws, &GlmOptions::new(100, 1e-10)).unwrap();
    assert_eq!(weighted.fit.coefficients, fit.fit.coefficients);
    assert!(fit_logit_laplace(design, None, 0.0, &mut ws, &GlmOptions::default()).is_err());
}
