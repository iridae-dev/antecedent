//! Tests for the multinomial logit fits.
#![allow(unused_imports)]

use super::*;
use crate::faer_backend::FaerBackend;
use antecedent_kernels::norm_cdf;

#[test]
fn multinomial_binary_matches_logistic_slope_sign() {
    let n = 80usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0u32; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = if i % 10 == 0 { 1 - (t as u32) } else { t as u32 };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_multinomial_logit(
        MultinomialDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y_category: &y,
            n_categories: 2,
        },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-6),
    )
    .unwrap();
    assert!(fit.converged);
    assert!(!fit.separated);
    assert!(fit.coefficients[..2].iter().all(|&c| c == 0.0));
    // Saturated two-group model: P(y = 1 | t = 0) = 4/40, P(y = 1 | t = 1) = 36/40, so the
    // maximum-likelihood intercept is logit(0.1) = -ln 9 and the slope is logit(0.9) - logit(0.1)
    // = 2 ln 9.
    let ln9 = 9.0_f64.ln();
    assert!((fit.coefficients[2] + ln9).abs() < 1e-5, "intercept={}", fit.coefficients[2]);
    assert!((fit.coefficients[3] - 2.0 * ln9).abs() < 1e-5, "slope={}", fit.coefficients[3]);
}

#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn multinomial_intercept_only_three_class() {
    let n = 90usize;
    let x = vec![1.0; n];
    let mut y = vec![0u32; n];
    for i in 0..n {
        y[i] = (i % 3) as u32;
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_multinomial_logit(
        MultinomialDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 1,
            y_category: &y,
            n_categories: 3,
        },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-8),
    )
    .unwrap();
    assert!(fit.converged);
    // Equal class sizes → free intercepts ≈ 0.
    assert!(fit.coefficients[0].abs() == 0.0, "{:?}", fit.coefficients);
    assert!(fit.coefficients[1].abs() < 1e-6, "{:?}", fit.coefficients);
    assert!(fit.coefficients[2].abs() < 1e-6, "{:?}", fit.coefficients);
}

#[test]
fn multinomial_three_class_recovers_parent_signal() {
    // Y ∈ {0,1,2}; when X=0 prefer class 0, when X=1 prefer class 2.
    // Enough class mixing to avoid complete separation.
    let n = 180usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0u32; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = match (t < 0.5, i % 5) {
            (true, 0) | (false, 1) => 1,
            (true, 1) => 2,
            (true, _) | (false, 0) => 0,
            (false, _) => 2,
        };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_multinomial_logit(
        MultinomialDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y_category: &y,
            n_categories: 3,
        },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-8),
    )
    .unwrap();
    assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
    assert!(fit.coefficients[..2].iter().all(|&c| c == 0.0));
    // The design is saturated (two groups of 90), so the maximum likelihood is the log count
    // ratio against class 0. X = 0: counts (54, 18, 18); X = 1: counts (18, 18, 54). Class 1:
    // intercept ln(18/54) = -ln 3, slope ln 3. Class 2: intercept -ln 3, slope 2 ln 3.
    let ln3 = 3.0_f64.ln();
    let expected = [-ln3, ln3, -ln3, 2.0 * ln3];
    for (got, want) in fit.coefficients[2..].iter().zip(expected) {
        assert!((got - want).abs() < 1e-5, "{:?} vs {expected:?}", fit.coefficients);
    }
}

#[test]
fn weighted_multinomial_matches_fractional_category_counts() {
    for k in [2, 3] {
        let x = vec![1.0; k];
        let y: Vec<u32> = (0..k).map(|i| i as u32).collect();
        let weights: Vec<f64> = (0..k).map(|i| i as f64 + 0.5).collect();
        let fit = fit_multinomial_logit_weighted(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: k,
                ncols: 1,
                y_category: &y,
                n_categories: k,
            },
            Some(&weights),
            &FaerBackend,
            &mut LeastSquaresWorkspace::default(),
            &GlmOptions::default(),
        )
        .unwrap();
        assert!(fit.converged);
        for j in 1..k {
            assert!((fit.coefficients[j] - (weights[j] / weights[0]).ln()).abs() < 1e-5);
        }
    }
}

#[test]
fn multinomial_collinear_design_errors_at_every_sample_size() {
    // Intercept + x + a duplicate of x: the coefficients on x and x' are not identified.
    // An unconditional 1e-8 ridge used to make this fit "converge" for small n and error
    // only once the Gram scale exceeded 1e4; the verdict is now a property of the design.
    for n in [60usize, 6000] {
        let mut x = vec![0.0; n * 3];
        let mut y = vec![0u32; n];
        for i in 0..n {
            let t = i as f64 / n as f64;
            x[i] = 1.0;
            x[n + i] = t;
            x[2 * n + i] = t;
            y[i] = ((i * 7 + i / 3) % 3) as u32;
        }
        let err = fit_multinomial_logit(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 3,
                y_category: &y,
                n_categories: 3,
            },
            &FaerBackend,
            &mut LeastSquaresWorkspace::default(),
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap_err();
        assert!(matches!(err, StatsError::RankDeficient { .. }), "n={n} err={err}");
    }
}
