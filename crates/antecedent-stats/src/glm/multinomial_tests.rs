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
    assert!(fit.coefficients[3] > 0.5, "slope={}", fit.coefficients[3]);
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
    assert!(fit.coefficients[1].abs() < 0.2, "{:?}", fit.coefficients);
    assert!(fit.coefficients[2].abs() < 0.2, "{:?}", fit.coefficients);
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
    // Class 2 vs 0: positive slope on X.
    let slope_class2 = fit.coefficients[2 * 2 + 1];
    assert!(slope_class2 > 0.5, "class2 slope={slope_class2}");
    // Softmax at X=1 should put most mass on class 2.
    let eta1 = fit.coefficients[2] + fit.coefficients[3];
    let eta2 = fit.coefficients[4] + fit.coefficients[5];
    let m = eta1.max(eta2).max(0.0);
    let p0 = (-m).exp();
    let p1 = (eta1 - m).exp();
    let p2 = (eta2 - m).exp();
    let z = p0 + p1 + p2;
    assert!(p2 / z > p0 / z && p2 / z > p1 / z, "p={:?}", [p0 / z, p1 / z, p2 / z]);
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
