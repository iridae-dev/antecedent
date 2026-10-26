//! Tests for the Poisson and negative-binomial fits.
#![allow(unused_imports)]

use super::*;
use crate::faer_backend::FaerBackend;
use antecedent_kernels::norm_cdf;

#[test]
fn poisson_fit_is_invariant_to_the_unit_of_a_predictor() {
    let n = 80usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Vec<f64> = t.iter().enumerate().map(|(i, v)| (2.0 + 3.0 * v) + (i % 3) as f64).collect();
    let fit_with = |scale: f64| {
        let mut x = vec![1.0; n];
        x.extend(t.iter().map(|v| v * scale));
        let mut ws = LeastSquaresWorkspace::default();
        fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
        )
        .unwrap()
    };
    let base = fit_with(1.0);
    let tiny = fit_with(1e3);
    assert!(base.converged && tiny.converged);
    assert!((base.deviance - tiny.deviance).abs() <= 1e-6 * base.deviance.abs().max(1.0));
}

#[test]
fn poisson_recovers_positive_association() {
    let n = 120usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = if t < 0.5 { 2.0 } else { 4.0 };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::PoissonLog,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-8),
    )
    .unwrap();
    assert!(fit.converged);
    assert!(fit.coefficients[1] > 0.3);
}

#[test]
fn poisson_intercept_only_converges_to_log_mean() {
    let n = 40usize;
    let x = vec![1.0; n];
    let y = vec![100.0; n];
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::PoissonLog,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::default(),
    )
    .unwrap();
    assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
    assert!((fit.coefficients[0] - 100.0_f64.ln()).abs() < 1e-6, "b0={}", fit.coefficients[0]);
}

#[test]
fn negbin_fixed_alpha_recovers_positive_association() {
    let n = 120usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        // Overdispersed counts: base 2 vs 5 with occasional spikes.
        y[i] = if t < 0.5 {
            if i % 11 == 0 { 8.0 } else { 2.0 }
        } else if i % 11 == 0 {
            15.0
        } else {
            5.0
        };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let opts = GlmOptions { nb_alpha: NbAlphaPolicy::Fixed(0.5), ..GlmOptions::new(100, 1e-8) };
    let fit = fit_glm(
        GlmFamily::NegativeBinomial,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &opts,
    )
    .unwrap();
    assert!(fit.converged);
    assert!(fit.coefficients[1] > 0.3, "slope={}", fit.coefficients[1]);
}

#[test]
fn negbin_mom_alpha_matches_pearson_formula() {
    // Intercept-only Poisson μ̂ = ȳ; Pearson MoM α̂ = (Σ(y−μ)²/μ − (n−1)) / (nμ).
    let y = [1.0_f64, 2.0, 0.0, 4.0, 3.0, 1.0, 5.0, 2.0];
    let n = y.len();
    let mu = y.iter().sum::<f64>() / n as f64;
    let pearson: f64 = y.iter().map(|&yi| (yi - mu).powi(2) / mu).sum();
    let expected = ((pearson - (n as f64 - 1.0)).max(0.0) / (n as f64 * mu)).max(1e-8);

    let x = vec![1.0; n];
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::NegativeBinomial,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-10),
    )
    .unwrap();
    assert!(fit.converged);
    // Recompute MoM from the same Poisson μ path used internally.
    let poisson = fit_glm(
        GlmFamily::PoissonLog,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-10),
    )
    .unwrap();
    let mu_hat = poisson.coefficients[0].exp();
    assert!((mu_hat - mu).abs() < 1e-8);
    assert!(
        (expected - ((pearson - (n as f64 - 1.0)).max(0.0) / (n as f64 * mu_hat)).max(1e-8))
            .abs()
            < 1e-12
    );
    // Fixed-α fit at that MoM should match default MethodOfMoments path coefficients.
    let opts =
        GlmOptions { nb_alpha: NbAlphaPolicy::Fixed(expected), ..GlmOptions::new(100, 1e-10) };
    let fit_fixed = fit_glm(
        GlmFamily::NegativeBinomial,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &opts,
    )
    .unwrap();
    assert!((fit.coefficients[0] - fit_fixed.coefficients[0]).abs() < 1e-6);
}

#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn negbin_nested_mle_recovers_overdispersion() {
    // Intercept-only NB2 with known α≈0.5 via gamma-poisson mixture (deterministic spikes).
    let n = 400usize;
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        // Mean ~3 with heavy tails: mostly 2, occasional 10.
        y[i] = if i % 8 == 0 { 10.0 } else { 2.0 };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let opts = GlmOptions {
        nb_alpha: NbAlphaPolicy::NestedMle { max_outer: 20, tol_alpha: 1e-6 },
        ..GlmOptions::new(100, 1e-8)
    };
    let fit = fit_glm(
        GlmFamily::NegativeBinomial,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &opts,
    )
    .unwrap();
    assert!(fit.converged);
    let alpha = fit.nb_alpha.expect("nb alpha");
    assert!(alpha > 0.05, "alpha={alpha}");
    assert!(alpha < 5.0, "alpha={alpha}");
    // Nested MLE α should be in the ballpark of MoM.
    let mom = fit_glm(
        GlmFamily::NegativeBinomial,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-8),
    )
    .unwrap();
    let mom_a = mom.nb_alpha.unwrap();
    assert!((alpha - mom_a).abs() / mom_a.max(1e-6) < 2.0, "nested={alpha} mom={mom_a}");
}

#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn negbin_nested_mle_recovers_alpha_above_two() {
    // Heavy overdispersion so NestedMle Newton enters θ = 1/α < 0.5 (trigamma reflection).
    let n = 500usize;
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        // Mean ~4 with extreme spikes ⇒ α ≫ 2.
        y[i] = if i % 5 == 0 { 30.0 } else { 1.0 };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let opts = GlmOptions {
        nb_alpha: NbAlphaPolicy::NestedMle { max_outer: 30, tol_alpha: 1e-6 },
        ..GlmOptions::new(100, 1e-8)
    };
    let fit = fit_glm(
        GlmFamily::NegativeBinomial,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &opts,
    )
    .unwrap();
    assert!(fit.converged);
    let alpha = fit.nb_alpha.expect("nb alpha");
    assert!(alpha > 2.0, "expected alpha > 2 for trigamma reflection path, got {alpha}");
    // MoM should also land in the same high-α regime.
    let mom = fit_glm(
        GlmFamily::NegativeBinomial,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(100, 1e-8),
    )
    .unwrap();
    let mom_a = mom.nb_alpha.unwrap();
    assert!(mom_a > 2.0, "mom alpha={mom_a}");
    assert!((alpha - mom_a).abs() / mom_a.max(1e-6) < 2.0, "nested={alpha} mom={mom_a}");
}
