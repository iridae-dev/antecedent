//! Tests for the Gaussian fit.
#![allow(unused_imports)]

use super::*;
use crate::faer_backend::FaerBackend;
use antecedent_kernels::norm_cdf;

#[test]
fn gaussian_recovers_linear_slope() {
    let n = 100usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = (i % 2) as f64;
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = 1.0 + 2.0 * t;
    }
    let mut ws = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::GaussianIdentity,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::default(),
    )
    .unwrap();
    assert!((fit.coefficients[1] - 2.0).abs() < 1e-8);
}
