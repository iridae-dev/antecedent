//! Native measurement of the Laplace GLM control: allocator bytes and calls, live-byte peak,
//! peak resident set and the engine's own counts across repeated complete fits.
//!
//! It runs the same fit as `laplace_glm` (same design, prior and options) under
//! `antecedent_core::measurement_record::measure`. Timing stays with the Criterion bench;
//! this pass writes `$ANTECEDENT_MEASUREMENT_OUT/laplace_glm/measurement.json`.
//! `ANTECEDENT_LANE_ROWS` sets the row count (default 500).
#![allow(
    missing_docs,
    reason = "benchmark crate root: the allocator static is the only public item"
)]

use antecedent_core::ExecutionContext;
use antecedent_kernels::native_probe::{CountingAllocator, measured_pass};
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, GaussianCoefficientPrior, LaplaceWorkspace,
    PriorSet, PriorSpec, fit_laplace_glm,
};

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator::new();

fn main() {
    let n: usize =
        std::env::var("ANTECEDENT_LANE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    let mut x = vec![0.0; n * 3];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let z = (r as f64) * 0.01;
        x[r] = 1.0;
        x[n + r] = z;
        x[2 * n + r] = z * z;
        y[r] = 0.2 + 0.5 * z - 0.1 * z * z;
    }
    let prior = PriorSet {
        specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(3, 10.0))],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 3, y: &y, weights: None, offsets: None };
    let opts = BayesFitOptions { n_draws: 256, seed: 1, max_iter: 40, grad_tol: 1e-8 };
    let ctx = ExecutionContext::for_tests(1);
    let mut ws = LaplaceWorkspace::default();
    // Warm once so the workspace is sized, as the Criterion bench does.
    fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws).unwrap();
    measured_pass("laplace_glm", 5, &ctx, || {
        fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws)
            .map(|fit| fit.map[0])
    });
}
