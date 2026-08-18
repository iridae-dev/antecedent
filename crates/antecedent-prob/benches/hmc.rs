//! HMC GLM workspace-reuse benchmark.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, GaussianCoefficientPrior, HmcOptions,
    LaplaceWorkspace, PriorSet, PriorSpec, fit_hmc_glm,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn make_design(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for r in 0..n {
        x[r] = 1.0;
        y[r] = 2.0 + ((r % 4) as f64 - 1.5) * 0.05;
    }
    (x, y)
}

fn bench_hmc(c: &mut Criterion) {
    let n = 30usize;
    let (x, y) = make_design(n);
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
    // Tiny known-σ² intercept-only GLM: enough to run leapfrog + workspace
    // reuse. Publication (ESS ≥ 100 / R̂ ≤ 1.01) is a unit-test gate, not this
    // smoke — 2×40 draws cannot meet ESS ≥ 100.
    let opts = BayesFitOptions { n_draws: 40, seed: 5, max_iter: 50, grad_tol: 1e-8 };
    let hmc = HmcOptions {
        n_chains: 2,
        n_warmup: 20,
        leapfrog_steps: 4,
        step_size: 0.08,
        target_accept: 0.8,
        mass: 1.0,
    };

    let mut ws = LaplaceWorkspace::default();
    ws.prepare(n, 1, opts.n_draws.saturating_mul(hmc.n_chains));
    let grow_before = ws.grow_count;

    c.bench_function("hmc_gaussian_n30_p1", |b| {
        b.iter(|| {
            let fit = fit_hmc_glm(
                BayesLikelihood::GaussianIdentity,
                black_box(design),
                black_box(&prior),
                black_box(&opts),
                black_box(hmc),
                &mut ws,
            );
            match fit {
                Ok(ok) => black_box(ok.map[0]),
                Err(e) => black_box(e.to_string().len() as f64),
            }
        });
    });

    assert_eq!(ws.grow_count, grow_before, "HMC workspace must not grow across repeated fits");
}

criterion_group!(benches, bench_hmc);
criterion_main!(benches);
