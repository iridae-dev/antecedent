//! Laplace GLM workspace-reuse benchmark (Phase 6 exit criterion).
#![allow(missing_docs, clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use causal_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, GaussianCoefficientPrior, LaplaceWorkspace,
    PriorSet, PriorSpec, fit_laplace_glm,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn make_design(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n * 3];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let z = (r as f64) * 0.01;
        x[r] = 1.0;
        x[n + r] = z;
        x[2 * n + r] = z * z;
        y[r] = 0.2 + 0.5 * z - 0.1 * z * z;
    }
    (x, y)
}

fn bench_laplace(c: &mut Criterion) {
    let n = 500usize;
    let (x, y) = make_design(n);
    let prior = PriorSet {
        specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
            3, 10.0,
        ))],
        contrast: None,
        categorical: Vec::new(),
    };
    let design = BayesDesignRef {
        x_colmajor: &x,
        nrows: n,
        ncols: 3,
        y: &y,
        weights: None,
        offsets: None,
    };
    let opts = BayesFitOptions {
        n_draws: 256,
        seed: 1,
        max_iter: 40,
        grad_tol: 1e-8,
    };

    let mut ws = LaplaceWorkspace::default();
    // Warm-up prepare so timed loop measures reuse, not first allocation.
    ws.prepare(n, 3, opts.n_draws);
    let grow_before = ws.grow_count;

    c.bench_function("laplace_gaussian_n500_p3", |b| {
        b.iter(|| {
            let fit = fit_laplace_glm(
                BayesLikelihood::GaussianIdentity,
                black_box(design),
                black_box(&prior),
                black_box(&opts),
                &mut ws,
            )
            .unwrap();
            black_box(fit.map[0]);
        });
    });

    assert_eq!(
        ws.grow_count, grow_before,
        "Laplace workspace must not grow across repeated fits"
    );
}

criterion_group!(benches, bench_laplace);
criterion_main!(benches);
