//! Linear-algebra microbenchmarks for the faer 0.24 substrate.
#![allow(
    missing_docs,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, SandwichKind, cholesky_spd,
    coefficient_covariance, fit_ridge, fit_wls, form_xtx,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use faer::Mat;
use faer::linalg::solvers::{ColPivQr, Qr};

struct Design {
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

fn design(nrows: usize, ncols: usize) -> Design {
    let mut x = vec![0.0; nrows * ncols];
    let mut y = vec![0.0; nrows];
    let mut w = vec![1.0; nrows];
    for r in 0..nrows {
        x[r] = 1.0;
        y[r] = 0.25 + 0.01 * r as f64;
        w[r] = 0.5 + (r % 5) as f64 * 0.1;
        for c in 1..ncols {
            let freq = (c as f64 + 1.3) * 0.17;
            x[c * nrows + r] = ((r as f64 + 1.0) * freq).sin() + 0.01 * r as f64 * c as f64;
            y[r] += x[c * nrows + r] * (c as f64 * 0.2);
        }
    }
    Design { x, y, w, nrows, ncols }
}

fn mat_from_colmajor(x: &[f64], nrows: usize, ncols: usize) -> Mat<f64> {
    Mat::<f64>::from_fn(nrows, ncols, |r, c| x[c * nrows + r])
}

fn bench_linalg(c: &mut Criterion) {
    let ols = design(400, 12);
    let wide = design(80, 40);
    let tall = design(4000, 8);

    c.bench_function("gram_xtx_n400_p12", |b| {
        let mut xtx = vec![0.0; ols.ncols * ols.ncols];
        b.iter(|| {
            form_xtx(black_box(&ols.x), ols.nrows, ols.ncols, &mut xtx);
            black_box(&xtx);
        });
    });

    c.bench_function("crossprod_xty_n400_p12", |b| {
        let mut xty = vec![0.0; ols.ncols];
        b.iter(|| {
            for col in 0..ols.ncols {
                let mut s = 0.0;
                let col_data = &ols.x[col * ols.nrows..(col + 1) * ols.nrows];
                for r in 0..ols.nrows {
                    s += col_data[r] * ols.y[r];
                }
                xty[col] = s;
            }
            black_box(&xty);
        });
    });

    c.bench_function("weighted_xtwx_n400_p12", |b| {
        let mut xtwx = vec![0.0; ols.ncols * ols.ncols];
        b.iter(|| {
            xtwx.fill(0.0);
            for c1 in 0..ols.ncols {
                for c2 in c1..ols.ncols {
                    let mut acc = 0.0;
                    let col1 = &ols.x[c1 * ols.nrows..(c1 + 1) * ols.nrows];
                    let col2 = &ols.x[c2 * ols.nrows..(c2 + 1) * ols.nrows];
                    for r in 0..ols.nrows {
                        acc += col1[r] * ols.w[r] * col2[r];
                    }
                    xtwx[c1 * ols.ncols + c2] = acc;
                    xtwx[c2 * ols.ncols + c1] = acc;
                }
            }
            black_box(&xtwx);
        });
    });

    c.bench_function("qr_least_squares_n400_p12", |b| {
        let mut ws = LeastSquaresWorkspace::default();
        b.iter(|| {
            let fit = FaerBackend
                .least_squares(black_box(&ols.x), ols.nrows, ols.ncols, &ols.y, &mut ws)
                .unwrap();
            black_box(fit.rss);
        });
    });

    c.bench_function("pivoted_qr_n400_p12", |b| {
        b.iter(|| {
            let a = mat_from_colmajor(black_box(&ols.x), ols.nrows, ols.ncols);
            let qr = ColPivQr::new(a.as_ref());
            black_box(qr.thin_R().nrows());
        });
    });

    c.bench_function("unpivoted_qr_n400_p12", |b| {
        b.iter(|| {
            let a = mat_from_colmajor(black_box(&ols.x), ols.nrows, ols.ncols);
            let qr = Qr::new(a.as_ref());
            black_box(qr.thin_R().nrows());
        });
    });

    c.bench_function("cholesky_p12", |b| {
        let mut xtx = vec![0.0; ols.ncols * ols.ncols];
        form_xtx(&ols.x, ols.nrows, ols.ncols, &mut xtx);
        for i in 0..ols.ncols {
            xtx[i * ols.ncols + i] += 1e-8;
        }
        b.iter(|| {
            black_box(cholesky_spd(black_box(&xtx), ols.ncols).unwrap());
        });
    });

    c.bench_function("svd_n400_p12", |b| {
        b.iter(|| {
            let a = mat_from_colmajor(black_box(&ols.x), ols.nrows, ols.ncols);
            let svd = a.svd().expect("svd");
            black_box(svd.S()[0]);
        });
    });

    c.bench_function("ridge_n400_p12", |b| {
        let mut ws = LeastSquaresWorkspace::default();
        b.iter(|| {
            let fit = fit_ridge(
                black_box(&ols.x),
                ols.nrows,
                ols.ncols,
                &ols.y,
                0.1,
                &FaerBackend,
                &mut ws,
            )
            .unwrap();
            black_box(fit.rss);
        });
    });

    c.bench_function("wls_n400_p12", |b| {
        let mut ws = LeastSquaresWorkspace::default();
        b.iter(|| {
            let fit = fit_wls(
                black_box(&ols.x),
                ols.nrows,
                ols.ncols,
                &ols.y,
                &ols.w,
                &FaerBackend,
                &mut ws,
            )
            .unwrap();
            black_box(fit.rss);
        });
    });

    c.bench_function("sandwich_hc1_n400_p12", |b| {
        let mut ws = LeastSquaresWorkspace::default();
        let fit = FaerBackend.least_squares(&ols.x, ols.nrows, ols.ncols, &ols.y, &mut ws).unwrap();
        b.iter(|| {
            let cov = coefficient_covariance(
                black_box(&ols.x),
                ols.nrows,
                ols.ncols,
                &fit.residuals,
                SandwichKind::Hc1,
            )
            .unwrap();
            black_box(cov[0]);
        });
    });

    c.bench_function("matvec_n4000_p8", |b| {
        let beta: Vec<f64> = (0..tall.ncols).map(|i| i as f64 * 0.1).collect();
        let mut out = vec![0.0; tall.nrows];
        b.iter(|| {
            out.fill(0.0);
            for c in 0..tall.ncols {
                let col = &tall.x[c * tall.nrows..(c + 1) * tall.nrows];
                let bc = beta[c];
                for r in 0..tall.nrows {
                    out[r] += col[r] * bc;
                }
            }
            black_box(&out);
        });
    });

    c.bench_function("qr_least_squares_wide_n80_p40", |b| {
        let mut ws = LeastSquaresWorkspace::default();
        b.iter(|| {
            let fit = FaerBackend
                .least_squares(black_box(&wide.x), wide.nrows, wide.ncols, &wide.y, &mut ws)
                .unwrap();
            black_box(fit.rss);
        });
    });
}

criterion_group!(benches, bench_linalg);
criterion_main!(benches);
