//! Compare the promoted borrowed dense Gram path with an independent scalar reference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, reason = "criterion generates benchmark entry points")]

use antecedent_stats::form_xtx;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn scalar_gram(x: &[f64], n: usize, p: usize, out: &mut [f64]) {
    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            for r in 0..n {
                sum += x[i * n + r] * x[j * n + r];
            }
            out[i * p + j] = sum;
            out[j * p + i] = sum;
        }
    }
}

fn bench_blas_layout(c: &mut Criterion) {
    for (n, p) in [(64, 8), (4096, 32), (4096, 128)] {
        let x: Vec<_> = (0..p)
            .flat_map(|col| {
                (0..n)
                    .map(move |row| ((f64::from(row) + 1.0) * (f64::from(col) + 1.3) * 0.17).sin())
            })
            .collect();
        let (n, p) = (usize::try_from(n).unwrap(), usize::try_from(p).unwrap());
        let mut expected = vec![0.0; p * p];
        let mut actual = vec![0.0; p * p];
        scalar_gram(&x, n, p, &mut expected);
        form_xtx(&x, n, p, &mut actual);
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()));
        }
        c.bench_function(&format!("gram/scalar/n_{n}/p_{p}"), |b| {
            b.iter(|| scalar_gram(black_box(&x), n, p, black_box(&mut expected)));
        });
        c.bench_function(&format!("gram/production/n_{n}/p_{p}"), |b| {
            b.iter(|| form_xtx(black_box(&x), n, p, black_box(&mut actual)));
        });
    }
}

criterion_group!(benches, bench_blas_layout);
criterion_main!(benches);
