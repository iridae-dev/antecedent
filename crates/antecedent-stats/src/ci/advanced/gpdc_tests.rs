//! Tests for the GPDC tests (exact and Nystrom).
#![allow(unused_imports)]

use super::*;
use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConfidenceMethod, SignificanceMethod,
};
use crate::test_util::centered_uniform as lcg_noise;
use antecedent_core::derive_stream;

#[test]
fn gpdc_runs() {
    let n = 40usize;
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let x: Vec<f64> = z.iter().map(|v| v + 0.01).collect();
    let y: Vec<f64> = z.iter().map(|v| 2.0 * v).collect();
    let cols: [&[f64]; 3] = [&x, &y, &z];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let z_flat = [2usize];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &z_flat,
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::default(),
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let out = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    assert!(out.results[0].statistic.is_finite());
    assert!((0.0..=1.0).contains(&out.results[0].p_value));
}

#[test]
fn gpdc_permutation_pvalue_separates_dependence() {
    let n = 60usize;
    let x = lcg_noise(n, 3);
    let y_dep = x.clone();
    let y_ind = lcg_noise(n, 4);
    let cols: [&[f64]; 3] = [&x, &y_dep, &y_ind];
    let queries = [
        CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 },
        CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 },
    ];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &[],
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::default(),
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(11);
    let out = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    // The dependent pair (y = x) has dCor 1, which no permutation reaches: the add-one p is
    // at its floor 1/50. The independent pair is decided on the statistic, not on a
    // Monte Carlo p-value that is uniform under H0 and would trip a fixed cut-off by chance.
    assert!(
        (out.results[0].statistic - 1.0).abs() < 1e-9,
        "dCor(x, x) is 1, got {}",
        out.results[0].statistic
    );
    assert!(
        (out.results[0].p_value - 0.02).abs() < 1e-12,
        "dependent p={} (add-one floor over 49 permutations)",
        out.results[0].p_value
    );
    assert!(
        out.results[0].statistic > out.results[1].statistic + 0.3,
        "dCor dependent={} independent={}",
        out.results[0].statistic,
        out.results[1].statistic
    );
}

/// stats-ci-4: standardised Z lets GPDC condition when Z is scaled by ~100.
#[test]
fn gpdc_scaled_z_null_and_power() {
    let n = 80usize;
    let trials = 80u32;
    let alpha = 0.05;
    let scale = 100.0;
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(0x0069_DC5C_u64);
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let z_flat = [2usize];
    let ci = Gpdc::new();
    let mut null_rej = 0u32;
    let mut alt_rej = 0u32;
    for t in 0..trials {
        let noise_z = lcg_noise(n, 100 + u64::from(t));
        let noise_x = lcg_noise(n, 200 + u64::from(t));
        let noise_y = lcg_noise(n, 300 + u64::from(t));
        let z: Vec<f64> = noise_z.iter().map(|e| scale * e).collect();
        // Null: X ⊥ Y | Z with both driven by Z.
        let x_null: Vec<f64> = z.iter().zip(&noise_x).map(|(zi, e)| 0.01 * zi + e).collect();
        let y_null: Vec<f64> = z.iter().zip(&noise_y).map(|(zi, e)| 0.02 * zi + e).collect();
        let cols_null: [&[f64]; 3] = [&x_null, &y_null, &z];
        let req = CiBatchRequest {
            columns: &cols_null,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let out = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        if out.results[0].p_value < alpha {
            null_rej += 1;
        }
        // Alt: residual X–Y dependence after conditioning on Z.
        let x_alt = x_null.clone();
        let y_alt: Vec<f64> = x_alt.iter().zip(&y_null).map(|(xi, yi)| xi + yi).collect();
        let cols_alt: [&[f64]; 3] = [&x_alt, &y_alt, &z];
        let req_alt = CiBatchRequest {
            columns: &cols_alt,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let out_alt = ci.test_batch_adhoc(&req_alt, &mut ws, &ctx).unwrap();
        if out_alt.results[0].p_value < alpha {
            alt_rej += 1;
        }
    }
    let type_i = f64::from(null_rej) / f64::from(trials);
    let power = f64::from(alt_rej) / f64::from(trials);
    // Observed: 4 of 80 null rejections (nominal 0.05) and all 80 alternatives rejected.
    assert!(type_i <= 0.15, "scaled-Z GPDC null rejects near 1 without standardisation: {type_i}");
    assert!(power >= 0.9, "scaled-Z GPDC lost power after standardisation: {power}");
}

#[test]
fn gpdc_refuses_constant_series_and_budgets_the_conditional_kernel() {
    let n = 30usize;
    let x = lcg_noise(n, 9);
    let y = vec![3.0; n];
    let cols: [&[f64]; 2] = [&x, &y];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &[],
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::default(),
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(3);
    let err = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
    assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");

    // An unconditional test never builds an n x n matrix (distance correlation is
    // O(n log n) in time and O(n) in memory), so it runs at any size; a conditional one
    // factorizes a dense kernel and is quoted against the caller's budget.
    let big = 4_001usize;
    let bx = lcg_noise(big, 5);
    let by = lcg_noise(big, 6);
    let bz = lcg_noise(big, 7);
    let big_cols: [&[f64]; 3] = [&bx, &by, &bz];
    let unconditional = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    let big_req = CiBatchRequest {
        columns: &big_cols,
        queries: &unconditional,
        z_flat: &[],
        significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
        confidence: ConfidenceMethod::default(),
    };
    let done = Gpdc::new().test_batch_adhoc(&big_req, &mut ws, &ctx).unwrap();
    let dcor = crate::dcor::distance_correlation(
        &big_cols[0].iter().map(|v| v - bx.iter().sum::<f64>() / big as f64).collect::<Vec<_>>(),
        &big_cols[1].iter().map(|v| v - by.iter().sum::<f64>() / big as f64).collect::<Vec<_>>(),
    );
    assert!((done.results[0].statistic - dcor).abs() < 1e-12);
    let conditional = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let big_req = CiBatchRequest { queries: &conditional, z_flat: &[2], ..big_req };
    // The dense kernel and its factor (2 n^2 * 8 bytes) plus the standardized Z block do
    // not fit 64 MiB: the quote is refused with its numbers before any kernel exists.
    let mut limited = ExecutionContext::for_tests(3);
    limited.set_memory_limit(Some(64 << 20));
    let rows = big as u64;
    let needed = 2 * rows * rows * 8 + rows * 8;
    match Gpdc::new().test_batch_adhoc(&big_req, &mut ws, &limited) {
        Err(StatsError::ResourceLimit { required_bytes, limit_bytes, message }) => {
            assert_eq!((required_bytes, limit_bytes), (needed, 64 << 20));
            assert!(message.contains("gpdc_nystrom"), "{message}");
        }
        other => panic!("expected the kernel quote to be refused, got {other:?}"),
    }
    assert_eq!(limited.admission().available().bytes(), 64 << 20);
    let quote = Gpdc::new().quote(&big_req, &ExecutionContext::for_tests(3)).unwrap().unwrap();
    assert_eq!((quote.rows, quote.factorizations, quote.cache_slots), (rows, 1, 2));
    assert_eq!(quote.work, rows * rows * rows + rows * rows);
}

/// The residual is `y_c - K a` with `(K + lambda I) a = y_c` solved by an independent dense
/// Gauss-Jordan; the implementation returns `lambda a` from the Cholesky solve.
#[test]
fn gpdc_residual_matches_dense_gp_regression() {
    let n = 12usize;
    let z: Vec<f64> = (0..n).map(|i| 0.4 * i as f64 + (i as f64 * 0.7).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (z[i] * 0.5).sin() + 0.1 * ((i * 5 % 7) as f64)).collect();
    let gp = Gpdc::new();
    let cols: [&[f64]; 2] = [&y, &z];
    let chol = gp_factor(&cols, &[1], n, &gp).unwrap();
    let got = gp_residual(&y, Some(&chol), gp.ridge).unwrap();

    let mz = z.iter().sum::<f64>() / n as f64;
    let sd = (z.iter().map(|v| (v - mz) * (v - mz)).sum::<f64>() / n as f64).sqrt();
    let zs: Vec<f64> = z.iter().map(|v| (v - mz) / sd).collect();
    let my = y.iter().sum::<f64>() / n as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - my).collect();
    let kmat = |i: usize, j: usize| (-0.5 * (zs[i] - zs[j]) * (zs[i] - zs[j])).exp();
    // Augmented [K + lambda I | y_c], Gauss-Jordan with partial pivoting.
    let mut a = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = kmat(i, j) + if i == j { gp.ridge } else { 0.0 };
        }
        a[i][n] = yc[i];
    }
    for c in 0..n {
        let piv = (c..n).max_by(|&p, &q| a[p][c].abs().total_cmp(&a[q][c].abs())).unwrap();
        a.swap(c, piv);
        let d = a[c][c];
        for j in 0..=n {
            a[c][j] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = a[r][c];
                for j in 0..=n {
                    let v = a[c][j];
                    a[r][j] -= f * v;
                }
            }
        }
    }
    let alpha: Vec<f64> = (0..n).map(|i| a[i][n]).collect();
    for i in 0..n {
        let pred: f64 = (0..n).map(|j| kmat(i, j) * alpha[j]).sum();
        let want = yc[i] - pred;
        assert!((got[i] - want).abs() < 1e-8, "{i}: {} vs {want}", got[i]);
    }
}

/// Columns `[x, y, z1, z2]` with a smooth nonlinear confounder `(z1, z2)` and independent
/// unit noises `e1`, `e2`: `x = 1.5 sin(1.5 z1) + 0.8 z2 + e1`,
/// `y = 1.5 tanh(1.5 z1) - 0.8 z2 + e2 + y_on_x * x`. With `y_on_x = 0`, `X ⊥ Y | Z`.
fn nystrom_sample(n: usize, seed: u64, y_on_x: f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x6E59);
    let z1: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
    let z2: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
    let e1: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
    let e2: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
    let x: Vec<f64> = (0..n).map(|i| 1.5 * (1.5 * z1[i]).sin() + 0.8 * z2[i] + e1[i]).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 1.5 * (1.5 * z1[i]).tanh() - 0.8 * z2[i] + e2[i] + y_on_x * x[i]).collect();
    (vec![x, y, z1, z2], e1, e2)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

/// Independent dense computation of the rank-`m` predictor on the same inducing rows: an
/// explicit `n x m` kernel matrix, naive normal equations and a dense solve, with none of
/// the row-block accumulation of the implementation.
fn dense_nystrom_residual(
    columns: &[&[f64]],
    z: &[usize],
    y: &[f64],
    gp: &GpdcNystrom,
) -> Vec<f64> {
    let n = y.len();
    let m = gp.inducing_points.min(n);
    let z_std = standardized_z(columns, z, n).unwrap();
    let zdim = z.len();
    let rows = nystrom_inducing_rows(n, m, gp.inducing_seed, z);
    let ls2 = gp.length_scale * gp.length_scale;
    let kernel = |a: usize, b: usize| {
        let d2: f64 = (0..zdim)
            .map(|d| {
                let diff = z_std[a * zdim + d] - z_std[b * zdim + d];
                diff * diff
            })
            .sum();
        (-0.5 * d2 / ls2).exp()
    };
    let knm: Vec<Vec<f64>> = (0..n).map(|i| rows.iter().map(|&r| kernel(i, r)).collect()).collect();
    let mut normal = vec![0.0; m * m];
    for j in 0..m {
        for l in 0..m {
            let mut acc: f64 = knm.iter().map(|row| row[j] * row[l]).sum();
            acc += gp.ridge * kernel(rows[j], rows[l]);
            normal[j * m + l] = acc;
        }
    }
    let mean_diagonal = (0..m).map(|j| normal[j * m + j]).sum::<f64>() / m as f64;
    for j in 0..m {
        normal[j * m + j] += NYSTROM_JITTER * mean_diagonal;
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let rhs: Vec<f64> =
        (0..m).map(|j| knm.iter().zip(&centered).map(|(row, yc)| row[j] * yc).sum()).collect();
    let chol = crate::gram::cholesky_spd(&normal, m).unwrap();
    let c = crate::gram::chol_solve(&chol, m, &rhs).unwrap();
    (0..n).map(|i| centered[i] - knm[i].iter().zip(&c).map(|(k, ci)| k * ci).sum::<f64>()).collect()
}

#[test]
fn gpdc_nystrom_residuals_match_a_dense_reference_beyond_the_exact_limit() {
    let n = 6_000usize;
    let (data, _, _) = nystrom_sample(n, 21, 0.0);
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let gp = GpdcNystrom { inducing_points: 24, ..GpdcNystrom::new() };
    let ctx = ExecutionContext::for_tests(5);
    let fit = nystrom_fit(&cols, &[2, 3], n, &gp, &ctx).unwrap();
    let [rx, ry] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
    for (got, target) in [(&rx, cols[0]), (&ry, cols[1])] {
        let want = dense_nystrom_residual(&cols, &[2, 3], target, &gp);
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-5, "worst residual gap {worst}");
    }
}

#[test]
fn gpdc_nystrom_residual_recovers_the_noise_of_a_nonlinear_confounder() {
    let n = 6_000usize;
    let (data, e1, e2) = nystrom_sample(n, 22, 0.0);
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let ctx = ExecutionContext::for_tests(6);
    let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), &ctx).unwrap();
    let [rx, ry] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
    // The truth is known: what is left after removing the smooth Z effect is the noise.
    assert!(pearson(&rx, &e1) > 0.99, "x residual vs noise {}", pearson(&rx, &e1));
    assert!(pearson(&ry, &e2) > 0.99, "y residual vs noise {}", pearson(&ry, &e2));
    // Removing Z must remove the dependence it induced between X and Y.
    let raw = crate::dcor::distance_correlation(cols[0], cols[1]);
    let cleaned = crate::dcor::distance_correlation(&rx, &ry);
    assert!(raw > 0.08 && cleaned < 0.05, "dCor before {raw}, after {cleaned}");
}

#[test]
fn gpdc_nystrom_agrees_with_the_exact_gp_where_both_run() {
    let n = 1_200usize;
    let (data, _, _) = nystrom_sample(n, 23, 0.0);
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let ctx = ExecutionContext::for_tests(7);
    let exact_factor = gp_factor(&cols, &[2, 3], n, &Gpdc::new()).unwrap();
    let exact = gp_residual(cols[0], Some(&exact_factor), Gpdc::new().ridge).unwrap();
    let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), &ctx).unwrap();
    let [low_rank, _] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
    assert!(pearson(&exact, &low_rank) > 0.99, "{}", pearson(&exact, &low_rank));
}

#[test]
fn gpdc_nystrom_is_identical_at_any_thread_count() {
    let n = 5_000usize;
    let (data, _, _) = nystrom_sample(n, 24, 0.0);
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let serial = ExecutionContext::for_tests(8);
    let mut threaded = ExecutionContext::for_tests(8);
    threaded.parallelism =
        antecedent_core::Parallelism::bounded(antecedent_core::NonZeroThreadCount::new(4).unwrap());
    let run = |ctx: &ExecutionContext| {
        let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), ctx).unwrap();
        nystrom_residuals(Some(&fit), [cols[0], cols[1]], ctx).unwrap()
    };
    let (a, b) = (run(&serial), run(&threaded));
    for (u, v) in a.iter().zip(&b) {
        assert!(u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

#[test]
fn gpdc_nystrom_separates_dependence_given_the_confounder_beyond_the_exact_limit() {
    let n = 6_000usize;
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
    let z_flat = [2usize, 3];
    let ctx = ExecutionContext::for_tests(9);
    let run = |y_on_x: f64| {
        let (data, _, _) = nystrom_sample(n, 25, y_on_x);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
    };
    let dependent = run(0.4);
    assert!(dependent.p_value <= 0.02 + 1e-12, "dependent p={}", dependent.p_value);
    assert!(dependent.statistic > 0.15, "dCor {}", dependent.statistic);
    let independent = run(0.0);
    assert!(independent.statistic < 0.05, "dCor {}", independent.statistic);
    assert!(independent.statistic + 0.1 < dependent.statistic);
}

#[test]
fn gpdc_nystrom_refuses_invalid_rank_and_budget_instead_of_degrading() {
    let n = 300usize;
    let (data, _, _) = nystrom_sample(n, 26, 0.0);
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &[2],
        significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
        confidence: ConfidenceMethod::None,
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(10);
    for inducing_points in [0, GPDC_NYSTROM_MAX_INDUCING_POINTS + 1] {
        let gp = GpdcNystrom { inducing_points, ..GpdcNystrom::new() };
        assert!(matches!(gp.test_batch_adhoc(&req, &mut ws, &ctx), Err(StatsError::Shape { .. })));
    }
    let mut limited = ExecutionContext::for_tests(10);
    limited.set_memory_limit(Some(1_024));
    assert!(matches!(
        GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &limited),
        Err(StatsError::ResourceLimit { .. })
    ));
    let cancelled = ExecutionContext::for_tests(10);
    cancelled.cancellation.cancel();
    assert!(matches!(
        GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &cancelled),
        Err(StatsError::Cancelled)
    ));
    // More inducing points than rows: every row is one.
    let all = GpdcNystrom { inducing_points: 1_024, ..GpdcNystrom::new() };
    assert!(all.test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
}

#[test]
#[ignore = "acceptance workload: 100k rows; run with --ignored in a release build"]
fn gpdc_nystrom_one_hundred_thousand_row_acceptance() {
    let n = 100_000usize;
    let (data, e1, _) = nystrom_sample(n, 27, 0.4);
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &[2, 3],
        significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
        confidence: ConfidenceMethod::None,
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::production(11, 4);
    let started = std::time::Instant::now();
    let out = GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    eprintln!("100k-row GPDC Nystrom took {:?}", started.elapsed());
    assert!(out.results[0].p_value <= 0.02 + 1e-12, "p={}", out.results[0].p_value);
    let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), &ctx).unwrap();
    let [rx, _] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
    assert!(pearson(&rx, &e1) > 0.99);
}

/// At 2000 rows the exact GPDC test keeps its construction: the statistic and p-value are,
/// bit for bit, those of the dense kernel factorization, GP residuals and residual
/// permutation null it has always run, and the low-rank test is a different construction.
#[test]
fn exact_conditional_gpdc_retained_at_small_sizes() {
    let n = 2_000usize;
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
    let z_flat = [2usize, 3];
    let significance = SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 };
    for (y_on_x, label) in [(0.0, "independent"), (0.3, "dependent")] {
        let (data, _, _) = nystrom_sample(n, 6_075_547, y_on_x);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance,
            confidence: ConfidenceMethod::None,
        };
        let ctx = ExecutionContext::for_tests(12);
        let mut ws = CiWorkspace::default();
        let got = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0];

        let gp = Gpdc::new();
        let factor = gp_factor(&cols, &z_flat, n, &gp).unwrap();
        let rx = gp_residual(cols[0], Some(&factor), gp.ridge).unwrap();
        let ry = gp_residual(cols[1], Some(&factor), gp.ridge).unwrap();
        let want = residual_dcor_permutation_result(
            &rx,
            &ry,
            ResidualNull {
                salt: 0x69DC_u64 ^ query_stream_salt(&[0], &[1], &z_flat),
                n_perm: nonparametric_permutation_count(significance),
                block_size: 1,
            },
            &ctx,
        );
        assert_eq!(got.statistic.to_bits(), want.statistic.to_bits(), "{label} statistic");
        assert_eq!(got.p_value.to_bits(), want.p_value.to_bits(), "{label} p-value");

        let low_rank = GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0];
        assert_ne!(low_rank.statistic.to_bits(), got.statistic.to_bits(), "{label}");
    }
    // Under a budget that cannot hold the kernel the exact test is refused, and the
    // refusal names the low-rank construction without switching to it.
    let (data, _, _) = nystrom_sample(4_001, 6_075_547, 0.0);
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &z_flat,
        significance,
        confidence: ConfidenceMethod::None,
    };
    let mut tight = ExecutionContext::for_tests(12);
    tight.set_memory_limit(Some(64 << 20));
    let refused =
        Gpdc::new().test_batch_adhoc(&req, &mut CiWorkspace::default(), &tight).unwrap_err();
    assert!(matches!(refused, StatsError::ResourceLimit { .. }), "{refused}");
    assert!(refused.to_string().contains("gpdc_nystrom"), "{refused}");
}

/// Five conditional queries over three conditioning sets, in an order that forces a rebuild
/// when only one factor is kept: A, B, A, C, A.
fn cache_sequence_request<'a>(
    cols: &'a [&'a [f64]],
    queries: &'a [CiQuery],
    z_flat: &'a [usize],
) -> CiBatchRequest<'a> {
    CiBatchRequest {
        columns: cols,
        queries,
        z_flat,
        significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
        confidence: ConfidenceMethod::None,
    }
}

#[test]
fn exact_gpdc_quote_matches_observed_factorizations_and_follows_the_callers_memory() {
    use crate::test_util::centered_uniform as lcg_noise;
    let n = 40usize;
    let data: Vec<Vec<f64>> = (0..5).map(|seed| lcg_noise(n, 70 + seed)).collect();
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let z_flat = [2usize, 3, 4];
    let queries: Vec<CiQuery> = [2usize, 3, 2, 4, 2]
        .iter()
        .map(|&z| CiQuery { x: 0, y: 1, z_start: z - 2, z_len: 1 })
        .collect();
    let request = cache_sequence_request(&cols, &queries, &z_flat);

    let per_factor = GpdcKernel::new(n as u64, 1).estimate().unwrap();
    let factor_bytes = (n * n * 8) as u64;
    let run = |ctx: &ExecutionContext| {
        let before = GPDC_FACTORIZATIONS.with(std::cell::Cell::get);
        let out = Gpdc::new().test_batch_adhoc(&request, &mut CiWorkspace::default(), ctx);
        (out, GPDC_FACTORIZATIONS.with(std::cell::Cell::get) - before)
    };

    // Room for two factors beside the working kernel: A, B, A hit, C evicts A, A rebuilt.
    let mut wide = ExecutionContext::for_tests(9);
    wide.set_memory_limit(Some(per_factor.memory_bytes() + factor_bytes));
    let wide_quote = Gpdc::new().quote(&request, &wide).unwrap().unwrap();
    assert_eq!((wide_quote.cache_slots, wide_quote.factorizations), (2, 4));
    assert_eq!(wide_quote.peak_bytes, per_factor.memory_bytes() + factor_bytes);
    assert_eq!(wide_quote.work, 4 * per_factor.work());
    let (wide_out, wide_observed) = run(&wide);
    let wide_out = wide_out.unwrap();
    assert_eq!(wide_observed, wide_quote.factorizations);
    assert_eq!(wide.admission().available().bytes(), per_factor.memory_bytes() + factor_bytes);

    // One factor's worth: the cache narrows to one slot, every change of set rebuilds.
    let mut narrow = ExecutionContext::for_tests(9);
    narrow.set_memory_limit(Some(per_factor.memory_bytes()));
    let narrow_quote = Gpdc::new().quote(&request, &narrow).unwrap().unwrap();
    assert_eq!((narrow_quote.cache_slots, narrow_quote.factorizations), (1, 5));
    assert_eq!(narrow_quote.peak_bytes, per_factor.memory_bytes());
    let (narrow_out, narrow_observed) = run(&narrow);
    let narrow_out = narrow_out.unwrap();
    assert_eq!(narrow_observed, 5);
    assert_eq!(narrow.admission().available().bytes(), per_factor.memory_bytes());
    // The cache width changes the work, never the answer.
    for (a, b) in wide_out.results.iter().zip(&narrow_out.results) {
        assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
        assert_eq!(a.p_value.to_bits(), b.p_value.to_bits());
    }

    // Below one factorization the batch is refused with nothing built or leased.
    let mut starved = ExecutionContext::for_tests(9);
    starved.set_memory_limit(Some(per_factor.memory_bytes() - 1));
    let (starved_out, starved_observed) = run(&starved);
    assert!(matches!(
        starved_out,
        Err(StatsError::ResourceLimit { required_bytes, .. })
            if required_bytes == per_factor.memory_bytes()
    ));
    assert_eq!(starved_observed, 0);
    assert_eq!(starved.admission().available().bytes(), per_factor.memory_bytes() - 1);
}

#[test]
fn exact_gpdc_work_limit_refuses_before_any_factorization() {
    use crate::test_util::centered_uniform as lcg_noise;
    let n = 30usize;
    let data: Vec<Vec<f64>> = (0..3).map(|seed| lcg_noise(n, 80 + seed)).collect();
    let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let request = cache_sequence_request(&cols, &queries, &[2]);
    let ctx = ExecutionContext::for_tests(4);
    let work = GpdcKernel::new(n as u64, 1).estimate().unwrap().work();
    assert_eq!(Gpdc::new().quote(&request, &ctx).unwrap().unwrap().work, work);

    let before = GPDC_FACTORIZATIONS.with(std::cell::Cell::get);
    let refused = Gpdc::new()
        .with_work_limit(work - 1)
        .test_batch_adhoc(&request, &mut CiWorkspace::default(), &ctx)
        .unwrap_err();
    assert_eq!(
        refused,
        StatsError::WorkLimit {
            message: "GPDC exact kernel factorization",
            required_work: work,
            limit_work: work - 1,
        }
    );
    assert_eq!(GPDC_FACTORIZATIONS.with(std::cell::Cell::get), before);
    // A limit equal to the quote is admitted and the factorization happens once.
    Gpdc::new()
        .with_work_limit(work)
        .test_batch_adhoc(&request, &mut CiWorkspace::default(), &ctx)
        .unwrap();
    assert_eq!(GPDC_FACTORIZATIONS.with(std::cell::Cell::get), before + 1);
    // An unconditional batch has no kernel to quote.
    let unconditional = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    let free = cache_sequence_request(&cols, &unconditional, &[]);
    assert_eq!(Gpdc::new().with_work_limit(0).quote(&free, &ctx).unwrap(), None);
}
