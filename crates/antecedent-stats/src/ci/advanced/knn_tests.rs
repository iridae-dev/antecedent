//! Tests for the kNN distance-dependence tests.
#![allow(unused_imports)]

use super::*;
use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConfidenceMethod, SignificanceMethod,
};
use crate::test_util::centered_uniform as lcg_noise;
use antecedent_core::derive_stream;

#[test]
fn knn_rebuilds_index_for_different_pairs_in_one_batch() {
    let n = 60usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y_tight: Vec<f64> = x.iter().map(|v| v + 0.001).collect();
    let y_spread: Vec<f64> = (0..n).map(|i| ((i * 37 + 11) % 60) as f64).collect();
    let cols: [&[f64]; 3] = [&x, &y_tight, &y_spread];
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
    let ctx = ExecutionContext::for_tests(12);
    let out = KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    let s1 = out.results[0].statistic;
    let s2 = out.results[1].statistic;
    assert!((s1 - s2).abs() > 1e-6, "same n/dim pairs must not share a cached index: {s1} vs {s2}");
    assert!(s1 > s2, "tight pair should have smaller kth distances: {s1} vs {s2}");
}

#[test]
fn knn_null_preserves_yz_dependence() {
    // X ⊥ Y | Z with Y strongly driven by Z (three well-separated Z levels): the
    // within-strata null must not report systematically tiny p-values.
    let n = 90usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 3) as f64 * 5.0).collect();
    let noise = lcg_noise(n, 6);
    let y: Vec<f64> = z.iter().zip(&noise).map(|(v, e)| v + 0.1 * e).collect();
    let x = lcg_noise(n, 5);
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
    // Under H0 the p-value is uniform, so any single seed has a ~5% chance of a small p.
    // A null that ignored Y|Z would put every seed near the floor; require most seeds to
    // sit above 0.05.
    let mut ws = CiWorkspace::default();
    let ps: Vec<f64> = (14..20u64)
        .map(|seed| {
            let ctx = ExecutionContext::for_tests(seed);
            KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0].p_value
        })
        .collect();
    let above = ps.iter().filter(|&&p| p > 0.05).count();
    assert!(above >= 4, "kNN conditional null reports tiny p-values: {ps:?}");
}

/// stats-ci-1: continuous Z with Y←Z must not yield Type I ≈ 1 under X ⊥ Y | Z.
#[test]
fn knn_continuous_z_null_type_i_not_near_one() {
    let n = 120usize;
    let trials = 200u32;
    let alpha = 0.05;
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(0x000C_11C0_u64);
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let z_flat = [2usize];
    let ci = KnnDependence::new(5);
    let mut rejects = 0u32;
    for t in 0..trials {
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, derive_stream(0xC11, u64::from(t)));
        let z: Vec<f64> = (0..n)
            .map(|_| {
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                // Box–Muller half: approximate N(0,1)
                let v = (rng.next_u64() as f64) / (u64::MAX as f64);
                let r = (-2.0 * (u.max(1e-12)).ln()).sqrt();
                r * (2.0 * std::f64::consts::PI * v).cos()
            })
            .collect();
        let y: Vec<f64> = z
            .iter()
            .map(|&zi| {
                let e = (rng.next_u64() as f64) / (u64::MAX as f64) - 0.5;
                zi + 0.35 * e
            })
            .collect();
        let x: Vec<f64> =
            (0..n).map(|_| (rng.next_u64() as f64) / (u64::MAX as f64) - 0.5).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let out = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        if out.results[0].p_value < alpha {
            rejects += 1;
        }
    }
    let rate = f64::from(rejects) / f64::from(trials);
    // Coarse tercile nulls reject ≈ 1.0 here; local neighbourhoods must stay far below that.
    assert!(
        rate < 0.25,
        "continuous-Z kNN type I near 1 (or badly inflated): {rate} ({rejects}/{trials})"
    );
}

/// Bivariate Z with X and Y both driven by the first Z coordinate, so X ⊥ Y | Z. Continuous
/// bivariate Z used to be grouped by the norm of Z, pairing z = (2, 0) with z = (-2, 0) and
/// rejecting every trial; it is now refused. Discrete bivariate Z keeps exact level strata
/// and a controlled Type I rate.
#[test]
fn knn_bivariate_z_conditional_null() {
    let n = 100usize;
    let trials = 60u32;
    let alpha = 0.05;
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(0x2D_2D_u64);
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
    let z_flat = [2usize, 3];
    let ci = KnnDependence::new(5);
    let mut rejects = 0u32;
    for t in 0..trials {
        let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, derive_stream(0x2D2, u64::from(t)));
        let mut normal = || {
            let u = (rng.next_u64() as f64) / (u64::MAX as f64);
            let v = (rng.next_u64() as f64) / (u64::MAX as f64);
            (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        };
        let z1: Vec<f64> = (0..n).map(|_| normal()).collect();
        let z2: Vec<f64> = (0..n).map(|_| normal()).collect();
        let x: Vec<f64> = z1.iter().map(|&a| a + 0.35 * normal()).collect();
        let y: Vec<f64> = z1.iter().map(|&a| a + 0.35 * normal()).collect();
        let cols: [&[f64]; 4] = [&x, &y, &z1, &z2];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let err = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
        assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");

        let d1: Vec<f64> = z1.iter().map(|v| v.round().clamp(-1.0, 1.0)).collect();
        let d2: Vec<f64> = z2.iter().map(|v| f64::from(u8::from(*v > 0.0))).collect();
        let x: Vec<f64> = d1.iter().map(|&a| 2.0 * a + 0.35 * normal()).collect();
        let y: Vec<f64> = d1.iter().map(|&a| 2.0 * a + 0.35 * normal()).collect();
        let cols: [&[f64]; 4] = [&x, &y, &d1, &d2];
        let req = CiBatchRequest { columns: &cols, ..req };
        if ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0].p_value < alpha {
            rejects += 1;
        }
    }
    let rate = f64::from(rejects) / f64::from(trials);
    // alpha + 3 binomial SE at 60 trials is about 0.135.
    assert!(rate < 0.14, "discrete bivariate-Z kNN type I inflated: {rate} ({rejects}/{trials})");
}

#[test]
fn knn_refuses_constant_series() {
    let n = 30usize;
    let x = lcg_noise(n, 8);
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
    let err = KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
    assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");
}

#[test]
fn knn_dependence_runs_beyond_the_former_ten_thousand_row_limit() {
    let n = 12_000usize;
    let mut rng = ExecutionContext::for_tests(31).rng.stream_for(StreamDomain::Test, 0x4B4E);
    let x: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
    let y_dep: Vec<f64> =
        x.iter().map(|v| v + 0.3 * antecedent_kernels::standard_normal(&mut rng)).collect();
    let y_ind: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
    let cols: [&[f64]; 3] = [&x, &y_dep, &y_ind];
    let queries = [
        CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 },
        CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 },
    ];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &[],
        significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
        confidence: ConfidenceMethod::None,
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(32);
    let out = KnnDependence::new(5).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    // Every null replicate is less dependent than the observed pair: the add-one floor.
    assert!((out.results[0].p_value - 0.1).abs() < 1e-12, "{}", out.results[0].p_value);
    assert!(out.results[0].statistic > out.results[1].statistic);
}
