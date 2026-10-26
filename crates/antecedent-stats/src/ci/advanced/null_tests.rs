//! Tests for the permutation-null behaviour shared by the nonparametric tests.
#![allow(unused_imports)]

use super::*;
use crate::ci::types::{CiBatchRequest, CiQuery, CiWorkspace, ConfidenceMethod, SignificanceMethod};
use crate::test_util::centered_uniform as lcg_noise;
use antecedent_core::derive_stream;

/// A test that cannot honour `block_size` must refuse it rather than discard it — but these
/// three draw that line in different places, and this pins all three.
///
/// `KnnDependence` and `SymbolicCmi` build their null as an exchange within Z strata
/// (`z_permutation_strata` / `discrete_strata`). **With a conditioning set** those strata are
/// Z-level or local-neighbourhood groups whose members are scattered across time: preserving `Y|Z` needs
/// permutation within scattered index sets, preserving serial dependence needs contiguous
/// runs, and the two cannot both hold. A caller asking for `block_size = 20` to preserve
/// 20-step serial dependence must not silently receive an ordinary within-stratum exchange,
/// which under-disperses the null and inflates Type I error for autocorrelated data — the one
/// case the parameter exists for. So that request is an error.
///
/// **With an empty conditioning set** the conflict disappears: both stratifiers degenerate to
/// a single stratum holding every row in original time order, so a contiguous-block
/// permutation is exactly as well defined as it is for `ParCorr`, and both tests honour it.
/// That is not a corner case — PC1's first level (`cond_size = 0`) tests unconditionally, and
/// `GSquared` already drew the line here.
///
/// `Gpdc` honours `block_size` in both shapes because it residualizes X and Y on Z via GP
/// regression *before* permuting (`gp_residual`), leaving nothing to stratify.
///
/// `block_size = 1` requests no blocking and is always accepted.
#[test]
fn block_preserving_requests_honoured_or_refused_per_conditioning_set() {
    static UNCONDITIONAL: [CiQuery; 1] = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    static CONDITIONAL: [CiQuery; 1] = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];

    // Rounded so symbols recur: distinct continuous values would each be a singleton
    // category, leaving `SymbolicCmi` no permutation support.
    let x: Vec<f64> = (0..60).map(|i| (f64::from(i) * 0.3).sin().round()).collect();
    let y: Vec<f64> = (0..60).map(|i| (f64::from(i) * 0.3).cos().round()).collect();
    let z: Vec<f64> = (0..60).map(|i| (f64::from(i) * 0.11).sin().round()).collect();
    let cols: [&[f64]; 3] = [&x, &y, &z];
    let z_flat = [2usize];
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);

    let req_for = |queries: &'static [CiQuery], block_size: usize| CiBatchRequest {
        columns: &cols,
        queries,
        z_flat: &z_flat,
        significance: SignificanceMethod::BlockShuffle { replicates: 19, block_size },
        confidence: ConfidenceMethod::default(),
    };
    let conditional: &'static [CiQuery] = &CONDITIONAL;
    let unconditional: &'static [CiQuery] = &UNCONDITIONAL;

    for block_size in [5usize, 20] {
        // Conditioned: the stratifying tests must refuse; Gpdc must not.
        let req = req_for(conditional, block_size);
        assert!(
            KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_err(),
            "KnnDependence accepted block_size={block_size} with a conditioning set, \
             which it cannot honour"
        );
        assert!(
            SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_err(),
            "SymbolicCmi accepted block_size={block_size} with a conditioning set, \
             which it cannot honour"
        );
        assert!(
            Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok(),
            "Gpdc refused block_size={block_size}, which it honours by residualizing"
        );

        // Unconditional: a single time-ordered stratum, so all three honour blocking.
        let req = req_for(unconditional, block_size);
        assert!(
            KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_ok(),
            "KnnDependence refused block_size={block_size} with an empty conditioning set, \
             where a contiguous-block permutation is well defined"
        );
        assert!(
            SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok(),
            "SymbolicCmi refused block_size={block_size} with an empty conditioning set, \
             where a contiguous-block permutation is well defined"
        );
        assert!(Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
    }

    // block_size = 1 imposes no blocking, so it must still run in both shapes.
    for queries in [conditional, unconditional] {
        let req = req_for(queries, 1);
        assert!(KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
        assert!(SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
        assert!(Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
    }
}

/// stats-ci-5: non-finite Z must not become a finite stratum / rank result.
#[test]
fn nonfinite_z_and_rank_path_return_errors() {
    let n = 30usize;
    let mut z: Vec<f64> = (0..n).map(|i| i as f64).collect();
    z[7] = f64::NAN;
    let x = lcg_noise(n, 1);
    let y = lcg_noise(n, 2);
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
    assert!(
        KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_err(),
        "KnnDependence must refuse non-finite Z"
    );
    assert!(
        coarse_z_strata(&cols, &[2], n).is_err(),
        "coarse_z_strata / z_permutation_strata must refuse non-finite Z"
    );

    let mut x_nan = x.clone();
    x_nan[3] = f64::INFINITY;
    let z_ok: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
    let cols_rank: [&[f64]; 3] = [&x_nan, &y, &z_ok];
    let req_rank = CiBatchRequest {
        columns: &cols_rank,
        queries: &queries,
        z_flat: &z_flat,
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::default(),
    };
    assert!(
        MixedKnnDependence::new(3).test_batch_adhoc(&req_rank, &mut ws, &ctx).is_err(),
        "MixedKnnDependence rank path must refuse non-finite input"
    );
}

/// The permutation stream is keyed by the query, not its position in the batch, for every
/// nonparametric test.
#[test]
fn nonparametric_p_values_are_independent_of_batch_position() {
    let n = 60usize;
    let x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 5) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i * 3 + 1) % 4) as f64).collect();
    let w: Vec<f64> = (0..n).map(|i| ((i * 5 + 2) % 6) as f64).collect();
    let cols: [&[f64]; 3] = [&x, &y, &w];
    let target = CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 };
    let other = CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 };
    let sig = SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 };
    let tests: [(&str, &dyn ConditionalIndependenceTest); 3] = [
        ("knn", &KnnDependence::new(3)),
        ("symbolic", &SymbolicCmi::new()),
        ("gpdc", &Gpdc::new()),
    ];
    for (name, ci) in tests {
        let run = |queries: &[CiQuery]| {
            let req = CiBatchRequest {
                columns: &cols,
                queries,
                z_flat: &[],
                significance: sig,
                confidence: ConfidenceMethod::None,
            };
            let mut ws = CiWorkspace::default();
            let ctx = ExecutionContext::for_tests(77);
            ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap()
        };
        let alone = run(&[target]);
        let second = run(&[other, target]);
        assert_eq!(
            alone.results[0].p_value.to_bits(),
            second.results[1].p_value.to_bits(),
            "{name}: p depends on batch position"
        );
    }
}

#[test]
fn permutation_tests_report_their_resolution() {
    let analytic = SignificanceMethod::Analytic;
    let b99 = SignificanceMethod::BlockShuffle { replicates: 99, block_size: 1 };
    for ci in [
        &KnnDependence::new(3) as &dyn ConditionalIndependenceTest,
        &MixedKnnDependence::new(3),
        &SymbolicCmi::new(),
        &Gpdc::new(),
    ] {
        assert!((ci.min_attainable_p(analytic) - 0.02).abs() < 1e-15);
        assert!((ci.min_attainable_p(b99) - 0.01).abs() < 1e-15);
    }
    assert!(crate::ci::ensure_alpha_resolvable(0.02, 0.01).is_err());
    assert!(crate::ci::ensure_alpha_resolvable(0.02, 0.05).is_ok());
}
