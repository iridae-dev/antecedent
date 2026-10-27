//! Tests for the symbolic CMI test.
#![allow(unused_imports)]

use super::*;
use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConfidenceMethod, SignificanceMethod,
};
use crate::test_util::centered_uniform as lcg_noise;
use antecedent_core::derive_stream;

#[test]
fn symbolic_mi_positive_on_copy() {
    let x: Vec<f64> = (0..100).map(|i| f64::from(u32::try_from(i % 4).unwrap_or(0))).collect();
    let y = x.clone();
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
    let ctx = ExecutionContext::for_tests(1);
    let out = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    assert!(out.results[0].statistic > 0.5);
}

/// Non-integer symbols are labels: shifting every symbol by 0.3 (and conditioning on a
/// fractional Z) yields the same statistic and permutation p-value as the integer coding.
#[test]
fn symbolic_treats_non_integer_values_as_symbols() {
    let n = 90usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| ((i / 3) % 3) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i / 3 + i / 9) % 3) as f64).collect();
    let shift = |v: &[f64]| v.iter().map(|a| a * 1.5 + 0.3).collect::<Vec<f64>>();
    let (xs, ys, zs) = (shift(&x), shift(&y), shift(&z));
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let run = |cols: [&[f64]; 3]| {
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[2],
            significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(4);
        SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
    };
    let coded = run([&x, &y, &z]);
    let labelled = run([&xs, &ys, &zs]);
    assert!(coded.statistic > 0.0);
    assert_eq!(coded.statistic.to_bits(), labelled.statistic.to_bits());
    assert_eq!(coded.p_value.to_bits(), labelled.p_value.to_bits());
}

/// When every X symbol occurs once, every rearrangement of Y is a relabelled copy of the
/// observed table, so the null never moves and the add-one p-value is exactly 1. That is a
/// valid, conservative p-value (the data cannot reject independence), not an error.
#[test]
fn symbolic_null_that_cannot_move_gives_conservative_p_of_one() {
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let cols: [&[f64]; 2] = [&x, &y];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &[],
        significance: SignificanceMethod::BlockShuffle { replicates: 19, block_size: 1 },
        confidence: ConfidenceMethod::default(),
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(3);
    let out = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    assert_eq!(out.results[0].p_value.to_bits(), 1.0_f64.to_bits());
}

/// Y = Z exactly: within every Z stratum Y is constant, so no permutation can change the
/// statistic and the add-one p-value is 1 by construction. That used to be reported as
/// "independent" (p = 1); it is no information, and is refused.
#[test]
fn symbolic_refuses_y_determined_by_z() {
    let n = 200usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
    let y = z.clone();
    let x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 5) as f64).collect();
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
    let ctx = ExecutionContext::for_tests(13);
    let err = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
    assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");
}

/// Within-stratum Y variation, X independent of Y given Z: the null has support, and the
/// p-value is an honest large one, not the degenerate p = 1.
#[test]
fn symbolic_null_preserves_yz_dependence() {
    let n = 240usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    // Y = Z plus an independent binary wobble; X independent of both given Z.
    let y: Vec<f64> = (0..n).map(|i| (i % 3) as f64 * 2.0 + ((i / 3) % 2) as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 5) as f64).collect();
    let cols: [&[f64]; 3] = [&x, &y, &z];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    let z_flat = [2usize];
    let req = CiBatchRequest {
        columns: &cols,
        queries: &queries,
        z_flat: &z_flat,
        significance: SignificanceMethod::BlockShuffle { replicates: 199, block_size: 1 },
        confidence: ConfidenceMethod::default(),
    };
    let mut ws = CiWorkspace::default();
    let ctx = ExecutionContext::for_tests(13);
    let out = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    assert!(out.results[0].p_value > 0.05, "p={}", out.results[0].p_value);
}

#[test]
fn symbolic_refuses_non_finite_symbols() {
    let n = 40usize;
    let mut x: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i / 3) % 3) as f64).collect();
    x[5] = f64::NAN;
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
    let ctx = ExecutionContext::for_tests(2);
    assert!(SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_err());
}
