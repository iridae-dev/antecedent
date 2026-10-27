//! Tests for the oracle test.
#![allow(unused_imports)]

use super::*;
use crate::ci::types::{
    CiBatchRequest, CiQuery, CiWorkspace, ConfidenceMethod, SignificanceMethod,
};
use crate::test_util::centered_uniform as lcg_noise;
use antecedent_core::derive_stream;

#[test]
fn oracle_marks_dependence() {
    let oracle = OracleCi::new([(0usize, 1usize)]);
    let x = [1.0, 2.0, 3.0];
    let y = [1.0, 2.0, 3.0];
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
    let out = oracle.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
    assert!((out.results[0].p_value - 0.0).abs() < f64::EPSILON);
}
