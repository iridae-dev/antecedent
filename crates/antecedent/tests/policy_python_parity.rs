//! A price-floor policy sampled from a fitted model is bit-identical in Rust and Python.
//!
//! Both sides fit the same deterministic data with the default mechanism selection and draw
//! with the same seed; the checksum of every drawn bit is pinned here and in
//! `python/tests/test_intervention_programs.py`. Python builds program text and calls this
//! same entry point, so a difference would mean a second implementation had crept in.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::gcm::{fit_gcm, sample_do};
use antecedent_core::{CausalRng, ExecutionContext, Intervention, ValueExpr, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

/// Sum (mod 2^64) of the IEEE-754 bits of every draw of the policy run in the Python test.
const PRICE_FLOOR_DRAW_CHECKSUM: u64 = 15_163_479_412_849_571_691;

#[test]
fn a_price_floor_draws_the_same_bits_from_rust_as_from_python() {
    let price: Vec<f64> =
        (0..600_u32).map(|i| 11.0 + (f64::from((i * 37) % 101) - 50.0) * 0.04).collect();
    let demand: Vec<f64> = (0..600_u32)
        .zip(&price)
        .map(|(i, p)| 50.0 - 0.8 * p + (f64::from((i * 13) % 23) - 11.0) * 0.05)
        .collect();
    let data =
        TabularData::from_f64_columns([("price", price.as_slice()), ("demand", demand.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let fitted = fit_gcm(graph, &data).unwrap();

    let floor = Intervention::transform(
        VariableId::from_raw(0),
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated())),
    );
    let ctx = ExecutionContext::for_tests(7);
    let draws =
        sample_do(&fitted.model, &[floor], 512, &mut CausalRng::from_seed(7), &ctx).unwrap();
    let checksum = draws.values.iter().fold(0_u64, |sum, v| sum.wrapping_add(v.to_bits()));
    assert_eq!(checksum, PRICE_FLOOR_DRAW_CHECKSUM);
    assert!(draws.column(0).unwrap().iter().all(|p| *p >= 10.0));
}
