//! The compiled `IvBounds` study and the direct `binary_iv_ate_bounds` call solve the same
//! Balke-Pearl program.
//!
//! The direct entry takes the observed law `P(Y=y, D=d | Z=z)`; the compiled study reads the law
//! from rows and solves it with the same kernel function. On a table whose per-arm cell counts
//! are the counts behind the law, the two endpoints agree bit for bit, whether the rows are
//! unit rows or one weighted row per cell.
//!
//! The direct entry reports the identified set and no sampling interval, so there is no direct
//! interval to compare. The compiled route's interval and calibration status are read from its
//! own graphless support row, which this file does not change.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "fixture counts convert exactly and the bits are compared exactly"
)]

use antecedent::identify::{BinaryIvLaw, binary_iv_ate_bounds};
use antecedent::{CausalQuery, IvBoundsQuery, RefuteSuite, Study};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Rows per instrument arm of each `(y, d)` cell, order `(0,0), (1,0), (0,1), (1,1)`.
const CELLS: [[usize; 4]; 2] = [[51, 29, 6, 14], [21, 9, 16, 54]];
/// A second law with unequal arm sizes.
const UNEQUAL: [[usize; 4]; 2] = [[40, 22, 9, 4], [17, 8, 31, 70]];

/// The law the counts describe, formed as the row reader forms it: a cell's mass over its arm's.
fn law_of(counts: &[[usize; 4]; 2]) -> BinaryIvLaw {
    let mut cells = [[0.0; 4]; 2];
    for (arm, row) in counts.iter().enumerate() {
        let total: f64 = row.iter().map(|&count| count as f64).sum();
        for (cell, &count) in row.iter().enumerate() {
            cells[arm][cell] = count as f64 / total;
        }
    }
    BinaryIvLaw { cells }
}

/// One row per unit, columns y, d, z.
fn unit_rows(counts: &[[usize; 4]; 2]) -> TabularData {
    let (mut y, mut d, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for (arm, row) in counts.iter().enumerate() {
        for (cell, &count) in row.iter().enumerate() {
            for _ in 0..count {
                y.push((cell % 2) as f64);
                d.push((cell / 2) as f64);
                z.push(arm as f64);
            }
        }
    }
    TabularData::from_f64_columns([("y", &y[..]), ("d", &d[..]), ("z", &z[..])]).unwrap()
}

/// One weighted row per cell, columns y, d, z, w.
fn weighted_rows(counts: &[[usize; 4]; 2]) -> TabularData {
    let (mut y, mut d, mut z, mut w) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (arm, row) in counts.iter().enumerate() {
        for (cell, &count) in row.iter().enumerate() {
            y.push((cell % 2) as f64);
            d.push((cell / 2) as f64);
            z.push(arm as f64);
            w.push(count as f64);
        }
    }
    TabularData::from_f64_columns([("y", &y[..]), ("d", &d[..]), ("z", &z[..]), ("w", &w[..])])
        .unwrap()
}

fn compiled(data: TabularData, weighted: bool) -> antecedent_estimate::IvBoundsEstimate {
    let query = IvBoundsQuery::new(v(0), v(1), v(2));
    let query = if weighted { query.with_weight(v(3)) } else { query };
    let ctx = ExecutionContext::for_tests(11);
    let result = Study::tabular(data)
        .query(CausalQuery::IvBounds(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    result.iv_bounds().expect("a bounds payload").clone()
}

fn assert_twin(counts: &[[usize; 4]; 2]) {
    let law = law_of(counts);
    let direct = binary_iv_ate_bounds(law).unwrap();
    for (label, fit) in [
        ("unit rows", compiled(unit_rows(counts), false)),
        ("weighted cells", compiled(weighted_rows(counts), true)),
    ] {
        assert_eq!(fit.lower.to_bits(), direct.lower.to_bits(), "{label}: lower");
        assert_eq!(fit.upper.to_bits(), direct.upper.to_bits(), "{label}: upper");
        for arm in 0..2 {
            for cell in 0..4 {
                assert_eq!(
                    fit.law.cells[arm][cell].to_bits(),
                    law.cells[arm][cell].to_bits(),
                    "{label}: law cell ({arm}, {cell})"
                );
            }
        }
        assert!(fit.interval.is_none() && fit.posterior.is_none(), "{label}");
    }
}

#[test]
fn compiled_iv_bounds_equal_the_direct_balke_pearl_bounds_on_the_mixture_law() {
    assert_twin(&CELLS);
}

#[test]
fn compiled_iv_bounds_equal_the_direct_balke_pearl_bounds_with_unequal_arms() {
    assert_twin(&UNEQUAL);
}

#[test]
fn a_law_the_instrument_premises_cannot_produce_is_refused_by_both() {
    // The instrument moves the outcome among the never-takers' arm: no response-type law fits.
    let counts: [[usize; 4]; 2] = [[50, 0, 0, 0], [0, 50, 0, 0]];
    let law = law_of(&counts);
    let direct = binary_iv_ate_bounds(law).expect_err("the inequality is violated");
    assert!(direct.to_string().contains("instrumental inequality"), "{direct}");
    let ctx = ExecutionContext::for_tests(11);
    let compiled = Study::tabular(unit_rows(&counts))
        .query(CausalQuery::IvBounds(IvBoundsQuery::new(v(0), v(1), v(2))))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .expect_err("the compiled study refuses the same law");
    assert!(compiled.to_string().contains("instrumental inequality"), "{compiled}");
}
