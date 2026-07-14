//! Shared test helpers for causal-data unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};

use crate::column::{Float64Column, OwnedColumn, ValidityBitmap};
use crate::dataset::TimeSeriesData;
use crate::storage::OwnedColumnarStorage;
use crate::temporal::{SamplingRegularity, TimeIndex};

/// Build a float64 series with `vars` columns named `v0..` of length `n`.
pub(crate) fn float_series(n: usize, vars: usize) -> TimeSeriesData {
    float_series_inner(n, vars, None, None)
}

/// Like [`float_series`] but with row `invalid_row` of `v0` marked missing.
pub(crate) fn float_series_with_gap(n: usize, vars: usize, invalid_row: usize) -> TimeSeriesData {
    float_series_inner(n, vars, Some(invalid_row), None)
}

/// Like [`float_series`] but with row `hidden_row` excluded by an analysis mask.
pub(crate) fn float_series_with_mask(n: usize, vars: usize, hidden_row: usize) -> TimeSeriesData {
    float_series_inner(n, vars, None, Some(hidden_row))
}

fn float_series_inner(
    n: usize,
    vars: usize,
    invalid_row: Option<usize>,
    hidden_row: Option<usize>,
) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for i in 0..vars {
        b.add_variable(
            format!("v{i}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut cols = Vec::with_capacity(vars);
    for v in 0..vars {
        let values: Vec<f64> = (0..n).map(|t| (t as f64) + 100.0 * v as f64).collect();
        let validity = match invalid_row {
            Some(row) if v == 0 => bitmap_without(n, row),
            _ => ValidityBitmap::all_valid(n),
        };
        cols.push(OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(v as u32), Arc::from(values), validity)
                .unwrap(),
        ));
    }
    let mask = hidden_row.map(|row| bitmap_without(n, row));
    let storage = OwnedColumnarStorage::try_new(schema, cols, mask, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

/// All-valid bitmap of `n` bits with bit `row` cleared.
fn bitmap_without(n: usize, row: usize) -> ValidityBitmap {
    let mut bytes = vec![0xFFu8; n.div_ceil(8)];
    bytes[row / 8] &= !(1 << (row % 8));
    ValidityBitmap::from_bytes(bytes, n).unwrap()
}
