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
        cols.push(OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(v as u32),
                Arc::from(values),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ));
    }
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}
