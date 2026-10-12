//! Empty continuous tables shared by the static-discovery unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};

/// `ncols` all-zero continuous columns of `nrows` rows, variables `0..ncols`.
pub(crate) fn tabular_n(ncols: usize, nrows: usize) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for i in 0..ncols {
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
    let owned: Vec<OwnedColumn> = (0..ncols)
        .map(|i| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX)),
                    Arc::from(vec![0.0; nrows]),
                    ValidityBitmap::all_valid(nrows),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
    TabularData::new(storage)
}
