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

/// Named continuous columns, in order. Variable ids are `0..columns.len()`.
///
/// Callers keep the column values they already built, so a shared table does not
/// move an oracle edge.
pub(crate) fn continuous_table(names: &[&str], columns: Vec<Vec<f64>>) -> TabularData {
    let n = columns.first().map_or(0, Vec::len);
    let mut b = CausalSchemaBuilder::new();
    for name in names {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let owned: Vec<OwnedColumn> = columns
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX)),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
    TabularData::new(storage)
}

/// `ncols` all-zero continuous columns of `nrows` rows, variables `0..ncols`.
pub(crate) fn tabular_n(ncols: usize, nrows: usize) -> TabularData {
    let names: Vec<String> = (0..ncols).map(|i| format!("v{i}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    continuous_table(&refs, (0..ncols).map(|_| vec![0.0; nrows]).collect())
}
