//! Continuous columns shared by discovery integration tests.
//!
//! Callers keep the values they already built. Packing them here does not move
//! an oracle edge.

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};

/// Named continuous columns, in order. Column variable ids are `0..columns.len()`.
pub fn continuous_storage(
    names: &[impl AsRef<str>],
    columns: Vec<Vec<f64>>,
) -> OwnedColumnarStorage {
    let n = columns.first().map_or(0, Vec::len);
    let mut builder = CausalSchemaBuilder::new();
    for name in names {
        builder
            .add_variable(
                name.as_ref(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let owned = columns
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
    OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap()
}
