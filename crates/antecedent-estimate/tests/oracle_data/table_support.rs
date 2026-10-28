//! A finite continuous table builder shared by the reference tests (no assertions).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "each reference test uses a subset of the helpers")]

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};

/// Columns in order, with variable ids `0..columns.len()`.
pub fn table(columns: &[(&str, RoleHint, Vec<f64>)]) -> TabularData {
    let n = columns[0].2.len();
    let mut builder = CausalSchemaBuilder::new();
    for (i, (name, hint, _)) in columns.iter().enumerate() {
        builder
            .add_variable(
                format!("{name}{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(*hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let cols = columns
        .iter()
        .enumerate()
        .map(|(i, (_, _, values))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(values.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), cols, None, None).unwrap(),
    )
}

pub fn id(i: u32) -> VariableId {
    VariableId::from_raw(i)
}
