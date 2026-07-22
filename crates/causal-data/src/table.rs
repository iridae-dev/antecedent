//! [`TableView`] trait — public causal table API (ADR 0004).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{CausalSchema, VariableId};

use crate::column::ColumnView;
use crate::error::DataError;

/// Lossy `i64` → `f64` for the float analysis path (mantissa cannot hold all `i64`).
fn analysis_f64_from_i64(v: i64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        v as f64
    }
}

/// Borrowed table access used by algorithms.
pub trait TableView {
    /// Immutable causal schema.
    fn schema(&self) -> &CausalSchema;

    /// Number of rows.
    fn row_count(&self) -> usize;

    /// Column view for `id`.
    ///
    /// # Errors
    ///
    /// Unknown variable or type issues.
    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError>;

    /// Copy a column into an owned `f64` buffer.
    ///
    /// Native `Float64` columns are copied as-is. `Int64` and `Boolean` columns
    /// are coerced to `f64` (`true` → `1.0`, `false` → `0.0`); invalid rows become
    /// `NaN`. Other column kinds (categorical, timestamp, fixed vector) error.
    ///
    /// # Errors
    ///
    /// Unknown variable or unsupported column type.
    fn float64_values(&self, id: VariableId) -> Result<Vec<f64>, DataError> {
        match self.column(id)? {
            ColumnView::Float64(c) => Ok(c.values.to_vec()),
            ColumnView::Int64(c) => {
                let mut out = Vec::with_capacity(c.values.len());
                for (i, &v) in c.values.iter().enumerate() {
                    out.push(if c.validity.is_valid(i) {
                        analysis_f64_from_i64(v)
                    } else {
                        f64::NAN
                    });
                }
                Ok(out)
            }
            ColumnView::Boolean(c) => {
                let mut out = Vec::with_capacity(c.values.len());
                for (i, &v) in c.values.iter().enumerate() {
                    out.push(if c.validity.is_valid(i) { f64::from(v) } else { f64::NAN });
                }
                Ok(out)
            }
            _ => Err(DataError::TypeMismatch {
                id,
                expected: "float64 (or coercible int64/boolean)",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };

    use super::*;
    use crate::column::{BooleanColumn, Float64Column, Int64Column, OwnedColumn, ValidityBitmap};
    use crate::dataset::TabularData;
    use crate::storage::OwnedColumnarStorage;

    fn schema_n(n: usize) -> causal_core::CausalSchema {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..n {
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
        b.build().unwrap()
    }

    #[test]
    fn float64_values_coerces_int64_and_boolean() {
        let schema = schema_n(3);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from([1.5_f64, 2.5]),
                    ValidityBitmap::all_valid(2),
                )
                .unwrap(),
            ),
            OwnedColumn::Int64(
                Int64Column::new(
                    VariableId::from_raw(1),
                    Arc::<[i64]>::from([3_i64, 4]),
                    ValidityBitmap::all_valid(2),
                )
                .unwrap(),
            ),
            OwnedColumn::Boolean(
                BooleanColumn::new(
                    VariableId::from_raw(2),
                    Arc::<[u8]>::from([1_u8, 0]),
                    ValidityBitmap::all_valid(2),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        assert_eq!(data.float64_values(VariableId::from_raw(0)).unwrap(), vec![1.5, 2.5]);
        assert_eq!(data.float64_values(VariableId::from_raw(1)).unwrap(), vec![3.0, 4.0]);
        assert_eq!(data.float64_values(VariableId::from_raw(2)).unwrap(), vec![1.0, 0.0]);
    }
}
