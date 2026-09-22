//! Discrete column reader: one column as level codes.
//!
//! Every estimator that treats a column as a finite-domain variable (empirical
//! transport tables, the functional-distribution estimator) reads it through this
//! one function, so they agree on which rows are missing: a row is missing when its
//! cell is invalid **or** the row lies outside the storage analysis mask.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_core::{Value, VariableId};

use crate::column::ColumnView;
use crate::dataset::TabularData;
use crate::error::DataError;
use crate::table::TableView;

/// A discrete column as level codes.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteColumn {
    /// Per row, the index into [`Self::levels`], or [`Self::MISSING`] for a row whose cell
    /// is invalid or outside the analysis mask.
    pub codes: Vec<u32>,
    /// Distinct observed values in order of first appearance (missing rows contribute
    /// none); `codes` index into it.
    pub levels: Vec<Value>,
}

impl DiscreteColumn {
    /// Code of a missing (invalid or masked-out) row.
    pub const MISSING: u32 = u32::MAX;

    /// Value of row `row`, or `None` when it is missing.
    #[must_use]
    pub fn value(&self, row: usize) -> Option<&Value> {
        match self.codes.get(row).copied() {
            Some(code) if code != Self::MISSING => self.levels.get(code as usize),
            _ => None,
        }
    }
}

impl TabularData {
    /// Read column `id` as a discrete variable: float64, int64 or categorical cells become
    /// level codes; rows that are invalid or outside the analysis mask are
    /// [`DiscreteColumn::MISSING`].
    ///
    /// # Errors
    ///
    /// Unknown variable, a column type that is not float64 / int64 / categorical, or more
    /// distinct values than a `u32` code can index.
    pub fn discrete_column(&self, id: VariableId) -> Result<DiscreteColumn, DataError> {
        let view = self.column(id)?;
        let n = view.len();
        let validity = view.validity();
        let mask = self.storage().analysis_mask();
        let analyzed = |i: usize| validity.is_valid(i) && mask.is_none_or(|m| m.is_valid(i));
        let mut index: HashMap<Value, u32> = HashMap::new();
        let mut levels: Vec<Value> = Vec::new();
        let mut codes = Vec::with_capacity(n);
        let mut push = |value: Value| -> Result<u32, DataError> {
            if let Some(&code) = index.get(&value) {
                return Ok(code);
            }
            let code = u32::try_from(levels.len())
                .ok()
                .filter(|&c| c != DiscreteColumn::MISSING)
                .ok_or(DataError::InvalidArgument {
                message: "discrete column has more levels than a u32 code can index".into(),
            })?;
            index.insert(value.clone(), code);
            levels.push(value);
            Ok(code)
        };
        match view {
            ColumnView::Float64(c) => {
                for i in 0..n {
                    codes.push(if analyzed(i) {
                        push(Value::f64(c.values[i]))?
                    } else {
                        DiscreteColumn::MISSING
                    });
                }
            }
            ColumnView::Int64(c) => {
                for i in 0..n {
                    codes.push(if analyzed(i) {
                        push(Value::Int64(c.values[i]))?
                    } else {
                        DiscreteColumn::MISSING
                    });
                }
            }
            ColumnView::Categorical(c) => {
                for i in 0..n {
                    codes.push(if analyzed(i) {
                        push(Value::Category(c.codes[i].raw()))?
                    } else {
                        DiscreteColumn::MISSING
                    });
                }
            }
            _ => {
                return Err(DataError::TypeMismatch {
                    id,
                    expected: "float64, int64 or categorical",
                });
            }
        }
        Ok(DiscreteColumn { codes, levels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_follow_first_appearance_and_mark_masked_rows() {
        let data = TabularData::from_f64_columns([("x", &[5.0, 2.0, 9.0, 5.0, 2.0][..])]).unwrap();
        // Row 2 (the only 9.0) and row 3 are outside the analysis mask.
        let mask = crate::column::ValidityBitmap::from_bytes(vec![0b1_0011_u8], 5).unwrap();
        let masked = data.with_analysis_mask(mask).unwrap();
        let col = masked.discrete_column(VariableId::from_raw(0)).unwrap();
        // Masked rows are missing and contribute no level: 9.0 is not a level, and 5.0
        // and 2.0 keep their first-seen codes.
        assert_eq!(col.codes, vec![0, 1, DiscreteColumn::MISSING, DiscreteColumn::MISSING, 1]);
        assert_eq!(col.levels, vec![Value::f64(5.0), Value::f64(2.0)]);
        assert_eq!(col.value(4), Some(&Value::f64(2.0)));
        assert_eq!(col.value(2), None);
        assert_eq!(col.value(99), None);
        // Unmasked, every row has a level and 9.0 is one of them.
        let full = data.discrete_column(VariableId::from_raw(0)).unwrap();
        assert_eq!(full.levels, vec![Value::f64(5.0), Value::f64(2.0), Value::f64(9.0)]);
        assert_eq!(full.codes, vec![0, 1, 2, 0, 1]);
    }
}
