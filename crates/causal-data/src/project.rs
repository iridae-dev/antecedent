//! Column projection: narrow a table to the variables needed after identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet};

use causal_core::{CausalSchemaBuilder, VariableId};

use crate::dataset::TabularData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

/// Dense-id remap produced by [`TabularData::project`].
///
/// Maps original (source) variable ids onto contiguous projected ids `0..k-1`.
#[derive(Clone, Debug)]
pub struct IdRemap {
    old_to_new: HashMap<VariableId, VariableId>,
}

impl IdRemap {
    /// Map an original id to its projected dense id.
    ///
    /// # Errors
    ///
    /// [`DataError::UnknownVariable`] when `old` was not included in the projection.
    pub fn map(&self, old: VariableId) -> Result<VariableId, DataError> {
        self.old_to_new.get(&old).copied().ok_or(DataError::UnknownVariable { id: old })
    }

    /// Number of projected columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.old_to_new.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.old_to_new.is_empty()
    }
}

impl TabularData {
    /// Project onto `ids` (order preserved, duplicates dropped).
    ///
    /// Column value buffers are Arc-shared when possible; schema and column ids
    /// are rebuilt as contiguous `0..k-1`. Analysis mask and weights are retained.
    ///
    /// # Errors
    ///
    /// Unknown variable, empty selection, or schema construction failure.
    pub fn project(&self, ids: &[VariableId]) -> Result<(Self, IdRemap), DataError> {
        let mut seen = HashSet::new();
        let mut ordered: Vec<VariableId> = Vec::with_capacity(ids.len());
        for &id in ids {
            if seen.insert(id) {
                ordered.push(id);
            }
        }
        if ordered.is_empty() {
            return Err(DataError::EmptySelection {
                context: "column projection: no variables requested",
            });
        }

        let storage = self.storage();
        let schema = storage.schema();
        let mut builder = CausalSchemaBuilder::new();
        let mut cols = Vec::with_capacity(ordered.len());
        let mut old_to_new = HashMap::with_capacity(ordered.len());

        for (new_idx, &old_id) in ordered.iter().enumerate() {
            let meta = schema.get(old_id).map_err(|_| DataError::UnknownVariable { id: old_id })?;
            builder
                .add_variable(
                    std::sync::Arc::clone(&meta.name),
                    meta.value_type.clone(),
                    meta.role_hints,
                    meta.unit.clone(),
                    meta.category_domain,
                    meta.measurement.clone(),
                )
                .map_err(|e| DataError::Schema(e.to_string()))?;
            let new_id = VariableId::from_raw(u32::try_from(new_idx).map_err(|_| {
                DataError::InvalidArgument {
                    message: "projected schema exceeds VariableId range".into(),
                }
            })?);
            old_to_new.insert(old_id, new_id);
            let col = storage
                .columns()
                .get(old_id.as_usize())
                .ok_or(DataError::UnknownVariable { id: old_id })?;
            cols.push(col.with_id(new_id));
        }

        let new_schema = builder.build().map_err(|e| DataError::Schema(e.to_string()))?;
        let new_storage = OwnedColumnarStorage::try_new(
            new_schema,
            cols,
            storage.analysis_mask().cloned(),
            storage.weights().map(std::sync::Arc::from),
        )?;
        Ok((Self::new(new_storage), IdRemap { old_to_new }))
    }
}

/// Deduplicate variable ids while preserving first-seen order.
#[must_use]
pub fn dedupe_variable_ids(ids: impl IntoIterator<Item = VariableId>) -> Vec<VariableId> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for id in ids {
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss)]

    use super::*;
    use crate::column::{Float64Column, OwnedColumn, ValidityBitmap};
    use causal_core::{MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use std::sync::Arc;

    fn float_table(names: &[&str], rows: usize) -> TabularData {
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
        let cols: Vec<OwnedColumn> = names
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let id = VariableId::from_raw(u32::try_from(i).unwrap());
                let values: Arc<[f64]> =
                    (0..rows).map(|r| (r + i * 100) as f64).collect::<Vec<_>>().into();
                OwnedColumn::Float64(
                    Float64Column::new(id, values, ValidityBitmap::all_valid(rows)).unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
    }

    #[test]
    fn project_preserves_values_and_remaps_ids() {
        let data = float_table(&["noise", "t", "y", "z", "extra"], 4);
        let t = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let z = VariableId::from_raw(3);
        let (proj, remap) = data.project(&[t, y, z]).unwrap();
        assert_eq!(proj.schema().len(), 3);
        assert_eq!(proj.schema().get(VariableId::from_raw(0)).unwrap().name.as_ref(), "t");
        assert_eq!(proj.schema().get(VariableId::from_raw(1)).unwrap().name.as_ref(), "y");
        assert_eq!(proj.schema().get(VariableId::from_raw(2)).unwrap().name.as_ref(), "z");
        assert_eq!(remap.map(t).unwrap(), VariableId::from_raw(0));
        assert_eq!(remap.map(y).unwrap(), VariableId::from_raw(1));
        assert_eq!(remap.map(z).unwrap(), VariableId::from_raw(2));
        assert!(remap.map(VariableId::from_raw(0)).is_err());

        let view = proj.column(VariableId::from_raw(0)).unwrap();
        let crate::column::ColumnView::Float64(c) = view else {
            panic!("expected float64");
        };
        assert_eq!(c.values.as_slice(), &[100.0, 101.0, 102.0, 103.0]);
    }

    #[test]
    fn project_shares_float_buffers() {
        let data = float_table(&["t", "y", "noise"], 8);
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let before = match data.storage().columns()[0] {
            OwnedColumn::Float64(ref c) => c.values.as_slice().as_ptr(),
            _ => panic!("float"),
        };
        let (proj, _) = data.project(&[t, y]).unwrap();
        let after = match proj.storage().columns()[0] {
            OwnedColumn::Float64(ref c) => c.values.as_slice().as_ptr(),
            _ => panic!("float"),
        };
        assert_eq!(before, after);
    }
}
