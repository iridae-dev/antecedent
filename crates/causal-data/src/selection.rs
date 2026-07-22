//! Row selection and column transform helpers shared by estimators / refuters.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};

use crate::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
use crate::dataset::TabularData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

impl TabularData {
    /// Row mask: analysis mask ∩ validity of every listed column.
    ///
    /// # Errors
    ///
    /// Unknown variables, or no remaining complete cases.
    pub fn complete_case_mask(&self, ids: &[VariableId]) -> Result<Vec<bool>, DataError> {
        let n = self.row_count();
        let mut keep = vec![true; n];
        if let Some(mask) = self.storage().analysis_mask() {
            for (i, slot) in keep.iter_mut().enumerate() {
                *slot = mask.is_valid(i);
            }
        }
        for &id in ids {
            let validity = self.column(id)?.validity();
            for (i, slot) in keep.iter_mut().enumerate() {
                if *slot && !validity.is_valid(i) {
                    *slot = false;
                }
            }
        }
        if !keep.iter().any(|k| *k) {
            return Err(DataError::EmptySelection {
                context: "complete-case mask after validity/analysis filtering",
            });
        }
        Ok(keep)
    }

    /// Extract float64 values for rows where `keep[i]` is true.
    ///
    /// # Errors
    ///
    /// Unknown / non-float64 column, or keep length mismatch.
    pub fn float64_masked(&self, id: VariableId, keep: &[bool]) -> Result<Vec<f64>, DataError> {
        if keep.len() != self.row_count() {
            return Err(DataError::LengthMismatch {
                expected: self.row_count(),
                actual: keep.len(),
                context: "complete-case keep mask",
            });
        }
        let ColumnView::Float64(c) = self.column(id)? else {
            return Err(DataError::TypeMismatch { id, expected: "float64" });
        };
        let mut out = Vec::with_capacity(keep.iter().filter(|k| **k).count());
        for (i, &k) in keep.iter().enumerate() {
            if k {
                out.push(c.values[i]);
            }
        }
        Ok(out)
    }

    /// Replace one float64 column; preserve other columns, analysis mask, and weights.
    ///
    /// The replacement column is marked all-valid (caller supplies a complete vector).
    ///
    /// # Errors
    ///
    /// Unknown id, length mismatch, or non-float target.
    pub fn with_replaced_float(
        &self,
        id: VariableId,
        values: Arc<[f64]>,
    ) -> Result<Self, DataError> {
        let n = self.row_count();
        if values.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: values.len(),
                context: "replacement float column",
            });
        }
        let storage = self.storage();
        let mut cols: Vec<OwnedColumn> = storage.columns().to_vec();
        let idx = id.as_usize();
        if idx >= cols.len() {
            return Err(DataError::UnknownVariable { id });
        }
        if !matches!(cols[idx], OwnedColumn::Float64(_)) {
            return Err(DataError::TypeMismatch { id, expected: "float64" });
        }
        cols[idx] =
            OwnedColumn::Float64(Float64Column::new(id, values, ValidityBitmap::all_valid(n))?);
        let storage = OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            cols,
            storage.analysis_mask().cloned(),
            storage.weights().map(Arc::from),
        )?;
        Ok(Self::new(storage))
    }

    /// Restrict analysis to rows where `mask` is valid, intersected (AND) with any existing
    /// analysis mask; preserves columns, validity, and weights.
    ///
    /// # Errors
    ///
    /// Mask length mismatch.
    pub fn with_analysis_mask(&self, mask: ValidityBitmap) -> Result<Self, DataError> {
        let storage = self.storage();
        let n = storage.row_count();
        if mask.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: mask.len(),
                context: "analysis mask",
            });
        }
        let combined = match storage.analysis_mask() {
            Some(existing) => {
                let mut bytes = vec![0u8; n.div_ceil(8)];
                for i in 0..n {
                    if existing.is_valid(i) && mask.is_valid(i) {
                        bytes[i / 8] |= 1 << (i % 8);
                    }
                }
                ValidityBitmap::from_bytes(bytes, n)?
            }
            None => mask,
        };
        let new_storage = OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            storage.columns().to_vec(),
            Some(combined),
            storage.weights().map(Arc::from),
        )?;
        Ok(Self::new(new_storage))
    }

    /// Append a continuous float64 covariate; preserve existing columns/mask/weights.
    ///
    /// # Errors
    ///
    /// Length mismatch or schema construction failure.
    pub fn with_appended_float(
        &self,
        name: &str,
        values: Arc<[f64]>,
    ) -> Result<(Self, VariableId), DataError> {
        let n = self.row_count();
        if values.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: values.len(),
                context: "appended float column",
            });
        }
        let storage = self.storage();
        let mut builder = CausalSchemaBuilder::new();
        for v in storage.schema().variables() {
            builder
                .add_variable(
                    Arc::clone(&v.name),
                    v.value_type.clone(),
                    v.role_hints,
                    v.unit.clone(),
                    v.category_domain,
                    v.measurement.clone(),
                )
                .map_err(|e| DataError::Schema(e.to_string()))?;
        }
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(|e| DataError::Schema(e.to_string()))?;
        let schema = builder.build().map_err(|e| DataError::Schema(e.to_string()))?;
        let new_id = VariableId::from_raw(u32::try_from(schema.len() - 1).map_err(|_| {
            DataError::InvalidArgument { message: "schema exceeds VariableId range".into() }
        })?);
        let mut cols: Vec<OwnedColumn> = storage.columns().to_vec();
        cols.push(OwnedColumn::Float64(Float64Column::new(
            new_id,
            values,
            ValidityBitmap::all_valid(n),
        )?));
        let storage = OwnedColumnarStorage::try_new(
            schema,
            cols,
            storage.analysis_mask().cloned(),
            storage.weights().map(Arc::from),
        )?;
        Ok((Self::new(storage), new_id))
    }
}
