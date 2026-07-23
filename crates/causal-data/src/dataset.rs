//! Concrete tabular and time-series dataset types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType, VariableId,
};

use crate::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::{SamplingRegularity, TimeIndex};

/// Tabular IID dataset.
#[derive(Clone, Debug)]
pub struct TabularData {
    storage: OwnedColumnarStorage,
}

impl TabularData {
    /// Wrap owned storage.
    #[must_use]
    pub fn new(storage: OwnedColumnarStorage) -> Self {
        Self { storage }
    }

    /// Build continuous `f64` columns from named slices (equal length).
    ///
    /// Schema variables are continuous with empty role hints, in iterator order.
    /// Dense [`VariableId`]s align with column order (`0..n`).
    ///
    /// # Errors
    ///
    /// Empty input, length mismatch across columns, or schema construction failure.
    pub fn from_f64_columns<'a, S>(
        columns: impl IntoIterator<Item = (S, &'a [f64])>,
    ) -> Result<Self, DataError>
    where
        S: Into<Arc<str>>,
    {
        let collected: Vec<(Arc<str>, &'a [f64])> =
            columns.into_iter().map(|(n, v)| (n.into(), v)).collect();
        if collected.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "from_f64_columns requires at least one column".into(),
            });
        }
        let n = collected[0].1.len();
        let mut b = CausalSchemaBuilder::new();
        for (name, values) in &collected {
            if values.len() != n {
                return Err(DataError::LengthMismatch {
                    expected: n,
                    actual: values.len(),
                    context: "from_f64_columns",
                });
            }
            b.add_variable(
                Arc::clone(name),
                ValueType::Continuous,
                SmallRoleSet::empty(),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(|e| DataError::Schema(e.to_string()))?;
        }
        let schema = b.build().map_err(|e| DataError::Schema(e.to_string()))?;
        Self::try_from_schema_f64(
            schema,
            collected.iter().map(|(name, values)| (name.as_ref(), *values)),
        )
    }

    /// Bind named `f64` slices to an existing schema (names must match; order may differ).
    ///
    /// # Errors
    ///
    /// Missing/extra names, length mismatch, or storage construction failure.
    pub fn try_from_schema_f64<'a>(
        schema: CausalSchema,
        columns: impl IntoIterator<Item = (&'a str, &'a [f64])>,
    ) -> Result<Self, DataError> {
        let n_vars = schema.len();
        let mut by_name: std::collections::HashMap<&str, &[f64]> = columns.into_iter().collect();
        if by_name.len() != n_vars {
            return Err(DataError::InvalidArgument {
                message: format!("expected {} columns for schema, got {}", n_vars, by_name.len()),
            });
        }
        let mut row_count = None;
        let mut owned = Vec::with_capacity(n_vars);
        for var in schema.variables() {
            let Some(values) = by_name.remove(var.name.as_ref()) else {
                return Err(DataError::InvalidArgument {
                    message: format!("missing column for schema variable '{}'", var.name),
                });
            };
            let n = *row_count.get_or_insert(values.len());
            if values.len() != n {
                return Err(DataError::LengthMismatch {
                    expected: n,
                    actual: values.len(),
                    context: "try_from_schema_f64",
                });
            }
            let col = Float64Column::new(
                var.id,
                Arc::<[f64]>::from(values.to_vec()),
                ValidityBitmap::all_valid(n),
            )?;
            owned.push(OwnedColumn::Float64(col));
        }
        if !by_name.is_empty() {
            let extra: Vec<_> = by_name.keys().copied().collect();
            return Err(DataError::InvalidArgument {
                message: format!("columns not in schema: {extra:?}"),
            });
        }
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None)?;
        Ok(Self::new(storage))
    }

    /// Borrow underlying storage.
    #[must_use]
    pub fn storage(&self) -> &OwnedColumnarStorage {
        &self.storage
    }
}

impl TableView for TabularData {
    fn schema(&self) -> &CausalSchema {
        self.storage.schema()
    }

    fn row_count(&self) -> usize {
        self.storage.row_count()
    }

    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        self.storage.column(id)
    }
}

/// Temporal / time-series dataset with time index metadata.
#[derive(Clone, Debug)]
pub struct TimeSeriesData {
    storage: OwnedColumnarStorage,
    time_index: TimeIndex,
}

impl TimeSeriesData {
    /// Construct ensuring time index length matches row count.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn try_new(
        storage: OwnedColumnarStorage,
        time_index: TimeIndex,
    ) -> Result<Self, DataError> {
        if time_index.length != storage.row_count() {
            return Err(DataError::LengthMismatch {
                expected: storage.row_count(),
                actual: time_index.length,
                context: "time index",
            });
        }
        Ok(Self { storage, time_index })
    }

    /// Build a regularly sampled series from named `f64` columns.
    ///
    /// # Errors
    ///
    /// Propagates [`TabularData::from_f64_columns`] errors.
    pub fn from_f64_columns<'a, S>(
        columns: impl IntoIterator<Item = (S, &'a [f64])>,
        interval_ns: u64,
    ) -> Result<Self, DataError>
    where
        S: Into<Arc<str>>,
    {
        let tabular = TabularData::from_f64_columns(columns)?;
        let length = tabular.row_count();
        Self::try_new(
            tabular.storage().clone(),
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length },
        )
    }

    /// Time index metadata.
    #[must_use]
    pub fn time_index(&self) -> &TimeIndex {
        &self.time_index
    }

    /// Borrow storage.
    #[must_use]
    pub fn storage(&self) -> &OwnedColumnarStorage {
        &self.storage
    }

    /// Pointer identity of the columnar Arc (tests: planning must not clone payloads).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn columnar_ptr(&self) -> *const [OwnedColumn] {
        Arc::as_ptr(self.storage.columns_arc())
    }

    /// Restrict analysis to rows where `mask` is valid, intersected (AND) with any existing
    /// analysis mask; preserves columns, validity, weights, and time index.
    ///
    /// # Errors
    ///
    /// Mask length mismatch.
    pub fn with_analysis_mask(&self, mask: ValidityBitmap) -> Result<Self, DataError> {
        let storage = &self.storage;
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
        Self::try_new(new_storage, self.time_index.clone())
    }
}

impl TableView for TimeSeriesData {
    fn schema(&self) -> &CausalSchema {
        self.storage.schema()
    }

    fn row_count(&self) -> usize {
        self.storage.row_count()
    }

    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        self.storage.column(id)
    }
}
