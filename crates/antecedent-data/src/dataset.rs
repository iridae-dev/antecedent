//! Concrete tabular and time-series dataset types.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType, VariableId,
};

use crate::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;
use crate::temporal::{SamplingRegularity, TimeIndex};

/// The table a derived table's rows were taken from: the digest of that table's rows and, when
/// rows were selected, the index in it of each derived row.
///
/// A column projection keeps every row in order, so the rows of two projections of one table
/// are the same observations. A row selection names the rows it kept. Deriving again from a
/// derived table composes the indices, so the origin is always the first table that was not
/// itself derived. The digest covers the rows' columns and validity, not an analysis mask or
/// weights ([`OwnedColumnarStorage::rows_digest`]): the same rows under another mask or
/// weighting are the same observations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowOrigin {
    table: [u8; 32],
    rows: Option<Arc<[u64]>>,
}

impl RowOrigin {
    /// Every row of the table whose content digest is `table`, in order.
    #[must_use]
    pub const fn whole(table: [u8; 32]) -> Self {
        Self { table, rows: None }
    }

    /// Rows digest ([`OwnedColumnarStorage::rows_digest`]) of the source table.
    #[must_use]
    pub const fn table(&self) -> [u8; 32] {
        self.table
    }

    /// Index in the source table of each row, when the rows were selected; `None` when every
    /// row was kept in order.
    #[must_use]
    pub fn rows(&self) -> Option<&[u64]> {
        self.rows.as_deref()
    }

    /// The origin of rows `selected` (indices into the rows this origin describes).
    fn select(&self, selected: &[usize]) -> Result<Self, DataError> {
        let rows = match &self.rows {
            None => selected
                .iter()
                .map(|&row| u64::try_from(row).map_err(|_| DataError::overflow("row index")))
                .collect::<Result<Vec<u64>, DataError>>()?,
            Some(parent) => selected
                .iter()
                .map(|&row| {
                    parent.get(row).copied().ok_or_else(|| {
                        DataError::invalid_value("row selection", "exceeds the table")
                    })
                })
                .collect::<Result<Vec<u64>, DataError>>()?,
        };
        Ok(Self { table: self.table, rows: Some(Arc::from(rows)) })
    }
}

/// Tabular IID dataset.
#[derive(Clone, Debug)]
pub struct TabularData {
    storage: OwnedColumnarStorage,
    /// Declared unit of every row. A declaration beside the data, outside its content digest.
    unit_key: Option<Arc<crate::unit_key::UnitKey>>,
    /// The table this one's rows were taken from, set by a projection or a row selection.
    /// Beside the data, outside its content digest.
    row_origin: Option<Arc<RowOrigin>>,
}

impl TabularData {
    /// Wrap owned storage.
    #[must_use]
    pub fn new(storage: OwnedColumnarStorage) -> Self {
        Self { storage, unit_key: None, row_origin: None }
    }

    /// The table this one's rows were taken from, when it is a projection or a row selection
    /// of another table ([`Self::project`], [`Self::select_rows`]); `None` for a table that
    /// was not derived from another.
    #[must_use]
    pub fn row_origin(&self) -> Option<&RowOrigin> {
        self.row_origin.as_deref()
    }

    /// The origin of this table's rows: the recorded origin, or every row of this table.
    pub(crate) fn origin_of_rows(&self) -> Arc<RowOrigin> {
        self.row_origin
            .clone()
            .unwrap_or_else(|| Arc::new(RowOrigin::whole(self.storage.rows_digest())))
    }

    /// `derived`, whose rows are this table's rows in order, remembering their origin.
    pub(crate) fn with_rows_of(&self, mut derived: Self) -> Self {
        derived.row_origin = Some(self.origin_of_rows());
        derived
    }

    /// The rows `rows` of this table, in that order (a row may repeat), with every column, the
    /// analysis mask and the weights gathered through them. The result remembers its origin
    /// ([`Self::row_origin`]): the index of each row in the first table that was not itself
    /// derived from another. The declared unit key is dropped.
    ///
    /// # Errors
    ///
    /// An empty selection, a row index past the table, or gathered weights that are not a
    /// valid weighting (no positive total over the analysis rows).
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self, DataError> {
        if rows.is_empty() {
            return Err(DataError::EmptySelection { context: "row selection: no rows requested" });
        }
        let storage = &self.storage;
        if rows.iter().any(|&row| row >= storage.row_count()) {
            return Err(DataError::invalid_value("row selection", "exceeds the table"));
        }
        let origin = self.origin_of_rows().select(rows)?;
        let columns = (0..storage.columns().len())
            .map(|index| {
                let id = VariableId::from_raw(
                    u32::try_from(index).map_err(|_| DataError::overflow("column index"))?,
                );
                storage.column(id)?.gather_rows(rows)
            })
            .collect::<Result<Vec<_>, DataError>>()?;
        let mask = storage.analysis_mask().map(|mask| mask.gather_rows(rows)).transpose()?;
        let weights = storage
            .weights()
            .map(|weights| rows.iter().map(|&row| weights[row]).collect::<Arc<[f64]>>());
        let derived = storage.derive(storage.schema().clone(), columns, mask, weights)?;
        Ok(Self { storage: derived, unit_key: None, row_origin: Some(Arc::new(origin)) })
    }

    /// This table holding `lease` for as long as it, its clones and the tables derived from
    /// its columns live, whatever the column types (see
    /// [`OwnedColumnarStorage::with_retention`]). Keeps the declared unit key.
    #[must_use]
    pub fn with_retention(
        mut self,
        lease: Arc<antecedent_core::execution::AdmissionLease>,
    ) -> Self {
        self.storage = self.storage.with_retention(lease);
        self
    }

    /// This table with the declared unit of each row.
    ///
    /// A study that bound its units refuses a replacement table whose key differs, even at the
    /// same row count; a table without a key inherits the study's binding. Every derived table
    /// (a projection, a schema change) drops the key.
    ///
    /// # Errors
    ///
    /// [`DataError::InvalidValue`] when the key labels another number of rows.
    pub fn with_unit_key(mut self, key: crate::unit_key::UnitKey) -> Result<Self, DataError> {
        if key.rows() != self.storage.row_count() {
            return Err(DataError::invalid_value(
                "unit key rows",
                "must equal the row count of the table",
            ));
        }
        self.unit_key = Some(Arc::new(key));
        Ok(self)
    }

    /// The declared unit of each row, when one was attached.
    #[must_use]
    pub fn unit_key(&self) -> Option<&crate::unit_key::UnitKey> {
        self.unit_key.as_deref()
    }

    /// Build continuous `f64` columns from named slices (equal length).
    ///
    /// Schema variables are continuous with empty role hints, in iterator order.
    /// Dense [`VariableId`]s align with column order (`0..n`). A `NaN` value is a
    /// missing cell: its row is marked invalid, exactly as a null would be.
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
            return Err(DataError::empty("from_f64_columns columns"));
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
            .map_err(DataError::from)?;
        }
        let schema = b.build().map_err(DataError::from)?;
        Self::try_from_schema_f64(
            schema,
            collected.iter().map(|(name, values)| (name.as_ref(), *values)),
        )
    }

    /// Bind named `f64` slices to an existing schema (names must match; order may differ).
    ///
    /// `NaN` values are missing cells (their rows are marked invalid).
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
            return Err(DataError::LengthMismatch {
                expected: n_vars,
                actual: by_name.len(),
                context: "try_from_schema_f64 column count",
            });
        }
        let mut row_count = None;
        let mut owned = Vec::with_capacity(n_vars);
        for var in schema.variables() {
            let Some(values) = by_name.remove(var.name.as_ref()) else {
                return Err(DataError::absent("column for schema variable", var.name.to_string()));
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
                Arc::<[f64]>::from(values),
                ValidityBitmap::all_valid(n),
            )?;
            owned.push(OwnedColumn::Float64(col));
        }
        if !by_name.is_empty() {
            let extra: Vec<_> = by_name.keys().copied().collect();
            return Err(DataError::absent("schema variable for column", format!("{extra:?}")));
        }
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None)?;
        Ok(Self::new(storage))
    }

    /// Borrow underlying storage.
    #[must_use]
    pub fn storage(&self) -> &OwnedColumnarStorage {
        &self.storage
    }

    /// This table as a unified source snapshot.
    #[must_use]
    pub fn source(&self) -> crate::source::OwnedSource {
        crate::source::OwnedSource::from_tabular(self)
    }

    /// The same columns, mask and weights under another schema (a rename, a role or a unit
    /// change). The result is a different source: the snapshot binds the schema.
    ///
    /// # Errors
    ///
    /// A schema whose variable count is not the column count.
    pub fn with_schema(&self, schema: CausalSchema) -> Result<Self, DataError> {
        let renamed = Self::new(self.storage.with_schema(schema)?);
        Ok(match &self.row_origin {
            Some(origin) => Self { row_origin: Some(Arc::clone(origin)), ..renamed },
            None => renamed,
        })
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
    /// `NaN` values are missing cells (their rows are marked invalid).
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

    /// This series as a unified source snapshot under `modality`; the series time index is
    /// bound into its contract.
    ///
    /// # Errors
    ///
    /// `modality` is [`crate::source::SourceModality::Tabular`].
    pub fn source(
        &self,
        modality: crate::source::SourceModality,
    ) -> Result<crate::source::OwnedSource, DataError> {
        crate::source::OwnedSource::from_time_series(self, modality)
    }

    /// The same observations under another time index (a different cadence or regularity).
    /// The storage digest is unchanged and the source contract is not: the snapshot binds
    /// the time index, so lazy views and incremental digests of the earlier index no longer
    /// match.
    ///
    /// # Errors
    ///
    /// The index length is not the row count.
    pub fn with_time_index(&self, time_index: TimeIndex) -> Result<Self, DataError> {
        Self::try_new(self.storage.clone(), time_index)
    }

    /// The same observations under another schema (a rename, a role or a unit change),
    /// keeping the time index.
    ///
    /// # Errors
    ///
    /// A schema whose variable count is not the column count.
    pub fn with_schema(&self, schema: CausalSchema) -> Result<Self, DataError> {
        Self::try_new(self.storage.with_schema(schema)?, self.time_index.clone())
    }

    /// Replace one float64 column; preserve time index, other columns, mask, and weights.
    ///
    /// # Errors
    ///
    /// Unknown id, length mismatch, or non-float target.
    pub fn with_replaced_float(
        &self,
        id: VariableId,
        values: Arc<[f64]>,
    ) -> Result<Self, DataError> {
        let replaced = TabularData::new(self.storage.clone()).with_replaced_float(id, values)?;
        Self::try_new(replaced.storage().clone(), self.time_index.clone())
    }

    /// Attach observation weights (one per time step), replacing any existing ones; preserves
    /// columns, validity, the analysis mask and the time index.
    ///
    /// # Errors
    ///
    /// A weight count that is not the series length, or weights that are not finite and
    /// non-negative with positive total over the analysis rows.
    pub fn with_weights(&self, weights: Arc<[f64]>) -> Result<Self, DataError> {
        let storage = &self.storage;
        if weights.len() != storage.row_count() {
            return Err(DataError::LengthMismatch {
                expected: storage.row_count(),
                actual: weights.len(),
                context: "observation weights",
            });
        }
        let new_storage = storage.derive(
            storage.schema().clone(),
            storage.columns().to_vec(),
            storage.analysis_mask().cloned(),
            Some(weights),
        )?;
        Self::try_new(new_storage, self.time_index.clone())
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
        let new_storage = storage.derive(
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
