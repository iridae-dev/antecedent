//! Arrow-backed adapters. Arrow types stay inside this module (ADR 0004).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroU32;
use std::sync::Arc;

use antecedent_core::{
    CategoryDomainId, CausalSchema, CausalSchemaBuilder, DiagnosticSet, MeasurementSpec, RoleHint,
    ScalarType, SmallRoleSet, ValueType, VariableId,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowDictionaryKeyType, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float64Array, LargeStringArray,
    PrimitiveArray, RecordBatch, StringArray, StringViewArray,
};
use arrow_schema::{DataType, TimeUnit};

use crate::arrow_ffi::{
    ArrowCColumn, ArrowCStream, float64_column_from_array, validity_from_array,
};
use crate::buffer::F64Buffer;
use crate::categorical::Contrast;
use crate::categorical::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
};
use crate::column::{
    FixedVectorColumn, Float64Column, Int64Column, OwnedColumn, TimestampColumn, ValidityBitmap,
};
use crate::dataset::TabularData;
use crate::error::DataError;
use crate::materialize::{
    EstimatorStreamNeed, MaterializationReason, PreparationControls, PreparationProduct,
    SchemaColumn, SchemaConversion, SchemaValueKind, StreamingScan, finish_scan,
    materialization_diagnostic, prepare_table,
};
use crate::missing::{ChunkMissing, MissingConversion, dropped_row_digest};
use crate::sample_policy::MissingPolicy;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

mod chunks;
mod convert;
mod schema;
#[cfg(test)]
mod tests;

pub(crate) use self::convert::finish_load;
#[cfg(any(feature = "parquet", feature = "ipc"))]
pub(crate) use self::convert::level_text;
pub use self::convert::tabular_from_arrow_c_columns;
pub use self::convert::tabular_from_record_batch;

use self::chunks::{
    align_chunks, checked_row_total, concat_chunks, empty_column, owned_payload_bytes,
    remember_chunk_count, same_column_kind, variable_id,
};
use self::convert::column_from_array;
use self::schema::{
    detach_foreign, frame_payload_bytes, materialize_frame, resolve_order, scan_frame,
};

/// Result of loading Arrow input into library-owned storage.
#[derive(Clone, Debug)]
pub struct ArrowLoadResult {
    /// Loaded tabular data.
    pub data: TabularData,
    /// Copy / materialization diagnostics.
    pub diagnostics: DiagnosticSet,
    /// Total bytes copied into owned buffers.
    pub bytes_copied: u64,
    /// Total bytes borrowed zero-copy from foreign buffers.
    pub bytes_borrowed: u64,
}

/// One logical column kept as separate chunks.
#[derive(Clone, Debug)]
struct FrameColumn {
    name: String,
    chunks: Vec<OwnedColumn>,
}

/// Chunk-preserving Arrow frame.
///
/// A stream import does not concatenate batches. [`Self::contiguous`] copies
/// when a consumer needs one buffer per column. [`Self::freeze`] copies any
/// borrowed float64 buffer so the resulting study does not share the exporter's
/// memory.
#[derive(Clone, Debug)]
pub struct ArrowFrameView {
    columns: Vec<FrameColumn>,
    diagnostics: DiagnosticSet,
    bytes_copied: u64,
    bytes_borrowed: u64,
}

/// A study whose columns are owned.
///
/// The bytes were copied at freeze. A later write through an exporter alias is
/// not a write to this study.
#[derive(Clone, Debug)]
pub struct FrozenArrowStudy {
    data: TabularData,
}

impl FrozenArrowStudy {
    /// Borrow the frozen table.
    #[must_use]
    pub fn data(&self) -> &TabularData {
        &self.data
    }

    /// Row count of the frozen table.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.data.row_count()
    }

    /// Refuse to bind `alias` onto the frozen columns.
    ///
    /// The alias is released. The study's columns stay the bytes captured at
    /// freeze.
    ///
    /// # Errors
    ///
    /// Always. A frozen study refuses a mutable foreign alias.
    pub fn bind_foreign_alias(&self, alias: ArrowCColumn) -> Result<(), DataError> {
        drop(alias);
        Err(DataError::unsupported(
            "freezing a mutable foreign alias",
            "its columns were copied at freeze and a later write through the exporter is not the study",
        ))
    }
}

impl ArrowFrameView {
    /// Import CDI columns without concatenating chunks.
    ///
    /// # Errors
    ///
    /// Empty input, a CDI failure, mismatched chunk lengths, or an unsupported
    /// or inexact type.
    pub fn from_c_columns(columns: Vec<ArrowCColumn>) -> Result<Self, DataError> {
        if columns.is_empty() {
            return Err(DataError::empty("Arrow CDI import columns"));
        }
        let mut frame_columns = Vec::with_capacity(columns.len());
        let mut diagnostics = DiagnosticSet::new();
        let mut bytes_copied = 0u64;
        let mut bytes_borrowed = 0u64;
        let mut n_chunks = None;
        for (index, column) in columns.into_iter().enumerate() {
            let id = variable_id(index)?;
            let domain_id = CategoryDomainId::from_raw(id.raw());
            let name = column.name.clone();
            let arrays = column.into_chunk_arrays()?;
            let mut chunks = Vec::with_capacity(arrays.len());
            for array in arrays {
                let (owned, borrowed, copied, diag) = column_from_array(id, domain_id, array)?;
                bytes_borrowed += borrowed;
                bytes_copied += copied;
                if let Some(diag) = diag {
                    diagnostics.push(diag);
                }
                chunks.push(owned);
            }
            remember_chunk_count(&mut n_chunks, chunks.len())?;
            frame_columns.push(FrameColumn { name, chunks });
        }
        align_chunks(&frame_columns)?;
        Ok(Self { columns: frame_columns, diagnostics, bytes_copied, bytes_borrowed })
    }

    /// Import a C stream. Each record batch stays one chunk.
    ///
    /// # Errors
    ///
    /// When the stream cannot be read, a batch disagrees with the stream schema, or a column
    /// type cannot be kept exact.
    pub fn from_stream(stream: ArrowCStream) -> Result<Self, DataError> {
        let (schema, batches) = stream.into_batches()?;
        Self::from_schema_batches(&schema, batches)
    }

    /// [`Self::from_stream`] bounded by the bytes the stream's batches retain.
    ///
    /// # Errors
    ///
    /// As [`Self::from_stream`], or when the batches read so far retain more than `max_bytes`.
    pub fn from_stream_within(stream: ArrowCStream, max_bytes: u64) -> Result<Self, DataError> {
        let (schema, batches) = stream.into_batches_within(max_bytes)?;
        Self::from_schema_batches(&schema, batches)
    }

    /// Import record batches without concatenating them.
    ///
    /// # Errors
    ///
    /// Empty input, mismatched schemas, or an unsupported column type.
    pub fn from_record_batches(batches: Vec<RecordBatch>) -> Result<Self, DataError> {
        let Some(first) = batches.first() else {
            return Err(DataError::empty("Arrow record-batch import batches"));
        };
        let schema = first.schema();
        Self::from_schema_batches(&schema, batches)
    }

    /// Column names in schema order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|column| column.name.as_str())
    }

    /// Number of columns.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Number of chunks. Every column has this many chunks.
    #[must_use]
    pub fn n_chunks(&self) -> usize {
        self.columns.first().map_or(0, |column| column.chunks.len())
    }

    /// Borrow one column's chunks. The slices are not concatenated.
    ///
    /// # Errors
    ///
    /// When `index` is past the last column.
    pub fn column_chunks(&self, index: usize) -> Result<&[OwnedColumn], DataError> {
        self.columns
            .get(index)
            .map(|column| column.chunks.as_slice())
            .ok_or(DataError::out_of_range("Arrow frame column", index, self.columns.len()))
    }

    /// Logical row count, the sum of chunk lengths.
    ///
    /// The sum uses `usize` and is not capped at `u32::MAX`.
    ///
    /// # Errors
    ///
    /// When the sum overflows `usize`.
    pub fn row_count(&self) -> Result<usize, DataError> {
        let Some(column) = self.columns.first() else {
            return Ok(0);
        };
        checked_row_total(column.chunks.iter().map(OwnedColumn::len))
    }

    /// Copy chunks into one buffer per column.
    ///
    /// A frame that is already one chunk keeps a borrowed float64 buffer.
    ///
    /// # Errors
    ///
    /// When chunk types or category domains disagree, or the row sum overflows.
    pub fn contiguous(&self) -> Result<ArrowLoadResult, DataError> {
        if self.n_chunks() <= 1 {
            return self.load_single_chunk();
        }
        let mut columns = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            columns.push(concat_chunks(&column.chunks)?);
        }
        let mut bytes_copied = 0u64;
        for column in &columns {
            bytes_copied += owned_payload_bytes(column)?;
        }
        let mut diagnostics = self.diagnostics.clone();
        diagnostics.push(materialization_diagnostic(
            MaterializationReason::ForeignBufferIncompatible,
            bytes_copied,
        ));
        finish_load(&self.names_owned(), columns, diagnostics, bytes_copied, 0)
    }

    /// Copy the frame into a study that does not share exporter memory.
    ///
    /// # Errors
    ///
    /// When [`Self::contiguous`] fails.
    pub fn freeze(&self) -> Result<FrozenArrowStudy, DataError> {
        let loaded = self.contiguous()?;
        let columns = loaded
            .data
            .storage()
            .columns()
            .iter()
            .map(detach_foreign)
            .collect::<Result<Vec<_>, _>>()?;
        let schema = loaded.data.schema().clone();
        let storage = OwnedColumnarStorage::try_new(schema, columns, None, None)?;
        Ok(FrozenArrowStudy { data: TabularData::new(storage) })
    }

    fn load_single_chunk(&self) -> Result<ArrowLoadResult, DataError> {
        let mut columns = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            match column.chunks.as_slice() {
                [one] => columns.push(one.clone()),
                [] => {
                    return Err(DataError::empty("Arrow column chunks to load"));
                }
                _ => {
                    return Err(DataError::precondition(
                        "loading an Arrow column",
                        "a single chunk; concatenate the chunks explicitly",
                    ));
                }
            }
        }
        finish_load(
            &self.names_owned(),
            columns,
            self.diagnostics.clone(),
            self.bytes_copied,
            self.bytes_borrowed,
        )
    }

    fn names_owned(&self) -> Vec<String> {
        self.columns.iter().map(|column| column.name.clone()).collect()
    }

    fn from_schema_batches(
        schema: &arrow_schema::Schema,
        batches: Vec<RecordBatch>,
    ) -> Result<Self, DataError> {
        if schema.fields().is_empty() {
            return Err(DataError::empty("Arrow stream columns"));
        }
        let mut frame_columns = Vec::with_capacity(schema.fields().len());
        for (index, field) in schema.fields().iter().enumerate() {
            let _ = variable_id(index)?;
            frame_columns.push(FrameColumn { name: field.name().clone(), chunks: Vec::new() });
        }
        let mut diagnostics = DiagnosticSet::new();
        let mut bytes_copied = 0u64;
        let mut bytes_borrowed = 0u64;
        for batch in batches {
            if batch.num_columns() != frame_columns.len() {
                return Err(DataError::LengthMismatch {
                    expected: frame_columns.len(),
                    actual: batch.num_columns(),
                    context: "Arrow stream batch width",
                });
            }
            for (index, field) in schema.fields().iter().enumerate() {
                let batch_schema = batch.schema();
                let batch_field = batch_schema.field(index);
                if batch_field.name() != field.name()
                    || batch_field.data_type() != field.data_type()
                {
                    return Err(DataError::mismatch(
                        "Arrow stream batch column and the stream schema",
                    ));
                }
                let id = variable_id(index)?;
                let (owned, borrowed, copied, diag) = column_from_array(
                    id,
                    CategoryDomainId::from_raw(id.raw()),
                    Arc::clone(batch.column(index)),
                )?;
                bytes_borrowed += borrowed;
                bytes_copied += copied;
                if let Some(diag) = diag {
                    diagnostics.push(diag);
                }
                if let Some(previous) = frame_columns[index].chunks.last() {
                    same_column_kind(previous, &owned)?;
                }
                frame_columns[index].chunks.push(owned);
            }
        }
        if frame_columns.iter().all(|column| column.chunks.is_empty()) {
            for (index, field) in schema.fields().iter().enumerate() {
                let id = variable_id(index)?;
                frame_columns[index].chunks.push(empty_column(
                    id,
                    CategoryDomainId::from_raw(id.raw()),
                    field.data_type(),
                )?);
            }
        }
        align_chunks(&frame_columns)?;
        Ok(Self { columns: frame_columns, diagnostics, bytes_copied, bytes_borrowed })
    }
}

/// What an estimator receives from a chunked Arrow frame.
#[derive(Clone, Debug)]
pub enum EstimatorFrame {
    /// One finite table. The estimator does not see the chunks.
    Finite {
        /// Materialized table.
        table: TabularData,
        /// Schema, category, missing-value, and admission receipt.
        product: PreparationProduct,
        /// Copy diagnostics, including an explicit materialization when chunks were joined.
        diagnostics: DiagnosticSet,
    },
    /// A scan. Its receipt is not a bounded end-to-end route.
    Scan(StreamingScan),
}

impl ArrowFrameView {
    /// Deliver this frame to an estimator.
    ///
    /// `order` is column indexes. Empty means schema order. An estimator that
    /// cannot stream either receives one table or a refusal. The refusal happens
    /// before chunks are concatenated. An estimator that can stream receives the
    /// chunks; that scan is not a bounded end-to-end route.
    ///
    /// # Errors
    ///
    /// The backend, order, schema, category, missing-value policy, or budget
    /// refuses the plan, or concatenation fails after admission.
    pub fn open_for_estimator(
        &self,
        order: &[usize],
        missing: MissingPolicy,
        categories: &[(VariableId, Contrast)],
        controls: PreparationControls,
    ) -> Result<EstimatorFrame, DataError> {
        controls.backend.admit().map_err(|_| {
            DataError::unsupported(
                "estimator preparation on the lazy backend",
                "the lazy backend is not adopted; prepare through the direct_arrow path",
            )
        })?;
        let order = resolve_order(self.n_columns(), order)?;
        let bytes = frame_payload_bytes(self)?;
        let budget = controls.budget.map_or(bytes, crate::materialize::PreparationBudget::bytes);
        let noun = match controls.stream {
            EstimatorStreamNeed::CanStream => "streaming scratch",
            EstimatorStreamNeed::CannotStream => "materialization",
        };
        if bytes > budget {
            return Err(DataError::budget_refused(noun, bytes, Some(budget)));
        }
        match controls.stream {
            EstimatorStreamNeed::CanStream => Ok(EstimatorFrame::Scan(scan_frame(
                self, &order, missing, categories, bytes, budget,
            )?)),
            EstimatorStreamNeed::CannotStream => {
                materialize_frame(self, &order, missing, categories, controls, budget)
            }
        }
    }
}
