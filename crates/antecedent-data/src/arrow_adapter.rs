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
use crate::categorical::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
};
use crate::column::{
    FixedVectorColumn, Float64Column, Int64Column, OwnedColumn, TimestampColumn, ValidityBitmap,
};
use crate::dataset::TabularData;
use crate::categorical::Contrast;
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

/// Load float64 and fixed-size float64-list columns from an Arrow record batch.
///
/// Always copies into library-owned buffers (in-process `RecordBatch` path).
/// Prefer [`tabular_from_arrow_c_columns`] for Arrow C Data Interface zero-copy.
/// Null cells are invalid and hold `NaN`; a fixed-size-list row with a null
/// component is an invalid row.
///
/// # Errors
///
/// Unsupported column types, empty batches, schema construction failures, or more
/// than `u32::MAX` columns.
pub fn tabular_from_record_batch(batch: &RecordBatch) -> Result<ArrowLoadResult, DataError> {
    if batch.num_columns() == 0 {
        return Err(DataError::InvalidArgument { message: "record batch has no columns".into() });
    }
    let mut builder = CausalSchemaBuilder::new();
    let mut columns = Vec::with_capacity(batch.num_columns());
    let mut diagnostics = DiagnosticSet::new();
    let mut bytes_copied = 0u64;
    let n_rows = batch.num_rows();

    for (i, field) in batch.schema().fields().iter().enumerate() {
        push_record_column(
            &mut RecordImport {
                builder: &mut builder,
                columns: &mut columns,
                diagnostics: &mut diagnostics,
                bytes_copied: &mut bytes_copied,
            },
            i,
            field.name(),
            batch.column(i),
            n_rows,
        )?;
    }

    let schema: CausalSchema = builder.build().map_err(DataError::from)?;
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None)?;
    Ok(ArrowLoadResult {
        data: TabularData::new(storage),
        diagnostics,
        bytes_copied,
        bytes_borrowed: 0,
    })
}

struct RecordImport<'a> {
    builder: &'a mut CausalSchemaBuilder,
    columns: &'a mut Vec<OwnedColumn>,
    diagnostics: &'a mut DiagnosticSet,
    bytes_copied: &'a mut u64,
}

fn push_record_column(
    import: &mut RecordImport<'_>,
    index: usize,
    name: &str,
    array: &ArrayRef,
    n_rows: usize,
) -> Result<(), DataError> {
    let id = VariableId::from_raw(u32::try_from(index).map_err(|_| {
        DataError::InvalidArgument { message: "too many Arrow columns for VariableId".into() }
    })?);
    if let Some(floats) = array.as_any().downcast_ref::<Float64Array>() {
        import
            .builder
            .add_variable(
                Arc::<str>::from(name),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(DataError::from)?;
        let (col, copied) = float64_owned_from_array(id, floats, n_rows)?;
        *import.bytes_copied += copied;
        import.diagnostics.push(materialization_diagnostic(
            MaterializationReason::ForeignBufferIncompatible,
            copied,
        ));
        import.columns.push(OwnedColumn::Float64(col));
        return Ok(());
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let dim = usize::try_from(list.value_length()).map_err(|_| DataError::InvalidArgument {
            message: "FixedSizeList width must fit usize".into(),
        })?;
        if dim == 0 {
            return Err(DataError::InvalidArgument {
                message: "FixedSizeList width must be > 0".into(),
            });
        }
        let width = NonZeroU32::new(u32::try_from(dim).map_err(|_| {
            DataError::InvalidArgument { message: "FixedSizeList width must fit u32".into() }
        })?)
        .ok_or(DataError::InvalidArgument { message: "FixedSizeList width must be > 0".into() })?;
        import
            .builder
            .add_variable(
                Arc::<str>::from(name),
                ValueType::Vector { width, element: ScalarType::Float64 },
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(DataError::from)?;
        let (col, copied) = fixed_vector_from_list(id, list, n_rows, dim)?;
        *import.bytes_copied += copied;
        import.diagnostics.push(materialization_diagnostic(
            MaterializationReason::ForeignBufferIncompatible,
            copied,
        ));
        import.columns.push(OwnedColumn::FixedVector(col));
        return Ok(());
    }
    let domain_id = CategoryDomainId::from_raw(id.raw());
    let Some(col) = typed_owned_column(id, domain_id, array)? else {
        return Err(DataError::TypeMismatch {
            id,
            expected: "float64, int, timestamp, dictionary, or FixedSizeList<float64>",
        });
    };
    let (value_type, roles, domain) = schema_binding(&col)?;
    import
        .builder
        .add_variable(
            Arc::<str>::from(name),
            value_type,
            roles,
            None,
            domain,
            MeasurementSpec::default(),
        )
        .map_err(DataError::from)?;
    let copied = owned_payload_bytes(&col)?;
    *import.bytes_copied += copied;
    import
        .diagnostics
        .push(materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied));
    import.columns.push(col);
    Ok(())
}

fn float64_owned_from_array(
    id: VariableId,
    floats: &Float64Array,
    n_rows: usize,
) -> Result<(Float64Column, u64), DataError> {
    // One bulk copy of the value buffer straight into the shared slice; only a column that
    // actually has nulls pays a per-row pass, to write NaN under each missing cell (so
    // value-only readers never see an observation) and to build the bitmap.
    let mut validity_bytes = vec![0u8; n_rows.div_ceil(8)];
    let values: Arc<[f64]> = if floats.null_count() == 0 {
        let (full, tail) = (n_rows / 8, n_rows % 8);
        validity_bytes[..full].fill(0xFF);
        if tail > 0 {
            validity_bytes[full] = (1u8 << tail) - 1;
        }
        Arc::from(&floats.values()[..])
    } else {
        let mut patched = floats.values().to_vec();
        for row in 0..n_rows {
            if floats.is_null(row) {
                patched[row] = f64::NAN;
            } else {
                validity_bytes[row / 8] |= 1 << (row % 8);
            }
        }
        Arc::from(patched)
    };
    let copied = (values.len() * core::mem::size_of::<f64>() + validity_bytes.len()) as u64;
    let col = Float64Column::new(
        id,
        F64Buffer::owned(values),
        ValidityBitmap::from_bytes(validity_bytes, n_rows)?,
    )?;
    Ok((col, copied))
}

fn fixed_vector_from_list(
    id: VariableId,
    list: &FixedSizeListArray,
    n_rows: usize,
    dim: usize,
) -> Result<(FixedVectorColumn, u64), DataError> {
    let values = list.values();
    let floats = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(DataError::TypeMismatch { id, expected: "FixedSizeList<float64>" })?;
    // Validity is per row, so a vector with any null component is a missing row;
    // every component of a missing row holds NaN.
    let mut flat = Vec::with_capacity(n_rows.saturating_mul(dim));
    let mut validity_bytes = vec![0u8; n_rows.div_ceil(8)];
    for row in 0..n_rows {
        let start = row.saturating_mul(dim);
        if list.is_null(row) || (start..start + dim).any(|idx| floats.is_null(idx)) {
            flat.extend(std::iter::repeat_n(f64::NAN, dim));
            continue;
        }
        validity_bytes[row / 8] |= 1 << (row % 8);
        // `value` is logical; the buffer is addressed from the array offset.
        let raw = floats.values();
        let phys = floats.offset() + start;
        let end = phys + dim;
        if end > raw.len() {
            return Err(DataError::InvalidArgument {
                message: "FixedSizeList value buffer is shorter than its row width".into(),
            });
        }
        flat.extend_from_slice(&raw[phys..end]);
    }
    let copied = (flat.len() * core::mem::size_of::<f64>() + validity_bytes.len()) as u64;
    let col = FixedVectorColumn::new(
        id,
        dim,
        Arc::from(flat),
        ValidityBitmap::from_bytes(validity_bytes, n_rows)?,
    )?;
    Ok((col, copied))
}

/// Load Arrow C Data Interface columns, preferring zero-copy for float64.
///
/// Consumes each [`ArrowCColumn`]'s FFI structs. One chunk of a contiguous
/// float64 value buffer is borrowed; validity bitmaps are copied. A float64
/// chunk with nulls is borrowed only when every null slot already holds `NaN`;
/// otherwise its values are copied with `NaN` written under each null.
/// Integer ids, timestamps, and dictionary categories are stored as
/// [`OwnedColumn::Int64`], [`OwnedColumn::Timestamp`], and
/// [`OwnedColumn::Categorical`]. Several chunks are concatenated because the
/// returned table is one buffer per column; use [`ArrowFrameView`] to keep them.
/// Row count is a `usize` sum of the chunks. It is not capped at `u32::MAX`.
///
/// # Errors
///
/// Empty input, an unsupported type, a CDI or schema failure, a category or
/// timestamp that cannot be kept exact, or more than `u32::MAX` columns.
pub fn tabular_from_arrow_c_columns(
    columns: Vec<ArrowCColumn>,
) -> Result<ArrowLoadResult, DataError> {
    let frame = ArrowFrameView::from_c_columns(columns)?;
    if frame.n_chunks() <= 1 { frame.load_single_chunk() } else { frame.contiguous() }
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
        Err(DataError::InvalidArgument {
            message: "a frozen study refuses a mutable foreign alias; its columns were copied at \
                      freeze and a later write through the exporter is not the study"
                .into(),
        })
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
            return Err(DataError::InvalidArgument {
                message: "Arrow CDI import needs ≥1 column".into(),
            });
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
    /// When the stream cannot be read, a batch disagrees with the stream
    /// schema, or a column type cannot be kept exact.
    pub fn from_stream(stream: ArrowCStream) -> Result<Self, DataError> {
        let (schema, batches) = stream.into_batches()?;
        Self::from_schema_batches(&schema, batches)
    }

    /// Import record batches without concatenating them.
    ///
    /// # Errors
    ///
    /// Empty input, mismatched schemas, or an unsupported column type.
    pub fn from_record_batches(batches: Vec<RecordBatch>) -> Result<Self, DataError> {
        let Some(first) = batches.first() else {
            return Err(DataError::InvalidArgument {
                message: "Arrow record-batch import needs ≥1 batch".into(),
            });
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
        self.columns.get(index).map(|column| column.chunks.as_slice()).ok_or(
            DataError::InvalidArgument { message: format!("Arrow frame has no column {index}") },
        )
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
                    return Err(DataError::InvalidArgument {
                        message: "Arrow column has no chunks to load".into(),
                    });
                }
                _ => {
                    return Err(DataError::InvalidArgument {
                        message: "Arrow column has several chunks; concatenate them explicitly"
                            .into(),
                    });
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
            return Err(DataError::InvalidArgument {
                message: "Arrow stream has no columns".into(),
            });
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
                    return Err(DataError::InvalidArgument {
                        message: format!(
                            "Arrow stream batch column {} disagrees with the stream schema",
                            field.name()
                        ),
                    });
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

/// Sum chunk lengths. A total above `u32::MAX` is a valid `usize` on a 64-bit
/// target; this function does not reject it.
fn checked_row_total(chunk_rows: impl IntoIterator<Item = usize>) -> Result<usize, DataError> {
    let mut total = 0usize;
    for rows in chunk_rows {
        total = total.checked_add(rows).ok_or_else(|| DataError::InvalidArgument {
            message: "Arrow chunk row counts overflow usize".into(),
        })?;
    }
    Ok(total)
}

fn remember_chunk_count(seen: &mut Option<usize>, count: usize) -> Result<(), DataError> {
    match *seen {
        None => *seen = Some(count),
        Some(expected) if expected == count => {}
        Some(expected) => {
            return Err(DataError::LengthMismatch {
                expected,
                actual: count,
                context: "Arrow CDI chunk counts",
            });
        }
    }
    Ok(())
}

fn align_chunks(columns: &[FrameColumn]) -> Result<(), DataError> {
    let Some(first) = columns.first() else {
        return Ok(());
    };
    for column in columns.iter().skip(1) {
        if column.chunks.len() != first.chunks.len() {
            return Err(DataError::LengthMismatch {
                expected: first.chunks.len(),
                actual: column.chunks.len(),
                context: "Arrow CDI chunk counts",
            });
        }
        for (index, chunk) in column.chunks.iter().enumerate() {
            let expected = first.chunks[index].len();
            if chunk.len() != expected {
                return Err(DataError::LengthMismatch {
                    expected,
                    actual: chunk.len(),
                    context: "Arrow CDI column lengths",
                });
            }
        }
    }
    Ok(())
}

fn variable_id(index: usize) -> Result<VariableId, DataError> {
    let raw = u32::try_from(index).map_err(|_| DataError::InvalidArgument {
        message: "too many Arrow columns for VariableId".into(),
    })?;
    Ok(VariableId::from_raw(raw))
}

fn finish_load(
    names: &[String],
    columns: Vec<OwnedColumn>,
    diagnostics: DiagnosticSet,
    bytes_copied: u64,
    bytes_borrowed: u64,
) -> Result<ArrowLoadResult, DataError> {
    let mut builder = CausalSchemaBuilder::new();
    for (name, column) in names.iter().zip(columns.iter()) {
        let (value_type, roles, domain) = schema_binding(column)?;
        builder
            .add_variable(
                Arc::<str>::from(name.as_str()),
                value_type,
                roles,
                None,
                domain,
                MeasurementSpec::default(),
            )
            .map_err(DataError::from)?;
    }
    let schema = builder.build().map_err(DataError::from)?;
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None)?;
    Ok(ArrowLoadResult {
        data: TabularData::new(storage),
        diagnostics,
        bytes_copied,
        bytes_borrowed,
    })
}

fn column_from_array(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: ArrayRef,
) -> Result<(OwnedColumn, u64, u64, Option<antecedent_core::Diagnostic>), DataError> {
    if array.data_type() == &DataType::Float64 {
        return float64_column_from_array(id, array);
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let dim = usize::try_from(list.value_length()).map_err(|_| DataError::InvalidArgument {
            message: "FixedSizeList width must fit usize".into(),
        })?;
        let (col, copied) = fixed_vector_from_list(id, list, list.len(), dim)?;
        let diag =
            materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied);
        return Ok((OwnedColumn::FixedVector(col), 0, copied, Some(diag)));
    }
    let Some(col) = typed_owned_column(id, domain_id, &array)? else {
        return Err(DataError::TypeMismatch {
            id,
            expected: "float64, int, timestamp, dictionary, or FixedSizeList<float64>",
        });
    };
    let copied = owned_payload_bytes(&col)?;
    let diag = materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied);
    Ok((col, 0, copied, Some(diag)))
}

fn typed_owned_column(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: &ArrayRef,
) -> Result<Option<OwnedColumn>, DataError> {
    let column = match array.data_type() {
        DataType::Int8 => int64_from_signed::<Int8Type>(id, array.as_primitive())?,
        DataType::Int16 => int64_from_signed::<Int16Type>(id, array.as_primitive())?,
        DataType::Int32 => int64_from_signed::<Int32Type>(id, array.as_primitive())?,
        DataType::Int64 => int64_from_signed::<Int64Type>(id, array.as_primitive())?,
        DataType::UInt8 => int64_from_unsigned::<UInt8Type>(id, array.as_primitive())?,
        DataType::UInt16 => int64_from_unsigned::<UInt16Type>(id, array.as_primitive())?,
        DataType::UInt32 => int64_from_unsigned::<UInt32Type>(id, array.as_primitive())?,
        DataType::UInt64 => int64_from_unsigned::<UInt64Type>(id, array.as_primitive())?,
        DataType::Timestamp(unit, _) => timestamp_column(id, array.as_ref(), *unit)?,
        DataType::Dictionary(_, _) => categorical_column(id, domain_id, array.as_ref())?,
        _ => return Ok(None),
    };
    Ok(Some(column))
}

fn int64_from_signed<T>(id: VariableId, array: &PrimitiveArray<T>) -> Result<OwnedColumn, DataError>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: Into<i64>,
{
    let (validity, _) = validity_from_array(array)?;
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        let value = if array.is_null(row) { 0 } else { array.value(row).into() };
        values.push(value);
    }
    Ok(OwnedColumn::Int64(Int64Column::new(id, values, validity)?))
}

fn int64_from_unsigned<T>(
    id: VariableId,
    array: &PrimitiveArray<T>,
) -> Result<OwnedColumn, DataError>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: Into<u64>,
{
    let (validity, _) = validity_from_array(array)?;
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(0);
            continue;
        }
        let raw = array.value(row).into();
        let value = i64::try_from(raw).map_err(|_| DataError::InvalidArgument {
            message: format!("unsigned id {raw} does not fit an exact int64"),
        })?;
        values.push(value);
    }
    Ok(OwnedColumn::Int64(Int64Column::new(id, values, validity)?))
}

fn timestamp_column(
    id: VariableId,
    array: &dyn Array,
    unit: TimeUnit,
) -> Result<OwnedColumn, DataError> {
    let (validity, _) = validity_from_array(array)?;
    let scale: i64 = match unit {
        TimeUnit::Nanosecond => 1,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Second => 1_000_000_000,
    };
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(0);
            continue;
        }
        let raw = timestamp_raw(array, unit, row);
        let nanos = raw.checked_mul(scale).ok_or_else(|| DataError::InvalidArgument {
            message: format!("timestamp {raw} in {unit:?} overflows exact nanoseconds"),
        })?;
        values.push(nanos);
    }
    Ok(OwnedColumn::Timestamp(TimestampColumn::new(id, values, validity)?))
}

fn timestamp_raw(array: &dyn Array, unit: TimeUnit, row: usize) -> i64 {
    match unit {
        TimeUnit::Nanosecond => array.as_primitive::<TimestampNanosecondType>().value(row),
        TimeUnit::Microsecond => array.as_primitive::<TimestampMicrosecondType>().value(row),
        TimeUnit::Millisecond => array.as_primitive::<TimestampMillisecondType>().value(row),
        TimeUnit::Second => array.as_primitive::<TimestampSecondType>().value(row),
    }
}

fn categorical_column(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: &dyn Array,
) -> Result<OwnedColumn, DataError> {
    let DataType::Dictionary(key, _) = array.data_type() else {
        return Err(DataError::TypeMismatch { id, expected: "dictionary" });
    };
    match key.as_ref() {
        DataType::Int8 => dict_from::<Int8Type>(id, domain_id, array.as_dictionary()),
        DataType::Int16 => dict_from::<Int16Type>(id, domain_id, array.as_dictionary()),
        DataType::Int32 => dict_from::<Int32Type>(id, domain_id, array.as_dictionary()),
        DataType::Int64 => dict_from::<Int64Type>(id, domain_id, array.as_dictionary()),
        DataType::UInt8 => dict_from::<UInt8Type>(id, domain_id, array.as_dictionary()),
        DataType::UInt16 => dict_from::<UInt16Type>(id, domain_id, array.as_dictionary()),
        DataType::UInt32 => dict_from::<UInt32Type>(id, domain_id, array.as_dictionary()),
        other => Err(DataError::InvalidArgument {
            message: format!("dictionary key type {other} is not an exact category code"),
        }),
    }
}

fn dict_from<K>(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: &DictionaryArray<K>,
) -> Result<OwnedColumn, DataError>
where
    K: ArrowDictionaryKeyType,
{
    let labels = dictionary_labels(array.values())?;
    let domain = Arc::new(CategoryDomain::try_new(
        domain_id,
        labels,
        array.is_ordered(),
        None,
        UnknownCategoryPolicy::Fail,
    )?);
    let (validity, _) = validity_from_array(array)?;
    let mut codes = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        let code = match array.key(row) {
            None => 0,
            Some(key) => u32::try_from(key).map_err(|_| DataError::InvalidArgument {
                message: format!("dictionary key {key} does not fit a category code"),
            })?,
        };
        codes.push(CategoryCode::from_raw(code));
    }
    Ok(OwnedColumn::Categorical(CategoricalColumn::try_new(id, codes, validity, domain)?))
}

fn dictionary_labels(values: &ArrayRef) -> Result<Vec<CategoryLevel>, DataError> {
    if values.is_empty() {
        return Err(DataError::InvalidArgument {
            message: "dictionary category domain has no levels".into(),
        });
    }
    let mut labels = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            return Err(DataError::InvalidArgument {
                message: format!("dictionary level {row} is null"),
            });
        }
        labels.push(CategoryLevel { label: Arc::<str>::from(level_text(values.as_ref(), row)?) });
    }
    Ok(labels)
}

fn level_text(values: &dyn Array, row: usize) -> Result<String, DataError> {
    if let Some(text) = values.as_any().downcast_ref::<StringArray>() {
        return Ok(text.value(row).to_owned());
    }
    if let Some(text) = values.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(text.value(row).to_owned());
    }
    if let Some(text) = values.as_any().downcast_ref::<StringViewArray>() {
        return Ok(text.value(row).to_owned());
    }
    let rendered = match values.data_type() {
        DataType::Int8 => values.as_primitive::<Int8Type>().value(row).to_string(),
        DataType::Int16 => values.as_primitive::<Int16Type>().value(row).to_string(),
        DataType::Int32 => values.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => values.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::UInt8 => values.as_primitive::<UInt8Type>().value(row).to_string(),
        DataType::UInt16 => values.as_primitive::<UInt16Type>().value(row).to_string(),
        DataType::UInt32 => values.as_primitive::<UInt32Type>().value(row).to_string(),
        DataType::UInt64 => values.as_primitive::<UInt64Type>().value(row).to_string(),
        other => {
            return Err(DataError::InvalidArgument {
                message: format!("dictionary values of type {other} are not category labels"),
            });
        }
    };
    Ok(rendered)
}

fn schema_binding(
    column: &OwnedColumn,
) -> Result<(ValueType, SmallRoleSet, Option<CategoryDomainId>), DataError> {
    Ok(match column {
        OwnedColumn::Float64(_) => {
            (ValueType::Continuous, SmallRoleSet::from_hint(RoleHint::Context), None)
        }
        OwnedColumn::Int64(_) => {
            (ValueType::Count, SmallRoleSet::from_hint(RoleHint::UnitId), None)
        }
        OwnedColumn::Timestamp(_) => {
            (ValueType::Count, SmallRoleSet::from_hint(RoleHint::Time), None)
        }
        OwnedColumn::Boolean(_) => {
            (ValueType::Binary, SmallRoleSet::from_hint(RoleHint::Context), None)
        }
        OwnedColumn::Categorical(column) => {
            let value_type =
                if column.domain.ordered { ValueType::Ordinal } else { ValueType::Categorical };
            (value_type, SmallRoleSet::from_hint(RoleHint::Context), Some(column.domain.id))
        }
        OwnedColumn::FixedVector(column) => {
            let width = NonZeroU32::new(u32::try_from(column.dim).map_err(|_| {
                DataError::InvalidArgument { message: "FixedSizeList width must fit u32".into() }
            })?)
            .ok_or_else(|| DataError::InvalidArgument {
                message: "FixedSizeList width must be > 0".into(),
            })?;
            (
                ValueType::Vector { width, element: ScalarType::Float64 },
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
            )
        }
    })
}

fn owned_payload_bytes(column: &OwnedColumn) -> Result<u64, DataError> {
    let (n_values, width) = match column {
        OwnedColumn::Float64(column) => (column.len(), core::mem::size_of::<f64>()),
        OwnedColumn::Int64(column) => (column.len(), core::mem::size_of::<i64>()),
        OwnedColumn::Timestamp(column) => (column.len(), core::mem::size_of::<i64>()),
        OwnedColumn::Boolean(column) => (column.len(), 1usize),
        OwnedColumn::Categorical(column) => (column.len(), core::mem::size_of::<CategoryCode>()),
        OwnedColumn::FixedVector(column) => (column.values.len(), core::mem::size_of::<f64>()),
    };
    let values = n_values.checked_mul(width).ok_or_else(|| DataError::InvalidArgument {
        message: "column payload size overflows usize".into(),
    })?;
    let validity = column.as_view().validity().len().div_ceil(8);
    let total = values.checked_add(validity).ok_or_else(|| DataError::InvalidArgument {
        message: "column payload size overflows usize".into(),
    })?;
    u64::try_from(total).map_err(|_| DataError::InvalidArgument {
        message: "column payload size does not fit u64".into(),
    })
}

fn empty_column(
    id: VariableId,
    domain_id: CategoryDomainId,
    data_type: &DataType,
) -> Result<OwnedColumn, DataError> {
    let empty: ArrayRef = arrow_array::new_empty_array(data_type);
    let (column, _, _, _) = column_from_array(id, domain_id, empty)?;
    Ok(column)
}

fn same_column_kind(previous: &OwnedColumn, next: &OwnedColumn) -> Result<(), DataError> {
    let same = match (previous, next) {
        (OwnedColumn::Float64(_), OwnedColumn::Float64(_))
        | (OwnedColumn::Int64(_), OwnedColumn::Int64(_))
        | (OwnedColumn::Timestamp(_), OwnedColumn::Timestamp(_))
        | (OwnedColumn::Boolean(_), OwnedColumn::Boolean(_)) => true,
        (OwnedColumn::FixedVector(left), OwnedColumn::FixedVector(right)) => left.dim == right.dim,
        (OwnedColumn::Categorical(left), OwnedColumn::Categorical(right)) => {
            left.domain == right.domain
        }
        _ => false,
    };
    if same {
        Ok(())
    } else {
        Err(DataError::InvalidArgument {
            message: "Arrow chunks of one column disagree in type or category domain".into(),
        })
    }
}

fn concat_chunks(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let Some(first) = chunks.first() else {
        return Err(DataError::InvalidArgument {
            message: "Arrow column has no chunks to concatenate".into(),
        });
    };
    if chunks.len() == 1 {
        return Ok(first.clone());
    }
    for chunk in chunks.iter().skip(1) {
        same_column_kind(first, chunk)?;
    }
    match first {
        OwnedColumn::Float64(_) => concat_float(chunks),
        OwnedColumn::Int64(_) => concat_int(chunks),
        OwnedColumn::Timestamp(_) => concat_timestamp(chunks),
        OwnedColumn::Categorical(_) => concat_categorical(chunks),
        OwnedColumn::FixedVector(_) => concat_vectors(chunks),
        OwnedColumn::Boolean(_) => Err(DataError::InvalidArgument {
            message: "boolean Arrow chunks are not concatenated".into(),
        }),
    }
}

fn concat_float(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let mut id = None;
    for chunk in chunks {
        let OwnedColumn::Float64(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "float64" });
        };
        id = Some(column.id);
        values.extend_from_slice(column.values.as_slice());
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let id = id.ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow column has no chunks to concatenate".into(),
    })?;
    Ok(OwnedColumn::Float64(Float64Column::new(
        id,
        F64Buffer::owned(Arc::<[f64]>::from(values)),
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

fn concat_int(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let mut id = None;
    for chunk in chunks {
        let OwnedColumn::Int64(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "int64" });
        };
        id = Some(column.id);
        values.extend_from_slice(&column.values);
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let id = id.ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow column has no chunks to concatenate".into(),
    })?;
    Ok(OwnedColumn::Int64(Int64Column::new(
        id,
        values,
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

fn concat_timestamp(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let mut id = None;
    for chunk in chunks {
        let OwnedColumn::Timestamp(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "timestamp" });
        };
        id = Some(column.id);
        values.extend_from_slice(&column.values_ns);
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let id = id.ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow column has no chunks to concatenate".into(),
    })?;
    Ok(OwnedColumn::Timestamp(TimestampColumn::new(
        id,
        values,
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

fn concat_categorical(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut codes = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let OwnedColumn::Categorical(first) = &chunks[0] else {
        return Err(DataError::TypeMismatch { id: chunks[0].id(), expected: "dictionary" });
    };
    let mut remapped = 0usize;
    for chunk in chunks {
        let OwnedColumn::Categorical(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "dictionary" });
        };
        codes.extend_from_slice(&column.codes);
        remapped += column.n_remapped_unknown;
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let mut owned = CategoricalColumn::try_new(
        first.id,
        codes,
        ValidityBitmap::from_bytes(validity, len)?,
        Arc::clone(&first.domain),
    )?;
    owned.n_remapped_unknown = remapped;
    Ok(OwnedColumn::Categorical(owned))
}

fn concat_vectors(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let OwnedColumn::FixedVector(first) = &chunks[0] else {
        return Err(DataError::TypeMismatch { id: chunks[0].id(), expected: "fixed vector" });
    };
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    for chunk in chunks {
        let OwnedColumn::FixedVector(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "fixed vector" });
        };
        values.extend_from_slice(&column.values);
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    Ok(OwnedColumn::FixedVector(FixedVectorColumn::new(
        first.id,
        first.dim,
        values,
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

fn append_validity(
    bytes: &mut Vec<u8>,
    len: usize,
    source: &ValidityBitmap,
) -> Result<usize, DataError> {
    let new_len = len.checked_add(source.len()).ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow chunk row counts overflow usize".into(),
    })?;
    bytes.resize(new_len.div_ceil(8), 0);
    for row in 0..source.len() {
        if source.is_valid(row) {
            let bit = len + row;
            bytes[bit / 8] |= 1 << (bit % 8);
        }
    }
    Ok(new_len)
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
        controls.backend.admit().map_err(|refusal| DataError::InvalidArgument {
            message: refusal.to_string(),
        })?;
        let order = resolve_order(self.n_columns(), order)?;
        let bytes = frame_payload_bytes(self)?;
        let budget = controls.budget.map_or(bytes, crate::materialize::PreparationBudget::bytes);
        let noun = match controls.stream {
            EstimatorStreamNeed::CanStream => "streaming scratch",
            EstimatorStreamNeed::CannotStream => "materialization",
        };
        if bytes > budget {
            return Err(DataError::InvalidArgument {
                message: format!(
                    "preparation refused: {noun} of {bytes} bytes exceeds budget {budget}"
                ),
            });
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

fn resolve_order(n_columns: usize, order: &[usize]) -> Result<Vec<usize>, DataError> {
    let order = if order.is_empty() { (0..n_columns).collect::<Vec<_>>() } else { order.to_vec() };
    let mut seen = vec![false; n_columns];
    for &index in &order {
        if index >= n_columns || seen[index] {
            return Err(DataError::InvalidArgument {
                message: format!(
                    "preparation order index {index} is outside the frame or repeated"
                ),
            });
        }
        seen[index] = true;
    }
    Ok(order)
}

fn frame_payload_bytes(frame: &ArrowFrameView) -> Result<u64, DataError> {
    let mut total = 0u64;
    for column in &frame.columns {
        for chunk in &column.chunks {
            total = total.checked_add(owned_payload_bytes(chunk)?).ok_or_else(|| {
                DataError::InvalidArgument { message: "column payload size overflows".into() }
            })?;
        }
    }
    Ok(total)
}

fn scan_frame(
    frame: &ArrowFrameView,
    order: &[usize],
    missing: MissingPolicy,
    categories: &[(VariableId, Contrast)],
    bytes: u64,
    budget: u64,
) -> Result<StreamingScan, DataError> {
    let schema = scan_schema(frame, order)?;
    let categories = scan_categories(frame, order, categories)?;
    let missing = scan_missing(frame, order, missing)?;
    let chunk_rows = chunk_row_counts(frame)?;
    finish_scan(schema, &categories, missing, &chunk_rows, bytes, budget)
}

fn scan_categories(
    frame: &ArrowFrameView,
    order: &[usize],
    categories: &[(VariableId, Contrast)],
) -> Result<Vec<crate::categorical::CategoryConversion>, DataError> {
    let mut recorded = Vec::new();
    for &index in order {
        let Some(chunk) = frame.columns[index].chunks.first() else {
            continue;
        };
        let OwnedColumn::Categorical(column) = chunk else {
            continue;
        };
        let id = variable_id(index)?;
        let Some((_, contrast)) = categories.iter().find(|(variable, _)| *variable == id) else {
            return Err(DataError::InvalidArgument {
                message: format!(
                    "category conversion requires an explicit contrast for variable {id}"
                ),
            });
        };
        recorded.push(crate::categorical::expand_contrast(column, contrast)?.conversion);
    }
    Ok(recorded)
}

fn scan_schema(frame: &ArrowFrameView, order: &[usize]) -> Result<SchemaConversion, DataError> {
    let mut columns = Vec::with_capacity(order.len());
    for &index in order {
        let column = &frame.columns[index];
        let kind = column.chunks.first().map_or(SchemaValueKind::Empty, storage_kind);
        columns.push(SchemaColumn {
            name: Arc::<str>::from(column.name.as_str()),
            id: variable_id(index)?,
            value_kind: kind,
        });
    }
    Ok(SchemaConversion { columns: Arc::from(columns) })
}

const fn storage_kind(column: &OwnedColumn) -> SchemaValueKind {
    match column {
        OwnedColumn::Float64(_) => SchemaValueKind::Float64,
        OwnedColumn::Int64(_) => SchemaValueKind::Int64,
        OwnedColumn::Boolean(_) => SchemaValueKind::Boolean,
        OwnedColumn::Categorical(_) => SchemaValueKind::Categorical,
        OwnedColumn::Timestamp(_) => SchemaValueKind::Timestamp,
        OwnedColumn::FixedVector(_) => SchemaValueKind::Vector,
    }
}

fn scan_missing(
    frame: &ArrowFrameView,
    order: &[usize],
    policy: MissingPolicy,
) -> Result<MissingConversion, DataError> {
    let mut chunks = Vec::with_capacity(frame.n_chunks());
    for index in 0..frame.n_chunks() {
        chunks.push(chunk_missing(frame, order, index)?);
    }
    MissingConversion::from_scan(policy, chunks)
}

fn chunk_missing(
    frame: &ArrowFrameView,
    order: &[usize],
    index: usize,
) -> Result<ChunkMissing, DataError> {
    let rows = frame.columns.first().map_or(0, |column| {
        column.chunks.get(index).map_or(0, OwnedColumn::len)
    });
    let mut dropped = Vec::new();
    for row in 0..rows {
        let invalid = order.iter().any(|&column| {
            !frame.columns[column].chunks[index].as_view().validity().is_valid(row)
        });
        if invalid {
            let row = u64::try_from(row).map_err(|_| DataError::InvalidArgument {
                message: "chunk row does not fit u64".into(),
            })?;
            dropped.push(row);
        }
    }
    let rows_kept = rows.checked_sub(dropped.len()).ok_or_else(|| DataError::InvalidArgument {
        message: "chunk kept rows overflow".into(),
    })?;
    Ok(ChunkMissing {
        rows_total: rows,
        rows_kept,
        dropped_digest: Arc::from(dropped_row_digest(rows, &dropped)),
    })
}

fn chunk_row_counts(frame: &ArrowFrameView) -> Result<Vec<u64>, DataError> {
    let Some(column) = frame.columns.first() else {
        return Ok(Vec::new());
    };
    column
        .chunks
        .iter()
        .map(|chunk| u64::try_from(chunk.len()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| DataError::InvalidArgument {
            message: "chunk row count does not fit u64".into(),
        })
}

fn materialize_frame(
    frame: &ArrowFrameView,
    order: &[usize],
    missing: MissingPolicy,
    categories: &[(VariableId, Contrast)],
    controls: PreparationControls,
    budget: u64,
) -> Result<EstimatorFrame, DataError> {
    let loaded = frame.contiguous()?;
    let ids = order.iter().copied().map(variable_id).collect::<Result<Vec<_>, _>>()?;
    let mut finite = controls;
    finite.stream = EstimatorStreamNeed::CannotStream;
    let mut product = prepare_table(&loaded.data, &ids, missing, categories, finite)?;
    let mut diagnostics = loaded.diagnostics;
    if frame.n_chunks() > 1 {
        product = product.into_materialized_from_scan()?;
        let copied = product.receipt.admission().materialized_bytes().unwrap_or(budget);
        diagnostics.push(materialization_diagnostic(
            MaterializationReason::EstimatorCannotStream,
            copied,
        ));
    }
    Ok(EstimatorFrame::Finite { table: loaded.data, product, diagnostics })
}

fn detach_foreign(column: &OwnedColumn) -> Result<OwnedColumn, DataError> {
    match column {
        OwnedColumn::Float64(column) if column.values.is_foreign() => {
            let values = Arc::<[f64]>::from(column.values.as_slice());
            Ok(OwnedColumn::Float64(Float64Column::new(
                column.id,
                F64Buffer::owned(values),
                column.validity.clone(),
            )?))
        }
        other => Ok(other.clone()),
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "values are copied bit-for-bit")]
mod tests {
    use antecedent_core::VariableId;
    use arrow_array::Float64Array;
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::arrow_ffi::ArrowCColumn;
    use crate::table::TableView;

    #[test]
    fn arrow_load_copies_and_exposes_table_view() {
        let schema = Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]);
        let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let y = Float64Array::from(vec![Some(4.0), Some(5.0), Some(6.0)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(x), Arc::new(y)]).unwrap();

        let loaded = tabular_from_record_batch(&batch).unwrap();
        assert!(loaded.bytes_copied > 0);
        assert_eq!(loaded.bytes_borrowed, 0);
        assert!(!loaded.diagnostics.is_empty());
        assert_eq!(loaded.data.row_count(), 3);
        let col = loaded.data.column(VariableId::from_raw(0)).unwrap();
        match col {
            crate::column::ColumnView::Float64(c) => {
                assert!(c.validity.is_valid(0));
                assert!(!c.validity.is_valid(1));
                assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
                assert!(!c.values.is_foreign());
            }
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn arrow_load_fixed_size_list_float64() {
        use arrow_array::FixedSizeListArray;
        use arrow_buffer::NullBuffer;

        let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let list = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float64, true)),
            2,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)]).unwrap();
        let loaded = tabular_from_record_batch(&batch).unwrap();
        assert_eq!(loaded.data.row_count(), 3);
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::FixedVector(c) => {
                assert_eq!(c.dim, 2);
                assert!(c.validity.is_valid(0));
                assert!(!c.validity.is_valid(1));
                assert!(c.validity.is_valid(2));
                assert!((c.values[0] - 1.0).abs() < f64::EPSILON);
                assert!((c.values[1] - 2.0).abs() < f64::EPSILON);
                assert!((c.values[4] - 5.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected FixedVector"),
        }
    }

    fn float_col(data: &TabularData, id: u32) -> &Float64Column {
        match data.column(VariableId::from_raw(id)).unwrap() {
            crate::column::ColumnView::Float64(c) => c,
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn arrow_null_float64_cell_reads_nan_on_every_path() {
        let schema = Schema::new(vec![Field::new("x", DataType::Float64, true)]);
        let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(x)]).unwrap();
        let loaded = tabular_from_record_batch(&batch).unwrap();
        let id = VariableId::from_raw(0);
        assert!(!float_col(&loaded.data, 0).validity.is_valid(1));
        let slice = loaded.data.float64_slice(id).unwrap();
        assert!(slice[1].is_nan(), "borrowed slice exposed {} for a null cell", slice[1]);
        assert!(loaded.data.float64_cow(id).unwrap()[1].is_nan());
        let values = loaded.data.float64_values(id).unwrap();
        assert!(values[1].is_nan());
        assert_eq!((values[0], values[2]), (1.0, 3.0));
    }

    #[test]
    fn arrow_cdi_nulls_copy_with_nan_when_slots_hold_placeholders() {
        // arrow-rs writes 0.0 under null slots, like most exporters.
        let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(x.values()[1], 0.0);
        let loaded =
            tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &x).unwrap()])
                .unwrap();
        assert_eq!(loaded.bytes_borrowed, 0);
        let c = float_col(&loaded.data, 0);
        assert!(!c.values.is_foreign());
        assert!(!c.validity.is_valid(1));
        assert!(c.values[1].is_nan());
        assert!(loaded.data.float64_slice(VariableId::from_raw(0)).unwrap()[1].is_nan());
        assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn arrow_cdi_nulls_stay_zero_copy_when_slots_hold_nan() {
        use arrow_buffer::{NullBuffer, ScalarBuffer};
        let x = Float64Array::new(
            ScalarBuffer::from(vec![1.0, f64::NAN, 3.0]),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let loaded =
            tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &x).unwrap()])
                .unwrap();
        assert!(loaded.bytes_borrowed > 0);
        let c = float_col(&loaded.data, 0);
        assert!(c.values.is_foreign());
        assert!(!c.validity.is_valid(1));
        assert!(c.values[1].is_nan());
    }

    #[test]
    fn arrow_fixed_size_list_null_component_invalidates_row() {
        use arrow_array::FixedSizeListArray;
        use arrow_buffer::NullBuffer;

        let values =
            Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0), None, Some(5.0), Some(6.0)]);
        let item = Arc::new(Field::new("item", DataType::Float64, true));
        let list = FixedSizeListArray::new(
            Arc::clone(&item),
            2,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let schema = Schema::new(vec![Field::new("v", DataType::FixedSizeList(item, 2), true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)]).unwrap();
        let loaded = tabular_from_record_batch(&batch).unwrap();
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::FixedVector(c) => {
                assert!(c.validity.is_valid(0));
                assert_eq!(&c.values[0..2], &[1.0, 2.0]);
                // Row 1 has a null component; row 2 is a null list slot.
                assert!(!c.validity.is_valid(1));
                assert!(!c.validity.is_valid(2));
                assert!(c.values[2..].iter().all(|v| v.is_nan()));
            }
            _ => panic!("expected FixedVector"),
        }
    }

    #[test]
    fn arrow_cdi_zero_copy_borrows_values() {
        let x = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let y = Float64Array::from(vec![4.0, 5.0, 6.0]);
        let loaded = tabular_from_arrow_c_columns(vec![
            ArrowCColumn::from_arrow_array("x", &x).unwrap(),
            ArrowCColumn::from_arrow_array("y", &y).unwrap(),
        ])
        .unwrap();
        assert!(loaded.bytes_borrowed > 0);
        assert_eq!(loaded.data.row_count(), 3);
        let col = loaded.data.column(VariableId::from_raw(0)).unwrap();
        match col {
            crate::column::ColumnView::Float64(c) => {
                assert!(c.values.is_foreign());
                assert!((c.values[0] - 1.0).abs() < f64::EPSILON);
                assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn row_total_is_not_capped_at_u32_max() {
        let max_u32 = usize::try_from(u32::MAX).expect("row total");
        let wide = max_u32 + 2;
        assert_eq!(checked_row_total([wide]).unwrap(), wide);
        assert_eq!(checked_row_total([max_u32, 2]).unwrap(), wide);
        assert!(checked_row_total([usize::MAX, 1]).is_err());
    }

    #[test]
    fn stream_keeps_chunks_until_contiguous() {
        use arrow_array::{Int64Array, RecordBatchIterator};
        use arrow_schema::ArrowError;

        let big = (1_i64 << 53) + 3;
        let schema = Arc::new(Schema::new(vec![Field::new("unit_id", DataType::Int64, true)]));
        let first = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![Some(1_i64), Some(big)]))],
        )
        .unwrap();
        let second = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![Some(4_i64)]))],
        )
        .unwrap();
        let empty = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(Vec::<i64>::new()))],
        )
        .unwrap();
        let batches: Vec<Result<RecordBatch, ArrowError>> = vec![Ok(first), Ok(empty), Ok(second)];
        let reader = RecordBatchIterator::new(batches.into_iter(), Arc::clone(&schema));
        let frame = ArrowFrameView::from_stream(crate::arrow_ffi::ArrowCStream::from_reader(
            Box::new(reader),
        ))
        .unwrap();
        assert_eq!(frame.n_chunks(), 3);
        let chunks = frame.column_chunks(0).unwrap();
        match &chunks[0] {
            crate::column::OwnedColumn::Int64(column) => {
                assert_eq!(column.values.as_ref(), &[1, big]);
                assert_eq!(column.len(), 2);
            }
            _ => panic!("expected int64 id"),
        }
        assert_eq!(chunks[1].len(), 0);
        let loaded = frame.contiguous().unwrap();
        assert_eq!(loaded.data.row_count(), 3);
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::Int64(column) => {
                assert_eq!(column.values.as_ref(), &[1, big, 4]);
            }
            _ => panic!("expected int64 id"),
        }
        let variable = &loaded.data.schema().variables()[0];
        assert!(variable.role_hints.contains(antecedent_core::RoleHint::UnitId));
    }

    #[test]
    fn sliced_offset_reads_the_logical_window() {
        use arrow_array::{Int64Array, TimestampMicrosecondArray};

        let ids = Int64Array::from(vec![Some(9_i64), None, Some(7), Some(8)]);
        let window = ids.slice(1, 3);
        let loaded = tabular_from_arrow_c_columns(vec![
            ArrowCColumn::from_arrow_array("unit_id", &window).unwrap(),
        ])
        .unwrap();
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::Int64(column) => {
                assert!(!column.validity.is_valid(0));
                assert!(column.validity.is_valid(1));
                assert_eq!(column.values[1], 7);
                assert_eq!(column.values[2], 8);
                assert_ne!(column.values[0], 9);
            }
            _ => panic!("expected int64"),
        }

        let stamps = TimestampMicrosecondArray::from(vec![Some(9_i64), Some(5), None, Some(7)]);
        let window = stamps.slice(1, 3);
        let loaded = tabular_from_arrow_c_columns(vec![
            ArrowCColumn::from_arrow_array("when", &window).unwrap(),
        ])
        .unwrap();
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::Timestamp(column) => {
                assert_eq!(column.len(), 3);
                assert_eq!(column.values_ns[0], 5_000);
                assert!(!column.validity.is_valid(1));
                assert!(column.validity.is_valid(0));
                assert_eq!(column.values_ns[2], 7_000);
            }
            _ => panic!("expected timestamp"),
        }
        let variable = &loaded.data.schema().variables()[0];
        assert!(variable.role_hints.contains(antecedent_core::RoleHint::Time));
    }

    #[test]
    fn dictionary_categories_keep_codes_and_labels() {
        use arrow_array::types::Int32Type;
        use arrow_array::{DictionaryArray, Int32Array, StringArray};

        let values =
            Arc::new(StringArray::from(vec!["red", "blue", "green"])) as arrow_array::ArrayRef;
        let keys = Int32Array::from(vec![Some(2), Some(0), None]);
        let dict = DictionaryArray::<Int32Type>::new(keys, Arc::clone(&values));
        let sliced = dict.slice(1, 2);
        let loaded = tabular_from_arrow_c_columns(vec![
            ArrowCColumn::from_arrow_array("arm", &sliced).unwrap(),
        ])
        .unwrap();
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::Categorical(column) => {
                assert_eq!(column.codes[0].raw(), 0);
                assert!(!column.validity.is_valid(1));
                assert_eq!(column.domain.levels.len(), 3);
                assert_eq!(column.domain.levels[0].label.as_ref(), "red");
                assert_eq!(column.domain.levels[2].label.as_ref(), "green");
                assert!(!column.domain.ordered);
            }
            _ => panic!("expected categorical"),
        }
    }

    #[test]
    fn stream_error_releases_the_reader() {
        use std::sync::atomic::{AtomicBool, Ordering};

        use arrow_schema::ArrowError;

        struct ReleaseOnDrop(Arc<AtomicBool>);
        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        struct Flaky {
            schema: Arc<Schema>,
            step: usize,
            _guard: ReleaseOnDrop,
        }
        impl Iterator for Flaky {
            type Item = Result<RecordBatch, ArrowError>;
            fn next(&mut self) -> Option<Self::Item> {
                if self.step == 0 {
                    self.step = 1;
                    let batch = RecordBatch::try_new(
                        Arc::clone(&self.schema),
                        vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
                    )
                    .unwrap();
                    Some(Ok(batch))
                } else {
                    Some(Err(ArrowError::ComputeError("batch failed".into())))
                }
            }
        }
        impl arrow_array::RecordBatchReader for Flaky {
            fn schema(&self) -> arrow_schema::SchemaRef {
                Arc::clone(&self.schema)
            }
        }
        let released = Arc::new(AtomicBool::new(false));
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let reader = Flaky { schema, step: 0, _guard: ReleaseOnDrop(Arc::clone(&released)) };
        let loaded = ArrowFrameView::from_stream(crate::arrow_ffi::ArrowCStream::from_reader(
            Box::new(reader),
        ));
        assert!(loaded.is_err());
        assert!(released.load(Ordering::SeqCst));
    }

    #[test]
    fn second_column_failure_drops_the_first_import() {
        use arrow_array::StringArray;

        let good = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let bad = StringArray::from(vec!["no"]);
        let loaded = tabular_from_arrow_c_columns(vec![
            ArrowCColumn::from_arrow_array("x", &good).unwrap(),
            ArrowCColumn::from_arrow_array("s", &bad).unwrap(),
        ]);
        assert!(loaded.is_err());
    }

    #[test]
    fn categorical_chunks_keep_the_reference_and_the_domain_id() {
        fn chunk(domain_id: u32, reference: u32) -> OwnedColumn {
            let domain = Arc::new(
                CategoryDomain::try_new(
                    CategoryDomainId::from_raw(domain_id),
                    vec![
                        CategoryLevel { label: Arc::<str>::from("a") },
                        CategoryLevel { label: Arc::<str>::from("b") },
                    ],
                    false,
                    Some(CategoryCode::from_raw(reference)),
                    UnknownCategoryPolicy::Fail,
                )
                .unwrap(),
            );
            OwnedColumn::Categorical(
                CategoricalColumn::try_new(
                    VariableId::from_raw(1),
                    vec![CategoryCode::from_raw(0)],
                    ValidityBitmap::all_valid(1),
                    domain,
                )
                .unwrap(),
            )
        }

        let matched = concat_chunks(&[chunk(7, 0), chunk(7, 0)]).unwrap();
        let OwnedColumn::Categorical(column) = matched else {
            panic!("categorical");
        };
        assert_eq!(column.domain.reference, Some(CategoryCode::from_raw(0)));
        assert!(concat_chunks(&[chunk(7, 0), chunk(7, 1)]).is_err());
        assert!(concat_chunks(&[chunk(7, 0), chunk(8, 0)]).is_err());
    }

    fn two_chunk_frame() -> ArrowFrameView {
        use arrow_array::RecordBatch;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let first = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
        )
        .unwrap();
        let second = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(vec![3.0]))],
        )
        .unwrap();
        ArrowFrameView::from_record_batches(vec![first, second]).unwrap()
    }

    #[test]
    fn an_estimator_that_cannot_stream_materializes_or_refuses() {
        use crate::materialize::{EstimatorStreamNeed, PreparationBudget, PreparationControls};
        use crate::sample_policy::MissingPolicy;

        let frame = two_chunk_frame();
        let mut controls = PreparationControls::finite_resident();
        controls.budget = Some(PreparationBudget::new(0));
        let refused = frame.open_for_estimator(&[], MissingPolicy::CompleteCase, &[], controls);
        let message = refused.expect_err("budget").to_string();
        assert!(message.contains("exceeds budget 0"), "{message}");

        controls.budget = None;
        let delivered =
            frame.open_for_estimator(&[], MissingPolicy::CompleteCase, &[], controls).unwrap();
        let EstimatorFrame::Finite { table, product, diagnostics } = delivered else {
            panic!("cannot-stream delivery must be a finite table");
        };
        assert_eq!(table.row_count(), 3);
        assert_eq!(product.receipt.proved_rows(), Some(3));
        assert!(!product.receipt.estimator_sees_unbounded_stream());
        assert!(product.receipt.materialized_from_scan());
        // No budget was given, so the copy is recorded as unmetered rather than admitted.
        assert!(!product.receipt.is_bounded_end_to_end_route());
        assert!(diagnostics.entries.iter().any(|entry| {
            entry.code.contains("materialize.estimator")
        }));

        controls.stream = EstimatorStreamNeed::CanStream;
        let scan =
            frame.open_for_estimator(&[], MissingPolicy::CompleteCase, &[], controls).unwrap();
        let EstimatorFrame::Scan(scan) = scan else {
            panic!("a streaming estimator keeps the chunks");
        };
        assert!(scan.receipt().estimator_sees_unbounded_stream());
        assert!(!scan.receipt().is_bounded_end_to_end_route());
        assert!(!scan.receipt().missing().unwrap().population_bound);
    }
}
