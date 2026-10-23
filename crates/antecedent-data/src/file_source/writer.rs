//! Write library-owned tables as Parquet or Arrow IPC files.
//!
//! The writer is the inverse of the reader for every kind it reads: float64,
//! int64, boolean, nanosecond timestamps, and categories as dictionaries with
//! `u32` keys whose values are the domain labels in order (the order flag is
//! kept). Invalid cells are written as nulls. Fixed-width vector columns are
//! refused. Parquet is written uncompressed, one row group per `rows_per_group`
//! rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;
use std::sync::Arc;

use arrow_array::types::UInt32Type;
use arrow_array::{
    ArrayRef, BooleanArray, DictionaryArray, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};

use crate::column::OwnedColumn;
use crate::dataset::TabularData;
use crate::global_index::GlobalId;
use crate::table::TableView;

use super::TableFileFormat;
use super::error::FileSourceError;

/// How a table is written.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WriteOptions {
    /// Rows per Parquet row group and per Arrow IPC record batch.
    pub rows_per_group: usize,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self { rows_per_group: super::DEFAULT_CHUNK_ROWS }
    }
}

/// Write `data` to `path` as `format` and return the rows written.
///
/// # Errors
///
/// The format is not compiled, a column is a fixed-width vector, `rows_per_group`
/// is zero, or the file cannot be written.
pub fn write_table_file(
    path: impl AsRef<Path>,
    format: TableFileFormat,
    data: &TabularData,
    options: WriteOptions,
) -> Result<GlobalId, FileSourceError> {
    let path = path.as_ref();
    if !format.is_compiled() {
        return Err(format.unavailable());
    }
    if options.rows_per_group == 0 {
        return Err(FileSourceError::InvalidOptions {
            detail: "rows_per_group must be at least one row".into(),
        });
    }
    let (schema, columns) = arrow_columns(data)?;
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(|error| super::provider::format_error(path, &error))?;
    let rows = batch.num_rows();
    let mut batches = Vec::new();
    let mut offset = 0;
    while offset < rows {
        let take = options.rows_per_group.min(rows - offset);
        batches.push(batch.slice(offset, take));
        offset += take;
    }
    if batches.is_empty() {
        batches.push(batch);
    }
    match format {
        TableFileFormat::Parquet => write_parquet(path, &schema, &batches, options)?,
        TableFileFormat::ArrowIpcFile => write_ipc(path, &schema, &batches, false)?,
        TableFileFormat::ArrowIpcStream => write_ipc(path, &schema, &batches, true)?,
    }
    Ok(GlobalId::new(rows as u64))
}

fn arrow_columns(data: &TabularData) -> Result<(SchemaRef, Vec<ArrayRef>), FileSourceError> {
    let storage = data.storage();
    let mut fields = Vec::with_capacity(storage.columns().len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(storage.columns().len());
    for column in storage.columns() {
        let id = column.id();
        let name =
            storage.schema().variables().iter().find(|variable| variable.id == id).map_or_else(
                || format!("column_{}", id.raw()),
                |variable| variable.name.to_string(),
            );
        let validity = column.as_view().validity();
        let valid = |row: usize| validity.is_valid(row);
        let (field, array): (Field, ArrayRef) = match column {
            OwnedColumn::Float64(column) => {
                let values = column.values.as_slice();
                let array: Float64Array =
                    (0..values.len()).map(|row| valid(row).then_some(values[row])).collect();
                (Field::new(name, DataType::Float64, true), Arc::new(array))
            }
            OwnedColumn::Int64(column) => {
                let array: Int64Array = (0..column.values.len())
                    .map(|row| valid(row).then_some(column.values[row]))
                    .collect();
                (Field::new(name, DataType::Int64, true), Arc::new(array))
            }
            OwnedColumn::Boolean(column) => {
                let array: BooleanArray = (0..column.values.len())
                    .map(|row| valid(row).then_some(column.values[row] != 0))
                    .collect();
                (Field::new(name, DataType::Boolean, true), Arc::new(array))
            }
            OwnedColumn::Timestamp(column) => {
                let array: TimestampNanosecondArray = (0..column.values_ns.len())
                    .map(|row| valid(row).then_some(column.values_ns[row]))
                    .collect();
                (
                    Field::new(name, DataType::Timestamp(TimeUnit::Nanosecond, None), true),
                    Arc::new(array),
                )
            }
            OwnedColumn::Categorical(column) => {
                let keys: UInt32Array = (0..column.codes.len())
                    .map(|row| valid(row).then_some(column.codes[row].raw()))
                    .collect();
                let labels: StringArray =
                    column.domain.levels.iter().map(|level| Some(&*level.label)).collect();
                let array = DictionaryArray::<UInt32Type>::try_new(keys, Arc::new(labels))
                    .map_err(|error| FileSourceError::InvalidOptions {
                        detail: format!("category column '{name}': {error}"),
                    })?;
                let data_type =
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8));
                #[allow(deprecated, reason = "the order flag is set where the field is built")]
                let field = Field::new_dict(name, data_type, true, 0, column.domain.ordered);
                (field, Arc::new(array))
            }
            OwnedColumn::FixedVector(_) => {
                return Err(FileSourceError::UnsupportedColumn {
                    column: name,
                    arrow_type: "FixedSizeList<Float64>".to_owned(),
                    next_attempt: "write each vector component as its own float column",
                });
            }
        };
        fields.push(field);
        arrays.push(array);
    }
    if fields.is_empty() {
        return Err(FileSourceError::InvalidOptions { detail: "the table has no column".into() });
    }
    Ok((Arc::new(Schema::new(fields)), arrays))
}

#[cfg(feature = "parquet")]
fn write_parquet(
    path: &Path,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    options: WriteOptions,
) -> Result<(), FileSourceError> {
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let file = std::fs::File::create(path).map_err(|error| FileSourceError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let properties =
        WriterProperties::builder().set_max_row_group_size(options.rows_per_group).build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(schema), Some(properties))
        .map_err(|error| super::provider::format_error(path, &error))?;
    for batch in batches {
        writer.write(batch).map_err(|error| super::provider::format_error(path, &error))?;
        writer.flush().map_err(|error| super::provider::format_error(path, &error))?;
    }
    writer.close().map_err(|error| super::provider::format_error(path, &error))?;
    Ok(())
}

#[cfg(not(feature = "parquet"))]
fn write_parquet(
    _path: &Path,
    _schema: &SchemaRef,
    _batches: &[RecordBatch],
    _options: WriteOptions,
) -> Result<(), FileSourceError> {
    Err(TableFileFormat::Parquet.unavailable())
}

#[cfg(feature = "ipc")]
fn write_ipc(
    path: &Path,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    stream: bool,
) -> Result<(), FileSourceError> {
    use arrow_ipc::writer::{FileWriter, StreamWriter};

    let file = std::fs::File::create(path).map_err(|error| FileSourceError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let fail = |error: arrow_schema::ArrowError| super::provider::format_error(path, &error);
    if stream {
        let mut writer = StreamWriter::try_new(file, schema).map_err(fail)?;
        for batch in batches {
            writer.write(batch).map_err(fail)?;
        }
        writer.finish().map_err(fail)?;
    } else {
        let mut writer = FileWriter::try_new(file, schema).map_err(fail)?;
        for batch in batches {
            writer.write(batch).map_err(fail)?;
        }
        writer.finish().map_err(fail)?;
    }
    Ok(())
}

#[cfg(not(feature = "ipc"))]
fn write_ipc(
    _path: &Path,
    _schema: &SchemaRef,
    _batches: &[RecordBatch],
    stream: bool,
) -> Result<(), FileSourceError> {
    let format =
        if stream { TableFileFormat::ArrowIpcStream } else { TableFileFormat::ArrowIpcFile };
    Err(format.unavailable())
}
