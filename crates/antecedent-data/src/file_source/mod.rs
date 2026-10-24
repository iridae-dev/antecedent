//! File-backed table sources: Parquet files, directories of Parquet files, Arrow
//! IPC files and streams, and Arrow C streams.
//!
//! A [`TableFile`] reads only its schema and footer when it is opened. A read is
//! planned from the required-column closure ([`crate::column_closure`]): the
//! scientific target together with every validator, support, weight, split, and
//! complete-case column. Only those columns are decoded, and a projection that
//! drops one of them is refused before any byte is read.
//!
//! Rows stream in bounded chunks. Each chunk is charged to the execution
//! context's admission pool *before* it is decoded and again before its
//! library-owned columns are built, so a pool that cannot hold one chunk is a
//! refusal that names the bytes and the next attempt (a smaller chunk or a
//! larger pool), and leaves the pool as it was. An Arrow IPC batch keeps the
//! size its writer chose, so its decode storage is sized from the message
//! metadata (body length, inflated buffers, dictionaries and per-array
//! structure) and leased in full before the reader allocates any of it; a batch
//! the pool cannot hold is refused undecoded. Parquet is sized from the footer's
//! column sizes, including the plain bytes of byte-array columns. Rows are addressed by checked
//! [`GlobalId`]s, so a table may hold more than `2^32` rows; a row is narrowed
//! to a [`ChunkIndex`] only inside its own bounded chunk.
//!
//! Every read computes the full-content identity as it streams
//! ([`FileContentIdentity`]); the identity is issued only after a complete first
//! scan, and it changes whenever a value changes. A route that needs the whole
//! table resident gets a [`MaterializationQuote`] first and is refused before
//! reading when the pool cannot hold it; a route that only needs sufficient
//! statistics streams the chunks and holds one at a time.
//!
//! Types are kept exactly: floats as IEEE-754 binary64 (narrower floats widen
//! exactly), integers as `i64`, dates and timestamps as nanoseconds, booleans,
//! and dictionaries or strings as category codes over a domain that keeps the
//! dictionary order. Nested, decimal, and binary columns are refused with the
//! next attempt rather than cast. Nulls are validity masks; a null float cell is
//! invalid and holds `NaN`, as the missing-data policies expect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod convert;
mod error;
mod identity;
#[cfg(feature = "ipc")]
mod ipc_scan;
mod materialize;
mod provider;
mod writer;

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema, SchemaRef};

use crate::arrow_ffi::ArrowCStream;
use crate::column::OwnedColumn;
use crate::column_closure::{ClosureInputs, ColumnClosure, ColumnRole};
use crate::global_index::{ChunkIndex, GlobalId};

use convert::ColumnState;
use identity::ColumnDigest;
use provider::{BatchProvider, DecodeNeed};

pub use error::{FileSourceError, FileSourceErrorKind, MemoryStage};
pub use identity::{FILE_IDENTITY_VERSION, FileContentIdentity};
pub use materialize::{MaterializationQuote, MaterializedTable};
pub use writer::{WriteOptions, write_table_file};

/// Rows per chunk when the caller does not choose.
pub const DEFAULT_CHUNK_ROWS: usize = 65_536;

/// Bytes per row assumed for a string or dictionary column whose file records
/// no column sizes (Arrow IPC). A decoded batch above the quote is charged
/// before its columns are built.
pub const DEFAULT_TEXT_BYTES_PER_ROW: u64 = 64;

/// On-disk table format.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TableFileFormat {
    /// Apache Parquet; a directory is read as a dataset of Parquet files.
    Parquet,
    /// Arrow IPC file (Feather version 2).
    ArrowIpcFile,
    /// Arrow IPC stream.
    ArrowIpcStream,
}

impl TableFileFormat {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::ArrowIpcFile => "arrow_ipc_file",
            Self::ArrowIpcStream => "arrow_ipc_stream",
        }
    }

    /// Format named `name` (`parquet`, `arrow_ipc_file` or `feather`, `arrow_ipc_stream`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "parquet" => Some(Self::Parquet),
            "arrow_ipc_file" | "ipc" | "arrow" | "feather" => Some(Self::ArrowIpcFile),
            "arrow_ipc_stream" | "arrows" => Some(Self::ArrowIpcStream),
            _ => None,
        }
    }

    /// Format implied by the extension of `path`.
    ///
    /// `.parquet`, `.parq` and `.pq` are Parquet; `.arrow`, `.feather` and `.ipc`
    /// are Arrow IPC files; `.arrows` is an Arrow IPC stream.
    #[must_use]
    pub fn from_path(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        match extension.as_str() {
            "parquet" | "parq" | "pq" => Some(Self::Parquet),
            "arrow" | "feather" | "ipc" => Some(Self::ArrowIpcFile),
            "arrows" => Some(Self::ArrowIpcStream),
            _ => None,
        }
    }

    /// Cargo feature of `antecedent-data` that compiles this reader.
    #[must_use]
    pub const fn feature(self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::ArrowIpcFile | Self::ArrowIpcStream => "ipc",
        }
    }

    /// Whether this build reads the format.
    #[must_use]
    pub const fn is_compiled(self) -> bool {
        match self {
            Self::Parquet => cfg!(feature = "parquet"),
            Self::ArrowIpcFile | Self::ArrowIpcStream => cfg!(feature = "ipc"),
        }
    }

    pub(crate) const fn unavailable(self) -> FileSourceError {
        FileSourceError::FormatUnavailable { format: self.as_str(), feature: self.feature() }
    }
}

impl fmt::Display for TableFileFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a file column is held once it is read.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum FileColumnKind {
    /// IEEE-754 binary64. Binary32 and binary16 columns widen exactly.
    Float64,
    /// Exact signed 64-bit integer, from any integer width.
    Int64,
    /// Boolean.
    Boolean,
    /// Nanoseconds since the epoch, from a timestamp or a date.
    Timestamp {
        /// Time zone the file declares, if any.
        timezone: Option<Arc<str>>,
    },
    /// Category codes over the dictionary values or string labels.
    Categorical {
        /// Whether the file declares the dictionary order meaningful.
        ordered: bool,
    },
    /// A type the reader refuses without a lossy cast.
    Unsupported {
        /// Arrow type of the column.
        arrow_type: Arc<str>,
    },
}

impl FileColumnKind {
    /// Stable name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Float64 => "float64",
            Self::Int64 => "int64",
            Self::Boolean => "boolean",
            Self::Timestamp { .. } => "timestamp",
            Self::Categorical { .. } => "categorical",
            Self::Unsupported { .. } => "unsupported",
        }
    }

    /// Bytes per row of the library-owned value buffer, without the validity mask.
    #[must_use]
    pub const fn owned_width_bytes(&self) -> Option<u64> {
        match self {
            Self::Float64 | Self::Int64 | Self::Timestamp { .. } => Some(8),
            Self::Boolean => Some(1),
            Self::Categorical { .. } => Some(4),
            Self::Unsupported { .. } => None,
        }
    }

    fn of(field: &arrow_schema::Field) -> Self {
        match field.data_type() {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => Self::Float64,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => Self::Int64,
            DataType::Boolean => Self::Boolean,
            DataType::Timestamp(_, timezone) => Self::Timestamp { timezone: timezone.clone() },
            DataType::Date32 | DataType::Date64 => Self::Timestamp { timezone: None },
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                Self::Categorical { ordered: false }
            }
            DataType::Dictionary(key, value)
                if dictionary_key_is_exact(key) && dictionary_value_is_label(value) =>
            {
                Self::Categorical { ordered: field.dict_is_ordered().unwrap_or(false) }
            }
            other => Self::Unsupported { arrow_type: Arc::from(other.to_string()) },
        }
    }
}

impl fmt::Display for FileColumnKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { arrow_type } => write!(f, "unsupported ({arrow_type})"),
            Self::Categorical { ordered: true } => f.write_str("ordered categorical"),
            other => f.write_str(other.as_str()),
        }
    }
}

const fn dictionary_key_is_exact(key: &DataType) -> bool {
    matches!(
        key,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
    )
}

const fn dictionary_value_is_label(value: &DataType) -> bool {
    matches!(
        value,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

/// One column of a file schema.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FileColumn {
    /// Column name.
    pub name: Arc<str>,
    /// How the column is held once read.
    pub kind: FileColumnKind,
    /// Whether the schema allows nulls.
    pub nullable: bool,
    /// Arrow type name, for diagnostics.
    pub arrow_type: Arc<str>,
}

/// Bounds of one read.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ReadOptions {
    /// Rows per chunk. Parquet decodes at most this many rows at a time.
    pub chunk_rows: usize,
    /// Bytes per row assumed for a string or dictionary column whose file records
    /// no column sizes.
    pub text_bytes_per_row: u64,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self { chunk_rows: DEFAULT_CHUNK_ROWS, text_bytes_per_row: DEFAULT_TEXT_BYTES_PER_ROW }
    }
}

impl ReadOptions {
    /// Default options with `chunk_rows` rows per chunk.
    #[must_use]
    pub fn with_chunk_rows(chunk_rows: usize) -> Self {
        Self { chunk_rows, ..Self::default() }
    }
}

/// Bytes charged to the admission pool for one chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChunkQuote {
    /// Rows in the chunk.
    pub rows: usize,
    /// Decoded Arrow buffers of the projected columns, with their null masks.
    pub decode_bytes: u64,
    /// Library-owned columns, including the copy that seals each buffer.
    pub owned_bytes: u64,
}

impl ChunkQuote {
    /// Decode plus owned bytes.
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.decode_bytes.saturating_add(self.owned_bytes)
    }
}

/// A projected, bounded read of a table source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadPlan {
    closure: ColumnClosure,
    columns: Vec<FileColumn>,
    /// Plan position to file column index.
    file_index: Vec<usize>,
    /// Ascending file column indices the provider decodes.
    projection: Vec<usize>,
    /// Plan position to the column position inside a projected batch.
    batch_position: Vec<usize>,
    chunk_rows: usize,
    decode_bytes_per_row: u64,
    owned_bytes_per_row: u64,
    /// Bytes the reader holds for the whole stream beside its chunks (the
    /// parsed Parquet footer).
    reader_bytes: u64,
}

impl ReadPlan {
    /// Bytes the file reader holds for the whole stream beside any chunk: the
    /// parsed footer of a Parquet file, which grows with its row groups.
    #[must_use]
    pub const fn reader_bytes(&self) -> u64 {
        self.reader_bytes
    }

    /// The required-column closure this plan reads.
    #[must_use]
    pub fn closure(&self) -> &ColumnClosure {
        &self.closure
    }

    /// Projected columns, in closure order.
    #[must_use]
    pub fn columns(&self) -> &[FileColumn] {
        &self.columns
    }

    /// Projected column names, in closure order.
    #[must_use]
    pub fn names(&self) -> Vec<Arc<str>> {
        self.columns.iter().map(|column| Arc::clone(&column.name)).collect()
    }

    /// Position of `name` in the plan.
    #[must_use]
    pub fn position(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|column| &*column.name == name)
    }

    /// File column indices that are decoded, ascending.
    #[must_use]
    pub fn projection(&self) -> &[usize] {
        &self.projection
    }

    /// Rows per chunk.
    #[must_use]
    pub const fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    /// The same plan with `chunk_rows` rows per chunk.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::InvalidOptions`] when `chunk_rows` is zero.
    pub fn with_chunk_rows(&self, chunk_rows: usize) -> Result<Self, FileSourceError> {
        if chunk_rows == 0 {
            return Err(zero_chunk());
        }
        Ok(Self { chunk_rows, ..self.clone() })
    }

    /// Owned bytes per row, without validity masks.
    #[must_use]
    pub const fn owned_bytes_per_row(&self) -> u64 {
        self.owned_bytes_per_row
    }

    /// Quote for one full chunk.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::CountOverflow`] when the bytes do not fit in `u64`.
    pub fn chunk_quote(&self) -> Result<ChunkQuote, FileSourceError> {
        self.quote_rows(self.chunk_rows)
    }

    /// Quote for a chunk of `rows` rows.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::CountOverflow`] when the bytes do not fit in `u64`.
    pub fn quote_rows(&self, rows: usize) -> Result<ChunkQuote, FileSourceError> {
        let overflow = || FileSourceError::CountOverflow { context: "chunk byte quote" };
        let rows_u64 = u64::try_from(rows).map_err(|_| overflow())?;
        let masks =
            rows_u64.div_ceil(8).checked_mul(self.columns.len() as u64).ok_or_else(overflow)?;
        let decode = rows_u64
            .checked_mul(self.decode_bytes_per_row)
            .and_then(|bytes| bytes.checked_add(masks))
            .ok_or_else(overflow)?;
        let owned = rows_u64
            .checked_mul(self.owned_bytes_per_row)
            .and_then(|bytes| bytes.checked_add(masks))
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or_else(overflow)?;
        Ok(ChunkQuote { rows, decode_bytes: decode, owned_bytes: owned })
    }
}

fn zero_chunk() -> FileSourceError {
    FileSourceError::InvalidOptions { detail: "chunk_rows must be at least one row".into() }
}

/// Checked position of the next row of a stream.
///
/// Positions are `u64` [`GlobalId`]s. Advancing past `u64::MAX` is refused; a
/// position above `u32::MAX` is ordinary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct RowCursor {
    next: u64,
}

impl RowCursor {
    /// Cursor at row zero.
    #[must_use]
    pub const fn new() -> Self {
        Self { next: 0 }
    }

    /// Cursor at `start`.
    #[must_use]
    pub const fn starting_at(start: GlobalId) -> Self {
        Self { next: start.get() }
    }

    /// Global id of the next row.
    #[must_use]
    pub const fn position(self) -> GlobalId {
        GlobalId::new(self.next)
    }

    /// Advance by `rows` and return the global id of the first of them.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::CountOverflow`] when the position would pass `u64::MAX`.
    pub fn advance(&mut self, rows: u64) -> Result<GlobalId, FileSourceError> {
        let start = self.next;
        self.next = start
            .checked_add(rows)
            .ok_or(FileSourceError::CountOverflow { context: "global row id" })?;
        Ok(GlobalId::new(start))
    }
}

/// Offset of global row `row` inside the chunk that starts at `chunk_start`.
///
/// # Errors
///
/// [`FileSourceError::InvalidOptions`] when `row` precedes the chunk or its
/// offset does not fit a [`ChunkIndex`]; the message names the next attempt.
pub fn chunk_local_index(
    chunk_start: GlobalId,
    row: GlobalId,
) -> Result<ChunkIndex, FileSourceError> {
    let offset = row.get().checked_sub(chunk_start.get()).ok_or_else(|| {
        FileSourceError::InvalidOptions {
            detail: format!("row {} precedes the chunk at {}", row.get(), chunk_start.get()),
        }
    })?;
    GlobalId::new(offset)
        .to_chunk_index()
        .map_err(|error| FileSourceError::InvalidOptions { detail: error.to_string() })
}

/// Schema of a table source and the per-column decode estimate.
#[derive(Clone, Debug)]
struct SchemaView {
    schema: SchemaRef,
    columns: Vec<FileColumn>,
    decode_per_row: Vec<Option<u64>>,
    reader_bytes: u64,
}

impl SchemaView {
    fn new(schema: SchemaRef, decode_per_row: Vec<Option<u64>>, reader_bytes: u64) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|field| FileColumn {
                name: Arc::from(field.name().as_str()),
                kind: FileColumnKind::of(field),
                nullable: field.is_nullable(),
                arrow_type: Arc::from(field.data_type().to_string()),
            })
            .collect();
        Self { schema, columns, decode_per_row, reader_bytes }
    }

    fn plan(
        &self,
        closure: &ColumnClosure,
        options: ReadOptions,
    ) -> Result<ReadPlan, FileSourceError> {
        if options.chunk_rows == 0 {
            return Err(zero_chunk());
        }
        if closure.columns().is_empty() {
            return Err(FileSourceError::InvalidOptions {
                detail: "the required-column closure names no column".into(),
            });
        }
        let mut file_index = Vec::with_capacity(closure.columns().len());
        let mut resolved = Vec::with_capacity(closure.columns().len());
        for required in closure.columns() {
            let name = required.role.as_str();
            let mut matches =
                self.columns.iter().enumerate().filter(|(_, column)| &*column.name == name);
            let Some((index, column)) = matches.next() else {
                return Err(FileSourceError::UnknownColumn {
                    column: name.to_owned(),
                    available: self.columns.iter().map(|column| column.name.to_string()).collect(),
                });
            };
            if matches.next().is_some() {
                return Err(FileSourceError::SchemaMismatch {
                    path: PathBuf::new(),
                    detail: format!("column name '{name}' occurs more than once"),
                });
            }
            if let FileColumnKind::Unsupported { .. } = column.kind {
                return Err(convert::unsupported_type(name, self.schema.field(index).data_type()));
            }
            file_index.push(index);
            resolved.push(ColumnRole::new(name));
        }
        closure.project(&resolved)?;
        let mut projection = file_index.clone();
        projection.sort_unstable();
        let batch_position =
            file_index.iter().map(|index| projection.binary_search(index).unwrap_or(0)).collect();
        let mut decode_bytes_per_row = 0_u64;
        let mut owned_bytes_per_row = 0_u64;
        let mut columns = Vec::with_capacity(file_index.len());
        for &index in &file_index {
            let column = self.columns[index].clone();
            let arrow = arrow_decode_width(self.schema.field(index).data_type(), options);
            let recorded = self.decode_per_row.get(index).copied().flatten().unwrap_or(0);
            decode_bytes_per_row = decode_bytes_per_row.saturating_add(arrow.max(recorded));
            owned_bytes_per_row =
                owned_bytes_per_row.saturating_add(column.kind.owned_width_bytes().unwrap_or(8));
            columns.push(column);
        }
        Ok(ReadPlan {
            closure: closure.clone(),
            columns,
            file_index,
            projection,
            batch_position,
            chunk_rows: options.chunk_rows,
            decode_bytes_per_row,
            owned_bytes_per_row,
            reader_bytes: self.reader_bytes,
        })
    }
}

fn arrow_decode_width(data_type: &DataType, options: ReadOptions) -> u64 {
    let text = options.text_bytes_per_row;
    match data_type {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
        DataType::Utf8 => 4_u64.saturating_add(text),
        DataType::LargeUtf8 => 8_u64.saturating_add(text),
        DataType::Utf8View => 16_u64.saturating_add(text),
        DataType::Dictionary(key, _) => arrow_decode_width(key, options).saturating_add(text),
        _ => 8,
    }
}

fn same_schema(first: &Schema, next: &Schema, path: &Path) -> Result<(), FileSourceError> {
    if first.fields().len() != next.fields().len() {
        return Err(FileSourceError::SchemaMismatch {
            path: path.to_path_buf(),
            detail: format!(
                "{} columns where the first file has {}",
                next.fields().len(),
                first.fields().len()
            ),
        });
    }
    for (left, right) in first.fields().iter().zip(next.fields().iter()) {
        if left.name() != right.name() || left.data_type() != right.data_type() {
            return Err(FileSourceError::SchemaMismatch {
                path: path.to_path_buf(),
                detail: format!(
                    "column {} {} where the first file has {} {}",
                    right.name(),
                    right.data_type(),
                    left.name(),
                    left.data_type()
                ),
            });
        }
    }
    Ok(())
}

fn list_dataset(directory: &Path) -> Result<Vec<PathBuf>, FileSourceError> {
    let io = |error: std::io::Error| FileSourceError::Io {
        path: directory.to_path_buf(),
        message: error.to_string(),
    };
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(io)? {
        let path = entry.map_err(io)?.path();
        let hidden = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with('.') || name.starts_with('_'));
        if hidden || !path.is_file() {
            continue;
        }
        if TableFileFormat::from_path(&path) == Some(TableFileFormat::Parquet) {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(FileSourceError::EmptyDataset { path: directory.to_path_buf() });
    }
    Ok(paths)
}

/// A table stored in one Parquet or Arrow IPC file, or in a directory of
/// Parquet files that share one schema.
///
/// Opening reads the schema and the footer only. Reads are planned with
/// [`Self::plan`] and executed chunk by chunk with [`Self::reader`], or
/// resident with [`Self::materialize`].
#[derive(Clone, Debug)]
pub struct TableFile {
    format: TableFileFormat,
    paths: Vec<PathBuf>,
    view: SchemaView,
    declared_rows: Option<GlobalId>,
}

impl TableFile {
    /// Open `path`, inferring the format from its extension. A directory is a
    /// dataset of the `.parquet` files directly inside it, in file-name order
    /// (names starting with `.` or `_` are skipped).
    ///
    /// # Errors
    ///
    /// The format cannot be inferred or is not compiled, the file cannot be
    /// read, or dataset files disagree on their schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FileSourceError> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::open_as(path, TableFileFormat::Parquet);
        }
        let format = TableFileFormat::from_path(path)
            .ok_or_else(|| FileSourceError::UnknownFormat { path: path.to_path_buf() })?;
        Self::open_as(path, format)
    }

    /// Open `path` as `format`.
    ///
    /// # Errors
    ///
    /// As [`Self::open`]. A directory is accepted only as a Parquet dataset.
    pub fn open_as(
        path: impl AsRef<Path>,
        format: TableFileFormat,
    ) -> Result<Self, FileSourceError> {
        let path = path.as_ref();
        if !format.is_compiled() {
            return Err(format.unavailable());
        }
        let paths = if path.is_dir() {
            if format != TableFileFormat::Parquet {
                return Err(FileSourceError::InvalidOptions {
                    detail: format!(
                        "a directory is read as a dataset of Parquet files, not as {format}"
                    ),
                });
            }
            list_dataset(path)?
        } else {
            vec![path.to_path_buf()]
        };
        let first = provider::inspect(&paths[0], format)?;
        let mut rows = first.rows;
        let mut decode_per_row = first.decode_per_row;
        // Files of a dataset are opened one after another, so the largest footer bounds it.
        let mut reader_bytes = first.reader_bytes;
        for other in &paths[1..] {
            let next = provider::inspect(other, format)?;
            same_schema(&first.schema, &next.schema, other)?;
            rows = match (rows, next.rows) {
                (Some(total), Some(more)) => Some(
                    total
                        .checked_add(more)
                        .ok_or(FileSourceError::CountOverflow { context: "dataset row count" })?,
                ),
                _ => None,
            };
            reader_bytes = reader_bytes.max(next.reader_bytes);
            for (slot, more) in decode_per_row.iter_mut().zip(next.decode_per_row) {
                *slot = match (*slot, more) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    (left, right) => left.or(right),
                };
            }
        }
        Ok(Self {
            format,
            paths,
            view: SchemaView::new(first.schema, decode_per_row, reader_bytes),
            declared_rows: rows.map(GlobalId::new),
        })
    }

    /// Format of the files.
    #[must_use]
    pub const fn format(&self) -> TableFileFormat {
        self.format
    }

    /// Files read, in order.
    #[must_use]
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Every column of the schema, readable or not.
    #[must_use]
    pub fn columns(&self) -> &[FileColumn] {
        &self.view.columns
    }

    /// Column named `name`.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&FileColumn> {
        self.view.columns.iter().find(|column| &*column.name == name)
    }

    /// Row count the footers record. Arrow IPC records none.
    #[must_use]
    pub const fn declared_rows(&self) -> Option<GlobalId> {
        self.declared_rows
    }

    /// Plan a read of the columns `closure` requires.
    ///
    /// # Errors
    ///
    /// A required column is absent, repeated, or of a refused type; the closure
    /// refuses the projection; or `chunk_rows` is zero.
    pub fn plan(
        &self,
        closure: &ColumnClosure,
        options: ReadOptions,
    ) -> Result<ReadPlan, FileSourceError> {
        self.view.plan(closure, options)
    }

    /// Plan a read of `names`, each a target of the closure.
    ///
    /// # Errors
    ///
    /// As [`Self::plan`].
    pub fn plan_columns<S: AsRef<str>>(
        &self,
        names: &[S],
        options: ReadOptions,
    ) -> Result<ReadPlan, FileSourceError> {
        let closure = ClosureInputs::new(names.iter().map(ColumnRole::new)).close();
        self.view.plan(&closure, options)
    }

    /// Plan a read of every readable column, in file order.
    ///
    /// # Errors
    ///
    /// As [`Self::plan`]; also when no column is readable.
    pub fn plan_all(&self, options: ReadOptions) -> Result<ReadPlan, FileSourceError> {
        let names: Vec<&str> = self
            .view
            .columns
            .iter()
            .filter(|column| !matches!(column.kind, FileColumnKind::Unsupported { .. }))
            .map(|column| &*column.name)
            .collect();
        self.plan_columns(&names, options)
    }

    /// Chunked reader for `plan`, charging `ctx`'s admission pool.
    ///
    /// # Errors
    ///
    /// The plan does not belong to this schema, or the files cannot be opened.
    pub fn reader(
        &self,
        plan: &ReadPlan,
        ctx: &ExecutionContext,
    ) -> Result<TableChunkReader, FileSourceError> {
        self.check_plan(plan)?;
        let provider =
            provider::open_all(self.format, &self.paths, &plan.projection, plan.chunk_rows);
        TableChunkReader::new(
            Origin::File { format: self.format, paths: self.paths.clone() },
            plan.clone(),
            ctx,
            provider,
        )
    }

    /// Stream every chunk of `plan` and return the content identity.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::reader`] and [`TableChunkReader::next_chunk`].
    pub fn scan_identity(
        &self,
        plan: &ReadPlan,
        ctx: &ExecutionContext,
    ) -> Result<ScanReceipt, FileSourceError> {
        let mut reader = self.reader(plan, ctx)?;
        while reader.next_chunk()?.is_some() {}
        reader.finish()
    }

    /// Memory quote for holding `plan` resident, when the row count is recorded.
    ///
    /// # Errors
    ///
    /// The quote overflows `u64`.
    pub fn quote(&self, plan: &ReadPlan) -> Result<Option<MaterializationQuote>, FileSourceError> {
        self.declared_rows.map(|rows| MaterializationQuote::for_rows(rows, plan)).transpose()
    }

    /// Read `plan` into one resident table.
    ///
    /// When the row count is recorded, the whole quote is admitted before the
    /// first byte is read, and a pool that cannot hold it is refused with the
    /// exact quote and the next attempt: stream a sufficient-statistics route,
    /// or raise the pool. The returned table holds its resident lease.
    ///
    /// # Errors
    ///
    /// The memory refusal above, and the errors of [`Self::reader`].
    pub fn materialize(
        &self,
        plan: &ReadPlan,
        ctx: &ExecutionContext,
    ) -> Result<MaterializedTable, FileSourceError> {
        let quote = self.quote(plan)?;
        if let Some(quote) = &quote {
            quote.admit(ctx.admission.available().bytes(), plan.chunk_rows)?;
        }
        let reader = self.reader(plan, ctx)?;
        materialize::materialize(reader, quote, ctx)
    }

    fn check_plan(&self, plan: &ReadPlan) -> Result<(), FileSourceError> {
        for (column, &index) in plan.columns.iter().zip(&plan.file_index) {
            if self.view.columns.get(index) != Some(column) {
                return Err(FileSourceError::InvalidOptions {
                    detail: format!("the plan column '{}' is not in this table", column.name),
                });
            }
        }
        Ok(())
    }
}

/// An Arrow C stream read once, in bounded chunks.
pub struct TableStream {
    view: SchemaView,
    reader: arrow_array::ffi_stream::ArrowArrayStreamReader,
}

impl fmt::Debug for TableStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableStream").field("columns", &self.view.columns).finish_non_exhaustive()
    }
}

impl TableStream {
    /// Import the schema of `stream`. No batch is read.
    ///
    /// # Errors
    ///
    /// The stream schema cannot be imported.
    pub fn new(stream: ArrowCStream) -> Result<Self, FileSourceError> {
        use arrow_array::RecordBatchReader;

        let reader = stream.into_reader()?;
        let schema = reader.schema();
        let width = schema.fields().len();
        Ok(Self { view: SchemaView::new(schema, vec![None; width], 0), reader })
    }

    /// Every column of the stream schema.
    #[must_use]
    pub fn columns(&self) -> &[FileColumn] {
        &self.view.columns
    }

    /// Plan a read of the columns `closure` requires.
    ///
    /// # Errors
    ///
    /// As [`TableFile::plan`].
    pub fn plan(
        &self,
        closure: &ColumnClosure,
        options: ReadOptions,
    ) -> Result<ReadPlan, FileSourceError> {
        self.view.plan(closure, options)
    }

    /// Chunked reader for `plan`. The stream is consumed and cannot be rewound.
    ///
    /// # Errors
    ///
    /// The plan does not belong to this schema.
    pub fn reader(
        self,
        plan: &ReadPlan,
        ctx: &ExecutionContext,
    ) -> Result<TableChunkReader, FileSourceError> {
        let provider = provider::StreamBatches::new(self.reader, plan.projection.clone());
        TableChunkReader::new(Origin::Stream, plan.clone(), ctx, Box::new(provider))
    }
}

enum Origin {
    File { format: TableFileFormat, paths: Vec<PathBuf> },
    Stream,
}

struct Pending {
    batch: RecordBatch,
    offset: usize,
    held_bytes: u64,
    _leases: Vec<AdmissionLease>,
}

/// One bounded chunk of library-owned columns.
///
/// The chunk holds the admission lease that paid for its columns; dropping
/// the chunk returns it.
pub struct FileChunk {
    start: GlobalId,
    rows: usize,
    columns: Vec<OwnedColumn>,
    names: Arc<[Arc<str>]>,
    lease: AdmissionLease,
}

impl fmt::Debug for FileChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileChunk")
            .field("start", &self.start)
            .field("rows", &self.rows)
            .field("names", &self.names)
            .finish_non_exhaustive()
    }
}

impl FileChunk {
    /// Global id of the first row.
    #[must_use]
    pub const fn start(&self) -> GlobalId {
        self.start
    }

    /// Rows in the chunk.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Columns in plan order.
    #[must_use]
    pub fn columns(&self) -> &[OwnedColumn] {
        &self.columns
    }

    /// Column names in plan order.
    #[must_use]
    pub fn names(&self) -> &[Arc<str>] {
        &self.names
    }

    /// Column named `name`.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&OwnedColumn> {
        self.names
            .iter()
            .position(|candidate| &**candidate == name)
            .map(|index| &self.columns[index])
    }

    /// Bytes the chunk's lease still holds.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        self.lease.remainder().bytes()
    }

    /// Local offset of global row `row` in this chunk.
    ///
    /// # Errors
    ///
    /// The row is not inside this chunk.
    pub fn local_index(&self, row: GlobalId) -> Result<ChunkIndex, FileSourceError> {
        let index = chunk_local_index(self.start, row)?;
        if (index.get() as usize) < self.rows {
            Ok(index)
        } else {
            Err(FileSourceError::InvalidOptions {
                detail: format!("row {} is past the chunk at {}", row.get(), self.start.get()),
            })
        }
    }
}

/// What a complete scan read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScanReceipt {
    rows: GlobalId,
    chunks: u64,
    identity: FileContentIdentity,
    columns: Vec<FileColumn>,
    levels: Vec<Vec<Arc<str>>>,
    peak_chunk_bytes: u64,
}

impl ScanReceipt {
    /// Rows read.
    #[must_use]
    pub const fn rows(&self) -> GlobalId {
        self.rows
    }

    /// Chunks delivered.
    #[must_use]
    pub const fn chunks(&self) -> u64 {
        self.chunks
    }

    /// Full-content identity of the projected columns.
    #[must_use]
    pub const fn identity(&self) -> FileContentIdentity {
        self.identity
    }

    /// Columns read, in plan order.
    #[must_use]
    pub fn columns(&self) -> &[FileColumn] {
        &self.columns
    }

    /// Category labels of the column at plan position `position`, in domain order.
    #[must_use]
    pub fn levels(&self, position: usize) -> &[Arc<str>] {
        self.levels.get(position).map_or(&[], Vec::as_slice)
    }

    /// Largest bytes leased at once for one chunk (decoded batch plus owned columns).
    #[must_use]
    pub const fn peak_chunk_bytes(&self) -> u64 {
        self.peak_chunk_bytes
    }
}

/// Reads a planned table in bounded chunks, charging each to the admission pool.
pub struct TableChunkReader {
    origin: Origin,
    plan: ReadPlan,
    ctx: ExecutionContext,
    provider: Option<Box<dyn BatchProvider>>,
    states: Vec<ColumnState>,
    names: Arc<[Arc<str>]>,
    cursor: RowCursor,
    pending: Option<Pending>,
    chunks: u64,
    exhausted: bool,
    peak_chunk_bytes: u64,
    receipt: Option<ScanReceipt>,
}

impl fmt::Debug for TableChunkReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableChunkReader")
            .field("names", &self.names)
            .field("position", &self.cursor.position())
            .field("chunks", &self.chunks)
            .field("exhausted", &self.exhausted)
            .finish_non_exhaustive()
    }
}

impl TableChunkReader {
    fn new(
        origin: Origin,
        plan: ReadPlan,
        ctx: &ExecutionContext,
        provider: Box<dyn BatchProvider>,
    ) -> Result<Self, FileSourceError> {
        let mut states = Vec::with_capacity(plan.columns.len());
        for (position, column) in plan.columns.iter().enumerate() {
            let position = u32::try_from(position).map_err(|_| FileSourceError::CountOverflow {
                context: "projected column count",
            })?;
            states.push(ColumnState::new(column.clone(), position));
        }
        let names: Arc<[Arc<str>]> = plan.names().into();
        Ok(Self {
            origin,
            plan,
            ctx: ctx.clone(),
            provider: Some(provider),
            states,
            names,
            cursor: RowCursor::new(),
            pending: None,
            chunks: 0,
            exhausted: false,
            peak_chunk_bytes: 0,
            receipt: None,
        })
    }

    /// The plan being read.
    #[must_use]
    pub fn plan(&self) -> &ReadPlan {
        &self.plan
    }

    /// Global id of the next row.
    #[must_use]
    pub const fn position(&self) -> GlobalId {
        self.cursor.position()
    }

    /// Whether every row has been delivered.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Next chunk of at most [`ReadPlan::chunk_rows`] rows, or `None` at the end.
    ///
    /// The decode quote is leased before a batch is decoded, and the owned
    /// quote before its columns are built. A refused lease returns an error and
    /// leaves the pool and the read position unchanged.
    ///
    /// # Errors
    ///
    /// A memory refusal, a decode or conversion refusal, or cancellation.
    pub fn next_chunk(&mut self) -> Result<Option<FileChunk>, FileSourceError> {
        loop {
            if self.ctx.cancellation.is_cancelled() {
                self.pending = None;
                return Err(FileSourceError::Cancelled);
            }
            if self.exhausted {
                return Ok(None);
            }
            let Some(pending) = self.pending.as_ref() else {
                if !self.pull()? {
                    self.exhausted = true;
                    return Ok(None);
                }
                continue;
            };
            let total = pending.batch.num_rows();
            if pending.offset >= total {
                self.pending = None;
                continue;
            }
            let take = (total - pending.offset).min(self.plan.chunk_rows);
            let held = pending.held_bytes;
            let quote = self.plan.quote_rows(take)?;
            let lease = self.lease(MemoryStage::ChunkColumns, quote.owned_bytes)?;
            self.peak_chunk_bytes =
                self.peak_chunk_bytes.max(held.saturating_add(quote.owned_bytes));
            let Some(pending) = self.pending.as_mut() else {
                continue;
            };
            let slice = pending.batch.slice(pending.offset, take);
            pending.offset += take;
            return self.convert(&slice, take, lease).map(Some);
        }
    }

    fn pull(&mut self) -> Result<bool, FileSourceError> {
        let quote = self.plan.chunk_quote()?;
        let provider = self.provider.as_mut().ok_or(FileSourceError::NotRewindable)?;
        // A provider that can size its next batch from metadata is leased for
        // the whole decode (message body, inflated buffers, array structure)
        // before one byte is read, so a writer-sized batch is refused here and
        // never materialized.
        let (reserve, stage) = match provider.next_decode_need()? {
            DecodeNeed::Exhausted => return Ok(false),
            DecodeNeed::Bytes(need) if need > quote.decode_bytes => {
                (need, MemoryStage::DecodedBatch)
            }
            DecodeNeed::Bytes(_) | DecodeNeed::Unknown => {
                (quote.decode_bytes, MemoryStage::ChunkDecode)
            }
        };
        let decode = self.lease(stage, reserve)?;
        let provider = self.provider.as_mut().ok_or(FileSourceError::NotRewindable)?;
        let Some(batch) = provider.next_batch()? else {
            return Ok(false);
        };
        // Arrow reports a decoded column's whole backing allocation, so columns sliced from one
        // message body would each be charged for all of it. The sliced span is what they own.
        let actual = batch
            .columns()
            .iter()
            .map(|column| {
                column
                    .to_data()
                    .get_slice_memory_size()
                    .unwrap_or_else(|_| column.get_array_memory_size())
            })
            .sum::<usize>() as u64;
        let mut held = reserve;
        let mut leases = vec![decode];
        if actual > reserve {
            let extra = actual - reserve;
            leases.push(
                self.lease(MemoryStage::DecodedBatch, extra)
                    .map_err(|error| self.producer_remedy(error, actual))?,
            );
            held = actual;
        }
        self.pending = Some(Pending { batch, offset: 0, held_bytes: held, _leases: leases });
        Ok(true)
    }

    fn convert(
        &mut self,
        slice: &RecordBatch,
        rows: usize,
        lease: AdmissionLease,
    ) -> Result<FileChunk, FileSourceError> {
        let start = self.cursor.position();
        let mut columns = Vec::with_capacity(self.states.len());
        for (state, &position) in self.states.iter_mut().zip(&self.plan.batch_position) {
            columns.push(state.convert(slice.column(position).as_ref(), start)?);
        }
        self.cursor.advance(rows as u64)?;
        self.chunks += 1;
        Ok(FileChunk { start, rows, columns, names: Arc::clone(&self.names), lease })
    }

    /// An Arrow C stream's batch was sized by its producer, so the remedy names the producer.
    fn producer_remedy(&self, error: FileSourceError, batch_bytes: u64) -> FileSourceError {
        match (&self.origin, error) {
            (
                Origin::Stream,
                FileSourceError::MemoryRefused {
                    stage: MemoryStage::DecodedBatch,
                    required_bytes,
                    available_bytes,
                    chunk_rows,
                    ..
                },
            ) => FileSourceError::MemoryRefused {
                stage: MemoryStage::DecodedBatch,
                required_bytes,
                available_bytes,
                chunk_rows,
                next_attempt: format!(
                    "the producer sent a {batch_bytes}-byte batch; request batches of at most \
                     {chunk_rows} rows from the producer before export (for example pyarrow \
                     `to_reader(max_chunksize={chunk_rows})`), or raise the admission pool by {} \
                     bytes",
                    required_bytes.saturating_sub(available_bytes)
                ),
            },
            (_, error) => error,
        }
    }

    fn lease(&self, stage: MemoryStage, bytes: u64) -> Result<AdmissionLease, FileSourceError> {
        match self.ctx.acquire_admission(0, bytes) {
            Ok(lease) => Ok(lease),
            Err(AdmissionError::Cancelled) => Err(FileSourceError::Cancelled),
            Err(AdmissionError::ExceedsRemainder { remaining_bytes, .. }) => {
                Err(memory_refusal(stage, bytes, remaining_bytes, self.plan.chunk_rows))
            }
            Err(_) => Err(memory_refusal(
                stage,
                bytes,
                self.ctx.admission.available().bytes(),
                self.plan.chunk_rows,
            )),
        }
    }

    /// Start a new pass from the first row.
    ///
    /// Category codes assigned in an earlier pass are kept. The identity of
    /// the new pass is computed afresh.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::NotRewindable`] for an Arrow C stream, or the files
    /// cannot be reopened.
    pub fn rewind(&mut self) -> Result<(), FileSourceError> {
        let Origin::File { format, paths } = &self.origin else {
            return Err(FileSourceError::NotRewindable);
        };
        self.pending = None;
        self.provider = None;
        self.provider =
            Some(provider::open_all(*format, paths, &self.plan.projection, self.plan.chunk_rows));
        for state in &mut self.states {
            state.restart();
        }
        self.cursor = RowCursor::new();
        self.chunks = 0;
        self.exhausted = false;
        self.receipt = None;
        Ok(())
    }

    /// Receipt of the completed pass, with the content identity.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::ScanIncomplete`] before the last chunk was read.
    pub fn finish(&mut self) -> Result<ScanReceipt, FileSourceError> {
        if let Some(receipt) = &self.receipt {
            return Ok(receipt.clone());
        }
        if !self.exhausted {
            return Err(FileSourceError::ScanIncomplete);
        }
        let rows = self.cursor.position();
        let digests: Vec<[u8; 32]> = self.states.iter_mut().map(ColumnState::take_digest).collect();
        let parts: Vec<ColumnDigest<'_>> = self
            .states
            .iter()
            .zip(&digests)
            .map(|(state, digest)| ColumnDigest {
                name: &state.column.name,
                kind: &state.column.kind,
                levels: state.levels(),
                digest: *digest,
            })
            .collect();
        let identity = FileContentIdentity::seal(rows, &parts);
        let receipt = ScanReceipt {
            rows,
            chunks: self.chunks,
            identity,
            columns: self.plan.columns.clone(),
            levels: self.states.iter().map(|state| state.levels().to_vec()).collect(),
            peak_chunk_bytes: self.peak_chunk_bytes,
        };
        self.receipt = Some(receipt.clone());
        Ok(receipt)
    }

    pub(crate) fn final_domain(
        &mut self,
        position: usize,
    ) -> Result<Option<Arc<crate::categorical::CategoryDomain>>, FileSourceError> {
        match self.states.get_mut(position) {
            Some(state) => state.domain(),
            None => Ok(None),
        }
    }
}

pub(crate) fn memory_refusal(
    stage: MemoryStage,
    required_bytes: u64,
    available_bytes: u64,
    chunk_rows: usize,
) -> FileSourceError {
    let deficit = required_bytes.saturating_sub(available_bytes);
    let next_attempt = match stage {
        MemoryStage::ChunkDecode | MemoryStage::ChunkColumns => format!(
            "lower chunk_rows below {chunk_rows}, or raise the admission pool by {deficit} bytes"
        ),
        MemoryStage::DecodedBatch => format!(
            "rewrite the file with smaller record batches or row groups, or raise the admission \
             pool by {deficit} bytes"
        ),
        MemoryStage::Materialization => format!(
            "stream a sufficient-statistics summary route, which holds one chunk at a time, or \
             raise the admission pool to at least {required_bytes} bytes"
        ),
    };
    FileSourceError::MemoryRefused {
        stage,
        required_bytes,
        available_bytes,
        chunk_rows,
        next_attempt,
    }
}
