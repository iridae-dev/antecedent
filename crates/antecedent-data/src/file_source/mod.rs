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

mod format;
mod plan;
mod schema;
mod chunks;

pub use self::format::TableFileFormat;
pub use self::format::FileColumnKind;
pub use self::format::FileColumn;
pub use self::plan::ReadOptions;
pub use self::plan::ChunkQuote;
pub use self::plan::ReadPlan;
pub use self::plan::RowCursor;
pub use self::plan::chunk_local_index;
pub use self::chunks::TableStream;
pub use self::chunks::FileChunk;
pub use self::chunks::ScanReceipt;
pub use self::chunks::TableChunkReader;

use self::schema::{SchemaView, list_dataset, same_schema};
use self::chunks::{Origin};

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
