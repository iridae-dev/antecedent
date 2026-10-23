//! Refusals of the file-backed table source.
//!
//! Every refusal that a caller can act on names its next attempt. A memory
//! refusal carries the exact bytes that were quoted and the bytes the
//! admission pool still had, so the caller can lower the chunk, stream a
//! summary route, or raise the pool by a known amount.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::path::PathBuf;

use crate::column_closure::ClosureError;
use crate::error::DataError;
use crate::global_index::GlobalId;

/// What the reader was about to allocate when the admission pool refused it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum MemoryStage {
    /// Decoding one bounded chunk from the file.
    ChunkDecode,
    /// A decoded record batch larger than the decode quote. Arrow IPC batches
    /// keep the size their writer chose.
    DecodedBatch,
    /// Library-owned columns of one chunk.
    ChunkColumns,
    /// Resident columns of a materialized table.
    Materialization,
}

impl MemoryStage {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChunkDecode => "chunk_decode",
            Self::DecodedBatch => "decoded_batch",
            Self::ChunkColumns => "chunk_columns",
            Self::Materialization => "materialization",
        }
    }
}

/// How a caller should treat a [`FileSourceError`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum FileSourceErrorKind {
    /// The file could not be opened, decoded, or does not match its dataset.
    Data,
    /// The request names a column, type, value, or option the source refuses.
    InvalidArgument,
    /// The admission pool cannot hold the quoted bytes.
    Resource,
    /// The execution context was cancelled.
    Cancelled,
}

/// A file-backed table source refused to open, plan, or read.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FileSourceError {
    /// The operating system refused to open or read a path.
    Io {
        /// Path that was being read.
        path: PathBuf,
        /// Operating-system message.
        message: String,
    },
    /// The bytes are not a readable Parquet or Arrow IPC table.
    Format {
        /// Path that was being decoded.
        path: PathBuf,
        /// Decoder message.
        message: String,
    },
    /// The extension does not name a supported format.
    UnknownFormat {
        /// Path whose format could not be inferred.
        path: PathBuf,
    },
    /// The reader for this format was not compiled into this build.
    FormatUnavailable {
        /// Format name.
        format: &'static str,
        /// Cargo feature that compiles it.
        feature: &'static str,
    },
    /// A dataset directory holds no Parquet file.
    EmptyDataset {
        /// Directory that was listed.
        path: PathBuf,
    },
    /// A dataset file disagrees with the first file, or a column name repeats.
    SchemaMismatch {
        /// File that disagreed.
        path: PathBuf,
        /// What disagreed.
        detail: String,
    },
    /// A required column is not in the file.
    UnknownColumn {
        /// Requested name.
        column: String,
        /// Names the file has.
        available: Vec<String>,
    },
    /// A required column has a type that cannot be read without a lossy cast.
    UnsupportedColumn {
        /// Column name.
        column: String,
        /// Arrow type of the column.
        arrow_type: String,
        /// What to do instead.
        next_attempt: &'static str,
    },
    /// A value cannot be kept exactly in a library-owned column.
    UnsupportedValue {
        /// Column name.
        column: String,
        /// Global row of the value.
        row: GlobalId,
        /// What was refused.
        detail: String,
        /// What to do instead.
        next_attempt: &'static str,
    },
    /// A required column holds a missing value and the route declared no missing-data policy.
    MissingValue {
        /// Column name.
        column: String,
        /// Global row of the missing cell.
        row: GlobalId,
        /// What to do instead.
        next_attempt: &'static str,
    },
    /// The required-column closure refused the projection.
    Closure(ClosureError),
    /// Read options are not usable.
    InvalidOptions {
        /// What was refused.
        detail: String,
    },
    /// The admission pool cannot hold the quoted bytes. Nothing was allocated
    /// for the refused stage and the pool is unchanged.
    MemoryRefused {
        /// What was about to be allocated.
        stage: MemoryStage,
        /// Bytes quoted for that stage.
        required_bytes: u64,
        /// Bytes the pool still had.
        available_bytes: u64,
        /// Rows per chunk of the plan.
        chunk_rows: usize,
        /// What to do instead.
        next_attempt: String,
    },
    /// A row or byte count does not fit in `u64`.
    CountOverflow {
        /// Which count overflowed.
        context: &'static str,
    },
    /// The identity is issued only after a complete first scan.
    ScanIncomplete,
    /// An Arrow C stream is read once and cannot be rewound.
    NotRewindable,
    /// The execution context was cancelled. Chunk leases were returned.
    Cancelled,
    /// A library-owned column refused the converted values.
    Data(DataError),
}

impl FileSourceError {
    /// How a caller should treat this refusal.
    #[must_use]
    pub const fn kind(&self) -> FileSourceErrorKind {
        match self {
            Self::Io { .. }
            | Self::Format { .. }
            | Self::EmptyDataset { .. }
            | Self::SchemaMismatch { .. }
            | Self::Data(_) => FileSourceErrorKind::Data,
            Self::MemoryRefused { .. } | Self::CountOverflow { .. } => {
                FileSourceErrorKind::Resource
            }
            Self::Cancelled => FileSourceErrorKind::Cancelled,
            Self::UnknownFormat { .. }
            | Self::FormatUnavailable { .. }
            | Self::UnknownColumn { .. }
            | Self::UnsupportedColumn { .. }
            | Self::UnsupportedValue { .. }
            | Self::MissingValue { .. }
            | Self::Closure(_)
            | Self::InvalidOptions { .. }
            | Self::ScanIncomplete
            | Self::NotRewindable => FileSourceErrorKind::InvalidArgument,
        }
    }

    /// The next attempt this refusal names, when it names one.
    #[must_use]
    pub fn next_attempt(&self) -> Option<String> {
        match self {
            Self::UnsupportedColumn { next_attempt, .. }
            | Self::UnsupportedValue { next_attempt, .. }
            | Self::MissingValue { next_attempt, .. } => Some((*next_attempt).to_owned()),
            Self::MemoryRefused { next_attempt, .. } => Some(next_attempt.clone()),
            Self::UnknownFormat { .. } => {
                Some("name the format: parquet, arrow_ipc_file, or arrow_ipc_stream".to_owned())
            }
            Self::FormatUnavailable { feature, .. } => {
                Some(format!("build antecedent-data with the {feature} feature"))
            }
            Self::Closure(ClosureError::Refused(refusal)) => {
                Some(format!("retain column '{}' in the projection", refusal.retain))
            }
            Self::ScanIncomplete => Some("read every chunk before asking for the identity".into()),
            Self::NotRewindable => {
                Some("open the table from a file, which can be read again".to_owned())
            }
            _ => None,
        }
    }
}

impl fmt::Display for FileSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "cannot read {}: {message}", path.display()),
            Self::Format { path, message } => {
                write!(f, "{} is not a readable table file: {message}", path.display())
            }
            Self::UnknownFormat { path } => write!(
                f,
                "cannot infer the table format of {}; next attempt: name the format: parquet, \
                 arrow_ipc_file, or arrow_ipc_stream",
                path.display()
            ),
            Self::FormatUnavailable { format, feature } => write!(
                f,
                "the {format} reader is not compiled into this build; next attempt: build \
                 antecedent-data with the {feature} feature"
            ),
            Self::EmptyDataset { path } => {
                write!(f, "dataset directory {} holds no .parquet file", path.display())
            }
            Self::SchemaMismatch { path, detail } => {
                write!(f, "{} does not match the table schema: {detail}", path.display())
            }
            Self::UnknownColumn { column, available } => write!(
                f,
                "required column '{column}' is not in the file; available: {}",
                available.join(", ")
            ),
            Self::UnsupportedColumn { column, arrow_type, next_attempt } => write!(
                f,
                "column '{column}' has type {arrow_type}, which cannot be read without a lossy \
                 cast; next attempt: {next_attempt}"
            ),
            Self::UnsupportedValue { column, row, detail, next_attempt } => write!(
                f,
                "column '{column}' row {}: {detail}; next attempt: {next_attempt}",
                row.get()
            ),
            Self::MissingValue { column, row, next_attempt } => write!(
                f,
                "column '{column}' row {} is missing and no missing-data policy was declared; \
                 next attempt: {next_attempt}",
                row.get()
            ),
            Self::Closure(error) => write!(f, "required-column closure: {error}"),
            Self::InvalidOptions { detail } => write!(f, "invalid read options: {detail}"),
            Self::MemoryRefused {
                stage,
                required_bytes,
                available_bytes,
                chunk_rows,
                next_attempt,
            } => write!(
                f,
                "{} needs {required_bytes} bytes and the admission pool holds \
                 {available_bytes} (chunk of {chunk_rows} rows); next attempt: {next_attempt}",
                stage.as_str()
            ),
            Self::CountOverflow { context } => write!(f, "{context} does not fit in u64"),
            Self::ScanIncomplete => f.write_str(
                "the content identity is issued only after a complete first scan; next attempt: \
                 read every chunk before asking for the identity",
            ),
            Self::NotRewindable => f.write_str(
                "an Arrow C stream is read once and cannot be rewound; next attempt: open the \
                 table from a file, which can be read again",
            ),
            Self::Cancelled => f.write_str("the file read was cancelled"),
            Self::Data(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FileSourceError {}

impl From<DataError> for FileSourceError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

impl From<ClosureError> for FileSourceError {
    fn from(error: ClosureError) -> Self {
        Self::Closure(error)
    }
}
