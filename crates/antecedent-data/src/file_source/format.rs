//! File formats and the column kinds a file schema maps to.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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

    pub(super) fn of(field: &arrow_schema::Field) -> Self {
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

pub(super) const fn dictionary_key_is_exact(key: &DataType) -> bool {
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

pub(super) const fn dictionary_value_is_label(value: &DataType) -> bool {
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
