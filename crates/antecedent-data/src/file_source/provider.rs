//! Record-batch providers for Parquet, Arrow IPC files and streams, and Arrow C streams.
//!
//! Arrow and Parquet types stay inside this module tree (ADR 0004). A provider
//! yields record batches of the projected columns only: Parquet decodes just
//! the projected root columns, at most `batch_rows` rows per batch; Arrow IPC
//! reads the projected columns of each record batch at the size its writer
//! chose, after the batch's message sizes are read from metadata so the caller
//! can lease them first ([`BatchProvider::next_decode_need`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs::File;
use std::path::{Path, PathBuf};

use arrow_array::RecordBatch;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_schema::SchemaRef;

use super::TableFileFormat;
use super::error::FileSourceError;

/// Bytes the next batch will hold while it is decoded, when the source knows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(not(feature = "ipc"), allow(dead_code, reason = "only the IPC provider sizes batches"))]
pub(crate) enum DecodeNeed {
    /// The source cannot size its next batch before decoding it.
    Unknown,
    /// No batch remains.
    Exhausted,
    /// Bytes the reader holds for the next batch: its message body, inflated
    /// buffers, metadata and per-array structure.
    Bytes(u64),
}

/// Source of decoded record batches.
pub(crate) trait BatchProvider {
    /// Size of the next batch, read from metadata without decoding it.
    ///
    /// A source that returns [`DecodeNeed::Bytes`] guarantees decoding the next
    /// batch allocates no more than that; the caller leases it first.
    fn next_decode_need(&mut self) -> Result<DecodeNeed, FileSourceError> {
        Ok(DecodeNeed::Unknown)
    }

    /// Next batch, or `None` when the source is exhausted.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>, FileSourceError>;
}

/// Schema and size facts read from a file without decoding its rows.
pub(crate) struct Inspected {
    pub schema: SchemaRef,
    /// Row count from the file footer, when the format records one.
    pub rows: Option<u64>,
    /// Largest uncompressed bytes per row of each top-level column, when the
    /// format records column sizes.
    pub decode_per_row: Vec<Option<u64>>,
}

pub(crate) fn inspect(path: &Path, format: TableFileFormat) -> Result<Inspected, FileSourceError> {
    match format {
        TableFileFormat::Parquet => parquet_inspect(path),
        TableFileFormat::ArrowIpcFile => ipc_inspect(path, false),
        TableFileFormat::ArrowIpcStream => ipc_inspect(path, true),
    }
}

/// Batches of `paths` in order, projected to the ascending file indices `projection`.
///
/// Each file is opened when its first batch is requested.
pub(crate) fn open_all(
    format: TableFileFormat,
    paths: &[PathBuf],
    projection: &[usize],
    batch_rows: usize,
) -> Box<dyn BatchProvider> {
    Box::new(Chained {
        format,
        paths: paths.to_vec(),
        projection: projection.to_vec(),
        batch_rows,
        next: 0,
        current: None,
    })
}

struct Chained {
    format: TableFileFormat,
    paths: Vec<PathBuf>,
    projection: Vec<usize>,
    batch_rows: usize,
    next: usize,
    current: Option<Box<dyn BatchProvider>>,
}

impl BatchProvider for Chained {
    fn next_decode_need(&mut self) -> Result<DecodeNeed, FileSourceError> {
        loop {
            if self.current.is_none() {
                let Some(path) = self.paths.get(self.next) else {
                    return Ok(DecodeNeed::Exhausted);
                };
                self.current =
                    Some(open_one(self.format, path, &self.projection, self.batch_rows)?);
                self.next += 1;
            }
            let Some(current) = self.current.as_mut() else {
                continue;
            };
            match current.next_decode_need()? {
                DecodeNeed::Exhausted => self.current = None,
                other => return Ok(other),
            }
        }
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, FileSourceError> {
        loop {
            if self.current.is_none() {
                let Some(path) = self.paths.get(self.next) else {
                    return Ok(None);
                };
                self.current =
                    Some(open_one(self.format, path, &self.projection, self.batch_rows)?);
                self.next += 1;
            }
            if let Some(current) = self.current.as_mut() {
                if let Some(batch) = current.next_batch()? {
                    return Ok(Some(batch));
                }
            }
            self.current = None;
        }
    }
}

fn open_one(
    format: TableFileFormat,
    path: &Path,
    projection: &[usize],
    batch_rows: usize,
) -> Result<Box<dyn BatchProvider>, FileSourceError> {
    match format {
        TableFileFormat::Parquet => parquet_open(path, projection, batch_rows),
        TableFileFormat::ArrowIpcFile => ipc_open(path, projection, false),
        TableFileFormat::ArrowIpcStream => ipc_open(path, projection, true),
    }
}

pub(crate) fn open_file(path: &Path) -> Result<File, FileSourceError> {
    File::open(path).map_err(|error| FileSourceError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

pub(crate) fn format_error(path: &Path, error: &impl std::fmt::Display) -> FileSourceError {
    FileSourceError::Format { path: path.to_path_buf(), message: error.to_string() }
}

#[cfg(feature = "parquet")]
fn parquet_inspect(path: &Path) -> Result<Inspected, FileSourceError> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let builder = ParquetRecordBatchReaderBuilder::try_new(open_file(path)?)
        .map_err(|error| format_error(path, &error))?;
    let schema = SchemaRef::clone(builder.schema());
    let metadata = builder.metadata();
    let rows = u64::try_from(metadata.file_metadata().num_rows())
        .map_err(|_| format_error(path, &"the footer records a negative row count"))?;
    let descriptor = builder.parquet_schema();
    let width = schema.fields().len();
    let mut decode_per_row: Vec<Option<u64>> = vec![None; width];
    for group in metadata.row_groups() {
        let Ok(group_rows) = u64::try_from(group.num_rows()) else {
            continue;
        };
        if group_rows == 0 {
            continue;
        }
        let mut bytes = vec![0_u64; width];
        for (leaf, chunk) in group.columns().iter().enumerate() {
            let root = descriptor.get_column_root_idx(leaf);
            if let Some(slot) = bytes.get_mut(root) {
                let stored = u64::try_from(chunk.uncompressed_size()).unwrap_or(0);
                // A dictionary- or run-encoded byte-array chunk expands when it
                // is decoded; the footer records the plain value bytes, and an
                // Arrow offset takes four more bytes per value.
                let expanded = chunk
                    .unencoded_byte_array_data_bytes()
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .zip(u64::try_from(chunk.num_values()).ok())
                    .map_or(0, |(bytes, values)| bytes.saturating_add(values.saturating_mul(4)));
                *slot = slot.saturating_add(stored.max(expanded));
            }
        }
        for (slot, total) in decode_per_row.iter_mut().zip(bytes) {
            let estimate = total.div_ceil(group_rows);
            *slot = Some(slot.map_or(estimate, |seen| seen.max(estimate)));
        }
    }
    Ok(Inspected { schema, rows: Some(rows), decode_per_row })
}

#[cfg(not(feature = "parquet"))]
fn parquet_inspect(_path: &Path) -> Result<Inspected, FileSourceError> {
    Err(TableFileFormat::Parquet.unavailable())
}

#[cfg(feature = "parquet")]
fn parquet_open(
    path: &Path,
    projection: &[usize],
    batch_rows: usize,
) -> Result<Box<dyn BatchProvider>, FileSourceError> {
    use parquet::arrow::ProjectionMask;
    use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

    struct ParquetBatches {
        reader: ParquetRecordBatchReader,
        path: PathBuf,
    }

    impl BatchProvider for ParquetBatches {
        fn next_batch(&mut self) -> Result<Option<RecordBatch>, FileSourceError> {
            self.reader.next().transpose().map_err(|error| format_error(&self.path, &error))
        }
    }

    let builder = ParquetRecordBatchReaderBuilder::try_new(open_file(path)?)
        .map_err(|error| format_error(path, &error))?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), projection.iter().copied());
    let reader = builder
        .with_projection(mask)
        .with_batch_size(batch_rows)
        .build()
        .map_err(|error| format_error(path, &error))?;
    Ok(Box::new(ParquetBatches { reader, path: path.to_path_buf() }))
}

#[cfg(not(feature = "parquet"))]
fn parquet_open(
    _path: &Path,
    _projection: &[usize],
    _batch_rows: usize,
) -> Result<Box<dyn BatchProvider>, FileSourceError> {
    Err(TableFileFormat::Parquet.unavailable())
}

#[cfg(feature = "ipc")]
fn ipc_inspect(path: &Path, stream: bool) -> Result<Inspected, FileSourceError> {
    use arrow_ipc::reader::{FileReader, StreamReader};
    use std::io::BufReader;

    let reader = BufReader::new(open_file(path)?);
    let schema = if stream {
        StreamReader::try_new(reader, None).map_err(|error| format_error(path, &error))?.schema()
    } else {
        FileReader::try_new(reader, None).map_err(|error| format_error(path, &error))?.schema()
    };
    let width = schema.fields().len();
    Ok(Inspected { schema, rows: None, decode_per_row: vec![None; width] })
}

#[cfg(not(feature = "ipc"))]
fn ipc_inspect(_path: &Path, stream: bool) -> Result<Inspected, FileSourceError> {
    let format =
        if stream { TableFileFormat::ArrowIpcStream } else { TableFileFormat::ArrowIpcFile };
    Err(format.unavailable())
}

#[cfg(feature = "ipc")]
fn ipc_open(
    path: &Path,
    projection: &[usize],
    stream: bool,
) -> Result<Box<dyn BatchProvider>, FileSourceError> {
    use arrow_ipc::reader::{FileReader, StreamReader};
    use std::io::BufReader;

    struct IpcBatches<R> {
        reader: R,
        path: PathBuf,
        costs: super::ipc_scan::IpcCosts,
        next: usize,
    }

    impl<R> BatchProvider for IpcBatches<R>
    where
        R: Iterator<Item = Result<RecordBatch, arrow_schema::ArrowError>>,
    {
        fn next_decode_need(&mut self) -> Result<DecodeNeed, FileSourceError> {
            let Some(cost) = self.costs.batches.get(self.next) else {
                return Ok(DecodeNeed::Exhausted);
            };
            cost.checked_add(self.costs.dictionaries)
                .map(DecodeNeed::Bytes)
                .ok_or(FileSourceError::CountOverflow { context: "ipc batch decode bytes" })
        }

        fn next_batch(&mut self) -> Result<Option<RecordBatch>, FileSourceError> {
            self.next += 1;
            self.reader.next().transpose().map_err(|error| format_error(&self.path, &error))
        }
    }

    let costs = if stream {
        super::ipc_scan::scan_stream(path)?
    } else {
        super::ipc_scan::scan_file(path)?
    };
    let reader = BufReader::new(open_file(path)?);
    let projection = Some(projection.to_vec());
    let path = path.to_path_buf();
    if stream {
        let reader = StreamReader::try_new(reader, projection)
            .map_err(|error| format_error(&path, &error))?;
        Ok(Box::new(IpcBatches { reader, path, costs, next: 0 }))
    } else {
        let reader =
            FileReader::try_new(reader, projection).map_err(|error| format_error(&path, &error))?;
        Ok(Box::new(IpcBatches { reader, path, costs, next: 0 }))
    }
}

#[cfg(not(feature = "ipc"))]
fn ipc_open(
    _path: &Path,
    _projection: &[usize],
    stream: bool,
) -> Result<Box<dyn BatchProvider>, FileSourceError> {
    let format =
        if stream { TableFileFormat::ArrowIpcStream } else { TableFileFormat::ArrowIpcFile };
    Err(format.unavailable())
}

/// Batches of an imported Arrow C stream, projected to `projection`.
pub(crate) struct StreamBatches {
    reader: ArrowArrayStreamReader,
    projection: Vec<usize>,
}

impl StreamBatches {
    pub(crate) fn new(reader: ArrowArrayStreamReader, projection: Vec<usize>) -> Self {
        Self { reader, projection }
    }
}

impl BatchProvider for StreamBatches {
    fn next_batch(&mut self) -> Result<Option<RecordBatch>, FileSourceError> {
        let origin = Path::new("<arrow c stream>");
        let Some(batch) = self.reader.next() else {
            return Ok(None);
        };
        let batch = batch.map_err(|error| format_error(origin, &error))?;
        batch.project(&self.projection).map(Some).map_err(|error| format_error(origin, &error))
    }
}
