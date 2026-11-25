//! Read the validity buffers of eligible Arrow IPC columns by seeking over their values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{Read, Seek};
use std::path::Path;

use antecedent_core::ExecutionContext;
use arrow_ipc::{
    BodyCompression, Buffer, CompressionType, MetadataVersion, RecordBatch, root_as_footer,
    root_as_message,
};
use arrow_schema::{DataType, Schema, TimeUnit};

use super::ipc_scan::{read_at, read_footer_bytes, read_metadata, read_prefix};
use super::provider::{format_error, open_file};
use super::validity::{MaskBuilder, finish};
use super::{FileSourceError, MemoryStage, ReadPlan, TableFileFormat, admit_file_bytes};
use crate::column_closure::{ColumnRole, ObservationFrame};

pub(super) fn scan(
    path: &Path,
    format: TableFileFormat,
    schema: &Schema,
    plan: &ReadPlan,
    ctx: &ExecutionContext,
) -> Result<Option<ObservationFrame>, FileSourceError> {
    let mut file = open_file(path)?;
    let length = file
        .metadata()
        .map_err(|error| FileSourceError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?
        .len();
    scan_from(&mut file, path, length, format, schema, plan, ctx)
}

fn scan_from<R: Read + Seek>(
    file: &mut R,
    path: &Path,
    length: u64,
    format: TableFileFormat,
    schema: &Schema,
    plan: &ReadPlan,
    ctx: &ExecutionContext,
) -> Result<Option<ObservationFrame>, FileSourceError> {
    let Some(positions) = bitmap_positions(schema, plan) else { return Ok(None) };
    let mut masks: Vec<MaskBuilder> =
        (0..positions.len()).map(|_| MaskBuilder::default()).collect();
    let mut leases = Vec::new();
    let mut visit = |file: &mut R, offset: u64| -> Result<Option<u64>, FileSourceError> {
        if ctx.cancellation.is_cancelled() {
            return Err(FileSourceError::Cancelled);
        }
        let (meta_start, meta_len) = read_prefix(file, offset, length, path)?;
        if meta_len == 0 {
            return Ok(Some(length));
        }
        let _meta_lease =
            admit_file_bytes(ctx, MemoryStage::ChunkDecode, meta_len, plan.chunk_rows())?;
        let metadata = read_metadata(file, meta_start, meta_len, length, path)?;
        let message = root_as_message(&metadata).map_err(|error| format_error(path, &error))?;
        let (body, end) = body_range(path, meta_start, meta_len, message.bodyLength(), length)?;
        if let Some(found) = message.header_as_schema() {
            check_schema(path, schema, found)?;
        }
        let Some(batch) = message.header_as_record_batch() else { return Ok(Some(end)) };
        if message.version() < MetadataVersion::V4 {
            return Ok(None);
        }
        let body = BatchBody {
            path,
            ctx,
            chunk_rows: plan.chunk_rows(),
            start: body,
            end,
            compression: batch.compression(),
        };
        leases.push(append_batch(file, &body, batch, &positions, &mut masks)?);
        Ok(Some(end))
    };
    if format == TableFileFormat::ArrowIpcStream {
        let mut position = 0;
        while position + 4 <= length {
            let Some(end) = visit(file, position)? else { return Ok(None) };
            position = end;
        }
    } else {
        let footer = read_footer_bytes(file, length, path)?;
        let footer = root_as_footer(&footer).map_err(|error| format_error(path, &error))?;
        let found =
            footer.schema().ok_or_else(|| format_error(path, &"missing IPC footer schema"))?;
        check_schema(path, schema, found)?;
        if let Some(blocks) = footer.recordBatches() {
            for block in blocks {
                let offset = u64::try_from(block.offset())
                    .map_err(|_| format_error(path, &"negative IPC batch offset"))?;
                if visit(file, offset)?.is_none() {
                    return Ok(None);
                }
            }
        }
    }
    finish(plan.columns().iter().map(|column| ColumnRole::new(&column.name)), masks).map(Some)
}

type ColumnBuffers = (usize, usize, Option<MissingValue>);

fn append_batch(
    file: &mut (impl Read + Seek),
    body: &BatchBody<'_>,
    batch: RecordBatch<'_>,
    positions: &[ColumnBuffers],
    masks: &mut [MaskBuilder],
) -> Result<antecedent_core::execution::AdmissionLease, FileSourceError> {
    let (path, ctx, chunk_rows) = (body.path, body.ctx, body.chunk_rows);
    let rows = usize::try_from(batch.length())
        .map_err(|_| format_error(path, &"invalid IPC batch row count"))?;
    let nodes = batch.nodes().ok_or_else(|| format_error(path, &"missing IPC nodes"))?;
    let buffers = batch.buffers().ok_or_else(|| format_error(path, &"missing IPC buffers"))?;
    let mask_bytes = rows.div_ceil(8);
    let resident = (mask_bytes as u64)
        .checked_mul(positions.len() as u64)
        .ok_or(FileSourceError::CountOverflow { context: "IPC observation masks" })?;
    let lease = admit_file_bytes(ctx, MemoryStage::ChunkColumns, resident, chunk_rows)?;
    for (mask, &(node, buffer, missing)) in masks.iter_mut().zip(positions) {
        if node >= nodes.len() || buffer >= buffers.len() {
            return Err(format_error(path, &"IPC validity layout is incomplete"));
        }
        let node = nodes.get(node);
        let null_count = usize::try_from(node.null_count())
            .map_err(|_| format_error(path, &"invalid IPC null count"))?;
        if usize::try_from(node.length()).ok() != Some(rows) || null_count > rows {
            return Err(format_error(path, &"invalid IPC validity dimensions"));
        }
        if null_count == rows {
            mask.append(rows, |_| false)?;
            continue;
        }
        let _mask_lease =
            admit_file_bytes(ctx, MemoryStage::ChunkDecode, mask_bytes as u64, chunk_rows)?;
        let bitmap = if null_count == 0 {
            None
        } else {
            Some(body.read_buffer(file, buffers.get(buffer), mask_bytes)?)
        };
        let present =
            |row: usize| bitmap.as_ref().is_none_or(|bits| bits[row / 8] & (1 << (row % 8)) != 0);
        if let Some(missing) = missing {
            if buffer + 1 >= buffers.len() {
                return Err(format_error(path, &"IPC value layout is incomplete"));
            }
            let width = missing.width();
            let value_bytes = rows
                .checked_mul(width)
                .ok_or(FileSourceError::CountOverflow { context: "IPC sentinel values" })?;
            let _value_lease =
                admit_file_bytes(ctx, MemoryStage::ChunkDecode, value_bytes as u64, chunk_rows)?;
            let values = body.read_buffer(file, buffers.get(buffer + 1), value_bytes)?;
            mask.append(rows, |row| {
                present(row) && !missing.is_missing(&values[row * width..][..width])
            })?;
        } else {
            mask.append(rows, present)?;
        }
    }
    Ok(lease)
}

fn body_range(
    path: &Path,
    position: u64,
    metadata_len: u64,
    body_len: i64,
    limit: u64,
) -> Result<(u64, u64), FileSourceError> {
    let bounds = (|| {
        let start = position.checked_add(metadata_len)?;
        let end = start.checked_add(u64::try_from(body_len).ok()?)?;
        (end <= limit).then_some((start, end))
    })();
    bounds.ok_or_else(|| format_error(path, &"IPC buffer runs past its containing message"))
}

fn check_schema(
    path: &Path,
    schema: &Schema,
    found: arrow_ipc::Schema<'_>,
) -> Result<(), FileSourceError> {
    let received =
        arrow_ipc::convert::try_fb_to_schema(found).map_err(|error| format_error(path, &error))?;
    if found.endianness() != arrow_ipc::Endianness::Little || &received != schema {
        return Err(FileSourceError::SourceChanged {
            path: path.to_path_buf(),
            detail: "the observation scan schema differs from open".to_owned(),
        });
    }
    Ok(())
}

struct BatchBody<'a> {
    path: &'a Path,
    ctx: &'a ExecutionContext,
    chunk_rows: usize,
    start: u64,
    end: u64,
    compression: Option<BodyCompression<'a>>,
}

impl BatchBody<'_> {
    fn read_buffer(
        &self,
        reader: &mut (impl Read + Seek),
        buffer: &Buffer,
        expected: usize,
    ) -> Result<Vec<u8>, FileSourceError> {
        let (_, start) = body_range(self.path, self.start, 0, buffer.offset(), self.end)?;
        let (_, end) = body_range(self.path, start, 0, buffer.length(), self.end)?;
        if let Some(compression) = &self.compression {
            if end - start < 8 {
                return Err(format_error(self.path, &"IPC compressed buffer has no length prefix"));
            }
            let mut prefix = [0; 8];
            read_at(reader, start, &mut prefix, self.path)?;
            let expanded = i64::from_le_bytes(prefix);
            if expanded == -1 {
                return self.read_uncompressed(reader, start + 8, end, expected);
            }
            if usize::try_from(expanded).ok() != Some(expected) {
                return Err(format_error(self.path, &"invalid IPC uncompressed length"));
            }
            let len = end - start - 8;
            let _lease =
                admit_file_bytes(self.ctx, MemoryStage::ChunkDecode, len, self.chunk_rows)?;
            let mut compressed = vec![
                0;
                usize::try_from(len).map_err(|_| {
                    FileSourceError::CountOverflow { context: "compressed IPC buffer" }
                })?
            ];
            read_at(reader, start + 8, &mut compressed, self.path)?;
            let bytes = match compression.codec() {
                CompressionType::ZSTD => zstd::bulk::decompress(&compressed, expected),
                CompressionType::LZ4_FRAME => read_lz4(&compressed, expected),
                _ => return Err(format_error(self.path, &"unsupported IPC compression")),
            }
            .map_err(|error| format_error(self.path, &error))?;
            if bytes.len() != expected {
                return Err(format_error(self.path, &"decoded IPC buffer has the wrong length"));
            }
            Ok(bytes)
        } else {
            self.read_uncompressed(reader, start, end, expected)
        }
    }

    fn read_uncompressed(
        &self,
        reader: &mut (impl Read + Seek),
        start: u64,
        end: u64,
        expected: usize,
    ) -> Result<Vec<u8>, FileSourceError> {
        if end - start < expected as u64 {
            return Err(format_error(self.path, &"IPC observation buffer is too short"));
        }
        let mut bytes = vec![0; expected];
        read_at(reader, start, &mut bytes, self.path)?;
        Ok(bytes)
    }
}

fn read_lz4(compressed: &[u8], expected: usize) -> std::io::Result<Vec<u8>> {
    let mut decoder = lz4::Decoder::new(compressed)?;
    let mut bytes = vec![0; expected];
    decoder.read_exact(&mut bytes)?;
    if decoder.read(&mut [0])? != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "LZ4 buffer expands beyond its declared size",
        ));
    }
    let (_, result) = decoder.finish();
    result.map(|()| bytes)
}

#[derive(Clone, Copy)]
enum MissingValue {
    Float16,
    Float32,
    Float64,
    NaT,
}

impl MissingValue {
    fn width(self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 => 4,
            Self::Float64 | Self::NaT => 8,
        }
    }

    fn is_missing(self, bytes: &[u8]) -> bool {
        match self {
            Self::Float16 => u16::from_le_bytes([bytes[0], bytes[1]]) & 0x7fff > 0x7c00,
            Self::Float32 => f32::from_le_bytes(bytes.try_into().unwrap_or_default()).is_nan(),
            Self::Float64 => f64::from_le_bytes(bytes.try_into().unwrap_or_default()).is_nan(),
            Self::NaT => i64::from_le_bytes(bytes.try_into().unwrap_or_default()) == i64::MIN,
        }
    }
}

fn bitmap_positions(schema: &Schema, plan: &ReadPlan) -> Option<Vec<ColumnBuffers>> {
    let mut by_index = Vec::with_capacity(schema.fields().len());
    let mut nodes = 0;
    let mut buffers = 0;
    for field in schema.fields().iter().take(*plan.projection().last()? + 1) {
        let dtype = field.data_type();
        by_index.push((nodes, buffers));
        let layout = field_layout(dtype)?;
        nodes += layout.0;
        buffers += layout.1;
    }
    plan.file_index
        .iter()
        .map(|&index| {
            let kind = schema.field(index).data_type();
            let missing = match kind {
                DataType::Float16 => Some(MissingValue::Float16),
                DataType::Float32 => Some(MissingValue::Float32),
                DataType::Float64 => Some(MissingValue::Float64),
                DataType::Timestamp(TimeUnit::Nanosecond, _) => Some(MissingValue::NaT),
                DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Utf8
                | DataType::LargeUtf8 => None,
                _ => return None,
            };
            let (node, buffer) = by_index[index];
            Some((node, buffer, missing))
        })
        .collect()
}

fn field_layout(kind: &DataType) -> Option<(usize, usize)> {
    let leaf = match kind {
        DataType::Null => 0,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => 3,
        DataType::Boolean | DataType::Dictionary(_, _) | DataType::FixedSizeBinary(_) => 2,
        kind if kind.is_primitive() => 2,
        DataType::List(field) | DataType::LargeList(field) | DataType::Map(field, _) => {
            let (nodes, buffers) = field_layout(field.data_type())?;
            return Some((nodes + 1, buffers + 2));
        }
        DataType::FixedSizeList(field, _) => {
            let (nodes, buffers) = field_layout(field.data_type())?;
            return Some((nodes + 1, buffers + 1));
        }
        DataType::Struct(fields) => {
            return fields.iter().try_fold((1, 1), |(nodes, buffers), field| {
                let layout = field_layout(field.data_type())?;
                Some((nodes + layout.0, buffers + layout.1))
            });
        }
        _ => return None,
    };
    Some((1, leaf))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch, StringArray};
    use arrow_ipc::writer::{FileWriter, IpcWriteOptions};

    use super::*;
    use crate::column_closure::ClosureInputs;
    use crate::file_source::ReadOptions;
    use crate::file_source::schema::SchemaView;
    use crate::file_source::validity::tests::TracedReader;

    #[test]
    fn lz4_bitmap_checks_the_decoded_length() {
        use std::io::Write as _;

        let mut encoder = lz4::EncoderBuilder::new().build(Vec::new()).unwrap();
        let bitmap = [0b1010_1101; 1024];
        encoder.write_all(&bitmap).unwrap();
        let (compressed, result) = encoder.finish();
        result.unwrap();
        assert_eq!(read_lz4(&compressed, bitmap.len()).unwrap(), bitmap);
        assert!(read_lz4(&compressed, bitmap.len() - 1).is_err());
    }

    #[test]
    fn mixed_ipc_cohort_reads_float_and_bitmap_but_skips_label_values() {
        let path = Path::new("test.arrow");
        let text = "large value ".repeat(20_000);
        let batch = RecordBatch::try_from_iter([
            ("x", Arc::new(Float64Array::from(vec![1.0, 2.0, f64::NAN, 4.0])) as _),
            (
                "label",
                Arc::new(StringArray::from(vec![Some(text.as_str()), None, Some("b"), Some("c")]))
                    as _,
            ),
        ])
        .unwrap();
        let schema = batch.schema();
        let selected = ["label", "x"].map(ColumnRole::new);
        let closure = ClosureInputs::new(selected.clone()).close();
        let plan = SchemaView::new(Arc::clone(&schema), vec![None; 2], 0)
            .plan(&closure, ReadOptions::default())
            .unwrap();
        for compression in [None, Some(CompressionType::ZSTD)] {
            let options = IpcWriteOptions::default().try_with_compression(compression).unwrap();
            let mut bytes = Vec::new();
            let mut writer =
                FileWriter::try_new_with_options(&mut bytes, &schema, options).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
            drop(writer);
            let length = bytes.len() as u64;
            let mut cursor = Cursor::new(bytes);
            let footer = read_footer_bytes(&mut cursor, length, path).unwrap();
            let footer = root_as_footer(&footer).unwrap();
            let block = footer.recordBatches().unwrap().get(0);
            let (start, len) =
                read_prefix(&mut cursor, u64::try_from(block.offset()).unwrap(), length, path)
                    .unwrap();
            let metadata = read_metadata(&mut cursor, start, len, length, path).unwrap();
            let message = root_as_message(&metadata).unwrap();
            let buffers = message.header_as_record_batch().unwrap().buffers().unwrap();
            let span = |index| {
                let buffer = buffers.get(index);
                let at = start + len + u64::try_from(buffer.offset()).unwrap();
                at..at + u64::try_from(buffer.length()).unwrap()
            };
            let mut traced = TracedReader::new(cursor);
            let frame = scan_from(
                &mut traced,
                path,
                length,
                TableFileFormat::ArrowIpcFile,
                &schema,
                &plan,
                &ExecutionContext::for_tests(1),
            )
            .unwrap()
            .unwrap();
            assert_eq!(frame.complete_case(&selected).unwrap().rows(), &[0, 3]);
            assert!(traced.read_intersects(span(1)), "float values determine NaN");
            assert!(traced.read_intersects(span(2)), "nullable label validity determines cohort");
            assert!(!traced.read_intersects(span(3)), "label offsets are unnecessary");
            assert!(!traced.read_intersects(span(4)), "label value bytes never enter the reader");
        }
    }
}
