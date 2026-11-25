//! Definition-level-only reads for scalar Parquet columns.
//!
//! In data page V2, definition levels precede the independently compressed values.
//! Page V1 compresses them together, so a mixed-null V1 column uses the Arrow decoder.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use antecedent_core::ExecutionContext;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::statistics::Statistics;

use super::provider::{format_error, open_file};
use super::validity::MaskBuilder;
use super::{FileColumnKind, FileSourceError, MemoryStage, ReadPlan, admit_file_bytes};

pub(super) fn scan(
    paths: &[impl AsRef<Path>],
    plan: &ReadPlan,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<MaskBuilder>>, FileSourceError> {
    let mut masks = plan
        .columns()
        .iter()
        .map(|column| {
            matches!(
                column.kind,
                FileColumnKind::Int64
                    | FileColumnKind::Boolean
                    | FileColumnKind::Categorical { .. }
            )
            .then(MaskBuilder::default)
        })
        .collect::<Vec<_>>();
    for path in paths.iter().map(AsRef::as_ref) {
        if masks.iter().all(Option::is_none) {
            break;
        }
        let mut file = open_file(path)?;
        let arrow = ParquetRecordBatchReaderBuilder::try_new(
            file.try_clone().map_err(|e| format_error(path, &e))?,
        )
        .map_err(|error| format_error(path, &error))?;
        let schema = arrow.parquet_schema();
        let metadata = arrow.metadata();
        for (&root, mask) in plan.file_index.iter().zip(&mut masks) {
            let Some(builder) = mask else { continue };
            let mut leaves =
                (0..schema.num_columns()).filter(|&leaf| schema.get_column_root_idx(leaf) == root);
            let Some(leaf) = leaves.next() else {
                *mask = None;
                continue;
            };
            if leaves.next().is_some() || schema.column(leaf).max_rep_level() != 0 {
                *mask = None;
                continue;
            }
            let definition = schema.column(leaf).max_def_level();
            for group in metadata.row_groups() {
                if ctx.cancellation.is_cancelled() {
                    return Err(FileSourceError::Cancelled);
                }
                let rows = usize::try_from(group.num_rows())
                    .map_err(|_| format_error(path, &"invalid Parquet group row count"))?;
                let column = group.column(leaf);
                let nulls = column.statistics().and_then(Statistics::null_count_opt);
                if definition == 0 || nulls == Some(0) {
                    builder.append(rows, |_| true)?;
                } else if nulls == Some(rows as u64) {
                    builder.append(rows, |_| false)?;
                } else if definition != 1
                    || !append_v2(path, &mut file, column, rows, builder, ctx, plan.chunk_rows())?
                {
                    *mask = None;
                    break;
                }
            }
        }
    }
    Ok(masks)
}

fn append_v2(
    path: &Path,
    reader: &mut (impl Read + Seek),
    column: &ColumnChunkMetaData,
    rows: usize,
    mask: &mut MaskBuilder,
    ctx: &ExecutionContext,
    chunk_rows: usize,
) -> Result<bool, FileSourceError> {
    let bad = |detail: &str| format_error(path, &detail);
    let mut position = u64::try_from(column.data_page_offset())
        .map_err(|_| bad("negative Parquet data page offset"))?;
    let start = u64::try_from(column.dictionary_page_offset().unwrap_or(column.data_page_offset()))
        .map_err(|_| bad("negative Parquet column offset"))?;
    let end = u64::try_from(column.compressed_size())
        .ok()
        .and_then(|bytes| start.checked_add(bytes))
        .ok_or_else(|| bad("invalid Parquet column bounds"))?;
    let mut remaining = rows;
    while remaining > 0 && position < end {
        if ctx.cancellation.is_cancelled() {
            return Err(FileSourceError::Cancelled);
        }
        reader.seek(SeekFrom::Start(position)).map_err(|error| format_error(path, &error))?;
        let (header, body) = read_header(path, reader, position, end)?;
        let Some(page) = header.v2 else { return Ok(false) };
        position = body
            .checked_add(header.body_len)
            .filter(|&next| next <= end)
            .ok_or_else(|| bad("Parquet page exceeds its column bounds"))?;
        if page.rows > remaining || page.nulls > page.rows || page.levels as u64 > header.body_len {
            return Err(bad("invalid Parquet V2 scalar page"));
        }
        match page.nulls {
            0 => mask.append(page.rows, |_| true)?,
            nulls if nulls == page.rows => mask.append(page.rows, |_| false)?,
            _ => {
                let _lease = admit_file_bytes(
                    ctx,
                    MemoryStage::ChunkDecode,
                    page.levels as u64,
                    chunk_rows,
                )?;
                let mut levels = vec![0; page.levels];
                reader.read_exact(&mut levels).map_err(|error| format_error(path, &error))?;
                append_levels(path, &levels, page.rows, mask)?;
            }
        }
        remaining -= page.rows;
    }
    if remaining != 0 {
        return Err(bad("Parquet pages did not cover the declared rows"));
    }
    Ok(true)
}

fn append_levels(
    path: &Path,
    mut bytes: &[u8],
    mut rows: usize,
    mask: &mut MaskBuilder,
) -> Result<(), FileSourceError> {
    let bad = || format_error(path, &"invalid Parquet V2 definition levels");
    while rows != 0 {
        let header = read_varint(&mut bytes).map_err(|_| bad())?;
        let count = usize::try_from(header >> 1).map_err(|_| bad())?;
        if count == 0 {
            return Err(bad());
        }
        if header & 1 == 0 {
            let Some((&flag, tail)) = bytes.split_first() else { return Err(bad()) };
            if count > rows || flag > 1 {
                return Err(bad());
            }
            mask.append(count, |_| flag == 1)?;
            bytes = tail;
            rows -= count;
        } else {
            let (packed, tail) = bytes.split_at_checked(count).ok_or_else(bad)?;
            let used = count.checked_mul(8).ok_or_else(bad)?.min(rows);
            mask.append(used, |row| packed[row / 8] & (1 << (row % 8)) != 0)?;
            bytes = tail;
            rows -= used;
        }
    }
    Ok(())
}

struct PageHeader {
    body_len: u64,
    v2: Option<DataPageV2>,
}

struct DataPageV2 {
    rows: usize,
    nulls: usize,
    levels: usize,
}

// The reader visits only the Compact Protocol fields used by a Parquet page header.
// Binary statistics are skipped with a seek so even very large min/max labels stay unread.
fn read_header(
    path: &Path,
    reader: &mut (impl Read + Seek),
    start: u64,
    end: u64,
) -> Result<(PageHeader, u64), FileSourceError> {
    let mut input = CompactHeader { reader, remaining: end - start };
    let map = |error| format_error(path, &error);
    let mut fields = 0;
    let mut kind = None;
    let mut body_len = None;
    let mut v2 = None;
    while let Some((field, tag)) = input.field(&mut fields).map_err(map)? {
        match (field, tag) {
            (1, 5) => {
                kind = Some(input.nonnegative().map_err(map)?);
                if kind != Some(3) {
                    return Ok((PageHeader { body_len: 0, v2: None }, start));
                }
            }
            (3, 5) => body_len = Some(input.nonnegative().map_err(map)?),
            (8, 12) => v2 = Some(input.page_v2().map_err(map)?),
            (_, tag) => input.skip(tag, 0).map_err(map)?,
        }
    }
    let body_len = body_len.ok_or_else(|| format_error(path, &"missing Parquet page size"))?;
    if kind != Some(3) || v2.is_none() {
        return Err(format_error(path, &"invalid Parquet V2 header"));
    }
    Ok((PageHeader { body_len, v2 }, end - input.remaining))
}

struct CompactHeader<R> {
    reader: R,
    remaining: u64,
}

impl<R: Read + Seek> CompactHeader<R> {
    fn field(&mut self, preceding: &mut i64) -> std::io::Result<Option<(i64, u8)>> {
        let mut field = [0];
        self.read_exact(&mut field)?;
        if field[0] == 0 {
            return Ok(None);
        }
        let delta = field[0] >> 4;
        *preceding = if delta == 0 {
            signed(read_varint(self)?)
        } else {
            preceding.checked_add(i64::from(delta)).ok_or_else(invalid_header)?
        };
        Ok(Some((*preceding, field[0] & 15)))
    }

    fn nonnegative(&mut self) -> std::io::Result<u64> {
        u64::try_from(signed(read_varint(self)?)).map_err(|_| invalid_header())
    }

    fn page_v2(&mut self) -> std::io::Result<DataPageV2> {
        let mut fields = 0;
        let (mut values, mut rows, mut nulls, mut levels, mut repetitions) =
            (None, None, None, None, None);
        while let Some((field, tag)) = self.field(&mut fields)? {
            let at = match (field, tag) {
                (1, 5) => &mut values,
                (2, 5) => &mut nulls,
                (3, 5) => &mut rows,
                (5, 5) => &mut levels,
                (6, 5) => &mut repetitions,
                (_, tag) => {
                    self.skip(tag, 0)?;
                    continue;
                }
            };
            *at = Some(usize::try_from(self.nonnegative()?).map_err(|_| invalid_header())?);
        }
        match (values, rows, nulls, levels, repetitions) {
            (Some(values), Some(rows), Some(nulls), Some(levels), Some(0)) if values == rows => {
                Ok(DataPageV2 { rows, nulls, levels })
            }
            _ => Err(invalid_header()),
        }
    }

    fn skip(&mut self, tag: u8, depth: u8) -> std::io::Result<()> {
        match tag {
            1 | 2 => Ok(()),
            3 => self.advance(1),
            4..=6 => read_varint(self).map(drop),
            7 => self.advance(8),
            8 => {
                let bytes = read_varint(self)?;
                self.advance(bytes)
            }
            12 if depth < 8 => {
                let mut fields = 0;
                while let Some((_, tag)) = self.field(&mut fields)? {
                    self.skip(tag, depth + 1)?;
                }
                Ok(())
            }
            _ => Err(invalid_header()),
        }
    }

    fn advance(&mut self, bytes: u64) -> std::io::Result<()> {
        self.remaining = self.remaining.checked_sub(bytes).ok_or_else(invalid_header)?;
        self.reader.seek(SeekFrom::Current(i64::try_from(bytes).map_err(|_| invalid_header())?))?;
        Ok(())
    }
}

impl<R: Read> Read for CompactHeader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let max = buffer.len().min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
        let count = self.reader.read(&mut buffer[..max])?;
        self.remaining -= count as u64;
        Ok(count)
    }
}

fn read_varint(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let mut byte = [0];
        reader.read_exact(&mut byte)?;
        let bits = u64::from(byte[0] & 0x7f);
        if shift == 63 && bits > 1 {
            return Err(invalid_header());
        }
        value |= bits << shift;
        if byte[0] < 0x80 {
            return Ok(value);
        }
    }
    Err(invalid_header())
}

fn signed(value: u64) -> i64 {
    let magnitude = (value >> 1).cast_signed();
    if value & 1 == 0 { magnitude } else { !magnitude }
}

fn invalid_header() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid Parquet compact page header")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch, StringArray};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};

    use super::*;
    use crate::column_closure::ColumnRole;
    use crate::file_source::validity::{finish, tests::TracedReader};
    use crate::file_source::{ReadOptions, TableFile};

    #[test]
    fn mixed_parquet_cohort_skips_v2_value_bytes_even_when_compressed() {
        let path = std::env::temp_dir()
            .join(format!("antecedent-validity-v2-{}.parquet", std::process::id()));
        let text = "repeated large value ".repeat(20_000);
        let batch = RecordBatch::try_from_iter([
            ("x", Arc::new(Float64Array::from(vec![1.0, 2.0, f64::NAN, 4.0])) as _),
            (
                "label",
                Arc::new(StringArray::from(vec![Some(text.as_str()), None, Some("b"), Some("c")]))
                    as _,
            ),
        ])
        .unwrap();
        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_compression(Compression::SNAPPY)
            .build();
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let mut file = open_file(&path).unwrap();
        let arrow = ParquetRecordBatchReaderBuilder::try_new(file.try_clone().unwrap()).unwrap();
        let column = arrow.metadata().row_group(0).column(1);
        let start = u64::try_from(column.data_page_offset()).unwrap();
        let end = start + u64::try_from(column.compressed_size()).unwrap();
        file.seek(SeekFrom::Start(start)).unwrap();
        let (header, body) = read_header(&path, &mut file, start, end).unwrap();
        let def_len = header.v2.unwrap().levels as u64;
        let mut traced = TracedReader::new(file);
        let mut mask = MaskBuilder::default();
        let ctx = ExecutionContext::for_tests(1);
        assert!(append_v2(&path, &mut traced, column, 4, &mut mask, &ctx, 128).unwrap());
        assert!(traced.read_intersects(body..body + def_len));
        assert!(!traced.read_intersects(body + def_len..body + header.body_len));
        let role = ColumnRole::new("label");
        let frame = finish([role.clone()], vec![mask]).unwrap();
        assert_eq!(frame.complete_case(&[role]).unwrap().rows(), &[0, 2, 3]);

        let source = TableFile::open(&path).unwrap();
        let plan = source.plan_columns(&["label", "x"], ReadOptions::default()).unwrap();
        let frame = source.scan_observations(&plan, &ctx).unwrap();
        assert_eq!(
            frame.complete_case(&["label", "x"].map(ColumnRole::new)).unwrap().rows(),
            &[0, 3]
        );
        std::fs::remove_file(path).unwrap();
    }
}
