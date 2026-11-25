//! Observation masks for file-backed cohort checks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::RowMask;
use arrow_array::Array;
use arrow_array::cast::AsArray as _;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, TimestampNanosecondType};
use arrow_schema::{DataType, TimeUnit};

use super::{FileSourceError, MemoryStage, TableChunkReader, admit_file_bytes};
use crate::column_closure::{ColumnRole, ObservationFrame};
use crate::error::DataError;

#[derive(Default)]
pub(super) struct MaskBuilder {
    bytes: Vec<u8>,
    rows: usize,
}

impl MaskBuilder {
    pub(super) fn append(
        &mut self,
        rows: usize,
        mut observed: impl FnMut(usize) -> bool,
    ) -> Result<(), FileSourceError> {
        let first = self.rows;
        self.rows = first
            .checked_add(rows)
            .ok_or(FileSourceError::CountOverflow { context: "cohort row count" })?;
        self.bytes.resize(self.rows.div_ceil(8), 0);
        for row in (0..rows).filter(|&row| observed(row)) {
            let offset = first + row;
            self.bytes[offset / 8] |= 1 << (offset % 8);
        }
        Ok(())
    }

    fn finish(self) -> Result<RowMask, FileSourceError> {
        RowMask::from_bytes(self.bytes, self.rows)
            .map_err(|_| DataError::mismatch("cohort validity buffer").into())
    }
}

pub(super) fn decoded_masks(
    mut reader: TableChunkReader,
    funded: bool,
) -> Result<Vec<MaskBuilder>, FileSourceError> {
    let mut masks: Vec<MaskBuilder> =
        (0..reader.plan.columns().len()).map(|_| MaskBuilder::default()).collect();
    let mut leases = Vec::new();
    loop {
        if reader.ctx.cancellation.is_cancelled() {
            return Err(FileSourceError::Cancelled);
        }
        if !reader.pull()? {
            break;
        }
        let Some(batch) = reader.pending.take() else { break };
        let rows = batch.batch.num_rows();
        if !funded {
            let bytes = u64::try_from(rows.div_ceil(8))
                .ok()
                .and_then(|bytes| bytes.checked_mul(masks.len() as u64))
                .ok_or(FileSourceError::CountOverflow { context: "cohort validity bytes" })?;
            leases.push(admit_file_bytes(
                &reader.ctx,
                MemoryStage::ChunkColumns,
                bytes,
                reader.plan.chunk_rows(),
            )?);
        }
        for (mask, &position) in masks.iter_mut().zip(&reader.plan.batch_position) {
            append_array(mask, batch.batch.column(position).as_ref())?;
        }
    }
    Ok(masks)
}

pub(super) fn finish(
    names: impl IntoIterator<Item = ColumnRole>,
    masks: Vec<MaskBuilder>,
) -> Result<ObservationFrame, FileSourceError> {
    let columns = names
        .into_iter()
        .zip(masks)
        .map(|(name, mask)| mask.finish().map(|mask| (name, mask)))
        .collect::<Result<Vec<_>, _>>()?;
    ObservationFrame::try_from_masks(columns).map_err(Into::into)
}

fn append_array(mask: &mut MaskBuilder, array: &dyn Array) -> Result<(), FileSourceError> {
    match array.data_type() {
        DataType::Float16 => {
            let values = array.as_primitive::<Float16Type>();
            mask.append(values.len(), |row| values.is_valid(row) && !values.value(row).is_nan())
        }
        DataType::Float32 => {
            let values = array.as_primitive::<Float32Type>();
            mask.append(values.len(), |row| values.is_valid(row) && !values.value(row).is_nan())
        }
        DataType::Float64 => {
            let values = array.as_primitive::<Float64Type>();
            mask.append(values.len(), |row| values.is_valid(row) && !values.value(row).is_nan())
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let values = array.as_primitive::<TimestampNanosecondType>();
            mask.append(values.len(), |row| values.is_valid(row) && values.value(row) != i64::MIN)
        }
        _ => {
            let validity = array.logical_nulls();
            mask.append(array.len(), |row| validity.as_ref().is_none_or(|v| v.is_valid(row)))
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::io::{Read, Seek, SeekFrom};
    use std::ops::Range;

    pub(in super::super) struct TracedReader<R> {
        inner: R,
        read: Vec<Range<u64>>,
    }

    impl<R> TracedReader<R> {
        pub(in super::super) fn new(inner: R) -> Self {
            Self { inner, read: Vec::new() }
        }

        pub(in super::super) fn read_intersects(&self, range: Range<u64>) -> bool {
            self.read.iter().any(|r| r.start < range.end && range.start < r.end)
        }
    }

    impl<R: Read + Seek> Read for TracedReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let start = self.inner.stream_position()?;
            let n = self.inner.read(buf)?;
            if n != 0 {
                self.read.push(start..start + n as u64);
            }
            Ok(n)
        }
    }

    impl<R: Seek> Seek for TracedReader<R> {
        fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(from)
        }
    }
}
