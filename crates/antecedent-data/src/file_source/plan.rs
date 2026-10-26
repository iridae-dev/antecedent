//! Read options, chunk quotes, read plans and row cursors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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
    pub(super) closure: ColumnClosure,
    pub(super) columns: Vec<FileColumn>,
    /// Plan position to file column index.
    pub(super) file_index: Vec<usize>,
    /// Ascending file column indices the provider decodes.
    pub(super) projection: Vec<usize>,
    /// Plan position to the column position inside a projected batch.
    pub(super) batch_position: Vec<usize>,
    pub(super) chunk_rows: usize,
    pub(super) decode_bytes_per_row: u64,
    pub(super) owned_bytes_per_row: u64,
    /// Bytes the reader holds for the whole stream beside its chunks (the
    /// parsed Parquet footer).
    pub(super) reader_bytes: u64,
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

pub(super) fn zero_chunk() -> FileSourceError {
    FileSourceError::InvalidOptions { detail: "chunk_rows must be at least one row".into() }
}

/// Checked position of the next row of a stream.
///
/// Positions are `u64` [`GlobalId`]s. Advancing past `u64::MAX` is refused; a
/// position above `u32::MAX` is ordinary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct RowCursor {
    pub(super) next: u64,
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
