//! Chunk streams and readers over an opened table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::schema::*;
use super::*;

/// An Arrow C stream read once, in bounded chunks.
pub struct TableStream {
    pub(super) view: SchemaView,
    pub(super) reader: arrow_array::ffi_stream::ArrowArrayStreamReader,
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

pub(super) enum Origin {
    File { format: TableFileFormat, paths: Vec<PathBuf> },
    Stream,
}

pub(super) struct Pending {
    pub(super) batch: RecordBatch,
    pub(super) offset: usize,
    pub(super) held_bytes: u64,
    pub(super) _leases: Vec<AdmissionLease>,
}

/// One bounded chunk of library-owned columns.
///
/// The chunk holds the admission lease that paid for its columns; dropping
/// the chunk returns it.
pub struct FileChunk {
    pub(super) start: GlobalId,
    pub(super) rows: usize,
    pub(super) columns: Vec<OwnedColumn>,
    pub(super) names: Arc<[Arc<str>]>,
    pub(super) lease: AdmissionLease,
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
    pub(super) rows: GlobalId,
    pub(super) chunks: u64,
    pub(super) identity: FileContentIdentity,
    pub(super) columns: Vec<FileColumn>,
    pub(super) levels: Vec<Vec<Arc<str>>>,
    pub(super) peak_chunk_bytes: u64,
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
    pub(super) origin: Origin,
    pub(super) plan: ReadPlan,
    pub(super) ctx: ExecutionContext,
    pub(super) provider: Option<Box<dyn BatchProvider>>,
    pub(super) states: Vec<ColumnState>,
    pub(super) names: Arc<[Arc<str>]>,
    pub(super) cursor: RowCursor,
    pub(super) pending: Option<Pending>,
    pub(super) chunks: u64,
    pub(super) exhausted: bool,
    pub(super) peak_chunk_bytes: u64,
    pub(super) receipt: Option<ScanReceipt>,
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
    pub(super) fn new(
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

    pub(super) fn pull(&mut self) -> Result<bool, FileSourceError> {
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

    pub(super) fn convert(
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
    pub(super) fn producer_remedy(
        &self,
        error: FileSourceError,
        batch_bytes: u64,
    ) -> FileSourceError {
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

    pub(super) fn lease(
        &self,
        stage: MemoryStage,
        bytes: u64,
    ) -> Result<AdmissionLease, FileSourceError> {
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
