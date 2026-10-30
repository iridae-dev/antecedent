//! Resident materialization of a file-backed table, quoted and admitted before reading.
//!
//! The quote is a route-memory declaration: the resident value buffers
//! (`rows` times the declared width of every projected column), their validity
//! masks, one chunk of scratch (decode plus owned columns), and the copy that
//! seals each resident buffer. When the row count is recorded, the whole quote
//! is compared with the admission pool before the first byte is read; a pool
//! that is short is refused with the exact quote and the next attempt.
//!
//! The materialized table is snapshotted under the source contract and its
//! full-content digest is verified once (the required first scan of the lazy
//! identity); row ranges of it then have their own lazy-view identities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::{
    DiagnosticSet, EncodedWidth, ExecutionContext, RouteBudget, RouteMemoryDeclaration,
    RouteMemoryError, VariableId,
};

use crate::categorical::{CategoricalColumn, CategoryCode, CategoryDomain};
use crate::column::{
    BooleanColumn, Float64Column, Int64Column, OwnedColumn, TimestampColumn, ValidityBitmap,
};
use crate::dataset::TabularData;
use crate::error::DataError;
use crate::global_index::GlobalId;
use crate::lazy_identity::{FullContentIdentity, LazyViewIdentity};
use crate::source::{OwnedSource, SourceModality};

use super::error::{FileSourceError, MemoryStage};
use super::{FileColumnKind, ReadPlan, ScanReceipt, TableChunkReader, memory_refusal};

/// Bytes a resident read of a plan needs, declared before reading.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MaterializationQuote {
    rows: GlobalId,
    declaration: RouteMemoryDeclaration,
    values_bytes: u64,
    mask_bytes: u64,
    chunk_bytes: u64,
    sealing_copy_bytes: u64,
    total_bytes: u64,
}

impl MaterializationQuote {
    /// Quote for `rows` rows of `plan`. Computed in `u64`; nothing is allocated,
    /// so a row count above `2^32` is quoted like any other.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::CountOverflow`] when the bytes do not fit in `u64`.
    pub fn for_rows(rows: GlobalId, plan: &ReadPlan) -> Result<Self, FileSourceError> {
        let overflow = || FileSourceError::CountOverflow { context: "materialization quote" };
        let chunk_bytes = plan.chunk_quote()?.total_bytes();
        let mask_bytes =
            rows.get().div_ceil(8).checked_mul(plan.columns().len() as u64).ok_or_else(overflow)?;
        let values_bytes =
            rows.get().checked_mul(plan.owned_bytes_per_row()).ok_or_else(overflow)?;
        let sealing_copy_bytes = values_bytes.checked_add(mask_bytes).ok_or_else(overflow)?;
        let mut declaration = RouteMemoryDeclaration::new().rows(rows.get());
        for column in plan.columns() {
            let width =
                column.kind.owned_width_bytes().and_then(EncodedWidth::new).ok_or_else(|| {
                    FileSourceError::InvalidOptions {
                        detail: format!("column '{}' has no declared width", column.name),
                    }
                })?;
            declaration = declaration.encoded_column(width);
        }
        let declaration = declaration
            .masks(mask_bytes)
            .design_expansion(0)
            .learner_fold_copies(0, 0)
            .solver_scratch(chunk_bytes)
            .retained_scores_draws(0)
            .output(sealing_copy_bytes)
            .workers(1);
        let estimate = declaration.estimate().map_err(|_| overflow())?;
        Ok(Self {
            rows,
            declaration,
            values_bytes: estimate.encoded_width(),
            mask_bytes,
            chunk_bytes,
            sealing_copy_bytes,
            total_bytes: estimate.total(),
        })
    }

    /// Rows quoted.
    #[must_use]
    pub const fn rows(&self) -> GlobalId {
        self.rows
    }

    /// Resident value buffers.
    #[must_use]
    pub const fn values_bytes(&self) -> u64 {
        self.values_bytes
    }

    /// Resident validity masks.
    #[must_use]
    pub const fn mask_bytes(&self) -> u64 {
        self.mask_bytes
    }

    /// One chunk of decode and conversion scratch.
    #[must_use]
    pub const fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    /// The copy that seals each resident buffer.
    #[must_use]
    pub const fn sealing_copy_bytes(&self) -> u64 {
        self.sealing_copy_bytes
    }

    /// Bytes held for the whole read, beyond the per-chunk scratch.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.chunk_bytes)
    }

    /// Every component together.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Refuse when `available` bytes cannot hold the quote.
    ///
    /// # Errors
    ///
    /// [`FileSourceError::MemoryRefused`] at [`MemoryStage::Materialization`],
    /// naming the next attempt: stream a summary route, or raise the pool.
    pub fn admit(&self, available: u64, chunk_rows: usize) -> Result<(), FileSourceError> {
        match self.declaration.admit(RouteBudget::new(available)) {
            Ok(_) => Ok(()),
            Err(RouteMemoryError::ExceedsBudget { estimated_bytes, budget_bytes }) => {
                Err(memory_refusal(
                    MemoryStage::Materialization,
                    estimated_bytes,
                    budget_bytes,
                    chunk_rows,
                ))
            }
            Err(_) => Err(FileSourceError::CountOverflow { context: "materialization quote" }),
        }
    }
}

/// A file-backed table read into library-owned columns.
///
/// The table holds the admission leases that paid for its resident columns;
/// dropping it returns them. [`Self::into_data`] keeps the columns and returns
/// the leases.
pub struct MaterializedTable {
    data: TabularData,
    source: OwnedSource,
    full: FullContentIdentity,
    receipt: ScanReceipt,
    quote: Option<MaterializationQuote>,
    leases: Vec<AdmissionLease>,
}

impl fmt::Debug for MaterializedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaterializedTable")
            .field("rows", &self.receipt.rows())
            .field("identity", &self.receipt.identity())
            .field("quote", &self.quote)
            .finish_non_exhaustive()
    }
}

impl MaterializedTable {
    /// The resident table.
    #[must_use]
    pub fn data(&self) -> &TabularData {
        &self.data
    }

    /// The resident table. The leases that charged it are returned to the pool.
    #[must_use]
    pub fn into_data(self) -> TabularData {
        self.data
    }

    /// What the scan read, with the file content identity.
    #[must_use]
    pub fn receipt(&self) -> &ScanReceipt {
        &self.receipt
    }

    /// The quote admitted before reading, when the row count was recorded.
    #[must_use]
    pub const fn quote(&self) -> Option<&MaterializationQuote> {
        self.quote.as_ref()
    }

    /// Bytes the resident leases still hold.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        self.leases.iter().map(|lease| lease.remainder().bytes()).fold(0, u64::saturating_add)
    }

    /// Verified full-content digest of the resident storage (columns, masks, weights).
    #[must_use]
    pub const fn snapshot_digest(&self) -> [u8; 32] {
        self.full.digest()
    }

    /// Digest of the snapshot contract (modality, order, splits, schema names, time index).
    #[must_use]
    pub const fn contract_binding(&self) -> [u8; 32] {
        self.full.binding()
    }

    /// Lazy-view identity of the rows `rows` of this snapshot.
    ///
    /// It is domain-separated from [`Self::snapshot_digest`] and bound to the
    /// verified parent digest and contract.
    ///
    /// # Errors
    ///
    /// The range is outside the table.
    pub fn row_range_identity(&self, rows: Range<usize>) -> Result<[u8; 32], FileSourceError> {
        let chunk = self.source.chunk(rows.start, rows.end)?;
        Ok(LazyViewIdentity::from_chunk(&self.full, &chunk)?.digest())
    }
}

pub(crate) fn materialize(
    mut reader: TableChunkReader,
    quote: Option<MaterializationQuote>,
    ctx: &ExecutionContext,
) -> Result<MaterializedTable, FileSourceError> {
    let plan = reader.plan().clone();
    let chunk_rows = plan.chunk_rows();
    let mut leases = Vec::new();
    let mut reserve = 0_usize;
    if let Some(quote) = &quote {
        reserve = usize::try_from(quote.rows().get())
            .map_err(|_| FileSourceError::CountOverflow { context: "resident row count" })?;
        leases.push(resident_lease(ctx, quote.resident_bytes(), chunk_rows)?);
    }
    let mut builders = plan
        .columns()
        .iter()
        .map(|column| Builder::new(&column.kind, reserve))
        .collect::<Result<Vec<_>, _>>()?;
    let mut received = 0_u64;
    while let Some(chunk) = reader.next_chunk()? {
        let added = u64::try_from(chunk.rows())
            .map_err(|_| FileSourceError::CountOverflow { context: "resident row count" })?;
        let next = received
            .checked_add(added)
            .ok_or(FileSourceError::CountOverflow { context: "resident row count" })?;
        if quote.as_ref().is_some_and(|q| next > q.rows().get()) {
            return Err(FileSourceError::Data(DataError::mismatch("declared file row count")));
        }
        if quote.is_none() {
            let bytes = plan.quote_rows(chunk.rows())?.owned_bytes;
            leases.push(resident_lease(ctx, bytes, chunk_rows)?);
        }
        for (builder, column) in builders.iter_mut().zip(chunk.columns()) {
            builder.append(column)?;
        }
        received = next;
    }
    let receipt = reader.finish()?;
    if quote.as_ref().is_some_and(|q| receipt.rows() != q.rows()) {
        return Err(FileSourceError::Data(DataError::mismatch("declared file row count")));
    }
    let mut columns = Vec::with_capacity(builders.len());
    for (position, builder) in builders.into_iter().enumerate() {
        let raw = u32::try_from(position)
            .map_err(|_| FileSourceError::CountOverflow { context: "projected column count" })?;
        let domain = reader.final_domain(position)?;
        columns.push(builder.finish(VariableId::from_raw(raw), domain)?);
    }
    let names: Vec<String> = plan.columns().iter().map(|column| column.name.to_string()).collect();
    let loaded = crate::arrow_adapter::finish_load(&names, columns, DiagnosticSet::new(), 0, 0)?;
    let source = OwnedSource::from_storage(loaded.data.storage().clone(), SourceModality::Tabular);
    let full = FullContentIdentity::try_from_source(&source)?;
    Ok(MaterializedTable { data: loaded.data, source, full, receipt, quote, leases })
}

fn resident_lease(
    ctx: &ExecutionContext,
    bytes: u64,
    chunk_rows: usize,
) -> Result<AdmissionLease, FileSourceError> {
    match ctx.acquire_admission(0, bytes) {
        Ok(lease) => Ok(lease),
        Err(AdmissionError::Cancelled) => Err(FileSourceError::Cancelled),
        Err(AdmissionError::ExceedsRemainder { remaining_bytes, .. }) => {
            Err(memory_refusal(MemoryStage::Materialization, bytes, remaining_bytes, chunk_rows))
        }
        Err(_) => Err(memory_refusal(
            MemoryStage::Materialization,
            bytes,
            ctx.admission.available().bytes(),
            chunk_rows,
        )),
    }
}

enum Values {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Bool(Vec<u8>),
    Time(Vec<i64>),
    Codes(Vec<CategoryCode>),
}

struct Builder {
    values: Values,
    validity: Vec<u8>,
    len: usize,
}

impl Builder {
    fn new(kind: &FileColumnKind, reserve: usize) -> Result<Self, FileSourceError> {
        let refused = |_| {
            FileSourceError::Data(DataError::external(
                "allocator",
                "refused the resident column buffer",
            ))
        };
        let values = match kind {
            FileColumnKind::Float64 => {
                let mut values = Vec::new();
                values.try_reserve_exact(reserve).map_err(refused)?;
                Values::Float(values)
            }
            FileColumnKind::Int64 => {
                let mut values = Vec::new();
                values.try_reserve_exact(reserve).map_err(refused)?;
                Values::Int(values)
            }
            FileColumnKind::Boolean => {
                let mut values = Vec::new();
                values.try_reserve_exact(reserve).map_err(refused)?;
                Values::Bool(values)
            }
            FileColumnKind::Timestamp { .. } => {
                let mut values = Vec::new();
                values.try_reserve_exact(reserve).map_err(refused)?;
                Values::Time(values)
            }
            FileColumnKind::Categorical { .. } => {
                let mut values = Vec::new();
                values.try_reserve_exact(reserve).map_err(refused)?;
                Values::Codes(values)
            }
            FileColumnKind::Unsupported { arrow_type } => {
                return Err(FileSourceError::InvalidOptions {
                    detail: format!("a column of type {arrow_type} is not materialized"),
                });
            }
        };
        let mut validity = Vec::new();
        validity.try_reserve_exact(reserve.div_ceil(8)).map_err(refused)?;
        Ok(Self { values, validity, len: 0 })
    }

    fn append(&mut self, column: &OwnedColumn) -> Result<(), FileSourceError> {
        match (&mut self.values, column) {
            (Values::Float(values), OwnedColumn::Float64(column)) => {
                values.extend_from_slice(column.values.as_slice());
            }
            (Values::Int(values), OwnedColumn::Int64(column)) => {
                values.extend_from_slice(&column.values);
            }
            (Values::Bool(values), OwnedColumn::Boolean(column)) => {
                values.extend_from_slice(&column.values);
            }
            (Values::Time(values), OwnedColumn::Timestamp(column)) => {
                values.extend_from_slice(&column.values_ns);
            }
            (Values::Codes(values), OwnedColumn::Categorical(column)) => {
                values.extend_from_slice(&column.codes);
            }
            _ => {
                return Err(FileSourceError::Data(DataError::TypeMismatch {
                    id: column.id(),
                    expected: "the column kind of the first chunk",
                }));
            }
        }
        let source = column.as_view().validity();
        let added = source.len();
        let new_len = self
            .len
            .checked_add(added)
            .ok_or(FileSourceError::CountOverflow { context: "resident row count" })?;
        self.validity.resize(new_len.div_ceil(8), 0);
        let all_valid = source.is_all_valid();
        for row in 0..added {
            if all_valid || source.is_valid(row) {
                let bit = self.len + row;
                self.validity[bit / 8] |= 1 << (bit % 8);
            }
        }
        self.len = new_len;
        Ok(())
    }

    fn finish(
        self,
        id: VariableId,
        domain: Option<Arc<CategoryDomain>>,
    ) -> Result<OwnedColumn, FileSourceError> {
        let validity = ValidityBitmap::from_bytes(self.validity, self.len)?;
        let column = match self.values {
            Values::Float(values) => {
                OwnedColumn::Float64(Float64Column::new(id, values, validity)?)
            }
            Values::Int(values) => OwnedColumn::Int64(Int64Column::new(id, values, validity)?),
            Values::Bool(values) => OwnedColumn::Boolean(BooleanColumn::new(id, values, validity)?),
            Values::Time(values) => {
                OwnedColumn::Timestamp(TimestampColumn::new(id, values, validity)?)
            }
            Values::Codes(codes) => {
                let domain = domain.ok_or({
                    FileSourceError::Data(DataError::TypeMismatch { id, expected: "categorical" })
                })?;
                OwnedColumn::Categorical(CategoricalColumn::try_new(id, codes, validity, domain)?)
            }
        };
        Ok(column)
    }
}
