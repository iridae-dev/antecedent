//! File-backed tables at scale: Parquet files, Parquet datasets, Arrow IPC
//! files and streams, read in bounded chunks under the admission pool.
//!
//! [`TableFile`] opens a file or a directory of Parquet files and reads only
//! the schema and footer. A read is planned from the required-column closure,
//! so only the target, validator, support, weight, split and complete-case
//! columns are decoded. Each chunk is charged to the study's admission pool
//! before it is decoded and before its columns are built, and rows are
//! addressed by checked 64-bit global ids.
//!
//! Two routes consume a file:
//!
//! - A route that needs the whole design resident calls
//!   [`TableFile::materialize`], which quotes the resident bytes first and is
//!   refused before reading when the pool cannot hold them. The refusal names
//!   the exact quote and the next attempt: stream a summary route, or raise
//!   the pool.
//! - [`summarize_weighted_gaussian`] streams the file through the bounded
//!   weighted Gaussian / OLS capacity route. It keeps one chunk resident at a
//!   time and returns the cross-products, coefficients, `HC0` covariance,
//!   leverage and residual checks, so a table larger than memory is
//!   summarized under a fixed ceiling.
//!
//! Calibrated intervals for large tables are a separate claim; reading a large
//! table does not extend a route's calibrated scope.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::rewrite_legality::{
    FrameFingerprint, FrameLedger, Rewrite, RewriteRefusal, check_executed, value_row_id,
};
use antecedent_core::ExecutionContext;
use antecedent_data::OwnedColumn;
use antecedent_io_core::capacity_account::{DesignShape, WeightedRowSource, run_weighted_capacity};

pub use antecedent_data::column_closure::{
    ClosureError, ClosureInputs, Cohort, ColumnClosure, ColumnDuty, ColumnFilter, ColumnRole,
    ObservationFrame, RequiredColumn,
};
pub use antecedent_data::file_source::{
    ChunkQuote, DEFAULT_CHUNK_ROWS, FileChunk, FileColumn, FileColumnKind, FileContentIdentity,
    FileSourceError, FileSourceErrorKind, MaterializationQuote, MaterializedTable, MemoryStage,
    ReadOptions, ReadPlan, RowCursor, ScanReceipt, TableChunkReader, TableFile, TableFileFormat,
    TableStream, WriteOptions, chunk_local_index, write_table_file,
};
pub use antecedent_io_core::capacity_account::{
    CapacityControl, CapacityOptions, CeilingCharges, WeightedCapacityError,
    WeightedCapacityOutput, WeightedCapacityResult,
};

/// Largest integer magnitude that is exact in binary64.
const EXACT_INTEGER: i64 = 1 << 53;

/// Name of the intercept column in [`FileGaussianSummary::design_columns`].
pub const INTERCEPT_COLUMN: &str = "(intercept)";

/// What the summary route does with a row that misses a required value.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SummaryMissing {
    /// Refuse the run at the first missing value, naming its column and row.
    #[default]
    Refuse,
    /// Leave out every row that misses a covariate, the response, or the
    /// weight, and count the rows left out.
    CompleteCase,
}

impl SummaryMissing {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Refuse => "refuse",
            Self::CompleteCase => "complete_case",
        }
    }

    /// Policy named `name` (`refuse` or `complete_case`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "refuse" => Some(Self::Refuse),
            "complete_case" => Some(Self::CompleteCase),
            _ => None,
        }
    }
}

/// Columns of a weighted Gaussian summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GaussianSummarySpec {
    /// Design columns, in order.
    pub covariates: Vec<String>,
    /// Response column.
    pub response: String,
    /// Weight column; every row weighs one when absent.
    pub weight: Option<String>,
    /// Whether the design starts with an intercept column.
    pub intercept: bool,
    /// Missing-value policy.
    pub missing: SummaryMissing,
}

impl GaussianSummarySpec {
    /// Unweighted summary of `response` on an intercept and `covariates`, refusing missing values.
    #[must_use]
    pub fn new(covariates: Vec<String>, response: impl Into<String>) -> Self {
        Self {
            covariates,
            response: response.into(),
            weight: None,
            intercept: true,
            missing: SummaryMissing::Refuse,
        }
    }

    /// Design column names, with the intercept first when there is one.
    #[must_use]
    pub fn design_columns(&self) -> Vec<String> {
        let mut columns = Vec::with_capacity(self.covariates.len() + 1);
        if self.intercept {
            columns.push(INTERCEPT_COLUMN.to_owned());
        }
        columns.extend(self.covariates.iter().cloned());
        columns
    }
}

/// A file summarized by the weighted Gaussian capacity route.
#[derive(Debug)]
pub struct FileGaussianSummary {
    /// The capacity route's output: the resident summary, or its spill.
    pub output: WeightedCapacityOutput,
    /// Design columns in coefficient order.
    pub design_columns: Vec<String>,
    /// Rows read in the last pass.
    pub rows_read: u64,
    /// Rows that entered the summary.
    pub rows_used: u64,
    /// Rows left out under [`SummaryMissing::CompleteCase`].
    pub rows_dropped: u64,
    /// Full-content identity of the projected columns, from the last complete pass.
    pub file_identity: FileContentIdentity,
}

/// Why a file summary was refused.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileTableError {
    /// The file source refused to plan or read.
    #[error(transparent)]
    Source(#[from] FileSourceError),
    /// The capacity route refused the summary.
    #[error("weighted capacity route: {0}")]
    Capacity(#[from] WeightedCapacityError),
    /// The summary did not carry the rows, weights and independence of the
    /// complete-case cohort the file declared.
    #[error("sufficient-statistics rewrite: {0}")]
    Rewrite(#[from] RewriteRefusal),
}

/// A column projection or filter pushdown was refused, or it ran and did not keep the cohort.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PushdownCheckError {
    /// The closure refused the pushdown, or the frame was malformed.
    #[error(transparent)]
    Closure(#[from] ClosureError),
    /// The pushdown ran, and the rows it delivers are not the declared cohort.
    #[error("pushdown rewrite: {0}")]
    Rewrite(#[from] RewriteRefusal),
}

fn cohort_fingerprint(rows: impl IntoIterator<Item = usize>) -> FrameFingerprint {
    let mut ledger =
        FrameLedger::try_new("complete_case_cohort", "static_cross_section", "unweighted")
            .unwrap_or_else(|_| unreachable!("fixed non-blank frame ids"));
    for row in rows {
        let row = row as u64;
        ledger.record_numeric(row, 0, row, 0);
    }
    ledger.fingerprint()
}

/// Push `filter` through `closure`, then check the rows it delivers against the cohort.
///
/// The cohort is the complete-case rows on every closure column. The pushdown is admitted
/// only when the cohort it returns, and the rows a scan with the filter applied would
/// deliver, are that cohort.
///
/// # Errors
///
/// [`PushdownCheckError::Closure`] when the closure refuses the filter;
/// [`PushdownCheckError::Rewrite`] when the delivered rows differ from the cohort.
pub fn checked_push_filter(
    closure: &antecedent_data::column_closure::ColumnClosure,
    frame: &ObservationFrame,
    filter: &ColumnFilter,
) -> Result<Cohort, PushdownCheckError> {
    let declared = closure.complete_case(frame)?;
    let pushed = closure.push_filter(frame, filter)?;
    let delivered = declared.rows().iter().copied().filter(|row| !filter.drops(*row));
    check_executed(
        Rewrite::COLUMN_FILTER_PUSHDOWN,
        &cohort_fingerprint(declared.rows().iter().copied()),
        &[cohort_fingerprint(pushed.rows().iter().copied()), cohort_fingerprint(delivered)],
    )?;
    Ok(pushed)
}

/// Project to `keep`, then check that completing cases on the kept columns is the cohort.
///
/// [`antecedent_data::column_closure::ColumnClosure::project`] refuses a projection that
/// drops a required column. A projection that keeps extra columns is not refused there, but
/// complete-case over the projected frame then also drops rows missing an extra column.
///
/// # Errors
///
/// [`PushdownCheckError::Closure`] when the projection drops a required column;
/// [`PushdownCheckError::Rewrite`] when the projected complete-case rows are not the cohort.
pub fn checked_projection(
    closure: &antecedent_data::column_closure::ColumnClosure,
    frame: &ObservationFrame,
    keep: &[antecedent_data::column_closure::ColumnRole],
) -> Result<Cohort, PushdownCheckError> {
    closure.project(keep)?;
    let declared = closure.complete_case(frame)?;
    let projected = frame.complete_case(keep)?;
    check_executed(
        Rewrite::COLUMN_FILTER_PUSHDOWN,
        &cohort_fingerprint(declared.rows().iter().copied()),
        &[cohort_fingerprint(projected.rows().iter().copied())],
    )?;
    Ok(projected)
}

/// Stream `file` through the weighted Gaussian / OLS capacity route.
///
/// The read is planned from `spec`'s columns only, at
/// [`CapacityOptions::batch_rows`] rows per chunk; each chunk is charged to
/// `ctx`'s admission pool and to the route's ceiling before it is decoded.
/// Covariates, response and weight must be float, integer (exact in binary64)
/// or boolean columns. The route reads the file twice (cross-products, then
/// diagnostics) and refuses a file whose rows changed between the passes.
///
/// # Errors
///
/// [`FileTableError::Source`] when a column is absent or not numeric, a value
/// is missing under [`SummaryMissing::Refuse`], a chunk does not fit the pool,
/// or the read is cancelled; [`FileTableError::Capacity`] when the route's
/// ceiling, weights, design or solve refuse the summary.
pub fn summarize_weighted_gaussian(
    file: &TableFile,
    spec: &GaussianSummarySpec,
    control: CapacityControl,
    options: &CapacityOptions,
    ctx: &ExecutionContext,
) -> Result<FileGaussianSummary, FileTableError> {
    summarize_with_fault(file, spec, control, options, ctx, None)
}

/// [`summarize_weighted_gaussian`] with a defect injected into the lowering stage.
///
/// A test seam: it must return [`FileTableError::Rewrite`] for any fault that changes the
/// rows, weights, or values the summary received.
///
/// # Errors
///
/// As [`summarize_weighted_gaussian`].
#[doc(hidden)]
pub fn summarize_weighted_gaussian_faulted(
    file: &TableFile,
    spec: &GaussianSummarySpec,
    control: CapacityControl,
    options: &CapacityOptions,
    ctx: &ExecutionContext,
    fault: RowFault,
) -> Result<FileGaussianSummary, FileTableError> {
    summarize_with_fault(file, spec, control, options, ctx, Some(fault))
}

fn summarize_with_fault(
    file: &TableFile,
    spec: &GaussianSummarySpec,
    control: CapacityControl,
    options: &CapacityOptions,
    ctx: &ExecutionContext,
    fault: Option<RowFault>,
) -> Result<FileGaussianSummary, FileTableError> {
    let mut source = FileWeightedRows::new(file, spec, options.batch_rows.max(1), ctx, fault)?;
    let outcome = run_weighted_capacity(control, &mut source, options);
    if let Some(error) = source.error.take() {
        return Err(FileTableError::Source(error));
    }
    let output = outcome?;
    if let Some(refusal) = source.refusal.take() {
        return Err(FileTableError::Rewrite(refusal));
    }
    let receipt = source.receipt.take().ok_or(FileSourceError::ScanIncomplete)?;
    Ok(FileGaussianSummary {
        output,
        design_columns: spec.design_columns(),
        rows_read: source.rows_read,
        rows_used: source.rows_used,
        rows_dropped: source.rows_dropped,
        file_identity: receipt.identity(),
    })
}

/// One executed row with the values it carried, for the cohort and summary ledgers.
fn record_row(
    ledger: &mut FrameLedger,
    global: u64,
    values: impl IntoIterator<Item = f64>,
    weight: f64,
) {
    ledger.record(
        &value_row_id(global, values),
        "0",
        &global.to_string(),
        &format!("{:016x}", weight.to_bits()),
    );
}

/// A defect injected into the lowering stage of a file summary.
///
/// This is a test seam: it changes what the summary receives for one row, so a test can
/// show that [`summarize_weighted_gaussian_faulted`] refuses a lowering that is not the
/// declared cohort. The row is the global row id.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowFault {
    /// The row never reaches the summary.
    DropRow(u64),
    /// The row reaches the summary twice.
    DuplicateRow(u64),
    /// The row reaches the summary with its response shifted by one.
    AlterResponse(u64),
}

/// A planned file read presented as weighted design rows.
struct FileWeightedRows<'a> {
    file: &'a TableFile,
    plan: ReadPlan,
    ctx: ExecutionContext,
    reader: Option<TableChunkReader>,
    chunk: Option<FileChunk>,
    offset: usize,
    covariates: Vec<usize>,
    response: usize,
    weight: Option<usize>,
    intercept: bool,
    missing: SummaryMissing,
    bound: Option<usize>,
    rows_read: u64,
    rows_used: u64,
    rows_dropped: u64,
    error: Option<FileSourceError>,
    receipt: Option<ScanReceipt>,
    /// Empty ledger for this read; each pass starts from a copy.
    fresh: FrameLedger,
    /// Rows the projected read delivered, before any missing-value rule.
    projected: FrameLedger,
    fault: Option<RowFault>,
    /// Rows observed on every closure column, with the weight each carries.
    cohort: FrameLedger,
    /// Rows that entered the summary, with the weight written for each.
    lowered: FrameLedger,
    /// First pass whose summary did not carry the cohort.
    refusal: Option<RewriteRefusal>,
}

impl<'a> FileWeightedRows<'a> {
    fn new(
        file: &'a TableFile,
        spec: &GaussianSummarySpec,
        chunk_rows: usize,
        ctx: &ExecutionContext,
        fault: Option<RowFault>,
    ) -> Result<Self, FileSourceError> {
        let mut target: Vec<ColumnRole> = spec.covariates.iter().map(ColumnRole::new).collect();
        target.push(ColumnRole::new(&spec.response));
        let mut inputs = ClosureInputs::new(target);
        if let Some(weight) = &spec.weight {
            inputs = inputs.weight(ColumnRole::new(weight));
        }
        let plan = file.plan(&inputs.close(), ReadOptions::with_chunk_rows(chunk_rows))?;
        for column in plan.columns() {
            if !matches!(
                column.kind,
                FileColumnKind::Float64 | FileColumnKind::Int64 | FileColumnKind::Boolean
            ) {
                return Err(FileSourceError::UnsupportedColumn {
                    column: column.name.to_string(),
                    arrow_type: column.arrow_type.to_string(),
                    next_attempt: "encode the column as numeric indicators or as a number before \
                                   the Gaussian summary",
                });
            }
        }
        let position = |name: &str| {
            plan.position(name).ok_or_else(|| FileSourceError::UnknownColumn {
                column: name.to_owned(),
                available: plan.names().iter().map(ToString::to_string).collect(),
            })
        };
        let covariates =
            spec.covariates.iter().map(|name| position(name)).collect::<Result<Vec<_>, _>>()?;
        let response = position(&spec.response)?;
        let weight = spec.weight.as_deref().map(position).transpose()?;
        let population = format!("complete_case:{}", plan.names().join(","));
        let law = spec
            .weight
            .as_ref()
            .map_or_else(|| "unit_weights".to_owned(), |column| format!("weights:{column}"));
        let fresh = FrameLedger::try_new(&population, "static_cross_section", &law)
            .map_err(|refusal| FileSourceError::InvalidOptions { detail: refusal.to_string() })?;
        Ok(Self {
            file,
            plan,
            ctx: ctx.clone(),
            reader: None,
            chunk: None,
            offset: 0,
            covariates,
            response,
            weight,
            intercept: spec.intercept,
            missing: spec.missing,
            bound: None,
            rows_read: 0,
            rows_used: 0,
            rows_dropped: 0,
            error: None,
            receipt: None,
            cohort: fresh.clone(),
            lowered: fresh.clone(),
            projected: fresh.clone(),
            fault,
            fresh,
            refusal: None,
        })
    }

    fn fail(&mut self, error: FileSourceError) -> WeightedCapacityError {
        if error == FileSourceError::Cancelled {
            self.error = Some(error);
            return WeightedCapacityError::Cancelled;
        }
        let message = error.to_string();
        self.error = Some(error);
        WeightedCapacityError::RowSource { message }
    }

    /// Compare the summary's rows and weights with the cohort this pass saw.
    ///
    /// Also compares the rows the projected read delivered with the rows the file holds, so
    /// a projection that skipped or repeated a chunk is a refusal too.
    fn check_lowering(&mut self, file_rows: u64) {
        if self.refusal.is_some() {
            return;
        }
        let mut file = self.fresh.clone();
        for row in 0..file_rows {
            file.record_numeric(row, 0, row, 0);
        }
        let pushed = check_executed(
            Rewrite::COLUMN_FILTER_PUSHDOWN,
            &file.fingerprint(),
            &[self.projected.fingerprint()],
        );
        let lowered = check_executed(
            Rewrite::SUFFICIENT_STATISTICS,
            &self.cohort.fingerprint(),
            &[self.lowered.fingerprint()],
        );
        self.refusal = pushed.and(lowered).err();
    }

    /// The next non-empty chunk, or `None` after the last one (when the receipt is taken).
    fn advance(&mut self) -> Result<bool, WeightedCapacityError> {
        self.chunk = None;
        self.offset = 0;
        let Some(reader) = self.reader.as_mut() else {
            return Err(WeightedCapacityError::DecoderNotPrepared);
        };
        match reader.next_chunk() {
            Ok(Some(chunk)) => {
                self.chunk = Some(chunk);
                Ok(true)
            }
            Ok(None) => match reader.finish() {
                Ok(receipt) => {
                    let read_rows = receipt.rows().get();
                    self.receipt = Some(receipt);
                    self.check_lowering(read_rows);
                    Ok(false)
                }
                Err(error) => Err(self.fail(error)),
            },
            Err(error) => Err(self.fail(error)),
        }
    }
}

/// Value of row `row` of `column` as binary64, or `None` when it is missing.
fn cell(
    column: &OwnedColumn,
    row: usize,
    name: &str,
    chunk: &FileChunk,
) -> Result<Option<f64>, FileSourceError> {
    let valid = column.as_view().validity().is_valid(row);
    if !valid {
        return Ok(None);
    }
    match column {
        OwnedColumn::Float64(values) => Ok(Some(values.values.as_slice()[row])),
        OwnedColumn::Int64(values) => {
            let value = values.values[row];
            if value.unsigned_abs() > EXACT_INTEGER.unsigned_abs() {
                return Err(FileSourceError::UnsupportedValue {
                    column: name.to_owned(),
                    row: antecedent_data::GlobalId::new(chunk.start().get() + row as u64),
                    detail: format!("integer {value} is not exact in binary64"),
                    next_attempt: "rescale the integer column, or read it as a category",
                });
            }
            Ok(Some(value as f64))
        }
        OwnedColumn::Boolean(values) => Ok(Some(f64::from(values.values[row]))),
        _ => Err(FileSourceError::UnsupportedColumn {
            column: name.to_owned(),
            arrow_type: "non-numeric".to_owned(),
            next_attempt: "encode the column as numeric indicators or as a number before the \
                           Gaussian summary",
        }),
    }
}

impl WeightedRowSource for FileWeightedRows<'_> {
    fn ncols(&self) -> usize {
        self.covariates.len() + usize::from(self.intercept)
    }

    fn input_ownership_bytes(&self) -> u64 {
        0
    }

    fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError> {
        if batch_rows == 0 {
            return Err(WeightedCapacityError::BatchBound);
        }
        self.plan
            .with_chunk_rows(batch_rows)
            .and_then(|plan| plan.chunk_quote())
            .map(ChunkQuote::total_bytes)
            .map_err(|error| WeightedCapacityError::RowSource { message: error.to_string() })
    }

    fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError> {
        if batch_rows == 0 {
            return Err(WeightedCapacityError::BatchBound);
        }
        match self.plan.with_chunk_rows(batch_rows) {
            Ok(plan) => self.plan = plan,
            Err(error) => return Err(self.fail(error)),
        }
        self.reader = None;
        self.bound = Some(batch_rows);
        Ok(())
    }

    fn allocated_decoder_bytes(&self) -> u64 {
        self.chunk.as_ref().map_or(0, FileChunk::held_bytes)
    }

    fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
        self.chunk = None;
        self.offset = 0;
        self.rows_read = 0;
        self.rows_used = 0;
        self.rows_dropped = 0;
        self.cohort = self.fresh.clone();
        self.lowered = self.fresh.clone();
        self.projected = self.fresh.clone();
        let rewound = if let Some(reader) = self.reader.as_mut() {
            reader.rewind()
        } else {
            match self.file.reader(&self.plan, &self.ctx) {
                Ok(reader) => {
                    self.reader = Some(reader);
                    Ok(())
                }
                Err(error) => Err(error),
            }
        };
        rewound.map_err(|error| self.fail(error))
    }

    fn read_batch(
        &mut self,
        max_rows: usize,
        x_rowmajor: &mut [f64],
        y: &mut [f64],
        weights: &mut [f64],
    ) -> Result<usize, WeightedCapacityError> {
        let bound = self.bound.ok_or(WeightedCapacityError::DecoderNotPrepared)?;
        if max_rows == 0 || max_rows > bound {
            return Err(WeightedCapacityError::ReadPastBatch { requested: max_rows, bound });
        }
        let ncols = self.ncols();
        let cells = max_rows.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
        if x_rowmajor.len() < cells || y.len() < max_rows || weights.len() < max_rows {
            return Err(WeightedCapacityError::Shape(DesignShape::BatchWindow));
        }
        let mut written = 0;
        while written < max_rows {
            let exhausted = self.chunk.as_ref().is_none_or(|chunk| self.offset >= chunk.rows());
            if exhausted && !self.advance()? {
                break;
            }
            let Some(chunk) = self.chunk.as_ref() else {
                break;
            };
            let row = self.offset;
            self.offset += 1;
            self.rows_read += 1;
            let global = chunk.start().get() + row as u64;
            let mut values = Vec::with_capacity(self.covariates.len() + 2);
            let mut missing = None;
            let order = self
                .covariates
                .iter()
                .copied()
                .chain(std::iter::once(self.response))
                .chain(self.weight);
            for position in order {
                let name = &chunk.names()[position];
                match cell(&chunk.columns()[position], row, name, chunk) {
                    Ok(Some(value)) => values.push(value),
                    Ok(None) => {
                        missing = Some((name.to_string(), chunk.start().get() + row as u64));
                        break;
                    }
                    Err(error) => return Err(self.fail(error)),
                }
            }
            self.projected.record_numeric(global, 0, global, 0);
            if missing.is_none() {
                let weight = if self.weight.is_some() { values[values.len() - 1] } else { 1.0 };
                let observed = values[..self.covariates.len() + 1].iter().copied();
                record_row(&mut self.cohort, global, observed, weight);
            }
            if let Some((column, global)) = missing {
                match self.missing {
                    SummaryMissing::CompleteCase => {
                        self.rows_dropped += 1;
                        continue;
                    }
                    SummaryMissing::Refuse => {
                        return Err(self.fail(FileSourceError::MissingValue {
                            column,
                            row: antecedent_data::GlobalId::new(global),
                            next_attempt: "declare complete-case for the summary, or impute the \
                                           column before reading it",
                        }));
                    }
                }
            }
            let covariates = self.covariates.len();
            let (copies, shift) = match self.fault {
                Some(RowFault::DropRow(row)) if row == global => (0, 0.0),
                Some(RowFault::DuplicateRow(row)) if row == global && written + 1 < max_rows => {
                    (2, 0.0)
                }
                Some(RowFault::AlterResponse(row)) if row == global => (1, 1.0),
                _ => (1, 0.0),
            };
            for _ in 0..copies {
                let start = written * ncols;
                let mut slot = start;
                if self.intercept {
                    x_rowmajor[slot] = 1.0;
                    slot += 1;
                }
                x_rowmajor[slot..slot + covariates].copy_from_slice(&values[..covariates]);
                y[written] = values[covariates] + shift;
                weights[written] = if self.weight.is_some() { values[covariates + 1] } else { 1.0 };
                // What the summary received, read back from the buffers it was given.
                record_row(
                    &mut self.lowered,
                    global,
                    x_rowmajor[slot..slot + covariates].iter().copied().chain([y[written]]),
                    weights[written],
                );
                self.rows_used += 1;
                written += 1;
            }
        }
        Ok(written)
    }
}
