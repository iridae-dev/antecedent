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

use antecedent_core::ExecutionContext;
use antecedent_data::OwnedColumn;
use antecedent_io_core::capacity_account::{DesignShape, WeightedRowSource, run_weighted_capacity};

pub use antecedent_data::column_closure::{
    ClosureInputs, ColumnClosure, ColumnDuty, ColumnRole, RequiredColumn,
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
    let mut source = FileWeightedRows::new(file, spec, options.batch_rows.max(1), ctx)?;
    let outcome = run_weighted_capacity(control, &mut source, options);
    if let Some(error) = source.error.take() {
        return Err(FileTableError::Source(error));
    }
    let output = outcome?;
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
}

impl<'a> FileWeightedRows<'a> {
    fn new(
        file: &'a TableFile,
        spec: &GaussianSummarySpec,
        chunk_rows: usize,
        ctx: &ExecutionContext,
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
                    self.receipt = Some(receipt);
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
            let start = written * ncols;
            let mut slot = start;
            if self.intercept {
                x_rowmajor[slot] = 1.0;
                slot += 1;
            }
            let covariates = self.covariates.len();
            x_rowmajor[slot..slot + covariates].copy_from_slice(&values[..covariates]);
            y[written] = values[covariates];
            weights[written] = if self.weight.is_some() { values[covariates + 1] } else { 1.0 };
            self.rows_used += 1;
            written += 1;
        }
        Ok(written)
    }
}
