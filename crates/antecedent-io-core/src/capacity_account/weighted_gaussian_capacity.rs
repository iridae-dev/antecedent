//! File-backed and resident weighted Gaussian / OLS under one memory ceiling.
//!
//! Both [`CapacityControl::Glm`] and [`CapacityControl::GaussianBayes`] return
//! the same summary: weighted coefficients, `HC0` covariance, leverage, residual
//! checks, and the Gaussian cross-products `X'WX`, `X'Wy`, `y'Wy`, and `Σw`.
//! The coefficient call is [`antecedent_stats::fit_wls`] on the Cholesky factor
//! of that Gram, which is the weighted normal equation. Rows are read in batches
//! no larger than [`CapacityOptions::batch_rows`].
//!
//! The cross-products are accumulated around running weighted means (a
//! weighted Welford update per row), so the sums do not compound rounding
//! across rows. The raw `X'WX`, `X'Wy`, and `y'Wy` are rebuilt from the
//! centered moments once at the end. The solve still runs on the Gram, so its
//! conditioning is the Gram's. The row stream is hashed in both passes and a
//! source that changes between them is refused.
//!
//! [`OutputRetention::SummaryOnly`] returns that summary resident.
//! [`OutputRetention::DiskBacked`] writes the finished summary, and nothing
//! else, to [`SpillOptions::path`] and returns a [`SpillHandle`] to it. The
//! row batches already stream from the source in both modes, so the spill is
//! the coefficients, the `HC0` covariance, the cross-products, the leverage and
//! residual summaries, the identity digest, the fit diagnostics, and the
//! charges. The file is a digest-checked frame written to a temporary name,
//! renamed into place, and read back. [`read_spilled_summary`] returns the same
//! [`WeightedCapacityResult`] a resident run returns, bit for bit, and refuses
//! a file that was changed or truncated. The spill bytes are reserved from
//! [`SpillOptions::pool`] before anything is written and are held by the
//! handle until [`SpillHandle::remove`] or drop. A run that errors or is
//! cancelled removes its temporary file, leaves no target, and returns the
//! reservation.
//!
//! The mechanism-registry fitter `fit_linear_gaussian_weighted` is `pub(super)`
//! in `antecedent-model` and is not a call site from this crate.
//! [`antecedent_prob::gaussian_target`] is crate-private as well. This route
//! does not time a host and does not record a measured resident set.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::TryReserveError;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use antecedent_core::device_reservation::{
    DeviceReservationError, DeviceReservationLease, DeviceReservationPool,
};
use antecedent_core::execution::CancellationToken;
use antecedent_core::execution_facts::OutputRetention;
use antecedent_stats::{
    FaerBackend, FitDiagnostics, LeastSquaresWorkspace, StatsError, cholesky_spd, fit_wls,
    invert_square,
};

use super::spill_file::{self, AllocFail, Declared, IoFail, SpillFault};

const MAGIC: &[u8; 4] = b"WGOL";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 20;
const HEADER_BYTES: u64 = 20;
const IDENTITY_DOMAIN: &str = "antecedent.capacity.weighted-gaussian.v1";
/// Fit backends the spill layout can name. A backend outside this list is not spilled.
const SPILL_BACKENDS: [&str; 1] = ["faer"];

/// Which capacity control is being run.
///
/// The two controls share one result and one retention mode. Gaussian lowering
/// here is the cross-product summary, not a second scan inside a sampler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityControl {
    /// Weighted Gaussian linear fit, the `glm` control's likelihood.
    Glm,
    /// The same weighted summary, the `gaussian_bayes` control's likelihood.
    GaussianBayes,
}

/// Batch bound, byte ceiling, and the retention mode both controls use.
#[derive(Clone, Debug)]
pub struct CapacityOptions {
    /// Maximum source rows materialized in one read.
    pub batch_rows: usize,
    /// Bytes available for ownership, identity, decoders, spill, and output.
    pub ceiling_bytes: u64,
    /// Summary-only keeps the summary resident. Disk-backed spills it to
    /// [`Self::spill`], which it requires.
    pub retention: OutputRetention,
    /// Where a disk-backed run writes its summary.
    pub spill: Option<SpillOptions>,
    /// Checked once per non-empty batch in each pass, before the spill is
    /// reserved, and before the spill file is renamed into place.
    pub cancellation: Option<CancellationToken>,
}

impl CapacityOptions {
    /// Options with no spill target and no cancellation token.
    #[must_use]
    pub const fn new(batch_rows: usize, ceiling_bytes: u64, retention: OutputRetention) -> Self {
        Self { batch_rows, ceiling_bytes, retention, spill: None, cancellation: None }
    }
}

/// Where a disk-backed summary goes and what it may take from the spill pool.
#[derive(Clone, Debug)]
pub struct SpillOptions {
    /// Target file. Its directory has to exist; a file already there is replaced.
    pub path: PathBuf,
    /// Pool the spill bytes are reserved from before the file is written.
    /// `None` leaves the ceiling as the only limit.
    pub pool: Option<DeviceReservationPool>,
}

/// Bytes reserved against [`CapacityOptions::ceiling_bytes`].
///
/// The total is the sum of the five charges. It is a reservation from the
/// shapes, not a measurement of a host allocator. A disk-backed run charges the
/// exact frame length as [`Self::spill_bytes`] and charges the buffer that
/// encodes and verifies the frame, plus the finished summary it encodes, under
/// [`Self::decoder_bytes`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CeilingCharges {
    /// Resident source bytes. A file handle charges nothing for the unread body.
    pub input_ownership_bytes: u64,
    /// Digest plus the one-row staging buffer that feeds it.
    pub content_identity_bytes: u64,
    /// Batch window, file decoder, and every solver and diagnostic buffer.
    pub decoder_bytes: u64,
    /// Frame bytes written to the spill file. Zero unless retention is disk-backed.
    pub spill_bytes: u64,
    /// Summary bytes when retention is summary-only, or the spill handle when
    /// retention is disk-backed.
    pub output_bytes: u64,
    /// Sum of the five charges.
    pub total_bytes: u64,
}

/// Leverage of the scaled weighted design.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeverageSummary {
    /// Largest diagonal hat value.
    pub max: f64,
    /// Sum of the diagonal hat values.
    pub sum: f64,
    /// Rows whose leverage sits within `1e-8` of 1.
    pub saturated_rows: u64,
}

/// Residual checks from the weighted fit. Not a stored residual vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualCheck {
    /// `Σ w (y − xβ)²` evaluated as the squared scaled residual.
    pub weighted_rss: f64,
    /// Largest absolute raw residual `y − xβ`.
    pub max_abs_residual: f64,
    /// Rows whose weight is strictly positive.
    pub positive_weight_rows: u64,
}

/// Weighted Gaussian cross-products the coefficient solve consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianCrossProducts {
    /// `Σ w`.
    pub n_eff: f64,
    /// Row-major `X'WX`.
    pub xtwx: Vec<f64>,
    /// `X'Wy`.
    pub xtwy: Vec<f64>,
    /// `y'Wy`.
    pub ytwy: f64,
}

/// Summary shared by both capacity controls.
#[derive(Clone, Debug)]
pub struct WeightedCapacityResult {
    /// Control that was requested.
    pub control: CapacityControl,
    /// Retention mode the charges used.
    pub retention: OutputRetention,
    /// Weighted least-squares coefficients from [`fit_wls`].
    pub coefficients: Vec<f64>,
    /// `HC0` covariance of the scaled design, row-major.
    pub robust_covariance: Vec<f64>,
    /// Leverage summary. Empty leverage is not a successful result.
    pub leverage: LeverageSummary,
    /// Residual summary. A dropped residual pass leaves this at zero.
    pub residual: ResidualCheck,
    /// Gaussian cross-products.
    pub cross_products: GaussianCrossProducts,
    /// Streaming digest of the column count and of every `(x, y, w)` row.
    pub content_identity: [u8; 32],
    /// Rank and condition report from the [`fit_wls`] call.
    pub fit: FitDiagnostics,
    /// Ceiling reservation for this run.
    pub charges: CeilingCharges,
    /// Non-empty batch reads in the cross-product pass.
    pub batches: u64,
    /// Cross-product pass plus the diagnostic pass.
    pub source_passes: u64,
}

/// Which shape rule a design broke.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DesignShape {
    /// The design has no columns, or no rows to infer them from.
    #[error("no design columns")]
    ZeroColumns,
    /// The weight vector is not one entry per row.
    #[error("weights length does not match rows")]
    WeightsLength,
    /// The design length is not a whole number of rows.
    #[error("design length is not rows times columns")]
    DesignLength,
    /// A row does not have the design width.
    #[error("row width does not match the design")]
    Row,
    /// A batch window is shorter than the requested read.
    #[error("batch window is shorter than the read")]
    BatchWindow,
    /// The solver returned a different number of coefficients than columns.
    #[error("solve returned the wrong number of coefficients")]
    SolveWidth,
}

/// Why a weight was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum WeightFault {
    /// The weight is below zero.
    #[error("negative")]
    Negative,
    /// The weight is NaN or infinite.
    #[error("not finite")]
    NotFinite,
}

/// How a row file failed to match its layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RowFileFault {
    /// The first four bytes are not the row-file magic.
    #[error("bad weighted-row magic")]
    BadMagic,
    /// The version field is not one this reader decodes.
    #[error("unsupported weighted-row version {found}")]
    UnsupportedVersion {
        /// Version in the header.
        found: u32,
    },
    /// A header field was shorter than its width.
    #[error("short header")]
    ShortHeader,
    /// A record was shorter than its width.
    #[error("short record")]
    ShortRecord,
    /// The file length is not the header length plus `nrows` whole records.
    ///
    /// Trailing bytes and truncation both land here.
    #[error("file is {actual_bytes} bytes but its header describes {expected_bytes}")]
    LengthMismatch {
        /// Bytes the header describes.
        expected_bytes: u64,
        /// Bytes the file holds.
        actual_bytes: u64,
    },
}

/// Why a weighted capacity run refused to return a summary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WeightedCapacityError {
    /// `batch_rows` was zero.
    #[error("batch bound must be at least one row")]
    BatchBound,
    /// A read asked for more rows than the prepared batch bound.
    #[error("a read of {requested} rows exceeds the batch bound of {bound}")]
    ReadPastBatch {
        /// Rows the caller asked to materialize.
        requested: usize,
        /// Prepared bound.
        bound: usize,
    },
    /// Full reusable retention would keep every source row.
    #[error("full reusable retention would keep every source row")]
    RetentionKeepsRows,
    /// Disk-backed retention needs [`CapacityOptions::spill`].
    #[error("disk-backed retention needs a spill path")]
    SpillPathRequired,
    /// The spill file failed its length, digest, or layout check.
    #[error("spill file: {0}")]
    Spill(#[from] SpillFault),
    /// The spill pool refused the reservation.
    #[error("spill reservation refused")]
    SpillPool(#[source] DeviceReservationError),
    /// The cancellation token was set. Nothing was written or kept.
    #[error("the capacity run was cancelled")]
    Cancelled,
    /// Column count or slice lengths do not describe one design.
    #[error("design shape: {0}")]
    Shape(DesignShape),
    /// A weight is negative or not finite.
    #[error("weight at row {row}: {reason}")]
    Weight {
        /// Zero-based row in read order.
        row: u64,
        /// Which rule the weight broke.
        reason: WeightFault,
    },
    /// A design entry or a response is not finite.
    #[error("non-finite design entry at row {row}")]
    NonFinite {
        /// Zero-based row in read order.
        row: u64,
    },
    /// Every weight was zero, or the source had no rows.
    #[error("no positively weighted rows")]
    Empty,
    /// The weighted Gram is not positive definite.
    #[error("singular weighted Gram")]
    Singular,
    /// [`fit_wls`] refused the reduced design.
    #[error("weighted solve refused the reduced design")]
    Solve(#[source] StatsError),
    /// The charges do not fit in the ceiling.
    #[error("ceiling {ceiling} cannot hold {total} bytes ({over_by} over)")]
    OverCeiling {
        /// Declared ceiling.
        ceiling: u64,
        /// Charges that were refused.
        total: u64,
        /// Bytes by which the reservation exceeds the ceiling.
        over_by: u64,
        /// The refused reservation.
        charges: CeilingCharges,
    },
    /// A byte count or a row count overflowed.
    #[error("capacity byte count overflowed")]
    Overflow,
    /// The allocator refused a buffer this run had reserved room for.
    #[error("could not allocate {bytes} bytes for {what}")]
    Allocation {
        /// Which buffer was being allocated.
        what: &'static str,
        /// Bytes requested.
        bytes: u64,
        /// The allocator's refusal.
        #[source]
        source: TryReserveError,
    },
    /// The row file could not be opened, read, or seeked.
    #[error("row file: {context}")]
    Io {
        /// The operation that failed.
        context: &'static str,
        /// The operating-system error.
        #[source]
        source: io::Error,
    },
    /// The row file did not match this layout.
    #[error("row file: {0}")]
    RowFile(RowFileFault),
    /// The row stream hashed differently in the second pass.
    #[error("source rows changed between the cross-product pass and the diagnostic pass")]
    SourceChanged {
        /// Identity from the cross-product pass.
        first: [u8; 32],
        /// Identity from the diagnostic pass.
        second: [u8; 32],
    },
    /// [`WeightedRowSource::prepare_decoder`] was not called for this bound.
    #[error("decoder was not prepared for this batch bound")]
    DecoderNotPrepared,
}

impl From<IoFail> for WeightedCapacityError {
    fn from(fail: IoFail) -> Self {
        Self::Io { context: fail.context, source: fail.source }
    }
}

impl From<AllocFail> for WeightedCapacityError {
    fn from(fail: AllocFail) -> Self {
        Self::Allocation { what: "spill frame", bytes: fail.bytes, source: fail.source }
    }
}

/// The spill file of a disk-backed run and the reservation that pays for it.
///
/// The handle names the file, the length of its frame, and the digest the
/// frame has to carry. [`read_spilled_summary`] refuses a file that disagrees.
/// Dropping the handle returns the pool reservation and leaves the file;
/// [`Self::remove`] deletes the file as well.
#[derive(Debug)]
pub struct SpillHandle {
    path: PathBuf,
    frame_bytes: u64,
    digest: [u8; 32],
    lease: Option<DeviceReservationLease>,
}

impl SpillHandle {
    /// Path the frame was renamed to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Length of the whole frame in bytes.
    #[must_use]
    pub const fn frame_bytes(&self) -> u64 {
        self.frame_bytes
    }

    /// Digest of the payload, as the frame carries it.
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Whether a pool reservation is still held for this file.
    #[must_use]
    pub const fn holds_reservation(&self) -> bool {
        self.lease.is_some()
    }

    /// Delete the file and return the reservation. A file already gone is not an error.
    ///
    /// # Errors
    ///
    /// The file exists and could not be removed. The reservation is returned either way.
    pub fn remove(self) -> Result<(), WeightedCapacityError> {
        spill_file::remove_file(&self.path)?;
        Ok(())
    }
}

/// What a disk-backed run keeps resident: the handle and the run's counters.
///
/// [`read_spilled_summary`] turns the handle back into the summary.
#[derive(Debug)]
pub struct SpilledCapacity {
    /// Control that was requested.
    pub control: CapacityControl,
    /// Always [`OutputRetention::DiskBacked`].
    pub retention: OutputRetention,
    /// The spill file.
    pub handle: SpillHandle,
    /// Ceiling reservation for this run.
    pub charges: CeilingCharges,
    /// Non-empty batch reads in the cross-product pass.
    pub batches: u64,
    /// Cross-product pass plus the diagnostic pass.
    pub source_passes: u64,
    /// Streaming digest of the column count and of every `(x, y, w)` row.
    pub content_identity: [u8; 32],
}

/// What one capacity run returns, by retention mode.
#[derive(Debug)]
#[non_exhaustive]
pub enum WeightedCapacityOutput {
    /// [`OutputRetention::SummaryOnly`]: the summary, resident.
    Resident(WeightedCapacityResult),
    /// [`OutputRetention::DiskBacked`]: a handle to the summary on disk.
    Spilled(SpilledCapacity),
}

impl WeightedCapacityOutput {
    /// Ceiling reservation for this run.
    #[must_use]
    pub const fn charges(&self) -> CeilingCharges {
        match self {
            Self::Resident(result) => result.charges,
            Self::Spilled(spilled) => spilled.charges,
        }
    }

    /// The resident summary, if this run kept one.
    #[must_use]
    pub fn into_resident(self) -> Option<WeightedCapacityResult> {
        match self {
            Self::Resident(result) => Some(result),
            Self::Spilled(_) => None,
        }
    }

    /// The spill, if this run wrote one.
    #[must_use]
    pub fn into_spilled(self) -> Option<SpilledCapacity> {
        match self {
            Self::Spilled(spilled) => Some(spilled),
            Self::Resident(_) => None,
        }
    }
}

/// One batch of weighted rows, at most `max_rows` long.
pub trait WeightedRowSource {
    /// Design columns, including an intercept column when the caller wants one.
    #[must_use = "the design width selects the Gram"]
    fn ncols(&self) -> usize;

    /// Bytes of source this run already holds.
    ///
    /// A resident borrow charges the slices the caller is keeping. A file
    /// charges none of the unread body.
    #[must_use = "owned source bytes are part of the ceiling"]
    fn input_ownership_bytes(&self) -> u64;

    /// Bytes [`Self::prepare_decoder`] will request for `batch_rows`.
    ///
    /// # Errors
    ///
    /// `batch_rows` is zero, or the byte count overflows.
    fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError>;

    /// Allocate the decoder for `batch_rows` after the ceiling has accepted it.
    ///
    /// # Errors
    ///
    /// The bound is zero, the reservation overflows, or the allocator refuses.
    fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError>;

    /// Bytes requested from the allocator for the decoder. Zero before preparation.
    #[must_use = "the prepared decoder is compared with the batch bound"]
    fn allocated_decoder_bytes(&self) -> u64;

    /// Return to the first row.
    ///
    /// # Errors
    ///
    /// The file could not be seeked, or its length no longer matches its header.
    fn rewind(&mut self) -> Result<(), WeightedCapacityError>;

    /// Copy at most `max_rows` into the front of each slice.
    ///
    /// Returns the number of rows written. Zero means the source is exhausted.
    ///
    /// # Errors
    ///
    /// The decoder was not prepared, `max_rows` exceeds the bound, a slice is
    /// shorter than `max_rows`, or the file ends inside a row.
    fn read_batch(
        &mut self,
        max_rows: usize,
        x_rowmajor: &mut [f64],
        y: &mut [f64],
        weights: &mut [f64],
    ) -> Result<usize, WeightedCapacityError>;
}

/// Row-major resident design. The route copies one batch at a time.
#[derive(Clone, Debug)]
pub struct ResidentWeightedRows<'a> {
    x: &'a [f64],
    y: &'a [f64],
    weights: &'a [f64],
    ncols: usize,
    nrows: usize,
    cursor: usize,
    bound: Option<usize>,
}

impl<'a> ResidentWeightedRows<'a> {
    /// Borrow a row-major design, a response, and a weight per row.
    ///
    /// # Errors
    ///
    /// The column count is zero, or the slices are not one rectangular design.
    pub fn try_new(
        x_rowmajor: &'a [f64],
        y: &'a [f64],
        weights: &'a [f64],
    ) -> Result<Self, WeightedCapacityError> {
        if y.len() != weights.len() {
            return Err(WeightedCapacityError::Shape(DesignShape::WeightsLength));
        }
        let ncols = if y.is_empty() {
            return Err(WeightedCapacityError::Shape(DesignShape::ZeroColumns));
        } else if x_rowmajor.len() % y.len() != 0 {
            return Err(WeightedCapacityError::Shape(DesignShape::DesignLength));
        } else {
            x_rowmajor.len() / y.len()
        };
        if ncols == 0 {
            return Err(WeightedCapacityError::Shape(DesignShape::ZeroColumns));
        }
        Ok(Self { x: x_rowmajor, y, weights, ncols, nrows: y.len(), cursor: 0, bound: None })
    }
}

impl WeightedRowSource for ResidentWeightedRows<'_> {
    fn ncols(&self) -> usize {
        self.ncols
    }

    fn input_ownership_bytes(&self) -> u64 {
        let cells = self
            .x
            .len()
            .checked_add(self.y.len())
            .and_then(|sum| sum.checked_add(self.weights.len()));
        cells
            .and_then(|count| u64::try_from(count).ok())
            .and_then(|count| count.checked_mul(8))
            .unwrap_or(u64::MAX)
    }

    fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError> {
        if batch_rows == 0 {
            return Err(WeightedCapacityError::BatchBound);
        }
        Ok(0)
    }

    fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError> {
        if batch_rows == 0 {
            return Err(WeightedCapacityError::BatchBound);
        }
        self.bound = Some(batch_rows);
        Ok(())
    }

    fn allocated_decoder_bytes(&self) -> u64 {
        0
    }

    fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
        self.cursor = 0;
        Ok(())
    }

    fn read_batch(
        &mut self,
        max_rows: usize,
        x_rowmajor: &mut [f64],
        y: &mut [f64],
        weights: &mut [f64],
    ) -> Result<usize, WeightedCapacityError> {
        let bound = self.bound.ok_or(WeightedCapacityError::DecoderNotPrepared)?;
        require_batch_request(max_rows, bound)?;
        require_window(max_rows, self.ncols, x_rowmajor, y, weights)?;
        let take = max_rows.min(self.nrows.saturating_sub(self.cursor));
        let rows = self.cursor..self.cursor + take;
        let cells = rows.start * self.ncols..rows.end * self.ncols;
        x_rowmajor[..take * self.ncols].copy_from_slice(&self.x[cells]);
        y[..take].copy_from_slice(&self.y[rows.clone()]);
        weights[..take].copy_from_slice(&self.weights[rows]);
        self.cursor += take;
        Ok(take)
    }
}

/// Little-endian row file: magic `WGOL`, version, `ncols`, `nrows`, then
/// `nrows` records of `ncols` design values, the response, and the weight.
#[derive(Debug)]
pub struct FileWeightedRows {
    file: File,
    ncols: usize,
    nrows: u64,
    rows_left: u64,
    expected_len: u64,
    bound: Option<usize>,
    decoder: Vec<u8>,
    reserved_bytes: u64,
}

impl FileWeightedRows {
    /// Open a file written by [`encode_weighted_row_file`].
    ///
    /// The body stays on disk until [`WeightedRowSource::prepare_decoder`].
    ///
    /// # Errors
    ///
    /// The header is short, the magic or version does not match, a count does
    /// not fit, or the file length is not the header plus `nrows` whole
    /// records (trailing bytes or truncation).
    pub fn open(path: &Path) -> Result<Self, WeightedCapacityError> {
        let mut file = File::open(path).map_err(io_failure("open the row file"))?;
        let mut header = [0_u8; HEADER_LEN];
        file.read_exact(&mut header).map_err(io_failure("read the row-file header"))?;
        if &header[0..4] != MAGIC {
            return Err(WeightedCapacityError::RowFile(RowFileFault::BadMagic));
        }
        let version = read_u32(&header[4..8])?;
        if version != VERSION {
            return Err(WeightedCapacityError::RowFile(RowFileFault::UnsupportedVersion {
                found: version,
            }));
        }
        let ncols_u = read_u32(&header[8..12])?;
        let nrows = read_u64(&header[12..20])?;
        let ncols = usize::try_from(ncols_u).map_err(|_| WeightedCapacityError::Overflow)?;
        if ncols == 0 {
            return Err(WeightedCapacityError::Shape(DesignShape::ZeroColumns));
        }
        let expected_len = batch_byte_len(1, ncols)?
            .checked_mul(nrows)
            .and_then(|body| body.checked_add(HEADER_BYTES))
            .ok_or(WeightedCapacityError::Overflow)?;
        let rows = Self {
            file,
            ncols,
            nrows,
            rows_left: nrows,
            expected_len,
            bound: None,
            decoder: Vec::new(),
            reserved_bytes: 0,
        };
        rows.verify_length()?;
        Ok(rows)
    }

    fn verify_length(&self) -> Result<(), WeightedCapacityError> {
        let actual_bytes = self.file.metadata().map_err(io_failure("measure the row file"))?.len();
        if actual_bytes == self.expected_len {
            Ok(())
        } else {
            Err(WeightedCapacityError::RowFile(RowFileFault::LengthMismatch {
                expected_bytes: self.expected_len,
                actual_bytes,
            }))
        }
    }
}

impl WeightedRowSource for FileWeightedRows {
    fn ncols(&self) -> usize {
        self.ncols
    }

    fn input_ownership_bytes(&self) -> u64 {
        0
    }

    fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError> {
        batch_byte_len(batch_rows, self.ncols)
    }

    fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError> {
        let reserved = batch_byte_len(batch_rows, self.ncols)?;
        let reserved_usize =
            usize::try_from(reserved).map_err(|_| WeightedCapacityError::Overflow)?;
        let mut decoder = Vec::new();
        decoder
            .try_reserve_exact(reserved_usize)
            .map_err(|source| allocation_failure("file decoder", reserved, source))?;
        self.decoder = decoder;
        self.reserved_bytes = reserved;
        self.bound = Some(batch_rows);
        Ok(())
    }

    fn allocated_decoder_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
        self.verify_length()?;
        self.file
            .seek(SeekFrom::Start(HEADER_BYTES))
            .map_err(io_failure("seek to the first row"))?;
        self.rows_left = self.nrows;
        Ok(())
    }

    fn read_batch(
        &mut self,
        max_rows: usize,
        x_rowmajor: &mut [f64],
        y: &mut [f64],
        weights: &mut [f64],
    ) -> Result<usize, WeightedCapacityError> {
        let bound = self.bound.ok_or(WeightedCapacityError::DecoderNotPrepared)?;
        require_batch_request(max_rows, bound)?;
        require_window(max_rows, self.ncols, x_rowmajor, y, weights)?;
        let available =
            usize::try_from(self.rows_left).map_err(|_| WeightedCapacityError::Overflow)?;
        let take = max_rows.min(available);
        if take == 0 {
            return Ok(0);
        }
        let bytes = batch_byte_len(take, self.ncols)?;
        let need = usize::try_from(bytes).map_err(|_| WeightedCapacityError::Overflow)?;
        if need > self.decoder.capacity() {
            return Err(WeightedCapacityError::ReadPastBatch { requested: take, bound });
        }
        self.decoder.resize(need, 0);
        self.file
            .read_exact(&mut self.decoder[..need])
            .map_err(io_failure("read a batch of records"))?;
        parse_records(&self.decoder[..need], self.ncols, take, x_rowmajor, y, weights)?;
        self.rows_left -= u64::try_from(take).map_err(|_| WeightedCapacityError::Overflow)?;
        Ok(take)
    }
}

/// Encode the row file [`FileWeightedRows`] reads.
///
/// # Errors
///
/// The slices are not one rectangular design, or a count does not fit.
pub fn encode_weighted_row_file(
    x_rowmajor: &[f64],
    y: &[f64],
    weights: &[f64],
) -> Result<Vec<u8>, WeightedCapacityError> {
    if y.len() != weights.len() || y.is_empty() || x_rowmajor.len() % y.len() != 0 {
        return Err(WeightedCapacityError::Shape(DesignShape::DesignLength));
    }
    let ncols = x_rowmajor.len() / y.len();
    if ncols == 0 {
        return Err(WeightedCapacityError::Shape(DesignShape::ZeroColumns));
    }
    let ncols_u = u32::try_from(ncols).map_err(|_| WeightedCapacityError::Overflow)?;
    let nrows = u64::try_from(y.len()).map_err(|_| WeightedCapacityError::Overflow)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&ncols_u.to_le_bytes());
    bytes.extend_from_slice(&nrows.to_le_bytes());
    for ((x_row, response), weight) in x_rowmajor.chunks_exact(ncols).zip(y).zip(weights) {
        for value in x_row.iter().chain([response, weight]) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

/// Run one capacity control.
///
/// Resident and file sources of the same rows share this function, the same
/// retention mode, and the same charges formula. [`OutputRetention::FullReusable`]
/// is refused: it would keep every source row. [`OutputRetention::SummaryOnly`]
/// returns [`WeightedCapacityOutput::Resident`]; [`OutputRetention::DiskBacked`]
/// returns [`WeightedCapacityOutput::Spilled`]. Both carry the same numbers.
///
/// # Errors
///
/// The ceiling is too small, the batch bound is zero, a weight or a row is not
/// usable, the Gram is singular, the source cannot be read, the source rows
/// differ between the two passes, a disk-backed run has no spill path or the
/// spill pool or the file system refuses it, or the run is cancelled.
pub fn run_weighted_capacity(
    control: CapacityControl,
    source: &mut dyn WeightedRowSource,
    options: &CapacityOptions,
) -> Result<WeightedCapacityOutput, WeightedCapacityError> {
    let plan = plan_run(&*source, options)?;
    source.prepare_decoder(plan.batch_rows)?;
    source.rewind()?;
    let mut scan = Scan::open(plan.batch_rows, plan.ncols, options.cancellation.clone())?;
    let first = cross_product_pass(source, &mut scan, plan.ncols)?;
    let solved = solve_weighted(&first.folded.cross_products, plan.ncols)?;
    source.rewind()?;
    let diagnostics = diagnostic_pass(source, &mut scan, &solved, first.digest)?;
    drop(scan);
    let result = Assembly {
        control,
        retention: options.retention,
        charges: plan.charges,
        batches: first.batches,
        content_identity: first.digest,
        folded: first.folded,
        solved,
        diagnostics,
    }
    .into_result()?;
    match options.retention {
        OutputRetention::DiskBacked => {
            spill_summary(result, options).map(WeightedCapacityOutput::Spilled)
        }
        _ => Ok(WeightedCapacityOutput::Resident(result)),
    }
}

/// Checked shape, retention, and ceiling reservation for one run.
struct RunPlan {
    ncols: usize,
    batch_rows: usize,
    charges: CeilingCharges,
}

fn plan_run(
    source: &dyn WeightedRowSource,
    options: &CapacityOptions,
) -> Result<RunPlan, WeightedCapacityError> {
    let ncols = source.ncols();
    if ncols == 0 {
        return Err(WeightedCapacityError::Shape(DesignShape::ZeroColumns));
    }
    if options.batch_rows == 0 {
        return Err(WeightedCapacityError::BatchBound);
    }
    let source_decoder = source.decoder_reservation(options.batch_rows)?;
    let charges = plan_charges(ncols, options, source.input_ownership_bytes(), source_decoder)?;
    if charges.total_bytes > options.ceiling_bytes {
        return Err(WeightedCapacityError::OverCeiling {
            ceiling: options.ceiling_bytes,
            total: charges.total_bytes,
            over_by: charges.total_bytes - options.ceiling_bytes,
            charges,
        });
    }
    Ok(RunPlan { ncols, batch_rows: options.batch_rows, charges })
}

/// Buffers both passes share: the batch window and the row hash.
struct Scan {
    window: Window,
    identity: RowIdentity,
    cancellation: Option<CancellationToken>,
}

impl Scan {
    fn open(
        batch_rows: usize,
        ncols: usize,
        cancellation: Option<CancellationToken>,
    ) -> Result<Self, WeightedCapacityError> {
        Ok(Self {
            window: Window::allocate(batch_rows, ncols)?,
            identity: RowIdentity::open(ncols)?,
            cancellation,
        })
    }
}

fn check_cancelled(token: Option<&CancellationToken>) -> Result<(), WeightedCapacityError> {
    if token.is_some_and(CancellationToken::is_cancelled) {
        Err(WeightedCapacityError::Cancelled)
    } else {
        Ok(())
    }
}

struct FirstPass {
    folded: Folded,
    batches: u64,
    digest: [u8; 32],
}

fn cross_product_pass(
    source: &mut dyn WeightedRowSource,
    scan: &mut Scan,
    ncols: usize,
) -> Result<FirstPass, WeightedCapacityError> {
    let mut fold = Fold::new(ncols)?;
    let Scan { window, identity, cancellation } = scan;
    let batches =
        scan_rows(source, window, cancellation.as_ref(), |index, row, response, weight| {
            fold.push(index, row, response, weight)?;
            identity.push(row, response, weight);
            Ok(())
        })?;
    if fold.positive_rows == 0 {
        return Err(WeightedCapacityError::Empty);
    }
    Ok(FirstPass { folded: fold.finish(), batches, digest: identity.digest() })
}

fn diagnostic_pass(
    source: &mut dyn WeightedRowSource,
    scan: &mut Scan,
    solved: &Solved,
    expected: [u8; 32],
) -> Result<DiagSummary, WeightedCapacityError> {
    let mut diagnostics = DiagAccum::new(&solved.coefficients, &solved.bread)?;
    let Scan { window, identity, cancellation } = scan;
    identity.reset();
    scan_rows(source, window, cancellation.as_ref(), |index, row, response, weight| {
        diagnostics.push(index, row, response, weight)?;
        identity.push(row, response, weight);
        Ok(())
    })?;
    let second = identity.digest();
    if second != expected {
        return Err(WeightedCapacityError::SourceChanged { first: expected, second });
    }
    Ok(diagnostics.finish())
}

struct Window {
    batch_rows: usize,
    ncols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
}

impl Window {
    fn allocate(batch_rows: usize, ncols: usize) -> Result<Self, WeightedCapacityError> {
        let cells = batch_rows.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
        Ok(Self {
            batch_rows,
            ncols,
            x: zeros(cells, "batch design window")?,
            y: zeros(batch_rows, "batch response window")?,
            w: zeros(batch_rows, "batch weight window")?,
        })
    }
}

/// Weighted moments around running means.
///
/// Each positive row updates the means and the centered co-moments with the
/// weighted Welford step, so the sums stay on the scale of the spread rather
/// than the scale of the raw products. [`Self::finish`] rebuilds the raw
/// cross-products once.
struct Fold {
    ncols: usize,
    mean: Vec<f64>,
    offset: Vec<f64>,
    comoment: Vec<f64>,
    cross: Vec<f64>,
    mean_y: f64,
    y_ss: f64,
    n_eff: f64,
    positive_rows: u64,
}

/// Raw cross-products rebuilt from the centered moments.
struct Folded {
    cross_products: GaussianCrossProducts,
    positive_rows: u64,
}

impl Fold {
    fn new(ncols: usize) -> Result<Self, WeightedCapacityError> {
        let gram = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
        Ok(Self {
            ncols,
            mean: zeros(ncols, "column means")?,
            offset: zeros(ncols, "row offsets")?,
            comoment: zeros(gram, "centered Gram")?,
            cross: zeros(ncols, "centered cross-moment")?,
            mean_y: 0.0,
            y_ss: 0.0,
            n_eff: 0.0,
            positive_rows: 0,
        })
    }

    fn push(
        &mut self,
        index: u64,
        row: &[f64],
        y: f64,
        weight: f64,
    ) -> Result<(), WeightedCapacityError> {
        validate_row(index, row, self.ncols, y, weight)?;
        if !weight_is_positive(weight) {
            return Ok(());
        }
        let total = self.n_eff + weight;
        let share = weight / total;
        for ((offset, mean), &x) in self.offset.iter_mut().zip(self.mean.iter_mut()).zip(row) {
            *offset = x - *mean;
            *mean += share * *offset;
        }
        let y_offset = y - self.mean_y;
        self.mean_y += share * y_offset;
        let y_remainder = y - self.mean_y;
        // Upper triangle only; `finish` mirrors it.
        for (i, (&offset, line)) in
            self.offset.iter().zip(self.comoment.chunks_exact_mut(self.ncols)).enumerate()
        {
            let lead = weight * offset;
            for ((cell, &x), &mean) in line[i..].iter_mut().zip(&row[i..]).zip(&self.mean[i..]) {
                *cell += lead * (x - mean);
            }
            self.cross[i] += lead * y_remainder;
        }
        self.y_ss += weight * y_offset * y_remainder;
        self.n_eff = total;
        self.positive_rows =
            self.positive_rows.checked_add(1).ok_or(WeightedCapacityError::Overflow)?;
        Ok(())
    }

    /// Raw `X'WX = C + Σw·m·m'`, `X'Wy = c + Σw·m·m_y`, `y'Wy = s + Σw·m_y²`.
    ///
    /// The centered buffers become the raw ones in place; the means and the
    /// row offsets are released.
    fn finish(self) -> Folded {
        let Self {
            ncols,
            mean,
            offset,
            comoment: mut xtwx,
            cross: mut xtwy,
            mean_y,
            y_ss,
            n_eff,
            positive_rows,
        } = self;
        drop(offset);
        for (i, (line, &lead)) in xtwx.chunks_exact_mut(ncols).zip(&mean).enumerate() {
            for (cell, &other) in line[i..].iter_mut().zip(&mean[i..]) {
                *cell += n_eff * lead * other;
            }
        }
        for i in 1..ncols {
            for j in 0..i {
                xtwx[i * ncols + j] = xtwx[j * ncols + i];
            }
        }
        for (cell, &lead) in xtwy.iter_mut().zip(&mean) {
            *cell += n_eff * lead * mean_y;
        }
        let ytwy = y_ss + n_eff * mean_y * mean_y;
        Folded { cross_products: GaussianCrossProducts { n_eff, xtwx, xtwy, ytwy }, positive_rows }
    }
}

struct RowIdentity {
    hasher: blake3::Hasher,
    ncols_tag: u32,
    staging: Vec<u8>,
}

impl RowIdentity {
    fn open(ncols: usize) -> Result<Self, WeightedCapacityError> {
        let row_bytes = batch_byte_len(1, ncols)?;
        let width = usize::try_from(row_bytes).map_err(|_| WeightedCapacityError::Overflow)?;
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(width)
            .map_err(|source| allocation_failure("row staging", row_bytes, source))?;
        let ncols_tag = u32::try_from(ncols).map_err(|_| WeightedCapacityError::Overflow)?;
        Ok(Self { hasher: Self::seeded(ncols_tag), ncols_tag, staging })
    }

    fn seeded(ncols_tag: u32) -> blake3::Hasher {
        let mut hasher = blake3::Hasher::new_derive_key(IDENTITY_DOMAIN);
        hasher.update(&ncols_tag.to_le_bytes());
        hasher
    }

    /// Start a fresh digest for the next pass without a second staging buffer.
    fn reset(&mut self) {
        self.hasher = Self::seeded(self.ncols_tag);
    }

    fn push(&mut self, row: &[f64], y: f64, weight: f64) {
        self.staging.clear();
        for value in row.iter().chain(&[y, weight]) {
            self.staging.extend_from_slice(&value.to_le_bytes());
        }
        self.hasher.update(&self.staging);
    }

    fn digest(&self) -> [u8; 32] {
        self.hasher.finalize().into()
    }
}

struct Solved {
    coefficients: Vec<f64>,
    diagnostics: FitDiagnostics,
    bread: Vec<f64>,
}

/// Leverage, residual, and `HC0` meat accumulators. Borrows the solve.
struct DiagAccum<'a> {
    beta: &'a [f64],
    bread: &'a [f64],
    meat: Vec<f64>,
    scratch: Vec<f64>,
    max_leverage: f64,
    sum_leverage: f64,
    saturated_rows: u64,
    weighted_rss: f64,
    max_abs_residual: f64,
}

/// What the diagnostic pass leaves behind once the solve borrow ends.
struct DiagSummary {
    meat: Vec<f64>,
    max_leverage: f64,
    sum_leverage: f64,
    saturated_rows: u64,
    weighted_rss: f64,
    max_abs_residual: f64,
}

impl<'a> DiagAccum<'a> {
    fn new(beta: &'a [f64], bread: &'a [f64]) -> Result<Self, WeightedCapacityError> {
        let p = beta.len();
        let gram = p.checked_mul(p).ok_or(WeightedCapacityError::Overflow)?;
        Ok(Self {
            beta,
            bread,
            meat: zeros(gram, "HC0 meat")?,
            scratch: zeros(p, "leverage scratch")?,
            max_leverage: 0.0,
            sum_leverage: 0.0,
            saturated_rows: 0,
            weighted_rss: 0.0,
            max_abs_residual: 0.0,
        })
    }

    fn push(
        &mut self,
        index: u64,
        row: &[f64],
        y: f64,
        weight: f64,
    ) -> Result<(), WeightedCapacityError> {
        validate_row(index, row, self.beta.len(), y, weight)?;
        if !weight_is_positive(weight) {
            return Ok(());
        }
        let p = self.beta.len();
        let scale = weight.sqrt();
        let predicted: f64 = self.beta.iter().zip(row).map(|(&coef, &value)| coef * value).sum();
        let residual = y - predicted;
        let scaled_residual = scale * residual;
        for (line, &x1) in self.meat.chunks_exact_mut(p).zip(row) {
            let lead = scaled_residual.powi(2) * (scale * x1);
            for (cell, &x2) in line.iter_mut().zip(row) {
                *cell += lead * (scale * x2);
            }
        }
        for (slot, bread_row) in self.scratch.iter_mut().zip(self.bread.chunks_exact(p)) {
            *slot = bread_row.iter().zip(row).map(|(&b, &x)| b * (scale * x)).sum();
        }
        let hat: f64 = row.iter().zip(&self.scratch).map(|(&x, &s)| (scale * x) * s).sum();
        self.sum_leverage += hat;
        if hat > self.max_leverage {
            self.max_leverage = hat;
        }
        if hat.partial_cmp(&(1.0 - 1.0e-8)).is_some_and(Ordering::is_ge) {
            self.saturated_rows =
                self.saturated_rows.checked_add(1).ok_or(WeightedCapacityError::Overflow)?;
        }
        self.weighted_rss += scaled_residual.powi(2);
        let abs_residual = residual.abs();
        if abs_residual > self.max_abs_residual {
            self.max_abs_residual = abs_residual;
        }
        Ok(())
    }

    fn finish(self) -> DiagSummary {
        DiagSummary {
            meat: self.meat,
            max_leverage: self.max_leverage,
            sum_leverage: self.sum_leverage,
            saturated_rows: self.saturated_rows,
            weighted_rss: self.weighted_rss,
            max_abs_residual: self.max_abs_residual,
        }
    }
}

/// Everything the two passes produced, ready to become the result.
struct Assembly {
    control: CapacityControl,
    retention: OutputRetention,
    charges: CeilingCharges,
    batches: u64,
    content_identity: [u8; 32],
    folded: Folded,
    solved: Solved,
    diagnostics: DiagSummary,
}

impl Assembly {
    fn into_result(self) -> Result<WeightedCapacityResult, WeightedCapacityError> {
        let Self {
            control,
            retention,
            charges,
            batches,
            content_identity,
            folded,
            solved,
            diagnostics,
        } = self;
        let p = solved.coefficients.len();
        let robust_covariance = sandwich_product(&solved.bread, &diagnostics.meat, p)?;
        Ok(WeightedCapacityResult {
            control,
            retention,
            coefficients: solved.coefficients,
            robust_covariance,
            leverage: LeverageSummary {
                max: diagnostics.max_leverage,
                sum: diagnostics.sum_leverage,
                saturated_rows: diagnostics.saturated_rows,
            },
            residual: ResidualCheck {
                weighted_rss: diagnostics.weighted_rss,
                max_abs_residual: diagnostics.max_abs_residual,
                positive_weight_rows: folded.positive_rows,
            },
            cross_products: folded.cross_products,
            content_identity,
            fit: solved.diagnostics,
            charges,
            batches,
            source_passes: 2,
        })
    }
}

fn scan_rows(
    source: &mut dyn WeightedRowSource,
    window: &mut Window,
    cancellation: Option<&CancellationToken>,
    mut on_row: impl FnMut(u64, &[f64], f64, f64) -> Result<(), WeightedCapacityError>,
) -> Result<u64, WeightedCapacityError> {
    let mut batches = 0_u64;
    let mut index = 0_u64;
    loop {
        let n =
            source.read_batch(window.batch_rows, &mut window.x, &mut window.y, &mut window.w)?;
        if n == 0 {
            break;
        }
        check_cancelled(cancellation)?;
        batches = batches.checked_add(1).ok_or(WeightedCapacityError::Overflow)?;
        let rows = window.x.chunks_exact(window.ncols).zip(&window.y).zip(&window.w).take(n);
        for ((row, &response), &weight) in rows {
            on_row(index, row, response, weight)?;
            index = index.checked_add(1).ok_or(WeightedCapacityError::Overflow)?;
        }
    }
    Ok(batches)
}

fn spill_summary(
    result: WeightedCapacityResult,
    options: &CapacityOptions,
) -> Result<SpilledCapacity, WeightedCapacityError> {
    let spill = options.spill.as_ref().ok_or(WeightedCapacityError::SpillPathRequired)?;
    spill_with(result, options.cancellation.as_ref(), spill, spill_file::write_whole)
}

/// Reserve, encode, write, and read back one summary. `write` stores the frame
/// in the temporary file, so a test can make that step fail.
#[allow(
    clippy::needless_pass_by_value,
    reason = "spilling consumes the in-memory summary and releases it after committing the frame"
)]
fn spill_with(
    result: WeightedCapacityResult,
    cancellation: Option<&CancellationToken>,
    spill: &SpillOptions,
    write: impl FnOnce(&mut File, &[u8]) -> io::Result<()>,
) -> Result<SpilledCapacity, WeightedCapacityError> {
    check_cancelled(cancellation)?;
    let lease = spill
        .pool
        .as_ref()
        .map(|pool| pool.try_grant(0, result.charges.spill_bytes, false))
        .transpose()
        .map_err(WeightedCapacityError::SpillPool)?;
    let payload_len = spill_payload_bytes(result.coefficients.len())?;
    let mut frame = spill_file::begin_frame(payload_len)?;
    encode_summary(&result, &mut frame)?;
    spill_file::seal_frame(&mut frame);
    if u64::try_from(frame.len()).ok() != Some(result.charges.spill_bytes) {
        return Err(WeightedCapacityError::Spill(SpillFault::PayloadLayout));
    }
    let receipt =
        spill_file::commit_frame::<WeightedCapacityError>(&spill.path, frame, write, || {
            check_cancelled(cancellation)
        })?;
    Ok(SpilledCapacity {
        control: result.control,
        retention: result.retention,
        handle: SpillHandle {
            path: spill.path.clone(),
            frame_bytes: receipt.frame_bytes,
            digest: receipt.digest,
            lease,
        },
        charges: result.charges,
        batches: result.batches,
        source_passes: result.source_passes,
        content_identity: result.content_identity,
    })
}

/// Read a spilled summary back.
///
/// The result is the one a resident run returns, with [`OutputRetention::DiskBacked`]
/// as its retention and the charges of the spilled run. The read allocates one
/// buffer of [`SpillHandle::frame_bytes`], which the run already charged under
/// [`CeilingCharges::decoder_bytes`].
///
/// # Errors
///
/// The file is missing, has a length other than the handle's, or fails its
/// magic, version, digest, or layout check.
pub fn read_spilled_summary(
    handle: &SpillHandle,
) -> Result<WeightedCapacityResult, WeightedCapacityError> {
    let declared = Declared { frame_bytes: handle.frame_bytes, digest: handle.digest };
    let frame = spill_file::read_verified::<WeightedCapacityError>(&handle.path, declared)?;
    let payload = spill_file::verify_frame(&frame)?;
    decode_summary(payload)
}

fn control_code(control: CapacityControl) -> u8 {
    match control {
        CapacityControl::Glm => 0,
        CapacityControl::GaussianBayes => 1,
    }
}

fn put_f64s(out: &mut Vec<u8>, values: &[f64]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Canonical little-endian payload: control, `ncols`, the matrices and vectors,
/// the scalar summaries, the identity, the fit diagnostics, the charges, and
/// the counters, in that order.
fn encode_summary(
    result: &WeightedCapacityResult,
    out: &mut Vec<u8>,
) -> Result<(), WeightedCapacityError> {
    let p = result.coefficients.len();
    let gram = p.checked_mul(p).ok_or(WeightedCapacityError::Overflow)?;
    let shaped = p > 0
        && result.robust_covariance.len() == gram
        && result.cross_products.xtwx.len() == gram
        && result.cross_products.xtwy.len() == p;
    if !shaped {
        return Err(WeightedCapacityError::Spill(SpillFault::PayloadLayout));
    }
    let backend = SPILL_BACKENDS
        .iter()
        .position(|name| *name == result.fit.backend)
        .and_then(|index| u8::try_from(index).ok())
        .ok_or(WeightedCapacityError::Spill(SpillFault::UnknownBackend))?;
    let ncols = u32::try_from(p).map_err(|_| WeightedCapacityError::Overflow)?;
    let rank = u64::try_from(result.fit.rank).map_err(|_| WeightedCapacityError::Overflow)?;
    out.push(control_code(result.control));
    out.extend_from_slice(&ncols.to_le_bytes());
    put_f64s(out, &result.coefficients);
    put_f64s(out, &result.robust_covariance);
    put_f64s(out, &result.cross_products.xtwx);
    put_f64s(out, &result.cross_products.xtwy);
    put_f64s(out, &[result.cross_products.n_eff, result.cross_products.ytwy]);
    put_f64s(out, &[result.leverage.max, result.leverage.sum]);
    put_u64(out, result.leverage.saturated_rows);
    put_f64s(out, &[result.residual.weighted_rss, result.residual.max_abs_residual]);
    put_u64(out, result.residual.positive_weight_rows);
    out.extend_from_slice(&result.content_identity);
    put_u64(out, rank);
    out.push(u8::from(result.fit.rcond.is_some()));
    put_f64s(out, &[result.fit.rcond.unwrap_or(0.0)]);
    out.push(backend);
    out.extend_from_slice(&result.fit.grow_count.to_le_bytes());
    let charges = &result.charges;
    for word in [
        charges.input_ownership_bytes,
        charges.content_identity_bytes,
        charges.decoder_bytes,
        charges.spill_bytes,
        charges.output_bytes,
        charges.total_bytes,
        result.batches,
        result.source_passes,
    ] {
        put_u64(out, word);
    }
    Ok(())
}

/// Cursor over a spill payload. Every read that runs past the end is a layout fault.
struct PayloadReader<'a> {
    rest: &'a [u8],
}

impl<'a> PayloadReader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], WeightedCapacityError> {
        if self.rest.len() < len {
            return Err(WeightedCapacityError::Spill(SpillFault::PayloadLayout));
        }
        let (head, tail) = self.rest.split_at(len);
        self.rest = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], WeightedCapacityError> {
        self.take(N)?
            .try_into()
            .map_err(|_| WeightedCapacityError::Spill(SpillFault::PayloadLayout))
    }

    fn u8(&mut self) -> Result<u8, WeightedCapacityError> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, WeightedCapacityError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, WeightedCapacityError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> Result<f64, WeightedCapacityError> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    fn f64s(&mut self, len: usize, what: &'static str) -> Result<Vec<f64>, WeightedCapacityError> {
        let raw = self.take(len.checked_mul(8).ok_or(WeightedCapacityError::Overflow)?)?;
        let mut values = zeros(len, what)?;
        for (slot, chunk) in values.iter_mut().zip(raw.chunks_exact(8)) {
            *slot = decode_f64(chunk)?;
        }
        Ok(values)
    }
}

fn decode_summary(payload: &[u8]) -> Result<WeightedCapacityResult, WeightedCapacityError> {
    let layout = || WeightedCapacityError::Spill(SpillFault::PayloadLayout);
    let mut reader = PayloadReader { rest: payload };
    let control = match reader.u8()? {
        0 => CapacityControl::Glm,
        1 => CapacityControl::GaussianBayes,
        _ => return Err(layout()),
    };
    let p = usize::try_from(reader.u32()?).map_err(|_| WeightedCapacityError::Overflow)?;
    if p == 0 || u64::try_from(payload.len()).ok() != Some(spill_payload_bytes(p)?) {
        return Err(layout());
    }
    let gram = p * p;
    let coefficients = reader.f64s(p, "spilled coefficients")?;
    let robust_covariance = reader.f64s(gram, "spilled covariance")?;
    let xtwx = reader.f64s(gram, "spilled X'WX")?;
    let xtwy = reader.f64s(p, "spilled X'Wy")?;
    let (n_eff, ytwy) = (reader.f64()?, reader.f64()?);
    let leverage =
        LeverageSummary { max: reader.f64()?, sum: reader.f64()?, saturated_rows: reader.u64()? };
    let residual = ResidualCheck {
        weighted_rss: reader.f64()?,
        max_abs_residual: reader.f64()?,
        positive_weight_rows: reader.u64()?,
    };
    let content_identity = reader.array::<32>()?;
    let rank = usize::try_from(reader.u64()?).map_err(|_| WeightedCapacityError::Overflow)?;
    let has_rcond = match reader.u8()? {
        0 => false,
        1 => true,
        _ => return Err(layout()),
    };
    let rcond = reader.f64()?;
    let backend = SPILL_BACKENDS
        .get(usize::from(reader.u8()?))
        .copied()
        .ok_or(WeightedCapacityError::Spill(SpillFault::UnknownBackend))?;
    let grow_count = reader.u32()?;
    let charges = CeilingCharges {
        input_ownership_bytes: reader.u64()?,
        content_identity_bytes: reader.u64()?,
        decoder_bytes: reader.u64()?,
        spill_bytes: reader.u64()?,
        output_bytes: reader.u64()?,
        total_bytes: reader.u64()?,
    };
    let batches = reader.u64()?;
    let source_passes = reader.u64()?;
    if !reader.rest.is_empty() {
        return Err(layout());
    }
    Ok(WeightedCapacityResult {
        control,
        retention: OutputRetention::DiskBacked,
        coefficients,
        robust_covariance,
        leverage,
        residual,
        cross_products: GaussianCrossProducts { n_eff, xtwx, xtwy, ytwy },
        content_identity,
        fit: FitDiagnostics::new(rank, has_rcond.then_some(rcond), backend, grow_count),
        charges,
        batches,
        source_passes,
    })
}

/// `L` is the row-major Cholesky factor of `X'WX`. That buffer is also the
/// column-major layout of `L'`. [`fit_wls`] on `L'` with right-hand side
/// `L⁻¹ X'Wy` is the weighted normal equation.
fn solve_weighted(
    cross_products: &GaussianCrossProducts,
    p: usize,
) -> Result<Solved, WeightedCapacityError> {
    let factor = cholesky_spd(&cross_products.xtwx, p).ok_or(WeightedCapacityError::Singular)?;
    let reduced_y = forward_substitute(&factor, p, &cross_products.xtwy)?;
    let mut ones = zeros(p, "unit weights")?;
    ones.fill(1.0);
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_wls(&factor, p, p, &reduced_y, &ones, &FaerBackend, &mut workspace)
        .map_err(WeightedCapacityError::Solve)?;
    if fit.coefficients.len() != p {
        return Err(WeightedCapacityError::Shape(DesignShape::SolveWidth));
    }
    let bread = invert_square(&cross_products.xtwx, p).ok_or(WeightedCapacityError::Singular)?;
    Ok(Solved { coefficients: fit.coefficients, diagnostics: fit.diagnostics, bread })
}

fn forward_substitute(
    factor: &[f64],
    p: usize,
    rhs: &[f64],
) -> Result<Vec<f64>, WeightedCapacityError> {
    let mut solved = zeros(p, "forward substitution")?;
    for (i, (factor_row, &b)) in factor.chunks_exact(p).zip(rhs).enumerate() {
        let diag = factor_row[i];
        if diag.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(WeightedCapacityError::Singular);
        }
        let acc = factor_row[..i].iter().zip(&solved[..i]).fold(b, |acc, (&l, &s)| acc - l * s);
        solved[i] = acc / diag;
    }
    Ok(solved)
}

/// `bread · meat · bread` for square row-major matrices.
fn sandwich_product(
    bread: &[f64],
    meat: &[f64],
    ncols: usize,
) -> Result<Vec<f64>, WeightedCapacityError> {
    let cells = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
    let mut tmp = zeros(cells, "sandwich intermediate")?;
    multiply_square(bread, meat, ncols, &mut tmp);
    let mut out = zeros(cells, "robust covariance")?;
    multiply_square(&tmp, bread, ncols, &mut out);
    Ok(out)
}

fn multiply_square(left: &[f64], right: &[f64], n: usize, out: &mut [f64]) {
    for (out_row, left_row) in out.chunks_exact_mut(n).zip(left.chunks_exact(n)) {
        for (j, cell) in out_row.iter_mut().enumerate() {
            *cell = left_row.iter().zip(right.iter().skip(j).step_by(n)).map(|(a, b)| a * b).sum();
        }
    }
}

fn plan_charges(
    ncols: usize,
    options: &CapacityOptions,
    input_ownership_bytes: u64,
    source_decoder_bytes: u64,
) -> Result<CeilingCharges, WeightedCapacityError> {
    let payload = payload_bytes(ncols)?;
    let (output_bytes, spill_bytes, staging_bytes) = match options.retention {
        OutputRetention::SummaryOnly => (payload, 0, 0),
        OutputRetention::DiskBacked => {
            let spill = options.spill.as_ref().ok_or(WeightedCapacityError::SpillPathRequired)?;
            let frame = spill_frame_bytes(ncols)?;
            // The finished summary is encoded into one frame buffer, which is
            // dropped before the read-back allocates the next one.
            let staging = frame.checked_add(payload).ok_or(WeightedCapacityError::Overflow)?;
            (handle_bytes(&spill.path)?, frame, staging)
        }
        OutputRetention::FullReusable => return Err(WeightedCapacityError::RetentionKeepsRows),
    };
    let identity = content_identity_bytes(ncols)?;
    let decoder = working_bytes(options.batch_rows, ncols, source_decoder_bytes)?
        .checked_add(staging_bytes)
        .ok_or(WeightedCapacityError::Overflow)?;
    let total_bytes = input_ownership_bytes
        .checked_add(identity)
        .and_then(|sum| sum.checked_add(decoder))
        .and_then(|sum| sum.checked_add(spill_bytes))
        .and_then(|sum| sum.checked_add(output_bytes))
        .ok_or(WeightedCapacityError::Overflow)?;
    Ok(CeilingCharges {
        input_ownership_bytes,
        content_identity_bytes: identity,
        decoder_bytes: decoder,
        spill_bytes,
        output_bytes,
        total_bytes,
    })
}

/// Resident bytes of a [`SpillHandle`] naming `path`.
fn handle_bytes(path: &Path) -> Result<u64, WeightedCapacityError> {
    let path_bytes =
        u64::try_from(path.as_os_str().len()).map_err(|_| WeightedCapacityError::Overflow)?;
    size_u64::<SpillHandle>()?.checked_add(path_bytes).ok_or(WeightedCapacityError::Overflow)
}

/// Exact length of the spill payload for `ncols` columns: seven fixed `f64`
/// values and `2p² + 2p` matrix and vector cells, eleven `u64` counters, and
/// forty-three bytes of identity, flags, codes, and the grow count.
fn spill_payload_bytes(ncols: usize) -> Result<u64, WeightedCapacityError> {
    let gram = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
    let floats = gram
        .checked_mul(2)
        .and_then(|cells| cells.checked_add(ncols.checked_mul(2)?))
        .and_then(|cells| cells.checked_add(7))
        .and_then(|cells| cells.checked_add(11))
        .ok_or(WeightedCapacityError::Overflow)?;
    f64_bytes(floats)?.checked_add(43).ok_or(WeightedCapacityError::Overflow)
}

fn spill_frame_bytes(ncols: usize) -> Result<u64, WeightedCapacityError> {
    spill_file::frame_len(spill_payload_bytes(ncols)?).ok_or(WeightedCapacityError::Overflow)
}

fn payload_bytes(ncols: usize) -> Result<u64, WeightedCapacityError> {
    let gram = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
    let floats = gram
        .checked_mul(2)
        .and_then(|cells| cells.checked_add(ncols.checked_mul(2)?))
        .and_then(|cells| cells.checked_add(6))
        .ok_or(WeightedCapacityError::Overflow)?;
    let float_bytes = f64_bytes(floats)?;
    let header = size_u64::<FitDiagnostics>()?
        + size_u64::<CapacityControl>()?
        + size_u64::<OutputRetention>()?
        + 4 * size_u64::<u64>()?;
    float_bytes.checked_add(header).ok_or(WeightedCapacityError::Overflow)
}

fn content_identity_bytes(ncols: usize) -> Result<u64, WeightedCapacityError> {
    batch_byte_len(1, ncols)?.checked_add(32).ok_or(WeightedCapacityError::Overflow)
}

fn working_bytes(
    batch_rows: usize,
    ncols: usize,
    source_decoder_bytes: u64,
) -> Result<u64, WeightedCapacityError> {
    let window = batch_byte_len(batch_rows, ncols)?;
    let solver = solver_bytes(ncols)?;
    window
        .checked_add(source_decoder_bytes)
        .and_then(|sum| sum.checked_add(solver))
        .ok_or(WeightedCapacityError::Overflow)
}

/// Reservation for every buffer the fold, the solve, and the diagnostic pass
/// allocate beyond the output payload. Twelve `p×p` buffers, twenty-four
/// length-`p` buffers, and two hundred fifty-six floats of fixed slack. The sum is taken
/// as if the buffers were live together, which covers the Cholesky factor,
/// the `fit_wls` copies and its backend factorization, the inversion
/// scratch, the meat, and the sandwich intermediate. This is a reservation, not
/// a measured allocation.
fn solver_bytes(ncols: usize) -> Result<u64, WeightedCapacityError> {
    let gram = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
    let floats = gram
        .checked_mul(12)
        .and_then(|cells| cells.checked_add(ncols.checked_mul(24)?))
        .and_then(|cells| cells.checked_add(256))
        .ok_or(WeightedCapacityError::Overflow)?;
    f64_bytes(floats)
}

fn batch_byte_len(batch_rows: usize, ncols: usize) -> Result<u64, WeightedCapacityError> {
    if batch_rows == 0 {
        return Err(WeightedCapacityError::BatchBound);
    }
    let stride = ncols.checked_add(2).ok_or(WeightedCapacityError::Overflow)?;
    let cells = batch_rows.checked_mul(stride).ok_or(WeightedCapacityError::Overflow)?;
    f64_bytes(cells)
}

fn f64_bytes(cells: usize) -> Result<u64, WeightedCapacityError> {
    u64::try_from(cells)
        .map_err(|_| WeightedCapacityError::Overflow)?
        .checked_mul(8)
        .ok_or(WeightedCapacityError::Overflow)
}

fn size_u64<T>() -> Result<u64, WeightedCapacityError> {
    u64::try_from(std::mem::size_of::<T>()).map_err(|_| WeightedCapacityError::Overflow)
}

fn allocation_failure(
    what: &'static str,
    bytes: u64,
    source: TryReserveError,
) -> WeightedCapacityError {
    WeightedCapacityError::Allocation { what, bytes, source }
}

fn io_failure(context: &'static str) -> impl FnOnce(io::Error) -> WeightedCapacityError {
    move |source| WeightedCapacityError::Io { context, source }
}

fn zeros(len: usize, what: &'static str) -> Result<Vec<f64>, WeightedCapacityError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|source| {
        let bytes = u64::try_from(len).ok().and_then(|n| n.checked_mul(8)).unwrap_or(u64::MAX);
        allocation_failure(what, bytes, source)
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn validate_row(
    index: u64,
    row: &[f64],
    ncols: usize,
    y: f64,
    weight: f64,
) -> Result<(), WeightedCapacityError> {
    if row.len() != ncols {
        return Err(WeightedCapacityError::Shape(DesignShape::Row));
    }
    if !weight.is_finite() {
        return Err(WeightedCapacityError::Weight { row: index, reason: WeightFault::NotFinite });
    }
    if weight.partial_cmp(&0.0) == Some(Ordering::Less) {
        return Err(WeightedCapacityError::Weight { row: index, reason: WeightFault::Negative });
    }
    if !y.is_finite() || row.iter().any(|value| !value.is_finite()) {
        return Err(WeightedCapacityError::NonFinite { row: index });
    }
    Ok(())
}

#[must_use]
fn weight_is_positive(weight: f64) -> bool {
    weight.partial_cmp(&0.0) == Some(Ordering::Greater)
}

fn require_batch_request(max_rows: usize, bound: usize) -> Result<(), WeightedCapacityError> {
    if max_rows == 0 || max_rows > bound {
        Err(WeightedCapacityError::ReadPastBatch { requested: max_rows, bound })
    } else {
        Ok(())
    }
}

fn require_window(
    max_rows: usize,
    ncols: usize,
    x_rowmajor: &[f64],
    y: &[f64],
    weights: &[f64],
) -> Result<(), WeightedCapacityError> {
    let cells = max_rows.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
    if x_rowmajor.len() < cells || y.len() < max_rows || weights.len() < max_rows {
        Err(WeightedCapacityError::Shape(DesignShape::BatchWindow))
    } else {
        Ok(())
    }
}

fn parse_records(
    bytes: &[u8],
    ncols: usize,
    take: usize,
    x_rowmajor: &mut [f64],
    y: &mut [f64],
    weights: &mut [f64],
) -> Result<(), WeightedCapacityError> {
    let record_bytes =
        usize::try_from(batch_byte_len(1, ncols)?).map_err(|_| WeightedCapacityError::Overflow)?;
    let needed = record_bytes.checked_mul(take).ok_or(WeightedCapacityError::Overflow)?;
    if bytes.len() < needed {
        return Err(short_record());
    }
    let records = bytes
        .chunks_exact(record_bytes)
        .zip(x_rowmajor.chunks_exact_mut(ncols))
        .zip(y.iter_mut())
        .zip(weights.iter_mut())
        .take(take);
    for (((record, x_row), y_slot), weight_slot) in records {
        let mut cells = record.chunks_exact(8).map(decode_f64);
        for slot in x_row {
            *slot = cells.next().ok_or_else(short_record)??;
        }
        *y_slot = cells.next().ok_or_else(short_record)??;
        *weight_slot = cells.next().ok_or_else(short_record)??;
    }
    Ok(())
}

fn short_record() -> WeightedCapacityError {
    WeightedCapacityError::RowFile(RowFileFault::ShortRecord)
}

fn decode_f64(chunk: &[u8]) -> Result<f64, WeightedCapacityError> {
    let bits: [u8; 8] = chunk.try_into().map_err(|_| short_record())?;
    Ok(f64::from_le_bytes(bits))
}

fn read_u32(bytes: &[u8]) -> Result<u32, WeightedCapacityError> {
    let chunk: [u8; 4] =
        bytes.try_into().map_err(|_| WeightedCapacityError::RowFile(RowFileFault::ShortHeader))?;
    Ok(u32::from_le_bytes(chunk))
}

fn read_u64(bytes: &[u8]) -> Result<u64, WeightedCapacityError> {
    let chunk: [u8; 8] =
        bytes.try_into().map_err(|_| WeightedCapacityError::RowFile(RowFileFault::ShortHeader))?;
    Ok(u64::from_le_bytes(chunk))
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;
    use std::path::PathBuf;

    use antecedent_core::device_reservation::DeviceReservationPool;
    use antecedent_core::execution::CancellationToken;
    use antecedent_core::execution_facts::OutputRetention;
    use antecedent_stats::{
        FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, SandwichKind,
        coefficient_covariance, fit_glm_weighted, fit_wls,
    };

    use super::{
        CapacityControl, CapacityOptions, FileWeightedRows, ResidentWeightedRows, RowFileFault,
        SpillFault, SpillOptions, SpilledCapacity, WeightedCapacityError, WeightedCapacityOutput,
        WeightedCapacityResult, WeightedRowSource, encode_weighted_row_file, read_spilled_summary,
        spill_with, zeros,
    };

    /// The resident summary of a run. A disk-backed run is a test failure here.
    fn run_weighted_capacity(
        control: CapacityControl,
        source: &mut dyn WeightedRowSource,
        options: &CapacityOptions,
    ) -> Result<WeightedCapacityResult, WeightedCapacityError> {
        match super::run_weighted_capacity(control, source, options)? {
            WeightedCapacityOutput::Resident(result) => Ok(result),
            WeightedCapacityOutput::Spilled(_) => panic!("expected a resident summary"),
        }
    }

    fn run_spilled(
        control: CapacityControl,
        source: &mut dyn WeightedRowSource,
        options: &CapacityOptions,
    ) -> Result<SpilledCapacity, WeightedCapacityError> {
        match super::run_weighted_capacity(control, source, options)? {
            WeightedCapacityOutput::Spilled(spilled) => Ok(spilled),
            WeightedCapacityOutput::Resident(_) => panic!("expected a spill"),
        }
    }

    /// Per-thread live and peak byte counts, so concurrent tests do not see
    /// each other. Every call forwards to the system allocator unchanged.
    #[allow(
        unsafe_code,
        reason = "test-only counting allocator that forwards to the system allocator"
    )]
    mod counting {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        struct Counting;

        thread_local! {
            // const-initialized: no lazy allocation, so counting cannot recurse.
            static LIVE: Cell<isize> = const { Cell::new(0) };
            static PEAK: Cell<isize> = const { Cell::new(0) };
        }

        fn grow(bytes: usize) {
            let delta = isize::try_from(bytes).unwrap_or(isize::MAX);
            let _ = LIVE.try_with(|live| {
                let now = live.get().saturating_add(delta);
                live.set(now);
                let _ = PEAK.try_with(|peak| {
                    if now > peak.get() {
                        peak.set(now);
                    }
                });
            });
        }

        fn shrink(bytes: usize) {
            let delta = isize::try_from(bytes).unwrap_or(isize::MAX);
            let _ = LIVE.try_with(|live| live.set(live.get().saturating_sub(delta)));
        }

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                // SAFETY: forwarding to the system allocator with the same layout.
                let ptr = unsafe { System.alloc(layout) };
                if !ptr.is_null() {
                    grow(layout.size());
                }
                ptr
            }

            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                // SAFETY: forwarding to the system allocator with the same layout.
                let ptr = unsafe { System.alloc_zeroed(layout) };
                if !ptr.is_null() {
                    grow(layout.size());
                }
                ptr
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                // SAFETY: `ptr` came from this allocator with this layout.
                let moved = unsafe { System.realloc(ptr, layout, new_size) };
                if !moved.is_null() {
                    shrink(layout.size());
                    grow(new_size);
                }
                moved
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: `ptr` came from this allocator with this layout.
                unsafe { System.dealloc(ptr, layout) };
                shrink(layout.size());
            }
        }

        #[global_allocator]
        static ALLOCATOR: Counting = Counting;

        /// Run `work` and report the most bytes it held above where it started.
        pub fn peak_above_start<T>(work: impl FnOnce() -> T) -> (T, u64) {
            let start = LIVE.with(Cell::get);
            PEAK.with(|peak| peak.set(start));
            let out = work();
            let peak = PEAK.with(Cell::get);
            (out, u64::try_from(peak.saturating_sub(start)).unwrap_or(0))
        }
    }

    fn design() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        let y = vec![1.0, 2.2, 2.8, 4.4, 100.0];
        let weights = vec![1.0, 2.0, 3.0, 4.0, 0.0];
        (x, y, weights)
    }

    /// Intercept, a centered covariate, a covariate far from zero, and a
    /// third covariate. Every fifth row from the fourth, and the last row,
    /// carry zero weight.
    fn wide_design(rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let (mut x, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..rows {
            let a = next() * 2.0 - 1.0;
            let b = 1000.0 + 4.0 * next();
            let c = next();
            let noise = next() - 0.5;
            let draw = next();
            x.extend_from_slice(&[1.0, a, b, c]);
            y.push(2.0 + 3.0 * a - 0.5 * (b - 1000.0) + 1.5 * c + noise);
            w.push(if row % 5 == 3 || row + 1 == rows { 0.0 } else { 0.25 + draw });
        }
        (x, y, w)
    }

    fn options(retention: OutputRetention) -> CapacityOptions {
        CapacityOptions::new(1, u64::MAX, retention)
    }

    fn bits_eq(left: &[f64], right: &[f64]) -> bool {
        left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
    }

    fn assert_declared(result: &WeightedCapacityResult) {
        assert_eq!(result.source_passes, 2);
        assert_eq!(result.batches, 5);
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.robust_covariance.len(), 4);
        assert_eq!(result.cross_products.xtwx.len(), 4);
        assert_eq!(result.cross_products.xtwy.len(), 2);
        assert_eq!(result.residual.positive_weight_rows, 4);
        assert!((result.cross_products.n_eff - 10.0).abs() < 1.0e-9);
        assert!(result.residual.weighted_rss > 0.0);
        assert!(result.residual.max_abs_residual > 0.0);
        assert!(result.leverage.max > 0.0);
        assert!((result.leverage.sum - 2.0).abs() < 1.0e-4);
        assert!(result.charges.content_identity_bytes >= 64);
        assert!(result.charges.decoder_bytes >= 32);
        let parts = result.charges.input_ownership_bytes
            + result.charges.content_identity_bytes
            + result.charges.decoder_bytes
            + result.charges.spill_bytes
            + result.charges.output_bytes;
        assert_eq!(result.charges.total_bytes, parts);
    }

    fn same_science(left: &WeightedCapacityResult, right: &WeightedCapacityResult) {
        same_numbers(left, right);
        assert_eq!(left.retention, right.retention);
    }

    /// Every number and count of the summary, compared bit for bit.
    fn same_numbers(left: &WeightedCapacityResult, right: &WeightedCapacityResult) {
        assert!(bits_eq(&left.coefficients, &right.coefficients));
        assert!(bits_eq(&left.robust_covariance, &right.robust_covariance));
        assert!(bits_eq(&left.cross_products.xtwx, &right.cross_products.xtwx));
        assert!(bits_eq(&left.cross_products.xtwy, &right.cross_products.xtwy));
        assert_eq!(left.cross_products.n_eff.to_bits(), right.cross_products.n_eff.to_bits());
        assert_eq!(left.cross_products.ytwy.to_bits(), right.cross_products.ytwy.to_bits());
        assert_eq!(left.leverage.max.to_bits(), right.leverage.max.to_bits());
        assert_eq!(left.leverage.sum.to_bits(), right.leverage.sum.to_bits());
        assert_eq!(left.residual.weighted_rss.to_bits(), right.residual.weighted_rss.to_bits());
        assert_eq!(left.leverage.saturated_rows, right.leverage.saturated_rows,);
        assert_eq!(
            left.residual.max_abs_residual.to_bits(),
            right.residual.max_abs_residual.to_bits()
        );
        assert_eq!(left.fit, right.fit);
        assert_eq!(left.content_identity, right.content_identity);
        assert_eq!(left.source_passes, right.source_passes);
        assert_eq!(left.residual.positive_weight_rows, right.residual.positive_weight_rows);
    }

    struct RemoveFile(PathBuf);

    impl Drop for RemoveFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn write_rows(name: &str, bytes: &[u8]) -> RemoveFile {
        let pid = std::process::id();
        let path =
            std::env::temp_dir().join(format!("weighted-gaussian-capacity-{pid}-{name}.wgol"));
        std::fs::write(&path, bytes).expect("write rows");
        RemoveFile(path)
    }

    struct Probe<'a> {
        inner: ResidentWeightedRows<'a>,
        max_request: usize,
        reads: u64,
    }

    impl WeightedRowSource for Probe<'_> {
        fn ncols(&self) -> usize {
            self.inner.ncols()
        }

        fn input_ownership_bytes(&self) -> u64 {
            self.inner.input_ownership_bytes()
        }

        fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError> {
            self.inner.decoder_reservation(batch_rows)
        }

        fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError> {
            self.inner.prepare_decoder(batch_rows)
        }

        fn allocated_decoder_bytes(&self) -> u64 {
            self.inner.allocated_decoder_bytes()
        }

        fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
            self.inner.rewind()
        }

        fn read_batch(
            &mut self,
            max_rows: usize,
            x_rowmajor: &mut [f64],
            y: &mut [f64],
            weights: &mut [f64],
        ) -> Result<usize, WeightedCapacityError> {
            self.max_request = self.max_request.max(max_rows);
            self.reads = self.reads.saturating_add(1);
            self.inner.read_batch(max_rows, x_rowmajor, y, weights)
        }
    }

    /// Serves the true rows on the first pass and a nudged first response on
    /// every pass after it.
    struct ChangesAfterFirstPass<'a> {
        inner: ResidentWeightedRows<'a>,
        rewinds: u32,
    }

    impl WeightedRowSource for ChangesAfterFirstPass<'_> {
        fn ncols(&self) -> usize {
            self.inner.ncols()
        }

        fn input_ownership_bytes(&self) -> u64 {
            self.inner.input_ownership_bytes()
        }

        fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError> {
            self.inner.decoder_reservation(batch_rows)
        }

        fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError> {
            self.inner.prepare_decoder(batch_rows)
        }

        fn allocated_decoder_bytes(&self) -> u64 {
            self.inner.allocated_decoder_bytes()
        }

        fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
            self.rewinds += 1;
            self.inner.rewind()
        }

        fn read_batch(
            &mut self,
            max_rows: usize,
            x_rowmajor: &mut [f64],
            y: &mut [f64],
            weights: &mut [f64],
        ) -> Result<usize, WeightedCapacityError> {
            let took = self.inner.read_batch(max_rows, x_rowmajor, y, weights)?;
            if took > 0 && self.rewinds >= 2 {
                y[0] += 1.0;
            }
            Ok(took)
        }
    }

    #[test]
    fn resident_and_file_rows_match_under_one_retention() {
        let (x, y, weights) = design();
        let options = options(OutputRetention::SummaryOnly);
        let mut resident = ResidentWeightedRows::try_new(&x, &y, &weights).expect("resident");
        let resident_fit = run_weighted_capacity(CapacityControl::Glm, &mut resident, &options)
            .expect("resident fit");
        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");
        let file_guard = write_rows("match", &bytes);
        let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
        assert_eq!(file.allocated_decoder_bytes(), 0);
        let file_fit =
            run_weighted_capacity(CapacityControl::Glm, &mut file, &options).expect("file fit");
        assert_declared(&resident_fit);
        assert_declared(&file_fit);
        same_science(&resident_fit, &file_fit);
        assert_eq!(resident_fit.retention, OutputRetention::SummaryOnly);
        assert_eq!(file.allocated_decoder_bytes(), 32);
        assert!(file.allocated_decoder_bytes() < 160);
        assert_eq!(resident_fit.charges.input_ownership_bytes, 160);
        assert_eq!(file_fit.charges.input_ownership_bytes, 0);
        assert_eq!(file_fit.charges.decoder_bytes, resident_fit.charges.decoder_bytes + 32);
    }

    #[test]
    fn glm_and_gaussian_bayes_return_the_same_summary() {
        let (x, y, weights) = design();
        let options = options(OutputRetention::SummaryOnly);
        let mut glm_source = ResidentWeightedRows::try_new(&x, &y, &weights).expect("glm");
        let mut bayes_source = ResidentWeightedRows::try_new(&x, &y, &weights).expect("bayes");
        let glm =
            run_weighted_capacity(CapacityControl::Glm, &mut glm_source, &options).expect("glm");
        let bayes =
            run_weighted_capacity(CapacityControl::GaussianBayes, &mut bayes_source, &options)
                .expect("bayes");
        assert_eq!(glm.control, CapacityControl::Glm);
        assert_eq!(bayes.control, CapacityControl::GaussianBayes);
        assert_declared(&glm);
        assert_declared(&bayes);
        same_science(&glm, &bayes);
        assert_eq!(glm.charges.total_bytes, bayes.charges.total_bytes);
    }

    #[test]
    fn a_dropped_diagnostic_or_an_ignored_weight_fails() {
        let (x, y, weights) = design();
        let unit = vec![1.0; y.len()];
        let options = options(OutputRetention::SummaryOnly);
        for control in [CapacityControl::Glm, CapacityControl::GaussianBayes] {
            let mut weighted = ResidentWeightedRows::try_new(&x, &y, &weights).expect("weighted");
            let mut unweighted = ResidentWeightedRows::try_new(&x, &y, &unit).expect("unit");
            let fit =
                run_weighted_capacity(control, &mut weighted, &options).expect("weighted fit");
            let flat = run_weighted_capacity(control, &mut unweighted, &options).expect("unit fit");
            assert_declared(&fit);
            assert!((fit.coefficients[0] - flat.coefficients[0]).abs() > 1.0e-3);
            assert_matches_resident_oracle(&fit, &x, &y, &weights, 2);
        }
    }

    struct Scaled {
        x: Vec<f64>,
        y: Vec<f64>,
        ones: Vec<f64>,
        nrows: usize,
    }

    fn scaled_design(x: &[f64], y: &[f64], weights: &[f64], ncols: usize) -> Scaled {
        let nrows = y.len();
        let mut design = vec![0.0; nrows * ncols];
        let mut response = vec![0.0; nrows];
        for row in 0..nrows {
            let scale = weights[row].sqrt();
            response[row] = y[row] * scale;
            for col in 0..ncols {
                design[col * nrows + row] = x[row * ncols + col] * scale;
            }
        }
        Scaled { x: design, y: response, ones: vec![1.0; nrows], nrows }
    }

    fn scaled_residuals(scaled: &Scaled, ncols: usize, beta: &[f64]) -> Vec<f64> {
        (0..scaled.nrows)
            .map(|row| {
                let predicted: f64 =
                    (0..ncols).map(|col| beta[col] * scaled.x[col * scaled.nrows + row]).sum();
                scaled.y[row] - predicted
            })
            .collect()
    }

    fn assert_close(got: &[f64], want: &[f64], relative: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= relative * scale, "{what}: {g} vs {w}");
        }
    }

    /// The resident path as the oracle: `fit_wls` on the row-scaled design,
    /// the `HC0` covariance of that design, and the naive weighted
    /// cross-products.
    fn assert_matches_resident_oracle(
        fit: &WeightedCapacityResult,
        x: &[f64],
        y: &[f64],
        weights: &[f64],
        ncols: usize,
    ) {
        let scaled = scaled_design(x, y, weights, ncols);
        let mut workspace = LeastSquaresWorkspace::default();
        let reference = fit_wls(
            &scaled.x,
            scaled.nrows,
            ncols,
            &scaled.y,
            &scaled.ones,
            &FaerBackend,
            &mut workspace,
        )
        .expect("fit_wls");
        assert_close(&fit.coefficients, &reference.coefficients, 1.0e-6, "coefficients");
        let residuals = scaled_residuals(&scaled, ncols, &fit.coefficients);
        let robust =
            coefficient_covariance(&scaled.x, scaled.nrows, ncols, &residuals, SandwichKind::Hc0)
                .expect("hc0");
        assert_close(&fit.robust_covariance, &robust, 1.0e-5, "robust covariance");
        let mut xtwx = vec![0.0; ncols * ncols];
        let mut xtwy = vec![0.0; ncols];
        let mut ytwy = 0.0;
        let mut n_eff = 0.0;
        for ((row, &response), &weight) in x.chunks_exact(ncols).zip(y).zip(weights) {
            n_eff += weight;
            ytwy += weight * response * response;
            for (i, &xi) in row.iter().enumerate() {
                xtwy[i] += weight * xi * response;
                for (j, &xj) in row.iter().enumerate() {
                    xtwx[i * ncols + j] += weight * xi * xj;
                }
            }
        }
        assert_close(&fit.cross_products.xtwx, &xtwx, 1.0e-9, "X'WX");
        assert_close(&fit.cross_products.xtwy, &xtwy, 1.0e-9, "X'Wy");
        assert_close(&[fit.cross_products.ytwy], &[ytwy], 1.0e-9, "y'Wy");
        assert_close(&[fit.cross_products.n_eff], &[n_eff], 1.0e-12, "sum of weights");
    }

    #[test]
    fn every_batch_size_matches_the_resident_oracle_in_both_controls_and_sources() {
        let rows = 23;
        let (x, y, weights) = wide_design(rows);
        let positive = u64::try_from(weights.iter().filter(|w| **w > 0.0).count()).expect("count");
        assert!(positive < u64::try_from(rows).expect("rows"));
        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");
        let file_guard = write_rows("equivalence", &bytes);
        let mut first: Option<WeightedCapacityResult> = None;
        for batch_rows in [1, 2, 5, 7, 22, 23, 64] {
            let options = CapacityOptions::new(batch_rows, u64::MAX, OutputRetention::SummaryOnly);
            let expected_batches = u64::try_from(rows.div_ceil(batch_rows)).expect("batches");
            for control in [CapacityControl::Glm, CapacityControl::GaussianBayes] {
                let mut resident =
                    ResidentWeightedRows::try_new(&x, &y, &weights).expect("resident");
                let from_memory =
                    run_weighted_capacity(control, &mut resident, &options).expect("resident fit");
                let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
                let from_file =
                    run_weighted_capacity(control, &mut file, &options).expect("file fit");
                for fit in [&from_memory, &from_file] {
                    assert_eq!(fit.control, control);
                    assert_eq!(fit.retention, OutputRetention::SummaryOnly);
                    assert_eq!(fit.batches, expected_batches);
                    assert_eq!(fit.source_passes, 2);
                    assert_eq!(fit.residual.positive_weight_rows, positive);
                    assert_matches_resident_oracle(fit, &x, &y, &weights, 4);
                }
                same_science(&from_memory, &from_file);
                assert_eq!(from_memory.charges.output_bytes, from_file.charges.output_bytes);
                assert_eq!(from_file.charges.input_ownership_bytes, 0);
                match &first {
                    Some(reference) => same_science(reference, &from_memory),
                    None => first = Some(from_memory),
                }
            }
        }
    }

    /// A unique, empty directory that is removed when it goes out of scope.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("weighted-gaussian-spill-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }

        fn entries(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .expect("read test directory")
                .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn disk_options(batch_rows: usize, dir: &TempDir, file: &str) -> CapacityOptions {
        CapacityOptions {
            spill: Some(SpillOptions { path: dir.join(file), pool: None }),
            ..CapacityOptions::new(batch_rows, u64::MAX, OutputRetention::DiskBacked)
        }
    }

    fn spill_of(
        x: &[f64],
        y: &[f64],
        weights: &[f64],
        options: &CapacityOptions,
    ) -> Result<SpilledCapacity, WeightedCapacityError> {
        let mut source = ResidentWeightedRows::try_new(x, y, weights).expect("resident");
        run_spilled(CapacityControl::Glm, &mut source, options)
    }

    /// The `glm` Gaussian fitter as the oracle: coefficients and weighted deviance.
    fn assert_matches_glm_oracle(
        fit: &WeightedCapacityResult,
        x: &[f64],
        y: &[f64],
        weights: &[f64],
        ncols: usize,
    ) {
        let nrows = y.len();
        let mut colmajor = vec![0.0; nrows * ncols];
        for (row, values) in x.chunks_exact(ncols).enumerate() {
            for (col, &value) in values.iter().enumerate() {
                colmajor[col * nrows + row] = value;
            }
        }
        let design = GlmDesignRef { x_colmajor: &colmajor, nrows, ncols, y };
        let mut workspace = LeastSquaresWorkspace::default();
        let reference = fit_glm_weighted(
            GlmFamily::GaussianIdentity,
            design,
            weights,
            &FaerBackend,
            &mut workspace,
            &GlmOptions::default(),
        )
        .expect("glm oracle");
        assert_close(&fit.coefficients, &reference.coefficients, 1.0e-6, "glm coefficients");
        assert_close(
            &[fit.residual.weighted_rss],
            &[reference.deviance],
            1.0e-6,
            "weighted deviance",
        );
    }

    #[test]
    fn disk_backed_summary_is_bit_identical_to_the_resident_summary() {
        let rows = 23;
        let (x, y, weights) = wide_design(rows);
        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");
        let file_guard = write_rows("spill-identical", &bytes);
        let dir = TempDir::new("identical");
        for batch_rows in [1, 7, 23] {
            for control in [CapacityControl::Glm, CapacityControl::GaussianBayes] {
                let summary_options =
                    CapacityOptions::new(batch_rows, u64::MAX, OutputRetention::SummaryOnly);
                let mut resident =
                    ResidentWeightedRows::try_new(&x, &y, &weights).expect("resident");
                let kept = run_weighted_capacity(control, &mut resident, &summary_options)
                    .expect("resident fit");
                let options = disk_options(batch_rows, &dir, "summary.spill");
                let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
                let spilled = run_spilled(control, &mut file, &options).expect("spill");
                assert_eq!(spilled.control, control);
                assert_eq!(spilled.retention, OutputRetention::DiskBacked);
                assert_eq!(spilled.batches, kept.batches);
                assert_eq!(spilled.source_passes, 2);
                assert_eq!(spilled.content_identity, kept.content_identity);
                let charges = spilled.charges;
                assert!(charges.spill_bytes > 0);
                assert_eq!(charges.spill_bytes, spilled.handle.frame_bytes());
                let on_disk = std::fs::metadata(spilled.handle.path()).expect("spill file").len();
                assert_eq!(on_disk, charges.spill_bytes);
                assert_eq!(
                    charges.total_bytes,
                    charges.input_ownership_bytes
                        + charges.content_identity_bytes
                        + charges.decoder_bytes
                        + charges.spill_bytes
                        + charges.output_bytes
                );
                assert!(charges.decoder_bytes > kept.charges.decoder_bytes);
                assert_eq!(dir.entries(), vec!["summary.spill".to_owned()]);

                let back = read_spilled_summary(&spilled.handle).expect("read back");
                same_numbers(&kept, &back);
                assert_eq!(back.retention, OutputRetention::DiskBacked);
                assert_eq!(back.control, control);
                assert_eq!(back.batches, kept.batches);
                assert_eq!(back.charges, charges);
                assert_matches_glm_oracle(&back, &x, &y, &weights, 4);
                spilled.handle.remove().expect("remove");
                assert!(dir.entries().is_empty());
            }
        }
    }

    #[test]
    fn disk_backed_summary_matches_the_glm_oracle_for_the_small_design() {
        let (x, y, weights) = design();
        let dir = TempDir::new("oracle");
        let spilled =
            spill_of(&x, &y, &weights, &disk_options(1, &dir, "small.spill")).expect("spill");
        let back = read_spilled_summary(&spilled.handle).expect("read back");
        assert_declared(&back);
        assert_matches_glm_oracle(&back, &x, &y, &weights, 2);
        assert_matches_resident_oracle(&back, &x, &y, &weights, 2);
    }

    #[test]
    fn spill_bytes_count_against_the_ceiling() {
        let (x, y, weights) = design();
        let dir = TempDir::new("ceiling");
        let probe = spill_of(&x, &y, &weights, &disk_options(1, &dir, "run.spill"))
            .expect("unbounded spill")
            .charges;
        assert!(probe.spill_bytes > 0);
        let run_at = |ceiling_bytes: u64| {
            let options = CapacityOptions { ceiling_bytes, ..disk_options(1, &dir, "run.spill") };
            spill_of(&x, &y, &weights, &options)
        };
        let within = run_at(probe.total_bytes).expect("exactly the reservation fits");
        assert_eq!(within.charges, probe);
        within.handle.remove().expect("remove");
        for lowered in [probe.spill_bytes, 1] {
            match run_at(probe.total_bytes - lowered) {
                Err(WeightedCapacityError::OverCeiling { over_by, total, charges, .. }) => {
                    assert_eq!(over_by, lowered);
                    assert_eq!(total, probe.total_bytes);
                    assert_eq!(charges, probe);
                }
                other => panic!("expected a ceiling refusal, got {other:?}"),
            }
        }
        assert_eq!(dir.entries(), Vec::<String>::new());
    }

    #[test]
    fn disk_backed_retention_needs_a_spill_path_for_both_controls_and_sources() {
        let (x, y, weights) = design();
        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");
        let file_guard = write_rows("disk-backed", &bytes);
        let options = options(OutputRetention::DiskBacked);
        for control in [CapacityControl::Glm, CapacityControl::GaussianBayes] {
            let mut resident = ResidentWeightedRows::try_new(&x, &y, &weights).expect("resident");
            assert!(matches!(
                super::run_weighted_capacity(control, &mut resident, &options),
                Err(WeightedCapacityError::SpillPathRequired)
            ));
            let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
            assert!(matches!(
                super::run_weighted_capacity(control, &mut file, &options),
                Err(WeightedCapacityError::SpillPathRequired)
            ));
            assert_eq!(file.allocated_decoder_bytes(), 0);
        }
    }

    #[test]
    fn the_spill_pool_is_reserved_before_the_write_and_returned_afterwards() {
        let (x, y, weights) = design();
        let dir = TempDir::new("pool");
        let frame = spill_of(&x, &y, &weights, &disk_options(1, &dir, "probe.spill"))
            .expect("probe")
            .charges
            .spill_bytes;
        std::fs::remove_file(dir.join("probe.spill")).expect("clear probe");
        let with_pool = |pool: &DeviceReservationPool, file: &str| {
            let mut options = disk_options(1, &dir, file);
            options.spill = Some(SpillOptions { path: dir.join(file), pool: Some(pool.clone()) });
            spill_of(&x, &y, &weights, &options)
        };

        let small = DeviceReservationPool::fixed(0, frame - 1);
        match with_pool(&small, "refused.spill") {
            Err(WeightedCapacityError::SpillPool(_)) => {}
            other => panic!("expected a pool refusal, got {other:?}"),
        }
        assert_eq!(small.available().spill_bytes(), frame - 1);
        assert!(dir.entries().is_empty());

        let pool = DeviceReservationPool::fixed(0, frame + 10);
        let spilled = with_pool(&pool, "granted.spill").expect("granted");
        assert!(spilled.handle.holds_reservation());
        assert_eq!(pool.available().spill_bytes(), 10);
        assert!(with_pool(&pool, "second.spill").is_err());
        assert_eq!(dir.entries(), vec!["granted.spill".to_owned()]);
        spilled.handle.remove().expect("remove");
        assert_eq!(pool.available().spill_bytes(), frame + 10);
        assert!(dir.entries().is_empty());

        let kept = with_pool(&pool, "dropped.spill").expect("granted again");
        let path = kept.handle.path().to_path_buf();
        drop(kept);
        assert_eq!(pool.available().spill_bytes(), frame + 10);
        assert!(path.exists());
    }

    #[test]
    fn cancellation_leaves_no_file_and_returns_the_reservation() {
        let (x, y, weights) = design();
        let dir = TempDir::new("cancel");
        let pool = DeviceReservationPool::fixed(0, 1 << 20);
        // Five batches per pass: ten scan checks, then one before the pool is
        // taken and one before the rename.
        let run_with = |checks: Option<usize>| {
            let mut options = disk_options(1, &dir, "cancelled.spill");
            options.spill =
                Some(SpillOptions { path: dir.join("cancelled.spill"), pool: Some(pool.clone()) });
            options.cancellation = checks.map(CancellationToken::cancel_after_checks);
            spill_of(&x, &y, &weights, &options)
        };
        for checks in [0, 3, 7, 10, 11] {
            match run_with(Some(checks)) {
                Err(WeightedCapacityError::Cancelled) => {}
                other => panic!("checks {checks}: expected a cancellation, got {other:?}"),
            }
            assert!(dir.entries().is_empty(), "checks {checks}: files left behind");
            assert_eq!(pool.available().spill_bytes(), 1 << 20, "checks {checks}");
        }
        let finished = run_with(Some(12)).expect("no check trips");
        assert_eq!(dir.entries(), vec!["cancelled.spill".to_owned()]);
        finished.handle.remove().expect("remove");
        assert!(dir.entries().is_empty());
    }

    #[test]
    fn a_failed_write_or_rename_leaves_no_temporary_or_partial_file() {
        let (x, y, weights) = design();
        let dir = TempDir::new("write-error");
        let seed = spill_of(&x, &y, &weights, &disk_options(1, &dir, "seed.spill")).expect("seed");
        let result = read_spilled_summary(&seed.handle).expect("read seed");
        let kept = dir.entries();

        let target = SpillOptions { path: dir.join("failed.spill"), pool: None };
        let failing = |file: &mut std::fs::File, frame: &[u8]| {
            use std::io::Write as _;
            file.write_all(&frame[..frame.len() / 2])?;
            Err(std::io::Error::other("injected write failure"))
        };
        match spill_with(result.clone(), None, &target, failing) {
            Err(WeightedCapacityError::Io { context: "write the spill file", source }) => {
                assert_eq!(source.to_string(), "injected write failure");
            }
            other => panic!("expected the injected write failure, got {other:?}"),
        }
        assert_eq!(dir.entries(), kept);

        let blocked = SpillOptions { path: dir.join("blocked.spill"), pool: None };
        std::fs::create_dir(&blocked.path).expect("block the target");
        match spill_with(result, None, &blocked, super::spill_file::write_whole) {
            Err(WeightedCapacityError::Io {
                context: "move the spill file into place", ..
            }) => {}
            other => panic!("expected a rename failure, got {other:?}"),
        }
        let mut expected = kept;
        expected.push("blocked.spill".to_owned());
        expected.sort();
        assert_eq!(dir.entries(), expected);
    }

    #[test]
    fn a_changed_truncated_or_swapped_spill_file_is_refused_on_read() {
        let (x, y, weights) = design();
        let dir = TempDir::new("tamper");
        let spilled =
            spill_of(&x, &y, &weights, &disk_options(1, &dir, "a.spill")).expect("spill a");
        let path = spilled.handle.path().to_path_buf();
        let original = std::fs::read(&path).expect("read frame");
        let refused = |bytes: &[u8]| {
            std::fs::write(&path, bytes).expect("write tampered frame");
            read_spilled_summary(&spilled.handle).expect_err("tampered frame")
        };

        let mut flipped = original.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(matches!(
            refused(&flipped),
            WeightedCapacityError::Spill(SpillFault::DigestMismatch)
        ));
        let mut header_digest = original.clone();
        header_digest[20] ^= 1;
        assert!(matches!(
            refused(&header_digest),
            WeightedCapacityError::Spill(SpillFault::DigestMismatch)
        ));
        let mut magic = original.clone();
        magic[0] ^= 1;
        assert!(matches!(refused(&magic), WeightedCapacityError::Spill(SpillFault::BadMagic)));
        assert!(matches!(
            refused(&original[..original.len() - 1]),
            WeightedCapacityError::Spill(SpillFault::LengthMismatch { .. })
        ));
        let mut longer = original.clone();
        longer.push(0);
        assert!(matches!(
            refused(&longer),
            WeightedCapacityError::Spill(SpillFault::LengthMismatch { .. })
        ));

        // A well-formed frame from another run has the same length and a different digest.
        let other_y: Vec<f64> = y.iter().map(|value| value + 1.0).collect();
        let other =
            spill_of(&x, &other_y, &weights, &disk_options(1, &dir, "b.spill")).expect("spill b");
        assert_eq!(other.handle.frame_bytes(), spilled.handle.frame_bytes());
        let swapped = std::fs::read(other.handle.path()).expect("read b");
        assert!(matches!(
            refused(&swapped),
            WeightedCapacityError::Spill(SpillFault::DigestMismatch)
        ));

        std::fs::write(&path, &original).expect("restore");
        read_spilled_summary(&spilled.handle).expect("restored frame reads");
        std::fs::remove_file(&path).expect("remove");
        assert!(matches!(
            read_spilled_summary(&spilled.handle),
            Err(WeightedCapacityError::Io { .. })
        ));
    }

    #[test]
    fn a_disk_backed_run_fits_a_ceiling_smaller_than_the_row_data() {
        let rows = 3000;
        let (x, y, weights) = wide_design(rows);
        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");
        let row_file_bytes = u64::try_from(bytes.len()).expect("length");
        let file_guard = write_rows("bounded", &bytes);
        drop(bytes);
        let dir = TempDir::new("bounded");
        let run_at = |ceiling_bytes: u64| {
            let options =
                CapacityOptions { ceiling_bytes, ..disk_options(8, &dir, "bounded.spill") };
            let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
            run_spilled(CapacityControl::GaussianBayes, &mut file, &options)
        };
        let probe = run_at(u64::MAX).expect("probe").charges;
        assert!(
            probe.total_bytes < row_file_bytes,
            "ceiling {} is not below the {row_file_bytes} row bytes",
            probe.total_bytes
        );
        let options = CapacityOptions {
            ceiling_bytes: probe.total_bytes,
            ..disk_options(8, &dir, "bounded.spill")
        };
        let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
        let (outcome, peak) = counting::peak_above_start(|| {
            super::run_weighted_capacity(CapacityControl::GaussianBayes, &mut file, &options)
        });
        let spilled = outcome.expect("fits the ceiling").into_spilled().expect("spilled");
        assert!(
            peak <= spilled.charges.total_bytes,
            "peak {peak} exceeds the reservation {}",
            spilled.charges.total_bytes
        );
        let back = read_spilled_summary(&spilled.handle).expect("read back");
        assert_matches_glm_oracle(&back, &x, &y, &weights, 4);
        assert!(matches!(
            run_at(probe.total_bytes - 1),
            Err(WeightedCapacityError::OverCeiling { over_by: 1, .. })
        ));
    }

    #[test]
    fn read_past_the_batch_bound_is_refused() {
        let (x, y, weights) = design();
        let options = options(OutputRetention::SummaryOnly);
        let mut probe = Probe {
            inner: ResidentWeightedRows::try_new(&x, &y, &weights).expect("probe"),
            max_request: 0,
            reads: 0,
        };
        let fit =
            run_weighted_capacity(CapacityControl::Glm, &mut probe, &options).expect("probe fit");
        assert_declared(&fit);
        assert_eq!(probe.max_request, 1);
        assert!(probe.max_request < y.len());
        assert!(probe.reads >= 10);
        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");
        let file_guard = write_rows("bound", &bytes);
        let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
        let unprepared = file.read_batch(1, &mut [0.0; 2], &mut [0.0; 1], &mut [0.0; 1]);
        assert!(matches!(unprepared, Err(WeightedCapacityError::DecoderNotPrepared)));
        let file_fit = run_weighted_capacity(CapacityControl::GaussianBayes, &mut file, &options)
            .expect("file");
        assert_declared(&file_fit);
        assert_eq!(file.allocated_decoder_bytes(), 32);
        let mut wide_x = [0.0; 4];
        let mut wide_y = [0.0; 2];
        let mut wide_w = [0.0; 2];
        let past = file.read_batch(2, &mut wide_x, &mut wide_y, &mut wide_w);
        assert!(matches!(
            past,
            Err(WeightedCapacityError::ReadPastBatch { requested: 2, bound: 1 })
        ));
    }

    #[test]
    fn each_charge_alone_decides_the_ceiling() {
        let (x, y, weights) = design();
        let base = options(OutputRetention::SummaryOnly);
        let run_at = |ceiling_bytes: u64, control: CapacityControl| {
            let mut source = ResidentWeightedRows::try_new(&x, &y, &weights).expect("source");
            run_weighted_capacity(
                control,
                &mut source,
                &CapacityOptions { ceiling_bytes, ..base.clone() },
            )
        };
        let summary = run_at(u64::MAX, CapacityControl::Glm).expect("summary");
        let charges = summary.charges;
        assert_eq!(charges.spill_bytes, 0);
        assert_eq!(charges.input_ownership_bytes, 160);
        let lowered = [
            ("input ownership", charges.input_ownership_bytes),
            ("content identity", charges.content_identity_bytes),
            ("decoder", charges.decoder_bytes),
            ("output", charges.output_bytes),
        ];
        for (index, (name, amount)) in lowered.iter().enumerate() {
            assert!(*amount > 0, "{name} charge is zero");
            for (other, (_, other_amount)) in lowered.iter().enumerate() {
                assert!(
                    index == other || amount != other_amount,
                    "{name} charge equals another charge, so the refusal could not name it"
                );
            }
            let control =
                if index % 2 == 0 { CapacityControl::Glm } else { CapacityControl::GaussianBayes };
            match run_at(charges.total_bytes - amount, control) {
                Err(WeightedCapacityError::OverCeiling {
                    ceiling,
                    total,
                    over_by,
                    charges: refused,
                }) => {
                    assert_eq!(ceiling, charges.total_bytes - amount, "{name}");
                    assert_eq!(total, charges.total_bytes, "{name}");
                    assert_eq!(over_by, *amount, "{name}: the overage is the lowered charge");
                    assert_eq!(refused, charges, "{name}: the refusal carries the reservation");
                }
                other => panic!("{name}: expected a ceiling refusal, got {other:?}"),
            }
        }
        assert!(run_at(charges.total_bytes, CapacityControl::Glm).is_ok());
        match run_at(charges.total_bytes - 1, CapacityControl::Glm) {
            Err(WeightedCapacityError::OverCeiling { over_by: 1, .. }) => {}
            other => panic!("one byte under the total must be refused by one byte: {other:?}"),
        }
        let mut kept = ResidentWeightedRows::try_new(&x, &y, &weights).expect("kept");
        let full = CapacityOptions { retention: OutputRetention::FullReusable, ..base.clone() };
        assert!(matches!(
            run_weighted_capacity(CapacityControl::Glm, &mut kept, &full),
            Err(WeightedCapacityError::RetentionKeepsRows)
        ));
    }

    #[test]
    fn measured_peak_allocation_stays_within_the_reserved_total() {
        let (x, y, weights) = wide_design(200);
        let options = CapacityOptions::new(16, u64::MAX, OutputRetention::SummaryOnly);
        let mut resident = ResidentWeightedRows::try_new(&x, &y, &weights).expect("resident");
        let (resident_fit, resident_peak) = counting::peak_above_start(|| {
            run_weighted_capacity(CapacityControl::Glm, &mut resident, &options)
        });
        let resident_fit = resident_fit.expect("resident fit");
        // The borrowed slices were allocated before the measurement began.
        let resident_reserved =
            resident_fit.charges.total_bytes - resident_fit.charges.input_ownership_bytes;
        assert!(resident_peak > 0);
        assert!(
            resident_peak <= resident_reserved,
            "resident peak {resident_peak} exceeds the reservation {resident_reserved}"
        );

        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");
        let file_guard = write_rows("peak", &bytes);
        drop(bytes);
        let mut file = FileWeightedRows::open(&file_guard.0).expect("open");
        let (file_fit, file_peak) = counting::peak_above_start(|| {
            run_weighted_capacity(CapacityControl::GaussianBayes, &mut file, &options)
        });
        let file_fit = file_fit.expect("file fit");
        assert!(file_peak > 0);
        assert!(
            file_peak <= file_fit.charges.total_bytes,
            "file peak {file_peak} exceeds the reservation {}",
            file_fit.charges.total_bytes
        );
        same_science(&resident_fit, &file_fit);
    }

    #[test]
    fn a_source_that_changes_between_passes_is_refused() {
        let (x, y, weights) = wide_design(23);
        let mut source = ChangesAfterFirstPass {
            inner: ResidentWeightedRows::try_new(&x, &y, &weights).expect("resident"),
            rewinds: 0,
        };
        let options = CapacityOptions::new(5, u64::MAX, OutputRetention::SummaryOnly);
        match run_weighted_capacity(CapacityControl::Glm, &mut source, &options) {
            Err(WeightedCapacityError::SourceChanged { first, second }) => {
                assert_ne!(first, second)
            }
            other => panic!("expected a changed-source refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_row_file_that_disagrees_with_its_header_is_refused() {
        let (x, y, weights) = design();
        let bytes = encode_weighted_row_file(&x, &y, &weights).expect("encode");

        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&[0_u8; 8]);
        let trailing_guard = write_rows("trailing", &trailing);
        assert!(matches!(
            FileWeightedRows::open(&trailing_guard.0),
            Err(WeightedCapacityError::RowFile(RowFileFault::LengthMismatch { .. }))
        ));

        let truncated_guard = write_rows("truncated", &bytes[..bytes.len() - 8]);
        match FileWeightedRows::open(&truncated_guard.0) {
            Err(WeightedCapacityError::RowFile(RowFileFault::LengthMismatch {
                expected_bytes,
                actual_bytes,
            })) => assert_eq!(expected_bytes, actual_bytes + 8),
            other => panic!("expected a length mismatch, got {other:?}"),
        }

        let mut wrong_magic = bytes;
        wrong_magic[0] = b'X';
        let magic_guard = write_rows("magic", &wrong_magic);
        assert!(matches!(
            FileWeightedRows::open(&magic_guard.0),
            Err(WeightedCapacityError::RowFile(RowFileFault::BadMagic))
        ));
    }

    #[test]
    fn errors_keep_their_source() {
        let missing = std::env::temp_dir().join("weighted-gaussian-capacity-does-not-exist.wgol");
        match FileWeightedRows::open(&missing) {
            Err(error @ WeightedCapacityError::Io { .. }) => {
                assert!(error.source().is_some());
            }
            other => panic!("expected an io error, got {other:?}"),
        }
        match zeros(usize::MAX / 16, "probe buffer") {
            Err(error @ WeightedCapacityError::Allocation { what: "probe buffer", .. }) => {
                assert!(error.source().is_some());
            }
            other => panic!("expected an allocation failure, got {other:?}"),
        }
    }
}
