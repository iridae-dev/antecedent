//! File-backed and resident weighted Gaussian / OLS under one memory ceiling.
//!
//! Both [`CapacityControl::Glm`] and [`CapacityControl::GaussianBayes`] return
//! the same summary: weighted coefficients, `HC0` covariance, leverage, residual
//! checks, and the Gaussian cross-products `X'WX`, `X'Wy`, `y'Wy`, and `Σw`.
//! The coefficient call is [`antecedent_stats::fit_wls`] on the Cholesky factor
//! of that Gram, which is the weighted normal equation. Rows are read in batches
//! no larger than [`CapacityOptions::batch_rows`].
//!
//! The mechanism-registry fitter `fit_linear_gaussian_weighted` is `pub(super)`
//! in `antecedent-model` and is not a call site from this crate.
//! [`antecedent_prob::gaussian_target`] is crate-private as well. This route
//! does not time a host and does not record a measured resident set.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use antecedent_core::execution_facts::OutputRetention;
use antecedent_stats::{
    FaerBackend, FitDiagnostics, LeastSquaresWorkspace, cholesky_spd, fit_wls, invert_square,
};

const MAGIC: &[u8; 4] = b"WGOL";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 20;
const HEADER_BYTES: u64 = 20;
const IDENTITY_DOMAIN: &str = "antecedent.capacity.weighted-gaussian.v1";

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapacityOptions {
    /// Maximum source rows materialized in one read.
    pub batch_rows: usize,
    /// Bytes available for ownership, identity, decoders, spill, and output.
    pub ceiling_bytes: u64,
    /// Summary-only keeps the summary resident. Disk-backed charges it as spill.
    pub retention: OutputRetention,
}

/// Bytes reserved against [`CapacityOptions::ceiling_bytes`].
///
/// The total is the sum of the five charges. It is a reservation from the
/// shapes, not a measurement of a host allocator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CeilingCharges {
    /// Resident source bytes. A file handle charges nothing for the unread body.
    pub input_ownership_bytes: u64,
    /// Digest plus the one-row staging buffer that feeds it.
    pub content_identity_bytes: u64,
    /// Batch window, file decoder, and the reduced `fit_wls` buffers.
    pub decoder_bytes: u64,
    /// Declared summary bytes when retention is disk-backed. No path is stored.
    pub spill_bytes: u64,
    /// Summary bytes when retention is summary-only.
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

/// Why a weighted capacity run refused to return a summary.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
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
    /// Column count or slice lengths do not describe one design.
    #[error("design shape: {0}")]
    Shape(&'static str),
    /// A weight is negative or not finite.
    #[error("weight at row {row}: {reason}")]
    Weight {
        /// Zero-based row in read order.
        row: u64,
        /// `negative` or `not finite`.
        reason: &'static str,
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
    #[error("weighted solve: {0}")]
    Solve(String),
    /// The five charges do not fit in the ceiling.
    #[error("ceiling {ceiling} cannot hold {total} bytes")]
    OverCeiling {
        /// Declared ceiling.
        ceiling: u64,
        /// Charges that were refused.
        total: u64,
        /// The refused reservation.
        charges: CeilingCharges,
    },
    /// A byte count or a row count overflowed.
    #[error("capacity byte count overflowed")]
    Overflow,
    /// The row file could not be read or did not match this layout.
    #[error("row file: {0}")]
    Io(String),
    /// [`WeightedRowSource::prepare_decoder`] was not called for this bound.
    #[error("decoder was not prepared for this batch bound")]
    DecoderNotPrepared,
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
    /// The bound is zero or the reservation overflows.
    fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError>;

    /// Bytes requested from the allocator for the decoder. Zero before preparation.
    #[must_use = "the prepared decoder is compared with the batch bound"]
    fn allocated_decoder_bytes(&self) -> u64;

    /// Return to the first row.
    ///
    /// # Errors
    ///
    /// The file could not be seeked.
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
            return Err(WeightedCapacityError::Shape("weights length != rows"));
        }
        let ncols = if y.is_empty() {
            return Err(WeightedCapacityError::Shape("ncols"));
        } else if x_rowmajor.len() % y.len() != 0 {
            return Err(WeightedCapacityError::Shape("design length"));
        } else {
            x_rowmajor.len() / y.len()
        };
        if ncols == 0 {
            return Err(WeightedCapacityError::Shape("ncols"));
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
        for row in 0..take {
            let src = self.cursor + row;
            let start = row * self.ncols;
            x_rowmajor[start..start + self.ncols]
                .copy_from_slice(&self.x[src * self.ncols..(src + 1) * self.ncols]);
            y[row] = self.y[src];
            weights[row] = self.weights[src];
        }
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
    /// The header is short, the magic or version does not match, or a count
    /// does not fit.
    pub fn open(path: &Path) -> Result<Self, WeightedCapacityError> {
        let mut file = File::open(path).map_err(|err| WeightedCapacityError::Io(err.to_string()))?;
        let mut header = [0_u8; HEADER_LEN];
        file.read_exact(&mut header).map_err(|err| WeightedCapacityError::Io(err.to_string()))?;
        if &header[0..4] != MAGIC {
            return Err(WeightedCapacityError::Io("bad weighted-row magic".into()));
        }
        let version = read_u32(&header[4..8])?;
        if version != VERSION {
            return Err(WeightedCapacityError::Io("unsupported weighted-row version".into()));
        }
        let ncols_u = read_u32(&header[8..12])?;
        let nrows = read_u64(&header[12..20])?;
        let ncols = usize::try_from(ncols_u).map_err(|_| WeightedCapacityError::Overflow)?;
        if ncols == 0 {
            return Err(WeightedCapacityError::Shape("ncols"));
        }
        Ok(Self {
            file,
            ncols,
            nrows,
            rows_left: nrows,
            bound: None,
            decoder: Vec::new(),
            reserved_bytes: 0,
        })
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
        self.decoder = Vec::with_capacity(reserved_usize);
        self.reserved_bytes = reserved;
        self.bound = Some(batch_rows);
        Ok(())
    }

    fn allocated_decoder_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
        self.file
            .seek(SeekFrom::Start(HEADER_BYTES))
            .map_err(|err| WeightedCapacityError::Io(err.to_string()))?;
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
            .map_err(|err| WeightedCapacityError::Io(err.to_string()))?;
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
        return Err(WeightedCapacityError::Shape("design length"));
    }
    let ncols = x_rowmajor.len() / y.len();
    let ncols_u = u32::try_from(ncols).map_err(|_| WeightedCapacityError::Overflow)?;
    let nrows = u64::try_from(y.len()).map_err(|_| WeightedCapacityError::Overflow)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&ncols_u.to_le_bytes());
    bytes.extend_from_slice(&nrows.to_le_bytes());
    for row in 0..y.len() {
        let start = row * ncols;
        for value in &x_rowmajor[start..start + ncols] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&y[row].to_le_bytes());
        bytes.extend_from_slice(&weights[row].to_le_bytes());
    }
    Ok(bytes)
}

/// Run one capacity control.
///
/// Resident and file sources of the same rows share this function, the same
/// retention mode, and the same charges formula. [`OutputRetention::FullReusable`]
/// is refused: it would keep every source row.
///
/// # Errors
///
/// The ceiling is too small, the batch bound is zero, a weight or a row is not
/// usable, the Gram is singular, or the source cannot be read.
pub fn run_weighted_capacity(
    control: CapacityControl,
    source: &mut dyn WeightedRowSource,
    options: &CapacityOptions,
) -> Result<WeightedCapacityResult, WeightedCapacityError> {
    let ncols = source.ncols();
    if ncols == 0 {
        return Err(WeightedCapacityError::Shape("ncols"));
    }
    if options.batch_rows == 0 {
        return Err(WeightedCapacityError::BatchBound);
    }
    if options.retention == OutputRetention::FullReusable {
        return Err(WeightedCapacityError::RetentionKeepsRows);
    }
    let source_decoder = source.decoder_reservation(options.batch_rows)?;
    let charges = plan_charges(ncols, options, source.input_ownership_bytes(), source_decoder)?;
    if charges.total_bytes > options.ceiling_bytes {
        return Err(WeightedCapacityError::OverCeiling {
            ceiling: options.ceiling_bytes,
            total: charges.total_bytes,
            charges,
        });
    }
    source.prepare_decoder(options.batch_rows)?;
    source.rewind()?;
    let mut window = Window::allocate(options.batch_rows, ncols)?;
    let mut fold = Fold::new(ncols)?;
    let mut identity = RowIdentity::open(ncols)?;
    let batches = scan_rows(source, &mut window, |index, row, response, weight| {
        fold.push(index, row, response, weight)?;
        identity.push(row, response, weight)
    })?;
    if fold.positive_rows == 0 {
        return Err(WeightedCapacityError::Empty);
    }
    let solved = solve_weighted(&fold)?;
    let digest = identity.finish();
    source.rewind()?;
    let mut diagnostics = DiagAccum::from_solve(&solved)?;
    let _second = scan_rows(source, &mut window, |index, row, response, weight| {
        diagnostics.push(index, row, response, weight)
    })?;
    Ok(assemble(control, options.retention, charges, batches, digest, fold, solved, diagnostics))
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
            x: zeros(cells)?,
            y: zeros(batch_rows)?,
            w: zeros(batch_rows)?,
        })
    }
}

struct Fold {
    ncols: usize,
    xtwx: Vec<f64>,
    xtwy: Vec<f64>,
    ytwy: f64,
    n_eff: f64,
    positive_rows: u64,
}

impl Fold {
    fn new(ncols: usize) -> Result<Self, WeightedCapacityError> {
        let gram = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
        Ok(Self {
            ncols,
            xtwx: zeros(gram)?,
            xtwy: zeros(ncols)?,
            ytwy: 0.0,
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
        let scale = weight.sqrt();
        let scaled_y = scale * y;
        for (c1, &x1) in row.iter().enumerate() {
            let sx1 = scale * x1;
            self.xtwy[c1] += sx1 * scaled_y;
            for (offset, &x2) in row.iter().enumerate().skip(c1) {
                let value = sx1 * (scale * x2);
                self.xtwx[c1 * self.ncols + offset] += value;
                if c1 != offset {
                    self.xtwx[offset * self.ncols + c1] += value;
                }
            }
        }
        self.ytwy += scaled_y.powi(2);
        self.n_eff += weight;
        self.positive_rows =
            self.positive_rows.checked_add(1).ok_or(WeightedCapacityError::Overflow)?;
        Ok(())
    }
}

struct RowIdentity {
    hasher: blake3::Hasher,
    staging: Vec<u8>,
}

impl RowIdentity {
    fn open(ncols: usize) -> Result<Self, WeightedCapacityError> {
        let row_bytes = batch_byte_len(1, ncols)?;
        let width = usize::try_from(row_bytes).map_err(|_| WeightedCapacityError::Overflow)?;
        let mut staging = Vec::new();
        staging.try_reserve_exact(width).map_err(|_| WeightedCapacityError::Overflow)?;
        let mut hasher = blake3::Hasher::new_derive_key(IDENTITY_DOMAIN);
        let ncols_u = u32::try_from(ncols).map_err(|_| WeightedCapacityError::Overflow)?;
        hasher.update(&ncols_u.to_le_bytes());
        Ok(Self { hasher, staging })
    }

    fn push(&mut self, row: &[f64], y: f64, weight: f64) -> Result<(), WeightedCapacityError> {
        self.staging.clear();
        for value in row.iter().chain(std::iter::once(&y)).chain(std::iter::once(&weight)) {
            self.staging.extend_from_slice(&value.to_le_bytes());
        }
        self.hasher.update(&self.staging);
        Ok(())
    }

    fn finish(self) -> [u8; 32] {
        self.hasher.finalize().into()
    }
}

struct Solved {
    coefficients: Vec<f64>,
    diagnostics: FitDiagnostics,
    bread: Vec<f64>,
}

struct DiagAccum {
    beta: Vec<f64>,
    bread: Vec<f64>,
    meat: Vec<f64>,
    scratch: Vec<f64>,
    max_leverage: f64,
    sum_leverage: f64,
    saturated_rows: u64,
    weighted_rss: f64,
    max_abs_residual: f64,
}

impl DiagAccum {
    fn from_solve(solved: &Solved) -> Result<Self, WeightedCapacityError> {
        let p = solved.coefficients.len();
        let gram = p.checked_mul(p).ok_or(WeightedCapacityError::Overflow)?;
        Ok(Self {
            beta: solved.coefficients.clone(),
            bread: solved.bread.clone(),
            meat: zeros(gram)?,
            scratch: zeros(p)?,
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
        let mut predicted = 0.0;
        for (&coef, &value) in self.beta.iter().zip(row) {
            predicted += coef * value;
        }
        let residual = y - predicted;
        let scaled_residual = scale * residual;
        for (c1, &x1) in row.iter().enumerate() {
            let sx1 = scale * x1;
            for (c2, &x2) in row.iter().enumerate() {
                self.meat[c1 * p + c2] += scaled_residual.powi(2) * sx1 * (scale * x2);
            }
        }
        for a in 0..p {
            let mut acc = 0.0;
            for b in 0..p {
                acc += self.bread[a * p + b] * (scale * row[b]);
            }
            self.scratch[a] = acc;
        }
        let mut hat = 0.0;
        for a in 0..p {
            hat += (scale * row[a]) * self.scratch[a];
        }
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
}

fn assemble(
    control: CapacityControl,
    retention: OutputRetention,
    charges: CeilingCharges,
    batches: u64,
    content_identity: [u8; 32],
    fold: Fold,
    solved: Solved,
    diagnostics: DiagAccum,
) -> WeightedCapacityResult {
    let p = solved.coefficients.len();
    WeightedCapacityResult {
        control,
        retention,
        coefficients: solved.coefficients,
        robust_covariance: sandwich_product(&diagnostics.bread, &diagnostics.meat, p),
        leverage: LeverageSummary {
            max: diagnostics.max_leverage,
            sum: diagnostics.sum_leverage,
            saturated_rows: diagnostics.saturated_rows,
        },
        residual: ResidualCheck {
            weighted_rss: diagnostics.weighted_rss,
            max_abs_residual: diagnostics.max_abs_residual,
            positive_weight_rows: fold.positive_rows,
        },
        cross_products: GaussianCrossProducts {
            n_eff: fold.n_eff,
            xtwx: fold.xtwx,
            xtwy: fold.xtwy,
            ytwy: fold.ytwy,
        },
        content_identity,
        fit: solved.diagnostics,
        charges,
        batches,
        source_passes: 2,
    }
}

fn scan_rows(
    source: &mut dyn WeightedRowSource,
    window: &mut Window,
    mut on_row: impl FnMut(u64, &[f64], f64, f64) -> Result<(), WeightedCapacityError>,
) -> Result<u64, WeightedCapacityError> {
    let mut batches = 0_u64;
    let mut index = 0_u64;
    loop {
        let n = source.read_batch(window.batch_rows, &mut window.x, &mut window.y, &mut window.w)?;
        if n == 0 {
            break;
        }
        batches = batches.checked_add(1).ok_or(WeightedCapacityError::Overflow)?;
        for row in 0..n {
            let start = row * window.ncols;
            on_row(index, &window.x[start..start + window.ncols], window.y[row], window.w[row])?;
            index = index.checked_add(1).ok_or(WeightedCapacityError::Overflow)?;
        }
    }
    Ok(batches)
}

/// `L` is the row-major Cholesky factor of `X'WX`. That buffer is also the
/// column-major layout of `L'`. [`fit_wls`] on `L'` with right-hand side
/// `L⁻¹ X'Wy` is the weighted normal equation.
fn solve_weighted(fold: &Fold) -> Result<Solved, WeightedCapacityError> {
    let p = fold.ncols;
    let factor = cholesky_spd(&fold.xtwx, p).ok_or(WeightedCapacityError::Singular)?;
    let reduced_y = forward_substitute(&factor, p, &fold.xtwy)?;
    let ones = vec![1.0; p];
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_wls(&factor, p, p, &reduced_y, &ones, &FaerBackend, &mut workspace)
        .map_err(|err| WeightedCapacityError::Solve(err.to_string()))?;
    if fit.coefficients.len() != p {
        return Err(WeightedCapacityError::Shape("solve width"));
    }
    let bread = invert_square(&fold.xtwx, p).ok_or(WeightedCapacityError::Singular)?;
    Ok(Solved { coefficients: fit.coefficients, diagnostics: fit.diagnostics, bread })
}

fn forward_substitute(
    factor: &[f64],
    p: usize,
    rhs: &[f64],
) -> Result<Vec<f64>, WeightedCapacityError> {
    let mut solved = zeros(p)?;
    for i in 0..p {
        let mut acc = rhs[i];
        for j in 0..i {
            acc -= factor[i * p + j] * solved[j];
        }
        let diag = factor[i * p + i];
        if diag.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(WeightedCapacityError::Singular);
        }
        solved[i] = acc / diag;
    }
    Ok(solved)
}

fn sandwich_product(bread: &[f64], meat: &[f64], ncols: usize) -> Vec<f64> {
    let mut tmp = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            let mut acc = 0.0;
            for k in 0..ncols {
                acc += bread[i * ncols + k] * meat[k * ncols + j];
            }
            tmp[i * ncols + j] = acc;
        }
    }
    let mut out = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            let mut acc = 0.0;
            for k in 0..ncols {
                acc += tmp[i * ncols + k] * bread[k * ncols + j];
            }
            out[i * ncols + j] = acc;
        }
    }
    out
}

fn plan_charges(
    ncols: usize,
    options: &CapacityOptions,
    input_ownership_bytes: u64,
    source_decoder_bytes: u64,
) -> Result<CeilingCharges, WeightedCapacityError> {
    let payload = payload_bytes(ncols)?;
    let (output_bytes, spill_bytes) = match options.retention {
        OutputRetention::SummaryOnly => (payload, 0),
        OutputRetention::DiskBacked => (0, payload),
        OutputRetention::FullReusable => return Err(WeightedCapacityError::RetentionKeepsRows),
    };
    let identity = content_identity_bytes(ncols)?;
    let decoder = working_bytes(options.batch_rows, ncols, source_decoder_bytes)?;
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

/// Reservation for the reduced [`fit_wls`] call and the diagnostic scratch.
/// Seven `p×p` buffers and six length-`p` buffers. This is not a measured allocation.
fn solver_bytes(ncols: usize) -> Result<u64, WeightedCapacityError> {
    let gram = ncols.checked_mul(ncols).ok_or(WeightedCapacityError::Overflow)?;
    let floats = gram
        .checked_mul(7)
        .and_then(|cells| cells.checked_add(ncols.checked_mul(6)?))
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

fn zeros(len: usize) -> Result<Vec<f64>, WeightedCapacityError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| WeightedCapacityError::Overflow)?;
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
        return Err(WeightedCapacityError::Shape("row"));
    }
    if !weight.is_finite() {
        return Err(WeightedCapacityError::Weight { row: index, reason: "not finite" });
    }
    if weight.partial_cmp(&0.0) == Some(Ordering::Less) {
        return Err(WeightedCapacityError::Weight { row: index, reason: "negative" });
    }
    if !y.is_finite() || row.iter().any(|value| !value.is_finite()) {
        return Err(WeightedCapacityError::NonFinite { row: index });
    }
    Ok(())
}

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
        Err(WeightedCapacityError::Shape("batch window"))
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
    let record = ncols + 2;
    for row in 0..take {
        let base = row * record;
        for col in 0..ncols {
            x_rowmajor[row * ncols + col] = read_f64_at(bytes, base + col)?;
        }
        y[row] = read_f64_at(bytes, base + ncols)?;
        weights[row] = read_f64_at(bytes, base + ncols + 1)?;
    }
    Ok(())
}

fn read_f64_at(bytes: &[u8], index: usize) -> Result<f64, WeightedCapacityError> {
    let start = index.checked_mul(8).ok_or(WeightedCapacityError::Overflow)?;
    let end = start.checked_add(8).ok_or(WeightedCapacityError::Overflow)?;
    let chunk = bytes.get(start..end).ok_or(WeightedCapacityError::Io("short record".into()))?;
    let bits: [u8; 8] =
        chunk.try_into().map_err(|_| WeightedCapacityError::Io("short record".into()))?;
    Ok(f64::from_le_bytes(bits))
}

fn read_u32(bytes: &[u8]) -> Result<u32, WeightedCapacityError> {
    let chunk: [u8; 4] =
        bytes.try_into().map_err(|_| WeightedCapacityError::Io("short header".into()))?;
    Ok(u32::from_le_bytes(chunk))
}

fn read_u64(bytes: &[u8]) -> Result<u64, WeightedCapacityError> {
    let chunk: [u8; 8] =
        bytes.try_into().map_err(|_| WeightedCapacityError::Io("short header".into()))?;
    Ok(u64::from_le_bytes(chunk))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use antecedent_core::execution_facts::OutputRetention;
    use antecedent_stats::{
        FaerBackend, LeastSquaresWorkspace, SandwichKind, coefficient_covariance, fit_wls,
    };

    use super::{
        CapacityControl, CapacityOptions, FileWeightedRows, ResidentWeightedRows,
        WeightedCapacityError, WeightedCapacityResult, WeightedRowSource, encode_weighted_row_file,
        run_weighted_capacity,
    };

    fn design() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        let y = vec![1.0, 2.2, 2.8, 4.4, 100.0];
        let weights = vec![1.0, 2.0, 3.0, 4.0, 0.0];
        (x, y, weights)
    }

    fn options(retention: OutputRetention) -> CapacityOptions {
        CapacityOptions { batch_rows: 1, ceiling_bytes: u64::MAX, retention }
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
        assert!(bits_eq(&left.coefficients, &right.coefficients));
        assert!(bits_eq(&left.robust_covariance, &right.robust_covariance));
        assert!(bits_eq(&left.cross_products.xtwx, &right.cross_products.xtwx));
        assert!(bits_eq(&left.cross_products.xtwy, &right.cross_products.xtwy));
        assert_eq!(left.cross_products.n_eff.to_bits(), right.cross_products.n_eff.to_bits());
        assert_eq!(left.cross_products.ytwy.to_bits(), right.cross_products.ytwy.to_bits());
        assert_eq!(left.leverage.max.to_bits(), right.leverage.max.to_bits());
        assert_eq!(left.leverage.sum.to_bits(), right.leverage.sum.to_bits());
        assert_eq!(left.residual.weighted_rss.to_bits(), right.residual.weighted_rss.to_bits());
        assert_eq!(left.content_identity, right.content_identity);
        assert_eq!(left.retention, right.retention);
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

        fn decoder_reservation(
            &self,
            batch_rows: usize,
        ) -> Result<u64, WeightedCapacityError> {
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
        let bayes = run_weighted_capacity(
            CapacityControl::GaussianBayes,
            &mut bayes_source,
            &options,
        )
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
            let scaled = scaled_design(&x, &y, &weights);
            let mut workspace = LeastSquaresWorkspace::default();
            let reference = fit_wls(
                &scaled.x,
                scaled.nrows,
                2,
                &scaled.y,
                &scaled.ones,
                &FaerBackend,
                &mut workspace,
            )
            .expect("fit_wls");
            assert!((fit.coefficients[0] - reference.coefficients[0]).abs() < 1.0e-5);
            assert!((fit.coefficients[1] - reference.coefficients[1]).abs() < 1.0e-5);
            let residuals = scaled_residuals(&scaled, &fit.coefficients);
            let robust = coefficient_covariance(
                &scaled.x,
                scaled.nrows,
                2,
                &residuals,
                SandwichKind::Hc0,
            )
            .expect("hc0");
            for (got, expected) in fit.robust_covariance.iter().zip(&robust) {
                assert!((got - expected).abs() < 1.0e-4);
            }
        }
    }

    struct Scaled {
        x: Vec<f64>,
        y: Vec<f64>,
        ones: Vec<f64>,
        nrows: usize,
    }

    fn scaled_design(x: &[f64], y: &[f64], weights: &[f64]) -> Scaled {
        let nrows = y.len();
        let ncols = 2;
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

    fn scaled_residuals(scaled: &Scaled, beta: &[f64]) -> Vec<f64> {
        (0..scaled.nrows)
            .map(|row| {
                let predicted = beta[0] * scaled.x[row] + beta[1] * scaled.x[scaled.nrows + row];
                scaled.y[row] - predicted
            })
            .collect()
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
    fn ceiling_counts_ownership_identity_decoder_spill_and_output() {
        let (x, y, weights) = design();
        let summary_options = options(OutputRetention::SummaryOnly);
        let mut resident = ResidentWeightedRows::try_new(&x, &y, &weights).expect("resident");
        let summary = run_weighted_capacity(CapacityControl::Glm, &mut resident, &summary_options)
            .expect("summary");
        assert_eq!(summary.charges.spill_bytes, 0);
        assert!(summary.charges.output_bytes > 0);
        assert_eq!(summary.charges.input_ownership_bytes, 160);
        let mut tight = summary_options;
        tight.ceiling_bytes = summary.charges.total_bytes - summary.charges.input_ownership_bytes;
        let mut again = ResidentWeightedRows::try_new(&x, &y, &weights).expect("again");
        assert!(matches!(
            run_weighted_capacity(CapacityControl::Glm, &mut again, &tight),
            Err(WeightedCapacityError::OverCeiling { .. })
        ));
        tight.ceiling_bytes = summary.charges.total_bytes - summary.charges.content_identity_bytes;
        let mut identity = ResidentWeightedRows::try_new(&x, &y, &weights).expect("identity");
        assert!(matches!(
            run_weighted_capacity(CapacityControl::GaussianBayes, &mut identity, &tight),
            Err(WeightedCapacityError::OverCeiling { .. })
        ));
        tight.ceiling_bytes = summary.charges.total_bytes - summary.charges.decoder_bytes;
        let mut decoder = ResidentWeightedRows::try_new(&x, &y, &weights).expect("decoder");
        assert!(matches!(
            run_weighted_capacity(CapacityControl::Glm, &mut decoder, &tight),
            Err(WeightedCapacityError::OverCeiling { .. })
        ));
        tight.ceiling_bytes = summary.charges.total_bytes - summary.charges.output_bytes;
        let mut output = ResidentWeightedRows::try_new(&x, &y, &weights).expect("output");
        assert!(matches!(
            run_weighted_capacity(CapacityControl::Glm, &mut output, &tight),
            Err(WeightedCapacityError::OverCeiling { .. })
        ));
        let disk_options = options(OutputRetention::DiskBacked);
        let mut disk_source = ResidentWeightedRows::try_new(&x, &y, &weights).expect("disk");
        let disk =
            run_weighted_capacity(CapacityControl::GaussianBayes, &mut disk_source, &disk_options)
                .expect("disk");
        assert_eq!(disk.retention, OutputRetention::DiskBacked);
        assert_eq!(disk.charges.output_bytes, 0);
        assert_eq!(disk.charges.spill_bytes, summary.charges.output_bytes);
        same_science_ignoring_retention(&summary, &disk);
        let mut below_spill = disk_options;
        below_spill.ceiling_bytes = disk.charges.total_bytes - disk.charges.spill_bytes;
        let mut spill = ResidentWeightedRows::try_new(&x, &y, &weights).expect("spill");
        assert!(matches!(
            run_weighted_capacity(CapacityControl::Glm, &mut spill, &below_spill),
            Err(WeightedCapacityError::OverCeiling { .. })
        ));
        let mut kept = ResidentWeightedRows::try_new(&x, &y, &weights).expect("kept");
        let mut full = summary_options;
        full.retention = OutputRetention::FullReusable;
        assert!(matches!(
            run_weighted_capacity(CapacityControl::Glm, &mut kept, &full),
            Err(WeightedCapacityError::RetentionKeepsRows)
        ));
    }

    fn same_science_ignoring_retention(
        left: &WeightedCapacityResult,
        right: &WeightedCapacityResult,
    ) {
        assert!(bits_eq(&left.coefficients, &right.coefficients));
        assert!(bits_eq(&left.robust_covariance, &right.robust_covariance));
        assert_eq!(left.content_identity, right.content_identity);
        assert_eq!(left.source_passes, right.source_passes);
        assert_eq!(left.residual.positive_weight_rows, right.residual.positive_weight_rows);
    }
}
