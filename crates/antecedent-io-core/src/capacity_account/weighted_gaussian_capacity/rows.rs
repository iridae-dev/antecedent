use super::{
    DesignShape, HEADER_BYTES, HEADER_LEN, MAGIC, RowFileFault, VERSION, WeightedCapacityError,
    allocation_failure, batch_byte_len, io_failure, parse_records, read_u32, read_u64,
    require_batch_request, require_window,
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

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
        } else if !x_rowmajor.len().is_multiple_of(y.len()) {
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
    if y.len() != weights.len() || y.is_empty() || !x_rowmajor.len().is_multiple_of(y.len()) {
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
