//! Refresh of a linear least-squares summary against its retained rows.
//!
//! [`OlsRefresh`] keeps a [`LinearOlsSuffStats`] summary, the rows it came from, the source
//! the rows belong to, a running source digest and a ledger of row counts. The contract is
//! [`crate::route_contract`] for [`crate::RefreshRoute::LinearOlsGram`]:
//!
//! * appended rows and a summarized window of the same source update the summary in place;
//! * an evicted row, a changed count weight, a changed mask or a replaced source drops the
//!   summary and rebuilds it from the retained rows ([`RefreshOutcome::Recomputed`]);
//! * rows of another source are refused: a summary of one source is never extended with
//!   another's rows;
//! * summary-only retention keeps no rows, so every rebuild is refused with its next attempt.
//!
//! [`OlsRefresh::check_against_fresh`] rebuilds the complete result (Gram, cross-product,
//! coefficients, residual variance), the source identity and the ledger from the retained rows
//! alone and compares each with the cached value. A weight is a nonnegative integer count: a
//! row with weight `k` enters the summary `k` times. A fractional weight is not representable
//! here and is refused rather than rounded.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::CanonicalWriter;

use crate::disk_retention::{SpillDecoder, SpillEncoder, SpillHandle, SpillRegistry, Spillable};
use crate::error::StateError;
use crate::retention::{RetainedOperation, RetentionMode};
use crate::suff_stats::{
    LinearOlsSuffStats, all_close, checked_row_ids, merge_within_tolerance, repeated_row,
    source_digest_start,
};

const SOURCE_TAG: &str = "antecedent.state.ols_refresh.source";

/// One retained observation.
#[derive(Clone, Debug, PartialEq)]
pub struct RefreshRow {
    /// Identity of the observation within its source. Unique per summary.
    pub id: u64,
    /// Design row.
    pub x: Vec<f64>,
    /// Response.
    pub y: f64,
    /// Integer count weight. Zero drops the row from the summary and keeps it retained.
    pub count: u32,
    /// Masked rows are retained and left out of the summary.
    pub masked: bool,
}

impl RefreshRow {
    /// A unit-weight, unmasked row.
    #[must_use]
    pub const fn new(id: u64, x: Vec<f64>, y: f64) -> Self {
        Self { id, x, y, count: 1, masked: false }
    }

    const fn live(&self) -> bool {
        !self.masked && self.count > 0
    }

    fn validate(&self, ncols: usize) -> Result<(), StateError> {
        if self.x.len() != ncols {
            return Err(StateError::Shape(format!("row len {} != ncols {ncols}", self.x.len())));
        }
        if !self.y.is_finite() || self.x.iter().any(|value| !value.is_finite()) {
            return Err(StateError::Numerical("design row and response must be finite".into()));
        }
        Ok(())
    }
}

/// How a change reached the cached result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefreshOutcome {
    /// The summary was updated in place.
    Updated,
    /// The summary was dropped and rebuilt from the retained rows.
    Recomputed,
}

/// Row counts a summary rests on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OlsLedgerState {
    /// Retained rows, masked or not.
    pub rows: u64,
    /// Rows that enter the summary (unmasked, positive count).
    pub live_rows: u64,
    /// Sum of the counts of the live rows. Equals the summary's `n`.
    pub effective_n: u64,
}

impl OlsLedgerState {
    fn add(&mut self, row: &RefreshRow) {
        self.rows = self.rows.saturating_add(1);
        if row.live() {
            self.live_rows = self.live_rows.saturating_add(1);
            self.effective_n = self.effective_n.saturating_add(u64::from(row.count));
        }
    }
}

/// The complete result of a summary: statistics, diagnostics, source identity and ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct OlsSnapshot {
    /// Source the rows belong to.
    pub source: Arc<str>,
    /// Observation count.
    pub n: u64,
    /// Row-major Gram.
    pub xtx: Vec<f64>,
    /// Cross-product.
    pub xty: Vec<f64>,
    /// Sum of squared responses.
    pub yty: f64,
    /// Least-squares coefficients, when the Gram is nonsingular.
    pub beta: Option<Vec<f64>>,
    /// Residual variance, when `n > p`.
    pub residual_variance: Option<f64>,
    /// Digest of the source and the ordered rows with their counts and masks.
    pub source_digest: String,
    /// Row counts.
    pub ledger: OlsLedgerState,
}

/// A least-squares summary together with what is needed to check or rebuild it.
#[derive(Clone, Debug)]
pub struct OlsRefresh {
    ncols: usize,
    source: Arc<str>,
    rows: Vec<RefreshRow>,
    ids: BTreeSet<u64>,
    retention: RetentionMode,
    stats: LinearOlsSuffStats,
    digest: CanonicalWriter,
    ledger: OlsLedgerState,
    recomputes: u64,
}

fn digest_start(source: &str, ncols: usize) -> CanonicalWriter {
    source_digest_start(SOURCE_TAG, source, ncols)
}

fn digest_row(writer: &mut CanonicalWriter, row: &RefreshRow) {
    writer
        .word(row.id)
        .reals(&row.x)
        .real(row.y)
        .word(u64::from(row.count))
        .word(u64::from(row.masked));
}

fn add_to_stats(stats: &mut LinearOlsSuffStats, row: &RefreshRow) -> Result<(), StateError> {
    if !row.live() {
        return Ok(());
    }
    if row.count == 1 {
        return stats.append_row(&row.x, row.y);
    }
    // Build a repeated-row summary using Chan merges. A power-of-two summary
    // doubles at most 31 times; combining the selected bits takes at most 32
    // merges. Stage it separately so a refused merge leaves `stats` unchanged.
    let mut power = LinearOlsSuffStats::new(stats.ncols())?;
    power.append_row(&row.x, row.y)?;
    let mut repeated = LinearOlsSuffStats::new(stats.ncols())?;
    let mut remaining = row.count;
    while remaining != 0 {
        if remaining & 1 != 0 {
            repeated.merge(&power)?;
        }
        remaining >>= 1;
        if remaining != 0 {
            let duplicate = power.clone();
            power.merge(&duplicate)?;
        }
    }
    stats.merge(&repeated)
}

/// Everything a summary holds, derived from `rows` alone.
fn derive(
    ncols: usize,
    source: &str,
    rows: &[RefreshRow],
) -> Result<(LinearOlsSuffStats, CanonicalWriter, OlsLedgerState), StateError> {
    let mut stats = LinearOlsSuffStats::new(ncols)?;
    let mut digest = digest_start(source, ncols);
    let mut ledger = OlsLedgerState { rows: 0, live_rows: 0, effective_n: 0 };
    for row in rows {
        add_to_stats(&mut stats, row)?;
        digest_row(&mut digest, row);
        ledger.add(row);
    }
    Ok((stats, digest, ledger))
}

fn refuse_retention(operation: RetainedOperation, mode: RetentionMode) -> StateError {
    StateError::InvalidEvent(format!(
        "{mode} cannot serve {operation}; reload the rows of this source and rebuild from them"
    ))
}

impl OlsRefresh {
    /// An empty summary of `ncols` columns for `source`, retaining rows.
    ///
    /// # Errors
    /// `ncols` is zero.
    pub fn new(source: impl Into<Arc<str>>, ncols: usize) -> Result<Self, StateError> {
        Self::from_rows(source, ncols, Vec::new())
    }

    /// A summary rebuilt from `rows`.
    ///
    /// # Errors
    /// `ncols` is zero, a row is malformed, or an id repeats.
    pub fn from_rows(
        source: impl Into<Arc<str>>,
        ncols: usize,
        rows: Vec<RefreshRow>,
    ) -> Result<Self, StateError> {
        if ncols == 0 {
            return Err(StateError::Shape("ncols is 0".into()));
        }
        let source: Arc<str> = source.into();
        let ids = checked_ids(ncols, &rows, &BTreeSet::new())?;
        let (stats, digest, ledger) = derive(ncols, &source, &rows)?;
        Ok(Self {
            ncols,
            source,
            rows,
            ids,
            retention: RetentionMode::FullReusable,
            stats,
            digest,
            ledger,
            recomputes: 0,
        })
    }

    /// Source the summary describes.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Retained rows. Empty under summary-only retention.
    #[must_use]
    pub fn rows(&self) -> &[RefreshRow] {
        &self.rows
    }

    /// The cached summary.
    #[must_use]
    pub const fn stats(&self) -> &LinearOlsSuffStats {
        &self.stats
    }

    /// Row counts of the cached summary.
    #[must_use]
    pub const fn ledger(&self) -> OlsLedgerState {
        self.ledger
    }

    /// How many times the summary was rebuilt from the retained rows.
    #[must_use]
    pub const fn recomputes(&self) -> u64 {
        self.recomputes
    }

    /// Retention mode.
    #[must_use]
    pub const fn retention(&self) -> RetentionMode {
        self.retention
    }

    /// Stop retaining rows. The summary, ledger and source identity stay, and every later
    /// change that needs the rows is refused.
    #[must_use]
    pub fn into_summary_only(mut self) -> Self {
        self.rows.clear();
        self.retention = RetentionMode::SummaryOnly;
        self
    }

    fn check_source(&self, source: &str) -> Result<(), StateError> {
        if source == self.source.as_ref() {
            return Ok(());
        }
        Err(StateError::StaleCache(format!(
            "rows are from source {source}, the summary is of {}; replace the source and \
             recompute from its rows",
            self.source
        )))
    }

    fn take(&mut self, row: &RefreshRow) {
        digest_row(&mut self.digest, row);
        self.ledger.add(row);
        self.ids.insert(row.id);
        if self.retention == RetentionMode::FullReusable {
            self.rows.push(row.clone());
        }
    }

    /// Append one row of the same source. Exact in place.
    ///
    /// # Errors
    /// Another source, a malformed row or a repeated id. Nothing changes.
    pub fn append_row(
        &mut self,
        source: &str,
        row: &RefreshRow,
    ) -> Result<RefreshOutcome, StateError> {
        self.check_source(source)?;
        row.validate(self.ncols)?;
        if self.ids.contains(&row.id) {
            return Err(repeated_row(row.id));
        }
        add_to_stats(&mut self.stats, row)?;
        self.take(row);
        Ok(RefreshOutcome::Updated)
    }

    /// Append a contiguous window of the same source by merging its summary. Exact in place.
    ///
    /// # Errors
    /// Another source, a malformed row or a repeated id (also within the window). Nothing
    /// changes.
    pub fn append_window(
        &mut self,
        source: &str,
        rows: &[RefreshRow],
    ) -> Result<RefreshOutcome, StateError> {
        self.check_source(source)?;
        checked_ids(self.ncols, rows, &self.ids)?;
        let mut window = LinearOlsSuffStats::new(self.ncols)?;
        for row in rows {
            add_to_stats(&mut window, row)?;
        }
        self.stats.merge(&window)?;
        for row in rows {
            self.take(row);
        }
        Ok(RefreshOutcome::Updated)
    }

    /// Remove one retained row and rebuild the summary from the rows that remain.
    ///
    /// # Errors
    /// Rows are not retained, or the id is unknown.
    pub fn evict_row(&mut self, id: u64) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::ExactRowEviction)?;
        let position = self.position(id)?;
        self.rows.remove(position);
        self.ids.remove(&id);
        self.rebuild()
    }

    /// Keep the most recently appended `max_rows` rows and drop the rest, then rebuild.
    ///
    /// # Errors
    /// Rows are not retained.
    pub fn retain_last(&mut self, max_rows: usize) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::ExactRowEviction)?;
        let drop_count = self.rows.len().saturating_sub(max_rows);
        for row in self.rows.drain(..drop_count) {
            self.ids.remove(&row.id);
        }
        self.rebuild()
    }

    /// Change the count weight of one retained row and rebuild the summary.
    ///
    /// # Errors
    /// Rows are not retained, or the id is unknown.
    pub fn set_count(&mut self, id: u64, count: u32) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::InPlaceWeightOrMaskUpdate)?;
        let position = self.position(id)?;
        if let Some(row) = self.rows.get_mut(position) {
            row.count = count;
        }
        self.rebuild()
    }

    /// Change a weight given as a real. Only a nonnegative integer is a count.
    ///
    /// # Errors
    /// A fractional, negative or non-finite weight (refused, not rounded), or the errors of
    /// [`Self::set_count`].
    pub fn set_weight(&mut self, id: u64, weight: f64) -> Result<RefreshOutcome, StateError> {
        let count = integer_count(weight).ok_or_else(|| {
            StateError::InvalidEvent(format!(
                "weight {weight} is not a nonnegative integer count; supply integer counts or \
                 fit the weighted least-squares route, which this summary does not carry"
            ))
        })?;
        self.set_count(id, count)
    }

    /// Change the mask of one retained row and rebuild the summary.
    ///
    /// # Errors
    /// Rows are not retained, or the id is unknown.
    pub fn set_mask(&mut self, id: u64, masked: bool) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::InPlaceWeightOrMaskUpdate)?;
        let position = self.position(id)?;
        if let Some(row) = self.rows.get_mut(position) {
            row.masked = masked;
        }
        self.rebuild()
    }

    /// Replace the source and its rows. Nothing of the old source survives.
    ///
    /// # Errors
    /// A malformed row or a repeated id. Nothing changes.
    pub fn replace_source(
        &mut self,
        source: impl Into<Arc<str>>,
        rows: Vec<RefreshRow>,
    ) -> Result<RefreshOutcome, StateError> {
        let source: Arc<str> = source.into();
        let ids = checked_ids(self.ncols, &rows, &BTreeSet::new())?;
        let (stats, digest, ledger) = derive(self.ncols, &source, &rows)?;
        self.source = source;
        self.ids = ids;
        self.stats = stats;
        self.digest = digest;
        self.ledger = ledger;
        self.rows = if self.retention == RetentionMode::FullReusable { rows } else { Vec::new() };
        self.recomputes = self.recomputes.saturating_add(1);
        Ok(RefreshOutcome::Recomputed)
    }

    /// Write the retained rows under `registry` as a disk-backed result.
    ///
    /// # Errors
    /// Rows are not retained, or the errors of [`SpillRegistry::spill`].
    pub fn spill(
        &self,
        registry: &Arc<SpillRegistry>,
        key: &str,
    ) -> Result<SpillHandle, StateError> {
        registry.spill(key, self)
    }

    /// A column was inserted. Refused: the retained rows carry no values for it.
    ///
    /// # Errors
    /// Always. The next attempt is to replace the source with rows of the wider design.
    pub fn insert_column(&self) -> Result<RefreshOutcome, StateError> {
        Err(StateError::InvalidEvent(
            "a column cannot be inserted into retained rows that carry no values for it; \
             replace the source with rows of the wider design"
                .into(),
        ))
    }

    /// Refuse `operation` when this block's retention cannot serve it.
    pub(crate) fn require_rows(&self, operation: RetainedOperation) -> Result<(), StateError> {
        self.require(operation)
    }

    fn require(&self, operation: RetainedOperation) -> Result<(), StateError> {
        self.retention.require(operation).map_err(|_| refuse_retention(operation, self.retention))
    }

    fn position(&self, id: u64) -> Result<usize, StateError> {
        self.rows
            .iter()
            .position(|row| row.id == id)
            .ok_or_else(|| StateError::UnknownId(format!("row {id}")))
    }

    fn rebuild(&mut self) -> Result<RefreshOutcome, StateError> {
        let (stats, digest, ledger) = derive(self.ncols, &self.source, &self.rows)?;
        self.stats = stats;
        self.digest = digest;
        self.ledger = ledger;
        self.recomputes = self.recomputes.saturating_add(1);
        Ok(RefreshOutcome::Recomputed)
    }

    /// The complete cached result.
    #[must_use]
    pub fn snapshot(&self) -> OlsSnapshot {
        snapshot_of(&self.source, &self.stats, &self.digest, self.ledger)
    }

    /// The complete result rebuilt from the retained rows alone.
    ///
    /// # Errors
    /// Rows are not retained.
    pub fn fresh_snapshot(&self) -> Result<OlsSnapshot, StateError> {
        self.require(RetainedOperation::RefitFromRetainedRows)?;
        let (stats, digest, ledger) = derive(self.ncols, &self.source, &self.rows)?;
        Ok(snapshot_of(&self.source, &stats, &digest, ledger))
    }

    /// Compare the cached result with a fresh reconstruction: statistics within
    /// [`merge_within_tolerance`]; coefficients and residual variance within it or, for a
    /// regressor far from zero, as the same fit (see
    /// [`LinearOlsSuffStats::coefficients_agree`]); source identity, source and ledger exactly.
    ///
    /// # Errors
    /// Rows are not retained, or the first field that disagrees.
    pub fn check_against_fresh(&self) -> Result<(), StateError> {
        self.require(RetainedOperation::RefitFromRetainedRows)?;
        let (stats, digest, ledger) = derive(self.ncols, &self.source, &self.rows)?;
        let fresh = snapshot_of(&self.source, &stats, &digest, ledger);
        let cached = self.snapshot();
        let mismatch = |field: &str| {
            Err(StateError::StaleCache(format!("cached {field} differs from the fresh rebuild")))
        };
        if cached.source != fresh.source {
            return mismatch("source");
        }
        if cached.source_digest != fresh.source_digest {
            return mismatch("source digest");
        }
        if cached.ledger != fresh.ledger || cached.n != fresh.n {
            return mismatch("ledger");
        }
        if !all_close(&cached.xtx, &fresh.xtx) || !all_close(&cached.xty, &fresh.xty) {
            return mismatch("Gram or cross-product");
        }
        if !merge_within_tolerance(cached.yty, fresh.yty) {
            return mismatch("sum of squared responses");
        }
        let coefficients = match (cached.beta.as_deref(), fresh.beta.as_deref()) {
            (Some(left), Some(right)) => stats.coefficients_agree(left, right),
            (None, None) => true,
            _ => false,
        };
        if !coefficients {
            return mismatch("coefficients");
        }
        if !stats.variances_agree(cached.residual_variance, fresh.residual_variance) {
            return mismatch("residual variance");
        }
        Ok(())
    }
}

fn checked_ids(
    ncols: usize,
    rows: &[RefreshRow],
    taken: &BTreeSet<u64>,
) -> Result<BTreeSet<u64>, StateError> {
    checked_row_ids(rows, taken, |row| row.validate(ncols).map(|()| row.id))
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the weight is finite, integral and within 0..=u32::MAX before the cast"
)]
fn integer_count(weight: f64) -> Option<u32> {
    let integral = weight.is_finite() && weight >= 0.0 && (weight - weight.round()).abs() <= 0.0;
    (integral && weight <= f64::from(u32::MAX)).then_some(weight as u32)
}

fn snapshot_of(
    source: &Arc<str>,
    stats: &LinearOlsSuffStats,
    digest: &CanonicalWriter,
    ledger: OlsLedgerState,
) -> OlsSnapshot {
    let beta = stats.solve_beta().ok();
    let residual_variance = beta.as_deref().and_then(|beta| stats.residual_variance(beta));
    OlsSnapshot {
        source: Arc::clone(source),
        n: stats.n(),
        xtx: stats.xtx().to_vec(),
        xty: stats.xty().to_vec(),
        yty: stats.yty(),
        beta,
        residual_variance,
        source_digest: digest.finish_hex(),
        ledger,
    }
}

impl Spillable for OlsRefresh {
    const KIND: &'static str = "antecedent.state.ols_refresh";

    fn encode(&self) -> Result<Vec<u8>, StateError> {
        self.require(RetainedOperation::RowReconstruction)?;
        let mut encoder = SpillEncoder::new();
        encoder.text(&self.source).word(self.ncols as u64).word(self.rows.len() as u64);
        for row in &self.rows {
            encoder
                .word(row.id)
                .reals(&row.x)
                .real(row.y)
                .word(u64::from(row.count))
                .flag(row.masked);
        }
        Ok(encoder.finish())
    }

    fn decode(bytes: &[u8]) -> Result<Self, StateError> {
        let mut decoder = SpillDecoder::new(bytes);
        let source = decoder.text()?;
        let ncols = usize::try_from(decoder.word()?)
            .map_err(|_| StateError::StaleCache("spilled column count is too large".into()))?;
        let count = decoder.count(8)?;
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let id = decoder.word()?;
            let x = decoder.reals()?;
            let y = decoder.real()?;
            let weight = u32::try_from(decoder.word()?)
                .map_err(|_| StateError::StaleCache("spilled count is too large".into()))?;
            let masked = decoder.flag()?;
            rows.push(RefreshRow { id, x, y, count: weight, masked });
        }
        decoder.finish()?;
        Self::from_rows(source, ncols, rows)
    }
}
