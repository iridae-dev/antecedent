//! Refresh of a streaming mean and scatter against its retained observations.
//!
//! [`CovarianceRefresh`] keeps a [`StreamingCovariance`], the observation vectors it came from,
//! the source they belong to, a running source digest and a ledger of row counts. The contract
//! is [`crate::route_contract`] for [`crate::RefreshRoute::StreamingCovariance`]:
//!
//! * appended observations and a summarized window of the same source update the scatter in
//!   place;
//! * an evicted observation, a changed mask, a slid window or a replaced source drops the
//!   scatter and rebuilds it from the retained observations
//!   ([`RefreshOutcome::Recomputed`]);
//! * observations of another source are refused;
//! * summary-only retention keeps no observations, so every rebuild is refused;
//! * disk-backed retention is [`CovarianceRefresh::spill`].
//!
//! [`CovarianceRefresh::check_against_fresh`] rebuilds the count, mean, scatter, sample
//! covariance, source identity and ledger from the retained observations alone and compares
//! each with the cached value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::CanonicalWriter;

use crate::disk_retention::{SpillDecoder, SpillEncoder, SpillHandle, SpillRegistry, Spillable};
use crate::error::StateError;
use crate::ols_refresh::{OlsLedgerState, RefreshOutcome};
use crate::retention::{RetainedOperation, RetentionMode};
use crate::suff_stats::{StreamingCovariance, merge_within_tolerance};

const SOURCE_TAG: &str = "antecedent.state.covariance_refresh.source";

/// One retained observation vector.
#[derive(Clone, Debug, PartialEq)]
pub struct CovarianceRow {
    /// Identity of the observation within its source. Unique per summary.
    pub id: u64,
    /// The observation vector.
    pub z: Vec<f64>,
    /// Masked observations are retained and left out of the scatter.
    pub masked: bool,
}

impl CovarianceRow {
    /// An unmasked observation.
    #[must_use]
    pub const fn new(id: u64, z: Vec<f64>) -> Self {
        Self { id, z, masked: false }
    }

    fn validate(&self, dim: usize) -> Result<(), StateError> {
        if self.z.len() != dim {
            return Err(StateError::Shape(format!("row len {} != dim {dim}", self.z.len())));
        }
        if self.z.iter().any(|value| !value.is_finite()) {
            return Err(StateError::Numerical("observations must be finite".into()));
        }
        Ok(())
    }
}

/// The complete result of a scatter summary.
#[derive(Clone, Debug, PartialEq)]
pub struct CovarianceSnapshot {
    /// Source the observations belong to.
    pub source: Arc<str>,
    /// Observation count.
    pub n: u64,
    /// Running mean.
    pub mean: Vec<f64>,
    /// Unnormalized scatter.
    pub m2: Vec<f64>,
    /// Sample covariance, when `n >= 2`.
    pub sample_covariance: Option<Vec<f64>>,
    /// Digest of the source and the ordered observations with their masks.
    pub source_digest: String,
    /// Row counts. `effective_n` equals the live row count.
    pub ledger: OlsLedgerState,
}

/// A streaming scatter together with what is needed to check or rebuild it.
#[derive(Clone, Debug)]
pub struct CovarianceRefresh {
    dim: usize,
    source: Arc<str>,
    rows: Vec<CovarianceRow>,
    ids: BTreeSet<u64>,
    retention: RetentionMode,
    stats: StreamingCovariance,
    digest: CanonicalWriter,
    ledger: OlsLedgerState,
    recomputes: u64,
}

fn digest_start(source: &str, dim: usize) -> CanonicalWriter {
    let mut writer = CanonicalWriter::new(SOURCE_TAG);
    writer.text(source).word(dim as u64);
    writer
}

fn digest_row(writer: &mut CanonicalWriter, row: &CovarianceRow) {
    writer.word(row.id).reals(&row.z).word(u64::from(row.masked));
}

fn ledger_add(ledger: &mut OlsLedgerState, row: &CovarianceRow) {
    ledger.rows = ledger.rows.saturating_add(1);
    if !row.masked {
        ledger.live_rows = ledger.live_rows.saturating_add(1);
        ledger.effective_n = ledger.effective_n.saturating_add(1);
    }
}

const EMPTY_LEDGER: OlsLedgerState = OlsLedgerState { rows: 0, live_rows: 0, effective_n: 0 };

fn derive(
    dim: usize,
    source: &str,
    rows: &[CovarianceRow],
) -> Result<(StreamingCovariance, CanonicalWriter, OlsLedgerState), StateError> {
    let mut stats = StreamingCovariance::new(dim)?;
    let mut digest = digest_start(source, dim);
    let mut ledger = EMPTY_LEDGER;
    for row in rows {
        if !row.masked {
            stats.append(&row.z)?;
        }
        digest_row(&mut digest, row);
        ledger_add(&mut ledger, row);
    }
    Ok((stats, digest, ledger))
}

fn repeated(id: u64) -> StateError {
    StateError::InvalidEvent(format!(
        "observation {id} is already in the summary; each observation enters once, so drop the \
         duplicate or evict the earlier one first"
    ))
}

fn checked_ids(
    dim: usize,
    rows: &[CovarianceRow],
    taken: &BTreeSet<u64>,
) -> Result<BTreeSet<u64>, StateError> {
    let mut ids = BTreeSet::new();
    for row in rows {
        row.validate(dim)?;
        if taken.contains(&row.id) || !ids.insert(row.id) {
            return Err(repeated(row.id));
        }
    }
    Ok(ids)
}

impl CovarianceRefresh {
    /// A scatter of `dim` columns for `source`, rebuilt from `rows`.
    ///
    /// # Errors
    /// `dim` is zero, an observation is malformed, or an id repeats.
    pub fn from_rows(
        source: impl Into<Arc<str>>,
        dim: usize,
        rows: Vec<CovarianceRow>,
    ) -> Result<Self, StateError> {
        if dim == 0 {
            return Err(StateError::Shape("dim is 0".into()));
        }
        let source: Arc<str> = source.into();
        let ids = checked_ids(dim, &rows, &BTreeSet::new())?;
        let (stats, digest, ledger) = derive(dim, &source, &rows)?;
        Ok(Self {
            dim,
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

    /// An empty scatter.
    ///
    /// # Errors
    /// `dim` is zero.
    pub fn new(source: impl Into<Arc<str>>, dim: usize) -> Result<Self, StateError> {
        Self::from_rows(source, dim, Vec::new())
    }

    /// Source the summary describes.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Retained observations. Empty under summary-only retention.
    #[must_use]
    pub fn rows(&self) -> &[CovarianceRow] {
        &self.rows
    }

    /// The cached scatter.
    #[must_use]
    pub const fn stats(&self) -> &StreamingCovariance {
        &self.stats
    }

    /// Row counts of the cached scatter.
    #[must_use]
    pub const fn ledger(&self) -> OlsLedgerState {
        self.ledger
    }

    /// How many times the scatter was rebuilt from the retained observations.
    #[must_use]
    pub const fn recomputes(&self) -> u64 {
        self.recomputes
    }

    /// Retention mode.
    #[must_use]
    pub const fn retention(&self) -> RetentionMode {
        self.retention
    }

    /// Stop retaining observations. Later changes that need them are refused.
    #[must_use]
    pub fn into_summary_only(mut self) -> Self {
        self.rows.clear();
        self.retention = RetentionMode::SummaryOnly;
        self
    }

    fn require(&self, operation: RetainedOperation) -> Result<(), StateError> {
        self.retention.require(operation).map_err(|_| {
            StateError::InvalidEvent(format!(
                "{} cannot serve {operation}; reload the observations of this source and \
                 rebuild from them",
                self.retention
            ))
        })
    }

    fn check_source(&self, source: &str) -> Result<(), StateError> {
        if source == self.source.as_ref() {
            return Ok(());
        }
        Err(StateError::StaleCache(format!(
            "observations are from source {source}, the summary is of {}; replace the source \
             and recompute from its observations",
            self.source
        )))
    }

    fn take(&mut self, row: &CovarianceRow) {
        digest_row(&mut self.digest, row);
        ledger_add(&mut self.ledger, row);
        self.ids.insert(row.id);
        if self.retention == RetentionMode::FullReusable {
            self.rows.push(row.clone());
        }
    }

    /// Append one observation of the same source. Exact in place.
    ///
    /// # Errors
    /// Another source, a malformed observation or a repeated id. Nothing changes.
    pub fn append_row(
        &mut self,
        source: &str,
        row: &CovarianceRow,
    ) -> Result<RefreshOutcome, StateError> {
        self.check_source(source)?;
        row.validate(self.dim)?;
        if self.ids.contains(&row.id) {
            return Err(repeated(row.id));
        }
        if !row.masked {
            self.stats.append(&row.z)?;
        }
        self.take(row);
        Ok(RefreshOutcome::Updated)
    }

    /// Append a contiguous window of the same source by merging its scatter. Exact in place.
    ///
    /// # Errors
    /// Another source, a malformed observation or a repeated id (also within the window).
    /// Nothing changes.
    pub fn append_window(
        &mut self,
        source: &str,
        rows: &[CovarianceRow],
    ) -> Result<RefreshOutcome, StateError> {
        self.check_source(source)?;
        checked_ids(self.dim, rows, &self.ids)?;
        let mut window = StreamingCovariance::new(self.dim)?;
        for row in rows.iter().filter(|row| !row.masked) {
            window.append(&row.z)?;
        }
        self.stats.merge(&window)?;
        for row in rows {
            self.take(row);
        }
        Ok(RefreshOutcome::Updated)
    }

    fn position(&self, id: u64) -> Result<usize, StateError> {
        self.rows
            .iter()
            .position(|row| row.id == id)
            .ok_or_else(|| StateError::UnknownId(format!("observation {id}")))
    }

    fn rebuild(&mut self) -> Result<RefreshOutcome, StateError> {
        let (stats, digest, ledger) = derive(self.dim, &self.source, &self.rows)?;
        self.stats = stats;
        self.digest = digest;
        self.ledger = ledger;
        self.recomputes = self.recomputes.saturating_add(1);
        Ok(RefreshOutcome::Recomputed)
    }

    /// Remove one retained observation and rebuild the scatter from the rest.
    ///
    /// # Errors
    /// Observations are not retained, or the id is unknown.
    pub fn evict_row(&mut self, id: u64) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::ExactRowEviction)?;
        let position = self.position(id)?;
        self.rows.remove(position);
        self.ids.remove(&id);
        self.rebuild()
    }

    /// Keep the most recently appended `max_rows` observations, then rebuild.
    ///
    /// # Errors
    /// Observations are not retained.
    pub fn retain_last(&mut self, max_rows: usize) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::ExactRowEviction)?;
        let drop_count = self.rows.len().saturating_sub(max_rows);
        for row in self.rows.drain(..drop_count) {
            self.ids.remove(&row.id);
        }
        self.rebuild()
    }

    /// Change the mask of one retained observation and rebuild the scatter.
    ///
    /// # Errors
    /// Observations are not retained, or the id is unknown.
    pub fn set_mask(&mut self, id: u64, masked: bool) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::InPlaceWeightOrMaskUpdate)?;
        let position = self.position(id)?;
        if let Some(row) = self.rows.get_mut(position) {
            row.masked = masked;
        }
        self.rebuild()
    }

    /// Replace the source and its observations. Nothing of the old source survives.
    ///
    /// # Errors
    /// A malformed observation or a repeated id. Nothing changes.
    pub fn replace_source(
        &mut self,
        source: impl Into<Arc<str>>,
        rows: Vec<CovarianceRow>,
    ) -> Result<RefreshOutcome, StateError> {
        let source: Arc<str> = source.into();
        let ids = checked_ids(self.dim, &rows, &BTreeSet::new())?;
        let (stats, digest, ledger) = derive(self.dim, &source, &rows)?;
        self.source = source;
        self.ids = ids;
        self.stats = stats;
        self.digest = digest;
        self.ledger = ledger;
        self.rows = if self.retention == RetentionMode::FullReusable { rows } else { Vec::new() };
        self.recomputes = self.recomputes.saturating_add(1);
        Ok(RefreshOutcome::Recomputed)
    }

    /// The complete cached result.
    #[must_use]
    pub fn snapshot(&self) -> CovarianceSnapshot {
        snapshot_of(&self.source, &self.stats, &self.digest, self.ledger)
    }

    /// The complete result rebuilt from the retained observations alone.
    ///
    /// # Errors
    /// Observations are not retained.
    pub fn fresh_snapshot(&self) -> Result<CovarianceSnapshot, StateError> {
        self.require(RetainedOperation::RefitFromRetainedRows)?;
        let (stats, digest, ledger) = derive(self.dim, &self.source, &self.rows)?;
        Ok(snapshot_of(&self.source, &stats, &digest, ledger))
    }

    /// Compare the cached result with a fresh reconstruction: mean, scatter and covariance
    /// within [`merge_within_tolerance`]; count, source identity, source and ledger exactly.
    ///
    /// # Errors
    /// Observations are not retained, or the first field that disagrees.
    pub fn check_against_fresh(&self) -> Result<(), StateError> {
        let fresh = self.fresh_snapshot()?;
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
        if !close(&cached.mean, &fresh.mean) || !close(&cached.m2, &fresh.m2) {
            return mismatch("mean or scatter");
        }
        let covariance = match (&cached.sample_covariance, &fresh.sample_covariance) {
            (None, None) => true,
            (Some(a), Some(b)) => close(a, b),
            _ => false,
        };
        if !covariance {
            return mismatch("sample covariance");
        }
        Ok(())
    }

    /// Write the retained observations under `registry` as a disk-backed result.
    ///
    /// # Errors
    /// Observations are not retained, or the errors of [`SpillRegistry::spill`].
    pub fn spill(
        &self,
        registry: &Arc<SpillRegistry>,
        key: &str,
    ) -> Result<SpillHandle, StateError> {
        registry.spill(key, self)
    }
}

fn close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| merge_within_tolerance(*a, *b))
}

fn snapshot_of(
    source: &Arc<str>,
    stats: &StreamingCovariance,
    digest: &CanonicalWriter,
    ledger: OlsLedgerState,
) -> CovarianceSnapshot {
    CovarianceSnapshot {
        source: Arc::clone(source),
        n: stats.n(),
        mean: stats.mean().to_vec(),
        m2: stats.m2().to_vec(),
        sample_covariance: stats.sample_covariance(),
        source_digest: digest.finish_hex(),
        ledger,
    }
}

impl Spillable for CovarianceRefresh {
    const KIND: &'static str = "antecedent.state.covariance_refresh";

    fn encode(&self) -> Result<Vec<u8>, StateError> {
        self.require(RetainedOperation::RowReconstruction)?;
        let mut encoder = SpillEncoder::new();
        encoder.text(&self.source).word(self.dim as u64).word(self.rows.len() as u64);
        for row in &self.rows {
            encoder.word(row.id).reals(&row.z).flag(row.masked);
        }
        Ok(encoder.finish())
    }

    fn decode(bytes: &[u8]) -> Result<Self, StateError> {
        let mut decoder = SpillDecoder::new(bytes);
        let source = decoder.text()?;
        let dim = usize::try_from(decoder.word()?)
            .map_err(|_| StateError::StaleCache("spilled dimension is too large".into()))?;
        let count = decoder.count(8)?;
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let id = decoder.word()?;
            let z = decoder.reals()?;
            let masked = decoder.flag()?;
            rows.push(CovarianceRow { id, z, masked });
        }
        decoder.finish()?;
        Self::from_rows(source, dim, rows)
    }
}
