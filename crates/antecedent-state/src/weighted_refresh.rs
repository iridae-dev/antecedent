//! Refresh of a weighted least-squares summary against its retained rows.
//!
//! [`WeightedOlsRefresh`] is the real-weight form of [`crate::OlsRefresh`]. A row carries a
//! finite nonnegative precision weight `w` (a case weight, a survey or inverse-probability
//! weight after normalisation, or any fractional count). The summary keeps `Σw`, `Σw²`, the
//! weighted Gram `XᵀWX`, `XᵀWy`, `yᵀWy`, and a weighted centered scatter of `[x | y]` for a
//! translation-stable residual scale. The contract is [`crate::route_contract`] for
//! [`crate::RefreshRoute::LinearOlsGram`]:
//!
//! * appended rows and a summarized window of the same source update the summary in place;
//! * an evicted row, a changed weight (one row or every row), a changed mask, a slid window
//!   or a replaced source drops the summary and rebuilds it from the retained rows
//!   ([`RefreshOutcome::Recomputed`]);
//! * rows of another source are refused;
//! * summary-only retention keeps no rows, so every rebuild is refused with its next attempt;
//! * disk-backed retention is [`WeightedOlsRefresh::spill`], which writes the retained rows
//!   under a [`SpillRegistry`] byte budget and reloads them as the resident state.
//!
//! [`WeightedOlsRefresh::check_against_fresh`] rebuilds the complete result (weights, Gram,
//! coefficients, residual variance, coefficient standard errors, effective sample size), the
//! source identity and the ledger from the retained rows alone and compares each with the
//! cached value. The residual variance is `SSE_w / (live rows − p)`: weights are precision
//! weights, so scaling every weight by `c` scales it by `c`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::CanonicalWriter;
use antecedent_stats::invert_square;

use crate::disk_retention::{SpillDecoder, SpillEncoder, SpillHandle, SpillRegistry, Spillable};
use crate::error::StateError;
use crate::ols_refresh::RefreshOutcome;
use crate::retention::{RetainedOperation, RetentionMode};
use crate::suff_stats::{merge_within_tolerance, solve_gram};

const SOURCE_TAG: &str = "antecedent.state.weighted_refresh.source";

/// One retained weighted observation.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedRow {
    /// Identity of the observation within its source. Unique per summary.
    pub id: u64,
    /// Design row.
    pub x: Vec<f64>,
    /// Response.
    pub y: f64,
    /// Finite nonnegative precision weight. Zero keeps the row retained and out of the summary.
    pub weight: f64,
    /// Masked rows are retained and left out of the summary.
    pub masked: bool,
}

impl WeightedRow {
    /// A unit-weight, unmasked row.
    #[must_use]
    pub const fn new(id: u64, x: Vec<f64>, y: f64) -> Self {
        Self { id, x, y, weight: 1.0, masked: false }
    }

    /// A weighted, unmasked row.
    #[must_use]
    pub const fn weighted(id: u64, x: Vec<f64>, y: f64, weight: f64) -> Self {
        Self { id, x, y, weight, masked: false }
    }

    fn live(&self) -> bool {
        !self.masked && self.weight > 0.0
    }

    fn validate(&self, ncols: usize) -> Result<(), StateError> {
        if self.x.len() != ncols {
            return Err(StateError::Shape(format!("row len {} != ncols {ncols}", self.x.len())));
        }
        if !self.y.is_finite() || self.x.iter().any(|value| !value.is_finite()) {
            return Err(StateError::Numerical("design row and response must be finite".into()));
        }
        check_weight(self.weight)
    }
}

fn check_weight(weight: f64) -> Result<(), StateError> {
    if weight.is_finite() && weight >= 0.0 {
        Ok(())
    } else {
        Err(StateError::InvalidEvent(format!(
            "weight {weight} is not finite and nonnegative; supply a nonnegative weight, or mask \
             the row"
        )))
    }
}

/// Row counts and total weight a weighted summary rests on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedLedger {
    /// Retained rows, masked or not.
    pub rows: u64,
    /// Rows that enter the summary (unmasked, positive weight).
    pub live_rows: u64,
    /// Sum of the weights of the live rows.
    pub total_weight: f64,
}

/// Weighted sufficient statistics, built only from live rows.
#[derive(Clone, Debug, PartialEq)]
struct WeightedStats {
    ncols: usize,
    live: u64,
    sw: f64,
    sw2: f64,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    yty: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl WeightedStats {
    fn new(ncols: usize) -> Result<Self, StateError> {
        let joint = ncols
            .checked_add(1)
            .ok_or_else(|| StateError::Shape("weighted joint dimension overflow".into()))?;
        let gram = ncols
            .checked_mul(ncols)
            .ok_or_else(|| StateError::Shape("weighted gram dimension overflow".into()))?;
        let cells = joint
            .checked_mul(joint)
            .ok_or_else(|| StateError::Shape("weighted joint dimension overflow".into()))?;
        Ok(Self {
            ncols,
            live: 0,
            sw: 0.0,
            sw2: 0.0,
            xtx: vec![0.0; gram],
            xty: vec![0.0; ncols],
            yty: 0.0,
            mean: vec![0.0; joint],
            m2: vec![0.0; cells],
        })
    }

    fn add(&mut self, row: &WeightedRow) {
        if !row.live() {
            return;
        }
        let p = self.ncols;
        let dim = p + 1;
        let w = row.weight;
        for i in 0..p {
            self.xty[i] += w * row.x[i] * row.y;
            for j in 0..p {
                self.xtx[i * p + j] += w * row.x[i] * row.x[j];
            }
        }
        self.yty += w * row.y * row.y;
        let mut z = row.x.clone();
        z.push(row.y);
        let sw_next = self.sw + w;
        let delta: Vec<f64> = z.iter().zip(&self.mean).map(|(value, mean)| value - mean).collect();
        for (mean, step) in self.mean.iter_mut().zip(&delta) {
            *mean += step * (w / sw_next);
        }
        for (i, &delta_i) in delta.iter().enumerate() {
            for (j, (&z_j, &mean_j)) in z.iter().zip(&self.mean).enumerate() {
                self.m2[i * dim + j] += w * delta_i * (z_j - mean_j);
            }
        }
        self.sw = sw_next;
        self.sw2 += w * w;
        self.live = self.live.saturating_add(1);
    }

    fn merge(&mut self, other: &Self) {
        if other.live == 0 {
            return;
        }
        if self.live == 0 {
            *self = other.clone();
            return;
        }
        let dim = self.ncols + 1;
        let total = self.sw + other.sw;
        let scale = self.sw * other.sw / total;
        let delta: Vec<f64> = other.mean.iter().zip(&self.mean).map(|(b, a)| b - a).collect();
        for i in 0..dim {
            for j in 0..dim {
                self.m2[i * dim + j] += other.m2[i * dim + j] + delta[i] * delta[j] * scale;
            }
        }
        for (mean, step) in self.mean.iter_mut().zip(&delta) {
            *mean += step * (other.sw / total);
        }
        for (slot, add) in self.xtx.iter_mut().zip(&other.xtx) {
            *slot += add;
        }
        for (slot, add) in self.xty.iter_mut().zip(&other.xty) {
            *slot += add;
        }
        self.yty += other.yty;
        self.sw = total;
        self.sw2 += other.sw2;
        self.live = self.live.saturating_add(other.live);
    }

    fn beta(&self) -> Option<Vec<f64>> {
        solve_gram(&self.xtx, &self.xty, self.ncols).ok()
    }

    fn residual_variance(&self, beta: &[f64]) -> Option<f64> {
        let p = self.ncols;
        if beta.len() != p || usize::try_from(self.live).is_ok_and(|live| live <= p) {
            return None;
        }
        let dim = p + 1;
        let mut sse = self.m2[p * dim + p];
        for i in 0..p {
            sse -= 2.0 * beta[i] * self.m2[i * dim + p];
            for j in 0..p {
                sse += beta[i] * self.m2[i * dim + j] * beta[j];
            }
        }
        let mut r_bar = self.mean[p];
        for (i, &coefficient) in beta.iter().enumerate() {
            r_bar -= coefficient * self.mean[i];
        }
        sse += self.sw * r_bar * r_bar;
        if !sse.is_finite() {
            return None;
        }
        if sse < 0.0 {
            let scale = self.m2[p * dim + p].abs().max(1.0);
            if sse.abs() > 1e-12 * scale {
                return None;
            }
            sse = 0.0;
        }
        Some(sse / (self.live as f64 - p as f64))
    }
}

/// The complete result of a weighted summary.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedSnapshot {
    /// Source the rows belong to.
    pub source: Arc<str>,
    /// Sum of the live weights.
    pub total_weight: f64,
    /// Kish effective sample size `(Σw)² / Σw²`, zero with no live rows.
    pub effective_sample_size: f64,
    /// Row-major weighted Gram `XᵀWX`.
    pub xtx: Vec<f64>,
    /// `XᵀWy`.
    pub xty: Vec<f64>,
    /// `yᵀWy`.
    pub yty: f64,
    /// Weighted least-squares coefficients, when the Gram is nonsingular.
    pub beta: Option<Vec<f64>>,
    /// Residual variance, when there are more live rows than columns.
    pub residual_variance: Option<f64>,
    /// Model-based coefficient standard errors `sqrt(σ² (XᵀWX)⁻¹ᵢᵢ)`.
    pub beta_se: Option<Vec<f64>>,
    /// Digest of the source and the ordered rows with their weights and masks.
    pub source_digest: String,
    /// Row counts and total weight.
    pub ledger: WeightedLedger,
}

/// A weighted least-squares summary together with what is needed to check or rebuild it.
#[derive(Clone, Debug)]
pub struct WeightedOlsRefresh {
    ncols: usize,
    source: Arc<str>,
    rows: Vec<WeightedRow>,
    ids: BTreeSet<u64>,
    retention: RetentionMode,
    stats: WeightedStats,
    digest: CanonicalWriter,
    ledger: WeightedLedger,
    recomputes: u64,
}

fn digest_start(source: &str, ncols: usize) -> CanonicalWriter {
    let mut writer = CanonicalWriter::new(SOURCE_TAG);
    writer.text(source).word(ncols as u64);
    writer
}

fn digest_row(writer: &mut CanonicalWriter, row: &WeightedRow) {
    writer.word(row.id).reals(&row.x).real(row.y).real(row.weight).word(u64::from(row.masked));
}

fn ledger_add(ledger: &mut WeightedLedger, row: &WeightedRow) {
    ledger.rows = ledger.rows.saturating_add(1);
    if row.live() {
        ledger.live_rows = ledger.live_rows.saturating_add(1);
        ledger.total_weight += row.weight;
    }
}

const EMPTY_LEDGER: WeightedLedger = WeightedLedger { rows: 0, live_rows: 0, total_weight: 0.0 };

fn derive(
    ncols: usize,
    source: &str,
    rows: &[WeightedRow],
) -> Result<(WeightedStats, CanonicalWriter, WeightedLedger), StateError> {
    let mut stats = WeightedStats::new(ncols)?;
    let mut digest = digest_start(source, ncols);
    let mut ledger = EMPTY_LEDGER;
    for row in rows {
        stats.add(row);
        digest_row(&mut digest, row);
        ledger_add(&mut ledger, row);
    }
    Ok((stats, digest, ledger))
}

fn repeated(id: u64) -> StateError {
    StateError::InvalidEvent(format!(
        "row {id} is already in the summary; each observation enters once, so drop the \
         duplicate or evict the earlier row first"
    ))
}

fn checked_ids(
    ncols: usize,
    rows: &[WeightedRow],
    taken: &BTreeSet<u64>,
) -> Result<BTreeSet<u64>, StateError> {
    let mut ids = BTreeSet::new();
    for row in rows {
        row.validate(ncols)?;
        if taken.contains(&row.id) || !ids.insert(row.id) {
            return Err(repeated(row.id));
        }
    }
    Ok(ids)
}

impl WeightedOlsRefresh {
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
    /// `ncols` is zero, a row is malformed or has an invalid weight, or an id repeats.
    pub fn from_rows(
        source: impl Into<Arc<str>>,
        ncols: usize,
        rows: Vec<WeightedRow>,
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

    /// Number of design columns.
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.ncols
    }

    /// Retained rows. Empty under summary-only retention.
    #[must_use]
    pub fn rows(&self) -> &[WeightedRow] {
        &self.rows
    }

    /// Row counts and total weight of the cached summary.
    #[must_use]
    pub const fn ledger(&self) -> WeightedLedger {
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

    fn require(&self, operation: RetainedOperation) -> Result<(), StateError> {
        self.retention.require(operation).map_err(|_| {
            StateError::InvalidEvent(format!(
                "{} cannot serve {operation}; reload the rows of this source and rebuild from them",
                self.retention
            ))
        })
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

    fn take(&mut self, row: &WeightedRow) {
        digest_row(&mut self.digest, row);
        ledger_add(&mut self.ledger, row);
        self.ids.insert(row.id);
        if self.retention == RetentionMode::FullReusable {
            self.rows.push(row.clone());
        }
    }

    /// Append one row of the same source. Exact in place.
    ///
    /// # Errors
    /// Another source, a malformed row or weight, or a repeated id. Nothing changes.
    pub fn append_row(
        &mut self,
        source: &str,
        row: &WeightedRow,
    ) -> Result<RefreshOutcome, StateError> {
        self.check_source(source)?;
        row.validate(self.ncols)?;
        if self.ids.contains(&row.id) {
            return Err(repeated(row.id));
        }
        self.stats.add(row);
        self.take(row);
        Ok(RefreshOutcome::Updated)
    }

    /// Append a contiguous window of the same source by merging its summary. Exact in place.
    ///
    /// # Errors
    /// Another source, a malformed row or weight, or a repeated id (also within the window).
    /// Nothing changes.
    pub fn append_window(
        &mut self,
        source: &str,
        rows: &[WeightedRow],
    ) -> Result<RefreshOutcome, StateError> {
        self.check_source(source)?;
        checked_ids(self.ncols, rows, &self.ids)?;
        let mut window = WeightedStats::new(self.ncols)?;
        for row in rows {
            window.add(row);
        }
        self.stats.merge(&window);
        for row in rows {
            self.take(row);
        }
        Ok(RefreshOutcome::Updated)
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

    /// Rebuild the summary from the retained rows.
    ///
    /// # Errors
    /// Rows are not retained.
    pub fn refresh(&mut self) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::RefitFromRetainedRows)?;
        self.rebuild()
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

    /// Change the weight of one retained row and rebuild the summary.
    ///
    /// # Errors
    /// Rows are not retained, the id is unknown, or the weight is not finite and nonnegative.
    pub fn set_weight(&mut self, id: u64, weight: f64) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::InPlaceWeightOrMaskUpdate)?;
        check_weight(weight)?;
        let position = self.position(id)?;
        if let Some(row) = self.rows.get_mut(position) {
            row.weight = weight;
        }
        self.rebuild()
    }

    /// Multiply every retained weight by `factor` and rebuild the summary.
    ///
    /// # Errors
    /// Rows are not retained, or `factor` is not finite and nonnegative.
    pub fn scale_weights(&mut self, factor: f64) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::InPlaceWeightOrMaskUpdate)?;
        check_weight(factor)?;
        for row in &mut self.rows {
            row.weight *= factor;
        }
        self.rebuild()
    }

    /// Replace every retained weight from `weights`, given in retained-row order, and rebuild.
    ///
    /// # Errors
    /// Rows are not retained, the length differs from the retained rows, or a weight is not
    /// finite and nonnegative. Nothing changes on an error.
    pub fn replace_weights(&mut self, weights: &[f64]) -> Result<RefreshOutcome, StateError> {
        self.require(RetainedOperation::InPlaceWeightOrMaskUpdate)?;
        if weights.len() != self.rows.len() {
            return Err(StateError::Shape(format!(
                "{} weights for {} retained rows",
                weights.len(),
                self.rows.len()
            )));
        }
        for weight in weights {
            check_weight(*weight)?;
        }
        for (row, weight) in self.rows.iter_mut().zip(weights) {
            row.weight = *weight;
        }
        self.rebuild()
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
    /// A malformed row or weight, or a repeated id. Nothing changes.
    pub fn replace_source(
        &mut self,
        source: impl Into<Arc<str>>,
        rows: Vec<WeightedRow>,
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

    /// The complete cached result.
    #[must_use]
    pub fn snapshot(&self) -> WeightedSnapshot {
        snapshot_of(&self.source, &self.stats, &self.digest, self.ledger)
    }

    /// The complete result rebuilt from the retained rows alone.
    ///
    /// # Errors
    /// Rows are not retained.
    pub fn fresh_snapshot(&self) -> Result<WeightedSnapshot, StateError> {
        self.require(RetainedOperation::RefitFromRetainedRows)?;
        let (stats, digest, ledger) = derive(self.ncols, &self.source, &self.rows)?;
        Ok(snapshot_of(&self.source, &stats, &digest, ledger))
    }

    /// Compare the cached result with a fresh reconstruction: statistics, coefficients,
    /// standard errors, residual variance and weights within [`merge_within_tolerance`];
    /// source identity, source and row counts exactly.
    ///
    /// # Errors
    /// Rows are not retained, or the first field that disagrees.
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
        if cached.ledger.rows != fresh.ledger.rows
            || cached.ledger.live_rows != fresh.ledger.live_rows
            || !merge_within_tolerance(cached.ledger.total_weight, fresh.ledger.total_weight)
        {
            return mismatch("ledger");
        }
        if !merge_within_tolerance(cached.total_weight, fresh.total_weight)
            || !merge_within_tolerance(cached.effective_sample_size, fresh.effective_sample_size)
        {
            return mismatch("total weight or effective sample size");
        }
        if !all_close(&cached.xtx, &fresh.xtx) || !all_close(&cached.xty, &fresh.xty) {
            return mismatch("Gram or cross-product");
        }
        if !merge_within_tolerance(cached.yty, fresh.yty) {
            return mismatch("weighted sum of squared responses");
        }
        if !optional_close(cached.beta.as_deref(), fresh.beta.as_deref()) {
            return mismatch("coefficients");
        }
        if !optional_close(cached.beta_se.as_deref(), fresh.beta_se.as_deref()) {
            return mismatch("coefficient standard errors");
        }
        let variance = match (cached.residual_variance, fresh.residual_variance) {
            (None, None) => true,
            (Some(a), Some(b)) => merge_within_tolerance(a, b),
            _ => false,
        };
        if !variance {
            return mismatch("residual variance");
        }
        Ok(())
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
}

fn all_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| merge_within_tolerance(*a, *b))
}

fn optional_close(left: Option<&[f64]>, right: Option<&[f64]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => all_close(a, b),
        _ => false,
    }
}

fn snapshot_of(
    source: &Arc<str>,
    stats: &WeightedStats,
    digest: &CanonicalWriter,
    ledger: WeightedLedger,
) -> WeightedSnapshot {
    let beta = stats.beta();
    let residual_variance = beta.as_deref().and_then(|beta| stats.residual_variance(beta));
    let beta_se = residual_variance.and_then(|variance| {
        let inverse = invert_square(&stats.xtx, stats.ncols)?;
        (0..stats.ncols)
            .map(|i| {
                let value = variance * inverse[i * stats.ncols + i];
                (value.is_finite() && value >= 0.0).then(|| value.sqrt())
            })
            .collect::<Option<Vec<f64>>>()
    });
    let effective_sample_size = if stats.sw2 > 0.0 { stats.sw * stats.sw / stats.sw2 } else { 0.0 };
    WeightedSnapshot {
        source: Arc::clone(source),
        total_weight: stats.sw,
        effective_sample_size,
        xtx: stats.xtx.clone(),
        xty: stats.xty.clone(),
        yty: stats.yty,
        beta,
        residual_variance,
        beta_se,
        source_digest: digest.finish_hex(),
        ledger,
    }
}

impl Spillable for WeightedOlsRefresh {
    const KIND: &'static str = "antecedent.state.weighted_ols_refresh";

    fn encode(&self) -> Result<Vec<u8>, StateError> {
        self.require(RetainedOperation::RowReconstruction)?;
        let mut encoder = SpillEncoder::new();
        encoder.text(&self.source).word(self.ncols as u64).word(self.rows.len() as u64);
        for row in &self.rows {
            encoder.word(row.id).reals(&row.x).real(row.y).real(row.weight).flag(row.masked);
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
            let weight = decoder.real()?;
            let masked = decoder.flag()?;
            rows.push(WeightedRow { id, x, y, weight, masked });
        }
        decoder.finish()?;
        Self::from_rows(source, ncols, rows)
    }
}
