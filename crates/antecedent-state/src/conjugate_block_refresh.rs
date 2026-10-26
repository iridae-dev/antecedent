//! Refresh of a conjugate posterior against the retained rows of each of its blocks.
//!
//! [`ConjugateBlockRefresh`] pairs a [`ConjugateOlsPosterior`] with one [`OlsRefresh`] per
//! labelled block, so the posterior can be checked against, and rebuilt from, the observations
//! it was built on. The contract is [`crate::route_contract`] for
//! [`crate::RefreshRoute::ConjugateGaussianPosterior`]:
//!
//! * a new block under a new label updates the posterior in place
//!   ([`RefreshOutcome::Updated`]);
//! * rows added to, evicted from, reweighted in or masked in an existing block, and a changed
//!   prior, rebuild the posterior from every retained block ([`RefreshOutcome::Recomputed`]);
//! * a block label used twice is refused: each observation enters the likelihood once;
//! * a block whose rows are not retained cannot be rebuilt, and the refusal names the attempt.
//!
//! [`ConjugateBlockRefresh::check_against_fresh`] rebuilds every block summary from its
//! retained rows, rebuilds the posterior from those summaries under the prior, and compares
//! the posterior mean, precision, scale, shape, rate and count and the identity of the prior
//! and the blocks with the cached posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::CanonicalWriter;

use crate::conjugate_refresh::{ConjugateOlsPosterior, ConjugateOlsPrior, ConjugateOlsSummary};
use crate::disk_retention::{SpillDecoder, SpillEncoder, SpillHandle, SpillRegistry, Spillable};
use crate::error::StateError;
use crate::ols_refresh::{OlsRefresh, RefreshOutcome, RefreshRow};
use crate::retention::{RetainedOperation, RetentionMode};
use crate::suff_stats::{LinearOlsSuffStats, merge_within_tolerance};

const IDENTITY_TAG: &str = "antecedent.state.conjugate_block_refresh.identity";

/// A conjugate posterior and the retained rows of every block that built it.
#[derive(Clone, Debug)]
pub struct ConjugateBlockRefresh {
    prior: ConjugateOlsPrior,
    blocks: BTreeMap<Arc<str>, OlsRefresh>,
    posterior: ConjugateOlsPosterior,
    retention: RetentionMode,
    recomputes: u64,
}

fn nonempty_blocks(blocks: &BTreeMap<Arc<str>, OlsRefresh>) -> Vec<(&str, &LinearOlsSuffStats)> {
    blocks
        .iter()
        .filter(|(_, block)| block.stats().n() > 0)
        .map(|(label, block)| (label.as_ref(), block.stats()))
        .collect()
}

fn posterior_from(
    prior: &ConjugateOlsPrior,
    blocks: &BTreeMap<Arc<str>, OlsRefresh>,
) -> Result<ConjugateOlsPosterior, StateError> {
    let mut posterior = ConjugateOlsPosterior::new(prior);
    for (label, stats) in nonempty_blocks(blocks) {
        posterior.update(prior, label, stats)?;
    }
    Ok(posterior)
}

/// The summary of a block's retained rows, built from the rows alone.
fn fresh_stats(block: &OlsRefresh) -> Result<LinearOlsSuffStats, StateError> {
    let mut stats = LinearOlsSuffStats::new(block.stats().ncols())?;
    for row in block.rows() {
        if row.masked {
            continue;
        }
        for _ in 0..row.count {
            stats.append_row(&row.x, row.y)?;
        }
    }
    Ok(stats)
}

impl ConjugateBlockRefresh {
    /// The posterior of no blocks under `prior`.
    #[must_use]
    pub fn new(prior: ConjugateOlsPrior) -> Self {
        let posterior = ConjugateOlsPosterior::new(&prior);
        Self {
            prior,
            blocks: BTreeMap::new(),
            posterior,
            retention: RetentionMode::FullReusable,
            recomputes: 0,
        }
    }

    /// The prior the posterior answers under.
    #[must_use]
    pub const fn prior(&self) -> &ConjugateOlsPrior {
        &self.prior
    }

    /// Retention mode of the blocks.
    #[must_use]
    pub const fn retention(&self) -> RetentionMode {
        self.retention
    }

    /// Stop retaining rows in every block, present and later. The posterior and the block
    /// summaries stay; every later change that needs the rows is refused with its next
    /// attempt.
    #[must_use]
    pub fn into_summary_only(mut self) -> Self {
        self.blocks = std::mem::take(&mut self.blocks)
            .into_iter()
            .map(|(label, block)| (label, block.into_summary_only()))
            .collect();
        self.retention = RetentionMode::SummaryOnly;
        self
    }

    /// How many times the posterior was rebuilt from the retained blocks.
    #[must_use]
    pub const fn recomputes(&self) -> u64 {
        self.recomputes
    }

    /// Labels of the blocks, in order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.blocks.keys().map(AsRef::as_ref)
    }

    /// The retained block `label`.
    #[must_use]
    pub fn block(&self, label: &str) -> Option<&OlsRefresh> {
        self.blocks.get(label)
    }

    fn rebuild(&mut self) -> Result<RefreshOutcome, StateError> {
        self.posterior = posterior_from(&self.prior, &self.blocks)?;
        self.recomputes = self.recomputes.saturating_add(1);
        Ok(RefreshOutcome::Recomputed)
    }

    /// Add a block of rows of `source` under a new label. Exact in place.
    ///
    /// # Errors
    /// The label was already used, the block holds no observations, a row is malformed or an
    /// id repeats. Nothing changes.
    pub fn add_block(
        &mut self,
        label: &str,
        source: &str,
        rows: Vec<RefreshRow>,
    ) -> Result<RefreshOutcome, StateError> {
        if self.blocks.contains_key(label) {
            return Err(StateError::InvalidEvent(format!(
                "block {label} was already used; extend the block, or use a new label"
            )));
        }
        let block = OlsRefresh::from_rows(source, self.prior.ncols(), rows)?;
        let block = if self.retention == RetentionMode::FullReusable {
            block
        } else {
            block.into_summary_only()
        };
        self.posterior.update(&self.prior, label, block.stats())?;
        self.blocks.insert(Arc::from(label), block);
        Ok(RefreshOutcome::Updated)
    }

    fn block_mut(&mut self, label: &str) -> Result<&mut OlsRefresh, StateError> {
        self.blocks.get_mut(label).ok_or_else(|| StateError::UnknownId(format!("block {label}")))
    }

    /// Append rows of the same source to an existing block, then rebuild the posterior: the
    /// posterior cannot take the same label twice.
    ///
    /// # Errors
    /// Unknown label, another source, a malformed row or a repeated id. Nothing changes.
    pub fn extend_block(
        &mut self,
        label: &str,
        source: &str,
        rows: &[RefreshRow],
    ) -> Result<RefreshOutcome, StateError> {
        self.block_mut(label)?.append_window(source, rows)?;
        self.rebuild()
    }

    /// Evict one retained row of a block, then rebuild the posterior.
    ///
    /// # Errors
    /// Unknown label or id, or the block does not retain rows.
    pub fn evict_row(&mut self, label: &str, id: u64) -> Result<RefreshOutcome, StateError> {
        self.block_mut(label)?.evict_row(id)?;
        self.rebuild()
    }

    /// Keep the most recently appended `max_rows` rows of a block, then rebuild the posterior.
    ///
    /// # Errors
    /// Unknown label, or the block does not retain rows.
    pub fn retain_last(
        &mut self,
        label: &str,
        max_rows: usize,
    ) -> Result<RefreshOutcome, StateError> {
        self.block_mut(label)?.retain_last(max_rows)?;
        self.rebuild()
    }

    /// Change one row's count weight in a block, then rebuild the posterior.
    ///
    /// # Errors
    /// Unknown label or id, or the block does not retain rows.
    pub fn set_count(
        &mut self,
        label: &str,
        id: u64,
        count: u32,
    ) -> Result<RefreshOutcome, StateError> {
        self.block_mut(label)?.set_count(id, count)?;
        self.rebuild()
    }

    /// Change one row's mask in a block, then rebuild the posterior.
    ///
    /// # Errors
    /// Unknown label or id, or the block does not retain rows.
    pub fn set_mask(
        &mut self,
        label: &str,
        id: u64,
        masked: bool,
    ) -> Result<RefreshOutcome, StateError> {
        self.block_mut(label)?.set_mask(id, masked)?;
        self.rebuild()
    }

    /// Replace the prior and rebuild the posterior from every retained block under it.
    ///
    /// # Errors
    /// The new prior has another column count, it already used one of the block labels, or a
    /// block cannot be rebuilt. Nothing changes.
    pub fn change_prior(&mut self, prior: ConjugateOlsPrior) -> Result<RefreshOutcome, StateError> {
        if prior.ncols() != self.prior.ncols() {
            return Err(StateError::Shape(format!(
                "prior has {} columns, the blocks have {}",
                prior.ncols(),
                self.prior.ncols()
            )));
        }
        let posterior = self.posterior.reprior(&prior, &nonempty_blocks(&self.blocks))?;
        self.posterior = posterior;
        self.prior = prior;
        self.recomputes = self.recomputes.saturating_add(1);
        Ok(RefreshOutcome::Recomputed)
    }

    /// The cached posterior summary.
    ///
    /// # Errors
    /// A singular posterior precision.
    pub fn summary(&self) -> Result<ConjugateOlsSummary, StateError> {
        self.posterior.summary(&self.prior)
    }

    /// Identity of the prior and of every block's source and rows, from the cached blocks.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut writer = CanonicalWriter::new(IDENTITY_TAG);
        writer.text(&self.prior.digest());
        for (label, block) in &self.blocks {
            writer.text(label).text(&block.snapshot().source_digest);
        }
        writer.finish_hex()
    }

    /// The identity rebuilt from the retained rows alone.
    ///
    /// # Errors
    /// A block does not retain rows.
    pub fn fresh_identity(&self) -> Result<String, StateError> {
        let mut writer = CanonicalWriter::new(IDENTITY_TAG);
        writer.text(&self.prior.digest());
        for (label, block) in &self.blocks {
            writer.text(label).text(&block.fresh_snapshot()?.source_digest);
        }
        Ok(writer.finish_hex())
    }

    /// The posterior summary rebuilt from the retained rows alone, borrowing them: no block
    /// is copied and the cached posterior is not consulted.
    ///
    /// # Errors
    /// A block does not retain rows, or the rebuilt posterior precision is singular.
    pub fn fresh_summary(&self) -> Result<ConjugateOlsSummary, StateError> {
        let mut fresh = ConjugateOlsPosterior::new(&self.prior);
        for (label, block) in &self.blocks {
            block.require_rows(RetainedOperation::RefitFromRetainedRows)?;
            let stats = fresh_stats(block)?;
            if stats.n() > 0 {
                fresh.update(&self.prior, label, &stats)?;
            }
        }
        fresh.summary(&self.prior)
    }

    /// Compare the cached posterior with one rebuilt from the retained rows: every block
    /// against its own fresh rebuild, then the posterior mean, precision, scale, shape, rate
    /// and count within [`merge_within_tolerance`], and the identity exactly.
    ///
    /// # Errors
    /// A block does not retain rows, or the first field that disagrees.
    pub fn check_against_fresh(&self) -> Result<(), StateError> {
        let mismatch = |field: &str| {
            Err(StateError::StaleCache(format!("cached {field} differs from the fresh rebuild")))
        };
        let mut fresh = ConjugateOlsPosterior::new(&self.prior);
        for (label, block) in &self.blocks {
            block.check_against_fresh()?;
            let stats = fresh_stats(block)?;
            if stats.n() > 0 {
                fresh.update(&self.prior, label, &stats)?;
            }
        }
        if self.identity() != self.fresh_identity()? {
            return mismatch("identity");
        }
        let cached = self.summary()?;
        let rebuilt = fresh.summary(&self.prior)?;
        let close = |left: &[f64], right: &[f64]| {
            left.len() == right.len()
                && left.iter().zip(right).all(|(a, b)| merge_within_tolerance(*a, *b))
        };
        if cached.n != rebuilt.n {
            return mismatch("observation count");
        }
        if !close(&cached.mean, &rebuilt.mean) {
            return mismatch("posterior mean");
        }
        if !close(&cached.precision, &rebuilt.precision) || !close(&cached.scale, &rebuilt.scale) {
            return mismatch("posterior precision");
        }
        if !merge_within_tolerance(cached.shape, rebuilt.shape)
            || !merge_within_tolerance(cached.rate, rebuilt.rate)
        {
            return mismatch("posterior shape or rate");
        }
        Ok(())
    }

    /// Write the prior and every block's retained rows under `registry`.
    ///
    /// # Errors
    /// A block does not retain rows, the prior already used blocks, or the errors of
    /// [`SpillRegistry::spill`].
    pub fn spill(
        &self,
        registry: &Arc<SpillRegistry>,
        key: &str,
    ) -> Result<SpillHandle, StateError> {
        registry.spill(key, self)
    }
}

impl Spillable for ConjugateBlockRefresh {
    const KIND: &'static str = "antecedent.state.conjugate_block_refresh";

    fn encode(&self) -> Result<Vec<u8>, StateError> {
        if self.prior.used_labels().next().is_some() {
            return Err(StateError::InvalidEvent(
                "a prior built from a posterior is not spilled; spill the blocks under the \
                 original prior"
                    .into(),
            ));
        }
        let mut encoder = SpillEncoder::new();
        encoder
            .reals(self.prior.mean())
            .reals(self.prior.precision())
            .real(self.prior.shape())
            .real(self.prior.rate())
            .word(self.blocks.len() as u64);
        for (label, block) in &self.blocks {
            block.retention().require(RetainedOperation::RowReconstruction).map_err(|_| {
                StateError::InvalidEvent(format!(
                    "block {label} does not retain rows, so it cannot be spilled"
                ))
            })?;
            encoder.text(label).text(block.source()).word(block.rows().len() as u64);
            for row in block.rows() {
                encoder
                    .word(row.id)
                    .reals(&row.x)
                    .real(row.y)
                    .word(u64::from(row.count))
                    .flag(row.masked);
            }
        }
        Ok(encoder.finish())
    }

    fn decode(bytes: &[u8]) -> Result<Self, StateError> {
        let mut decoder = SpillDecoder::new(bytes);
        let mean = decoder.reals()?;
        let precision = decoder.reals()?;
        let shape = decoder.real()?;
        let rate = decoder.real()?;
        let prior = ConjugateOlsPrior::new(mean, precision, shape, rate)?;
        let block_count = decoder.count(8)?;
        let mut blocks = BTreeMap::new();
        for _ in 0..block_count {
            let label = decoder.text()?;
            let source = decoder.text()?;
            let row_count = decoder.count(8)?;
            let mut rows = Vec::with_capacity(row_count);
            for _ in 0..row_count {
                let id = decoder.word()?;
                let x = decoder.reals()?;
                let y = decoder.real()?;
                let count = u32::try_from(decoder.word()?)
                    .map_err(|_| StateError::StaleCache("spilled count is too large".into()))?;
                let masked = decoder.flag()?;
                rows.push(RefreshRow { id, x, y, count, masked });
            }
            blocks.insert(Arc::from(label), OlsRefresh::from_rows(source, prior.ncols(), rows)?);
        }
        decoder.finish()?;
        let posterior = posterior_from(&prior, &blocks)?;
        Ok(Self { prior, blocks, posterior, retention: RetentionMode::FullReusable, recomputes: 0 })
    }
}
