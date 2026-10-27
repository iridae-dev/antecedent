//! Retention of the replicates a bootstrap produced.
//!
//! [`RetainedBootstrap`] keeps the replicate estimates and a running count, mean and sum of
//! squares of them. The standard error and the replicate accounting come from the running
//! summary, so a later batch of replicates extends the result in place under any retention
//! mode. A percentile interval needs the replicates themselves, so a summary-only result
//! ([`RetainedBootstrap::into_summary_only`]) refuses it with the next attempt.
//!
//! A disk-backed result is [`SpilledBootstrap`]: [`RetainedBootstrap::spill`] writes the
//! replicates through a [`SpillRegistry`] after admitting the frame against the operation's
//! admission pool, and [`SpilledBootstrap::reload`] admits the same bytes again before it
//! reads the artifact back.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{BootstrapAccounting, ReplicatePolicy};
use antecedent_state::disk_retention::{SpillDecoder, SpillEncoder};
use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_state::{SpillHandle, SpillRegistry, Spillable, StateError, merge_within_tolerance};

use crate::error::CausalError;
use crate::retained_refresh::{reload_admitted, spill_admitted};

/// The replicate estimates of one bootstrap and the summary they stand for.
#[derive(Clone, Debug)]
pub struct RetainedBootstrap {
    estimates: Vec<f64>,
    failed: u32,
    count: u64,
    mean: f64,
    m2: f64,
    retention: RetentionMode,
}

fn invalid(message: String) -> CausalError {
    CausalError::State(StateError::InvalidEvent(message))
}

impl RetainedBootstrap {
    /// A result of `estimates` (one per successful replicate) and `failed` replicates that did
    /// not produce an estimate.
    ///
    /// # Errors
    /// An estimate is not finite: a replicate without an estimate is counted in `failed`.
    pub fn from_replicates(mut estimates: Vec<f64>, failed: u32) -> Result<Self, CausalError> {
        let mut result = Self {
            estimates: Vec::new(),
            failed: 0,
            count: 0,
            mean: 0.0,
            m2: 0.0,
            retention: RetentionMode::SummaryOnly,
        };
        result.append(&estimates, failed)?;
        // Reuse the caller's owned buffer after the shared validation/moment pass.
        // Compact unused capacity rather than retaining an arbitrarily padded input.
        estimates.shrink_to_fit();
        result.estimates = estimates;
        result.retention = RetentionMode::FullReusable;
        Ok(result)
    }

    /// Retention mode.
    #[must_use]
    pub const fn retention(&self) -> RetentionMode {
        self.retention
    }

    /// The replicate estimates. Empty under summary-only retention.
    #[must_use]
    pub fn estimates(&self) -> &[f64] {
        &self.estimates
    }

    /// Stop retaining the replicates. The standard error and the accounting stay.
    #[must_use]
    pub fn into_summary_only(mut self) -> Self {
        self.estimates.clear();
        self.estimates.shrink_to_fit();
        self.retention = RetentionMode::SummaryOnly;
        self
    }

    /// Add a batch of replicates, in place and under any retention mode.
    ///
    /// # Errors
    /// An estimate is not finite. Nothing changes.
    pub fn append(&mut self, estimates: &[f64], failed: u32) -> Result<(), CausalError> {
        if let Some(bad) = estimates.iter().find(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "replicate estimate {bad} is not finite; count a replicate without an estimate \
                 as failed"
            )));
        }
        for value in estimates {
            self.count += 1;
            let delta = value - self.mean;
            self.mean += delta / self.count as f64;
            self.m2 += delta * (value - self.mean);
        }
        self.failed = self.failed.saturating_add(failed);
        if self.retention == RetentionMode::FullReusable {
            self.estimates.extend_from_slice(estimates);
        }
        Ok(())
    }

    /// Replicate counts and the standard error when the replicate rule admits one.
    #[must_use]
    pub fn accounting(&self) -> BootstrapAccounting {
        let ok = u32::try_from(self.count).unwrap_or(u32::MAX);
        let admitted = ReplicatePolicy::BOOTSTRAP
            .decide(ok.saturating_add(self.failed), ok, self.failed)
            .is_ok();
        let se = (admitted && self.count >= 2).then(|| (self.m2 / (self.count - 1) as f64).sqrt());
        BootstrapAccounting {
            se,
            replicates_ok: Some(ok),
            replicates_failed: Some(self.failed),
            cancelled: false,
            early_stopped: false,
        }
    }

    /// Mean of the replicate estimates, when there is one.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Equal-tailed percentile interval at `level`, by linear interpolation between order
    /// statistics.
    ///
    /// # Errors
    /// The replicates are not retained (the next attempt is to rerun the bootstrap, or to keep
    /// full reusable retention), `level` is outside `(0, 1)`, or fewer than two replicates
    /// survived.
    pub fn percentile_interval(&self, level: f64) -> Result<(f64, f64), CausalError> {
        self.retention.require(RetainedOperation::RowReconstruction).map_err(|refusal| {
            invalid(format!(
                "{refusal}; rerun the bootstrap, or keep full reusable retention of its replicates"
            ))
        })?;
        if !(level > 0.0 && level < 1.0) {
            return Err(invalid(format!("level {level} is not in (0, 1)")));
        }
        if self.estimates.len() < 2 {
            return Err(invalid("a percentile interval needs at least two replicates".into()));
        }
        let mut sorted = self.estimates.clone();
        sorted.sort_by(f64::total_cmp);
        let tail = (1.0 - level) / 2.0;
        Ok((quantile(&sorted, tail), quantile(&sorted, 1.0 - tail)))
    }

    /// Compare the running summary with one rebuilt from the retained replicates: count
    /// exactly, mean and sum of squares within the merge tolerance.
    ///
    /// # Errors
    /// The replicates are not retained, or the summary disagrees.
    pub fn check_against_fresh(&self) -> Result<(), CausalError> {
        self.retention.require(RetainedOperation::RefitFromRetainedRows).map_err(|refusal| {
            invalid(format!("{refusal}; rerun the bootstrap to compare against its replicates"))
        })?;
        let fresh = Self::from_replicates(self.estimates.clone(), self.failed)?;
        if fresh.count != self.count
            || !merge_within_tolerance(fresh.mean, self.mean)
            || !merge_within_tolerance(fresh.m2, self.m2)
        {
            return Err(CausalError::State(StateError::StaleCache(
                "cached bootstrap summary differs from the fresh rebuild".into(),
            )));
        }
        Ok(())
    }

    /// Upper bound of the bytes the frame of this result takes: what a spill or a reload asks the
    /// admission pool for. The constant covers the encoder framing, as for the refresh family.
    #[must_use]
    pub fn frame_bytes(&self) -> u64 {
        8 * self.estimates.len() as u64 + 256
    }

    /// Write the replicates under `registry` as a disk-backed result, after admitting the frame
    /// against the operation's admission pool.
    ///
    /// # Errors
    /// The admission pool cannot hold the frame, the replicates are not retained, the frame is
    /// larger than the whole budget of `registry`, or the file cannot be written.
    pub fn spill(
        &self,
        ctx: &ExecutionContext,
        registry: &Arc<SpillRegistry>,
        key: &str,
    ) -> Result<SpilledBootstrap, CausalError> {
        let handle = spill_admitted(ctx, registry, key, self, self.frame_bytes())?;
        Ok(SpilledBootstrap { handle })
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the position is between zero and the replicate count"
)]
fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let below = position.floor();
    let fraction = position - below;
    let index = below as usize;
    match sorted.get(index + 1) {
        Some(above) => sorted[index] + fraction * (above - sorted[index]),
        None => sorted[index],
    }
}

impl Spillable for RetainedBootstrap {
    const KIND: &'static str = "antecedent.facade.bootstrap_replicates";

    fn encode(&self) -> Result<Vec<u8>, StateError> {
        self.retention.require(RetainedOperation::RowReconstruction).map_err(|refusal| {
            StateError::InvalidEvent(format!(
                "{refusal}; rerun the bootstrap, or keep full reusable retention of its replicates"
            ))
        })?;
        let mut encoder = SpillEncoder::new();
        encoder.reals(&self.estimates).word(u64::from(self.failed));
        Ok(encoder.finish())
    }

    fn decode(bytes: &[u8]) -> Result<Self, StateError> {
        let mut decoder = SpillDecoder::new(bytes);
        let estimates = decoder.reals()?;
        let failed = u32::try_from(decoder.word()?)
            .map_err(|_| StateError::StaleCache("spilled failure count is too large".into()))?;
        decoder.finish()?;
        Self::from_replicates(estimates, failed).map_err(|error| match error {
            CausalError::State(state) => state,
            other => StateError::StaleCache(other.to_string()),
        })
    }
}

/// A bootstrap result written to disk. Its mode is [`RetentionMode::DiskBacked`]: it serves a
/// read of the artifact and refuses every resident operation with the reload that serves it.
#[derive(Clone, Debug)]
pub struct SpilledBootstrap {
    handle: SpillHandle,
}

impl SpilledBootstrap {
    /// The mode of a spilled result.
    #[must_use]
    pub const fn mode(&self) -> RetentionMode {
        RetentionMode::DiskBacked
    }

    /// The handle on the artifact.
    #[must_use]
    pub const fn handle(&self) -> &SpillHandle {
        &self.handle
    }

    /// Whether the artifact is still on disk (it is not once evicted).
    #[must_use]
    pub fn on_disk(&self) -> bool {
        self.handle.on_disk()
    }

    /// Keys evicted from the registry to make room for this spill.
    #[must_use]
    pub fn evicted(&self) -> &[String] {
        self.handle.evicted()
    }

    /// Accept `operation` on the spilled result, or refuse it with the next attempt.
    ///
    /// # Errors
    /// The disk-backed mode cannot serve `operation`.
    pub fn require(&self, operation: RetainedOperation) -> Result<(), CausalError> {
        Ok(self.handle.require(operation)?)
    }

    /// Read the artifact back as the resident result, after admitting its frame against the
    /// operation's admission pool.
    ///
    /// # Errors
    /// The admission pool cannot hold the frame, the artifact was evicted (the next attempt is
    /// to rerun the bootstrap), or the file was changed.
    pub fn reload(&self, ctx: &ExecutionContext) -> Result<RetainedBootstrap, CausalError> {
        reload_admitted(ctx, &self.handle)
    }
}
