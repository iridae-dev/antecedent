//! Stage progress clock and progressive stage-result streaming.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use causal_core::ExecutionContext;
use causal_estimate::EffectEstimate;
use causal_identify::{IdentificationResult, IdentifiedEstimand};
use causal_validate::{PredictiveCheckReport, RefutationReport};

use crate::error::CausalError;

/// Canonical progressive-execute stage ids (stable ProgressSink contract).
pub const STAGE_IDENTIFY: &str = "identify";
pub const STAGE_ESTIMATE_POINT: &str = "estimate_point";
pub const STAGE_UNCERTAINTY: &str = "uncertainty";
pub const STAGE_VALIDATE: &str = "validate";

/// Intermediate stage payload emitted before the final [`crate::CausalAnalysisResult`].
///
/// Same logical plan throughout; only sample size / uncertainty fills deepen across stages.
#[derive(Clone, Debug)]
pub enum AnalysisStageEvent {
    /// Identification fail-fast complete.
    Identify {
        /// Full identification artifact.
        identification: IdentificationResult,
        /// Primary estimand selected for estimation.
        estimand: IdentifiedEstimand,
    },
    /// Point estimate (analytic SE may be present; bootstrap / posterior fills absent).
    Point {
        /// Point estimate without bootstrap / posterior uncertainty fills.
        estimate: EffectEstimate,
    },
    /// Bootstrap / posterior uncertainty fills attached.
    Uncertainty {
        /// Estimate with SE / replicate accounting filled.
        estimate: EffectEstimate,
    },
    /// Refuters / predictive checks complete.
    Validate {
        /// Refutation reports (may be empty).
        refutations: Vec<RefutationReport>,
        /// Prior/posterior predictive checks (Bayesian; may be empty).
        predictive_checks: Vec<PredictiveCheckReport>,
    },
}

impl AnalysisStageEvent {
    /// Stable stage id matching [`STAGE_IDENTIFY`] / … constants.
    #[must_use]
    pub fn stage_id(&self) -> &'static str {
        match self {
            Self::Identify { .. } => STAGE_IDENTIFY,
            Self::Point { .. } => STAGE_ESTIMATE_POINT,
            Self::Uncertainty { .. } => STAGE_UNCERTAINTY,
            Self::Validate { .. } => STAGE_VALIDATE,
        }
    }
}

/// Optional sink for streamed intermediate stage payloads (parallel to [`causal_core::ProgressSink`]).
pub trait StageResultSink: Send + Sync {
    /// Called at each progressive stage boundary with a usable partial payload.
    fn on_stage(&self, event: &AnalysisStageEvent);
}

/// Records per-stage timings and reports progress / cancellation.
#[derive(Debug)]
pub(crate) struct StageClock {
    started: Instant,
    stage_started: Instant,
    timings: Vec<(Arc<str>, u64)>,
    cancelled: bool,
}

impl StageClock {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self { started: now, stage_started: now, timings: Vec::with_capacity(4), cancelled: false }
    }

    /// Begin a stage: report progress and check cancellation.
    ///
    /// # Errors
    ///
    /// [`CausalError::Cancelled`] when cancellation is already requested.
    pub(crate) fn begin(
        &mut self,
        ctx: &ExecutionContext,
        stage: &'static str,
        fraction: f64,
    ) -> Result<(), CausalError> {
        if ctx.cancellation.is_cancelled() {
            self.cancelled = true;
            return Err(CausalError::Cancelled { stage });
        }
        if let Some(p) = &ctx.progress {
            p.report(fraction, stage);
        }
        self.stage_started = Instant::now();
        Ok(())
    }

    /// Finish the current stage and record its wall time.
    pub(crate) fn finish(&mut self, stage: &'static str) {
        let ns = u64::try_from(self.stage_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.timings.push((Arc::from(stage), ns));
    }

    /// Mark cancellation observed after a usable partial result exists.
    pub(crate) fn mark_cancelled(&mut self) {
        self.cancelled = true;
    }

    #[must_use]
    pub(crate) fn cancelled(&self) -> bool {
        self.cancelled
    }

    #[must_use]
    pub(crate) fn timings(&self) -> Vec<(Arc<str>, u64)> {
        self.timings.clone()
    }

    #[must_use]
    pub(crate) fn wall_time_ns(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Emit a stage event when a sink is configured.
pub(crate) fn emit_stage(sink: Option<&Arc<dyn StageResultSink>>, event: &AnalysisStageEvent) {
    if let Some(s) = sink {
        s.on_stage(event);
    }
}
