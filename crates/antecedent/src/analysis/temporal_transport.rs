//! Prepared exact execution of one finite two-step temporal transport sequence.
//!
//! Preparation decides the whole sequence once, as one longitudinal intervention
//! on the explicit two-slice unrolled selection diagram, under one shared search
//! budget, and compiles it against supplied exact laws; estimation evaluates the
//! retained plan and never re-identifies. The claim is exact and point-only:
//! temporal sampling intervals, initial-state uncertainty and new-period refresh
//! are 2.3A, so an interval request refuses. A same-window evidence refresh
//! re-estimates under the same proof; a changed horizon, sequence or window is a
//! new preparation.
use super::StudyBuilder;
use super::transport_common::estimate_err;
use antecedent_core::{EvidenceCatalog, ExecutionContext, SearchLimits, Value};
use antecedent_estimate::temporal_transport::{
    PreparedTemporalSequence, TemporalSequenceReport, prepare_temporal_sequence,
};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::sid::temporal_sequence::{
    TemporalRefusal, TemporalSequenceSpec, decide_temporal_transport_sequence,
};
use antecedent_io::IoError;
use antecedent_io::temporal_transport_artifact::{
    TemporalSequenceArtifactWire, TemporalTransportConsumeLimits, temporal_decision_error,
    temporal_refusal,
};

/// A decided, compiled two-step sequence.
#[derive(Clone, Debug)]
pub struct PreparedTemporalTransport {
    inner: PreparedTemporalSequence,
    catalog: EvidenceCatalog,
}

impl StudyBuilder {
    /// Decide the whole two-step sequence once as one longitudinal intervention
    /// and compile it against supplied exact laws.
    ///
    /// `budget` is the one shared search budget: the identification search, its
    /// verification replays and the growth of the history lattice all charge it,
    /// against the context's hard memory limit and cancellation. A stop is a
    /// receipt (`temporal_transport.history_budget`), never a non-identification
    /// verdict.
    ///
    /// # Errors
    /// A refused specification, sequence or bound; a decision that identified
    /// nothing (with its own reason); a law outside the window or the declared
    /// domains; a reached history outside certified support
    /// (`temporal_transport.history_outside_support`); or provider failures.
    #[allow(clippy::too_many_arguments)] // Every premise of the preparation, explicitly.
    pub fn temporal_transport_sequence(
        spec: &TemporalSequenceSpec,
        sequence: &[Value],
        source: &str,
        target: &str,
        catalog: EvidenceCatalog,
        budget: SearchLimits,
        data: ExactTransportData,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTemporalTransport, IoError> {
        let decision = decide_temporal_transport_sequence(
            spec, sequence, source, target, &catalog, budget, ctx,
        )
        .map_err(temporal_decision_error)?;
        let inner = prepare_temporal_sequence(decision, data, limits, ctx).map_err(estimate_err)?;
        Ok(PreparedTemporalTransport { inner, catalog })
    }
}

impl PreparedTemporalTransport {
    /// Evaluate the whole sequence's history-aware functional.
    ///
    /// # Errors
    /// A numerical failure, a history outside support the evaluator surfaces, or
    /// cancellation.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<TemporalSequenceReport, IoError> {
        self.inner.evaluate(ctx).map_err(estimate_err)
    }

    /// Re-estimate against new laws of the same measurement window; the proof is
    /// kept. Laws over another window or horizon refuse with
    /// `temporal_transport.horizon`: they need a new preparation.
    ///
    /// # Errors
    /// A changed window, or laws outside the domains or support.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let inner = self.inner.refresh(data, ctx).map_err(estimate_err)?;
        Ok(Self { inner, catalog: self.catalog.clone() })
    }

    /// An interval is not licensed: temporal sampling and initial-state
    /// uncertainty are 2.3A. This always refuses with
    /// `estimator_inference_mismatch`.
    ///
    /// # Errors
    /// Always.
    pub fn interval(&self) -> Result<(), IoError> {
        Err(temporal_refusal(TemporalRefusal::interval_requested()))
    }

    /// The frozen decision, compiled plan and support.
    #[must_use]
    pub const fn prepared(&self) -> &PreparedTemporalSequence {
        &self.inner
    }

    /// Export a report with every premise needed for independent replay.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn export(&self, report: &TemporalSequenceReport) -> Result<Vec<u8>, IoError> {
        TemporalSequenceArtifactWire::checked(&self.inner, &self.catalog, report)?.export()
    }
}

/// Independently replay a temporal sequence artifact and return the recomputed report.
///
/// # Errors
/// Any limit, digest, reconstruction, decision or report mismatch.
pub fn consume_temporal_transport_artifact(
    bytes: &[u8],
    limits: TemporalTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<TemporalSequenceReport, IoError> {
    Ok(TemporalSequenceArtifactWire::consume_with_limits(bytes, limits, ctx)?.1)
}
