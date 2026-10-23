//! Prepared exact execution of one finite two-step temporal transport sequence.
//!
//! Preparation decides the whole sequence once, as one longitudinal intervention
//! on the explicit two-slice unrolled selection diagram, under one shared search
//! budget, and compiles it against supplied exact laws; estimation evaluates the
//! retained plan and never re-identifies. The claim is exact and point-only:
//! temporal sampling intervals, initial-state uncertainty and new-period refresh
//! are not part of this route, so an interval request refuses. A same-window evidence refresh
//! re-estimates under the same proof; a changed horizon, sequence or window is a
//! new preparation. The public horizon route reads a history-indexed law on the
//! scientific target. Unit histories and source experiments do not supply it.

use std::collections::BTreeMap;

use super::StudyBuilder;
use super::transport_common::{EvidenceSearch, estimate_err};
use antecedent_core::scientific_target::{HistoryIndexedTargetLaw, ScientificTarget};
use antecedent_core::{EvidenceCatalog, ExecutionContext, Value, VariableId, reason_code};
use antecedent_estimate::temporal_transport::{
    PreparedTemporalSequence, TemporalSequenceReport, prepare_temporal_sequence,
};
use antecedent_expr::{ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData, LawOrigin};
use antecedent_identify::sid::temporal_sequence::{
    TemporalRefusal, TemporalSequenceSpec, decide_temporal_transport_sequence,
};
use antecedent_io::IoError;
use antecedent_io::temporal_transport_artifact::{
    TemporalSequenceArtifactWire, TemporalTransportConsumeLimits, temporal_decision_error,
    temporal_refusal,
};

const TARGET_LAW_MISSING: &str = "temporal_transport.target_law_missing";
const TARGET_LAW_MISMATCH: &str = "temporal_transport.target_law_mismatch";

/// A decided, compiled two-step sequence.
#[derive(Clone, Debug)]
pub struct PreparedTemporalTransport {
    inner: PreparedTemporalSequence,
    catalog: EvidenceCatalog,
}

/// The intervention sequence and the source and target populations it is transported between.
#[derive(Clone, Copy)]
pub struct SequenceEndpoints<'a> {
    /// Interventions applied at each step.
    pub sequence: &'a [Value],
    /// Source population.
    pub source: &'a str,
    /// Target population.
    pub target: &'a str,
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
    pub fn temporal_transport_sequence(
        spec: &TemporalSequenceSpec,
        endpoints: SequenceEndpoints<'_>,
        search: EvidenceSearch,
        data: ExactTransportData,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTemporalTransport, IoError> {
        let SequenceEndpoints { sequence, source, target } = endpoints;
        let EvidenceSearch { catalog, budget } = search;
        let decision = decide_temporal_transport_sequence(
            spec, sequence, source, target, &catalog, budget, ctx,
        )
        .map_err(temporal_decision_error)?;
        let inner = prepare_temporal_sequence(decision, data, limits, ctx).map_err(estimate_err)?;
        Ok(PreparedTemporalTransport { inner, catalog })
    }

    /// Decide the licensed two-step horizon against a scientific target that
    /// carries the observational law over history-indexed experiments.
    ///
    /// The law is read from `target` and checked against the observational
    /// marginal of `data` on the law's coordinates. A missing law, a law that
    /// is only a unit-history frequency, and a horizon other than two refuse.
    /// Compilation of the sequence is the existing transport preparation.
    ///
    /// # Errors
    /// The horizon is not two, the target has no shared horizon of two, the
    /// target has no history-indexed law, the data has no supplied observational
    /// law of the target population on those coordinates, that law disagrees
    /// with the target, or [`Self::temporal_transport_sequence`] refuses.
    #[allow(
        clippy::too_many_arguments,
        reason = "the horizon route adds the scientific target to the sequence arguments"
    )]
    pub fn temporal_transport_sequence_for_target(
        spec: &TemporalSequenceSpec,
        endpoints: SequenceEndpoints<'_>,
        search: EvidenceSearch,
        data: ExactTransportData,
        target: &ScientificTarget,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTemporalTransport, IoError> {
        if spec.horizon() != 2 {
            return Err(temporal_refusal(TemporalRefusal::horizon(format!(
                "horizon {} is outside the licensed horizon of 2",
                spec.horizon()
            ))));
        }
        match target.shared_temporal_horizon() {
            Some(2) => {}
            Some(horizon) => {
                return Err(temporal_refusal(TemporalRefusal::horizon(format!(
                    "horizon {horizon} is outside the licensed horizon of 2"
                ))));
            }
            None => {
                return Err(temporal_refusal(TemporalRefusal::horizon(
                    "the scientific target has no shared two-step horizon".to_owned(),
                )));
            }
        }
        let declared = target.history_indexed_law().ok_or_else(missing_target_law)?;
        let derived =
            Self::history_indexed_target_law(&data, endpoints.target, declared.coordinates())?;
        if derived != *declared {
            return Err(law_refusal(
                TARGET_LAW_MISMATCH,
                "the history-indexed law is not the observational marginal of the target laws",
            ));
        }
        Self::temporal_transport_sequence(spec, endpoints, search, data, limits, ctx)
    }

    /// Marginal of the target population's supplied observational law onto `coordinates`.
    ///
    /// An empirical joint, including one fit from unit histories, is not this law.
    ///
    /// # Errors
    /// No single supplied observational law of `population` covers the coordinates,
    /// or that marginal is not a finite probability law.
    #[must_use = "a missing observational law is a refusal, not an empty law"]
    pub fn history_indexed_target_law(
        data: &ExactTransportData,
        population: &str,
        coordinates: &[VariableId],
    ) -> Result<HistoryIndexedTargetLaw, IoError> {
        let law = observational_law(data, population, coordinates)?;
        marginal_history_law(law, coordinates)
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
    /// uncertainty are not part of this route. This always refuses with
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

fn missing_target_law() -> IoError {
    law_refusal(
        TARGET_LAW_MISSING,
        "unit histories do not supply the target law over history-indexed experiments",
    )
}

fn law_refusal(detail: &'static str, message: &str) -> IoError {
    temporal_refusal(TemporalRefusal::new(
        reason_code!("transport_missing_evidence"),
        detail,
        message.to_owned(),
    ))
}

fn observational_law<'a>(
    data: &'a ExactTransportData,
    population: &str,
    coordinates: &[VariableId],
) -> Result<&'a ExactDiscreteLaw, IoError> {
    let mut found = None;
    for law in data.laws() {
        if law.population() != population
            || !law.interventions().is_empty()
            || law.origin() != LawOrigin::SuppliedExact
        {
            continue;
        }
        let covers = coordinates
            .iter()
            .all(|coordinate| law.axes().iter().any(|axis| axis.variable == *coordinate));
        if !covers {
            continue;
        }
        if found.is_some() {
            return Err(law_refusal(
                TARGET_LAW_MISMATCH,
                "more than one supplied observational law covers the history-indexed experiments",
            ));
        }
        found = Some(law);
    }
    found.ok_or_else(missing_target_law)
}

fn marginal_history_law(
    law: &ExactDiscreteLaw,
    coordinates: &[VariableId],
) -> Result<HistoryIndexedTargetLaw, IoError> {
    let axes = law.axes();
    let mut strides = vec![1_usize; axes.len()];
    let mut next = 1_usize;
    for index in (0..axes.len()).rev() {
        strides[index] = next;
        next = next.checked_mul(axes[index].values.len()).ok_or_else(|| {
            law_refusal(TARGET_LAW_MISMATCH, "the observational law is too wide to marginalize")
        })?;
    }
    let mut axis_of = Vec::with_capacity(coordinates.len());
    for coordinate in coordinates {
        let position =
            axes.iter().position(|axis| axis.variable == *coordinate).ok_or_else(|| {
                law_refusal(
                    TARGET_LAW_MISSING,
                    "unit histories do not supply the target law over history-indexed experiments",
                )
            })?;
        axis_of.push(position);
    }
    let mut merged: BTreeMap<Vec<u32>, f64> = BTreeMap::new();
    for (cell, mass) in law.probabilities().iter().copied().enumerate() {
        let mut remainder = cell;
        let mut levels = Vec::with_capacity(axes.len());
        for stride in &strides {
            if *stride == 0 {
                return Err(law_refusal(
                    TARGET_LAW_MISMATCH,
                    "the observational law is too wide to marginalize",
                ));
            }
            levels.push(remainder / *stride);
            remainder %= *stride;
        }
        let mut projected = Vec::with_capacity(axis_of.len());
        for axis in &axis_of {
            let level = u32::try_from(levels[*axis]).map_err(|_| {
                law_refusal(
                    TARGET_LAW_MISMATCH,
                    "a history level index does not fit in a coordinate word",
                )
            })?;
            projected.push(level);
        }
        let entry = merged.entry(projected).or_insert(0.0);
        *entry += mass;
    }
    HistoryIndexedTargetLaw::new(law.snapshot_identity(), coordinates.iter().copied(), merged)
        .map_err(|error| law_refusal(TARGET_LAW_MISMATCH, &error.to_string()))
}
