//! Independent artifacts for one finite two-step temporal transport sequence.
//!
//! Format version 1. The artifact records the unrolled selection diagram, the
//! coordinates' places in time, the horizon and the ordered action sequence, the
//! limits of the one shared search budget, the evidence catalog, the supplied
//! exact laws, and the full report: the checked derivation, the point response,
//! the history/horizon-local support, every invariance used per time slice and
//! the time-varying confounders. A consumer trusts nothing: it rebuilds the
//! specification, re-decides the whole sequence under the producer's recorded
//! limits (none may exceed its own), recompiles and re-evaluates, and accepts
//! only a report identical to the stored one, bit for bit.
//!
//! The scientific identity binds the graph, selections, slots, variable names,
//! horizon, action sequence, question, budgets and evaluation limits. The laws
//! are data identity: a same-window evidence refresh replaces them under the same
//! identity, while a changed horizon or sequence changes the identity. A separate
//! data digest binds the catalog's and every law's snapshot identities, so a
//! consumer detects a swapped catalog or law snapshot even when the replayed
//! point happens not to change.

use crate::{
    IoError, admg_from_wire, admg_to_wire, exact_law_wire::ExactLawWire,
    mz_transport_artifact::MzTransportPointWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    transport_scenario_artifact::ScenarioCoordinateWire, wire::AdmgWire,
};
use antecedent_core::{ExecutionContext, IdentityDomain, SearchLimits, Value, VariableId};
use antecedent_estimate::temporal_transport::{
    PreparedTemporalSequence, TEMPORAL_EXACT_PROVIDER, TemporalSequenceReport,
    prepare_temporal_sequence,
};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData, LawOrigin};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    SidLimits,
    sid::SidDerivationRecord,
    sid::temporal_sequence::{
        TemporalOutcome, TemporalRefusal, TemporalSequenceError, TemporalSequenceSpec,
        TemporalSlots, decide_temporal_transport_sequence,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const TEMPORAL_TRANSPORT_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const TEMPORAL_TRANSPORT_ARTIFACT_FEATURE: &str = "temporal_transport_sequence_v1";

/// Why a temporal sequence artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TemporalTransportArtifactError {
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A recorded limit or stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The recomputed report differs from the stored report.
    #[error("temporal sequence report does not replay")]
    ReportMismatch,
    /// The stored catalog or laws disagree with the recorded data digest.
    #[error("temporal sequence data identity mismatch")]
    DataIdentityMismatch,
    /// The report was not produced by the prepared sequence it is exported with.
    #[error("the report was not produced by this prepared sequence")]
    ForeignReport,
    /// The stored laws disagree with the recorded provider.
    #[error("temporal sequence provider mismatch: {0}")]
    ProviderMismatch(&'static str),
}

/// A refused sequence, window or bound as a reason-coded refusal.
#[must_use]
#[allow(clippy::needless_pass_by_value)] // A `map_err` adapter.
#[doc(hidden)]
pub fn temporal_refusal(refusal: TemporalRefusal) -> IoError {
    IoError::Refused {
        code: refusal.code,
        message: format!("{}: {}", refusal.detail, refusal.message),
    }
}

/// A decision failure as an IO error: a refusal keeps its code.
#[must_use]
#[doc(hidden)]
pub fn temporal_decision_error(error: TemporalSequenceError) -> IoError {
    match error {
        TemporalSequenceError::Refused(refusal) => temporal_refusal(refusal),
        TemporalSequenceError::Identification(inner) => inner.into(),
    }
}

/// Consumer bounds. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct TemporalTransportConsumeLimits {
    /// Largest shared search budget (operations and depth) the consumer will
    /// replay. A producer memory limit above the consumer context's hard limit
    /// is refused too.
    pub budget: SearchLimits,
    /// Largest exact-evaluation limits the consumer will replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
}

impl Default for TemporalTransportConsumeLimits {
    fn default() -> Self {
        let sid = SidLimits::default();
        Self {
            budget: SearchLimits { operations: sid.steps, depth: sid.depth },
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
        }
    }
}

/// Where each coordinate sits in time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TemporalSlotsWire {
    /// Baseline covariates.
    pub baseline: Vec<u32>,
    /// Covariates observed before each action.
    pub covariates: [Vec<u32>; 2],
    /// The action of each step.
    pub actions: [u32; 2],
    /// The outcome.
    pub outcome: u32,
}

/// One history's support at one step.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HistoryRowWire {
    /// Step: 1 for an initial state, 2 for a complete history.
    pub step: u8,
    /// Covariate values.
    pub history: Vec<ValueWire>,
    /// Target mass of the history, when a target law covered it.
    pub target_mass: Option<f64>,
    /// `supported`, `unreached` or `outside_support`.
    pub status: String,
}

/// One invariance assumed at a time slice.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InvarianceWire {
    /// Time slice.
    pub slice: u8,
    /// Coordinate.
    pub variable: u32,
    /// `invariant` or `differs_by_selection`.
    pub assumption: String,
    /// Whether the derived formula reads a source factor for it.
    pub borrowed_from_source: bool,
}

/// One catalog regime at one time slice.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SliceEvidenceWire {
    /// Time slice.
    pub slice: u8,
    /// Population.
    pub population: String,
    /// Regime.
    pub regime: u32,
    /// Interventions in this slice.
    pub interventions: Vec<u32>,
    /// Measured coordinates in this slice.
    pub measured: Vec<u32>,
    /// Whether the checked derivation cites this regime.
    pub cited_by_derivation: bool,
}

/// The stored report.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalReportWire {
    /// Horizon.
    pub horizon: usize,
    /// The ordered action sequence.
    pub sequence: Vec<ValueWire>,
    /// Checked derivation of the whole sequence.
    pub proof: SidDerivationRecord,
    /// Point response of the outcome.
    pub point: MzTransportPointWire,
    /// Mean of the outcome.
    pub mean: f64,
    /// Coordinates of a step-1 support row.
    pub initial_coordinates: Vec<u32>,
    /// Coordinates of a step-2 support row.
    pub history_coordinates: Vec<u32>,
    /// Whether a target law covered the reachability masses.
    pub target_law_used: bool,
    /// History/horizon-local support.
    pub rows: Vec<HistoryRowWire>,
    /// Invariances per time slice.
    pub invariances: Vec<InvarianceWire>,
    /// Time-varying confounders of the second action.
    pub time_varying_confounders: Vec<u32>,
    /// Evidence per time slice.
    pub evidence: Vec<SliceEvidenceWire>,
    /// Always `point_only`.
    pub inference_claim: String,
}

impl TemporalReportWire {
    /// Encode a report together with the decision's proof record.
    ///
    /// # Errors
    /// A decision that did not identify the sequence has no report.
    pub fn from_report(
        prepared: &PreparedTemporalSequence,
        report: &TemporalSequenceReport,
    ) -> Result<Self, IoError> {
        let TemporalOutcome::Identified(bound) = &prepared.decision().outcome else {
            return Err(IoError::Convert("only an identified sequence has a report".into()));
        };
        let raw = |ids: &[VariableId]| ids.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Ok(Self {
            horizon: report.horizon,
            sequence: report.sequence.iter().map(ValueWire::from_value).collect(),
            proof: bound.derivation().to_record(),
            point: MzTransportPointWire::from_distribution(&report.distribution),
            mean: report.mean,
            initial_coordinates: raw(&report.support.initial_coordinates),
            history_coordinates: raw(&report.support.history_coordinates),
            target_law_used: report.support.target_law_used,
            rows: report
                .support
                .rows
                .iter()
                .map(|r| HistoryRowWire {
                    step: r.step,
                    history: r.history.iter().map(ValueWire::from_value).collect(),
                    target_mass: r.target_mass,
                    status: r.status.into(),
                })
                .collect(),
            invariances: report
                .invariances
                .iter()
                .map(|i| InvarianceWire {
                    slice: i.slice,
                    variable: i.variable.raw(),
                    assumption: i.assumption.name().into(),
                    borrowed_from_source: i.borrowed_from_source,
                })
                .collect(),
            time_varying_confounders: raw(&report.time_varying_confounders),
            evidence: report
                .evidence
                .iter()
                .map(|e| SliceEvidenceWire {
                    slice: e.slice,
                    population: e.population.to_string(),
                    regime: e.regime.raw(),
                    interventions: raw(&e.interventions),
                    measured: raw(&e.measured),
                    cited_by_derivation: e.cited_by_derivation,
                })
                .collect(),
            inference_claim: report.inference_claim.into(),
        })
    }
}

/// Versioned temporal sequence execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalSequenceArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Horizon (two).
    pub horizon: usize,
    /// The unrolled causal graph.
    pub graph: AdmgWire,
    /// Selection targets: time-indexed mechanism differences.
    pub selections: Vec<u32>,
    /// Where each coordinate sits in time.
    pub slots: TemporalSlotsWire,
    /// Named coordinates, in variable order.
    pub coordinates: Vec<ScenarioCoordinateWire>,
    /// Source population.
    pub source: String,
    /// Target population.
    pub target: String,
    /// The ordered action sequence.
    pub sequence: Vec<ValueWire>,
    /// Operation limit of the shared search budget.
    pub search_operations: usize,
    /// Depth limit of the shared search budget.
    pub search_depth: usize,
    /// Hard memory limit the producer decided under.
    pub search_memory_bytes: Option<u64>,
    /// Evidence catalog.
    pub catalog: EvidenceCatalogWire,
    /// Provider identity.
    pub provider: String,
    /// Laws in canonical order.
    pub laws: Vec<ExactLawWire>,
    /// Support budget.
    pub max_support_rows: usize,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// The full report.
    pub report: TemporalReportWire,
    /// Digest of the graph, selections, slots, names, question, sequence,
    /// horizon and limits: its scientific identity.
    pub premises_digest: String,
    /// Digest of the data identity: every catalog binding's snapshot and dataset
    /// ids and every law's snapshot (refresh-replaceable, outside the premises).
    pub data_digest: String,
}

/// `(regime, snapshot, dataset)` of every catalog binding and
/// `(population, regime, snapshot)` of every law, sorted.
fn data_digest(
    catalog: &antecedent_core::EvidenceCatalog,
    laws: &[ExactLawWire],
) -> Result<String, IoError> {
    let mut bindings = catalog
        .bindings
        .iter()
        .map(|b| {
            (
                b.regime.raw(),
                b.snapshot_identity.to_string(),
                b.dataset_identity.as_deref().map(str::to_owned),
            )
        })
        .collect::<Vec<_>>();
    bindings.sort();
    let mut snapshots = laws
        .iter()
        .map(|law| (law.population.clone(), law.regime, law.snapshot.clone()))
        .collect::<Vec<_>>();
    snapshots.sort();
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("temporal_transport_data_v1", bindings, snapshots),
    )?
    .to_hex())
}

/// Scientific identity of a temporal sequence execution: the unrolled graph and
/// selections, the slots, the named coordinates, the question, the horizon, the
/// ordered action sequence, and every budget the report depends on. The laws are
/// data identity and are not part of it.
///
/// # Errors
/// Encoding failure.
#[doc(hidden)]
pub fn temporal_sequence_identity(wire: &TemporalSequenceArtifactWire) -> Result<String, IoError> {
    let mut graph = wire.graph.clone();
    graph.directed.sort_unstable();
    graph.bidirected.sort_unstable();
    let mut selections = wire.selections.clone();
    selections.sort_unstable();
    let mut coordinates = wire.coordinates.clone();
    coordinates.sort_by_key(|c| c.variable);
    let mut slots = wire.slots.clone();
    slots.baseline.sort_unstable();
    slots.covariates.iter_mut().for_each(|c| c.sort_unstable());
    let budget = (
        (wire.search_operations, wire.search_depth, wire.search_memory_bytes),
        (wire.operation_limit, wire.depth_limit, wire.max_support_rows),
    );
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &(
            "temporal_transport_sequence_v1",
            wire.horizon,
            graph,
            selections,
            slots,
            coordinates,
            (&wire.source, &wire.target),
            &wire.sequence,
            budget,
        ),
    )?
    .to_hex())
}

/// A supplied exact provider carries no fitted tables.
fn check_provider(
    provider: &str,
    data: &ExactTransportData,
) -> Result<(), TemporalTransportArtifactError> {
    let mismatch = TemporalTransportArtifactError::ProviderMismatch;
    if provider != TEMPORAL_EXACT_PROVIDER {
        return Err(mismatch("unknown provider"));
    }
    if data.laws().iter().any(|law| law.origin() != LawOrigin::SuppliedExact) {
        return Err(mismatch("supplied exact laws carry no fitted tables"));
    }
    Ok(())
}

fn canonical_laws(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = data
        .laws()
        .iter()
        .map(|law| {
            let wire = ExactLawWire::from_law(law);
            crate::to_cbor(&(&wire.population, wire.regime, &wire.interventions))
                .map(|key| (key, wire))
        })
        .collect::<Result<Vec<_>, _>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

fn slots_wire(slots: &TemporalSlots) -> TemporalSlotsWire {
    let raw = |ids: &[VariableId]| ids.iter().map(|v| v.raw()).collect::<Vec<_>>();
    TemporalSlotsWire {
        baseline: raw(&slots.baseline),
        covariates: [raw(&slots.covariates[0]), raw(&slots.covariates[1])],
        actions: [slots.actions[0].raw(), slots.actions[1].raw()],
        outcome: slots.outcome.raw(),
    }
}

impl TemporalSequenceArtifactWire {
    /// Build an artifact from a prepared sequence and the report it produced.
    /// The limits recorded are the ones the decision ran under.
    ///
    /// # Errors
    /// The premises do not encode, or the decision identified nothing.
    pub fn checked(
        prepared: &PreparedTemporalSequence,
        catalog: &antecedent_core::EvidenceCatalog,
        report: &TemporalSequenceReport,
    ) -> Result<Self, IoError> {
        if !report.is_from(prepared) {
            return Err(TemporalTransportArtifactError::ForeignReport.into());
        }
        let decision = prepared.decision();
        let limits = prepared.limits();
        let laws = canonical_laws(prepared.data())?;
        let mut wire = Self {
            version: TEMPORAL_TRANSPORT_ARTIFACT_VERSION,
            required_features: vec![TEMPORAL_TRANSPORT_ARTIFACT_FEATURE.into()],
            horizon: decision.spec.horizon(),
            graph: admg_to_wire(decision.spec.diagram().causal_graph())?,
            selections: decision
                .spec
                .diagram()
                .selection_targets()
                .iter()
                .map(|v| v.raw())
                .collect(),
            slots: slots_wire(decision.spec.slots()),
            coordinates: decision
                .spec
                .schema()
                .iter()
                .map(ScenarioCoordinateWire::from_coordinate)
                .collect(),
            source: decision.query.source.to_string(),
            target: decision.query.target.to_string(),
            sequence: decision.sequence.iter().map(ValueWire::from_value).collect(),
            search_operations: decision.limits.budget.operations,
            search_depth: decision.limits.budget.depth,
            search_memory_bytes: decision.limits.memory_limit_bytes,
            catalog: EvidenceCatalogWire::from_catalog(catalog),
            provider: prepared.provider().into(),
            data_digest: data_digest(catalog, &laws)?,
            laws,
            max_support_rows: prepared.data().max_support_rows(),
            operation_limit: limits.operations,
            depth_limit: limits.depth,
            report: TemporalReportWire::from_report(prepared, report)?,
            premises_digest: String::new(),
        };
        wire.premises_digest = temporal_sequence_identity(&wire)?;
        Ok(wire)
    }

    /// The data-identity digest these stored catalog bindings and law
    /// snapshots would carry. A consumer never trusts it: re-sealing mutated
    /// data with it still fails replay.
    ///
    /// # Errors
    /// The catalog does not decode, or an encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        data_digest(&self.catalog.to_catalog()?, &self.laws)
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version first.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], a decoding failure, or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != TEMPORAL_TRANSPORT_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [TEMPORAL_TRANSPORT_ARTIFACT_FEATURE] {
            return Err(
                TemporalTransportArtifactError::UnsupportedSemantics("required features").into()
            );
        }
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &TemporalTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(), TemporalTransportArtifactError> {
        let exceeded = TemporalTransportArtifactError::LimitsExceeded;
        if self.search_operations > limits.budget.operations
            || self.search_depth > limits.budget.depth
        {
            return Err(exceeded("search budget"));
        }
        if let (Some(producer), Some(consumer)) =
            (self.search_memory_bytes, ctx.memory.hard_limit_bytes)
        {
            if producer > consumer {
                return Err(exceeded("search memory"));
            }
        }
        if self.operation_limit > limits.evaluation.operations
            || self.depth_limit > limits.evaluation.depth
        {
            return Err(exceeded("evaluation limits"));
        }
        if self.max_support_rows > limits.max_support_rows {
            return Err(exceeded("support rows"));
        }
        if self.laws.len() > limits.max_laws {
            return Err(exceeded("law count"));
        }
        Ok(())
    }

    fn spec(&self) -> Result<TemporalSequenceSpec, IoError> {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        let coordinates = self
            .coordinates
            .iter()
            .map(ScenarioCoordinateWire::to_coordinate)
            .collect::<Result<Vec<_>, _>>()?;
        let diagram = SelectionDiagram::try_new(
            admg_from_wire(&self.graph)?,
            Arc::<[VariableId]>::from(ids(&self.selections)),
        )
        .map_err(|e| IoError::Convert(e.to_string()))?;
        let slots = TemporalSlots {
            baseline: ids(&self.slots.baseline),
            covariates: [ids(&self.slots.covariates[0]), ids(&self.slots.covariates[1])],
            actions: [
                VariableId::from_raw(self.slots.actions[0]),
                VariableId::from_raw(self.slots.actions[1]),
            ],
            outcome: VariableId::from_raw(self.slots.outcome),
        };
        TemporalSequenceSpec::try_new(self.horizon, slots, diagram, coordinates)
            .map_err(temporal_refusal)
    }

    /// Decode and replay everything, accepting only an identical report.
    ///
    /// # Errors
    /// A limit, digest, reconstruction, decision or report mismatch.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: TemporalTransportConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, TemporalSequenceReport), IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits, ctx)?;
        if temporal_sequence_identity(&wire)? != wire.premises_digest {
            return Err(TemporalTransportArtifactError::PremisesMismatch.into());
        }
        if data_digest(&wire.catalog.to_catalog()?, &wire.laws)? != wire.data_digest {
            return Err(TemporalTransportArtifactError::DataIdentityMismatch.into());
        }
        let spec = wire.spec()?;
        let sequence = wire.sequence.iter().map(ValueWire::to_value).collect::<Vec<Value>>();
        let catalog = wire.catalog.to_catalog()?;
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let data = ExactTransportData::try_new(laws, wire.max_support_rows)
            .map_err(|e| IoError::Convert(e.to_string()))?;
        check_provider(&wire.provider, &data)?;
        // Replay under the producer's memory limit, which the consumer's own
        // hard limit already bounds, so a memory truncation reproduces.
        let mut replay = ctx.clone();
        if wire.search_memory_bytes.is_some() {
            replay.memory.hard_limit_bytes = wire.search_memory_bytes;
        }
        let ctx = &replay;
        let decision = decide_temporal_transport_sequence(
            &spec,
            &sequence,
            &wire.source,
            &wire.target,
            &catalog,
            SearchLimits { operations: wire.search_operations, depth: wire.search_depth },
            ctx,
        )
        .map_err(temporal_decision_error)?;
        let prepared = prepare_temporal_sequence(
            decision,
            data,
            ExactEvaluationLimits { operations: wire.operation_limit, depth: wire.depth_limit },
            ctx,
        )?;
        let report = prepared.evaluate(ctx)?;
        let replayed = TemporalReportWire::from_report(&prepared, &report)?;
        if crate::to_cbor(&replayed)? != crate::to_cbor(&wire.report)? {
            return Err(TemporalTransportArtifactError::ReportMismatch.into());
        }
        Ok((wire, report))
    }
}
