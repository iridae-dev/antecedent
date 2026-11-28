//! Portable learned-regime discovery through the compiled structure search.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::execution::{KernelPolicy, RngFactory};
use antecedent_core::{
    CanonicalWriter, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RegimeId, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{DataError, SamplingRegularity, TabularData, TimeIndex, TimeSeriesData};
use antecedent_io::{IoError, from_cbor, to_cbor};
use antecedent_stats::FdrAdjustment;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::analysis::composition::{
    ArtifactDescription, BundleStage, NodeFailure, NodeKind, NodeVerifier, UpstreamNode,
    VerifiedNode,
};
use crate::discovery::{
    BackgroundKnowledge, DiscoverParams, MultiDatasetConstraints, RegimeCountPolicy,
    RpcmciDiscoveryResult, RpcmciLearning,
};
use crate::discovery_defaults::resolve_ci;
use crate::support::unit::{SupportKey, UnitId, require_program};
use crate::{
    AcceptedGraph, CausalError, CompiledExecution, DiscoveredStructure, DiscoverySpec, Study,
};

const LEGACY_MAGIC: &[u8] = b"ANTE-RPCMCI-LEARNED-1\0";
const SCHEMA_MAGIC: &[u8] = b"ANTE-RPCMCI-LEARNED-2\0";
const MAX_BYTES: usize = 16 * 1024 * 1024;
const REPLAY_UNIT: UnitId = UnitId::new("discovery.learned_rpcmci_replay");
const RUN_ENTRY: &str = "antecedent.discovery.RPCMCI.run_portable";
const CONSUME_ENTRY: &str = "antecedent.rpcmci_portable.consume_learned_rpcmci";

fn require_replay_unit(
    ctx: &ExecutionContext,
    entry: &'static str,
) -> Result<(), RpcmciReplayError> {
    require_program(ctx, &SupportKey::Unit(REPLAY_UNIT), entry)
        .map_err(|refusal| CausalError::from(refusal).into())
}

/// Retained native, unweighted partial-correlation RPCMCI declaration and complete series.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedRpcmciRequest {
    /// Unique column names in retained graph order.
    pub names: Vec<String>,
    /// Complete numeric columns from one regularly spaced series.
    pub columns: Vec<Vec<f64>>,
    /// Length of one sampling interval in nanoseconds.
    pub interval_ns: u64,
    /// Candidate count, or maximum when `select_count` is true.
    pub count: u32,
    /// Choose among counts by the declared classification-BIC criterion.
    pub select_count: bool,
    /// Seeded initializations per candidate count.
    pub starts: u32,
    /// Maximum accepted alternating-update attempts per start.
    pub max_iterations: u32,
    /// Cost of switching a label.
    pub switch_penalty: f64,
    /// Optional minimum duration for a contiguous regime.
    pub min_regime_len: Option<usize>,
    /// Admitted maximum lag of each PCMCI+ child.
    pub max_lag: u32,
    /// Within-regime PCMCI+ decision threshold.
    pub alpha: f64,
    /// Apply the registered Benjamini–Hochberg adjustment.
    pub fdr: bool,
    /// Maximum conditioning-set size in each PCMCI+ child.
    pub max_cond_size: usize,
}

/// Identity, execution or scope refusal for a portable learned RPCMCI result.
#[derive(Debug, Error)]
pub enum RpcmciReplayError {
    /// Invalid or changed portable declaration.
    #[error("portable RPCMCI: {0}")]
    Invalid(&'static str),
    /// Replay did not have sufficient execution resources.
    #[error("portable RPCMCI resource: {0}")]
    Resource(String),
    /// Execution cancellation.
    #[error("portable RPCMCI was cancelled")]
    Cancelled,
    /// Codec failure.
    #[error(transparent)]
    Io(#[from] IoError),
    /// Original causal compiler or reviewed graph refused.
    #[error(transparent)]
    Causal(#[from] CausalError),
}

/// Independent learned result readout retained with the compiled discovery.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedRpcmciReport {
    /// Program identity reconstructed under the retained declaration.
    pub program_identity: String,
    /// Exact data snapshot the compiled search used.
    pub data_identity: String,
    /// The seed from which the initialization starts are derived.
    pub seed: u64,
    /// Regime of each original source observation, in source order.
    pub assignments: Vec<u32>,
    /// Per-regime graph fingerprints, ordered by the reported regime id.
    pub graphs: Vec<(u32, String)>,
    /// Explicitly selected count.
    pub selected_count: u32,
    /// Best-fit classification scores by count.
    pub candidate_scores: Vec<(u32, f64)>,
    /// Counts refused for insufficient or degenerate support.
    pub infeasible_counts: Vec<u32>,
    /// The selected start stopped before its iteration bound.
    pub converged: bool,
    /// Accepted updates for the selected start.
    pub iterations: u32,
    /// Successful initializations at the selected count.
    pub successful_starts: usize,
    /// Agreement of assignments across aligned successful starts.
    pub assignment_agreement: Option<f64>,
    /// Per-regime exact graph stability across aligned successful starts.
    pub graph_agreement: Vec<Option<f64>>,
    /// Native route diagnostics and their plain-language messages.
    pub diagnostics: Vec<(String, String)>,
}

/// Evidence scope of one graph selected from the replayed regime collection.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedRpcmciRegimeScope {
    /// Source artifact identity, binding the whole assignment and stability report.
    pub artifact_identity: String,
    /// Original time-series snapshot read by the compiled search.
    pub data_identity: String,
    /// The selected regime id in the original assignment.
    pub regime: u32,
    /// Identity of the exact regime graph.
    pub graph_identity: String,
    /// Zero-based source rows labeled with the selected regime.
    pub assigned_rows: Vec<usize>,
    /// Zero-based endpoints of complete original windows used for graph discovery.
    pub window_end_rows: Vec<usize>,
    /// Largest required lookback of each retained window, twice the declared lag.
    pub lookback: usize,
    /// Exact graph stability among successful initializations after label alignment.
    pub graph_agreement: Option<f64>,
    /// Agreement of all source assignments across successful initializations.
    pub assignment_agreement: Option<f64>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    request: LearnedRpcmciRequest,
    report: LearnedRpcmciReport,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum SourceSchema {
    ContinuousContextV1,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SchemaWire<T> {
    source_schema: SourceSchema,
    result: T,
}

/// Replayed discovery with the raw evidence needed by the independent reader.
pub struct LearnedRpcmciArtifact {
    result: RpcmciDiscoveryResult,
    wire: Wire,
    source_schema: Option<SourceSchema>,
    identity: String,
    plan: crate::planner::PlanView,
}

impl LearnedRpcmciRequest {
    fn validate(&self) -> Result<(), RpcmciReplayError> {
        let width = self.names.len();
        let rows = self.columns.first().map_or(0, Vec::len);
        if width == 0
            || self.names.iter().any(|name| name.trim().is_empty() || name.len() > 256)
            || self.names.iter().collect::<BTreeSet<_>>().len() != width
            || self.columns.len() != width
            || rows < 10
            || self
                .columns
                .iter()
                .any(|column| column.len() != rows || column.iter().any(|x| !x.is_finite()))
            || self.interval_ns == 0
            || self.max_lag == 0
            || !self.alpha.is_finite()
            || !(0.0..1.0).contains(&self.alpha)
            || self.alpha == 0.0
        {
            return Err(RpcmciReplayError::Invalid(
                "supply a bounded complete regularly spaced series, positive lag and valid CI controls",
            ));
        }
        Ok(())
    }

    fn source_series(&self, schema: Option<SourceSchema>) -> Result<TimeSeriesData, DataError> {
        let columns =
            self.names.iter().map(String::as_str).zip(self.columns.iter().map(Vec::as_slice));
        match schema {
            None => TimeSeriesData::from_f64_columns(columns, self.interval_ns),
            Some(SourceSchema::ContinuousContextV1) => {
                let mut builder = CausalSchemaBuilder::new();
                for name in &self.names {
                    builder.add_variable(
                        name.as_str(),
                        ValueType::Continuous,
                        SmallRoleSet::from_hint(RoleHint::Context),
                        None,
                        None,
                        MeasurementSpec::default(),
                    )?;
                }
                let table = TabularData::try_from_schema_f64(builder.build()?, columns)?;
                let index = TimeIndex {
                    regularity: SamplingRegularity::Regular { interval_ns: self.interval_ns },
                    length: self.columns[0].len(),
                };
                TimeSeriesData::try_new(table.storage().clone(), index)
            }
        }
    }

    /// Invoke the normal causal compiler and retain its full learned-regime report.
    ///
    /// Continuous columns carry the same context role as ordinary numeric ingestion.
    ///
    /// # Errors
    /// Invalid declaration, unsupported route, failed discovery or resource stop.
    pub fn run(self, ctx: &ExecutionContext) -> Result<LearnedRpcmciArtifact, RpcmciReplayError> {
        require_replay_unit(ctx, RUN_ENTRY)?;
        self.run_with_schema(Some(SourceSchema::ContinuousContextV1), ctx)
    }

    fn discovery_spec(&self) -> Result<DiscoverySpec, RpcmciReplayError> {
        let ci = resolve_ci("parcorr", None)?;
        let params = DiscoverParams {
            max_lag: self.max_lag,
            alpha: self.alpha,
            fdr: self.fdr.then(FdrAdjustment::bh),
            ci,
            multi_dataset: MultiDatasetConstraints::default(),
            max_cond_size: self.max_cond_size,
            knowledge: BackgroundKnowledge::default(),
        };
        let options = RpcmciLearning {
            count: if self.select_count {
                RegimeCountPolicy::Bic { max_regimes: self.count }
            } else {
                RegimeCountPolicy::Fixed(self.count)
            },
            starts: self.starts,
            max_iterations: self.max_iterations,
            switch_penalty: self.switch_penalty,
        };
        Ok(DiscoverySpec::rpcmci_learn(params, self.min_regime_len, options, "parcorr")?)
    }

    fn run_with_schema(
        self,
        source_schema: Option<SourceSchema>,
        ctx: &ExecutionContext,
    ) -> Result<LearnedRpcmciArtifact, RpcmciReplayError> {
        self.validate()?;
        let bytes = series_bytes(self.columns[0].len(), self.columns.len())?;
        let _lease = ctx.acquire_admission(0, bytes).map_err(|error| {
            if ctx.cancellation.is_cancelled() {
                RpcmciReplayError::Cancelled
            } else {
                RpcmciReplayError::Resource(error.to_string())
            }
        })?;
        let mut portable = ctx.clone();
        portable.kernel_policy = KernelPolicy::scalar_only();
        let spec = self.discovery_spec()?;
        let data = self.source_series(source_schema).map_err(CausalError::rejected)?;
        let width = u32::try_from(self.names.len()).expect("the admitted series width fits a u32");
        let variables = (0..width).map(VariableId::from_raw).collect::<Vec<_>>();
        let study = Study::series(data).discovery(spec, variables).build()?;
        let execution = CompiledExecution::run(&study, &portable).map_err(|error| match error {
            CausalError::Cancelled { .. } => RpcmciReplayError::Cancelled,
            CausalError::Resource { message }
            | CausalError::Discovery(antecedent_discovery::DiscoveryError::Resource(message)) => {
                RpcmciReplayError::Resource(message)
            }
            other => other.into(),
        })?;
        let program_identity = execution
            .lowered_identity()
            .ok_or(RpcmciReplayError::Invalid("compiled discovery identity is absent"))?
            .to_hex();
        let data_identity = execution
            .result()
            .executed_contract()
            .ok_or(RpcmciReplayError::Invalid("compiled discovery snapshot is absent"))?
            .inspection_data_snapshot
            .to_hex();
        let Some(discovery) = execution.result().discovery() else {
            return Err(RpcmciReplayError::Invalid("compiled discovery produced no structure"));
        };
        let DiscoveredStructure::Regimes(result) = &discovery.structure else {
            return Err(RpcmciReplayError::Invalid(
                "compiled discovery produced no regime structure",
            ));
        };
        let learning = result.learning.as_ref().ok_or(RpcmciReplayError::Invalid(
            "compiled discovery produced no learning diagnostics",
        ))?;
        let graphs = result
            .graphs
            .graphs
            .iter()
            .map(|(regime, graph)| {
                let graph = antecedent_io::temporal_cpdag_identity(graph)?;
                Ok((
                    regime.raw(),
                    CanonicalWriter::new("antecedent.rpcmci.temporal_graph.v1")
                        .bytes(&to_cbor(&graph)?)
                        .finish_hex(),
                ))
            })
            .collect::<Result<Vec<_>, IoError>>()?;
        let report = LearnedRpcmciReport {
            program_identity,
            data_identity,
            seed: learning.seed,
            assignments: result.assignments.regimes.iter().map(|regime| regime.raw()).collect(),
            graphs,
            selected_count: learning.selected_count,
            candidate_scores: learning.candidate_scores.clone(),
            infeasible_counts: learning.infeasible_counts.clone(),
            converged: learning.converged,
            iterations: learning.iterations,
            successful_starts: learning.successful_starts,
            assignment_agreement: learning.assignment_agreement,
            graph_agreement: learning.graph_agreement.clone(),
            diagnostics: result
                .diagnostics
                .iter()
                .map(|item| (item.code.to_string(), item.message.to_string()))
                .collect(),
        };
        let plan = execution.prepared().plan_view()?;
        let wire = Wire { request: self, report };
        let identity = wire_identity(&wire, source_schema)?;
        Ok(LearnedRpcmciArtifact { wire, source_schema, identity, result: result.clone(), plan })
    }
}

fn series_bytes(rows: usize, width: usize) -> Result<u64, RpcmciReplayError> {
    let rows = u64::try_from(rows).map_err(|_| {
        RpcmciReplayError::Resource("learned RPCMCI series quote overflowed".into())
    })?;
    let width = u64::try_from(width).map_err(|_| {
        RpcmciReplayError::Resource("learned RPCMCI series quote overflowed".into())
    })?;
    rows.checked_mul(width)
        .and_then(|cells| cells.checked_mul(16))
        .ok_or_else(|| RpcmciReplayError::Resource("learned RPCMCI series quote overflowed".into()))
}

fn encode_wire(wire: &Wire, source_schema: Option<SourceSchema>) -> Result<Vec<u8>, IoError> {
    match source_schema {
        Some(source_schema) => to_cbor(&SchemaWire { source_schema, result: wire }),
        None => to_cbor(wire),
    }
}

fn wire_identity(wire: &Wire, source_schema: Option<SourceSchema>) -> Result<String, IoError> {
    let tag = if source_schema.is_some() {
        "antecedent.rpcmci.learned_result.v2"
    } else {
        "antecedent.rpcmci.learned_result.v1"
    };
    Ok(CanonicalWriter::new(tag).bytes(&encode_wire(wire, source_schema)?).finish_hex())
}

impl LearnedRpcmciArtifact {
    /// Artifact identity to retain separately from the encoded result.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Full diagnostic report reproduced on replay.
    #[must_use]
    pub fn report(&self) -> &LearnedRpcmciReport {
        &self.wire.report
    }

    /// The usual inspectable compiler plan reconstructed from the retained source.
    #[must_use]
    pub fn plan_view(&self) -> &crate::planner::PlanView {
        &self.plan
    }

    /// Names in regime-graph variable order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.wire.request.names
    }

    /// The compiled discovery consumer's original labeled collection.
    #[must_use]
    pub fn result(&self) -> &RpcmciDiscoveryResult {
        &self.result
    }

    /// Read the original row and lag-window scope for one verified regime graph.
    ///
    /// The rows index the recorded series. Use its original lag windows to avoid
    /// combining observations from opposite sides of a transition.
    ///
    /// # Errors
    /// The chosen id is absent from the original regime report.
    pub fn regime_scope(&self, regime: u32) -> Result<LearnedRpcmciRegimeScope, RpcmciReplayError> {
        let report = self.report();
        let (index, (_, graph_identity)) = report
            .graphs
            .iter()
            .enumerate()
            .find(|(_, (id, _))| *id == regime)
            .ok_or(RpcmciReplayError::Invalid("choose a regime id from the retained report"))?;
        let lookback = 2 * self.wire.request.max_lag as usize;
        let assigned_rows = report
            .assignments
            .iter()
            .enumerate()
            .filter_map(|(index, &id)| (id == regime).then_some(index))
            .collect();
        let window_end_rows = report
            .assignments
            .windows(lookback + 1)
            .enumerate()
            .filter_map(|(start, window)| {
                window.iter().all(|&id| id == regime).then_some(start + lookback)
            })
            .collect();
        Ok(LearnedRpcmciRegimeScope {
            artifact_identity: self.identity.clone(),
            data_identity: report.data_identity.clone(),
            regime,
            graph_identity: graph_identity.clone(),
            assigned_rows,
            window_end_rows,
            lookback,
            graph_agreement: report.graph_agreement[index],
            assignment_agreement: report.assignment_agreement,
        })
    }

    /// Review one explicitly chosen regime for a subsequent single-graph consumer.
    ///
    /// Use [the regime scope](Self::regime_scope) for the original lag windows
    /// and the graph's stability across initializations.
    ///
    /// # Errors
    /// Unknown regime or unresolved conflict marks in its graph.
    pub fn accept_regime(&self, regime: u32) -> Result<AcceptedGraph, RpcmciReplayError> {
        let Some(graph) = self.result.graphs.get(RegimeId::from_raw(regime)) else {
            return Err(RpcmciReplayError::Invalid("choose a regime id from the retained report"));
        };
        Ok(AcceptedGraph::temporal_cpdag(graph.clone())?.with_discovery_algorithm("rpcmci"))
    }

    /// Export the source and exact diagnostics for a fresh compiler.
    ///
    /// # Errors
    /// Codec failure or artifact byte limit.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RpcmciReplayError> {
        let magic = if self.source_schema.is_some() { SCHEMA_MAGIC } else { LEGACY_MAGIC };
        let mut bytes = magic.to_vec();
        bytes.extend(encode_wire(&self.wire, self.source_schema)?);
        if bytes.len() > MAX_BYTES {
            return Err(IoError::TooLarge.into());
        }
        Ok(bytes)
    }
}

/// Consume against a retained identity and rebuild the source, compiler and output.
///
/// # Errors
/// Versions or identities differ, recomputation differs, or the execution stops.
pub fn consume_learned_rpcmci(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<LearnedRpcmciArtifact, RpcmciReplayError> {
    replay(bytes, Some(expected_identity), ctx)
}

pub(super) fn recognizes(bytes: &[u8]) -> bool {
    bytes.starts_with(SCHEMA_MAGIC) || bytes.starts_with(LEGACY_MAGIC)
}

fn decode_wire(bytes: &[u8]) -> Result<(Wire, Option<SourceSchema>), RpcmciReplayError> {
    if let Some(body) = bytes.strip_prefix(SCHEMA_MAGIC) {
        let wire: SchemaWire<Wire> = from_cbor(body)?;
        Ok((wire.result, Some(wire.source_schema)))
    } else if let Some(body) = bytes.strip_prefix(LEGACY_MAGIC) {
        Ok((from_cbor(body)?, None))
    } else {
        Err(RpcmciReplayError::Invalid("unknown artifact version"))
    }
}

fn replay(
    bytes: &[u8],
    expected: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<LearnedRpcmciArtifact, RpcmciReplayError> {
    if bytes.len() > MAX_BYTES {
        return Err(IoError::TooLarge.into());
    }
    let _lease =
        ctx.acquire_admission(0, (bytes.len() as u64).saturating_mul(4)).map_err(|error| {
            if ctx.cancellation.is_cancelled() {
                RpcmciReplayError::Cancelled
            } else {
                RpcmciReplayError::Resource(error.to_string())
            }
        })?;
    let (wire, source_schema) = decode_wire(bytes)?;
    let identity = wire_identity(&wire, source_schema)?;
    if expected.is_some_and(|retained| retained != identity) {
        return Err(RpcmciReplayError::Invalid(
            "the bundle differs from the retained RPCMCI artifact identity",
        ));
    }
    require_replay_unit(ctx, CONSUME_ENTRY)?;
    let mut replay_context = ctx.clone();
    replay_context.rng = RngFactory::from_seed(wire.report.seed);
    let live = wire.request.run_with_schema(source_schema, &replay_context)?;
    if live.identity != identity || live.report() != &wire.report {
        return Err(RpcmciReplayError::Invalid(
            "the regime graphs or diagnostic report differ on replay",
        ));
    }
    Ok(live)
}

pub(super) struct Verifier(pub ExecutionContext);

fn bundle_failure(error: &RpcmciReplayError) -> NodeFailure {
    let stage = match error {
        RpcmciReplayError::Cancelled | RpcmciReplayError::Causal(CausalError::Cancelled { .. }) => {
            BundleStage::ExecutionCancelled
        }
        RpcmciReplayError::Resource(_)
        | RpcmciReplayError::Causal(CausalError::Resource { .. }) => BundleStage::ResourceRefused,
        RpcmciReplayError::Io(IoError::TooLarge) => BundleStage::Oversized,
        _ => BundleStage::SwappedEvidence,
    };
    NodeFailure { stage, reason: error.to_string() }
}

pub(super) fn describe(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<ArtifactDescription, NodeFailure> {
    let artifact = replay(bytes, None, ctx).map_err(|error| bundle_failure(&error))?;
    let report = artifact.report();
    let mut values = vec![("regime_count".into(), f64::from(report.selected_count))];
    if let Some(agreement) = report.assignment_agreement {
        values.push(("assignment_agreement".into(), agreement));
    }
    for (&(id, _), agreement) in report.graphs.iter().zip(&report.graph_agreement) {
        if let Some(agreement) = agreement {
            values.push((format!("regime.{id}.graph_agreement"), *agreement));
        }
    }
    Ok(ArtifactDescription {
        kind: NodeKind::LearnedRpcmci,
        identity: artifact.identity().to_owned(),
        facts: BTreeMap::from([
            ("discovery_program".into(), report.program_identity.clone()),
            ("discovery_snapshot".into(), report.data_identity.clone()),
        ]),
        values,
    })
}

impl NodeVerifier for Verifier {
    fn kind(&self) -> NodeKind {
        NodeKind::LearnedRpcmci
    }

    fn verify(
        &self,
        _id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let artifact = describe(bytes, &self.0)?;
        Ok(VerifiedNode {
            identity: artifact.identity,
            facts: artifact.facts,
            values: artifact.values,
        })
    }
}

#[cfg(test)]
mod source_schema_tests {
    use super::*;

    const OLD_BYTES: &[u8] =
        include_bytes!("../../../../conformance/discovery/rpcmci_learned_v1.artifact");
    const OLD_ID: &str = "d1873b241ac72b140a2b06ab6d8494396d7ddb3ab3863d60027bd93a3fb57b10";
    const OLD_DATA: &str = "7a299295a8283a4c75afa108418a6b96895821504a38a6be02c476b2b279b87e";
    const CONTEXT_DATA: &str = "2918d6e1d9d1c7e529c80f9295b463bfbf3908e7c4a98bde88d1a942c74ddbcf";

    #[test]
    fn bundle_replay_preserves_cancellation_and_resource_stops() {
        let ctx = ExecutionContext::for_tests(0);
        ctx.cancellation.cancel();
        assert_eq!(describe(OLD_BYTES, &ctx).unwrap_err().stage, BundleStage::ExecutionCancelled);

        for (cause, stage) in [
            (CausalError::Cancelled { stage: "discover" }, BundleStage::ExecutionCancelled),
            (
                CausalError::Resource { message: "insufficient memory".into() },
                BundleStage::ResourceRefused,
            ),
        ] {
            let error = RpcmciReplayError::Causal(cause);
            let message = error.to_string();
            let failure = bundle_failure(&error);
            assert_eq!(failure.stage, stage);
            assert_eq!(failure.reason, message);
        }
        let ctx = ExecutionContext::for_tests(0);
        assert_eq!(
            describe(b"bad envelope", &ctx).unwrap_err().stage,
            BundleStage::SwappedEvidence
        );
    }

    #[test]
    fn v1_wire_keeps_its_original_layout_and_identity() {
        let (wire, schema) = decode_wire(OLD_BYTES).unwrap();
        assert!(schema.is_none());
        assert_eq!(wire_identity(&wire, schema).unwrap(), OLD_ID);
        assert_eq!(wire.report.data_identity, OLD_DATA);
        assert_eq!(encode_wire(&wire, schema).unwrap(), &OLD_BYTES[LEGACY_MAGIC.len()..]);

        let mut wrong_magic = SCHEMA_MAGIC.to_vec();
        wrong_magic.extend_from_slice(&OLD_BYTES[LEGACY_MAGIC.len()..]);
        assert!(decode_wire(&wrong_magic).is_err());
    }

    #[test]
    fn both_retained_schema_versions_replay_without_rebinding_data() {
        let (frozen, schema) = decode_wire(OLD_BYTES).unwrap();
        let ctx = ExecutionContext::production(frozen.report.seed, 1);
        // Compute the legacy diagnostic bits on this target before independently
        // serializing its original envelope. The frozen test above pins framing
        // without depending on the platform's floating-point logarithm.
        let old = frozen.request.clone().run_with_schema(schema, &ctx).unwrap();
        let mut bytes = LEGACY_MAGIC.to_vec();
        bytes.extend(to_cbor(&old.wire).unwrap());
        let old_id = CanonicalWriter::new("antecedent.rpcmci.learned_result.v1")
            .bytes(&bytes[LEGACY_MAGIC.len()..])
            .finish_hex();
        let legacy = consume_learned_rpcmci(&bytes, &old_id, &ctx).unwrap();
        assert_eq!(legacy.report().data_identity, OLD_DATA);
        assert_eq!(legacy.to_bytes().unwrap(), bytes);

        let new = frozen.request.run(&ctx).unwrap();
        assert_eq!(new.report().data_identity, CONTEXT_DATA);
        assert_eq!(new.report().program_identity, old.report().program_identity);
        assert_eq!(new.report().assignments, old.report().assignments);
        assert_eq!(new.report().graphs, old.report().graphs);
        let bytes = new.to_bytes().unwrap();
        assert!(bytes.starts_with(SCHEMA_MAGIC));
        assert!(consume_learned_rpcmci(&bytes, &old_id, &ctx).is_err());
        assert_eq!(
            consume_learned_rpcmci(&bytes, new.identity(), &ctx).unwrap().report(),
            new.report()
        );
    }
}
