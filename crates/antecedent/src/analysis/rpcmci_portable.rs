//! Portable learned-regime discovery through the compiled structure search.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::execution::{KernelPolicy, RngFactory};
use antecedent_core::{CanonicalWriter, ExecutionContext, RegimeId, VariableId};
use antecedent_data::TimeSeriesData;
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
use crate::{
    AcceptedGraph, CausalError, CompiledExecution, DiscoveredStructure, DiscoverySpec, Study,
};

const MAGIC: &[u8] = b"ANTE-RPCMCI-LEARNED-1\0";
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_CELLS: usize = 1_000_000;

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

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    request: LearnedRpcmciRequest,
    report: LearnedRpcmciReport,
}

/// Replayed discovery with the raw evidence needed by the independent reader.
pub struct LearnedRpcmciArtifact {
    result: RpcmciDiscoveryResult,
    wire: Wire,
    identity: String,
    plan: crate::planner::PlanView,
}

impl LearnedRpcmciRequest {
    fn validate(&self) -> Result<(), RpcmciReplayError> {
        let width = self.names.len();
        let rows = self.columns.first().map_or(0, Vec::len);
        if !(1..=64).contains(&width)
            || self.names.iter().any(|name| name.trim().is_empty() || name.len() > 256)
            || self.names.iter().collect::<BTreeSet<_>>().len() != width
            || self.columns.len() != width
            || rows < 10
            || rows.saturating_mul(width) > MAX_CELLS
            || self
                .columns
                .iter()
                .any(|column| column.len() != rows || column.iter().any(|x| !x.is_finite()))
            || self.interval_ns == 0
            || !(1..=8).contains(&self.max_lag)
            || self.max_cond_size > 8
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

    /// Invoke the normal causal compiler and retain its full learned-regime report.
    ///
    /// # Errors
    /// Invalid declaration, unsupported route, failed discovery or resource stop.
    pub fn run(self, ctx: &ExecutionContext) -> Result<LearnedRpcmciArtifact, RpcmciReplayError> {
        self.validate()?;
        let bytes = (self.columns[0].len() as u64)
            .saturating_mul(self.columns.len() as u64)
            .saturating_mul(16);
        let _lease = ctx.acquire_admission(0, bytes).map_err(|error| {
            if ctx.cancellation.is_cancelled() {
                RpcmciReplayError::Cancelled
            } else {
                RpcmciReplayError::Resource(error.to_string())
            }
        })?;
        let mut portable = ctx.clone();
        portable.kernel_policy = KernelPolicy::scalar_only();
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
        let spec = DiscoverySpec::rpcmci_learn(params, self.min_regime_len, options, "parcorr")?;
        let data = TimeSeriesData::from_f64_columns(
            self.names.iter().map(String::as_str).zip(self.columns.iter().map(Vec::as_slice)),
            self.interval_ns,
        )
        .map_err(CausalError::rejected)?;
        let variables = (0..self.names.len() as u32).map(VariableId::from_raw).collect::<Vec<_>>();
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
        let identity = wire_identity(&wire)?;
        Ok(LearnedRpcmciArtifact { wire, identity, result: result.clone(), plan })
    }
}

fn wire_identity(wire: &Wire) -> Result<String, IoError> {
    Ok(CanonicalWriter::new("antecedent.rpcmci.learned_result.v1")
        .bytes(&to_cbor(wire)?)
        .finish_hex())
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

    /// Review one explicitly chosen regime for a subsequent single-graph consumer.
    ///
    /// # Errors
    /// Unknown regime or unresolved conflict marks in its graph.
    pub fn accept_regime(&self, regime: u32) -> Result<AcceptedGraph, RpcmciReplayError> {
        let Some(graph) = self.result.graphs.get(RegimeId::from_raw(regime)) else {
            return Err(RpcmciReplayError::Invalid("choose a regime id from the retained report"));
        };
        Ok(AcceptedGraph::temporal_cpdag(graph.clone())?)
    }

    /// Export the source and exact diagnostics for a fresh compiler.
    ///
    /// # Errors
    /// Codec failure or artifact byte limit.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RpcmciReplayError> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(to_cbor(&self.wire)?);
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
    bytes.starts_with(MAGIC)
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
    let wire: Wire = from_cbor(
        bytes.strip_prefix(MAGIC).ok_or(RpcmciReplayError::Invalid("unknown artifact version"))?,
    )?;
    let identity = wire_identity(&wire)?;
    if expected.is_some_and(|retained| retained != identity) {
        return Err(RpcmciReplayError::Invalid(
            "the bundle differs from the retained RPCMCI artifact identity",
        ));
    }
    let mut replay_context = ctx.clone();
    replay_context.rng = RngFactory::from_seed(wire.report.seed);
    let live = wire.request.run(&replay_context)?;
    if live.identity != identity || live.report() != &wire.report {
        return Err(RpcmciReplayError::Invalid(
            "the regime graphs or diagnostic report differ on replay",
        ));
    }
    Ok(live)
}

pub(super) struct Verifier(pub ExecutionContext);

pub(super) fn describe(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<ArtifactDescription, NodeFailure> {
    let artifact = replay(bytes, None, ctx).map_err(|error| {
        let stage = match error {
            RpcmciReplayError::Cancelled => BundleStage::ExecutionCancelled,
            RpcmciReplayError::Resource(_)
            | RpcmciReplayError::Causal(CausalError::Resource { .. }) => {
                BundleStage::ResourceRefused
            }
            RpcmciReplayError::Io(IoError::TooLarge) => BundleStage::Oversized,
            _ => BundleStage::SwappedEvidence,
        };
        NodeFailure { stage, reason: error.to_string() }
    })?;
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
