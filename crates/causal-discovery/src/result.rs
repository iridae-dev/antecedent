//! Discovery result types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, Lag, VariableId};
use causal_graph::{
    TemporalCpdag, TemporalCpdagReview, TemporalDag, TemporalGraphReview, TemporalPag,
    TemporalPagReview,
};

/// One lagged parent `(variable, lag)`.
pub type LaggedParent = (VariableId, Lag);

/// Key for a directed PC separation event `(source, source_lag, target, target_lag)`.
pub type SepsetKey = (VariableId, Lag, VariableId, Lag);

/// PC separating sets recorded during parent selection.
pub type PcSepsets = std::collections::HashMap<SepsetKey, std::sync::Arc<[LaggedParent]>>;

/// Directed lagged link.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LaggedLink {
    /// Source variable.
    pub source: VariableId,
    /// Source lag (positive = past).
    pub source_lag: Lag,
    /// Target variable (typically contemporaneous).
    pub target: VariableId,
    /// Target lag (usually contemporaneous).
    pub target_lag: Lag,
}

/// Graph evidence for a discovered graph of class `G`.
#[derive(Clone, Debug)]
pub struct GraphEvidence<G> {
    /// Graph summary (DAG, CPDAG, …).
    pub graph: G,
    /// Per-edge evidence keyed by compact lagged links (not high-level node objects).
    pub edge_evidence: Arc<[EdgeEvidence]>,
    /// Kept links with MCI statistics (aligned with [`Self::edge_evidence`] for callers).
    pub links: Arc<[ScoredLink]>,
    /// How the evidence was produced.
    pub source: EvidenceSource,
}

/// How graph evidence was obtained.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceSource {
    /// Temporal discovery (PCMCI / PCMCI+).
    Discovery {
        /// Algorithm id.
        algorithm: Arc<str>,
    },
    /// Expert / manually supplied.
    Expert,
}

/// Per-edge statistical / orientation evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeEvidence {
    /// Compact lagged edge key.
    pub link: LaggedLink,
    /// Dependence statistic (e.g. partial correlation).
    pub statistic: Option<f64>,
    /// Raw p-value.
    pub p_value: Option<f64>,
    /// Multiple-testing adjusted p-value when available.
    pub adjusted_p_value: Option<f64>,
    /// Optional confidence interval for the statistic.
    pub interval: Option<(f64, f64)>,
    /// Separating sets used during PC / orientation (may be empty).
    pub separating_sets: Arc<[Arc<[LaggedParent]>]>,
    /// Short provenance tags (e.g. `mci`, `orient.meek_r1`).
    pub provenance: Arc<[Arc<str>]>,
}

impl EdgeEvidence {
    /// Build from a scored MCI link.
    #[must_use]
    pub fn from_scored(link: ScoredLink, sepsets: Arc<[Arc<[LaggedParent]>]>) -> Self {
        Self {
            link: link.link,
            statistic: Some(link.statistic),
            p_value: Some(link.p_value),
            adjusted_p_value: None,
            interval: None,
            separating_sets: sepsets,
            provenance: Arc::from([Arc::from("mci")]),
        }
    }
}

/// PCMCI (lagged) graph evidence.
pub type DagGraphEvidence = GraphEvidence<TemporalDag>;

/// PCMCI+ graph evidence.
pub type CpdagGraphEvidence = GraphEvidence<TemporalCpdag>;

/// Link with MCI statistic / p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredLink {
    /// Link.
    pub link: LaggedLink,
    /// Partial correlation (MCI).
    pub statistic: f64,
    /// Raw (unadjusted) p-value.
    pub p_value: f64,
    /// Adjusted p-value when multiple-testing correction ran over the MCI family.
    pub adjusted_p_value: Option<f64>,
}

/// Algorithm metadata.
#[derive(Clone, Debug)]
pub struct AlgorithmRecord {
    /// Algorithm id.
    pub id: Arc<str>,
    /// Configuration digest / label.
    pub config: Arc<str>,
}

/// One discovery iteration summary.
#[derive(Clone, Debug, Default)]
pub struct DiscoveryIteration {
    /// Conditioning-set size for PC phase, or label.
    pub label: Arc<str>,
    /// CI tests performed.
    pub ci_tests: u64,
}

/// Discovery diagnostic.
#[derive(Clone, Debug)]
pub struct DiscoveryDiagnostic {
    /// Code.
    pub code: Arc<str>,
    /// Message.
    pub message: Arc<str>,
}

/// Performance counters.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiscoveryPerformanceRecord {
    /// CI tests executed.
    pub ci_tests: u64,
    /// Links retained after MCI (and optional FDR).
    pub links_retained: u64,
    /// Targets processed.
    pub targets: u64,
    /// Bytes in the prepared lagged frame.
    pub lagged_frame_bytes: u64,
    /// Worker threads used for target-wise parallel phases.
    pub worker_threads: u32,
}

/// Full discovery result parameterized by graph class and review artifact.
#[derive(Clone, Debug)]
pub struct DiscoveryResult<G, R> {
    /// Evidence.
    pub evidence: GraphEvidence<G>,
    /// Review artifact listing pending edges / orientations.
    pub review: R,
    /// Algorithm.
    pub algorithm: AlgorithmRecord,
    /// Assumptions.
    pub assumptions: AssumptionSet,
    /// Iterations.
    pub iterations: Vec<DiscoveryIteration>,
    /// Diagnostics.
    pub diagnostics: Vec<DiscoveryDiagnostic>,
    /// Performance.
    pub performance: DiscoveryPerformanceRecord,
    /// PC separating sets: `(source, source_lag, target, target_lag) → conditioning set`.
    pub sepsets: PcSepsets,
}

/// Lagged PCMCI discovery result (`TemporalDag` evidence + DAG review).
pub type DagDiscoveryResult = DiscoveryResult<TemporalDag, TemporalGraphReview>;

/// PCMCI+ discovery result (`TemporalCpdag` evidence + CPDAG review).
pub type CpdagDiscoveryResult = DiscoveryResult<TemporalCpdag, TemporalCpdagReview>;

/// LPCMCI discovery result (`TemporalPag` evidence + PAG review).
pub type PagDiscoveryResult = DiscoveryResult<TemporalPag, TemporalPagReview>;

/// Graph evidence specialized to a temporal PAG.
pub type PagGraphEvidence = GraphEvidence<TemporalPag>;
