//! Discovery result types (DESIGN.md §13.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, Lag, VariableId};
use causal_graph::TemporalDag;

/// Directed lagged link.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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

/// Graph evidence for a discovered temporal DAG.
#[derive(Clone, Debug)]
pub struct GraphEvidence {
    /// Temporal DAG summary.
    pub graph: TemporalDag,
    /// Kept links with MCI statistics.
    pub links: Arc<[ScoredLink]>,
}

/// Link with MCI statistic / p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredLink {
    /// Link.
    pub link: LaggedLink,
    /// Partial correlation (MCI).
    pub statistic: f64,
    /// P-value (possibly FDR-adjusted later).
    pub p_value: f64,
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
    /// Links retained after MCI.
    pub links_retained: u64,
    /// Targets processed.
    pub targets: u64,
}

/// Full discovery result.
#[derive(Clone, Debug)]
pub struct DiscoveryResult {
    /// Evidence.
    pub evidence: GraphEvidence,
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
}
