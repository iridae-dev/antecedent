//! Coarse discovery stage APIs for the facade (DESIGN.md §21 / §25.2).
//!
//! Bindings and analysis call these instead of constructing PCMCI-family
//! algorithms directly when the facade covers the stage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::{MultiEnvironmentData, TimeSeriesData};
use causal_discovery::{
    CpdagDiscoveryResult, DagDiscoveryResult, DiscoveryWorkspace, JpcmciPlus, Lpcmci, Pcmci,
    PcmciPlus, PagDiscoveryResult, RegimeAssignment, Rpcmci, RpcmciDiscoveryResult,
};
use causal_graph::{DenseNodeId, Endpoint, TemporalPag};
use causal_stats::ConditionalIndependence;

use crate::discovery_defaults::{
    contemporaneous_constraints, pcmci_constraints, DEFAULT_RPCMCI_MIN_REGIME_LEN,
};
use crate::error::AnalysisError;

/// Parameters shared by PCMCI-family discovery stage calls.
#[derive(Clone)]
pub struct DiscoverParams {
    /// Maximum lag.
    pub max_lag: u32,
    /// Significance level.
    pub alpha: f64,
    /// Apply FDR control when the algorithm supports it.
    pub fdr: bool,
    /// Conditional-independence test (resolved via [`crate::discovery_defaults::resolve_ci`]).
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
}

impl std::fmt::Debug for DiscoverParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscoverParams")
            .field("max_lag", &self.max_lag)
            .field("alpha", &self.alpha)
            .field("fdr", &self.fdr)
            .field("ci", &"<dyn ConditionalIndependence>")
            .finish()
    }
}

/// Run lagged PCMCI.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_pcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<DagDiscoveryResult, AnalysisError> {
    let pcmci = Pcmci::new()
        .with_fdr(params.fdr)
        .with_constraints(pcmci_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    pcmci.run(data, variables, &mut ws, ctx).map_err(AnalysisError::from)
}

/// Run PCMCI+.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_pcmci_plus(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, AnalysisError> {
    let plus = PcmciPlus::new()
        .with_fdr(params.fdr)
        .with_constraints(contemporaneous_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    plus.run(data, variables, &mut ws, ctx).map_err(AnalysisError::from)
}

/// Run LPCMCI.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_lpcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<PagDiscoveryResult, AnalysisError> {
    let alg = Lpcmci::new()
        .with_fdr(params.fdr)
        .with_constraints(contemporaneous_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(AnalysisError::from)
}

/// Run J-PCMCI+ over multi-environment series.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_jpcmci_plus(
    data: &MultiEnvironmentData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, AnalysisError> {
    let alg = JpcmciPlus::new()
        .with_fdr(params.fdr)
        .with_constraints(contemporaneous_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(AnalysisError::from)
}

/// Run RPCMCI with a supplied regime assignment.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_rpcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    assignment: &RegimeAssignment,
    params: &DiscoverParams,
    min_regime_len: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<RpcmciDiscoveryResult, AnalysisError> {
    let plus = PcmciPlus::new()
        .with_fdr(params.fdr)
        .with_constraints(contemporaneous_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let alg = Rpcmci::new()
        .with_min_regime_len(min_regime_len.unwrap_or(DEFAULT_RPCMCI_MIN_REGIME_LEN))
        .with_pcmci_plus(plus);
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, assignment, &mut ws, ctx)
        .map_err(AnalysisError::from)
}

/// Count definite directed edges in a temporal PAG (Tail–Arrow or Arrow–Tail).
#[must_use]
pub fn pag_definite_directed_edge_count(pag: &TemporalPag) -> u64 {
    let mut directed = 0u64;
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(i as u32);
        for (b, at_a, at_b) in pag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if matches!(
                (at_a, at_b),
                (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Arrow, Endpoint::Tail)
            ) {
                directed += 1;
            }
        }
    }
    directed
}
