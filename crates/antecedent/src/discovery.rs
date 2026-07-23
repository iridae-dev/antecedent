//! Coarse discovery stage APIs for the facade.
//!
//! Bindings and analysis call these instead of constructing PCMCI-family
//! algorithms directly when the facade covers the stage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{MultiEnvironmentData, TabularData, TimeSeriesData};
pub use antecedent_discovery::{
    CiScreenedPosterior, CiSoftWeight, ContextKind, CpdagDiscoveryResult, DagDiscoveryResult,
    DbnPosterior, DirectLingam, DiscoveryPerformanceRecord, EXACT_ENUM_MAX_NODES,
    ExactDagPosterior, Fci, Ges, GraphPosterior, GraphPosteriorEngine, GraphPrior, JpcmciNodeRole,
    JpcmciPlus, Lpcmci, MultiDatasetConstraints, Notears, NotearsDiscoveryResult, OrderMcmc,
    PagDiscoveryResult, Pc, RegimeAssignment, RegimeGraphCollection, Rfci, Rpcmci,
    RpcmciDiscoveryResult, ScoredLink, SpaceDummyCiMode, StaticCpdagDiscoveryResult,
    StaticDagDiscoveryResult, StaticPagDiscoveryResult, StructureMcmc, TimeDummyCiMode,
    two_regime_half_split,
};
use antecedent_discovery::{DiscoveryWorkspace, Pcmci, PcmciPlus};
use antecedent_graph::{DenseNodeId, Endpoint, TemporalPag};
use antecedent_state::GraphScoreFamily;
use antecedent_stats::{ConditionalIndependence, FdrAdjustment};

use crate::discovery_defaults::{
    DEFAULT_RPCMCI_MIN_REGIME_LEN, contemporaneous_constraints, jpcmci_constraints,
    pcmci_constraints, static_pc_constraints,
};
use crate::error::CausalError;

/// Prior + score family for Bayesian graph-posterior discovery.
#[derive(Clone, Debug)]
pub struct BayesianDiscoverParams {
    /// Structural prior (default: uniform over constraint-valid DAGs).
    pub prior: GraphPrior,
    /// Local score family (currently Gaussian BIC only).
    pub score_family: GraphScoreFamily,
}

impl Default for BayesianDiscoverParams {
    fn default() -> Self {
        Self { prior: GraphPrior::uniform(), score_family: GraphScoreFamily::GaussianBic }
    }
}

/// MCMC schedule for order / structure / CI-screened / DBN posterior search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphMcmcSchedule {
    /// Number of chains (≥ 2 for R-hat).
    pub n_chains: u32,
    /// Warmup draws discarded per chain.
    pub n_warmup: u32,
    /// Post-warmup draws retained per chain (before thinning).
    pub n_draws: u32,
    /// Keep every `thin`-th post-warmup draw.
    pub thin: u32,
}

impl Default for GraphMcmcSchedule {
    fn default() -> Self {
        Self { n_chains: 4, n_warmup: 500, n_draws: 1000, thin: 1 }
    }
}

/// Parameters shared by PCMCI-family discovery stage calls.
#[derive(Clone)]
pub struct DiscoverParams {
    /// Maximum lag.
    pub max_lag: u32,
    /// Significance level.
    pub alpha: f64,
    /// Multiple-testing adjustment (`None` = off).
    pub fdr: Option<antecedent_stats::FdrAdjustment>,
    /// Conditional-independence test (resolved via [`crate::discovery_defaults::resolve_ci`]).
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// Multi-dataset / context settings (J-PCMCI+); ignored by single-series algorithms.
    pub multi_dataset: MultiDatasetConstraints,
}

impl std::fmt::Debug for DiscoverParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscoverParams")
            .field("max_lag", &self.max_lag)
            .field("alpha", &self.alpha)
            .field("fdr", &self.fdr)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("multi_dataset", &self.multi_dataset)
            .finish()
    }
}

/// Parameters for static (non-temporal) discovery stage calls.
#[derive(Clone)]
pub struct StaticDiscoverParams {
    /// Significance level.
    pub alpha: f64,
    /// Max conditioning-set size.
    pub max_cond_size: usize,
    /// Multiple-testing adjustment (`None` = off).
    pub fdr: Option<FdrAdjustment>,
    /// Conditional-independence test.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// GES only: soft PC-skeleton screening for Insert candidates.
    pub screen_pc: bool,
    /// GES only: T/H subset enumeration cap (`None` → default 12).
    pub max_subset: Option<usize>,
}

impl std::fmt::Debug for StaticDiscoverParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticDiscoverParams")
            .field("alpha", &self.alpha)
            .field("max_cond_size", &self.max_cond_size)
            .field("fdr", &self.fdr)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("screen_pc", &self.screen_pc)
            .field("max_subset", &self.max_subset)
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
) -> Result<DagDiscoveryResult, CausalError> {
    let pcmci = Pcmci::new()
        .with_fdr_adjustment(params.fdr)
        .with_constraints(pcmci_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    pcmci.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
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
) -> Result<CpdagDiscoveryResult, CausalError> {
    let plus = PcmciPlus::new()
        .with_fdr_adjustment(params.fdr)
        .with_constraints(contemporaneous_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    plus.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
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
) -> Result<PagDiscoveryResult, CausalError> {
    let alg = Lpcmci::new()
        .with_fdr_adjustment(params.fdr)
        .with_constraints(contemporaneous_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
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
) -> Result<CpdagDiscoveryResult, CausalError> {
    let alg = JpcmciPlus::new()
        .with_fdr_adjustment(params.fdr)
        .with_constraints(jpcmci_constraints(
            params.max_lag,
            params.alpha,
            params.multi_dataset.clone(),
        ))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
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
) -> Result<RpcmciDiscoveryResult, CausalError> {
    let plus = PcmciPlus::new()
        .with_fdr_adjustment(params.fdr)
        .with_constraints(contemporaneous_constraints(params.max_lag, params.alpha))
        .with_ci(Arc::clone(&params.ci));
    let alg = Rpcmci::new()
        .with_min_regime_len(min_regime_len.unwrap_or(DEFAULT_RPCMCI_MIN_REGIME_LEN))
        .with_pcmci_plus(plus);
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, assignment, &mut ws, ctx).map_err(CausalError::from)
}

/// Run static PC over tabular data.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_pc(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticCpdagDiscoveryResult, CausalError> {
    let fdr = params.fdr.map(|f| f.with_exclude_contemporaneous(false));
    let alg = Pc::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_pc_constraints(params.alpha, params.max_cond_size))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
}

/// Run classic static FCI over tabular data → PAG.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_fci(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticPagDiscoveryResult, CausalError> {
    let fdr = params.fdr.map(|f| f.with_exclude_contemporaneous(false));
    let alg = Fci::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_pc_constraints(params.alpha, params.max_cond_size))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
}

/// Run classic static RFCI over tabular data → PAG.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_rfci(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticPagDiscoveryResult, CausalError> {
    let fdr = params.fdr.map(|f| f.with_exclude_contemporaneous(false));
    let alg = Rfci::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_pc_constraints(params.alpha, params.max_cond_size))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
}

/// Run GES (Gaussian BIC) over tabular data → CPDAG.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_ges(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticCpdagDiscoveryResult, CausalError> {
    let fdr = params.fdr.map(|f| f.with_exclude_contemporaneous(false));
    let alg = Ges::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_pc_constraints(params.alpha, params.max_cond_size))
        .with_ci(Arc::clone(&params.ci))
        .with_pc_screening(params.screen_pc)
        .with_max_subset(params.max_subset);
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
}

/// Run `DirectLiNGAM` over tabular data → DAG.
///
/// # Errors
///
/// Discovery failures.
pub fn discover_lingam(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    prune_threshold: f64,
    ctx: &ExecutionContext,
) -> Result<StaticDagDiscoveryResult, CausalError> {
    let alg = DirectLingam::new()
        .with_constraints(static_pc_constraints(params.alpha, params.max_cond_size))
        .with_prune_threshold(prune_threshold);
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
}

/// Discover a static DAG with NOTEARS (continuous SEM; ).
///
/// Returns the hard DAG review plus the soft weight matrix for mechanism seeding.
///
/// # Errors
///
/// Discovery / solver failures.
pub fn discover_notears(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    lambda: f64,
    threshold: f64,
    standardize: bool,
    ctx: &ExecutionContext,
) -> Result<NotearsDiscoveryResult, CausalError> {
    let alg = Notears::new()
        .with_constraints(static_pc_constraints(params.alpha, params.max_cond_size))
        .with_lambda(lambda)
        .with_threshold(threshold)
        .with_standardize(standardize);
    let mut ws = DiscoveryWorkspace::default();
    alg.run(data, variables, &mut ws, ctx).map_err(CausalError::from)
}

/// Exact DAG posterior enumeration (`n ≤ 6`, Gaussian BIC).
///
/// # Errors
///
/// Discovery failures (oversized graph, empty support, score/data errors).
pub fn discover_exact_dag_posterior(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let eng = ExactDagPosterior::new();
    let mut ws = DiscoveryWorkspace::default();
    eng.run(data, variables, &params.prior, params.score_family, &mut ws, ctx)
        .map_err(CausalError::from)
}

/// Order MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
pub fn discover_order_mcmc(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    require_diagnostics_gate: bool,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let eng = OrderMcmc::new()
        .with_schedule(schedule.n_chains, schedule.n_warmup, schedule.n_draws, schedule.thin)
        .with_diagnostics_gate(require_diagnostics_gate);
    let mut ws = DiscoveryWorkspace::default();
    eng.run(data, variables, &params.prior, params.score_family, &mut ws, ctx)
        .map_err(CausalError::from)
}

/// Structure MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
pub fn discover_structure_mcmc(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let eng = StructureMcmc::new().with_schedule(
        schedule.n_chains,
        schedule.n_warmup,
        schedule.n_draws,
        schedule.thin,
    );
    let mut ws = DiscoveryWorkspace::default();
    eng.run(data, variables, &params.prior, params.score_family, &mut ws, ctx)
        .map_err(CausalError::from)
}

/// CI-screened candidate-edge posterior (PC skeleton → structure MCMC).
///
/// # Errors
///
/// Screening, empty skeleton, or MCMC failures.
pub fn discover_ci_screened_posterior(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    screen: &StaticDiscoverParams,
    schedule: &GraphMcmcSchedule,
    soft_weight: CiSoftWeight,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let fdr = screen.fdr.map(|f| f.with_exclude_contemporaneous(false));
    let mcmc = StructureMcmc::new().with_schedule(
        schedule.n_chains,
        schedule.n_warmup,
        schedule.n_draws,
        schedule.thin,
    );
    let eng = CiScreenedPosterior::new()
        .with_constraints(static_pc_constraints(screen.alpha, screen.max_cond_size))
        .with_ci(Arc::clone(&screen.ci))
        .with_soft_weight(soft_weight)
        .with_mcmc(mcmc);
    // Preserve FDR from screen (CiScreenedPosterior::new sets a default).
    let mut eng = eng;
    eng.fdr = fdr;
    let mut ws = DiscoveryWorkspace::default();
    eng.run(data, variables, &params.prior, params.score_family, &mut ws, ctx)
        .map_err(CausalError::from)
}

/// Bounded-lag DBN template posterior from a time series (Gaussian BIC).
///
/// # Errors
///
/// Short series, unsupported size, score, or empty support.
pub fn discover_dbn_posterior(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    max_lag: u32,
    force_mcmc: bool,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let eng = DbnPosterior::new(max_lag).with_force_mcmc(force_mcmc).with_mcmc_schedule(
        schedule.n_chains,
        schedule.n_warmup,
        schedule.n_draws,
    );
    eng.run(data, variables, &params.prior, params.score_family, ctx).map_err(CausalError::from)
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
