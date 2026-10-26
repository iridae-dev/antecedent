//! Discovery parameters, result types and the search engines the compiled
//! structure-discovery route runs.
//!
//! A structure search is a compiled route: build a [`crate::DiscoverySpec`]
//! from the parameters here, start the study with
//! [`crate::StudyBuilder::discovery`], prepare it and estimate. The prepared
//! handle returns the search engine's own result together with a
//! [`crate::CausalContract`] and a claim; [`crate::accept_discovered`]
//! accepts the structure through its review gate and records its provenance.
//!
//! The `discover_*` functions are the engines that route runs. They are hidden
//! from the public surface: called directly they return a structure with no
//! contract, plan, claim or provenance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{MultiEnvironmentData, TabularData, TimeSeriesData};
pub use antecedent_discovery::{
    BackgroundKnowledge, CiScreenedPosterior, CiSoftWeight, ContextKind, CpdagDiscoveryResult,
    DagDiscoveryResult, DbnPosterior, DirectLingam, DiscoveryPerformanceRecord,
    EXACT_ENUM_MAX_NODES, EXACT_MARGINAL_MAX_NODES, ExactDagMarginals, ExactDagPosterior, Fci, Ges,
    GraphPosterior, GraphPosteriorAtomKind, GraphPosteriorEngine, GraphPrior, JpcmciNodeRole,
    JpcmciPlus, LaggedParent, Lpcmci, MultiDatasetConstraints, Notears, NotearsDiscoveryResult,
    OrderMcmc, PagDiscoveryResult, Pc, PcSepsets, RegimeAssignment, RegimeGraphCollection, Rfci,
    Rpcmci, RpcmciDiscoveryResult, SamplingSummary, ScoredLink, SpaceDummyCiMode,
    StaticCpdagDiscoveryResult, StaticDagDiscoveryResult, StaticPagDiscoveryResult, StructureMcmc,
    StructureStability, TimeDummyCiMode, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    two_regime_half_split,
};
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Pcmci, PcmciPlus};
use antecedent_graph::{DenseNodeId, Endpoint, TemporalPag};
pub use antecedent_prob::InferenceDiagnostics;
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

/// Debug text for a `ConditionalIndependence` object. The trait has no `Debug` supertrait.
const DYN_CI_DEBUG: &str = "<dyn ConditionalIndependence>";

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
    /// Max conditioning-set size in the PC1 / skeleton phase (same cap `PC`/`GES`/`FCI`/
    /// `RFCI`/`LiNGAM`/`NOTEARS` expose via `StaticDiscoverParams::max_cond_size`).
    pub max_cond_size: usize,
    /// Background knowledge: required and forbidden links, tiers and a parent limit. PCMCI and
    /// PCMCI+ honour it and refuse what they cannot; LPCMCI, J-PCMCI+ and RPCMCI take none and
    /// are never handed any by the bindings.
    pub knowledge: BackgroundKnowledge,
}

impl std::fmt::Debug for DiscoverParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscoverParams")
            .field("max_lag", &self.max_lag)
            .field("alpha", &self.alpha)
            .field("fdr", &self.fdr)
            .field("ci", &DYN_CI_DEBUG)
            .field("multi_dataset", &self.multi_dataset)
            .field("max_cond_size", &self.max_cond_size)
            .field("knowledge", &self.knowledge)
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
    /// Background knowledge: required and forbidden links, tiers and a parent limit. PC, FCI,
    /// RFCI and GES honour it and refuse what they cannot (see the discovery guide for each
    /// algorithm's semantics); `LiNGAM` and NOTEARS pass it to their own constraint handling.
    pub knowledge: BackgroundKnowledge,
}

impl std::fmt::Debug for StaticDiscoverParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticDiscoverParams")
            .field("alpha", &self.alpha)
            .field("max_cond_size", &self.max_cond_size)
            .field("fdr", &self.fdr)
            .field("ci", &DYN_CI_DEBUG)
            .field("screen_pc", &self.screen_pc)
            .field("max_subset", &self.max_subset)
            .field("knowledge", &self.knowledge)
            .finish()
    }
}

/// `constraints` with `knowledge` set on them.
fn constrained(
    mut constraints: DiscoveryConstraints,
    knowledge: &BackgroundKnowledge,
) -> DiscoveryConstraints {
    knowledge.apply_to(&mut constraints);
    constraints
}

/// A temporal discoverer configured from `params`: its FDR adjustment, the given
/// `constraints` and the conditional-independence test.
macro_rules! temporal_discoverer {
    ($algorithm:expr, $params:expr, $constraints:expr) => {
        $algorithm
            .with_fdr_adjustment($params.fdr)
            .with_constraints($constraints)
            .with_ci(Arc::clone(&$params.ci))
    };
}

/// Run lagged PCMCI.
///
/// # Errors
///
/// Discovery failures, or background knowledge the search cannot honour.
#[doc(hidden)]
pub fn discover_pcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<DagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(pcmci_for(params).run(data, variables, &mut ws, ctx)?)
}

fn pcmci_for(params: &DiscoverParams) -> Pcmci {
    temporal_discoverer!(
        Pcmci::new(),
        params,
        constrained(
            pcmci_constraints(params.max_lag, params.alpha, params.max_cond_size),
            &params.knowledge,
        )
    )
}

/// Run PCMCI over the units of a panel.
///
/// Every unit's lag windows are built inside that unit and the rows are pooled, so no lagged
/// parent crosses a unit boundary. The pooled design is that of one structural model shared by
/// exchangeable units; a shift in level or scale between units is not modelled.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_pcmci_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<DagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(pcmci_for(params).run_panel(units, variables, &mut ws, ctx)?)
}

/// Run PCMCI+.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_pcmci_plus(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(pcmci_plus_for(params).run(data, variables, &mut ws, ctx)?)
}

fn pcmci_plus_for(params: &DiscoverParams) -> PcmciPlus {
    temporal_discoverer!(
        PcmciPlus::new(),
        params,
        constrained(
            contemporaneous_constraints(params.max_lag, params.alpha, params.max_cond_size),
            &params.knowledge,
        )
    )
}

/// Run PCMCI+ over the units of a panel (see [`discover_pcmci_panel`]).
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_pcmci_plus_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(pcmci_plus_for(params).run_panel(units, variables, &mut ws, ctx)?)
}

/// Run LPCMCI.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_lpcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<PagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(lpcmci_for(params).run(data, variables, &mut ws, ctx)?)
}

fn lpcmci_for(params: &DiscoverParams) -> Lpcmci {
    temporal_discoverer!(
        Lpcmci::new(),
        params,
        contemporaneous_constraints(params.max_lag, params.alpha, params.max_cond_size)
    )
}

/// Run LPCMCI over the units of a panel (see [`discover_pcmci_panel`]).
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_lpcmci_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<PagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(lpcmci_for(params).run_panel(units, variables, &mut ws, ctx)?)
}

/// Run J-PCMCI+ over multi-environment series.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_jpcmci_plus(
    data: &MultiEnvironmentData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    let alg = temporal_discoverer!(
        JpcmciPlus::new(),
        params,
        jpcmci_constraints(
            params.max_lag,
            params.alpha,
            params.max_cond_size,
            params.multi_dataset.clone(),
        )
    );
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, &mut ws, ctx)?)
}

/// Run RPCMCI with a supplied regime assignment.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_rpcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    assignment: &RegimeAssignment,
    params: &DiscoverParams,
    min_regime_len: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<RpcmciDiscoveryResult, CausalError> {
    let plus = temporal_discoverer!(
        PcmciPlus::new(),
        params,
        contemporaneous_constraints(params.max_lag, params.alpha, params.max_cond_size)
    );
    let alg = Rpcmci::new()
        .with_min_regime_len(min_regime_len.unwrap_or(DEFAULT_RPCMCI_MIN_REGIME_LEN))
        .with_pcmci_plus(plus);
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, assignment, &mut ws, ctx)?)
}

/// The constraints of a static run: the defaults for `params` with its background knowledge.
fn static_constraints(params: &StaticDiscoverParams) -> DiscoveryConstraints {
    constrained(static_pc_constraints(params.alpha, params.max_cond_size), &params.knowledge)
}

/// Static discovery tests every pair, so the contemporaneous exclusion is always off.
fn static_fdr(params: &StaticDiscoverParams) -> Option<FdrAdjustment> {
    params.fdr.map(|f| f.with_exclude_contemporaneous(false))
}

/// Run static PC over tabular data.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_pc(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticCpdagDiscoveryResult, CausalError> {
    let fdr = static_fdr(params);
    let alg = Pc::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_constraints(params))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, &mut ws, ctx)?)
}

/// Run classic static FCI over tabular data → PAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_fci(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticPagDiscoveryResult, CausalError> {
    let fdr = static_fdr(params);
    let alg = Fci::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_constraints(params))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, &mut ws, ctx)?)
}

/// Run classic static RFCI over tabular data → PAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_rfci(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticPagDiscoveryResult, CausalError> {
    let fdr = static_fdr(params);
    let alg = Rfci::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_constraints(params))
        .with_ci(Arc::clone(&params.ci));
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, &mut ws, ctx)?)
}

/// Run GES (Gaussian BIC) over tabular data → CPDAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_ges(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticCpdagDiscoveryResult, CausalError> {
    let fdr = static_fdr(params);
    let alg = Ges::new()
        .with_fdr_adjustment(fdr)
        .with_constraints(static_constraints(params))
        .with_ci(Arc::clone(&params.ci))
        .with_pc_screening(params.screen_pc)
        .with_max_subset(params.max_subset);
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, &mut ws, ctx)?)
}

/// Run `DirectLiNGAM` over tabular data → DAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
pub fn discover_lingam(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    prune_threshold: f64,
    ctx: &ExecutionContext,
) -> Result<StaticDagDiscoveryResult, CausalError> {
    let alg = DirectLingam::new()
        .with_constraints(static_constraints(params))
        .with_prune_threshold(prune_threshold);
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, &mut ws, ctx)?)
}

/// Discover a static DAG with NOTEARS (continuous SEM).
///
/// Returns the hard DAG review plus the soft weight matrix for mechanism seeding and the row
/// accounting of the run. `missing` declares how missing values are handled: `None` refuses a
/// table with a missing value, `Some(CompleteCase)` drops the incomplete rows and reports them.
///
/// # Errors
///
/// Discovery / solver failures, and `missing_policy_required` for a missing value without a
/// declared policy.
#[expect(
    clippy::too_many_arguments,
    reason = "public entry point whose parameters mirror the binding-level keyword list"
)]
#[doc(hidden)]
pub fn discover_notears_with(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    lambda: f64,
    threshold: f64,
    standardize: bool,
    missing: Option<antecedent_data::MissingPolicy>,
    ctx: &ExecutionContext,
) -> Result<NotearsDiscoveryResult, CausalError> {
    let mut alg = Notears::new()
        .with_constraints(static_constraints(params))
        .with_lambda(lambda)
        .with_threshold(threshold)
        .with_standardize(standardize);
    alg.missing = missing;
    let mut ws = DiscoveryWorkspace::default();
    Ok(alg.run(data, variables, &mut ws, ctx)?)
}

/// Discover a static DAG with NOTEARS (continuous SEM), refusing a table with a missing value.
///
/// [`discover_notears_with`] with no declared missing-data policy.
///
/// # Errors
///
/// Discovery / solver failures, and `missing_policy_required` for a missing value.
#[doc(hidden)]
pub fn discover_notears(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    lambda: f64,
    threshold: f64,
    standardize: bool,
    ctx: &ExecutionContext,
) -> Result<NotearsDiscoveryResult, CausalError> {
    discover_notears_with(data, variables, params, lambda, threshold, standardize, None, ctx)
}

/// Exact DAG posterior enumeration (`n ≤ 6`, Gaussian BIC).
///
/// # Errors
///
/// Discovery failures (oversized graph, empty support, score/data errors).
#[doc(hidden)]
pub fn discover_exact_dag_posterior(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let eng = ExactDagPosterior::new();
    let mut ws = DiscoveryWorkspace::default();
    // The engine labels the posterior with the algorithm that produced it: a problem above the
    // enumeration cap is routed to order MCMC and keeps that label.
    Ok(eng.run(data, variables, &params.prior, params.score_family, &mut ws, ctx)?)
}

/// Exact posterior over PAG classes of the Gaussian maximal ancestral graphs of at most four
/// variables, scored by BIC.
///
/// # Errors
///
/// More than four variables, missing or non-finite values, or no finite score.
#[doc(hidden)]
pub fn discover_exact_pag_posterior(
    data: &TabularData,
    variables: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    Ok(antecedent_discovery::exact_pag_posterior(data, variables, ctx)?)
}

/// Metropolis posterior over PAG classes of the Gaussian maximal ancestral graphs of two to
/// eight variables.
///
/// # Errors
///
/// A bad schedule or variable count, missing or non-finite values, or no scorable graph.
#[doc(hidden)]
pub fn discover_pag_mcmc(
    data: &TabularData,
    variables: &[VariableId],
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    Ok(antecedent_discovery::pag_posterior_mcmc(
        data,
        variables,
        antecedent_discovery::PagMcmcSchedule {
            chains: schedule.n_chains as usize,
            warmup: schedule.n_warmup as usize,
            draws: schedule.n_draws as usize,
            thin: schedule.thin as usize,
        },
        ctx,
    )?)
}

/// Exact edge marginals of the DAG posterior for `7 ≤ n ≤ 12` variables (Gaussian BIC).
///
/// Above [`EXACT_ENUM_MAX_NODES`] the enumerating engine cannot list the graphs, but the
/// marginal `P(i → j)` of every ordered pair and the normalizing constant are still exact
/// (sink recursion over node subsets). Graph atoms, a MAP graph and any quantity that needs the
/// list of graphs are not computed and are not approximated.
///
/// # Errors
///
/// More than [`EXACT_MARGINAL_MAX_NODES`] variables, invalid constraints, score/data errors.
#[doc(hidden)]
pub fn discover_exact_dag_marginals(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<ExactDagMarginals, CausalError> {
    Ok(antecedent_discovery::exact_dag_marginals(
        data,
        variables,
        &params.prior,
        params.score_family,
        ctx,
    )?)
}

/// Order MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
#[doc(hidden)]
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
    Ok(eng
        .run(data, variables, &params.prior, params.score_family, &mut ws, ctx)?
        .with_algorithm("order_mcmc"))
}

/// Structure MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
#[doc(hidden)]
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
    Ok(eng
        .run(data, variables, &params.prior, params.score_family, &mut ws, ctx)?
        .with_algorithm("structure_mcmc"))
}

/// CI-screened candidate-edge posterior (PC skeleton → structure MCMC).
///
/// # Errors
///
/// Screening, empty skeleton, or MCMC failures.
#[doc(hidden)]
pub fn discover_ci_screened_posterior(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    screen: &StaticDiscoverParams,
    schedule: &GraphMcmcSchedule,
    soft_weight: CiSoftWeight,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let fdr = static_fdr(screen);
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
    Ok(eng
        .run(data, variables, &params.prior, params.score_family, &mut ws, ctx)?
        .with_algorithm("ci_screened_posterior"))
}

/// Bounded-lag DBN template posterior from a time series (Gaussian BIC).
///
/// # Errors
///
/// Short series, unsupported size, score, or empty support.
#[doc(hidden)]
pub fn discover_dbn_posterior(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    max_lag: u32,
    force_mcmc: bool,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    Ok(dbn_posterior_for(max_lag, force_mcmc, schedule)
        .run(data, variables, &params.prior, params.score_family, ctx)?
        .with_algorithm("dbn_posterior"))
}

fn dbn_posterior_for(max_lag: u32, force_mcmc: bool, schedule: &GraphMcmcSchedule) -> DbnPosterior {
    DbnPosterior::new(max_lag).with_force_mcmc(force_mcmc).with_mcmc_schedule(
        schedule.n_chains,
        schedule.n_warmup,
        schedule.n_draws,
    )
}

/// Infer the DBN template posterior over the units of a panel (see [`discover_pcmci_panel`]).
///
/// # Errors
///
/// No usable rows, unsupported size, score, or empty support.
#[doc(hidden)]
pub fn discover_dbn_posterior_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    max_lag: u32,
    force_mcmc: bool,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    Ok(dbn_posterior_for(max_lag, force_mcmc, schedule)
        .run_panel(units, variables, &params.prior, params.score_family, ctx)?
        .with_algorithm("dbn_posterior"))
}

/// Count definite directed edges in a temporal PAG (Tail–Arrow or Arrow–Tail).
#[must_use]
pub fn pag_definite_directed_edge_count(pag: &TemporalPag) -> u64 {
    let mut directed = 0u64;
    for i in 0..pag.node_count() {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "node ids are u32 by construction (DenseNodeId), so node positions fit u32"
        )]
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
