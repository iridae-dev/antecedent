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
//! The search engines the route runs live in the private `engines` module. The
//! hidden `discover_*` functions here are deprecated wrappers over them: called
//! directly they return a structure with no contract, plan, claim or provenance.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    EXACT_ENUM_MAX_NODES, EXACT_MARGINAL_MAX_NODES, EdgePriors, ExactDagMarginals,
    ExactDagPosterior, Fci, Ges, GraphPosterior, GraphPosteriorAtomKind, GraphPosteriorEngine,
    GraphPrior, JpcmciNodeRole, JpcmciPlus, LaggedParent, Lpcmci, MultiDatasetConstraints, Notears,
    NotearsDiscoveryResult, OrderMcmc, PagDiscoveryResult, Pc, PcSepsets, RegimeAssignment,
    RegimeGraphCollection, Rfci, Rpcmci, RpcmciDiscoveryResult, SamplingSummary, ScoredLink,
    ScreenedMarginals, SoftTiers, SpaceDummyCiMode, StaticCpdagDiscoveryResult,
    StaticDagDiscoveryResult, StaticPagDiscoveryResult, StructureMcmc, StructureStability,
    TimeDummyCiMode, adjacency_mask_from_cpdag, adjacency_masks_from_pag, two_regime_half_split,
};
use antecedent_graph::{DenseNodeId, Endpoint, TemporalPag};
pub use antecedent_prob::InferenceDiagnostics;
use antecedent_state::GraphScoreFamily;
use antecedent_stats::{ConditionalIndependence, FdrAdjustment};

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

pub(crate) mod engines;

/// Run lagged PCMCI.
///
/// # Errors
///
/// Discovery failures, or background knowledge the search cannot honour.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<DagDiscoveryResult, CausalError> {
    engines::discover_pcmci(data, variables, params, ctx)
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
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pcmci_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<DagDiscoveryResult, CausalError> {
    engines::discover_pcmci_panel(units, variables, params, ctx)
}

/// Run PCMCI+.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pcmci_plus(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    engines::discover_pcmci_plus(data, variables, params, ctx)
}

/// Run PCMCI+ over the units of a panel (see [`discover_pcmci_panel`]).
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pcmci_plus_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    engines::discover_pcmci_plus_panel(units, variables, params, ctx)
}

/// Run LPCMCI.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_lpcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<PagDiscoveryResult, CausalError> {
    engines::discover_lpcmci(data, variables, params, ctx)
}

/// Run LPCMCI over the units of a panel (see [`discover_pcmci_panel`]).
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_lpcmci_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<PagDiscoveryResult, CausalError> {
    engines::discover_lpcmci_panel(units, variables, params, ctx)
}

/// Run J-PCMCI+ over multi-environment series.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_jpcmci_plus(
    data: &MultiEnvironmentData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    engines::discover_jpcmci_plus(data, variables, params, ctx)
}

/// Run RPCMCI with a supplied regime assignment.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_rpcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    assignment: &RegimeAssignment,
    params: &DiscoverParams,
    min_regime_len: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<RpcmciDiscoveryResult, CausalError> {
    engines::discover_rpcmci(data, variables, assignment, params, min_regime_len, ctx)
}

/// Run static PC over tabular data.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pc(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticCpdagDiscoveryResult, CausalError> {
    engines::discover_pc(data, variables, params, ctx)
}

/// Run classic static FCI over tabular data → PAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_fci(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticPagDiscoveryResult, CausalError> {
    engines::discover_fci(data, variables, params, ctx)
}

/// Run classic static RFCI over tabular data → PAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_rfci(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticPagDiscoveryResult, CausalError> {
    engines::discover_rfci(data, variables, params, ctx)
}

/// Run GES (Gaussian BIC) over tabular data → CPDAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_ges(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<StaticCpdagDiscoveryResult, CausalError> {
    engines::discover_ges(data, variables, params, ctx)
}

/// Run `DirectLiNGAM` over tabular data → DAG.
///
/// # Errors
///
/// Discovery failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_lingam(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    prune_threshold: f64,
    ctx: &ExecutionContext,
) -> Result<StaticDagDiscoveryResult, CausalError> {
    engines::discover_lingam(data, variables, params, prune_threshold, ctx)
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
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
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
    engines::discover_notears_with(
        data,
        variables,
        params,
        lambda,
        threshold,
        standardize,
        missing,
        ctx,
    )
}

/// Discover a static DAG with NOTEARS (continuous SEM), refusing a table with a missing value.
///
/// [`discover_notears_with`] with no declared missing-data policy.
///
/// # Errors
///
/// Discovery / solver failures, and `missing_policy_required` for a missing value.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_notears(
    data: &TabularData,
    variables: &[VariableId],
    params: &StaticDiscoverParams,
    lambda: f64,
    threshold: f64,
    standardize: bool,
    ctx: &ExecutionContext,
) -> Result<NotearsDiscoveryResult, CausalError> {
    engines::discover_notears(data, variables, params, lambda, threshold, standardize, ctx)
}

/// Exact DAG posterior enumeration (`n ≤ 6`, Gaussian BIC).
///
/// # Errors
///
/// Discovery failures (oversized graph, empty support, score/data errors).
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_exact_dag_posterior(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_exact_dag_posterior(data, variables, params, ctx)
}

/// Exact posterior over PAG classes of the Gaussian maximal ancestral graphs of at most four
/// variables, scored by BIC.
///
/// # Errors
///
/// More than four variables, missing or non-finite values, or no finite score.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_exact_pag_posterior(
    data: &TabularData,
    variables: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_exact_pag_posterior(data, variables, ctx)
}

/// Metropolis posterior over PAG classes of the Gaussian maximal ancestral graphs of two to
/// eight variables.
///
/// # Errors
///
/// A bad schedule or variable count, missing or non-finite values, or no scorable graph.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pag_mcmc(
    data: &TabularData,
    variables: &[VariableId],
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_pag_mcmc(data, variables, schedule, ctx)
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
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_exact_dag_marginals(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<ExactDagMarginals, CausalError> {
    engines::discover_exact_dag_marginals(data, variables, params, ctx)
}

/// Exact DAG posterior pooled across imputations: the exact posterior of every completed table
/// (structure searched afresh on each), mixed with equal weight.
///
/// Every table must hold the same variables in the same order. The pooled mass of a graph is
/// the mean of its per-imputation masses; the posterior is tagged
/// `pooled_imputations(exact_dag_posterior)` and its diagnostics state the method.
///
/// # Errors
///
/// No table, a discovery failure on any table, or posteriors that do not pool.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pooled_exact_dag_posterior(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_pooled_exact_dag_posterior(tables, variables, params, ctx)
}

/// Exact DAG edge marginals pooled across imputations: the means of the per-imputation
/// marginals, and the log of the mean of the per-imputation normalizers.
///
/// # Errors
///
/// No table, a discovery failure on any table, or marginals that do not pool.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pooled_exact_dag_marginals(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<ExactDagMarginals, CausalError> {
    engines::discover_pooled_exact_dag_marginals(tables, variables, params, ctx)
}

/// Order MCMC pooled across imputations: the chains run on every completed table.
///
/// # Errors
///
/// No table, a discovery failure on any table (including the diagnostics gate), or posteriors
/// that do not pool.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pooled_order_mcmc(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    require_diagnostics_gate: bool,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_pooled_order_mcmc(
        tables,
        variables,
        params,
        schedule,
        require_diagnostics_gate,
        ctx,
    )
}

/// Structure MCMC pooled across imputations.
///
/// # Errors
///
/// No table, a discovery failure on any table, or posteriors that do not pool.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pooled_structure_mcmc(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_pooled_structure_mcmc(tables, variables, params, schedule, ctx)
}

/// CI-screened posterior pooled across imputations: the screen and the chains run on every
/// completed table.
///
/// # Errors
///
/// No table, a discovery failure on any table, or posteriors that do not pool.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pooled_ci_screened_posterior(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    screen: &StaticDiscoverParams,
    schedule: &GraphMcmcSchedule,
    soft_weight: CiSoftWeight,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_pooled_ci_screened_posterior(
        tables,
        variables,
        params,
        screen,
        schedule,
        soft_weight,
        ctx,
    )
}

/// Bounded-lag DBN template posterior pooled across completed series.
///
/// # Errors
///
/// No series, a discovery failure on any series, or posteriors that do not pool.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_pooled_dbn_posterior(
    series: &[TimeSeriesData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    max_lag: u32,
    force_mcmc: bool,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_pooled_dbn_posterior(
        series, variables, params, max_lag, force_mcmc, schedule, ctx,
    )
}

/// Order MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_order_mcmc(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    require_diagnostics_gate: bool,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_order_mcmc(data, variables, params, schedule, require_diagnostics_gate, ctx)
}

/// Structure MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_structure_mcmc(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_structure_mcmc(data, variables, params, schedule, ctx)
}

/// CI-screened candidate-edge posterior (PC skeleton → structure MCMC).
///
/// # Errors
///
/// Screening, empty skeleton, or MCMC failures.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_ci_screened_posterior(
    data: &TabularData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    screen: &StaticDiscoverParams,
    schedule: &GraphMcmcSchedule,
    soft_weight: CiSoftWeight,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_ci_screened_posterior(
        data,
        variables,
        params,
        screen,
        schedule,
        soft_weight,
        ctx,
    )
}

/// Screen a wide variable set by conditional independence, then compute the exact edge
/// marginals of at most [`EXACT_MARGINAL_MAX_NODES`] variables around `focus`, with the screen's
/// edges as per-edge prior probabilities. Every marginal is conditional on the screen.
///
/// # Errors
///
/// Screening failures, an empty or unknown `focus`, a set size above the exact ceiling,
/// probabilities outside `(0, 1)`, or the exact computation's refusals.
#[doc(hidden)]
#[allow(clippy::too_many_arguments, reason = "one screened question, stated field by field")]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_screened_exact_marginals(
    data: &TabularData,
    variables: &[VariableId],
    focus: &[VariableId],
    max_variables: usize,
    params: &BayesianDiscoverParams,
    screen: &StaticDiscoverParams,
    in_probability: f64,
    out_probability: f64,
    ctx: &ExecutionContext,
) -> Result<ExactDagMarginals, CausalError> {
    engines::discover_screened_exact_marginals(
        data,
        variables,
        focus,
        max_variables,
        params,
        screen,
        in_probability,
        out_probability,
        ctx,
    )
}

/// Bounded-lag DBN template posterior from a time series (Gaussian BIC).
///
/// # Errors
///
/// Short series, unsupported size, score, or empty support.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_dbn_posterior(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    max_lag: u32,
    force_mcmc: bool,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_dbn_posterior(data, variables, params, max_lag, force_mcmc, schedule, ctx)
}

/// Infer the DBN template posterior over the units of a panel (see [`discover_pcmci_panel`]).
///
/// # Errors
///
/// No usable rows, unsupported size, score, or empty support.
#[doc(hidden)]
#[deprecated(
    since = "3.1.0",
    note = "runs the search with no contract, claim or provenance; use the compiled route: `StudyBuilder::discovery` with a `DiscoverySpec`, then `prepare` and `estimate`"
)]
pub fn discover_dbn_posterior_panel(
    units: &[TimeSeriesData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    max_lag: u32,
    force_mcmc: bool,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    engines::discover_dbn_posterior_panel(
        units, variables, params, max_lag, force_mcmc, schedule, ctx,
    )
}

/// The posterior pooled across imputations: the equal-weight mixture of the per-imputation
/// posteriors (draw sets, for a sampled search), whichever search produced them.
///
/// The pooled posterior states the method in its algorithm tag and diagnostics notes and keeps
/// each imputation's Monte-Carlo diagnostics; see [`GraphPosterior::pooled_over_imputations`].
///
/// # Errors
///
/// No posterior, or posteriors that do not pool.
pub fn pool_posteriors_over_imputations(
    posteriors: &[GraphPosterior],
) -> Result<GraphPosterior, CausalError> {
    Ok(GraphPosterior::pooled_over_imputations(posteriors)?)
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

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Discovery errors.
pub use antecedent_discovery::DiscoveryError;
/// Shared PCMCI engine: PC-style parents + MCI.
pub use antecedent_discovery::DiscoveryWorkspace;
/// Discovery result types.
pub use antecedent_discovery::LaggedLink;
/// Public lagged PCMCI algorithm.
pub use antecedent_discovery::Pcmci;
/// PCMCI+ returning a temporal CPDAG.
pub use antecedent_discovery::PcmciPlus;
/// Graph prior / posterior types and Bayesian discovery engine trait.
pub use antecedent_discovery::n_directed_edges;
/// Packed directed-edge set of one graph-posterior atom.
pub use antecedent_discovery::{AdjacencyMask, LagMask};
/// Packed directed-edge set of one graph-posterior atom.
pub mod adjacency_mask {
    pub use antecedent_discovery::adjacency_mask::{
        adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag, has_edge,
        mask_is_dag,
    };
}
/// Discovery constraints with compiled masks (§13.4 / §13.8).
pub mod constraints {
    pub use antecedent_discovery::constraints::{CiMaskType, DiscoveryConstraints};
}
/// Selective discovery: reuse of conditional-independence statistics across discovery requests.
pub mod selective {
    pub use antecedent_discovery::selective::{
        Mark, StructureEdge, StructureNode, request_downstream_interval, request_recovery_claim,
        request_score_search_reuse,
    };
}
/// Lagged temporal-ADMG posterior atoms.
pub mod temporal_admg_atoms {
    pub use antecedent_discovery::temporal_admg_atoms::with_lagged_edge;
}

/// Signed-uniform structural graph probabilities for a partially observed table.
pub mod observed_marginals {
    pub use antecedent_discovery::observed_marginals::{
        ObservedMarginals, signed_uniform_observed_marginals,
    };
}
