//! The search engines the compiled structure-discovery route runs.
//!
//! Each engine builds the search from its parameters and runs it, returning the search's own
//! result with no contract, plan, claim or provenance. The compiled route calls these
//! directly; the hidden public `discover_*` functions of the parent module are deprecated
//! wrappers over them.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{MultiEnvironmentData, TabularData, TimeSeriesData};
use antecedent_discovery::DiscoveryConstraints;
use antecedent_discovery::{
    BackgroundKnowledge, CiScreenedPosterior, CiSoftWeight, CpdagDiscoveryResult,
    DagDiscoveryResult, DbnPosterior, DirectLingam, DiscoveryWorkspace, ExactDagMarginals,
    ExactDagPosterior, Fci, Ges, GraphPosterior, JpcmciPlus, Lpcmci, Notears,
    NotearsDiscoveryResult, OrderMcmc, PagDiscoveryResult, Pc, Pcmci, PcmciPlus, RegimeAssignment,
    Rfci, Rpcmci, RpcmciDiscoveryResult, StaticCpdagDiscoveryResult, StaticDagDiscoveryResult,
    StaticPagDiscoveryResult, StructureMcmc,
};
use antecedent_stats::FdrAdjustment;

use super::{BayesianDiscoverParams, DiscoverParams, GraphMcmcSchedule, StaticDiscoverParams};
use crate::discovery_defaults::{
    DEFAULT_RPCMCI_MIN_REGIME_LEN, contemporaneous_constraints, jpcmci_constraints,
    pcmci_constraints, static_pc_constraints,
};
use crate::error::CausalError;

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
pub(crate) fn discover_pcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<DagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(pcmci_for(params).run(data, variables, &mut ws, ctx)?)
}

/// The PCMCI engine a temporal search configured from `params` runs.
pub(crate) fn pcmci_for(params: &DiscoverParams) -> Pcmci {
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
pub(crate) fn discover_pcmci_panel(
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
pub(crate) fn discover_pcmci_plus(
    data: &TimeSeriesData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(pcmci_plus_for(params).run(data, variables, &mut ws, ctx)?)
}

/// The PCMCI+ engine a temporal search configured from `params` runs.
pub(crate) fn pcmci_plus_for(params: &DiscoverParams) -> PcmciPlus {
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
pub(crate) fn discover_pcmci_plus_panel(
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
pub(crate) fn discover_lpcmci(
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
pub(crate) fn discover_lpcmci_panel(
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
pub(crate) fn discover_jpcmci_plus(
    data: &MultiEnvironmentData,
    variables: &[VariableId],
    params: &DiscoverParams,
    ctx: &ExecutionContext,
) -> Result<CpdagDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(jpcmci_plus_for(params).run(data, variables, &mut ws, ctx)?)
}

/// The J-PCMCI+ engine a multi-environment search configured from `params` runs.
pub(crate) fn jpcmci_plus_for(params: &DiscoverParams) -> JpcmciPlus {
    temporal_discoverer!(
        JpcmciPlus::new(),
        params,
        jpcmci_constraints(
            params.max_lag,
            params.alpha,
            params.max_cond_size,
            params.multi_dataset.clone(),
        )
    )
}

/// Run RPCMCI with a supplied regime assignment.
///
/// # Errors
///
/// Discovery failures.
pub(crate) fn discover_rpcmci(
    data: &TimeSeriesData,
    variables: &[VariableId],
    assignment: &RegimeAssignment,
    params: &DiscoverParams,
    min_regime_len: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<RpcmciDiscoveryResult, CausalError> {
    let mut ws = DiscoveryWorkspace::default();
    Ok(rpcmci_for(params, min_regime_len).run(data, variables, assignment, &mut ws, ctx)?)
}

/// The regime-PCMCI engine a search configured from `params` and `min_regime_len` runs.
pub(crate) fn rpcmci_for(params: &DiscoverParams, min_regime_len: Option<usize>) -> Rpcmci {
    let plus = temporal_discoverer!(
        PcmciPlus::new(),
        params,
        contemporaneous_constraints(params.max_lag, params.alpha, params.max_cond_size)
    );
    Rpcmci::new()
        .with_min_regime_len(min_regime_len.unwrap_or(DEFAULT_RPCMCI_MIN_REGIME_LEN))
        .with_pcmci_plus(plus)
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
pub(crate) fn discover_pc(
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
pub(crate) fn discover_fci(
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
pub(crate) fn discover_rfci(
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
pub(crate) fn discover_ges(
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
pub(crate) fn discover_lingam(
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
pub(crate) fn discover_notears_with(
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
pub(crate) fn discover_notears(
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
pub(crate) fn discover_exact_dag_posterior(
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
pub(crate) fn discover_exact_pag_posterior(
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
pub(crate) fn discover_pag_mcmc(
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
/// Above `EXACT_ENUM_MAX_NODES` the enumerating engine cannot list the graphs, but the
/// marginal `P(i → j)` of every ordered pair and the normalizing constant are still exact
/// (sink recursion over node subsets). Graph atoms, a MAP graph and any quantity that needs the
/// list of graphs are not computed and are not approximated.
///
/// # Errors
///
/// A visit quote or table the execution does not hold, invalid constraints, score/data errors.
pub(crate) fn discover_exact_dag_marginals(
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
pub(crate) fn discover_pooled_exact_dag_posterior(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    if params.score_family == antecedent_state::GraphScoreFamily::GaussianSignedUniform {
        return Err(antecedent_discovery::DiscoveryError::unsupported(
            "GaussianSignedUniform requires complete observations; the model has no observed-data likelihood for pooled imputations",
        ).into());
    }
    let posteriors = tables
        .iter()
        .map(|table| discover_exact_dag_posterior(table, variables, params, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphPosterior::pooled_over_imputations(&posteriors)?)
}

/// Exact DAG edge marginals pooled across imputations: the means of the per-imputation
/// marginals, and the log of the mean of the per-imputation normalizers.
///
/// # Errors
///
/// No table, a discovery failure on any table, or marginals that do not pool.
pub(crate) fn discover_pooled_exact_dag_marginals(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    ctx: &ExecutionContext,
) -> Result<ExactDagMarginals, CausalError> {
    if params.score_family == antecedent_state::GraphScoreFamily::GaussianSignedUniform {
        return Err(antecedent_discovery::DiscoveryError::unsupported(
            "GaussianSignedUniform requires complete observations; the model has no observed-data likelihood for pooled imputations",
        ).into());
    }
    let parts = tables
        .iter()
        .map(|table| discover_exact_dag_marginals(table, variables, params, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ExactDagMarginals::pooled_over_imputations(&parts)?)
}

/// Order MCMC pooled across imputations: the chains run on every completed table.
///
/// # Errors
///
/// No table, a discovery failure on any table (including the diagnostics gate), or posteriors
/// that do not pool.
pub(crate) fn discover_pooled_order_mcmc(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    require_diagnostics_gate: bool,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let posteriors = tables
        .iter()
        .map(|table| {
            discover_order_mcmc(table, variables, params, schedule, require_diagnostics_gate, ctx)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphPosterior::pooled_over_imputations(&posteriors)?)
}

/// Structure MCMC pooled across imputations.
///
/// # Errors
///
/// No table, a discovery failure on any table, or posteriors that do not pool.
pub(crate) fn discover_pooled_structure_mcmc(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let posteriors = tables
        .iter()
        .map(|table| discover_structure_mcmc(table, variables, params, schedule, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphPosterior::pooled_over_imputations(&posteriors)?)
}

/// CI-screened posterior pooled across imputations: the screen and the chains run on every
/// completed table.
///
/// # Errors
///
/// No table, a discovery failure on any table, or posteriors that do not pool.
pub(crate) fn discover_pooled_ci_screened_posterior(
    tables: &[TabularData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    screen: &StaticDiscoverParams,
    schedule: &GraphMcmcSchedule,
    soft_weight: CiSoftWeight,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let posteriors = tables
        .iter()
        .map(|table| {
            discover_ci_screened_posterior(
                table,
                variables,
                params,
                screen,
                schedule,
                soft_weight,
                ctx,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphPosterior::pooled_over_imputations(&posteriors)?)
}

/// Bounded-lag DBN template posterior pooled across completed series.
///
/// # Errors
///
/// No series, a discovery failure on any series, or posteriors that do not pool.
pub(crate) fn discover_pooled_dbn_posterior(
    series: &[TimeSeriesData],
    variables: &[VariableId],
    params: &BayesianDiscoverParams,
    max_lag: u32,
    force_mcmc: bool,
    schedule: &GraphMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, CausalError> {
    let posteriors = series
        .iter()
        .map(|one| {
            discover_dbn_posterior(one, variables, params, max_lag, force_mcmc, schedule, ctx)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphPosterior::pooled_over_imputations(&posteriors)?)
}

/// Order MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
pub(crate) fn discover_order_mcmc(
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
    let posterior = eng.run(data, variables, &params.prior, params.score_family, &mut ws, ctx)?;
    // A problem too large for the order sampler is routed to structure MCMC and tagged so.
    if posterior.algorithm().is_some_and(|tag| &**tag == "order_mcmc_via_structure_mcmc") {
        return Ok(posterior);
    }
    Ok(posterior.with_algorithm("order_mcmc"))
}

/// Structure MCMC DAG posterior (Gaussian BIC).
///
/// # Errors
///
/// Discovery / diagnostics-gate failures.
pub(crate) fn discover_structure_mcmc(
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
pub(crate) fn discover_ci_screened_posterior(
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
        // A test that fails to reject a weak edge has shown low power, not independence: keep
        // the pair proposable unless the marginal likelihood decisively supports its absence,
        // so the published credible set can reach the true structure at every sample size.
        .with_retain_undecided(true)
        .with_mcmc(mcmc);
    // Preserve FDR from screen (CiScreenedPosterior::new sets a default).
    let mut eng = eng;
    eng.fdr = fdr;
    let mut ws = DiscoveryWorkspace::default();
    Ok(eng
        .run(data, variables, &params.prior, params.score_family, &mut ws, ctx)?
        .with_algorithm("ci_screened_posterior"))
}

/// Screen a wide variable set by conditional independence, then compute the exact edge
/// marginals of at most `EXACT_MARGINAL_MAX_NODES` variables around `focus`, with the screen's
/// edges as per-edge prior probabilities. Every marginal is conditional on the screen.
///
/// # Errors
///
/// Screening failures, an empty or unknown `focus`, a set size above the exact ceiling,
/// probabilities outside `(0, 1)`, or the exact computation's refusals.
#[allow(clippy::too_many_arguments, reason = "one screened question, stated field by field")]
pub(crate) fn discover_screened_exact_marginals(
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
    let eng = CiScreenedPosterior::new()
        .with_constraints(static_pc_constraints(screen.alpha, screen.max_cond_size))
        .with_ci(Arc::clone(&screen.ci))
        .with_retain_undecided(true);
    let mut eng = eng;
    eng.fdr = static_fdr(screen);
    let mut ws = DiscoveryWorkspace::default();
    let screened = eng.screen_then_exact_marginals(
        data,
        variables,
        focus,
        max_variables,
        &params.prior,
        in_probability,
        out_probability,
        params.score_family,
        &mut ws,
        ctx,
    )?;
    Ok(screened.marginals)
}

/// Bounded-lag DBN template posterior from a time series (Gaussian BIC).
///
/// # Errors
///
/// Short series, unsupported size, score, or empty support.
pub(crate) fn discover_dbn_posterior(
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
pub(crate) fn discover_dbn_posterior_panel(
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
