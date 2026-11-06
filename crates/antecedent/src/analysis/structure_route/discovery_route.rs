//! Discovery search dispatch and admission.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CELL_BYTES, CausalError, CiSoftWeight, DiscoverData, DiscoverParams, DiscoverQuery,
    DiscoveredStructure, DiscoveryMethod, DiscoverySpec, ExecutionContext, GraphMcmcSchedule,
    PanelData, PanelUnit, ScreenWeight, SearchParams, StaticDiscoverParams, TabularData,
    TimeSeriesData, VariableId, saturating_u32,
};

/// The units of a panel as the per-unit series a panel search reads.
fn panel_units(panel: &PanelData) -> Vec<TimeSeriesData> {
    panel.units().iter().map(|unit| unit.series.clone()).collect()
}

/// A panel of `units`, unit `i` labelled `i`.
///
/// # Errors
///
/// No unit, or units whose schemas differ.
pub fn panel_of_units(units: Vec<TimeSeriesData>) -> Result<PanelData, CausalError> {
    let units: Vec<PanelUnit> = units
        .into_iter()
        .enumerate()
        .map(|(index, series)| PanelUnit { unit_id: saturating_u32(index), series })
        .collect();
    PanelData::try_new(units).map_err(CausalError::rejected)
}

/// Bytes the search leases before it runs: the lagged frame it builds and the
/// link, separating-set and draw tables it keeps.
pub(crate) fn discovery_quote(query: &DiscoverQuery, rows: u64) -> u64 {
    let width = query.variables.len() as u64;
    let lags = u64::from(query.bounds.max_lag).saturating_add(1);
    let frame = rows.saturating_mul(width).saturating_mul(lags).saturating_mul(CELL_BYTES);
    let nodes = width.saturating_mul(lags);
    let links = nodes.saturating_mul(nodes).saturating_mul(64);
    let draws = query.bounds.schedule.map_or(0, |schedule| {
        u64::from(schedule.chains)
            .saturating_mul(u64::from(schedule.draws))
            .saturating_mul(width.saturating_mul(width).div_ceil(8).saturating_add(16))
    });
    frame.saturating_add(links).saturating_add(draws)
}

/// Lease `bytes` from the context's admission pool.
pub(crate) fn admit(
    ctx: &ExecutionContext,
    bytes: u64,
    what: &'static str,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: what }
        }
        other => CausalError::Resource {
            message: format!(
                "the {what} needs {bytes} bytes and the admission pool refused them: {other}; \
                 next attempt: narrow the request or raise the pool"
            ),
        },
    })
}

fn wrong_data(method: &DiscoveryMethod, data: &DiscoverData<'_>) -> CausalError {
    crate::compile_reason!(
        "data_modality_not_supported",
        "the {} search does not run on {} data",
        method.id(),
        data.modality()
    )
}

fn schedule_of(spec: &DiscoverySpec) -> Result<GraphMcmcSchedule, CausalError> {
    spec.schedule.ok_or_else(|| {
        crate::compile_reason!(
            "required_option_missing",
            "the {} posterior needs an MCMC schedule",
            spec.method.id()
        )
    })
}

/// Run the search `spec` states over `data`, with the engine every
/// `discover_*` entry runs.
pub(crate) fn run_discovery(
    spec: &DiscoverySpec,
    query: &DiscoverQuery,
    data: &DiscoverData<'_>,
    design: Option<antecedent_core::DeclaredSamplingDesign>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredStructure, CausalError> {
    if let Some(inner) = spec.unwrapped() {
        // A wrapper runs the search it wraps and reads its answer.
        return match spec.stability_replicates() {
            Some(replicates) => super::super::structure_stability::run_stability(
                &inner, replicates, query, data, design, ctx,
            ),
            None => match run_discovery(&inner, query, data, design, ctx)? {
                DiscoveredStructure::Posterior(posterior) => {
                    Ok(DiscoveredStructure::Posterior(posterior.to_equivalence_classes()?))
                }
                other => Err(crate::compile_reason!(
                    "route_not_supported",
                    "the {} search returned no listed graphs to collapse into equivalence \
                     classes (it returned {})",
                    spec.method().id(),
                    other.algorithm_id().unwrap_or("no structure")
                )),
            },
        };
    }
    let search = Search { spec, vars: &query.variables, data, ctx };
    for algorithm in DISCOVERY_ALGORITHMS {
        if let Some(found) = algorithm(&search)? {
            return Ok(found);
        }
    }
    Err(wrong_data(&spec.method, data))
}

/// The request one discovery algorithm reads.
struct Search<'a> {
    spec: &'a DiscoverySpec,
    vars: &'a [VariableId],
    data: &'a DiscoverData<'a>,
    ctx: &'a ExecutionContext,
}

/// One algorithm's entry: `Ok(None)` when the request's method, parameter family or data
/// modality is not this algorithm's cell, so the next entry is tried.
type DiscoveryAlgorithm = fn(&Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError>;

/// Every algorithm cell, tried in order; adding an algorithm adds one entry here.
const DISCOVERY_ALGORITHMS: &[DiscoveryAlgorithm] = &[
    run_pc,
    run_fci,
    run_rfci,
    run_ges,
    run_lingam,
    run_notears,
    run_pcmci,
    run_pcmci_plus,
    run_lpcmci,
    run_jpcmci_plus,
    run_rpcmci,
    run_exact_dag_posterior,
    run_exact_pag_posterior,
    run_pag_mcmc,
    run_exact_dag_marginals,
    run_order_mcmc,
    run_structure_mcmc,
    run_ci_screened_posterior,
    run_dbn_posterior,
];

type TabularEngine<P, R> =
    fn(&TabularData, &[VariableId], &P, &ExecutionContext) -> Result<R, CausalError>;

/// A static search run over tabular rows.
fn static_tabular<R>(
    search: &Search<'_>,
    method_matches: bool,
    engine: TabularEngine<StaticDiscoverParams, R>,
    wrap: fn(R) -> DiscoveredStructure,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (true, SearchParams::Static(p), DiscoverData::Tabular(d)) =
        (method_matches, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    Ok(Some(wrap(engine(d, search.vars, p, search.ctx)?)))
}

fn run_pc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Pc),
        crate::discovery::discover_pc,
        DiscoveredStructure::StaticCpdag,
    )
}

fn run_fci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Fci),
        crate::discovery::discover_fci,
        DiscoveredStructure::StaticPag,
    )
}

fn run_rfci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Rfci),
        crate::discovery::discover_rfci,
        DiscoveredStructure::StaticPag,
    )
}

fn run_ges(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Ges { .. }),
        crate::discovery::discover_ges,
        DiscoveredStructure::StaticCpdag,
    )
}

fn run_lingam(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::DirectLingam { prune_threshold },
        SearchParams::Static(p),
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_lingam(d, search.vars, p, *prune_threshold, search.ctx)?;
    Ok(Some(DiscoveredStructure::StaticDag(found)))
}

fn run_notears(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::Notears { lambda, threshold, standardize },
        SearchParams::Static(p),
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_notears_with(
        d,
        search.vars,
        p,
        *lambda,
        *threshold,
        *standardize,
        search.spec.missing,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Notears(found)))
}

/// A temporal search that runs on one series or on a panel's per-unit series.
fn temporal_series_or_panel<R>(
    search: &Search<'_>,
    method_matches: bool,
    on_series: SeriesEngine<R>,
    on_panel: PanelEngine<R>,
    wrap: fn(R) -> DiscoveredStructure,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (true, SearchParams::Temporal(p)) = (method_matches, &search.spec.search) else {
        return Ok(None);
    };
    let found = match search.data {
        DiscoverData::Series(s) => on_series(s, search.vars, p, search.ctx)?,
        DiscoverData::Panel(panel) => on_panel(&panel_units(panel), search.vars, p, search.ctx)?,
        _ => return Ok(None),
    };
    Ok(Some(wrap(found)))
}

type SeriesEngine<R> = fn(
    &TimeSeriesData,
    &[VariableId],
    &DiscoverParams,
    &ExecutionContext,
) -> Result<R, CausalError>;
type PanelEngine<R> = fn(
    &[TimeSeriesData],
    &[VariableId],
    &DiscoverParams,
    &ExecutionContext,
) -> Result<R, CausalError>;

fn run_pcmci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    temporal_series_or_panel(
        search,
        matches!(search.spec.method, DiscoveryMethod::Pcmci),
        engine::discover_pcmci,
        engine::discover_pcmci_panel,
        DiscoveredStructure::TemporalDag,
    )
}

fn run_pcmci_plus(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    temporal_series_or_panel(
        search,
        matches!(search.spec.method, DiscoveryMethod::PcmciPlus),
        engine::discover_pcmci_plus,
        engine::discover_pcmci_plus_panel,
        DiscoveredStructure::TemporalCpdag,
    )
}

fn run_lpcmci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    temporal_series_or_panel(
        search,
        matches!(search.spec.method, DiscoveryMethod::Lpcmci),
        engine::discover_lpcmci,
        engine::discover_lpcmci_panel,
        DiscoveredStructure::TemporalPag,
    )
}

fn run_jpcmci_plus(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::JpcmciPlus, SearchParams::Temporal(p), DiscoverData::MultiEnv(m)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_jpcmci_plus(m, search.vars, p, search.ctx)?;
    Ok(Some(DiscoveredStructure::TemporalCpdag(found)))
}

fn run_rpcmci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::Rpcmci { .. }, SearchParams::Temporal(p), DiscoverData::Series(s)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let regimes = search.spec.regimes.as_ref().ok_or_else(|| {
        crate::compile_reason!("required_option_missing", "regime PCMCI needs a regime assignment")
    })?;
    let found = crate::discovery::discover_rpcmci(
        s,
        search.vars,
        regimes,
        p,
        search.spec.min_regime_len,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Regimes(found)))
}

fn run_exact_dag_posterior(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::ExactDagPosterior, SearchParams::Bayesian(b), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_exact_dag_posterior(d, search.vars, b, search.ctx)?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_exact_pag_posterior(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::ExactPagPosterior, SearchParams::Bayesian(_), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_exact_pag_posterior(d, search.vars, search.ctx)?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_pag_mcmc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::PagMcmc, SearchParams::Bayesian(_), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_pag_mcmc(
        d,
        search.vars,
        &schedule_of(search.spec)?,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_exact_dag_marginals(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::ExactDagMarginals, SearchParams::Bayesian(b), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_exact_dag_marginals(d, search.vars, b, search.ctx)?;
    Ok(Some(DiscoveredStructure::EdgeMarginals(found)))
}

fn run_order_mcmc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::OrderMcmc { diagnostics_gate },
        SearchParams::Bayesian(b),
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_order_mcmc(
        d,
        search.vars,
        b,
        &schedule_of(search.spec)?,
        *diagnostics_gate,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_structure_mcmc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::StructureMcmc, SearchParams::Bayesian(b), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_structure_mcmc(
        d,
        search.vars,
        b,
        &schedule_of(search.spec)?,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_ci_screened_posterior(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::CiScreenedPosterior { weight },
        SearchParams::Screened { bayesian, screen },
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let soft_weight = match weight {
        ScreenWeight::None => CiSoftWeight::None,
        ScreenWeight::BayesFactor => CiSoftWeight::BayesFactor,
        ScreenWeight::PosteriorDependence => CiSoftWeight::PosteriorDependence,
    };
    let found = crate::discovery::discover_ci_screened_posterior(
        d,
        search.vars,
        bayesian,
        screen,
        &schedule_of(search.spec)?,
        soft_weight,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_dbn_posterior(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    let (DiscoveryMethod::DbnPosterior { force_mcmc }, SearchParams::Bayesian(b)) =
        (&search.spec.method, &search.spec.search)
    else {
        return Ok(None);
    };
    let spec = search.spec;
    let found = match search.data {
        DiscoverData::Series(s) => engine::discover_dbn_posterior(
            s,
            search.vars,
            b,
            spec.max_lag,
            *force_mcmc,
            &schedule_of(spec)?,
            search.ctx,
        )?,
        DiscoverData::Panel(panel) => engine::discover_dbn_posterior_panel(
            &panel_units(panel),
            search.vars,
            b,
            spec.max_lag,
            *force_mcmc,
            &schedule_of(spec)?,
            search.ctx,
        )?,
        _ => return Ok(None),
    };
    Ok(Some(DiscoveredStructure::Posterior(found)))
}
