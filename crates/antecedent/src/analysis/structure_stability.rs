//! Sampling distribution of a frequentist structure search, through the compiler.
//!
//! The search runs once on the study's data (the reference) and then on `replicates`
//! resamples drawn under the declared sampling design: independent subjects draw rows with
//! replacement, a dependent series draws circular blocks, and a panel draws whole units.
//! The structure each run returns is tabulated by [`StructureStability`]: how often every
//! node pair was adjacent, carried every pair of endpoint marks, and how often every whole
//! structure came back. A resample on which the search fails is kept as failed mass; it is
//! never redrawn. Each resample has its own keyed stream, so the answer does not depend on
//! the thread count.
//!
//! The table is a bootstrap distribution of the search. It is not a posterior and it is not
//! one structure, so it carries no review gate. It executes unlicensed and its coverage is
//! unmeasured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalRng, DeclaredSamplingDesign, DiscoverQuery, ExecutionContext, StreamDomain, VariableId,
};
use antecedent_data::{
    MultiEnvironmentData, PanelData, ResamplingPlan, TableView, TabularData, TimeSeriesData,
    circular_block_length, fill_circular_block_indexes, resample_timeseries,
};
use antecedent_discovery::selective::DiscoveredStructure as CanonicalStructure;
use antecedent_discovery::{RegimeAssignment, StructureStability};
use antecedent_kernels::unbiased_index;

use super::structure_route::{
    DiscoverData, DiscoveredStructure, DiscoverySpec, admit, panel_of_units, run_discovery,
};
use crate::error::CausalError;

/// Stage name of a stability search, for cancellation.
const STABILITY_STAGE: &str = "discover.stability";
/// Base key of the per-resample streams.
const STREAM_KEY: u64 = 0x5354_4142_494C_4954;
/// The golden-ratio increment that spreads resample indexes over the stream space.
const STREAM_SPREAD: u64 = 0x9E37_79B9_7F4A_7C15;
/// Bytes of one resampled cell.
const CELL_BYTES: u64 = 8;

/// One resample of the study's data.
enum Resample {
    Table(TabularData),
    Series(TimeSeriesData),
    Panel(PanelData),
    Environments(MultiEnvironmentData),
}

impl Resample {
    fn view(&self) -> DiscoverData<'_> {
        match self {
            Self::Table(data) => DiscoverData::Tabular(data),
            Self::Series(data) => DiscoverData::Series(data),
            Self::Panel(data) => DiscoverData::Panel(data),
            Self::Environments(data) => DiscoverData::MultiEnv(data),
        }
    }
}

/// Refuse a resample the declared sampling design does not allow.
///
/// With no declaration a table is independent subjects, one row each, a series is one
/// dependent series, and a panel is independent subject trajectories.
fn check_design(
    design: Option<DeclaredSamplingDesign>,
    data: &DiscoverData<'_>,
) -> Result<(), CausalError> {
    use antecedent_core::ResamplingRule;
    let (default, rule) = match data {
        DiscoverData::Tabular(_) | DiscoverData::Panel(_) => {
            (DeclaredSamplingDesign::IndependentSubjectTrajectories, ResamplingRule::RowSubjectIid)
        }
        // Each environment is its own dependent series, resampled in blocks on its own.
        DiscoverData::Series(_) | DiscoverData::MultiEnv(_) => {
            (DeclaredSamplingDesign::DependentTimeSeries, ResamplingRule::ContiguousBlock)
        }
    };
    let design = design.unwrap_or(default);
    let panel = matches!(data, DiscoverData::Panel(_));
    // A panel's units are drawn whole, which both subject and cluster designs allow.
    let allowed = if panel {
        design != DeclaredSamplingDesign::DependentTimeSeries
    } else {
        design.apply_resampling(rule).is_ok()
    };
    if allowed {
        Ok(())
    } else {
        Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the declared sampling design does not allow this resample: a table is resampled \
             by independent subjects, a series by contiguous blocks and a panel by whole \
             units. Next attempt: declare the design the data was sampled under, or search \
             once for the point structure"
        ))
    }
}

/// A row bootstrap of `data` over the searched columns.
fn resample_rows(
    data: &TabularData,
    variables: &[VariableId],
    rng: &mut CausalRng,
) -> Result<TabularData, CausalError> {
    let storage = data.storage();
    if storage.analysis_mask().is_some() || storage.weights().is_some() {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "a row resample of a masked or weighted table would change its population; next \
             attempt: resample the unmasked, unweighted rows the search reads"
        ));
    }
    let n = data.row_count();
    let rows: Vec<usize> = (0..n).map(|_| unbiased_index(rng, n)).collect();
    let mut replacements = Vec::with_capacity(variables.len());
    for &variable in variables {
        let column = data.float64_slice(variable)?;
        if column.iter().any(|value| !value.is_finite()) {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "a stability search resamples complete rows; next attempt: drop the rows with \
                 a missing searched value, then search and tabulate on the same rows"
            ));
        }
        let gathered: std::sync::Arc<[f64]> = rows.iter().map(|&row| column[row]).collect();
        replacements.push((variable, gathered));
    }
    Ok(data.with_replaced_floats(&replacements)?)
}

fn draw(
    data: &DiscoverData<'_>,
    query: &DiscoverQuery,
    rng: &mut CausalRng,
) -> Result<Resample, CausalError> {
    match data {
        DiscoverData::Tabular(table) => {
            Ok(Resample::Table(resample_rows(table, &query.variables, rng)?))
        }
        DiscoverData::Series(series) => {
            let rows = series.row_count();
            let span =
                usize::try_from(query.bounds.max_lag).unwrap_or(usize::MAX).saturating_add(1);
            let plan = ResamplingPlan::CircularBlock { length: circular_block_length(span, rows) };
            let mut scratch = Vec::new();
            Ok(Resample::Series(resample_timeseries(series, plan, rng, &mut scratch)?))
        }
        DiscoverData::Panel(panel) => {
            let units = panel.units();
            let drawn: Vec<TimeSeriesData> = (0..units.len())
                .map(|_| units[unbiased_index(rng, units.len())].series.clone())
                .collect();
            Ok(Resample::Panel(panel_of_units(drawn)?))
        }
        DiscoverData::MultiEnv(multi) => {
            let span =
                usize::try_from(query.bounds.max_lag).unwrap_or(usize::MAX).saturating_add(1);
            let mut scratch = Vec::new();
            let drawn = multi
                .environments()
                .iter()
                .map(|environment| {
                    let plan = ResamplingPlan::CircularBlock {
                        length: circular_block_length(span, environment.row_count()),
                    };
                    resample_timeseries(environment, plan, rng, &mut scratch)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Resample::Environments(MultiEnvironmentData::try_new(drawn)?))
        }
    }
}

/// The canonical structure a search returned, or `None` for a result with no single graph.
fn canonical(structure: &DiscoveredStructure) -> Option<CanonicalStructure> {
    Some(match structure {
        DiscoveredStructure::StaticCpdag(result) => {
            CanonicalStructure::from_cpdag(&result.evidence.graph)
        }
        DiscoveredStructure::StaticPag(result) => {
            CanonicalStructure::from_pag(&result.evidence.graph)
        }
        DiscoveredStructure::StaticDag(result) => {
            CanonicalStructure::from_dag(&result.evidence.graph)
        }
        DiscoveredStructure::Notears(result) => {
            CanonicalStructure::from_dag(&result.discovery.evidence.graph)
        }
        DiscoveredStructure::TemporalDag(result) => {
            CanonicalStructure::from_temporal_dag(&result.evidence.graph)
        }
        DiscoveredStructure::TemporalCpdag(result) => {
            CanonicalStructure::from_temporal_cpdag(&result.evidence.graph)
        }
        DiscoveredStructure::TemporalPag(result) => {
            CanonicalStructure::from_temporal_pag(&result.evidence.graph)
        }
        _ => return None,
    })
}

fn not_a_graph(base: &str) -> CausalError {
    crate::compile_reason!(
        "route_not_supported",
        "the {base} search returned no single graph to tabulate; a stability search re-runs \
         PC, FCI, RFCI, GES, LiNGAM, NOTEARS, PCMCI, PCMCI+, LPCMCI or J-PCMCI+"
    )
}

/// Run `base` on the full sample and on `replicates` resamples and tabulate what it returned.
///
/// `base` is the search with its stability wrapper removed.
///
/// # Errors
///
/// The full-sample search failing, a resample the declared design refuses, fewer than half
/// of the resamples returning a structure, or cancellation.
pub(crate) fn run_stability(
    base: &DiscoverySpec,
    replicates: u32,
    query: &DiscoverQuery,
    data: &DiscoverData<'_>,
    design: Option<DeclaredSamplingDesign>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredStructure, CausalError> {
    check_design(design, data)?;
    if matches!(base.method(), antecedent_core::DiscoveryMethod::Rpcmci { .. }) {
        return run_regime_stability(base, replicates, query, data, design, ctx);
    }
    let base_id = base.method().id();
    let lease_bytes = data
        .rows()
        .saturating_mul(query.variables.len() as u64)
        .saturating_mul(CELL_BYTES)
        .saturating_mul(2);
    let _lease = admit(ctx, lease_bytes, "structure stability resample")?;
    let reference = run_discovery(base, query, data, design, ctx)?;
    let reference = canonical(&reference).ok_or_else(|| not_a_graph(base_id))?;
    let mut structures = Vec::new();
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STABILITY_STAGE });
        }
        let mut rng = ctx.rng.stream_for(
            StreamDomain::Resample,
            STREAM_KEY.wrapping_add(u64::from(replicate).wrapping_mul(STREAM_SPREAD)),
        );
        let resample = match draw(data, query, &mut rng) {
            Ok(resample) => resample,
            // The design and the data refuse the same way on every resample: report it.
            Err(error) if replicate == 0 => return Err(error),
            Err(CausalError::Cancelled { stage }) => return Err(CausalError::Cancelled { stage }),
            Err(_) => continue,
        };
        match run_discovery(base, query, &resample.view(), design, ctx) {
            Ok(found) => structures.extend(canonical(&found)),
            Err(CausalError::Cancelled { stage }) => return Err(CausalError::Cancelled { stage }),
            // A failed resample is failed mass in the tabulation, never redrawn.
            Err(_) => {}
        }
    }
    let stability = StructureStability::tabulate(reference, &structures, replicates)?;
    Ok(DiscoveredStructure::Stability(stability))
}

/// The series with its rows gathered along `map`.
fn gather_series(series: &TimeSeriesData, map: &[usize]) -> Result<TimeSeriesData, CausalError> {
    let mut gathered = series.clone();
    for variable in series.schema().variables() {
        let column = series.float64_slice(variable.id)?;
        let values: std::sync::Arc<[f64]> = map.iter().map(|&row| column[row]).collect();
        gathered = gathered.with_replaced_float(variable.id, values)?;
    }
    Ok(gathered)
}

/// Regime PCMCI over circular-block resamples that carry their regime labels.
///
/// Each resample gathers the series and its labels along one block map, so a row keeps the
/// regime it was observed in, and the search is re-run on the gathered pair. One table is
/// tabulated per regime of the full-sample run; a resample that returns no graph for a regime
/// counts as failed mass for that regime.
fn run_regime_stability(
    base: &DiscoverySpec,
    replicates: u32,
    query: &DiscoverQuery,
    data: &DiscoverData<'_>,
    design: Option<DeclaredSamplingDesign>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredStructure, CausalError> {
    let DiscoverData::Series(series) = data else {
        return Err(crate::unsupported_reason!(
            "data_modality_not_licensed",
            "regime PCMCI labels the regimes of one series; next attempt: search one series"
        ));
    };
    let labels = base.regime_assignment().ok_or_else(|| {
        crate::compile_reason!("required_option_missing", "regime PCMCI needs a regime assignment")
    })?;
    let rows = series.row_count();
    if labels.regimes.len() != rows {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the regime assignment labels {} rows and the series has {rows}",
            labels.regimes.len()
        ));
    }
    let _lease = admit(
        ctx,
        data.rows()
            .saturating_mul(query.variables.len() as u64)
            .saturating_mul(CELL_BYTES)
            .saturating_mul(2),
        "structure stability resample",
    )?;
    let DiscoveredStructure::Regimes(reference) = run_discovery(base, query, data, design, ctx)?
    else {
        return Err(not_a_graph(base.method().id()));
    };
    let mut references: Vec<(antecedent_core::RegimeId, CanonicalStructure)> = reference
        .graphs
        .graphs
        .iter()
        .map(|(regime, graph)| (*regime, CanonicalStructure::from_temporal_cpdag(graph)))
        .collect();
    let mut collected: Vec<Vec<CanonicalStructure>> = vec![Vec::new(); references.len()];
    let span = usize::try_from(query.bounds.max_lag).unwrap_or(usize::MAX).saturating_add(1);
    let block = circular_block_length(span, rows);
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STABILITY_STAGE });
        }
        let mut rng = ctx.rng.stream_for(
            StreamDomain::Resample,
            STREAM_KEY.wrapping_add(u64::from(replicate).wrapping_mul(STREAM_SPREAD)),
        );
        let mut map = vec![0_usize; rows];
        fill_circular_block_indexes(rows, block, &mut rng, &mut map)?;
        let gathered = gather_series(series, &map)?;
        let regimes: Vec<antecedent_core::RegimeId> =
            map.iter().map(|&row| labels.regimes[row]).collect();
        let resampled = base.with_regime_assignment(RegimeAssignment::try_new(regimes)?);
        match run_discovery(&resampled, query, &DiscoverData::Series(&gathered), design, ctx) {
            Ok(DiscoveredStructure::Regimes(found)) => {
                for (slot, (regime, _)) in references.iter().enumerate() {
                    if let Some(graph) = found.graphs.get(*regime) {
                        collected[slot].push(CanonicalStructure::from_temporal_cpdag(graph));
                    }
                }
            }
            Err(CausalError::Cancelled { stage }) => return Err(CausalError::Cancelled { stage }),
            // A failed resample is failed mass in every regime's table, never redrawn.
            Ok(_) | Err(_) => {}
        }
    }
    let mut tables = Vec::with_capacity(references.len());
    for ((regime, structure), found) in references.drain(..).zip(&collected) {
        tables.push((regime, StructureStability::tabulate(structure, found, replicates)?));
    }
    Ok(DiscoveredStructure::RegimeStability(tables))
}
