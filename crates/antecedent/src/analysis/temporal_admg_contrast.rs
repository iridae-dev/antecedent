//! Pulse, sustained and conditional contrasts on a temporal ADMG, prepared on the window.
//!
//! The template is unrolled over the periods the policy needs
//! ([`antecedent_identify::temporal_admg`]), the contrast is identified on that window as a
//! functional over its coordinates, and the estimator reads one lag-aligned row of the series per
//! window position: the window table. Its column `period * variables + variable` holds that
//! variable at that period of the window, so the identified functional's variable ids are the
//! table's column ids and the unfolding indexer of the schema names them.
//!
//! Rows of the window table overlap: row `i` holds the series times `i .. i + periods`, so a
//! row-level resample or Dirichlet draw would treat shared observations as independent. The
//! estimator resamples (frequentist) or reweights (Bayesian) circular blocks of consecutive rows,
//! never shorter than the window, which is the declared sampling design of one dependent series.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    Lag, ResponseQuery, TargetPopulation, TemporalEffectQuery, TemporalIndexer, VariableId,
};
use antecedent_data::{
    LaggedFrame, LaggedFrameOptions, MaskPolicy, MissingPolicy, ReferencePointPolicy, TableView,
    TabularData, TimeSeriesData,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{TemporalAdmg, UnrolledTemporalAdmg};
use antecedent_identify::IdentificationResult;
use antecedent_identify::temporal_admg::{
    AdmittedTemporalAdmgWindowPlan, TemporalAdmgWindowPlan,
    identify_temporal_admg_contrast_over_within,
};

use crate::analysis::prepared::{
    CachedTemporalHorizonIdentification, CachedTemporalIdentification,
};
use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, select_estimand};

/// Fewest complete windows the plug-in is fitted on. Below this the empirical conditional tables
/// are not a law of the series, whatever the estimator.
pub(crate) const MIN_WINDOWS: usize = 8;

/// Next attempts named by every refusal of an unidentified or unsupported window contrast.
const NEXT_ATTEMPTS: &str = "next attempts: a TemporalDag when the lag structure has no latent \
     confounding, a shorter horizon or fewer treated periods so the window holds no hedge, an \
     observed proxy for the latent cause in the template, or the contemporaneous projection \
     (AverageEffect on the TemporalAdmg)";

/// A contrast identified on the unrolled window of a temporal ADMG.
#[derive(Clone, Debug)]
pub(crate) struct WindowContrast {
    /// Identification over the window coordinates (the claim the result carries).
    pub(crate) identification: IdentificationResult,
    /// The functional estimand the plug-in evaluates.
    pub(crate) estimand: IdentifiedEstimand,
    /// Where the policy, the outcome and the history sit in the window.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the tests read the plan; the run recomputes it from the template"
        )
    )]
    pub(crate) plan: AdmittedTemporalAdmgWindowPlan,
    /// Retain the identification owner's admitted schedule and supporting window.
    _identification_owner: antecedent_identify::temporal_admg::AdmittedTemporalAdmgIdentification,
    /// The unrolled window the functional is stated on.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the tests read the window; the run recomputes it from the template"
        )
    )]
    pub(crate) window: UnrolledTemporalAdmg,
    /// The window's variable namespace as the unfolding indexer of the schema it is numbered
    /// over: coordinate `period * variables + variable` is the node at offset `period - origin`.
    pub(crate) indexer: TemporalIndexer,
}

/// Admit the temporary template built from one posterior atom before its
/// quadratic contemporaneous scan and graph construction.
pub(crate) fn admit_temporal_atom_template(
    atom: &antecedent_discovery::TemporalAdmgAtom,
    n_vars: usize,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal atom" });
    }
    let masks = [&atom.contemporaneous, &atom.lagged_directed, &atom.lagged_bidirected];
    let words = masks
        .iter()
        .try_fold(0u64, |sum, mask| sum.checked_add(mask.words().len() as u64))
        .ok_or_else(|| CausalError::Resource { message: "temporal atom size overflow".into() })?;
    budget.charge_work(words).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let mut bits = 0u64;
    for mask in masks {
        for &word in mask.words() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal atom" });
            }
            bits = bits.checked_add(u64::from(word.count_ones())).ok_or_else(|| {
                CausalError::Resource { message: "temporal atom edge overflow".into() }
            })?;
        }
    }
    let n = n_vars as u64;
    let nodes = bits
        .checked_mul(2)
        .and_then(|v| v.checked_add(n))
        .ok_or_else(|| CausalError::Resource { message: "temporal atom nodes overflow".into() })?;
    if nodes > budget.limits().max_graph_nodes || bits > budget.limits().max_graph_edges {
        return Err(CausalError::Resource { message: "temporal atom graph limit".into() });
    }
    let work = n
        .checked_mul(n)
        .and_then(|v| {
            bits.checked_mul(nodes.checked_add(bits)?.checked_add(1)?)
                .and_then(|e| v.checked_add(e))
        })
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| CausalError::Resource { message: "temporal atom work overflow".into() })?;
    budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let bytes = nodes
        .checked_mul(512)
        .and_then(|v| bits.checked_mul(512).and_then(|e| v.checked_add(e)))
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| CausalError::Resource { message: "temporal atom bytes overflow".into() })?;
    if bytes > budget.limits().max_bytes {
        return Err(CausalError::Resource { message: "temporal atom byte limit".into() });
    }
    ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal atom" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })
}

/// A derivative identified by adjustment on the unrolled window of a temporal ADMG.
#[derive(Clone, Debug)]
pub(crate) struct WindowDerivative {
    /// Identification over the window coordinates (the claim the result carries).
    pub(crate) identification: IdentificationResult,
    /// The back-door estimand whose adjustment set the slope regression conditions on.
    pub(crate) estimand: IdentifiedEstimand,
    /// The window's variable namespace as the unfolding indexer of the schema it is numbered
    /// over, as for [`WindowContrast::indexer`].
    pub(crate) indexer: TemporalIndexer,
}

/// `query` restricted to the single horizon `horizon`: same functional, policy and history
/// bound, so each horizon of a multi-horizon derivative is identified and estimated alone.
///
/// # Errors
///
/// A query without a temporal attachment, or a horizon the attachment refuses.
pub(crate) fn single_horizon_query(
    query: &ResponseQuery,
    horizon: u32,
) -> Result<ResponseQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "a temporal derivative needs ResponseQuery.temporal".into(),
    })?;
    let mut single = query.clone();
    single.temporal = Some(antecedent_core::TemporalResponseSpec::new(
        vec![horizon],
        temporal.policy.clone(),
        temporal.max_history_lag,
    )?);
    Ok(single)
}

/// [`identify_window_derivative`] at every horizon of `query`, in horizon order.
///
/// # Errors
///
/// The refusal of any horizon.
pub(crate) fn identify_window_derivatives(
    template: &TemporalAdmg,
    query: &ResponseQuery,
    variable_count: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<(u32, WindowDerivative)>, CausalError> {
    let horizons =
        query.temporal.as_ref().map(|temporal| temporal.horizons.as_ref()).ok_or_else(|| {
            CausalError::Compile {
                message: "a temporal derivative needs ResponseQuery.temporal".into(),
            }
        })?;
    budget
        .charge_work(horizons.len() as u64)
        .map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let mut out = Vec::with_capacity(horizons.len());
    for &horizon in horizons {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal identification" });
        }
        let single = single_horizon_query(query, horizon)?;
        out.push((
            horizon,
            identify_window_derivative_within(template, &single, variable_count, budget, ctx)?,
        ));
    }
    Ok(out)
}

/// Identify the derivative or Jacobian `query` at its one horizon on `template`, by one
/// adjustment set of the unrolled window valid for every treatment/outcome pair.
///
/// # Errors
///
/// A population other than the observed one; a window with no adjustment set whose general ID
/// run finds a hedge (the typed refusal names the variables and periods that carry it); a
/// window that general ID identifies only through a non-adjustment functional, which no
/// continuous-treatment estimator evaluates; or a window the graph crate cannot number.
pub(crate) fn identify_window_derivative_within(
    template: &TemporalAdmg,
    query: &ResponseQuery,
    variable_count: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<WindowDerivative, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "a temporal derivative needs ResponseQuery.temporal".into(),
    })?;
    let horizon = match temporal.horizons.as_ref() {
        [horizon] => *horizon,
        _ => {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
    };
    let (treatment, outcome) =
        crate::analysis::proof_support::require_primary_pair(&query.functional)?;
    // The unit pulse contrast at the horizon under the response's policy and history bound:
    // the same probe every temporal route identifies a horizon with.
    let probe = TemporalEffectQuery {
        treatment,
        outcome,
        policy: temporal.policy.clone(),
        control: antecedent_core::Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
        active: antecedent_core::Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
        horizon_steps: horizon,
        max_history_lag: temporal.max_history_lag,
        target_population: query.target_population.clone(),
        effect_modifiers: std::sync::Arc::from([]),
    };
    let probe = &probe;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: format!(
                "a derivative on a TemporalAdmg answers for the observed population; a \
                 reweighted or local target population is a different estimand \
                 ({NEXT_ATTEMPTS})"
            ),
        });
    }
    let found = antecedent_identify::identify_temporal_admg_response_adjustment_within(
        template,
        probe,
        query,
        variable_count,
        budget,
        ctx,
    )?;
    if found.result.estimands.is_empty() {
        // No adjustment set: a front-door set of the window identifies the mean through its
        // mediators.
        if let Some(front_door) =
            antecedent_identify::identify_temporal_admg_front_door_derivative_within(
                template,
                probe,
                query,
                variable_count,
                budget,
                ctx,
            )?
        {
            let estimand = front_door.result.estimands[0].clone();
            let indexer = TemporalIndexer::new(
                variable_count,
                front_door.plan.origin,
                front_door.plan.periods - front_door.plan.origin,
            )?;
            return Ok(WindowDerivative { identification: front_door.result, estimand, indexer });
        }
        // Otherwise the complete algorithm decides whether this is a hedge.
        let contrast = identify_temporal_admg_contrast_over_within(
            template,
            probe,
            variable_count,
            budget,
            ctx,
        )?;
        if let Some(obstruction) = &contrast.obstruction {
            return Err(CausalError::not_identified(
                contrast.result.status,
                false,
                &format!("{obstruction}; {NEXT_ATTEMPTS}"),
            ));
        }
        return Err(CausalError::Unsupported {
            message: "the temporal derivative is identified on the unrolled window by the \
                      general ID algorithm but by neither an adjustment set nor a front-door \
                      set: it is a general ID functional of the observed law, and no \
                      continuous-treatment estimator evaluates such a functional",
        });
    }
    let estimand = found.result.estimands[0].clone();
    if found.result.estimands.iter().any(|other| other.adjustment_set != estimand.adjustment_set) {
        return Err(CausalError::Unsupported {
            message: "multi-pair temporal derivative estimation requires one common adjustment \
                      set",
        });
    }
    let indexer = TemporalIndexer::new(
        variable_count,
        found.plan.origin,
        found.plan.periods - found.plan.origin,
    )?;
    let mut identification = found.result;
    if window_has_no_directed_path(&found.window, &found.plan, query) {
        identification.diagnostics.push(no_directed_path_note(horizon));
    }
    Ok(WindowDerivative { identification, estimand, indexer })
}

/// Diagnostic code of a horizon whose window holds no directed path from the treatment to the
/// outcome: the derivative is the structural zero, whatever the observed association.
pub(crate) const NO_DIRECTED_PATH_CODE: &str = "response.temporal_admg.no_directed_path";

/// Whether the window ADMG has no directed path from any treatment (at the treated period) to
/// any outcome (at the outcome period) of a point or average derivative. Such a do-intervention
/// leaves the outcome's law unchanged, so the derivative is exactly zero.
fn window_has_no_directed_path(
    window: &UnrolledTemporalAdmg,
    plan: &TemporalAdmgWindowPlan,
    query: &ResponseQuery,
) -> bool {
    use antecedent_core::ResponseFunctional as F;
    if !matches!(query.functional, F::PointDerivative { .. } | F::AverageDerivative { .. }) {
        return false;
    }
    let (Some(&treated), admg) = (plan.treated.first(), window.admg()) else {
        return false;
    };
    let dense = |variable: VariableId, period: u32| {
        window.coordinate(variable, period).and_then(|slot| admg.dense_of_static(slot))
    };
    let mut pairs = 0_usize;
    for treatment in query.functional.treatment_ids() {
        for outcome in query.functional.outcome_ids() {
            let (Some(from), Some(to)) =
                (dense(treatment, treated), dense(outcome, plan.outcome_period))
            else {
                return false;
            };
            if from == to || admg.reaches(from, to) {
                return false;
            }
            pairs += 1;
        }
    }
    pairs > 0
}

/// The typed note that a horizon's derivative is the structural zero of an absent causal path.
fn no_directed_path_note(horizon: u32) -> antecedent_core::Diagnostic {
    antecedent_core::Diagnostic::new(
        NO_DIRECTED_PATH_CODE,
        antecedent_core::DiagnosticKind::Scientific,
        antecedent_core::DiagnosticSeverity::Info,
        format!(
            "horizon {horizon}: the declared graph has no directed path from the treatment to \
             the outcome, so the do-effect is exactly zero; the published value is that \
             structural zero, not the regression slope, which is a pure association here"
        ),
    )
    .with_fields([("horizon", horizon.to_string())])
}

/// Identify the pulse, sustained or conditional contrast of `query` on `template`, numbering the
/// window over the `variable_count` variables of the series schema.
///
/// # Errors
///
/// A population other than the observed one, an unsupported intervention or policy, a modifier
/// the treated periods can reach, a hedge on the window (the typed refusal names the variables
/// and periods that carry it and the next attempts), or no estimand the functional estimator
/// evaluates.
#[cfg(test)]
pub(crate) fn identify_window_contrast(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
    variable_count: u32,
) -> Result<WindowContrast, CausalError> {
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    identify_window_contrast_within(template, query, variable_count, &mut budget, &ctx)
}

pub(crate) fn identify_window_contrast_within(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
    variable_count: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<WindowContrast, CausalError> {
    if query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: format!(
                "a contrast on a TemporalAdmg answers for the observed population; a reweighted \
                 or local target population is a different estimand ({NEXT_ATTEMPTS})"
            ),
        });
    }
    let found =
        identify_temporal_admg_contrast_over_within(template, query, variable_count, budget, ctx)?;
    if let Some(obstruction) = &found.obstruction {
        return Err(CausalError::not_identified(
            found.result.status,
            false,
            &format!("{obstruction}; {NEXT_ATTEMPTS}"),
        ));
    }
    let estimand = select_estimand(&found.result, EstimatorId::FunctionalEffect)?;
    let indexer = TemporalIndexer::new(
        variable_count,
        found.plan.origin,
        found.plan.periods - found.plan.origin,
    )?;
    Ok(WindowContrast {
        identification: found.result.clone(),
        estimand,
        plan: found.plan.clone(),
        window: found.window.clone(),
        indexer,
        _identification_owner: found,
    })
}

/// The identification a prepared handle or a compiled program retains for the contrast, in the
/// shape the temporal routes share: one product at the query's horizon with the unfolding
/// indexer that names every window coordinate by variable and offset.
pub(crate) fn retained_identification(
    contrast: WindowContrast,
    query: &TemporalEffectQuery,
) -> CachedTemporalIdentification {
    CachedTemporalIdentification {
        by_horizon: std::sync::Arc::from([CachedTemporalHorizonIdentification {
            horizon: query.horizon_steps,
            identification: contrast.identification,
            estimand: contrast.estimand,
            indexer: contrast.indexer,
        }]),
    }
}

/// The variables the template or the query names: the only columns the window table must
/// hold values for.
pub(crate) fn used_variables(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
) -> Vec<VariableId> {
    let mut used: Vec<VariableId> = template
        .nodes()
        .iter()
        .filter_map(|node| match node {
            antecedent_core::NodeRef::Lagged { variable, .. } => Some(*variable),
            _ => None,
        })
        .chain([query.treatment, query.outcome])
        .chain(query.effect_modifiers.iter().copied())
        .collect();
    used.sort_unstable();
    used.dedup();
    used
}

/// Window coordinates the estimator must find in the table: the treatment at each treated
/// period, the outcome, and each modifier at the first treated period.
pub(crate) fn window_coordinates(
    window: &UnrolledTemporalAdmg,
    plan: &TemporalAdmgWindowPlan,
    query: &TemporalEffectQuery,
) -> Vec<VariableId> {
    let first = plan.treated.iter().copied().min().unwrap_or(plan.origin);
    plan.treated
        .iter()
        .filter_map(|&period| window.coordinate(query.treatment, period))
        .chain(window.coordinate(query.outcome, plan.outcome_period))
        .chain(
            query
                .effect_modifiers
                .iter()
                .filter_map(|&modifier| window.coordinate(modifier, first)),
        )
        .collect()
}

/// The lag-aligned window table of `series`: one row per window position, one column per window
/// coordinate, named `<variable>@<period>`.
///
/// Row `i` holds the series times `i .. i + periods`; the column of coordinate
/// `period * variables + variable` is the series value `periods - 1 - period` lags before the
/// row's last time, so the functional's variable ids index the table directly. A variable the
/// template and the query do not name (`used` lists the ones they do) is a column of missing
/// cells: the estimator never reads it, and a gap in a variable nobody asked about refuses
/// nothing.
///
/// # Errors
///
/// A missing value or a masked row in a used variable (a window cannot straddle one), or a
/// series shorter than [`MIN_WINDOWS`] windows.
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "the policy is passed by reference as every kernel-policy consumer takes it"
)]
#[derive(Clone, Debug)]
pub(crate) struct AdmittedWindowTable(std::sync::Arc<AdmittedWindowTableInner>);
#[derive(Debug)]
struct AdmittedWindowTableInner {
    table: TabularData,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for AdmittedWindowTable {
    type Target = TabularData;
    fn deref(&self) -> &Self::Target {
        &self.0.table
    }
}
impl TableView for AdmittedWindowTable {
    fn schema(&self) -> &antecedent_core::CausalSchema {
        self.0.table.schema()
    }
    fn row_count(&self) -> usize {
        self.0.table.row_count()
    }
    fn column(
        &self,
        id: VariableId,
    ) -> Result<antecedent_data::ColumnView<'_>, antecedent_data::DataError> {
        self.0.table.column(id)
    }
}

pub(crate) fn window_table(
    series: &TimeSeriesData,
    window: &UnrolledTemporalAdmg,
    used: &[VariableId],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<AdmittedWindowTable, CausalError> {
    window_table_over(series, window.variables(), window.periods(), used, budget, ctx)
}

/// [`window_table`] for a window known only by its layout: `periods` consecutive times of the
/// schema's `variables`, column `period * variables + variable`. The layout of every window
/// numbered by a `TemporalIndexer` (the finite unfolding of a temporal PAG), which has no
/// unrolled ADMG to read the layout from.
///
/// # Errors
///
/// As [`window_table`].
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "the policy is passed by reference as every kernel-policy consumer takes it"
)]
pub(crate) fn window_table_over(
    series: &TimeSeriesData,
    variables: &[VariableId],
    periods: u32,
    used: &[VariableId],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<AdmittedWindowTable, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal window table" });
    }
    let max_lag = periods
        .checked_sub(1)
        .ok_or_else(|| CausalError::Resource { message: "zero temporal window periods".into() })?;
    let columns = (variables.len() as u64)
        .checked_mul(u64::from(periods))
        .ok_or_else(|| CausalError::Resource { message: "window column count overflow".into() })?;
    if columns > budget.limits().max_graph_nodes {
        return Err(CausalError::Resource { message: "window column limit".into() });
    }
    let column_capacity = usize::try_from(columns).map_err(|_| CausalError::Resource {
        message: "window column count exceeds address space".into(),
    })?;
    let rows = series.row_count() as u64;
    let cells = columns
        .checked_mul(rows)
        .ok_or_else(|| CausalError::Resource { message: "window cell count overflow".into() })?;
    let compare = (variables.len() as u64)
        .checked_mul(used.len() as u64)
        .ok_or_else(|| CausalError::Resource { message: "window lookup work overflow".into() })?;
    let mut name_bytes = 0u64;
    budget
        .charge_work(variables.len() as u64)
        .map_err(|e| CausalError::Resource { message: e.to_string() })?;
    for &variable in variables {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal window table" });
        }
        name_bytes = name_bytes
            .checked_add(series.schema().get(variable)?.name.len() as u64)
            .ok_or_else(|| CausalError::Resource { message: "window names overflow".into() })?;
    }
    let names = name_bytes
        .checked_mul(u64::from(periods))
        .ok_or_else(|| CausalError::Resource { message: "window names overflow".into() })?;
    let work = cells
        .checked_mul(4)
        .and_then(|n| n.checked_add(compare))
        .and_then(|n| n.checked_add(columns))
        .and_then(|n| n.checked_add(names))
        .ok_or_else(|| CausalError::Resource { message: "window table work overflow".into() })?;
    budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let bytes = cells
        .checked_mul(64)
        .and_then(|n| columns.checked_mul(256).and_then(|c| n.checked_add(c)))
        .and_then(|n| n.checked_add(names))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| CausalError::Resource { message: "window table byte overflow".into() })?;
    if bytes > budget.limits().max_bytes {
        return Err(CausalError::Resource { message: "window table byte limit".into() });
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal window table" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?;
    let read: Vec<VariableId> =
        variables.iter().copied().filter(|variable| used.contains(variable)).collect();
    let frame = LaggedFrame::from_series_with_options_and_context(
        series,
        &read,
        max_lag,
        ReferencePointPolicy::SeriesOrigin,
        LaggedFrameOptions { mask: MaskPolicy::Honor, missing: MissingPolicy::ErrorOnMissing },
        &ctx.kernel_policy,
        ctx,
    )
    .map_err(|error| {
        if matches!(error, antecedent_data::DataError::Cancelled) {
            return CausalError::Cancelled { stage: "temporal window table" };
        }
        crate::compile_reason!(
            "missing_policy_required",
            "the lag-aligned window table of a temporal window contrast reads {periods} \
             consecutive times of every template variable and cannot straddle a missing or \
             masked value ({error}); next attempt: impute or trim the series to a complete \
             stretch, or analyse the stretches separately"
        )
    })?;
    if frame.n_effective() < MIN_WINDOWS {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the series holds {} complete windows of {periods} times and a temporal window \
             contrast needs at least {MIN_WINDOWS}; next attempt: a longer series, or a shorter \
             horizon so the window spans fewer times",
            frame.n_effective()
        ));
    }
    let schema = series.schema();
    let absent = vec![f64::NAN; frame.n_effective()];
    let mut names = Vec::with_capacity(column_capacity);
    let mut columns: Vec<&[f64]> = Vec::with_capacity(names.capacity());
    for period in 0..periods {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal window table" });
        }
        let lag = Lag::from_raw(max_lag - period);
        for &variable in variables {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal window table" });
            }
            names.push(format!("{}@{period}", schema.get(variable)?.name));
            columns.push(
                frame.column_index(variable, lag).map_or(absent.as_slice(), |i| frame.column(i)),
            );
        }
    }
    let table =
        TabularData::from_f64_columns(names.into_iter().zip(columns)).map_err(CausalError::from)?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal window table" });
    }
    Ok(AdmittedWindowTable(std::sync::Arc::new(AdmittedWindowTableInner { table, _lease: lease })))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{ExecutionContext, IdentificationStatus, TemporalNodeKey};
    use antecedent_estimate::{FunctionalDistributionWorkspace, FunctionalEffect, RowResampling};
    use antecedent_graph::DenseNodeId;

    use super::*;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    /// x(t-1) -> y(t) and nothing else: the effect of the treatment on the next period's outcome
    /// is the contrast of the conditional means given the treatment.
    fn lagged_cause() -> TemporalAdmg {
        let mut graph = TemporalAdmg::empty();
        let x1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
        let y0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        graph
    }

    /// The treatment at time t on the outcome at time t + 1.
    fn next_period() -> TemporalEffectQuery {
        TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2)
    }

    /// A deterministic binary series whose outcome copies the previous treatment, except at the
    /// listed times, where it flips.
    fn series(len: usize, flips: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let mut state = 0x9E37_79B9_u32;
        let mut x = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x.push(f64::from((state >> 16) & 1));
        }
        let y = (0..len)
            .map(|t| {
                let copy = if t == 0 { 0.0 } else { x[t - 1] };
                if flips.contains(&t) { 1.0 - copy } else { copy }
            })
            .collect();
        (x, y)
    }

    fn data(x: &[f64], y: &[f64]) -> TimeSeriesData {
        TimeSeriesData::from_f64_columns([("x", x), ("y", y)], 1).unwrap()
    }

    fn table_of(
        data: &TimeSeriesData,
        graph: &TemporalAdmg,
        query: &TemporalEffectQuery,
        contrast: &WindowContrast,
    ) -> AdmittedWindowTable {
        let used = used_variables(graph, query);
        let ctx = ExecutionContext::for_tests(10);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        window_table(data, &contrast.window, &used, &mut budget, &ctx).unwrap()
    }

    #[test]
    fn the_window_table_aligns_each_coordinate_to_its_series_time() {
        let (x, y) = series(40, &[]);
        let series_data = data(&x, &y);
        let query = next_period();
        let contrast = identify_window_contrast(&lagged_cause(), &query, 2).unwrap();
        let periods = contrast.window.periods() as usize;
        let table = table_of(&series_data, &lagged_cause(), &query, &contrast);
        assert_eq!(table.row_count(), x.len() - periods + 1);
        assert_eq!(table.schema().len(), 2 * periods);
        let cell = |name: &str, row: usize| {
            let id = table.schema().id_of(name).unwrap();
            let antecedent_data::ColumnView::Float64(column) = table.column(id).unwrap() else {
                panic!("float column")
            };
            column.values.as_slice()[row]
        };
        for row in 0..table.row_count() {
            for period in 0..periods {
                assert!((cell(&format!("x@{period}"), row) - x[row + period]).abs() < 1e-12);
                assert!((cell(&format!("y@{period}"), row) - y[row + period]).abs() < 1e-12);
            }
        }
        // The treated and outcome coordinates are the table's own column ids.
        let treated = contrast.window.coordinate(v(0), contrast.plan.treated[0]).unwrap();
        assert_eq!(table.schema().get(treated).unwrap().name.as_ref(), "x@1");
        let outcome = contrast.window.coordinate(v(1), contrast.plan.outcome_period).unwrap();
        assert_eq!(table.schema().get(outcome).unwrap().name.as_ref(), "y@2");
    }

    #[test]
    fn the_retained_indexer_names_every_window_coordinate_by_variable_and_offset() {
        let query = next_period();
        let contrast = identify_window_contrast(&lagged_cause(), &query, 2).unwrap();
        let treated = contrast.window.coordinate(v(0), contrast.plan.treated[0]).unwrap();
        let outcome = contrast.window.coordinate(v(1), contrast.plan.outcome_period).unwrap();
        assert_eq!(
            contrast.indexer.key_of(treated.raw()).unwrap(),
            TemporalNodeKey { variable: v(0), offset: 0 }
        );
        assert_eq!(
            contrast.indexer.key_of(outcome.raw()).unwrap(),
            TemporalNodeKey { variable: v(1), offset: 1 }
        );
        assert_eq!(contrast.indexer.dense_len(), contrast.window.admg().node_count());
        let retained = retained_identification(contrast, &query);
        assert!(retained.get(query.horizon_steps).is_some());
    }

    #[test]
    fn the_plug_in_recovers_the_conditional_mean_contrast_counted_from_the_series() {
        // y(t) copies x(t-1) except at four times, so the contrast is strictly inside (0, 1) and
        // counted independently below.
        let (x, y) = series(120, &[7, 31, 58, 90]);
        let series_data = data(&x, &y);
        let query = next_period();
        let contrast = identify_window_contrast(&lagged_cause(), &query, 2).unwrap();
        assert_eq!(
            contrast.identification.status,
            IdentificationStatus::NonparametricallyIdentified
        );
        let table = table_of(&series_data, &lagged_cause(), &query, &contrast);
        let ctx = ExecutionContext::for_tests(11);
        let fitter = FunctionalEffect::new();
        let prepared = fitter
            .prepare(
                &table,
                &contrast.estimand,
                &contrast.identification.arena,
                contrast.identification.required_assumptions.clone(),
                &window_coordinates(&contrast.window, &contrast.plan, &query),
            )
            .unwrap();
        let mut workspace = FunctionalDistributionWorkspace::default();
        let estimate = fitter.estimate(&prepared, &mut workspace, &ctx).unwrap();
        // Windows are series times (i, i + 1, i + 2); the treatment sits at i + 1 and the outcome
        // at i + 2, so count y(t + 1) given x(t) over t in 1 ..= len - 2.
        let mean_given = |level: f64| {
            let rows: Vec<f64> = (1..x.len() - 1)
                .filter(|t| (x[*t] - level).abs() < 1e-12)
                .map(|t| y[t + 1])
                .collect();
            rows.iter().sum::<f64>() / rows.len() as f64
        };
        let expected = mean_given(1.0) - mean_given(0.0);
        assert!(expected > 0.5 && expected < 1.0, "{expected}");
        assert!((estimate.ate - expected).abs() < 1e-12, "{} vs {expected}", estimate.ate);
        // The reversed lag direction would answer a different number.
        let reversed = |level: f64| {
            let rows: Vec<f64> = (1..x.len() - 1)
                .filter(|t| (x[*t + 1] - level).abs() < 1e-12)
                .map(|t| y[t])
                .collect();
            rows.iter().sum::<f64>() / rows.len() as f64
        };
        assert!((reversed(1.0) - reversed(0.0) - expected).abs() > 1e-3);
    }

    #[test]
    fn overlapping_windows_resample_in_blocks_never_shorter_than_the_window() {
        let (x, y) = series(60, &[5]);
        let series_data = data(&x, &y);
        let query = next_period();
        let contrast = identify_window_contrast(&lagged_cause(), &query, 2).unwrap();
        let table = table_of(&series_data, &lagged_cause(), &query, &contrast);
        let span = contrast.window.periods();
        let ctx = ExecutionContext::for_tests(12);
        let fitter = FunctionalEffect { bootstrap_replicates: 40, ..FunctionalEffect::new() }
            .with_row_resampling(
                RowResampling::WindowBlocks { span },
                antecedent_estimate::util::SamplingDeclaration::Undeclared,
            );
        let prepared = fitter
            .prepare(
                &table,
                &contrast.estimand,
                &contrast.identification.arena,
                contrast.identification.required_assumptions.clone(),
                &window_coordinates(&contrast.window, &contrast.plan, &query),
            )
            .unwrap();
        let mut workspace = FunctionalDistributionWorkspace::default();
        let blocked = fitter.estimate(&prepared, &mut workspace, &ctx).unwrap();
        let block = fitter.row_resampling.block_length(table.row_count()).unwrap();
        assert!(block >= span as usize, "{block} < {span}");
        assert_eq!(
            blocked.uncertainty.block.family,
            Some(antecedent_estimate::CircularBlockFamily::RowBlock)
        );
    }

    #[test]
    fn a_hedged_window_is_a_typed_refusal_naming_the_variables_and_the_next_attempts() {
        let mut graph = lagged_cause();
        // Latent confounding between the lagged treatment and the outcome it causes.
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let error = identify_window_contrast(&graph, &next_period(), 2).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("effect_not_identified"), "{text}");
        assert!(text.contains("hedge") && text.contains('@'), "{text}");
        assert!(text.contains("next attempts"), "{text}");
    }

    #[test]
    fn window_table_refuses_tight_limits_and_clones_retain_lease() {
        let (x, y) = series(40, &[]);
        let series_data = data(&x, &y);
        let graph = lagged_cause();
        let query = next_period();
        let contrast = identify_window_contrast(&graph, &query, 2).unwrap();
        let used = used_variables(&graph, &query);
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(10).with_admission(pool.clone());
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let before = pool.available().bytes();
        let table = window_table(&series_data, &contrast.window, &used, &mut budget, &ctx).unwrap();
        let held = pool.available().bytes();
        assert!(held < before);
        let copy = table.clone();
        drop(table);
        assert_eq!(pool.available().bytes(), held);
        drop(copy);
        assert_eq!(pool.available().bytes(), before);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_bytes = 1;
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        assert!(matches!(
            window_table(&series_data, &contrast.window, &used, &mut budget, &ctx),
            Err(CausalError::Resource { .. })
        ));
    }

    #[test]
    fn a_gap_in_a_variable_the_graph_does_not_name_refuses_nothing_but_a_gap_in_a_used_one_does() {
        let (x, y) = series(40, &[]);
        let mut unrelated = vec![0.0; 40];
        unrelated[20] = f64::NAN;
        let query = next_period();
        let graph = lagged_cause();
        let contrast = identify_window_contrast(&graph, &query, 3).unwrap();
        let ctx = ExecutionContext::for_tests(10);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let used = used_variables(&graph, &query);
        let three = TimeSeriesData::from_f64_columns(
            [("x", &x[..]), ("y", &y[..]), ("z", &unrelated[..])],
            1,
        )
        .unwrap();
        let table = window_table(&three, &contrast.window, &used, &mut budget, &ctx).unwrap();
        assert_eq!(table.schema().len(), 3 * contrast.window.periods() as usize);
        let mut gappy = y.clone();
        gappy[20] = f64::NAN;
        let refused =
            TimeSeriesData::from_f64_columns([("x", &x[..]), ("y", &gappy[..])], 1).unwrap();
        let contrast = identify_window_contrast(&graph, &query, 2).unwrap();
        let error = window_table(&refused, &contrast.window, &used, &mut budget, &ctx).unwrap_err();
        assert!(error.to_string().contains("missing"), "{error}");
    }

    #[test]
    fn a_short_series_is_refused_with_its_next_attempt() {
        let (x, y) = series(9, &[]);
        let query = next_period();
        let graph = lagged_cause();
        let contrast = identify_window_contrast(&graph, &query, 2).unwrap();
        let used = used_variables(&graph, &query);
        let ctx = ExecutionContext::for_tests(10);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let error =
            window_table(&data(&x, &y), &contrast.window, &used, &mut budget, &ctx).unwrap_err();
        assert!(error.to_string().contains("next attempt"), "{error}");
    }
}
