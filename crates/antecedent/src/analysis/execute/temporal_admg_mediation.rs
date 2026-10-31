// SPDX-License-Identifier: MIT OR Apache-2.0
//! Natural mediation on one series under a temporal ADMG.
//!
//! The template is unrolled over the periods the pulse and its outcome horizon need (as for a
//! pulse contrast), so a cause that precedes the window is a latent common cause of its
//! in-window effects. On that window ADMG the treatment is the pulse period's coordinate, the
//! outcome the horizon period's coordinate, and the mediators are the declared mediator
//! variables at every period from the pulse to the outcome. The path-specific identifier then
//! decides the total, natural direct and natural indirect effects, including the recanting
//! district criterion on the window, and the decomposition is estimated on the lag-aligned
//! window table with the same joint uncertainty as a static mixed graph: circular blocks of
//! consecutive windows (never shorter than the window) or one Dirichlet weight per block.

use antecedent_core::{Intervention, MediationQuery, TemporalEffectQuery, VariableId};
use antecedent_data::TableView;
use std::sync::Arc;

use super::mediation_mixed::{ComponentRun, component_slot, identify_components};
use super::{
    CausalError, EstimatorId, ExecutionContext, GraphRef, Instant, PhysicalExecutionPlan,
    StudyResult, TimeSeriesData, report_identify_compute,
};
use crate::analysis::temporal_admg_contrast::{used_variables, window_table};
use crate::planner::RouteDefaults;

/// Owns the copied intervention payload and its admission together.
pub(crate) struct AdmittedMediationEffect {
    effect: TemporalEffectQuery,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for AdmittedMediationEffect {
    type Target = TemporalEffectQuery;
    fn deref(&self) -> &TemporalEffectQuery {
        &self.effect
    }
}

/// The pulse contrast whose window the mediation reads: same treatment, outcome and horizon.
pub(crate) fn window_effect_query_within(
    query: &MediationQuery,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<AdmittedMediationEffect, CausalError> {
    let work = (query.mediators.len() as u64).checked_add(query.horizons.len() as u64).ok_or_else(
        || CausalError::Resource { message: "temporal mediation validation overflow".into() },
    )?;
    budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let validation = antecedent_core::admit_temporal_intervention_validation(
        [&query.control, &query.active],
        budget,
        ctx,
    )
    .map_err(|e| {
        CausalError::from(antecedent_estimate::EstimationError::TemporalMaterialization(e))
    })?;
    query.validate()?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal mediation" });
    }
    let horizon = match query.horizons.as_ref() {
        [horizon] => *horizon,
        _ => {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "mediation on a temporal ADMG or PAG window reads one outcome horizon; next attempt: one \
                 study per horizon"
            ));
        }
    };
    let mut effect =
        TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0).with_horizon_steps(horizon);
    effect.control = query.control.clone();
    effect.active = query.active.clone();
    effect.target_population = query.target_population.clone();
    Ok(AdmittedMediationEffect { effect, _lease: validation })
}

/// The window problem of a mediation under a temporal ADMG: the unrolled window, the
/// pulse contrast on its coordinates and the mediators between the pulse and the outcome.
pub(crate) struct AdmgWindowProblem {
    effect: AdmittedMediationEffect,
    plan: antecedent_identify::temporal_admg::AdmittedTemporalAdmgWindowPlan,
    _mediator_lease: Arc<antecedent_core::execution::AdmissionLease>,
    window: antecedent_graph::UnrolledTemporalAdmg,
    treatment: VariableId,
    outcome: VariableId,
    mediators: Vec<VariableId>,
    window_query: MediationQuery,
}

/// State the window problem of `query` on `template`, reading `variable_count` variables.
///
/// # Errors
///
/// The query reads more than one horizon, names a variable the template lacks, has no
/// mediator in the window, or intervenes other than by `Set`.
pub(crate) fn admg_window_problem_within(
    template: &antecedent_graph::TemporalAdmg,
    query: &MediationQuery,
    variable_count: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<AdmgWindowProblem, CausalError> {
    let effect = window_effect_query_within(query, budget, ctx)?;
    let plan = antecedent_identify::temporal_admg::plan_temporal_admg_window_within(
        template, &effect, budget, ctx,
    )?;
    let window = template.unroll_window_over_within(plan.periods, variable_count, budget, ctx)?;
    let unknown = |variable: VariableId| CausalError::Compile {
        message: format!(
            "variable {} is not a variable of the temporal ADMG template; name a variable \
                 that appears in the template at some lag",
            variable.raw()
        ),
    };
    let first = plan.treated.first().copied().ok_or_else(|| CausalError::Compile {
        message: "the pulse has no treated period".into(),
    })?;
    let treatment =
        window.coordinate(query.treatment, first).ok_or_else(|| unknown(query.treatment))?;
    let outcome = window
        .coordinate(query.outcome, plan.outcome_period)
        .ok_or_else(|| unknown(query.outcome))?;
    let periods =
        if first <= plan.outcome_period { u64::from(plan.outcome_period - first) + 1 } else { 0 };
    let count = periods.checked_mul(query.mediators.len() as u64).ok_or_else(|| {
        CausalError::Resource { message: "temporal mediator count overflow".into() }
    })?;
    budget
        .charge_work(count)
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    let bytes = count.checked_mul(64).and_then(|n| n.checked_add(512)).ok_or_else(|| {
        CausalError::Resource { message: "temporal mediator bytes overflow".into() }
    })?;
    if bytes > budget.limits().max_bytes {
        return Err(CausalError::Resource { message: "temporal mediator byte limit".into() });
    }
    let mediator_lease = Arc::new(ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal mediation" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?);
    let mut mediators = Vec::new();
    mediators
        .try_reserve_exact(usize::try_from(count).map_err(|_| CausalError::Resource {
            message: "temporal mediator count exceeds address space".into(),
        })?)
        .map_err(|_| CausalError::Resource {
            message: "temporal mediator allocation failed".into(),
        })?;
    for &variable in query.mediators.iter() {
        for period in first..=plan.outcome_period {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal mediation" });
            }
            if let Some(node) = window.coordinate(variable, period) {
                if node != treatment && node != outcome {
                    mediators.push(node);
                }
            }
        }
    }
    if mediators.is_empty() {
        return Err(CausalError::Compile {
            message: "no mediator occurs in the window between the pulse and the outcome; a \
                          mediator must lie between the treatment period and the outcome period"
                .into(),
        });
    }
    let level = |intervention: &Intervention| match intervention {
        Intervention::Set { value, .. } => Ok(Intervention::set(treatment, value.clone())),
        _ => Err(crate::unsupported_reason!(
            "option_not_applicable",
            "mediation under a temporal ADMG requires Set interventions on both sides"
        )),
    };
    let mut window_query =
        MediationQuery::binary(treatment, outcome, mediators.clone(), query.contrast);
    window_query.control = level(&query.control)?;
    window_query.active = level(&query.active)?;
    window_query.target_population = query.target_population.clone();

    Ok(AdmgWindowProblem {
        effect,
        plan,
        window,
        treatment,
        outcome,
        mediators,
        window_query,
        _mediator_lease: mediator_lease,
    })
}

impl super::Study {
    /// Identify and estimate the mediation decomposition on the unrolled window of a
    /// `TemporalAdmg`.
    ///
    /// # Errors
    ///
    /// The requested contrast is not identified on the window (the refusal names the
    /// recanting district or hedge), a custom validator was attached, the series holds a gap
    /// inside a window, or more than one outcome horizon was requested.
    pub(super) fn execute_temporal_admg_mediation(
        &self,
        series: &TimeSeriesData,
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if !self.declared.custom_validators.is_empty() {
            return Err(crate::unsupported_reason!(
                "validators_not_applicable",
                "a custom validator refutes one scalar average effect and none applies to a \
                 mediation decomposition on the window of a temporal ADMG"
            ));
        }
        let started = Instant::now();
        let GraphRef::TemporalAdmg(template) = self.graph.graph_ref() else {
            return Err(CausalError::Compile {
                message: "temporal ADMG mediation requires a supplied TemporalAdmg".into(),
            });
        };
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "temporal ADMG mediation requires estimator functional.effect; got \
                     {estimator_id}"
                ),
            });
        }
        let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
            CausalError::Compile { message: "series schema exceeds the variable id range".into() }
        })?;
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let problem =
            admg_window_problem_within(template, query, variable_count, &mut budget, ctx)?;
        let AdmgWindowProblem {
            effect,
            plan,
            window,
            treatment,
            outcome,
            mediators,
            window_query,
            _mediator_lease,
        } = problem;
        // The window is identified once, at compile, and the program retains the components;
        // a plan that retained none identifies here.
        let mut identified =
            if let Some(crate::planner::lowered::RetainedExecutionProof::Admg(identified)) =
                crate::analysis::lowering::plan_retained_execution_proof(self, physical)
            {
                identified.clone()
            } else {
                report_identify_compute(ctx);
                identify_components(window.admg(), &window_query, identifier_id, estimator_id)?
            };
        for (result, _) in identified.iter_mut().flatten() {
            result.retain_temporal_lease(Arc::clone(&_mediator_lease));
        }
        let mut used = used_variables(template, &effect);
        used.extend(query.mediators.iter().copied());
        used.sort_unstable();
        used.dedup();
        let table = window_table(series, &window, &used, &mut budget, ctx)?;
        let mut roles = vec![treatment, outcome];
        roles.extend(mediators.iter().copied());
        let (resampling, design) = self.window_row_resampling(plan.periods)?;
        let disclosure = super::temporal_admg_path::window_disclosure(
            &plan,
            table.row_count(),
            resampling.block_length(table.row_count()),
            0,
        );
        self.estimate_components(
            &table,
            ComponentRun {
                identified,
                requested: component_slot(query.contrast),
                roles,
                reported: (query.treatment, query.outcome),
                resampling: (resampling, design),
                structure: "unrolled temporal window",
                strategy: (identifier_id, estimator_id),
                started,
                extra_diagnostics: vec![disclosure],
            },
            physical,
            ctx,
        )
    }

    /// The window identification compile retains with the program, for a mediation under a
    /// temporal ADMG or PAG; `None` leaves identification (and its refusal) to execution.
    pub(crate) fn retain_window_mediation_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Option<crate::planner::lowered::RetainedExecutionProof>, CausalError> {
        if matches!(self.graph.graph_ref(), GraphRef::TemporalPag(_)) {
            let (super::DataInput::Temporal(series) | super::DataInput::Event(series)) = &self.data
            else {
                return Ok(None);
            };
            let Ok(count) = u32::try_from(series.schema().len()) else {
                return Ok(None);
            };
            self.retain_pag_window_identification(count, ctx)
        } else {
            self.retain_admg_window_mediation_identification(plan, ctx)
        }
    }

    fn retain_admg_window_mediation_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Option<crate::planner::lowered::RetainedExecutionProof>, CausalError> {
        let (super::DataInput::Temporal(series) | super::DataInput::Event(series)) = &self.data
        else {
            return Ok(None);
        };
        let Ok(variable_count) = u32::try_from(series.schema().len()) else {
            return Ok(None);
        };
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        match self.graph.graph_ref() {
            GraphRef::TemporalAdmg(template) => {
                let antecedent_core::CausalQuery::Mediation(query) = &self.query else {
                    return Ok(None);
                };
                let Ok((identifier_id, estimator_id)) = plan.binding().resolve(RouteDefaults::PATH)
                else {
                    return Ok(None);
                };
                if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
                    return Ok(None);
                }
                let problem = match admg_window_problem_within(
                    template,
                    query,
                    variable_count,
                    &mut budget,
                    ctx,
                ) {
                    Ok(problem) => problem,
                    Err(error @ (CausalError::Resource { .. } | CausalError::Cancelled { .. })) => {
                        return Err(error);
                    }
                    Err(_) => return Ok(None),
                };
                let mut identified = match identify_components(
                    problem.window.admg(),
                    &problem.window_query,
                    identifier_id,
                    estimator_id,
                ) {
                    Ok(identified) => identified,
                    Err(error @ (CausalError::Resource { .. } | CausalError::Cancelled { .. })) => {
                        return Err(error);
                    }
                    Err(_) => return Ok(None),
                };
                for (result, _) in identified.iter_mut().flatten() {
                    result.retain_temporal_lease(Arc::clone(&problem._mediator_lease));
                }
                Ok(Some(crate::planner::lowered::RetainedExecutionProof::Admg(identified)))
            }
            _ => Ok(None),
        }
    }
}

/// The `(variable, offset)` of every coordinate of the window a mediation under a temporal ADMG
/// or PAG unrolls, indexed by the window coordinate; `None` for any other graph or query.
///
/// An identification over the window names its nodes by window coordinate, so a reader needs
/// this map to place them: the offset is the period relative to the policy origin.
pub(crate) struct NamespaceData {
    pub(crate) nodes: Vec<(u32, i32)>,
    pub(crate) lease: Arc<antecedent_core::execution::AdmissionLease>,
}
pub(crate) fn admit_namespace(
    count: u64,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Arc<antecedent_core::execution::AdmissionLease>, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal namespace" });
    }
    budget.charge_work(count).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    // Covers the coordinate vector, wire conversion, and their transient copies.
    let bytes = count.checked_mul(128).and_then(|v| v.checked_add(4096)).ok_or_else(|| {
        CausalError::Resource { message: "temporal namespace byte overflow".into() }
    })?;
    if bytes > budget.limits().max_bytes {
        return Err(CausalError::Resource { message: "temporal namespace byte limit".into() });
    }
    Ok(Arc::new(ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal namespace" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?))
}

pub(crate) fn mediation_window_namespace(
    graph: GraphRef<'_>,
    query: &antecedent_core::CausalQuery,
    variable_count: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Option<NamespaceData>, CausalError> {
    let antecedent_core::CausalQuery::Mediation(query) = query else {
        return Ok(None);
    };
    let indexer = match graph {
        GraphRef::TemporalAdmg(template) => {
            let problem = admg_window_problem_within(template, query, variable_count, budget, ctx)?;
            let nodes = u64::from(problem.plan.periods)
                .checked_mul(problem.window.variables().len() as u64)
                .ok_or_else(|| CausalError::Resource {
                    message: "temporal namespace size overflow".into(),
                })?;
            let lease = admit_namespace(nodes, budget, ctx)?;
            let origin = i32::try_from(problem.plan.origin).map_err(|_| CausalError::Resource {
                message: "temporal namespace origin overflow".into(),
            })?;
            let mut result = Vec::new();
            result
                .try_reserve_exact(usize::try_from(nodes).map_err(|_| CausalError::Resource {
                    message: "temporal namespace too large".into(),
                })?)
                .map_err(|_| CausalError::Resource {
                    message: "temporal namespace allocation".into(),
                })?;
            for coordinate in 0..nodes {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: "temporal namespace" });
                }
                let raw = u32::try_from(coordinate).map_err(|_| CausalError::Resource {
                    message: "temporal namespace coordinate overflow".into(),
                })?;
                let (variable, period) = problem
                    .window
                    .locate(VariableId::from_raw(raw))
                    .ok_or_else(|| CausalError::Compile {
                        message: "invalid temporal namespace coordinate".into(),
                    })?;
                let offset =
                    i32::try_from(i64::from(period) - i64::from(origin)).map_err(|_| {
                        CausalError::Resource {
                            message: "temporal namespace offset overflow".into(),
                        }
                    })?;
                result.push((variable.raw(), offset));
            }
            return Ok(Some(NamespaceData { nodes: result, lease }));
        }
        GraphRef::TemporalPag(template) => {
            let effect = window_effect_query_within(query, budget, ctx)?;
            antecedent_identify::temporal_pag_mediation_window_within(
                template,
                &effect,
                query,
                variable_count,
                budget,
                ctx,
            )?
            .indexer
        }
        _ => return Ok(None),
    };
    let nodes = indexer.dense_len() as u64;
    let lease = admit_namespace(nodes, budget, ctx)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(indexer.dense_len())
        .map_err(|_| CausalError::Resource { message: "temporal namespace allocation".into() })?;
    for dense in 0..indexer.dense_len() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal namespace" });
        }
        let raw = u32::try_from(dense).map_err(|_| CausalError::Resource {
            message: "temporal namespace coordinate overflow".into(),
        })?;
        let key = indexer
            .key_of(raw)
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        result.push((key.variable.raw(), key.offset));
    }
    Ok(Some(NamespaceData { nodes: result, lease }))
}
