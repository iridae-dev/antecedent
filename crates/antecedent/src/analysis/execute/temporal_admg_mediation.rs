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
use antecedent_identify::temporal_admg::plan_temporal_admg_window;

use super::mediation_mixed::{ComponentRun, component_slot, identify_components};
use super::{
    CausalError, EstimatorId, ExecutionContext, GraphRef, Instant, PhysicalExecutionPlan,
    StudyResult, TimeSeriesData, report_identify_compute,
};
use crate::analysis::temporal_admg_contrast::{used_variables, window_table};
use crate::planner::RouteDefaults;

/// The pulse contrast whose window the mediation reads: same treatment, outcome and horizon.
pub(crate) fn window_effect_query(
    query: &MediationQuery,
) -> Result<TemporalEffectQuery, CausalError> {
    query.validate()?;
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
    Ok(effect)
}

/// The window problem of a mediation under a temporal ADMG: the unrolled window, the
/// pulse contrast on its coordinates and the mediators between the pulse and the outcome.
pub(crate) struct AdmgWindowProblem {
    effect: TemporalEffectQuery,
    plan: antecedent_identify::temporal_admg::TemporalAdmgWindowPlan,
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
pub(crate) fn admg_window_problem(
    template: &antecedent_graph::TemporalAdmg,
    query: &MediationQuery,
    variable_count: u32,
) -> Result<AdmgWindowProblem, CausalError> {
    let effect = window_effect_query(query)?;
    let plan = plan_temporal_admg_window(template, &effect)?;
    let window = template.unroll_window_over(plan.periods, variable_count)?;
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
    let mut mediators = Vec::new();
    for &variable in query.mediators.iter() {
        for period in first..=plan.outcome_period {
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

    Ok(AdmgWindowProblem { effect, plan, window, treatment, outcome, mediators, window_query })
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
        if !self.custom_validators.is_empty() {
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
        let problem = admg_window_problem(template, query, variable_count)?;
        let AdmgWindowProblem { effect, plan, window, treatment, outcome, mediators, window_query } =
            problem;
        // The window is identified once, at compile, and the program retains the components;
        // a plan that retained none identifies here.
        let identified = match crate::analysis::lowering::plan_retained_execution_proof(self, physical) {
            Some(crate::planner::lowered::RetainedExecutionProof::Admg(identified)) => identified.clone(),
            _ => {
                report_identify_compute(ctx);
                identify_components(window.admg(), &window_query, identifier_id, estimator_id)?
            }
        };
        let mut used = used_variables(template, &effect);
        used.extend(query.mediators.iter().copied());
        used.sort_unstable();
        used.dedup();
        let table = window_table(series, &window, &used, &ctx.kernel_policy)?;
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
    ) -> Option<crate::planner::lowered::RetainedExecutionProof> {
        let (super::DataInput::Temporal(series) | super::DataInput::Event(series)) = &self.data
        else {
            return None;
        };
        let variable_count = u32::try_from(series.schema().len()).ok()?;
        match self.graph.graph_ref() {
            GraphRef::TemporalAdmg(template) => {
                let antecedent_core::CausalQuery::Mediation(query) = &self.query else {
                    return None;
                };
                let (identifier_id, estimator_id) =
                    plan.binding().resolve(RouteDefaults::PATH).ok()?;
                if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
                    return None;
                }
                let problem = admg_window_problem(template, query, variable_count).ok()?;
                let identified = identify_components(
                    problem.window.admg(),
                    &problem.window_query,
                    identifier_id,
                    estimator_id,
                )
                .ok()?;
                Some(crate::planner::lowered::RetainedExecutionProof::Admg(identified))
            }
            GraphRef::TemporalPag(_) => self.retain_pag_window_identification(variable_count),
            _ => None,
        }
    }
}
