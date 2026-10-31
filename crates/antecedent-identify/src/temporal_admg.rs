//! Pulse and sustained contrasts on a [`TemporalAdmg`], identified on its unrolled window.
//!
//! The template is unrolled over enough periods to hold the policy's treated steps, the
//! outcome and the history the template's lags reach back to, and the complete ID algorithm
//! runs on that window as a static ADMG: both contrast sides are identified, or the hedge
//! that blocks identification is reported in template terms (variable and period). A
//! variable that occupies several lags therefore no longer needs a static projection.
//! A cause that precedes the window is treated as a latent common cause of its in-window
//! effects, and history deeper than that is the initial state, as in the unfolding of a
//! `TemporalDag`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, Intervention, TemporalEffectQuery, Value, VariableId};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DerivationMeta, EstimandMethod, ExprNode, IdentifiedEstimand,
    OutcomeExprId,
};
use antecedent_graph::{TemporalAdmg, UnrolledTemporalAdmg};

use crate::error::IdentificationError;
use crate::id::{IdIdentifier, ScheduleContrast};
use crate::idc::IdcIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::prepared::PreparedAdmg;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};

/// Where the policy, the outcome and the history sit in the unrolled window.
///
/// The window is the same for every consumer of the contrast: identification states the
/// functional over its coordinates, and the estimator reads one lag-aligned row of the series
/// per window position.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalAdmgWindowPlan {
    /// Periods in the window: the last treated or outcome period plus one.
    pub periods: u32,
    /// Period the policy origin sits at: the template's longest lag, plus the earliest
    /// negative treated offset.
    pub origin: u32,
    /// Longest lag the template reaches back.
    pub history: u32,
    /// Treated periods, in the policy's offset order.
    pub treated: Vec<u32>,
    /// Period the outcome is read at.
    pub outcome_period: u32,
}

/// An admitted window plan. Clones share the plan and its lease; an explicit
/// clone of the borrowed plan is a separately caller-owned allocation.
#[derive(Clone, Debug)]
pub struct AdmittedTemporalAdmgWindowPlan(Arc<AdmittedTemporalAdmgWindowPlanInner>);
#[derive(Debug)]
struct AdmittedTemporalAdmgWindowPlanInner {
    plan: TemporalAdmgWindowPlan,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl PartialEq for AdmittedTemporalAdmgWindowPlan {
    fn eq(&self, other: &Self) -> bool {
        self.0.plan == other.0.plan
    }
}
impl Eq for AdmittedTemporalAdmgWindowPlan {}
impl std::ops::Deref for AdmittedTemporalAdmgWindowPlan {
    type Target = TemporalAdmgWindowPlan;
    fn deref(&self) -> &TemporalAdmgWindowPlan {
        &self.0.plan
    }
}

/// Plan a temporal ADMG window under explicit logical and memory ceilings.
/// No offset is expanded before its size is admitted. Cancellation and resource
/// exhaustion remain operational failures, not non-identification results.
/// # Errors
/// Invalid query, unrepresentable window, limit, cancellation or admission.
pub fn plan_temporal_admg_window_within(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<AdmittedTemporalAdmgWindowPlan, IdentificationError> {
    use crate::error::IdentificationBudget;
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    budget
        .charge_work(query.effect_modifiers.len() as u64)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let validation_lease = antecedent_core::admit_temporal_intervention_validation(
        [&query.control, &query.active],
        budget,
        ctx,
    )
    .map_err(crate::error::temporal_materialization_error)?;
    query
        .validate()
        .map_err(|_| IdentificationError::unsupported("invalid temporal effect query"))?;
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    drop(validation_lease);
    let count = query.policy.offset_count().map_err(IdentificationError::construction)?;
    budget
        .charge_offsets(count)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let limits = budget.limits();
    let work = count
        .checked_mul(2)
        .and_then(|n| n.checked_add(template.nodes().len() as u64))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    budget
        .charge_work(work)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let outcome =
        query.horizon_steps.checked_sub(1).and_then(|n| i32::try_from(n).ok()).ok_or_else(
            || IdentificationError::out_of_range("temporal outcome offset exceeds i32"),
        )?;
    let mut history = 0u32;
    for node in template.nodes() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        if let antecedent_core::NodeRef::Lagged { lag, .. } = node {
            history = history.max(lag.raw());
        }
    }
    let mut earliest = 0i32;
    for offset in query.policy.offsets().map_err(IdentificationError::construction)? {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        earliest = earliest.min(offset);
    }
    let origin = history
        .checked_add(earliest.unsigned_abs())
        .ok_or_else(|| IdentificationError::out_of_range("temporal origin exceeds u32"))?;
    let period = |offset: i32| -> Result<u32, IdentificationError> {
        u32::try_from(i64::from(origin) + i64::from(offset)).map_err(|_| {
            IdentificationError::out_of_range(
                "treated or outcome period exceeds window coordinates",
            )
        })
    };
    let outcome_period = period(outcome)?;
    let bytes = count
        .checked_mul(8)
        .and_then(|n| n.checked_add(512))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
    if bytes > limits.max_bytes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let len = usize::try_from(count)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Memory))?;
    let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
        _ => IdentificationError::budget(IdentificationBudget::Memory),
    })?;
    let mut treated = Vec::new();
    treated
        .try_reserve_exact(len)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Memory))?;
    let mut last = outcome_period;
    for offset in query.policy.offsets().map_err(IdentificationError::construction)? {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        let next = period(offset)?;
        last = last.max(next);
        treated.push(next);
    }
    let periods = last
        .checked_add(1)
        .ok_or_else(|| IdentificationError::out_of_range("temporal window exceeds u32"))?;
    Ok(AdmittedTemporalAdmgWindowPlan(Arc::new(AdmittedTemporalAdmgWindowPlanInner {
        plan: TemporalAdmgWindowPlan { periods, origin, history, treated, outcome_period },
        _lease: lease,
    })))
}

/// A scheduled temporal contrast whose window, plan and lease share ownership.
pub type TemporalAdmgIdentification = AdmittedTemporalAdmgIdentification;

/// Where the policy, the outcome and the history of `query` sit in the unrolled window of
/// `template`.
///
/// The policy origin sits after as many history periods as the template's longest lag, the
/// outcome at `origin + horizon_steps - 1` (the reading every temporal route gives the
/// horizon), and the treated periods at the policy's offsets.
///
/// # Errors
///
/// An invalid query, a policy with a negative offset beyond the history, or a window the
/// period numbering cannot hold.
pub fn plan_temporal_admg_window(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
) -> Result<AdmittedTemporalAdmgWindowPlan, IdentificationError> {
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    plan_temporal_admg_window_within(template, query, &mut budget, &ctx)
}

/// Identify the pulse, sustained or conditional contrast of `query` on `template`, numbering the
/// window over the variable ids up to the largest the template or the query names.
///
/// # Errors
///
/// As [`identify_temporal_admg_contrast_over`].
pub fn identify_temporal_admg_contrast(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
) -> Result<TemporalAdmgIdentification, IdentificationError> {
    use crate::error::IdentificationBudget;
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let work = (template.nodes().len() as u64)
        .checked_add(query.effect_modifiers.len() as u64)
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    budget
        .charge_work(work)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let mut largest = query.treatment.raw().max(query.outcome.raw());
    for node in template.nodes() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        if let antecedent_core::NodeRef::Lagged { variable, .. } = node {
            largest = largest.max(variable.raw());
        }
    }
    for modifier in query.effect_modifiers.iter() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        largest = largest.max(modifier.raw());
    }
    let count = largest
        .checked_add(1)
        .ok_or_else(|| IdentificationError::out_of_range("temporal variable count exceeds u32"))?;
    identify_temporal_admg_contrast_over_within(template, query, count, &mut budget, &ctx)
}

/// Identify the pulse, sustained or conditional contrast of `query` on `template`, with
/// `variable_count` coordinates per period of the window (the schema's variable count, so a
/// window coordinate is `period * variable_count + variable`).
///
/// # Errors
///
/// A policy with a negative offset beyond the history, a non-`Set` intervention, a treatment,
/// outcome or modifier that is not a template variable, a repeated modifier, a modifier the
/// treated periods can reach, a variable outside `0 .. variable_count`, or a window the graph
/// crate cannot hold. A contrast the window does not identify is not an error: the result is
/// not identified and [`TemporalAdmgIdentification::obstruction`] names the hedge.
pub fn identify_temporal_admg_contrast_over(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
    variable_count: u32,
) -> Result<TemporalAdmgIdentification, IdentificationError> {
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    identify_temporal_admg_contrast_over_within(template, query, variable_count, &mut budget, &ctx)
}

/// Context-bounded identification. The Arc-owned result shares its window,
/// schedule, and retained admission across clones. A caller-created copy of a
/// borrowed field is owned and accounted by that caller.
#[derive(Clone, Debug)]
pub struct AdmittedTemporalAdmgIdentification(Arc<AdmittedTemporalAdmgIdentificationData>);
/// Borrowable fields of an admitted temporal ADMG identification.
#[derive(Debug)]
pub struct AdmittedTemporalAdmgIdentificationData {
    /// Identification output over the admitted window.
    pub result: IdentificationResult,
    /// Admitted window.
    pub window: UnrolledTemporalAdmg,
    /// Admitted temporal plan.
    pub plan: AdmittedTemporalAdmgWindowPlan,
    /// Treatment coordinates.
    pub schedule: Vec<VariableId>,
    /// Outcome coordinate.
    pub outcome: VariableId,
    /// Modifier coordinates.
    pub modifiers: Vec<VariableId>,
    /// Identification obstruction, when present.
    pub obstruction: Option<String>,
    _lease: Arc<antecedent_core::execution::AdmissionLease>,
}
impl std::ops::Deref for AdmittedTemporalAdmgIdentification {
    type Target = AdmittedTemporalAdmgIdentificationData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Identify a temporal contrast with cumulative work and retained window admission.
/// # Errors
/// Invalid query, unsupported interventions, graph, cancellation, or resource refusal.
pub fn identify_temporal_admg_contrast_over_within(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
    variable_count: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<AdmittedTemporalAdmgIdentification, IdentificationError> {
    use crate::error::IdentificationBudget;
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    let set_level = |intervention: &Intervention| -> Result<Value, IdentificationError> {
        match intervention {
            Intervention::Set { value, .. } => Ok(value.clone()),
            _ => Err(IdentificationError::unsupported(
                "a temporal ADMG contrast requires Set interventions on both sides",
            )),
        }
    };
    let active = set_level(&query.active)?;
    let control = set_level(&query.control)?;
    let plan = plan_temporal_admg_window_within(template, query, budget, ctx)?;
    let window = template
        .unroll_window_over_within(plan.periods, variable_count, budget, ctx)
        .map_err(|error| match error {
            antecedent_graph::GraphError::TemporalMaterialization(error) => {
                crate::error::temporal_materialization_error(error)
            }
            other => IdentificationError::construction(other),
        })?;
    let count = plan.treated.len() as u64;
    let modifier_count = query.effect_modifiers.len() as u64;
    let nodes = window.admg().node_count() as u64;
    budget
        .charge_work(nodes)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let mut edges = 0u64;
    for i in 0..window.admg().node_count() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        edges = edges
            .checked_add(
                window.admg().children(antecedent_graph::DenseNodeId::from_index(i)).len() as u64
            )
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    }
    let traversals = count
        .checked_mul(modifier_count)
        .and_then(|n| n.checked_mul(nodes.checked_add(edges)?.checked_add(1)?))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    let work = traversals
        .checked_add(count)
        .and_then(|n| modifier_count.checked_mul(modifier_count).and_then(|m| n.checked_add(m)))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    budget
        .charge_work(work)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let bytes = nodes
        .checked_mul(256)
        .and_then(|n| count.checked_mul(128).and_then(|v| n.checked_add(v)))
        .and_then(|n| modifier_count.checked_mul(128).and_then(|v| n.checked_add(v)))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
    if bytes > budget.limits().max_bytes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let lease = Arc::new(ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
        _ => IdentificationError::budget(IdentificationBudget::Memory),
    })?);
    let mut schedule = Vec::new();
    schedule
        .try_reserve_exact(plan.treated.len())
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Memory))?;
    for &period in &plan.treated {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        schedule.push(
            window
                .coordinate(query.treatment, period)
                .ok_or(IdentificationError::UnknownVariable { id: query.treatment })?,
        );
    }
    let outcome = window
        .coordinate(query.outcome, plan.outcome_period)
        .ok_or(IdentificationError::UnknownVariable { id: query.outcome })?;
    let modifiers = modifier_coordinates_within(&window, &plan, query, &schedule, ctx)?;
    let prepared = PreparedAdmg::from_unrolled_within(window.clone(), budget, ctx)?;
    let causal_query = CausalQuery::TemporalEffect(query.clone());
    let mut result = if modifiers.is_empty() {
        let mut workspace = IdentificationWorkspace::default();
        IdIdentifier::new().identify_schedule_contrast(
            &prepared,
            outcome,
            &ScheduleContrast {
                schedule: &schedule,
                active_level: &active,
                control_level: &control,
            },
            causal_query,
            &mut workspace,
        )?
    } else {
        identify_conditional_contrast_prepared(
            &prepared,
            outcome,
            &ConditionalSchedule {
                schedule: &schedule,
                modifiers: &modifiers,
                active_level: &active,
                control_level: &control,
            },
            causal_query,
        )?
    };
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    result.derivation.push("temporal_admg.unrolled_window", format!(
        "complete ID on the {}-period unrolling of the temporal ADMG: {} treated period(s), outcome at period {}, history {}, {} effect modifier(s)",
        window.periods(), schedule.len(), plan.outcome_period, plan.history, modifiers.len()));
    result.retain_temporal_lease(Arc::clone(&lease));
    let obstruction = (result.status != IdentificationStatus::NonparametricallyIdentified)
        .then(|| describe_obstruction(&result, &window));
    Ok(AdmittedTemporalAdmgIdentification(Arc::new(AdmittedTemporalAdmgIdentificationData {
        result,
        window,
        plan,
        schedule,
        outcome,
        modifiers,
        obstruction,
        _lease: lease,
    })))
}

fn modifier_coordinates_within(
    window: &UnrolledTemporalAdmg,
    plan: &TemporalAdmgWindowPlan,
    query: &TemporalEffectQuery,
    schedule: &[VariableId],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<VariableId>, IdentificationError> {
    use crate::error::IdentificationBudget;
    const CHANGED_MODIFIER: &str = "an effect modifier of a temporal ADMG conditional contrast must be read before \
         the treated periods can change it (not the treatment, and not a descendant of a \
         treated period); next attempt: condition on a variable measured at the first \
         treated period that no treated period reaches, or ask for the unconditional \
         contrast";
    let first = plan.treated.iter().copied().min().unwrap_or(plan.origin);
    let mut modifiers = query.effect_modifiers.to_vec();
    modifiers.sort_unstable();
    if modifiers.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IdentificationError::unsupported(
            "effect modifiers of a temporal ADMG conditional contrast must be unique",
        ));
    }
    let graph = window.admg();
    let mut visited = vec![false; graph.node_count()];
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(graph.node_count())
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Memory))?;
    let mut out = Vec::with_capacity(modifiers.len());
    for modifier in modifiers {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        let coordinate = window
            .coordinate(modifier, first)
            .ok_or(IdentificationError::UnknownVariable { id: modifier })?;
        let target = graph.dense_of_static(coordinate);
        for treated in schedule {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            if *treated == coordinate {
                return Err(IdentificationError::unsupported(CHANGED_MODIFIER));
            }
            let (Some(from), Some(target)) = (graph.dense_of_static(*treated), target) else {
                continue;
            };
            visited.fill(false);
            stack.clear();
            visited[from.as_usize()] = true;
            stack.push(from);
            while let Some(node) = stack.pop() {
                if ctx.cancellation.is_cancelled() {
                    return Err(IdentificationError::Cancelled);
                }
                for child in graph.children(node) {
                    if ctx.cancellation.is_cancelled() {
                        return Err(IdentificationError::Cancelled);
                    }
                    if *child == target {
                        return Err(IdentificationError::unsupported(CHANGED_MODIFIER));
                    }
                    if !visited[child.as_usize()] {
                        visited[child.as_usize()] = true;
                        stack.push(*child);
                    }
                }
            }
        }
        out.push(coordinate);
    }
    Ok(out)
}

/// The treated schedule of a conditional contrast with the modifiers it is averaged over.
struct ConditionalSchedule<'a> {
    schedule: &'a [VariableId],
    modifiers: &'a [VariableId],
    active_level: &'a Value,
    control_level: &'a Value,
}

/// IDC at the two levels of the schedule, assembled into the mean contrast whose modifier
/// coordinates stay free.
fn identify_conditional_contrast_prepared(
    prepared: &PreparedAdmg,
    outcome: VariableId,
    request: &ConditionalSchedule<'_>,
    query: CausalQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let idc = IdcIdentifier::new();
    let mut workspace = IdentificationWorkspace::default();
    let mut law = |level: &Value| -> Result<IdentificationResult, IdentificationError> {
        let interventions: Vec<Intervention> = request
            .schedule
            .iter()
            .map(|coordinate| Intervention::set(*coordinate, level.clone()))
            .collect();
        idc.identify_conditional_at_levels(
            prepared,
            &[outcome],
            &interventions,
            request.modifiers,
            &mut workspace,
        )
    };
    let active = law(request.active_level)?;
    let control = law(request.control_level)?;
    for arm in [&active, &control] {
        if arm.status != IdentificationStatus::NonparametricallyIdentified
            || arm.estimands.len() != 1
        {
            let mut failed = arm.clone();
            failed.query = query;
            failed.derivation.push(
                "temporal_admg.conditional_window",
                "IDC did not identify one of the two conditional interventional laws on the \
                 unrolled window",
            );
            return Ok(failed);
        }
    }
    let mut arena = CausalExprArena::new();
    let left_root = arena.import(&active.arena, active.estimands[0].functional);
    let right_root = arena.import(&control.arena, control.estimands[0].functional);
    let left = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: left_root,
    });
    let right = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: right_root,
    });
    let contrast = arena.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
    let functional = arena.simplify(contrast).map_err(IdentificationError::expression)?;
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            "temporal_admg.conditional_window",
            Some(Arc::from(
                "difference of the active and control conditional interventional laws on the \
                 unrolled window, the modifiers read at the first treated period",
            )),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        EstimandMethod::GeneralId.as_str(),
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut assumptions = active.required_assumptions.clone();
    assumptions.extend_unique(&control.required_assumptions.entries);
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "temporal_admg.conditional_window",
        "difference of the active and control conditional interventional laws, each identified \
         by IDC at its hard-set schedule on the unrolled window; the modifiers stay free \
         variables averaged over their observed law",
    );
    Ok(IdentificationResult::identified(
        query,
        vec![estimand],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord { candidates_examined: 2, sets_returned: 1 },
    ))
}

fn describe_obstruction(result: &IdentificationResult, window: &UnrolledTemporalAdmg) -> String {
    let name = |coordinates: &[VariableId]| {
        coordinates
            .iter()
            .map(|c| {
                window.locate(*c).map_or_else(
                    || format!("#{}", c.raw()),
                    |(variable, period)| format!("v{}@{period}", variable.raw()),
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    match &result.hedge {
        Some(hedge) => format!(
            "not identified: the unrolled window contains a hedge; the C-forest {{{}}} contains \
             the smaller C-forest {{{}}} rooted at the outcome, so latent confounding connects \
             the treated periods to their effects on the outcome",
            name(&hedge.f),
            name(&hedge.f_prime)
        ),
        None => "not identified on the unrolled window".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{Lag, TemporalEffectQuery};
    use antecedent_graph::DenseNodeId;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    #[test]
    fn admitted_window_plan_preserves_small_semantics_and_refuses_huge_inputs() {
        use antecedent_core::execution::AdmissionPool;
        use antecedent_core::{ExecutionContext, TemporalMaterializationLimits};
        let limits = TemporalMaterializationLimits {
            max_offsets: 8,
            max_work: 1024,
            max_bytes: 16384,
            max_graph_nodes: 128,
            max_graph_edges: 128,
        };
        let pool = AdmissionPool::fixed(1, 16384);
        let ctx = ExecutionContext::for_tests(4).with_admission(pool.clone());
        let graph = template(false);
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2);
        let expected = plan_temporal_admg_window(&graph, &query).unwrap();
        let admitted = plan_temporal_admg_window_within(
            &graph,
            &query,
            &mut antecedent_core::TemporalBudget::new(limits),
            &ctx,
        )
        .unwrap();
        assert_eq!(&*admitted, &*expected);
        let free = pool.available().bytes();
        let clone = admitted.clone();
        drop(admitted);
        assert_eq!(pool.available().bytes(), free);
        drop(clone);
        assert_eq!(pool.available().bytes(), 16384);
        let mut huge = query.clone();
        huge.policy = antecedent_core::TemporalPolicy::sustained(i32::MIN, i32::MAX);
        assert!(matches!(
            plan_temporal_admg_window_within(
                &graph,
                &huge,
                &mut antecedent_core::TemporalBudget::new(limits),
                &ctx
            ),
            Err(IdentificationError::Budget { .. })
        ));
        let mut huge_horizon = query;
        huge_horizon.horizon_steps = u32::MAX;
        assert!(
            plan_temporal_admg_window_within(
                &graph,
                &huge_horizon,
                &mut antecedent_core::TemporalBudget::new(limits),
                &ctx
            )
            .is_err()
        );
        ctx.cancellation.cancel();
        assert!(matches!(
            plan_temporal_admg_window_within(
                &graph,
                &huge_horizon,
                &mut antecedent_core::TemporalBudget::new(limits),
                &ctx
            ),
            Err(IdentificationError::Cancelled)
        ));
    }

    #[test]
    fn bounded_identification_shares_window_and_plan_lease_on_clone() {
        use antecedent_core::execution::AdmissionPool;
        let pool = AdmissionPool::fixed(1, 1_000_000);
        let ctx = antecedent_core::ExecutionContext::for_tests(9).with_admission(pool.clone());
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2);
        let before = pool.available().bytes();
        let found = identify_temporal_admg_contrast_over_within(
            &template(false),
            &query,
            2,
            &mut budget,
            &ctx,
        )
        .unwrap();
        assert_eq!(found.result.status, IdentificationStatus::NonparametricallyIdentified);
        let held = pool.available().bytes();
        assert!(held < before);
        let cloned = found.clone();
        drop(found);
        assert_eq!(pool.available().bytes(), held);
        drop(cloned);
        assert_eq!(pool.available().bytes(), before);
    }

    /// x(t-1) -> x(t), y(t-1) -> x(t), and x(t-1) -> y(t) <- y(t-1); optionally x(t-1) <-> y(t).
    fn template(confounded: bool) -> TemporalAdmg {
        let mut graph = TemporalAdmg::empty();
        let x1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
        let x0 = graph.add_lagged(v(0), Lag::CONTEMPORANEOUS).unwrap();
        let y1 = graph.add_lagged(v(1), Lag::from_raw(1)).unwrap();
        let y0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x1, x0).unwrap();
        graph.insert_directed(y1, x0).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        graph.insert_directed(y1, y0).unwrap();
        if confounded {
            graph.insert_bidirected(x1, y0).unwrap();
        }
        graph
    }

    #[test]
    fn a_multi_lag_template_is_identified_when_history_is_observed() {
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2);
        let identified = identify_temporal_admg_contrast(&template(false), &query).unwrap();
        assert_eq!(
            identified.result.status,
            IdentificationStatus::NonparametricallyIdentified,
            "{:?}",
            identified.obstruction
        );
        assert!(identified.obstruction.is_none());
    }

    #[test]
    fn latent_confounding_of_treatment_and_its_next_period_outcome_is_a_named_hedge() {
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2);
        let blocked = identify_temporal_admg_contrast(&template(true), &query).unwrap();
        assert_eq!(blocked.result.status, IdentificationStatus::NotIdentified);
        let reason = blocked.obstruction.as_deref().expect("the obstruction is named");
        assert!(reason.contains("hedge") && reason.contains('@'), "{reason}");
    }

    #[test]
    fn a_modifier_that_is_the_outcome_is_refused() {
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_effect_modifiers([v(1)]);
        assert!(identify_temporal_admg_contrast(&template(false), &query).is_err());
    }

    /// The two-variable template plus a covariate `w` (variable 2) with `w(t-1) -> w(t)`,
    /// `w(t-1) -> x(t)` and `w(t) -> y(t)`; optionally `x(t) -> w(t)`, which makes the
    /// covariate a consequence of the treatment.
    fn template_with_covariate(treatment_reaches_it: bool) -> TemporalAdmg {
        let mut graph = template(false);
        let w1 = graph.add_lagged(v(2), Lag::from_raw(1)).unwrap();
        let w0 = graph.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
        let x0 = DenseNodeId::from_raw(1);
        let y0 = DenseNodeId::from_raw(3);
        graph.insert_directed(w1, w0).unwrap();
        graph.insert_directed(w1, x0).unwrap();
        graph.insert_directed(w0, y0).unwrap();
        if treatment_reaches_it {
            graph.insert_directed(x0, w0).unwrap();
        }
        graph
    }

    #[test]
    fn a_pre_treatment_covariate_identifies_the_conditional_contrast_on_the_window() {
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
            .with_horizon_steps(2)
            .with_effect_modifiers([v(2)]);
        let identified =
            identify_temporal_admg_contrast(&template_with_covariate(false), &query).unwrap();
        assert_eq!(
            identified.result.status,
            IdentificationStatus::NonparametricallyIdentified,
            "{:?}",
            identified.obstruction
        );
        let [modifier] = identified.modifiers[..] else { panic!("one modifier coordinate") };
        let (variable, period) = identified.window.locate(modifier).unwrap();
        assert_eq!(variable, v(2));
        assert_eq!(period, identified.plan.treated[0]);
        assert_eq!(identified.result.estimands.len(), 1);
    }

    #[test]
    fn a_covariate_the_treatment_reaches_is_refused_with_its_next_attempt() {
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
            .with_horizon_steps(2)
            .with_effect_modifiers([v(2)]);
        let error =
            identify_temporal_admg_contrast(&template_with_covariate(true), &query).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("descendant") && text.contains("next attempt"), "{text}");
    }

    #[test]
    fn the_window_plan_matches_the_identified_window() {
        let query = TemporalEffectQuery::sustained(v(0), v(1), 1, 1.0).with_horizon_steps(3);
        let plan = plan_temporal_admg_window(&template(false), &query).unwrap();
        let identified = identify_temporal_admg_contrast(&template(false), &query).unwrap();
        assert_eq!(plan, identified.plan);
        assert_eq!(identified.window.periods(), plan.periods);
        assert_eq!(plan.treated.len(), 2);
    }
}
