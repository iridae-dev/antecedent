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

/// Identification of a scheduled contrast on a temporal ADMG, with its window.
#[derive(Clone, Debug)]
pub struct TemporalAdmgIdentification {
    /// The identification result over the window's coordinates.
    pub result: IdentificationResult,
    /// The window the result is stated on.
    pub window: UnrolledTemporalAdmg,
    /// Where the policy, the outcome and the history sit in the window.
    pub plan: TemporalAdmgWindowPlan,
    /// Window coordinate of the treatment at each treated period, in period order.
    pub schedule: Vec<VariableId>,
    /// Window coordinate of the outcome.
    pub outcome: VariableId,
    /// Window coordinate of each effect modifier, read at the first treated period; empty for
    /// an unconditional contrast.
    pub modifiers: Vec<VariableId>,
    /// What blocks identification, in template terms; `None` when identified.
    pub obstruction: Option<String>,
}

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
) -> Result<TemporalAdmgWindowPlan, IdentificationError> {
    query
        .validate()
        .map_err(|_| IdentificationError::unsupported("invalid temporal effect query"))?;
    let offsets =
        query.policy.active_offsets().map_err(|error| IdentificationError::construction(error))?;
    let history = template
        .nodes()
        .iter()
        .filter_map(|node| match node {
            antecedent_core::NodeRef::Lagged { lag, .. } => Some(lag.raw()),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let earliest = offsets.iter().copied().min().unwrap_or(0).min(0);
    let origin = history + earliest.unsigned_abs();
    let periods_of = |offset: i32| -> Result<u32, IdentificationError> {
        u32::try_from(i64::from(origin) + i64::from(offset)).map_err(|_| {
            IdentificationError::out_of_range("a treated step precedes the unrolled window")
        })
    };
    // The same reading as every temporal route: the outcome sits at offset `horizon_steps - 1`
    // from the policy origin, the offset the treated steps are counted from.
    let outcome_period = periods_of(query.outcome_offset())?;
    let mut treated = Vec::with_capacity(offsets.len());
    for &offset in offsets.iter() {
        treated.push(periods_of(offset)?);
    }
    let last = treated.iter().copied().max().unwrap_or(0).max(outcome_period);
    Ok(TemporalAdmgWindowPlan { periods: last + 1, origin, history, treated, outcome_period })
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
    let largest = template
        .nodes()
        .iter()
        .filter_map(|node| match node {
            antecedent_core::NodeRef::Lagged { variable, .. } => Some(variable.raw()),
            _ => None,
        })
        .chain([query.treatment.raw(), query.outcome.raw()])
        .chain(query.effect_modifiers.iter().map(|modifier| modifier.raw()))
        .max()
        .unwrap_or(0);
    identify_temporal_admg_contrast_over(template, query, largest + 1)
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
    let set_level = |intervention: &Intervention| -> Result<Value, IdentificationError> {
        match intervention {
            Intervention::Set { value, .. } => Ok(value.clone()),
            _ => Err(IdentificationError::unsupported(
                "a temporal ADMG contrast requires Set interventions on both sides",
            )),
        }
    };
    let plan = plan_temporal_admg_window(template, query)?;
    let active = set_level(&query.active)?;
    let control = set_level(&query.control)?;
    let window = template
        .unroll_window_over(plan.periods, variable_count)
        .map_err(|error| IdentificationError::construction(error))?;
    let unknown = |variable: VariableId| IdentificationError::UnknownVariable { id: variable };
    let schedule = plan
        .treated
        .iter()
        .map(|&period| window.coordinate(query.treatment, period).ok_or(unknown(query.treatment)))
        .collect::<Result<Vec<_>, _>>()?;
    let outcome =
        window.coordinate(query.outcome, plan.outcome_period).ok_or(unknown(query.outcome))?;
    let modifiers = modifier_coordinates(&window, &plan, query, &schedule)?;
    let causal_query = CausalQuery::TemporalEffect(query.clone());
    let mut result = if modifiers.is_empty() {
        let prepared = PreparedAdmg::new(window.admg().clone())?;
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
        identify_conditional_contrast(
            &window,
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
    result.derivation.push(
        "temporal_admg.unrolled_window",
        format!(
            "complete ID on the {}-period unrolling of the temporal ADMG: {} treated period(s), \
             outcome at period {}, history {}, {} effect modifier(s)",
            window.periods(),
            schedule.len(),
            plan.outcome_period,
            plan.history,
            modifiers.len()
        ),
    );
    let obstruction = (result.status != IdentificationStatus::NonparametricallyIdentified)
        .then(|| describe_obstruction(&result, &window));
    Ok(TemporalAdmgIdentification {
        result,
        window,
        plan,
        schedule,
        outcome,
        modifiers,
        obstruction,
    })
}

/// Window coordinates of the effect modifiers, read at the first treated period.
///
/// A modifier is the state at decision time, so it must precede every intervention it
/// conditions: one the treated periods can reach is a different variable under the policy.
fn modifier_coordinates(
    window: &UnrolledTemporalAdmg,
    plan: &TemporalAdmgWindowPlan,
    query: &TemporalEffectQuery,
    schedule: &[VariableId],
) -> Result<Vec<VariableId>, IdentificationError> {
    if query.effect_modifiers.is_empty() {
        return Ok(Vec::new());
    }
    let first = plan.treated.iter().copied().min().unwrap_or(plan.origin);
    let mut template_modifiers: Vec<VariableId> = query.effect_modifiers.to_vec();
    template_modifiers.sort_unstable();
    if template_modifiers.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IdentificationError::unsupported(
            "effect modifiers of a temporal ADMG conditional contrast must be unique",
        ));
    }
    let mut coordinates = Vec::with_capacity(template_modifiers.len());
    for modifier in template_modifiers {
        let coordinate = window
            .coordinate(modifier, first)
            .ok_or(IdentificationError::UnknownVariable { id: modifier })?;
        let node = window.admg().dense_of_static(coordinate);
        let reached = schedule.iter().any(|treated| {
            *treated == coordinate
                || matches!(
                    (window.admg().dense_of_static(*treated), node),
                    (Some(from), Some(to)) if window.admg().reaches(from, to)
                )
        });
        if reached {
            return Err(IdentificationError::unsupported(
                "an effect modifier of a temporal ADMG conditional contrast must be read before \
                 the treated periods can change it (not the treatment, and not a descendant of a \
                 treated period); next attempt: condition on a variable measured at the first \
                 treated period that no treated period reaches, or ask for the unconditional \
                 contrast",
            ));
        }
        coordinates.push(coordinate);
    }
    Ok(coordinates)
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
fn identify_conditional_contrast(
    window: &UnrolledTemporalAdmg,
    outcome: VariableId,
    request: &ConditionalSchedule<'_>,
    query: CausalQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let idc = IdcIdentifier::new();
    let prepared = idc.prepare(window.admg())?;
    let mut workspace = IdentificationWorkspace::default();
    let mut law = |level: &Value| -> Result<IdentificationResult, IdentificationError> {
        let interventions: Vec<Intervention> = request
            .schedule
            .iter()
            .map(|coordinate| Intervention::set(*coordinate, level.clone()))
            .collect();
        idc.identify_conditional_at_levels(
            &prepared,
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
        let reason = blocked.obstruction.expect("the obstruction is named");
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
