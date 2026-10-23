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

use antecedent_core::{CausalQuery, Intervention, TemporalEffectQuery, Value, VariableId};
use antecedent_graph::{TemporalAdmg, UnrolledTemporalAdmg};

use crate::error::IdentificationError;
use crate::id::{IdIdentifier, ScheduleContrast};
use crate::identifier::IdentificationWorkspace;
use crate::prepared::PreparedAdmg;
use crate::result::{IdentificationResult, IdentificationStatus};

/// Identification of a scheduled contrast on a temporal ADMG, with its window.
#[derive(Clone, Debug)]
pub struct TemporalAdmgIdentification {
    /// The identification result over the window's coordinates.
    pub result: IdentificationResult,
    /// The window the result is stated on.
    pub window: UnrolledTemporalAdmg,
    /// Window coordinate of the treatment at each treated period, in period order.
    pub schedule: Vec<VariableId>,
    /// Window coordinate of the outcome.
    pub outcome: VariableId,
    /// What blocks identification, in template terms; `None` when identified.
    pub obstruction: Option<String>,
}

/// Identify the pulse or sustained contrast of `query` on `template`.
///
/// The policy origin sits after as many history periods as the template's longest lag, the
/// outcome at `origin + horizon_steps`, and the treated periods at the policy's offsets.
///
/// # Errors
///
/// A modifier-conditional query (the conditional law needs IDC on the same window), a policy
/// with a negative offset beyond the history, a non-`Set` intervention, a treatment or outcome
/// that is not a template variable, or a window the graph crate cannot hold. A contrast the
/// window does not identify is not an error: the result is not identified and
/// [`TemporalAdmgIdentification::obstruction`] names the hedge.
pub fn identify_temporal_admg_contrast(
    template: &TemporalAdmg,
    query: &TemporalEffectQuery,
) -> Result<TemporalAdmgIdentification, IdentificationError> {
    query
        .validate()
        .map_err(|_| IdentificationError::unsupported("invalid temporal effect query"))?;
    if !query.effect_modifiers.is_empty() {
        return Err(IdentificationError::unsupported(
            "a conditional contrast on a temporal ADMG needs IDC on the unrolled window; the \
             unconditional pulse and sustained contrasts are identified here",
        ));
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
    let offsets = query
        .policy
        .active_offsets()
        .map_err(|error| IdentificationError::msg(error.to_string()))?;
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
        u32::try_from(i64::from(origin) + i64::from(offset))
            .map_err(|_| IdentificationError::msg("a treated step precedes the unrolled window"))
    };
    let outcome_period = periods_of(i32::try_from(query.horizon_steps).unwrap_or(i32::MAX))?;
    let mut treated = Vec::with_capacity(offsets.len());
    for &offset in offsets.iter() {
        treated.push(periods_of(offset)?);
    }
    let last = treated.iter().copied().max().unwrap_or(0).max(outcome_period);
    let window = template
        .unroll_window(last + 1)
        .map_err(|error| IdentificationError::msg(error.to_string()))?;
    let unknown = |variable: VariableId| IdentificationError::UnknownVariable { id: variable };
    let schedule = treated
        .iter()
        .map(|&period| window.coordinate(query.treatment, period).ok_or(unknown(query.treatment)))
        .collect::<Result<Vec<_>, _>>()?;
    let outcome = window.coordinate(query.outcome, outcome_period).ok_or(unknown(query.outcome))?;
    let prepared = PreparedAdmg::new(window.admg().clone())?;
    let mut workspace = IdentificationWorkspace::default();
    let mut result = IdIdentifier::new().identify_schedule_contrast(
        &prepared,
        outcome,
        &ScheduleContrast { schedule: &schedule, active_level: &active, control_level: &control },
        CausalQuery::TemporalEffect(query.clone()),
        &mut workspace,
    )?;
    result.derivation.push(
        "temporal_admg.unrolled_window",
        format!(
            "complete ID on the {}-period unrolling of the temporal ADMG: {} treated period(s), \
             outcome at period {outcome_period}, history {history}",
            window.periods(),
            schedule.len()
        ),
    );
    let obstruction = (result.status == IdentificationStatus::NotIdentified)
        .then(|| describe_obstruction(&result, &window));
    Ok(TemporalAdmgIdentification { result, window, schedule, outcome, obstruction })
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
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(1);
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
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(1);
        let blocked = identify_temporal_admg_contrast(&template(true), &query).unwrap();
        assert_eq!(blocked.result.status, IdentificationStatus::NotIdentified);
        let reason = blocked.obstruction.expect("the obstruction is named");
        assert!(reason.contains("hedge") && reason.contains('@'), "{reason}");
    }

    #[test]
    fn a_conditional_query_is_not_silently_answered_unconditionally() {
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_effect_modifiers([v(1)]);
        assert!(identify_temporal_admg_contrast(&template(false), &query).is_err());
    }
}
