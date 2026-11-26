//! Full-source replay of the randomized delayed-entry incidence interval.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::AnalysisResultContractWire;
use super::dr_replay::{SourceReplay, SourceWitnessState};
use crate::contract_reason::ContractReason;
use crate::distribution_replay::ReplayLimits;
use crate::source_witness::{verify_source_witness, verify_source_witness_schema};
use crate::{AnalysisResultHeader, AnalysisResultWire, CausalQueryWire};
use antecedent_stats::survival::{
    KnownCensoring, SurvivalEndpoint, SurvivalRows,
    randomized_survival_bootstrap_intervals_controlled, randomized_survival_ipcw_summary,
    randomized_survival_summary,
};

pub(super) fn declares_incidence_replay(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
) -> bool {
    contract.estimator.as_deref() == Some("randomized.survival_product_limit")
        && matches!(&contract.target.query, CausalQueryWire::Survival(query)
            if query.target_cause.is_some() && query.delayed_entry.is_some())
        && body.survival.as_ref().is_some_and(|curve| curve.difference_at_tau_interval.is_some())
}

#[allow(
    clippy::too_many_lines,
    reason = "source identity, replay bounds and the retained output are checked in order"
)]
pub(super) fn replay_delayed_entry_incidence(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
    source: Option<&SourceReplay<'_>>,
    limits: &ReplayLimits,
) -> Result<bool, ContractReason> {
    if !declares_incidence_replay(contract, body) {
        return Ok(false);
    }
    use ContractReason::{
        BodySurvivalIncidence, DependenciesCheckedSurvivalIncidenceOperation,
        SurvivalIncidenceReplayResource, SurvivalIncidenceReplaySource,
    };
    let (CausalQueryWire::Survival(query), Some(curve)) = (&contract.target.query, &body.survival)
    else {
        return Err(BodySurvivalIncidence);
    };
    let (Some(target_cause), Some(entry_column)) = (query.target_cause, query.delayed_entry) else {
        return Err(BodySurvivalIncidence);
    };
    let source = source.ok_or(DependenciesCheckedSurvivalIncidenceOperation)?;
    let requested = curve.bootstrap_replicates_requested.ok_or(BodySurvivalIncidence)?;
    if contract.inference_binding.as_ref().map(|binding| binding.bootstrap_replicates)
        != Some(requested)
    {
        return Err(BodySurvivalIncidence);
    }
    let seed = contract.execution.as_ref().ok_or(BodySurvivalIncidence)?.seed;
    let SourceWitnessState::Present(witness) = source.state else {
        return Err(SurvivalIncidenceReplaySource);
    };
    let snapshot = contract.data_snapshot.as_ref().ok_or(SurvivalIncidenceReplaySource)?;
    let observation = contract.observation.as_ref().ok_or(SurvivalIncidenceReplaySource)?;
    let cancelled = || source.bounds.cancellation.is_cancelled();
    let known_width = query.censoring_columns.len() as u64;
    let working_memory = witness
        .row_count
        .saturating_mul(
            (witness.columns.len() as u64)
                .saturating_add(known_width.saturating_mul(3))
                .saturating_add(30),
        )
        .saturating_mul(8)
        .saturating_add(u64::from(requested).saturating_mul(16));
    if cancelled() || working_memory > source.bounds.memory_bytes {
        return Err(SurvivalIncidenceReplayResource);
    }
    verify_source_witness(witness, &header.variable_names, snapshot, &source.bounds.witness)
        .and_then(|()| verify_source_witness_schema(witness, &header.variable_names, observation))
        .map_err(|_| SurvivalIncidenceReplaySource)?;
    let column = |id: u32| {
        witness
            .columns
            .get(id as usize)
            .map(|col| col.values.as_slice())
            .ok_or(SurvivalIncidenceReplaySource)
    };
    let duration = column(query.duration)?;
    let event = column(query.event)?;
    let entry = column(entry_column)?;
    let event_knots = duration
        .iter()
        .zip(event)
        .filter(|&(&time, &cause)| cause > 0.0 && time <= query.tau)
        .map(|(&time, _)| time.to_bits())
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let grid_work = known_width.max(event_knots).saturating_add(4);
    let operation_count =
        witness.row_count.saturating_mul(grid_work).saturating_mul(u64::from(requested) + 1);
    if cancelled() || operation_count > limits.max_operations as u64 {
        return Err(SurvivalIncidenceReplayResource);
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the source events are checked for integrality and range before conversion"
    )]
    let events = event
        .iter()
        .map(|&value| {
            if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= i64::MAX as f64
            {
                Ok(value as i64)
            } else {
                Err(BodySurvivalIncidence)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let treated = column(query.treatment)?
        .iter()
        .map(|&value| match value {
            0.0 => Ok(false),
            1.0 => Ok(true),
            _ => Err(BodySurvivalIncidence),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let active_count = treated.iter().filter(|&&arm| arm).count();
    let mut probabilities = Vec::new();
    let known = if query.censoring_columns.is_empty() {
        None
    } else {
        let columns =
            query.censoring_columns.iter().map(|&id| column(id)).collect::<Result<Vec<_>, _>>()?;
        probabilities.reserve(duration.len().saturating_mul(columns.len()));
        for row in 0..duration.len() {
            probabilities.extend(columns.iter().map(|col| col[row]));
        }
        Some(KnownCensoring {
            times: &query.censoring_times,
            survival: &probabilities,
            minimum_probability: query.censoring_probability_floor.ok_or(BodySurvivalIncidence)?,
        })
    };
    let rows = SurvivalRows {
        duration,
        event_code: &events,
        treated: &treated,
        delayed_entry: Some(entry),
    };
    let endpoint = SurvivalEndpoint::CumulativeIncidence { target_cause };
    let summary = match known {
        Some(known) => randomized_survival_ipcw_summary(&rows, &known, query.tau, endpoint)
            .map(|value| value.0),
        None => randomized_survival_summary(&rows, query.tau, endpoint),
    }
    .map_err(|_| BodySurvivalIncidence)?;
    let interval = randomized_survival_bootstrap_intervals_controlled(
        &rows,
        known.as_ref(),
        query.tau,
        endpoint,
        requested,
        seed,
        cancelled,
    )
    .map_err(|_| {
        if cancelled() { SurvivalIncidenceReplayResource } else { BodySurvivalIncidence }
    })?;
    let same = |replayed: &[f64], stored: &[f64]| {
        replayed.len() == stored.len()
            && replayed.iter().zip(stored).all(|(left, right)| left.to_bits() == right.to_bits())
    };
    if !same(&summary.times, &curve.times)
        || !same(&summary.control, &curve.control)
        || !same(&summary.treated, &curve.treated)
        || curve.assignment_counts != Some([duration.len() - active_count, active_count])
        || curve.minimum_event_risk_set != summary.minimum_event_risk_set
        || curve.bootstrap_replicates_ok != Some(interval.replicates_ok)
        || !curve
            .difference_at_tau_interval
            .is_some_and(|stored| same(&interval.difference_at_tau, &stored))
    {
        return Err(BodySurvivalIncidence);
    }
    Ok(true)
}
