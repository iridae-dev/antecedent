//! Full-source verification of longitudinal regime intervals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::AnalysisResultContractWire;
use super::dr_replay::{SourceReplay, SourceWitnessState};
use crate::analysis_result_artifact::LongitudinalScheduleResultWire;
use crate::contract_reason::ContractReason;
use crate::query_wire::LongitudinalScheduleTargetWire;
use crate::source_witness::{verify_source_witness, verify_source_witness_schema};
use crate::{AnalysisResultHeader, AnalysisResultWire, CausalQueryWire};
use antecedent_stats::fitted_regime_q::{FittedRegimeRows, estimate_fitted_regime_q};
use antecedent_stats::longitudinal_policy::{
    compare_longitudinal_schedules, learn_longitudinal_schedule,
};
use antecedent_stats::longitudinal_regime::{
    SequentialDrInputs, SubjectHistories, evaluate_sequential_dr_value_with_subject_scores,
    sequential_dr_pointwise_interval_95,
};

pub(super) fn declares_longitudinal_replay(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
) -> bool {
    matches!(&contract.target.query, CausalQueryWire::LongitudinalRegime(query)
        if query.periods == 4 && query.method == "sequential_dr"
            || query.periods == 2 && matches!(query.method.as_str(), "fitted_g_formula" | "compare_schedules" | "learn_schedule"))
        && body
            .longitudinal_regime
            .as_ref()
            .is_some_and(|fit| fit.value_interval_95.is_some() || fit.schedule_family.is_some())
}

pub(super) fn replay_longitudinal_interval(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
    source: Option<&SourceReplay<'_>>,
) -> Result<bool, ContractReason> {
    if !declares_longitudinal_replay(contract, body) {
        return Ok(false);
    }
    let (CausalQueryWire::LongitudinalRegime(query), Some(fit)) =
        (&contract.target.query, &body.longitudinal_regime)
    else {
        return Ok(false);
    };
    let source = source.ok_or(ContractReason::DependenciesCheckedLongitudinalRegimeOperation)?;
    let SourceWitnessState::Present(witness) = source.state else {
        return Err(ContractReason::LongitudinalRegimeReplaySource);
    };
    let snapshot =
        contract.data_snapshot.as_ref().ok_or(ContractReason::LongitudinalRegimeReplaySource)?;
    let observation =
        contract.observation.as_ref().ok_or(ContractReason::LongitudinalRegimeReplaySource)?;
    let storage_and_score_bytes = witness
        .row_count
        .saturating_mul((witness.columns.len() as u64).saturating_add(1).saturating_mul(8));
    if source.bounds.cancellation.is_cancelled()
        || storage_and_score_bytes > source.bounds.memory_bytes
    {
        return Err(ContractReason::LongitudinalRegimeReplayResource);
    }
    verify_source_witness(witness, &header.variable_names, snapshot, &source.bounds.witness)
        .and_then(|()| verify_source_witness_schema(witness, &header.variable_names, observation))
        .map_err(|_| ContractReason::LongitudinalRegimeReplaySource)?;
    let outcomes = witness
        .columns
        .get(query.outcome as usize)
        .ok_or(ContractReason::LongitudinalRegimeReplaySource)?;
    if let Some(declaration) = query.schedule_family {
        let execution = contract.execution.as_ref().ok_or(ContractReason::IdentitiesExecution)?;
        let mut ctx = antecedent_core::ExecutionContext::production(execution.seed, 1)
            .with_memory_limit(Some(source.bounds.memory_bytes));
        ctx.cancellation = source.bounds.cancellation.clone();
        let (result, value, se, interval) = match declaration.target {
            LongitudinalScheduleTargetWire::Contrasts { reference, contrasts } => {
                let comparison = compare_longitudinal_schedules(
                    &query.treatment_history,
                    &outcomes.values,
                    declaration.schedules,
                    reference,
                    contrasts,
                    declaration.draws as usize,
                    &ctx,
                )
                .map_err(schedule_replay_error)?;
                let value = comparison
                    .values
                    .iter()
                    .find(|&&(code, _)| code == reference)
                    .map(|&(_, value)| value)
                    .ok_or(ContractReason::BodyLongitudinalRegime)?;
                (LongitudinalScheduleResultWire::from(&comparison), value, None, None)
            }
            LongitudinalScheduleTargetWire::Learn { training_fold } => {
                let learned = learn_longitudinal_schedule(
                    &query.treatment_history,
                    &outcomes.values,
                    &query.fold_ids,
                    declaration.schedules,
                    training_fold,
                    declaration.draws as usize,
                    &ctx,
                )
                .map_err(schedule_replay_error)?;
                (
                    LongitudinalScheduleResultWire::from(&learned),
                    learned.selected_value,
                    Some(learned.selected_standard_error),
                    Some(learned.selected_interval),
                )
            }
        };
        let agrees = fit.schedule_family.as_ref() == Some(&result)
            && fit.value.to_bits() == value.to_bits()
            && fit.value_standard_error.map(f64::to_bits) == se.map(f64::to_bits)
            && fit.value_interval_95.map(|bounds| bounds.map(f64::to_bits))
                == interval.map(|bounds| bounds.map(f64::to_bits));
        return agrees.then_some(true).ok_or(ContractReason::BodyLongitudinalRegime);
    }
    let reported_bounds = fit.value_interval_95.ok_or(ContractReason::BodyLongitudinalRegime)?;
    if let Some(declaration) = query.fitted_q {
        let baseline = witness
            .columns
            .get(declaration.baseline as usize)
            .ok_or(ContractReason::LongitudinalRegimeReplaySource)?;
        let actions = query
            .regime_actions
            .first_chunk::<2>()
            .copied()
            .ok_or(ContractReason::BodyLongitudinalRegime)?;
        let execution = contract.execution.as_ref().ok_or(ContractReason::IdentitiesExecution)?;
        let mut ctx = antecedent_core::ExecutionContext::production(execution.seed, 1)
            .with_memory_limit(Some(source.bounds.memory_bytes));
        ctx.cancellation = source.bounds.cancellation.clone();
        let estimated = estimate_fitted_regime_q(
            &FittedRegimeRows {
                baseline: &baseline.values,
                actions: &query.treatment_history,
                outcomes: &outcomes.values,
                folds: &query.fold_ids,
            },
            actions,
            declaration.external_training_fold,
            declaration.bootstrap_replicates as usize,
            &ctx,
        )
        .map_err(|error| match error {
            antecedent_stats::StatsError::Admission(_)
            | antecedent_stats::StatsError::Cancelled => {
                ContractReason::LongitudinalRegimeReplayResource
            }
            _ => ContractReason::BodyLongitudinalRegime,
        })?;
        let agrees = estimated.value.to_bits() == fit.value.to_bits()
            && Some(estimated.standard_error.to_bits())
                == fit.value_standard_error.map(f64::to_bits)
            && estimated.interval.map(f64::to_bits) == reported_bounds.map(f64::to_bits)
            && Some(estimated.replicates_ok) == fit.fitted_q_bootstrap_replicates_ok
            && (estimated.evaluation_subjects as f64).to_bits()
                == fit.effective_sample_size.to_bits();
        return agrees.then_some(true).ok_or(ContractReason::BodyLongitudinalRegime);
    }
    let (summary, scores) = evaluate_sequential_dr_value_with_subject_scores(
        &outcomes.values,
        &SubjectHistories {
            treatment: &query.treatment_history,
            treatment_probability: &query.treatment_probabilities,
            outcome_observed: &query.outcome_observed,
            censoring_probability: &query.censoring_probabilities,
            periods: query.periods,
            minimum_probability: query.minimum_probability,
        },
        &SequentialDrInputs {
            actions: &query.regime_actions,
            observation_history: &query.observation_history,
            q_predictions: &query.q_predictions,
        },
    )
    .map_err(|_| ContractReason::BodyLongitudinalRegime)?;
    let observed = query.outcome_observed.iter().filter(|&&on| on).count();
    let matches = query
        .outcome_observed
        .iter()
        .zip(query.treatment_history.chunks_exact(query.periods))
        .zip(query.regime_actions.chunks_exact(query.periods))
        .filter(|&((&observed, treatment), actions)| observed && treatment == actions)
        .count();
    let (se, bounds) =
        sequential_dr_pointwise_interval_95(&summary, &scores, query.periods, observed, matches)
            .ok_or(ContractReason::BodyLongitudinalRegime)?;
    if source.bounds.cancellation.is_cancelled() {
        return Err(ContractReason::LongitudinalRegimeReplayResource);
    }
    if [
        (summary.value, fit.value),
        (summary.effective_sample_size, fit.effective_sample_size),
        (summary.matched_observed_fraction, fit.matched_observed_fraction),
        (summary.maximum_weight, fit.maximum_weight),
        (summary.minimum_action_probability, fit.minimum_action_probability),
        (summary.minimum_censoring_probability, fit.minimum_censoring_probability),
        (se, fit.value_standard_error.unwrap_or(f64::NAN)),
        (bounds[0], reported_bounds[0]),
        (bounds[1], reported_bounds[1]),
    ]
    .into_iter()
    .any(|(replayed, recorded)| replayed.to_bits() != recorded.to_bits())
    {
        return Err(ContractReason::BodyLongitudinalRegime);
    }
    Ok(true)
}

fn schedule_replay_error(error: antecedent_stats::StatsError) -> ContractReason {
    match error {
        antecedent_stats::StatsError::Admission(_) | antecedent_stats::StatsError::Cancelled => {
            ContractReason::LongitudinalRegimeReplayResource
        }
        _ => ContractReason::BodyLongitudinalRegime,
    }
}
