//! Conversion of aligned sequential evaluations to the reliability carrier.

use super::{OffPolicyBootstrap, RolloutEstimate, SequentialResult};
use crate::decision_reliability::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws, DrawSegment, SourceParadigm,
    StreamId, StreamPurpose, WithinStreamDependence,
};

fn invalid(message: &'static str) -> super::SequentialEvaluationError {
    refusal!(
        "invalid_argument",
        "sequential_evaluation.reliability_carrier",
        message,
        "evaluate a fixed finite policy set under one model and common stream"
    )
}

fn carrier(
    columns: Vec<(String, Vec<f64>)>,
    unit: &str,
    source: SourceParadigm,
    seed: u64,
    label: String,
) -> SequentialResult<DecisionDraws> {
    if unit.trim().is_empty() || columns.len() < 2 {
        return Err(invalid("a utility unit and at least two policies are required"));
    }
    let stream = StreamId::new(seed, StreamPurpose::Evaluation, 0, label);
    let candidates = columns
        .into_iter()
        .map(|(id, utilities)| {
            let segment = DrawSegment::whole(stream.clone(), utilities.len());
            CandidateDraws { id, utilities, segment }
        })
        .collect();
    let draws = DecisionDraws {
        utility_unit: unit.into(),
        producer: "sequential_evaluation".into(),
        source,
        coupling: CandidateCoupling::SharedSimulation,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates,
    };
    draws.validate().map_err(|_| invalid("invalid or inconsistent policy draw carrier"))?;
    Ok(draws)
}

/// Align rollout utilities only when the model and semantic streams agree.
///
/// # Errors
/// Different model, stream or rollout counts, or an invalid candidate set.
pub fn rollout_decision_draws(
    rows: &[RolloutEstimate],
    unit: &str,
) -> SequentialResult<DecisionDraws> {
    let first = rows.first().ok_or_else(|| invalid("no policies supplied"))?;
    if rows.iter().any(|row| {
        row.model_identity != first.model_identity
            || row.seed != first.seed
            || row.rollouts != first.rollouts
            || row.stream_scheme != first.stream_scheme
            || row.utilities.len() as u64 != row.rollouts
    }) {
        return Err(invalid("rollout models or common streams differ"));
    }
    carrier(
        rows.iter().map(|row| (row.policy_identity.clone(), row.utilities.clone())).collect(),
        unit,
        SourceParadigm::KnownLaw,
        first.seed,
        format!("rollout:{}:{}", first.model_identity, first.stream_scheme),
    )
}

/// Align whole-trajectory bootstrap estimates as an uncalibrated sampling law.
///
/// # Errors
/// Different models, behaviour policies, or an invalid candidate set.
pub fn bootstrap_decision_draws(
    result: &OffPolicyBootstrap,
    unit: &str,
) -> SequentialResult<DecisionDraws> {
    let first = result.estimates.first().ok_or_else(|| invalid("no policies supplied"))?;
    if result.estimates.len() != result.replicates.len()
        || result.estimates.iter().any(|row| {
            row.model_identity != first.model_identity
                || row.behaviour_identity != first.behaviour_identity
                || row.trajectories != first.trajectories
        })
    {
        return Err(invalid("bootstrap models, logging policies or trajectory counts differ"));
    }
    carrier(
        result
            .estimates
            .iter()
            .zip(&result.replicates)
            .map(|(row, values)| (row.policy_identity.clone(), values.clone()))
            .collect(),
        unit,
        SourceParadigm::SamplingDistribution,
        result.seed,
        format!(
            "trajectory_bootstrap:{}:{}:{}",
            first.model_identity, first.behaviour_identity, result.log_identity
        ),
    )
}
