//! Replay functional response grids and nested counterfactual results.

use crate::contract_reason::ContractReason;
use crate::{AnalysisResultWire, CausalQueryWire};

use super::moments::GRAM_ENTRY_REPLAY;
use super::{
    AnalysisResultContractWire, apply_moment_inverse, close_replay_value, invert_linear_moments,
    is_functional_effect_estimator_contract,
};

#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::contract_section) fn verify_nested_counterfactual_result(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
) -> Result<(), ContractReason> {
    let crate::CausalQueryWire::NestedCounterfactual {
        treatment,
        mediator,
        outcome,
        control_bits,
        active_bits,
    } = contract.target.query
    else {
        return Err(ContractReason::ProgramCheckedNestedCounterfactualTarget);
    };
    let operation = contract
        .program
        .as_ref()
        .and_then(|program| program.checked_nested_counterfactual.as_ref())
        .ok_or(ContractReason::ProgramCheckedNestedCounterfactual)?;
    if contract.graph_class != "Dag"
        || !matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        || !matches!(contract.identifier.as_deref(), None | Some("path_specific.natural"))
        || !matches!(contract.estimator.as_deref(), None | Some("mediation.linear"))
        || contract
            .program
            .as_ref()
            .and_then(|program| program.commitments.resolved_estimator.as_deref())
            != Some("mediation.linear")
        || !matches!(
            contract.inference_binding.as_ref().map(|binding| binding.inference.as_str()),
            Some("frequentist" | "bayesian")
        )
    {
        return Err(ContractReason::ProgramCheckedNestedCounterfactualScope);
    }
    let bayesian_point =
        contract.inference_binding.as_ref().is_some_and(|binding| binding.inference == "bayesian");
    if operation.format != 1
        || operation.treatment != treatment
        || operation.mediator != mediator
        || operation.outcome != outcome
        || operation.control_bits != control_bits
        || operation.active_bits != active_bits
        || operation.model != "linear_gaussian"
        || operation.procedure != "natural_direct_shared_exogenous"
        || !f64::from_bits(active_bits).is_finite()
        || !f64::from_bits(control_bits).is_finite()
    {
        return Err(ContractReason::ProgramCheckedNestedCounterfactualBinding);
    }
    let graph = contract
        .identification
        .as_ref()
        .map(|identity| &identity.graph)
        .ok_or(ContractReason::IdentitiesIdentification)?;
    let crate::GraphIdentityWire::Dag(dag) = graph else {
        return Err(ContractReason::ProgramCheckedNestedCounterfactualGraph);
    };
    let mut edges = dag.edges.clone();
    edges.sort_unstable();
    let mut expected_edges = vec![(treatment, mediator), (treatment, outcome), (mediator, outcome)];
    expected_edges.sort_unstable();
    if dag.node_count != 3 || edges != expected_edges {
        return Err(ContractReason::ProgramCheckedNestedCounterfactualGraph);
    }
    if !body.identification.estimands.iter().any(|estimand| {
        estimand.method == "path_specific.natural" && estimand.mediators == [mediator]
    }) {
        return Err(ContractReason::ProgramCheckedNestedCounterfactualIdentification);
    }
    let snapshot = contract
        .data_snapshot
        .as_ref()
        .ok_or(ContractReason::DependenciesNestedCounterfactualFitMoments)?;
    let fit = snapshot
        .nested_counterfactual_fit
        .as_ref()
        .ok_or(ContractReason::DependenciesNestedCounterfactualFitMoments)?;
    if fit.format != 1
        || fit.complete_case_rows < 4
        || fit.complete_case_rows > snapshot.row_count
        || fit.gram.iter().chain(&fit.outcome_cross).any(|value| !value.is_finite())
        || !GRAM_ENTRY_REPLAY.close(fit.gram[0], fit.complete_case_rows as f64)
        || (0..3).any(|row| {
            (0..3).any(|column| {
                !GRAM_ENTRY_REPLAY.close(fit.gram[row * 3 + column], fit.gram[column * 3 + row])
            })
        })
    {
        return Err(ContractReason::DependenciesNestedCounterfactualFitMoments);
    }
    let actual = body.estimate.ok_or(ContractReason::BodyNestedCounterfactualEstimate)?;
    if !actual.is_finite() {
        return Err(ContractReason::BodyNestedCounterfactualEstimate);
    }
    if bayesian_point
        && (body.standard_error.is_some_and(f64::is_finite)
            || body.interval_lower.is_some()
            || body.interval_upper.is_some()
            || body.posterior_artifact.is_some())
    {
        return Err(ContractReason::BodyNestedCounterfactualEstimate);
    }
    let expected = natural_direct_from_fit(fit, snapshot.row_count, active_bits, control_bits)?;
    if !close_replay_value(actual, expected) {
        return Err(ContractReason::BodyNestedCounterfactualEstimate);
    }
    Ok(())
}

/// A temporal nested result is the natural direct effect of the published
/// outcome-regression moments, not an arbitrary finite scalar.
///
/// The estimate is always that plug-in (under Bayesian inference the moments carry
/// the slope prior, so it is the posterior-mean natural direct effect). Beside it the
/// result may carry the uncertainty its inference mode publishes:
///
/// * Frequentist: the shared circular-block standard error and its interval. Both
///   endpoints are present together, ordered, contain the estimate, and the
///   standard error is finite and positive. A posterior artifact is refused.
/// * Bayesian: the slope-posterior summary (standard deviation and equal-tailed
///   interval, both endpoints together and ordered) and optionally its posterior
///   artifact. The interval is a quantile summary of draws, so it is not required to
///   contain the plug-in.
///
/// With no inference binding only the bare point is accepted.
pub(in crate::contract_section) fn verify_temporal_nested_point(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
) -> Result<(), ContractReason> {
    let estimate = body.estimate.filter(|value| value.is_finite());
    let Some(estimate) = estimate else {
        return Err(ContractReason::BodyTemporalNestedPoint);
    };
    let inference = contract.inference_binding.as_ref().map(|binding| binding.inference.as_str());
    let interval = match (body.interval_lower, body.interval_upper) {
        (None, None) => None,
        (Some(lower), Some(upper)) if lower.is_finite() && upper.is_finite() && lower <= upper => {
            Some((lower, upper))
        }
        _ => return Err(ContractReason::BodyTemporalNestedPoint),
    };
    let standard_error = match body.standard_error {
        Some(se) if se.is_finite() => {
            if se <= 0.0 {
                return Err(ContractReason::BodyTemporalNestedPoint);
            }
            Some(se)
        }
        _ => None,
    };
    let carries_uncertainty =
        interval.is_some() || standard_error.is_some() || body.posterior_artifact.is_some();
    match inference {
        _ if !carries_uncertainty => {}
        Some("frequentist") => {
            let (Some((lower, upper)), Some(_)) = (interval, standard_error) else {
                return Err(ContractReason::BodyTemporalNestedPoint);
            };
            if body.posterior_artifact.is_some() || estimate < lower || estimate > upper {
                return Err(ContractReason::BodyTemporalNestedPoint);
            }
        }
        Some("bayesian") => {
            // The posterior summary is either the body's ordered interval or the posterior
            // artifact that carries its draws and quantiles; a bare standard error is neither.
            if interval.is_none() && body.posterior_artifact.is_none() {
                return Err(ContractReason::BodyTemporalNestedPoint);
            }
        }
        _ => return Err(ContractReason::BodyTemporalNestedPoint),
    }
    let crate::CausalQueryWire::NestedCounterfactual { control_bits, active_bits, .. } =
        contract.target.query
    else {
        return Err(ContractReason::BodyTemporalNestedPoint);
    };
    let snapshot = contract
        .data_snapshot
        .as_ref()
        .ok_or(ContractReason::DependenciesNestedCounterfactualFitMoments)?;
    let fit = snapshot
        .nested_counterfactual_fit
        .as_ref()
        .ok_or(ContractReason::DependenciesNestedCounterfactualFitMoments)?;
    let expected = natural_direct_from_fit(fit, snapshot.row_count, active_bits, control_bits)
        .map_err(|reason| match reason {
            ContractReason::BodyNestedCounterfactualEstimate => {
                ContractReason::BodyTemporalNestedPoint
            }
            other => other,
        })?;
    if !close_replay_value(estimate, expected) {
        return Err(ContractReason::BodyTemporalNestedPoint);
    }
    Ok(())
}

pub(in crate::contract_section) fn natural_direct_from_fit(
    fit: &crate::NestedCounterfactualFitWire,
    snapshot_rows: u64,
    active_bits: u64,
    control_bits: u64,
) -> Result<f64, ContractReason> {
    if fit.format != 1
        || fit.complete_case_rows < 4
        || fit.complete_case_rows > snapshot_rows
        || fit.gram.iter().chain(&fit.outcome_cross).any(|value| !value.is_finite())
        || !GRAM_ENTRY_REPLAY.close(fit.gram[0], fit.complete_case_rows as f64)
        || (0..3).any(|row| {
            (0..3).any(|column| {
                !GRAM_ENTRY_REPLAY.close(fit.gram[row * 3 + column], fit.gram[column * 3 + row])
            })
        })
    {
        return Err(ContractReason::DependenciesNestedCounterfactualFitMoments);
    }
    let beta = invert_linear_moments(&fit.gram, 3)
        .and_then(|inverse| apply_moment_inverse(&inverse, &fit.outcome_cross, 3))
        .ok_or(ContractReason::DependenciesNestedCounterfactualFitRank)?;
    let expected = beta[1] * (f64::from_bits(active_bits) - f64::from_bits(control_bits));
    if !expected.is_finite() {
        return Err(ContractReason::BodyNestedCounterfactualEstimate);
    }
    Ok(expected)
}

pub(in crate::contract_section) fn is_scalar_functional_effect_contract(
    contract: &AnalysisResultContractWire,
) -> bool {
    is_functional_effect_estimator_contract(contract)
        && matches!(
            contract.target.query,
            crate::CausalQueryWire::AverageEffect { .. }
                | crate::CausalQueryWire::PathSpecific(_)
                | crate::CausalQueryWire::NestedCounterfactual { .. }
        )
}

pub(in crate::contract_section) fn is_response_functional_effect_contract(
    contract: &AnalysisResultContractWire,
) -> bool {
    is_functional_effect_estimator_contract(contract)
        && matches!(contract.target.query, crate::CausalQueryWire::Response(_))
}

/// Most grid points of a replayed response curve, whether the target declares them as explicit
/// values or as a linspace, and so most members a checked response-grid family may carry.
///
/// Every member is replayed against the same [`crate::distribution_replay::ReplayLimits`]: the
/// cells and operations each member is admitted for are charged to one budget, so the whole
/// grid costs at most the limits, whatever its length.
pub(in crate::contract_section) const MAX_RESPONSE_GRID_POINTS: usize = 100_000;

pub(in crate::contract_section) fn replay_functional_response_grid(
    target: &CausalQueryWire,
    family: &crate::CheckedFunctionalResponseGridWire,
    reported: &crate::ResponseIdentificationWire,
    laws: &crate::DistributionFactorLawsWire,
    limits: &crate::distribution_replay::ReplayLimits,
) -> Result<(), ContractReason> {
    use crate::{GridSpecWire, ResponseFunctionalWire, ResponseValueWire};
    let CausalQueryWire::Response(target) = target else {
        return Err(ContractReason::ProgramFunctionalEffectResponseGridTarget);
    };
    let ResponseFunctionalWire::MeanCurve { outcome, treatment } = &target.functional else {
        return Err(ContractReason::ProgramFunctionalEffectResponseGridTargetKind);
    };
    if family.format != 1 || family.outcome != *outcome || family.treatment != treatment.variable {
        return Err(ContractReason::ProgramFunctionalEffectResponseGridTargetBinding);
    }
    let expected_grid = match &treatment.grid {
        GridSpecWire::Values(values) => {
            if values.len() > MAX_RESPONSE_GRID_POINTS {
                return Err(ContractReason::FunctionalEffectResponseGridResourceLimit);
            }
            values.clone()
        }
        GridSpecWire::Linspace { start, end, points } => {
            let points = usize::try_from(*points)
                .map_err(|_| ContractReason::FunctionalEffectResponseGrid)?;
            if points == 0 || points > MAX_RESPONSE_GRID_POINTS {
                return Err(ContractReason::FunctionalEffectResponseGridResourceLimit);
            }
            if points == 1 {
                vec![*start]
            } else {
                (0..points)
                    .map(|index| start + (end - start) * index as f64 / (points - 1) as f64)
                    .collect()
            }
        }
    };
    if expected_grid.len() != family.members.len() || expected_grid.is_empty() {
        return Err(ContractReason::ProgramFunctionalEffectResponseGridMemberCount);
    }
    let crate::ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface {
        grid: reported_grid,
        dimension: 1,
        mean: means,
    }) = reported
    else {
        return Err(ContractReason::FunctionalEffectResponseResultShape);
    };
    if reported_grid.len() != expected_grid.len()
        || means.len() != expected_grid.len()
        || reported_grid
            .iter()
            .zip(&expected_grid)
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(ContractReason::FunctionalEffectResponseGridResultMismatch);
    }
    if laws.format != 1
        || laws.provider != "empirical_table"
        || laws.missing_row_policy != "joint_complete_case"
        || laws.complete_case_rows == 0
        || laws.source_rows < laws.complete_case_rows
    {
        return Err(ContractReason::FunctionalEffectProviderProvenance);
    }
    let mut remaining = *limits;
    for ((member, expected), reported) in family.members.iter().zip(expected_grid).zip(means) {
        if member.grid_value_bits != expected.to_bits() {
            return Err(ContractReason::ProgramFunctionalEffectResponseGridMemberValue);
        }
        let CausalQueryWire::Response(query) = &member.query else {
            return Err(ContractReason::ProgramFunctionalEffectResponseGridMemberQueryKind);
        };
        let ResponseFunctionalWire::InterventionResponse { outcome: member_outcome, interventions } =
            &query.functional
        else {
            return Err(ContractReason::ProgramFunctionalEffectResponseGridMemberFunctional);
        };
        if member_outcome != outcome
            || interventions.len() != 1
            || !matches!(interventions.first(), Some(crate::InterventionWire::Set { variable, value })
                if *variable == treatment.variable && value.to_value() == antecedent_core::Value::Float64(expected))
        {
            return Err(ContractReason::ProgramFunctionalEffectResponseGridMemberIntervention);
        }
        if member.identification.estimands.iter().all(|estimand| {
            estimand.method != "general.id" || estimand.functional != member.program.source
        }) || member.program.source != member.program.executable
        {
            return Err(ContractReason::ProgramFunctionalEffectResponseGridMemberProgram);
        }
        // One budget for the whole grid: a member is admitted for what the earlier members
        // left, and a spent budget refuses the next member.
        let charge = crate::distribution_replay::replay_functional_scalar(
            &member.query,
            Some(*reported),
            &member.program,
            laws,
            &remaining,
        )?;
        remaining = remaining.remaining_after(charge);
    }
    Ok(())
}
