//! Estimator-specific contract replay.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::consume::{require_nested_digest, require_payload_digest};

use antecedent_core::IdentityDomain;

use super::AnalysisResultContractWire;
use crate::contract_reason::ContractReason;
use crate::contract_wire::{CheckedAipwLoweringWire, checked_aipw_rows_digest};
use crate::identity::{PayloadDigestWire, digest_wire, score_reuse_digest, target_weights_digest};
use crate::query_wire::TargetPopulationWire;
use crate::{AnalysisResultWire, CausalQueryWire};

/// Whether the contract advertises `estimator` or its program resolved to it.
///
/// Studies built without an explicit estimator leave `contract.estimator`
/// unset and commit the resolver's choice only in the program, so every
/// estimator-specific verification gate must consult both.
pub(in crate::contract_section) fn contract_uses_estimator(
    contract: &AnalysisResultContractWire,
    estimator: &str,
) -> bool {
    contract.estimator.as_deref() == Some(estimator)
        || contract.program.as_ref().is_some_and(|program| {
            program.commitments.resolved_estimator.as_deref() == Some(estimator)
        })
}

pub(in crate::contract_section) fn is_functional_effect_estimator_contract(
    contract: &AnalysisResultContractWire,
) -> bool {
    contract_uses_estimator(contract, "functional.effect")
}

pub(in crate::contract_section) fn is_functional_distribution_contract(
    contract: &AnalysisResultContractWire,
) -> bool {
    contract_uses_estimator(contract, "functional.distribution")
}

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
        || contract.structure_source != "explicit"
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
        || (fit.gram[0] - fit.complete_case_rows as f64).abs() > 1e-9
        || (0..3).any(|row| {
            (0..3).any(|column| {
                (fit.gram[row * 3 + column] - fit.gram[column * 3 + row]).abs() > 1e-9
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
    if (actual - expected).abs() > 1e-8 * expected.abs().max(1.0) {
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
    if (estimate - expected).abs() > 1e-8 * expected.abs().max(1.0) {
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
        || (fit.gram[0] - fit.complete_case_rows as f64).abs() > 1e-9
        || (0..3).any(|row| {
            (0..3).any(|column| {
                (fit.gram[row * 3 + column] - fit.gram[column * 3 + row]).abs() > 1e-9
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

pub(in crate::contract_section) fn replay_functional_response_grid(
    target: &CausalQueryWire,
    family: &crate::CheckedFunctionalResponseGridWire,
    reported: &crate::ResponseIdentificationWire,
    laws: &crate::DistributionFactorLawsWire,
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
        GridSpecWire::Values(values) => values.clone(),
        GridSpecWire::Linspace { start, end, points } => {
            let points = usize::try_from(*points)
                .map_err(|_| ContractReason::FunctionalEffectResponseGrid)?;
            if points == 0 || points > 100_000 {
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
        crate::distribution_replay::replay_functional_scalar(
            &member.query,
            Some(*reported),
            &member.program,
            laws,
        )?;
    }
    Ok(())
}

pub(in crate::contract_section) fn verify_checked_linear_adjustment(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let lowering = contract
        .program
        .as_ref()
        .and_then(|program| program.checked_linear_adjustment_lowering.as_ref());
    // The checked linear-adjustment lowering currently covers the supplied
    // or accepted DAG ATE. Other graph classes may select the same numerical
    // estimator inside an envelope, but they do not have this lowering yet.
    let is_linear = contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, crate::CausalQueryWire::AverageEffect { .. })
        && (contract.estimator.as_deref() == Some("linear.adjustment.ate")
            || contract.program.as_ref().is_some_and(|program| {
                program.commitments.resolved_estimator.as_deref() == Some("linear.adjustment.ate")
            }));
    if is_linear && lowering.is_none() {
        unresolved.push(ContractReason::ProgramCheckedLinearAdjustmentLowering);
        return;
    }
    let Some(lowering) = lowering else { return };
    if !verify_linear_adjustment_lowering(contract, lowering) {
        unresolved.push(ContractReason::ProgramCheckedLinearAdjustmentBinding);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "The lowering verifier compares one complete checked product across its target, design, and evidence."
)]
pub(in crate::contract_section) fn verify_linear_adjustment_lowering(
    contract: &AnalysisResultContractWire,
    lowering: &crate::CheckedLinearAdjustmentLoweringWire,
) -> bool {
    let (Some(product), Some(program), Some(snapshot), Some(binding)) = (
        contract.identification_product.as_ref(),
        contract.program.as_ref(),
        contract.data_snapshot.as_ref(),
        contract.inference_binding.as_ref(),
    ) else {
        return false;
    };
    if lowering.format != 1
        || lowering.backend != "faer"
        || lowering.population != TargetPopulationWire::AllObserved
        || lowering.complete_case_rows == 0
        || lowering.complete_case_rows > snapshot.row_count
        || lowering.design_columns.first().map(String::as_str) != Some("intercept")
        || lowering.design_columns.get(1).map(String::as_str) != Some("treatment")
        || lowering.design_columns.len() != lowering.adjustment.len() + 2
        || lowering
            .design_columns
            .iter()
            .skip(2)
            .zip(&lowering.adjustment)
            .any(|(role, id)| role != &format!("covariate:{id}"))
        || !valid_linear_fit_kind(&lowering.fit_kind)
        || !matches!(
            lowering.se_kind.as_str(),
            "homoskedastic"
                | "hc0"
                | "hc1"
                | "hc2"
                | "hc3"
                | "cluster"
                | "multiway"
                | "newey_west"
                | "panel_cluster_hac"
        )
        || matches!(lowering.se_kind.as_str(), "newey_west" | "panel_cluster_hac")
            != lowering.se_lag.is_some()
    {
        return false;
    }
    let Some(estimand) = product.estimands.iter().find(|estimand| {
        estimand.functional == lowering.functional
            && estimand.method == "backdoor.adjustment"
            && estimand.adjustment_set == lowering.adjustment
    }) else {
        return false;
    };
    let target_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            active,
            control,
            outcome_functional,
            target_population,
            effect_modifiers,
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && intervention_bits(active, *treatment) == Some(lowering.active_bits)
                && intervention_bits(control, *treatment) == Some(lowering.control_bits)
                && matches!(outcome_functional, crate::query_wire::OutcomeFunctionalWire::Mean)
                && target_population == &lowering.population
                && effect_modifiers.is_empty()
        }
        _ => false,
    };
    if !product.estimands.iter().any(|candidate| candidate.functional == lowering.functional) {
        return false;
    }
    let Ok(mut arena) = crate::expr_arena_from_wire(&lowering.arena) else {
        return false;
    };
    let treatment = antecedent_core::VariableId::from_raw(lowering.treatment);
    let outcome = antecedent_core::VariableId::from_raw(lowering.outcome);
    let adjustment = lowering
        .adjustment
        .iter()
        .copied()
        .map(antecedent_core::VariableId::from_raw)
        .collect::<Vec<_>>();
    let expected_source = arena.backdoor_ate(
        treatment,
        outcome,
        &adjustment,
        antecedent_core::Value::Float64(f64::from_bits(lowering.active_bits)),
        antecedent_core::Value::Float64(f64::from_bits(lowering.control_bits)),
    );
    let z = arena.intern_var_set(adjustment.iter().copied());
    let y = arena.intern_var_set([outcome]);
    let tz = arena.intern_var_set(std::iter::once(treatment).chain(adjustment.iter().copied()));
    let empty = arena.empty_var_set();
    let no_intervention = arena.empty_intervention_set();
    let marginal_z = arena.intern_distribution(
        z,
        empty,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let outcome_given_tz = arena.intern_distribution(
        y,
        tz,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let factors = arena.intern_list([outcome_given_tz, marginal_z]);
    let product_expr = arena.intern(antecedent_expr::ExprNode::Product(factors));
    let averaged =
        arena.intern(antecedent_expr::ExprNode::SumOut { variables: z, expr: product_expr });
    let expected_executable = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(outcome),
        distribution: averaged,
    });
    let expected_arena = crate::expr_arena_to_wire(&arena).ok();
    let commitments_match = program.commitments.resolved_estimator.as_deref()
        == Some("linear.adjustment.ate")
        && program.commitments.inference == "frequentist"
        && !program.commitments.prior_required
        && program.commitments.se_kind.as_deref() == Some(lowering.se_kind.as_str())
        && program.commitments.interval_method == lowering.interval_method
        && lowering.interval_method
            == if lowering.bootstrap_replicates > 0 { "bootstrap_se" } else { "analytic_se" }
        && binding.inference == "frequentist"
        && binding.bayesian.is_none()
        && binding.bootstrap_replicates == lowering.bootstrap_replicates;
    let binding_matches = match binding.estimator_spec.as_ref() {
        Some(crate::EstimatorSpecWire::Default(name)) => name == "linear.adjustment.ate",
        Some(crate::EstimatorSpecWire::LinearAdjustmentAte(config)) => {
            config.bootstrap_replicates == lowering.bootstrap_replicates
                && config.se_kind.as_deref().unwrap_or("homoskedastic") == lowering.se_kind
                && config.backend == lowering.backend
                && config.fit_kind.as_ref().is_none_or(|(family, parameter)| {
                    let expected = match lowering.fit_kind.split_once(':') {
                        Some((kind, bits)) => (kind, u64::from_str_radix(bits, 16).ok()),
                        None => (lowering.fit_kind.as_str(), None),
                    };
                    family == expected.0 && *parameter == expected.1
                })
        }
        None => true,
        _ => false,
    };
    target_matches
        && estimand.adjustment_set == lowering.adjustment
        && expected_source.raw() == lowering.functional
        && expected_executable.raw() == lowering.executable
        && expected_arena.as_ref() == Some(&lowering.arena)
        && commitments_match
        && binding_matches
}

pub(in crate::contract_section) fn valid_linear_fit_kind(fit_kind: &str) -> bool {
    match fit_kind {
        "ols" => true,
        _ => fit_kind.split_once(':').is_some_and(|(kind, bits)| {
            let Ok(bits) = u64::from_str_radix(bits, 16) else { return false };
            let value = f64::from_bits(bits);
            value.is_finite() && value > 0.0 && matches!(kind, "ridge" | "lasso" | "huber")
        }),
    }
}

pub(in crate::contract_section) fn verify_linear_fit_moments(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
    lowering: &crate::CheckedLinearAdjustmentLoweringWire,
) -> Result<(), ContractReason> {
    if lowering.fit_kind != "ols"
        || lowering.se_kind != "homoskedastic"
        || lowering.bootstrap_replicates != 0
        || lowering.interval_method != "analytic_se"
        || !matches!(&lowering.population, crate::TargetPopulationWire::AllObserved)
    {
        return Err(ContractReason::DependenciesLinearFitSufficientStatistics);
    }
    let snapshot = contract
        .data_snapshot
        .as_ref()
        .ok_or(ContractReason::DependenciesLinearFitSufficientStatistics)?;
    let moments = snapshot
        .linear_fit_moments
        .as_ref()
        .ok_or(ContractReason::DependenciesLinearFitSufficientStatistics)?;
    let p = lowering.design_columns.len();
    if moments.format != 1
        || usize::try_from(moments.columns).ok() != Some(p)
        || moments.complete_case_rows != lowering.complete_case_rows
        || moments.complete_case_rows > snapshot.row_count
        || !(2..=64).contains(&p)
        || moments.complete_case_rows <= p as u64
        || moments.gram.len() != p * p
        || moments.outcome_cross.len() != p
        || !moments.outcome_square.is_finite()
        || moments.gram.iter().chain(&moments.outcome_cross).any(|v| !v.is_finite())
    {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let scale = moments.gram.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0
        || !scale.is_finite()
        || (0..p).any(|row| {
            (0..p).any(|column| {
                (moments.gram[row * p + column] - moments.gram[column * p + row]).abs()
                    > 1e-10 * scale.max(1.0)
            })
        })
    {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let inverse = invert_linear_moments(&moments.gram, p)
        .ok_or(ContractReason::DataSnapshotLinearFitMomentsRank)?;
    let beta = apply_moment_inverse(&inverse, &moments.outcome_cross, p)
        .ok_or(ContractReason::DataSnapshotLinearFitMomentsRank)?;
    let residual_square = moments.outcome_square
        - beta.iter().zip(&moments.outcome_cross).map(|(b, xy)| b * xy).sum::<f64>();
    let tolerance = 1e-9 * moments.outcome_square.abs().max(1.0);
    if !residual_square.is_finite() || residual_square < -tolerance {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let dof = (moments.complete_case_rows - p as u64) as f64;
    let treatment_column = lowering
        .design_columns
        .iter()
        .position(|column| column == "treatment")
        .ok_or(ContractReason::ProgramCheckedLinearAdjustmentBinding)?;
    if lowering.design_columns.iter().filter(|column| *column == "treatment").count() != 1
        || lowering.design_columns.first().map(String::as_str) != Some("intercept")
    {
        return Err(ContractReason::ProgramCheckedLinearAdjustmentBinding);
    }
    // The residual sum of squares is a difference of two sums of products of the data, so
    // an exact fit leaves only roundoff of the order of eps * outcome_square. That is a zero
    // residual, not a standard error of its square root.
    let residual_square = if residual_square.abs() <= 1e-12 * moments.outcome_square.abs() {
        0.0
    } else {
        residual_square
    };
    let variance =
        (residual_square.max(0.0) / dof) * inverse[treatment_column * p + treatment_column];
    if !variance.is_finite() || variance < 0.0 {
        return Err(ContractReason::DataSnapshotLinearFitMoments);
    }
    let delta = f64::from_bits(lowering.active_bits) - f64::from_bits(lowering.control_bits);
    let expected = beta[treatment_column] * delta;
    let expected_se = delta.abs() * variance.sqrt();
    let actual = body.estimate.ok_or(ContractReason::BodyEstimate)?;
    if !close_replay_value(actual, expected) {
        return Err(ContractReason::BodyEstimate);
    }
    let actual_se = body.standard_error.ok_or(ContractReason::BodyStandardError)?;
    if !close_replay_value(actual_se, expected_se) {
        return Err(ContractReason::BodyStandardError);
    }
    Ok(())
}

/// Inverse of a row-major `p x p` moment matrix, or `None` when it is singular, non-finite or
/// overflows. The replay deliberately stays off the estimator's fit path and shares only the
/// scale-free equilibrated Gauss-Jordan inverse of `antecedent-stats`, so a unit change in a
/// design column moves neither the refusal nor the verdict.
pub(in crate::contract_section) fn invert_linear_moments(
    gram: &[f64],
    p: usize,
) -> Option<Vec<f64>> {
    let inverse = antecedent_stats::invert_square(gram, p)?;
    inverse.iter().all(|v| v.is_finite()).then_some(inverse)
}

/// `inverse * cross` for a row-major `p x p` inverse; `None` when the product is not finite.
pub(in crate::contract_section) fn apply_moment_inverse(
    inverse: &[f64],
    cross: &[f64],
    p: usize,
) -> Option<Vec<f64>> {
    let beta: Vec<f64> = (0..p)
        .map(|row| (0..p).map(|column| inverse[row * p + column] * cross[column]).sum())
        .collect();
    beta.iter().all(|v| v.is_finite()).then_some(beta)
}

pub(in crate::contract_section) fn close_replay_value(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 1e-8 * expected.abs().max(1.0)
}

pub(in crate::contract_section) fn verify_checked_iv(
    contract: &AnalysisResultContractWire,
    lowering: &crate::CheckedIvLoweringWire,
) -> bool {
    let (Some(product), Some(program), Some(snapshot)) = (
        contract.identification_product.as_ref(),
        contract.program.as_ref(),
        contract.data_snapshot.as_ref(),
    ) else {
        return false;
    };
    let Some(_estimand) = product.estimands.iter().find(|estimand| {
        estimand.functional == lowering.functional
            && estimand.method == "iv"
            && estimand.instruments == [lowering.instrument]
            && estimand.adjustment_set == lowering.adjustment
    }) else {
        return false;
    };
    let (query_treatment, query_outcome, active, control, mean_outcome) =
        match &contract.target.query {
            crate::CausalQueryWire::AverageEffect {
                treatment,
                outcome,
                active,
                control,
                outcome_functional,
                ..
            } => (
                *treatment,
                *outcome,
                active,
                control,
                matches!(outcome_functional, crate::query_wire::OutcomeFunctionalWire::Mean),
            ),
            _ => return false,
        };
    // An undeclared estimator defers to the program's resolved choice; the
    // commitment check below still requires the two to agree.
    let expected_procedure =
        match contract.estimator.as_deref().or(program.commitments.resolved_estimator.as_deref()) {
            Some("iv.wald") => "wald",
            Some("iv.2sls") => "two_stage_least_squares",
            _ => return false,
        };
    let expected_weak_uncertainty = match lowering.se_kind.as_str() {
        "homoskedastic" => "anderson_rubin_if_weak",
        "hc0" | "hc1" | "hc2" | "hc3" | "cluster" | "multiway" | "newey_west"
        | "panel_cluster_hac" => "withheld_non_homoskedastic",
        _ => return false,
    };
    let Ok(mut arena) = crate::expr_arena_from_wire(&product.arena) else {
        return false;
    };
    let Ok(expected_functional) = arena.iv_wald(
        antecedent_core::VariableId::from_raw(lowering.treatment),
        antecedent_core::VariableId::from_raw(lowering.outcome),
        &[antecedent_core::VariableId::from_raw(lowering.instrument)],
        &antecedent_core::Value::Float64(f64::from_bits(lowering.active_bits)),
        &antecedent_core::Value::Float64(f64::from_bits(lowering.control_bits)),
    ) else {
        return false;
    };
    let expected_arena = crate::expr_arena_to_wire(&arena).ok();
    lowering.format == 1
        && lowering.procedure == expected_procedure
        && lowering.treatment == query_treatment
        && lowering.outcome == query_outcome
        && intervention_bits(active, query_treatment) == Some(lowering.active_bits)
        && intervention_bits(control, query_treatment) == Some(lowering.control_bits)
        && mean_outcome
        && lowering.instrument_active_bits == 1.0_f64.to_bits()
        && lowering.instrument_control_bits == 0.0_f64.to_bits()
        && lowering.complete_case_rows > 0
        && lowering.complete_case_rows <= snapshot.row_count
        && program.commitments.resolved_estimator.as_deref()
            == Some(expected_procedure_id(expected_procedure))
        && program.commitments.se_kind.as_deref() == Some(lowering.se_kind.as_str())
        && lowering.weak_instrument_uncertainty == expected_weak_uncertainty
        && expected_functional.raw() == lowering.functional
        && expected_arena.as_ref() == Some(&product.arena)
}

pub(in crate::contract_section) fn expected_procedure_id(procedure: &str) -> &'static str {
    match procedure {
        "wald" => "iv.wald",
        _ => "iv.2sls",
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "The lowering verifier checks the complete front-door product and its source bindings."
)]
pub(in crate::contract_section) fn verify_frontdoor_lowering(
    contract: &AnalysisResultContractWire,
    lowering: &crate::CheckedFrontDoorLoweringWire,
) -> bool {
    let Some(product) = contract.identification_product.as_ref() else { return false };
    let Some(program) = contract.program.as_ref() else { return false };
    let Some(estimand) = product.estimands.iter().find(|estimand| {
        estimand.functional == lowering.functional
            && estimand.mediators == lowering.mediators
            && estimand.method == "frontdoor"
    }) else {
        return false;
    };
    let target_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            active,
            control,
            outcome_functional,
            ..
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && intervention_bits(active, *treatment) == Some(lowering.active_bits)
                && intervention_bits(control, *treatment) == Some(lowering.control_bits)
                && matches!(outcome_functional, crate::query_wire::OutcomeFunctionalWire::Mean)
        }
        _ => false,
    };
    let expected_variables: Vec<_> = contract
        .target
        .schema
        .variables
        .iter()
        .map(|variable| (variable.id, variable.name.clone()))
        .collect();
    let program_valid = crate::functional_program_from_wire(
        &crate::FunctionalProgramWire {
            arena: lowering.arena.clone(),
            source: lowering.functional,
            executable: lowering.executable,
            variables: expected_variables,
        },
        antecedent_expr::ProgramLimits::default(),
    )
    .is_ok();
    let lowering_matches_frontdoor = (|| {
        let mut arena = crate::expr_arena_from_wire(&product.arena).ok()?;
        let treatment = antecedent_core::VariableId::from_raw(lowering.treatment);
        let outcome = antecedent_core::VariableId::from_raw(lowering.outcome);
        let mediators = lowering
            .mediators
            .iter()
            .copied()
            .map(antecedent_core::VariableId::from_raw)
            .collect::<Vec<_>>();
        let source = arena.frontdoor_ate(
            treatment,
            outcome,
            &mediators,
            antecedent_core::Value::Float64(f64::from_bits(lowering.active_bits)),
            antecedent_core::Value::Float64(f64::from_bits(lowering.control_bits)),
        );
        let executable = frontdoor_observational_root(&mut arena, treatment, outcome, &mediators);
        let expected_arena = crate::expr_arena_to_wire(&arena).ok()?;
        Some(
            source.raw() == lowering.functional
                && executable.raw() == lowering.executable
                && expected_arena == lowering.arena,
        )
    })()
    .unwrap_or(false);
    let uncertainty_matches = lowering.uncertainty
        == format!(
            "{}:{}",
            program.commitments.interval_method,
            program.commitments.se_kind.as_deref().unwrap_or("unspecified")
        );
    let overlap_matches = contract.inference_binding.as_ref().is_some_and(|binding| {
        if let Some(overlap) = binding.overlap_policy.as_deref() {
            return overlap == lowering.overlap;
        }
        match binding.estimator_spec.as_ref() {
            Some(crate::EstimatorSpecWire::FrontDoorTwoStage(config)) => {
                overlap_policy_tag(&config.overlap) == lowering.overlap
            }
            Some(crate::EstimatorSpecWire::Default(id)) if id == "frontdoor.linear_two_stage" => {
                lowering.overlap == "explicit_override"
            }
            _ => false,
        }
    });
    target_matches
        && program_valid
        && lowering_matches_frontdoor
        && lowering.format == 1
        && lowering.procedure == "linear_path_product"
        && matches!(contract.estimator.as_deref(), None | Some("frontdoor.linear_two_stage"))
        && program.commitments.resolved_estimator.as_deref() == Some("frontdoor.linear_two_stage")
        && uncertainty_matches
        && overlap_matches
        && lowering.complete_case_rows > 0
        && !lowering.mediators.is_empty()
        && lowering.treatment != lowering.outcome
        && lowering.mediators.iter().all(|id| *id != lowering.treatment && *id != lowering.outcome)
        && estimand.method == "frontdoor"
}

pub(in crate::contract_section) fn frontdoor_observational_root(
    arena: &mut antecedent_expr::CausalExprArena,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    mediators: &[antecedent_core::VariableId],
) -> antecedent_expr::ExprId {
    use antecedent_expr::{DomainRef, ExprNode, OutcomeExprId};

    let m = arena.intern_var_set(mediators.iter().copied());
    let y = arena.intern_var_set([outcome]);
    let t = arena.intern_var_set([treatment]);
    let m_and_t = arena.intern_var_set(mediators.iter().copied().chain([treatment]));
    let empty = arena.empty_var_set();
    let no_bindings = arena.empty_intervention_set();
    let m_given_t = arena.intern_distribution(m, t, no_bindings, DomainRef::Observational);
    let y_given_m_t = arena.intern_distribution(y, m_and_t, no_bindings, DomainRef::Observational);
    let t_marginal = arena.intern_distribution(t, empty, no_bindings, DomainRef::Observational);
    let inner_factors = arena.intern_list([y_given_m_t, t_marginal]);
    let inner_product = arena.intern(ExprNode::Product(inner_factors));
    let inner_sum = arena.intern(ExprNode::SumOut { variables: t, expr: inner_product });
    let outer_factors = arena.intern_list([m_given_t, inner_sum]);
    let outer_product = arena.intern(ExprNode::Product(outer_factors));
    let outer_sum = arena.intern(ExprNode::SumOut { variables: m, expr: outer_product });
    arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: outer_sum,
    })
}

pub(in crate::contract_section) fn intervention_bits(
    wire: &crate::query_wire::InterventionWire,
    variable: u32,
) -> Option<u64> {
    match wire {
        crate::query_wire::InterventionWire::Set { variable: target, value }
            if *target == variable =>
        {
            match value {
                crate::query_wire::ValueWire::Float64(value) => Some(value.to_bits()),
                crate::query_wire::ValueWire::Int64(value) => Some((*value as f64).to_bits()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether a study-level overlap policy tag (as `inference_binding.overlap_policy`
/// spells it) trims exactly as the checked AIPW lowering does; no tag means the
/// estimator's default, untrimmed policy ran.
pub(in crate::contract_section) fn overlap_tag_trim_matches(
    tag: Option<&str>,
    lowering: &CheckedAipwLoweringWire,
) -> bool {
    let expected = format!(
        ":trim={}",
        lowering.trim_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}"))
    );
    match tag {
        None => lowering.trim_bits.is_none(),
        Some(tag) => tag.starts_with("require_diagnostics:") && tag.ends_with(&expected),
    }
}

pub(in crate::contract_section) fn overlap_policy_tag(policy: &crate::OverlapPolicyWire) -> String {
    match policy {
        crate::OverlapPolicyWire::ExplicitOverride => "explicit_override".into(),
        crate::OverlapPolicyWire::RequireDiagnostics { clip_bits, trim_bits } => format!(
            "require_diagnostics:clip={}:trim={}",
            clip_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}")),
            trim_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}")),
        ),
    }
}

/// Cross-layer references: every layer must describe the same question,
/// observation contract, graph class, schema, and inference family.
pub(in crate::contract_section) fn verify_layer_links(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let target = &contract.target;
    if let Some(identification) = &contract.identification {
        match digest_wire(IdentityDomain::Target, &target.question()) {
            Ok(question) if question.as_bytes() == &identification.target_question => {}
            _ => unresolved.push(ContractReason::IdentificationTargetQuestion),
        }
        if identification.observation != contract.identities.observation {
            unresolved.push(ContractReason::IdentificationObservation);
        }
        if identification.graph_class != contract.graph_class {
            unresolved.push(ContractReason::IdentificationGraphClass);
        }
        if identification.structure_source != contract.structure_source {
            unresolved.push(ContractReason::IdentificationStructureSource);
        }
        if identification
            .schema_names
            .as_ref()
            .is_some_and(|names| *names != target.schema.variable_names())
        {
            unresolved.push(ContractReason::IdentificationSchemaNames);
        }
        let depends_on: &[u32] = match target.query.target_population() {
            Some(TargetPopulationWire::CustomDistribution { depends_on, .. }) => depends_on,
            _ => &[],
        };
        if identification.population_depends_on != depends_on {
            unresolved.push(ContractReason::IdentificationPopulationDependsOn);
        }
    }
    verify_score_reuse_link(contract, unresolved);
    verify_target_weights_link(contract, unresolved);
    if let Some(observation) = &contract.observation {
        if observation.schema != target.schema {
            unresolved.push(ContractReason::ObservationSchema);
        }
    }
    if let Some(program) = &contract.program {
        let commitments = &program.commitments;
        if commitments.identifier != contract.identifier {
            unresolved.push(ContractReason::ProgramCommitmentsIdentifier);
        }
        if commitments.estimator != contract.estimator {
            unresolved.push(ContractReason::ProgramCommitmentsEstimator);
        }
        if let Some(binding) = &contract.inference_binding {
            if commitments.inference != binding.inference {
                unresolved.push(ContractReason::InferenceBindingInference);
            }
            if commitments.validation_suite != binding.validation_suite {
                unresolved.push(ContractReason::InferenceBindingValidationSuite);
            }
            if binding.bayesian.is_some() != (binding.inference == "bayesian") {
                unresolved.push(ContractReason::InferenceBindingBayesian);
            }
            if commitments.prior_required != binding.bayesian.is_some() {
                unresolved.push(ContractReason::ProgramCommitmentsPriorRequired);
            }
        }
    }
}

/// Whether a checked AIPW lowering is internally well formed: a known format
/// and uncertainty method, a lag exactly for the lagged methods, and a
/// procedure the trim rule selects (cross-fitted scores run five folds without
/// a trim, the common-support refit runs no folds with one).
pub(in crate::contract_section) fn checked_aipw_lowering_well_formed(
    lowering: &CheckedAipwLoweringWire,
) -> bool {
    let valid_method = matches!(
        lowering.se_kind.as_str(),
        "homoskedastic"
            | "hc0"
            | "hc1"
            | "hc2"
            | "hc3"
            | "cluster"
            | "multiway"
            | "newey_west"
            | "panel_cluster_hac"
    );
    let lag_valid = matches!(lowering.se_kind.as_str(), "newey_west" | "panel_cluster_hac")
        == lowering.se_lag.is_some();
    let valid_procedure = match lowering.procedure.as_str() {
        "cross_fitted_logistic_ols" => lowering.folds == 5 && lowering.trim_bits.is_none(),
        "trimmed_logistic_ols" => lowering.folds == 0 && lowering.trim_bits.is_some(),
        _ => false,
    };
    lowering.format == 1 && valid_method && lag_valid && valid_procedure
}

pub(in crate::contract_section) fn verify_checked_aipw(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let lowering =
        contract.program.as_ref().and_then(|program| program.checked_aipw_lowering.as_ref());
    if is_aipw_contract(contract) && lowering.is_none() {
        unresolved.push(ContractReason::ProgramCheckedAipwLowering);
        return;
    }
    let Some(lowering) = lowering else { return };
    if !checked_aipw_lowering_well_formed(lowering) {
        unresolved.push(ContractReason::ProgramCheckedAipwLowering);
    }
    if !is_aipw_contract(contract) {
        unresolved.push(ContractReason::ProgramCheckedAipwBinding);
    }
    let commitments = contract.program.as_ref().map(|program| &program.commitments);
    if commitments.is_none_or(|commitments| {
        commitments.inference != "frequentist"
            || commitments.prior_required
            || commitments.se_kind.as_deref() != Some(lowering.se_kind.as_str())
            || commitments.interval_method
                != if lowering.bootstrap_replicates > 0 { "bootstrap_se" } else { "analytic_se" }
    }) {
        unresolved.push(ContractReason::ProgramCheckedAipwUncertainty);
    }
    if contract.inference_binding.as_ref().is_none_or(|binding| {
        let choices_match = match binding.estimator_spec.as_ref() {
            Some(crate::EstimatorSpecWire::Aipw(config)) => {
                config.bootstrap_replicates == lowering.bootstrap_replicates
                    && config.se_kind.as_deref() == Some(lowering.se_kind.as_str())
                    && matches!(
                        config.overlap,
                        crate::OverlapPolicyWire::RequireDiagnostics { trim_bits, .. }
                            if trim_bits == lowering.trim_bits
                    )
            }
            Some(crate::EstimatorSpecWire::Default(estimator)) if estimator == "aipw" => {
                lowering.se_kind == "homoskedastic"
                    && binding.bootstrap_replicates == lowering.bootstrap_replicates
                    && overlap_tag_trim_matches(binding.overlap_policy.as_deref(), lowering)
            }
            None => {
                binding.bootstrap_replicates == lowering.bootstrap_replicates
                    && overlap_tag_trim_matches(binding.overlap_policy.as_deref(), lowering)
            }
            _ => false,
        };
        binding.inference != "frequentist" || binding.bayesian.is_some() || !choices_match
    }) {
        unresolved.push(ContractReason::ProgramCheckedAipwUncertainty);
    }
    let query_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            control,
            active,
            target_population: crate::TargetPopulationWire::AllObserved,
            outcome_functional: crate::query_wire::OutcomeFunctionalWire::Mean,
            effect_modifiers,
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && effect_modifiers.is_empty()
                && is_set_value(control, lowering.treatment, 0.0)
                && is_set_value(active, lowering.treatment, 1.0)
                && lowering.population == "all_observed"
        }
        _ => false,
    };
    let estimand_matches = contract.identification_product.as_ref().is_some_and(|product| {
        product.arena.nodes.get(lowering.functional as usize).is_some()
            && product.estimands.iter().any(|estimand| {
                estimand.functional == lowering.functional
                    && estimand.adjustment_set == lowering.adjustment
                    && estimand.instruments.is_empty()
                    && estimand.mediators.is_empty()
            })
    });
    let adjustments_unique =
        lowering.adjustment.iter().copied().collect::<std::collections::BTreeSet<_>>().len()
            == lowering.adjustment.len()
            && !lowering.adjustment.contains(&lowering.treatment)
            && !lowering.adjustment.contains(&lowering.outcome);
    if !query_matches
        || !estimand_matches
        || !adjustments_unique
        || !matches!(
            lowering.procedure.as_str(),
            "cross_fitted_logistic_ols" | "trimmed_logistic_ols"
        )
    {
        unresolved.push(ContractReason::ProgramCheckedAipwBinding);
    }
}

pub(in crate::contract_section) fn is_set_value(
    intervention: &crate::InterventionWire,
    variable: u32,
    value: f64,
) -> bool {
    matches!(intervention,
        crate::InterventionWire::Set { variable: actual, value: crate::ValueWire::Float64(actual_value) }
            if *actual == variable && actual_value.to_bits() == value.to_bits())
}

pub(in crate::contract_section) fn verify_checked_aipw_rows(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let row_binding = contract.checked_aipw_rows.as_ref();
    if row_binding.is_some() != contract.identities.checked_aipw_rows.is_some() {
        unresolved.push(ContractReason::CheckedAipwRowBinding);
        return;
    }
    let Some(row_binding) = row_binding else {
        if is_aipw_contract(contract)
            || contract
                .program
                .as_ref()
                .is_some_and(|program| program.checked_aipw_lowering.is_some())
        {
            unresolved.push(ContractReason::CheckedAipwRowBinding);
        }
        return;
    };
    let digest_matches = checked_aipw_rows_digest(row_binding)
        .is_ok_and(|digest| Some(digest) == contract.identities.checked_aipw_rows);
    let snapshot_matches = row_binding.data_snapshot == contract.identities.data_snapshot
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| {
            row_binding.rows.len() as u64 <= snapshot.row_count
                && row_binding.rows.iter().all(|row| *row < snapshot.row_count)
        });
    let ordered_unique =
        !row_binding.rows.is_empty() && row_binding.rows.windows(2).all(|pair| pair[0] < pair[1]);
    if row_binding.format != 1
        || !digest_matches
        || !snapshot_matches
        || !ordered_unique
        || !is_aipw_contract(contract)
    {
        unresolved.push(ContractReason::CheckedAipwRowBinding);
    }
}

/// Whether the artifact executed AIPW, whether the caller declared the
/// estimator or the planner resolved it (a tiered closure resolves to AIPW
/// without a declaration).
pub(in crate::contract_section) fn is_aipw_contract(contract: &AnalysisResultContractWire) -> bool {
    contract_uses_estimator(contract, "aipw")
}

/// Score-reuse layer: present with its digest, fitted on this snapshot, and
/// keyed by this identification.
pub(in crate::contract_section) fn verify_score_reuse_link(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if contract.score_reuse.is_some() != contract.identities.score_reuse.is_some() {
        unresolved.push(ContractReason::IdentitiesScoreReuse);
        return;
    }
    let Some(score) = &contract.score_reuse else {
        return;
    };
    require_payload_digest(
        unresolved,
        ContractReason::IdentitiesScoreReuse,
        Some(score),
        contract.identities.score_reuse.as_ref(),
        score_reuse_digest,
    );
    require_nested_digest(
        unresolved,
        ContractReason::ScoreReuseDataSnapshot,
        Some(score.data_snapshot),
        Some(&contract.identities.data_snapshot),
    );
    require_nested_digest(
        unresolved,
        ContractReason::ScoreReuseIdentification,
        score.identification,
        Some(&contract.identities.identification),
    );
}

/// Row-weight layer: re-derive the target-weights identity from the weights the
/// section carries, and check that the target population, the snapshot and the
/// score table all name it.
pub(in crate::contract_section) fn verify_target_weights_link(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let row_weights = match contract.target.query.target_population() {
        Some(TargetPopulationWire::RowWeights { weights, depends_on }) => {
            Some((*weights, depends_on.clone()))
        }
        _ => None,
    };
    if contract.target_weights.is_some() != contract.identities.target_weights.is_some() {
        unresolved.push(ContractReason::IdentitiesTargetWeights);
        return;
    }
    let Some(section) = &contract.target_weights else {
        if row_weights.is_some() {
            unresolved.push(ContractReason::TargetPopulation);
        }
        return;
    };
    let identity = &section.identity;
    require_payload_digest(
        unresolved,
        ContractReason::IdentitiesTargetWeights,
        Some(identity),
        contract.identities.target_weights.as_ref(),
        target_weights_digest,
    );
    let rows = PayloadDigestWire::f64s("target_weights.rows", &section.values);
    let values_ok = rows.len == identity.row_count
        && rows.digest == identity.weights
        && section.values.iter().all(|weight| weight.is_finite() && *weight >= 0.0);
    if !values_ok {
        unresolved.push(ContractReason::TargetWeightsValues);
    }
    require_nested_digest(
        unresolved,
        ContractReason::TargetWeightsDataSnapshot,
        Some(identity.data_snapshot),
        Some(&contract.identities.data_snapshot),
    );
    require_nested_digest(
        unresolved,
        ContractReason::TargetWeightsScoreReuse,
        Some(identity.score_reuse),
        contract.identities.score_reuse.as_ref(),
    );
    if contract.score_reuse.as_ref().is_some_and(|score| score.row_index.len != identity.row_count)
    {
        unresolved.push(ContractReason::TargetWeightsRowCount);
    }
    match row_weights {
        Some((weights, depends_on))
            if Some(weights) == contract.identities.target_weights
                && depends_on == identity.depends_on => {}
        _ => unresolved.push(ContractReason::TargetPopulation),
    }
}
