//! Verify instrumental-variable and frontdoor lowering bindings.

use super::{AnalysisResultContractWire, overlap_policy_tag};

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
        | "panel_cluster_hac" => "robust_se_without_weak_instrument_interval",
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
