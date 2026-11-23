//! Replay and validate checked AIPW contract bindings.

use crate::contract_reason::ContractReason;
use crate::contract_wire::{CheckedAipwLoweringWire, checked_aipw_rows_digest};

use super::{
    AnalysisResultContractWire, carried_adjustment_is_backdoor, contract_uses_estimator,
    overlap_tag_trim_matches,
};

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
            && !lowering.adjustment.contains(&lowering.outcome)
            && carried_adjustment_is_backdoor(
                contract,
                lowering.treatment,
                lowering.outcome,
                &lowering.adjustment,
            );
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
