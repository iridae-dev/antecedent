//! Validate retained score, fit and target-weight identity links.

use crate::CausalQueryWire;
use crate::contract_reason::ContractReason;
use crate::identity::{
    PayloadDigestWire, adjusted_fit_reuse_digest, score_reuse_digest, target_weights_digest,
};
use crate::query_wire::TargetPopulationWire;

use super::{
    AnalysisResultContractWire, contract_uses_estimator, require_nested_digest,
    require_payload_digest,
};

/// Score-reuse layer: present with its digest, fitted on this snapshot, and
/// keyed by this identification.
pub(in crate::contract_section) fn verify_score_reuse_link(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if contract.score_reuse.is_some() && contract.adjusted_fit_reuse.is_some() {
        unresolved.push(ContractReason::IdentitiesScoreReuse);
        return;
    }
    if contract.adjusted_fit_reuse.is_some() {
        verify_adjusted_fit_reuse_link(contract, unresolved);
        return;
    }
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

fn verify_adjusted_fit_reuse_link(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let Some(fit) = &contract.adjusted_fit_reuse else {
        return;
    };
    let estimator = match fit.model.as_str() {
        "additive_linear_mean_v1" => "linear.adjustment.ate",
        "glm_response_mean_v1" => "glm.adjustment",
        _ => {
            unresolved.push(ContractReason::IdentitiesScoreReuse);
            return;
        }
    };
    let rows = fit.row_index.len;
    let roles_ok = fit.treatment != fit.outcome
        && fit.adjustment_set.iter().enumerate().all(|(i, v)| {
            *v != fit.treatment && *v != fit.outcome && !fit.adjustment_set[..i].contains(v)
        });
    let target_ok = matches!(&contract.target.query, CausalQueryWire::AverageEffect {
        treatment,outcome,outcome_functional:crate::query_wire::OutcomeFunctionalWire::Mean,effect_modifiers,.. }
        if *treatment==fit.treatment && *outcome==fit.outcome && effect_modifiers.is_empty());
    let covariance_ok = matches!(
        fit.covariance.as_str(),
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
    if fit.format != 1
        || !contract_uses_estimator(contract, estimator)
        || !roles_ok
        || !target_ok
        || !covariance_ok
        || rows == 0
        || fit.n_columns < 2
        || rows.checked_mul(fit.n_columns) != Some(fit.design.len)
        || fit.inference_binding != contract.identities.inference_binding
        || contract.inference_binding.as_ref().is_none_or(|b| b.inference != "frequentist")
    {
        unresolved.push(ContractReason::IdentitiesScoreReuse);
    }
    require_payload_digest(
        unresolved,
        ContractReason::IdentitiesScoreReuse,
        Some(fit),
        contract.identities.score_reuse.as_ref(),
        adjusted_fit_reuse_digest,
    );
    require_nested_digest(
        unresolved,
        ContractReason::ScoreReuseDataSnapshot,
        Some(fit.data_snapshot),
        Some(&contract.identities.data_snapshot),
    );
    require_nested_digest(
        unresolved,
        ContractReason::ScoreReuseIdentification,
        Some(fit.identification),
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
    if contract
        .adjusted_fit_reuse
        .as_ref()
        .is_some_and(|fit| fit.row_index.len != identity.row_count)
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
