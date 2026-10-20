//! Contract verification against an analysis-result body.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::consume::{
    claim_domains, claim_id_matches, claim_kind_name, claim_value_matches_body, contract_seal_of,
    identification_product_matches_body, require_nested_digest, require_payload_digest,
    require_present, require_slot, support_empirical, validate_mixture_masses, weight_basis_name,
};
use super::replay::{
    contract_uses_estimator, is_aipw_contract, is_functional_distribution_contract,
    is_response_functional_effect_contract, is_scalar_functional_effect_contract,
    replay_functional_response_grid, verify_checked_aipw, verify_checked_aipw_rows,
    verify_checked_iv, verify_checked_linear_adjustment, verify_frontdoor_lowering,
    verify_layer_links, verify_linear_fit_moments, verify_nested_counterfactual_result,
    verify_temporal_nested_point,
};

use std::sync::Arc;

use antecedent_core::IdentityDomain;

use super::{AnalysisResultContractWire, ClaimSectionWire};
use crate::contract_reason::{ContractReason, publish_reasons};
use crate::identity::{
    data_snapshot_digest, digest_wire, execution_digest, identification_digest,
    identification_product_digest_wire, inference_binding_digest, program_digest,
};
use crate::{AnalysisResultHeader, AnalysisResultWire, CausalQueryWire};

/// Shared producer/consumer check: stored payloads, body query, and claim.
///
/// Missing rehashable payloads are unresolved — attested digests are not enough.
#[must_use]
pub fn verify_contract_against_body(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
) -> Vec<Arc<str>> {
    publish_reasons(contract_unresolved(header, body, contract))
}

pub(in crate::contract_section) fn contract_unresolved(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
) -> Vec<ContractReason> {
    let mut unresolved = verify_stored_payloads(contract);
    note_linear_fit_replay(contract, body, &mut unresolved);
    note_checked_program_targets(contract, &mut unresolved);
    if contract.target.query != body.query {
        unresolved.push(ContractReason::BodyQuery);
    }
    note_replay_dependencies(contract, body, &mut unresolved);
    note_functional_and_nested_replay(contract, body, &mut unresolved);
    note_published_claim(header, body, contract, &mut unresolved);
    unresolved
}

pub(in crate::contract_section) fn note_linear_fit_replay(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if let Some(lowering) = contract
        .program
        .as_ref()
        .and_then(|program| program.checked_linear_adjustment_lowering.as_ref())
    {
        unresolved
            .retain(|reason| *reason != ContractReason::DependenciesLinearFitSufficientStatistics);
        if let Err(reason) = verify_linear_fit_moments(contract, body, lowering) {
            unresolved.push(reason);
        }
    }
}

pub(in crate::contract_section) fn note_checked_program_targets(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if contract.program.as_ref().is_some_and(|program| {
        [
            program.functional_program.is_some(),
            program.checked_aipw_lowering.is_some(),
            program.checked_frontdoor_lowering.is_some(),
            program.checked_iv_lowering.is_some(),
            program.checked_linear_adjustment_lowering.is_some(),
            program.checked_functional_response_grid.is_some(),
            program.checked_nested_counterfactual.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count()
            > 1
    }) {
        unresolved.push(ContractReason::ProgramCheckedPayloadConflict);
    }
    if let Some(program) = contract.program.as_ref() {
        let estimator_matches = |expected: &[&str]| {
            contract
                .estimator
                .as_deref()
                .into_iter()
                .chain(program.commitments.resolved_estimator.as_deref())
                .all(|name| expected.contains(&name))
                && (contract.estimator.is_some()
                    || program.commitments.resolved_estimator.is_some())
        };
        let average_effect = matches!(contract.target.query, CausalQueryWire::AverageEffect { .. });
        if program.functional_program.is_some()
            && !((estimator_matches(&["functional.effect"])
                && matches!(
                    contract.target.query,
                    CausalQueryWire::AverageEffect { .. }
                        | CausalQueryWire::ConditionalEffect { .. }
                        | CausalQueryWire::PathSpecific(_)
                ))
                || (estimator_matches(&["functional.distribution"])
                    && matches!(contract.target.query, CausalQueryWire::Distribution(_))))
        {
            unresolved.push(ContractReason::ProgramFunctionalProgramTarget);
        }
        for (present, matches, reason) in [
            (
                program.checked_aipw_lowering.is_some(),
                average_effect && estimator_matches(&["aipw"]),
                ContractReason::ProgramCheckedAipwLoweringTarget,
            ),
            (
                program.checked_frontdoor_lowering.is_some(),
                average_effect && estimator_matches(&["frontdoor.linear_two_stage"]),
                ContractReason::ProgramCheckedFrontdoorLoweringTarget,
            ),
            (
                program.checked_iv_lowering.is_some(),
                average_effect && estimator_matches(&["iv.wald", "iv.2sls"]),
                ContractReason::ProgramCheckedIvLoweringTarget,
            ),
            (
                program.checked_linear_adjustment_lowering.is_some(),
                average_effect && estimator_matches(&["linear.adjustment.ate"]),
                ContractReason::ProgramCheckedLinearAdjustmentLoweringTarget,
            ),
            (
                program.checked_functional_response_grid.is_some(),
                is_response_functional_effect_contract(contract),
                ContractReason::ProgramCheckedFunctionalResponseGridTarget,
            ),
            (
                program.checked_nested_counterfactual.is_some(),
                matches!(contract.target.query, CausalQueryWire::NestedCounterfactual { .. })
                    && estimator_matches(&["mediation.linear"]),
                ContractReason::ProgramCheckedNestedCounterfactualTarget,
            ),
        ] {
            if present && !matches {
                unresolved.push(reason);
            }
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "Replay-dependency gaps stay one ordered match so a route cannot be named twice."
)]
pub(in crate::contract_section) fn note_replay_dependencies(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
    unresolved: &mut Vec<ContractReason>,
) {
    // These prepared routes execute from retained model or estimator operations,
    // but their current result wires do not carry a portable operation and its
    // fitted data dependencies. Keep the artifact readable and name the missing
    // dependency instead of silently treating a checked in-process run as
    // independently replayable.
    let resolved_estimator = contract
        .program
        .as_ref()
        .and_then(|program| program.commitments.resolved_estimator.as_deref())
        .or(contract.estimator.as_deref());
    match resolved_estimator {
        Some("bayesian.basis.gcomp")
            if matches!(contract.target.query, CausalQueryWire::AverageEffect { .. }) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianBasisAteOperation);
        }
        Some("bayesian.basis.gcomp")
            if matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. }) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianBasisCateOperation);
        }
        Some("bayesian.robust_ate") => {
            unresolved.push(ContractReason::DependenciesCheckedBayesianRobustAteOperation);
        }
        Some("glm.adjustment") => {
            unresolved.push(ContractReason::DependenciesCheckedGlmOperation);
        }
        Some("rd.sharp") => {
            unresolved.push(ContractReason::DependenciesCheckedRdOperation);
        }
        Some("propensity.weighting" | "propensity.matching") => {
            unresolved.push(ContractReason::DependenciesCheckedPropensityOperation);
        }
        Some("response.kennedy_dr" | "response.riesz_ade" | "response.gam_derivative")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if matches!(query.functional,
                    crate::ResponseFunctionalWire::PointDerivative { .. }
                    | crate::ResponseFunctionalWire::AverageDerivative { .. }
                    | crate::ResponseFunctionalWire::DirectionalDerivative { .. }
                    | crate::ResponseFunctionalWire::Jacobian { .. })) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedDerivativeResponseOperation);
        }
        Some("response.kennedy_dr")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if matches!(query.functional, crate::ResponseFunctionalWire::MeanCurve { .. })) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedResponseGridOperation);
        }
        Some("response.intervention_gcomp" | "cell.aipw")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if matches!(query.functional, crate::ResponseFunctionalWire::InterventionResponse { .. })) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedInterventionResponseOperation);
        }
        // A static Bayesian response publishes a posterior over the retained
        // adjustment target; the sealed static response operation is its replay
        // dependency exactly as for the frequentist estimators above.
        Some("response.bayesian")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if query.temporal.is_none()
                    && matches!(query.functional, crate::ResponseFunctionalWire::MeanCurve { .. })) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedResponseGridOperation);
        }
        Some("response.bayesian")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if query.temporal.is_none()
                    && matches!(query.functional, crate::ResponseFunctionalWire::InterventionResponse { .. })) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedInterventionResponseOperation);
        }
        Some(
            "interference.ht_hajek"
            | "interference.cluster_neyman"
            | "interference.bayesian_gaussian",
        ) if matches!(contract.target.query, CausalQueryWire::Interference { .. }) => {
            unresolved.push(ContractReason::DependenciesCheckedInterferenceOperation);
        }
        Some("iv.bayesian_joint_linear")
            if matches!(contract.target.query, CausalQueryWire::AverageEffect { .. }) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianIvOperation);
        }
        Some("rd.bayesian_local_linear")
            if matches!(contract.target.query, CausalQueryWire::AverageEffect { .. }) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianRdOperation);
        }
        Some("bayesian.panel_did")
            if matches!(contract.target.query, CausalQueryWire::PanelDid(_)) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianPanelDidOperation);
        }
        Some("bayesian.synthetic_control")
            if matches!(contract.target.query, CausalQueryWire::SyntheticControl(_)) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianSyntheticControlOperation);
        }
        Some("bayesian.randomized_quantile")
            if matches!(contract.target.query, CausalQueryWire::AverageEffect { .. }) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianQuantileOperation);
        }
        Some("bayesian.piecewise_survival")
            if matches!(contract.target.query, CausalQueryWire::Survival(_)) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianSurvivalOperation);
        }
        Some("bayesian.continuous_dose_policy")
            if matches!(contract.target.query, CausalQueryWire::ContinuousDoseResponse(_)) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianDosePolicyOperation);
        }
        Some("bayesian.local_ratio")
            if matches!(contract.target.query, CausalQueryWire::LocalPolynomialRatio(_)) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianLocalRatioOperation);
        }
        Some(
            "bayesian.randomized_finite_population"
            | "bayesian.randomized_superpopulation"
            | "bayesian.switchback_effect",
        )
            if matches!(contract.target.query, CausalQueryWire::RandomizedEffect(_)) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedBayesianRandomizedOperation);
        }
        Some("transport.trial_ipw" | "transport.trial_bayesian_bootstrap")
            if matches!(contract.target.query, CausalQueryWire::Transport(_)) =>
        {
            unresolved.push(ContractReason::DependenciesCheckedTransportTrialOperation);
        }
        Some("gcm.fit" | "gcm.fit.bayesian")
            if matches!(contract.target.query, CausalQueryWire::Counterfactual { .. }) =>
        {
            unresolved.push(ContractReason::DependenciesFittedCounterfactualMechanisms);
        }
        _ => {}
    }
    if resolved_estimator == Some("linear.adjustment.ate")
        && body.identification.status == "graph_dependent"
        && body.structural_response.is_some()
    {
        unresolved.push(ContractReason::DependenciesCheckedUnknownTieredAverageOperation);
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("linear.adjustment.ate")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(if matches!(contract.graph_class.as_str(), "Cpdag" | "Pag") {
            ContractReason::DependenciesCheckedClassGraphPosteriorEffectOperation
        } else {
            ContractReason::DependenciesCheckedGraphPosteriorEffectOperation
        });
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Dag" | "Cpdag" | "Pag")
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("bayesian.gcomp")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(if contract.graph_class == "Dag" {
            ContractReason::DependenciesCheckedBayesianGraphPosteriorAteOperation
        } else {
            ContractReason::DependenciesCheckedClassGraphPosteriorEffectOperation
        });
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Dag" | "Cpdag" | "Pag")
        && matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. })
        && resolved_estimator == Some("conditional.linear.adjustment")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(if contract.graph_class == "Dag" {
            ContractReason::DependenciesCheckedGraphPosteriorEffectOperation
        } else {
            ContractReason::DependenciesCheckedClassGraphPosteriorEffectOperation
        });
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Dag" | "Cpdag" | "Pag")
        && matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. })
        && resolved_estimator == Some("conditional.bayesian")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(if contract.graph_class == "Dag" {
            ContractReason::DependenciesCheckedBayesianGraphPosteriorAteOperation
        } else {
            ContractReason::DependenciesCheckedClassGraphPosteriorEffectOperation
        });
    }
    if matches!(contract.graph_class.as_str(), "Cpdag" | "Pag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && contract.program.as_ref().is_some_and(|program| {
            (resolved_estimator == Some("linear.adjustment.ate")
                && program.commitments.inference.eq_ignore_ascii_case("frequentist"))
                || (resolved_estimator == Some("bayesian.gcomp")
                    && program.commitments.inference.eq_ignore_ascii_case("bayesian"))
        })
    {
        unresolved.push(ContractReason::DependenciesCheckedStaticClassEffectOperation);
    }
    if matches!(contract.graph_class.as_str(), "Cpdag" | "Pag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. })
        && resolved_estimator == Some("conditional.bayesian")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedBayesianClassConditionalOperation);
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Admg" | "TemporalAdmg")
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("functional.effect")
    {
        unresolved.push(ContractReason::DependenciesCheckedGraphPosteriorEffectOperation);
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Admg" | "TemporalAdmg")
        && matches!(contract.target.query, CausalQueryWire::Response(_))
        && resolved_estimator == Some("functional.effect")
    {
        unresolved.push(ContractReason::DependenciesCheckedAdmgGraphPosteriorResponseOperation);
    }
    let static_response_estimator = matches!(
        resolved_estimator,
        Some("response.intervention_gcomp" | "response.kennedy_dr" | "response.bayesian")
    );
    if matches!(contract.graph_class.as_str(), "Cpdag" | "Pag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(&contract.target.query, CausalQueryWire::Response(query)
        if query.temporal.is_none()
            && matches!(
                query.functional,
                crate::ResponseFunctionalWire::MeanCurve { .. }
                    | crate::ResponseFunctionalWire::InterventionResponse { .. }
            ))
        && static_response_estimator
    {
        // The completion envelope and per-completion mixing are the replay
        // dependency; one scalar program cannot represent the class family.
        unresolved.push(ContractReason::DependenciesCheckedStaticClassResponseOperation);
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Dag" | "Cpdag" | "Pag")
        && matches!(&contract.target.query, CausalQueryWire::Response(query)
        if query.temporal.is_none()
            && matches!(
                query.functional,
                crate::ResponseFunctionalWire::MeanCurve { .. }
                    | crate::ResponseFunctionalWire::InterventionResponse { .. }
            ))
        && static_response_estimator
    {
        // The atom-wise checked execution and frozen graph mass are the
        // replay dependency; one scalar program cannot represent the family.
        unresolved.push(ContractReason::DependenciesCheckedGraphPosteriorResponseOperation);
    }
    if matches!(contract.graph_class.as_str(), "Dag" | "Cpdag" | "Pag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. })
        && resolved_estimator == Some("conditional.linear.adjustment")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(ContractReason::DependenciesCheckedConditionalEffectOperation);
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::Mediation { .. })
        && resolved_estimator == Some("mediation.linear")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(ContractReason::DependenciesCheckedMediationOperation);
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::Mediation { .. })
        && resolved_estimator == Some("mediation.linear")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedBayesianMediationOperation);
    }
    if matches!(contract.graph_class.as_str(), "Dag" | "TemporalDag")
        && contract.structure_source == "explicit"
        && matches!(
            contract.target.query,
            CausalQueryWire::AnomalyAttribution { .. } | CausalQueryWire::ChangeAttribution { .. }
        )
        && ((resolved_estimator == Some("gcm.fit")
            && contract.program.as_ref().is_some_and(|program| {
                program.commitments.inference.eq_ignore_ascii_case("frequentist")
            }))
            || (matches!(contract.target.query, CausalQueryWire::AnomalyAttribution { .. })
                && resolved_estimator == Some("gcm.fit.bayesian")
                && contract.program.as_ref().is_some_and(|program| {
                    program.commitments.inference.eq_ignore_ascii_case("bayesian")
                }))
            || (matches!(contract.target.query, CausalQueryWire::ChangeAttribution { .. })
                && resolved_estimator == Some("gcm.attribution.bayesian")
                && contract.program.as_ref().is_some_and(|program| {
                    program.commitments.inference.eq_ignore_ascii_case("bayesian")
                })))
    {
        unresolved.push(ContractReason::DependenciesCheckedAttributionOperation);
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. })
        && resolved_estimator == Some("conditional.bayesian")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedBayesianConditionalOperation);
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("bayesian.gcomp")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedBayesianDagAteOperation);
    }
    let program_inference =
        contract.program.as_ref().map(|program| program.commitments.inference.to_ascii_lowercase());
    let temporal_response_procedure = matches!(
        (resolved_estimator, program_inference.as_deref()),
        (Some("temporal.response.gcomp"), Some("frequentist"))
            | (Some("response.temporal.bayesian"), Some("bayesian"))
    );
    let direct_temporal_response = matches!(&contract.target.query, CausalQueryWire::Response(query)
        if matches!(
            query.functional,
            crate::ResponseFunctionalWire::MeanCurve { .. }
                | crate::ResponseFunctionalWire::InterventionResponse { .. }
        ) && matches!(query.observation, crate::ObservationSpecWire::Complete));
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && direct_temporal_response
        && temporal_response_procedure
    {
        unresolved.push(ContractReason::DependenciesCheckedTemporalResponseOperation);
    }
    if matches!(contract.graph_class.as_str(), "TemporalCpdag" | "TemporalPag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && direct_temporal_response
        && temporal_response_procedure
    {
        unresolved.push(ContractReason::DependenciesCheckedTemporalClassResponseOperation);
    }
    if matches!(contract.graph_class.as_str(), "TemporalDag" | "TemporalCpdag" | "TemporalPag")
        && contract.structure_source == "graph_posterior"
        && direct_temporal_response
        && temporal_response_procedure
    {
        unresolved.push(ContractReason::DependenciesCheckedTemporalGraphPosteriorResponseOperation);
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(
            resolved_estimator,
            Some("temporal.linear.adjustment" | "temporal.sequential.gcomp")
        )
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(ContractReason::DependenciesCheckedTemporalDagEffectOperation);
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && resolved_estimator == Some("bayesian.temporal.gcomp")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedBayesianTemporalDagEffectOperation);
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && resolved_estimator == Some("temporal.sequential.gcomp")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedBayesianTemporalDagEffectOperation);
    }
    if matches!(contract.graph_class.as_str(), "TemporalDag" | "TemporalCpdag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted" | "graph_posterior")
        && matches!(contract.target.query, CausalQueryWire::Mediation { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(resolved_estimator, Some("temporal.mediation" | "temporal.mediation.bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedTemporalMediationOperation);
    }
    if matches!(contract.graph_class.as_str(), "TemporalCpdag" | "TemporalPag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(
            resolved_estimator,
            Some("temporal.linear.adjustment" | "temporal.sequential.gcomp")
        )
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(ContractReason::DependenciesCheckedTemporalClassEffectOperation);
    }
    if matches!(contract.graph_class.as_str(), "TemporalCpdag" | "TemporalPag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(
            resolved_estimator,
            Some("bayesian.temporal.gcomp" | "temporal.sequential.gcomp")
        )
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(ContractReason::DependenciesCheckedBayesianTemporalClassEffectOperation);
    }
    if matches!(contract.graph_class.as_str(), "TemporalDag" | "TemporalCpdag" | "TemporalPag")
        && contract.structure_source == "graph_posterior"
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(
            resolved_estimator,
            Some(
                "temporal.linear.adjustment"
                    | "bayesian.temporal.gcomp"
                    | "temporal.sequential.gcomp"
            )
        )
    {
        unresolved.push(ContractReason::DependenciesCheckedTemporalGraphPosteriorEffectOperation);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::contract_section) fn note_functional_and_nested_replay(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if is_functional_distribution_contract(contract) && body.interventional_distribution.is_none() {
        // Pre-atom artifacts remain decodable, but cannot independently verify
        // the distribution that produced the scalar summary.
        unresolved.push(ContractReason::BodyInterventionalDistribution);
    } else if crate::analysis_result_artifact::validate_interventional_distribution(body).is_err() {
        // Decoding already refuses such bodies; a caller verifying an
        // in-memory body must see the same refusal.
        unresolved.push(ContractReason::BodyInterventionalDistribution);
    }
    let factor_laws = contract
        .data_snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref());
    let posterior_distribution = is_functional_distribution_contract(contract)
        && (contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some()));
    let functional_effect = is_scalar_functional_effect_contract(contract);
    let graph_posterior_functional_effect = contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Admg" | "TemporalAdmg")
        && contract
            .program
            .as_ref()
            .and_then(|program| program.commitments.resolved_estimator.as_deref())
            .or(contract.estimator.as_deref())
            == Some("functional.effect");
    let posterior_functional_effect = functional_effect
        && (contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some()));
    if posterior_distribution {
        // Posterior atoms require joint per-draw factor tables and draw
        // identity. Empirical marginal laws cannot replay their weights.
        unresolved.push(ContractReason::DependenciesDistributionPosteriorFactorDraws);
    }
    if is_functional_distribution_contract(contract) && factor_laws.is_none() {
        // The checked expression proves what to evaluate, while the data snapshot
        // only carries partition digests. Without portable factor laws a detached
        // consumer cannot recompute the atom probabilities.
        unresolved.push(ContractReason::DependenciesDistributionFactorLaws);
    }
    if functional_effect
        && factor_laws.is_none()
        && !posterior_functional_effect
        && !graph_posterior_functional_effect
    {
        unresolved.push(ContractReason::DependenciesFunctionalEffectFactorLaws);
    }
    if posterior_functional_effect && !graph_posterior_functional_effect {
        // The scalar provider law is enough for a frequentist point replay, but Bayesian
        // results require the shared row weights for every posterior draw.
        unresolved.push(ContractReason::DependenciesFunctionalEffectPosteriorDraws);
    }
    if is_response_functional_effect_contract(contract) {
        let graph_posterior = contract.structure_source == "graph_posterior"
            && matches!(contract.graph_class.as_str(), "Admg" | "TemporalAdmg")
            && contract
                .program
                .as_ref()
                .and_then(|program| program.commitments.resolved_estimator.as_deref())
                .or(contract.estimator.as_deref())
                == Some("functional.effect");
        let posterior = contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some());
        if graph_posterior {
            // The atom-wise checked execution and frozen graph mass are the
            // replay dependency; one scalar program cannot represent the family.
        } else if posterior {
            unresolved.push(ContractReason::DependenciesFunctionalEffectResponsePosteriorDraws);
        } else {
            match (
                contract
                    .program
                    .as_ref()
                    .and_then(|program| program.checked_functional_response_grid.as_ref()),
                contract
                    .data_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref()),
                body.response.as_ref(),
            ) {
                (Some(grid), Some(laws), Some(response)) => {
                    if let Err(reason) = replay_functional_response_grid(
                        &contract.target.query,
                        grid,
                        &response.estimate,
                        laws,
                    ) {
                        unresolved.push(reason);
                    }
                }
                (None, _, _) => {
                    unresolved.push(ContractReason::ProgramFunctionalEffectResponseGrid)
                }
                (_, None, _) => {
                    unresolved.push(ContractReason::DependenciesFunctionalEffectResponseFactorLaws)
                }
                (_, _, None) => unresolved.push(ContractReason::BodyResponse),
            }
        }
    }
    if body.identification.query != body.query {
        unresolved.push(ContractReason::BodyIdentificationQuery);
    }
    if is_functional_distribution_contract(contract) && !posterior_distribution {
        match (
            body.interventional_distribution.as_ref(),
            contract.program.as_ref().and_then(|program| program.functional_program.as_ref()),
            contract
                .data_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref()),
        ) {
            (Some(result), Some(program), Some(laws)) => {
                if let Err(reason) = crate::distribution_replay::replay_distribution_atoms(
                    &body.query,
                    result,
                    program,
                    laws,
                ) {
                    unresolved.push(reason);
                }
            }
            _ => {
                if !unresolved
                    .iter()
                    .any(|reason| *reason == ContractReason::DependenciesDistributionFactorLaws)
                {
                    unresolved.push(ContractReason::DependenciesDistributionFactorLaws);
                }
            }
        }
    }
    if functional_effect && !posterior_functional_effect && !graph_posterior_functional_effect {
        match (
            contract.program.as_ref().and_then(|program| program.functional_program.as_ref()),
            contract
                .data_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref()),
        ) {
            (Some(program), Some(laws)) => {
                if let Err(reason) = crate::distribution_replay::replay_functional_scalar(
                    &body.query,
                    body.estimate,
                    program,
                    laws,
                ) {
                    unresolved.push(reason);
                }
            }
            (_, None) => {
                if !unresolved
                    .iter()
                    .any(|reason| *reason == ContractReason::DependenciesFunctionalEffectFactorLaws)
                {
                    unresolved.push(ContractReason::DependenciesFunctionalEffectFactorLaws);
                }
            }
            (None, Some(_)) => unresolved.push(ContractReason::ProgramFunctionalProgram),
        }
    }
    if matches!(contract.target.query, crate::CausalQueryWire::NestedCounterfactual { .. }) {
        if matches!(contract.graph_class.as_str(), "TemporalDag" | "TemporalCpdag" | "TemporalPag")
        {
            if let Err(reason) = verify_temporal_nested_point(contract, body) {
                unresolved.push(reason);
            }
        } else if let Err(reason) = verify_nested_counterfactual_result(contract, body) {
            unresolved.push(reason);
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::contract_section) fn note_published_claim(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    if contract.target.schema.variable_names() != header.variable_names {
        unresolved.push(ContractReason::HeaderVariableNames);
    }
    if let Some(slot) = contract.reasoning.identification.value.as_ref() {
        if slot.status != body.identification.status {
            unresolved.push(ContractReason::IdentificationStatus);
        }
        if validate_mixture_masses(
            slot.identified_mass,
            slot.unidentified_mass,
            slot.unevaluable_mass,
            slot.incomplete_search_mass,
        )
        .is_err()
        {
            unresolved.push(ContractReason::ReasoningIdentificationMasses);
        }
        if let Some(structural) = &body.structural_response {
            let masses_match = (slot.identified_mass - structural.identified_mass).abs() <= 1e-12
                && (slot.unidentified_mass - structural.unidentified_mass).abs() <= 1e-12
                && (slot.unevaluable_mass - structural.unevaluable_mass).abs() <= 1e-12
                && (slot.incomplete_search_mass - structural.subsampled_out_mass).abs() <= 1e-12;
            if !masses_match {
                unresolved.push(ContractReason::ReasoningIdentificationMasses);
            }
            if slot.weight_basis.as_deref() != Some(weight_basis_name(structural.weight_basis)) {
                unresolved.push(ContractReason::ReasoningIdentificationWeightBasis);
            }
        }
    }
    if let Some(product) = &contract.identification_product {
        if !identification_product_matches_body(product, body) {
            unresolved.push(ContractReason::BodyIdentificationProduct);
        }
    } else if contract.identities.identification_product.is_some() {
        unresolved.push(ContractReason::IdentitiesIdentificationProduct);
    }
    if contract_uses_estimator(contract, "frontdoor.linear_two_stage")
        && !body.assumptions.iter().any(|assumption| {
            matches!(&assumption.assumption,
                crate::trace::AssumptionTagWire::ParametricRestriction { id, .. }
                    if id == "frontdoor.linear_path_product")
        })
    {
        unresolved.push(ContractReason::BodyFrontdoorAssumption);
    }
    require_present(
        unresolved,
        ContractReason::IdentitiesIdentification,
        contract.identification.as_ref(),
    );
    require_present(unresolved, ContractReason::IdentitiesProgram, contract.program.as_ref());
    require_present(
        unresolved,
        ContractReason::IdentitiesInferenceBinding,
        contract.inference_binding.as_ref(),
    );
    require_present(
        unresolved,
        ContractReason::IdentitiesObservation,
        contract.observation.as_ref(),
    );
    require_present(
        unresolved,
        ContractReason::IdentitiesDataSnapshot,
        contract.data_snapshot.as_ref(),
    );
    if let Some(claim) = &contract.claim {
        if !claim_id_matches(contract, claim, body) {
            unresolved.push(ContractReason::ClaimId);
        }
        if claim.kind == "point" && !claim_value_matches_body(claim, body) {
            unresolved.push(ContractReason::ClaimValue);
        }
        if claim.kind != "point" && claim.value_bits.is_some() {
            unresolved.push(ContractReason::ClaimValue);
        }
        let identification = contract.reasoning.identification.value.as_ref();
        if claim.kind != claim_kind_name(body, identification) {
            unresolved.push(ContractReason::ClaimKind);
        }
        let support = contract.reasoning.support.value.as_ref();
        if let Some(slot) = support {
            let empirical = support_empirical(&body.refutations)
                .unwrap_or_else(|| "unavailable:not_evaluated".into());
            if slot.empirical != empirical {
                unresolved.push(ContractReason::ReasoningSupportEmpirical);
            }
        }
        let domains = claim_domains(support, identification);
        if claim.identification_domain != domains.identification
            || claim.support_domain != domains.support
            || claim.evaluated_domain != domains.evaluated
        {
            unresolved.push(ContractReason::ClaimDomains);
        }
        if claim.execution.is_some() && contract.execution.is_none() {
            unresolved.push(ContractReason::IdentitiesExecution);
        }
        verify_claim_calibration(contract, claim, unresolved);
    }
}

/// Producer validation permits readable exports with explicitly unavailable
/// replay dependencies. Independent consumption receives the unfiltered
/// refusals from [`verify_contract_against_body`]. A malformed present payload
/// remains fatal.
pub(crate) fn verify_contract_for_encoding(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
) -> Vec<Arc<str>> {
    publish_reasons(producer_encoding_unresolved(
        contract,
        contract_unresolved(header, body, contract),
    ))
}

pub(in crate::contract_section) fn producer_encoding_unresolved(
    contract: &AnalysisResultContractWire,
    mut unresolved: Vec<ContractReason>,
) -> Vec<ContractReason> {
    unresolved.retain(|key| !key.omitted_when_encoding());
    // Preserve portable structural artifacts while making the missing replay
    // input explicit to independent consumers.
    if is_functional_distribution_contract(contract) {
        unresolved.retain(|key| *key != ContractReason::DependenciesDistributionFactorLaws);
        unresolved
            .retain(|key| *key != ContractReason::DependenciesDistributionPosteriorFactorDraws);
    }
    if is_scalar_functional_effect_contract(contract) {
        unresolved.retain(|key| *key != ContractReason::DependenciesFunctionalEffectFactorLaws);
        unresolved.retain(|key| *key != ContractReason::DependenciesFunctionalEffectPosteriorDraws);
    }
    if is_response_functional_effect_contract(contract) {
        unresolved.retain(|key| *key != ContractReason::ProgramFunctionalEffectResponseGrid);
        unresolved
            .retain(|key| *key != ContractReason::DependenciesFunctionalEffectResponseFactorLaws);
        unresolved.retain(|key| {
            *key != ContractReason::DependenciesFunctionalEffectResponsePosteriorDraws
        });
    }
    if contract
        .program
        .as_ref()
        .is_some_and(|program| program.checked_linear_adjustment_lowering.is_some())
    {
        unresolved.retain(|key| *key != ContractReason::DependenciesLinearFitSufficientStatistics);
    }
    if is_aipw_contract(contract) && contract.program.is_some() {
        if contract.program.as_ref().is_some_and(|program| program.checked_aipw_lowering.is_none())
        {
            unresolved.retain(|key| *key != ContractReason::ProgramCheckedAipwLowering);
        }
        if contract.checked_aipw_rows.is_none() && contract.identities.checked_aipw_rows.is_none() {
            unresolved.retain(|key| *key != ContractReason::CheckedAipwRowBinding);
        }
    }
    let is_iv = matches!(contract.estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        || contract.program.as_ref().is_some_and(|program| {
            matches!(program.commitments.resolved_estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        });
    if is_iv
        && contract.program.as_ref().is_some_and(|program| program.checked_iv_lowering.is_none())
    {
        unresolved.retain(|key| *key != ContractReason::ProgramCheckedIvLowering);
    }
    unresolved
}

/// Re-derive the claim's calibration slot from the basis it carries and check
/// that basis against the payloads this contract rehashes.
pub(in crate::contract_section) fn verify_claim_calibration(
    contract: &AnalysisResultContractWire,
    claim: &ClaimSectionWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let slot = &claim.calibration;
    let statuses_ok = std::iter::once(slot).chain(slot.secondary.iter()).all(|slot| {
        matches!(slot.status.as_str(), "calibrated" | "scope_not_assessed" | "unavailable")
    });
    if !statuses_ok || crate::calibration::rederive_calibration(slot) != *slot {
        unresolved.push(ContractReason::ClaimCalibration);
    }
    if contract
        .reasoning
        .support
        .value
        .as_ref()
        .is_some_and(|support| support.matrix_status == "off_axis")
        && std::iter::once(slot)
            .chain(slot.secondary.iter())
            .any(|entry| entry.status == "calibrated")
    {
        unresolved.push(ContractReason::ClaimCalibrationOffAxis);
    }
    let consistent = std::iter::once(slot)
        .chain(slot.secondary.iter())
        .filter_map(|slot| slot.basis.as_ref())
        .all(|basis| calibration_basis_matches_contract(contract, basis));
    if !consistent {
        unresolved.push(ContractReason::ClaimCalibrationBasis);
    }
}

/// The basis fields a consumer can recompute from rehashed payloads: the
/// support coordinate (query, graph axis, structure, inference), the resolved
/// estimator, the snapshot row count and modality, and the identification
/// masses.
pub(in crate::contract_section) fn calibration_basis_matches_contract(
    contract: &AnalysisResultContractWire,
    basis: &crate::calibration::CalibrationBasisWire,
) -> bool {
    let key = &basis.key;
    let coordinate_ok = contract
        .reasoning
        .support
        .value
        .as_ref()
        .and_then(|slot| slot.matrix_coordinate.as_deref())
        .is_some_and(|coordinate| {
            let mut parts = coordinate.split(':');
            let (Some(query), Some(graph), Some(structure), Some(inference)) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return false;
            };
            let structure =
                if structure == "graph_posterior" { "graph_posterior" } else { "fixed" };
            query == key.query
                && graph == key.graph_class
                && structure == key.structure
                && inference == key.inference
        })
        || contract.reasoning.support.value.as_ref().is_some_and(|slot| {
            // Exact graphless DiD licenses are bound to the executed result.
            // The calibration key still names the compiled query because the
            // geometric support registry does not contain graphless rows.
            slot.matrix_status == "licensed"
                && slot.matrix_coordinate.as_deref().is_some_and(|coordinate| {
                    coordinate.starts_with("graphless:difference_in_differences/")
                })
                && matches!(contract.target.query, crate::CausalQueryWire::PanelDid(_))
                && key.query == "Unknown"
                && key.graph_class == contract.graph_class
                && key.structure == "fixed"
        })
        || contract.reasoning.support.value.as_ref().is_some_and(|slot| {
            // Off-axis design results retain their interval construction in
            // the claim even though the support matrix has no licensed cell.
            // The query-specific payload validator verifies its support and
            // endpoints; calibration remains unavailable on this axis.
            slot.matrix_status == "off_axis"
                && slot.matrix_coordinate.is_none()
                && key.query == "Unknown"
                && (key.interval_method == "none"
                    || (key.inference.eq_ignore_ascii_case("bayesian")
                        && key.interval_method == "posterior_quantile")
                    || (matches!(contract.target.query, crate::CausalQueryWire::PanelDid(_))
                        && key.interval_method == "analytic_se"
                        && key.dependence == "cluster"))
                && key.graph_class == contract.graph_class
                && key.structure == "fixed"
        });
    let program_ok = contract.program.as_ref().is_some_and(|program| {
        let commitments = &program.commitments;
        commitments.inference.eq_ignore_ascii_case(&key.inference)
            && commitments
                .resolved_estimator
                .as_deref()
                .is_none_or(|resolved| resolved == key.estimator)
    });
    let snapshot_ok = contract.data_snapshot.as_ref().is_some_and(|snapshot| {
        basis.scope.row_count <= snapshot.row_count
            && snapshot.modality == key.modality
            && match snapshot.modality.as_str() {
                "panel" => key.dependence == "panel_cluster",
                "tabular" => {
                    key.dependence == "iid"
                        || (matches!(contract.target.query, crate::CausalQueryWire::PanelDid(_))
                            && key.dependence == "cluster")
                }
                _ => key.dependence != "panel_cluster",
            }
    });
    let identification_ok = contract.reasoning.identification.value.as_ref().is_none_or(|slot| {
        let unidentified =
            slot.unidentified_mass + slot.unevaluable_mass + slot.incomplete_search_mass;
        (unidentified - basis.scope.unidentified_mass).abs() <= 1e-9
    });
    coordinate_ok && program_ok && snapshot_ok && identification_ok
}

#[allow(
    clippy::too_many_lines,
    reason = "Stored-payload verification must retain the complete cross-layer digest checklist."
)]
pub(in crate::contract_section) fn verify_stored_payloads(
    contract: &AnalysisResultContractWire,
) -> Vec<ContractReason> {
    let mut unresolved = Vec::new();
    require_slot(
        &mut unresolved,
        ContractReason::ReasoningIdentification,
        &contract.reasoning.identification,
    );
    require_slot(&mut unresolved, ContractReason::ReasoningSupport, &contract.reasoning.support);
    require_slot(
        &mut unresolved,
        ContractReason::ReasoningUncertainty,
        &contract.reasoning.uncertainty,
    );
    require_slot(
        &mut unresolved,
        ContractReason::ReasoningAssumptions,
        &contract.reasoning.assumptions,
    );
    match digest_wire(IdentityDomain::Target, &contract.target) {
        Ok(target) if target.as_bytes() == &contract.identities.target => {}
        Ok(_) => unresolved.push(ContractReason::IdentitiesTarget),
        Err(_) => unresolved.push(ContractReason::TargetPayload),
    }
    require_payload_digest(
        &mut unresolved,
        ContractReason::IdentitiesIdentification,
        contract.identification.as_ref(),
        Some(&contract.identities.identification),
        identification_digest,
    );
    require_payload_digest(
        &mut unresolved,
        ContractReason::IdentitiesIdentificationProduct,
        contract.identification_product.as_ref(),
        contract.identities.identification_product.as_ref(),
        identification_product_digest_wire,
    );
    require_payload_digest(
        &mut unresolved,
        ContractReason::IdentitiesProgram,
        contract.program.as_ref(),
        Some(&contract.identities.program),
        program_digest,
    );
    let graph_posterior_functional_effect = contract.structure_source == "graph_posterior"
        && matches!(contract.graph_class.as_str(), "Admg" | "TemporalAdmg")
        && contract
            .program
            .as_ref()
            .and_then(|program| program.commitments.resolved_estimator.as_deref())
            .or(contract.estimator.as_deref())
            == Some("functional.effect");
    let functional = contract
        .program
        .as_ref()
        .and_then(|item| item.functional_program.as_ref())
        .filter(|_| !graph_posterior_functional_effect);
    let posterior_distribution = is_functional_distribution_contract(contract)
        && (contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some()));
    if is_functional_distribution_contract(contract)
        && functional.is_none()
        && !posterior_distribution
    {
        // Older artifacts stay readable but cannot advertise a verified
        // executable distribution without the program that was evaluated.
        unresolved.push(ContractReason::ProgramFunctionalProgram);
    }
    if is_scalar_functional_effect_contract(contract)
        && functional.is_none()
        && !graph_posterior_functional_effect
    {
        unresolved.push(ContractReason::ProgramFunctionalProgram);
    }
    if let Some(program) = functional.filter(|_| {
        !posterior_distribution
            && (is_functional_distribution_contract(contract)
                || is_scalar_functional_effect_contract(contract))
    }) {
        if crate::functional_program_from_wire(program, antecedent_expr::ProgramLimits::default())
            .is_err()
        {
            unresolved.push(ContractReason::ProgramFunctionalProgram);
        }
        let expected_variables: Vec<_> = contract
            .target
            .schema
            .variables
            .iter()
            .map(|variable| (variable.id, variable.name.clone()))
            .collect();
        let matches_identification =
            contract.identification_product.as_ref().is_some_and(|product| {
                program.arena == product.arena
                    && program.source == program.executable
                    && product
                        .estimands
                        .iter()
                        .any(|estimand| estimand.functional == program.source)
            });
        let distribution_binding = is_functional_distribution_contract(contract)
            && matches!(contract.target.query, crate::CausalQueryWire::Distribution(_));
        let effect_method = match &contract.target.query {
            crate::CausalQueryWire::AverageEffect { .. } => Some("general.id"),
            crate::CausalQueryWire::PathSpecific(_) => Some("path_specific.natural"),
            _ => None,
        };
        let effect_binding = is_scalar_functional_effect_contract(contract)
            && effect_method.is_some_and(|method| {
                contract.identification_product.as_ref().is_some_and(|product| {
                    product.estimands.iter().any(|estimand| {
                        estimand.functional == program.source && estimand.method == method
                    })
                })
            });
        if (!distribution_binding && !effect_binding)
            || program.variables != expected_variables
            || !matches_identification
        {
            unresolved.push(ContractReason::ProgramFunctionalBinding);
        }
    }
    let frontdoor =
        contract.program.as_ref().and_then(|item| item.checked_frontdoor_lowering.as_ref());
    if contract_uses_estimator(contract, "frontdoor.linear_two_stage") && frontdoor.is_none() {
        unresolved.push(ContractReason::ProgramCheckedFrontdoorLowering);
    }
    if let Some(frontdoor) = frontdoor {
        if !verify_frontdoor_lowering(contract, frontdoor) {
            unresolved.push(ContractReason::ProgramFrontdoorBinding);
        }
    }
    let checked_iv = contract.program.as_ref().and_then(|item| item.checked_iv_lowering.as_ref());
    let is_checked_iv = matches!(contract.estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        || contract.program.as_ref().is_some_and(|item| {
            matches!(item.commitments.resolved_estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        });
    if is_checked_iv && checked_iv.is_none() {
        unresolved.push(ContractReason::ProgramCheckedIvLowering);
    }
    if let Some(iv) = checked_iv {
        if !verify_checked_iv(contract, iv) {
            unresolved.push(ContractReason::ProgramCheckedIvBinding);
        }
    }
    verify_checked_linear_adjustment(contract, &mut unresolved);
    verify_checked_aipw(contract, &mut unresolved);
    require_payload_digest(
        &mut unresolved,
        ContractReason::IdentitiesInferenceBinding,
        contract.inference_binding.as_ref(),
        Some(&contract.identities.inference_binding),
        inference_binding_digest,
    );
    require_payload_digest(
        &mut unresolved,
        ContractReason::IdentitiesObservation,
        contract.observation.as_ref(),
        Some(&contract.identities.observation),
        |observation| digest_wire(IdentityDomain::Observation, observation),
    );
    require_payload_digest(
        &mut unresolved,
        ContractReason::IdentitiesDataSnapshot,
        contract.data_snapshot.as_ref(),
        Some(&contract.identities.data_snapshot),
        data_snapshot_digest,
    );
    require_payload_digest(
        &mut unresolved,
        ContractReason::IdentitiesExecution,
        contract.execution.as_ref(),
        contract.identities.execution.as_ref(),
        execution_digest,
    );
    require_nested_digest(
        &mut unresolved,
        ContractReason::IdentitiesExecution,
        contract.identities.execution,
        contract.claim.as_ref().and_then(|claim| claim.execution.as_ref()),
    );
    require_nested_digest(
        &mut unresolved,
        ContractReason::ProgramTarget,
        contract.program.as_ref().map(|item| item.target),
        Some(&contract.identities.target),
    );
    require_nested_digest(
        &mut unresolved,
        ContractReason::ProgramIdentification,
        contract.program.as_ref().map(|item| item.identification),
        Some(&contract.identities.identification),
    );
    require_nested_digest(
        &mut unresolved,
        ContractReason::ProgramIdentificationProduct,
        contract.program.as_ref().and_then(|item| item.identification_product),
        contract.identities.identification_product.as_ref(),
    );
    require_nested_digest(
        &mut unresolved,
        ContractReason::DataSnapshotObservation,
        contract.data_snapshot.as_ref().map(|item| item.observation),
        Some(&contract.identities.observation),
    );
    verify_checked_aipw_rows(contract, &mut unresolved);
    verify_layer_links(contract, &mut unresolved);
    match contract_seal_of(contract) {
        Ok(seal) if seal == contract.seal => {}
        _ => unresolved.push(ContractReason::ContractSeal),
    }
    unresolved
}
