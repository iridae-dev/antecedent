//! The retained operation's inspection record, projected for `plan_view`.
//!
//! Every sealed route that publishes an inspection record projects it here, so a reader of
//! `plan_view().operation` sees the target, procedure and frozen plan the route will run
//! without executing it. Routes that publish none name only their kind.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::PyPreparedAnalysis;

impl PyPreparedAnalysis {
    fn variable_name(&self, id: antecedent_core::VariableId) -> String {
        self.names.get(id.as_usize()).cloned().unwrap_or_else(|| id.to_string())
    }

    /// The retained operation's inspection record, by its stable name.
    pub(super) fn operation_detail(&self) -> serde_json::Value {
        use antecedent::analysis::RetainedOperation as R;
        use serde_json::json;
        let operation = self.inner.retained_operation();
        let name = |id: antecedent_core::VariableId| self.variable_name(id);
        let names = |ids: &[antecedent_core::VariableId]| {
            ids.iter().copied().map(|id| self.variable_name(id)).collect::<Vec<_>>()
        };
        let mut detail = match &operation {
            R::CheckedConditional(info) => json!({
                "query": format!("{:?}", info.query),
                "treatment": name(info.query.inner.treatment),
                "outcome": name(info.query.inner.outcome),
                "modifiers": names(&info.query.inner.effect_modifiers[..]),
                "control": format!("{:?}", info.query.inner.control),
                "active": format!("{:?}", info.query.inner.active),
                "functional": format!("{:?}", info.query.inner.outcome_functional),
                "population": format!("{:?}", info.query.inner.target_population),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "procedure": info.procedure.as_ref(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "design_roles": names(&info.design_roles[..]),
                "source_rows": info.source_rows.as_ref(),
            }),
            R::StaticMediation(info) | R::BayesianStaticMediation(info) => json!({
                "query": format!("{:?}", info.query),
                "treatment": name(info.query.treatment),
                "outcome": name(info.query.outcome),
                "mediators": names(&info.query.mediators[..]),
                "contrast": format!("{:?}", info.query.contrast),
                "control": format!("{:?}", info.query.control),
                "active": format!("{:?}", info.query.active),
                "population": format!("{:?}", info.query.target_population),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "bootstrap_replicates": info.bootstrap_replicates,
                "adjustment_set": names(&info.adjustment_set[..]),
                "functional_roots": info.functional_roots,
                "graph_edges": info.graph_edges.as_ref(),
            }),
            R::BayesianGcomp { operation, validation } => {
                let query = operation.query();
                let mut detail = json!({
                    "query": format!("{query:?}"),
                    "treatment": name(query.treatment),
                    "outcome": name(query.outcome),
                    "population": format!("{:?}", query.target_population),
                    "identifier": "backdoor.adjustment",
                    "estimator": "bayesian.gcomp",
                    "validation": validation.validation_suite_id().unwrap_or("none"),
                    "adjustment_set": names(&operation.estimand().adjustment_set[..]),
                });
                if let antecedent::InferenceMode::Bayesian(config) = operation.inference() {
                    detail["backend"] = json!(format!("{:?}", config.backend));
                    detail["likelihood"] = json!(format!("{:?}", config.likelihood));
                    detail["n_draws"] = json!(config.n_draws);
                    detail["prior_scale"] = json!(config.prior_scale);
                    detail["transferred_prior"] = json!(config.prior_artifact.is_some());
                    detail["composed_prior"] = json!(config.external_compose.is_some());
                }
                detail
            }
            R::StaticDagResponse(info) => json!({
                "query": format!("{:?}", info.query.functional),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "grid_members": info.grid_members.as_ref(),
            }),
            R::GraphPosteriorEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "conditional": info.conditional,
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "graph_keys": info.graph_keys.as_ref(),
                "weights": info.weights.as_ref(),
                "identified": info.identified.iter().map(|flag| format!("{flag:?}")).collect::<Vec<_>>(),
                "bootstrap_replicates": info.bootstrap_replicates,
            }),
            R::TemporalDagEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "policy": format!("{:?}", info.query.policy),
                "control": format!("{:?}", info.query.control),
                "active": format!("{:?}", info.query.active),
                "horizon_steps": info.query.horizon_steps,
                "population": format!("{:?}", info.query.target_population),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "bootstrap_replicates": info.bootstrap_replicates,
                "adjustment_set": info.adjustment_set.iter().map(|key| format!("{key:?}")).collect::<Vec<_>>(),
                "graph_signature": info.graph_signature.as_ref(),
            }),
            R::TemporalDagResponse(info) => json!({
                "query": format!("{:?}", info.query.functional),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "uncertainty": info.uncertainty.as_ref(),
                "grid_members": info.grid_members.as_ref(),
                "horizons": info.horizons.as_ref(),
            }),
            R::CheckedPropensity(info) => json!({
                "estimator": info.estimator.as_str(),
                "adjustment_set": names(&info.adjustment_set[..]),
                "population": format!("{:?}", info.population),
                "source_rows": info.source_rows.iter().map(|id| id.get()).collect::<Vec<_>>(),
                "uncertainty": info.uncertainty.as_ref(),
                "bootstrap_replicates": info.bootstrap_replicates,
            }),
            R::Attribution(info) => json!({
                "query": format!("{:?}", info.query),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "graph_edges": info.graph_edges.as_ref(),
                "identification_status": format!("{:?}", info.identification_status),
            }),
            R::BayesianBasisAte(info) => json!({
                "query": format!("{:?}", info.query),
                "treatment": name(info.query.treatment),
                "outcome": name(info.query.outcome),
                "adjustment_set": names(&info.adjustment_set[..]),
                "posterior_draws": info.posterior_draws,
                "prior_scale": info.prior_scale,
            }),
            R::BayesianRobustAte(info) => json!({
                "query": format!("{:?}", info.query),
                "treatment": name(info.query.treatment),
                "outcome": name(info.query.outcome),
                "adjustment_set": names(&info.adjustment_set[..]),
                "posterior_draws": info.posterior_draws,
                "row_ids": info.row_ids.as_ref(),
                "fold_ids": info.fold_ids.as_ref(),
            }),
            R::BayesianSpecialist(info) => json!({
                "query": format!("{:?}", info.query),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "n_draws": info.n_draws,
                "prior_scale": info.prior_scale,
                "rd": info.rd.as_ref().map(|rd| format!("{rd:?}")),
                "plan_id": info.plan_id.as_ref(),
            }),
            R::BayesianClassConditional(info) => json!({
                "query": format!("{:?}", info.query),
                "graph_class": info.graph_class.as_str(),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "inference": format!("{:?}", info.inference),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "adjustment_set": names(&info.adjustment_set[..]),
                "completion_count": info.completion_count,
                "identified_mass": info.identified_mass,
                "unresolved_mass": info.unresolved_mass,
            }),
            R::BayesianGraphPosteriorAte(info) => json!({
                "query": format!("{:?}", info.query),
                "conditional": info.conditional,
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "graph_keys": info.graph_keys.as_ref(),
                "weights": info.weights.as_ref(),
                "identified": info.identified.iter().map(|flag| format!("{flag:?}")).collect::<Vec<_>>(),
                "inference": info.inference.as_ref(),
            }),
            R::BayesianTemporalDagEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "posterior_draws": info.posterior_draws,
                "graph_signature": info.graph_signature.as_ref(),
            }),
            R::BayesianTemporalClassEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "graph_class": info.graph_class.as_str(),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "posterior_draws": info.posterior_draws,
                "class_prior_supplied": info.class_prior_supplied,
                "completion_count": info.completion_count,
                "completion_limit": info.completion_limit,
                "identified_mass": info.identified_mass,
                "unidentified_mass": info.unidentified_mass,
                "truncated_completions": info.truncated_completions,
                "graph_version": info.graph_version,
            }),
            R::CellAipwResponse(info) => json!({
                "query": format!("{:?}", info.query.functional),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "origin": format!("{:?}", info.origin),
                "requested_arm": info.requested_arm,
                "adjustment_set": names(&info.adjustment_set[..]),
                "within_tier": info.within_tier.as_ref().map(|tier| format!("{tier:?}")),
            }),
            R::ClassGraphPosteriorEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "conditional": info.conditional,
                "inference": info.inference.as_ref(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "graph_keys": info.graph_keys.as_ref(),
                "weights": info.weights.as_ref(),
                "identified": info.identified.iter().map(|flag| format!("{flag:?}")).collect::<Vec<_>>(),
                "class_atom_count": info.class_atom_count,
                "bootstrap_replicates": info.bootstrap_replicates,
            }),
            R::DerivativeResponse(info) => json!({
                "query": format!("{:?}", info.query.functional),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "adjustment_set": names(&info.adjustment_set[..]),
                "max_derivative_cells": info.max_derivative_cells,
            }),
            R::StaticClassEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "graph_class": info.graph_class.as_str(),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "inference": format!("{:?}", info.inference),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "completion_count": info.completion_count,
                "identified_mass": info.identified_mass,
                "unresolved_mass": info.unresolved_mass,
            }),
            R::TemporalClassEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "graph_class": info.graph_class.as_str(),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "bootstrap_replicates": info.bootstrap_replicates,
                "completion_count": info.completion_count,
                "completion_limit": info.completion_limit,
                "identified_mass": info.identified_mass,
                "unidentified_mass": info.unidentified_mass,
                "truncated_completions": info.truncated_completions,
                "graph_version": info.graph_version,
            }),
            R::TemporalClassResponse(info) => json!({
                "query": format!("{:?}", info.query.functional),
                "graph_class": info.graph_class.as_str(),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "inference": info.inference.as_ref(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "bootstrap_replicates": info.bootstrap_replicates,
                "horizons": info.horizons.as_ref(),
                "completion_count": info.completion_count,
                "completion_limit": info.completion_limit,
                "identified_mass": info.identified_mass,
                "unidentified_mass": info.unidentified_mass,
                "graph_version": info.graph_version,
            }),
            R::TemporalGraphPosteriorEffect(info) => json!({
                "query": format!("{:?}", info.query),
                "atom_kind": format!("{:?}", info.atom_kind),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "inference": info.inference.as_ref(),
                "graph_keys": info.graph_keys.as_ref(),
                "weights": info.weights.as_ref(),
                "identified": info.identified.iter().map(|flag| format!("{flag:?}")).collect::<Vec<_>>(),
                "identified_atom_count": info.identified_atom_count,
                "bootstrap_replicates": info.bootstrap_replicates,
            }),
            R::TemporalGraphPosteriorResponse(info) | R::GraphPosteriorResponse(info) => {
                let mut record = json!({
                    "query": format!("{:?}", info.query.functional),
                    "atom_kind": format!("{:?}", info.atom_kind),
                    "identifier": info.identifier.as_str(),
                    "estimator": info.estimator.as_str(),
                    "inference": info.inference.as_ref(),
                    "validation": info.validation.validation_suite_id().unwrap_or("none"),
                    "graph_keys": info.graph_keys.as_ref(),
                    "weights": info.weights.as_ref(),
                    "identified": info.identified.iter().map(|flag| format!("{flag:?}")).collect::<Vec<_>>(),
                });
                // A temporal response publishes the horizons its atoms were proved over.
                if let Some(spec) = info.query.temporal.as_ref() {
                    record["horizons"] = json!(spec.horizons.as_ref());
                }
                record
            }
            R::TemporalClassMediation(info) => json!({
                "query": format!("{:?}", info.query),
                "graph_class": info.graph_class.as_str(),
                "structure_source": info.structure_source.as_str(),
                "identifier": info.identifier.as_ref(),
                "estimator": info.estimator.as_str(),
                "validation": info.validation.validation_suite_id().unwrap_or("none"),
                "bootstrap_replicates": info.bootstrap_replicates,
                "horizons": info.horizons.as_ref(),
                "proof_masses": info.proof_masses.iter().map(|mass| json!({
                    "horizon": mass.horizon,
                    "atom_count": mass.atom_count,
                    "basis": mass.basis.as_str(),
                    "identified_mass": mass.identified_mass,
                    "unidentified_mass": mass.unidentified_mass,
                    "truncated_completions": mass.truncated_completions,
                })).collect::<Vec<_>>(),
                "atom_count": info.atom_count,
                "identified_mass": info.identified_mass,
                "unidentified_mass": info.unidentified_mass,
                "proof_signature": info.proof_signature.as_ref(),
                "scientific_targets": info.scientific_targets.iter().map(|target| json!({
                    "horizon": target.horizon,
                    "atom_key": target.atom_key,
                    "completion_index": target.completion_index,
                    "atom_weight": target.atom_weight,
                    "completion_weight": target.completion_weight,
                    "atom_identified": target.atom_identified,
                    "identification_status": target.identification_status.as_ref().map(|status| format!("{status:?}")),
                    "scientific_target": target.scientific_target.as_ref().map(|t| format!("{t:?}")),
                })).collect::<Vec<_>>(),
            }),
            R::TransportTrial(info) => json!({
                "query": format!("{:?}", info.query),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "rule": info.rule.as_ref(),
                "posterior_draws": info.posterior_draws,
                "plan_id": info.plan_id.as_ref(),
            }),
            R::Interference(info) => json!({
                "query": format!("{:?}", info.query),
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "unit_count": info.unit_count,
                "network_edge_count": info.network_edge_count,
            }),
            R::UnknownTieredAverage(info) => json!({
                "query": format!("{:?}", info.query),
                "treatment": name(info.query.treatment),
                "outcome": name(info.query.outcome),
                "tier_count": info.tier_count,
                "scenario_count": info.scenario_count,
                "identifier": info.identifier.as_str(),
                "estimator": info.estimator.as_str(),
                "plan_id": info.plan_id.as_ref(),
            }),
            _ => json!({}),
        };
        detail["kind"] = json!(operation.name());
        detail
    }
}
