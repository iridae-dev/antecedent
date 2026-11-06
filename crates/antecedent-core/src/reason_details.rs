//! Generated from `parity/reason_details.toml`. Do not edit.
//!
//! Every refusal detail id is a constant here: `reason_details::<family>::<NAME>`.

#![cfg_attr(rustfmt, rustfmt::skip)]

/// Every registered detail id, sorted.
pub const REASON_DETAILS: &[&str] = &[
    "adaptive_inference.analysis_invalid",
    "adaptive_inference.analysis_route_mismatch",
    "adaptive_inference.anytime_premise_contexts",
    "adaptive_inference.anytime_settings_invalid",
    "adaptive_inference.anytime_settings_missing",
    "adaptive_inference.arm_not_observed",
    "adaptive_inference.arm_out_of_range",
    "adaptive_inference.arms_invalid",
    "adaptive_inference.artifact_size_exceeded",
    "adaptive_inference.assigned_probability_zero",
    "adaptive_inference.assignment_not_replayed",
    "adaptive_inference.assignment_probabilities_missing",
    "adaptive_inference.batch_incomplete",
    "adaptive_inference.between_unit_variance_missing",
    "adaptive_inference.borrowing_fixed_after_first_look",
    "adaptive_inference.cancelled",
    "adaptive_inference.cell_incoherent",
    "adaptive_inference.cell_not_delivered",
    "adaptive_inference.choice_outcome_needs_choice_contract",
    "adaptive_inference.contexts_invalid",
    "adaptive_inference.correlation_invalid",
    "adaptive_inference.declared_looks_exceeded",
    "adaptive_inference.declared_units_exceeded",
    "adaptive_inference.delay_not_ignorable",
    "adaptive_inference.exact_law_states_exceeded",
    "adaptive_inference.exact_law_unavailable",
    "adaptive_inference.exact_law_work_exceeded",
    "adaptive_inference.exposure_differs_from_assignment",
    "adaptive_inference.exposure_history_missing",
    "adaptive_inference.exposure_invalid",
    "adaptive_inference.final_analysis_missing",
    "adaptive_inference.fixed_sample_data_dependent_stopping",
    "adaptive_inference.fixed_sample_design_adaptive",
    "adaptive_inference.fixed_sample_on_adaptive_data",
    "adaptive_inference.group_sequential_assignment_adaptive",
    "adaptive_inference.guarantee_not_transferable",
    "adaptive_inference.historical_borrowing_missing",
    "adaptive_inference.historical_prior_invalid",
    "adaptive_inference.independence_assumed_for_dependent_measurements",
    "adaptive_inference.input_invalid",
    "adaptive_inference.interim_analyses_missing",
    "adaptive_inference.interval_names_no_quantity",
    "adaptive_inference.inversion_premise_contexts",
    "adaptive_inference.ledger_limit",
    "adaptive_inference.look_out_of_range",
    "adaptive_inference.looks_not_ordered",
    "adaptive_inference.looks_overlap_unknown",
    "adaptive_inference.looks_share_observations",
    "adaptive_inference.measurements_invalid",
    "adaptive_inference.no_stagewise_adjustment_procedure",
    "adaptive_inference.no_stopping_reliability_procedure",
    "adaptive_inference.numerical_failure",
    "adaptive_inference.outcome_availability_invalid",
    "adaptive_inference.outcome_availability_missing",
    "adaptive_inference.outcome_dependent_delay",
    "adaptive_inference.outcome_invalid",
    "adaptive_inference.outcome_structure_missing",
    "adaptive_inference.pairing_not_supported",
    "adaptive_inference.paradigm_not_supported_for_cell",
    "adaptive_inference.paradigm_rule_mismatch",
    "adaptive_inference.policy_history_invalid",
    "adaptive_inference.policy_history_missing",
    "adaptive_inference.policy_selection_missing",
    "adaptive_inference.positivity_floor",
    "adaptive_inference.posterior_premise_outcome_dependent_delay",
    "adaptive_inference.probabilities_invalid",
    "adaptive_inference.probability_floor_invalid",
    "adaptive_inference.procedure_limit_exceeded",
    "adaptive_inference.repeated_measures_paradigm_not_derived",
    "adaptive_inference.repeated_measures_stopping_not_derived",
    "adaptive_inference.report_replay_mismatch",
    "adaptive_inference.retained_record_exceeded",
    "adaptive_inference.rule_mismatch",
    "adaptive_inference.selection_not_ignorable",
    "adaptive_inference.shape_mismatch",
    "adaptive_inference.simulation_memory_exceeded",
    "adaptive_inference.simulation_replications_exceeded",
    "adaptive_inference.simulation_work_exceeded",
    "adaptive_inference.stopping_not_replayed",
    "adaptive_inference.stopping_rule_missing",
    "adaptive_inference.structure_incomplete",
    "adaptive_inference.unit_out_of_range",
    "adaptive_inference.variance_invalid",
    "adaptive_inference.within_unit_correlation_missing",
    "adaptive_inference.within_unit_variance_missing",
    "adaptive_inference.wrong_contract",
    "adaptive_transport.arms",
    "adaptive_transport.assignment_below_floor",
    "adaptive_transport.assignment_monitoring_contract_missing",
    "adaptive_transport.design_not_adaptive",
    "adaptive_transport.diagram_binding_mismatch",
    "adaptive_transport.mechanism_declared_twice",
    "adaptive_transport.mechanism_status_missing",
    "adaptive_transport.no_analysed_units",
    "adaptive_transport.outcome_mechanism_not_invariant",
    "adaptive_transport.outcome_missing",
    "adaptive_transport.population_label",
    "adaptive_transport.selection_diagram_missing",
    "adaptive_transport.strata",
    "adaptive_transport.stratum_arm_unobserved",
    "adaptive_transport.stratum_law_declared_invariant",
    "adaptive_transport.target_is_source",
    "adaptive_transport.target_stratum_law",
    "adaptive_transport.target_stratum_law_missing",
    "adaptive_transport.unit_strata",
    "adaptive_transport.value_mismatch",
    "admg_conditional_posterior.band_not_constructed",
    "admg_conditional_posterior.empty_law",
    "admg_conditional_posterior.external_power_prior",
    "admg_conditional_posterior.mapped_prior",
    "admg_conditional_posterior.non_positive_scale",
    "admg_transport.bounds_exceeded",
    "admg_transport.budget",
    "admg_transport.consumer_limits",
    "admg_transport.counted_laws_unsupported",
    "admg_transport.data_identity_mismatch",
    "admg_transport.evidence_mismatch",
    "admg_transport.invalid_artifact",
    "admg_transport.invalid_catalog",
    "admg_transport.invalid_derivation",
    "admg_transport.invalid_query",
    "admg_transport.invalid_request",
    "admg_transport.missing_evidence",
    "admg_transport.not_certified",
    "admg_transport.premises_mismatch",
    "admg_transport.proven_non_transportable",
    "admg_transport.replay_mismatch",
    "admg_transport.support_failure",
    "aligned_joint_draws.identity_expected",
    "aligned_joint_draws.legacy_binding",
    "aligned_joint_draws.legacy_source",
    "aligned_joint_draws.marginals_not_joint",
    "aligned_joint_draws.outcome_coordinate",
    "batch_retarget.cancelled",
    "batch_retarget.contrast_member_failed",
    "batch_retarget.covariance_unavailable",
    "batch_retarget.duplicate_name",
    "batch_retarget.empty_family",
    "batch_retarget.functional_not_supported",
    "batch_retarget.incompatible_target",
    "batch_retarget.invalid_contrast",
    "batch_retarget.max_t_draws_out_of_range",
    "batch_retarget.max_t_invalid_level",
    "batch_retarget.mixed_snapshot",
    "batch_retarget.partial_family",
    "batch_retarget.point_only_member",
    "batch_retarget.scores_mismatch",
    "batch_retarget.scores_unavailable",
    "batch_retarget.scores_unavailable_after_estimate",
    "batch_retarget.unknown_claim",
    "batch_retarget.weighted_overlap_failed",
    "batch_retarget_posterior.band_not_constructed",
    "batch_retarget_posterior.empty_scores",
    "batch_retarget_posterior.external_power_prior",
    "batch_retarget_posterior.mapped_prior",
    "batch_retarget_posterior.non_positive_scale",
    "bayesian_transport.admission_refused",
    "bayesian_transport.cancelled",
    "bayesian_transport.feature_mismatch",
    "bayesian_transport.identification_not_certified",
    "bayesian_transport.identification_required",
    "bayesian_transport.invalid_input",
    "bayesian_transport.invalid_prior",
    "bayesian_transport.missing_law",
    "bayesian_transport.missing_source",
    "bayesian_transport.prior_data_overlap",
    "bayesian_transport.prior_dimension",
    "bayesian_transport.singular_fit",
    "bayesian_transport.source_dependence",
    "bayesian_transport.support_resource_limit",
    "bayesian_transport.too_many_draws",
    "bayesian_transport.too_many_parameters",
    "bayesian_transport.unsupported_graph",
    "bayesian_transport.weak_overlap",
    "binary_recovery_posterior.band_not_constructed",
    "binary_recovery_posterior.empty_law",
    "binary_recovery_posterior.external_power_prior",
    "binary_recovery_posterior.mapped_prior",
    "binary_recovery_posterior.non_positive_scale",
    "callback_protocol.artifact_invalid",
    "callback_protocol.blank_field",
    "callback_protocol.cancelled",
    "callback_protocol.diagnostic_only",
    "callback_protocol.duplicate_name",
    "callback_protocol.hidden_state_unobservable",
    "callback_protocol.identification_not_conferred",
    "callback_protocol.identity_mismatch",
    "callback_protocol.interval_not_supported",
    "callback_protocol.invalid_quantity",
    "callback_protocol.invalid_support",
    "callback_protocol.invocation_limit",
    "callback_protocol.invocation_outside_execution",
    "callback_protocol.joint_law_not_conferred",
    "callback_protocol.meaning_not_for_kind",
    "callback_protocol.native_trust_claimed",
    "callback_protocol.native_trust_not_conferred",
    "callback_protocol.nested_input_overlap_unknown",
    "callback_protocol.nested_input_unregistered",
    "callback_protocol.nested_inputs_mismatch",
    "callback_protocol.nested_inputs_share_evidence",
    "callback_protocol.no_idempotency_contract",
    "callback_protocol.no_operations",
    "callback_protocol.opaque_dependencies",
    "callback_protocol.operation_not_declared",
    "callback_protocol.operation_not_for_kind",
    "callback_protocol.output_not_a_law",
    "callback_protocol.output_not_finite",
    "callback_protocol.output_outside_support",
    "callback_protocol.output_shape",
    "callback_protocol.output_too_large",
    "callback_protocol.posterior_guarantee_not_supported",
    "callback_protocol.provider_failed",
    "callback_protocol.provider_identity_mismatch",
    "callback_protocol.provider_unavailable",
    "callback_protocol.query_invalid",
    "callback_protocol.replay_contract_incomplete",
    "callback_protocol.request_mismatch",
    "callback_protocol.retry_outside_policy",
    "callback_protocol.side_effects_not_allowed",
    "callback_protocol.too_many_items",
    "callback_protocol.trust_incomplete",
    "callback_protocol.update_declaration_mismatch",
    "callback_protocol.wall_limit_exceeded",
    "categorical_posterior.band_not_constructed",
    "categorical_posterior.external_power_prior",
    "categorical_posterior.mapped_prior",
    "categorical_posterior.non_positive_scale",
    "categorical_treatment.absent_level",
    "categorical_treatment.cancelled",
    "categorical_treatment.data_not_bound",
    "categorical_treatment.export_budget_exceeded",
    "categorical_treatment.invalid_identity",
    "categorical_treatment.invalid_level",
    "categorical_treatment.invalid_min_level_rows",
    "categorical_treatment.invalid_pair",
    "categorical_treatment.invalid_request",
    "categorical_treatment.limits_exceeded",
    "categorical_treatment.memory_budget_exceeded",
    "categorical_treatment.monotonicity_requires_ordered",
    "categorical_treatment.posterior_declared_rows",
    "categorical_treatment.row_count_mismatch",
    "categorical_treatment.sparse_level",
    "categorical_treatment.summary_inconsistent",
    "categorical_treatment.too_few_levels",
    "categorical_treatment.undeclared_level",
    "categorical_treatment.unknown_level",
    "categorical_treatment.unknown_reference",
    "categorical_treatment.wrong_contract",
    "chain_recovery_posterior.band_not_constructed",
    "chain_recovery_posterior.empty_law",
    "chain_recovery_posterior.external_power_prior",
    "chain_recovery_posterior.mapped_prior",
    "chain_recovery_posterior.non_positive_scale",
    "cluster_dml.cancelled",
    "cluster_dml.cluster_ids_missing",
    "cluster_dml.dyadic_giant_component",
    "cluster_dml.dyadic_shared_namespace",
    "cluster_dml.flexible_learner_not_supported",
    "cluster_dml.invalid_min_clusters",
    "cluster_dml.invalid_min_components",
    "cluster_dml.no_interval_method",
    "cluster_dml.penalized_propensity",
    "cluster_dml.scope",
    "cluster_dml.second_ids_missing",
    "cluster_dml.second_ids_unexpected",
    "cluster_dml.too_few_clusters",
    "cluster_dml.too_few_components",
    "clustered_posterior.band_not_constructed",
    "clustered_posterior.empty_scores",
    "clustered_posterior.external_power_prior",
    "clustered_posterior.mapped_prior",
    "clustered_posterior.non_positive_scale",
    "cohort_decision.cancelled",
    "cohort_decision.cohort_changed",
    "cohort_decision.cohort_empty",
    "cohort_decision.cohort_invalid",
    "cohort_decision.cohort_too_large",
    "cohort_decision.container_checksum_mismatch",
    "cohort_decision.container_invalid",
    "cohort_decision.contrast_mismatch",
    "cohort_decision.contrast_not_declared",
    "cohort_decision.extrapolative_not_accepted",
    "cohort_decision.feature_schema_mismatch",
    "cohort_decision.feature_type_mismatch",
    "cohort_decision.fitted_model_absent",
    "cohort_decision.fitted_model_invalid",
    "cohort_decision.identity_unexpected",
    "cohort_decision.interval_level_invalid",
    "cohort_decision.interval_not_supported",
    "cohort_decision.interval_requires_native_fit",
    "cohort_decision.invalid_request",
    "cohort_decision.joint_law_required",
    "cohort_decision.law_not_supplied",
    "cohort_decision.layout_invalid",
    "cohort_decision.link_not_identity",
    "cohort_decision.missing_artifact_id",
    "cohort_decision.model_identity_mismatch",
    "cohort_decision.native_fit_refused",
    "cohort_decision.negative_variance",
    "cohort_decision.non_finite_result",
    "cohort_decision.outcome_units_required",
    "cohort_decision.outside_support",
    "cohort_decision.oversized",
    "cohort_decision.parent_attested_not_native",
    "cohort_decision.parent_not_verified",
    "cohort_decision.parent_query_unsupported",
    "cohort_decision.population_mismatch",
    "cohort_decision.prediction_refused",
    "cohort_decision.replay_mismatch",
    "cohort_decision.shift_declaration_mismatch",
    "cohort_decision.shift_declaration_required",
    "cohort_decision.shift_declaration_unneeded",
    "cohort_decision.shift_support_failure",
    "cohort_decision.source_invalid",
    "cohort_decision.support_contract_absent",
    "cohort_decision.units_mismatch",
    "cohort_decision.unsupported_version",
    "cohort_decision.weights_invalid",
    "cohort_decision.weights_not_normalized",
    "cohort_decision.wrong_parent_identity",
    "compact_export.container_checksum_mismatch",
    "compact_export.container_invalid",
    "compact_export.covariance_asymmetric",
    "compact_export.covariance_not_psd",
    "compact_export.data_digest_mismatch",
    "compact_export.dimension_mismatch",
    "compact_export.duplicate_term",
    "compact_export.empty_mask_region",
    "compact_export.encode",
    "compact_export.identity_mismatch",
    "compact_export.identity_unexpected",
    "compact_export.inputs_not_canonical",
    "compact_export.invalid_degree",
    "compact_export.invalid_mask_region",
    "compact_export.invalid_quantity",
    "compact_export.invalid_query",
    "compact_export.invalid_request",
    "compact_export.invalid_support",
    "compact_export.layout_invalid",
    "compact_export.mask_not_canonical",
    "compact_export.mask_support_mismatch",
    "compact_export.masked_region",
    "compact_export.missing_artifact_id",
    "compact_export.missing_quantity",
    "compact_export.negative_variance",
    "compact_export.no_terms",
    "compact_export.non_finite_input",
    "compact_export.non_finite_result",
    "compact_export.non_finite_value",
    "compact_export.out_of_support",
    "compact_export.oversized",
    "compact_export.premises_digest_mismatch",
    "compact_export.query_type_mismatch",
    "compact_export.scope_mismatch",
    "compact_export.term_support_mismatch",
    "compact_export.terms_not_canonical",
    "compact_export.too_many_inputs",
    "compact_export.too_many_levels",
    "compact_export.too_many_mask_regions",
    "compact_export.too_many_terms",
    "compact_export.unknown_level",
    "compact_export.unknown_quantity",
    "compact_export.unsupported_version",
    "compiled_categorical_posterior.cancelled",
    "compiled_categorical_posterior.candidate",
    "compiled_categorical_posterior.execution_refused",
    "compiled_categorical_posterior.resource_limit",
    "compiled_categorical_posterior.source_invalid",
    "composite_recalc.invalid_request",
    "composite_recalc.native_state_mismatch",
    "composite_recalc.receipt_invalid",
    "composite_recalc.request_unsupported",
    "composition_boundary.artifact_rebuild_failed",
    "composition_boundary.atom_probabilities_invalid",
    "composition_boundary.conflicting_atoms_not_averaged",
    "composition_boundary.coordinate_missing",
    "composition_boundary.coordinate_unsupported",
    "composition_boundary.dependence_route_not_supported",
    "composition_boundary.invalid_input",
    "composition_boundary.mean_is_not_a_distribution",
    "composition_boundary.metadata_only_native_claim",
    "composition_boundary.native_required",
    "composition_boundary.non_affine_needs_joint_law",
    "composition_boundary.operation_not_declared",
    "composition_boundary.outcome_law_needs_joint_law",
    "composition_boundary.paired_draws_across_sources",
    "composition_boundary.shared_evidence_not_independent",
    "composition_boundary.unknown_dependence_is_not_independence",
    "composition_boundary.unknown_input",
    "composition_boundary.unsupported_action_not_comparable",
    "composition_boundary.verification_receipt_missing",
    "composition_bundle.callback_unavailable",
    "composition_bundle.edge_digest_mismatch",
    "composition_bundle.expected_identity_mismatch",
    "composition_bundle.graph_or_snapshot_mismatch",
    "composition_bundle.incompatible_version",
    "composition_bundle.invalid_container",
    "composition_bundle.node_not_found",
    "composition_bundle.oversized",
    "composition_bundle.provider_request_changed",
    "composition_bundle.swapped_evidence",
    "composition_bundle.tampered_quantity",
    "composition_bundle.unknown_node_kind",
    "composition_bundle.unsupported_law",
    "composition_provenance.digest_mismatch",
    "composition_provenance.duplicate_link",
    "composition_provenance.invalid_link",
    "composition_provenance.missing_stage",
    "composition_provenance.unknown_link",
    "composition_provenance.unresolved_parent",
    "composition_verification.artifact_invalid",
    "composition_verification.cancelled",
    "composition_verification.cycle",
    "composition_verification.duplicate_node",
    "composition_verification.empty_unsupported",
    "composition_verification.graph_invalid",
    "composition_verification.identity_mismatch",
    "composition_verification.inherited_without_edge",
    "composition_verification.input_retired",
    "composition_verification.invalid_text",
    "composition_verification.ledger_mismatch",
    "composition_verification.limit_exceeded",
    "composition_verification.provider_trust_withdrawn",
    "composition_verification.recompute_required",
    "composition_verification.replay_mismatch",
    "composition_verification.self_dependency",
    "composition_verification.stage_not_in_plan",
    "composition_verification.stage_unexecutable",
    "composition_verification.support_withdrawn",
    "composition_verification.unknown_claim",
    "composition_verification.unknown_node",
    "composition_verification.unknown_result",
    "conditional_study_ranking.action_not_unique",
    "conditional_study_ranking.artifact_invalid",
    "conditional_study_ranking.invalid_policy",
    "conditional_study_ranking.original_ranking_refused",
    "conditional_study_ranking.resource_limit",
    "conditional_study_ranking.source_mismatch",
    "conditional_study_ranking.source_unavailable",
    "constancy_posterior.band_not_constructed",
    "constancy_posterior.empty_contrast",
    "constancy_posterior.external_power_prior",
    "constancy_posterior.mapped_prior",
    "constancy_posterior.non_positive_scale",
    "coordinate_support.functional_not_described",
    "coordinate_support.invalid_grid",
    "coordinate_support.missing_evidence_relabeled",
    "coordinate_support.point_count_mismatch",
    "coordinate_support.regime_not_describable",
    "coordinate_support.summary_not_worst_label",
    "coordinate_support.temporal_path_not_described",
    "coordinate_support.units_required",
    "coordinate_support.unknown_variable",
    "counterfactual_id.bounds_exceeded",
    "counterfactual_id.budget",
    "counterfactual_id.conflicting_subscripts",
    "counterfactual_id.counterfactual_hedge",
    "counterfactual_id.graph_outside_contract",
    "counterfactual_id.interval_requested",
    "counterfactual_id.invalid_artifact",
    "counterfactual_id.invalid_query",
    "counterfactual_id.path_specific_not_supported",
    "counterfactual_id.positivity_violation",
    "counterfactual_id.query_outside_contract",
    "counterfactual_transport.bounds_exceeded",
    "counterfactual_transport.budget",
    "counterfactual_transport.checked_obstruction",
    "counterfactual_transport.coupling_unsupported",
    "counterfactual_transport.evidence_incompatible",
    "counterfactual_transport.factor_not_derivable",
    "counterfactual_transport.graph_outside_contract",
    "counterfactual_transport.invalid_artifact",
    "counterfactual_transport.invalid_catalog",
    "counterfactual_transport.invalid_derivation",
    "counterfactual_transport.invalid_evidence",
    "counterfactual_transport.invalid_query",
    "counterfactual_transport.invalid_witness",
    "counterfactual_transport.missing_law",
    "counterfactual_transport.missing_regime",
    "counterfactual_transport.no_sampled_inference",
    "counterfactual_transport.selected_sample_law",
    "counterfactual_transport.support_failure",
    "counterfactual_transport.zero_conditioning_mass",
    "counterfactual_transport_sampled.bounds_exceeded",
    "counterfactual_transport_sampled.cancelled",
    "counterfactual_transport_sampled.envelope_budget",
    "counterfactual_transport_sampled.envelope_undefined",
    "counterfactual_transport_sampled.interval_not_computable",
    "counterfactual_transport_sampled.invalid_artifact",
    "counterfactual_transport_sampled.invalid_counts",
    "counterfactual_transport_sampled.invalid_overlap",
    "counterfactual_transport_sampled.missing_table",
    "counterfactual_transport_sampled.overlap_declared_independent",
    "counterfactual_transport_sampled.overlap_unknown",
    "counterfactual_transport_sampled.sample_size_below_limit",
    "counterfactual_transport_sampled.shared_units_unlinked",
    "counterfactual_transport_sampled.sparsity_above_limit",
    "counterfactual_transport_sampled.table_not_cited",
    "counterfactual_transport_sampled.unseen_cell",
    "cpdag_scenarios.cancelled_not_exportable",
    "cpdag_scenarios.consumer_limit_exceeded",
    "cpdag_scenarios.data_identity_mismatch",
    "cpdag_scenarios.evidence_identity_mismatch",
    "cpdag_scenarios.exact_laws_only",
    "cpdag_scenarios.no_evidence_binding",
    "cpdag_scenarios.not_a_cpdag",
    "cpdag_scenarios.premises_mismatch",
    "cpdag_scenarios.provider_mismatch",
    "cpdag_scenarios.report_replay_mismatch",
    "cpdag_scenarios.selection_or_latent",
    "cpdag_scenarios.unsupported_semantics",
    "cross_world.graph_outside_contract",
    "cross_world.interval_requested",
    "cross_world.invalid_query",
    "cross_world.query_outside_contract",
    "cross_world.recanting_witness",
    "cross_world_execution.artifact_changed",
    "cross_world_execution.bounds_exceeded",
    "cross_world_execution.cancelled",
    "cross_world_execution.cell_not_in_matrix",
    "cross_world_execution.elimination_bound_exceeded",
    "cross_world_execution.envelope_member_structure_differs",
    "cross_world_execution.envelope_needs_parameter_set",
    "cross_world_execution.envelope_parameter_set_empty",
    "cross_world_execution.envelope_reading_required",
    "cross_world_execution.evidence_outside_model_support",
    "cross_world_execution.evidence_probability_underflow",
    "cross_world_execution.finite_fitting_not_executed",
    "cross_world_execution.incompatible_history",
    "cross_world_execution.invalid_artifact",
    "cross_world_execution.invalid_evidence",
    "cross_world_execution.invalid_identity",
    "cross_world_execution.invalid_quantity",
    "cross_world_execution.invalid_request",
    "cross_world_execution.invalid_scm",
    "cross_world_execution.invalid_world",
    "cross_world_execution.joint_state_bound_exceeded",
    "cross_world_execution.latent_confounding_not_identified",
    "cross_world_execution.mechanisms_not_declared",
    "cross_world_execution.no_interval_method",
    "cross_world_execution.no_posterior_method",
    "cross_world_execution.no_transported_row_derivation",
    "cross_world_execution.posterior_weight_underflow",
    "cross_world_execution.sampled_abduction_not_supported",
    "cross_world_execution.soft_intervention",
    "cross_world_execution.spread_needs_enumeration",
    "cross_world_uncertainty.account_mismatch",
    "cross_world_uncertainty.bounds_exceeded",
    "cross_world_uncertainty.incompatible_history",
    "cross_world_uncertainty.no_transported_path_derivation",
    "cross_world_uncertainty.result_mismatch",
    "cross_world_uncertainty.sampled_path_not_supported",
    "cross_world_uncertainty.unit_sampling_undeclared",
    "decision_adapters.bayes_over_identified_set",
    "decision_adapters.claim_draws_too_large",
    "decision_adapters.claims_sources_mismatch",
    "decision_adapters.completion_count_not_probability",
    "decision_adapters.criterion_not_supported_for_set",
    "decision_adapters.duplicate_action",
    "decision_adapters.evaluated_needs_draws",
    "decision_adapters.graph_claims_carry_no_probability",
    "decision_adapters.invalid_claims",
    "decision_adapters.invalid_interval",
    "decision_adapters.invalid_probability",
    "decision_adapters.invalid_utilities",
    "decision_adapters.missing_action",
    "decision_adapters.point_claim_is_single",
    "decision_adapters.policy_mismatch",
    "decision_adapters.probabilities_required",
    "decision_adapters.unevaluated_needs_reason",
    "decision_adapters.unknown_action",
    "decision_adapters.unknown_claim_kind",
    "decision_adapters.unknown_claim_status",
    "decision_adapters.unknown_support_status",
    "decision_admissibility.duplicate_rule",
    "decision_admissibility.invalid_rule",
    "decision_admissibility.unknown_action",
    "decision_admissibility.unknown_input",
    "decision_claims.identity_mismatch",
    "decision_claims.missing_scenario_utility",
    "decision_claims.probabilities_not_declared",
    "decision_claims.range_unavailable",
    "decision_claims.unresolved_scenarios",
    "decision_claims.unsupported_claim_shape",
    "decision_contract.duplicate_id",
    "decision_contract.invalid_declaration",
    "decision_contract.invalid_parameter",
    "decision_contract.invalid_quantity",
    "decision_contract.missing_source",
    "decision_contract.outside_decision_scope",
    "decision_contract.unknown_action",
    "decision_contract.unknown_input",
    "decision_evaluation.analytic_law_unsupported",
    "decision_evaluation.distribution_meaning",
    "decision_evaluation.joint_law_required",
    "decision_evaluation.mean_source_insufficient",
    "decision_evaluation.mean_source_not_replayable",
    "decision_evaluation.non_finite_utility",
    "decision_evaluation.quantity_not_found",
    "decision_evaluation.structure_inputs_required",
    "decision_evaluation.unsupported_coordinate",
    "decision_evidence.adjustment_not_admissible",
    "decision_evidence.cancelled",
    "decision_evidence.dependence_unknown",
    "decision_evidence.deployment_bridge_missing",
    "decision_evidence.frame_shape",
    "decision_evidence.incompatible_horizon",
    "decision_evidence.incompatible_intervention",
    "decision_evidence.incompatible_population",
    "decision_evidence.interference_unspecified",
    "decision_evidence.invalid_declaration",
    "decision_evidence.resource_limit",
    "decision_evidence.too_many_unknowns",
    "decision_evidence.unsupported_assignment_design",
    "decision_joint_scenarios.carrier",
    "decision_posterior.band_not_constructed",
    "decision_posterior.empty_law",
    "decision_posterior.external_power_prior",
    "decision_posterior.mapped_prior",
    "decision_posterior.non_positive_scale",
    "decision_reliability.artifact",
    "decision_reliability.artifact_id_missing",
    "decision_reliability.artifact_kind_mismatch",
    "decision_reliability.cancelled",
    "decision_reliability.candidate_set_mismatch",
    "decision_reliability.combined_uncertainty",
    "decision_reliability.decision_mismatch",
    "decision_reliability.dependence_unknown",
    "decision_reliability.draws_digest_mismatch",
    "decision_reliability.effective_sample_size_too_small",
    "decision_reliability.identity_mismatch",
    "decision_reliability.invalid_input",
    "decision_reliability.joint_law_required",
    "decision_reliability.memory_budget_exceeded",
    "decision_reliability.numerical",
    "decision_reliability.producer_mean_disagrees",
    "decision_reliability.producer_point_only",
    "decision_reliability.report_mismatch",
    "decision_reliability.route_not_supported",
    "decision_reliability.search_record_inconsistent",
    "decision_reliability.search_record_required",
    "decision_reliability.selection_stream_mismatch",
    "decision_reliability.utility_unit_mismatch",
    "decision_reliability.validation_not_independent",
    "decision_reliability.validation_stream_mismatch",
    "decision_robustness.contract_mismatch",
    "decision_robustness.invalid_profile",
    "decision_structural.invalid_atoms",
    "decision_structural.invalid_probabilities",
    "decision_structural.probabilities_required",
    "descriptive_comparison.admission_refused",
    "descriptive_comparison.arm_not_populated",
    "descriptive_comparison.arm_too_small",
    "descriptive_comparison.cancelled",
    "descriptive_comparison.column_attribution_not_identified",
    "descriptive_comparison.estimand_coding_mismatch",
    "descriptive_comparison.fold_exceeds_lease",
    "descriptive_comparison.gap_interval_unavailable",
    "descriptive_comparison.invalid_covariance",
    "descriptive_comparison.invalid_data",
    "descriptive_comparison.joint_covariance_unavailable",
    "descriptive_comparison.learner_exceeds_lease",
    "descriptive_comparison.population_not_all_observed",
    "descriptive_comparison.scale_not_supported",
    "descriptive_comparison.transform_domain",
    "design_ranking.bounds_exceeded",
    "design_ranking.contract_mismatch",
    "design_ranking.cost_units_mismatch",
    "design_ranking.design_error",
    "design_ranking.digest_mismatch",
    "design_ranking.duplicate_candidate",
    "design_ranking.empty",
    "design_ranking.identity_mismatch",
    "design_ranking.invalid_artifact",
    "design_ranking.invalid_request",
    "design_ranking.rank_mismatch",
    "design_ranking.replay_mismatch",
    "design_ranking.signal_inconsistent",
    "design_ranking.signal_mismatch",
    "design_ranking.source_digest_mismatch",
    "design_ranking.source_overlap",
    "design_ranking.source_overlap_unknown",
    "design_ranking.unsupported_version",
    "discovered_transport.class_enumeration_partial",
    "discovered_transport.class_not_identified",
    "discovered_transport.class_too_large",
    "discovered_transport.evidence_invalid",
    "discovered_transport.member_bound_zero",
    "discovered_transport.member_not_estimable",
    "discovered_transport.no_members",
    "discovered_transport.orientation_conflict",
    "discovered_transport.pag_class_ambiguous",
    "discovered_transport.pag_selection_bias_edge",
    "discovered_transport.search_incomplete",
    "discovered_transport.selection_target_unknown",
    "discovered_transport.temporal_query_invalid",
    "discovered_transport.temporal_structure_unsupported",
    "discovered_transport.trial_compatibility_invalid",
    "discovered_transport.trial_rejects_every_member",
    "discretized_scm.cancelled",
    "discretized_scm.composition_incoherent",
    "discretized_scm.degenerate_marginal",
    "discretized_scm.intervention_invalid",
    "discretized_scm.latent_confounding",
    "discretized_scm.levels_out_of_range",
    "discretized_scm.levels_undeclared",
    "discretized_scm.non_gaussian_mechanism",
    "discretized_scm.numerical",
    "discretized_scm.point_only",
    "discretized_scm.state_space_too_large",
    "discretized_scm.support_not_real_line",
    "discretized_scm.unknown_variable",
    "discretized_scm_artifact.claimed_numbers_differ",
    "discretized_scm_artifact.identity_mismatch",
    "discretized_scm_artifact.malformed",
    "discretized_scm_artifact.not_exportable",
    "discretized_scm_artifact.rederivation_refused",
    "discretized_scm_artifact.version_unsupported",
    "distribution_meaning.incompatible_operation",
    "distribution_meaning.legacy_posterior",
    "distribution_meaning.observational_not_interventional",
    "distribution_meaning.posterior_interval",
    "dml.arm_not_populated",
    "dml.cancelled",
    "dose_grid.bandwidth_outside_range",
    "dose_grid.cancelled",
    "dose_grid.consumer_limit_exceeded",
    "dose_grid.data_identity_mismatch",
    "dose_grid.derivative_without_claim",
    "dose_grid.graph_not_certified",
    "dose_grid.insufficient_local_weight",
    "dose_grid.invalid_lag",
    "dose_grid.invalid_request",
    "dose_grid.level_without_claim",
    "dose_grid.no_simultaneous_band_method",
    "dose_grid.non_finite_result",
    "dose_grid.premises_mismatch",
    "dose_grid.refused",
    "dose_grid.resource_limit",
    "dose_grid.resource_refused",
    "dose_grid.result_replay_mismatch",
    "dose_grid.too_few_clusters",
    "dose_grid.undecodable",
    "dose_grid.unsupported_dose",
    "dose_grid.unsupported_semantics",
    "dose_grid_posterior.band_not_constructed",
    "dose_grid_posterior.external_power_prior",
    "dose_grid_posterior.insufficient_weight",
    "dose_grid_posterior.mapped_prior",
    "dose_grid_posterior.non_positive_scale",
    "dose_response.bootstrap_below_floor",
    "dose_response.bounds_exceeded",
    "dose_response.cate_or_simultaneous",
    "dose_response.declared_folds_unbalanced",
    "dose_response.dose_density_floor",
    "dose_response.dose_density_invalid",
    "dose_response.dose_too_discrete",
    "dose_response.estimated_dose_density",
    "dose_response.grid_outside_dose_support",
    "dose_response.invalid_declared_rows",
    "dose_response.kernel_not_supported",
    "dose_response.learner_not_portable",
    "dose_response.local_dose_ess",
    "dose_response.membership_overlap",
    "dose_response.non_iid_design",
    "dose_response.posterior_declared_rows",
    "dose_response.quadrature_tolerance",
    "dose_response.target_not_smoothed",
    "effect_constancy.estimand_missing",
    "effect_constancy.incompatible_partitions",
    "effect_constancy.invalid_alpha",
    "effect_constancy.invalid_covariance",
    "effect_constancy.invalid_identity",
    "effect_constancy.invalid_request",
    "effect_constancy.invalid_standard_error",
    "effect_constancy.limits_exceeded",
    "effect_constancy.non_finite_estimate",
    "effect_constancy.too_few_partitions",
    "effect_constancy.too_many_partitions",
    "effect_constancy.unknown_reference",
    "effect_constancy.unsupported_partition",
    "effect_constancy.wrong_contract",
    "effect_constancy_consumer.artifact_invalid",
    "effect_constancy_consumer.catalog_binding_invalid",
    "effect_constancy_consumer.contrast_missing",
    "effect_constancy_consumer.decision_unsupported",
    "effect_constancy_consumer.invalid_request",
    "effect_constancy_consumer.limits_exceeded",
    "effect_constancy_consumer.partition_missing",
    "effect_constancy_consumer.quantity_mismatch",
    "effect_constancy_consumer.support_unsupported",
    "evidence_ledger.cancelled",
    "evidence_ledger.diagonal_covariance_shared_rows",
    "evidence_ledger.duplicate_result",
    "evidence_ledger.duplicate_row",
    "evidence_ledger.duplicate_source",
    "evidence_ledger.identity_mismatch",
    "evidence_ledger.interval_construction_unavailable",
    "evidence_ledger.interval_report_invalid",
    "evidence_ledger.invalid_declaration",
    "evidence_ledger.kind_mismatch",
    "evidence_ledger.limit_exceeded",
    "evidence_ledger.lineage_rows_mismatch",
    "evidence_ledger.native_and_transported",
    "evidence_ledger.operation_arity",
    "evidence_ledger.overlap_unknown",
    "evidence_ledger.reentry_derived_from_used_data",
    "evidence_ledger.reentry_duplicate",
    "evidence_ledger.reentry_prior_plus_likelihood",
    "evidence_ledger.reentry_transform",
    "evidence_ledger.repeated_source",
    "evidence_ledger.replay_mismatch",
    "evidence_ledger.result_not_valid",
    "evidence_ledger.shared_observations",
    "evidence_ledger.source_superseded",
    "evidence_ledger.source_withdrawn",
    "evidence_ledger.supersede_kind_mismatch",
    "evidence_ledger.synthesis_mismatch",
    "evidence_ledger.transport_identity_missing",
    "evidence_ledger.transported_as_native",
    "evidence_ledger.unknown_result",
    "evidence_ledger.unknown_source",
    "evidence_ledger.update_drops_rows",
    "evidence_obligations.invalid_obligation",
    "evidence_obligations.wrong_contract",
    "evidence_plan.budget",
    "evidence_plan.compile_refused",
    "evidence_plan.cost_units_mismatch",
    "evidence_plan.invalid_request",
    "evidence_plan.no_failure_to_plan",
    "evidence_plan.no_route",
    "evidence_plan.none_certified",
    "evsi.action_set_changed",
    "evsi.bound_violation",
    "evsi.cancelled",
    "evsi.cost_map_required",
    "evsi.cost_units_mismatch",
    "evsi.decision_values_incoherent",
    "evsi.decision_values_need_draws",
    "evsi.design_error",
    "evsi.duplicate_candidate",
    "evsi.execution_resource_limit",
    "evsi.invalid_cost",
    "evsi.invalid_request",
    "evsi.prior_states_mismatch",
    "evsi.receipt_mismatch",
    "evsi.sample_size_mismatch",
    "evsi.source_overlap",
    "evsi.source_overlap_unknown",
    "evsi.unscored_candidate",
    "external_mechanism.declaration_invalid",
    "external_mechanism.implementation_changed",
    "external_mechanism.implementation_unavailable",
    "external_mechanism.resolver_failed",
    "external_mechanism.resolver_missing",
    "external_object_verification.duplicate_probe",
    "external_object_verification.failed_probe",
    "external_object_verification.invalid_probe",
    "external_object_verification.missing_probe",
    "external_recalc.artifact_invalid",
    "external_recalc.attempt_limit",
    "external_recalc.callback_failed",
    "external_recalc.contract_mismatch",
    "external_recalc.idempotency_conflict",
    "external_recalc.idempotency_unsupported",
    "external_recalc.invalid_descriptor",
    "external_recalc.invalid_output",
    "external_recalc.invalid_request",
    "external_recalc.limits_exceeded",
    "external_recalc.output_unavailable",
    "external_recalc.policy_unsupported",
    "external_recalc.provider_mismatch",
    "external_recalc.provider_unavailable",
    "external_recalc.quantity_unsupported",
    "external_recalc.replay_mismatch",
    "external_recalc.replay_unsupported",
    "external_recalc.retry_unsafe",
    "external_response_binding.column_quantity",
    "external_response_binding.column_support_length",
    "external_response_binding.contract_not_identified",
    "external_response_binding.coordinate_conditioning",
    "external_response_binding.coordinate_functional",
    "external_response_binding.coordinate_horizon",
    "external_response_binding.coordinate_invalid_coordinate",
    "external_response_binding.coordinate_population",
    "external_response_binding.coordinate_regime",
    "external_response_binding.coordinate_transform",
    "external_response_binding.coordinate_units",
    "external_response_binding.coordinate_variable",
    "external_response_binding.dimension",
    "external_response_binding.distribution_meaning",
    "external_response_binding.empty_column",
    "external_response_binding.graph",
    "external_response_binding.invalid_contract",
    "external_response_binding.malformed_artifact",
    "external_response_binding.malformed_capability",
    "external_response_binding.malformed_equivalence",
    "external_response_binding.malformed_identification",
    "external_response_binding.malformed_meaning",
    "external_response_binding.malformed_probe",
    "external_response_binding.malformed_quantity",
    "external_response_binding.malformed_support",
    "external_response_binding.malformed_trust",
    "external_response_binding.missing_assumption",
    "external_response_binding.missing_evidence",
    "external_response_binding.non_finite_value",
    "external_response_binding.object_kind",
    "external_response_binding.object_quantities",
    "external_response_binding.posterior_kind",
    "external_response_binding.unchecked_observational_law",
    "external_scientific_providers.duplicate_declaration",
    "external_scientific_providers.invalid_declaration",
    "external_scientific_providers.invalid_identity",
    "external_trust_receipt.receipt_mismatch",
    "finite_action_posterior.band_not_constructed",
    "finite_action_posterior.empty_law",
    "finite_action_posterior.external_power_prior",
    "finite_action_posterior.mapped_prior",
    "finite_action_posterior.non_positive_scale",
    "frozen_scores.identity_mismatch",
    "functional_inverse_posterior.band_not_constructed",
    "functional_inverse_posterior.empty_law",
    "functional_inverse_posterior.external_power_prior",
    "functional_inverse_posterior.mapped_prior",
    "functional_inverse_posterior.non_positive_scale",
    "functional_inverse_query.artifact_kind_mismatch",
    "functional_inverse_query.bounds_exceeded",
    "functional_inverse_query.forward_origin_invalid",
    "functional_inverse_query.global_feasibility_claim",
    "functional_inverse_query.invalid_artifact",
    "functional_inverse_query.wrong_contract",
    "functional_source.expected_identity_mismatch",
    "functional_source.invalid_artifact",
    "functional_source.invalid_contract",
    "functional_source.invalid_source",
    "functional_source.limits_exceeded",
    "functional_source.missing_action",
    "functional_source.native_source_unavailable",
    "functional_source.original_law_required",
    "functional_source.output_mismatch",
    "functional_source.source_mismatch",
    "graph_falsification.ci_p_value_missing",
    "graph_falsification.ci_result_count",
    "graph_falsification.ci_test_not_frequentist",
    "graph_falsification.data_not_observed",
    "graph_falsification.duplicate_variable",
    "graph_falsification.empty_data",
    "graph_falsification.invalid_level",
    "graph_falsification.invalid_permutations",
    "graph_falsification.node_not_static",
    "graph_falsification.too_few_nodes",
    "graph_falsification.work_limit",
    "graph_imputation.atom_failure",
    "graph_imputation.atom_not_lowered",
    "graph_imputation.cancelled",
    "graph_imputation.candidate_refused",
    "graph_imputation.compiled",
    "graph_imputation.execution_refused",
    "graph_imputation.interval_unsupported",
    "graph_imputation.invalid_source",
    "graph_imputation.lowering",
    "graph_imputation.point_unsupported",
    "graph_imputation.resource_limit",
    "graph_imputation.search_stopped",
    "graph_imputation.supported_graph_unidentified",
    "graph_imputation.unidentified",
    "graph_imputation.unscorable",
    "graph_imputation.unsupported",
    "graph_imputation.wrong_dispatch",
    "graph_selection_imputation.columns",
    "graph_selection_imputation.data",
    "graph_selection_imputation.domain",
    "graph_selection_imputation.interval_not_measured",
    "graph_selection_imputation.model",
    "graph_selection_imputation.result",
    "graph_selection_imputation.variables",
    "identification_repair.bounds_exceeded",
    "identification_repair.budget",
    "identification_repair.cost_units_mismatch",
    "identification_repair.invalid_request",
    "identification_repair.not_a_failure",
    "identification_repair.wrong_contract",
    "initial_state.invalid_law",
    "initial_state.invalid_premises",
    "initial_state.support_gap",
    "initial_state.target_law_missing",
    "initial_state.too_many_states",
    "initial_state_posterior.band_not_constructed",
    "initial_state_posterior.empty_law",
    "initial_state_posterior.external_power_prior",
    "initial_state_posterior.mapped_prior",
    "initial_state_posterior.non_positive_scale",
    "interface_contract.artifact_invalid",
    "interface_contract.blank_field",
    "interface_contract.calibration_record_blank",
    "interface_contract.calibration_without_interval",
    "interface_contract.conditioning_mismatch",
    "interface_contract.cyclic_composition",
    "interface_contract.dependence_unknown",
    "interface_contract.duplicate_component",
    "interface_contract.duplicate_port",
    "interface_contract.functional_mismatch",
    "interface_contract.horizon_mismatch",
    "interface_contract.identity_mismatch",
    "interface_contract.inherited_input_unknown",
    "interface_contract.invalid_digest",
    "interface_contract.invalid_quantity",
    "interface_contract.invalid_support",
    "interface_contract.measurement_assumption_mismatch",
    "interface_contract.missing_operations",
    "interface_contract.operation_unavailable",
    "interface_contract.point_only_with_uncertainty",
    "interface_contract.population_mismatch",
    "interface_contract.port_multiply_fed",
    "interface_contract.posterior_without_posterior_meaning",
    "interface_contract.provenance_missing",
    "interface_contract.regime_mismatch",
    "interface_contract.semantic_identity_mismatch",
    "interface_contract.shared_evidence",
    "interface_contract.support_mismatch",
    "interface_contract.transform_mismatch",
    "interface_contract.uncertainty_meaning_mismatch",
    "interface_contract.uncertainty_meaning_missing",
    "interface_contract.uncertainty_meaning_required",
    "interface_contract.uncertainty_meaning_unused",
    "interface_contract.uncertainty_paradigm_mismatch",
    "interface_contract.units_mismatch",
    "interface_contract.unknown_component",
    "interface_contract.unknown_port",
    "interface_contract.unknown_provenance_stage",
    "interface_contract.variable_mismatch",
    "intrinsic_influence.correlated_noise",
    "intrinsic_influence.draw_budget_exceeded",
    "intrinsic_influence.invalid_settings",
    "intrinsic_influence.mechanism_unfitted",
    "intrinsic_influence.noise_independence_undeclared",
    "intrinsic_influence.noise_not_row_independent",
    "intrinsic_influence.structure_mismatch",
    "intrinsic_influence.target_not_continuous",
    "inverse.cancelled",
    "inverse.forward_not_mean_response",
    "inverse.forward_not_point_identified",
    "inverse.forward_not_supported",
    "inverse.invalid_action",
    "inverse.invalid_forward",
    "inverse.invalid_target",
    "inverse.observational_scenarios",
    "inverse.probability_target",
    "inverse.quantile_target",
    "inverse_forward_source.composed_coordinate_is_outcome_law",
    "inverse_forward_source.composed_family_mismatch",
    "inverse_forward_source.composed_law_not_materialized",
    "inverse_forward_source.composed_level_coding",
    "inverse_forward_source.composed_query_result_mismatch",
    "inverse_forward_source.composed_response_invalid",
    "inverse_forward_source.interval_not_supported",
    "inverse_forward_source.origin_invalid",
    "inverse_forward_source.per_action_not_sealable",
    "inverse_forward_source.point_only_law_requested",
    "inverse_forward_source.transported_not_accepted",
    "inverse_observational_scenarios.condition_on_outcome",
    "inverse_observational_scenarios.coordinate_missing",
    "inverse_observational_scenarios.coordinate_unsupported",
    "inverse_observational_scenarios.event_has_no_mass",
    "inverse_observational_scenarios.invalid_parameter",
    "inverse_observational_scenarios.invalid_scenarios",
    "inverse_observational_scenarios.joint_law_required",
    "inverse_observational_scenarios.meaning_not_observational",
    "inverse_observational_scenarios.non_finite_draw",
    "inverse_observational_scenarios.result_mismatch",
    "inverse_observational_scenarios.unknown_scenario",
    "joint_cells.cancelled",
    "joint_cells.cell_empty",
    "joint_cells.cell_ess_below_minimum",
    "joint_cells.columns",
    "joint_cells.component_count",
    "joint_cells.conditional_unsupported",
    "joint_cells.config",
    "joint_cells.contrast_cell_unsupported",
    "joint_cells.derived_constituent_copy_in_adjustment",
    "joint_cells.derived_constituent_in_adjustment",
    "joint_cells.derived_declaration_invalid",
    "joint_cells.derived_descendant_in_adjustment",
    "joint_cells.derived_duplicate_constituents",
    "joint_cells.derived_exclusion_not_in_adjustment",
    "joint_cells.derived_exclusion_rule_mismatch",
    "joint_cells.derived_illegal_value",
    "joint_cells.derived_intervention_not_defined",
    "joint_cells.derived_not_a_joint_cell",
    "joint_cells.derived_outcome_in_adjustment",
    "joint_cells.derived_rank_deficient",
    "joint_cells.derived_treatment_tracked_by_covariates",
    "joint_cells.derived_unknown_column",
    "joint_cells.learner",
    "joint_cells.learner_unavailable",
    "joint_cells.ml_learner_not_declared",
    "joint_cells.no_interval_method",
    "joint_cells.no_ml_interval_method",
    "joint_cells.no_rows",
    "joint_cells.no_supported_cell",
    "joint_cells.normalization_failed",
    "joint_cells.not_binary",
    "joint_cells.nuisance_provider",
    "joint_cells.ordering",
    "joint_cells.outcome_model_unsupported",
    "joint_law_propagation.alignment_mismatch",
    "joint_law_propagation.artifact_invalid",
    "joint_law_propagation.artifact_refused",
    "joint_law_propagation.bounds_exceeded",
    "joint_law_propagation.cancelled",
    "joint_law_propagation.covariance_insufficient_for_nonlinear",
    "joint_law_propagation.coverage_record_construction_mismatch",
    "joint_law_propagation.coverage_record_coordinate_mismatch",
    "joint_law_propagation.dependence_unknown",
    "joint_law_propagation.envelope_is_not_a_point",
    "joint_law_propagation.envelope_not_available",
    "joint_law_propagation.evidence_overlap_unknown",
    "joint_law_propagation.horizon_composition",
    "joint_law_propagation.horizon_draws_dose",
    "joint_law_propagation.horizon_draws_layout",
    "joint_law_propagation.horizon_draws_mean_curve",
    "joint_law_propagation.horizon_draws_not_joint",
    "joint_law_propagation.index_pairing_refused",
    "joint_law_propagation.invalid_request",
    "joint_law_propagation.mixture_weights_unknown",
    "joint_law_propagation.no_interval_construction",
    "joint_law_propagation.no_posterior_for_fixed_law_inputs",
    "joint_law_propagation.nonlinear_needs_aligned_joint_law",
    "joint_law_propagation.shared_evidence_declared_independent",
    "joint_law_propagation.spread_needs_supported_dependence",
    "joint_law_propagation.structure_not_verified",
    "joint_law_propagation.transport_pieces_incomplete",
    "joint_law_propagation.transported_dependence_required",
    "joint_sensitivity.bounds_exceeded",
    "joint_sensitivity.budget",
    "joint_sensitivity.budget_coupling",
    "joint_sensitivity.cancelled",
    "joint_sensitivity.consumer_limits",
    "joint_sensitivity.data_identity_mismatch",
    "joint_sensitivity.duplicate_factor",
    "joint_sensitivity.factor_count",
    "joint_sensitivity.fixed_graph_conditional",
    "joint_sensitivity.fixed_graph_parent",
    "joint_sensitivity.formula_incompatible",
    "joint_sensitivity.incomplete_kernel",
    "joint_sensitivity.invalid_fraction",
    "joint_sensitivity.invalid_threshold",
    "joint_sensitivity.invalid_tolerance",
    "joint_sensitivity.no_interval_method",
    "joint_sensitivity.premises_mismatch",
    "joint_sensitivity.provider_mismatch",
    "joint_sensitivity.replay_mismatch",
    "joint_sensitivity.source_target_discrepancy",
    "joint_sensitivity.stage_accounting",
    "joint_sensitivity.treatment_factor",
    "joint_sensitivity.union_labelled_ci",
    "joint_sensitivity.unknown_factor",
    "joint_sensitivity.unsupported_domain",
    "latent_class.cancelled",
    "latent_class.constant_outcome",
    "latent_class.data_required",
    "latent_class.degenerate_class",
    "latent_class.design_support_too_small",
    "latent_class.export_budget_exceeded",
    "latent_class.invalid_config",
    "latent_class.invalid_request",
    "latent_class.length_mismatch",
    "latent_class.likelihood_not_monotone",
    "latent_class.limits_exceeded",
    "latent_class.memory_budget_exceeded",
    "latent_class.non_finite_input",
    "latent_class.not_converged",
    "latent_class.randomization_not_declared",
    "latent_class.report_replay_mismatch",
    "latent_class.too_few_rows",
    "latent_class.treatment_constant",
    "latent_class.weak_class",
    "latent_class.wrong_contract",
    "latent_class_posterior.band_not_constructed",
    "latent_class_posterior.empty_design",
    "latent_class_posterior.external_power_prior",
    "latent_class_posterior.mapped_prior",
    "latent_class_posterior.non_positive_scale",
    "learned_continuous_posterior.band_not_constructed",
    "learned_continuous_posterior.empty_scores",
    "learned_continuous_posterior.external_power_prior",
    "learned_continuous_posterior.mapped_prior",
    "learned_continuous_posterior.non_positive_scale",
    "learned_joint_transport.admission_refused",
    "learned_joint_transport.cancelled",
    "learned_joint_transport.feature_mismatch",
    "learned_joint_transport.fit_failed",
    "learned_joint_transport.identification_not_certified",
    "learned_joint_transport.identification_required",
    "learned_joint_transport.invalid_basis",
    "learned_joint_transport.invalid_input",
    "learned_joint_transport.invalid_prior",
    "learned_joint_transport.missing_law",
    "learned_joint_transport.missing_source",
    "learned_joint_transport.prior_data_overlap",
    "learned_joint_transport.prior_dimension",
    "learned_joint_transport.rank_deficient",
    "learned_joint_transport.source_dependence",
    "learned_joint_transport.support_resource_limit",
    "learned_joint_transport.too_many_draws",
    "learned_joint_transport.too_many_parameters",
    "learned_joint_transport.unsupported_graph",
    "learned_joint_transport.weak_overlap",
    "learned_transport.bootstrap_below_floor",
    "learned_transport.bootstrap_not_supported",
    "learned_transport.cate_requested",
    "learned_transport.membership_overlap",
    "learned_transport.non_iid_design",
    "learned_transport.treatment_overlap",
    "mapped_posterior_transfer.coefficient_map",
    "mapped_posterior_transfer.parameter_posterior",
    "mapped_posterior_transfer.residual_variance",
    "matched_case_control.absolute_risk_not_identified",
    "matched_case_control.budget",
    "matched_case_control.estimand_not_supported",
    "matched_case_control.estimate_not_finite",
    "matched_case_control.invalid_data",
    "matched_case_control.invalid_level",
    "matched_case_control.no_informative_sets",
    "matched_case_control.sampling_design",
    "measured_inference.artifact_mismatch",
    "measured_inference.bounds_exceeded",
    "measured_inference.consumer_bounds_exceeded",
    "measured_inference.invalid_expectation",
    "measured_inference.invalid_report",
    "measured_inference.memory_budget_exceeded",
    "measured_inference.source_report_not_supported",
    "measured_inference.unsupported_route",
    "measured_interval.draws_not_converged",
    "measured_interval.invalid_source",
    "measured_interval.no_interval_of_this_kind",
    "measured_interval.source_mismatch",
    "measured_interval.variance_assumption_not_met",
    "measured_transport.input_bounds_exceeded",
    "measured_transport.sampler_mismatch",
    "measurement_model.admission_refused",
    "measurement_model.arm_mismatch",
    "measurement_model.artifact_invalid",
    "measurement_model.artifact_refused",
    "measurement_model.binary_outcome_continuous_error",
    "measurement_model.blank_field",
    "measurement_model.bootstrap_unstable",
    "measurement_model.cancelled",
    "measurement_model.confusion_ill_conditioned",
    "measurement_model.construct_mismatch",
    "measurement_model.continuous_export",
    "measurement_model.design_singular",
    "measurement_model.differential_axis_unsupported",
    "measurement_model.differential_declared",
    "measurement_model.differential_population_unestablished",
    "measurement_model.differential_undeclared",
    "measurement_model.error_variance_not_identified",
    "measurement_model.indicator_as_construct",
    "measurement_model.indicator_kind",
    "measurement_model.indicator_mismatch",
    "measurement_model.indicator_observed",
    "measurement_model.input_invalid",
    "measurement_model.insufficient_draws",
    "measurement_model.insufficient_replicates",
    "measurement_model.interval_level",
    "measurement_model.interval_requires_estimate",
    "measurement_model.intervention_meaning_not_carried",
    "measurement_model.invalid_model",
    "measurement_model.invariance_not_declared",
    "measurement_model.invariance_premise_required",
    "measurement_model.invariance_source_unknown",
    "measurement_model.invariance_target_mismatch",
    "measurement_model.learned_coordinates_not_intervenable",
    "measurement_model.mapping_endpoints",
    "measurement_model.no_model",
    "measurement_model.nothing_to_transport",
    "measurement_model.operation_not_supported",
    "measurement_model.outcome_model_not_fitted",
    "measurement_model.parameter",
    "measurement_model.population_mismatch",
    "measurement_model.posterior_bound_to_snapshot",
    "measurement_model.posterior_capability",
    "measurement_model.posterior_declaration",
    "measurement_model.posterior_identity",
    "measurement_model.posterior_kind",
    "measurement_model.posterior_native_trust",
    "measurement_model.posterior_quantity",
    "measurement_model.posterior_receipt",
    "measurement_model.proxy_as_construct",
    "measurement_model.regime_mismatch",
    "measurement_model.reliability_too_small",
    "measurement_model.representation_unmapped",
    "measurement_model.route_mismatch",
    "measurement_model.route_parameters",
    "measurement_model.structure_parameters",
    "measurement_model.transport_receipt_mismatch",
    "measurement_model.unknown_level",
    "measurement_model.unmodelled_proxy",
    "measurement_model.unrecognized",
    "measurement_model.variance_not_positive",
    "measurement_model.youden_too_small",
    "mechanism_composition.artifact_invalid",
    "mechanism_composition.artifact_kind_mismatch",
    "mechanism_composition.cancelled",
    "mechanism_composition.conditioning_extra_conditioner",
    "mechanism_composition.conditioning_parent_not_conditioned",
    "mechanism_composition.cycle_directed_cycle",
    "mechanism_composition.family_mixed",
    "mechanism_composition.identity_declaration_unknown_variable",
    "mechanism_composition.identity_duplicate_mechanism",
    "mechanism_composition.identity_edge_unknown_variable",
    "mechanism_composition.identity_parent_without_mechanism",
    "mechanism_composition.interval_invalid",
    "mechanism_composition.latent_family_unsupported",
    "mechanism_composition.latent_loading_undeclared",
    "mechanism_composition.latent_shared_with_absent_mechanism",
    "mechanism_composition.latent_sharing_not_declared",
    "mechanism_composition.latent_sharing_not_reciprocal",
    "mechanism_composition.latent_too_many",
    "mechanism_composition.numbers_unbound",
    "mechanism_composition.numerical_failure",
    "mechanism_composition.parents_declared_graph_mismatch",
    "mechanism_composition.population_invariance_unstated",
    "mechanism_composition.population_latent_blocks_selection_admissibility",
    "mechanism_composition.population_parent_context_differs_from_fit",
    "mechanism_composition.population_selection_arrow_into_mechanism",
    "mechanism_composition.population_undeclared_transport",
    "mechanism_composition.posterior_dependence_cross_block_duplicate",
    "mechanism_composition.posterior_dependence_cross_covariance_missing",
    "mechanism_composition.posterior_dependence_evidence_overlap_unknown",
    "mechanism_composition.posterior_dependence_independence_unjustified",
    "mechanism_composition.posterior_dependence_shared_evidence_declared_independent",
    "mechanism_composition.posterior_dependence_unknown",
    "mechanism_composition.premises_mismatch",
    "mechanism_composition.provenance_declared_mechanism_unacknowledged",
    "mechanism_composition.provenance_missing_artifact",
    "mechanism_composition.provenance_missing_evidence",
    "mechanism_composition.quantity_conditioned_event",
    "mechanism_composition.quantity_conditioning_mismatch",
    "mechanism_composition.quantity_functional_mismatch",
    "mechanism_composition.quantity_horizon_mismatch",
    "mechanism_composition.quantity_invalid_coordinate",
    "mechanism_composition.quantity_not_a_realization",
    "mechanism_composition.quantity_role_mismatch",
    "mechanism_composition.regime_parent_context_differs_from_fit",
    "mechanism_composition.regime_selection_diagram_not_applicable",
    "mechanism_composition.regime_undeclared_invariance",
    "mechanism_composition.request_unsupported",
    "mechanism_composition.result_mismatch",
    "mechanism_composition.shape_blank_coordinate",
    "mechanism_composition.shape_blank_identity",
    "mechanism_composition.shape_coefficient_parent_mismatch",
    "mechanism_composition.shape_duplicate_parent",
    "mechanism_composition.shape_empty_composition",
    "mechanism_composition.shape_invalid_domain",
    "mechanism_composition.shape_invalid_latents",
    "mechanism_composition.shape_invalid_table",
    "mechanism_composition.shape_mechanism_count",
    "mechanism_composition.shape_negative_scale",
    "mechanism_composition.shape_non_finite_parameter",
    "mechanism_composition.shape_too_many_mechanisms",
    "mechanism_composition.shape_too_many_parents",
    "mechanism_composition.slot_not_linear_gaussian",
    "mechanism_composition.support_disjoint",
    "mechanism_composition.support_intervention_outside_domain",
    "mechanism_composition.support_kind_mismatch",
    "mechanism_composition.support_missing_parent_configuration",
    "mechanism_composition.support_not_contained",
    "mechanism_composition.transform_mismatch",
    "mechanism_composition.uncertainty_assembled_not_psd",
    "mechanism_composition.uncertainty_cross_block_shape",
    "mechanism_composition.uncertainty_cross_block_unknown_mechanism",
    "mechanism_composition.uncertainty_family_unsupported",
    "mechanism_composition.uncertainty_joint_parameter_set_mismatch",
    "mechanism_composition.uncertainty_marginal_inconsistent_with_joint",
    "mechanism_composition.uncertainty_marginal_missing",
    "mechanism_composition.uncertainty_meaning_mismatch",
    "mechanism_composition.uncertainty_not_psd",
    "mechanism_composition.uncertainty_not_symmetric",
    "mechanism_composition.uncertainty_parameter_set_mismatch",
    "mechanism_composition.uncertainty_shape",
    "mechanism_composition.units_mismatch",
    "mechanism_discrepancy.degenerate_covariance",
    "mechanism_discrepancy.dependence_unknown",
    "mechanism_discrepancy.duplicate_name",
    "mechanism_discrepancy.incomparable_measurements",
    "mechanism_discrepancy.inconsistent_summary",
    "mechanism_discrepancy.invalid_alpha",
    "mechanism_discrepancy.invalid_artifact",
    "mechanism_discrepancy.invalid_identity",
    "mechanism_discrepancy.invalid_power",
    "mechanism_discrepancy.invalid_request",
    "mechanism_discrepancy.no_compared_coefficients",
    "mechanism_discrepancy.non_finite_value",
    "mechanism_discrepancy.rank_deficient_design",
    "mechanism_discrepancy.row_count_mismatch",
    "mechanism_discrepancy.sample_too_small",
    "mechanism_discrepancy.too_many_parents",
    "mechanism_discrepancy.wrong_contract",
    "mechanism_posterior.band_not_constructed",
    "mechanism_posterior.empty_summary",
    "mechanism_posterior.external_power_prior",
    "mechanism_posterior.mapped_prior",
    "mechanism_posterior.non_positive_scale",
    "mechanism_transport.diagram_source_mismatch",
    "mechanism_transport.diagram_target_mismatch",
    "mechanism_transport.duplicate_mechanism",
    "mechanism_transport.family_unsupported",
    "mechanism_transport.origin_not_fitted",
    "mechanism_transport.selection_arrow_into_latent",
    "mechanism_transport.target_mechanism_not_native",
    "mechanism_transport.transport_identity_mismatch",
    "mediation_posterior.band_not_constructed",
    "mediation_posterior.empty_design",
    "mediation_posterior.external_power_prior",
    "mediation_posterior.mapped_prior",
    "mediation_posterior.non_positive_scale",
    "missing_data.covariates_incomplete",
    "missing_data.multiple_imputation",
    "missing_data.policy_not_applicable",
    "missing_data.policy_required",
    "mixed_search.bounds_exceeded",
    "mixed_search.budget",
    "mixed_search.invalid_derivation",
    "mixed_search.invalid_query",
    "mixed_search.missing_joint",
    "mixed_search.named_route",
    "mixed_search.not_certified",
    "mixed_search.posterior_as_law",
    "mnar_fixed_shift.cancelled",
    "mnar_fixed_shift.constant_outcome",
    "mnar_fixed_shift.degenerate_interval",
    "mnar_fixed_shift.duplicate_subject",
    "mnar_fixed_shift.insufficient_observations",
    "mnar_fixed_shift.interval_not_supported",
    "mnar_fixed_shift.invalid_declaration",
    "mnar_fixed_shift.invalid_outcome",
    "mnar_fixed_shift.invalid_shift",
    "mnar_fixed_shift.invalid_source",
    "mnar_fixed_shift.lowering",
    "mnar_fixed_shift.preparation",
    "mnar_fixed_shift.preparation_cancelled",
    "mnar_fixed_shift.preparation_resource_limit",
    "mnar_fixed_shift.source_admission_cancelled",
    "mnar_fixed_shift.source_admission_resource_limit",
    "mnar_sensitivity.adjustment_not_valid",
    "mnar_sensitivity.bounds_exceeded",
    "mnar_sensitivity.composition_not_supported",
    "mnar_sensitivity.covariates_not_used",
    "mnar_sensitivity.covariates_required",
    "mnar_sensitivity.design_singular",
    "mnar_sensitivity.graph_not_supported",
    "mnar_sensitivity.insufficient_observed",
    "mnar_sensitivity.invalid_bootstrap",
    "mnar_sensitivity.invalid_data",
    "mnar_sensitivity.invalid_decision",
    "mnar_sensitivity.invalid_level",
    "mnar_sensitivity.invalid_loading",
    "mnar_sensitivity.invalid_tolerance",
    "mnar_sensitivity.mechanism_not_declared",
    "mnar_sensitivity.range_excludes_mar",
    "mnar_sensitivity.unbounded_range",
    "mnar_sensitivity.unsupported_estimator",
    "mnar_sensitivity.unsupported_mechanism",
    "mnar_sensitivity.unsupported_query",
    "mnar_sensitivity.unsupported_target",
    "msm_posterior.band_not_constructed",
    "msm_posterior.empty_law",
    "msm_posterior.external_power_prior",
    "msm_posterior.mapped_prior",
    "msm_posterior.non_positive_scale",
    "msm_sensitivity.bounds_exceeded",
    "msm_sensitivity.composition_not_supported",
    "msm_sensitivity.invalid_threshold",
    "msm_sensitivity.invalid_tolerance",
    "msm_sensitivity.lambda_below_one",
    "msm_sensitivity.lambda_range_empty",
    "msm_sensitivity.no_interval_method",
    "msm_sensitivity.outcome_law",
    "msm_sensitivity.positivity",
    "msm_sensitivity.stratum_mass",
    "multi_source_posterior.band_not_constructed",
    "multi_source_posterior.empty_law",
    "multi_source_posterior.external_power_prior",
    "multi_source_posterior.mapped_prior",
    "multi_source_posterior.non_positive_scale",
    "multiple_imputation.admission_refused",
    "multiple_imputation.ambiguous_variable_domain",
    "multiple_imputation.binary_under_joint_normal",
    "multiple_imputation.cancelled",
    "multiple_imputation.clusters_required",
    "multiple_imputation.column_length",
    "multiple_imputation.conditional_model_not_fitted",
    "multiple_imputation.dependent_series",
    "multiple_imputation.dynamics_joint_normal",
    "multiple_imputation.dynamics_lags",
    "multiple_imputation.graph_posterior_atoms",
    "multiple_imputation.graph_posterior_frequentist",
    "multiple_imputation.graph_posterior_query",
    "multiple_imputation.graph_posterior_series",
    "multiple_imputation.index",
    "multiple_imputation.interaction_variable",
    "multiple_imputation.interactions_joint_normal",
    "multiple_imputation.mixed_fits",
    "multiple_imputation.multi_horizon_posterior_not_joint",
    "multiple_imputation.non_finite",
    "multiple_imputation.not_at_random",
    "multiple_imputation.not_identified",
    "multiple_imputation.pool_df",
    "multiple_imputation.pool_non_finite",
    "multiple_imputation.pool_shape",
    "multiple_imputation.sampling_design",
    "multiple_imputation.spec_range",
    "multiple_imputation.stack_shape",
    "multiple_imputation.too_few_imputations",
    "multiple_imputation.units_labels",
    "multiple_imputation.units_shape",
    "multiple_imputation.variable_count",
    "multiple_imputation.variable_domain",
    "multiple_imputation.within_singular",
    "mz_transport.bounds_exceeded",
    "mz_transport.budget",
    "mz_transport.checked_obstruction",
    "mz_transport.empirical_counts_required",
    "mz_transport.fabricated_joint",
    "mz_transport.functional_graph_mismatch",
    "mz_transport.invalid_catalog",
    "mz_transport.invalid_derivation",
    "mz_transport.invalid_obstruction",
    "mz_transport.invalid_query",
    "mz_transport.missing_joint_regime",
    "mz_transport.search_incomplete",
    "native_attempt.invalid_callable",
    "native_claims.calibration_not_supported",
    "native_claims.contract_mismatch",
    "native_claims.grid_mismatch",
    "native_claims.invalid_context",
    "native_claims.joint_law_invalid",
    "native_claims.native_state_unavailable",
    "native_claims.no_source_requirement",
    "native_claims.no_supported_coordinate",
    "native_claims.point_support_missing",
    "native_claims.program_mismatch",
    "native_claims.projection_mismatch",
    "native_claims.provenance_mismatch",
    "native_claims.response_not_point_identified",
    "native_claims.source_not_supplied",
    "native_claims.units_conflict",
    "native_claims.unknown_variable",
    "native_claims.unsupported_estimand",
    "native_distribution.authority_required",
    "nested_markov.bayesian",
    "nested_markov.bayesian_artifact",
    "nested_markov.bayesian_artifact_limits",
    "nested_markov.bayesian_identity_mismatch",
    "nested_markov.bayesian_invalid_options",
    "nested_markov.bayesian_invalid_prior",
    "nested_markov.bayesian_sampler_scope",
    "nested_markov.cancelled",
    "nested_markov.constraint_violated",
    "nested_markov.fisher_artifact_limits",
    "nested_markov.fisher_identity_mismatch",
    "nested_markov.fisher_invalid_covariance",
    "nested_markov.fisher_invalid_level",
    "nested_markov.fisher_invalid_options",
    "nested_markov.fisher_sampling_design",
    "nested_markov.fisher_singular_information",
    "nested_markov.fit_not_converged",
    "nested_markov.fitted_boundary",
    "nested_markov.identification_unavailable",
    "nested_markov.invalid_counts",
    "nested_markov.invalid_input",
    "nested_markov.invalid_options",
    "nested_markov.iteration_bound",
    "nested_markov.measured_scope",
    "nested_markov.native_hook_unavailable",
    "nested_markov.not_normalized",
    "nested_markov.outside_binary_scope",
    "nested_markov.positivity",
    "nested_markov.producer_missing",
    "nonlinear_mediation.cancelled",
    "nonlinear_mediation.confounding",
    "nonlinear_mediation.cross_world_independence_not_declared",
    "nonlinear_mediation.data_required",
    "nonlinear_mediation.degenerate_mediator",
    "nonlinear_mediation.degenerate_mediator_residual",
    "nonlinear_mediation.degenerate_outcome",
    "nonlinear_mediation.degree_above_limit",
    "nonlinear_mediation.export_budget_exceeded",
    "nonlinear_mediation.identity_violation",
    "nonlinear_mediation.integration_error",
    "nonlinear_mediation.interventional_effects_not_supported",
    "nonlinear_mediation.invalid_degree",
    "nonlinear_mediation.invalid_nodes",
    "nonlinear_mediation.invalid_request",
    "nonlinear_mediation.invalid_support_bound",
    "nonlinear_mediation.invalid_tolerance",
    "nonlinear_mediation.length_mismatch",
    "nonlinear_mediation.limits_exceeded",
    "nonlinear_mediation.memory_budget_exceeded",
    "nonlinear_mediation.nodes_above_limit",
    "nonlinear_mediation.non_finite_input",
    "nonlinear_mediation.non_finite_option",
    "nonlinear_mediation.overlap",
    "nonlinear_mediation.quadrature_nodes_not_converged",
    "nonlinear_mediation.quadrature_weights_invalid",
    "nonlinear_mediation.rank_deficient",
    "nonlinear_mediation.report_replay_mismatch",
    "nonlinear_mediation.too_few_rows",
    "nonlinear_mediation.treatment_induced_confounding",
    "nonlinear_mediation.treatment_not_binary",
    "nonlinear_mediation.wrong_contract",
    "observation_query.arm_mismatch",
    "observation_query.artifact_invalid",
    "observation_query.artifact_refused",
    "observation_query.blank_field",
    "observation_query.cancelled",
    "observation_query.construct_mismatch",
    "observation_query.construct_observed_directly",
    "observation_query.count_fitted_bounds_exceeded",
    "observation_query.count_fitted_refused",
    "observation_query.declared_model_has_no_sampling_interval",
    "observation_query.differential_error_unsupported",
    "observation_query.evidence_covariance_singular",
    "observation_query.indicator_kind",
    "observation_query.indicator_mismatch",
    "observation_query.intervention_in_conditioning_slot",
    "observation_query.invalid_declaration",
    "observation_query.invalid_interval",
    "observation_query.invalid_request",
    "observation_query.level_unreachable",
    "observation_query.levels_for_continuous_model",
    "observation_query.measured_as_construct",
    "observation_query.measured_background",
    "observation_query.measured_signal_not_redrawn",
    "observation_query.measurement_route",
    "observation_query.mixed_conditioning_and_intervention",
    "observation_query.multiple_measured_signals",
    "observation_query.no_calibrated_probability",
    "observation_query.no_parameter_draws",
    "observation_query.outcome_assigned",
    "observation_query.population_mismatch",
    "observation_query.positive_level",
    "observation_query.real_values_for_finite_model",
    "observation_query.regime_mismatch",
    "observation_query.repeated_slot",
    "observation_query.request_unsupported",
    "observation_query.unknown_level",
    "observation_query.unknown_quantity",
    "observation_query.unknown_variable",
    "observation_query.value_not_finite",
    "observation_query.zero_probability_evidence",
    "paradigm_parity.paradigm_not_supported",
    "parent_set_effect.conflict_resolution_budget",
    "parent_set_effect.discovery_failed",
    "parent_set_effect.knowledge_invalid",
    "parent_set_effect.log_linear_input_not_positive",
    "parent_set_effect.outcome_precedes_treatment",
    "parent_set_effect.parent_limit_bound",
    "parent_set_effect.search_not_finished",
    "parent_set_effect.source_invalid",
    "parent_set_effect.source_not_canonical",
    "parent_set_effect.structure_class",
    "parent_set_effect.transformation_undeclared",
    "parent_set_effect.treatment_orientation_depends_on_conflict",
    "parent_set_effect.treatment_parents_ambiguous",
    "penalized_posterior.band_not_constructed",
    "penalized_posterior.empty_scores",
    "penalized_posterior.external_power_prior",
    "penalized_posterior.mapped_prior",
    "penalized_posterior.non_positive_scale",
    "penalized_propensity.cancelled",
    "penalized_propensity.effect_posterior",
    "penalized_propensity.fallback_not_supported",
    "penalized_propensity.fallback_with_penalty",
    "penalized_propensity.grid_allocation",
    "penalized_propensity.invalid_penalty",
    "penalized_propensity.scope",
    "penalized_propensity.selection_not_supported",
    "policy_value.action_not_populated",
    "policy_value.action_outside_support",
    "policy_value.action_propensity_below_floor",
    "policy_value.cancelled",
    "policy_value.evaluation_fraction_missing",
    "policy_value.evaluation_missing",
    "policy_value.evaluation_unknown",
    "policy_value.invalid_data",
    "policy_value.invalid_evaluation_fraction",
    "policy_value.invalid_folds",
    "policy_value.invalid_known_propensity",
    "policy_value.invalid_propensity_floor",
    "policy_value.invalid_request",
    "policy_value.invalid_support",
    "policy_value.objective_missing",
    "policy_value.objective_unknown",
    "policy_value.policy_actions_missing",
    "policy_value.policy_learned_on_evaluation_rows",
    "policy_value.policy_origin_missing",
    "policy_value.policy_origin_unknown",
    "policy_value.reference_outside_support",
    "policy_value.sparse_action",
    "policy_value.support_missing",
    "policy_value.too_few_evaluation_rows",
    "policy_value.too_few_rows",
    "policy_value.training_split_missing_action",
    "policy_value.undeclared_level",
    "predictive_assessment.bounds_exceeded",
    "predictive_assessment.cancelled",
    "predictive_assessment.design_invalid",
    "predictive_assessment.diagnostic_not_defined",
    "predictive_assessment.horizon_mismatch",
    "predictive_assessment.iid_not_supported_for_dependent_design",
    "predictive_assessment.in_sample",
    "predictive_assessment.interval_dependence_undeclared",
    "predictive_assessment.interval_law_family_mixed",
    "predictive_assessment.interval_not_supported",
    "predictive_assessment.interval_request_invalid",
    "predictive_assessment.invalid_artifact",
    "predictive_assessment.invalid_law",
    "predictive_assessment.invalid_observations",
    "predictive_assessment.law_callback_not_built",
    "predictive_assessment.log_score_infinite",
    "predictive_assessment.not_a_causal_claim",
    "predictive_assessment.outcome_kind",
    "predictive_assessment.overlap_unknown",
    "predictive_assessment.provider_identity_incomplete",
    "predictive_assessment.report_replay_mismatch",
    "predictive_assessment.score_not_defined",
    "predictive_assessment.score_not_probability",
    "predictive_assessment.serial_lag_below_horizon",
    "predictive_assessment.target_mismatch",
    "predictive_assessment.too_few_clusters",
    "predictive_assessment.unknown_source",
    "predictive_assessment.wrong_contract",
    "prior_signal.adapted_prior_not_draws",
    "prior_signal.artifact_not_in_catalog",
    "prior_signal.coefficient_prior_not_signal",
    "prior_signal.duplicate_source",
    "prior_signal.incompatible_artifact",
    "prior_signal.incomplete_posterior",
    "prior_signal.invalid_parameter",
    "prior_signal.invalid_request",
    "prior_signal.lineage_missing",
    "prior_signal.named_state_draws_required",
    "prior_signal.no_sources",
    "prior_signal.nonfinite_mapped_prior",
    "prior_signal.observation_lineage_missing",
    "prior_signal.observation_overlap_unknown",
    "prior_signal.observation_reuse",
    "prior_signal.original_posterior_unidentified",
    "prior_signal.posterior_draws_required",
    "prior_signal.prior_identity_mismatch",
    "prior_signal.prior_projection_mismatch",
    "prior_signal.prior_sources_overlap",
    "prior_signal.resolution_invalid",
    "prior_signal.signal_family_mismatch",
    "prior_signal.source_not_in_catalog_or_duplicate",
    "prior_signal.source_population_mismatch",
    "prior_signal.state_quantity_mismatch",
    "prior_signal.target_population_mismatch",
    "prior_signal.transport_policy_required",
    "prior_signal.unlock_list_not_signal",
    "prior_signal.unsupported_signal_provider",
    "program_binding.contract_identity_mismatch",
    "program_binding.dose_grid_changed",
    "program_binding.graph_only_identity",
    "program_binding.invalid_binding",
    "program_binding.malformed_quantity",
    "program_binding.no_identified_request",
    "program_binding.population_mismatch",
    "program_binding.quantities_override_mismatch",
    "program_binding.treatment_outcome_substitution",
    "proposal_arrival.assignment_mismatch",
    "proposal_arrival.base_catalog_changed",
    "proposal_arrival.base_catalog_missing",
    "proposal_arrival.base_provider_mismatch",
    "proposal_arrival.binding_missing",
    "proposal_arrival.bounds_exceeded",
    "proposal_arrival.cancelled",
    "proposal_arrival.candidate_missing",
    "proposal_arrival.duplicate_original_artifacts",
    "proposal_arrival.expected_proposal_mismatch",
    "proposal_arrival.family_not_supported",
    "proposal_arrival.identity_mismatch",
    "proposal_arrival.independent_joint_study_required",
    "proposal_arrival.invalid_request",
    "proposal_arrival.original_artifacts_required",
    "proposal_arrival.original_consumer_refused",
    "proposal_arrival.proof_not_bound",
    "proposal_arrival.provider_binding_mismatch",
    "proposal_arrival.raw_counts_required",
    "proposal_arrival.replay_mismatch",
    "proposal_arrival.sample_size_mismatch",
    "proposal_arrival.structural_arrival_refused",
    "proposal_receipt.arrival_cancelled",
    "proposal_receipt.arrival_empty",
    "proposal_receipt.arrival_family_mismatch",
    "proposal_receipt.arrival_invalid",
    "proposal_receipt.arrival_sample_mismatch",
    "proposal_receipt.base_failure_mismatch",
    "proposal_receipt.bundle_identity_mismatch",
    "proposal_receipt.bundle_inconsistent",
    "proposal_receipt.candidate_mismatch",
    "proposal_receipt.candidate_not_in_bundle",
    "proposal_receipt.candidate_not_in_ranking",
    "proposal_receipt.candidate_not_in_repair",
    "proposal_receipt.candidate_not_proposed",
    "proposal_receipt.candidate_set_mismatch",
    "proposal_receipt.contract_mismatch",
    "proposal_receipt.contract_required",
    "proposal_receipt.cost_mismatch",
    "proposal_receipt.delta_mismatch",
    "proposal_receipt.derivation_mismatch",
    "proposal_receipt.duplicate_candidate",
    "proposal_receipt.empty_bundle",
    "proposal_receipt.hypothetical_not_evidence",
    "proposal_receipt.identity_mismatch",
    "proposal_receipt.invalid_delta",
    "proposal_receipt.lineage_mismatch",
    "proposal_receipt.non_canonical_order",
    "proposal_receipt.obligation_mismatch",
    "proposal_receipt.ranking_digest_mismatch",
    "proposal_receipt.repair_digest_mismatch",
    "proposal_receipt.signal_mismatch",
    "proposal_receipt.source_digest_mismatch",
    "proposal_receipt.value_mismatch",
    "provider_capability_negotiation.capability_missing",
    "provider_capability_negotiation.request_mismatch",
    "provider_transport.artifact_invalid",
    "provider_transport.assumptions_missing",
    "provider_transport.assumptions_not_carried",
    "provider_transport.blank_field",
    "provider_transport.claim_not_in_target",
    "provider_transport.evidence_missing",
    "provider_transport.identity_mismatch",
    "provider_transport.input_not_external",
    "provider_transport.native_target_required",
    "provider_transport.native_trust_claimed",
    "provider_transport.object_invalid",
    "provider_transport.object_unsupported",
    "provider_transport.population_missing",
    "provider_transport.population_not_a_source",
    "provider_transport.quantity_invalid",
    "provider_transport.quantity_not_declared",
    "provider_transport.receipt_does_not_cover",
    "provider_transport.regime_missing",
    "provider_transport.transport_identity_incomplete",
    "provider_transport.trust_incomplete",
    "provider_transport.verification_required",
    "quadratic_posterior.band_not_constructed",
    "quadratic_posterior.external_power_prior",
    "quadratic_posterior.insufficient_weight",
    "quadratic_posterior.mapped_prior",
    "quadratic_posterior.non_positive_scale",
    "randomized_quantile_effect.adjustment_not_supported",
    "randomized_quantile_effect.arm_too_small",
    "randomized_quantile_effect.scope",
    "rank_drop_estimate.estimator_not_supported",
    "rank_drop_estimate.propensity_separates",
    "rank_drop_estimate.shape_not_supported",
    "rank_drop_estimate.span_not_preserved",
    "recalc.adjusted_arm_not_populated",
    "recalc.adjusted_design_rank_deficient",
    "recalc.adjusted_fit_refused",
    "recalc.adjusted_glm_boundary_saturated",
    "recalc.adjusted_glm_not_converged",
    "recalc.adjusted_glm_separated",
    "recalc.adjusted_invalid_model",
    "recalc.adjusted_model_data_invalid",
    "recalc.adjusted_prediction_out_of_support",
    "recalc.adjusted_prediction_schema_mismatch",
    "recalc.adjusted_receipt_invalid",
    "recalc.adjustment_not_identified",
    "recalc.aipw_requires_overlap_diagnostics",
    "recalc.bayesian_basis_prior_unsupported",
    "recalc.bayesian_effect_unavailable",
    "recalc.bayesian_external_compose_unsupported",
    "recalc.bayesian_fit_refused",
    "recalc.bayesian_inference_unsupported",
    "recalc.bayesian_invalid_data",
    "recalc.bayesian_invalid_graph",
    "recalc.bayesian_invalid_inference",
    "recalc.bayesian_invalid_query",
    "recalc.bayesian_invalid_summary",
    "recalc.bayesian_prior_invalid",
    "recalc.bayesian_prior_likelihood_double_use",
    "recalc.bayesian_prior_source_conflict",
    "recalc.bayesian_prior_source_unsupported",
    "recalc.bayesian_prior_source_unverified",
    "recalc.bayesian_workspace_limit",
    "recalc.blocked_by_refused_dependency",
    "recalc.bootstrap_replicates_unsupported",
    "recalc.cancelled",
    "recalc.capability_adapter_unavailable",
    "recalc.categorical_role_mismatch",
    "recalc.contrast_out_of_support",
    "recalc.contrast_unavailable",
    "recalc.declared_size_mismatch",
    "recalc.design_artifact_mismatch",
    "recalc.design_artifact_unverified",
    "recalc.design_data_unavailable",
    "recalc.design_effect_unavailable",
    "recalc.design_fit_refused",
    "recalc.design_model_data_invalid",
    "recalc.design_not_checked",
    "recalc.design_not_identified",
    "recalc.design_point_unavailable",
    "recalc.design_rank_deficient",
    "recalc.design_receipt_invalid",
    "recalc.dr_arm_not_populated",
    "recalc.dr_design_rank_deficient",
    "recalc.dr_effect_unavailable",
    "recalc.dr_fit_counter_mismatch",
    "recalc.dr_fit_refused",
    "recalc.dr_glm_boundary_saturated",
    "recalc.dr_glm_not_converged",
    "recalc.dr_glm_separated",
    "recalc.dr_invalid_folds",
    "recalc.dr_invalid_learner",
    "recalc.dr_invalid_model",
    "recalc.dr_invalid_overlap",
    "recalc.dr_invalid_score",
    "recalc.dr_model_data_invalid",
    "recalc.dr_not_identified",
    "recalc.dr_prediction_out_of_support",
    "recalc.dr_prediction_schema_mismatch",
    "recalc.dr_predictor_unavailable",
    "recalc.dr_receipt_invalid",
    "recalc.dr_scores_unavailable",
    "recalc.dr_source_mismatch",
    "recalc.dr_target_incompatible",
    "recalc.family_model_data_invalid",
    "recalc.family_not_identified",
    "recalc.family_receipt_invalid",
    "recalc.frozen_scores_rejected",
    "recalc.incompatible_contrast",
    "recalc.invalid_adjustment_set",
    "recalc.invalid_categorical_contrast",
    "recalc.invalid_changed_input",
    "recalc.invalid_contrast",
    "recalc.invalid_data",
    "recalc.invalid_declaration",
    "recalc.invalid_design_artifact",
    "recalc.invalid_design_roles",
    "recalc.invalid_evidence",
    "recalc.invalid_graph",
    "recalc.invalid_initial_state_law",
    "recalc.invalid_query",
    "recalc.invalid_rd_window",
    "recalc.invalid_request",
    "recalc.invalid_target_weights",
    "recalc.invalid_temporal_provider",
    "recalc.invalid_temporal_window",
    "recalc.iv_decision_unavailable",
    "recalc.iv_diagnostics_unavailable",
    "recalc.joint_bayesian_graph_unsupported",
    "recalc.joint_bayesian_invalid_data",
    "recalc.joint_bayesian_invalid_query",
    "recalc.joint_bayesian_refused",
    "recalc.limits_exceeded",
    "recalc.memory_budget_exceeded",
    "recalc.mz_action_out_of_support",
    "recalc.mz_invalid_request",
    "recalc.no_live_state",
    "recalc.off_grid_request",
    "recalc.primary_result_is_not_an_effect",
    "recalc.refused",
    "recalc.retarget_incompatible",
    "recalc.retarget_not_supported",
    "recalc.row_count_mismatch",
    "recalc.row_ids_mismatch",
    "recalc.score_stage_undeclared",
    "recalc.score_table_unavailable",
    "recalc.static_action_out_of_support",
    "recalc.static_engine_refused",
    "recalc.static_graph_unsupported",
    "recalc.static_invalid_contrast",
    "recalc.static_invalid_data",
    "recalc.static_invalid_factor",
    "recalc.static_invalid_query",
    "recalc.static_not_identified",
    "recalc.static_receipt_invalid",
    "recalc.static_required_factor_missing",
    "recalc.static_transport_budget_cancel",
    "recalc.static_transport_missing_evidence",
    "recalc.static_transport_missing_provider",
    "recalc.static_transport_non_transportable",
    "recalc.static_transport_not_certified",
    "recalc.static_transport_numerical_failure",
    "recalc.static_transport_support_failure",
    "recalc.static_transport_unsupported_evaluator",
    "recalc.temporal_artifact_invalid",
    "recalc.temporal_artifact_mismatch",
    "recalc.temporal_artifact_seed_mismatch",
    "recalc.temporal_budget_cancel",
    "recalc.temporal_data_invalid",
    "recalc.temporal_evaluator_unsupported",
    "recalc.temporal_fit_refused",
    "recalc.temporal_history_unsupported",
    "recalc.temporal_missing_evidence",
    "recalc.temporal_missing_provider",
    "recalc.temporal_numerical_failure",
    "recalc.temporal_scope_unsupported",
    "recalc.temporal_support_failure",
    "recalc.unavailable_data",
    "recalc.unavailable_fit",
    "recalc.unavailable_provider",
    "recalc.unknown_variable",
    "recalc.unsupported_aipw_score_configuration",
    "recalc.unsupported_request",
    "recalc.weighted_overlap_failed",
    "recalc_receipt.counts_inconsistent",
    "recalc_receipt.fresh_process_reuse",
    "recalc_receipt.identity_mismatch",
    "recalc_receipt.plan_mismatch",
    "recalc_receipt.totals_mismatch",
    "recovery.bounds_exceeded",
    "recovery.budget",
    "recovery.empirical_not_supported",
    "recovery.handoff_mismatch",
    "recovery.invalid_derivation",
    "recovery.invalid_observed_law",
    "recovery.invalid_query",
    "recovery.missing_margin",
    "recovery.nonrecoverable_witness",
    "recovery.positivity",
    "recovery.unsupported_mechanism",
    "recovery.witness_unavailable",
    "recovery_chain.budget",
    "recovery_chain.data_identity_mismatch",
    "recovery_chain.decision_replay_mismatch",
    "recovery_chain.invalid_derivation",
    "recovery_chain.invalid_observed_law",
    "recovery_chain.invalid_query",
    "recovery_chain.names_mismatch",
    "recovery_chain.nonrecoverable_witness",
    "recovery_chain.not_evaluated",
    "recovery_chain.positivity",
    "recovery_chain.premises_mismatch",
    "recovery_chain.recovered_replay_mismatch",
    "recovery_chain.undecodable",
    "recovery_chain.unsupported_mechanism",
    "recovery_chain.unsupported_semantics",
    "refresh_posterior.band_not_constructed",
    "refresh_posterior.empty_values",
    "refresh_posterior.external_power_prior",
    "refresh_posterior.mapped_prior",
    "refresh_posterior.non_positive_scale",
    "repair_artifact.bounds_exceeded",
    "repair_artifact.budget",
    "repair_artifact.invalid_artifact",
    "repair_artifact.replay_mismatch",
    "repair_artifact.unsupported_version",
    "response_gam_derivative.shared_complete_case_rows",
    "rollout.artifact_id_missing",
    "rollout.artifact_kind_mismatch",
    "rollout.expected_binding_mismatch",
    "rollout.finite_draw_state_required",
    "rollout.identity_mismatch",
    "rollout.invalid_request",
    "rollout.ordered_prior_mismatch",
    "rollout.ranking_invalid",
    "rollout.ranking_source_mismatch",
    "rollout.source_digest_mismatch",
    "rollout.source_evidence_mismatch",
    "rollout.source_standing_mismatch",
    "rollout.state_coordinate_mismatch",
    "rollout.state_interpretation_mismatch",
    "rollout.terminal_decision_mismatch",
    "rollout.unweighted_joint_state_required",
    "sampled_recovery.bounds_exceeded",
    "sampled_recovery.cancelled",
    "sampled_recovery.component_variance_only",
    "sampled_recovery.invalid_input",
    "sampled_recovery.native_hook_unavailable",
    "sampled_recovery.producer_missing",
    "sampled_recovery.receipt_mismatch",
    "sampled_recovery.too_many_failed_replicates",
    "sampled_recovery.unrecoverable_pattern",
    "sampling_weights.invalid",
    "sampling_weights.route_unsupported",
    "scenario_covariance.cancelled",
    "scenario_covariance.consumer_limit_exceeded",
    "scenario_covariance.data_identity_mismatch",
    "scenario_covariance.duplicate_scenario_id",
    "scenario_covariance.exact_enumeration_cap",
    "scenario_covariance.invalid_declaration",
    "scenario_covariance.invalid_dependence",
    "scenario_covariance.invalid_failure_bound",
    "scenario_covariance.invalid_functional",
    "scenario_covariance.invalid_row_table",
    "scenario_covariance.no_scenarios",
    "scenario_covariance.not_positive_semidefinite",
    "scenario_covariance.premises_mismatch",
    "scenario_covariance.replay_mismatch",
    "scenario_covariance.table_too_large",
    "scenario_covariance.too_few_replicates",
    "scenario_covariance.too_few_rows",
    "scenario_covariance.too_many_failed_replicates",
    "scenario_covariance.too_many_terms",
    "scenario_covariance.unknown_dependence",
    "scenario_covariance.unsupported_semantics",
    "scenario_covariance_posterior.band_not_constructed",
    "scenario_covariance_posterior.empty_matrix",
    "scenario_covariance_posterior.external_power_prior",
    "scenario_covariance_posterior.mapped_prior",
    "scenario_covariance_posterior.non_positive_scale",
    "scenario_invariance.derivation_inconsistent",
    "scenario_invariance.non_static_node",
    "scenario_invariance.not_estimated",
    "scenario_invariance.refused",
    "scenario_invariance.wrong_result_type",
    "scenarios.catalog_selections",
    "scenarios.coordinate_mismatch",
    "scenarios.count",
    "scenarios.duplicate_or_empty_name",
    "scenarios.duplicate_scenario",
    "scenarios.empty",
    "scenarios.equivalence_class_input",
    "scenarios.invalid_weights",
    "scenarios.shared_data_aggregate",
    "scenarios.unevaluated_budget",
    "scientific_bridge.artifact_invalid",
    "scientific_bridge.association_as_intervention",
    "scientific_bridge.assumption_evidence_missing",
    "scientific_bridge.assumption_missing",
    "scientific_bridge.blank_field",
    "scientific_bridge.cancelled",
    "scientific_bridge.chain_stage_not_affine",
    "scientific_bridge.conditioning_mismatch",
    "scientific_bridge.degenerate_validation",
    "scientific_bridge.dependence_not_a_covariance",
    "scientific_bridge.differential_error",
    "scientific_bridge.draws_meaning_mismatch",
    "scientific_bridge.draws_not_aligned",
    "scientific_bridge.draws_outside_domain",
    "scientific_bridge.draws_unusable",
    "scientific_bridge.duplicate_assumption",
    "scientific_bridge.endpoint_classes",
    "scientific_bridge.estimand_bridge_incomplete",
    "scientific_bridge.estimand_bridge_mismatch",
    "scientific_bridge.estimand_spend_inconsistent",
    "scientific_bridge.estimand_units_mismatch",
    "scientific_bridge.experimental_prior_without_estimand_bridge",
    "scientific_bridge.functional_not_supported",
    "scientific_bridge.horizon_mismatch",
    "scientific_bridge.identity_mismatch",
    "scientific_bridge.interval_not_supported",
    "scientific_bridge.interval_scope_outside_measured",
    "scientific_bridge.interval_source_uncertainty_required",
    "scientific_bridge.invalid_quantity",
    "scientific_bridge.joint_law_dimension_limit",
    "scientific_bridge.malformed_mapping",
    "scientific_bridge.mapping_not_allowed",
    "scientific_bridge.mapping_scope_mismatch",
    "scientific_bridge.mapping_uncertainty_invalid",
    "scientific_bridge.mapping_uncertainty_missing",
    "scientific_bridge.measurement_model_mismatch",
    "scientific_bridge.measurement_route_not_supported",
    "scientific_bridge.model_identity_mismatch",
    "scientific_bridge.model_score_as_probability",
    "scientific_bridge.non_finite_value",
    "scientific_bridge.nonlinear_law_insufficient",
    "scientific_bridge.nonlinear_mean_not_expectation",
    "scientific_bridge.outside_attainable_range",
    "scientific_bridge.outside_validated_range",
    "scientific_bridge.population_mismatch",
    "scientific_bridge.predictive_importance_as_causal_effect",
    "scientific_bridge.propagation_admission_refused",
    "scientific_bridge.propagation_overflow",
    "scientific_bridge.propagation_work_limit",
    "scientific_bridge.provenance",
    "scientific_bridge.proxy_as_construct",
    "scientific_bridge.quantity_mismatch",
    "scientific_bridge.regime_mismatch",
    "scientific_bridge.replay_mismatch",
    "scientific_bridge.scientific_endpoint_as_utility",
    "scientific_bridge.simulation_draw_limit",
    "scientific_bridge.source_dependence_invalid",
    "scientific_bridge.source_dependence_missing",
    "scientific_bridge.source_moment_not_finite",
    "scientific_bridge.source_not_draws",
    "scientific_bridge.source_not_point",
    "scientific_bridge.source_variance_invalid",
    "scientific_bridge.stated_preference_as_realized_behavior",
    "scientific_bridge.transport_diagram_scope_mismatch",
    "scientific_bridge.transport_premise_contradicted_by_selection_diagram",
    "scientific_bridge.transport_premise_mismatch",
    "scientific_bridge.transport_premise_missing",
    "scientific_bridge.transport_premise_unused",
    "scientific_bridge.uncertain_mapping_not_affine",
    "scientific_bridge.units_mismatch",
    "scientific_bridge.validation_r_squared_too_low",
    "scientific_bridge.validation_trials_too_few",
    "scientific_bridge.value_kind_not_carried",
    "scientific_bridge.value_kind_substitution",
    "scientific_bridge.youden_too_small",
    "sde.action_time_off_grid",
    "sde.input_outside_certified_envelope",
    "sde.milstein_noncommutative",
    "sde.unbounded_growth",
    "selective_discovery.analysis_rows_not_the_split_part",
    "selective_discovery.artifact_invalid",
    "selective_discovery.background_knowledge_contradictory",
    "selective_discovery.background_knowledge_not_applicable",
    "selective_discovery.background_knowledge_unknown_variable",
    "selective_discovery.components_incomplete",
    "selective_discovery.contemporaneous_links",
    "selective_discovery.contract_identity",
    "selective_discovery.contract_mismatch",
    "selective_discovery.data_mismatch",
    "selective_discovery.differences_are_not_selection_targets",
    "selective_discovery.downstream_interval_not_supported",
    "selective_discovery.identity_mismatch",
    "selective_discovery.invalid_contract",
    "selective_discovery.multi_population",
    "selective_discovery.nothing_retained",
    "selective_discovery.pooled_population_data",
    "selective_discovery.pooled_population_search",
    "selective_discovery.population_missing",
    "selective_discovery.population_repeated",
    "selective_discovery.populations_missing",
    "selective_discovery.raw_data_missing",
    "selective_discovery.receipt_invalid",
    "selective_discovery.receipt_limit",
    "selective_discovery.receipt_repeated",
    "selective_discovery.receipts_identity",
    "selective_discovery.recovery_claim_not_supported",
    "selective_discovery.replay_diverged",
    "selective_discovery.replay_miss",
    "selective_discovery.rng_mismatch",
    "selective_discovery.route_unknown",
    "selective_discovery.rows",
    "selective_discovery.score_search_not_supported",
    "selective_discovery.search_incomplete",
    "selective_discovery.split_fraction_invalid",
    "selective_discovery.split_mask_mismatch",
    "selective_discovery.split_not_disjoint",
    "selective_discovery.split_part_empty",
    "selective_discovery.split_part_weight_invalid",
    "selective_discovery.split_receipt_mismatch",
    "selective_discovery.split_row_excluded",
    "selective_discovery.split_row_out_of_range",
    "selective_discovery.split_rows_mismatch",
    "selective_discovery.split_rows_repeated",
    "selective_discovery.split_source_mismatch",
    "selective_discovery.split_unsupported_rows",
    "selective_discovery.split_weights_mismatch",
    "selective_discovery.structure_class",
    "selective_discovery.structure_marks",
    "selective_discovery.structure_nodes",
    "selective_discovery.test_identity_missing",
    "selective_discovery.unaccepted_discovery",
    "selective_discovery.unresolved_structure",
    "selective_discovery.variables_repeated",
    "selective_discovery.weighted_test_without_weights",
    "selective_discovery.weights_not_read_by_test",
    "selective_discovery.work_inconsistent",
    "sensitivity_decision_composition.bounds_exceeded",
    "sensitivity_decision_composition.composition_not_supported",
    "sensitivity_decision_composition.invalid_artifact",
    "sensitivity_decision_composition.invalid_surface",
    "sensitivity_decision_composition.sampling_not_reported",
    "sensitivity_decision_composition.unsupported_adaptation",
    "sensitivity_decision_composition.unsupported_coordinate",
    "sensitivity_decision_composition.wrong_contract",
    "sensitivity_source.invalid_artifact",
    "sensitivity_source.limits_exceeded",
    "sensitivity_source.lineage_digest",
    "sensitivity_source.source_unavailable",
    "sensitivity_source.surface_mismatch",
    "sequential_evaluation.accounting_residual",
    "sequential_evaluation.action_channel",
    "sequential_evaluation.action_cost",
    "sequential_evaluation.action_delayed_outcome",
    "sequential_evaluation.action_index",
    "sequential_evaluation.action_label",
    "sequential_evaluation.action_not_supported",
    "sequential_evaluation.action_support",
    "sequential_evaluation.action_transition",
    "sequential_evaluation.action_utility",
    "sequential_evaluation.artifact_id_missing",
    "sequential_evaluation.artifact_kind_mismatch",
    "sequential_evaluation.behaviour_support_missing",
    "sequential_evaluation.cancelled",
    "sequential_evaluation.channel_binding_mismatch",
    "sequential_evaluation.channel_construct_mismatch",
    "sequential_evaluation.channel_levels_mismatch",
    "sequential_evaluation.channel_rates",
    "sequential_evaluation.channel_route",
    "sequential_evaluation.commit_has_no_observation",
    "sequential_evaluation.comparison_kind_mismatch",
    "sequential_evaluation.comparison_label",
    "sequential_evaluation.comparisons",
    "sequential_evaluation.decision_receipt_mismatch",
    "sequential_evaluation.delayed_outcome_beyond_horizon",
    "sequential_evaluation.differential_channel",
    "sequential_evaluation.empty_log",
    "sequential_evaluation.exact_mismatch",
    "sequential_evaluation.history_receipt_mismatch",
    "sequential_evaluation.identity_mismatch",
    "sequential_evaluation.importance_weight_overflow",
    "sequential_evaluation.indicator_as_state",
    "sequential_evaluation.interval_not_supported",
    "sequential_evaluation.limit_exceeded",
    "sequential_evaluation.log_partially_counted",
    "sequential_evaluation.logged_trajectory_inconsistent",
    "sequential_evaluation.malformed",
    "sequential_evaluation.model_actions",
    "sequential_evaluation.model_field",
    "sequential_evaluation.model_horizon",
    "sequential_evaluation.model_initial",
    "sequential_evaluation.model_mismatch",
    "sequential_evaluation.model_resource_budget",
    "sequential_evaluation.model_states",
    "sequential_evaluation.model_terminal_utility",
    "sequential_evaluation.model_work",
    "sequential_evaluation.no_overlap",
    "sequential_evaluation.observation_impossible",
    "sequential_evaluation.observation_recorded_twice",
    "sequential_evaluation.off_policy_mismatch",
    "sequential_evaluation.past_horizon",
    "sequential_evaluation.policy_change_refused",
    "sequential_evaluation.policy_distribution",
    "sequential_evaluation.policy_incomplete",
    "sequential_evaluation.policy_mismatch",
    "sequential_evaluation.policy_version",
    "sequential_evaluation.record_already_counted",
    "sequential_evaluation.record_identity",
    "sequential_evaluation.resource_budget_exceeded",
    "sequential_evaluation.return_overflow",
    "sequential_evaluation.rollout_belief",
    "sequential_evaluation.rollout_mismatch",
    "sequential_evaluation.rollouts",
    "sequential_evaluation.session_label",
    "sequential_evaluation.static_score_table",
    "sequential_evaluation.stopping_boundary",
    "sequential_evaluation.temporal_transport_contract_missing",
    "sequential_evaluation.trajectory_identity",
    "sequential_evaluation.trajectory_recorded_twice",
    "sequential_evaluation.transported_policy_contract_missing",
    "sequential_evaluation.value_mismatch",
    "sequential_evaluation.version",
    "sequential_evaluation.voi_collect_action",
    "sequential_evaluation.voi_commit_actions",
    "sequential_evaluation.voi_enumeration_mismatch",
    "sequential_evaluation.voi_ranking",
    "sequential_evaluation.voi_scope",
    "sequential_evaluation.weighted_estimate_overflow",
    "sequential_evaluation.weighted_sum_overflow",
    "signal_provider.candidate_mismatch",
    "signal_provider.capability_missing",
    "signal_provider.computational_limit",
    "signal_provider.conditional_independence_missing",
    "signal_provider.invalid_object",
    "signal_provider.invalid_request",
    "signal_provider.invalid_table",
    "signal_provider.law_incoherent",
    "signal_provider.missing_lineage",
    "signal_provider.observation_quantity_mismatch",
    "signal_provider.posterior_incoherent",
    "signal_provider.posterior_quantity_mismatch",
    "signal_provider.prior_mismatch",
    "signal_provider.request_fingerprint_mismatch",
    "signal_provider.sample_size_invalid",
    "signal_provider.sample_size_mismatch",
    "signal_provider.trust_mismatch",
    "signal_provider.trust_native_claimed",
    "smoothed_dose_posterior.band_not_constructed",
    "smoothed_dose_posterior.cancelled",
    "smoothed_dose_posterior.certificate_mismatch",
    "smoothed_dose_posterior.empty_scores",
    "smoothed_dose_posterior.external_power_prior",
    "smoothed_dose_posterior.invalid_graph",
    "smoothed_dose_posterior.mapped_prior",
    "smoothed_dose_posterior.non_positive_scale",
    "smoothed_dose_posterior.not_certified",
    "smoothed_dose_posterior.resource_limit",
    "source_evidence.action_mismatch",
    "source_evidence.coordinate_mismatch",
    "source_evidence.coordinates_unavailable",
    "source_evidence.diagnostic_scope_mismatch",
    "source_evidence.forward_binding_mismatch",
    "source_evidence.invalid_artifact",
    "source_evidence.invalid_contract",
    "source_evidence.invalid_coordinates",
    "source_evidence.invalid_source_artifact",
    "source_evidence.limits_exceeded",
    "source_evidence.point_binding_mismatch",
    "source_evidence.resolution_scope_unsupported",
    "source_evidence.response_unavailable",
    "source_evidence.source_binding_mismatch",
    "source_evidence.source_encoding_failed",
    "source_evidence.source_execution_failed",
    "source_evidence.source_unverified",
    "source_projection.affine_contract_refused",
    "source_projection.attestation_unavailable",
    "source_projection.causal_source_unavailable",
    "source_projection.coordinate_mismatch",
    "source_projection.identity_mismatch",
    "source_projection.invalid_artifact",
    "source_projection.invalid_request",
    "source_projection.limits_exceeded",
    "source_projection.mean_binding_mismatch",
    "source_projection.mean_grid_required",
    "source_projection.original_consumer_refused",
    "source_projection.replay_mismatch",
    "source_projection.response_unavailable",
    "source_projection.source_binding_mismatch",
    "source_projection.unsupported_coordinate",
    "split_effect.degenerate_variance",
    "split_effect.design_singular",
    "split_effect.discovery_failed",
    "split_effect.outcome_precedes_treatment",
    "split_effect.search_not_finished",
    "split_effect.source_invalid",
    "split_effect.source_not_canonical",
    "split_effect.structure_class",
    "split_effect.structure_unresolved",
    "split_effect.too_few_rows",
    "structured_refusals.malformed_detail",
    "structured_refusals.missing_stage",
    "structured_refusals.unregistered_code",
    "study_candidate.declarations_required",
    "study_candidate.sample_below_obligation",
    "study_candidate.wrong_contract",
    "study_plan.arrival_mismatch",
    "study_plan.arrival_not_identified",
    "study_plan.bounds_exceeded",
    "study_plan.budget",
    "study_plan.invalid_artifact",
    "study_plan.invalid_candidate",
    "study_plan.invalid_query",
    "study_plan.no_failure_to_repair",
    "study_plan.none_certified",
    "study_plan.theorem_limited",
    "survival_effect.arm_not_populated",
    "survival_effect.cancelled",
    "survival_effect.censoring_assumption_missing",
    "survival_effect.censoring_assumption_unknown",
    "survival_effect.censoring_covariates_missing",
    "survival_effect.censoring_support",
    "survival_effect.horizon_beyond_follow_up",
    "survival_effect.horizon_missing",
    "survival_effect.invalid_data",
    "survival_effect.invalid_event",
    "survival_effect.invalid_floor",
    "survival_effect.invalid_folds",
    "survival_effect.invalid_horizon",
    "survival_effect.invalid_levels",
    "survival_effect.invalid_request",
    "survival_effect.too_few_rows",
    "survival_effect.training_fold_without_events",
    "survival_effect.treatment_positivity",
    "survival_effect.undeclared_level",
    "temporal_counterfactual.artifact_changed",
    "temporal_counterfactual.fit_mismatch",
    "temporal_counterfactual.history_name_invalid",
    "temporal_counterfactual.horizon_exceeded",
    "temporal_counterfactual.invalid_graph",
    "temporal_counterfactual.invalid_identity",
    "temporal_counterfactual.invalid_request",
    "temporal_counterfactual.latent_confounding",
    "temporal_counterfactual.limits_exceeded",
    "temporal_counterfactual.non_finite_history",
    "temporal_counterfactual.refuting_history",
    "temporal_counterfactual.shared_history_missing",
    "temporal_counterfactual.time_misaligned",
    "temporal_counterfactual.too_many_units",
    "temporal_counterfactual.unpaired_histories",
    "temporal_interval.consumer_unavailable",
    "temporal_interval.interval_not_supported",
    "temporal_interval.invalid_estimand",
    "temporal_interval.retired_method",
    "temporal_interval.retired_response_method",
    "temporal_interval.studentized_bounds_exceeded",
    "temporal_interval.studentized_estimator_not_certified",
    "temporal_interval.studentized_score_mismatch",
    "temporal_interval.studentized_unbalanced",
    "temporal_interval.studentized_zero_variance",
    "temporal_interval.too_few_units",
    "temporal_interval.too_many_replicates",
    "temporal_interval.unknown_units",
    "temporal_interval.unsupported_history",
    "temporal_policy_transport.history_alphabet_incompatible",
    "temporal_policy_transport.horizon_mismatch",
    "temporal_policy_transport.initial_law_declared_invariant",
    "temporal_policy_transport.invariance_binding_mismatch",
    "temporal_policy_transport.mechanism_declared_twice",
    "temporal_policy_transport.mechanism_status_missing",
    "temporal_policy_transport.mechanism_time_varying",
    "temporal_policy_transport.mechanism_unknown",
    "temporal_policy_transport.target_initial_law",
    "temporal_policy_transport.target_initial_law_missing",
    "temporal_policy_transport.target_is_source",
    "temporal_policy_transport.target_period",
    "temporal_policy_transport.target_support_outside_source",
    "temporal_policy_transport.time_invariance_missing",
    "temporal_policy_transport.value_mismatch",
    "temporal_refresh.factor_window_mismatch",
    "temporal_refresh.horizon_changed",
    "temporal_refresh.intervention_history_changed",
    "temporal_refresh.invalid_initial_law",
    "temporal_refresh.invalid_replacement",
    "temporal_refresh.lag_alignment_changed",
    "temporal_refresh.no_refresh",
    "temporal_refresh.premises_changed",
    "temporal_refresh.receipt_digest_mismatch",
    "temporal_refresh.receipt_mismatch",
    "temporal_refresh.stale_interval",
    "temporal_refresh.stale_snapshot",
    "temporal_refresh.window_missing",
    "temporal_transport.bounds_exceeded",
    "temporal_transport.checked_obstruction",
    "temporal_transport.history_budget",
    "temporal_transport.history_outside_support",
    "temporal_transport.horizon",
    "temporal_transport.interval_requested",
    "temporal_transport.invalid_sequence",
    "temporal_transport.invalid_spec",
    "temporal_transport.missing_evidence",
    "temporal_transport.search_incomplete",
    "temporal_transport.target_law_mismatch",
    "temporal_transport.target_law_missing",
    "tier_diagnostics.cancelled",
    "tier_diagnostics.evalue_embeds_interval_limit",
    "tier_diagnostics.evalue_not_computed",
    "tier_diagnostics.no_evalue_interval_method",
    "tier_diagnostics.not_a_tiered_average_result",
    "tier_diagnostics.overlap_not_carried",
    "tier_diagnostics.unknown_scenarios_no_evalue",
    "tier_diagnostics.unknown_scenarios_no_overlap",
    "transport_revalidation.claim_target_mismatch",
    "transport_revalidation.duplicate_claim",
    "transport_revalidation.identity_mismatch",
    "transport_revalidation.provider_trust_not_applicable",
    "transport_revalidation.replay_mismatch",
    "transport_revalidation.scene_invalid",
    "transport_revalidation.source_undeclared",
    "transport_revalidation.unknown_claim",
    "transport_revalidation.value_digest",
    "transport_sensitivity.blank_identity",
    "transport_sensitivity.coordinate_out_of_range",
    "transport_sensitivity.interval_not_composed",
    "transport_sensitivity.invalid_bound",
    "transport_sensitivity.invalid_grid",
    "transport_sensitivity.law_response_not_scalar",
    "transport_sensitivity.no_scalar_family",
    "transport_sensitivity.non_finite_point",
    "transport_sensitivity.not_transported",
    "transport_sensitivity.proportional_at_zero",
    "transport_sensitivity.quantity_not_in_target",
    "transport_sensitivity.route_undeclared",
    "transport_sensitivity.surface_refused",
    "transported_counterfactual.artifact_changed",
    "transported_counterfactual.cross_world_independence_missing",
    "transported_counterfactual.factor_missing",
    "transported_counterfactual.invalid_diagram",
    "transported_counterfactual.invalid_factor",
    "transported_counterfactual.invalid_identity",
    "transported_counterfactual.invalid_law",
    "transported_counterfactual.invalid_model",
    "transported_counterfactual.invalid_query",
    "transported_counterfactual.invalid_request",
    "transported_counterfactual.joint_theorem_not_derived",
    "transported_counterfactual.limits_exceeded",
    "transported_counterfactual.noise_law_not_shared",
    "transported_counterfactual.nonadditive_mechanism",
    "transported_counterfactual.overlap_failure",
    "transported_counterfactual.prerequisite_missing",
    "transported_counterfactual.selection_on_mechanism",
    "transported_decision.calibration_label_inflated",
    "transported_decision.coordinate_not_in_target",
    "transported_decision.exclusions_leave_no_comparison",
    "transported_decision.interval_not_supported",
    "transported_decision.invalid_value",
    "transported_decision.law_not_supplied",
    "transported_decision.mixed_transports",
    "transported_decision.native_label_on_transported_value",
    "transported_decision.not_transported",
    "transported_decision.route_needs_typed_source",
    "transported_decision.support_label_inflated",
    "transported_decision.support_rule_unmet",
    "transported_decision.transport_identity_incomplete",
    "transported_decision.transport_identity_not_digest",
    "transported_decision.transported_not_accepted",
    "transported_decision.uncertainty_requirement",
    "transported_decision_artifact.bounds_exceeded",
    "transported_decision_artifact.identity_expected",
    "transported_decision_artifact.invalid",
    "transported_decision_artifact.law_not_exportable",
    "transported_decision_artifact.replay_mismatch",
    "transported_effect.cohort_invalid",
    "transported_effect.cohort_not_numeric",
    "transported_effect.decision_source_mismatch",
    "transported_effect.declaration_invalid",
    "transported_effect.feature_schema_mismatch",
    "transported_effect.identity_mismatch",
    "transported_effect.interval_not_supported",
    "transported_effect.invalid_request",
    "transported_effect.law_not_supplied",
    "transported_effect.link_not_identity",
    "transported_effect.model_invalid",
    "transported_effect.model_not_portable",
    "transported_effect.modifier_set_mismatch",
    "transported_effect.no_overlap",
    "transported_effect.non_finite_result",
    "transported_effect.out_of_support",
    "transported_effect.outcome_units_required",
    "transported_effect.population_not_different",
    "transported_effect.prediction_refused",
    "transported_effect.selection_enters_mechanism",
    "transported_effect.shift_declaration_mismatch",
    "transported_effect.shift_declaration_required",
    "transported_effect.support_contract_absent",
    "transported_effect.weights_invalid",
    "transported_effect.weights_not_normalized",
    "transported_policy.diagram_binding_mismatch",
    "transported_policy.initial_law_declared_invariant",
    "transported_policy.mechanism_declared_twice",
    "transported_policy.mechanism_not_invariant",
    "transported_policy.mechanism_status_missing",
    "transported_policy.mechanism_unknown",
    "transported_policy.selection_diagram_missing",
    "transported_policy.target_initial_law",
    "transported_policy.target_initial_law_missing",
    "transported_policy.target_is_source",
    "transported_policy.target_population",
    "transported_policy.target_support_outside_source",
    "transported_policy.target_utility_digest",
    "transported_policy.target_utility_missing",
    "transported_policy.target_utility_provenance_missing",
    "transported_policy.target_utility_support",
    "transported_policy.target_utility_value",
    "transported_policy.utility_declared_invariant",
    "transported_policy.value_mismatch",
    "transported_synthesis.native_population_blank",
    "transported_synthesis.no_contributions",
    "transported_synthesis.numeric_refused",
    "transported_synthesis.overlap_not_modelled",
    "transported_synthesis.prior_role_mismatch",
    "transported_synthesis.shared_transport_needs_covariance",
    "transported_synthesis.source_listed_twice",
    "transported_synthesis.target_population_mismatch",
    "transported_synthesis.transport_identity_incomplete",
    "transported_synthesis.transport_variance_on_native_source",
    "transported_synthesis.transport_variance_undeclared",
    "transported_synthesis.unknown_source",
    "treatment_levels.cancelled",
    "treatment_levels.contrast_level_not_declared",
    "treatment_levels.full_population_undeclared_level",
    "treatment_levels.invalid_data",
    "treatment_levels.invalid_level_set",
    "treatment_levels.invalid_request",
    "treatment_levels.level_not_populated",
    "treatment_levels.levels_missing",
    "treatment_levels.sparse_level",
    "treatment_levels.training_fold_missing_level",
    "treatment_levels.undeclared_level",
    "two_step_posterior.band_not_constructed",
    "two_step_posterior.empty_law",
    "two_step_posterior.external_power_prior",
    "two_step_posterior.mapped_prior",
    "two_step_posterior.non_positive_scale",
    "typed_mechanism_transport.cycle",
    "typed_mechanism_transport.dependence_invalid",
    "typed_mechanism_transport.design_not_shared",
    "typed_mechanism_transport.fit_not_converged",
    "typed_mechanism_transport.fit_refused",
    "typed_mechanism_transport.invariance_contradicted",
    "typed_mechanism_transport.invariance_level_invalid",
    "typed_mechanism_transport.parameter_dependence_incomplete",
    "typed_mechanism_transport.parameter_dependence_unknown",
    "typed_mechanism_transport.parent_without_mechanism",
    "typed_mechanism_transport.parents_changed",
    "typed_mechanism_transport.regime_not_declared_invariant",
    "typed_mechanism_transport.selection_target_unknown_variable",
    "typed_mechanism_transport.shifted_without_target_evidence",
    "typed_mechanism_transport.target_specification_differs",
    "typed_mechanism_transport.transport_identity_mismatch",
    "typed_mechanism_transport.unknown_variable",
    "vector_posterior.band_not_constructed",
    "vector_posterior.external_power_prior",
    "vector_posterior.mapped_prior",
    "vector_posterior.non_positive_scale",
    "vector_treatment.adjustment_set_mismatch",
    "vector_treatment.cancelled",
    "vector_treatment.cluster_ids_required",
    "vector_treatment.cluster_unread",
    "vector_treatment.collinear_treatments",
    "vector_treatment.data_not_bound",
    "vector_treatment.degenerate_covariance",
    "vector_treatment.duplicate_contrast_coefficient",
    "vector_treatment.duplicate_name",
    "vector_treatment.empty_contrast",
    "vector_treatment.export_budget_exceeded",
    "vector_treatment.invalid_identity",
    "vector_treatment.invalid_lag",
    "vector_treatment.invalid_request",
    "vector_treatment.lag_required",
    "vector_treatment.lag_unread",
    "vector_treatment.leverage_one",
    "vector_treatment.limits_exceeded",
    "vector_treatment.memory_budget_exceeded",
    "vector_treatment.non_finite_value",
    "vector_treatment.rank_deficient_adjustment",
    "vector_treatment.row_count_mismatch",
    "vector_treatment.row_snapshot_mismatch",
    "vector_treatment.summary_inconsistent",
    "vector_treatment.summary_model_based_only",
    "vector_treatment.too_few_clusters",
    "vector_treatment.too_few_rows",
    "vector_treatment.too_few_treatments",
    "vector_treatment.too_many_treatments",
    "vector_treatment.treatment_without_variation",
    "vector_treatment.unknown_contrast_coefficient",
    "vector_treatment.weights_need_robust_covariance",
    "vector_treatment.wrong_contract",
];

/// Detail id and the reason codes it is refused under, sorted by id.
const DETAIL_CODES: &[(&str, &[&str])] = &[
    ("adaptive_inference.analysis_invalid", &["invalid_argument"]),
    ("adaptive_inference.analysis_route_mismatch", &["adaptive_assumption_violated"]),
    ("adaptive_inference.anytime_premise_contexts", &["adaptive_assumption_violated"]),
    ("adaptive_inference.anytime_settings_invalid", &["invalid_argument"]),
    ("adaptive_inference.anytime_settings_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.arm_not_observed", &["adaptive_assumption_violated"]),
    ("adaptive_inference.arm_out_of_range", &["invalid_argument"]),
    ("adaptive_inference.arms_invalid", &["invalid_argument"]),
    ("adaptive_inference.artifact_size_exceeded", &["route_not_supported"]),
    ("adaptive_inference.assigned_probability_zero", &["invalid_argument"]),
    ("adaptive_inference.assignment_not_replayed", &["adaptive_assumption_violated"]),
    ("adaptive_inference.assignment_probabilities_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.batch_incomplete", &["adaptive_assumption_violated"]),
    ("adaptive_inference.between_unit_variance_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.borrowing_fixed_after_first_look", &["adaptive_assumption_violated"]),
    ("adaptive_inference.cancelled", &["cancelled_no_claim"]),
    ("adaptive_inference.cell_incoherent", &["route_not_supported"]),
    ("adaptive_inference.cell_not_delivered", &["route_not_supported"]),
    ("adaptive_inference.choice_outcome_needs_choice_contract", &["invalid_argument"]),
    ("adaptive_inference.contexts_invalid", &["invalid_argument"]),
    ("adaptive_inference.correlation_invalid", &["invalid_argument"]),
    ("adaptive_inference.declared_looks_exceeded", &["invalid_argument"]),
    ("adaptive_inference.declared_units_exceeded", &["invalid_argument"]),
    ("adaptive_inference.delay_not_ignorable", &["adaptive_assumption_violated"]),
    ("adaptive_inference.exact_law_states_exceeded", &["invalid_argument"]),
    ("adaptive_inference.exact_law_unavailable", &["adaptive_assumption_violated"]),
    ("adaptive_inference.exact_law_work_exceeded", &["invalid_argument"]),
    ("adaptive_inference.exposure_differs_from_assignment", &["adaptive_assumption_violated"]),
    ("adaptive_inference.exposure_history_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.exposure_invalid", &["invalid_argument"]),
    ("adaptive_inference.final_analysis_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.fixed_sample_data_dependent_stopping", &["adaptive_assumption_violated"]),
    ("adaptive_inference.fixed_sample_design_adaptive", &["adaptive_assumption_violated"]),
    ("adaptive_inference.fixed_sample_on_adaptive_data", &["adaptive_design_ignored"]),
    ("adaptive_inference.group_sequential_assignment_adaptive", &["adaptive_assumption_violated"]),
    ("adaptive_inference.guarantee_not_transferable", &["adaptive_assumption_violated"]),
    ("adaptive_inference.historical_borrowing_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.historical_prior_invalid", &["invalid_argument"]),
    ("adaptive_inference.independence_assumed_for_dependent_measurements", &["adaptive_assumption_violated"]),
    ("adaptive_inference.input_invalid", &["invalid_argument"]),
    ("adaptive_inference.interim_analyses_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.interval_names_no_quantity", &["quantity_semantics_mismatch"]),
    ("adaptive_inference.inversion_premise_contexts", &["adaptive_assumption_violated"]),
    ("adaptive_inference.ledger_limit", &["invalid_argument"]),
    ("adaptive_inference.look_out_of_range", &["invalid_argument"]),
    ("adaptive_inference.looks_not_ordered", &["invalid_argument"]),
    ("adaptive_inference.looks_overlap_unknown", &["evidence_overlap_unknown"]),
    ("adaptive_inference.looks_share_observations", &["evidence_double_counted"]),
    ("adaptive_inference.measurements_invalid", &["invalid_argument"]),
    ("adaptive_inference.no_stagewise_adjustment_procedure", &["route_not_supported"]),
    ("adaptive_inference.no_stopping_reliability_procedure", &["estimator_inference_mismatch"]),
    ("adaptive_inference.numerical_failure", &["invalid_argument"]),
    ("adaptive_inference.outcome_availability_invalid", &["invalid_argument"]),
    ("adaptive_inference.outcome_availability_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.outcome_dependent_delay", &["adaptive_assumption_violated"]),
    ("adaptive_inference.outcome_invalid", &["invalid_argument"]),
    ("adaptive_inference.outcome_structure_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.pairing_not_supported", &["route_not_supported"]),
    ("adaptive_inference.paradigm_not_supported_for_cell", &["route_not_supported"]),
    ("adaptive_inference.paradigm_rule_mismatch", &["adaptive_assumption_violated"]),
    ("adaptive_inference.policy_history_invalid", &["invalid_argument"]),
    ("adaptive_inference.policy_history_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.policy_selection_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.positivity_floor", &["adaptive_assumption_violated"]),
    ("adaptive_inference.posterior_premise_outcome_dependent_delay", &["adaptive_assumption_violated"]),
    ("adaptive_inference.probabilities_invalid", &["invalid_argument"]),
    ("adaptive_inference.probability_floor_invalid", &["invalid_argument"]),
    ("adaptive_inference.procedure_limit_exceeded", &["invalid_argument"]),
    ("adaptive_inference.repeated_measures_paradigm_not_derived", &["adaptive_assumption_violated"]),
    ("adaptive_inference.repeated_measures_stopping_not_derived", &["invalid_argument"]),
    ("adaptive_inference.report_replay_mismatch", &["invalid_argument"]),
    ("adaptive_inference.retained_record_exceeded", &["invalid_argument"]),
    ("adaptive_inference.rule_mismatch", &["invalid_argument"]),
    ("adaptive_inference.selection_not_ignorable", &["adaptive_assumption_violated"]),
    ("adaptive_inference.shape_mismatch", &["invalid_argument"]),
    ("adaptive_inference.simulation_memory_exceeded", &["invalid_argument"]),
    ("adaptive_inference.simulation_replications_exceeded", &["invalid_argument"]),
    ("adaptive_inference.simulation_work_exceeded", &["invalid_argument"]),
    ("adaptive_inference.stopping_not_replayed", &["adaptive_assumption_violated"]),
    ("adaptive_inference.stopping_rule_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.structure_incomplete", &["adaptive_assumption_violated"]),
    ("adaptive_inference.unit_out_of_range", &["invalid_argument"]),
    ("adaptive_inference.variance_invalid", &["invalid_argument"]),
    ("adaptive_inference.within_unit_correlation_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.within_unit_variance_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_inference.wrong_contract", &["route_not_supported"]),
    ("adaptive_transport.arms", &["invalid_argument"]),
    ("adaptive_transport.assignment_below_floor", &["adaptive_assumption_violated"]),
    ("adaptive_transport.assignment_monitoring_contract_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_transport.design_not_adaptive", &["route_not_supported"]),
    ("adaptive_transport.diagram_binding_mismatch", &["transport_missing_evidence"]),
    ("adaptive_transport.mechanism_declared_twice", &["invalid_argument"]),
    ("adaptive_transport.mechanism_status_missing", &["transport_missing_evidence"]),
    ("adaptive_transport.no_analysed_units", &["transport_support_failure"]),
    ("adaptive_transport.outcome_mechanism_not_invariant", &["transport_missing_evidence"]),
    ("adaptive_transport.outcome_missing", &["adaptive_contract_incomplete"]),
    ("adaptive_transport.population_label", &["invalid_argument"]),
    ("adaptive_transport.selection_diagram_missing", &["transport_missing_evidence"]),
    ("adaptive_transport.strata", &["invalid_argument"]),
    ("adaptive_transport.stratum_arm_unobserved", &["transport_support_failure"]),
    ("adaptive_transport.stratum_law_declared_invariant", &["invalid_argument"]),
    ("adaptive_transport.target_is_source", &["invalid_argument"]),
    ("adaptive_transport.target_stratum_law", &["invalid_argument"]),
    ("adaptive_transport.target_stratum_law_missing", &["transport_missing_evidence"]),
    ("adaptive_transport.unit_strata", &["invalid_argument"]),
    ("adaptive_transport.value_mismatch", &["invalid_argument"]),
    ("admg_conditional_posterior.band_not_constructed", &["route_not_supported"]),
    ("admg_conditional_posterior.empty_law", &["invalid_argument"]),
    ("admg_conditional_posterior.external_power_prior", &["route_not_supported"]),
    ("admg_conditional_posterior.mapped_prior", &["route_not_supported"]),
    ("admg_conditional_posterior.non_positive_scale", &["invalid_argument"]),
    ("admg_transport.bounds_exceeded", &["route_not_supported"]),
    ("admg_transport.budget", &["transport_budget_cancel"]),
    ("admg_transport.consumer_limits", &["route_not_supported"]),
    ("admg_transport.counted_laws_unsupported", &["route_not_supported"]),
    ("admg_transport.data_identity_mismatch", &["transport_not_certified"]),
    ("admg_transport.evidence_mismatch", &["transport_not_certified"]),
    ("admg_transport.invalid_artifact", &["invalid_argument"]),
    ("admg_transport.invalid_catalog", &["invalid_argument"]),
    ("admg_transport.invalid_derivation", &["transport_not_certified"]),
    ("admg_transport.invalid_query", &["invalid_argument"]),
    ("admg_transport.invalid_request", &["invalid_argument"]),
    ("admg_transport.missing_evidence", &["transport_missing_evidence"]),
    ("admg_transport.not_certified", &["transport_not_certified"]),
    ("admg_transport.premises_mismatch", &["transport_not_certified"]),
    ("admg_transport.proven_non_transportable", &["transport_proven_non_transportable"]),
    ("admg_transport.replay_mismatch", &["transport_not_certified"]),
    ("admg_transport.support_failure", &["transport_support_failure"]),
    ("aligned_joint_draws.identity_expected", &["quantity_semantics_mismatch"]),
    ("aligned_joint_draws.legacy_binding", &["quantity_semantics_mismatch"]),
    ("aligned_joint_draws.legacy_source", &["quantity_semantics_mismatch"]),
    ("aligned_joint_draws.marginals_not_joint", &["joint_law_required"]),
    ("aligned_joint_draws.outcome_coordinate", &["quantity_semantics_mismatch"]),
    ("batch_retarget.cancelled", &["cancelled_no_claim"]),
    ("batch_retarget.contrast_member_failed", &["route_not_supported"]),
    ("batch_retarget.covariance_unavailable", &["route_not_supported"]),
    ("batch_retarget.duplicate_name", &["invalid_argument"]),
    ("batch_retarget.empty_family", &["invalid_argument"]),
    ("batch_retarget.functional_not_supported", &["route_not_supported"]),
    ("batch_retarget.incompatible_target", &["invalid_argument"]),
    ("batch_retarget.invalid_contrast", &["invalid_argument"]),
    ("batch_retarget.max_t_draws_out_of_range", &["invalid_argument"]),
    ("batch_retarget.max_t_invalid_level", &["invalid_argument"]),
    ("batch_retarget.mixed_snapshot", &["row_weights_bound_to_snapshot"]),
    ("batch_retarget.partial_family", &["route_not_supported"]),
    ("batch_retarget.point_only_member", &["route_not_supported"]),
    ("batch_retarget.scores_mismatch", &["invalid_argument"]),
    ("batch_retarget.scores_unavailable", &["score_table_unavailable"]),
    ("batch_retarget.scores_unavailable_after_estimate", &["score_table_unavailable"]),
    ("batch_retarget.unknown_claim", &["invalid_argument"]),
    ("batch_retarget.weighted_overlap_failed", &["positivity_violated"]),
    ("batch_retarget_posterior.band_not_constructed", &["route_not_supported"]),
    ("batch_retarget_posterior.empty_scores", &["invalid_argument"]),
    ("batch_retarget_posterior.external_power_prior", &["route_not_supported"]),
    ("batch_retarget_posterior.mapped_prior", &["route_not_supported"]),
    ("batch_retarget_posterior.non_positive_scale", &["invalid_argument"]),
    ("bayesian_transport.admission_refused", &["transport_budget_cancel"]),
    ("bayesian_transport.cancelled", &["transport_budget_cancel"]),
    ("bayesian_transport.feature_mismatch", &["invalid_argument"]),
    ("bayesian_transport.identification_not_certified", &["transport_missing_evidence"]),
    ("bayesian_transport.identification_required", &["transport_missing_evidence"]),
    ("bayesian_transport.invalid_input", &["invalid_argument"]),
    ("bayesian_transport.invalid_prior", &["invalid_argument"]),
    ("bayesian_transport.missing_law", &["joint_law_required"]),
    ("bayesian_transport.missing_source", &["transport_missing_evidence"]),
    ("bayesian_transport.prior_data_overlap", &["invalid_argument"]),
    ("bayesian_transport.prior_dimension", &["prior_dimension_mismatch"]),
    ("bayesian_transport.singular_fit", &["design_rank_deficient"]),
    ("bayesian_transport.source_dependence", &["sampling_dependence_unknown"]),
    ("bayesian_transport.support_resource_limit", &["execution_resource_limit"]),
    ("bayesian_transport.too_many_draws", &["invalid_argument"]),
    ("bayesian_transport.too_many_parameters", &["invalid_argument"]),
    ("bayesian_transport.unsupported_graph", &["route_not_supported"]),
    ("bayesian_transport.weak_overlap", &["transport_support_failure"]),
    ("binary_recovery_posterior.band_not_constructed", &["route_not_supported"]),
    ("binary_recovery_posterior.empty_law", &["invalid_argument"]),
    ("binary_recovery_posterior.external_power_prior", &["route_not_supported"]),
    ("binary_recovery_posterior.mapped_prior", &["route_not_supported"]),
    ("binary_recovery_posterior.non_positive_scale", &["invalid_argument"]),
    ("callback_protocol.artifact_invalid", &["invalid_argument"]),
    ("callback_protocol.blank_field", &["callback_protocol_invalid"]),
    ("callback_protocol.cancelled", &["cancelled_no_claim"]),
    ("callback_protocol.diagnostic_only", &["callback_reuse_ineligible"]),
    ("callback_protocol.duplicate_name", &["callback_protocol_invalid"]),
    ("callback_protocol.hidden_state_unobservable", &["callback_reuse_ineligible"]),
    ("callback_protocol.identification_not_conferred", &["callback_protocol_invalid"]),
    ("callback_protocol.identity_mismatch", &["route_not_supported"]),
    ("callback_protocol.interval_not_supported", &["uncertainty_representation_insufficient"]),
    ("callback_protocol.invalid_quantity", &["callback_protocol_invalid"]),
    ("callback_protocol.invalid_support", &["callback_protocol_invalid"]),
    ("callback_protocol.invocation_limit", &["callback_limit_exceeded"]),
    ("callback_protocol.invocation_outside_execution", &["callback_side_effect_forbidden"]),
    ("callback_protocol.joint_law_not_conferred", &["joint_law_required"]),
    ("callback_protocol.meaning_not_for_kind", &["callback_protocol_invalid"]),
    ("callback_protocol.native_trust_claimed", &["callback_protocol_invalid"]),
    ("callback_protocol.native_trust_not_conferred", &["callback_protocol_invalid"]),
    ("callback_protocol.nested_input_overlap_unknown", &["sampling_dependence_unknown"]),
    ("callback_protocol.nested_input_unregistered", &["sampling_dependence_unknown"]),
    ("callback_protocol.nested_inputs_mismatch", &["callback_protocol_invalid"]),
    ("callback_protocol.nested_inputs_share_evidence", &["evidence_double_counted"]),
    ("callback_protocol.no_idempotency_contract", &["callback_reuse_ineligible"]),
    ("callback_protocol.no_operations", &["callback_protocol_invalid"]),
    ("callback_protocol.opaque_dependencies", &["callback_reuse_ineligible"]),
    ("callback_protocol.operation_not_declared", &["external_capability_missing"]),
    ("callback_protocol.operation_not_for_kind", &["callback_protocol_invalid"]),
    ("callback_protocol.output_not_a_law", &["callback_output_invalid"]),
    ("callback_protocol.output_not_finite", &["callback_output_invalid"]),
    ("callback_protocol.output_outside_support", &["callback_output_invalid"]),
    ("callback_protocol.output_shape", &["callback_output_invalid"]),
    ("callback_protocol.output_too_large", &["callback_limit_exceeded"]),
    ("callback_protocol.posterior_guarantee_not_supported", &["uncertainty_representation_insufficient"]),
    ("callback_protocol.provider_failed", &["callback_provider_unavailable"]),
    ("callback_protocol.provider_identity_mismatch", &["external_binding_mismatch"]),
    ("callback_protocol.provider_unavailable", &["callback_provider_unavailable"]),
    ("callback_protocol.query_invalid", &["callback_output_invalid"]),
    ("callback_protocol.replay_contract_incomplete", &["callback_protocol_invalid"]),
    ("callback_protocol.request_mismatch", &["external_binding_mismatch"]),
    ("callback_protocol.retry_outside_policy", &["callback_side_effect_forbidden"]),
    ("callback_protocol.side_effects_not_allowed", &["callback_side_effect_forbidden"]),
    ("callback_protocol.too_many_items", &["callback_protocol_invalid"]),
    ("callback_protocol.trust_incomplete", &["callback_protocol_invalid"]),
    ("callback_protocol.update_declaration_mismatch", &["callback_protocol_invalid"]),
    ("callback_protocol.wall_limit_exceeded", &["callback_limit_exceeded"]),
    ("categorical_posterior.band_not_constructed", &["route_not_supported"]),
    ("categorical_posterior.external_power_prior", &["route_not_supported"]),
    ("categorical_posterior.mapped_prior", &["route_not_supported"]),
    ("categorical_posterior.non_positive_scale", &["invalid_argument"]),
    ("categorical_treatment.absent_level", &["arm_not_populated"]),
    ("categorical_treatment.cancelled", &["cancelled_no_claim"]),
    ("categorical_treatment.data_not_bound", &["route_not_supported"]),
    ("categorical_treatment.export_budget_exceeded", &["route_not_supported"]),
    ("categorical_treatment.invalid_identity", &["invalid_argument"]),
    ("categorical_treatment.invalid_level", &["invalid_argument"]),
    ("categorical_treatment.invalid_min_level_rows", &["invalid_argument"]),
    ("categorical_treatment.invalid_pair", &["invalid_argument"]),
    ("categorical_treatment.invalid_request", &["invalid_argument"]),
    ("categorical_treatment.limits_exceeded", &["invalid_argument"]),
    ("categorical_treatment.memory_budget_exceeded", &["route_not_supported"]),
    ("categorical_treatment.monotonicity_requires_ordered", &["route_not_supported"]),
    ("categorical_treatment.posterior_declared_rows", &["route_not_supported"]),
    ("categorical_treatment.row_count_mismatch", &["route_not_supported"]),
    ("categorical_treatment.sparse_level", &["arm_not_populated"]),
    ("categorical_treatment.summary_inconsistent", &["invalid_argument"]),
    ("categorical_treatment.too_few_levels", &["invalid_argument"]),
    ("categorical_treatment.undeclared_level", &["invalid_argument"]),
    ("categorical_treatment.unknown_level", &["invalid_argument"]),
    ("categorical_treatment.unknown_reference", &["invalid_argument"]),
    ("categorical_treatment.wrong_contract", &["route_not_supported"]),
    ("chain_recovery_posterior.band_not_constructed", &["route_not_supported"]),
    ("chain_recovery_posterior.empty_law", &["invalid_argument"]),
    ("chain_recovery_posterior.external_power_prior", &["route_not_supported"]),
    ("chain_recovery_posterior.mapped_prior", &["route_not_supported"]),
    ("chain_recovery_posterior.non_positive_scale", &["invalid_argument"]),
    ("cluster_dml.cancelled", &["cancelled_no_claim"]),
    ("cluster_dml.cluster_ids_missing", &["required_option_missing"]),
    ("cluster_dml.dyadic_giant_component", &["dyadic_dependence_not_supported"]),
    ("cluster_dml.dyadic_shared_namespace", &["dyadic_dependence_not_supported"]),
    ("cluster_dml.flexible_learner_not_supported", &["route_not_supported"]),
    ("cluster_dml.invalid_min_clusters", &["invalid_argument"]),
    ("cluster_dml.invalid_min_components", &["invalid_argument"]),
    ("cluster_dml.no_interval_method", &["cluster_interval_not_supported"]),
    ("cluster_dml.penalized_propensity", &["route_not_supported"]),
    ("cluster_dml.scope", &["route_not_supported"]),
    ("cluster_dml.second_ids_missing", &["required_option_missing"]),
    ("cluster_dml.second_ids_unexpected", &["invalid_argument"]),
    ("cluster_dml.too_few_clusters", &["too_few_clusters"]),
    ("cluster_dml.too_few_components", &["too_few_clusters"]),
    ("clustered_posterior.band_not_constructed", &["route_not_supported"]),
    ("clustered_posterior.empty_scores", &["invalid_argument"]),
    ("clustered_posterior.external_power_prior", &["route_not_supported"]),
    ("clustered_posterior.mapped_prior", &["route_not_supported"]),
    ("clustered_posterior.non_positive_scale", &["invalid_argument"]),
    ("cohort_decision.cancelled", &["invalid_argument"]),
    ("cohort_decision.cohort_changed", &["invalid_argument"]),
    ("cohort_decision.cohort_empty", &["invalid_argument"]),
    ("cohort_decision.cohort_invalid", &["invalid_argument"]),
    ("cohort_decision.cohort_too_large", &["invalid_argument"]),
    ("cohort_decision.container_checksum_mismatch", &["invalid_argument"]),
    ("cohort_decision.container_invalid", &["invalid_argument"]),
    ("cohort_decision.contrast_mismatch", &["quantity_semantics_mismatch"]),
    ("cohort_decision.contrast_not_declared", &["quantity_semantics_mismatch"]),
    ("cohort_decision.extrapolative_not_accepted", &["not_supported"]),
    ("cohort_decision.feature_schema_mismatch", &["schema_mismatch"]),
    ("cohort_decision.feature_type_mismatch", &["schema_mismatch"]),
    ("cohort_decision.fitted_model_absent", &["route_not_supported"]),
    ("cohort_decision.fitted_model_invalid", &["invalid_argument"]),
    ("cohort_decision.identity_unexpected", &["invalid_argument"]),
    ("cohort_decision.interval_level_invalid", &["invalid_argument"]),
    ("cohort_decision.interval_not_supported", &["route_not_supported"]),
    ("cohort_decision.interval_requires_native_fit", &["route_not_supported"]),
    ("cohort_decision.invalid_request", &["invalid_argument"]),
    ("cohort_decision.joint_law_required", &["joint_law_required"]),
    ("cohort_decision.law_not_supplied", &["decision_contract_unsatisfied"]),
    ("cohort_decision.layout_invalid", &["invalid_argument"]),
    ("cohort_decision.link_not_identity", &["quantity_semantics_mismatch"]),
    ("cohort_decision.missing_artifact_id", &["invalid_argument"]),
    ("cohort_decision.model_identity_mismatch", &["invalid_argument"]),
    ("cohort_decision.native_fit_refused", &["invalid_argument"]),
    ("cohort_decision.negative_variance", &["invalid_argument"]),
    ("cohort_decision.non_finite_result", &["invalid_argument"]),
    ("cohort_decision.outcome_units_required", &["invalid_argument"]),
    ("cohort_decision.outside_support", &["invalid_argument"]),
    ("cohort_decision.oversized", &["invalid_argument"]),
    ("cohort_decision.parent_attested_not_native", &["invalid_argument"]),
    ("cohort_decision.parent_not_verified", &["invalid_argument"]),
    ("cohort_decision.parent_query_unsupported", &["invalid_argument"]),
    ("cohort_decision.population_mismatch", &["quantity_semantics_mismatch"]),
    ("cohort_decision.prediction_refused", &["invalid_argument"]),
    ("cohort_decision.replay_mismatch", &["invalid_argument"]),
    ("cohort_decision.shift_declaration_mismatch", &["transport_not_certified"]),
    ("cohort_decision.shift_declaration_required", &["transport_missing_evidence"]),
    ("cohort_decision.shift_declaration_unneeded", &["quantity_semantics_mismatch"]),
    ("cohort_decision.shift_support_failure", &["transport_support_failure"]),
    ("cohort_decision.source_invalid", &["invalid_argument"]),
    ("cohort_decision.support_contract_absent", &["invalid_argument"]),
    ("cohort_decision.units_mismatch", &["quantity_semantics_mismatch"]),
    ("cohort_decision.unsupported_version", &["invalid_argument"]),
    ("cohort_decision.weights_invalid", &["invalid_argument"]),
    ("cohort_decision.weights_not_normalized", &["invalid_argument"]),
    ("cohort_decision.wrong_parent_identity", &["invalid_argument"]),
    ("compact_export.container_checksum_mismatch", &["invalid_argument"]),
    ("compact_export.container_invalid", &["invalid_argument"]),
    ("compact_export.covariance_asymmetric", &["invalid_argument"]),
    ("compact_export.covariance_not_psd", &["invalid_argument"]),
    ("compact_export.data_digest_mismatch", &["invalid_argument"]),
    ("compact_export.dimension_mismatch", &["invalid_argument"]),
    ("compact_export.duplicate_term", &["invalid_argument"]),
    ("compact_export.empty_mask_region", &["invalid_argument"]),
    ("compact_export.encode", &["invalid_argument"]),
    ("compact_export.identity_mismatch", &["invalid_argument"]),
    ("compact_export.identity_unexpected", &["invalid_argument"]),
    ("compact_export.inputs_not_canonical", &["invalid_argument"]),
    ("compact_export.invalid_degree", &["invalid_argument"]),
    ("compact_export.invalid_mask_region", &["invalid_argument"]),
    ("compact_export.invalid_quantity", &["invalid_argument"]),
    ("compact_export.invalid_query", &["invalid_argument"]),
    ("compact_export.invalid_request", &["invalid_argument"]),
    ("compact_export.invalid_support", &["invalid_argument"]),
    ("compact_export.layout_invalid", &["invalid_argument"]),
    ("compact_export.mask_not_canonical", &["invalid_argument"]),
    ("compact_export.mask_support_mismatch", &["invalid_argument"]),
    ("compact_export.masked_region", &["invalid_argument"]),
    ("compact_export.missing_artifact_id", &["invalid_argument"]),
    ("compact_export.missing_quantity", &["invalid_argument"]),
    ("compact_export.negative_variance", &["invalid_argument"]),
    ("compact_export.no_terms", &["invalid_argument"]),
    ("compact_export.non_finite_input", &["invalid_argument"]),
    ("compact_export.non_finite_result", &["invalid_argument"]),
    ("compact_export.non_finite_value", &["invalid_argument"]),
    ("compact_export.out_of_support", &["invalid_argument"]),
    ("compact_export.oversized", &["invalid_argument"]),
    ("compact_export.premises_digest_mismatch", &["invalid_argument"]),
    ("compact_export.query_type_mismatch", &["invalid_argument"]),
    ("compact_export.scope_mismatch", &["invalid_argument"]),
    ("compact_export.term_support_mismatch", &["invalid_argument"]),
    ("compact_export.terms_not_canonical", &["invalid_argument"]),
    ("compact_export.too_many_inputs", &["invalid_argument"]),
    ("compact_export.too_many_levels", &["invalid_argument"]),
    ("compact_export.too_many_mask_regions", &["invalid_argument"]),
    ("compact_export.too_many_terms", &["invalid_argument"]),
    ("compact_export.unknown_level", &["invalid_argument"]),
    ("compact_export.unknown_quantity", &["invalid_argument"]),
    ("compact_export.unsupported_version", &["invalid_argument"]),
    ("compiled_categorical_posterior.cancelled", &["cancelled_no_claim"]),
    ("compiled_categorical_posterior.candidate", &["invalid_argument"]),
    ("compiled_categorical_posterior.execution_refused", &["invalid_argument"]),
    ("compiled_categorical_posterior.resource_limit", &["execution_resource_limit"]),
    ("compiled_categorical_posterior.source_invalid", &["invalid_argument"]),
    ("composite_recalc.invalid_request", &["invalid_argument"]),
    ("composite_recalc.native_state_mismatch", &["invalid_argument"]),
    ("composite_recalc.receipt_invalid", &["invalid_argument"]),
    ("composite_recalc.request_unsupported", &["route_not_supported"]),
    ("composition_boundary.artifact_rebuild_failed", &["invalid_argument"]),
    ("composition_boundary.atom_probabilities_invalid", &["decision_contract_unsatisfied"]),
    ("composition_boundary.conflicting_atoms_not_averaged", &["scenario_aggregate_not_supported"]),
    ("composition_boundary.coordinate_missing", &["decision_contract_unsatisfied"]),
    ("composition_boundary.coordinate_unsupported", &["decision_contract_unsatisfied"]),
    ("composition_boundary.dependence_route_not_supported", &["sampling_dependence_unknown"]),
    ("composition_boundary.invalid_input", &["invalid_argument"]),
    ("composition_boundary.mean_is_not_a_distribution", &["decision_contract_unsatisfied"]),
    ("composition_boundary.metadata_only_native_claim", &["attested_not_reverifiable"]),
    ("composition_boundary.native_required", &["attested_not_reverifiable"]),
    ("composition_boundary.non_affine_needs_joint_law", &["decision_contract_unsatisfied"]),
    ("composition_boundary.operation_not_declared", &["invalid_argument"]),
    ("composition_boundary.outcome_law_needs_joint_law", &["decision_contract_unsatisfied"]),
    ("composition_boundary.paired_draws_across_sources", &["joint_law_required"]),
    ("composition_boundary.shared_evidence_not_independent", &["scenario_aggregate_not_supported"]),
    ("composition_boundary.unknown_dependence_is_not_independence", &["sampling_dependence_unknown"]),
    ("composition_boundary.unknown_input", &["invalid_argument"]),
    ("composition_boundary.unsupported_action_not_comparable", &["decision_contract_unsatisfied"]),
    ("composition_boundary.verification_receipt_missing", &["external_verification_failed"]),
    ("composition_bundle.callback_unavailable", &["external_capability_missing"]),
    ("composition_bundle.edge_digest_mismatch", &["external_binding_mismatch"]),
    ("composition_bundle.expected_identity_mismatch", &["external_binding_mismatch"]),
    ("composition_bundle.graph_or_snapshot_mismatch", &["external_binding_mismatch"]),
    ("composition_bundle.incompatible_version", &["schema_mismatch"]),
    ("composition_bundle.invalid_container", &["invalid_argument"]),
    ("composition_bundle.node_not_found", &["invalid_argument"]),
    ("composition_bundle.oversized", &["invalid_argument"]),
    ("composition_bundle.provider_request_changed", &["external_binding_mismatch"]),
    ("composition_bundle.swapped_evidence", &["external_binding_mismatch"]),
    ("composition_bundle.tampered_quantity", &["quantity_semantics_mismatch"]),
    ("composition_bundle.unknown_node_kind", &["invalid_argument"]),
    ("composition_bundle.unsupported_law", &["joint_law_required"]),
    ("composition_provenance.digest_mismatch", &["external_binding_mismatch"]),
    ("composition_provenance.duplicate_link", &["invalid_argument"]),
    ("composition_provenance.invalid_link", &["invalid_argument"]),
    ("composition_provenance.missing_stage", &["external_binding_mismatch"]),
    ("composition_provenance.unknown_link", &["external_binding_mismatch"]),
    ("composition_provenance.unresolved_parent", &["external_binding_mismatch"]),
    ("composition_verification.artifact_invalid", &["invalid_argument"]),
    ("composition_verification.cancelled", &["cancelled_no_claim"]),
    ("composition_verification.cycle", &["invalid_argument"]),
    ("composition_verification.duplicate_node", &["invalid_argument"]),
    ("composition_verification.empty_unsupported", &["invalid_argument"]),
    ("composition_verification.graph_invalid", &["invalid_argument"]),
    ("composition_verification.identity_mismatch", &["route_not_supported"]),
    ("composition_verification.inherited_without_edge", &["invalid_argument"]),
    ("composition_verification.input_retired", &["claim_support_withdrawn"]),
    ("composition_verification.invalid_text", &["invalid_argument"]),
    ("composition_verification.ledger_mismatch", &["invalid_argument"]),
    ("composition_verification.limit_exceeded", &["invalid_argument"]),
    ("composition_verification.provider_trust_withdrawn", &["claim_support_withdrawn"]),
    ("composition_verification.recompute_required", &["claim_revalidation_required"]),
    ("composition_verification.replay_mismatch", &["invalid_argument"]),
    ("composition_verification.self_dependency", &["invalid_argument"]),
    ("composition_verification.stage_not_in_plan", &["invalid_argument"]),
    ("composition_verification.stage_unexecutable", &["claim_support_withdrawn"]),
    ("composition_verification.support_withdrawn", &["claim_support_withdrawn"]),
    ("composition_verification.unknown_claim", &["invalid_argument"]),
    ("composition_verification.unknown_node", &["invalid_argument"]),
    ("composition_verification.unknown_result", &["invalid_argument"]),
    ("conditional_study_ranking.action_not_unique", &["route_not_supported"]),
    ("conditional_study_ranking.artifact_invalid", &["invalid_argument"]),
    ("conditional_study_ranking.invalid_policy", &["invalid_argument"]),
    ("conditional_study_ranking.original_ranking_refused", &["invalid_argument"]),
    ("conditional_study_ranking.resource_limit", &["invalid_argument"]),
    ("conditional_study_ranking.source_mismatch", &["invalid_argument"]),
    ("conditional_study_ranking.source_unavailable", &["route_not_supported"]),
    ("constancy_posterior.band_not_constructed", &["route_not_supported"]),
    ("constancy_posterior.empty_contrast", &["invalid_argument"]),
    ("constancy_posterior.external_power_prior", &["route_not_supported"]),
    ("constancy_posterior.mapped_prior", &["route_not_supported"]),
    ("constancy_posterior.non_positive_scale", &["invalid_argument"]),
    ("coordinate_support.functional_not_described", &["route_not_supported"]),
    ("coordinate_support.invalid_grid", &["invalid_argument"]),
    ("coordinate_support.missing_evidence_relabeled", &["quantity_semantics_mismatch"]),
    ("coordinate_support.point_count_mismatch", &["invalid_argument"]),
    ("coordinate_support.regime_not_describable", &["route_not_supported"]),
    ("coordinate_support.summary_not_worst_label", &["quantity_semantics_mismatch"]),
    ("coordinate_support.temporal_path_not_described", &["route_not_supported"]),
    ("coordinate_support.units_required", &["invalid_argument"]),
    ("coordinate_support.unknown_variable", &["unknown_variable"]),
    ("counterfactual_id.bounds_exceeded", &["route_not_supported"]),
    ("counterfactual_id.budget", &["transport_budget_cancel"]),
    ("counterfactual_id.conflicting_subscripts", &["route_not_supported"]),
    ("counterfactual_id.counterfactual_hedge", &["route_not_supported"]),
    ("counterfactual_id.graph_outside_contract", &["route_not_supported"]),
    ("counterfactual_id.interval_requested", &["estimator_inference_mismatch"]),
    ("counterfactual_id.invalid_artifact", &["invalid_argument"]),
    ("counterfactual_id.invalid_query", &["invalid_argument"]),
    ("counterfactual_id.path_specific_not_supported", &["route_not_supported"]),
    ("counterfactual_id.positivity_violation", &["invalid_argument"]),
    ("counterfactual_id.query_outside_contract", &["route_not_supported"]),
    ("counterfactual_transport.bounds_exceeded", &["route_not_supported"]),
    ("counterfactual_transport.budget", &["transport_budget_cancel"]),
    ("counterfactual_transport.checked_obstruction", &["transport_proven_non_transportable"]),
    ("counterfactual_transport.coupling_unsupported", &["route_not_supported"]),
    ("counterfactual_transport.evidence_incompatible", &["transport_numerical_failure"]),
    ("counterfactual_transport.factor_not_derivable", &["transport_not_certified"]),
    ("counterfactual_transport.graph_outside_contract", &["route_not_supported"]),
    ("counterfactual_transport.invalid_artifact", &["invalid_argument"]),
    ("counterfactual_transport.invalid_catalog", &["invalid_argument"]),
    ("counterfactual_transport.invalid_derivation", &["transport_not_certified"]),
    ("counterfactual_transport.invalid_evidence", &["invalid_argument"]),
    ("counterfactual_transport.invalid_query", &["invalid_argument"]),
    ("counterfactual_transport.invalid_witness", &["transport_not_certified"]),
    ("counterfactual_transport.missing_law", &["transport_missing_provider"]),
    ("counterfactual_transport.missing_regime", &["transport_missing_evidence"]),
    ("counterfactual_transport.no_sampled_inference", &["route_not_supported"]),
    ("counterfactual_transport.selected_sample_law", &["route_not_supported"]),
    ("counterfactual_transport.support_failure", &["transport_support_failure"]),
    ("counterfactual_transport.zero_conditioning_mass", &["transport_support_failure"]),
    ("counterfactual_transport_sampled.bounds_exceeded", &["route_not_supported"]),
    ("counterfactual_transport_sampled.cancelled", &["transport_budget_cancel"]),
    ("counterfactual_transport_sampled.envelope_budget", &["transport_budget_cancel"]),
    ("counterfactual_transport_sampled.envelope_undefined", &["transport_support_failure"]),
    ("counterfactual_transport_sampled.interval_not_computable", &["transport_support_failure"]),
    ("counterfactual_transport_sampled.invalid_artifact", &["invalid_argument"]),
    ("counterfactual_transport_sampled.invalid_counts", &["invalid_argument"]),
    ("counterfactual_transport_sampled.invalid_overlap", &["invalid_argument"]),
    ("counterfactual_transport_sampled.missing_table", &["transport_missing_provider"]),
    ("counterfactual_transport_sampled.overlap_declared_independent", &["evidence_double_counted"]),
    ("counterfactual_transport_sampled.overlap_unknown", &["sampling_dependence_unknown"]),
    ("counterfactual_transport_sampled.sample_size_below_limit", &["transport_support_failure"]),
    ("counterfactual_transport_sampled.shared_units_unlinked", &["route_not_supported"]),
    ("counterfactual_transport_sampled.sparsity_above_limit", &["transport_support_failure"]),
    ("counterfactual_transport_sampled.table_not_cited", &["invalid_argument"]),
    ("counterfactual_transport_sampled.unseen_cell", &["transport_support_failure"]),
    ("cpdag_scenarios.cancelled_not_exportable", &["cancelled_no_claim"]),
    ("cpdag_scenarios.consumer_limit_exceeded", &["transport_budget_cancel"]),
    ("cpdag_scenarios.data_identity_mismatch", &["invalid_argument"]),
    ("cpdag_scenarios.evidence_identity_mismatch", &["invalid_argument"]),
    ("cpdag_scenarios.exact_laws_only", &["route_not_supported"]),
    ("cpdag_scenarios.no_evidence_binding", &["invalid_argument"]),
    ("cpdag_scenarios.not_a_cpdag", &["invalid_argument"]),
    ("cpdag_scenarios.premises_mismatch", &["invalid_argument"]),
    ("cpdag_scenarios.provider_mismatch", &["invalid_argument"]),
    ("cpdag_scenarios.report_replay_mismatch", &["invalid_argument"]),
    ("cpdag_scenarios.selection_or_latent", &["route_not_supported"]),
    ("cpdag_scenarios.unsupported_semantics", &["route_not_supported"]),
    ("cross_world.graph_outside_contract", &["route_not_supported"]),
    ("cross_world.interval_requested", &["estimator_inference_mismatch"]),
    ("cross_world.invalid_query", &["invalid_argument"]),
    ("cross_world.query_outside_contract", &["route_not_supported"]),
    ("cross_world.recanting_witness", &["cross_world_not_identified"]),
    ("cross_world_execution.artifact_changed", &["route_not_supported"]),
    ("cross_world_execution.bounds_exceeded", &["invalid_argument"]),
    ("cross_world_execution.cancelled", &["transport_budget_cancel"]),
    ("cross_world_execution.cell_not_in_matrix", &["route_not_supported"]),
    ("cross_world_execution.elimination_bound_exceeded", &["route_not_supported"]),
    ("cross_world_execution.envelope_member_structure_differs", &["invalid_argument"]),
    ("cross_world_execution.envelope_needs_parameter_set", &["route_not_supported"]),
    ("cross_world_execution.envelope_parameter_set_empty", &["invalid_argument"]),
    ("cross_world_execution.envelope_reading_required", &["invalid_argument"]),
    ("cross_world_execution.evidence_outside_model_support", &["route_not_supported"]),
    ("cross_world_execution.evidence_probability_underflow", &["route_not_supported"]),
    ("cross_world_execution.finite_fitting_not_executed", &["route_not_supported"]),
    ("cross_world_execution.incompatible_history", &["route_not_supported"]),
    ("cross_world_execution.invalid_artifact", &["invalid_argument"]),
    ("cross_world_execution.invalid_evidence", &["invalid_argument"]),
    ("cross_world_execution.invalid_identity", &["invalid_argument"]),
    ("cross_world_execution.invalid_quantity", &["invalid_argument"]),
    ("cross_world_execution.invalid_request", &["invalid_argument"]),
    ("cross_world_execution.invalid_scm", &["invalid_argument"]),
    ("cross_world_execution.invalid_world", &["invalid_argument"]),
    ("cross_world_execution.joint_state_bound_exceeded", &["route_not_supported"]),
    ("cross_world_execution.latent_confounding_not_identified", &["cross_world_not_identified"]),
    ("cross_world_execution.mechanisms_not_declared", &["cross_world_not_identified"]),
    ("cross_world_execution.no_interval_method", &["sampling_dependence_unknown"]),
    ("cross_world_execution.no_posterior_method", &["uncertainty_representation_insufficient"]),
    ("cross_world_execution.no_transported_row_derivation", &["route_not_supported"]),
    ("cross_world_execution.posterior_weight_underflow", &["route_not_supported"]),
    ("cross_world_execution.sampled_abduction_not_supported", &["estimator_inference_mismatch"]),
    ("cross_world_execution.soft_intervention", &["route_not_supported"]),
    ("cross_world_execution.spread_needs_enumeration", &["route_not_supported"]),
    ("cross_world_uncertainty.account_mismatch", &["invalid_argument"]),
    ("cross_world_uncertainty.bounds_exceeded", &["invalid_argument"]),
    ("cross_world_uncertainty.incompatible_history", &["route_not_supported"]),
    ("cross_world_uncertainty.no_transported_path_derivation", &["transport_not_certified"]),
    ("cross_world_uncertainty.result_mismatch", &["invalid_argument"]),
    ("cross_world_uncertainty.sampled_path_not_supported", &["estimator_inference_mismatch"]),
    ("cross_world_uncertainty.unit_sampling_undeclared", &["sampling_dependence_unknown"]),
    ("decision_adapters.bayes_over_identified_set", &["decision_contract_unsatisfied"]),
    ("decision_adapters.claim_draws_too_large", &["invalid_argument"]),
    ("decision_adapters.claims_sources_mismatch", &["invalid_argument"]),
    ("decision_adapters.completion_count_not_probability", &["decision_contract_unsatisfied"]),
    ("decision_adapters.criterion_not_supported_for_set", &["decision_contract_unsatisfied"]),
    ("decision_adapters.duplicate_action", &["invalid_argument"]),
    ("decision_adapters.evaluated_needs_draws", &["invalid_argument"]),
    ("decision_adapters.graph_claims_carry_no_probability", &["invalid_argument"]),
    ("decision_adapters.invalid_claims", &["invalid_argument"]),
    ("decision_adapters.invalid_interval", &["invalid_argument"]),
    ("decision_adapters.invalid_probability", &["invalid_argument"]),
    ("decision_adapters.invalid_utilities", &["invalid_argument"]),
    ("decision_adapters.missing_action", &["invalid_argument"]),
    ("decision_adapters.point_claim_is_single", &["invalid_argument"]),
    ("decision_adapters.policy_mismatch", &["decision_contract_unsatisfied"]),
    ("decision_adapters.probabilities_required", &["decision_contract_unsatisfied"]),
    ("decision_adapters.unevaluated_needs_reason", &["invalid_argument"]),
    ("decision_adapters.unknown_action", &["invalid_argument"]),
    ("decision_adapters.unknown_claim_kind", &["invalid_argument"]),
    ("decision_adapters.unknown_claim_status", &["invalid_argument"]),
    ("decision_adapters.unknown_support_status", &["invalid_argument"]),
    ("decision_admissibility.duplicate_rule", &["invalid_argument"]),
    ("decision_admissibility.invalid_rule", &["invalid_argument"]),
    ("decision_admissibility.unknown_action", &["invalid_argument"]),
    ("decision_admissibility.unknown_input", &["invalid_argument"]),
    ("decision_claims.identity_mismatch", &["invalid_argument"]),
    ("decision_claims.missing_scenario_utility", &["decision_contract_unsatisfied"]),
    ("decision_claims.probabilities_not_declared", &["decision_contract_unsatisfied"]),
    ("decision_claims.range_unavailable", &["decision_contract_unsatisfied"]),
    ("decision_claims.unresolved_scenarios", &["decision_contract_unsatisfied"]),
    ("decision_claims.unsupported_claim_shape", &["invalid_argument"]),
    ("decision_contract.duplicate_id", &["invalid_argument"]),
    ("decision_contract.invalid_declaration", &["invalid_argument"]),
    ("decision_contract.invalid_parameter", &["invalid_argument"]),
    ("decision_contract.invalid_quantity", &["quantity_semantics_mismatch"]),
    ("decision_contract.missing_source", &["decision_contract_unsatisfied"]),
    ("decision_contract.outside_decision_scope", &["quantity_semantics_mismatch"]),
    ("decision_contract.unknown_action", &["invalid_argument"]),
    ("decision_contract.unknown_input", &["invalid_argument"]),
    ("decision_evaluation.analytic_law_unsupported", &["route_not_supported"]),
    ("decision_evaluation.distribution_meaning", &["distribution_meaning_mismatch"]),
    ("decision_evaluation.joint_law_required", &["joint_law_required"]),
    ("decision_evaluation.mean_source_insufficient", &["decision_contract_unsatisfied"]),
    ("decision_evaluation.mean_source_not_replayable", &["route_not_supported"]),
    ("decision_evaluation.non_finite_utility", &["decision_contract_unsatisfied"]),
    ("decision_evaluation.quantity_not_found", &["quantity_semantics_mismatch"]),
    ("decision_evaluation.structure_inputs_required", &["route_not_supported"]),
    ("decision_evaluation.unsupported_coordinate", &["quantity_semantics_mismatch"]),
    ("decision_evidence.adjustment_not_admissible", &[]),
    ("decision_evidence.cancelled", &[]),
    ("decision_evidence.dependence_unknown", &[]),
    ("decision_evidence.deployment_bridge_missing", &[]),
    ("decision_evidence.frame_shape", &[]),
    ("decision_evidence.incompatible_horizon", &[]),
    ("decision_evidence.incompatible_intervention", &[]),
    ("decision_evidence.incompatible_population", &["decision_evidence_scope_mismatch"]),
    ("decision_evidence.interference_unspecified", &[]),
    ("decision_evidence.invalid_declaration", &[]),
    ("decision_evidence.resource_limit", &[]),
    ("decision_evidence.too_many_unknowns", &[]),
    ("decision_evidence.unsupported_assignment_design", &[]),
    ("decision_joint_scenarios.carrier", &["decision_contract_unsatisfied"]),
    ("decision_posterior.band_not_constructed", &["route_not_supported"]),
    ("decision_posterior.empty_law", &["invalid_argument"]),
    ("decision_posterior.external_power_prior", &["route_not_supported"]),
    ("decision_posterior.mapped_prior", &["route_not_supported"]),
    ("decision_posterior.non_positive_scale", &["invalid_argument"]),
    ("decision_reliability.artifact", &["invalid_argument"]),
    ("decision_reliability.artifact_id_missing", &["invalid_argument"]),
    ("decision_reliability.artifact_kind_mismatch", &["invalid_argument"]),
    ("decision_reliability.cancelled", &["cancelled_no_claim"]),
    ("decision_reliability.candidate_set_mismatch", &["invalid_argument"]),
    ("decision_reliability.combined_uncertainty", &["uncertainty_components_not_combined"]),
    ("decision_reliability.decision_mismatch", &["invalid_argument"]),
    ("decision_reliability.dependence_unknown", &["sampling_dependence_unknown"]),
    ("decision_reliability.draws_digest_mismatch", &["invalid_argument"]),
    ("decision_reliability.effective_sample_size_too_small", &["route_not_supported"]),
    ("decision_reliability.identity_mismatch", &["invalid_argument"]),
    ("decision_reliability.invalid_input", &["invalid_argument"]),
    ("decision_reliability.joint_law_required", &["joint_law_required"]),
    ("decision_reliability.memory_budget_exceeded", &["invalid_argument"]),
    ("decision_reliability.numerical", &["invalid_argument"]),
    ("decision_reliability.producer_mean_disagrees", &["distribution_meaning_mismatch"]),
    ("decision_reliability.producer_point_only", &["joint_law_required"]),
    ("decision_reliability.report_mismatch", &["invalid_argument"]),
    ("decision_reliability.route_not_supported", &["route_not_supported"]),
    ("decision_reliability.search_record_inconsistent", &["invalid_argument"]),
    ("decision_reliability.search_record_required", &["candidate_search_record_required"]),
    ("decision_reliability.selection_stream_mismatch", &["invalid_argument"]),
    ("decision_reliability.utility_unit_mismatch", &["invalid_argument"]),
    ("decision_reliability.validation_not_independent", &["validation_stream_not_independent"]),
    ("decision_reliability.validation_stream_mismatch", &["invalid_argument"]),
    ("decision_robustness.contract_mismatch", &["decision_contract_unsatisfied"]),
    ("decision_robustness.invalid_profile", &["invalid_argument"]),
    ("decision_structural.invalid_atoms", &["invalid_argument"]),
    ("decision_structural.invalid_probabilities", &["decision_contract_unsatisfied"]),
    ("decision_structural.probabilities_required", &["decision_contract_unsatisfied"]),
    ("descriptive_comparison.admission_refused", &[]),
    ("descriptive_comparison.arm_not_populated", &["arm_not_populated"]),
    ("descriptive_comparison.arm_too_small", &["invalid_argument"]),
    ("descriptive_comparison.cancelled", &["cancelled_no_claim"]),
    ("descriptive_comparison.column_attribution_not_identified", &["effect_not_identified"]),
    ("descriptive_comparison.estimand_coding_mismatch", &["route_not_supported"]),
    ("descriptive_comparison.fold_exceeds_lease", &[]),
    ("descriptive_comparison.gap_interval_unavailable", &["route_not_supported"]),
    ("descriptive_comparison.invalid_covariance", &["invalid_argument"]),
    ("descriptive_comparison.invalid_data", &["invalid_argument"]),
    ("descriptive_comparison.joint_covariance_unavailable", &["required_option_missing"]),
    ("descriptive_comparison.learner_exceeds_lease", &[]),
    ("descriptive_comparison.population_not_all_observed", &["population_not_estimable"]),
    ("descriptive_comparison.scale_not_supported", &["route_not_supported"]),
    ("descriptive_comparison.transform_domain", &["invalid_argument"]),
    ("design_ranking.bounds_exceeded", &["route_not_supported"]),
    ("design_ranking.contract_mismatch", &["design_signal_invalid"]),
    ("design_ranking.cost_units_mismatch", &["design_cost_units_mismatch"]),
    ("design_ranking.design_error", &["design_signal_invalid"]),
    ("design_ranking.digest_mismatch", &["design_signal_invalid"]),
    ("design_ranking.duplicate_candidate", &["design_signal_invalid"]),
    ("design_ranking.empty", &["invalid_argument"]),
    ("design_ranking.identity_mismatch", &["design_signal_invalid"]),
    ("design_ranking.invalid_artifact", &["invalid_argument"]),
    ("design_ranking.invalid_request", &["invalid_argument"]),
    ("design_ranking.rank_mismatch", &["design_signal_invalid"]),
    ("design_ranking.replay_mismatch", &["design_signal_invalid"]),
    ("design_ranking.signal_inconsistent", &["design_signal_invalid"]),
    ("design_ranking.signal_mismatch", &["design_signal_invalid"]),
    ("design_ranking.source_digest_mismatch", &["design_signal_invalid"]),
    ("design_ranking.source_overlap", &["design_signal_invalid"]),
    ("design_ranking.source_overlap_unknown", &["evidence_overlap_unknown"]),
    ("design_ranking.unsupported_version", &["route_not_supported"]),
    ("discovered_transport.class_enumeration_partial", &["discovered_structure_unresolved"]),
    ("discovered_transport.class_not_identified", &["discovered_structure_unresolved"]),
    ("discovered_transport.class_too_large", &["discovered_structure_unresolved"]),
    ("discovered_transport.evidence_invalid", &["invalid_argument"]),
    ("discovered_transport.member_bound_zero", &["discovered_structure_unresolved"]),
    ("discovered_transport.member_not_estimable", &["discovered_structure_unresolved"]),
    ("discovered_transport.no_members", &["discovered_structure_unresolved"]),
    ("discovered_transport.orientation_conflict", &["discovered_structure_unresolved"]),
    ("discovered_transport.pag_class_ambiguous", &["discovered_structure_unresolved"]),
    ("discovered_transport.pag_selection_bias_edge", &["discovered_structure_unresolved"]),
    ("discovered_transport.search_incomplete", &["discovered_structure_unresolved"]),
    ("discovered_transport.selection_target_unknown", &["discovered_structure_unresolved"]),
    ("discovered_transport.temporal_query_invalid", &["discovered_structure_unresolved"]),
    ("discovered_transport.temporal_structure_unsupported", &["discovered_structure_unresolved"]),
    ("discovered_transport.trial_compatibility_invalid", &["discovered_structure_unresolved"]),
    ("discovered_transport.trial_rejects_every_member", &["discovered_structure_unresolved"]),
    ("discretized_scm.cancelled", &["cancelled_no_claim"]),
    ("discretized_scm.composition_incoherent", &["mechanism_composition_incoherent"]),
    ("discretized_scm.degenerate_marginal", &["route_not_supported"]),
    ("discretized_scm.intervention_invalid", &["invalid_argument"]),
    ("discretized_scm.latent_confounding", &["route_not_supported"]),
    ("discretized_scm.levels_out_of_range", &["invalid_argument"]),
    ("discretized_scm.levels_undeclared", &["invalid_argument"]),
    ("discretized_scm.non_gaussian_mechanism", &["route_not_supported"]),
    ("discretized_scm.numerical", &["invalid_argument"]),
    ("discretized_scm.point_only", &["uncertainty_representation_insufficient"]),
    ("discretized_scm.state_space_too_large", &["invalid_argument"]),
    ("discretized_scm.support_not_real_line", &["route_not_supported"]),
    ("discretized_scm.unknown_variable", &["invalid_argument"]),
    ("discretized_scm_artifact.claimed_numbers_differ", &["invalid_argument"]),
    ("discretized_scm_artifact.identity_mismatch", &["invalid_argument"]),
    ("discretized_scm_artifact.malformed", &["invalid_argument"]),
    ("discretized_scm_artifact.not_exportable", &["route_not_supported"]),
    ("discretized_scm_artifact.rederivation_refused", &["invalid_argument"]),
    ("discretized_scm_artifact.version_unsupported", &["schema_mismatch"]),
    ("distribution_meaning.incompatible_operation", &["distribution_meaning_mismatch"]),
    ("distribution_meaning.legacy_posterior", &["distribution_meaning_mismatch"]),
    ("distribution_meaning.observational_not_interventional", &["quantity_semantics_mismatch"]),
    ("distribution_meaning.posterior_interval", &["distribution_meaning_mismatch"]),
    ("dml.arm_not_populated", &["arm_not_populated"]),
    ("dml.cancelled", &["cancelled_no_claim"]),
    ("dose_grid.bandwidth_outside_range", &["invalid_argument"]),
    ("dose_grid.cancelled", &["transport_budget_cancel"]),
    ("dose_grid.consumer_limit_exceeded", &["invalid_argument"]),
    ("dose_grid.data_identity_mismatch", &["invalid_argument"]),
    ("dose_grid.derivative_without_claim", &["option_not_applicable"]),
    ("dose_grid.graph_not_certified", &["effect_not_identified"]),
    ("dose_grid.insufficient_local_weight", &["positivity_violated"]),
    ("dose_grid.invalid_lag", &["invalid_argument"]),
    ("dose_grid.invalid_request", &["invalid_argument"]),
    ("dose_grid.level_without_claim", &["option_not_applicable"]),
    ("dose_grid.no_simultaneous_band_method", &["route_not_supported"]),
    ("dose_grid.non_finite_result", &["transport_numerical_failure"]),
    ("dose_grid.premises_mismatch", &["invalid_argument"]),
    ("dose_grid.refused", &["invalid_argument"]),
    ("dose_grid.resource_limit", &["transport_budget_cancel"]),
    ("dose_grid.resource_refused", &["transport_budget_cancel"]),
    ("dose_grid.result_replay_mismatch", &["invalid_argument"]),
    ("dose_grid.too_few_clusters", &["invalid_argument"]),
    ("dose_grid.undecodable", &["invalid_argument"]),
    ("dose_grid.unsupported_dose", &["positivity_violated"]),
    ("dose_grid.unsupported_semantics", &["route_not_supported"]),
    ("dose_grid_posterior.band_not_constructed", &["route_not_supported"]),
    ("dose_grid_posterior.external_power_prior", &["route_not_supported"]),
    ("dose_grid_posterior.insufficient_weight", &["positivity_violated"]),
    ("dose_grid_posterior.mapped_prior", &["route_not_supported"]),
    ("dose_grid_posterior.non_positive_scale", &["invalid_argument"]),
    ("dose_response.bootstrap_below_floor", &["estimator_inference_mismatch"]),
    ("dose_response.bounds_exceeded", &["route_not_supported"]),
    ("dose_response.cate_or_simultaneous", &["route_not_supported"]),
    ("dose_response.declared_folds_unbalanced", &["invalid_argument"]),
    ("dose_response.dose_density_floor", &["transport_support_failure"]),
    ("dose_response.dose_density_invalid", &["transport_support_failure"]),
    ("dose_response.dose_too_discrete", &["treatment_support_too_discrete"]),
    ("dose_response.estimated_dose_density", &["route_not_supported"]),
    ("dose_response.grid_outside_dose_support", &["transport_support_failure"]),
    ("dose_response.invalid_declared_rows", &["invalid_argument"]),
    ("dose_response.kernel_not_supported", &["route_not_supported"]),
    ("dose_response.learner_not_portable", &["route_not_supported"]),
    ("dose_response.local_dose_ess", &["transport_support_failure"]),
    ("dose_response.membership_overlap", &["transport_support_failure"]),
    ("dose_response.non_iid_design", &["sampling_dependence_unknown"]),
    ("dose_response.posterior_declared_rows", &["route_not_supported"]),
    ("dose_response.quadrature_tolerance", &["transport_numerical_failure"]),
    ("dose_response.target_not_smoothed", &["route_not_supported"]),
    ("effect_constancy.estimand_missing", &["invalid_argument"]),
    ("effect_constancy.incompatible_partitions", &["route_not_supported"]),
    ("effect_constancy.invalid_alpha", &["invalid_argument"]),
    ("effect_constancy.invalid_covariance", &["invalid_argument"]),
    ("effect_constancy.invalid_identity", &["invalid_argument"]),
    ("effect_constancy.invalid_request", &["invalid_argument"]),
    ("effect_constancy.invalid_standard_error", &["invalid_argument"]),
    ("effect_constancy.limits_exceeded", &["invalid_argument"]),
    ("effect_constancy.non_finite_estimate", &["invalid_argument"]),
    ("effect_constancy.too_few_partitions", &["invalid_argument"]),
    ("effect_constancy.too_many_partitions", &["invalid_argument"]),
    ("effect_constancy.unknown_reference", &["invalid_argument"]),
    ("effect_constancy.unsupported_partition", &["route_not_supported"]),
    ("effect_constancy.wrong_contract", &["route_not_supported"]),
    ("effect_constancy_consumer.artifact_invalid", &["invalid_argument"]),
    ("effect_constancy_consumer.catalog_binding_invalid", &["invalid_argument"]),
    ("effect_constancy_consumer.contrast_missing", &["invalid_argument"]),
    ("effect_constancy_consumer.decision_unsupported", &["invalid_argument"]),
    ("effect_constancy_consumer.invalid_request", &["invalid_argument"]),
    ("effect_constancy_consumer.limits_exceeded", &["invalid_argument"]),
    ("effect_constancy_consumer.partition_missing", &["invalid_argument"]),
    ("effect_constancy_consumer.quantity_mismatch", &["invalid_argument"]),
    ("effect_constancy_consumer.support_unsupported", &["route_not_supported"]),
    ("evidence_ledger.cancelled", &["cancelled_no_claim"]),
    ("evidence_ledger.diagonal_covariance_shared_rows", &["evidence_double_counted"]),
    ("evidence_ledger.duplicate_result", &["invalid_argument"]),
    ("evidence_ledger.duplicate_row", &["evidence_double_counted"]),
    ("evidence_ledger.duplicate_source", &["evidence_double_counted"]),
    ("evidence_ledger.identity_mismatch", &["route_not_supported"]),
    ("evidence_ledger.interval_construction_unavailable", &["route_not_supported"]),
    ("evidence_ledger.interval_report_invalid", &["invalid_argument"]),
    ("evidence_ledger.invalid_declaration", &["invalid_argument"]),
    ("evidence_ledger.kind_mismatch", &["invalid_argument"]),
    ("evidence_ledger.limit_exceeded", &["invalid_argument"]),
    ("evidence_ledger.lineage_rows_mismatch", &["invalid_argument"]),
    ("evidence_ledger.native_and_transported", &["evidence_double_counted"]),
    ("evidence_ledger.operation_arity", &["invalid_argument"]),
    ("evidence_ledger.overlap_unknown", &["evidence_overlap_unknown"]),
    ("evidence_ledger.reentry_derived_from_used_data", &["evidence_double_counted"]),
    ("evidence_ledger.reentry_duplicate", &["evidence_double_counted"]),
    ("evidence_ledger.reentry_prior_plus_likelihood", &["evidence_double_counted"]),
    ("evidence_ledger.reentry_transform", &["evidence_double_counted"]),
    ("evidence_ledger.repeated_source", &["evidence_double_counted"]),
    ("evidence_ledger.replay_mismatch", &["invalid_argument"]),
    ("evidence_ledger.result_not_valid", &["evidence_source_withdrawn"]),
    ("evidence_ledger.shared_observations", &["evidence_double_counted"]),
    ("evidence_ledger.source_superseded", &["evidence_source_withdrawn"]),
    ("evidence_ledger.source_withdrawn", &["evidence_source_withdrawn"]),
    ("evidence_ledger.supersede_kind_mismatch", &["invalid_argument"]),
    ("evidence_ledger.synthesis_mismatch", &["invalid_argument"]),
    ("evidence_ledger.transport_identity_missing", &["invalid_argument"]),
    ("evidence_ledger.transported_as_native", &["invalid_argument"]),
    ("evidence_ledger.unknown_result", &["invalid_argument"]),
    ("evidence_ledger.unknown_source", &["invalid_argument"]),
    ("evidence_ledger.update_drops_rows", &["invalid_argument"]),
    ("evidence_obligations.invalid_obligation", &["invalid_argument"]),
    ("evidence_obligations.wrong_contract", &["transport_missing_evidence"]),
    ("evidence_plan.budget", &["transport_budget_cancel"]),
    ("evidence_plan.compile_refused", &["invalid_argument"]),
    ("evidence_plan.cost_units_mismatch", &["design_cost_units_mismatch"]),
    ("evidence_plan.invalid_request", &["invalid_argument"]),
    ("evidence_plan.no_failure_to_plan", &["invalid_argument"]),
    ("evidence_plan.no_route", &["invalid_argument"]),
    ("evidence_plan.none_certified", &["transport_not_certified"]),
    ("evsi.action_set_changed", &["decision_contract_unsatisfied"]),
    ("evsi.bound_violation", &["design_signal_invalid"]),
    ("evsi.cancelled", &["cancelled_no_claim"]),
    ("evsi.cost_map_required", &["design_cost_units_mismatch"]),
    ("evsi.cost_units_mismatch", &["design_cost_units_mismatch"]),
    ("evsi.decision_values_incoherent", &["design_signal_invalid"]),
    ("evsi.decision_values_need_draws", &["design_signal_invalid"]),
    ("evsi.design_error", &["design_signal_invalid"]),
    ("evsi.duplicate_candidate", &["design_signal_invalid"]),
    ("evsi.execution_resource_limit", &["execution_resource_limit"]),
    ("evsi.invalid_cost", &["invalid_argument"]),
    ("evsi.invalid_request", &["invalid_argument"]),
    ("evsi.prior_states_mismatch", &["design_signal_invalid"]),
    ("evsi.receipt_mismatch", &["design_signal_invalid"]),
    ("evsi.sample_size_mismatch", &["design_signal_invalid"]),
    ("evsi.source_overlap", &["design_signal_invalid"]),
    ("evsi.source_overlap_unknown", &["evidence_overlap_unknown"]),
    ("evsi.unscored_candidate", &["design_signal_invalid"]),
    ("external_mechanism.declaration_invalid", &["mechanism_declaration_invalid"]),
    ("external_mechanism.implementation_changed", &["mechanism_implementation_changed"]),
    ("external_mechanism.implementation_unavailable", &["mechanism_implementation_unavailable"]),
    ("external_mechanism.resolver_failed", &["mechanism_resolver_failed"]),
    ("external_mechanism.resolver_missing", &["mechanism_resolver_missing"]),
    ("external_object_verification.duplicate_probe", &["invalid_argument"]),
    ("external_object_verification.failed_probe", &["external_verification_failed"]),
    ("external_object_verification.invalid_probe", &["invalid_argument"]),
    ("external_object_verification.missing_probe", &["external_verification_failed"]),
    ("external_recalc.artifact_invalid", &["invalid_argument"]),
    ("external_recalc.attempt_limit", &["invalid_argument"]),
    ("external_recalc.callback_failed", &["invalid_argument"]),
    ("external_recalc.contract_mismatch", &["invalid_argument"]),
    ("external_recalc.idempotency_conflict", &["invalid_argument"]),
    ("external_recalc.idempotency_unsupported", &["invalid_argument"]),
    ("external_recalc.invalid_descriptor", &["invalid_argument"]),
    ("external_recalc.invalid_output", &["invalid_argument"]),
    ("external_recalc.invalid_request", &["invalid_argument"]),
    ("external_recalc.limits_exceeded", &["invalid_argument"]),
    ("external_recalc.output_unavailable", &["invalid_argument"]),
    ("external_recalc.policy_unsupported", &["route_not_supported"]),
    ("external_recalc.provider_mismatch", &["invalid_argument"]),
    ("external_recalc.provider_unavailable", &["invalid_argument"]),
    ("external_recalc.quantity_unsupported", &["route_not_supported"]),
    ("external_recalc.replay_mismatch", &["invalid_argument"]),
    ("external_recalc.replay_unsupported", &["route_not_supported"]),
    ("external_recalc.retry_unsafe", &["route_not_supported"]),
    ("external_response_binding.column_quantity", &["external_binding_mismatch"]),
    ("external_response_binding.column_support_length", &["invalid_argument"]),
    ("external_response_binding.contract_not_identified", &["effect_not_identified"]),
    ("external_response_binding.coordinate_conditioning", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_functional", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_horizon", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_invalid_coordinate", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_population", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_regime", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_transform", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_units", &["quantity_semantics_mismatch"]),
    ("external_response_binding.coordinate_variable", &["quantity_semantics_mismatch"]),
    ("external_response_binding.dimension", &["external_binding_mismatch"]),
    ("external_response_binding.distribution_meaning", &["distribution_meaning_mismatch"]),
    ("external_response_binding.empty_column", &["invalid_argument"]),
    ("external_response_binding.graph", &["external_binding_mismatch"]),
    ("external_response_binding.invalid_contract", &["invalid_argument"]),
    ("external_response_binding.malformed_artifact", &["invalid_argument"]),
    ("external_response_binding.malformed_capability", &["invalid_argument"]),
    ("external_response_binding.malformed_equivalence", &["invalid_argument"]),
    ("external_response_binding.malformed_identification", &["invalid_argument"]),
    ("external_response_binding.malformed_meaning", &["invalid_argument"]),
    ("external_response_binding.malformed_probe", &["invalid_argument"]),
    ("external_response_binding.malformed_quantity", &["invalid_argument"]),
    ("external_response_binding.malformed_support", &["invalid_argument"]),
    ("external_response_binding.malformed_trust", &["invalid_argument"]),
    ("external_response_binding.missing_assumption", &["external_binding_mismatch"]),
    ("external_response_binding.missing_evidence", &["external_binding_mismatch"]),
    ("external_response_binding.non_finite_value", &["invalid_argument"]),
    ("external_response_binding.object_kind", &["external_binding_mismatch"]),
    ("external_response_binding.object_quantities", &["external_binding_mismatch"]),
    ("external_response_binding.posterior_kind", &["external_binding_mismatch"]),
    ("external_response_binding.unchecked_observational_law", &["external_binding_mismatch"]),
    ("external_scientific_providers.duplicate_declaration", &["invalid_argument"]),
    ("external_scientific_providers.invalid_declaration", &["invalid_argument"]),
    ("external_scientific_providers.invalid_identity", &["invalid_argument"]),
    ("external_trust_receipt.receipt_mismatch", &["external_verification_failed"]),
    ("finite_action_posterior.band_not_constructed", &["route_not_supported"]),
    ("finite_action_posterior.empty_law", &["invalid_argument"]),
    ("finite_action_posterior.external_power_prior", &["route_not_supported"]),
    ("finite_action_posterior.mapped_prior", &["route_not_supported"]),
    ("finite_action_posterior.non_positive_scale", &["invalid_argument"]),
    ("frozen_scores.identity_mismatch", &["route_not_supported"]),
    ("functional_inverse_posterior.band_not_constructed", &["route_not_supported"]),
    ("functional_inverse_posterior.empty_law", &["invalid_argument"]),
    ("functional_inverse_posterior.external_power_prior", &["route_not_supported"]),
    ("functional_inverse_posterior.mapped_prior", &["route_not_supported"]),
    ("functional_inverse_posterior.non_positive_scale", &["invalid_argument"]),
    ("functional_inverse_query.artifact_kind_mismatch", &["invalid_argument"]),
    ("functional_inverse_query.bounds_exceeded", &["inverse_functional_unsupported"]),
    ("functional_inverse_query.forward_origin_invalid", &["inverse_functional_unsupported"]),
    ("functional_inverse_query.global_feasibility_claim", &["inverse_functional_unsupported"]),
    ("functional_inverse_query.invalid_artifact", &["invalid_argument"]),
    ("functional_inverse_query.wrong_contract", &["inverse_functional_unsupported"]),
    ("functional_source.expected_identity_mismatch", &["invalid_argument"]),
    ("functional_source.invalid_artifact", &["invalid_argument"]),
    ("functional_source.invalid_contract", &["invalid_argument"]),
    ("functional_source.invalid_source", &["invalid_argument"]),
    ("functional_source.limits_exceeded", &["invalid_argument"]),
    ("functional_source.missing_action", &["invalid_argument"]),
    ("functional_source.native_source_unavailable", &["invalid_argument"]),
    ("functional_source.original_law_required", &["route_not_supported"]),
    ("functional_source.output_mismatch", &["invalid_argument"]),
    ("functional_source.source_mismatch", &["invalid_argument"]),
    ("graph_falsification.ci_p_value_missing", &["diagnostic_not_available"]),
    ("graph_falsification.ci_result_count", &["diagnostic_not_available"]),
    ("graph_falsification.ci_test_not_frequentist", &["route_not_supported"]),
    ("graph_falsification.data_not_observed", &["invalid_argument"]),
    ("graph_falsification.duplicate_variable", &["invalid_argument"]),
    ("graph_falsification.empty_data", &["invalid_argument"]),
    ("graph_falsification.invalid_level", &["invalid_argument"]),
    ("graph_falsification.invalid_permutations", &["invalid_argument"]),
    ("graph_falsification.node_not_static", &["route_not_supported"]),
    ("graph_falsification.too_few_nodes", &["invalid_argument"]),
    ("graph_falsification.work_limit", &[]),
    ("graph_imputation.atom_failure", &[]),
    ("graph_imputation.atom_not_lowered", &[]),
    ("graph_imputation.cancelled", &["cancelled_no_claim"]),
    ("graph_imputation.candidate_refused", &["invalid_argument"]),
    ("graph_imputation.compiled", &["invalid_argument"]),
    ("graph_imputation.execution_refused", &["invalid_argument"]),
    ("graph_imputation.interval_unsupported", &["route_not_supported"]),
    ("graph_imputation.invalid_source", &[]),
    ("graph_imputation.lowering", &["invalid_argument"]),
    ("graph_imputation.point_unsupported", &[]),
    ("graph_imputation.resource_limit", &["execution_resource_limit"]),
    ("graph_imputation.search_stopped", &["cancelled_no_claim", "execution_resource_limit"]),
    ("graph_imputation.supported_graph_unidentified", &[]),
    ("graph_imputation.unidentified", &[]),
    ("graph_imputation.unscorable", &[]),
    ("graph_imputation.unsupported", &["route_not_supported"]),
    ("graph_imputation.wrong_dispatch", &[]),
    ("graph_selection_imputation.columns", &["invalid_argument"]),
    ("graph_selection_imputation.data", &["invalid_argument"]),
    ("graph_selection_imputation.domain", &["invalid_argument"]),
    ("graph_selection_imputation.interval_not_measured", &["not_supported"]),
    ("graph_selection_imputation.model", &["invalid_argument"]),
    ("graph_selection_imputation.result", &["invalid_argument"]),
    ("graph_selection_imputation.variables", &["invalid_argument"]),
    ("identification_repair.bounds_exceeded", &["invalid_argument"]),
    ("identification_repair.budget", &["transport_budget_cancel"]),
    ("identification_repair.cost_units_mismatch", &["design_cost_units_mismatch"]),
    ("identification_repair.invalid_request", &["invalid_argument"]),
    ("identification_repair.not_a_failure", &["invalid_argument"]),
    ("identification_repair.wrong_contract", &["transport_not_certified"]),
    ("initial_state.invalid_law", &["invalid_argument"]),
    ("initial_state.invalid_premises", &["invalid_argument"]),
    ("initial_state.support_gap", &["transport_support_failure"]),
    ("initial_state.target_law_missing", &["transport_missing_evidence"]),
    ("initial_state.too_many_states", &["route_not_supported"]),
    ("initial_state_posterior.band_not_constructed", &["route_not_supported"]),
    ("initial_state_posterior.empty_law", &["invalid_argument"]),
    ("initial_state_posterior.external_power_prior", &["route_not_supported"]),
    ("initial_state_posterior.mapped_prior", &["route_not_supported"]),
    ("initial_state_posterior.non_positive_scale", &["invalid_argument"]),
    ("interface_contract.artifact_invalid", &["invalid_argument"]),
    ("interface_contract.blank_field", &["interface_contract_invalid"]),
    ("interface_contract.calibration_record_blank", &["interface_contract_invalid"]),
    ("interface_contract.calibration_without_interval", &["interface_contract_invalid"]),
    ("interface_contract.conditioning_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.cyclic_composition", &["interface_contract_cyclic_composition"]),
    ("interface_contract.dependence_unknown", &["interface_contract_assumption_mismatch"]),
    ("interface_contract.duplicate_component", &["interface_contract_invalid"]),
    ("interface_contract.duplicate_port", &["interface_contract_invalid"]),
    ("interface_contract.functional_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.horizon_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.identity_mismatch", &["route_not_supported"]),
    ("interface_contract.inherited_input_unknown", &["interface_contract_invalid"]),
    ("interface_contract.invalid_digest", &["interface_contract_invalid"]),
    ("interface_contract.invalid_quantity", &["interface_contract_invalid"]),
    ("interface_contract.invalid_support", &["interface_contract_invalid"]),
    ("interface_contract.measurement_assumption_mismatch", &["interface_contract_assumption_mismatch"]),
    ("interface_contract.missing_operations", &["interface_contract_invalid"]),
    ("interface_contract.operation_unavailable", &["interface_contract_operation_unavailable"]),
    ("interface_contract.point_only_with_uncertainty", &["interface_contract_invalid"]),
    ("interface_contract.population_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.port_multiply_fed", &["interface_contract_invalid"]),
    ("interface_contract.posterior_without_posterior_meaning", &["interface_contract_invalid"]),
    ("interface_contract.provenance_missing", &["interface_contract_assumption_mismatch"]),
    ("interface_contract.regime_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.semantic_identity_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.shared_evidence", &["interface_contract_assumption_mismatch"]),
    ("interface_contract.support_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.transform_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.uncertainty_meaning_mismatch", &["interface_contract_uncertainty_mismatch"]),
    ("interface_contract.uncertainty_meaning_missing", &["interface_contract_uncertainty_mismatch"]),
    ("interface_contract.uncertainty_meaning_required", &["interface_contract_invalid"]),
    ("interface_contract.uncertainty_meaning_unused", &["interface_contract_invalid"]),
    ("interface_contract.uncertainty_paradigm_mismatch", &["interface_contract_uncertainty_mismatch"]),
    ("interface_contract.units_mismatch", &["interface_contract_quantity_mismatch"]),
    ("interface_contract.unknown_component", &["interface_contract_invalid"]),
    ("interface_contract.unknown_port", &["interface_contract_invalid"]),
    ("interface_contract.unknown_provenance_stage", &["interface_contract_invalid"]),
    ("interface_contract.variable_mismatch", &["interface_contract_quantity_mismatch"]),
    ("intrinsic_influence.correlated_noise", &["route_not_supported"]),
    ("intrinsic_influence.draw_budget_exceeded", &["route_not_supported"]),
    ("intrinsic_influence.invalid_settings", &["invalid_argument"]),
    ("intrinsic_influence.mechanism_unfitted", &["route_not_supported"]),
    ("intrinsic_influence.noise_independence_undeclared", &["route_not_supported"]),
    ("intrinsic_influence.noise_not_row_independent", &["route_not_supported"]),
    ("intrinsic_influence.structure_mismatch", &["invalid_argument"]),
    ("intrinsic_influence.target_not_continuous", &["route_not_supported"]),
    ("inverse.cancelled", &["transport_budget_cancel"]),
    ("inverse.forward_not_mean_response", &["route_not_supported"]),
    ("inverse.forward_not_point_identified", &["route_not_supported"]),
    ("inverse.forward_not_supported", &["not_supported"]),
    ("inverse.invalid_action", &["invalid_argument"]),
    ("inverse.invalid_forward", &["invalid_argument"]),
    ("inverse.invalid_target", &["invalid_argument"]),
    ("inverse.observational_scenarios", &["inverse_functional_unsupported"]),
    ("inverse.probability_target", &["inverse_functional_unsupported"]),
    ("inverse.quantile_target", &["inverse_functional_unsupported"]),
    ("inverse_forward_source.composed_coordinate_is_outcome_law", &["invalid_argument"]),
    ("inverse_forward_source.composed_family_mismatch", &["mechanism_composition_incoherent"]),
    ("inverse_forward_source.composed_law_not_materialized", &["decision_contract_unsatisfied"]),
    ("inverse_forward_source.composed_level_coding", &["mechanism_composition_incoherent"]),
    ("inverse_forward_source.composed_query_result_mismatch", &["invalid_argument"]),
    ("inverse_forward_source.composed_response_invalid", &["mechanism_composition_incoherent"]),
    ("inverse_forward_source.interval_not_supported", &["uncertainty_representation_insufficient"]),
    ("inverse_forward_source.origin_invalid", &["invalid_argument"]),
    ("inverse_forward_source.per_action_not_sealable", &["invalid_argument"]),
    ("inverse_forward_source.point_only_law_requested", &["decision_contract_unsatisfied"]),
    ("inverse_forward_source.transported_not_accepted", &["decision_contract_unsatisfied"]),
    ("inverse_observational_scenarios.condition_on_outcome", &["quantity_semantics_mismatch"]),
    ("inverse_observational_scenarios.coordinate_missing", &["quantity_semantics_mismatch"]),
    ("inverse_observational_scenarios.coordinate_unsupported", &["quantity_semantics_mismatch"]),
    ("inverse_observational_scenarios.event_has_no_mass", &["invalid_argument"]),
    ("inverse_observational_scenarios.invalid_parameter", &["invalid_argument"]),
    ("inverse_observational_scenarios.invalid_scenarios", &["invalid_argument"]),
    ("inverse_observational_scenarios.joint_law_required", &["joint_law_required"]),
    ("inverse_observational_scenarios.meaning_not_observational", &["distribution_meaning_mismatch"]),
    ("inverse_observational_scenarios.non_finite_draw", &["invalid_argument"]),
    ("inverse_observational_scenarios.result_mismatch", &["invalid_argument"]),
    ("inverse_observational_scenarios.unknown_scenario", &["invalid_argument"]),
    ("joint_cells.cancelled", &["cancelled_no_claim"]),
    ("joint_cells.cell_empty", &["arm_not_populated"]),
    ("joint_cells.cell_ess_below_minimum", &["joint_cell_unsupported"]),
    ("joint_cells.columns", &["invalid_argument"]),
    ("joint_cells.component_count", &["invalid_argument"]),
    ("joint_cells.conditional_unsupported", &["joint_cell_unsupported"]),
    ("joint_cells.config", &["invalid_argument"]),
    ("joint_cells.contrast_cell_unsupported", &["joint_cell_unsupported"]),
    ("joint_cells.derived_constituent_copy_in_adjustment", &["derived_treatment_invalid"]),
    ("joint_cells.derived_constituent_in_adjustment", &["derived_treatment_invalid"]),
    ("joint_cells.derived_declaration_invalid", &["derived_treatment_invalid"]),
    ("joint_cells.derived_descendant_in_adjustment", &["derived_treatment_invalid"]),
    ("joint_cells.derived_duplicate_constituents", &["derived_treatment_invalid"]),
    ("joint_cells.derived_exclusion_not_in_adjustment", &["derived_treatment_invalid"]),
    ("joint_cells.derived_exclusion_rule_mismatch", &["derived_treatment_invalid"]),
    ("joint_cells.derived_illegal_value", &["derived_treatment_invalid"]),
    ("joint_cells.derived_intervention_not_defined", &["derived_treatment_invalid"]),
    ("joint_cells.derived_not_a_joint_cell", &["derived_treatment_invalid"]),
    ("joint_cells.derived_outcome_in_adjustment", &["derived_treatment_invalid"]),
    ("joint_cells.derived_rank_deficient", &["design_rank_deficient"]),
    ("joint_cells.derived_treatment_tracked_by_covariates", &["derived_treatment_invalid"]),
    ("joint_cells.derived_unknown_column", &["invalid_argument"]),
    ("joint_cells.learner", &["invalid_argument"]),
    ("joint_cells.learner_unavailable", &["route_not_supported"]),
    ("joint_cells.ml_learner_not_declared", &["ml_nuisance_not_supported"]),
    ("joint_cells.no_interval_method", &["penalized_interval_not_supported"]),
    ("joint_cells.no_ml_interval_method", &["ml_nuisance_not_supported"]),
    ("joint_cells.no_rows", &["invalid_argument"]),
    ("joint_cells.no_supported_cell", &["joint_cell_unsupported"]),
    ("joint_cells.normalization_failed", &["route_not_supported"]),
    ("joint_cells.not_binary", &["invalid_argument"]),
    ("joint_cells.nuisance_provider", &["invalid_argument"]),
    ("joint_cells.ordering", &["invalid_argument"]),
    ("joint_cells.outcome_model_unsupported", &["joint_cell_unsupported"]),
    ("joint_law_propagation.alignment_mismatch", &["joint_law_required"]),
    ("joint_law_propagation.artifact_invalid", &["invalid_argument"]),
    ("joint_law_propagation.artifact_refused", &["invalid_argument"]),
    ("joint_law_propagation.bounds_exceeded", &["route_not_supported"]),
    ("joint_law_propagation.cancelled", &["cancelled_no_claim"]),
    ("joint_law_propagation.covariance_insufficient_for_nonlinear", &["joint_law_required"]),
    ("joint_law_propagation.coverage_record_construction_mismatch", &["quantity_semantics_mismatch"]),
    ("joint_law_propagation.coverage_record_coordinate_mismatch", &["quantity_semantics_mismatch"]),
    ("joint_law_propagation.dependence_unknown", &["sampling_dependence_unknown"]),
    ("joint_law_propagation.envelope_is_not_a_point", &["uncertainty_representation_insufficient"]),
    ("joint_law_propagation.envelope_not_available", &["route_not_supported"]),
    ("joint_law_propagation.evidence_overlap_unknown", &["sampling_dependence_unknown"]),
    ("joint_law_propagation.horizon_composition", &["invalid_argument", "option_not_applicable"]),
    ("joint_law_propagation.horizon_draws_dose", &["invalid_argument"]),
    ("joint_law_propagation.horizon_draws_layout", &["invalid_argument"]),
    ("joint_law_propagation.horizon_draws_mean_curve", &["option_not_applicable"]),
    ("joint_law_propagation.horizon_draws_not_joint", &["joint_law_required"]),
    ("joint_law_propagation.index_pairing_refused", &["joint_law_required"]),
    ("joint_law_propagation.invalid_request", &["invalid_argument"]),
    ("joint_law_propagation.mixture_weights_unknown", &["invalid_argument"]),
    ("joint_law_propagation.no_interval_construction", &["uncertainty_representation_insufficient"]),
    ("joint_law_propagation.no_posterior_for_fixed_law_inputs", &["uncertainty_representation_insufficient"]),
    ("joint_law_propagation.nonlinear_needs_aligned_joint_law", &["joint_law_required"]),
    ("joint_law_propagation.shared_evidence_declared_independent", &["evidence_double_counted"]),
    ("joint_law_propagation.spread_needs_supported_dependence", &["uncertainty_representation_insufficient"]),
    ("joint_law_propagation.structure_not_verified", &["route_not_supported"]),
    ("joint_law_propagation.transport_pieces_incomplete", &["joint_law_required"]),
    ("joint_law_propagation.transported_dependence_required", &["joint_law_required"]),
    ("joint_sensitivity.bounds_exceeded", &["route_not_supported"]),
    ("joint_sensitivity.budget", &["transport_budget_cancel"]),
    ("joint_sensitivity.budget_coupling", &["route_not_supported"]),
    ("joint_sensitivity.cancelled", &["transport_budget_cancel"]),
    ("joint_sensitivity.consumer_limits", &["route_not_supported"]),
    ("joint_sensitivity.data_identity_mismatch", &["transport_not_certified"]),
    ("joint_sensitivity.duplicate_factor", &["invalid_argument"]),
    ("joint_sensitivity.factor_count", &["route_not_supported"]),
    ("joint_sensitivity.fixed_graph_conditional", &["route_not_supported"]),
    ("joint_sensitivity.fixed_graph_parent", &["route_not_supported"]),
    ("joint_sensitivity.formula_incompatible", &["transport_not_certified"]),
    ("joint_sensitivity.incomplete_kernel", &["transport_support_failure"]),
    ("joint_sensitivity.invalid_fraction", &["invalid_argument"]),
    ("joint_sensitivity.invalid_threshold", &["invalid_argument"]),
    ("joint_sensitivity.invalid_tolerance", &["invalid_argument"]),
    ("joint_sensitivity.no_interval_method", &["route_not_supported"]),
    ("joint_sensitivity.premises_mismatch", &["transport_not_certified"]),
    ("joint_sensitivity.provider_mismatch", &["transport_missing_provider"]),
    ("joint_sensitivity.replay_mismatch", &["transport_not_certified"]),
    ("joint_sensitivity.source_target_discrepancy", &["route_not_supported"]),
    ("joint_sensitivity.stage_accounting", &["invalid_argument"]),
    ("joint_sensitivity.treatment_factor", &["invalid_argument"]),
    ("joint_sensitivity.union_labelled_ci", &["estimator_inference_mismatch"]),
    ("joint_sensitivity.unknown_factor", &["invalid_argument"]),
    ("joint_sensitivity.unsupported_domain", &["route_not_supported"]),
    ("latent_class.cancelled", &["cancelled_no_claim"]),
    ("latent_class.constant_outcome", &["population_not_estimable"]),
    ("latent_class.data_required", &["route_not_supported"]),
    ("latent_class.degenerate_class", &["population_not_estimable"]),
    ("latent_class.design_support_too_small", &["design_rank_deficient"]),
    ("latent_class.export_budget_exceeded", &["route_not_supported"]),
    ("latent_class.invalid_config", &["invalid_argument"]),
    ("latent_class.invalid_request", &["invalid_argument"]),
    ("latent_class.length_mismatch", &["invalid_argument"]),
    ("latent_class.likelihood_not_monotone", &["mechanism_fit_not_converged"]),
    ("latent_class.limits_exceeded", &["invalid_argument"]),
    ("latent_class.memory_budget_exceeded", &["route_not_supported"]),
    ("latent_class.non_finite_input", &["invalid_argument"]),
    ("latent_class.not_converged", &["mechanism_fit_not_converged"]),
    ("latent_class.randomization_not_declared", &["required_option_missing"]),
    ("latent_class.report_replay_mismatch", &["invalid_argument"]),
    ("latent_class.too_few_rows", &["invalid_argument"]),
    ("latent_class.treatment_constant", &["effect_not_identified"]),
    ("latent_class.weak_class", &["population_not_estimable"]),
    ("latent_class.wrong_contract", &["route_not_supported"]),
    ("latent_class_posterior.band_not_constructed", &["route_not_supported"]),
    ("latent_class_posterior.empty_design", &["invalid_argument"]),
    ("latent_class_posterior.external_power_prior", &["route_not_supported"]),
    ("latent_class_posterior.mapped_prior", &["route_not_supported"]),
    ("latent_class_posterior.non_positive_scale", &["invalid_argument"]),
    ("learned_continuous_posterior.band_not_constructed", &["route_not_supported"]),
    ("learned_continuous_posterior.empty_scores", &["invalid_argument"]),
    ("learned_continuous_posterior.external_power_prior", &["route_not_supported"]),
    ("learned_continuous_posterior.mapped_prior", &["route_not_supported"]),
    ("learned_continuous_posterior.non_positive_scale", &["invalid_argument"]),
    ("learned_joint_transport.admission_refused", &["transport_budget_cancel"]),
    ("learned_joint_transport.cancelled", &["transport_budget_cancel"]),
    ("learned_joint_transport.feature_mismatch", &["invalid_argument"]),
    ("learned_joint_transport.fit_failed", &["transport_numerical_failure"]),
    ("learned_joint_transport.identification_not_certified", &["transport_missing_evidence"]),
    ("learned_joint_transport.identification_required", &["transport_missing_evidence"]),
    ("learned_joint_transport.invalid_basis", &["invalid_argument"]),
    ("learned_joint_transport.invalid_input", &["invalid_argument"]),
    ("learned_joint_transport.invalid_prior", &["invalid_argument"]),
    ("learned_joint_transport.missing_law", &["joint_law_required"]),
    ("learned_joint_transport.missing_source", &["transport_missing_evidence"]),
    ("learned_joint_transport.prior_data_overlap", &["invalid_argument"]),
    ("learned_joint_transport.prior_dimension", &["prior_dimension_mismatch"]),
    ("learned_joint_transport.rank_deficient", &["design_rank_deficient"]),
    ("learned_joint_transport.source_dependence", &["sampling_dependence_unknown"]),
    ("learned_joint_transport.support_resource_limit", &["execution_resource_limit"]),
    ("learned_joint_transport.too_many_draws", &["invalid_argument"]),
    ("learned_joint_transport.too_many_parameters", &["invalid_argument"]),
    ("learned_joint_transport.unsupported_graph", &["route_not_supported"]),
    ("learned_joint_transport.weak_overlap", &["transport_support_failure"]),
    ("learned_transport.bootstrap_below_floor", &["estimator_inference_mismatch"]),
    ("learned_transport.bootstrap_not_supported", &["estimator_inference_mismatch"]),
    ("learned_transport.cate_requested", &["route_not_supported"]),
    ("learned_transport.membership_overlap", &["transport_support_failure"]),
    ("learned_transport.non_iid_design", &["sampling_dependence_unknown"]),
    ("learned_transport.treatment_overlap", &["transport_support_failure"]),
    ("mapped_posterior_transfer.coefficient_map", &["quantity_semantics_mismatch"]),
    ("mapped_posterior_transfer.parameter_posterior", &["distribution_meaning_mismatch"]),
    ("mapped_posterior_transfer.residual_variance", &["prior_transfer_not_hydrated"]),
    ("matched_case_control.absolute_risk_not_identified", &["effect_not_identified"]),
    ("matched_case_control.budget", &["transport_budget_cancel"]),
    ("matched_case_control.estimand_not_supported", &["route_not_supported"]),
    ("matched_case_control.estimate_not_finite", &["route_not_supported"]),
    ("matched_case_control.invalid_data", &["invalid_argument"]),
    ("matched_case_control.invalid_level", &["invalid_argument"]),
    ("matched_case_control.no_informative_sets", &["effect_not_identified"]),
    ("matched_case_control.sampling_design", &["invalid_argument"]),
    ("measured_inference.artifact_mismatch", &["invalid_argument"]),
    ("measured_inference.bounds_exceeded", &["route_not_supported"]),
    ("measured_inference.consumer_bounds_exceeded", &[]),
    ("measured_inference.invalid_expectation", &[]),
    ("measured_inference.invalid_report", &["invalid_argument"]),
    ("measured_inference.memory_budget_exceeded", &[]),
    ("measured_inference.source_report_not_supported", &[]),
    ("measured_inference.unsupported_route", &["route_not_supported"]),
    ("measured_interval.draws_not_converged", &["posterior_draws_not_converged"]),
    ("measured_interval.invalid_source", &["invalid_argument"]),
    ("measured_interval.no_interval_of_this_kind", &["route_not_supported"]),
    ("measured_interval.source_mismatch", &["invalid_argument"]),
    ("measured_interval.variance_assumption_not_met", &["route_not_supported"]),
    ("measured_transport.input_bounds_exceeded", &["invalid_argument"]),
    ("measured_transport.sampler_mismatch", &["invalid_argument"]),
    ("measurement_model.admission_refused", &["invalid_argument"]),
    ("measurement_model.arm_mismatch", &["measurement_binding_mismatch"]),
    ("measurement_model.artifact_invalid", &["invalid_argument"]),
    ("measurement_model.artifact_refused", &["measurement_binding_mismatch"]),
    ("measurement_model.binary_outcome_continuous_error", &["route_not_supported"]),
    ("measurement_model.blank_field", &["measurement_model_invalid"]),
    ("measurement_model.bootstrap_unstable", &["measurement_not_identified"]),
    ("measurement_model.cancelled", &["cancelled_no_claim"]),
    ("measurement_model.confusion_ill_conditioned", &["measurement_not_identified"]),
    ("measurement_model.construct_mismatch", &["measurement_binding_mismatch"]),
    ("measurement_model.continuous_export", &["route_not_supported"]),
    ("measurement_model.design_singular", &["design_rank_deficient"]),
    ("measurement_model.differential_axis_unsupported", &["differential_error_not_pooled"]),
    ("measurement_model.differential_declared", &["differential_error_not_pooled"]),
    ("measurement_model.differential_population_unestablished", &["differential_error_not_pooled"]),
    ("measurement_model.differential_undeclared", &["differential_error_not_pooled"]),
    ("measurement_model.error_variance_not_identified", &["measurement_not_identified"]),
    ("measurement_model.indicator_as_construct", &["proxy_offered_as_construct"]),
    ("measurement_model.indicator_kind", &["measurement_model_invalid"]),
    ("measurement_model.indicator_mismatch", &["measurement_binding_mismatch"]),
    ("measurement_model.indicator_observed", &["measurement_model_invalid"]),
    ("measurement_model.input_invalid", &["invalid_argument"]),
    ("measurement_model.insufficient_draws", &["measurement_not_identified"]),
    ("measurement_model.insufficient_replicates", &["measurement_not_identified"]),
    ("measurement_model.interval_level", &["invalid_argument"]),
    ("measurement_model.interval_requires_estimate", &["route_not_supported"]),
    ("measurement_model.intervention_meaning_not_carried", &["representation_mapping_required"]),
    ("measurement_model.invalid_model", &["measurement_model_invalid"]),
    ("measurement_model.invariance_not_declared", &["transport_not_certified"]),
    ("measurement_model.invariance_premise_required", &["transport_missing_evidence"]),
    ("measurement_model.invariance_source_unknown", &["transport_not_certified"]),
    ("measurement_model.invariance_target_mismatch", &["transport_not_certified"]),
    ("measurement_model.learned_coordinates_not_intervenable", &["representation_mapping_required"]),
    ("measurement_model.mapping_endpoints", &["representation_mapping_required"]),
    ("measurement_model.no_model", &["measurement_model_invalid"]),
    ("measurement_model.nothing_to_transport", &["route_not_supported"]),
    ("measurement_model.operation_not_supported", &["route_not_supported"]),
    ("measurement_model.outcome_model_not_fitted", &["measurement_not_identified"]),
    ("measurement_model.parameter", &["measurement_model_invalid"]),
    ("measurement_model.population_mismatch", &["measurement_binding_mismatch"]),
    ("measurement_model.posterior_bound_to_snapshot", &["route_not_supported"]),
    ("measurement_model.posterior_capability", &["measurement_posterior_not_trusted"]),
    ("measurement_model.posterior_declaration", &["measurement_posterior_not_trusted"]),
    ("measurement_model.posterior_identity", &["measurement_posterior_not_trusted"]),
    ("measurement_model.posterior_kind", &["measurement_posterior_not_trusted"]),
    ("measurement_model.posterior_native_trust", &["measurement_posterior_not_trusted"]),
    ("measurement_model.posterior_quantity", &["measurement_posterior_not_trusted"]),
    ("measurement_model.posterior_receipt", &["measurement_posterior_not_trusted"]),
    ("measurement_model.proxy_as_construct", &["proxy_offered_as_construct"]),
    ("measurement_model.regime_mismatch", &["measurement_binding_mismatch"]),
    ("measurement_model.reliability_too_small", &["measurement_not_identified"]),
    ("measurement_model.representation_unmapped", &["representation_mapping_required"]),
    ("measurement_model.route_mismatch", &["route_not_supported"]),
    ("measurement_model.route_parameters", &["measurement_model_invalid"]),
    ("measurement_model.structure_parameters", &["differential_error_not_pooled"]),
    ("measurement_model.transport_receipt_mismatch", &["measurement_binding_mismatch"]),
    ("measurement_model.unknown_level", &["invalid_argument"]),
    ("measurement_model.unmodelled_proxy", &["proxy_offered_as_construct"]),
    ("measurement_model.unrecognized", &["measurement_model_invalid"]),
    ("measurement_model.variance_not_positive", &["measurement_not_identified"]),
    ("measurement_model.youden_too_small", &["measurement_not_identified"]),
    ("mechanism_composition.artifact_invalid", &["invalid_argument"]),
    ("mechanism_composition.artifact_kind_mismatch", &["invalid_argument"]),
    ("mechanism_composition.cancelled", &["cancelled_no_claim"]),
    ("mechanism_composition.conditioning_extra_conditioner", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.conditioning_parent_not_conditioned", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.cycle_directed_cycle", &["graph_invalid"]),
    ("mechanism_composition.family_mixed", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.identity_declaration_unknown_variable", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.identity_duplicate_mechanism", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.identity_edge_unknown_variable", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.identity_parent_without_mechanism", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.interval_invalid", &["invalid_argument"]),
    ("mechanism_composition.latent_family_unsupported", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.latent_loading_undeclared", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.latent_shared_with_absent_mechanism", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.latent_sharing_not_declared", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.latent_sharing_not_reciprocal", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.latent_too_many", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.numbers_unbound", &["invalid_argument"]),
    ("mechanism_composition.numerical_failure", &["invalid_argument"]),
    ("mechanism_composition.parents_declared_graph_mismatch", &["graph_invalid"]),
    ("mechanism_composition.population_invariance_unstated", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.population_latent_blocks_selection_admissibility", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.population_parent_context_differs_from_fit", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.population_selection_arrow_into_mechanism", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.population_undeclared_transport", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.posterior_dependence_cross_block_duplicate", &["joint_law_required"]),
    ("mechanism_composition.posterior_dependence_cross_covariance_missing", &["joint_law_required"]),
    ("mechanism_composition.posterior_dependence_evidence_overlap_unknown", &["joint_law_required"]),
    ("mechanism_composition.posterior_dependence_independence_unjustified", &["joint_law_required"]),
    ("mechanism_composition.posterior_dependence_shared_evidence_declared_independent", &["joint_law_required"]),
    ("mechanism_composition.posterior_dependence_unknown", &["joint_law_required"]),
    ("mechanism_composition.premises_mismatch", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.provenance_declared_mechanism_unacknowledged", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.provenance_missing_artifact", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.provenance_missing_evidence", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.quantity_conditioned_event", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.quantity_conditioning_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.quantity_functional_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.quantity_horizon_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.quantity_invalid_coordinate", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.quantity_not_a_realization", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.quantity_role_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.regime_parent_context_differs_from_fit", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.regime_selection_diagram_not_applicable", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.regime_undeclared_invariance", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.request_unsupported", &["route_not_supported"]),
    ("mechanism_composition.result_mismatch", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.shape_blank_coordinate", &["invalid_argument"]),
    ("mechanism_composition.shape_blank_identity", &["invalid_argument"]),
    ("mechanism_composition.shape_coefficient_parent_mismatch", &["invalid_argument"]),
    ("mechanism_composition.shape_duplicate_parent", &["invalid_argument"]),
    ("mechanism_composition.shape_empty_composition", &["invalid_argument"]),
    ("mechanism_composition.shape_invalid_domain", &["invalid_argument"]),
    ("mechanism_composition.shape_invalid_latents", &["invalid_argument"]),
    ("mechanism_composition.shape_invalid_table", &["invalid_argument"]),
    ("mechanism_composition.shape_mechanism_count", &["invalid_argument"]),
    ("mechanism_composition.shape_negative_scale", &["invalid_argument"]),
    ("mechanism_composition.shape_non_finite_parameter", &["invalid_argument"]),
    ("mechanism_composition.shape_too_many_mechanisms", &["invalid_argument"]),
    ("mechanism_composition.shape_too_many_parents", &["invalid_argument"]),
    ("mechanism_composition.slot_not_linear_gaussian", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.support_disjoint", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.support_intervention_outside_domain", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.support_kind_mismatch", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.support_missing_parent_configuration", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.support_not_contained", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.transform_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_composition.uncertainty_assembled_not_psd", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_cross_block_shape", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_cross_block_unknown_mechanism", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_family_unsupported", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_joint_parameter_set_mismatch", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_marginal_inconsistent_with_joint", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_marginal_missing", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_meaning_mismatch", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_not_psd", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_not_symmetric", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_parameter_set_mismatch", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.uncertainty_shape", &["mechanism_composition_incoherent"]),
    ("mechanism_composition.units_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_discrepancy.degenerate_covariance", &["invalid_argument"]),
    ("mechanism_discrepancy.dependence_unknown", &["route_not_supported"]),
    ("mechanism_discrepancy.duplicate_name", &["invalid_argument"]),
    ("mechanism_discrepancy.incomparable_measurements", &["route_not_supported"]),
    ("mechanism_discrepancy.inconsistent_summary", &["invalid_argument"]),
    ("mechanism_discrepancy.invalid_alpha", &["invalid_argument"]),
    ("mechanism_discrepancy.invalid_artifact", &["invalid_argument"]),
    ("mechanism_discrepancy.invalid_identity", &["invalid_argument"]),
    ("mechanism_discrepancy.invalid_power", &["invalid_argument"]),
    ("mechanism_discrepancy.invalid_request", &["invalid_argument"]),
    ("mechanism_discrepancy.no_compared_coefficients", &["invalid_argument"]),
    ("mechanism_discrepancy.non_finite_value", &["invalid_argument"]),
    ("mechanism_discrepancy.rank_deficient_design", &["invalid_argument"]),
    ("mechanism_discrepancy.row_count_mismatch", &["invalid_argument"]),
    ("mechanism_discrepancy.sample_too_small", &["invalid_argument"]),
    ("mechanism_discrepancy.too_many_parents", &["invalid_argument"]),
    ("mechanism_discrepancy.wrong_contract", &["route_not_supported"]),
    ("mechanism_posterior.band_not_constructed", &["route_not_supported"]),
    ("mechanism_posterior.empty_summary", &["invalid_argument"]),
    ("mechanism_posterior.external_power_prior", &["route_not_supported"]),
    ("mechanism_posterior.mapped_prior", &["route_not_supported"]),
    ("mechanism_posterior.non_positive_scale", &["invalid_argument"]),
    ("mechanism_transport.diagram_source_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_transport.diagram_target_mismatch", &["quantity_semantics_mismatch"]),
    ("mechanism_transport.duplicate_mechanism", &["quantity_semantics_mismatch"]),
    ("mechanism_transport.family_unsupported", &["mechanism_composition_incoherent"]),
    ("mechanism_transport.origin_not_fitted", &["mechanism_composition_incoherent"]),
    ("mechanism_transport.selection_arrow_into_latent", &["quantity_semantics_mismatch"]),
    ("mechanism_transport.target_mechanism_not_native", &["quantity_semantics_mismatch"]),
    ("mechanism_transport.transport_identity_mismatch", &["quantity_semantics_mismatch"]),
    ("mediation_posterior.band_not_constructed", &["route_not_supported"]),
    ("mediation_posterior.empty_design", &["invalid_argument"]),
    ("mediation_posterior.external_power_prior", &["route_not_supported"]),
    ("mediation_posterior.mapped_prior", &["route_not_supported"]),
    ("mediation_posterior.non_positive_scale", &["invalid_argument"]),
    ("missing_data.covariates_incomplete", &["missing_policy_not_applicable"]),
    ("missing_data.multiple_imputation", &["missing_policy_not_applicable"]),
    ("missing_data.policy_not_applicable", &["missing_policy_not_applicable"]),
    ("missing_data.policy_required", &["missing_policy_required"]),
    ("mixed_search.bounds_exceeded", &["route_not_supported"]),
    ("mixed_search.budget", &["transport_budget_cancel"]),
    ("mixed_search.invalid_derivation", &["transport_not_certified"]),
    ("mixed_search.invalid_query", &["invalid_argument"]),
    ("mixed_search.missing_joint", &["transport_missing_evidence"]),
    ("mixed_search.named_route", &["route_not_supported"]),
    ("mixed_search.not_certified", &["transport_not_certified"]),
    ("mixed_search.posterior_as_law", &["route_not_supported"]),
    ("mnar_fixed_shift.cancelled", &["execution_cancelled"]),
    ("mnar_fixed_shift.constant_outcome", &["invalid_argument"]),
    ("mnar_fixed_shift.degenerate_interval", &["invalid_argument"]),
    ("mnar_fixed_shift.duplicate_subject", &["invalid_argument"]),
    ("mnar_fixed_shift.insufficient_observations", &["invalid_argument"]),
    ("mnar_fixed_shift.interval_not_supported", &["invalid_argument"]),
    ("mnar_fixed_shift.invalid_declaration", &["invalid_argument"]),
    ("mnar_fixed_shift.invalid_outcome", &["invalid_argument"]),
    ("mnar_fixed_shift.invalid_shift", &["invalid_argument"]),
    ("mnar_fixed_shift.invalid_source", &["invalid_argument"]),
    ("mnar_fixed_shift.lowering", &["invalid_argument"]),
    ("mnar_fixed_shift.preparation", &["invalid_argument"]),
    ("mnar_fixed_shift.preparation_cancelled", &["execution_cancelled"]),
    ("mnar_fixed_shift.preparation_resource_limit", &["execution_resource_limit"]),
    ("mnar_fixed_shift.source_admission_cancelled", &["execution_cancelled"]),
    ("mnar_fixed_shift.source_admission_resource_limit", &["execution_resource_limit"]),
    ("mnar_sensitivity.adjustment_not_valid", &["invalid_argument"]),
    ("mnar_sensitivity.bounds_exceeded", &["route_not_supported"]),
    ("mnar_sensitivity.composition_not_supported", &["route_not_supported"]),
    ("mnar_sensitivity.covariates_not_used", &["invalid_argument"]),
    ("mnar_sensitivity.covariates_required", &["invalid_argument"]),
    ("mnar_sensitivity.design_singular", &["invalid_argument"]),
    ("mnar_sensitivity.graph_not_supported", &["route_not_supported"]),
    ("mnar_sensitivity.insufficient_observed", &["invalid_argument"]),
    ("mnar_sensitivity.invalid_bootstrap", &["invalid_argument"]),
    ("mnar_sensitivity.invalid_data", &["invalid_argument"]),
    ("mnar_sensitivity.invalid_decision", &["invalid_argument"]),
    ("mnar_sensitivity.invalid_level", &["invalid_argument"]),
    ("mnar_sensitivity.invalid_loading", &["invalid_argument"]),
    ("mnar_sensitivity.invalid_tolerance", &["invalid_argument"]),
    ("mnar_sensitivity.mechanism_not_declared", &["required_option_missing"]),
    ("mnar_sensitivity.range_excludes_mar", &["invalid_argument"]),
    ("mnar_sensitivity.unbounded_range", &["invalid_argument"]),
    ("mnar_sensitivity.unsupported_estimator", &["route_not_supported"]),
    ("mnar_sensitivity.unsupported_mechanism", &["route_not_supported"]),
    ("mnar_sensitivity.unsupported_query", &["route_not_supported"]),
    ("mnar_sensitivity.unsupported_target", &["route_not_supported"]),
    ("msm_posterior.band_not_constructed", &["route_not_supported"]),
    ("msm_posterior.empty_law", &["invalid_argument"]),
    ("msm_posterior.external_power_prior", &["route_not_supported"]),
    ("msm_posterior.mapped_prior", &["route_not_supported"]),
    ("msm_posterior.non_positive_scale", &["invalid_argument"]),
    ("msm_sensitivity.bounds_exceeded", &["route_not_supported"]),
    ("msm_sensitivity.composition_not_supported", &["route_not_supported"]),
    ("msm_sensitivity.invalid_threshold", &["invalid_argument"]),
    ("msm_sensitivity.invalid_tolerance", &["invalid_argument"]),
    ("msm_sensitivity.lambda_below_one", &["invalid_argument"]),
    ("msm_sensitivity.lambda_range_empty", &["invalid_argument"]),
    ("msm_sensitivity.no_interval_method", &["uncertainty_representation_insufficient"]),
    ("msm_sensitivity.outcome_law", &["invalid_argument"]),
    ("msm_sensitivity.positivity", &["route_not_supported"]),
    ("msm_sensitivity.stratum_mass", &["invalid_argument"]),
    ("multi_source_posterior.band_not_constructed", &["route_not_supported"]),
    ("multi_source_posterior.empty_law", &["invalid_argument"]),
    ("multi_source_posterior.external_power_prior", &["route_not_supported"]),
    ("multi_source_posterior.mapped_prior", &["route_not_supported"]),
    ("multi_source_posterior.non_positive_scale", &["invalid_argument"]),
    ("multiple_imputation.admission_refused", &["invalid_argument"]),
    ("multiple_imputation.ambiguous_variable_domain", &["invalid_argument"]),
    ("multiple_imputation.binary_under_joint_normal", &["missing_policy_not_applicable"]),
    ("multiple_imputation.cancelled", &["cancelled_no_claim"]),
    ("multiple_imputation.clusters_required", &["missing_policy_not_applicable"]),
    ("multiple_imputation.column_length", &["invalid_argument"]),
    ("multiple_imputation.conditional_model_not_fitted", &["missing_policy_not_applicable"]),
    ("multiple_imputation.dependent_series", &["missing_policy_not_applicable"]),
    ("multiple_imputation.dynamics_joint_normal", &["missing_policy_not_applicable"]),
    ("multiple_imputation.dynamics_lags", &["invalid_argument"]),
    ("multiple_imputation.graph_posterior_atoms", &["missing_policy_not_applicable"]),
    ("multiple_imputation.graph_posterior_frequentist", &["missing_policy_not_applicable"]),
    ("multiple_imputation.graph_posterior_query", &["missing_policy_not_applicable"]),
    ("multiple_imputation.graph_posterior_series", &["missing_policy_not_applicable"]),
    ("multiple_imputation.index", &["invalid_argument"]),
    ("multiple_imputation.interaction_variable", &["invalid_argument"]),
    ("multiple_imputation.interactions_joint_normal", &["missing_policy_not_applicable"]),
    ("multiple_imputation.mixed_fits", &["invalid_argument"]),
    ("multiple_imputation.multi_horizon_posterior_not_joint", &["missing_policy_not_applicable"]),
    ("multiple_imputation.non_finite", &["invalid_argument"]),
    ("multiple_imputation.not_at_random", &["missing_policy_not_applicable"]),
    ("multiple_imputation.not_identified", &["missing_policy_not_applicable"]),
    ("multiple_imputation.pool_df", &["invalid_argument"]),
    ("multiple_imputation.pool_non_finite", &["invalid_argument"]),
    ("multiple_imputation.pool_shape", &["invalid_argument"]),
    ("multiple_imputation.sampling_design", &["invalid_argument"]),
    ("multiple_imputation.spec_range", &["invalid_argument"]),
    ("multiple_imputation.stack_shape", &["invalid_argument"]),
    ("multiple_imputation.too_few_imputations", &["invalid_argument"]),
    ("multiple_imputation.units_labels", &["invalid_argument"]),
    ("multiple_imputation.units_shape", &["invalid_argument"]),
    ("multiple_imputation.variable_count", &["invalid_argument"]),
    ("multiple_imputation.variable_domain", &["invalid_argument"]),
    ("multiple_imputation.within_singular", &["invalid_argument"]),
    ("mz_transport.bounds_exceeded", &["route_not_supported"]),
    ("mz_transport.budget", &["transport_budget_cancel"]),
    ("mz_transport.checked_obstruction", &["transport_proven_non_transportable"]),
    ("mz_transport.empirical_counts_required", &["transport_missing_provider"]),
    ("mz_transport.fabricated_joint", &["transport_not_certified"]),
    ("mz_transport.functional_graph_mismatch", &["invalid_argument"]),
    ("mz_transport.invalid_catalog", &["invalid_argument"]),
    ("mz_transport.invalid_derivation", &["transport_not_certified"]),
    ("mz_transport.invalid_obstruction", &["transport_not_certified"]),
    ("mz_transport.invalid_query", &["invalid_argument"]),
    ("mz_transport.missing_joint_regime", &["transport_missing_evidence"]),
    ("mz_transport.search_incomplete", &["transport_not_certified"]),
    ("native_attempt.invalid_callable", &["invalid_argument"]),
    ("native_claims.calibration_not_supported", &["invalid_argument"]),
    ("native_claims.contract_mismatch", &["invalid_argument"]),
    ("native_claims.grid_mismatch", &["quantity_semantics_mismatch"]),
    ("native_claims.invalid_context", &["invalid_argument"]),
    ("native_claims.joint_law_invalid", &["invalid_argument"]),
    ("native_claims.native_state_unavailable", &["invalid_argument"]),
    ("native_claims.no_source_requirement", &["invalid_argument"]),
    ("native_claims.no_supported_coordinate", &["quantity_semantics_mismatch"]),
    ("native_claims.point_support_missing", &["quantity_semantics_mismatch"]),
    ("native_claims.program_mismatch", &["invalid_argument"]),
    ("native_claims.projection_mismatch", &["invalid_argument"]),
    ("native_claims.provenance_mismatch", &["invalid_argument"]),
    ("native_claims.response_not_point_identified", &["effect_not_identified"]),
    ("native_claims.source_not_supplied", &["decision_contract_unsatisfied"]),
    ("native_claims.units_conflict", &["quantity_semantics_mismatch"]),
    ("native_claims.unknown_variable", &["unknown_variable"]),
    ("native_claims.unsupported_estimand", &["route_not_supported"]),
    ("native_distribution.authority_required", &["invalid_argument"]),
    ("nested_markov.bayesian", &["invalid_argument"]),
    ("nested_markov.bayesian_artifact", &["invalid_argument"]),
    ("nested_markov.bayesian_artifact_limits", &["invalid_argument"]),
    ("nested_markov.bayesian_identity_mismatch", &["invalid_argument"]),
    ("nested_markov.bayesian_invalid_options", &["invalid_argument"]),
    ("nested_markov.bayesian_invalid_prior", &["invalid_argument"]),
    ("nested_markov.bayesian_sampler_scope", &["route_not_supported"]),
    ("nested_markov.cancelled", &["transport_budget_cancel"]),
    ("nested_markov.constraint_violated", &["transport_numerical_failure"]),
    ("nested_markov.fisher_artifact_limits", &["invalid_argument"]),
    ("nested_markov.fisher_identity_mismatch", &["invalid_argument"]),
    ("nested_markov.fisher_invalid_covariance", &["invalid_argument"]),
    ("nested_markov.fisher_invalid_level", &["invalid_argument"]),
    ("nested_markov.fisher_invalid_options", &["invalid_argument"]),
    ("nested_markov.fisher_sampling_design", &["invalid_argument"]),
    ("nested_markov.fisher_singular_information", &["invalid_argument"]),
    ("nested_markov.fit_not_converged", &["transport_numerical_failure"]),
    ("nested_markov.fitted_boundary", &["transport_numerical_failure"]),
    ("nested_markov.identification_unavailable", &["effect_not_identified"]),
    ("nested_markov.invalid_counts", &["invalid_argument"]),
    ("nested_markov.invalid_input", &["invalid_argument"]),
    ("nested_markov.invalid_options", &["invalid_argument"]),
    ("nested_markov.iteration_bound", &["invalid_argument"]),
    ("nested_markov.measured_scope", &["route_not_supported"]),
    ("nested_markov.native_hook_unavailable", &["not_supported"]),
    ("nested_markov.not_normalized", &["transport_numerical_failure"]),
    ("nested_markov.outside_binary_scope", &["route_not_supported"]),
    ("nested_markov.positivity", &["transport_support_failure"]),
    ("nested_markov.producer_missing", &["not_supported"]),
    ("nonlinear_mediation.cancelled", &["cancelled_no_claim"]),
    ("nonlinear_mediation.confounding", &["effect_not_identified"]),
    ("nonlinear_mediation.cross_world_independence_not_declared", &["cross_world_not_identified"]),
    ("nonlinear_mediation.data_required", &["route_not_supported"]),
    ("nonlinear_mediation.degenerate_mediator", &["invalid_argument"]),
    ("nonlinear_mediation.degenerate_mediator_residual", &["invalid_argument"]),
    ("nonlinear_mediation.degenerate_outcome", &["invalid_argument"]),
    ("nonlinear_mediation.degree_above_limit", &["invalid_argument"]),
    ("nonlinear_mediation.export_budget_exceeded", &["route_not_supported"]),
    ("nonlinear_mediation.identity_violation", &["mechanism_fit_not_converged"]),
    ("nonlinear_mediation.integration_error", &["mechanism_fit_not_converged"]),
    ("nonlinear_mediation.interventional_effects_not_supported", &["route_not_supported"]),
    ("nonlinear_mediation.invalid_degree", &["invalid_argument"]),
    ("nonlinear_mediation.invalid_nodes", &["invalid_argument"]),
    ("nonlinear_mediation.invalid_request", &["invalid_argument"]),
    ("nonlinear_mediation.invalid_support_bound", &["invalid_argument"]),
    ("nonlinear_mediation.invalid_tolerance", &["invalid_argument"]),
    ("nonlinear_mediation.length_mismatch", &["invalid_argument"]),
    ("nonlinear_mediation.limits_exceeded", &["invalid_argument"]),
    ("nonlinear_mediation.memory_budget_exceeded", &["route_not_supported"]),
    ("nonlinear_mediation.nodes_above_limit", &["invalid_argument"]),
    ("nonlinear_mediation.non_finite_input", &["invalid_argument"]),
    ("nonlinear_mediation.non_finite_option", &["invalid_argument"]),
    ("nonlinear_mediation.overlap", &["arm_not_populated"]),
    ("nonlinear_mediation.quadrature_nodes_not_converged", &["mechanism_fit_not_converged"]),
    ("nonlinear_mediation.quadrature_weights_invalid", &["mechanism_fit_not_converged"]),
    ("nonlinear_mediation.rank_deficient", &["design_rank_deficient"]),
    ("nonlinear_mediation.report_replay_mismatch", &["invalid_argument"]),
    ("nonlinear_mediation.too_few_rows", &["design_rank_deficient"]),
    ("nonlinear_mediation.treatment_induced_confounding", &["cross_world_not_identified"]),
    ("nonlinear_mediation.treatment_not_binary", &["invalid_argument"]),
    ("nonlinear_mediation.wrong_contract", &["route_not_supported"]),
    ("observation_query.arm_mismatch", &["measurement_binding_mismatch"]),
    ("observation_query.artifact_invalid", &["invalid_argument"]),
    ("observation_query.artifact_refused", &["invalid_argument"]),
    ("observation_query.blank_field", &["invalid_argument"]),
    ("observation_query.cancelled", &["cancelled_no_claim"]),
    ("observation_query.construct_mismatch", &["measurement_binding_mismatch"]),
    ("observation_query.construct_observed_directly", &["invalid_argument"]),
    ("observation_query.count_fitted_bounds_exceeded", &["invalid_argument"]),
    ("observation_query.count_fitted_refused", &["invalid_argument"]),
    ("observation_query.declared_model_has_no_sampling_interval", &["uncertainty_representation_insufficient"]),
    ("observation_query.differential_error_unsupported", &["differential_error_not_pooled"]),
    ("observation_query.evidence_covariance_singular", &["invalid_argument"]),
    ("observation_query.indicator_kind", &["invalid_argument"]),
    ("observation_query.indicator_mismatch", &["measurement_binding_mismatch"]),
    ("observation_query.intervention_in_conditioning_slot", &["quantity_semantics_mismatch"]),
    ("observation_query.invalid_declaration", &["invalid_argument"]),
    ("observation_query.invalid_interval", &["invalid_argument"]),
    ("observation_query.invalid_request", &["invalid_argument"]),
    ("observation_query.level_unreachable", &["evidence_cannot_reach_level"]),
    ("observation_query.levels_for_continuous_model", &["invalid_argument"]),
    ("observation_query.measured_as_construct", &["proxy_offered_as_construct"]),
    ("observation_query.measured_background", &["route_not_supported"]),
    ("observation_query.measured_signal_not_redrawn", &["route_not_supported"]),
    ("observation_query.measurement_route", &["route_not_supported"]),
    ("observation_query.mixed_conditioning_and_intervention", &["route_not_supported"]),
    ("observation_query.multiple_measured_signals", &["route_not_supported"]),
    ("observation_query.no_calibrated_probability", &["uncertainty_representation_insufficient"]),
    ("observation_query.no_parameter_draws", &["invalid_argument"]),
    ("observation_query.outcome_assigned", &["invalid_argument"]),
    ("observation_query.population_mismatch", &["measurement_binding_mismatch"]),
    ("observation_query.positive_level", &["invalid_argument"]),
    ("observation_query.real_values_for_finite_model", &["invalid_argument"]),
    ("observation_query.regime_mismatch", &["measurement_binding_mismatch"]),
    ("observation_query.repeated_slot", &["invalid_argument"]),
    ("observation_query.request_unsupported", &["route_not_supported"]),
    ("observation_query.unknown_level", &["invalid_argument"]),
    ("observation_query.unknown_quantity", &["invalid_argument"]),
    ("observation_query.unknown_variable", &["unknown_variable"]),
    ("observation_query.value_not_finite", &["invalid_argument"]),
    ("observation_query.zero_probability_evidence", &["invalid_argument"]),
    ("paradigm_parity.paradigm_not_supported", &["route_not_supported"]),
    ("parent_set_effect.conflict_resolution_budget", &["discovered_structure_unresolved"]),
    ("parent_set_effect.discovery_failed", &["invalid_argument"]),
    ("parent_set_effect.knowledge_invalid", &["invalid_argument"]),
    ("parent_set_effect.log_linear_input_not_positive", &["invalid_argument"]),
    ("parent_set_effect.outcome_precedes_treatment", &["invalid_argument"]),
    ("parent_set_effect.parent_limit_bound", &["discovered_structure_unresolved"]),
    ("parent_set_effect.search_not_finished", &["discovered_structure_unresolved"]),
    ("parent_set_effect.source_invalid", &["invalid_argument"]),
    ("parent_set_effect.source_not_canonical", &["invalid_argument"]),
    ("parent_set_effect.structure_class", &["invalid_argument"]),
    ("parent_set_effect.transformation_undeclared", &["invalid_argument"]),
    ("parent_set_effect.treatment_orientation_depends_on_conflict", &["discovered_structure_unresolved"]),
    ("parent_set_effect.treatment_parents_ambiguous", &["discovered_structure_unresolved"]),
    ("penalized_posterior.band_not_constructed", &["route_not_supported"]),
    ("penalized_posterior.empty_scores", &["invalid_argument"]),
    ("penalized_posterior.external_power_prior", &["route_not_supported"]),
    ("penalized_posterior.mapped_prior", &["route_not_supported"]),
    ("penalized_posterior.non_positive_scale", &["invalid_argument"]),
    ("penalized_propensity.cancelled", &["cancelled_no_claim"]),
    ("penalized_propensity.effect_posterior", &["route_not_supported"]),
    ("penalized_propensity.fallback_not_supported", &["nuisance_fallback_not_supported"]),
    ("penalized_propensity.fallback_with_penalty", &["invalid_argument"]),
    ("penalized_propensity.grid_allocation", &["execution_resource_limit"]),
    ("penalized_propensity.invalid_penalty", &["invalid_argument"]),
    ("penalized_propensity.scope", &["route_not_supported"]),
    ("penalized_propensity.selection_not_supported", &["selection_inference_not_supported"]),
    ("policy_value.action_not_populated", &["arm_not_populated"]),
    ("policy_value.action_outside_support", &["positivity_violated"]),
    ("policy_value.action_propensity_below_floor", &["positivity_violated"]),
    ("policy_value.cancelled", &["cancelled_no_claim"]),
    ("policy_value.evaluation_fraction_missing", &["required_option_missing"]),
    ("policy_value.evaluation_missing", &["required_option_missing"]),
    ("policy_value.evaluation_unknown", &["invalid_argument"]),
    ("policy_value.invalid_data", &["invalid_argument"]),
    ("policy_value.invalid_evaluation_fraction", &["invalid_argument"]),
    ("policy_value.invalid_folds", &["invalid_argument"]),
    ("policy_value.invalid_known_propensity", &["invalid_argument"]),
    ("policy_value.invalid_propensity_floor", &["invalid_argument"]),
    ("policy_value.invalid_request", &["invalid_argument"]),
    ("policy_value.invalid_support", &["invalid_argument"]),
    ("policy_value.objective_missing", &["required_option_missing"]),
    ("policy_value.objective_unknown", &["invalid_argument"]),
    ("policy_value.policy_actions_missing", &["required_option_missing"]),
    ("policy_value.policy_learned_on_evaluation_rows", &["policy_learned_on_evaluation_rows"]),
    ("policy_value.policy_origin_missing", &["required_option_missing"]),
    ("policy_value.policy_origin_unknown", &["invalid_argument"]),
    ("policy_value.reference_outside_support", &["invalid_argument"]),
    ("policy_value.sparse_action", &["arm_not_populated"]),
    ("policy_value.support_missing", &["required_option_missing"]),
    ("policy_value.too_few_evaluation_rows", &["invalid_argument"]),
    ("policy_value.too_few_rows", &["invalid_argument"]),
    ("policy_value.training_split_missing_action", &["arm_not_populated"]),
    ("policy_value.undeclared_level", &["treatment_level_undeclared"]),
    ("predictive_assessment.bounds_exceeded", &["route_not_supported"]),
    ("predictive_assessment.cancelled", &["cancelled_no_claim"]),
    ("predictive_assessment.design_invalid", &["invalid_argument"]),
    ("predictive_assessment.diagnostic_not_defined", &["invalid_argument"]),
    ("predictive_assessment.horizon_mismatch", &["quantity_semantics_mismatch"]),
    ("predictive_assessment.iid_not_supported_for_dependent_design", &["sampling_dependence_unknown"]),
    ("predictive_assessment.in_sample", &["evidence_double_counted"]),
    ("predictive_assessment.interval_dependence_undeclared", &["sampling_dependence_unknown"]),
    ("predictive_assessment.interval_law_family_mixed", &["route_not_supported"]),
    ("predictive_assessment.interval_not_supported", &["quantity_semantics_mismatch"]),
    ("predictive_assessment.interval_request_invalid", &["invalid_argument"]),
    ("predictive_assessment.invalid_artifact", &["invalid_argument"]),
    ("predictive_assessment.invalid_law", &["invalid_argument"]),
    ("predictive_assessment.invalid_observations", &["invalid_argument"]),
    ("predictive_assessment.law_callback_not_built", &["route_not_supported"]),
    ("predictive_assessment.log_score_infinite", &["invalid_argument"]),
    ("predictive_assessment.not_a_causal_claim", &["route_not_supported"]),
    ("predictive_assessment.outcome_kind", &["invalid_argument"]),
    ("predictive_assessment.overlap_unknown", &["evidence_overlap_unknown"]),
    ("predictive_assessment.provider_identity_incomplete", &["invalid_argument"]),
    ("predictive_assessment.report_replay_mismatch", &["invalid_argument"]),
    ("predictive_assessment.score_not_defined", &["route_not_supported"]),
    ("predictive_assessment.score_not_probability", &["route_not_supported"]),
    ("predictive_assessment.serial_lag_below_horizon", &["sampling_dependence_unknown"]),
    ("predictive_assessment.target_mismatch", &["quantity_semantics_mismatch"]),
    ("predictive_assessment.too_few_clusters", &["too_few_clusters"]),
    ("predictive_assessment.unknown_source", &["invalid_argument"]),
    ("predictive_assessment.wrong_contract", &["route_not_supported"]),
    ("prior_signal.adapted_prior_not_draws", &["invalid_argument"]),
    ("prior_signal.artifact_not_in_catalog", &["design_signal_invalid"]),
    ("prior_signal.coefficient_prior_not_signal", &["design_signal_invalid"]),
    ("prior_signal.duplicate_source", &["design_signal_invalid"]),
    ("prior_signal.incompatible_artifact", &["design_signal_invalid"]),
    ("prior_signal.incomplete_posterior", &["design_signal_invalid"]),
    ("prior_signal.invalid_parameter", &["invalid_argument"]),
    ("prior_signal.invalid_request", &["invalid_argument"]),
    ("prior_signal.lineage_missing", &["design_signal_invalid"]),
    ("prior_signal.named_state_draws_required", &["design_signal_invalid"]),
    ("prior_signal.no_sources", &["design_signal_invalid"]),
    ("prior_signal.nonfinite_mapped_prior", &["invalid_argument"]),
    ("prior_signal.observation_lineage_missing", &["design_signal_invalid"]),
    ("prior_signal.observation_overlap_unknown", &["evidence_overlap_unknown"]),
    ("prior_signal.observation_reuse", &["design_signal_invalid"]),
    ("prior_signal.original_posterior_unidentified", &["design_signal_invalid"]),
    ("prior_signal.posterior_draws_required", &["design_signal_invalid"]),
    ("prior_signal.prior_identity_mismatch", &["invalid_argument"]),
    ("prior_signal.prior_projection_mismatch", &["invalid_argument"]),
    ("prior_signal.prior_sources_overlap", &["design_signal_invalid"]),
    ("prior_signal.resolution_invalid", &["invalid_argument"]),
    ("prior_signal.signal_family_mismatch", &["design_signal_invalid"]),
    ("prior_signal.source_not_in_catalog_or_duplicate", &["design_signal_invalid"]),
    ("prior_signal.source_population_mismatch", &["invalid_argument"]),
    ("prior_signal.state_quantity_mismatch", &["invalid_argument"]),
    ("prior_signal.target_population_mismatch", &["invalid_argument"]),
    ("prior_signal.transport_policy_required", &["transport_policy_required"]),
    ("prior_signal.unlock_list_not_signal", &["design_signal_invalid"]),
    ("prior_signal.unsupported_signal_provider", &["route_not_supported"]),
    ("program_binding.contract_identity_mismatch", &["external_binding_mismatch"]),
    ("program_binding.dose_grid_changed", &["quantity_semantics_mismatch"]),
    ("program_binding.graph_only_identity", &["external_binding_mismatch"]),
    ("program_binding.invalid_binding", &["invalid_argument"]),
    ("program_binding.malformed_quantity", &["invalid_argument"]),
    ("program_binding.no_identified_request", &["invalid_argument"]),
    ("program_binding.population_mismatch", &["quantity_semantics_mismatch"]),
    ("program_binding.quantities_override_mismatch", &["quantity_semantics_mismatch"]),
    ("program_binding.treatment_outcome_substitution", &["external_binding_mismatch"]),
    ("proposal_arrival.assignment_mismatch", &["invalid_argument"]),
    ("proposal_arrival.base_catalog_changed", &["invalid_argument"]),
    ("proposal_arrival.base_catalog_missing", &["invalid_argument"]),
    ("proposal_arrival.base_provider_mismatch", &["invalid_argument"]),
    ("proposal_arrival.binding_missing", &["invalid_argument"]),
    ("proposal_arrival.bounds_exceeded", &["route_not_supported"]),
    ("proposal_arrival.cancelled", &["route_not_supported"]),
    ("proposal_arrival.candidate_missing", &["invalid_argument"]),
    ("proposal_arrival.duplicate_original_artifacts", &["invalid_argument"]),
    ("proposal_arrival.expected_proposal_mismatch", &["invalid_argument"]),
    ("proposal_arrival.family_not_supported", &["route_not_supported"]),
    ("proposal_arrival.identity_mismatch", &["invalid_argument"]),
    ("proposal_arrival.independent_joint_study_required", &["route_not_supported"]),
    ("proposal_arrival.invalid_request", &["invalid_argument"]),
    ("proposal_arrival.original_artifacts_required", &["invalid_argument"]),
    ("proposal_arrival.original_consumer_refused", &["invalid_argument"]),
    ("proposal_arrival.proof_not_bound", &["invalid_argument"]),
    ("proposal_arrival.provider_binding_mismatch", &["invalid_argument"]),
    ("proposal_arrival.raw_counts_required", &["invalid_argument"]),
    ("proposal_arrival.replay_mismatch", &["invalid_argument"]),
    ("proposal_arrival.sample_size_mismatch", &["invalid_argument"]),
    ("proposal_arrival.structural_arrival_refused", &["invalid_argument"]),
    ("proposal_receipt.arrival_cancelled", &["transport_budget_cancel"]),
    ("proposal_receipt.arrival_empty", &["invalid_argument"]),
    ("proposal_receipt.arrival_family_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.arrival_invalid", &["invalid_argument"]),
    ("proposal_receipt.arrival_sample_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.base_failure_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.bundle_identity_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.bundle_inconsistent", &["design_signal_invalid"]),
    ("proposal_receipt.candidate_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.candidate_not_in_bundle", &["design_signal_invalid"]),
    ("proposal_receipt.candidate_not_in_ranking", &["design_signal_invalid"]),
    ("proposal_receipt.candidate_not_in_repair", &["design_signal_invalid"]),
    ("proposal_receipt.candidate_not_proposed", &["design_signal_invalid"]),
    ("proposal_receipt.candidate_set_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.contract_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.contract_required", &["invalid_argument"]),
    ("proposal_receipt.cost_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.delta_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.derivation_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.duplicate_candidate", &["invalid_argument"]),
    ("proposal_receipt.empty_bundle", &["invalid_argument"]),
    ("proposal_receipt.hypothetical_not_evidence", &["transport_missing_evidence"]),
    ("proposal_receipt.identity_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.invalid_delta", &["invalid_argument"]),
    ("proposal_receipt.lineage_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.non_canonical_order", &["design_signal_invalid"]),
    ("proposal_receipt.obligation_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.ranking_digest_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.repair_digest_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.signal_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.source_digest_mismatch", &["design_signal_invalid"]),
    ("proposal_receipt.value_mismatch", &["design_signal_invalid"]),
    ("provider_capability_negotiation.capability_missing", &["external_capability_missing"]),
    ("provider_capability_negotiation.request_mismatch", &["external_binding_mismatch"]),
    ("provider_transport.artifact_invalid", &["invalid_argument"]),
    ("provider_transport.assumptions_missing", &["interface_contract_invalid"]),
    ("provider_transport.assumptions_not_carried", &["interface_contract_assumption_mismatch"]),
    ("provider_transport.blank_field", &["interface_contract_invalid"]),
    ("provider_transport.claim_not_in_target", &["interface_contract_quantity_mismatch"]),
    ("provider_transport.evidence_missing", &["interface_contract_invalid"]),
    ("provider_transport.identity_mismatch", &["route_not_supported"]),
    ("provider_transport.input_not_external", &["interface_contract_assumption_mismatch"]),
    ("provider_transport.native_target_required", &["interface_contract_assumption_mismatch"]),
    ("provider_transport.native_trust_claimed", &["interface_contract_invalid"]),
    ("provider_transport.object_invalid", &["interface_contract_invalid"]),
    ("provider_transport.object_unsupported", &["interface_contract_invalid"]),
    ("provider_transport.population_missing", &["interface_contract_invalid"]),
    ("provider_transport.population_not_a_source", &["interface_contract_quantity_mismatch"]),
    ("provider_transport.quantity_invalid", &["interface_contract_invalid"]),
    ("provider_transport.quantity_not_declared", &["external_binding_mismatch"]),
    ("provider_transport.receipt_does_not_cover", &["external_verification_failed"]),
    ("provider_transport.regime_missing", &["interface_contract_invalid"]),
    ("provider_transport.transport_identity_incomplete", &["interface_contract_invalid"]),
    ("provider_transport.trust_incomplete", &["interface_contract_invalid"]),
    ("provider_transport.verification_required", &["external_verification_failed"]),
    ("quadratic_posterior.band_not_constructed", &["route_not_supported"]),
    ("quadratic_posterior.external_power_prior", &["route_not_supported"]),
    ("quadratic_posterior.insufficient_weight", &["positivity_violated"]),
    ("quadratic_posterior.mapped_prior", &["route_not_supported"]),
    ("quadratic_posterior.non_positive_scale", &["invalid_argument"]),
    ("randomized_quantile_effect.adjustment_not_supported", &["route_not_supported"]),
    ("randomized_quantile_effect.arm_too_small", &["positivity_violated"]),
    ("randomized_quantile_effect.scope", &["route_not_supported"]),
    ("rank_drop_estimate.estimator_not_supported", &["route_not_supported"]),
    ("rank_drop_estimate.propensity_separates", &["rank_drop_not_supported"]),
    ("rank_drop_estimate.shape_not_supported", &["route_not_supported"]),
    ("rank_drop_estimate.span_not_preserved", &["rank_drop_not_supported"]),
    ("recalc.adjusted_arm_not_populated", &["arm_not_populated"]),
    ("recalc.adjusted_design_rank_deficient", &["design_rank_deficient"]),
    ("recalc.adjusted_fit_refused", &["route_not_supported"]),
    ("recalc.adjusted_glm_boundary_saturated", &["route_not_supported"]),
    ("recalc.adjusted_glm_not_converged", &["route_not_supported"]),
    ("recalc.adjusted_glm_separated", &["route_not_supported"]),
    ("recalc.adjusted_invalid_model", &["route_not_supported"]),
    ("recalc.adjusted_model_data_invalid", &["invalid_argument"]),
    ("recalc.adjusted_prediction_out_of_support", &["route_not_supported"]),
    ("recalc.adjusted_prediction_schema_mismatch", &["invalid_argument"]),
    ("recalc.adjusted_receipt_invalid", &["invalid_argument"]),
    ("recalc.adjustment_not_identified", &["effect_not_identified"]),
    ("recalc.aipw_requires_overlap_diagnostics", &["invalid_argument"]),
    ("recalc.bayesian_basis_prior_unsupported", &["route_not_supported"]),
    ("recalc.bayesian_effect_unavailable", &["invalid_argument"]),
    ("recalc.bayesian_external_compose_unsupported", &["route_not_supported"]),
    ("recalc.bayesian_fit_refused", &["route_not_supported"]),
    ("recalc.bayesian_inference_unsupported", &["route_not_supported"]),
    ("recalc.bayesian_invalid_data", &["invalid_argument"]),
    ("recalc.bayesian_invalid_graph", &["invalid_argument"]),
    ("recalc.bayesian_invalid_inference", &["invalid_argument"]),
    ("recalc.bayesian_invalid_query", &["invalid_argument"]),
    ("recalc.bayesian_invalid_summary", &["invalid_argument"]),
    ("recalc.bayesian_prior_invalid", &["invalid_argument"]),
    ("recalc.bayesian_prior_likelihood_double_use", &["invalid_argument"]),
    ("recalc.bayesian_prior_source_conflict", &["invalid_argument"]),
    ("recalc.bayesian_prior_source_unsupported", &["route_not_supported"]),
    ("recalc.bayesian_prior_source_unverified", &["route_not_supported"]),
    ("recalc.bayesian_workspace_limit", &["invalid_argument"]),
    ("recalc.blocked_by_refused_dependency", &["not_executed"]),
    ("recalc.bootstrap_replicates_unsupported", &["invalid_argument"]),
    ("recalc.cancelled", &["cancelled_no_claim"]),
    ("recalc.capability_adapter_unavailable", &["route_not_supported"]),
    ("recalc.categorical_role_mismatch", &["invalid_argument"]),
    ("recalc.contrast_out_of_support", &["route_not_supported"]),
    ("recalc.contrast_unavailable", &["invalid_argument"]),
    ("recalc.declared_size_mismatch", &["invalid_argument"]),
    ("recalc.design_artifact_mismatch", &["invalid_argument"]),
    ("recalc.design_artifact_unverified", &["route_not_supported"]),
    ("recalc.design_data_unavailable", &["score_table_unavailable"]),
    ("recalc.design_effect_unavailable", &["invalid_argument"]),
    ("recalc.design_fit_refused", &["route_not_supported"]),
    ("recalc.design_model_data_invalid", &["invalid_argument"]),
    ("recalc.design_not_checked", &["effect_not_identified"]),
    ("recalc.design_not_identified", &["effect_not_identified"]),
    ("recalc.design_point_unavailable", &["invalid_argument"]),
    ("recalc.design_rank_deficient", &["design_rank_deficient"]),
    ("recalc.design_receipt_invalid", &["invalid_argument"]),
    ("recalc.dr_arm_not_populated", &["arm_not_populated"]),
    ("recalc.dr_design_rank_deficient", &["design_rank_deficient"]),
    ("recalc.dr_effect_unavailable", &["invalid_argument"]),
    ("recalc.dr_fit_counter_mismatch", &["invalid_argument"]),
    ("recalc.dr_fit_refused", &["route_not_supported"]),
    ("recalc.dr_glm_boundary_saturated", &["route_not_supported"]),
    ("recalc.dr_glm_not_converged", &["route_not_supported"]),
    ("recalc.dr_glm_separated", &["route_not_supported"]),
    ("recalc.dr_invalid_folds", &["invalid_argument"]),
    ("recalc.dr_invalid_learner", &["invalid_argument"]),
    ("recalc.dr_invalid_model", &["invalid_argument"]),
    ("recalc.dr_invalid_overlap", &["invalid_argument"]),
    ("recalc.dr_invalid_score", &["invalid_argument"]),
    ("recalc.dr_model_data_invalid", &["invalid_argument"]),
    ("recalc.dr_not_identified", &["effect_not_identified"]),
    ("recalc.dr_prediction_out_of_support", &["route_not_supported"]),
    ("recalc.dr_prediction_schema_mismatch", &["invalid_argument"]),
    ("recalc.dr_predictor_unavailable", &["route_not_supported"]),
    ("recalc.dr_receipt_invalid", &["invalid_argument"]),
    ("recalc.dr_scores_unavailable", &["route_not_supported"]),
    ("recalc.dr_source_mismatch", &["invalid_argument"]),
    ("recalc.dr_target_incompatible", &["route_not_supported"]),
    ("recalc.family_model_data_invalid", &["invalid_argument"]),
    ("recalc.family_not_identified", &["effect_not_identified"]),
    ("recalc.family_receipt_invalid", &["invalid_argument"]),
    ("recalc.frozen_scores_rejected", &["invalid_argument"]),
    ("recalc.incompatible_contrast", &["route_not_supported"]),
    ("recalc.invalid_adjustment_set", &["invalid_argument"]),
    ("recalc.invalid_categorical_contrast", &["invalid_argument"]),
    ("recalc.invalid_changed_input", &["invalid_argument"]),
    ("recalc.invalid_contrast", &["invalid_argument"]),
    ("recalc.invalid_data", &["invalid_argument"]),
    ("recalc.invalid_declaration", &["invalid_argument"]),
    ("recalc.invalid_design_artifact", &["invalid_argument"]),
    ("recalc.invalid_design_roles", &["invalid_argument"]),
    ("recalc.invalid_evidence", &["invalid_argument"]),
    ("recalc.invalid_graph", &["invalid_argument"]),
    ("recalc.invalid_initial_state_law", &["invalid_argument"]),
    ("recalc.invalid_query", &["invalid_argument"]),
    ("recalc.invalid_rd_window", &["invalid_argument"]),
    ("recalc.invalid_request", &["invalid_argument"]),
    ("recalc.invalid_target_weights", &["invalid_argument"]),
    ("recalc.invalid_temporal_provider", &["invalid_argument"]),
    ("recalc.invalid_temporal_window", &["invalid_argument"]),
    ("recalc.iv_decision_unavailable", &["route_not_supported"]),
    ("recalc.iv_diagnostics_unavailable", &["invalid_argument"]),
    ("recalc.joint_bayesian_graph_unsupported", &["invalid_argument"]),
    ("recalc.joint_bayesian_invalid_data", &["invalid_argument"]),
    ("recalc.joint_bayesian_invalid_query", &["invalid_argument"]),
    ("recalc.joint_bayesian_refused", &["route_not_supported"]),
    ("recalc.limits_exceeded", &["invalid_argument"]),
    ("recalc.memory_budget_exceeded", &["invalid_argument"]),
    ("recalc.mz_action_out_of_support", &["route_not_supported"]),
    ("recalc.mz_invalid_request", &["invalid_argument"]),
    ("recalc.no_live_state", &["score_table_unavailable"]),
    ("recalc.off_grid_request", &["route_not_supported"]),
    ("recalc.primary_result_is_not_an_effect", &["invalid_argument"]),
    ("recalc.refused", &["not_executed"]),
    ("recalc.retarget_incompatible", &["construction_not_supported"]),
    ("recalc.retarget_not_supported", &["invalid_argument"]),
    ("recalc.row_count_mismatch", &["invalid_argument"]),
    ("recalc.row_ids_mismatch", &["invalid_argument"]),
    ("recalc.score_stage_undeclared", &["invalid_argument"]),
    ("recalc.score_table_unavailable", &["score_table_unavailable"]),
    ("recalc.static_action_out_of_support", &["route_not_supported"]),
    ("recalc.static_engine_refused", &["route_not_supported"]),
    ("recalc.static_graph_unsupported", &["route_not_supported"]),
    ("recalc.static_invalid_contrast", &["invalid_argument"]),
    ("recalc.static_invalid_data", &["invalid_argument"]),
    ("recalc.static_invalid_factor", &["invalid_argument"]),
    ("recalc.static_invalid_query", &["invalid_argument"]),
    ("recalc.static_not_identified", &["effect_not_identified"]),
    ("recalc.static_receipt_invalid", &["invalid_argument"]),
    ("recalc.static_required_factor_missing", &["route_not_supported"]),
    ("recalc.static_transport_budget_cancel", &["transport_budget_cancel"]),
    ("recalc.static_transport_missing_evidence", &["transport_missing_evidence"]),
    ("recalc.static_transport_missing_provider", &["transport_missing_provider"]),
    ("recalc.static_transport_non_transportable", &["transport_proven_non_transportable"]),
    ("recalc.static_transport_not_certified", &["transport_not_certified"]),
    ("recalc.static_transport_numerical_failure", &["transport_numerical_failure"]),
    ("recalc.static_transport_support_failure", &["transport_support_failure"]),
    ("recalc.static_transport_unsupported_evaluator", &["transport_unsupported_evaluator"]),
    ("recalc.temporal_artifact_invalid", &["invalid_argument"]),
    ("recalc.temporal_artifact_mismatch", &["invalid_argument"]),
    ("recalc.temporal_artifact_seed_mismatch", &["invalid_argument"]),
    ("recalc.temporal_budget_cancel", &["transport_budget_cancel"]),
    ("recalc.temporal_data_invalid", &["invalid_argument"]),
    ("recalc.temporal_evaluator_unsupported", &["transport_unsupported_evaluator"]),
    ("recalc.temporal_fit_refused", &["route_not_supported"]),
    ("recalc.temporal_history_unsupported", &["route_not_supported"]),
    ("recalc.temporal_missing_evidence", &["transport_missing_evidence"]),
    ("recalc.temporal_missing_provider", &["transport_missing_provider"]),
    ("recalc.temporal_numerical_failure", &["transport_numerical_failure"]),
    ("recalc.temporal_scope_unsupported", &["route_not_supported"]),
    ("recalc.temporal_support_failure", &["transport_support_failure"]),
    ("recalc.unavailable_data", &["score_table_unavailable"]),
    ("recalc.unavailable_fit", &["score_table_unavailable"]),
    ("recalc.unavailable_provider", &["external_capability_missing"]),
    ("recalc.unknown_variable", &["invalid_argument"]),
    ("recalc.unsupported_aipw_score_configuration", &["invalid_argument"]),
    ("recalc.unsupported_request", &["route_not_supported"]),
    ("recalc.weighted_overlap_failed", &["invalid_argument"]),
    ("recalc_receipt.counts_inconsistent", &["route_not_supported"]),
    ("recalc_receipt.fresh_process_reuse", &["score_table_unavailable"]),
    ("recalc_receipt.identity_mismatch", &["route_not_supported"]),
    ("recalc_receipt.plan_mismatch", &["route_not_supported"]),
    ("recalc_receipt.totals_mismatch", &["route_not_supported"]),
    ("recovery.bounds_exceeded", &["route_not_supported"]),
    ("recovery.budget", &["transport_budget_cancel"]),
    ("recovery.empirical_not_supported", &["route_not_supported"]),
    ("recovery.handoff_mismatch", &["invalid_argument"]),
    ("recovery.invalid_derivation", &["transport_not_certified"]),
    ("recovery.invalid_observed_law", &["invalid_argument"]),
    ("recovery.invalid_query", &["invalid_argument"]),
    ("recovery.missing_margin", &["transport_missing_evidence"]),
    ("recovery.nonrecoverable_witness", &["transport_proven_non_transportable"]),
    ("recovery.positivity", &["transport_support_failure"]),
    ("recovery.unsupported_mechanism", &["route_not_supported"]),
    ("recovery.witness_unavailable", &["transport_not_certified"]),
    ("recovery_chain.budget", &["transport_budget_cancel"]),
    ("recovery_chain.data_identity_mismatch", &["invalid_argument"]),
    ("recovery_chain.decision_replay_mismatch", &["invalid_argument"]),
    ("recovery_chain.invalid_derivation", &["transport_not_certified"]),
    ("recovery_chain.invalid_observed_law", &["invalid_argument"]),
    ("recovery_chain.invalid_query", &["invalid_argument"]),
    ("recovery_chain.names_mismatch", &["invalid_argument"]),
    ("recovery_chain.nonrecoverable_witness", &["transport_proven_non_transportable"]),
    ("recovery_chain.not_evaluated", &["invalid_argument"]),
    ("recovery_chain.positivity", &["transport_support_failure"]),
    ("recovery_chain.premises_mismatch", &["invalid_argument"]),
    ("recovery_chain.recovered_replay_mismatch", &["invalid_argument"]),
    ("recovery_chain.undecodable", &["invalid_argument"]),
    ("recovery_chain.unsupported_mechanism", &["route_not_supported"]),
    ("recovery_chain.unsupported_semantics", &["route_not_supported"]),
    ("refresh_posterior.band_not_constructed", &["route_not_supported"]),
    ("refresh_posterior.empty_values", &["invalid_argument"]),
    ("refresh_posterior.external_power_prior", &["route_not_supported"]),
    ("refresh_posterior.mapped_prior", &["route_not_supported"]),
    ("refresh_posterior.non_positive_scale", &["invalid_argument"]),
    ("repair_artifact.bounds_exceeded", &["route_not_supported"]),
    ("repair_artifact.budget", &["transport_budget_cancel"]),
    ("repair_artifact.invalid_artifact", &["transport_not_certified"]),
    ("repair_artifact.replay_mismatch", &["transport_not_certified"]),
    ("repair_artifact.unsupported_version", &["transport_not_certified"]),
    ("response_gam_derivative.shared_complete_case_rows", &["route_not_supported"]),
    ("rollout.artifact_id_missing", &["invalid_argument"]),
    ("rollout.artifact_kind_mismatch", &["invalid_argument"]),
    ("rollout.expected_binding_mismatch", &["invalid_argument"]),
    ("rollout.finite_draw_state_required", &["route_not_supported"]),
    ("rollout.identity_mismatch", &["invalid_argument"]),
    ("rollout.invalid_request", &["invalid_argument"]),
    ("rollout.ordered_prior_mismatch", &["invalid_argument"]),
    ("rollout.ranking_invalid", &["invalid_argument"]),
    ("rollout.ranking_source_mismatch", &["invalid_argument"]),
    ("rollout.source_digest_mismatch", &["invalid_argument"]),
    ("rollout.source_evidence_mismatch", &["invalid_argument"]),
    ("rollout.source_standing_mismatch", &["invalid_argument"]),
    ("rollout.state_coordinate_mismatch", &["invalid_argument"]),
    ("rollout.state_interpretation_mismatch", &["route_not_supported"]),
    ("rollout.terminal_decision_mismatch", &["invalid_argument"]),
    ("rollout.unweighted_joint_state_required", &["route_not_supported"]),
    ("sampled_recovery.bounds_exceeded", &["route_not_supported"]),
    ("sampled_recovery.cancelled", &["transport_budget_cancel"]),
    ("sampled_recovery.component_variance_only", &["estimator_inference_mismatch"]),
    ("sampled_recovery.invalid_input", &["invalid_argument"]),
    ("sampled_recovery.native_hook_unavailable", &["not_supported"]),
    ("sampled_recovery.producer_missing", &["not_supported"]),
    ("sampled_recovery.receipt_mismatch", &["invalid_argument"]),
    ("sampled_recovery.too_many_failed_replicates", &["route_not_supported"]),
    ("sampled_recovery.unrecoverable_pattern", &["route_not_supported"]),
    ("sampling_weights.invalid", &["sampling_weights_invalid"]),
    ("sampling_weights.route_unsupported", &["option_not_applicable"]),
    ("scenario_covariance.cancelled", &["cancelled_no_claim"]),
    ("scenario_covariance.consumer_limit_exceeded", &["execution_resource_limit"]),
    ("scenario_covariance.data_identity_mismatch", &["invalid_argument"]),
    ("scenario_covariance.duplicate_scenario_id", &["invalid_argument"]),
    ("scenario_covariance.exact_enumeration_cap", &["execution_resource_limit"]),
    ("scenario_covariance.invalid_declaration", &["invalid_argument"]),
    ("scenario_covariance.invalid_dependence", &["invalid_argument"]),
    ("scenario_covariance.invalid_failure_bound", &["invalid_argument"]),
    ("scenario_covariance.invalid_functional", &["invalid_argument"]),
    ("scenario_covariance.invalid_row_table", &["invalid_argument"]),
    ("scenario_covariance.no_scenarios", &["invalid_argument"]),
    ("scenario_covariance.not_positive_semidefinite", &["transport_numerical_failure"]),
    ("scenario_covariance.premises_mismatch", &["invalid_argument"]),
    ("scenario_covariance.replay_mismatch", &["invalid_argument"]),
    ("scenario_covariance.table_too_large", &["execution_resource_limit"]),
    ("scenario_covariance.too_few_replicates", &["invalid_argument"]),
    ("scenario_covariance.too_few_rows", &["invalid_argument"]),
    ("scenario_covariance.too_many_failed_replicates", &["transport_numerical_failure"]),
    ("scenario_covariance.too_many_terms", &["execution_resource_limit"]),
    ("scenario_covariance.unknown_dependence", &["route_not_supported"]),
    ("scenario_covariance.unsupported_semantics", &["route_not_supported"]),
    ("scenario_covariance_posterior.band_not_constructed", &["route_not_supported"]),
    ("scenario_covariance_posterior.empty_matrix", &["invalid_argument"]),
    ("scenario_covariance_posterior.external_power_prior", &["route_not_supported"]),
    ("scenario_covariance_posterior.mapped_prior", &["route_not_supported"]),
    ("scenario_covariance_posterior.non_positive_scale", &["invalid_argument"]),
    ("scenario_invariance.derivation_inconsistent", &["invalid_argument"]),
    ("scenario_invariance.non_static_node", &["invalid_argument"]),
    ("scenario_invariance.not_estimated", &["not_executed"]),
    ("scenario_invariance.refused", &["invalid_argument"]),
    ("scenario_invariance.wrong_result_type", &["invalid_argument"]),
    ("scenarios.catalog_selections", &["invalid_argument"]),
    ("scenarios.coordinate_mismatch", &["schema_mismatch"]),
    ("scenarios.count", &["route_not_supported"]),
    ("scenarios.duplicate_or_empty_name", &["invalid_argument"]),
    ("scenarios.duplicate_scenario", &["invalid_argument"]),
    ("scenarios.empty", &["invalid_argument"]),
    ("scenarios.equivalence_class_input", &["route_not_supported"]),
    ("scenarios.invalid_weights", &["invalid_argument"]),
    ("scenarios.shared_data_aggregate", &["scenario_aggregate_not_supported"]),
    ("scenarios.unevaluated_budget", &["transport_budget_cancel"]),
    ("scientific_bridge.artifact_invalid", &["invalid_argument"]),
    ("scientific_bridge.association_as_intervention", &["bridge_substitution_refused"]),
    ("scientific_bridge.assumption_evidence_missing", &["bridge_assumption_missing"]),
    ("scientific_bridge.assumption_missing", &["bridge_assumption_missing"]),
    ("scientific_bridge.blank_field", &["scientific_bridge_invalid"]),
    ("scientific_bridge.cancelled", &["cancelled_no_claim"]),
    ("scientific_bridge.chain_stage_not_affine", &["route_not_supported"]),
    ("scientific_bridge.conditioning_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.degenerate_validation", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.dependence_not_a_covariance", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.differential_error", &["differential_error_not_pooled"]),
    ("scientific_bridge.draws_meaning_mismatch", &["distribution_meaning_mismatch"]),
    ("scientific_bridge.draws_not_aligned", &["joint_law_required"]),
    ("scientific_bridge.draws_outside_domain", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.draws_unusable", &["scientific_bridge_invalid"]),
    ("scientific_bridge.duplicate_assumption", &["scientific_bridge_invalid"]),
    ("scientific_bridge.endpoint_classes", &["scientific_bridge_invalid"]),
    ("scientific_bridge.estimand_bridge_incomplete", &["scientific_bridge_invalid"]),
    ("scientific_bridge.estimand_bridge_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.estimand_spend_inconsistent", &["bridge_binding_mismatch"]),
    ("scientific_bridge.estimand_units_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.experimental_prior_without_estimand_bridge", &["bridge_assumption_missing"]),
    ("scientific_bridge.functional_not_supported", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.horizon_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.identity_mismatch", &["route_not_supported"]),
    ("scientific_bridge.interval_not_supported", &["route_not_supported"]),
    ("scientific_bridge.interval_scope_outside_measured", &["route_not_supported"]),
    ("scientific_bridge.interval_source_uncertainty_required", &["required_option_missing"]),
    ("scientific_bridge.invalid_quantity", &["scientific_bridge_invalid"]),
    ("scientific_bridge.joint_law_dimension_limit", &["execution_resource_limit"]),
    ("scientific_bridge.malformed_mapping", &["scientific_bridge_invalid"]),
    ("scientific_bridge.mapping_not_allowed", &["scientific_bridge_invalid"]),
    ("scientific_bridge.mapping_scope_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.mapping_uncertainty_invalid", &["scientific_bridge_invalid"]),
    ("scientific_bridge.mapping_uncertainty_missing", &["bridge_assumption_missing"]),
    ("scientific_bridge.measurement_model_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.measurement_route_not_supported", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.model_identity_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.model_score_as_probability", &["bridge_substitution_refused"]),
    ("scientific_bridge.non_finite_value", &["scientific_bridge_invalid"]),
    ("scientific_bridge.nonlinear_law_insufficient", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.nonlinear_mean_not_expectation", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.outside_attainable_range", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.outside_validated_range", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.population_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.predictive_importance_as_causal_effect", &["bridge_substitution_refused"]),
    ("scientific_bridge.propagation_admission_refused", &["execution_resource_limit"]),
    ("scientific_bridge.propagation_overflow", &["scientific_bridge_invalid"]),
    ("scientific_bridge.propagation_work_limit", &["execution_resource_limit"]),
    ("scientific_bridge.provenance", &["scientific_bridge_invalid"]),
    ("scientific_bridge.proxy_as_construct", &["proxy_offered_as_construct"]),
    ("scientific_bridge.quantity_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.regime_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.replay_mismatch", &["invalid_argument"]),
    ("scientific_bridge.scientific_endpoint_as_utility", &["bridge_substitution_refused"]),
    ("scientific_bridge.simulation_draw_limit", &["execution_resource_limit"]),
    ("scientific_bridge.source_dependence_invalid", &["scientific_bridge_invalid"]),
    ("scientific_bridge.source_dependence_missing", &["sampling_dependence_unknown"]),
    ("scientific_bridge.source_moment_not_finite", &["scientific_bridge_invalid"]),
    ("scientific_bridge.source_not_draws", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.source_not_point", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.source_variance_invalid", &["scientific_bridge_invalid"]),
    ("scientific_bridge.stated_preference_as_realized_behavior", &["bridge_substitution_refused"]),
    ("scientific_bridge.transport_diagram_scope_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.transport_premise_contradicted_by_selection_diagram", &["bridge_assumption_missing"]),
    ("scientific_bridge.transport_premise_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.transport_premise_missing", &["bridge_assumption_missing"]),
    ("scientific_bridge.transport_premise_unused", &["bridge_binding_mismatch"]),
    ("scientific_bridge.uncertain_mapping_not_affine", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.units_mismatch", &["bridge_binding_mismatch"]),
    ("scientific_bridge.validation_r_squared_too_low", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.validation_trials_too_few", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.value_kind_not_carried", &["bridge_mapping_not_supported"]),
    ("scientific_bridge.value_kind_substitution", &["bridge_substitution_refused"]),
    ("scientific_bridge.youden_too_small", &["measurement_not_identified"]),
    ("sde.action_time_off_grid", &["route_not_supported"]),
    ("sde.input_outside_certified_envelope", &["route_not_supported"]),
    ("sde.milstein_noncommutative", &["route_not_supported"]),
    ("sde.unbounded_growth", &["route_not_supported"]),
    ("selective_discovery.analysis_rows_not_the_split_part", &["invalid_argument"]),
    ("selective_discovery.artifact_invalid", &["invalid_argument"]),
    ("selective_discovery.background_knowledge_contradictory", &["invalid_argument"]),
    ("selective_discovery.background_knowledge_not_applicable", &["option_not_applicable"]),
    ("selective_discovery.background_knowledge_unknown_variable", &["unknown_variable"]),
    ("selective_discovery.components_incomplete", &["discovery_statistics_invalid"]),
    ("selective_discovery.contemporaneous_links", &["route_not_supported"]),
    ("selective_discovery.contract_identity", &["discovery_statistics_invalid"]),
    ("selective_discovery.contract_mismatch", &["discovery_statistics_invalid"]),
    ("selective_discovery.data_mismatch", &["discovery_statistics_invalid"]),
    ("selective_discovery.differences_are_not_selection_targets", &["discovered_structure_unresolved"]),
    ("selective_discovery.downstream_interval_not_supported", &["selection_inference_not_supported"]),
    ("selective_discovery.identity_mismatch", &["discovery_statistics_invalid"]),
    ("selective_discovery.invalid_contract", &["invalid_argument"]),
    ("selective_discovery.multi_population", &["route_not_supported"]),
    ("selective_discovery.nothing_retained", &["invalid_argument"]),
    ("selective_discovery.pooled_population_data", &["invalid_argument"]),
    ("selective_discovery.pooled_population_search", &["route_not_supported"]),
    ("selective_discovery.population_missing", &["invalid_argument"]),
    ("selective_discovery.population_repeated", &["invalid_argument"]),
    ("selective_discovery.populations_missing", &["invalid_argument"]),
    ("selective_discovery.raw_data_missing", &["discovery_statistics_invalid"]),
    ("selective_discovery.receipt_invalid", &["discovery_statistics_invalid"]),
    ("selective_discovery.receipt_limit", &["discovery_statistics_invalid"]),
    ("selective_discovery.receipt_repeated", &["discovery_statistics_invalid"]),
    ("selective_discovery.receipts_identity", &["discovery_statistics_invalid"]),
    ("selective_discovery.recovery_claim_not_supported", &["discovered_structure_unresolved"]),
    ("selective_discovery.replay_diverged", &["discovery_statistics_invalid"]),
    ("selective_discovery.replay_miss", &["discovery_statistics_invalid"]),
    ("selective_discovery.rng_mismatch", &["invalid_argument"]),
    ("selective_discovery.route_unknown", &["discovery_statistics_invalid"]),
    ("selective_discovery.rows", &["discovery_statistics_invalid"]),
    ("selective_discovery.score_search_not_supported", &["route_not_supported"]),
    ("selective_discovery.search_incomplete", &["discovered_structure_unresolved"]),
    ("selective_discovery.split_fraction_invalid", &["invalid_argument"]),
    ("selective_discovery.split_mask_mismatch", &["invalid_argument"]),
    ("selective_discovery.split_not_disjoint", &["invalid_argument"]),
    ("selective_discovery.split_part_empty", &["invalid_argument"]),
    ("selective_discovery.split_part_weight_invalid", &["invalid_argument"]),
    ("selective_discovery.split_receipt_mismatch", &["invalid_argument"]),
    ("selective_discovery.split_row_excluded", &["invalid_argument"]),
    ("selective_discovery.split_row_out_of_range", &["invalid_argument"]),
    ("selective_discovery.split_rows_mismatch", &["invalid_argument"]),
    ("selective_discovery.split_rows_repeated", &["invalid_argument"]),
    ("selective_discovery.split_source_mismatch", &["invalid_argument"]),
    ("selective_discovery.split_unsupported_rows", &["route_not_supported"]),
    ("selective_discovery.split_weights_mismatch", &["invalid_argument"]),
    ("selective_discovery.structure_class", &["discovery_statistics_invalid"]),
    ("selective_discovery.structure_marks", &["discovery_statistics_invalid"]),
    ("selective_discovery.structure_nodes", &["discovery_statistics_invalid"]),
    ("selective_discovery.test_identity_missing", &["invalid_argument"]),
    ("selective_discovery.unaccepted_discovery", &["discovered_structure_unresolved"]),
    ("selective_discovery.unresolved_structure", &["discovered_structure_unresolved"]),
    ("selective_discovery.variables_repeated", &["discovery_statistics_invalid"]),
    ("selective_discovery.weighted_test_without_weights", &["option_not_applicable"]),
    ("selective_discovery.weights_not_read_by_test", &["option_not_applicable"]),
    ("selective_discovery.work_inconsistent", &["discovery_statistics_invalid"]),
    ("sensitivity_decision_composition.bounds_exceeded", &["route_not_supported"]),
    ("sensitivity_decision_composition.composition_not_supported", &["route_not_supported"]),
    ("sensitivity_decision_composition.invalid_artifact", &["invalid_argument"]),
    ("sensitivity_decision_composition.invalid_surface", &["invalid_argument"]),
    ("sensitivity_decision_composition.sampling_not_reported", &["not_supported"]),
    ("sensitivity_decision_composition.unsupported_adaptation", &["route_not_supported"]),
    ("sensitivity_decision_composition.unsupported_coordinate", &["quantity_semantics_mismatch"]),
    ("sensitivity_decision_composition.wrong_contract", &["decision_contract_unsatisfied"]),
    ("sensitivity_source.invalid_artifact", &["invalid_argument"]),
    ("sensitivity_source.limits_exceeded", &["invalid_argument"]),
    ("sensitivity_source.lineage_digest", &["invalid_argument"]),
    ("sensitivity_source.source_unavailable", &["invalid_argument"]),
    ("sensitivity_source.surface_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.accounting_residual", &["invalid_argument"]),
    ("sequential_evaluation.action_channel", &["invalid_argument"]),
    ("sequential_evaluation.action_cost", &["invalid_argument"]),
    ("sequential_evaluation.action_delayed_outcome", &["invalid_argument"]),
    ("sequential_evaluation.action_index", &["invalid_argument"]),
    ("sequential_evaluation.action_label", &["invalid_argument"]),
    ("sequential_evaluation.action_not_supported", &["history_support_violated"]),
    ("sequential_evaluation.action_support", &["invalid_argument"]),
    ("sequential_evaluation.action_transition", &["invalid_argument"]),
    ("sequential_evaluation.action_utility", &["invalid_argument"]),
    ("sequential_evaluation.artifact_id_missing", &["invalid_argument"]),
    ("sequential_evaluation.artifact_kind_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.behaviour_support_missing", &["history_support_violated"]),
    ("sequential_evaluation.cancelled", &["cancelled_no_claim"]),
    ("sequential_evaluation.channel_binding_mismatch", &["measurement_binding_mismatch"]),
    ("sequential_evaluation.channel_construct_mismatch", &["measurement_binding_mismatch"]),
    ("sequential_evaluation.channel_levels_mismatch", &["measurement_binding_mismatch"]),
    ("sequential_evaluation.channel_rates", &["route_not_supported"]),
    ("sequential_evaluation.channel_route", &["route_not_supported"]),
    ("sequential_evaluation.commit_has_no_observation", &["invalid_argument"]),
    ("sequential_evaluation.comparison_kind_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.comparison_label", &["invalid_argument"]),
    ("sequential_evaluation.comparisons", &["invalid_argument"]),
    ("sequential_evaluation.decision_receipt_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.delayed_outcome_beyond_horizon", &["invalid_argument"]),
    ("sequential_evaluation.differential_channel", &["differential_error_not_pooled"]),
    ("sequential_evaluation.empty_log", &["invalid_argument"]),
    ("sequential_evaluation.exact_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.history_receipt_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.identity_mismatch", &["route_not_supported"]),
    ("sequential_evaluation.importance_weight_overflow", &["invalid_argument"]),
    ("sequential_evaluation.indicator_as_state", &["proxy_offered_as_construct"]),
    ("sequential_evaluation.interval_not_supported", &["uncertainty_representation_insufficient"]),
    ("sequential_evaluation.limit_exceeded", &["invalid_argument"]),
    ("sequential_evaluation.log_partially_counted", &["evidence_double_counted"]),
    ("sequential_evaluation.logged_trajectory_inconsistent", &["invalid_argument"]),
    ("sequential_evaluation.malformed", &["invalid_argument"]),
    ("sequential_evaluation.model_actions", &["invalid_argument"]),
    ("sequential_evaluation.model_field", &["invalid_argument"]),
    ("sequential_evaluation.model_horizon", &["invalid_argument"]),
    ("sequential_evaluation.model_initial", &["invalid_argument"]),
    ("sequential_evaluation.model_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.model_resource_budget", &["invalid_argument"]),
    ("sequential_evaluation.model_states", &["invalid_argument"]),
    ("sequential_evaluation.model_terminal_utility", &["invalid_argument"]),
    ("sequential_evaluation.model_work", &["invalid_argument"]),
    ("sequential_evaluation.no_overlap", &["history_support_violated"]),
    ("sequential_evaluation.observation_impossible", &["invalid_argument"]),
    ("sequential_evaluation.observation_recorded_twice", &["evidence_double_counted"]),
    ("sequential_evaluation.off_policy_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.past_horizon", &["invalid_argument"]),
    ("sequential_evaluation.policy_change_refused", &["route_not_supported"]),
    ("sequential_evaluation.policy_distribution", &["invalid_argument"]),
    ("sequential_evaluation.policy_incomplete", &["invalid_argument"]),
    ("sequential_evaluation.policy_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.policy_version", &["invalid_argument"]),
    ("sequential_evaluation.record_already_counted", &["evidence_double_counted"]),
    ("sequential_evaluation.record_identity", &["invalid_argument"]),
    ("sequential_evaluation.resource_budget_exceeded", &["invalid_argument"]),
    ("sequential_evaluation.return_overflow", &["invalid_argument"]),
    ("sequential_evaluation.rollout_belief", &["invalid_argument"]),
    ("sequential_evaluation.rollout_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.rollouts", &["invalid_argument"]),
    ("sequential_evaluation.session_label", &["invalid_argument"]),
    ("sequential_evaluation.static_score_table", &["route_not_supported"]),
    ("sequential_evaluation.stopping_boundary", &["invalid_argument"]),
    ("sequential_evaluation.temporal_transport_contract_missing", &["transport_missing_evidence"]),
    ("sequential_evaluation.trajectory_identity", &["invalid_argument"]),
    ("sequential_evaluation.trajectory_recorded_twice", &["evidence_double_counted"]),
    ("sequential_evaluation.transported_policy_contract_missing", &["transport_missing_evidence"]),
    ("sequential_evaluation.value_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.version", &["invalid_argument"]),
    ("sequential_evaluation.voi_collect_action", &["invalid_argument"]),
    ("sequential_evaluation.voi_commit_actions", &["invalid_argument"]),
    ("sequential_evaluation.voi_enumeration_mismatch", &["invalid_argument"]),
    ("sequential_evaluation.voi_ranking", &["invalid_argument"]),
    ("sequential_evaluation.voi_scope", &["route_not_supported"]),
    ("sequential_evaluation.weighted_estimate_overflow", &["invalid_argument"]),
    ("sequential_evaluation.weighted_sum_overflow", &["invalid_argument"]),
    ("signal_provider.candidate_mismatch", &["design_signal_invalid"]),
    ("signal_provider.capability_missing", &["external_capability_missing"]),
    ("signal_provider.computational_limit", &["design_signal_invalid"]),
    ("signal_provider.conditional_independence_missing", &["design_signal_invalid"]),
    ("signal_provider.invalid_object", &["external_binding_mismatch"]),
    ("signal_provider.invalid_request", &["invalid_argument"]),
    ("signal_provider.invalid_table", &["design_signal_invalid"]),
    ("signal_provider.law_incoherent", &["design_signal_invalid"]),
    ("signal_provider.missing_lineage", &["design_signal_invalid"]),
    ("signal_provider.observation_quantity_mismatch", &["quantity_semantics_mismatch"]),
    ("signal_provider.posterior_incoherent", &["design_signal_invalid"]),
    ("signal_provider.posterior_quantity_mismatch", &["quantity_semantics_mismatch"]),
    ("signal_provider.prior_mismatch", &["design_signal_invalid"]),
    ("signal_provider.request_fingerprint_mismatch", &["external_binding_mismatch"]),
    ("signal_provider.sample_size_invalid", &["design_signal_invalid"]),
    ("signal_provider.sample_size_mismatch", &["design_signal_invalid"]),
    ("signal_provider.trust_mismatch", &["external_binding_mismatch"]),
    ("signal_provider.trust_native_claimed", &["external_binding_mismatch"]),
    ("smoothed_dose_posterior.band_not_constructed", &["route_not_supported"]),
    ("smoothed_dose_posterior.cancelled", &["transport_budget_cancel"]),
    ("smoothed_dose_posterior.certificate_mismatch", &["invalid_argument"]),
    ("smoothed_dose_posterior.empty_scores", &["invalid_argument"]),
    ("smoothed_dose_posterior.external_power_prior", &["route_not_supported"]),
    ("smoothed_dose_posterior.invalid_graph", &["invalid_argument"]),
    ("smoothed_dose_posterior.mapped_prior", &["route_not_supported"]),
    ("smoothed_dose_posterior.non_positive_scale", &["invalid_argument"]),
    ("smoothed_dose_posterior.not_certified", &["invalid_argument"]),
    ("smoothed_dose_posterior.resource_limit", &["transport_budget_cancel"]),
    ("source_evidence.action_mismatch", &["invalid_argument"]),
    ("source_evidence.coordinate_mismatch", &["invalid_argument"]),
    ("source_evidence.coordinates_unavailable", &["invalid_argument"]),
    ("source_evidence.diagnostic_scope_mismatch", &["invalid_argument"]),
    ("source_evidence.forward_binding_mismatch", &["invalid_argument"]),
    ("source_evidence.invalid_artifact", &["invalid_argument"]),
    ("source_evidence.invalid_contract", &["invalid_argument"]),
    ("source_evidence.invalid_coordinates", &["invalid_argument"]),
    ("source_evidence.invalid_source_artifact", &["invalid_argument"]),
    ("source_evidence.limits_exceeded", &["invalid_argument"]),
    ("source_evidence.point_binding_mismatch", &["invalid_argument"]),
    ("source_evidence.resolution_scope_unsupported", &["invalid_argument"]),
    ("source_evidence.response_unavailable", &["invalid_argument"]),
    ("source_evidence.source_binding_mismatch", &["invalid_argument"]),
    ("source_evidence.source_encoding_failed", &["invalid_argument"]),
    ("source_evidence.source_execution_failed", &["invalid_argument"]),
    ("source_evidence.source_unverified", &["invalid_argument"]),
    ("source_projection.affine_contract_refused", &["decision_contract_unsatisfied"]),
    ("source_projection.attestation_unavailable", &["invalid_argument"]),
    ("source_projection.causal_source_unavailable", &["invalid_argument"]),
    ("source_projection.coordinate_mismatch", &["invalid_argument"]),
    ("source_projection.identity_mismatch", &["invalid_argument"]),
    ("source_projection.invalid_artifact", &["invalid_argument"]),
    ("source_projection.invalid_request", &["invalid_argument"]),
    ("source_projection.limits_exceeded", &["invalid_argument"]),
    ("source_projection.mean_binding_mismatch", &["invalid_argument"]),
    ("source_projection.mean_grid_required", &["invalid_argument"]),
    ("source_projection.original_consumer_refused", &["invalid_argument"]),
    ("source_projection.replay_mismatch", &["invalid_argument"]),
    ("source_projection.response_unavailable", &["invalid_argument"]),
    ("source_projection.source_binding_mismatch", &["invalid_argument"]),
    ("source_projection.unsupported_coordinate", &["invalid_argument"]),
    ("split_effect.degenerate_variance", &["invalid_argument"]),
    ("split_effect.design_singular", &["design_rank_deficient"]),
    ("split_effect.discovery_failed", &["invalid_argument"]),
    ("split_effect.outcome_precedes_treatment", &["invalid_argument"]),
    ("split_effect.search_not_finished", &["discovered_structure_unresolved"]),
    ("split_effect.source_invalid", &["invalid_argument"]),
    ("split_effect.source_not_canonical", &["invalid_argument"]),
    ("split_effect.structure_class", &["invalid_argument"]),
    ("split_effect.structure_unresolved", &["discovered_structure_unresolved"]),
    ("split_effect.too_few_rows", &["invalid_argument"]),
    ("structured_refusals.malformed_detail", &["invalid_argument"]),
    ("structured_refusals.missing_stage", &["invalid_argument"]),
    ("structured_refusals.unregistered_code", &["invalid_argument"]),
    ("study_candidate.declarations_required", &["design_signal_invalid"]),
    ("study_candidate.sample_below_obligation", &["design_signal_invalid"]),
    ("study_candidate.wrong_contract", &["design_signal_invalid"]),
    ("study_plan.arrival_mismatch", &["invalid_argument"]),
    ("study_plan.arrival_not_identified", &["transport_not_certified"]),
    ("study_plan.bounds_exceeded", &["route_not_supported"]),
    ("study_plan.budget", &["transport_budget_cancel"]),
    ("study_plan.invalid_artifact", &["transport_not_certified"]),
    ("study_plan.invalid_candidate", &["invalid_argument"]),
    ("study_plan.invalid_query", &["invalid_argument"]),
    ("study_plan.no_failure_to_repair", &["invalid_argument"]),
    ("study_plan.none_certified", &["transport_not_certified"]),
    ("study_plan.theorem_limited", &["transport_proven_non_transportable"]),
    ("survival_effect.arm_not_populated", &["arm_not_populated"]),
    ("survival_effect.cancelled", &["cancelled_no_claim"]),
    ("survival_effect.censoring_assumption_missing", &["required_option_missing"]),
    ("survival_effect.censoring_assumption_unknown", &["invalid_argument"]),
    ("survival_effect.censoring_covariates_missing", &["invalid_argument"]),
    ("survival_effect.censoring_support", &["positivity_violated"]),
    ("survival_effect.horizon_beyond_follow_up", &["positivity_violated"]),
    ("survival_effect.horizon_missing", &["required_option_missing"]),
    ("survival_effect.invalid_data", &["invalid_argument"]),
    ("survival_effect.invalid_event", &["invalid_argument"]),
    ("survival_effect.invalid_floor", &["invalid_argument"]),
    ("survival_effect.invalid_folds", &["invalid_argument"]),
    ("survival_effect.invalid_horizon", &["invalid_argument"]),
    ("survival_effect.invalid_levels", &["invalid_argument"]),
    ("survival_effect.invalid_request", &["invalid_argument"]),
    ("survival_effect.too_few_rows", &["invalid_argument"]),
    ("survival_effect.training_fold_without_events", &["arm_not_populated"]),
    ("survival_effect.treatment_positivity", &["positivity_violated"]),
    ("survival_effect.undeclared_level", &["treatment_level_undeclared"]),
    ("temporal_counterfactual.artifact_changed", &["route_not_supported"]),
    ("temporal_counterfactual.fit_mismatch", &["invalid_argument"]),
    ("temporal_counterfactual.history_name_invalid", &["invalid_argument"]),
    ("temporal_counterfactual.horizon_exceeded", &["invalid_argument"]),
    ("temporal_counterfactual.invalid_graph", &["invalid_argument"]),
    ("temporal_counterfactual.invalid_identity", &["invalid_argument"]),
    ("temporal_counterfactual.invalid_request", &["invalid_argument"]),
    ("temporal_counterfactual.latent_confounding", &["route_not_supported"]),
    ("temporal_counterfactual.limits_exceeded", &["invalid_argument"]),
    ("temporal_counterfactual.non_finite_history", &["invalid_argument"]),
    ("temporal_counterfactual.refuting_history", &["route_not_supported"]),
    ("temporal_counterfactual.shared_history_missing", &["route_not_supported"]),
    ("temporal_counterfactual.time_misaligned", &["route_not_supported"]),
    ("temporal_counterfactual.too_many_units", &["invalid_argument"]),
    ("temporal_counterfactual.unpaired_histories", &["route_not_supported"]),
    ("temporal_interval.consumer_unavailable", &["route_not_supported"]),
    ("temporal_interval.interval_not_supported", &["route_not_supported"]),
    ("temporal_interval.invalid_estimand", &["invalid_argument"]),
    ("temporal_interval.retired_method", &["route_not_supported"]),
    ("temporal_interval.retired_response_method", &["route_not_supported"]),
    ("temporal_interval.studentized_bounds_exceeded", &["route_not_supported"]),
    ("temporal_interval.studentized_estimator_not_certified", &["route_not_supported"]),
    ("temporal_interval.studentized_score_mismatch", &["route_not_supported"]),
    ("temporal_interval.studentized_unbalanced", &["route_not_supported"]),
    ("temporal_interval.studentized_zero_variance", &["route_not_supported"]),
    ("temporal_interval.too_few_units", &["too_few_clusters"]),
    ("temporal_interval.too_many_replicates", &["route_not_supported"]),
    ("temporal_interval.unknown_units", &["route_not_supported"]),
    ("temporal_interval.unsupported_history", &["route_not_supported"]),
    ("temporal_policy_transport.history_alphabet_incompatible", &["transport_missing_evidence"]),
    ("temporal_policy_transport.horizon_mismatch", &["route_not_supported"]),
    ("temporal_policy_transport.initial_law_declared_invariant", &["invalid_argument"]),
    ("temporal_policy_transport.invariance_binding_mismatch", &["transport_missing_evidence"]),
    ("temporal_policy_transport.mechanism_declared_twice", &["invalid_argument"]),
    ("temporal_policy_transport.mechanism_status_missing", &["transport_missing_evidence"]),
    ("temporal_policy_transport.mechanism_time_varying", &["transport_missing_evidence"]),
    ("temporal_policy_transport.mechanism_unknown", &["invalid_argument"]),
    ("temporal_policy_transport.target_initial_law", &["invalid_argument"]),
    ("temporal_policy_transport.target_initial_law_missing", &["transport_missing_evidence"]),
    ("temporal_policy_transport.target_is_source", &["invalid_argument"]),
    ("temporal_policy_transport.target_period", &["invalid_argument"]),
    ("temporal_policy_transport.target_support_outside_source", &["transport_support_failure"]),
    ("temporal_policy_transport.time_invariance_missing", &["transport_missing_evidence"]),
    ("temporal_policy_transport.value_mismatch", &["invalid_argument"]),
    ("temporal_refresh.factor_window_mismatch", &["invalid_argument"]),
    ("temporal_refresh.horizon_changed", &["route_not_supported"]),
    ("temporal_refresh.intervention_history_changed", &["route_not_supported"]),
    ("temporal_refresh.invalid_initial_law", &["invalid_argument"]),
    ("temporal_refresh.invalid_replacement", &["invalid_argument"]),
    ("temporal_refresh.lag_alignment_changed", &["route_not_supported"]),
    ("temporal_refresh.no_refresh", &["invalid_argument"]),
    ("temporal_refresh.premises_changed", &["route_not_supported"]),
    ("temporal_refresh.receipt_digest_mismatch", &["route_not_supported"]),
    ("temporal_refresh.receipt_mismatch", &["route_not_supported"]),
    ("temporal_refresh.stale_interval", &["route_not_supported"]),
    ("temporal_refresh.stale_snapshot", &["route_not_supported"]),
    ("temporal_refresh.window_missing", &["invalid_argument"]),
    ("temporal_transport.bounds_exceeded", &["route_not_supported"]),
    ("temporal_transport.checked_obstruction", &["transport_proven_non_transportable"]),
    ("temporal_transport.history_budget", &["transport_budget_cancel"]),
    ("temporal_transport.history_outside_support", &["transport_support_failure"]),
    ("temporal_transport.horizon", &["route_not_supported"]),
    ("temporal_transport.interval_requested", &["estimator_inference_mismatch"]),
    ("temporal_transport.invalid_sequence", &["invalid_argument"]),
    ("temporal_transport.invalid_spec", &["invalid_argument"]),
    ("temporal_transport.missing_evidence", &["transport_missing_evidence"]),
    ("temporal_transport.search_incomplete", &["transport_not_certified"]),
    ("temporal_transport.target_law_mismatch", &["transport_missing_evidence"]),
    ("temporal_transport.target_law_missing", &["transport_missing_evidence"]),
    ("tier_diagnostics.cancelled", &["cancelled_no_claim"]),
    ("tier_diagnostics.evalue_embeds_interval_limit", &["diagnostic_not_available"]),
    ("tier_diagnostics.evalue_not_computed", &["diagnostic_not_available"]),
    ("tier_diagnostics.no_evalue_interval_method", &["diagnostic_not_available"]),
    ("tier_diagnostics.not_a_tiered_average_result", &["invalid_argument"]),
    ("tier_diagnostics.overlap_not_carried", &["diagnostic_not_available"]),
    ("tier_diagnostics.unknown_scenarios_no_evalue", &["diagnostic_not_available"]),
    ("tier_diagnostics.unknown_scenarios_no_overlap", &["diagnostic_not_available"]),
    ("transport_revalidation.claim_target_mismatch", &["quantity_semantics_mismatch"]),
    ("transport_revalidation.duplicate_claim", &["invalid_argument"]),
    ("transport_revalidation.identity_mismatch", &["route_not_supported"]),
    ("transport_revalidation.provider_trust_not_applicable", &["invalid_argument"]),
    ("transport_revalidation.replay_mismatch", &["invalid_argument"]),
    ("transport_revalidation.scene_invalid", &["invalid_argument"]),
    ("transport_revalidation.source_undeclared", &["transport_missing_evidence"]),
    ("transport_revalidation.unknown_claim", &["invalid_argument"]),
    ("transport_revalidation.value_digest", &["invalid_argument"]),
    ("transport_sensitivity.blank_identity", &["invalid_argument"]),
    ("transport_sensitivity.coordinate_out_of_range", &["invalid_argument"]),
    ("transport_sensitivity.interval_not_composed", &["route_not_supported"]),
    ("transport_sensitivity.invalid_bound", &["invalid_argument"]),
    ("transport_sensitivity.invalid_grid", &["invalid_argument"]),
    ("transport_sensitivity.law_response_not_scalar", &["route_not_supported"]),
    ("transport_sensitivity.no_scalar_family", &["route_not_supported"]),
    ("transport_sensitivity.non_finite_point", &["invalid_argument"]),
    ("transport_sensitivity.not_transported", &["invalid_argument"]),
    ("transport_sensitivity.proportional_at_zero", &["invalid_argument"]),
    ("transport_sensitivity.quantity_not_in_target", &["quantity_semantics_mismatch"]),
    ("transport_sensitivity.route_undeclared", &["route_not_supported"]),
    ("transport_sensitivity.surface_refused", &["invalid_argument"]),
    ("transported_counterfactual.artifact_changed", &["route_not_supported"]),
    ("transported_counterfactual.cross_world_independence_missing", &["route_not_supported"]),
    ("transported_counterfactual.factor_missing", &["transport_missing_evidence"]),
    ("transported_counterfactual.invalid_diagram", &["invalid_argument"]),
    ("transported_counterfactual.invalid_factor", &["invalid_argument"]),
    ("transported_counterfactual.invalid_identity", &["invalid_argument"]),
    ("transported_counterfactual.invalid_law", &["invalid_argument"]),
    ("transported_counterfactual.invalid_model", &["invalid_argument"]),
    ("transported_counterfactual.invalid_query", &["invalid_argument"]),
    ("transported_counterfactual.invalid_request", &["invalid_argument"]),
    ("transported_counterfactual.joint_theorem_not_derived", &["route_not_supported"]),
    ("transported_counterfactual.limits_exceeded", &["invalid_argument"]),
    ("transported_counterfactual.noise_law_not_shared", &["route_not_supported"]),
    ("transported_counterfactual.nonadditive_mechanism", &["route_not_supported"]),
    ("transported_counterfactual.overlap_failure", &["transport_support_failure"]),
    ("transported_counterfactual.prerequisite_missing", &["transport_missing_evidence"]),
    ("transported_counterfactual.selection_on_mechanism", &["route_not_supported"]),
    ("transported_decision.calibration_label_inflated", &["transport_not_certified"]),
    ("transported_decision.coordinate_not_in_target", &["quantity_semantics_mismatch"]),
    ("transported_decision.exclusions_leave_no_comparison", &["decision_contract_unsatisfied"]),
    ("transported_decision.interval_not_supported", &["not_supported"]),
    ("transported_decision.invalid_value", &["invalid_argument"]),
    ("transported_decision.law_not_supplied", &["decision_contract_unsatisfied"]),
    ("transported_decision.mixed_transports", &["transport_not_certified"]),
    ("transported_decision.native_label_on_transported_value", &["transport_not_certified"]),
    ("transported_decision.not_transported", &["invalid_argument"]),
    ("transported_decision.route_needs_typed_source", &["invalid_argument"]),
    ("transported_decision.support_label_inflated", &["transport_not_certified"]),
    ("transported_decision.support_rule_unmet", &["transport_support_failure"]),
    ("transported_decision.transport_identity_incomplete", &["transport_missing_evidence"]),
    ("transported_decision.transport_identity_not_digest", &["transport_missing_evidence"]),
    ("transported_decision.transported_not_accepted", &["decision_contract_unsatisfied"]),
    ("transported_decision.uncertainty_requirement", &["decision_contract_unsatisfied"]),
    ("transported_decision_artifact.bounds_exceeded", &["invalid_argument"]),
    ("transported_decision_artifact.identity_expected", &["decision_contract_unsatisfied"]),
    ("transported_decision_artifact.invalid", &["invalid_argument"]),
    ("transported_decision_artifact.law_not_exportable", &["route_not_supported"]),
    ("transported_decision_artifact.replay_mismatch", &["invalid_argument"]),
    ("transported_effect.cohort_invalid", &["invalid_argument"]),
    ("transported_effect.cohort_not_numeric", &["invalid_argument"]),
    ("transported_effect.decision_source_mismatch", &["transport_not_certified"]),
    ("transported_effect.declaration_invalid", &["invalid_argument"]),
    ("transported_effect.feature_schema_mismatch", &["schema_mismatch"]),
    ("transported_effect.identity_mismatch", &["transport_not_certified"]),
    ("transported_effect.interval_not_supported", &["route_not_supported"]),
    ("transported_effect.invalid_request", &["invalid_argument"]),
    ("transported_effect.law_not_supplied", &["decision_contract_unsatisfied"]),
    ("transported_effect.link_not_identity", &["quantity_semantics_mismatch"]),
    ("transported_effect.model_invalid", &["invalid_argument"]),
    ("transported_effect.model_not_portable", &["invalid_argument"]),
    ("transported_effect.modifier_set_mismatch", &["transport_not_certified"]),
    ("transported_effect.no_overlap", &["transport_support_failure"]),
    ("transported_effect.non_finite_result", &["invalid_argument"]),
    ("transported_effect.out_of_support", &["transport_support_failure"]),
    ("transported_effect.outcome_units_required", &["invalid_argument"]),
    ("transported_effect.population_not_different", &["quantity_semantics_mismatch"]),
    ("transported_effect.prediction_refused", &["invalid_argument"]),
    ("transported_effect.selection_enters_mechanism", &["transport_not_certified"]),
    ("transported_effect.shift_declaration_mismatch", &["transport_not_certified"]),
    ("transported_effect.shift_declaration_required", &["transport_missing_evidence"]),
    ("transported_effect.support_contract_absent", &["invalid_argument"]),
    ("transported_effect.weights_invalid", &["invalid_argument"]),
    ("transported_effect.weights_not_normalized", &["invalid_argument"]),
    ("transported_policy.diagram_binding_mismatch", &["transport_missing_evidence"]),
    ("transported_policy.initial_law_declared_invariant", &["invalid_argument"]),
    ("transported_policy.mechanism_declared_twice", &["invalid_argument"]),
    ("transported_policy.mechanism_not_invariant", &["transport_missing_evidence"]),
    ("transported_policy.mechanism_status_missing", &["transport_missing_evidence"]),
    ("transported_policy.mechanism_unknown", &["invalid_argument"]),
    ("transported_policy.selection_diagram_missing", &["transport_missing_evidence"]),
    ("transported_policy.target_initial_law", &["invalid_argument"]),
    ("transported_policy.target_initial_law_missing", &["transport_missing_evidence"]),
    ("transported_policy.target_is_source", &["invalid_argument"]),
    ("transported_policy.target_population", &["invalid_argument"]),
    ("transported_policy.target_support_outside_source", &["transport_support_failure"]),
    ("transported_policy.target_utility_digest", &["invalid_argument"]),
    ("transported_policy.target_utility_missing", &["transport_missing_evidence"]),
    ("transported_policy.target_utility_provenance_missing", &["transport_missing_evidence"]),
    ("transported_policy.target_utility_support", &["invalid_argument"]),
    ("transported_policy.target_utility_value", &["invalid_argument"]),
    ("transported_policy.utility_declared_invariant", &["invalid_argument"]),
    ("transported_policy.value_mismatch", &["invalid_argument"]),
    ("transported_synthesis.native_population_blank", &["invalid_argument"]),
    ("transported_synthesis.no_contributions", &["invalid_argument"]),
    ("transported_synthesis.numeric_refused", &["invalid_argument"]),
    ("transported_synthesis.overlap_not_modelled", &["invalid_argument"]),
    ("transported_synthesis.prior_role_mismatch", &["invalid_argument"]),
    ("transported_synthesis.shared_transport_needs_covariance", &["joint_law_required"]),
    ("transported_synthesis.source_listed_twice", &["invalid_argument"]),
    ("transported_synthesis.target_population_mismatch", &["quantity_semantics_mismatch"]),
    ("transported_synthesis.transport_identity_incomplete", &["transport_missing_evidence"]),
    ("transported_synthesis.transport_variance_on_native_source", &["invalid_argument"]),
    ("transported_synthesis.transport_variance_undeclared", &["transport_missing_evidence"]),
    ("transported_synthesis.unknown_source", &["invalid_argument"]),
    ("treatment_levels.cancelled", &["cancelled_no_claim"]),
    ("treatment_levels.contrast_level_not_declared", &["invalid_argument"]),
    ("treatment_levels.full_population_undeclared_level", &["treatment_level_undeclared"]),
    ("treatment_levels.invalid_data", &["invalid_argument"]),
    ("treatment_levels.invalid_level_set", &["invalid_argument"]),
    ("treatment_levels.invalid_request", &["invalid_argument"]),
    ("treatment_levels.level_not_populated", &["arm_not_populated"]),
    ("treatment_levels.levels_missing", &["invalid_argument"]),
    ("treatment_levels.sparse_level", &["arm_not_populated"]),
    ("treatment_levels.training_fold_missing_level", &["arm_not_populated"]),
    ("treatment_levels.undeclared_level", &["treatment_level_undeclared"]),
    ("two_step_posterior.band_not_constructed", &["route_not_supported"]),
    ("two_step_posterior.empty_law", &["invalid_argument"]),
    ("two_step_posterior.external_power_prior", &["route_not_supported"]),
    ("two_step_posterior.mapped_prior", &["route_not_supported"]),
    ("two_step_posterior.non_positive_scale", &["invalid_argument"]),
    ("typed_mechanism_transport.cycle", &["mechanism_composition_incoherent"]),
    ("typed_mechanism_transport.dependence_invalid", &["uncertainty_representation_insufficient"]),
    ("typed_mechanism_transport.design_not_shared", &["quantity_semantics_mismatch"]),
    ("typed_mechanism_transport.fit_not_converged", &["mechanism_fit_not_converged"]),
    ("typed_mechanism_transport.fit_refused", &["mechanism_composition_incoherent"]),
    ("typed_mechanism_transport.invariance_contradicted", &["quantity_semantics_mismatch"]),
    ("typed_mechanism_transport.invariance_level_invalid", &["mechanism_composition_incoherent"]),
    ("typed_mechanism_transport.parameter_dependence_incomplete", &["uncertainty_representation_insufficient"]),
    ("typed_mechanism_transport.parameter_dependence_unknown", &["uncertainty_representation_insufficient"]),
    ("typed_mechanism_transport.parent_without_mechanism", &["mechanism_composition_incoherent"]),
    ("typed_mechanism_transport.parents_changed", &["quantity_semantics_mismatch"]),
    ("typed_mechanism_transport.regime_not_declared_invariant", &["quantity_semantics_mismatch"]),
    ("typed_mechanism_transport.selection_target_unknown_variable", &["quantity_semantics_mismatch"]),
    ("typed_mechanism_transport.shifted_without_target_evidence", &["transport_missing_evidence"]),
    ("typed_mechanism_transport.target_specification_differs", &["quantity_semantics_mismatch"]),
    ("typed_mechanism_transport.transport_identity_mismatch", &["quantity_semantics_mismatch"]),
    ("typed_mechanism_transport.unknown_variable", &["mechanism_composition_incoherent"]),
    ("vector_posterior.band_not_constructed", &["route_not_supported"]),
    ("vector_posterior.external_power_prior", &["route_not_supported"]),
    ("vector_posterior.mapped_prior", &["route_not_supported"]),
    ("vector_posterior.non_positive_scale", &["invalid_argument"]),
    ("vector_treatment.adjustment_set_mismatch", &["route_not_supported"]),
    ("vector_treatment.cancelled", &["cancelled_no_claim"]),
    ("vector_treatment.cluster_ids_required", &["invalid_argument"]),
    ("vector_treatment.cluster_unread", &["invalid_argument"]),
    ("vector_treatment.collinear_treatments", &["design_rank_deficient"]),
    ("vector_treatment.data_not_bound", &["route_not_supported"]),
    ("vector_treatment.degenerate_covariance", &["invalid_argument"]),
    ("vector_treatment.duplicate_contrast_coefficient", &["invalid_argument"]),
    ("vector_treatment.duplicate_name", &["invalid_argument"]),
    ("vector_treatment.empty_contrast", &["invalid_argument"]),
    ("vector_treatment.export_budget_exceeded", &["route_not_supported"]),
    ("vector_treatment.invalid_identity", &["invalid_argument"]),
    ("vector_treatment.invalid_lag", &["invalid_argument"]),
    ("vector_treatment.invalid_request", &["invalid_argument"]),
    ("vector_treatment.lag_required", &["invalid_argument"]),
    ("vector_treatment.lag_unread", &["invalid_argument"]),
    ("vector_treatment.leverage_one", &["invalid_argument"]),
    ("vector_treatment.limits_exceeded", &["invalid_argument"]),
    ("vector_treatment.memory_budget_exceeded", &["route_not_supported"]),
    ("vector_treatment.non_finite_value", &["invalid_argument"]),
    ("vector_treatment.rank_deficient_adjustment", &["design_rank_deficient"]),
    ("vector_treatment.row_count_mismatch", &["route_not_supported"]),
    ("vector_treatment.row_snapshot_mismatch", &["route_not_supported"]),
    ("vector_treatment.summary_inconsistent", &["invalid_argument"]),
    ("vector_treatment.summary_model_based_only", &["route_not_supported"]),
    ("vector_treatment.too_few_clusters", &["invalid_argument"]),
    ("vector_treatment.too_few_rows", &["invalid_argument"]),
    ("vector_treatment.too_few_treatments", &["invalid_argument"]),
    ("vector_treatment.too_many_treatments", &["invalid_argument"]),
    ("vector_treatment.treatment_without_variation", &["design_rank_deficient"]),
    ("vector_treatment.unknown_contrast_coefficient", &["invalid_argument"]),
    ("vector_treatment.weights_need_robust_covariance", &["option_not_applicable"]),
    ("vector_treatment.wrong_contract", &["route_not_supported"]),
];

/// The reason codes `detail` is refused under, or `None` for an unregistered id.
#[must_use]
pub fn codes_of(detail: &str) -> Option<&'static [&'static str]> {
    DETAIL_CODES
        .binary_search_by(|(id, _)| (*id).cmp(detail))
        .ok()
        .map(|at| DETAIL_CODES[at].1)
}

/// Details in the `adaptive_inference` family.
pub mod adaptive_inference {
    /// Detail `adaptive_inference.analysis_invalid`.
    pub const ANALYSIS_INVALID: &str = "adaptive_inference.analysis_invalid";
    /// Detail `adaptive_inference.analysis_route_mismatch`.
    pub const ANALYSIS_ROUTE_MISMATCH: &str = "adaptive_inference.analysis_route_mismatch";
    /// Detail `adaptive_inference.anytime_premise_contexts`.
    pub const ANYTIME_PREMISE_CONTEXTS: &str = "adaptive_inference.anytime_premise_contexts";
    /// Detail `adaptive_inference.anytime_settings_invalid`.
    pub const ANYTIME_SETTINGS_INVALID: &str = "adaptive_inference.anytime_settings_invalid";
    /// Detail `adaptive_inference.anytime_settings_missing`.
    pub const ANYTIME_SETTINGS_MISSING: &str = "adaptive_inference.anytime_settings_missing";
    /// Detail `adaptive_inference.arm_not_observed`.
    pub const ARM_NOT_OBSERVED: &str = "adaptive_inference.arm_not_observed";
    /// Detail `adaptive_inference.arm_out_of_range`.
    pub const ARM_OUT_OF_RANGE: &str = "adaptive_inference.arm_out_of_range";
    /// Detail `adaptive_inference.arms_invalid`.
    pub const ARMS_INVALID: &str = "adaptive_inference.arms_invalid";
    /// Detail `adaptive_inference.artifact_size_exceeded`.
    pub const ARTIFACT_SIZE_EXCEEDED: &str = "adaptive_inference.artifact_size_exceeded";
    /// Detail `adaptive_inference.assigned_probability_zero`.
    pub const ASSIGNED_PROBABILITY_ZERO: &str = "adaptive_inference.assigned_probability_zero";
    /// Detail `adaptive_inference.assignment_not_replayed`.
    pub const ASSIGNMENT_NOT_REPLAYED: &str = "adaptive_inference.assignment_not_replayed";
    /// Detail `adaptive_inference.assignment_probabilities_missing`.
    pub const ASSIGNMENT_PROBABILITIES_MISSING: &str = "adaptive_inference.assignment_probabilities_missing";
    /// Detail `adaptive_inference.batch_incomplete`.
    pub const BATCH_INCOMPLETE: &str = "adaptive_inference.batch_incomplete";
    /// Detail `adaptive_inference.between_unit_variance_missing`.
    pub const BETWEEN_UNIT_VARIANCE_MISSING: &str = "adaptive_inference.between_unit_variance_missing";
    /// Detail `adaptive_inference.borrowing_fixed_after_first_look`.
    pub const BORROWING_FIXED_AFTER_FIRST_LOOK: &str = "adaptive_inference.borrowing_fixed_after_first_look";
    /// Detail `adaptive_inference.cancelled`.
    pub const CANCELLED: &str = "adaptive_inference.cancelled";
    /// Detail `adaptive_inference.cell_incoherent`.
    pub const CELL_INCOHERENT: &str = "adaptive_inference.cell_incoherent";
    /// Detail `adaptive_inference.cell_not_delivered`.
    pub const CELL_NOT_DELIVERED: &str = "adaptive_inference.cell_not_delivered";
    /// Detail `adaptive_inference.choice_outcome_needs_choice_contract`.
    pub const CHOICE_OUTCOME_NEEDS_CHOICE_CONTRACT: &str = "adaptive_inference.choice_outcome_needs_choice_contract";
    /// Detail `adaptive_inference.contexts_invalid`.
    pub const CONTEXTS_INVALID: &str = "adaptive_inference.contexts_invalid";
    /// Detail `adaptive_inference.correlation_invalid`.
    pub const CORRELATION_INVALID: &str = "adaptive_inference.correlation_invalid";
    /// Detail `adaptive_inference.declared_looks_exceeded`.
    pub const DECLARED_LOOKS_EXCEEDED: &str = "adaptive_inference.declared_looks_exceeded";
    /// Detail `adaptive_inference.declared_units_exceeded`.
    pub const DECLARED_UNITS_EXCEEDED: &str = "adaptive_inference.declared_units_exceeded";
    /// Detail `adaptive_inference.delay_not_ignorable`.
    pub const DELAY_NOT_IGNORABLE: &str = "adaptive_inference.delay_not_ignorable";
    /// Detail `adaptive_inference.exact_law_states_exceeded`.
    pub const EXACT_LAW_STATES_EXCEEDED: &str = "adaptive_inference.exact_law_states_exceeded";
    /// Detail `adaptive_inference.exact_law_unavailable`.
    pub const EXACT_LAW_UNAVAILABLE: &str = "adaptive_inference.exact_law_unavailable";
    /// Detail `adaptive_inference.exact_law_work_exceeded`.
    pub const EXACT_LAW_WORK_EXCEEDED: &str = "adaptive_inference.exact_law_work_exceeded";
    /// Detail `adaptive_inference.exposure_differs_from_assignment`.
    pub const EXPOSURE_DIFFERS_FROM_ASSIGNMENT: &str = "adaptive_inference.exposure_differs_from_assignment";
    /// Detail `adaptive_inference.exposure_history_missing`.
    pub const EXPOSURE_HISTORY_MISSING: &str = "adaptive_inference.exposure_history_missing";
    /// Detail `adaptive_inference.exposure_invalid`.
    pub const EXPOSURE_INVALID: &str = "adaptive_inference.exposure_invalid";
    /// Detail `adaptive_inference.final_analysis_missing`.
    pub const FINAL_ANALYSIS_MISSING: &str = "adaptive_inference.final_analysis_missing";
    /// Detail `adaptive_inference.fixed_sample_data_dependent_stopping`.
    pub const FIXED_SAMPLE_DATA_DEPENDENT_STOPPING: &str = "adaptive_inference.fixed_sample_data_dependent_stopping";
    /// Detail `adaptive_inference.fixed_sample_design_adaptive`.
    pub const FIXED_SAMPLE_DESIGN_ADAPTIVE: &str = "adaptive_inference.fixed_sample_design_adaptive";
    /// Detail `adaptive_inference.fixed_sample_on_adaptive_data`.
    pub const FIXED_SAMPLE_ON_ADAPTIVE_DATA: &str = "adaptive_inference.fixed_sample_on_adaptive_data";
    /// Detail `adaptive_inference.group_sequential_assignment_adaptive`.
    pub const GROUP_SEQUENTIAL_ASSIGNMENT_ADAPTIVE: &str = "adaptive_inference.group_sequential_assignment_adaptive";
    /// Detail `adaptive_inference.guarantee_not_transferable`.
    pub const GUARANTEE_NOT_TRANSFERABLE: &str = "adaptive_inference.guarantee_not_transferable";
    /// Detail `adaptive_inference.historical_borrowing_missing`.
    pub const HISTORICAL_BORROWING_MISSING: &str = "adaptive_inference.historical_borrowing_missing";
    /// Detail `adaptive_inference.historical_prior_invalid`.
    pub const HISTORICAL_PRIOR_INVALID: &str = "adaptive_inference.historical_prior_invalid";
    /// Detail `adaptive_inference.independence_assumed_for_dependent_measurements`.
    pub const INDEPENDENCE_ASSUMED_FOR_DEPENDENT_MEASUREMENTS: &str = "adaptive_inference.independence_assumed_for_dependent_measurements";
    /// Detail `adaptive_inference.input_invalid`.
    pub const INPUT_INVALID: &str = "adaptive_inference.input_invalid";
    /// Detail `adaptive_inference.interim_analyses_missing`.
    pub const INTERIM_ANALYSES_MISSING: &str = "adaptive_inference.interim_analyses_missing";
    /// Detail `adaptive_inference.interval_names_no_quantity`.
    pub const INTERVAL_NAMES_NO_QUANTITY: &str = "adaptive_inference.interval_names_no_quantity";
    /// Detail `adaptive_inference.inversion_premise_contexts`.
    pub const INVERSION_PREMISE_CONTEXTS: &str = "adaptive_inference.inversion_premise_contexts";
    /// Detail `adaptive_inference.ledger_limit`.
    pub const LEDGER_LIMIT: &str = "adaptive_inference.ledger_limit";
    /// Detail `adaptive_inference.look_out_of_range`.
    pub const LOOK_OUT_OF_RANGE: &str = "adaptive_inference.look_out_of_range";
    /// Detail `adaptive_inference.looks_not_ordered`.
    pub const LOOKS_NOT_ORDERED: &str = "adaptive_inference.looks_not_ordered";
    /// Detail `adaptive_inference.looks_overlap_unknown`.
    pub const LOOKS_OVERLAP_UNKNOWN: &str = "adaptive_inference.looks_overlap_unknown";
    /// Detail `adaptive_inference.looks_share_observations`.
    pub const LOOKS_SHARE_OBSERVATIONS: &str = "adaptive_inference.looks_share_observations";
    /// Detail `adaptive_inference.measurements_invalid`.
    pub const MEASUREMENTS_INVALID: &str = "adaptive_inference.measurements_invalid";
    /// Detail `adaptive_inference.no_stagewise_adjustment_procedure`.
    pub const NO_STAGEWISE_ADJUSTMENT_PROCEDURE: &str = "adaptive_inference.no_stagewise_adjustment_procedure";
    /// Detail `adaptive_inference.no_stopping_reliability_procedure`.
    pub const NO_STOPPING_RELIABILITY_PROCEDURE: &str = "adaptive_inference.no_stopping_reliability_procedure";
    /// Detail `adaptive_inference.numerical_failure`.
    pub const NUMERICAL_FAILURE: &str = "adaptive_inference.numerical_failure";
    /// Detail `adaptive_inference.outcome_availability_invalid`.
    pub const OUTCOME_AVAILABILITY_INVALID: &str = "adaptive_inference.outcome_availability_invalid";
    /// Detail `adaptive_inference.outcome_availability_missing`.
    pub const OUTCOME_AVAILABILITY_MISSING: &str = "adaptive_inference.outcome_availability_missing";
    /// Detail `adaptive_inference.outcome_dependent_delay`.
    pub const OUTCOME_DEPENDENT_DELAY: &str = "adaptive_inference.outcome_dependent_delay";
    /// Detail `adaptive_inference.outcome_invalid`.
    pub const OUTCOME_INVALID: &str = "adaptive_inference.outcome_invalid";
    /// Detail `adaptive_inference.outcome_structure_missing`.
    pub const OUTCOME_STRUCTURE_MISSING: &str = "adaptive_inference.outcome_structure_missing";
    /// Detail `adaptive_inference.pairing_not_supported`.
    pub const PAIRING_NOT_SUPPORTED: &str = "adaptive_inference.pairing_not_supported";
    /// Detail `adaptive_inference.paradigm_not_supported_for_cell`.
    pub const PARADIGM_NOT_SUPPORTED_FOR_CELL: &str = "adaptive_inference.paradigm_not_supported_for_cell";
    /// Detail `adaptive_inference.paradigm_rule_mismatch`.
    pub const PARADIGM_RULE_MISMATCH: &str = "adaptive_inference.paradigm_rule_mismatch";
    /// Detail `adaptive_inference.policy_history_invalid`.
    pub const POLICY_HISTORY_INVALID: &str = "adaptive_inference.policy_history_invalid";
    /// Detail `adaptive_inference.policy_history_missing`.
    pub const POLICY_HISTORY_MISSING: &str = "adaptive_inference.policy_history_missing";
    /// Detail `adaptive_inference.policy_selection_missing`.
    pub const POLICY_SELECTION_MISSING: &str = "adaptive_inference.policy_selection_missing";
    /// Detail `adaptive_inference.positivity_floor`.
    pub const POSITIVITY_FLOOR: &str = "adaptive_inference.positivity_floor";
    /// Detail `adaptive_inference.posterior_premise_outcome_dependent_delay`.
    pub const POSTERIOR_PREMISE_OUTCOME_DEPENDENT_DELAY: &str = "adaptive_inference.posterior_premise_outcome_dependent_delay";
    /// Detail `adaptive_inference.probabilities_invalid`.
    pub const PROBABILITIES_INVALID: &str = "adaptive_inference.probabilities_invalid";
    /// Detail `adaptive_inference.probability_floor_invalid`.
    pub const PROBABILITY_FLOOR_INVALID: &str = "adaptive_inference.probability_floor_invalid";
    /// Detail `adaptive_inference.procedure_limit_exceeded`.
    pub const PROCEDURE_LIMIT_EXCEEDED: &str = "adaptive_inference.procedure_limit_exceeded";
    /// Detail `adaptive_inference.repeated_measures_paradigm_not_derived`.
    pub const REPEATED_MEASURES_PARADIGM_NOT_DERIVED: &str = "adaptive_inference.repeated_measures_paradigm_not_derived";
    /// Detail `adaptive_inference.repeated_measures_stopping_not_derived`.
    pub const REPEATED_MEASURES_STOPPING_NOT_DERIVED: &str = "adaptive_inference.repeated_measures_stopping_not_derived";
    /// Detail `adaptive_inference.report_replay_mismatch`.
    pub const REPORT_REPLAY_MISMATCH: &str = "adaptive_inference.report_replay_mismatch";
    /// Detail `adaptive_inference.retained_record_exceeded`.
    pub const RETAINED_RECORD_EXCEEDED: &str = "adaptive_inference.retained_record_exceeded";
    /// Detail `adaptive_inference.rule_mismatch`.
    pub const RULE_MISMATCH: &str = "adaptive_inference.rule_mismatch";
    /// Detail `adaptive_inference.selection_not_ignorable`.
    pub const SELECTION_NOT_IGNORABLE: &str = "adaptive_inference.selection_not_ignorable";
    /// Detail `adaptive_inference.shape_mismatch`.
    pub const SHAPE_MISMATCH: &str = "adaptive_inference.shape_mismatch";
    /// Detail `adaptive_inference.simulation_memory_exceeded`.
    pub const SIMULATION_MEMORY_EXCEEDED: &str = "adaptive_inference.simulation_memory_exceeded";
    /// Detail `adaptive_inference.simulation_replications_exceeded`.
    pub const SIMULATION_REPLICATIONS_EXCEEDED: &str = "adaptive_inference.simulation_replications_exceeded";
    /// Detail `adaptive_inference.simulation_work_exceeded`.
    pub const SIMULATION_WORK_EXCEEDED: &str = "adaptive_inference.simulation_work_exceeded";
    /// Detail `adaptive_inference.stopping_not_replayed`.
    pub const STOPPING_NOT_REPLAYED: &str = "adaptive_inference.stopping_not_replayed";
    /// Detail `adaptive_inference.stopping_rule_missing`.
    pub const STOPPING_RULE_MISSING: &str = "adaptive_inference.stopping_rule_missing";
    /// Detail `adaptive_inference.structure_incomplete`.
    pub const STRUCTURE_INCOMPLETE: &str = "adaptive_inference.structure_incomplete";
    /// Detail `adaptive_inference.unit_out_of_range`.
    pub const UNIT_OUT_OF_RANGE: &str = "adaptive_inference.unit_out_of_range";
    /// Detail `adaptive_inference.variance_invalid`.
    pub const VARIANCE_INVALID: &str = "adaptive_inference.variance_invalid";
    /// Detail `adaptive_inference.within_unit_correlation_missing`.
    pub const WITHIN_UNIT_CORRELATION_MISSING: &str = "adaptive_inference.within_unit_correlation_missing";
    /// Detail `adaptive_inference.within_unit_variance_missing`.
    pub const WITHIN_UNIT_VARIANCE_MISSING: &str = "adaptive_inference.within_unit_variance_missing";
    /// Detail `adaptive_inference.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "adaptive_inference.wrong_contract";
}

/// Details in the `adaptive_transport` family.
pub mod adaptive_transport {
    /// Detail `adaptive_transport.arms`.
    pub const ARMS: &str = "adaptive_transport.arms";
    /// Detail `adaptive_transport.assignment_below_floor`.
    pub const ASSIGNMENT_BELOW_FLOOR: &str = "adaptive_transport.assignment_below_floor";
    /// Detail `adaptive_transport.assignment_monitoring_contract_missing`.
    pub const ASSIGNMENT_MONITORING_CONTRACT_MISSING: &str = "adaptive_transport.assignment_monitoring_contract_missing";
    /// Detail `adaptive_transport.design_not_adaptive`.
    pub const DESIGN_NOT_ADAPTIVE: &str = "adaptive_transport.design_not_adaptive";
    /// Detail `adaptive_transport.diagram_binding_mismatch`.
    pub const DIAGRAM_BINDING_MISMATCH: &str = "adaptive_transport.diagram_binding_mismatch";
    /// Detail `adaptive_transport.mechanism_declared_twice`.
    pub const MECHANISM_DECLARED_TWICE: &str = "adaptive_transport.mechanism_declared_twice";
    /// Detail `adaptive_transport.mechanism_status_missing`.
    pub const MECHANISM_STATUS_MISSING: &str = "adaptive_transport.mechanism_status_missing";
    /// Detail `adaptive_transport.no_analysed_units`.
    pub const NO_ANALYSED_UNITS: &str = "adaptive_transport.no_analysed_units";
    /// Detail `adaptive_transport.outcome_mechanism_not_invariant`.
    pub const OUTCOME_MECHANISM_NOT_INVARIANT: &str = "adaptive_transport.outcome_mechanism_not_invariant";
    /// Detail `adaptive_transport.outcome_missing`.
    pub const OUTCOME_MISSING: &str = "adaptive_transport.outcome_missing";
    /// Detail `adaptive_transport.population_label`.
    pub const POPULATION_LABEL: &str = "adaptive_transport.population_label";
    /// Detail `adaptive_transport.selection_diagram_missing`.
    pub const SELECTION_DIAGRAM_MISSING: &str = "adaptive_transport.selection_diagram_missing";
    /// Detail `adaptive_transport.strata`.
    pub const STRATA: &str = "adaptive_transport.strata";
    /// Detail `adaptive_transport.stratum_arm_unobserved`.
    pub const STRATUM_ARM_UNOBSERVED: &str = "adaptive_transport.stratum_arm_unobserved";
    /// Detail `adaptive_transport.stratum_law_declared_invariant`.
    pub const STRATUM_LAW_DECLARED_INVARIANT: &str = "adaptive_transport.stratum_law_declared_invariant";
    /// Detail `adaptive_transport.target_is_source`.
    pub const TARGET_IS_SOURCE: &str = "adaptive_transport.target_is_source";
    /// Detail `adaptive_transport.target_stratum_law`.
    pub const TARGET_STRATUM_LAW: &str = "adaptive_transport.target_stratum_law";
    /// Detail `adaptive_transport.target_stratum_law_missing`.
    pub const TARGET_STRATUM_LAW_MISSING: &str = "adaptive_transport.target_stratum_law_missing";
    /// Detail `adaptive_transport.unit_strata`.
    pub const UNIT_STRATA: &str = "adaptive_transport.unit_strata";
    /// Detail `adaptive_transport.value_mismatch`.
    pub const VALUE_MISMATCH: &str = "adaptive_transport.value_mismatch";
}

/// Details in the `admg_conditional_posterior` family.
pub mod admg_conditional_posterior {
    /// Detail `admg_conditional_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "admg_conditional_posterior.band_not_constructed";
    /// Detail `admg_conditional_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "admg_conditional_posterior.empty_law";
    /// Detail `admg_conditional_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "admg_conditional_posterior.external_power_prior";
    /// Detail `admg_conditional_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "admg_conditional_posterior.mapped_prior";
    /// Detail `admg_conditional_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "admg_conditional_posterior.non_positive_scale";
}

/// Details in the `admg_transport` family.
pub mod admg_transport {
    /// Detail `admg_transport.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "admg_transport.bounds_exceeded";
    /// Detail `admg_transport.budget`.
    pub const BUDGET: &str = "admg_transport.budget";
    /// Detail `admg_transport.consumer_limits`.
    pub const CONSUMER_LIMITS: &str = "admg_transport.consumer_limits";
    /// Detail `admg_transport.counted_laws_unsupported`.
    pub const COUNTED_LAWS_UNSUPPORTED: &str = "admg_transport.counted_laws_unsupported";
    /// Detail `admg_transport.data_identity_mismatch`.
    pub const DATA_IDENTITY_MISMATCH: &str = "admg_transport.data_identity_mismatch";
    /// Detail `admg_transport.evidence_mismatch`.
    pub const EVIDENCE_MISMATCH: &str = "admg_transport.evidence_mismatch";
    /// Detail `admg_transport.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "admg_transport.invalid_artifact";
    /// Detail `admg_transport.invalid_catalog`.
    pub const INVALID_CATALOG: &str = "admg_transport.invalid_catalog";
    /// Detail `admg_transport.invalid_derivation`.
    pub const INVALID_DERIVATION: &str = "admg_transport.invalid_derivation";
    /// Detail `admg_transport.invalid_query`.
    pub const INVALID_QUERY: &str = "admg_transport.invalid_query";
    /// Detail `admg_transport.invalid_request`.
    pub const INVALID_REQUEST: &str = "admg_transport.invalid_request";
    /// Detail `admg_transport.missing_evidence`.
    pub const MISSING_EVIDENCE: &str = "admg_transport.missing_evidence";
    /// Detail `admg_transport.not_certified`.
    pub const NOT_CERTIFIED: &str = "admg_transport.not_certified";
    /// Detail `admg_transport.premises_mismatch`.
    pub const PREMISES_MISMATCH: &str = "admg_transport.premises_mismatch";
    /// Detail `admg_transport.proven_non_transportable`.
    pub const PROVEN_NON_TRANSPORTABLE: &str = "admg_transport.proven_non_transportable";
    /// Detail `admg_transport.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "admg_transport.replay_mismatch";
    /// Detail `admg_transport.support_failure`.
    pub const SUPPORT_FAILURE: &str = "admg_transport.support_failure";
}

/// Details in the `aligned_joint_draws` family.
pub mod aligned_joint_draws {
    /// Detail `aligned_joint_draws.identity_expected`.
    pub const IDENTITY_EXPECTED: &str = "aligned_joint_draws.identity_expected";
    /// Detail `aligned_joint_draws.legacy_binding`.
    pub const LEGACY_BINDING: &str = "aligned_joint_draws.legacy_binding";
    /// Detail `aligned_joint_draws.legacy_source`.
    pub const LEGACY_SOURCE: &str = "aligned_joint_draws.legacy_source";
    /// Detail `aligned_joint_draws.marginals_not_joint`.
    pub const MARGINALS_NOT_JOINT: &str = "aligned_joint_draws.marginals_not_joint";
    /// Detail `aligned_joint_draws.outcome_coordinate`.
    pub const OUTCOME_COORDINATE: &str = "aligned_joint_draws.outcome_coordinate";
}

/// Details in the `batch_retarget` family.
pub mod batch_retarget {
    /// Detail `batch_retarget.cancelled`.
    pub const CANCELLED: &str = "batch_retarget.cancelled";
    /// Detail `batch_retarget.contrast_member_failed`.
    pub const CONTRAST_MEMBER_FAILED: &str = "batch_retarget.contrast_member_failed";
    /// Detail `batch_retarget.covariance_unavailable`.
    pub const COVARIANCE_UNAVAILABLE: &str = "batch_retarget.covariance_unavailable";
    /// Detail `batch_retarget.duplicate_name`.
    pub const DUPLICATE_NAME: &str = "batch_retarget.duplicate_name";
    /// Detail `batch_retarget.empty_family`.
    pub const EMPTY_FAMILY: &str = "batch_retarget.empty_family";
    /// Detail `batch_retarget.functional_not_supported`.
    pub const FUNCTIONAL_NOT_SUPPORTED: &str = "batch_retarget.functional_not_supported";
    /// Detail `batch_retarget.incompatible_target`.
    pub const INCOMPATIBLE_TARGET: &str = "batch_retarget.incompatible_target";
    /// Detail `batch_retarget.invalid_contrast`.
    pub const INVALID_CONTRAST: &str = "batch_retarget.invalid_contrast";
    /// Detail `batch_retarget.max_t_draws_out_of_range`.
    pub const MAX_T_DRAWS_OUT_OF_RANGE: &str = "batch_retarget.max_t_draws_out_of_range";
    /// Detail `batch_retarget.max_t_invalid_level`.
    pub const MAX_T_INVALID_LEVEL: &str = "batch_retarget.max_t_invalid_level";
    /// Detail `batch_retarget.mixed_snapshot`.
    pub const MIXED_SNAPSHOT: &str = "batch_retarget.mixed_snapshot";
    /// Detail `batch_retarget.partial_family`.
    pub const PARTIAL_FAMILY: &str = "batch_retarget.partial_family";
    /// Detail `batch_retarget.point_only_member`.
    pub const POINT_ONLY_MEMBER: &str = "batch_retarget.point_only_member";
    /// Detail `batch_retarget.scores_mismatch`.
    pub const SCORES_MISMATCH: &str = "batch_retarget.scores_mismatch";
    /// Detail `batch_retarget.scores_unavailable`.
    pub const SCORES_UNAVAILABLE: &str = "batch_retarget.scores_unavailable";
    /// Detail `batch_retarget.scores_unavailable_after_estimate`.
    pub const SCORES_UNAVAILABLE_AFTER_ESTIMATE: &str = "batch_retarget.scores_unavailable_after_estimate";
    /// Detail `batch_retarget.unknown_claim`.
    pub const UNKNOWN_CLAIM: &str = "batch_retarget.unknown_claim";
    /// Detail `batch_retarget.weighted_overlap_failed`.
    pub const WEIGHTED_OVERLAP_FAILED: &str = "batch_retarget.weighted_overlap_failed";
}

/// Details in the `batch_retarget_posterior` family.
pub mod batch_retarget_posterior {
    /// Detail `batch_retarget_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "batch_retarget_posterior.band_not_constructed";
    /// Detail `batch_retarget_posterior.empty_scores`.
    pub const EMPTY_SCORES: &str = "batch_retarget_posterior.empty_scores";
    /// Detail `batch_retarget_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "batch_retarget_posterior.external_power_prior";
    /// Detail `batch_retarget_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "batch_retarget_posterior.mapped_prior";
    /// Detail `batch_retarget_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "batch_retarget_posterior.non_positive_scale";
}

/// Details in the `bayesian_transport` family.
pub mod bayesian_transport {
    /// Detail `bayesian_transport.admission_refused`.
    pub const ADMISSION_REFUSED: &str = "bayesian_transport.admission_refused";
    /// Detail `bayesian_transport.cancelled`.
    pub const CANCELLED: &str = "bayesian_transport.cancelled";
    /// Detail `bayesian_transport.feature_mismatch`.
    pub const FEATURE_MISMATCH: &str = "bayesian_transport.feature_mismatch";
    /// Detail `bayesian_transport.identification_not_certified`.
    pub const IDENTIFICATION_NOT_CERTIFIED: &str = "bayesian_transport.identification_not_certified";
    /// Detail `bayesian_transport.identification_required`.
    pub const IDENTIFICATION_REQUIRED: &str = "bayesian_transport.identification_required";
    /// Detail `bayesian_transport.invalid_input`.
    pub const INVALID_INPUT: &str = "bayesian_transport.invalid_input";
    /// Detail `bayesian_transport.invalid_prior`.
    pub const INVALID_PRIOR: &str = "bayesian_transport.invalid_prior";
    /// Detail `bayesian_transport.missing_law`.
    pub const MISSING_LAW: &str = "bayesian_transport.missing_law";
    /// Detail `bayesian_transport.missing_source`.
    pub const MISSING_SOURCE: &str = "bayesian_transport.missing_source";
    /// Detail `bayesian_transport.prior_data_overlap`.
    pub const PRIOR_DATA_OVERLAP: &str = "bayesian_transport.prior_data_overlap";
    /// Detail `bayesian_transport.prior_dimension`.
    pub const PRIOR_DIMENSION: &str = "bayesian_transport.prior_dimension";
    /// Detail `bayesian_transport.singular_fit`.
    pub const SINGULAR_FIT: &str = "bayesian_transport.singular_fit";
    /// Detail `bayesian_transport.source_dependence`.
    pub const SOURCE_DEPENDENCE: &str = "bayesian_transport.source_dependence";
    /// Detail `bayesian_transport.support_resource_limit`.
    pub const SUPPORT_RESOURCE_LIMIT: &str = "bayesian_transport.support_resource_limit";
    /// Detail `bayesian_transport.too_many_draws`.
    pub const TOO_MANY_DRAWS: &str = "bayesian_transport.too_many_draws";
    /// Detail `bayesian_transport.too_many_parameters`.
    pub const TOO_MANY_PARAMETERS: &str = "bayesian_transport.too_many_parameters";
    /// Detail `bayesian_transport.unsupported_graph`.
    pub const UNSUPPORTED_GRAPH: &str = "bayesian_transport.unsupported_graph";
    /// Detail `bayesian_transport.weak_overlap`.
    pub const WEAK_OVERLAP: &str = "bayesian_transport.weak_overlap";
}

/// Details in the `binary_recovery_posterior` family.
pub mod binary_recovery_posterior {
    /// Detail `binary_recovery_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "binary_recovery_posterior.band_not_constructed";
    /// Detail `binary_recovery_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "binary_recovery_posterior.empty_law";
    /// Detail `binary_recovery_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "binary_recovery_posterior.external_power_prior";
    /// Detail `binary_recovery_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "binary_recovery_posterior.mapped_prior";
    /// Detail `binary_recovery_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "binary_recovery_posterior.non_positive_scale";
}

/// Details in the `callback_protocol` family.
pub mod callback_protocol {
    /// Detail `callback_protocol.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "callback_protocol.artifact_invalid";
    /// Detail `callback_protocol.blank_field`.
    pub const BLANK_FIELD: &str = "callback_protocol.blank_field";
    /// Detail `callback_protocol.cancelled`.
    pub const CANCELLED: &str = "callback_protocol.cancelled";
    /// Detail `callback_protocol.diagnostic_only`.
    pub const DIAGNOSTIC_ONLY: &str = "callback_protocol.diagnostic_only";
    /// Detail `callback_protocol.duplicate_name`.
    pub const DUPLICATE_NAME: &str = "callback_protocol.duplicate_name";
    /// Detail `callback_protocol.hidden_state_unobservable`.
    pub const HIDDEN_STATE_UNOBSERVABLE: &str = "callback_protocol.hidden_state_unobservable";
    /// Detail `callback_protocol.identification_not_conferred`.
    pub const IDENTIFICATION_NOT_CONFERRED: &str = "callback_protocol.identification_not_conferred";
    /// Detail `callback_protocol.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "callback_protocol.identity_mismatch";
    /// Detail `callback_protocol.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "callback_protocol.interval_not_supported";
    /// Detail `callback_protocol.invalid_quantity`.
    pub const INVALID_QUANTITY: &str = "callback_protocol.invalid_quantity";
    /// Detail `callback_protocol.invalid_support`.
    pub const INVALID_SUPPORT: &str = "callback_protocol.invalid_support";
    /// Detail `callback_protocol.invocation_limit`.
    pub const INVOCATION_LIMIT: &str = "callback_protocol.invocation_limit";
    /// Detail `callback_protocol.invocation_outside_execution`.
    pub const INVOCATION_OUTSIDE_EXECUTION: &str = "callback_protocol.invocation_outside_execution";
    /// Detail `callback_protocol.joint_law_not_conferred`.
    pub const JOINT_LAW_NOT_CONFERRED: &str = "callback_protocol.joint_law_not_conferred";
    /// Detail `callback_protocol.meaning_not_for_kind`.
    pub const MEANING_NOT_FOR_KIND: &str = "callback_protocol.meaning_not_for_kind";
    /// Detail `callback_protocol.native_trust_claimed`.
    pub const NATIVE_TRUST_CLAIMED: &str = "callback_protocol.native_trust_claimed";
    /// Detail `callback_protocol.native_trust_not_conferred`.
    pub const NATIVE_TRUST_NOT_CONFERRED: &str = "callback_protocol.native_trust_not_conferred";
    /// Detail `callback_protocol.nested_input_overlap_unknown`.
    pub const NESTED_INPUT_OVERLAP_UNKNOWN: &str = "callback_protocol.nested_input_overlap_unknown";
    /// Detail `callback_protocol.nested_input_unregistered`.
    pub const NESTED_INPUT_UNREGISTERED: &str = "callback_protocol.nested_input_unregistered";
    /// Detail `callback_protocol.nested_inputs_mismatch`.
    pub const NESTED_INPUTS_MISMATCH: &str = "callback_protocol.nested_inputs_mismatch";
    /// Detail `callback_protocol.nested_inputs_share_evidence`.
    pub const NESTED_INPUTS_SHARE_EVIDENCE: &str = "callback_protocol.nested_inputs_share_evidence";
    /// Detail `callback_protocol.no_idempotency_contract`.
    pub const NO_IDEMPOTENCY_CONTRACT: &str = "callback_protocol.no_idempotency_contract";
    /// Detail `callback_protocol.no_operations`.
    pub const NO_OPERATIONS: &str = "callback_protocol.no_operations";
    /// Detail `callback_protocol.opaque_dependencies`.
    pub const OPAQUE_DEPENDENCIES: &str = "callback_protocol.opaque_dependencies";
    /// Detail `callback_protocol.operation_not_declared`.
    pub const OPERATION_NOT_DECLARED: &str = "callback_protocol.operation_not_declared";
    /// Detail `callback_protocol.operation_not_for_kind`.
    pub const OPERATION_NOT_FOR_KIND: &str = "callback_protocol.operation_not_for_kind";
    /// Detail `callback_protocol.output_not_a_law`.
    pub const OUTPUT_NOT_A_LAW: &str = "callback_protocol.output_not_a_law";
    /// Detail `callback_protocol.output_not_finite`.
    pub const OUTPUT_NOT_FINITE: &str = "callback_protocol.output_not_finite";
    /// Detail `callback_protocol.output_outside_support`.
    pub const OUTPUT_OUTSIDE_SUPPORT: &str = "callback_protocol.output_outside_support";
    /// Detail `callback_protocol.output_shape`.
    pub const OUTPUT_SHAPE: &str = "callback_protocol.output_shape";
    /// Detail `callback_protocol.output_too_large`.
    pub const OUTPUT_TOO_LARGE: &str = "callback_protocol.output_too_large";
    /// Detail `callback_protocol.posterior_guarantee_not_supported`.
    pub const POSTERIOR_GUARANTEE_NOT_SUPPORTED: &str = "callback_protocol.posterior_guarantee_not_supported";
    /// Detail `callback_protocol.provider_failed`.
    pub const PROVIDER_FAILED: &str = "callback_protocol.provider_failed";
    /// Detail `callback_protocol.provider_identity_mismatch`.
    pub const PROVIDER_IDENTITY_MISMATCH: &str = "callback_protocol.provider_identity_mismatch";
    /// Detail `callback_protocol.provider_unavailable`.
    pub const PROVIDER_UNAVAILABLE: &str = "callback_protocol.provider_unavailable";
    /// Detail `callback_protocol.query_invalid`.
    pub const QUERY_INVALID: &str = "callback_protocol.query_invalid";
    /// Detail `callback_protocol.replay_contract_incomplete`.
    pub const REPLAY_CONTRACT_INCOMPLETE: &str = "callback_protocol.replay_contract_incomplete";
    /// Detail `callback_protocol.request_mismatch`.
    pub const REQUEST_MISMATCH: &str = "callback_protocol.request_mismatch";
    /// Detail `callback_protocol.retry_outside_policy`.
    pub const RETRY_OUTSIDE_POLICY: &str = "callback_protocol.retry_outside_policy";
    /// Detail `callback_protocol.side_effects_not_allowed`.
    pub const SIDE_EFFECTS_NOT_ALLOWED: &str = "callback_protocol.side_effects_not_allowed";
    /// Detail `callback_protocol.too_many_items`.
    pub const TOO_MANY_ITEMS: &str = "callback_protocol.too_many_items";
    /// Detail `callback_protocol.trust_incomplete`.
    pub const TRUST_INCOMPLETE: &str = "callback_protocol.trust_incomplete";
    /// Detail `callback_protocol.update_declaration_mismatch`.
    pub const UPDATE_DECLARATION_MISMATCH: &str = "callback_protocol.update_declaration_mismatch";
    /// Detail `callback_protocol.wall_limit_exceeded`.
    pub const WALL_LIMIT_EXCEEDED: &str = "callback_protocol.wall_limit_exceeded";
}

/// Details in the `categorical_posterior` family.
pub mod categorical_posterior {
    /// Detail `categorical_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "categorical_posterior.band_not_constructed";
    /// Detail `categorical_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "categorical_posterior.external_power_prior";
    /// Detail `categorical_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "categorical_posterior.mapped_prior";
    /// Detail `categorical_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "categorical_posterior.non_positive_scale";
}

/// Details in the `categorical_treatment` family.
pub mod categorical_treatment {
    /// Detail `categorical_treatment.absent_level`.
    pub const ABSENT_LEVEL: &str = "categorical_treatment.absent_level";
    /// Detail `categorical_treatment.cancelled`.
    pub const CANCELLED: &str = "categorical_treatment.cancelled";
    /// Detail `categorical_treatment.data_not_bound`.
    pub const DATA_NOT_BOUND: &str = "categorical_treatment.data_not_bound";
    /// Detail `categorical_treatment.export_budget_exceeded`.
    pub const EXPORT_BUDGET_EXCEEDED: &str = "categorical_treatment.export_budget_exceeded";
    /// Detail `categorical_treatment.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "categorical_treatment.invalid_identity";
    /// Detail `categorical_treatment.invalid_level`.
    pub const INVALID_LEVEL: &str = "categorical_treatment.invalid_level";
    /// Detail `categorical_treatment.invalid_min_level_rows`.
    pub const INVALID_MIN_LEVEL_ROWS: &str = "categorical_treatment.invalid_min_level_rows";
    /// Detail `categorical_treatment.invalid_pair`.
    pub const INVALID_PAIR: &str = "categorical_treatment.invalid_pair";
    /// Detail `categorical_treatment.invalid_request`.
    pub const INVALID_REQUEST: &str = "categorical_treatment.invalid_request";
    /// Detail `categorical_treatment.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "categorical_treatment.limits_exceeded";
    /// Detail `categorical_treatment.memory_budget_exceeded`.
    pub const MEMORY_BUDGET_EXCEEDED: &str = "categorical_treatment.memory_budget_exceeded";
    /// Detail `categorical_treatment.monotonicity_requires_ordered`.
    pub const MONOTONICITY_REQUIRES_ORDERED: &str = "categorical_treatment.monotonicity_requires_ordered";
    /// Detail `categorical_treatment.posterior_declared_rows`.
    pub const POSTERIOR_DECLARED_ROWS: &str = "categorical_treatment.posterior_declared_rows";
    /// Detail `categorical_treatment.row_count_mismatch`.
    pub const ROW_COUNT_MISMATCH: &str = "categorical_treatment.row_count_mismatch";
    /// Detail `categorical_treatment.sparse_level`.
    pub const SPARSE_LEVEL: &str = "categorical_treatment.sparse_level";
    /// Detail `categorical_treatment.summary_inconsistent`.
    pub const SUMMARY_INCONSISTENT: &str = "categorical_treatment.summary_inconsistent";
    /// Detail `categorical_treatment.too_few_levels`.
    pub const TOO_FEW_LEVELS: &str = "categorical_treatment.too_few_levels";
    /// Detail `categorical_treatment.undeclared_level`.
    pub const UNDECLARED_LEVEL: &str = "categorical_treatment.undeclared_level";
    /// Detail `categorical_treatment.unknown_level`.
    pub const UNKNOWN_LEVEL: &str = "categorical_treatment.unknown_level";
    /// Detail `categorical_treatment.unknown_reference`.
    pub const UNKNOWN_REFERENCE: &str = "categorical_treatment.unknown_reference";
    /// Detail `categorical_treatment.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "categorical_treatment.wrong_contract";
}

/// Details in the `chain_recovery_posterior` family.
pub mod chain_recovery_posterior {
    /// Detail `chain_recovery_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "chain_recovery_posterior.band_not_constructed";
    /// Detail `chain_recovery_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "chain_recovery_posterior.empty_law";
    /// Detail `chain_recovery_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "chain_recovery_posterior.external_power_prior";
    /// Detail `chain_recovery_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "chain_recovery_posterior.mapped_prior";
    /// Detail `chain_recovery_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "chain_recovery_posterior.non_positive_scale";
}

/// Details in the `cluster_dml` family.
pub mod cluster_dml {
    /// Detail `cluster_dml.cancelled`.
    pub const CANCELLED: &str = "cluster_dml.cancelled";
    /// Detail `cluster_dml.cluster_ids_missing`.
    pub const CLUSTER_IDS_MISSING: &str = "cluster_dml.cluster_ids_missing";
    /// Detail `cluster_dml.dyadic_giant_component`.
    pub const DYADIC_GIANT_COMPONENT: &str = "cluster_dml.dyadic_giant_component";
    /// Detail `cluster_dml.dyadic_shared_namespace`.
    pub const DYADIC_SHARED_NAMESPACE: &str = "cluster_dml.dyadic_shared_namespace";
    /// Detail `cluster_dml.flexible_learner_not_supported`.
    pub const FLEXIBLE_LEARNER_NOT_SUPPORTED: &str = "cluster_dml.flexible_learner_not_supported";
    /// Detail `cluster_dml.invalid_min_clusters`.
    pub const INVALID_MIN_CLUSTERS: &str = "cluster_dml.invalid_min_clusters";
    /// Detail `cluster_dml.invalid_min_components`.
    pub const INVALID_MIN_COMPONENTS: &str = "cluster_dml.invalid_min_components";
    /// Detail `cluster_dml.no_interval_method`.
    pub const NO_INTERVAL_METHOD: &str = "cluster_dml.no_interval_method";
    /// Detail `cluster_dml.penalized_propensity`.
    pub const PENALIZED_PROPENSITY: &str = "cluster_dml.penalized_propensity";
    /// Detail `cluster_dml.scope`.
    pub const SCOPE: &str = "cluster_dml.scope";
    /// Detail `cluster_dml.second_ids_missing`.
    pub const SECOND_IDS_MISSING: &str = "cluster_dml.second_ids_missing";
    /// Detail `cluster_dml.second_ids_unexpected`.
    pub const SECOND_IDS_UNEXPECTED: &str = "cluster_dml.second_ids_unexpected";
    /// Detail `cluster_dml.too_few_clusters`.
    pub const TOO_FEW_CLUSTERS: &str = "cluster_dml.too_few_clusters";
    /// Detail `cluster_dml.too_few_components`.
    pub const TOO_FEW_COMPONENTS: &str = "cluster_dml.too_few_components";
}

/// Details in the `clustered_posterior` family.
pub mod clustered_posterior {
    /// Detail `clustered_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "clustered_posterior.band_not_constructed";
    /// Detail `clustered_posterior.empty_scores`.
    pub const EMPTY_SCORES: &str = "clustered_posterior.empty_scores";
    /// Detail `clustered_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "clustered_posterior.external_power_prior";
    /// Detail `clustered_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "clustered_posterior.mapped_prior";
    /// Detail `clustered_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "clustered_posterior.non_positive_scale";
}

/// Details in the `cohort_decision` family.
pub mod cohort_decision {
    /// Detail `cohort_decision.cancelled`.
    pub const CANCELLED: &str = "cohort_decision.cancelled";
    /// Detail `cohort_decision.cohort_changed`.
    pub const COHORT_CHANGED: &str = "cohort_decision.cohort_changed";
    /// Detail `cohort_decision.cohort_empty`.
    pub const COHORT_EMPTY: &str = "cohort_decision.cohort_empty";
    /// Detail `cohort_decision.cohort_invalid`.
    pub const COHORT_INVALID: &str = "cohort_decision.cohort_invalid";
    /// Detail `cohort_decision.cohort_too_large`.
    pub const COHORT_TOO_LARGE: &str = "cohort_decision.cohort_too_large";
    /// Detail `cohort_decision.container_checksum_mismatch`.
    pub const CONTAINER_CHECKSUM_MISMATCH: &str = "cohort_decision.container_checksum_mismatch";
    /// Detail `cohort_decision.container_invalid`.
    pub const CONTAINER_INVALID: &str = "cohort_decision.container_invalid";
    /// Detail `cohort_decision.contrast_mismatch`.
    pub const CONTRAST_MISMATCH: &str = "cohort_decision.contrast_mismatch";
    /// Detail `cohort_decision.contrast_not_declared`.
    pub const CONTRAST_NOT_DECLARED: &str = "cohort_decision.contrast_not_declared";
    /// Detail `cohort_decision.extrapolative_not_accepted`.
    pub const EXTRAPOLATIVE_NOT_ACCEPTED: &str = "cohort_decision.extrapolative_not_accepted";
    /// Detail `cohort_decision.feature_schema_mismatch`.
    pub const FEATURE_SCHEMA_MISMATCH: &str = "cohort_decision.feature_schema_mismatch";
    /// Detail `cohort_decision.feature_type_mismatch`.
    pub const FEATURE_TYPE_MISMATCH: &str = "cohort_decision.feature_type_mismatch";
    /// Detail `cohort_decision.fitted_model_absent`.
    pub const FITTED_MODEL_ABSENT: &str = "cohort_decision.fitted_model_absent";
    /// Detail `cohort_decision.fitted_model_invalid`.
    pub const FITTED_MODEL_INVALID: &str = "cohort_decision.fitted_model_invalid";
    /// Detail `cohort_decision.identity_unexpected`.
    pub const IDENTITY_UNEXPECTED: &str = "cohort_decision.identity_unexpected";
    /// Detail `cohort_decision.interval_level_invalid`.
    pub const INTERVAL_LEVEL_INVALID: &str = "cohort_decision.interval_level_invalid";
    /// Detail `cohort_decision.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "cohort_decision.interval_not_supported";
    /// Detail `cohort_decision.interval_requires_native_fit`.
    pub const INTERVAL_REQUIRES_NATIVE_FIT: &str = "cohort_decision.interval_requires_native_fit";
    /// Detail `cohort_decision.invalid_request`.
    pub const INVALID_REQUEST: &str = "cohort_decision.invalid_request";
    /// Detail `cohort_decision.joint_law_required`.
    pub const JOINT_LAW_REQUIRED: &str = "cohort_decision.joint_law_required";
    /// Detail `cohort_decision.law_not_supplied`.
    pub const LAW_NOT_SUPPLIED: &str = "cohort_decision.law_not_supplied";
    /// Detail `cohort_decision.layout_invalid`.
    pub const LAYOUT_INVALID: &str = "cohort_decision.layout_invalid";
    /// Detail `cohort_decision.link_not_identity`.
    pub const LINK_NOT_IDENTITY: &str = "cohort_decision.link_not_identity";
    /// Detail `cohort_decision.missing_artifact_id`.
    pub const MISSING_ARTIFACT_ID: &str = "cohort_decision.missing_artifact_id";
    /// Detail `cohort_decision.model_identity_mismatch`.
    pub const MODEL_IDENTITY_MISMATCH: &str = "cohort_decision.model_identity_mismatch";
    /// Detail `cohort_decision.native_fit_refused`.
    pub const NATIVE_FIT_REFUSED: &str = "cohort_decision.native_fit_refused";
    /// Detail `cohort_decision.negative_variance`.
    pub const NEGATIVE_VARIANCE: &str = "cohort_decision.negative_variance";
    /// Detail `cohort_decision.non_finite_result`.
    pub const NON_FINITE_RESULT: &str = "cohort_decision.non_finite_result";
    /// Detail `cohort_decision.outcome_units_required`.
    pub const OUTCOME_UNITS_REQUIRED: &str = "cohort_decision.outcome_units_required";
    /// Detail `cohort_decision.outside_support`.
    pub const OUTSIDE_SUPPORT: &str = "cohort_decision.outside_support";
    /// Detail `cohort_decision.oversized`.
    pub const OVERSIZED: &str = "cohort_decision.oversized";
    /// Detail `cohort_decision.parent_attested_not_native`.
    pub const PARENT_ATTESTED_NOT_NATIVE: &str = "cohort_decision.parent_attested_not_native";
    /// Detail `cohort_decision.parent_not_verified`.
    pub const PARENT_NOT_VERIFIED: &str = "cohort_decision.parent_not_verified";
    /// Detail `cohort_decision.parent_query_unsupported`.
    pub const PARENT_QUERY_UNSUPPORTED: &str = "cohort_decision.parent_query_unsupported";
    /// Detail `cohort_decision.population_mismatch`.
    pub const POPULATION_MISMATCH: &str = "cohort_decision.population_mismatch";
    /// Detail `cohort_decision.prediction_refused`.
    pub const PREDICTION_REFUSED: &str = "cohort_decision.prediction_refused";
    /// Detail `cohort_decision.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "cohort_decision.replay_mismatch";
    /// Detail `cohort_decision.shift_declaration_mismatch`.
    pub const SHIFT_DECLARATION_MISMATCH: &str = "cohort_decision.shift_declaration_mismatch";
    /// Detail `cohort_decision.shift_declaration_required`.
    pub const SHIFT_DECLARATION_REQUIRED: &str = "cohort_decision.shift_declaration_required";
    /// Detail `cohort_decision.shift_declaration_unneeded`.
    pub const SHIFT_DECLARATION_UNNEEDED: &str = "cohort_decision.shift_declaration_unneeded";
    /// Detail `cohort_decision.shift_support_failure`.
    pub const SHIFT_SUPPORT_FAILURE: &str = "cohort_decision.shift_support_failure";
    /// Detail `cohort_decision.source_invalid`.
    pub const SOURCE_INVALID: &str = "cohort_decision.source_invalid";
    /// Detail `cohort_decision.support_contract_absent`.
    pub const SUPPORT_CONTRACT_ABSENT: &str = "cohort_decision.support_contract_absent";
    /// Detail `cohort_decision.units_mismatch`.
    pub const UNITS_MISMATCH: &str = "cohort_decision.units_mismatch";
    /// Detail `cohort_decision.unsupported_version`.
    pub const UNSUPPORTED_VERSION: &str = "cohort_decision.unsupported_version";
    /// Detail `cohort_decision.weights_invalid`.
    pub const WEIGHTS_INVALID: &str = "cohort_decision.weights_invalid";
    /// Detail `cohort_decision.weights_not_normalized`.
    pub const WEIGHTS_NOT_NORMALIZED: &str = "cohort_decision.weights_not_normalized";
    /// Detail `cohort_decision.wrong_parent_identity`.
    pub const WRONG_PARENT_IDENTITY: &str = "cohort_decision.wrong_parent_identity";
}

/// Details in the `compact_export` family.
pub mod compact_export {
    /// Detail `compact_export.container_checksum_mismatch`.
    pub const CONTAINER_CHECKSUM_MISMATCH: &str = "compact_export.container_checksum_mismatch";
    /// Detail `compact_export.container_invalid`.
    pub const CONTAINER_INVALID: &str = "compact_export.container_invalid";
    /// Detail `compact_export.covariance_asymmetric`.
    pub const COVARIANCE_ASYMMETRIC: &str = "compact_export.covariance_asymmetric";
    /// Detail `compact_export.covariance_not_psd`.
    pub const COVARIANCE_NOT_PSD: &str = "compact_export.covariance_not_psd";
    /// Detail `compact_export.data_digest_mismatch`.
    pub const DATA_DIGEST_MISMATCH: &str = "compact_export.data_digest_mismatch";
    /// Detail `compact_export.dimension_mismatch`.
    pub const DIMENSION_MISMATCH: &str = "compact_export.dimension_mismatch";
    /// Detail `compact_export.duplicate_term`.
    pub const DUPLICATE_TERM: &str = "compact_export.duplicate_term";
    /// Detail `compact_export.empty_mask_region`.
    pub const EMPTY_MASK_REGION: &str = "compact_export.empty_mask_region";
    /// Detail `compact_export.encode`.
    pub const ENCODE: &str = "compact_export.encode";
    /// Detail `compact_export.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "compact_export.identity_mismatch";
    /// Detail `compact_export.identity_unexpected`.
    pub const IDENTITY_UNEXPECTED: &str = "compact_export.identity_unexpected";
    /// Detail `compact_export.inputs_not_canonical`.
    pub const INPUTS_NOT_CANONICAL: &str = "compact_export.inputs_not_canonical";
    /// Detail `compact_export.invalid_degree`.
    pub const INVALID_DEGREE: &str = "compact_export.invalid_degree";
    /// Detail `compact_export.invalid_mask_region`.
    pub const INVALID_MASK_REGION: &str = "compact_export.invalid_mask_region";
    /// Detail `compact_export.invalid_quantity`.
    pub const INVALID_QUANTITY: &str = "compact_export.invalid_quantity";
    /// Detail `compact_export.invalid_query`.
    pub const INVALID_QUERY: &str = "compact_export.invalid_query";
    /// Detail `compact_export.invalid_request`.
    pub const INVALID_REQUEST: &str = "compact_export.invalid_request";
    /// Detail `compact_export.invalid_support`.
    pub const INVALID_SUPPORT: &str = "compact_export.invalid_support";
    /// Detail `compact_export.layout_invalid`.
    pub const LAYOUT_INVALID: &str = "compact_export.layout_invalid";
    /// Detail `compact_export.mask_not_canonical`.
    pub const MASK_NOT_CANONICAL: &str = "compact_export.mask_not_canonical";
    /// Detail `compact_export.mask_support_mismatch`.
    pub const MASK_SUPPORT_MISMATCH: &str = "compact_export.mask_support_mismatch";
    /// Detail `compact_export.masked_region`.
    pub const MASKED_REGION: &str = "compact_export.masked_region";
    /// Detail `compact_export.missing_artifact_id`.
    pub const MISSING_ARTIFACT_ID: &str = "compact_export.missing_artifact_id";
    /// Detail `compact_export.missing_quantity`.
    pub const MISSING_QUANTITY: &str = "compact_export.missing_quantity";
    /// Detail `compact_export.negative_variance`.
    pub const NEGATIVE_VARIANCE: &str = "compact_export.negative_variance";
    /// Detail `compact_export.no_terms`.
    pub const NO_TERMS: &str = "compact_export.no_terms";
    /// Detail `compact_export.non_finite_input`.
    pub const NON_FINITE_INPUT: &str = "compact_export.non_finite_input";
    /// Detail `compact_export.non_finite_result`.
    pub const NON_FINITE_RESULT: &str = "compact_export.non_finite_result";
    /// Detail `compact_export.non_finite_value`.
    pub const NON_FINITE_VALUE: &str = "compact_export.non_finite_value";
    /// Detail `compact_export.out_of_support`.
    pub const OUT_OF_SUPPORT: &str = "compact_export.out_of_support";
    /// Detail `compact_export.oversized`.
    pub const OVERSIZED: &str = "compact_export.oversized";
    /// Detail `compact_export.premises_digest_mismatch`.
    pub const PREMISES_DIGEST_MISMATCH: &str = "compact_export.premises_digest_mismatch";
    /// Detail `compact_export.query_type_mismatch`.
    pub const QUERY_TYPE_MISMATCH: &str = "compact_export.query_type_mismatch";
    /// Detail `compact_export.scope_mismatch`.
    pub const SCOPE_MISMATCH: &str = "compact_export.scope_mismatch";
    /// Detail `compact_export.term_support_mismatch`.
    pub const TERM_SUPPORT_MISMATCH: &str = "compact_export.term_support_mismatch";
    /// Detail `compact_export.terms_not_canonical`.
    pub const TERMS_NOT_CANONICAL: &str = "compact_export.terms_not_canonical";
    /// Detail `compact_export.too_many_inputs`.
    pub const TOO_MANY_INPUTS: &str = "compact_export.too_many_inputs";
    /// Detail `compact_export.too_many_levels`.
    pub const TOO_MANY_LEVELS: &str = "compact_export.too_many_levels";
    /// Detail `compact_export.too_many_mask_regions`.
    pub const TOO_MANY_MASK_REGIONS: &str = "compact_export.too_many_mask_regions";
    /// Detail `compact_export.too_many_terms`.
    pub const TOO_MANY_TERMS: &str = "compact_export.too_many_terms";
    /// Detail `compact_export.unknown_level`.
    pub const UNKNOWN_LEVEL: &str = "compact_export.unknown_level";
    /// Detail `compact_export.unknown_quantity`.
    pub const UNKNOWN_QUANTITY: &str = "compact_export.unknown_quantity";
    /// Detail `compact_export.unsupported_version`.
    pub const UNSUPPORTED_VERSION: &str = "compact_export.unsupported_version";
}

/// Details in the `compiled_categorical_posterior` family.
pub mod compiled_categorical_posterior {
    /// Detail `compiled_categorical_posterior.cancelled`.
    pub const CANCELLED: &str = "compiled_categorical_posterior.cancelled";
    /// Detail `compiled_categorical_posterior.candidate`.
    pub const CANDIDATE: &str = "compiled_categorical_posterior.candidate";
    /// Detail `compiled_categorical_posterior.execution_refused`.
    pub const EXECUTION_REFUSED: &str = "compiled_categorical_posterior.execution_refused";
    /// Detail `compiled_categorical_posterior.resource_limit`.
    pub const RESOURCE_LIMIT: &str = "compiled_categorical_posterior.resource_limit";
    /// Detail `compiled_categorical_posterior.source_invalid`.
    pub const SOURCE_INVALID: &str = "compiled_categorical_posterior.source_invalid";
}

/// Details in the `composite_recalc` family.
pub mod composite_recalc {
    /// Detail `composite_recalc.invalid_request`.
    pub const INVALID_REQUEST: &str = "composite_recalc.invalid_request";
    /// Detail `composite_recalc.native_state_mismatch`.
    pub const NATIVE_STATE_MISMATCH: &str = "composite_recalc.native_state_mismatch";
    /// Detail `composite_recalc.receipt_invalid`.
    pub const RECEIPT_INVALID: &str = "composite_recalc.receipt_invalid";
    /// Detail `composite_recalc.request_unsupported`.
    pub const REQUEST_UNSUPPORTED: &str = "composite_recalc.request_unsupported";
}

/// Details in the `composition_boundary` family.
pub mod composition_boundary {
    /// Detail `composition_boundary.artifact_rebuild_failed`.
    pub const ARTIFACT_REBUILD_FAILED: &str = "composition_boundary.artifact_rebuild_failed";
    /// Detail `composition_boundary.atom_probabilities_invalid`.
    pub const ATOM_PROBABILITIES_INVALID: &str = "composition_boundary.atom_probabilities_invalid";
    /// Detail `composition_boundary.conflicting_atoms_not_averaged`.
    pub const CONFLICTING_ATOMS_NOT_AVERAGED: &str = "composition_boundary.conflicting_atoms_not_averaged";
    /// Detail `composition_boundary.coordinate_missing`.
    pub const COORDINATE_MISSING: &str = "composition_boundary.coordinate_missing";
    /// Detail `composition_boundary.coordinate_unsupported`.
    pub const COORDINATE_UNSUPPORTED: &str = "composition_boundary.coordinate_unsupported";
    /// Detail `composition_boundary.dependence_route_not_supported`.
    pub const DEPENDENCE_ROUTE_NOT_SUPPORTED: &str = "composition_boundary.dependence_route_not_supported";
    /// Detail `composition_boundary.invalid_input`.
    pub const INVALID_INPUT: &str = "composition_boundary.invalid_input";
    /// Detail `composition_boundary.mean_is_not_a_distribution`.
    pub const MEAN_IS_NOT_A_DISTRIBUTION: &str = "composition_boundary.mean_is_not_a_distribution";
    /// Detail `composition_boundary.metadata_only_native_claim`.
    pub const METADATA_ONLY_NATIVE_CLAIM: &str = "composition_boundary.metadata_only_native_claim";
    /// Detail `composition_boundary.native_required`.
    pub const NATIVE_REQUIRED: &str = "composition_boundary.native_required";
    /// Detail `composition_boundary.non_affine_needs_joint_law`.
    pub const NON_AFFINE_NEEDS_JOINT_LAW: &str = "composition_boundary.non_affine_needs_joint_law";
    /// Detail `composition_boundary.operation_not_declared`.
    pub const OPERATION_NOT_DECLARED: &str = "composition_boundary.operation_not_declared";
    /// Detail `composition_boundary.outcome_law_needs_joint_law`.
    pub const OUTCOME_LAW_NEEDS_JOINT_LAW: &str = "composition_boundary.outcome_law_needs_joint_law";
    /// Detail `composition_boundary.paired_draws_across_sources`.
    pub const PAIRED_DRAWS_ACROSS_SOURCES: &str = "composition_boundary.paired_draws_across_sources";
    /// Detail `composition_boundary.shared_evidence_not_independent`.
    pub const SHARED_EVIDENCE_NOT_INDEPENDENT: &str = "composition_boundary.shared_evidence_not_independent";
    /// Detail `composition_boundary.unknown_dependence_is_not_independence`.
    pub const UNKNOWN_DEPENDENCE_IS_NOT_INDEPENDENCE: &str = "composition_boundary.unknown_dependence_is_not_independence";
    /// Detail `composition_boundary.unknown_input`.
    pub const UNKNOWN_INPUT: &str = "composition_boundary.unknown_input";
    /// Detail `composition_boundary.unsupported_action_not_comparable`.
    pub const UNSUPPORTED_ACTION_NOT_COMPARABLE: &str = "composition_boundary.unsupported_action_not_comparable";
    /// Detail `composition_boundary.verification_receipt_missing`.
    pub const VERIFICATION_RECEIPT_MISSING: &str = "composition_boundary.verification_receipt_missing";
}

/// Details in the `composition_bundle` family.
pub mod composition_bundle {
    /// Detail `composition_bundle.callback_unavailable`.
    pub const CALLBACK_UNAVAILABLE: &str = "composition_bundle.callback_unavailable";
    /// Detail `composition_bundle.edge_digest_mismatch`.
    pub const EDGE_DIGEST_MISMATCH: &str = "composition_bundle.edge_digest_mismatch";
    /// Detail `composition_bundle.expected_identity_mismatch`.
    pub const EXPECTED_IDENTITY_MISMATCH: &str = "composition_bundle.expected_identity_mismatch";
    /// Detail `composition_bundle.graph_or_snapshot_mismatch`.
    pub const GRAPH_OR_SNAPSHOT_MISMATCH: &str = "composition_bundle.graph_or_snapshot_mismatch";
    /// Detail `composition_bundle.incompatible_version`.
    pub const INCOMPATIBLE_VERSION: &str = "composition_bundle.incompatible_version";
    /// Detail `composition_bundle.invalid_container`.
    pub const INVALID_CONTAINER: &str = "composition_bundle.invalid_container";
    /// Detail `composition_bundle.node_not_found`.
    pub const NODE_NOT_FOUND: &str = "composition_bundle.node_not_found";
    /// Detail `composition_bundle.oversized`.
    pub const OVERSIZED: &str = "composition_bundle.oversized";
    /// Detail `composition_bundle.provider_request_changed`.
    pub const PROVIDER_REQUEST_CHANGED: &str = "composition_bundle.provider_request_changed";
    /// Detail `composition_bundle.swapped_evidence`.
    pub const SWAPPED_EVIDENCE: &str = "composition_bundle.swapped_evidence";
    /// Detail `composition_bundle.tampered_quantity`.
    pub const TAMPERED_QUANTITY: &str = "composition_bundle.tampered_quantity";
    /// Detail `composition_bundle.unknown_node_kind`.
    pub const UNKNOWN_NODE_KIND: &str = "composition_bundle.unknown_node_kind";
    /// Detail `composition_bundle.unsupported_law`.
    pub const UNSUPPORTED_LAW: &str = "composition_bundle.unsupported_law";
}

/// Details in the `composition_provenance` family.
pub mod composition_provenance {
    /// Detail `composition_provenance.digest_mismatch`.
    pub const DIGEST_MISMATCH: &str = "composition_provenance.digest_mismatch";
    /// Detail `composition_provenance.duplicate_link`.
    pub const DUPLICATE_LINK: &str = "composition_provenance.duplicate_link";
    /// Detail `composition_provenance.invalid_link`.
    pub const INVALID_LINK: &str = "composition_provenance.invalid_link";
    /// Detail `composition_provenance.missing_stage`.
    pub const MISSING_STAGE: &str = "composition_provenance.missing_stage";
    /// Detail `composition_provenance.unknown_link`.
    pub const UNKNOWN_LINK: &str = "composition_provenance.unknown_link";
    /// Detail `composition_provenance.unresolved_parent`.
    pub const UNRESOLVED_PARENT: &str = "composition_provenance.unresolved_parent";
}

/// Details in the `composition_verification` family.
pub mod composition_verification {
    /// Detail `composition_verification.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "composition_verification.artifact_invalid";
    /// Detail `composition_verification.cancelled`.
    pub const CANCELLED: &str = "composition_verification.cancelled";
    /// Detail `composition_verification.cycle`.
    pub const CYCLE: &str = "composition_verification.cycle";
    /// Detail `composition_verification.duplicate_node`.
    pub const DUPLICATE_NODE: &str = "composition_verification.duplicate_node";
    /// Detail `composition_verification.empty_unsupported`.
    pub const EMPTY_UNSUPPORTED: &str = "composition_verification.empty_unsupported";
    /// Detail `composition_verification.graph_invalid`.
    pub const GRAPH_INVALID: &str = "composition_verification.graph_invalid";
    /// Detail `composition_verification.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "composition_verification.identity_mismatch";
    /// Detail `composition_verification.inherited_without_edge`.
    pub const INHERITED_WITHOUT_EDGE: &str = "composition_verification.inherited_without_edge";
    /// Detail `composition_verification.input_retired`.
    pub const INPUT_RETIRED: &str = "composition_verification.input_retired";
    /// Detail `composition_verification.invalid_text`.
    pub const INVALID_TEXT: &str = "composition_verification.invalid_text";
    /// Detail `composition_verification.ledger_mismatch`.
    pub const LEDGER_MISMATCH: &str = "composition_verification.ledger_mismatch";
    /// Detail `composition_verification.limit_exceeded`.
    pub const LIMIT_EXCEEDED: &str = "composition_verification.limit_exceeded";
    /// Detail `composition_verification.provider_trust_withdrawn`.
    pub const PROVIDER_TRUST_WITHDRAWN: &str = "composition_verification.provider_trust_withdrawn";
    /// Detail `composition_verification.recompute_required`.
    pub const RECOMPUTE_REQUIRED: &str = "composition_verification.recompute_required";
    /// Detail `composition_verification.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "composition_verification.replay_mismatch";
    /// Detail `composition_verification.self_dependency`.
    pub const SELF_DEPENDENCY: &str = "composition_verification.self_dependency";
    /// Detail `composition_verification.stage_not_in_plan`.
    pub const STAGE_NOT_IN_PLAN: &str = "composition_verification.stage_not_in_plan";
    /// Detail `composition_verification.stage_unexecutable`.
    pub const STAGE_UNEXECUTABLE: &str = "composition_verification.stage_unexecutable";
    /// Detail `composition_verification.support_withdrawn`.
    pub const SUPPORT_WITHDRAWN: &str = "composition_verification.support_withdrawn";
    /// Detail `composition_verification.unknown_claim`.
    pub const UNKNOWN_CLAIM: &str = "composition_verification.unknown_claim";
    /// Detail `composition_verification.unknown_node`.
    pub const UNKNOWN_NODE: &str = "composition_verification.unknown_node";
    /// Detail `composition_verification.unknown_result`.
    pub const UNKNOWN_RESULT: &str = "composition_verification.unknown_result";
}

/// Details in the `conditional_study_ranking` family.
pub mod conditional_study_ranking {
    /// Detail `conditional_study_ranking.action_not_unique`.
    pub const ACTION_NOT_UNIQUE: &str = "conditional_study_ranking.action_not_unique";
    /// Detail `conditional_study_ranking.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "conditional_study_ranking.artifact_invalid";
    /// Detail `conditional_study_ranking.invalid_policy`.
    pub const INVALID_POLICY: &str = "conditional_study_ranking.invalid_policy";
    /// Detail `conditional_study_ranking.original_ranking_refused`.
    pub const ORIGINAL_RANKING_REFUSED: &str = "conditional_study_ranking.original_ranking_refused";
    /// Detail `conditional_study_ranking.resource_limit`.
    pub const RESOURCE_LIMIT: &str = "conditional_study_ranking.resource_limit";
    /// Detail `conditional_study_ranking.source_mismatch`.
    pub const SOURCE_MISMATCH: &str = "conditional_study_ranking.source_mismatch";
    /// Detail `conditional_study_ranking.source_unavailable`.
    pub const SOURCE_UNAVAILABLE: &str = "conditional_study_ranking.source_unavailable";
}

/// Details in the `constancy_posterior` family.
pub mod constancy_posterior {
    /// Detail `constancy_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "constancy_posterior.band_not_constructed";
    /// Detail `constancy_posterior.empty_contrast`.
    pub const EMPTY_CONTRAST: &str = "constancy_posterior.empty_contrast";
    /// Detail `constancy_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "constancy_posterior.external_power_prior";
    /// Detail `constancy_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "constancy_posterior.mapped_prior";
    /// Detail `constancy_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "constancy_posterior.non_positive_scale";
}

/// Details in the `coordinate_support` family.
pub mod coordinate_support {
    /// Detail `coordinate_support.functional_not_described`.
    pub const FUNCTIONAL_NOT_DESCRIBED: &str = "coordinate_support.functional_not_described";
    /// Detail `coordinate_support.invalid_grid`.
    pub const INVALID_GRID: &str = "coordinate_support.invalid_grid";
    /// Detail `coordinate_support.missing_evidence_relabeled`.
    pub const MISSING_EVIDENCE_RELABELED: &str = "coordinate_support.missing_evidence_relabeled";
    /// Detail `coordinate_support.point_count_mismatch`.
    pub const POINT_COUNT_MISMATCH: &str = "coordinate_support.point_count_mismatch";
    /// Detail `coordinate_support.regime_not_describable`.
    pub const REGIME_NOT_DESCRIBABLE: &str = "coordinate_support.regime_not_describable";
    /// Detail `coordinate_support.summary_not_worst_label`.
    pub const SUMMARY_NOT_WORST_LABEL: &str = "coordinate_support.summary_not_worst_label";
    /// Detail `coordinate_support.temporal_path_not_described`.
    pub const TEMPORAL_PATH_NOT_DESCRIBED: &str = "coordinate_support.temporal_path_not_described";
    /// Detail `coordinate_support.units_required`.
    pub const UNITS_REQUIRED: &str = "coordinate_support.units_required";
    /// Detail `coordinate_support.unknown_variable`.
    pub const UNKNOWN_VARIABLE: &str = "coordinate_support.unknown_variable";
}

/// Details in the `counterfactual_id` family.
pub mod counterfactual_id {
    /// Detail `counterfactual_id.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "counterfactual_id.bounds_exceeded";
    /// Detail `counterfactual_id.budget`.
    pub const BUDGET: &str = "counterfactual_id.budget";
    /// Detail `counterfactual_id.conflicting_subscripts`.
    pub const CONFLICTING_SUBSCRIPTS: &str = "counterfactual_id.conflicting_subscripts";
    /// Detail `counterfactual_id.counterfactual_hedge`.
    pub const COUNTERFACTUAL_HEDGE: &str = "counterfactual_id.counterfactual_hedge";
    /// Detail `counterfactual_id.graph_outside_contract`.
    pub const GRAPH_OUTSIDE_CONTRACT: &str = "counterfactual_id.graph_outside_contract";
    /// Detail `counterfactual_id.interval_requested`.
    pub const INTERVAL_REQUESTED: &str = "counterfactual_id.interval_requested";
    /// Detail `counterfactual_id.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "counterfactual_id.invalid_artifact";
    /// Detail `counterfactual_id.invalid_query`.
    pub const INVALID_QUERY: &str = "counterfactual_id.invalid_query";
    /// Detail `counterfactual_id.path_specific_not_supported`.
    pub const PATH_SPECIFIC_NOT_SUPPORTED: &str = "counterfactual_id.path_specific_not_supported";
    /// Detail `counterfactual_id.positivity_violation`.
    pub const POSITIVITY_VIOLATION: &str = "counterfactual_id.positivity_violation";
    /// Detail `counterfactual_id.query_outside_contract`.
    pub const QUERY_OUTSIDE_CONTRACT: &str = "counterfactual_id.query_outside_contract";
}

/// Details in the `counterfactual_transport` family.
pub mod counterfactual_transport {
    /// Detail `counterfactual_transport.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "counterfactual_transport.bounds_exceeded";
    /// Detail `counterfactual_transport.budget`.
    pub const BUDGET: &str = "counterfactual_transport.budget";
    /// Detail `counterfactual_transport.checked_obstruction`.
    pub const CHECKED_OBSTRUCTION: &str = "counterfactual_transport.checked_obstruction";
    /// Detail `counterfactual_transport.coupling_unsupported`.
    pub const COUPLING_UNSUPPORTED: &str = "counterfactual_transport.coupling_unsupported";
    /// Detail `counterfactual_transport.evidence_incompatible`.
    pub const EVIDENCE_INCOMPATIBLE: &str = "counterfactual_transport.evidence_incompatible";
    /// Detail `counterfactual_transport.factor_not_derivable`.
    pub const FACTOR_NOT_DERIVABLE: &str = "counterfactual_transport.factor_not_derivable";
    /// Detail `counterfactual_transport.graph_outside_contract`.
    pub const GRAPH_OUTSIDE_CONTRACT: &str = "counterfactual_transport.graph_outside_contract";
    /// Detail `counterfactual_transport.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "counterfactual_transport.invalid_artifact";
    /// Detail `counterfactual_transport.invalid_catalog`.
    pub const INVALID_CATALOG: &str = "counterfactual_transport.invalid_catalog";
    /// Detail `counterfactual_transport.invalid_derivation`.
    pub const INVALID_DERIVATION: &str = "counterfactual_transport.invalid_derivation";
    /// Detail `counterfactual_transport.invalid_evidence`.
    pub const INVALID_EVIDENCE: &str = "counterfactual_transport.invalid_evidence";
    /// Detail `counterfactual_transport.invalid_query`.
    pub const INVALID_QUERY: &str = "counterfactual_transport.invalid_query";
    /// Detail `counterfactual_transport.invalid_witness`.
    pub const INVALID_WITNESS: &str = "counterfactual_transport.invalid_witness";
    /// Detail `counterfactual_transport.missing_law`.
    pub const MISSING_LAW: &str = "counterfactual_transport.missing_law";
    /// Detail `counterfactual_transport.missing_regime`.
    pub const MISSING_REGIME: &str = "counterfactual_transport.missing_regime";
    /// Detail `counterfactual_transport.no_sampled_inference`.
    pub const NO_SAMPLED_INFERENCE: &str = "counterfactual_transport.no_sampled_inference";
    /// Detail `counterfactual_transport.selected_sample_law`.
    pub const SELECTED_SAMPLE_LAW: &str = "counterfactual_transport.selected_sample_law";
    /// Detail `counterfactual_transport.support_failure`.
    pub const SUPPORT_FAILURE: &str = "counterfactual_transport.support_failure";
    /// Detail `counterfactual_transport.zero_conditioning_mass`.
    pub const ZERO_CONDITIONING_MASS: &str = "counterfactual_transport.zero_conditioning_mass";
}

/// Details in the `counterfactual_transport_sampled` family.
pub mod counterfactual_transport_sampled {
    /// Detail `counterfactual_transport_sampled.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "counterfactual_transport_sampled.bounds_exceeded";
    /// Detail `counterfactual_transport_sampled.cancelled`.
    pub const CANCELLED: &str = "counterfactual_transport_sampled.cancelled";
    /// Detail `counterfactual_transport_sampled.envelope_budget`.
    pub const ENVELOPE_BUDGET: &str = "counterfactual_transport_sampled.envelope_budget";
    /// Detail `counterfactual_transport_sampled.envelope_undefined`.
    pub const ENVELOPE_UNDEFINED: &str = "counterfactual_transport_sampled.envelope_undefined";
    /// Detail `counterfactual_transport_sampled.interval_not_computable`.
    pub const INTERVAL_NOT_COMPUTABLE: &str = "counterfactual_transport_sampled.interval_not_computable";
    /// Detail `counterfactual_transport_sampled.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "counterfactual_transport_sampled.invalid_artifact";
    /// Detail `counterfactual_transport_sampled.invalid_counts`.
    pub const INVALID_COUNTS: &str = "counterfactual_transport_sampled.invalid_counts";
    /// Detail `counterfactual_transport_sampled.invalid_overlap`.
    pub const INVALID_OVERLAP: &str = "counterfactual_transport_sampled.invalid_overlap";
    /// Detail `counterfactual_transport_sampled.missing_table`.
    pub const MISSING_TABLE: &str = "counterfactual_transport_sampled.missing_table";
    /// Detail `counterfactual_transport_sampled.overlap_declared_independent`.
    pub const OVERLAP_DECLARED_INDEPENDENT: &str = "counterfactual_transport_sampled.overlap_declared_independent";
    /// Detail `counterfactual_transport_sampled.overlap_unknown`.
    pub const OVERLAP_UNKNOWN: &str = "counterfactual_transport_sampled.overlap_unknown";
    /// Detail `counterfactual_transport_sampled.sample_size_below_limit`.
    pub const SAMPLE_SIZE_BELOW_LIMIT: &str = "counterfactual_transport_sampled.sample_size_below_limit";
    /// Detail `counterfactual_transport_sampled.shared_units_unlinked`.
    pub const SHARED_UNITS_UNLINKED: &str = "counterfactual_transport_sampled.shared_units_unlinked";
    /// Detail `counterfactual_transport_sampled.sparsity_above_limit`.
    pub const SPARSITY_ABOVE_LIMIT: &str = "counterfactual_transport_sampled.sparsity_above_limit";
    /// Detail `counterfactual_transport_sampled.table_not_cited`.
    pub const TABLE_NOT_CITED: &str = "counterfactual_transport_sampled.table_not_cited";
    /// Detail `counterfactual_transport_sampled.unseen_cell`.
    pub const UNSEEN_CELL: &str = "counterfactual_transport_sampled.unseen_cell";
}

/// Details in the `cpdag_scenarios` family.
pub mod cpdag_scenarios {
    /// Detail `cpdag_scenarios.cancelled_not_exportable`.
    pub const CANCELLED_NOT_EXPORTABLE: &str = "cpdag_scenarios.cancelled_not_exportable";
    /// Detail `cpdag_scenarios.consumer_limit_exceeded`.
    pub const CONSUMER_LIMIT_EXCEEDED: &str = "cpdag_scenarios.consumer_limit_exceeded";
    /// Detail `cpdag_scenarios.data_identity_mismatch`.
    pub const DATA_IDENTITY_MISMATCH: &str = "cpdag_scenarios.data_identity_mismatch";
    /// Detail `cpdag_scenarios.evidence_identity_mismatch`.
    pub const EVIDENCE_IDENTITY_MISMATCH: &str = "cpdag_scenarios.evidence_identity_mismatch";
    /// Detail `cpdag_scenarios.exact_laws_only`.
    pub const EXACT_LAWS_ONLY: &str = "cpdag_scenarios.exact_laws_only";
    /// Detail `cpdag_scenarios.no_evidence_binding`.
    pub const NO_EVIDENCE_BINDING: &str = "cpdag_scenarios.no_evidence_binding";
    /// Detail `cpdag_scenarios.not_a_cpdag`.
    pub const NOT_A_CPDAG: &str = "cpdag_scenarios.not_a_cpdag";
    /// Detail `cpdag_scenarios.premises_mismatch`.
    pub const PREMISES_MISMATCH: &str = "cpdag_scenarios.premises_mismatch";
    /// Detail `cpdag_scenarios.provider_mismatch`.
    pub const PROVIDER_MISMATCH: &str = "cpdag_scenarios.provider_mismatch";
    /// Detail `cpdag_scenarios.report_replay_mismatch`.
    pub const REPORT_REPLAY_MISMATCH: &str = "cpdag_scenarios.report_replay_mismatch";
    /// Detail `cpdag_scenarios.selection_or_latent`.
    pub const SELECTION_OR_LATENT: &str = "cpdag_scenarios.selection_or_latent";
    /// Detail `cpdag_scenarios.unsupported_semantics`.
    pub const UNSUPPORTED_SEMANTICS: &str = "cpdag_scenarios.unsupported_semantics";
}

/// Details in the `cross_world` family.
pub mod cross_world {
    /// Detail `cross_world.graph_outside_contract`.
    pub const GRAPH_OUTSIDE_CONTRACT: &str = "cross_world.graph_outside_contract";
    /// Detail `cross_world.interval_requested`.
    pub const INTERVAL_REQUESTED: &str = "cross_world.interval_requested";
    /// Detail `cross_world.invalid_query`.
    pub const INVALID_QUERY: &str = "cross_world.invalid_query";
    /// Detail `cross_world.query_outside_contract`.
    pub const QUERY_OUTSIDE_CONTRACT: &str = "cross_world.query_outside_contract";
    /// Detail `cross_world.recanting_witness`.
    pub const RECANTING_WITNESS: &str = "cross_world.recanting_witness";
}

/// Details in the `cross_world_execution` family.
pub mod cross_world_execution {
    /// Detail `cross_world_execution.artifact_changed`.
    pub const ARTIFACT_CHANGED: &str = "cross_world_execution.artifact_changed";
    /// Detail `cross_world_execution.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "cross_world_execution.bounds_exceeded";
    /// Detail `cross_world_execution.cancelled`.
    pub const CANCELLED: &str = "cross_world_execution.cancelled";
    /// Detail `cross_world_execution.cell_not_in_matrix`.
    pub const CELL_NOT_IN_MATRIX: &str = "cross_world_execution.cell_not_in_matrix";
    /// Detail `cross_world_execution.elimination_bound_exceeded`.
    pub const ELIMINATION_BOUND_EXCEEDED: &str = "cross_world_execution.elimination_bound_exceeded";
    /// Detail `cross_world_execution.envelope_member_structure_differs`.
    pub const ENVELOPE_MEMBER_STRUCTURE_DIFFERS: &str = "cross_world_execution.envelope_member_structure_differs";
    /// Detail `cross_world_execution.envelope_needs_parameter_set`.
    pub const ENVELOPE_NEEDS_PARAMETER_SET: &str = "cross_world_execution.envelope_needs_parameter_set";
    /// Detail `cross_world_execution.envelope_parameter_set_empty`.
    pub const ENVELOPE_PARAMETER_SET_EMPTY: &str = "cross_world_execution.envelope_parameter_set_empty";
    /// Detail `cross_world_execution.envelope_reading_required`.
    pub const ENVELOPE_READING_REQUIRED: &str = "cross_world_execution.envelope_reading_required";
    /// Detail `cross_world_execution.evidence_outside_model_support`.
    pub const EVIDENCE_OUTSIDE_MODEL_SUPPORT: &str = "cross_world_execution.evidence_outside_model_support";
    /// Detail `cross_world_execution.evidence_probability_underflow`.
    pub const EVIDENCE_PROBABILITY_UNDERFLOW: &str = "cross_world_execution.evidence_probability_underflow";
    /// Detail `cross_world_execution.finite_fitting_not_executed`.
    pub const FINITE_FITTING_NOT_EXECUTED: &str = "cross_world_execution.finite_fitting_not_executed";
    /// Detail `cross_world_execution.incompatible_history`.
    pub const INCOMPATIBLE_HISTORY: &str = "cross_world_execution.incompatible_history";
    /// Detail `cross_world_execution.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "cross_world_execution.invalid_artifact";
    /// Detail `cross_world_execution.invalid_evidence`.
    pub const INVALID_EVIDENCE: &str = "cross_world_execution.invalid_evidence";
    /// Detail `cross_world_execution.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "cross_world_execution.invalid_identity";
    /// Detail `cross_world_execution.invalid_quantity`.
    pub const INVALID_QUANTITY: &str = "cross_world_execution.invalid_quantity";
    /// Detail `cross_world_execution.invalid_request`.
    pub const INVALID_REQUEST: &str = "cross_world_execution.invalid_request";
    /// Detail `cross_world_execution.invalid_scm`.
    pub const INVALID_SCM: &str = "cross_world_execution.invalid_scm";
    /// Detail `cross_world_execution.invalid_world`.
    pub const INVALID_WORLD: &str = "cross_world_execution.invalid_world";
    /// Detail `cross_world_execution.joint_state_bound_exceeded`.
    pub const JOINT_STATE_BOUND_EXCEEDED: &str = "cross_world_execution.joint_state_bound_exceeded";
    /// Detail `cross_world_execution.latent_confounding_not_identified`.
    pub const LATENT_CONFOUNDING_NOT_IDENTIFIED: &str = "cross_world_execution.latent_confounding_not_identified";
    /// Detail `cross_world_execution.mechanisms_not_declared`.
    pub const MECHANISMS_NOT_DECLARED: &str = "cross_world_execution.mechanisms_not_declared";
    /// Detail `cross_world_execution.no_interval_method`.
    pub const NO_INTERVAL_METHOD: &str = "cross_world_execution.no_interval_method";
    /// Detail `cross_world_execution.no_posterior_method`.
    pub const NO_POSTERIOR_METHOD: &str = "cross_world_execution.no_posterior_method";
    /// Detail `cross_world_execution.no_transported_row_derivation`.
    pub const NO_TRANSPORTED_ROW_DERIVATION: &str = "cross_world_execution.no_transported_row_derivation";
    /// Detail `cross_world_execution.posterior_weight_underflow`.
    pub const POSTERIOR_WEIGHT_UNDERFLOW: &str = "cross_world_execution.posterior_weight_underflow";
    /// Detail `cross_world_execution.sampled_abduction_not_supported`.
    pub const SAMPLED_ABDUCTION_NOT_SUPPORTED: &str = "cross_world_execution.sampled_abduction_not_supported";
    /// Detail `cross_world_execution.soft_intervention`.
    pub const SOFT_INTERVENTION: &str = "cross_world_execution.soft_intervention";
    /// Detail `cross_world_execution.spread_needs_enumeration`.
    pub const SPREAD_NEEDS_ENUMERATION: &str = "cross_world_execution.spread_needs_enumeration";
}

/// Details in the `cross_world_uncertainty` family.
pub mod cross_world_uncertainty {
    /// Detail `cross_world_uncertainty.account_mismatch`.
    pub const ACCOUNT_MISMATCH: &str = "cross_world_uncertainty.account_mismatch";
    /// Detail `cross_world_uncertainty.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "cross_world_uncertainty.bounds_exceeded";
    /// Detail `cross_world_uncertainty.incompatible_history`.
    pub const INCOMPATIBLE_HISTORY: &str = "cross_world_uncertainty.incompatible_history";
    /// Detail `cross_world_uncertainty.no_transported_path_derivation`.
    pub const NO_TRANSPORTED_PATH_DERIVATION: &str = "cross_world_uncertainty.no_transported_path_derivation";
    /// Detail `cross_world_uncertainty.result_mismatch`.
    pub const RESULT_MISMATCH: &str = "cross_world_uncertainty.result_mismatch";
    /// Detail `cross_world_uncertainty.sampled_path_not_supported`.
    pub const SAMPLED_PATH_NOT_SUPPORTED: &str = "cross_world_uncertainty.sampled_path_not_supported";
    /// Detail `cross_world_uncertainty.unit_sampling_undeclared`.
    pub const UNIT_SAMPLING_UNDECLARED: &str = "cross_world_uncertainty.unit_sampling_undeclared";
}

/// Details in the `decision_adapters` family.
pub mod decision_adapters {
    /// Detail `decision_adapters.bayes_over_identified_set`.
    pub const BAYES_OVER_IDENTIFIED_SET: &str = "decision_adapters.bayes_over_identified_set";
    /// Detail `decision_adapters.claim_draws_too_large`.
    pub const CLAIM_DRAWS_TOO_LARGE: &str = "decision_adapters.claim_draws_too_large";
    /// Detail `decision_adapters.claims_sources_mismatch`.
    pub const CLAIMS_SOURCES_MISMATCH: &str = "decision_adapters.claims_sources_mismatch";
    /// Detail `decision_adapters.completion_count_not_probability`.
    pub const COMPLETION_COUNT_NOT_PROBABILITY: &str = "decision_adapters.completion_count_not_probability";
    /// Detail `decision_adapters.criterion_not_supported_for_set`.
    pub const CRITERION_NOT_SUPPORTED_FOR_SET: &str = "decision_adapters.criterion_not_supported_for_set";
    /// Detail `decision_adapters.duplicate_action`.
    pub const DUPLICATE_ACTION: &str = "decision_adapters.duplicate_action";
    /// Detail `decision_adapters.evaluated_needs_draws`.
    pub const EVALUATED_NEEDS_DRAWS: &str = "decision_adapters.evaluated_needs_draws";
    /// Detail `decision_adapters.graph_claims_carry_no_probability`.
    pub const GRAPH_CLAIMS_CARRY_NO_PROBABILITY: &str = "decision_adapters.graph_claims_carry_no_probability";
    /// Detail `decision_adapters.invalid_claims`.
    pub const INVALID_CLAIMS: &str = "decision_adapters.invalid_claims";
    /// Detail `decision_adapters.invalid_interval`.
    pub const INVALID_INTERVAL: &str = "decision_adapters.invalid_interval";
    /// Detail `decision_adapters.invalid_probability`.
    pub const INVALID_PROBABILITY: &str = "decision_adapters.invalid_probability";
    /// Detail `decision_adapters.invalid_utilities`.
    pub const INVALID_UTILITIES: &str = "decision_adapters.invalid_utilities";
    /// Detail `decision_adapters.missing_action`.
    pub const MISSING_ACTION: &str = "decision_adapters.missing_action";
    /// Detail `decision_adapters.point_claim_is_single`.
    pub const POINT_CLAIM_IS_SINGLE: &str = "decision_adapters.point_claim_is_single";
    /// Detail `decision_adapters.policy_mismatch`.
    pub const POLICY_MISMATCH: &str = "decision_adapters.policy_mismatch";
    /// Detail `decision_adapters.probabilities_required`.
    pub const PROBABILITIES_REQUIRED: &str = "decision_adapters.probabilities_required";
    /// Detail `decision_adapters.unevaluated_needs_reason`.
    pub const UNEVALUATED_NEEDS_REASON: &str = "decision_adapters.unevaluated_needs_reason";
    /// Detail `decision_adapters.unknown_action`.
    pub const UNKNOWN_ACTION: &str = "decision_adapters.unknown_action";
    /// Detail `decision_adapters.unknown_claim_kind`.
    pub const UNKNOWN_CLAIM_KIND: &str = "decision_adapters.unknown_claim_kind";
    /// Detail `decision_adapters.unknown_claim_status`.
    pub const UNKNOWN_CLAIM_STATUS: &str = "decision_adapters.unknown_claim_status";
    /// Detail `decision_adapters.unknown_support_status`.
    pub const UNKNOWN_SUPPORT_STATUS: &str = "decision_adapters.unknown_support_status";
}

/// Details in the `decision_admissibility` family.
pub mod decision_admissibility {
    /// Detail `decision_admissibility.duplicate_rule`.
    pub const DUPLICATE_RULE: &str = "decision_admissibility.duplicate_rule";
    /// Detail `decision_admissibility.invalid_rule`.
    pub const INVALID_RULE: &str = "decision_admissibility.invalid_rule";
    /// Detail `decision_admissibility.unknown_action`.
    pub const UNKNOWN_ACTION: &str = "decision_admissibility.unknown_action";
    /// Detail `decision_admissibility.unknown_input`.
    pub const UNKNOWN_INPUT: &str = "decision_admissibility.unknown_input";
}

/// Details in the `decision_claims` family.
pub mod decision_claims {
    /// Detail `decision_claims.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "decision_claims.identity_mismatch";
    /// Detail `decision_claims.missing_scenario_utility`.
    pub const MISSING_SCENARIO_UTILITY: &str = "decision_claims.missing_scenario_utility";
    /// Detail `decision_claims.probabilities_not_declared`.
    pub const PROBABILITIES_NOT_DECLARED: &str = "decision_claims.probabilities_not_declared";
    /// Detail `decision_claims.range_unavailable`.
    pub const RANGE_UNAVAILABLE: &str = "decision_claims.range_unavailable";
    /// Detail `decision_claims.unresolved_scenarios`.
    pub const UNRESOLVED_SCENARIOS: &str = "decision_claims.unresolved_scenarios";
    /// Detail `decision_claims.unsupported_claim_shape`.
    pub const UNSUPPORTED_CLAIM_SHAPE: &str = "decision_claims.unsupported_claim_shape";
}

/// Details in the `decision_contract` family.
pub mod decision_contract {
    /// Detail `decision_contract.duplicate_id`.
    pub const DUPLICATE_ID: &str = "decision_contract.duplicate_id";
    /// Detail `decision_contract.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "decision_contract.invalid_declaration";
    /// Detail `decision_contract.invalid_parameter`.
    pub const INVALID_PARAMETER: &str = "decision_contract.invalid_parameter";
    /// Detail `decision_contract.invalid_quantity`.
    pub const INVALID_QUANTITY: &str = "decision_contract.invalid_quantity";
    /// Detail `decision_contract.missing_source`.
    pub const MISSING_SOURCE: &str = "decision_contract.missing_source";
    /// Detail `decision_contract.outside_decision_scope`.
    pub const OUTSIDE_DECISION_SCOPE: &str = "decision_contract.outside_decision_scope";
    /// Detail `decision_contract.unknown_action`.
    pub const UNKNOWN_ACTION: &str = "decision_contract.unknown_action";
    /// Detail `decision_contract.unknown_input`.
    pub const UNKNOWN_INPUT: &str = "decision_contract.unknown_input";
}

/// Details in the `decision_evaluation` family.
pub mod decision_evaluation {
    /// Detail `decision_evaluation.analytic_law_unsupported`.
    pub const ANALYTIC_LAW_UNSUPPORTED: &str = "decision_evaluation.analytic_law_unsupported";
    /// Detail `decision_evaluation.distribution_meaning`.
    pub const DISTRIBUTION_MEANING: &str = "decision_evaluation.distribution_meaning";
    /// Detail `decision_evaluation.joint_law_required`.
    pub const JOINT_LAW_REQUIRED: &str = "decision_evaluation.joint_law_required";
    /// Detail `decision_evaluation.mean_source_insufficient`.
    pub const MEAN_SOURCE_INSUFFICIENT: &str = "decision_evaluation.mean_source_insufficient";
    /// Detail `decision_evaluation.mean_source_not_replayable`.
    pub const MEAN_SOURCE_NOT_REPLAYABLE: &str = "decision_evaluation.mean_source_not_replayable";
    /// Detail `decision_evaluation.non_finite_utility`.
    pub const NON_FINITE_UTILITY: &str = "decision_evaluation.non_finite_utility";
    /// Detail `decision_evaluation.quantity_not_found`.
    pub const QUANTITY_NOT_FOUND: &str = "decision_evaluation.quantity_not_found";
    /// Detail `decision_evaluation.structure_inputs_required`.
    pub const STRUCTURE_INPUTS_REQUIRED: &str = "decision_evaluation.structure_inputs_required";
    /// Detail `decision_evaluation.unsupported_coordinate`.
    pub const UNSUPPORTED_COORDINATE: &str = "decision_evaluation.unsupported_coordinate";
}

/// Details in the `decision_evidence` family.
pub mod decision_evidence {
    /// Detail `decision_evidence.adjustment_not_admissible`.
    pub const ADJUSTMENT_NOT_ADMISSIBLE: &str = "decision_evidence.adjustment_not_admissible";
    /// Detail `decision_evidence.cancelled`.
    pub const CANCELLED: &str = "decision_evidence.cancelled";
    /// Detail `decision_evidence.dependence_unknown`.
    pub const DEPENDENCE_UNKNOWN: &str = "decision_evidence.dependence_unknown";
    /// Detail `decision_evidence.deployment_bridge_missing`.
    pub const DEPLOYMENT_BRIDGE_MISSING: &str = "decision_evidence.deployment_bridge_missing";
    /// Detail `decision_evidence.frame_shape`.
    pub const FRAME_SHAPE: &str = "decision_evidence.frame_shape";
    /// Detail `decision_evidence.incompatible_horizon`.
    pub const INCOMPATIBLE_HORIZON: &str = "decision_evidence.incompatible_horizon";
    /// Detail `decision_evidence.incompatible_intervention`.
    pub const INCOMPATIBLE_INTERVENTION: &str = "decision_evidence.incompatible_intervention";
    /// Detail `decision_evidence.incompatible_population`.
    pub const INCOMPATIBLE_POPULATION: &str = "decision_evidence.incompatible_population";
    /// Detail `decision_evidence.interference_unspecified`.
    pub const INTERFERENCE_UNSPECIFIED: &str = "decision_evidence.interference_unspecified";
    /// Detail `decision_evidence.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "decision_evidence.invalid_declaration";
    /// Detail `decision_evidence.resource_limit`.
    pub const RESOURCE_LIMIT: &str = "decision_evidence.resource_limit";
    /// Detail `decision_evidence.too_many_unknowns`.
    pub const TOO_MANY_UNKNOWNS: &str = "decision_evidence.too_many_unknowns";
    /// Detail `decision_evidence.unsupported_assignment_design`.
    pub const UNSUPPORTED_ASSIGNMENT_DESIGN: &str = "decision_evidence.unsupported_assignment_design";
}

/// Details in the `decision_joint_scenarios` family.
pub mod decision_joint_scenarios {
    /// Detail `decision_joint_scenarios.carrier`.
    pub const CARRIER: &str = "decision_joint_scenarios.carrier";
}

/// Details in the `decision_posterior` family.
pub mod decision_posterior {
    /// Detail `decision_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "decision_posterior.band_not_constructed";
    /// Detail `decision_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "decision_posterior.empty_law";
    /// Detail `decision_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "decision_posterior.external_power_prior";
    /// Detail `decision_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "decision_posterior.mapped_prior";
    /// Detail `decision_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "decision_posterior.non_positive_scale";
}

/// Details in the `decision_reliability` family.
pub mod decision_reliability {
    /// Detail `decision_reliability.artifact`.
    pub const ARTIFACT: &str = "decision_reliability.artifact";
    /// Detail `decision_reliability.artifact_id_missing`.
    pub const ARTIFACT_ID_MISSING: &str = "decision_reliability.artifact_id_missing";
    /// Detail `decision_reliability.artifact_kind_mismatch`.
    pub const ARTIFACT_KIND_MISMATCH: &str = "decision_reliability.artifact_kind_mismatch";
    /// Detail `decision_reliability.cancelled`.
    pub const CANCELLED: &str = "decision_reliability.cancelled";
    /// Detail `decision_reliability.candidate_set_mismatch`.
    pub const CANDIDATE_SET_MISMATCH: &str = "decision_reliability.candidate_set_mismatch";
    /// Detail `decision_reliability.combined_uncertainty`.
    pub const COMBINED_UNCERTAINTY: &str = "decision_reliability.combined_uncertainty";
    /// Detail `decision_reliability.decision_mismatch`.
    pub const DECISION_MISMATCH: &str = "decision_reliability.decision_mismatch";
    /// Detail `decision_reliability.dependence_unknown`.
    pub const DEPENDENCE_UNKNOWN: &str = "decision_reliability.dependence_unknown";
    /// Detail `decision_reliability.draws_digest_mismatch`.
    pub const DRAWS_DIGEST_MISMATCH: &str = "decision_reliability.draws_digest_mismatch";
    /// Detail `decision_reliability.effective_sample_size_too_small`.
    pub const EFFECTIVE_SAMPLE_SIZE_TOO_SMALL: &str = "decision_reliability.effective_sample_size_too_small";
    /// Detail `decision_reliability.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "decision_reliability.identity_mismatch";
    /// Detail `decision_reliability.invalid_input`.
    pub const INVALID_INPUT: &str = "decision_reliability.invalid_input";
    /// Detail `decision_reliability.joint_law_required`.
    pub const JOINT_LAW_REQUIRED: &str = "decision_reliability.joint_law_required";
    /// Detail `decision_reliability.memory_budget_exceeded`.
    pub const MEMORY_BUDGET_EXCEEDED: &str = "decision_reliability.memory_budget_exceeded";
    /// Detail `decision_reliability.numerical`.
    pub const NUMERICAL: &str = "decision_reliability.numerical";
    /// Detail `decision_reliability.producer_mean_disagrees`.
    pub const PRODUCER_MEAN_DISAGREES: &str = "decision_reliability.producer_mean_disagrees";
    /// Detail `decision_reliability.producer_point_only`.
    pub const PRODUCER_POINT_ONLY: &str = "decision_reliability.producer_point_only";
    /// Detail `decision_reliability.report_mismatch`.
    pub const REPORT_MISMATCH: &str = "decision_reliability.report_mismatch";
    /// Detail `decision_reliability.route_not_supported`.
    pub const ROUTE_NOT_SUPPORTED: &str = "decision_reliability.route_not_supported";
    /// Detail `decision_reliability.search_record_inconsistent`.
    pub const SEARCH_RECORD_INCONSISTENT: &str = "decision_reliability.search_record_inconsistent";
    /// Detail `decision_reliability.search_record_required`.
    pub const SEARCH_RECORD_REQUIRED: &str = "decision_reliability.search_record_required";
    /// Detail `decision_reliability.selection_stream_mismatch`.
    pub const SELECTION_STREAM_MISMATCH: &str = "decision_reliability.selection_stream_mismatch";
    /// Detail `decision_reliability.utility_unit_mismatch`.
    pub const UTILITY_UNIT_MISMATCH: &str = "decision_reliability.utility_unit_mismatch";
    /// Detail `decision_reliability.validation_not_independent`.
    pub const VALIDATION_NOT_INDEPENDENT: &str = "decision_reliability.validation_not_independent";
    /// Detail `decision_reliability.validation_stream_mismatch`.
    pub const VALIDATION_STREAM_MISMATCH: &str = "decision_reliability.validation_stream_mismatch";
}

/// Details in the `decision_robustness` family.
pub mod decision_robustness {
    /// Detail `decision_robustness.contract_mismatch`.
    pub const CONTRACT_MISMATCH: &str = "decision_robustness.contract_mismatch";
    /// Detail `decision_robustness.invalid_profile`.
    pub const INVALID_PROFILE: &str = "decision_robustness.invalid_profile";
}

/// Details in the `decision_structural` family.
pub mod decision_structural {
    /// Detail `decision_structural.invalid_atoms`.
    pub const INVALID_ATOMS: &str = "decision_structural.invalid_atoms";
    /// Detail `decision_structural.invalid_probabilities`.
    pub const INVALID_PROBABILITIES: &str = "decision_structural.invalid_probabilities";
    /// Detail `decision_structural.probabilities_required`.
    pub const PROBABILITIES_REQUIRED: &str = "decision_structural.probabilities_required";
}

/// Details in the `descriptive_comparison` family.
pub mod descriptive_comparison {
    /// Detail `descriptive_comparison.admission_refused`.
    pub const ADMISSION_REFUSED: &str = "descriptive_comparison.admission_refused";
    /// Detail `descriptive_comparison.arm_not_populated`.
    pub const ARM_NOT_POPULATED: &str = "descriptive_comparison.arm_not_populated";
    /// Detail `descriptive_comparison.arm_too_small`.
    pub const ARM_TOO_SMALL: &str = "descriptive_comparison.arm_too_small";
    /// Detail `descriptive_comparison.cancelled`.
    pub const CANCELLED: &str = "descriptive_comparison.cancelled";
    /// Detail `descriptive_comparison.column_attribution_not_identified`.
    pub const COLUMN_ATTRIBUTION_NOT_IDENTIFIED: &str = "descriptive_comparison.column_attribution_not_identified";
    /// Detail `descriptive_comparison.estimand_coding_mismatch`.
    pub const ESTIMAND_CODING_MISMATCH: &str = "descriptive_comparison.estimand_coding_mismatch";
    /// Detail `descriptive_comparison.fold_exceeds_lease`.
    pub const FOLD_EXCEEDS_LEASE: &str = "descriptive_comparison.fold_exceeds_lease";
    /// Detail `descriptive_comparison.gap_interval_unavailable`.
    pub const GAP_INTERVAL_UNAVAILABLE: &str = "descriptive_comparison.gap_interval_unavailable";
    /// Detail `descriptive_comparison.invalid_covariance`.
    pub const INVALID_COVARIANCE: &str = "descriptive_comparison.invalid_covariance";
    /// Detail `descriptive_comparison.invalid_data`.
    pub const INVALID_DATA: &str = "descriptive_comparison.invalid_data";
    /// Detail `descriptive_comparison.joint_covariance_unavailable`.
    pub const JOINT_COVARIANCE_UNAVAILABLE: &str = "descriptive_comparison.joint_covariance_unavailable";
    /// Detail `descriptive_comparison.learner_exceeds_lease`.
    pub const LEARNER_EXCEEDS_LEASE: &str = "descriptive_comparison.learner_exceeds_lease";
    /// Detail `descriptive_comparison.population_not_all_observed`.
    pub const POPULATION_NOT_ALL_OBSERVED: &str = "descriptive_comparison.population_not_all_observed";
    /// Detail `descriptive_comparison.scale_not_supported`.
    pub const SCALE_NOT_SUPPORTED: &str = "descriptive_comparison.scale_not_supported";
    /// Detail `descriptive_comparison.transform_domain`.
    pub const TRANSFORM_DOMAIN: &str = "descriptive_comparison.transform_domain";
}

/// Details in the `design_ranking` family.
pub mod design_ranking {
    /// Detail `design_ranking.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "design_ranking.bounds_exceeded";
    /// Detail `design_ranking.contract_mismatch`.
    pub const CONTRACT_MISMATCH: &str = "design_ranking.contract_mismatch";
    /// Detail `design_ranking.cost_units_mismatch`.
    pub const COST_UNITS_MISMATCH: &str = "design_ranking.cost_units_mismatch";
    /// Detail `design_ranking.design_error`.
    pub const DESIGN_ERROR: &str = "design_ranking.design_error";
    /// Detail `design_ranking.digest_mismatch`.
    pub const DIGEST_MISMATCH: &str = "design_ranking.digest_mismatch";
    /// Detail `design_ranking.duplicate_candidate`.
    pub const DUPLICATE_CANDIDATE: &str = "design_ranking.duplicate_candidate";
    /// Detail `design_ranking.empty`.
    pub const EMPTY: &str = "design_ranking.empty";
    /// Detail `design_ranking.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "design_ranking.identity_mismatch";
    /// Detail `design_ranking.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "design_ranking.invalid_artifact";
    /// Detail `design_ranking.invalid_request`.
    pub const INVALID_REQUEST: &str = "design_ranking.invalid_request";
    /// Detail `design_ranking.rank_mismatch`.
    pub const RANK_MISMATCH: &str = "design_ranking.rank_mismatch";
    /// Detail `design_ranking.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "design_ranking.replay_mismatch";
    /// Detail `design_ranking.signal_inconsistent`.
    pub const SIGNAL_INCONSISTENT: &str = "design_ranking.signal_inconsistent";
    /// Detail `design_ranking.signal_mismatch`.
    pub const SIGNAL_MISMATCH: &str = "design_ranking.signal_mismatch";
    /// Detail `design_ranking.source_digest_mismatch`.
    pub const SOURCE_DIGEST_MISMATCH: &str = "design_ranking.source_digest_mismatch";
    /// Detail `design_ranking.source_overlap`.
    pub const SOURCE_OVERLAP: &str = "design_ranking.source_overlap";
    /// Detail `design_ranking.source_overlap_unknown`.
    pub const SOURCE_OVERLAP_UNKNOWN: &str = "design_ranking.source_overlap_unknown";
    /// Detail `design_ranking.unsupported_version`.
    pub const UNSUPPORTED_VERSION: &str = "design_ranking.unsupported_version";
}

/// Details in the `discovered_transport` family.
pub mod discovered_transport {
    /// Detail `discovered_transport.class_enumeration_partial`.
    pub const CLASS_ENUMERATION_PARTIAL: &str = "discovered_transport.class_enumeration_partial";
    /// Detail `discovered_transport.class_not_identified`.
    pub const CLASS_NOT_IDENTIFIED: &str = "discovered_transport.class_not_identified";
    /// Detail `discovered_transport.class_too_large`.
    pub const CLASS_TOO_LARGE: &str = "discovered_transport.class_too_large";
    /// Detail `discovered_transport.evidence_invalid`.
    pub const EVIDENCE_INVALID: &str = "discovered_transport.evidence_invalid";
    /// Detail `discovered_transport.member_bound_zero`.
    pub const MEMBER_BOUND_ZERO: &str = "discovered_transport.member_bound_zero";
    /// Detail `discovered_transport.member_not_estimable`.
    pub const MEMBER_NOT_ESTIMABLE: &str = "discovered_transport.member_not_estimable";
    /// Detail `discovered_transport.no_members`.
    pub const NO_MEMBERS: &str = "discovered_transport.no_members";
    /// Detail `discovered_transport.orientation_conflict`.
    pub const ORIENTATION_CONFLICT: &str = "discovered_transport.orientation_conflict";
    /// Detail `discovered_transport.pag_class_ambiguous`.
    pub const PAG_CLASS_AMBIGUOUS: &str = "discovered_transport.pag_class_ambiguous";
    /// Detail `discovered_transport.pag_selection_bias_edge`.
    pub const PAG_SELECTION_BIAS_EDGE: &str = "discovered_transport.pag_selection_bias_edge";
    /// Detail `discovered_transport.search_incomplete`.
    pub const SEARCH_INCOMPLETE: &str = "discovered_transport.search_incomplete";
    /// Detail `discovered_transport.selection_target_unknown`.
    pub const SELECTION_TARGET_UNKNOWN: &str = "discovered_transport.selection_target_unknown";
    /// Detail `discovered_transport.temporal_query_invalid`.
    pub const TEMPORAL_QUERY_INVALID: &str = "discovered_transport.temporal_query_invalid";
    /// Detail `discovered_transport.temporal_structure_unsupported`.
    pub const TEMPORAL_STRUCTURE_UNSUPPORTED: &str = "discovered_transport.temporal_structure_unsupported";
    /// Detail `discovered_transport.trial_compatibility_invalid`.
    pub const TRIAL_COMPATIBILITY_INVALID: &str = "discovered_transport.trial_compatibility_invalid";
    /// Detail `discovered_transport.trial_rejects_every_member`.
    pub const TRIAL_REJECTS_EVERY_MEMBER: &str = "discovered_transport.trial_rejects_every_member";
}

/// Details in the `discretized_scm` family.
pub mod discretized_scm {
    /// Detail `discretized_scm.cancelled`.
    pub const CANCELLED: &str = "discretized_scm.cancelled";
    /// Detail `discretized_scm.composition_incoherent`.
    pub const COMPOSITION_INCOHERENT: &str = "discretized_scm.composition_incoherent";
    /// Detail `discretized_scm.degenerate_marginal`.
    pub const DEGENERATE_MARGINAL: &str = "discretized_scm.degenerate_marginal";
    /// Detail `discretized_scm.intervention_invalid`.
    pub const INTERVENTION_INVALID: &str = "discretized_scm.intervention_invalid";
    /// Detail `discretized_scm.latent_confounding`.
    pub const LATENT_CONFOUNDING: &str = "discretized_scm.latent_confounding";
    /// Detail `discretized_scm.levels_out_of_range`.
    pub const LEVELS_OUT_OF_RANGE: &str = "discretized_scm.levels_out_of_range";
    /// Detail `discretized_scm.levels_undeclared`.
    pub const LEVELS_UNDECLARED: &str = "discretized_scm.levels_undeclared";
    /// Detail `discretized_scm.non_gaussian_mechanism`.
    pub const NON_GAUSSIAN_MECHANISM: &str = "discretized_scm.non_gaussian_mechanism";
    /// Detail `discretized_scm.numerical`.
    pub const NUMERICAL: &str = "discretized_scm.numerical";
    /// Detail `discretized_scm.point_only`.
    pub const POINT_ONLY: &str = "discretized_scm.point_only";
    /// Detail `discretized_scm.state_space_too_large`.
    pub const STATE_SPACE_TOO_LARGE: &str = "discretized_scm.state_space_too_large";
    /// Detail `discretized_scm.support_not_real_line`.
    pub const SUPPORT_NOT_REAL_LINE: &str = "discretized_scm.support_not_real_line";
    /// Detail `discretized_scm.unknown_variable`.
    pub const UNKNOWN_VARIABLE: &str = "discretized_scm.unknown_variable";
}

/// Details in the `discretized_scm_artifact` family.
pub mod discretized_scm_artifact {
    /// Detail `discretized_scm_artifact.claimed_numbers_differ`.
    pub const CLAIMED_NUMBERS_DIFFER: &str = "discretized_scm_artifact.claimed_numbers_differ";
    /// Detail `discretized_scm_artifact.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "discretized_scm_artifact.identity_mismatch";
    /// Detail `discretized_scm_artifact.malformed`.
    pub const MALFORMED: &str = "discretized_scm_artifact.malformed";
    /// Detail `discretized_scm_artifact.not_exportable`.
    pub const NOT_EXPORTABLE: &str = "discretized_scm_artifact.not_exportable";
    /// Detail `discretized_scm_artifact.rederivation_refused`.
    pub const REDERIVATION_REFUSED: &str = "discretized_scm_artifact.rederivation_refused";
    /// Detail `discretized_scm_artifact.version_unsupported`.
    pub const VERSION_UNSUPPORTED: &str = "discretized_scm_artifact.version_unsupported";
}

/// Details in the `distribution_meaning` family.
pub mod distribution_meaning {
    /// Detail `distribution_meaning.incompatible_operation`.
    pub const INCOMPATIBLE_OPERATION: &str = "distribution_meaning.incompatible_operation";
    /// Detail `distribution_meaning.legacy_posterior`.
    pub const LEGACY_POSTERIOR: &str = "distribution_meaning.legacy_posterior";
    /// Detail `distribution_meaning.observational_not_interventional`.
    pub const OBSERVATIONAL_NOT_INTERVENTIONAL: &str = "distribution_meaning.observational_not_interventional";
    /// Detail `distribution_meaning.posterior_interval`.
    pub const POSTERIOR_INTERVAL: &str = "distribution_meaning.posterior_interval";
}

/// Details in the `dml` family.
pub mod dml {
    /// Detail `dml.arm_not_populated`.
    pub const ARM_NOT_POPULATED: &str = "dml.arm_not_populated";
    /// Detail `dml.cancelled`.
    pub const CANCELLED: &str = "dml.cancelled";
}

/// Details in the `dose_grid` family.
pub mod dose_grid {
    /// Detail `dose_grid.bandwidth_outside_range`.
    pub const BANDWIDTH_OUTSIDE_RANGE: &str = "dose_grid.bandwidth_outside_range";
    /// Detail `dose_grid.cancelled`.
    pub const CANCELLED: &str = "dose_grid.cancelled";
    /// Detail `dose_grid.consumer_limit_exceeded`.
    pub const CONSUMER_LIMIT_EXCEEDED: &str = "dose_grid.consumer_limit_exceeded";
    /// Detail `dose_grid.data_identity_mismatch`.
    pub const DATA_IDENTITY_MISMATCH: &str = "dose_grid.data_identity_mismatch";
    /// Detail `dose_grid.derivative_without_claim`.
    pub const DERIVATIVE_WITHOUT_CLAIM: &str = "dose_grid.derivative_without_claim";
    /// Detail `dose_grid.graph_not_certified`.
    pub const GRAPH_NOT_CERTIFIED: &str = "dose_grid.graph_not_certified";
    /// Detail `dose_grid.insufficient_local_weight`.
    pub const INSUFFICIENT_LOCAL_WEIGHT: &str = "dose_grid.insufficient_local_weight";
    /// Detail `dose_grid.invalid_lag`.
    pub const INVALID_LAG: &str = "dose_grid.invalid_lag";
    /// Detail `dose_grid.invalid_request`.
    pub const INVALID_REQUEST: &str = "dose_grid.invalid_request";
    /// Detail `dose_grid.level_without_claim`.
    pub const LEVEL_WITHOUT_CLAIM: &str = "dose_grid.level_without_claim";
    /// Detail `dose_grid.no_simultaneous_band_method`.
    pub const NO_SIMULTANEOUS_BAND_METHOD: &str = "dose_grid.no_simultaneous_band_method";
    /// Detail `dose_grid.non_finite_result`.
    pub const NON_FINITE_RESULT: &str = "dose_grid.non_finite_result";
    /// Detail `dose_grid.premises_mismatch`.
    pub const PREMISES_MISMATCH: &str = "dose_grid.premises_mismatch";
    /// Detail `dose_grid.refused`.
    pub const REFUSED: &str = "dose_grid.refused";
    /// Detail `dose_grid.resource_limit`.
    pub const RESOURCE_LIMIT: &str = "dose_grid.resource_limit";
    /// Detail `dose_grid.resource_refused`.
    pub const RESOURCE_REFUSED: &str = "dose_grid.resource_refused";
    /// Detail `dose_grid.result_replay_mismatch`.
    pub const RESULT_REPLAY_MISMATCH: &str = "dose_grid.result_replay_mismatch";
    /// Detail `dose_grid.too_few_clusters`.
    pub const TOO_FEW_CLUSTERS: &str = "dose_grid.too_few_clusters";
    /// Detail `dose_grid.undecodable`.
    pub const UNDECODABLE: &str = "dose_grid.undecodable";
    /// Detail `dose_grid.unsupported_dose`.
    pub const UNSUPPORTED_DOSE: &str = "dose_grid.unsupported_dose";
    /// Detail `dose_grid.unsupported_semantics`.
    pub const UNSUPPORTED_SEMANTICS: &str = "dose_grid.unsupported_semantics";
}

/// Details in the `dose_grid_posterior` family.
pub mod dose_grid_posterior {
    /// Detail `dose_grid_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "dose_grid_posterior.band_not_constructed";
    /// Detail `dose_grid_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "dose_grid_posterior.external_power_prior";
    /// Detail `dose_grid_posterior.insufficient_weight`.
    pub const INSUFFICIENT_WEIGHT: &str = "dose_grid_posterior.insufficient_weight";
    /// Detail `dose_grid_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "dose_grid_posterior.mapped_prior";
    /// Detail `dose_grid_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "dose_grid_posterior.non_positive_scale";
}

/// Details in the `dose_response` family.
pub mod dose_response {
    /// Detail `dose_response.bootstrap_below_floor`.
    pub const BOOTSTRAP_BELOW_FLOOR: &str = "dose_response.bootstrap_below_floor";
    /// Detail `dose_response.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "dose_response.bounds_exceeded";
    /// Detail `dose_response.cate_or_simultaneous`.
    pub const CATE_OR_SIMULTANEOUS: &str = "dose_response.cate_or_simultaneous";
    /// Detail `dose_response.declared_folds_unbalanced`.
    pub const DECLARED_FOLDS_UNBALANCED: &str = "dose_response.declared_folds_unbalanced";
    /// Detail `dose_response.dose_density_floor`.
    pub const DOSE_DENSITY_FLOOR: &str = "dose_response.dose_density_floor";
    /// Detail `dose_response.dose_density_invalid`.
    pub const DOSE_DENSITY_INVALID: &str = "dose_response.dose_density_invalid";
    /// Detail `dose_response.dose_too_discrete`.
    pub const DOSE_TOO_DISCRETE: &str = "dose_response.dose_too_discrete";
    /// Detail `dose_response.estimated_dose_density`.
    pub const ESTIMATED_DOSE_DENSITY: &str = "dose_response.estimated_dose_density";
    /// Detail `dose_response.grid_outside_dose_support`.
    pub const GRID_OUTSIDE_DOSE_SUPPORT: &str = "dose_response.grid_outside_dose_support";
    /// Detail `dose_response.invalid_declared_rows`.
    pub const INVALID_DECLARED_ROWS: &str = "dose_response.invalid_declared_rows";
    /// Detail `dose_response.kernel_not_supported`.
    pub const KERNEL_NOT_SUPPORTED: &str = "dose_response.kernel_not_supported";
    /// Detail `dose_response.learner_not_portable`.
    pub const LEARNER_NOT_PORTABLE: &str = "dose_response.learner_not_portable";
    /// Detail `dose_response.local_dose_ess`.
    pub const LOCAL_DOSE_ESS: &str = "dose_response.local_dose_ess";
    /// Detail `dose_response.membership_overlap`.
    pub const MEMBERSHIP_OVERLAP: &str = "dose_response.membership_overlap";
    /// Detail `dose_response.non_iid_design`.
    pub const NON_IID_DESIGN: &str = "dose_response.non_iid_design";
    /// Detail `dose_response.posterior_declared_rows`.
    pub const POSTERIOR_DECLARED_ROWS: &str = "dose_response.posterior_declared_rows";
    /// Detail `dose_response.quadrature_tolerance`.
    pub const QUADRATURE_TOLERANCE: &str = "dose_response.quadrature_tolerance";
    /// Detail `dose_response.target_not_smoothed`.
    pub const TARGET_NOT_SMOOTHED: &str = "dose_response.target_not_smoothed";
}

/// Details in the `effect_constancy` family.
pub mod effect_constancy {
    /// Detail `effect_constancy.estimand_missing`.
    pub const ESTIMAND_MISSING: &str = "effect_constancy.estimand_missing";
    /// Detail `effect_constancy.incompatible_partitions`.
    pub const INCOMPATIBLE_PARTITIONS: &str = "effect_constancy.incompatible_partitions";
    /// Detail `effect_constancy.invalid_alpha`.
    pub const INVALID_ALPHA: &str = "effect_constancy.invalid_alpha";
    /// Detail `effect_constancy.invalid_covariance`.
    pub const INVALID_COVARIANCE: &str = "effect_constancy.invalid_covariance";
    /// Detail `effect_constancy.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "effect_constancy.invalid_identity";
    /// Detail `effect_constancy.invalid_request`.
    pub const INVALID_REQUEST: &str = "effect_constancy.invalid_request";
    /// Detail `effect_constancy.invalid_standard_error`.
    pub const INVALID_STANDARD_ERROR: &str = "effect_constancy.invalid_standard_error";
    /// Detail `effect_constancy.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "effect_constancy.limits_exceeded";
    /// Detail `effect_constancy.non_finite_estimate`.
    pub const NON_FINITE_ESTIMATE: &str = "effect_constancy.non_finite_estimate";
    /// Detail `effect_constancy.too_few_partitions`.
    pub const TOO_FEW_PARTITIONS: &str = "effect_constancy.too_few_partitions";
    /// Detail `effect_constancy.too_many_partitions`.
    pub const TOO_MANY_PARTITIONS: &str = "effect_constancy.too_many_partitions";
    /// Detail `effect_constancy.unknown_reference`.
    pub const UNKNOWN_REFERENCE: &str = "effect_constancy.unknown_reference";
    /// Detail `effect_constancy.unsupported_partition`.
    pub const UNSUPPORTED_PARTITION: &str = "effect_constancy.unsupported_partition";
    /// Detail `effect_constancy.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "effect_constancy.wrong_contract";
}

/// Details in the `effect_constancy_consumer` family.
pub mod effect_constancy_consumer {
    /// Detail `effect_constancy_consumer.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "effect_constancy_consumer.artifact_invalid";
    /// Detail `effect_constancy_consumer.catalog_binding_invalid`.
    pub const CATALOG_BINDING_INVALID: &str = "effect_constancy_consumer.catalog_binding_invalid";
    /// Detail `effect_constancy_consumer.contrast_missing`.
    pub const CONTRAST_MISSING: &str = "effect_constancy_consumer.contrast_missing";
    /// Detail `effect_constancy_consumer.decision_unsupported`.
    pub const DECISION_UNSUPPORTED: &str = "effect_constancy_consumer.decision_unsupported";
    /// Detail `effect_constancy_consumer.invalid_request`.
    pub const INVALID_REQUEST: &str = "effect_constancy_consumer.invalid_request";
    /// Detail `effect_constancy_consumer.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "effect_constancy_consumer.limits_exceeded";
    /// Detail `effect_constancy_consumer.partition_missing`.
    pub const PARTITION_MISSING: &str = "effect_constancy_consumer.partition_missing";
    /// Detail `effect_constancy_consumer.quantity_mismatch`.
    pub const QUANTITY_MISMATCH: &str = "effect_constancy_consumer.quantity_mismatch";
    /// Detail `effect_constancy_consumer.support_unsupported`.
    pub const SUPPORT_UNSUPPORTED: &str = "effect_constancy_consumer.support_unsupported";
}

/// Details in the `evidence_ledger` family.
pub mod evidence_ledger {
    /// Detail `evidence_ledger.cancelled`.
    pub const CANCELLED: &str = "evidence_ledger.cancelled";
    /// Detail `evidence_ledger.diagonal_covariance_shared_rows`.
    pub const DIAGONAL_COVARIANCE_SHARED_ROWS: &str = "evidence_ledger.diagonal_covariance_shared_rows";
    /// Detail `evidence_ledger.duplicate_result`.
    pub const DUPLICATE_RESULT: &str = "evidence_ledger.duplicate_result";
    /// Detail `evidence_ledger.duplicate_row`.
    pub const DUPLICATE_ROW: &str = "evidence_ledger.duplicate_row";
    /// Detail `evidence_ledger.duplicate_source`.
    pub const DUPLICATE_SOURCE: &str = "evidence_ledger.duplicate_source";
    /// Detail `evidence_ledger.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "evidence_ledger.identity_mismatch";
    /// Detail `evidence_ledger.interval_construction_unavailable`.
    pub const INTERVAL_CONSTRUCTION_UNAVAILABLE: &str = "evidence_ledger.interval_construction_unavailable";
    /// Detail `evidence_ledger.interval_report_invalid`.
    pub const INTERVAL_REPORT_INVALID: &str = "evidence_ledger.interval_report_invalid";
    /// Detail `evidence_ledger.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "evidence_ledger.invalid_declaration";
    /// Detail `evidence_ledger.kind_mismatch`.
    pub const KIND_MISMATCH: &str = "evidence_ledger.kind_mismatch";
    /// Detail `evidence_ledger.limit_exceeded`.
    pub const LIMIT_EXCEEDED: &str = "evidence_ledger.limit_exceeded";
    /// Detail `evidence_ledger.lineage_rows_mismatch`.
    pub const LINEAGE_ROWS_MISMATCH: &str = "evidence_ledger.lineage_rows_mismatch";
    /// Detail `evidence_ledger.native_and_transported`.
    pub const NATIVE_AND_TRANSPORTED: &str = "evidence_ledger.native_and_transported";
    /// Detail `evidence_ledger.operation_arity`.
    pub const OPERATION_ARITY: &str = "evidence_ledger.operation_arity";
    /// Detail `evidence_ledger.overlap_unknown`.
    pub const OVERLAP_UNKNOWN: &str = "evidence_ledger.overlap_unknown";
    /// Detail `evidence_ledger.reentry_derived_from_used_data`.
    pub const REENTRY_DERIVED_FROM_USED_DATA: &str = "evidence_ledger.reentry_derived_from_used_data";
    /// Detail `evidence_ledger.reentry_duplicate`.
    pub const REENTRY_DUPLICATE: &str = "evidence_ledger.reentry_duplicate";
    /// Detail `evidence_ledger.reentry_prior_plus_likelihood`.
    pub const REENTRY_PRIOR_PLUS_LIKELIHOOD: &str = "evidence_ledger.reentry_prior_plus_likelihood";
    /// Detail `evidence_ledger.reentry_transform`.
    pub const REENTRY_TRANSFORM: &str = "evidence_ledger.reentry_transform";
    /// Detail `evidence_ledger.repeated_source`.
    pub const REPEATED_SOURCE: &str = "evidence_ledger.repeated_source";
    /// Detail `evidence_ledger.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "evidence_ledger.replay_mismatch";
    /// Detail `evidence_ledger.result_not_valid`.
    pub const RESULT_NOT_VALID: &str = "evidence_ledger.result_not_valid";
    /// Detail `evidence_ledger.shared_observations`.
    pub const SHARED_OBSERVATIONS: &str = "evidence_ledger.shared_observations";
    /// Detail `evidence_ledger.source_superseded`.
    pub const SOURCE_SUPERSEDED: &str = "evidence_ledger.source_superseded";
    /// Detail `evidence_ledger.source_withdrawn`.
    pub const SOURCE_WITHDRAWN: &str = "evidence_ledger.source_withdrawn";
    /// Detail `evidence_ledger.supersede_kind_mismatch`.
    pub const SUPERSEDE_KIND_MISMATCH: &str = "evidence_ledger.supersede_kind_mismatch";
    /// Detail `evidence_ledger.synthesis_mismatch`.
    pub const SYNTHESIS_MISMATCH: &str = "evidence_ledger.synthesis_mismatch";
    /// Detail `evidence_ledger.transport_identity_missing`.
    pub const TRANSPORT_IDENTITY_MISSING: &str = "evidence_ledger.transport_identity_missing";
    /// Detail `evidence_ledger.transported_as_native`.
    pub const TRANSPORTED_AS_NATIVE: &str = "evidence_ledger.transported_as_native";
    /// Detail `evidence_ledger.unknown_result`.
    pub const UNKNOWN_RESULT: &str = "evidence_ledger.unknown_result";
    /// Detail `evidence_ledger.unknown_source`.
    pub const UNKNOWN_SOURCE: &str = "evidence_ledger.unknown_source";
    /// Detail `evidence_ledger.update_drops_rows`.
    pub const UPDATE_DROPS_ROWS: &str = "evidence_ledger.update_drops_rows";
}

/// Details in the `evidence_obligations` family.
pub mod evidence_obligations {
    /// Detail `evidence_obligations.invalid_obligation`.
    pub const INVALID_OBLIGATION: &str = "evidence_obligations.invalid_obligation";
    /// Detail `evidence_obligations.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "evidence_obligations.wrong_contract";
}

/// Details in the `evidence_plan` family.
pub mod evidence_plan {
    /// Detail `evidence_plan.budget`.
    pub const BUDGET: &str = "evidence_plan.budget";
    /// Detail `evidence_plan.compile_refused`.
    pub const COMPILE_REFUSED: &str = "evidence_plan.compile_refused";
    /// Detail `evidence_plan.cost_units_mismatch`.
    pub const COST_UNITS_MISMATCH: &str = "evidence_plan.cost_units_mismatch";
    /// Detail `evidence_plan.invalid_request`.
    pub const INVALID_REQUEST: &str = "evidence_plan.invalid_request";
    /// Detail `evidence_plan.no_failure_to_plan`.
    pub const NO_FAILURE_TO_PLAN: &str = "evidence_plan.no_failure_to_plan";
    /// Detail `evidence_plan.no_route`.
    pub const NO_ROUTE: &str = "evidence_plan.no_route";
    /// Detail `evidence_plan.none_certified`.
    pub const NONE_CERTIFIED: &str = "evidence_plan.none_certified";
}

/// Details in the `evsi` family.
pub mod evsi {
    /// Detail `evsi.action_set_changed`.
    pub const ACTION_SET_CHANGED: &str = "evsi.action_set_changed";
    /// Detail `evsi.bound_violation`.
    pub const BOUND_VIOLATION: &str = "evsi.bound_violation";
    /// Detail `evsi.cancelled`.
    pub const CANCELLED: &str = "evsi.cancelled";
    /// Detail `evsi.cost_map_required`.
    pub const COST_MAP_REQUIRED: &str = "evsi.cost_map_required";
    /// Detail `evsi.cost_units_mismatch`.
    pub const COST_UNITS_MISMATCH: &str = "evsi.cost_units_mismatch";
    /// Detail `evsi.decision_values_incoherent`.
    pub const DECISION_VALUES_INCOHERENT: &str = "evsi.decision_values_incoherent";
    /// Detail `evsi.decision_values_need_draws`.
    pub const DECISION_VALUES_NEED_DRAWS: &str = "evsi.decision_values_need_draws";
    /// Detail `evsi.design_error`.
    pub const DESIGN_ERROR: &str = "evsi.design_error";
    /// Detail `evsi.duplicate_candidate`.
    pub const DUPLICATE_CANDIDATE: &str = "evsi.duplicate_candidate";
    /// Detail `evsi.execution_resource_limit`.
    pub const EXECUTION_RESOURCE_LIMIT: &str = "evsi.execution_resource_limit";
    /// Detail `evsi.invalid_cost`.
    pub const INVALID_COST: &str = "evsi.invalid_cost";
    /// Detail `evsi.invalid_request`.
    pub const INVALID_REQUEST: &str = "evsi.invalid_request";
    /// Detail `evsi.prior_states_mismatch`.
    pub const PRIOR_STATES_MISMATCH: &str = "evsi.prior_states_mismatch";
    /// Detail `evsi.receipt_mismatch`.
    pub const RECEIPT_MISMATCH: &str = "evsi.receipt_mismatch";
    /// Detail `evsi.sample_size_mismatch`.
    pub const SAMPLE_SIZE_MISMATCH: &str = "evsi.sample_size_mismatch";
    /// Detail `evsi.source_overlap`.
    pub const SOURCE_OVERLAP: &str = "evsi.source_overlap";
    /// Detail `evsi.source_overlap_unknown`.
    pub const SOURCE_OVERLAP_UNKNOWN: &str = "evsi.source_overlap_unknown";
    /// Detail `evsi.unscored_candidate`.
    pub const UNSCORED_CANDIDATE: &str = "evsi.unscored_candidate";
}

/// Details in the `external_mechanism` family.
pub mod external_mechanism {
    /// Detail `external_mechanism.declaration_invalid`.
    pub const DECLARATION_INVALID: &str = "external_mechanism.declaration_invalid";
    /// Detail `external_mechanism.implementation_changed`.
    pub const IMPLEMENTATION_CHANGED: &str = "external_mechanism.implementation_changed";
    /// Detail `external_mechanism.implementation_unavailable`.
    pub const IMPLEMENTATION_UNAVAILABLE: &str = "external_mechanism.implementation_unavailable";
    /// Detail `external_mechanism.resolver_failed`.
    pub const RESOLVER_FAILED: &str = "external_mechanism.resolver_failed";
    /// Detail `external_mechanism.resolver_missing`.
    pub const RESOLVER_MISSING: &str = "external_mechanism.resolver_missing";
}

/// Details in the `external_object_verification` family.
pub mod external_object_verification {
    /// Detail `external_object_verification.duplicate_probe`.
    pub const DUPLICATE_PROBE: &str = "external_object_verification.duplicate_probe";
    /// Detail `external_object_verification.failed_probe`.
    pub const FAILED_PROBE: &str = "external_object_verification.failed_probe";
    /// Detail `external_object_verification.invalid_probe`.
    pub const INVALID_PROBE: &str = "external_object_verification.invalid_probe";
    /// Detail `external_object_verification.missing_probe`.
    pub const MISSING_PROBE: &str = "external_object_verification.missing_probe";
}

/// Details in the `external_recalc` family.
pub mod external_recalc {
    /// Detail `external_recalc.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "external_recalc.artifact_invalid";
    /// Detail `external_recalc.attempt_limit`.
    pub const ATTEMPT_LIMIT: &str = "external_recalc.attempt_limit";
    /// Detail `external_recalc.callback_failed`.
    pub const CALLBACK_FAILED: &str = "external_recalc.callback_failed";
    /// Detail `external_recalc.contract_mismatch`.
    pub const CONTRACT_MISMATCH: &str = "external_recalc.contract_mismatch";
    /// Detail `external_recalc.idempotency_conflict`.
    pub const IDEMPOTENCY_CONFLICT: &str = "external_recalc.idempotency_conflict";
    /// Detail `external_recalc.idempotency_unsupported`.
    pub const IDEMPOTENCY_UNSUPPORTED: &str = "external_recalc.idempotency_unsupported";
    /// Detail `external_recalc.invalid_descriptor`.
    pub const INVALID_DESCRIPTOR: &str = "external_recalc.invalid_descriptor";
    /// Detail `external_recalc.invalid_output`.
    pub const INVALID_OUTPUT: &str = "external_recalc.invalid_output";
    /// Detail `external_recalc.invalid_request`.
    pub const INVALID_REQUEST: &str = "external_recalc.invalid_request";
    /// Detail `external_recalc.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "external_recalc.limits_exceeded";
    /// Detail `external_recalc.output_unavailable`.
    pub const OUTPUT_UNAVAILABLE: &str = "external_recalc.output_unavailable";
    /// Detail `external_recalc.policy_unsupported`.
    pub const POLICY_UNSUPPORTED: &str = "external_recalc.policy_unsupported";
    /// Detail `external_recalc.provider_mismatch`.
    pub const PROVIDER_MISMATCH: &str = "external_recalc.provider_mismatch";
    /// Detail `external_recalc.provider_unavailable`.
    pub const PROVIDER_UNAVAILABLE: &str = "external_recalc.provider_unavailable";
    /// Detail `external_recalc.quantity_unsupported`.
    pub const QUANTITY_UNSUPPORTED: &str = "external_recalc.quantity_unsupported";
    /// Detail `external_recalc.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "external_recalc.replay_mismatch";
    /// Detail `external_recalc.replay_unsupported`.
    pub const REPLAY_UNSUPPORTED: &str = "external_recalc.replay_unsupported";
    /// Detail `external_recalc.retry_unsafe`.
    pub const RETRY_UNSAFE: &str = "external_recalc.retry_unsafe";
}

/// Details in the `external_response_binding` family.
pub mod external_response_binding {
    /// Detail `external_response_binding.column_quantity`.
    pub const COLUMN_QUANTITY: &str = "external_response_binding.column_quantity";
    /// Detail `external_response_binding.column_support_length`.
    pub const COLUMN_SUPPORT_LENGTH: &str = "external_response_binding.column_support_length";
    /// Detail `external_response_binding.contract_not_identified`.
    pub const CONTRACT_NOT_IDENTIFIED: &str = "external_response_binding.contract_not_identified";
    /// Detail `external_response_binding.coordinate_conditioning`.
    pub const COORDINATE_CONDITIONING: &str = "external_response_binding.coordinate_conditioning";
    /// Detail `external_response_binding.coordinate_functional`.
    pub const COORDINATE_FUNCTIONAL: &str = "external_response_binding.coordinate_functional";
    /// Detail `external_response_binding.coordinate_horizon`.
    pub const COORDINATE_HORIZON: &str = "external_response_binding.coordinate_horizon";
    /// Detail `external_response_binding.coordinate_invalid_coordinate`.
    pub const COORDINATE_INVALID_COORDINATE: &str = "external_response_binding.coordinate_invalid_coordinate";
    /// Detail `external_response_binding.coordinate_population`.
    pub const COORDINATE_POPULATION: &str = "external_response_binding.coordinate_population";
    /// Detail `external_response_binding.coordinate_regime`.
    pub const COORDINATE_REGIME: &str = "external_response_binding.coordinate_regime";
    /// Detail `external_response_binding.coordinate_transform`.
    pub const COORDINATE_TRANSFORM: &str = "external_response_binding.coordinate_transform";
    /// Detail `external_response_binding.coordinate_units`.
    pub const COORDINATE_UNITS: &str = "external_response_binding.coordinate_units";
    /// Detail `external_response_binding.coordinate_variable`.
    pub const COORDINATE_VARIABLE: &str = "external_response_binding.coordinate_variable";
    /// Detail `external_response_binding.dimension`.
    pub const DIMENSION: &str = "external_response_binding.dimension";
    /// Detail `external_response_binding.distribution_meaning`.
    pub const DISTRIBUTION_MEANING: &str = "external_response_binding.distribution_meaning";
    /// Detail `external_response_binding.empty_column`.
    pub const EMPTY_COLUMN: &str = "external_response_binding.empty_column";
    /// Detail `external_response_binding.graph`.
    pub const GRAPH: &str = "external_response_binding.graph";
    /// Detail `external_response_binding.invalid_contract`.
    pub const INVALID_CONTRACT: &str = "external_response_binding.invalid_contract";
    /// Detail `external_response_binding.malformed_artifact`.
    pub const MALFORMED_ARTIFACT: &str = "external_response_binding.malformed_artifact";
    /// Detail `external_response_binding.malformed_capability`.
    pub const MALFORMED_CAPABILITY: &str = "external_response_binding.malformed_capability";
    /// Detail `external_response_binding.malformed_equivalence`.
    pub const MALFORMED_EQUIVALENCE: &str = "external_response_binding.malformed_equivalence";
    /// Detail `external_response_binding.malformed_identification`.
    pub const MALFORMED_IDENTIFICATION: &str = "external_response_binding.malformed_identification";
    /// Detail `external_response_binding.malformed_meaning`.
    pub const MALFORMED_MEANING: &str = "external_response_binding.malformed_meaning";
    /// Detail `external_response_binding.malformed_probe`.
    pub const MALFORMED_PROBE: &str = "external_response_binding.malformed_probe";
    /// Detail `external_response_binding.malformed_quantity`.
    pub const MALFORMED_QUANTITY: &str = "external_response_binding.malformed_quantity";
    /// Detail `external_response_binding.malformed_support`.
    pub const MALFORMED_SUPPORT: &str = "external_response_binding.malformed_support";
    /// Detail `external_response_binding.malformed_trust`.
    pub const MALFORMED_TRUST: &str = "external_response_binding.malformed_trust";
    /// Detail `external_response_binding.missing_assumption`.
    pub const MISSING_ASSUMPTION: &str = "external_response_binding.missing_assumption";
    /// Detail `external_response_binding.missing_evidence`.
    pub const MISSING_EVIDENCE: &str = "external_response_binding.missing_evidence";
    /// Detail `external_response_binding.non_finite_value`.
    pub const NON_FINITE_VALUE: &str = "external_response_binding.non_finite_value";
    /// Detail `external_response_binding.object_kind`.
    pub const OBJECT_KIND: &str = "external_response_binding.object_kind";
    /// Detail `external_response_binding.object_quantities`.
    pub const OBJECT_QUANTITIES: &str = "external_response_binding.object_quantities";
    /// Detail `external_response_binding.posterior_kind`.
    pub const POSTERIOR_KIND: &str = "external_response_binding.posterior_kind";
    /// Detail `external_response_binding.unchecked_observational_law`.
    pub const UNCHECKED_OBSERVATIONAL_LAW: &str = "external_response_binding.unchecked_observational_law";
}

/// Details in the `external_scientific_providers` family.
pub mod external_scientific_providers {
    /// Detail `external_scientific_providers.duplicate_declaration`.
    pub const DUPLICATE_DECLARATION: &str = "external_scientific_providers.duplicate_declaration";
    /// Detail `external_scientific_providers.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "external_scientific_providers.invalid_declaration";
    /// Detail `external_scientific_providers.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "external_scientific_providers.invalid_identity";
}

/// Details in the `external_trust_receipt` family.
pub mod external_trust_receipt {
    /// Detail `external_trust_receipt.receipt_mismatch`.
    pub const RECEIPT_MISMATCH: &str = "external_trust_receipt.receipt_mismatch";
}

/// Details in the `finite_action_posterior` family.
pub mod finite_action_posterior {
    /// Detail `finite_action_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "finite_action_posterior.band_not_constructed";
    /// Detail `finite_action_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "finite_action_posterior.empty_law";
    /// Detail `finite_action_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "finite_action_posterior.external_power_prior";
    /// Detail `finite_action_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "finite_action_posterior.mapped_prior";
    /// Detail `finite_action_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "finite_action_posterior.non_positive_scale";
}

/// Details in the `frozen_scores` family.
pub mod frozen_scores {
    /// Detail `frozen_scores.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "frozen_scores.identity_mismatch";
}

/// Details in the `functional_inverse_posterior` family.
pub mod functional_inverse_posterior {
    /// Detail `functional_inverse_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "functional_inverse_posterior.band_not_constructed";
    /// Detail `functional_inverse_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "functional_inverse_posterior.empty_law";
    /// Detail `functional_inverse_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "functional_inverse_posterior.external_power_prior";
    /// Detail `functional_inverse_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "functional_inverse_posterior.mapped_prior";
    /// Detail `functional_inverse_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "functional_inverse_posterior.non_positive_scale";
}

/// Details in the `functional_inverse_query` family.
pub mod functional_inverse_query {
    /// Detail `functional_inverse_query.artifact_kind_mismatch`.
    pub const ARTIFACT_KIND_MISMATCH: &str = "functional_inverse_query.artifact_kind_mismatch";
    /// Detail `functional_inverse_query.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "functional_inverse_query.bounds_exceeded";
    /// Detail `functional_inverse_query.forward_origin_invalid`.
    pub const FORWARD_ORIGIN_INVALID: &str = "functional_inverse_query.forward_origin_invalid";
    /// Detail `functional_inverse_query.global_feasibility_claim`.
    pub const GLOBAL_FEASIBILITY_CLAIM: &str = "functional_inverse_query.global_feasibility_claim";
    /// Detail `functional_inverse_query.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "functional_inverse_query.invalid_artifact";
    /// Detail `functional_inverse_query.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "functional_inverse_query.wrong_contract";
}

/// Details in the `functional_source` family.
pub mod functional_source {
    /// Detail `functional_source.expected_identity_mismatch`.
    pub const EXPECTED_IDENTITY_MISMATCH: &str = "functional_source.expected_identity_mismatch";
    /// Detail `functional_source.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "functional_source.invalid_artifact";
    /// Detail `functional_source.invalid_contract`.
    pub const INVALID_CONTRACT: &str = "functional_source.invalid_contract";
    /// Detail `functional_source.invalid_source`.
    pub const INVALID_SOURCE: &str = "functional_source.invalid_source";
    /// Detail `functional_source.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "functional_source.limits_exceeded";
    /// Detail `functional_source.missing_action`.
    pub const MISSING_ACTION: &str = "functional_source.missing_action";
    /// Detail `functional_source.native_source_unavailable`.
    pub const NATIVE_SOURCE_UNAVAILABLE: &str = "functional_source.native_source_unavailable";
    /// Detail `functional_source.original_law_required`.
    pub const ORIGINAL_LAW_REQUIRED: &str = "functional_source.original_law_required";
    /// Detail `functional_source.output_mismatch`.
    pub const OUTPUT_MISMATCH: &str = "functional_source.output_mismatch";
    /// Detail `functional_source.source_mismatch`.
    pub const SOURCE_MISMATCH: &str = "functional_source.source_mismatch";
}

/// Details in the `graph_falsification` family.
pub mod graph_falsification {
    /// Detail `graph_falsification.ci_p_value_missing`.
    pub const CI_P_VALUE_MISSING: &str = "graph_falsification.ci_p_value_missing";
    /// Detail `graph_falsification.ci_result_count`.
    pub const CI_RESULT_COUNT: &str = "graph_falsification.ci_result_count";
    /// Detail `graph_falsification.ci_test_not_frequentist`.
    pub const CI_TEST_NOT_FREQUENTIST: &str = "graph_falsification.ci_test_not_frequentist";
    /// Detail `graph_falsification.data_not_observed`.
    pub const DATA_NOT_OBSERVED: &str = "graph_falsification.data_not_observed";
    /// Detail `graph_falsification.duplicate_variable`.
    pub const DUPLICATE_VARIABLE: &str = "graph_falsification.duplicate_variable";
    /// Detail `graph_falsification.empty_data`.
    pub const EMPTY_DATA: &str = "graph_falsification.empty_data";
    /// Detail `graph_falsification.invalid_level`.
    pub const INVALID_LEVEL: &str = "graph_falsification.invalid_level";
    /// Detail `graph_falsification.invalid_permutations`.
    pub const INVALID_PERMUTATIONS: &str = "graph_falsification.invalid_permutations";
    /// Detail `graph_falsification.node_not_static`.
    pub const NODE_NOT_STATIC: &str = "graph_falsification.node_not_static";
    /// Detail `graph_falsification.too_few_nodes`.
    pub const TOO_FEW_NODES: &str = "graph_falsification.too_few_nodes";
    /// Detail `graph_falsification.work_limit`.
    pub const WORK_LIMIT: &str = "graph_falsification.work_limit";
}

/// Details in the `graph_imputation` family.
pub mod graph_imputation {
    /// Detail `graph_imputation.atom_failure`.
    pub const ATOM_FAILURE: &str = "graph_imputation.atom_failure";
    /// Detail `graph_imputation.atom_not_lowered`.
    pub const ATOM_NOT_LOWERED: &str = "graph_imputation.atom_not_lowered";
    /// Detail `graph_imputation.cancelled`.
    pub const CANCELLED: &str = "graph_imputation.cancelled";
    /// Detail `graph_imputation.candidate_refused`.
    pub const CANDIDATE_REFUSED: &str = "graph_imputation.candidate_refused";
    /// Detail `graph_imputation.compiled`.
    pub const COMPILED: &str = "graph_imputation.compiled";
    /// Detail `graph_imputation.execution_refused`.
    pub const EXECUTION_REFUSED: &str = "graph_imputation.execution_refused";
    /// Detail `graph_imputation.interval_unsupported`.
    pub const INTERVAL_UNSUPPORTED: &str = "graph_imputation.interval_unsupported";
    /// Detail `graph_imputation.invalid_source`.
    pub const INVALID_SOURCE: &str = "graph_imputation.invalid_source";
    /// Detail `graph_imputation.lowering`.
    pub const LOWERING: &str = "graph_imputation.lowering";
    /// Detail `graph_imputation.point_unsupported`.
    pub const POINT_UNSUPPORTED: &str = "graph_imputation.point_unsupported";
    /// Detail `graph_imputation.resource_limit`.
    pub const RESOURCE_LIMIT: &str = "graph_imputation.resource_limit";
    /// Detail `graph_imputation.search_stopped`.
    pub const SEARCH_STOPPED: &str = "graph_imputation.search_stopped";
    /// Detail `graph_imputation.supported_graph_unidentified`.
    pub const SUPPORTED_GRAPH_UNIDENTIFIED: &str = "graph_imputation.supported_graph_unidentified";
    /// Detail `graph_imputation.unidentified`.
    pub const UNIDENTIFIED: &str = "graph_imputation.unidentified";
    /// Detail `graph_imputation.unscorable`.
    pub const UNSCORABLE: &str = "graph_imputation.unscorable";
    /// Detail `graph_imputation.unsupported`.
    pub const UNSUPPORTED: &str = "graph_imputation.unsupported";
    /// Detail `graph_imputation.wrong_dispatch`.
    pub const WRONG_DISPATCH: &str = "graph_imputation.wrong_dispatch";
}

/// Details in the `graph_selection_imputation` family.
pub mod graph_selection_imputation {
    /// Detail `graph_selection_imputation.columns`.
    pub const COLUMNS: &str = "graph_selection_imputation.columns";
    /// Detail `graph_selection_imputation.data`.
    pub const DATA: &str = "graph_selection_imputation.data";
    /// Detail `graph_selection_imputation.domain`.
    pub const DOMAIN: &str = "graph_selection_imputation.domain";
    /// Detail `graph_selection_imputation.interval_not_measured`.
    pub const INTERVAL_NOT_MEASURED: &str = "graph_selection_imputation.interval_not_measured";
    /// Detail `graph_selection_imputation.model`.
    pub const MODEL: &str = "graph_selection_imputation.model";
    /// Detail `graph_selection_imputation.result`.
    pub const RESULT: &str = "graph_selection_imputation.result";
    /// Detail `graph_selection_imputation.variables`.
    pub const VARIABLES: &str = "graph_selection_imputation.variables";
}

/// Details in the `identification_repair` family.
pub mod identification_repair {
    /// Detail `identification_repair.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "identification_repair.bounds_exceeded";
    /// Detail `identification_repair.budget`.
    pub const BUDGET: &str = "identification_repair.budget";
    /// Detail `identification_repair.cost_units_mismatch`.
    pub const COST_UNITS_MISMATCH: &str = "identification_repair.cost_units_mismatch";
    /// Detail `identification_repair.invalid_request`.
    pub const INVALID_REQUEST: &str = "identification_repair.invalid_request";
    /// Detail `identification_repair.not_a_failure`.
    pub const NOT_A_FAILURE: &str = "identification_repair.not_a_failure";
    /// Detail `identification_repair.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "identification_repair.wrong_contract";
}

/// Details in the `initial_state` family.
pub mod initial_state {
    /// Detail `initial_state.invalid_law`.
    pub const INVALID_LAW: &str = "initial_state.invalid_law";
    /// Detail `initial_state.invalid_premises`.
    pub const INVALID_PREMISES: &str = "initial_state.invalid_premises";
    /// Detail `initial_state.support_gap`.
    pub const SUPPORT_GAP: &str = "initial_state.support_gap";
    /// Detail `initial_state.target_law_missing`.
    pub const TARGET_LAW_MISSING: &str = "initial_state.target_law_missing";
    /// Detail `initial_state.too_many_states`.
    pub const TOO_MANY_STATES: &str = "initial_state.too_many_states";
}

/// Details in the `initial_state_posterior` family.
pub mod initial_state_posterior {
    /// Detail `initial_state_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "initial_state_posterior.band_not_constructed";
    /// Detail `initial_state_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "initial_state_posterior.empty_law";
    /// Detail `initial_state_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "initial_state_posterior.external_power_prior";
    /// Detail `initial_state_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "initial_state_posterior.mapped_prior";
    /// Detail `initial_state_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "initial_state_posterior.non_positive_scale";
}

/// Details in the `interface_contract` family.
pub mod interface_contract {
    /// Detail `interface_contract.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "interface_contract.artifact_invalid";
    /// Detail `interface_contract.blank_field`.
    pub const BLANK_FIELD: &str = "interface_contract.blank_field";
    /// Detail `interface_contract.calibration_record_blank`.
    pub const CALIBRATION_RECORD_BLANK: &str = "interface_contract.calibration_record_blank";
    /// Detail `interface_contract.calibration_without_interval`.
    pub const CALIBRATION_WITHOUT_INTERVAL: &str = "interface_contract.calibration_without_interval";
    /// Detail `interface_contract.conditioning_mismatch`.
    pub const CONDITIONING_MISMATCH: &str = "interface_contract.conditioning_mismatch";
    /// Detail `interface_contract.cyclic_composition`.
    pub const CYCLIC_COMPOSITION: &str = "interface_contract.cyclic_composition";
    /// Detail `interface_contract.dependence_unknown`.
    pub const DEPENDENCE_UNKNOWN: &str = "interface_contract.dependence_unknown";
    /// Detail `interface_contract.duplicate_component`.
    pub const DUPLICATE_COMPONENT: &str = "interface_contract.duplicate_component";
    /// Detail `interface_contract.duplicate_port`.
    pub const DUPLICATE_PORT: &str = "interface_contract.duplicate_port";
    /// Detail `interface_contract.functional_mismatch`.
    pub const FUNCTIONAL_MISMATCH: &str = "interface_contract.functional_mismatch";
    /// Detail `interface_contract.horizon_mismatch`.
    pub const HORIZON_MISMATCH: &str = "interface_contract.horizon_mismatch";
    /// Detail `interface_contract.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "interface_contract.identity_mismatch";
    /// Detail `interface_contract.inherited_input_unknown`.
    pub const INHERITED_INPUT_UNKNOWN: &str = "interface_contract.inherited_input_unknown";
    /// Detail `interface_contract.invalid_digest`.
    pub const INVALID_DIGEST: &str = "interface_contract.invalid_digest";
    /// Detail `interface_contract.invalid_quantity`.
    pub const INVALID_QUANTITY: &str = "interface_contract.invalid_quantity";
    /// Detail `interface_contract.invalid_support`.
    pub const INVALID_SUPPORT: &str = "interface_contract.invalid_support";
    /// Detail `interface_contract.measurement_assumption_mismatch`.
    pub const MEASUREMENT_ASSUMPTION_MISMATCH: &str = "interface_contract.measurement_assumption_mismatch";
    /// Detail `interface_contract.missing_operations`.
    pub const MISSING_OPERATIONS: &str = "interface_contract.missing_operations";
    /// Detail `interface_contract.operation_unavailable`.
    pub const OPERATION_UNAVAILABLE: &str = "interface_contract.operation_unavailable";
    /// Detail `interface_contract.point_only_with_uncertainty`.
    pub const POINT_ONLY_WITH_UNCERTAINTY: &str = "interface_contract.point_only_with_uncertainty";
    /// Detail `interface_contract.population_mismatch`.
    pub const POPULATION_MISMATCH: &str = "interface_contract.population_mismatch";
    /// Detail `interface_contract.port_multiply_fed`.
    pub const PORT_MULTIPLY_FED: &str = "interface_contract.port_multiply_fed";
    /// Detail `interface_contract.posterior_without_posterior_meaning`.
    pub const POSTERIOR_WITHOUT_POSTERIOR_MEANING: &str = "interface_contract.posterior_without_posterior_meaning";
    /// Detail `interface_contract.provenance_missing`.
    pub const PROVENANCE_MISSING: &str = "interface_contract.provenance_missing";
    /// Detail `interface_contract.regime_mismatch`.
    pub const REGIME_MISMATCH: &str = "interface_contract.regime_mismatch";
    /// Detail `interface_contract.semantic_identity_mismatch`.
    pub const SEMANTIC_IDENTITY_MISMATCH: &str = "interface_contract.semantic_identity_mismatch";
    /// Detail `interface_contract.shared_evidence`.
    pub const SHARED_EVIDENCE: &str = "interface_contract.shared_evidence";
    /// Detail `interface_contract.support_mismatch`.
    pub const SUPPORT_MISMATCH: &str = "interface_contract.support_mismatch";
    /// Detail `interface_contract.transform_mismatch`.
    pub const TRANSFORM_MISMATCH: &str = "interface_contract.transform_mismatch";
    /// Detail `interface_contract.uncertainty_meaning_mismatch`.
    pub const UNCERTAINTY_MEANING_MISMATCH: &str = "interface_contract.uncertainty_meaning_mismatch";
    /// Detail `interface_contract.uncertainty_meaning_missing`.
    pub const UNCERTAINTY_MEANING_MISSING: &str = "interface_contract.uncertainty_meaning_missing";
    /// Detail `interface_contract.uncertainty_meaning_required`.
    pub const UNCERTAINTY_MEANING_REQUIRED: &str = "interface_contract.uncertainty_meaning_required";
    /// Detail `interface_contract.uncertainty_meaning_unused`.
    pub const UNCERTAINTY_MEANING_UNUSED: &str = "interface_contract.uncertainty_meaning_unused";
    /// Detail `interface_contract.uncertainty_paradigm_mismatch`.
    pub const UNCERTAINTY_PARADIGM_MISMATCH: &str = "interface_contract.uncertainty_paradigm_mismatch";
    /// Detail `interface_contract.units_mismatch`.
    pub const UNITS_MISMATCH: &str = "interface_contract.units_mismatch";
    /// Detail `interface_contract.unknown_component`.
    pub const UNKNOWN_COMPONENT: &str = "interface_contract.unknown_component";
    /// Detail `interface_contract.unknown_port`.
    pub const UNKNOWN_PORT: &str = "interface_contract.unknown_port";
    /// Detail `interface_contract.unknown_provenance_stage`.
    pub const UNKNOWN_PROVENANCE_STAGE: &str = "interface_contract.unknown_provenance_stage";
    /// Detail `interface_contract.variable_mismatch`.
    pub const VARIABLE_MISMATCH: &str = "interface_contract.variable_mismatch";
}

/// Details in the `intrinsic_influence` family.
pub mod intrinsic_influence {
    /// Detail `intrinsic_influence.correlated_noise`.
    pub const CORRELATED_NOISE: &str = "intrinsic_influence.correlated_noise";
    /// Detail `intrinsic_influence.draw_budget_exceeded`.
    pub const DRAW_BUDGET_EXCEEDED: &str = "intrinsic_influence.draw_budget_exceeded";
    /// Detail `intrinsic_influence.invalid_settings`.
    pub const INVALID_SETTINGS: &str = "intrinsic_influence.invalid_settings";
    /// Detail `intrinsic_influence.mechanism_unfitted`.
    pub const MECHANISM_UNFITTED: &str = "intrinsic_influence.mechanism_unfitted";
    /// Detail `intrinsic_influence.noise_independence_undeclared`.
    pub const NOISE_INDEPENDENCE_UNDECLARED: &str = "intrinsic_influence.noise_independence_undeclared";
    /// Detail `intrinsic_influence.noise_not_row_independent`.
    pub const NOISE_NOT_ROW_INDEPENDENT: &str = "intrinsic_influence.noise_not_row_independent";
    /// Detail `intrinsic_influence.structure_mismatch`.
    pub const STRUCTURE_MISMATCH: &str = "intrinsic_influence.structure_mismatch";
    /// Detail `intrinsic_influence.target_not_continuous`.
    pub const TARGET_NOT_CONTINUOUS: &str = "intrinsic_influence.target_not_continuous";
}

/// Details in the `inverse` family.
pub mod inverse {
    /// Detail `inverse.cancelled`.
    pub const CANCELLED: &str = "inverse.cancelled";
    /// Detail `inverse.forward_not_mean_response`.
    pub const FORWARD_NOT_MEAN_RESPONSE: &str = "inverse.forward_not_mean_response";
    /// Detail `inverse.forward_not_point_identified`.
    pub const FORWARD_NOT_POINT_IDENTIFIED: &str = "inverse.forward_not_point_identified";
    /// Detail `inverse.forward_not_supported`.
    pub const FORWARD_NOT_SUPPORTED: &str = "inverse.forward_not_supported";
    /// Detail `inverse.invalid_action`.
    pub const INVALID_ACTION: &str = "inverse.invalid_action";
    /// Detail `inverse.invalid_forward`.
    pub const INVALID_FORWARD: &str = "inverse.invalid_forward";
    /// Detail `inverse.invalid_target`.
    pub const INVALID_TARGET: &str = "inverse.invalid_target";
    /// Detail `inverse.observational_scenarios`.
    pub const OBSERVATIONAL_SCENARIOS: &str = "inverse.observational_scenarios";
    /// Detail `inverse.probability_target`.
    pub const PROBABILITY_TARGET: &str = "inverse.probability_target";
    /// Detail `inverse.quantile_target`.
    pub const QUANTILE_TARGET: &str = "inverse.quantile_target";
}

/// Details in the `inverse_forward_source` family.
pub mod inverse_forward_source {
    /// Detail `inverse_forward_source.composed_coordinate_is_outcome_law`.
    pub const COMPOSED_COORDINATE_IS_OUTCOME_LAW: &str = "inverse_forward_source.composed_coordinate_is_outcome_law";
    /// Detail `inverse_forward_source.composed_family_mismatch`.
    pub const COMPOSED_FAMILY_MISMATCH: &str = "inverse_forward_source.composed_family_mismatch";
    /// Detail `inverse_forward_source.composed_law_not_materialized`.
    pub const COMPOSED_LAW_NOT_MATERIALIZED: &str = "inverse_forward_source.composed_law_not_materialized";
    /// Detail `inverse_forward_source.composed_level_coding`.
    pub const COMPOSED_LEVEL_CODING: &str = "inverse_forward_source.composed_level_coding";
    /// Detail `inverse_forward_source.composed_query_result_mismatch`.
    pub const COMPOSED_QUERY_RESULT_MISMATCH: &str = "inverse_forward_source.composed_query_result_mismatch";
    /// Detail `inverse_forward_source.composed_response_invalid`.
    pub const COMPOSED_RESPONSE_INVALID: &str = "inverse_forward_source.composed_response_invalid";
    /// Detail `inverse_forward_source.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "inverse_forward_source.interval_not_supported";
    /// Detail `inverse_forward_source.origin_invalid`.
    pub const ORIGIN_INVALID: &str = "inverse_forward_source.origin_invalid";
    /// Detail `inverse_forward_source.per_action_not_sealable`.
    pub const PER_ACTION_NOT_SEALABLE: &str = "inverse_forward_source.per_action_not_sealable";
    /// Detail `inverse_forward_source.point_only_law_requested`.
    pub const POINT_ONLY_LAW_REQUESTED: &str = "inverse_forward_source.point_only_law_requested";
    /// Detail `inverse_forward_source.transported_not_accepted`.
    pub const TRANSPORTED_NOT_ACCEPTED: &str = "inverse_forward_source.transported_not_accepted";
}

/// Details in the `inverse_observational_scenarios` family.
pub mod inverse_observational_scenarios {
    /// Detail `inverse_observational_scenarios.condition_on_outcome`.
    pub const CONDITION_ON_OUTCOME: &str = "inverse_observational_scenarios.condition_on_outcome";
    /// Detail `inverse_observational_scenarios.coordinate_missing`.
    pub const COORDINATE_MISSING: &str = "inverse_observational_scenarios.coordinate_missing";
    /// Detail `inverse_observational_scenarios.coordinate_unsupported`.
    pub const COORDINATE_UNSUPPORTED: &str = "inverse_observational_scenarios.coordinate_unsupported";
    /// Detail `inverse_observational_scenarios.event_has_no_mass`.
    pub const EVENT_HAS_NO_MASS: &str = "inverse_observational_scenarios.event_has_no_mass";
    /// Detail `inverse_observational_scenarios.invalid_parameter`.
    pub const INVALID_PARAMETER: &str = "inverse_observational_scenarios.invalid_parameter";
    /// Detail `inverse_observational_scenarios.invalid_scenarios`.
    pub const INVALID_SCENARIOS: &str = "inverse_observational_scenarios.invalid_scenarios";
    /// Detail `inverse_observational_scenarios.joint_law_required`.
    pub const JOINT_LAW_REQUIRED: &str = "inverse_observational_scenarios.joint_law_required";
    /// Detail `inverse_observational_scenarios.meaning_not_observational`.
    pub const MEANING_NOT_OBSERVATIONAL: &str = "inverse_observational_scenarios.meaning_not_observational";
    /// Detail `inverse_observational_scenarios.non_finite_draw`.
    pub const NON_FINITE_DRAW: &str = "inverse_observational_scenarios.non_finite_draw";
    /// Detail `inverse_observational_scenarios.result_mismatch`.
    pub const RESULT_MISMATCH: &str = "inverse_observational_scenarios.result_mismatch";
    /// Detail `inverse_observational_scenarios.unknown_scenario`.
    pub const UNKNOWN_SCENARIO: &str = "inverse_observational_scenarios.unknown_scenario";
}

/// Details in the `joint_cells` family.
pub mod joint_cells {
    /// Detail `joint_cells.cancelled`.
    pub const CANCELLED: &str = "joint_cells.cancelled";
    /// Detail `joint_cells.cell_empty`.
    pub const CELL_EMPTY: &str = "joint_cells.cell_empty";
    /// Detail `joint_cells.cell_ess_below_minimum`.
    pub const CELL_ESS_BELOW_MINIMUM: &str = "joint_cells.cell_ess_below_minimum";
    /// Detail `joint_cells.columns`.
    pub const COLUMNS: &str = "joint_cells.columns";
    /// Detail `joint_cells.component_count`.
    pub const COMPONENT_COUNT: &str = "joint_cells.component_count";
    /// Detail `joint_cells.conditional_unsupported`.
    pub const CONDITIONAL_UNSUPPORTED: &str = "joint_cells.conditional_unsupported";
    /// Detail `joint_cells.config`.
    pub const CONFIG: &str = "joint_cells.config";
    /// Detail `joint_cells.contrast_cell_unsupported`.
    pub const CONTRAST_CELL_UNSUPPORTED: &str = "joint_cells.contrast_cell_unsupported";
    /// Detail `joint_cells.derived_constituent_copy_in_adjustment`.
    pub const DERIVED_CONSTITUENT_COPY_IN_ADJUSTMENT: &str = "joint_cells.derived_constituent_copy_in_adjustment";
    /// Detail `joint_cells.derived_constituent_in_adjustment`.
    pub const DERIVED_CONSTITUENT_IN_ADJUSTMENT: &str = "joint_cells.derived_constituent_in_adjustment";
    /// Detail `joint_cells.derived_declaration_invalid`.
    pub const DERIVED_DECLARATION_INVALID: &str = "joint_cells.derived_declaration_invalid";
    /// Detail `joint_cells.derived_descendant_in_adjustment`.
    pub const DERIVED_DESCENDANT_IN_ADJUSTMENT: &str = "joint_cells.derived_descendant_in_adjustment";
    /// Detail `joint_cells.derived_duplicate_constituents`.
    pub const DERIVED_DUPLICATE_CONSTITUENTS: &str = "joint_cells.derived_duplicate_constituents";
    /// Detail `joint_cells.derived_exclusion_not_in_adjustment`.
    pub const DERIVED_EXCLUSION_NOT_IN_ADJUSTMENT: &str = "joint_cells.derived_exclusion_not_in_adjustment";
    /// Detail `joint_cells.derived_exclusion_rule_mismatch`.
    pub const DERIVED_EXCLUSION_RULE_MISMATCH: &str = "joint_cells.derived_exclusion_rule_mismatch";
    /// Detail `joint_cells.derived_illegal_value`.
    pub const DERIVED_ILLEGAL_VALUE: &str = "joint_cells.derived_illegal_value";
    /// Detail `joint_cells.derived_intervention_not_defined`.
    pub const DERIVED_INTERVENTION_NOT_DEFINED: &str = "joint_cells.derived_intervention_not_defined";
    /// Detail `joint_cells.derived_not_a_joint_cell`.
    pub const DERIVED_NOT_A_JOINT_CELL: &str = "joint_cells.derived_not_a_joint_cell";
    /// Detail `joint_cells.derived_outcome_in_adjustment`.
    pub const DERIVED_OUTCOME_IN_ADJUSTMENT: &str = "joint_cells.derived_outcome_in_adjustment";
    /// Detail `joint_cells.derived_rank_deficient`.
    pub const DERIVED_RANK_DEFICIENT: &str = "joint_cells.derived_rank_deficient";
    /// Detail `joint_cells.derived_treatment_tracked_by_covariates`.
    pub const DERIVED_TREATMENT_TRACKED_BY_COVARIATES: &str = "joint_cells.derived_treatment_tracked_by_covariates";
    /// Detail `joint_cells.derived_unknown_column`.
    pub const DERIVED_UNKNOWN_COLUMN: &str = "joint_cells.derived_unknown_column";
    /// Detail `joint_cells.learner`.
    pub const LEARNER: &str = "joint_cells.learner";
    /// Detail `joint_cells.learner_unavailable`.
    pub const LEARNER_UNAVAILABLE: &str = "joint_cells.learner_unavailable";
    /// Detail `joint_cells.ml_learner_not_declared`.
    pub const ML_LEARNER_NOT_DECLARED: &str = "joint_cells.ml_learner_not_declared";
    /// Detail `joint_cells.no_interval_method`.
    pub const NO_INTERVAL_METHOD: &str = "joint_cells.no_interval_method";
    /// Detail `joint_cells.no_ml_interval_method`.
    pub const NO_ML_INTERVAL_METHOD: &str = "joint_cells.no_ml_interval_method";
    /// Detail `joint_cells.no_rows`.
    pub const NO_ROWS: &str = "joint_cells.no_rows";
    /// Detail `joint_cells.no_supported_cell`.
    pub const NO_SUPPORTED_CELL: &str = "joint_cells.no_supported_cell";
    /// Detail `joint_cells.normalization_failed`.
    pub const NORMALIZATION_FAILED: &str = "joint_cells.normalization_failed";
    /// Detail `joint_cells.not_binary`.
    pub const NOT_BINARY: &str = "joint_cells.not_binary";
    /// Detail `joint_cells.nuisance_provider`.
    pub const NUISANCE_PROVIDER: &str = "joint_cells.nuisance_provider";
    /// Detail `joint_cells.ordering`.
    pub const ORDERING: &str = "joint_cells.ordering";
    /// Detail `joint_cells.outcome_model_unsupported`.
    pub const OUTCOME_MODEL_UNSUPPORTED: &str = "joint_cells.outcome_model_unsupported";
}

/// Details in the `joint_law_propagation` family.
pub mod joint_law_propagation {
    /// Detail `joint_law_propagation.alignment_mismatch`.
    pub const ALIGNMENT_MISMATCH: &str = "joint_law_propagation.alignment_mismatch";
    /// Detail `joint_law_propagation.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "joint_law_propagation.artifact_invalid";
    /// Detail `joint_law_propagation.artifact_refused`.
    pub const ARTIFACT_REFUSED: &str = "joint_law_propagation.artifact_refused";
    /// Detail `joint_law_propagation.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "joint_law_propagation.bounds_exceeded";
    /// Detail `joint_law_propagation.cancelled`.
    pub const CANCELLED: &str = "joint_law_propagation.cancelled";
    /// Detail `joint_law_propagation.covariance_insufficient_for_nonlinear`.
    pub const COVARIANCE_INSUFFICIENT_FOR_NONLINEAR: &str = "joint_law_propagation.covariance_insufficient_for_nonlinear";
    /// Detail `joint_law_propagation.coverage_record_construction_mismatch`.
    pub const COVERAGE_RECORD_CONSTRUCTION_MISMATCH: &str = "joint_law_propagation.coverage_record_construction_mismatch";
    /// Detail `joint_law_propagation.coverage_record_coordinate_mismatch`.
    pub const COVERAGE_RECORD_COORDINATE_MISMATCH: &str = "joint_law_propagation.coverage_record_coordinate_mismatch";
    /// Detail `joint_law_propagation.dependence_unknown`.
    pub const DEPENDENCE_UNKNOWN: &str = "joint_law_propagation.dependence_unknown";
    /// Detail `joint_law_propagation.envelope_is_not_a_point`.
    pub const ENVELOPE_IS_NOT_A_POINT: &str = "joint_law_propagation.envelope_is_not_a_point";
    /// Detail `joint_law_propagation.envelope_not_available`.
    pub const ENVELOPE_NOT_AVAILABLE: &str = "joint_law_propagation.envelope_not_available";
    /// Detail `joint_law_propagation.evidence_overlap_unknown`.
    pub const EVIDENCE_OVERLAP_UNKNOWN: &str = "joint_law_propagation.evidence_overlap_unknown";
    /// Detail `joint_law_propagation.horizon_composition`.
    pub const HORIZON_COMPOSITION: &str = "joint_law_propagation.horizon_composition";
    /// Detail `joint_law_propagation.horizon_draws_dose`.
    pub const HORIZON_DRAWS_DOSE: &str = "joint_law_propagation.horizon_draws_dose";
    /// Detail `joint_law_propagation.horizon_draws_layout`.
    pub const HORIZON_DRAWS_LAYOUT: &str = "joint_law_propagation.horizon_draws_layout";
    /// Detail `joint_law_propagation.horizon_draws_mean_curve`.
    pub const HORIZON_DRAWS_MEAN_CURVE: &str = "joint_law_propagation.horizon_draws_mean_curve";
    /// Detail `joint_law_propagation.horizon_draws_not_joint`.
    pub const HORIZON_DRAWS_NOT_JOINT: &str = "joint_law_propagation.horizon_draws_not_joint";
    /// Detail `joint_law_propagation.index_pairing_refused`.
    pub const INDEX_PAIRING_REFUSED: &str = "joint_law_propagation.index_pairing_refused";
    /// Detail `joint_law_propagation.invalid_request`.
    pub const INVALID_REQUEST: &str = "joint_law_propagation.invalid_request";
    /// Detail `joint_law_propagation.mixture_weights_unknown`.
    pub const MIXTURE_WEIGHTS_UNKNOWN: &str = "joint_law_propagation.mixture_weights_unknown";
    /// Detail `joint_law_propagation.no_interval_construction`.
    pub const NO_INTERVAL_CONSTRUCTION: &str = "joint_law_propagation.no_interval_construction";
    /// Detail `joint_law_propagation.no_posterior_for_fixed_law_inputs`.
    pub const NO_POSTERIOR_FOR_FIXED_LAW_INPUTS: &str = "joint_law_propagation.no_posterior_for_fixed_law_inputs";
    /// Detail `joint_law_propagation.nonlinear_needs_aligned_joint_law`.
    pub const NONLINEAR_NEEDS_ALIGNED_JOINT_LAW: &str = "joint_law_propagation.nonlinear_needs_aligned_joint_law";
    /// Detail `joint_law_propagation.shared_evidence_declared_independent`.
    pub const SHARED_EVIDENCE_DECLARED_INDEPENDENT: &str = "joint_law_propagation.shared_evidence_declared_independent";
    /// Detail `joint_law_propagation.spread_needs_supported_dependence`.
    pub const SPREAD_NEEDS_SUPPORTED_DEPENDENCE: &str = "joint_law_propagation.spread_needs_supported_dependence";
    /// Detail `joint_law_propagation.structure_not_verified`.
    pub const STRUCTURE_NOT_VERIFIED: &str = "joint_law_propagation.structure_not_verified";
    /// Detail `joint_law_propagation.transport_pieces_incomplete`.
    pub const TRANSPORT_PIECES_INCOMPLETE: &str = "joint_law_propagation.transport_pieces_incomplete";
    /// Detail `joint_law_propagation.transported_dependence_required`.
    pub const TRANSPORTED_DEPENDENCE_REQUIRED: &str = "joint_law_propagation.transported_dependence_required";
}

/// Details in the `joint_sensitivity` family.
pub mod joint_sensitivity {
    /// Detail `joint_sensitivity.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "joint_sensitivity.bounds_exceeded";
    /// Detail `joint_sensitivity.budget`.
    pub const BUDGET: &str = "joint_sensitivity.budget";
    /// Detail `joint_sensitivity.budget_coupling`.
    pub const BUDGET_COUPLING: &str = "joint_sensitivity.budget_coupling";
    /// Detail `joint_sensitivity.cancelled`.
    pub const CANCELLED: &str = "joint_sensitivity.cancelled";
    /// Detail `joint_sensitivity.consumer_limits`.
    pub const CONSUMER_LIMITS: &str = "joint_sensitivity.consumer_limits";
    /// Detail `joint_sensitivity.data_identity_mismatch`.
    pub const DATA_IDENTITY_MISMATCH: &str = "joint_sensitivity.data_identity_mismatch";
    /// Detail `joint_sensitivity.duplicate_factor`.
    pub const DUPLICATE_FACTOR: &str = "joint_sensitivity.duplicate_factor";
    /// Detail `joint_sensitivity.factor_count`.
    pub const FACTOR_COUNT: &str = "joint_sensitivity.factor_count";
    /// Detail `joint_sensitivity.fixed_graph_conditional`.
    pub const FIXED_GRAPH_CONDITIONAL: &str = "joint_sensitivity.fixed_graph_conditional";
    /// Detail `joint_sensitivity.fixed_graph_parent`.
    pub const FIXED_GRAPH_PARENT: &str = "joint_sensitivity.fixed_graph_parent";
    /// Detail `joint_sensitivity.formula_incompatible`.
    pub const FORMULA_INCOMPATIBLE: &str = "joint_sensitivity.formula_incompatible";
    /// Detail `joint_sensitivity.incomplete_kernel`.
    pub const INCOMPLETE_KERNEL: &str = "joint_sensitivity.incomplete_kernel";
    /// Detail `joint_sensitivity.invalid_fraction`.
    pub const INVALID_FRACTION: &str = "joint_sensitivity.invalid_fraction";
    /// Detail `joint_sensitivity.invalid_threshold`.
    pub const INVALID_THRESHOLD: &str = "joint_sensitivity.invalid_threshold";
    /// Detail `joint_sensitivity.invalid_tolerance`.
    pub const INVALID_TOLERANCE: &str = "joint_sensitivity.invalid_tolerance";
    /// Detail `joint_sensitivity.no_interval_method`.
    pub const NO_INTERVAL_METHOD: &str = "joint_sensitivity.no_interval_method";
    /// Detail `joint_sensitivity.premises_mismatch`.
    pub const PREMISES_MISMATCH: &str = "joint_sensitivity.premises_mismatch";
    /// Detail `joint_sensitivity.provider_mismatch`.
    pub const PROVIDER_MISMATCH: &str = "joint_sensitivity.provider_mismatch";
    /// Detail `joint_sensitivity.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "joint_sensitivity.replay_mismatch";
    /// Detail `joint_sensitivity.source_target_discrepancy`.
    pub const SOURCE_TARGET_DISCREPANCY: &str = "joint_sensitivity.source_target_discrepancy";
    /// Detail `joint_sensitivity.stage_accounting`.
    pub const STAGE_ACCOUNTING: &str = "joint_sensitivity.stage_accounting";
    /// Detail `joint_sensitivity.treatment_factor`.
    pub const TREATMENT_FACTOR: &str = "joint_sensitivity.treatment_factor";
    /// Detail `joint_sensitivity.union_labelled_ci`.
    pub const UNION_LABELLED_CI: &str = "joint_sensitivity.union_labelled_ci";
    /// Detail `joint_sensitivity.unknown_factor`.
    pub const UNKNOWN_FACTOR: &str = "joint_sensitivity.unknown_factor";
    /// Detail `joint_sensitivity.unsupported_domain`.
    pub const UNSUPPORTED_DOMAIN: &str = "joint_sensitivity.unsupported_domain";
}

/// Details in the `latent_class` family.
pub mod latent_class {
    /// Detail `latent_class.cancelled`.
    pub const CANCELLED: &str = "latent_class.cancelled";
    /// Detail `latent_class.constant_outcome`.
    pub const CONSTANT_OUTCOME: &str = "latent_class.constant_outcome";
    /// Detail `latent_class.data_required`.
    pub const DATA_REQUIRED: &str = "latent_class.data_required";
    /// Detail `latent_class.degenerate_class`.
    pub const DEGENERATE_CLASS: &str = "latent_class.degenerate_class";
    /// Detail `latent_class.design_support_too_small`.
    pub const DESIGN_SUPPORT_TOO_SMALL: &str = "latent_class.design_support_too_small";
    /// Detail `latent_class.export_budget_exceeded`.
    pub const EXPORT_BUDGET_EXCEEDED: &str = "latent_class.export_budget_exceeded";
    /// Detail `latent_class.invalid_config`.
    pub const INVALID_CONFIG: &str = "latent_class.invalid_config";
    /// Detail `latent_class.invalid_request`.
    pub const INVALID_REQUEST: &str = "latent_class.invalid_request";
    /// Detail `latent_class.length_mismatch`.
    pub const LENGTH_MISMATCH: &str = "latent_class.length_mismatch";
    /// Detail `latent_class.likelihood_not_monotone`.
    pub const LIKELIHOOD_NOT_MONOTONE: &str = "latent_class.likelihood_not_monotone";
    /// Detail `latent_class.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "latent_class.limits_exceeded";
    /// Detail `latent_class.memory_budget_exceeded`.
    pub const MEMORY_BUDGET_EXCEEDED: &str = "latent_class.memory_budget_exceeded";
    /// Detail `latent_class.non_finite_input`.
    pub const NON_FINITE_INPUT: &str = "latent_class.non_finite_input";
    /// Detail `latent_class.not_converged`.
    pub const NOT_CONVERGED: &str = "latent_class.not_converged";
    /// Detail `latent_class.randomization_not_declared`.
    pub const RANDOMIZATION_NOT_DECLARED: &str = "latent_class.randomization_not_declared";
    /// Detail `latent_class.report_replay_mismatch`.
    pub const REPORT_REPLAY_MISMATCH: &str = "latent_class.report_replay_mismatch";
    /// Detail `latent_class.too_few_rows`.
    pub const TOO_FEW_ROWS: &str = "latent_class.too_few_rows";
    /// Detail `latent_class.treatment_constant`.
    pub const TREATMENT_CONSTANT: &str = "latent_class.treatment_constant";
    /// Detail `latent_class.weak_class`.
    pub const WEAK_CLASS: &str = "latent_class.weak_class";
    /// Detail `latent_class.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "latent_class.wrong_contract";
}

/// Details in the `latent_class_posterior` family.
pub mod latent_class_posterior {
    /// Detail `latent_class_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "latent_class_posterior.band_not_constructed";
    /// Detail `latent_class_posterior.empty_design`.
    pub const EMPTY_DESIGN: &str = "latent_class_posterior.empty_design";
    /// Detail `latent_class_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "latent_class_posterior.external_power_prior";
    /// Detail `latent_class_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "latent_class_posterior.mapped_prior";
    /// Detail `latent_class_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "latent_class_posterior.non_positive_scale";
}

/// Details in the `learned_continuous_posterior` family.
pub mod learned_continuous_posterior {
    /// Detail `learned_continuous_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "learned_continuous_posterior.band_not_constructed";
    /// Detail `learned_continuous_posterior.empty_scores`.
    pub const EMPTY_SCORES: &str = "learned_continuous_posterior.empty_scores";
    /// Detail `learned_continuous_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "learned_continuous_posterior.external_power_prior";
    /// Detail `learned_continuous_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "learned_continuous_posterior.mapped_prior";
    /// Detail `learned_continuous_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "learned_continuous_posterior.non_positive_scale";
}

/// Details in the `learned_joint_transport` family.
pub mod learned_joint_transport {
    /// Detail `learned_joint_transport.admission_refused`.
    pub const ADMISSION_REFUSED: &str = "learned_joint_transport.admission_refused";
    /// Detail `learned_joint_transport.cancelled`.
    pub const CANCELLED: &str = "learned_joint_transport.cancelled";
    /// Detail `learned_joint_transport.feature_mismatch`.
    pub const FEATURE_MISMATCH: &str = "learned_joint_transport.feature_mismatch";
    /// Detail `learned_joint_transport.fit_failed`.
    pub const FIT_FAILED: &str = "learned_joint_transport.fit_failed";
    /// Detail `learned_joint_transport.identification_not_certified`.
    pub const IDENTIFICATION_NOT_CERTIFIED: &str = "learned_joint_transport.identification_not_certified";
    /// Detail `learned_joint_transport.identification_required`.
    pub const IDENTIFICATION_REQUIRED: &str = "learned_joint_transport.identification_required";
    /// Detail `learned_joint_transport.invalid_basis`.
    pub const INVALID_BASIS: &str = "learned_joint_transport.invalid_basis";
    /// Detail `learned_joint_transport.invalid_input`.
    pub const INVALID_INPUT: &str = "learned_joint_transport.invalid_input";
    /// Detail `learned_joint_transport.invalid_prior`.
    pub const INVALID_PRIOR: &str = "learned_joint_transport.invalid_prior";
    /// Detail `learned_joint_transport.missing_law`.
    pub const MISSING_LAW: &str = "learned_joint_transport.missing_law";
    /// Detail `learned_joint_transport.missing_source`.
    pub const MISSING_SOURCE: &str = "learned_joint_transport.missing_source";
    /// Detail `learned_joint_transport.prior_data_overlap`.
    pub const PRIOR_DATA_OVERLAP: &str = "learned_joint_transport.prior_data_overlap";
    /// Detail `learned_joint_transport.prior_dimension`.
    pub const PRIOR_DIMENSION: &str = "learned_joint_transport.prior_dimension";
    /// Detail `learned_joint_transport.rank_deficient`.
    pub const RANK_DEFICIENT: &str = "learned_joint_transport.rank_deficient";
    /// Detail `learned_joint_transport.source_dependence`.
    pub const SOURCE_DEPENDENCE: &str = "learned_joint_transport.source_dependence";
    /// Detail `learned_joint_transport.support_resource_limit`.
    pub const SUPPORT_RESOURCE_LIMIT: &str = "learned_joint_transport.support_resource_limit";
    /// Detail `learned_joint_transport.too_many_draws`.
    pub const TOO_MANY_DRAWS: &str = "learned_joint_transport.too_many_draws";
    /// Detail `learned_joint_transport.too_many_parameters`.
    pub const TOO_MANY_PARAMETERS: &str = "learned_joint_transport.too_many_parameters";
    /// Detail `learned_joint_transport.unsupported_graph`.
    pub const UNSUPPORTED_GRAPH: &str = "learned_joint_transport.unsupported_graph";
    /// Detail `learned_joint_transport.weak_overlap`.
    pub const WEAK_OVERLAP: &str = "learned_joint_transport.weak_overlap";
}

/// Details in the `learned_transport` family.
pub mod learned_transport {
    /// Detail `learned_transport.bootstrap_below_floor`.
    pub const BOOTSTRAP_BELOW_FLOOR: &str = "learned_transport.bootstrap_below_floor";
    /// Detail `learned_transport.bootstrap_not_supported`.
    pub const BOOTSTRAP_NOT_SUPPORTED: &str = "learned_transport.bootstrap_not_supported";
    /// Detail `learned_transport.cate_requested`.
    pub const CATE_REQUESTED: &str = "learned_transport.cate_requested";
    /// Detail `learned_transport.membership_overlap`.
    pub const MEMBERSHIP_OVERLAP: &str = "learned_transport.membership_overlap";
    /// Detail `learned_transport.non_iid_design`.
    pub const NON_IID_DESIGN: &str = "learned_transport.non_iid_design";
    /// Detail `learned_transport.treatment_overlap`.
    pub const TREATMENT_OVERLAP: &str = "learned_transport.treatment_overlap";
}

/// Details in the `mapped_posterior_transfer` family.
pub mod mapped_posterior_transfer {
    /// Detail `mapped_posterior_transfer.coefficient_map`.
    pub const COEFFICIENT_MAP: &str = "mapped_posterior_transfer.coefficient_map";
    /// Detail `mapped_posterior_transfer.parameter_posterior`.
    pub const PARAMETER_POSTERIOR: &str = "mapped_posterior_transfer.parameter_posterior";
    /// Detail `mapped_posterior_transfer.residual_variance`.
    pub const RESIDUAL_VARIANCE: &str = "mapped_posterior_transfer.residual_variance";
}

/// Details in the `matched_case_control` family.
pub mod matched_case_control {
    /// Detail `matched_case_control.absolute_risk_not_identified`.
    pub const ABSOLUTE_RISK_NOT_IDENTIFIED: &str = "matched_case_control.absolute_risk_not_identified";
    /// Detail `matched_case_control.budget`.
    pub const BUDGET: &str = "matched_case_control.budget";
    /// Detail `matched_case_control.estimand_not_supported`.
    pub const ESTIMAND_NOT_SUPPORTED: &str = "matched_case_control.estimand_not_supported";
    /// Detail `matched_case_control.estimate_not_finite`.
    pub const ESTIMATE_NOT_FINITE: &str = "matched_case_control.estimate_not_finite";
    /// Detail `matched_case_control.invalid_data`.
    pub const INVALID_DATA: &str = "matched_case_control.invalid_data";
    /// Detail `matched_case_control.invalid_level`.
    pub const INVALID_LEVEL: &str = "matched_case_control.invalid_level";
    /// Detail `matched_case_control.no_informative_sets`.
    pub const NO_INFORMATIVE_SETS: &str = "matched_case_control.no_informative_sets";
    /// Detail `matched_case_control.sampling_design`.
    pub const SAMPLING_DESIGN: &str = "matched_case_control.sampling_design";
}

/// Details in the `measured_inference` family.
pub mod measured_inference {
    /// Detail `measured_inference.artifact_mismatch`.
    pub const ARTIFACT_MISMATCH: &str = "measured_inference.artifact_mismatch";
    /// Detail `measured_inference.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "measured_inference.bounds_exceeded";
    /// Detail `measured_inference.consumer_bounds_exceeded`.
    pub const CONSUMER_BOUNDS_EXCEEDED: &str = "measured_inference.consumer_bounds_exceeded";
    /// Detail `measured_inference.invalid_expectation`.
    pub const INVALID_EXPECTATION: &str = "measured_inference.invalid_expectation";
    /// Detail `measured_inference.invalid_report`.
    pub const INVALID_REPORT: &str = "measured_inference.invalid_report";
    /// Detail `measured_inference.memory_budget_exceeded`.
    pub const MEMORY_BUDGET_EXCEEDED: &str = "measured_inference.memory_budget_exceeded";
    /// Detail `measured_inference.source_report_not_supported`.
    pub const SOURCE_REPORT_NOT_SUPPORTED: &str = "measured_inference.source_report_not_supported";
    /// Detail `measured_inference.unsupported_route`.
    pub const UNSUPPORTED_ROUTE: &str = "measured_inference.unsupported_route";
}

/// Details in the `measured_interval` family.
pub mod measured_interval {
    /// Detail `measured_interval.draws_not_converged`.
    pub const DRAWS_NOT_CONVERGED: &str = "measured_interval.draws_not_converged";
    /// Detail `measured_interval.invalid_source`.
    pub const INVALID_SOURCE: &str = "measured_interval.invalid_source";
    /// Detail `measured_interval.no_interval_of_this_kind`.
    pub const NO_INTERVAL_OF_THIS_KIND: &str = "measured_interval.no_interval_of_this_kind";
    /// Detail `measured_interval.source_mismatch`.
    pub const SOURCE_MISMATCH: &str = "measured_interval.source_mismatch";
    /// Detail `measured_interval.variance_assumption_not_met`.
    pub const VARIANCE_ASSUMPTION_NOT_MET: &str = "measured_interval.variance_assumption_not_met";
}

/// Details in the `measured_transport` family.
pub mod measured_transport {
    /// Detail `measured_transport.input_bounds_exceeded`.
    pub const INPUT_BOUNDS_EXCEEDED: &str = "measured_transport.input_bounds_exceeded";
    /// Detail `measured_transport.sampler_mismatch`.
    pub const SAMPLER_MISMATCH: &str = "measured_transport.sampler_mismatch";
}

/// Details in the `measurement_model` family.
pub mod measurement_model {
    /// Detail `measurement_model.admission_refused`.
    pub const ADMISSION_REFUSED: &str = "measurement_model.admission_refused";
    /// Detail `measurement_model.arm_mismatch`.
    pub const ARM_MISMATCH: &str = "measurement_model.arm_mismatch";
    /// Detail `measurement_model.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "measurement_model.artifact_invalid";
    /// Detail `measurement_model.artifact_refused`.
    pub const ARTIFACT_REFUSED: &str = "measurement_model.artifact_refused";
    /// Detail `measurement_model.binary_outcome_continuous_error`.
    pub const BINARY_OUTCOME_CONTINUOUS_ERROR: &str = "measurement_model.binary_outcome_continuous_error";
    /// Detail `measurement_model.blank_field`.
    pub const BLANK_FIELD: &str = "measurement_model.blank_field";
    /// Detail `measurement_model.bootstrap_unstable`.
    pub const BOOTSTRAP_UNSTABLE: &str = "measurement_model.bootstrap_unstable";
    /// Detail `measurement_model.cancelled`.
    pub const CANCELLED: &str = "measurement_model.cancelled";
    /// Detail `measurement_model.confusion_ill_conditioned`.
    pub const CONFUSION_ILL_CONDITIONED: &str = "measurement_model.confusion_ill_conditioned";
    /// Detail `measurement_model.construct_mismatch`.
    pub const CONSTRUCT_MISMATCH: &str = "measurement_model.construct_mismatch";
    /// Detail `measurement_model.continuous_export`.
    pub const CONTINUOUS_EXPORT: &str = "measurement_model.continuous_export";
    /// Detail `measurement_model.design_singular`.
    pub const DESIGN_SINGULAR: &str = "measurement_model.design_singular";
    /// Detail `measurement_model.differential_axis_unsupported`.
    pub const DIFFERENTIAL_AXIS_UNSUPPORTED: &str = "measurement_model.differential_axis_unsupported";
    /// Detail `measurement_model.differential_declared`.
    pub const DIFFERENTIAL_DECLARED: &str = "measurement_model.differential_declared";
    /// Detail `measurement_model.differential_population_unestablished`.
    pub const DIFFERENTIAL_POPULATION_UNESTABLISHED: &str = "measurement_model.differential_population_unestablished";
    /// Detail `measurement_model.differential_undeclared`.
    pub const DIFFERENTIAL_UNDECLARED: &str = "measurement_model.differential_undeclared";
    /// Detail `measurement_model.error_variance_not_identified`.
    pub const ERROR_VARIANCE_NOT_IDENTIFIED: &str = "measurement_model.error_variance_not_identified";
    /// Detail `measurement_model.indicator_as_construct`.
    pub const INDICATOR_AS_CONSTRUCT: &str = "measurement_model.indicator_as_construct";
    /// Detail `measurement_model.indicator_kind`.
    pub const INDICATOR_KIND: &str = "measurement_model.indicator_kind";
    /// Detail `measurement_model.indicator_mismatch`.
    pub const INDICATOR_MISMATCH: &str = "measurement_model.indicator_mismatch";
    /// Detail `measurement_model.indicator_observed`.
    pub const INDICATOR_OBSERVED: &str = "measurement_model.indicator_observed";
    /// Detail `measurement_model.input_invalid`.
    pub const INPUT_INVALID: &str = "measurement_model.input_invalid";
    /// Detail `measurement_model.insufficient_draws`.
    pub const INSUFFICIENT_DRAWS: &str = "measurement_model.insufficient_draws";
    /// Detail `measurement_model.insufficient_replicates`.
    pub const INSUFFICIENT_REPLICATES: &str = "measurement_model.insufficient_replicates";
    /// Detail `measurement_model.interval_level`.
    pub const INTERVAL_LEVEL: &str = "measurement_model.interval_level";
    /// Detail `measurement_model.interval_requires_estimate`.
    pub const INTERVAL_REQUIRES_ESTIMATE: &str = "measurement_model.interval_requires_estimate";
    /// Detail `measurement_model.intervention_meaning_not_carried`.
    pub const INTERVENTION_MEANING_NOT_CARRIED: &str = "measurement_model.intervention_meaning_not_carried";
    /// Detail `measurement_model.invalid_model`.
    pub const INVALID_MODEL: &str = "measurement_model.invalid_model";
    /// Detail `measurement_model.invariance_not_declared`.
    pub const INVARIANCE_NOT_DECLARED: &str = "measurement_model.invariance_not_declared";
    /// Detail `measurement_model.invariance_premise_required`.
    pub const INVARIANCE_PREMISE_REQUIRED: &str = "measurement_model.invariance_premise_required";
    /// Detail `measurement_model.invariance_source_unknown`.
    pub const INVARIANCE_SOURCE_UNKNOWN: &str = "measurement_model.invariance_source_unknown";
    /// Detail `measurement_model.invariance_target_mismatch`.
    pub const INVARIANCE_TARGET_MISMATCH: &str = "measurement_model.invariance_target_mismatch";
    /// Detail `measurement_model.learned_coordinates_not_intervenable`.
    pub const LEARNED_COORDINATES_NOT_INTERVENABLE: &str = "measurement_model.learned_coordinates_not_intervenable";
    /// Detail `measurement_model.mapping_endpoints`.
    pub const MAPPING_ENDPOINTS: &str = "measurement_model.mapping_endpoints";
    /// Detail `measurement_model.no_model`.
    pub const NO_MODEL: &str = "measurement_model.no_model";
    /// Detail `measurement_model.nothing_to_transport`.
    pub const NOTHING_TO_TRANSPORT: &str = "measurement_model.nothing_to_transport";
    /// Detail `measurement_model.operation_not_supported`.
    pub const OPERATION_NOT_SUPPORTED: &str = "measurement_model.operation_not_supported";
    /// Detail `measurement_model.outcome_model_not_fitted`.
    pub const OUTCOME_MODEL_NOT_FITTED: &str = "measurement_model.outcome_model_not_fitted";
    /// Detail `measurement_model.parameter`.
    pub const PARAMETER: &str = "measurement_model.parameter";
    /// Detail `measurement_model.population_mismatch`.
    pub const POPULATION_MISMATCH: &str = "measurement_model.population_mismatch";
    /// Detail `measurement_model.posterior_bound_to_snapshot`.
    pub const POSTERIOR_BOUND_TO_SNAPSHOT: &str = "measurement_model.posterior_bound_to_snapshot";
    /// Detail `measurement_model.posterior_capability`.
    pub const POSTERIOR_CAPABILITY: &str = "measurement_model.posterior_capability";
    /// Detail `measurement_model.posterior_declaration`.
    pub const POSTERIOR_DECLARATION: &str = "measurement_model.posterior_declaration";
    /// Detail `measurement_model.posterior_identity`.
    pub const POSTERIOR_IDENTITY: &str = "measurement_model.posterior_identity";
    /// Detail `measurement_model.posterior_kind`.
    pub const POSTERIOR_KIND: &str = "measurement_model.posterior_kind";
    /// Detail `measurement_model.posterior_native_trust`.
    pub const POSTERIOR_NATIVE_TRUST: &str = "measurement_model.posterior_native_trust";
    /// Detail `measurement_model.posterior_quantity`.
    pub const POSTERIOR_QUANTITY: &str = "measurement_model.posterior_quantity";
    /// Detail `measurement_model.posterior_receipt`.
    pub const POSTERIOR_RECEIPT: &str = "measurement_model.posterior_receipt";
    /// Detail `measurement_model.proxy_as_construct`.
    pub const PROXY_AS_CONSTRUCT: &str = "measurement_model.proxy_as_construct";
    /// Detail `measurement_model.regime_mismatch`.
    pub const REGIME_MISMATCH: &str = "measurement_model.regime_mismatch";
    /// Detail `measurement_model.reliability_too_small`.
    pub const RELIABILITY_TOO_SMALL: &str = "measurement_model.reliability_too_small";
    /// Detail `measurement_model.representation_unmapped`.
    pub const REPRESENTATION_UNMAPPED: &str = "measurement_model.representation_unmapped";
    /// Detail `measurement_model.route_mismatch`.
    pub const ROUTE_MISMATCH: &str = "measurement_model.route_mismatch";
    /// Detail `measurement_model.route_parameters`.
    pub const ROUTE_PARAMETERS: &str = "measurement_model.route_parameters";
    /// Detail `measurement_model.structure_parameters`.
    pub const STRUCTURE_PARAMETERS: &str = "measurement_model.structure_parameters";
    /// Detail `measurement_model.transport_receipt_mismatch`.
    pub const TRANSPORT_RECEIPT_MISMATCH: &str = "measurement_model.transport_receipt_mismatch";
    /// Detail `measurement_model.unknown_level`.
    pub const UNKNOWN_LEVEL: &str = "measurement_model.unknown_level";
    /// Detail `measurement_model.unmodelled_proxy`.
    pub const UNMODELLED_PROXY: &str = "measurement_model.unmodelled_proxy";
    /// Detail `measurement_model.unrecognized`.
    pub const UNRECOGNIZED: &str = "measurement_model.unrecognized";
    /// Detail `measurement_model.variance_not_positive`.
    pub const VARIANCE_NOT_POSITIVE: &str = "measurement_model.variance_not_positive";
    /// Detail `measurement_model.youden_too_small`.
    pub const YOUDEN_TOO_SMALL: &str = "measurement_model.youden_too_small";
}

/// Details in the `mechanism_composition` family.
pub mod mechanism_composition {
    /// Detail `mechanism_composition.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "mechanism_composition.artifact_invalid";
    /// Detail `mechanism_composition.artifact_kind_mismatch`.
    pub const ARTIFACT_KIND_MISMATCH: &str = "mechanism_composition.artifact_kind_mismatch";
    /// Detail `mechanism_composition.cancelled`.
    pub const CANCELLED: &str = "mechanism_composition.cancelled";
    /// Detail `mechanism_composition.conditioning_extra_conditioner`.
    pub const CONDITIONING_EXTRA_CONDITIONER: &str = "mechanism_composition.conditioning_extra_conditioner";
    /// Detail `mechanism_composition.conditioning_parent_not_conditioned`.
    pub const CONDITIONING_PARENT_NOT_CONDITIONED: &str = "mechanism_composition.conditioning_parent_not_conditioned";
    /// Detail `mechanism_composition.cycle_directed_cycle`.
    pub const CYCLE_DIRECTED_CYCLE: &str = "mechanism_composition.cycle_directed_cycle";
    /// Detail `mechanism_composition.family_mixed`.
    pub const FAMILY_MIXED: &str = "mechanism_composition.family_mixed";
    /// Detail `mechanism_composition.identity_declaration_unknown_variable`.
    pub const IDENTITY_DECLARATION_UNKNOWN_VARIABLE: &str = "mechanism_composition.identity_declaration_unknown_variable";
    /// Detail `mechanism_composition.identity_duplicate_mechanism`.
    pub const IDENTITY_DUPLICATE_MECHANISM: &str = "mechanism_composition.identity_duplicate_mechanism";
    /// Detail `mechanism_composition.identity_edge_unknown_variable`.
    pub const IDENTITY_EDGE_UNKNOWN_VARIABLE: &str = "mechanism_composition.identity_edge_unknown_variable";
    /// Detail `mechanism_composition.identity_parent_without_mechanism`.
    pub const IDENTITY_PARENT_WITHOUT_MECHANISM: &str = "mechanism_composition.identity_parent_without_mechanism";
    /// Detail `mechanism_composition.interval_invalid`.
    pub const INTERVAL_INVALID: &str = "mechanism_composition.interval_invalid";
    /// Detail `mechanism_composition.latent_family_unsupported`.
    pub const LATENT_FAMILY_UNSUPPORTED: &str = "mechanism_composition.latent_family_unsupported";
    /// Detail `mechanism_composition.latent_loading_undeclared`.
    pub const LATENT_LOADING_UNDECLARED: &str = "mechanism_composition.latent_loading_undeclared";
    /// Detail `mechanism_composition.latent_shared_with_absent_mechanism`.
    pub const LATENT_SHARED_WITH_ABSENT_MECHANISM: &str = "mechanism_composition.latent_shared_with_absent_mechanism";
    /// Detail `mechanism_composition.latent_sharing_not_declared`.
    pub const LATENT_SHARING_NOT_DECLARED: &str = "mechanism_composition.latent_sharing_not_declared";
    /// Detail `mechanism_composition.latent_sharing_not_reciprocal`.
    pub const LATENT_SHARING_NOT_RECIPROCAL: &str = "mechanism_composition.latent_sharing_not_reciprocal";
    /// Detail `mechanism_composition.latent_too_many`.
    pub const LATENT_TOO_MANY: &str = "mechanism_composition.latent_too_many";
    /// Detail `mechanism_composition.numbers_unbound`.
    pub const NUMBERS_UNBOUND: &str = "mechanism_composition.numbers_unbound";
    /// Detail `mechanism_composition.numerical_failure`.
    pub const NUMERICAL_FAILURE: &str = "mechanism_composition.numerical_failure";
    /// Detail `mechanism_composition.parents_declared_graph_mismatch`.
    pub const PARENTS_DECLARED_GRAPH_MISMATCH: &str = "mechanism_composition.parents_declared_graph_mismatch";
    /// Detail `mechanism_composition.population_invariance_unstated`.
    pub const POPULATION_INVARIANCE_UNSTATED: &str = "mechanism_composition.population_invariance_unstated";
    /// Detail `mechanism_composition.population_latent_blocks_selection_admissibility`.
    pub const POPULATION_LATENT_BLOCKS_SELECTION_ADMISSIBILITY: &str = "mechanism_composition.population_latent_blocks_selection_admissibility";
    /// Detail `mechanism_composition.population_parent_context_differs_from_fit`.
    pub const POPULATION_PARENT_CONTEXT_DIFFERS_FROM_FIT: &str = "mechanism_composition.population_parent_context_differs_from_fit";
    /// Detail `mechanism_composition.population_selection_arrow_into_mechanism`.
    pub const POPULATION_SELECTION_ARROW_INTO_MECHANISM: &str = "mechanism_composition.population_selection_arrow_into_mechanism";
    /// Detail `mechanism_composition.population_undeclared_transport`.
    pub const POPULATION_UNDECLARED_TRANSPORT: &str = "mechanism_composition.population_undeclared_transport";
    /// Detail `mechanism_composition.posterior_dependence_cross_block_duplicate`.
    pub const POSTERIOR_DEPENDENCE_CROSS_BLOCK_DUPLICATE: &str = "mechanism_composition.posterior_dependence_cross_block_duplicate";
    /// Detail `mechanism_composition.posterior_dependence_cross_covariance_missing`.
    pub const POSTERIOR_DEPENDENCE_CROSS_COVARIANCE_MISSING: &str = "mechanism_composition.posterior_dependence_cross_covariance_missing";
    /// Detail `mechanism_composition.posterior_dependence_evidence_overlap_unknown`.
    pub const POSTERIOR_DEPENDENCE_EVIDENCE_OVERLAP_UNKNOWN: &str = "mechanism_composition.posterior_dependence_evidence_overlap_unknown";
    /// Detail `mechanism_composition.posterior_dependence_independence_unjustified`.
    pub const POSTERIOR_DEPENDENCE_INDEPENDENCE_UNJUSTIFIED: &str = "mechanism_composition.posterior_dependence_independence_unjustified";
    /// Detail `mechanism_composition.posterior_dependence_shared_evidence_declared_independent`.
    pub const POSTERIOR_DEPENDENCE_SHARED_EVIDENCE_DECLARED_INDEPENDENT: &str = "mechanism_composition.posterior_dependence_shared_evidence_declared_independent";
    /// Detail `mechanism_composition.posterior_dependence_unknown`.
    pub const POSTERIOR_DEPENDENCE_UNKNOWN: &str = "mechanism_composition.posterior_dependence_unknown";
    /// Detail `mechanism_composition.premises_mismatch`.
    pub const PREMISES_MISMATCH: &str = "mechanism_composition.premises_mismatch";
    /// Detail `mechanism_composition.provenance_declared_mechanism_unacknowledged`.
    pub const PROVENANCE_DECLARED_MECHANISM_UNACKNOWLEDGED: &str = "mechanism_composition.provenance_declared_mechanism_unacknowledged";
    /// Detail `mechanism_composition.provenance_missing_artifact`.
    pub const PROVENANCE_MISSING_ARTIFACT: &str = "mechanism_composition.provenance_missing_artifact";
    /// Detail `mechanism_composition.provenance_missing_evidence`.
    pub const PROVENANCE_MISSING_EVIDENCE: &str = "mechanism_composition.provenance_missing_evidence";
    /// Detail `mechanism_composition.quantity_conditioned_event`.
    pub const QUANTITY_CONDITIONED_EVENT: &str = "mechanism_composition.quantity_conditioned_event";
    /// Detail `mechanism_composition.quantity_conditioning_mismatch`.
    pub const QUANTITY_CONDITIONING_MISMATCH: &str = "mechanism_composition.quantity_conditioning_mismatch";
    /// Detail `mechanism_composition.quantity_functional_mismatch`.
    pub const QUANTITY_FUNCTIONAL_MISMATCH: &str = "mechanism_composition.quantity_functional_mismatch";
    /// Detail `mechanism_composition.quantity_horizon_mismatch`.
    pub const QUANTITY_HORIZON_MISMATCH: &str = "mechanism_composition.quantity_horizon_mismatch";
    /// Detail `mechanism_composition.quantity_invalid_coordinate`.
    pub const QUANTITY_INVALID_COORDINATE: &str = "mechanism_composition.quantity_invalid_coordinate";
    /// Detail `mechanism_composition.quantity_not_a_realization`.
    pub const QUANTITY_NOT_A_REALIZATION: &str = "mechanism_composition.quantity_not_a_realization";
    /// Detail `mechanism_composition.quantity_role_mismatch`.
    pub const QUANTITY_ROLE_MISMATCH: &str = "mechanism_composition.quantity_role_mismatch";
    /// Detail `mechanism_composition.regime_parent_context_differs_from_fit`.
    pub const REGIME_PARENT_CONTEXT_DIFFERS_FROM_FIT: &str = "mechanism_composition.regime_parent_context_differs_from_fit";
    /// Detail `mechanism_composition.regime_selection_diagram_not_applicable`.
    pub const REGIME_SELECTION_DIAGRAM_NOT_APPLICABLE: &str = "mechanism_composition.regime_selection_diagram_not_applicable";
    /// Detail `mechanism_composition.regime_undeclared_invariance`.
    pub const REGIME_UNDECLARED_INVARIANCE: &str = "mechanism_composition.regime_undeclared_invariance";
    /// Detail `mechanism_composition.request_unsupported`.
    pub const REQUEST_UNSUPPORTED: &str = "mechanism_composition.request_unsupported";
    /// Detail `mechanism_composition.result_mismatch`.
    pub const RESULT_MISMATCH: &str = "mechanism_composition.result_mismatch";
    /// Detail `mechanism_composition.shape_blank_coordinate`.
    pub const SHAPE_BLANK_COORDINATE: &str = "mechanism_composition.shape_blank_coordinate";
    /// Detail `mechanism_composition.shape_blank_identity`.
    pub const SHAPE_BLANK_IDENTITY: &str = "mechanism_composition.shape_blank_identity";
    /// Detail `mechanism_composition.shape_coefficient_parent_mismatch`.
    pub const SHAPE_COEFFICIENT_PARENT_MISMATCH: &str = "mechanism_composition.shape_coefficient_parent_mismatch";
    /// Detail `mechanism_composition.shape_duplicate_parent`.
    pub const SHAPE_DUPLICATE_PARENT: &str = "mechanism_composition.shape_duplicate_parent";
    /// Detail `mechanism_composition.shape_empty_composition`.
    pub const SHAPE_EMPTY_COMPOSITION: &str = "mechanism_composition.shape_empty_composition";
    /// Detail `mechanism_composition.shape_invalid_domain`.
    pub const SHAPE_INVALID_DOMAIN: &str = "mechanism_composition.shape_invalid_domain";
    /// Detail `mechanism_composition.shape_invalid_latents`.
    pub const SHAPE_INVALID_LATENTS: &str = "mechanism_composition.shape_invalid_latents";
    /// Detail `mechanism_composition.shape_invalid_table`.
    pub const SHAPE_INVALID_TABLE: &str = "mechanism_composition.shape_invalid_table";
    /// Detail `mechanism_composition.shape_mechanism_count`.
    pub const SHAPE_MECHANISM_COUNT: &str = "mechanism_composition.shape_mechanism_count";
    /// Detail `mechanism_composition.shape_negative_scale`.
    pub const SHAPE_NEGATIVE_SCALE: &str = "mechanism_composition.shape_negative_scale";
    /// Detail `mechanism_composition.shape_non_finite_parameter`.
    pub const SHAPE_NON_FINITE_PARAMETER: &str = "mechanism_composition.shape_non_finite_parameter";
    /// Detail `mechanism_composition.shape_too_many_mechanisms`.
    pub const SHAPE_TOO_MANY_MECHANISMS: &str = "mechanism_composition.shape_too_many_mechanisms";
    /// Detail `mechanism_composition.shape_too_many_parents`.
    pub const SHAPE_TOO_MANY_PARENTS: &str = "mechanism_composition.shape_too_many_parents";
    /// Detail `mechanism_composition.slot_not_linear_gaussian`.
    pub const SLOT_NOT_LINEAR_GAUSSIAN: &str = "mechanism_composition.slot_not_linear_gaussian";
    /// Detail `mechanism_composition.support_disjoint`.
    pub const SUPPORT_DISJOINT: &str = "mechanism_composition.support_disjoint";
    /// Detail `mechanism_composition.support_intervention_outside_domain`.
    pub const SUPPORT_INTERVENTION_OUTSIDE_DOMAIN: &str = "mechanism_composition.support_intervention_outside_domain";
    /// Detail `mechanism_composition.support_kind_mismatch`.
    pub const SUPPORT_KIND_MISMATCH: &str = "mechanism_composition.support_kind_mismatch";
    /// Detail `mechanism_composition.support_missing_parent_configuration`.
    pub const SUPPORT_MISSING_PARENT_CONFIGURATION: &str = "mechanism_composition.support_missing_parent_configuration";
    /// Detail `mechanism_composition.support_not_contained`.
    pub const SUPPORT_NOT_CONTAINED: &str = "mechanism_composition.support_not_contained";
    /// Detail `mechanism_composition.transform_mismatch`.
    pub const TRANSFORM_MISMATCH: &str = "mechanism_composition.transform_mismatch";
    /// Detail `mechanism_composition.uncertainty_assembled_not_psd`.
    pub const UNCERTAINTY_ASSEMBLED_NOT_PSD: &str = "mechanism_composition.uncertainty_assembled_not_psd";
    /// Detail `mechanism_composition.uncertainty_cross_block_shape`.
    pub const UNCERTAINTY_CROSS_BLOCK_SHAPE: &str = "mechanism_composition.uncertainty_cross_block_shape";
    /// Detail `mechanism_composition.uncertainty_cross_block_unknown_mechanism`.
    pub const UNCERTAINTY_CROSS_BLOCK_UNKNOWN_MECHANISM: &str = "mechanism_composition.uncertainty_cross_block_unknown_mechanism";
    /// Detail `mechanism_composition.uncertainty_family_unsupported`.
    pub const UNCERTAINTY_FAMILY_UNSUPPORTED: &str = "mechanism_composition.uncertainty_family_unsupported";
    /// Detail `mechanism_composition.uncertainty_joint_parameter_set_mismatch`.
    pub const UNCERTAINTY_JOINT_PARAMETER_SET_MISMATCH: &str = "mechanism_composition.uncertainty_joint_parameter_set_mismatch";
    /// Detail `mechanism_composition.uncertainty_marginal_inconsistent_with_joint`.
    pub const UNCERTAINTY_MARGINAL_INCONSISTENT_WITH_JOINT: &str = "mechanism_composition.uncertainty_marginal_inconsistent_with_joint";
    /// Detail `mechanism_composition.uncertainty_marginal_missing`.
    pub const UNCERTAINTY_MARGINAL_MISSING: &str = "mechanism_composition.uncertainty_marginal_missing";
    /// Detail `mechanism_composition.uncertainty_meaning_mismatch`.
    pub const UNCERTAINTY_MEANING_MISMATCH: &str = "mechanism_composition.uncertainty_meaning_mismatch";
    /// Detail `mechanism_composition.uncertainty_not_psd`.
    pub const UNCERTAINTY_NOT_PSD: &str = "mechanism_composition.uncertainty_not_psd";
    /// Detail `mechanism_composition.uncertainty_not_symmetric`.
    pub const UNCERTAINTY_NOT_SYMMETRIC: &str = "mechanism_composition.uncertainty_not_symmetric";
    /// Detail `mechanism_composition.uncertainty_parameter_set_mismatch`.
    pub const UNCERTAINTY_PARAMETER_SET_MISMATCH: &str = "mechanism_composition.uncertainty_parameter_set_mismatch";
    /// Detail `mechanism_composition.uncertainty_shape`.
    pub const UNCERTAINTY_SHAPE: &str = "mechanism_composition.uncertainty_shape";
    /// Detail `mechanism_composition.units_mismatch`.
    pub const UNITS_MISMATCH: &str = "mechanism_composition.units_mismatch";
}

/// Details in the `mechanism_discrepancy` family.
pub mod mechanism_discrepancy {
    /// Detail `mechanism_discrepancy.degenerate_covariance`.
    pub const DEGENERATE_COVARIANCE: &str = "mechanism_discrepancy.degenerate_covariance";
    /// Detail `mechanism_discrepancy.dependence_unknown`.
    pub const DEPENDENCE_UNKNOWN: &str = "mechanism_discrepancy.dependence_unknown";
    /// Detail `mechanism_discrepancy.duplicate_name`.
    pub const DUPLICATE_NAME: &str = "mechanism_discrepancy.duplicate_name";
    /// Detail `mechanism_discrepancy.incomparable_measurements`.
    pub const INCOMPARABLE_MEASUREMENTS: &str = "mechanism_discrepancy.incomparable_measurements";
    /// Detail `mechanism_discrepancy.inconsistent_summary`.
    pub const INCONSISTENT_SUMMARY: &str = "mechanism_discrepancy.inconsistent_summary";
    /// Detail `mechanism_discrepancy.invalid_alpha`.
    pub const INVALID_ALPHA: &str = "mechanism_discrepancy.invalid_alpha";
    /// Detail `mechanism_discrepancy.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "mechanism_discrepancy.invalid_artifact";
    /// Detail `mechanism_discrepancy.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "mechanism_discrepancy.invalid_identity";
    /// Detail `mechanism_discrepancy.invalid_power`.
    pub const INVALID_POWER: &str = "mechanism_discrepancy.invalid_power";
    /// Detail `mechanism_discrepancy.invalid_request`.
    pub const INVALID_REQUEST: &str = "mechanism_discrepancy.invalid_request";
    /// Detail `mechanism_discrepancy.no_compared_coefficients`.
    pub const NO_COMPARED_COEFFICIENTS: &str = "mechanism_discrepancy.no_compared_coefficients";
    /// Detail `mechanism_discrepancy.non_finite_value`.
    pub const NON_FINITE_VALUE: &str = "mechanism_discrepancy.non_finite_value";
    /// Detail `mechanism_discrepancy.rank_deficient_design`.
    pub const RANK_DEFICIENT_DESIGN: &str = "mechanism_discrepancy.rank_deficient_design";
    /// Detail `mechanism_discrepancy.row_count_mismatch`.
    pub const ROW_COUNT_MISMATCH: &str = "mechanism_discrepancy.row_count_mismatch";
    /// Detail `mechanism_discrepancy.sample_too_small`.
    pub const SAMPLE_TOO_SMALL: &str = "mechanism_discrepancy.sample_too_small";
    /// Detail `mechanism_discrepancy.too_many_parents`.
    pub const TOO_MANY_PARENTS: &str = "mechanism_discrepancy.too_many_parents";
    /// Detail `mechanism_discrepancy.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "mechanism_discrepancy.wrong_contract";
}

/// Details in the `mechanism_posterior` family.
pub mod mechanism_posterior {
    /// Detail `mechanism_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "mechanism_posterior.band_not_constructed";
    /// Detail `mechanism_posterior.empty_summary`.
    pub const EMPTY_SUMMARY: &str = "mechanism_posterior.empty_summary";
    /// Detail `mechanism_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "mechanism_posterior.external_power_prior";
    /// Detail `mechanism_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "mechanism_posterior.mapped_prior";
    /// Detail `mechanism_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "mechanism_posterior.non_positive_scale";
}

/// Details in the `mechanism_transport` family.
pub mod mechanism_transport {
    /// Detail `mechanism_transport.diagram_source_mismatch`.
    pub const DIAGRAM_SOURCE_MISMATCH: &str = "mechanism_transport.diagram_source_mismatch";
    /// Detail `mechanism_transport.diagram_target_mismatch`.
    pub const DIAGRAM_TARGET_MISMATCH: &str = "mechanism_transport.diagram_target_mismatch";
    /// Detail `mechanism_transport.duplicate_mechanism`.
    pub const DUPLICATE_MECHANISM: &str = "mechanism_transport.duplicate_mechanism";
    /// Detail `mechanism_transport.family_unsupported`.
    pub const FAMILY_UNSUPPORTED: &str = "mechanism_transport.family_unsupported";
    /// Detail `mechanism_transport.origin_not_fitted`.
    pub const ORIGIN_NOT_FITTED: &str = "mechanism_transport.origin_not_fitted";
    /// Detail `mechanism_transport.selection_arrow_into_latent`.
    pub const SELECTION_ARROW_INTO_LATENT: &str = "mechanism_transport.selection_arrow_into_latent";
    /// Detail `mechanism_transport.target_mechanism_not_native`.
    pub const TARGET_MECHANISM_NOT_NATIVE: &str = "mechanism_transport.target_mechanism_not_native";
    /// Detail `mechanism_transport.transport_identity_mismatch`.
    pub const TRANSPORT_IDENTITY_MISMATCH: &str = "mechanism_transport.transport_identity_mismatch";
}

/// Details in the `mediation_posterior` family.
pub mod mediation_posterior {
    /// Detail `mediation_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "mediation_posterior.band_not_constructed";
    /// Detail `mediation_posterior.empty_design`.
    pub const EMPTY_DESIGN: &str = "mediation_posterior.empty_design";
    /// Detail `mediation_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "mediation_posterior.external_power_prior";
    /// Detail `mediation_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "mediation_posterior.mapped_prior";
    /// Detail `mediation_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "mediation_posterior.non_positive_scale";
}

/// Details in the `missing_data` family.
pub mod missing_data {
    /// Detail `missing_data.covariates_incomplete`.
    pub const COVARIATES_INCOMPLETE: &str = "missing_data.covariates_incomplete";
    /// Detail `missing_data.multiple_imputation`.
    pub const MULTIPLE_IMPUTATION: &str = "missing_data.multiple_imputation";
    /// Detail `missing_data.policy_not_applicable`.
    pub const POLICY_NOT_APPLICABLE: &str = "missing_data.policy_not_applicable";
    /// Detail `missing_data.policy_required`.
    pub const POLICY_REQUIRED: &str = "missing_data.policy_required";
}

/// Details in the `mixed_search` family.
pub mod mixed_search {
    /// Detail `mixed_search.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "mixed_search.bounds_exceeded";
    /// Detail `mixed_search.budget`.
    pub const BUDGET: &str = "mixed_search.budget";
    /// Detail `mixed_search.invalid_derivation`.
    pub const INVALID_DERIVATION: &str = "mixed_search.invalid_derivation";
    /// Detail `mixed_search.invalid_query`.
    pub const INVALID_QUERY: &str = "mixed_search.invalid_query";
    /// Detail `mixed_search.missing_joint`.
    pub const MISSING_JOINT: &str = "mixed_search.missing_joint";
    /// Detail `mixed_search.named_route`.
    pub const NAMED_ROUTE: &str = "mixed_search.named_route";
    /// Detail `mixed_search.not_certified`.
    pub const NOT_CERTIFIED: &str = "mixed_search.not_certified";
    /// Detail `mixed_search.posterior_as_law`.
    pub const POSTERIOR_AS_LAW: &str = "mixed_search.posterior_as_law";
}

/// Details in the `mnar_fixed_shift` family.
pub mod mnar_fixed_shift {
    /// Detail `mnar_fixed_shift.cancelled`.
    pub const CANCELLED: &str = "mnar_fixed_shift.cancelled";
    /// Detail `mnar_fixed_shift.constant_outcome`.
    pub const CONSTANT_OUTCOME: &str = "mnar_fixed_shift.constant_outcome";
    /// Detail `mnar_fixed_shift.degenerate_interval`.
    pub const DEGENERATE_INTERVAL: &str = "mnar_fixed_shift.degenerate_interval";
    /// Detail `mnar_fixed_shift.duplicate_subject`.
    pub const DUPLICATE_SUBJECT: &str = "mnar_fixed_shift.duplicate_subject";
    /// Detail `mnar_fixed_shift.insufficient_observations`.
    pub const INSUFFICIENT_OBSERVATIONS: &str = "mnar_fixed_shift.insufficient_observations";
    /// Detail `mnar_fixed_shift.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "mnar_fixed_shift.interval_not_supported";
    /// Detail `mnar_fixed_shift.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "mnar_fixed_shift.invalid_declaration";
    /// Detail `mnar_fixed_shift.invalid_outcome`.
    pub const INVALID_OUTCOME: &str = "mnar_fixed_shift.invalid_outcome";
    /// Detail `mnar_fixed_shift.invalid_shift`.
    pub const INVALID_SHIFT: &str = "mnar_fixed_shift.invalid_shift";
    /// Detail `mnar_fixed_shift.invalid_source`.
    pub const INVALID_SOURCE: &str = "mnar_fixed_shift.invalid_source";
    /// Detail `mnar_fixed_shift.lowering`.
    pub const LOWERING: &str = "mnar_fixed_shift.lowering";
    /// Detail `mnar_fixed_shift.preparation`.
    pub const PREPARATION: &str = "mnar_fixed_shift.preparation";
    /// Detail `mnar_fixed_shift.preparation_cancelled`.
    pub const PREPARATION_CANCELLED: &str = "mnar_fixed_shift.preparation_cancelled";
    /// Detail `mnar_fixed_shift.preparation_resource_limit`.
    pub const PREPARATION_RESOURCE_LIMIT: &str = "mnar_fixed_shift.preparation_resource_limit";
    /// Detail `mnar_fixed_shift.source_admission_cancelled`.
    pub const SOURCE_ADMISSION_CANCELLED: &str = "mnar_fixed_shift.source_admission_cancelled";
    /// Detail `mnar_fixed_shift.source_admission_resource_limit`.
    pub const SOURCE_ADMISSION_RESOURCE_LIMIT: &str = "mnar_fixed_shift.source_admission_resource_limit";
}

/// Details in the `mnar_sensitivity` family.
pub mod mnar_sensitivity {
    /// Detail `mnar_sensitivity.adjustment_not_valid`.
    pub const ADJUSTMENT_NOT_VALID: &str = "mnar_sensitivity.adjustment_not_valid";
    /// Detail `mnar_sensitivity.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "mnar_sensitivity.bounds_exceeded";
    /// Detail `mnar_sensitivity.composition_not_supported`.
    pub const COMPOSITION_NOT_SUPPORTED: &str = "mnar_sensitivity.composition_not_supported";
    /// Detail `mnar_sensitivity.covariates_not_used`.
    pub const COVARIATES_NOT_USED: &str = "mnar_sensitivity.covariates_not_used";
    /// Detail `mnar_sensitivity.covariates_required`.
    pub const COVARIATES_REQUIRED: &str = "mnar_sensitivity.covariates_required";
    /// Detail `mnar_sensitivity.design_singular`.
    pub const DESIGN_SINGULAR: &str = "mnar_sensitivity.design_singular";
    /// Detail `mnar_sensitivity.graph_not_supported`.
    pub const GRAPH_NOT_SUPPORTED: &str = "mnar_sensitivity.graph_not_supported";
    /// Detail `mnar_sensitivity.insufficient_observed`.
    pub const INSUFFICIENT_OBSERVED: &str = "mnar_sensitivity.insufficient_observed";
    /// Detail `mnar_sensitivity.invalid_bootstrap`.
    pub const INVALID_BOOTSTRAP: &str = "mnar_sensitivity.invalid_bootstrap";
    /// Detail `mnar_sensitivity.invalid_data`.
    pub const INVALID_DATA: &str = "mnar_sensitivity.invalid_data";
    /// Detail `mnar_sensitivity.invalid_decision`.
    pub const INVALID_DECISION: &str = "mnar_sensitivity.invalid_decision";
    /// Detail `mnar_sensitivity.invalid_level`.
    pub const INVALID_LEVEL: &str = "mnar_sensitivity.invalid_level";
    /// Detail `mnar_sensitivity.invalid_loading`.
    pub const INVALID_LOADING: &str = "mnar_sensitivity.invalid_loading";
    /// Detail `mnar_sensitivity.invalid_tolerance`.
    pub const INVALID_TOLERANCE: &str = "mnar_sensitivity.invalid_tolerance";
    /// Detail `mnar_sensitivity.mechanism_not_declared`.
    pub const MECHANISM_NOT_DECLARED: &str = "mnar_sensitivity.mechanism_not_declared";
    /// Detail `mnar_sensitivity.range_excludes_mar`.
    pub const RANGE_EXCLUDES_MAR: &str = "mnar_sensitivity.range_excludes_mar";
    /// Detail `mnar_sensitivity.unbounded_range`.
    pub const UNBOUNDED_RANGE: &str = "mnar_sensitivity.unbounded_range";
    /// Detail `mnar_sensitivity.unsupported_estimator`.
    pub const UNSUPPORTED_ESTIMATOR: &str = "mnar_sensitivity.unsupported_estimator";
    /// Detail `mnar_sensitivity.unsupported_mechanism`.
    pub const UNSUPPORTED_MECHANISM: &str = "mnar_sensitivity.unsupported_mechanism";
    /// Detail `mnar_sensitivity.unsupported_query`.
    pub const UNSUPPORTED_QUERY: &str = "mnar_sensitivity.unsupported_query";
    /// Detail `mnar_sensitivity.unsupported_target`.
    pub const UNSUPPORTED_TARGET: &str = "mnar_sensitivity.unsupported_target";
}

/// Details in the `msm_posterior` family.
pub mod msm_posterior {
    /// Detail `msm_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "msm_posterior.band_not_constructed";
    /// Detail `msm_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "msm_posterior.empty_law";
    /// Detail `msm_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "msm_posterior.external_power_prior";
    /// Detail `msm_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "msm_posterior.mapped_prior";
    /// Detail `msm_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "msm_posterior.non_positive_scale";
}

/// Details in the `msm_sensitivity` family.
pub mod msm_sensitivity {
    /// Detail `msm_sensitivity.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "msm_sensitivity.bounds_exceeded";
    /// Detail `msm_sensitivity.composition_not_supported`.
    pub const COMPOSITION_NOT_SUPPORTED: &str = "msm_sensitivity.composition_not_supported";
    /// Detail `msm_sensitivity.invalid_threshold`.
    pub const INVALID_THRESHOLD: &str = "msm_sensitivity.invalid_threshold";
    /// Detail `msm_sensitivity.invalid_tolerance`.
    pub const INVALID_TOLERANCE: &str = "msm_sensitivity.invalid_tolerance";
    /// Detail `msm_sensitivity.lambda_below_one`.
    pub const LAMBDA_BELOW_ONE: &str = "msm_sensitivity.lambda_below_one";
    /// Detail `msm_sensitivity.lambda_range_empty`.
    pub const LAMBDA_RANGE_EMPTY: &str = "msm_sensitivity.lambda_range_empty";
    /// Detail `msm_sensitivity.no_interval_method`.
    pub const NO_INTERVAL_METHOD: &str = "msm_sensitivity.no_interval_method";
    /// Detail `msm_sensitivity.outcome_law`.
    pub const OUTCOME_LAW: &str = "msm_sensitivity.outcome_law";
    /// Detail `msm_sensitivity.positivity`.
    pub const POSITIVITY: &str = "msm_sensitivity.positivity";
    /// Detail `msm_sensitivity.stratum_mass`.
    pub const STRATUM_MASS: &str = "msm_sensitivity.stratum_mass";
}

/// Details in the `multi_source_posterior` family.
pub mod multi_source_posterior {
    /// Detail `multi_source_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "multi_source_posterior.band_not_constructed";
    /// Detail `multi_source_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "multi_source_posterior.empty_law";
    /// Detail `multi_source_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "multi_source_posterior.external_power_prior";
    /// Detail `multi_source_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "multi_source_posterior.mapped_prior";
    /// Detail `multi_source_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "multi_source_posterior.non_positive_scale";
}

/// Details in the `multiple_imputation` family.
pub mod multiple_imputation {
    /// Detail `multiple_imputation.admission_refused`.
    pub const ADMISSION_REFUSED: &str = "multiple_imputation.admission_refused";
    /// Detail `multiple_imputation.ambiguous_variable_domain`.
    pub const AMBIGUOUS_VARIABLE_DOMAIN: &str = "multiple_imputation.ambiguous_variable_domain";
    /// Detail `multiple_imputation.binary_under_joint_normal`.
    pub const BINARY_UNDER_JOINT_NORMAL: &str = "multiple_imputation.binary_under_joint_normal";
    /// Detail `multiple_imputation.cancelled`.
    pub const CANCELLED: &str = "multiple_imputation.cancelled";
    /// Detail `multiple_imputation.clusters_required`.
    pub const CLUSTERS_REQUIRED: &str = "multiple_imputation.clusters_required";
    /// Detail `multiple_imputation.column_length`.
    pub const COLUMN_LENGTH: &str = "multiple_imputation.column_length";
    /// Detail `multiple_imputation.conditional_model_not_fitted`.
    pub const CONDITIONAL_MODEL_NOT_FITTED: &str = "multiple_imputation.conditional_model_not_fitted";
    /// Detail `multiple_imputation.dependent_series`.
    pub const DEPENDENT_SERIES: &str = "multiple_imputation.dependent_series";
    /// Detail `multiple_imputation.dynamics_joint_normal`.
    pub const DYNAMICS_JOINT_NORMAL: &str = "multiple_imputation.dynamics_joint_normal";
    /// Detail `multiple_imputation.dynamics_lags`.
    pub const DYNAMICS_LAGS: &str = "multiple_imputation.dynamics_lags";
    /// Detail `multiple_imputation.graph_posterior_atoms`.
    pub const GRAPH_POSTERIOR_ATOMS: &str = "multiple_imputation.graph_posterior_atoms";
    /// Detail `multiple_imputation.graph_posterior_frequentist`.
    pub const GRAPH_POSTERIOR_FREQUENTIST: &str = "multiple_imputation.graph_posterior_frequentist";
    /// Detail `multiple_imputation.graph_posterior_query`.
    pub const GRAPH_POSTERIOR_QUERY: &str = "multiple_imputation.graph_posterior_query";
    /// Detail `multiple_imputation.graph_posterior_series`.
    pub const GRAPH_POSTERIOR_SERIES: &str = "multiple_imputation.graph_posterior_series";
    /// Detail `multiple_imputation.index`.
    pub const INDEX: &str = "multiple_imputation.index";
    /// Detail `multiple_imputation.interaction_variable`.
    pub const INTERACTION_VARIABLE: &str = "multiple_imputation.interaction_variable";
    /// Detail `multiple_imputation.interactions_joint_normal`.
    pub const INTERACTIONS_JOINT_NORMAL: &str = "multiple_imputation.interactions_joint_normal";
    /// Detail `multiple_imputation.mixed_fits`.
    pub const MIXED_FITS: &str = "multiple_imputation.mixed_fits";
    /// Detail `multiple_imputation.multi_horizon_posterior_not_joint`.
    pub const MULTI_HORIZON_POSTERIOR_NOT_JOINT: &str = "multiple_imputation.multi_horizon_posterior_not_joint";
    /// Detail `multiple_imputation.non_finite`.
    pub const NON_FINITE: &str = "multiple_imputation.non_finite";
    /// Detail `multiple_imputation.not_at_random`.
    pub const NOT_AT_RANDOM: &str = "multiple_imputation.not_at_random";
    /// Detail `multiple_imputation.not_identified`.
    pub const NOT_IDENTIFIED: &str = "multiple_imputation.not_identified";
    /// Detail `multiple_imputation.pool_df`.
    pub const POOL_DF: &str = "multiple_imputation.pool_df";
    /// Detail `multiple_imputation.pool_non_finite`.
    pub const POOL_NON_FINITE: &str = "multiple_imputation.pool_non_finite";
    /// Detail `multiple_imputation.pool_shape`.
    pub const POOL_SHAPE: &str = "multiple_imputation.pool_shape";
    /// Detail `multiple_imputation.sampling_design`.
    pub const SAMPLING_DESIGN: &str = "multiple_imputation.sampling_design";
    /// Detail `multiple_imputation.spec_range`.
    pub const SPEC_RANGE: &str = "multiple_imputation.spec_range";
    /// Detail `multiple_imputation.stack_shape`.
    pub const STACK_SHAPE: &str = "multiple_imputation.stack_shape";
    /// Detail `multiple_imputation.too_few_imputations`.
    pub const TOO_FEW_IMPUTATIONS: &str = "multiple_imputation.too_few_imputations";
    /// Detail `multiple_imputation.units_labels`.
    pub const UNITS_LABELS: &str = "multiple_imputation.units_labels";
    /// Detail `multiple_imputation.units_shape`.
    pub const UNITS_SHAPE: &str = "multiple_imputation.units_shape";
    /// Detail `multiple_imputation.variable_count`.
    pub const VARIABLE_COUNT: &str = "multiple_imputation.variable_count";
    /// Detail `multiple_imputation.variable_domain`.
    pub const VARIABLE_DOMAIN: &str = "multiple_imputation.variable_domain";
    /// Detail `multiple_imputation.within_singular`.
    pub const WITHIN_SINGULAR: &str = "multiple_imputation.within_singular";
}

/// Details in the `mz_transport` family.
pub mod mz_transport {
    /// Detail `mz_transport.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "mz_transport.bounds_exceeded";
    /// Detail `mz_transport.budget`.
    pub const BUDGET: &str = "mz_transport.budget";
    /// Detail `mz_transport.checked_obstruction`.
    pub const CHECKED_OBSTRUCTION: &str = "mz_transport.checked_obstruction";
    /// Detail `mz_transport.empirical_counts_required`.
    pub const EMPIRICAL_COUNTS_REQUIRED: &str = "mz_transport.empirical_counts_required";
    /// Detail `mz_transport.fabricated_joint`.
    pub const FABRICATED_JOINT: &str = "mz_transport.fabricated_joint";
    /// Detail `mz_transport.functional_graph_mismatch`.
    pub const FUNCTIONAL_GRAPH_MISMATCH: &str = "mz_transport.functional_graph_mismatch";
    /// Detail `mz_transport.invalid_catalog`.
    pub const INVALID_CATALOG: &str = "mz_transport.invalid_catalog";
    /// Detail `mz_transport.invalid_derivation`.
    pub const INVALID_DERIVATION: &str = "mz_transport.invalid_derivation";
    /// Detail `mz_transport.invalid_obstruction`.
    pub const INVALID_OBSTRUCTION: &str = "mz_transport.invalid_obstruction";
    /// Detail `mz_transport.invalid_query`.
    pub const INVALID_QUERY: &str = "mz_transport.invalid_query";
    /// Detail `mz_transport.missing_joint_regime`.
    pub const MISSING_JOINT_REGIME: &str = "mz_transport.missing_joint_regime";
    /// Detail `mz_transport.search_incomplete`.
    pub const SEARCH_INCOMPLETE: &str = "mz_transport.search_incomplete";
}

/// Details in the `native_attempt` family.
pub mod native_attempt {
    /// Detail `native_attempt.invalid_callable`.
    pub const INVALID_CALLABLE: &str = "native_attempt.invalid_callable";
}

/// Details in the `native_claims` family.
pub mod native_claims {
    /// Detail `native_claims.calibration_not_supported`.
    pub const CALIBRATION_NOT_SUPPORTED: &str = "native_claims.calibration_not_supported";
    /// Detail `native_claims.contract_mismatch`.
    pub const CONTRACT_MISMATCH: &str = "native_claims.contract_mismatch";
    /// Detail `native_claims.grid_mismatch`.
    pub const GRID_MISMATCH: &str = "native_claims.grid_mismatch";
    /// Detail `native_claims.invalid_context`.
    pub const INVALID_CONTEXT: &str = "native_claims.invalid_context";
    /// Detail `native_claims.joint_law_invalid`.
    pub const JOINT_LAW_INVALID: &str = "native_claims.joint_law_invalid";
    /// Detail `native_claims.native_state_unavailable`.
    pub const NATIVE_STATE_UNAVAILABLE: &str = "native_claims.native_state_unavailable";
    /// Detail `native_claims.no_source_requirement`.
    pub const NO_SOURCE_REQUIREMENT: &str = "native_claims.no_source_requirement";
    /// Detail `native_claims.no_supported_coordinate`.
    pub const NO_SUPPORTED_COORDINATE: &str = "native_claims.no_supported_coordinate";
    /// Detail `native_claims.point_support_missing`.
    pub const POINT_SUPPORT_MISSING: &str = "native_claims.point_support_missing";
    /// Detail `native_claims.program_mismatch`.
    pub const PROGRAM_MISMATCH: &str = "native_claims.program_mismatch";
    /// Detail `native_claims.projection_mismatch`.
    pub const PROJECTION_MISMATCH: &str = "native_claims.projection_mismatch";
    /// Detail `native_claims.provenance_mismatch`.
    pub const PROVENANCE_MISMATCH: &str = "native_claims.provenance_mismatch";
    /// Detail `native_claims.response_not_point_identified`.
    pub const RESPONSE_NOT_POINT_IDENTIFIED: &str = "native_claims.response_not_point_identified";
    /// Detail `native_claims.source_not_supplied`.
    pub const SOURCE_NOT_SUPPLIED: &str = "native_claims.source_not_supplied";
    /// Detail `native_claims.units_conflict`.
    pub const UNITS_CONFLICT: &str = "native_claims.units_conflict";
    /// Detail `native_claims.unknown_variable`.
    pub const UNKNOWN_VARIABLE: &str = "native_claims.unknown_variable";
    /// Detail `native_claims.unsupported_estimand`.
    pub const UNSUPPORTED_ESTIMAND: &str = "native_claims.unsupported_estimand";
}

/// Details in the `native_distribution` family.
pub mod native_distribution {
    /// Detail `native_distribution.authority_required`.
    pub const AUTHORITY_REQUIRED: &str = "native_distribution.authority_required";
}

/// Details in the `nested_markov` family.
pub mod nested_markov {
    /// Detail `nested_markov.bayesian`.
    pub const BAYESIAN: &str = "nested_markov.bayesian";
    /// Detail `nested_markov.bayesian_artifact`.
    pub const BAYESIAN_ARTIFACT: &str = "nested_markov.bayesian_artifact";
    /// Detail `nested_markov.bayesian_artifact_limits`.
    pub const BAYESIAN_ARTIFACT_LIMITS: &str = "nested_markov.bayesian_artifact_limits";
    /// Detail `nested_markov.bayesian_identity_mismatch`.
    pub const BAYESIAN_IDENTITY_MISMATCH: &str = "nested_markov.bayesian_identity_mismatch";
    /// Detail `nested_markov.bayesian_invalid_options`.
    pub const BAYESIAN_INVALID_OPTIONS: &str = "nested_markov.bayesian_invalid_options";
    /// Detail `nested_markov.bayesian_invalid_prior`.
    pub const BAYESIAN_INVALID_PRIOR: &str = "nested_markov.bayesian_invalid_prior";
    /// Detail `nested_markov.bayesian_sampler_scope`.
    pub const BAYESIAN_SAMPLER_SCOPE: &str = "nested_markov.bayesian_sampler_scope";
    /// Detail `nested_markov.cancelled`.
    pub const CANCELLED: &str = "nested_markov.cancelled";
    /// Detail `nested_markov.constraint_violated`.
    pub const CONSTRAINT_VIOLATED: &str = "nested_markov.constraint_violated";
    /// Detail `nested_markov.fisher_artifact_limits`.
    pub const FISHER_ARTIFACT_LIMITS: &str = "nested_markov.fisher_artifact_limits";
    /// Detail `nested_markov.fisher_identity_mismatch`.
    pub const FISHER_IDENTITY_MISMATCH: &str = "nested_markov.fisher_identity_mismatch";
    /// Detail `nested_markov.fisher_invalid_covariance`.
    pub const FISHER_INVALID_COVARIANCE: &str = "nested_markov.fisher_invalid_covariance";
    /// Detail `nested_markov.fisher_invalid_level`.
    pub const FISHER_INVALID_LEVEL: &str = "nested_markov.fisher_invalid_level";
    /// Detail `nested_markov.fisher_invalid_options`.
    pub const FISHER_INVALID_OPTIONS: &str = "nested_markov.fisher_invalid_options";
    /// Detail `nested_markov.fisher_sampling_design`.
    pub const FISHER_SAMPLING_DESIGN: &str = "nested_markov.fisher_sampling_design";
    /// Detail `nested_markov.fisher_singular_information`.
    pub const FISHER_SINGULAR_INFORMATION: &str = "nested_markov.fisher_singular_information";
    /// Detail `nested_markov.fit_not_converged`.
    pub const FIT_NOT_CONVERGED: &str = "nested_markov.fit_not_converged";
    /// Detail `nested_markov.fitted_boundary`.
    pub const FITTED_BOUNDARY: &str = "nested_markov.fitted_boundary";
    /// Detail `nested_markov.identification_unavailable`.
    pub const IDENTIFICATION_UNAVAILABLE: &str = "nested_markov.identification_unavailable";
    /// Detail `nested_markov.invalid_counts`.
    pub const INVALID_COUNTS: &str = "nested_markov.invalid_counts";
    /// Detail `nested_markov.invalid_input`.
    pub const INVALID_INPUT: &str = "nested_markov.invalid_input";
    /// Detail `nested_markov.invalid_options`.
    pub const INVALID_OPTIONS: &str = "nested_markov.invalid_options";
    /// Detail `nested_markov.iteration_bound`.
    pub const ITERATION_BOUND: &str = "nested_markov.iteration_bound";
    /// Detail `nested_markov.measured_scope`.
    pub const MEASURED_SCOPE: &str = "nested_markov.measured_scope";
    /// Detail `nested_markov.native_hook_unavailable`.
    pub const NATIVE_HOOK_UNAVAILABLE: &str = "nested_markov.native_hook_unavailable";
    /// Detail `nested_markov.not_normalized`.
    pub const NOT_NORMALIZED: &str = "nested_markov.not_normalized";
    /// Detail `nested_markov.outside_binary_scope`.
    pub const OUTSIDE_BINARY_SCOPE: &str = "nested_markov.outside_binary_scope";
    /// Detail `nested_markov.positivity`.
    pub const POSITIVITY: &str = "nested_markov.positivity";
    /// Detail `nested_markov.producer_missing`.
    pub const PRODUCER_MISSING: &str = "nested_markov.producer_missing";
}

/// Details in the `nonlinear_mediation` family.
pub mod nonlinear_mediation {
    /// Detail `nonlinear_mediation.cancelled`.
    pub const CANCELLED: &str = "nonlinear_mediation.cancelled";
    /// Detail `nonlinear_mediation.confounding`.
    pub const CONFOUNDING: &str = "nonlinear_mediation.confounding";
    /// Detail `nonlinear_mediation.cross_world_independence_not_declared`.
    pub const CROSS_WORLD_INDEPENDENCE_NOT_DECLARED: &str = "nonlinear_mediation.cross_world_independence_not_declared";
    /// Detail `nonlinear_mediation.data_required`.
    pub const DATA_REQUIRED: &str = "nonlinear_mediation.data_required";
    /// Detail `nonlinear_mediation.degenerate_mediator`.
    pub const DEGENERATE_MEDIATOR: &str = "nonlinear_mediation.degenerate_mediator";
    /// Detail `nonlinear_mediation.degenerate_mediator_residual`.
    pub const DEGENERATE_MEDIATOR_RESIDUAL: &str = "nonlinear_mediation.degenerate_mediator_residual";
    /// Detail `nonlinear_mediation.degenerate_outcome`.
    pub const DEGENERATE_OUTCOME: &str = "nonlinear_mediation.degenerate_outcome";
    /// Detail `nonlinear_mediation.degree_above_limit`.
    pub const DEGREE_ABOVE_LIMIT: &str = "nonlinear_mediation.degree_above_limit";
    /// Detail `nonlinear_mediation.export_budget_exceeded`.
    pub const EXPORT_BUDGET_EXCEEDED: &str = "nonlinear_mediation.export_budget_exceeded";
    /// Detail `nonlinear_mediation.identity_violation`.
    pub const IDENTITY_VIOLATION: &str = "nonlinear_mediation.identity_violation";
    /// Detail `nonlinear_mediation.integration_error`.
    pub const INTEGRATION_ERROR: &str = "nonlinear_mediation.integration_error";
    /// Detail `nonlinear_mediation.interventional_effects_not_supported`.
    pub const INTERVENTIONAL_EFFECTS_NOT_SUPPORTED: &str = "nonlinear_mediation.interventional_effects_not_supported";
    /// Detail `nonlinear_mediation.invalid_degree`.
    pub const INVALID_DEGREE: &str = "nonlinear_mediation.invalid_degree";
    /// Detail `nonlinear_mediation.invalid_nodes`.
    pub const INVALID_NODES: &str = "nonlinear_mediation.invalid_nodes";
    /// Detail `nonlinear_mediation.invalid_request`.
    pub const INVALID_REQUEST: &str = "nonlinear_mediation.invalid_request";
    /// Detail `nonlinear_mediation.invalid_support_bound`.
    pub const INVALID_SUPPORT_BOUND: &str = "nonlinear_mediation.invalid_support_bound";
    /// Detail `nonlinear_mediation.invalid_tolerance`.
    pub const INVALID_TOLERANCE: &str = "nonlinear_mediation.invalid_tolerance";
    /// Detail `nonlinear_mediation.length_mismatch`.
    pub const LENGTH_MISMATCH: &str = "nonlinear_mediation.length_mismatch";
    /// Detail `nonlinear_mediation.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "nonlinear_mediation.limits_exceeded";
    /// Detail `nonlinear_mediation.memory_budget_exceeded`.
    pub const MEMORY_BUDGET_EXCEEDED: &str = "nonlinear_mediation.memory_budget_exceeded";
    /// Detail `nonlinear_mediation.nodes_above_limit`.
    pub const NODES_ABOVE_LIMIT: &str = "nonlinear_mediation.nodes_above_limit";
    /// Detail `nonlinear_mediation.non_finite_input`.
    pub const NON_FINITE_INPUT: &str = "nonlinear_mediation.non_finite_input";
    /// Detail `nonlinear_mediation.non_finite_option`.
    pub const NON_FINITE_OPTION: &str = "nonlinear_mediation.non_finite_option";
    /// Detail `nonlinear_mediation.overlap`.
    pub const OVERLAP: &str = "nonlinear_mediation.overlap";
    /// Detail `nonlinear_mediation.quadrature_nodes_not_converged`.
    pub const QUADRATURE_NODES_NOT_CONVERGED: &str = "nonlinear_mediation.quadrature_nodes_not_converged";
    /// Detail `nonlinear_mediation.quadrature_weights_invalid`.
    pub const QUADRATURE_WEIGHTS_INVALID: &str = "nonlinear_mediation.quadrature_weights_invalid";
    /// Detail `nonlinear_mediation.rank_deficient`.
    pub const RANK_DEFICIENT: &str = "nonlinear_mediation.rank_deficient";
    /// Detail `nonlinear_mediation.report_replay_mismatch`.
    pub const REPORT_REPLAY_MISMATCH: &str = "nonlinear_mediation.report_replay_mismatch";
    /// Detail `nonlinear_mediation.too_few_rows`.
    pub const TOO_FEW_ROWS: &str = "nonlinear_mediation.too_few_rows";
    /// Detail `nonlinear_mediation.treatment_induced_confounding`.
    pub const TREATMENT_INDUCED_CONFOUNDING: &str = "nonlinear_mediation.treatment_induced_confounding";
    /// Detail `nonlinear_mediation.treatment_not_binary`.
    pub const TREATMENT_NOT_BINARY: &str = "nonlinear_mediation.treatment_not_binary";
    /// Detail `nonlinear_mediation.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "nonlinear_mediation.wrong_contract";
}

/// Details in the `observation_query` family.
pub mod observation_query {
    /// Detail `observation_query.arm_mismatch`.
    pub const ARM_MISMATCH: &str = "observation_query.arm_mismatch";
    /// Detail `observation_query.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "observation_query.artifact_invalid";
    /// Detail `observation_query.artifact_refused`.
    pub const ARTIFACT_REFUSED: &str = "observation_query.artifact_refused";
    /// Detail `observation_query.blank_field`.
    pub const BLANK_FIELD: &str = "observation_query.blank_field";
    /// Detail `observation_query.cancelled`.
    pub const CANCELLED: &str = "observation_query.cancelled";
    /// Detail `observation_query.construct_mismatch`.
    pub const CONSTRUCT_MISMATCH: &str = "observation_query.construct_mismatch";
    /// Detail `observation_query.construct_observed_directly`.
    pub const CONSTRUCT_OBSERVED_DIRECTLY: &str = "observation_query.construct_observed_directly";
    /// Detail `observation_query.count_fitted_bounds_exceeded`.
    pub const COUNT_FITTED_BOUNDS_EXCEEDED: &str = "observation_query.count_fitted_bounds_exceeded";
    /// Detail `observation_query.count_fitted_refused`.
    pub const COUNT_FITTED_REFUSED: &str = "observation_query.count_fitted_refused";
    /// Detail `observation_query.declared_model_has_no_sampling_interval`.
    pub const DECLARED_MODEL_HAS_NO_SAMPLING_INTERVAL: &str = "observation_query.declared_model_has_no_sampling_interval";
    /// Detail `observation_query.differential_error_unsupported`.
    pub const DIFFERENTIAL_ERROR_UNSUPPORTED: &str = "observation_query.differential_error_unsupported";
    /// Detail `observation_query.evidence_covariance_singular`.
    pub const EVIDENCE_COVARIANCE_SINGULAR: &str = "observation_query.evidence_covariance_singular";
    /// Detail `observation_query.indicator_kind`.
    pub const INDICATOR_KIND: &str = "observation_query.indicator_kind";
    /// Detail `observation_query.indicator_mismatch`.
    pub const INDICATOR_MISMATCH: &str = "observation_query.indicator_mismatch";
    /// Detail `observation_query.intervention_in_conditioning_slot`.
    pub const INTERVENTION_IN_CONDITIONING_SLOT: &str = "observation_query.intervention_in_conditioning_slot";
    /// Detail `observation_query.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "observation_query.invalid_declaration";
    /// Detail `observation_query.invalid_interval`.
    pub const INVALID_INTERVAL: &str = "observation_query.invalid_interval";
    /// Detail `observation_query.invalid_request`.
    pub const INVALID_REQUEST: &str = "observation_query.invalid_request";
    /// Detail `observation_query.level_unreachable`.
    pub const LEVEL_UNREACHABLE: &str = "observation_query.level_unreachable";
    /// Detail `observation_query.levels_for_continuous_model`.
    pub const LEVELS_FOR_CONTINUOUS_MODEL: &str = "observation_query.levels_for_continuous_model";
    /// Detail `observation_query.measured_as_construct`.
    pub const MEASURED_AS_CONSTRUCT: &str = "observation_query.measured_as_construct";
    /// Detail `observation_query.measured_background`.
    pub const MEASURED_BACKGROUND: &str = "observation_query.measured_background";
    /// Detail `observation_query.measured_signal_not_redrawn`.
    pub const MEASURED_SIGNAL_NOT_REDRAWN: &str = "observation_query.measured_signal_not_redrawn";
    /// Detail `observation_query.measurement_route`.
    pub const MEASUREMENT_ROUTE: &str = "observation_query.measurement_route";
    /// Detail `observation_query.mixed_conditioning_and_intervention`.
    pub const MIXED_CONDITIONING_AND_INTERVENTION: &str = "observation_query.mixed_conditioning_and_intervention";
    /// Detail `observation_query.multiple_measured_signals`.
    pub const MULTIPLE_MEASURED_SIGNALS: &str = "observation_query.multiple_measured_signals";
    /// Detail `observation_query.no_calibrated_probability`.
    pub const NO_CALIBRATED_PROBABILITY: &str = "observation_query.no_calibrated_probability";
    /// Detail `observation_query.no_parameter_draws`.
    pub const NO_PARAMETER_DRAWS: &str = "observation_query.no_parameter_draws";
    /// Detail `observation_query.outcome_assigned`.
    pub const OUTCOME_ASSIGNED: &str = "observation_query.outcome_assigned";
    /// Detail `observation_query.population_mismatch`.
    pub const POPULATION_MISMATCH: &str = "observation_query.population_mismatch";
    /// Detail `observation_query.positive_level`.
    pub const POSITIVE_LEVEL: &str = "observation_query.positive_level";
    /// Detail `observation_query.real_values_for_finite_model`.
    pub const REAL_VALUES_FOR_FINITE_MODEL: &str = "observation_query.real_values_for_finite_model";
    /// Detail `observation_query.regime_mismatch`.
    pub const REGIME_MISMATCH: &str = "observation_query.regime_mismatch";
    /// Detail `observation_query.repeated_slot`.
    pub const REPEATED_SLOT: &str = "observation_query.repeated_slot";
    /// Detail `observation_query.request_unsupported`.
    pub const REQUEST_UNSUPPORTED: &str = "observation_query.request_unsupported";
    /// Detail `observation_query.unknown_level`.
    pub const UNKNOWN_LEVEL: &str = "observation_query.unknown_level";
    /// Detail `observation_query.unknown_quantity`.
    pub const UNKNOWN_QUANTITY: &str = "observation_query.unknown_quantity";
    /// Detail `observation_query.unknown_variable`.
    pub const UNKNOWN_VARIABLE: &str = "observation_query.unknown_variable";
    /// Detail `observation_query.value_not_finite`.
    pub const VALUE_NOT_FINITE: &str = "observation_query.value_not_finite";
    /// Detail `observation_query.zero_probability_evidence`.
    pub const ZERO_PROBABILITY_EVIDENCE: &str = "observation_query.zero_probability_evidence";
}

/// Details in the `paradigm_parity` family.
pub mod paradigm_parity {
    /// Detail `paradigm_parity.paradigm_not_supported`.
    pub const PARADIGM_NOT_SUPPORTED: &str = "paradigm_parity.paradigm_not_supported";
}

/// Details in the `parent_set_effect` family.
pub mod parent_set_effect {
    /// Detail `parent_set_effect.conflict_resolution_budget`.
    pub const CONFLICT_RESOLUTION_BUDGET: &str = "parent_set_effect.conflict_resolution_budget";
    /// Detail `parent_set_effect.discovery_failed`.
    pub const DISCOVERY_FAILED: &str = "parent_set_effect.discovery_failed";
    /// Detail `parent_set_effect.knowledge_invalid`.
    pub const KNOWLEDGE_INVALID: &str = "parent_set_effect.knowledge_invalid";
    /// Detail `parent_set_effect.log_linear_input_not_positive`.
    pub const LOG_LINEAR_INPUT_NOT_POSITIVE: &str = "parent_set_effect.log_linear_input_not_positive";
    /// Detail `parent_set_effect.outcome_precedes_treatment`.
    pub const OUTCOME_PRECEDES_TREATMENT: &str = "parent_set_effect.outcome_precedes_treatment";
    /// Detail `parent_set_effect.parent_limit_bound`.
    pub const PARENT_LIMIT_BOUND: &str = "parent_set_effect.parent_limit_bound";
    /// Detail `parent_set_effect.search_not_finished`.
    pub const SEARCH_NOT_FINISHED: &str = "parent_set_effect.search_not_finished";
    /// Detail `parent_set_effect.source_invalid`.
    pub const SOURCE_INVALID: &str = "parent_set_effect.source_invalid";
    /// Detail `parent_set_effect.source_not_canonical`.
    pub const SOURCE_NOT_CANONICAL: &str = "parent_set_effect.source_not_canonical";
    /// Detail `parent_set_effect.structure_class`.
    pub const STRUCTURE_CLASS: &str = "parent_set_effect.structure_class";
    /// Detail `parent_set_effect.transformation_undeclared`.
    pub const TRANSFORMATION_UNDECLARED: &str = "parent_set_effect.transformation_undeclared";
    /// Detail `parent_set_effect.treatment_orientation_depends_on_conflict`.
    pub const TREATMENT_ORIENTATION_DEPENDS_ON_CONFLICT: &str = "parent_set_effect.treatment_orientation_depends_on_conflict";
    /// Detail `parent_set_effect.treatment_parents_ambiguous`.
    pub const TREATMENT_PARENTS_AMBIGUOUS: &str = "parent_set_effect.treatment_parents_ambiguous";
}

/// Details in the `penalized_posterior` family.
pub mod penalized_posterior {
    /// Detail `penalized_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "penalized_posterior.band_not_constructed";
    /// Detail `penalized_posterior.empty_scores`.
    pub const EMPTY_SCORES: &str = "penalized_posterior.empty_scores";
    /// Detail `penalized_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "penalized_posterior.external_power_prior";
    /// Detail `penalized_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "penalized_posterior.mapped_prior";
    /// Detail `penalized_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "penalized_posterior.non_positive_scale";
}

/// Details in the `penalized_propensity` family.
pub mod penalized_propensity {
    /// Detail `penalized_propensity.cancelled`.
    pub const CANCELLED: &str = "penalized_propensity.cancelled";
    /// Detail `penalized_propensity.effect_posterior`.
    pub const EFFECT_POSTERIOR: &str = "penalized_propensity.effect_posterior";
    /// Detail `penalized_propensity.fallback_not_supported`.
    pub const FALLBACK_NOT_SUPPORTED: &str = "penalized_propensity.fallback_not_supported";
    /// Detail `penalized_propensity.fallback_with_penalty`.
    pub const FALLBACK_WITH_PENALTY: &str = "penalized_propensity.fallback_with_penalty";
    /// Detail `penalized_propensity.grid_allocation`.
    pub const GRID_ALLOCATION: &str = "penalized_propensity.grid_allocation";
    /// Detail `penalized_propensity.invalid_penalty`.
    pub const INVALID_PENALTY: &str = "penalized_propensity.invalid_penalty";
    /// Detail `penalized_propensity.scope`.
    pub const SCOPE: &str = "penalized_propensity.scope";
    /// Detail `penalized_propensity.selection_not_supported`.
    pub const SELECTION_NOT_SUPPORTED: &str = "penalized_propensity.selection_not_supported";
}

/// Details in the `policy_value` family.
pub mod policy_value {
    /// Detail `policy_value.action_not_populated`.
    pub const ACTION_NOT_POPULATED: &str = "policy_value.action_not_populated";
    /// Detail `policy_value.action_outside_support`.
    pub const ACTION_OUTSIDE_SUPPORT: &str = "policy_value.action_outside_support";
    /// Detail `policy_value.action_propensity_below_floor`.
    pub const ACTION_PROPENSITY_BELOW_FLOOR: &str = "policy_value.action_propensity_below_floor";
    /// Detail `policy_value.cancelled`.
    pub const CANCELLED: &str = "policy_value.cancelled";
    /// Detail `policy_value.evaluation_fraction_missing`.
    pub const EVALUATION_FRACTION_MISSING: &str = "policy_value.evaluation_fraction_missing";
    /// Detail `policy_value.evaluation_missing`.
    pub const EVALUATION_MISSING: &str = "policy_value.evaluation_missing";
    /// Detail `policy_value.evaluation_unknown`.
    pub const EVALUATION_UNKNOWN: &str = "policy_value.evaluation_unknown";
    /// Detail `policy_value.invalid_data`.
    pub const INVALID_DATA: &str = "policy_value.invalid_data";
    /// Detail `policy_value.invalid_evaluation_fraction`.
    pub const INVALID_EVALUATION_FRACTION: &str = "policy_value.invalid_evaluation_fraction";
    /// Detail `policy_value.invalid_folds`.
    pub const INVALID_FOLDS: &str = "policy_value.invalid_folds";
    /// Detail `policy_value.invalid_known_propensity`.
    pub const INVALID_KNOWN_PROPENSITY: &str = "policy_value.invalid_known_propensity";
    /// Detail `policy_value.invalid_propensity_floor`.
    pub const INVALID_PROPENSITY_FLOOR: &str = "policy_value.invalid_propensity_floor";
    /// Detail `policy_value.invalid_request`.
    pub const INVALID_REQUEST: &str = "policy_value.invalid_request";
    /// Detail `policy_value.invalid_support`.
    pub const INVALID_SUPPORT: &str = "policy_value.invalid_support";
    /// Detail `policy_value.objective_missing`.
    pub const OBJECTIVE_MISSING: &str = "policy_value.objective_missing";
    /// Detail `policy_value.objective_unknown`.
    pub const OBJECTIVE_UNKNOWN: &str = "policy_value.objective_unknown";
    /// Detail `policy_value.policy_actions_missing`.
    pub const POLICY_ACTIONS_MISSING: &str = "policy_value.policy_actions_missing";
    /// Detail `policy_value.policy_learned_on_evaluation_rows`.
    pub const POLICY_LEARNED_ON_EVALUATION_ROWS: &str = "policy_value.policy_learned_on_evaluation_rows";
    /// Detail `policy_value.policy_origin_missing`.
    pub const POLICY_ORIGIN_MISSING: &str = "policy_value.policy_origin_missing";
    /// Detail `policy_value.policy_origin_unknown`.
    pub const POLICY_ORIGIN_UNKNOWN: &str = "policy_value.policy_origin_unknown";
    /// Detail `policy_value.reference_outside_support`.
    pub const REFERENCE_OUTSIDE_SUPPORT: &str = "policy_value.reference_outside_support";
    /// Detail `policy_value.sparse_action`.
    pub const SPARSE_ACTION: &str = "policy_value.sparse_action";
    /// Detail `policy_value.support_missing`.
    pub const SUPPORT_MISSING: &str = "policy_value.support_missing";
    /// Detail `policy_value.too_few_evaluation_rows`.
    pub const TOO_FEW_EVALUATION_ROWS: &str = "policy_value.too_few_evaluation_rows";
    /// Detail `policy_value.too_few_rows`.
    pub const TOO_FEW_ROWS: &str = "policy_value.too_few_rows";
    /// Detail `policy_value.training_split_missing_action`.
    pub const TRAINING_SPLIT_MISSING_ACTION: &str = "policy_value.training_split_missing_action";
    /// Detail `policy_value.undeclared_level`.
    pub const UNDECLARED_LEVEL: &str = "policy_value.undeclared_level";
}

/// Details in the `predictive_assessment` family.
pub mod predictive_assessment {
    /// Detail `predictive_assessment.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "predictive_assessment.bounds_exceeded";
    /// Detail `predictive_assessment.cancelled`.
    pub const CANCELLED: &str = "predictive_assessment.cancelled";
    /// Detail `predictive_assessment.design_invalid`.
    pub const DESIGN_INVALID: &str = "predictive_assessment.design_invalid";
    /// Detail `predictive_assessment.diagnostic_not_defined`.
    pub const DIAGNOSTIC_NOT_DEFINED: &str = "predictive_assessment.diagnostic_not_defined";
    /// Detail `predictive_assessment.horizon_mismatch`.
    pub const HORIZON_MISMATCH: &str = "predictive_assessment.horizon_mismatch";
    /// Detail `predictive_assessment.iid_not_supported_for_dependent_design`.
    pub const IID_NOT_SUPPORTED_FOR_DEPENDENT_DESIGN: &str = "predictive_assessment.iid_not_supported_for_dependent_design";
    /// Detail `predictive_assessment.in_sample`.
    pub const IN_SAMPLE: &str = "predictive_assessment.in_sample";
    /// Detail `predictive_assessment.interval_dependence_undeclared`.
    pub const INTERVAL_DEPENDENCE_UNDECLARED: &str = "predictive_assessment.interval_dependence_undeclared";
    /// Detail `predictive_assessment.interval_law_family_mixed`.
    pub const INTERVAL_LAW_FAMILY_MIXED: &str = "predictive_assessment.interval_law_family_mixed";
    /// Detail `predictive_assessment.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "predictive_assessment.interval_not_supported";
    /// Detail `predictive_assessment.interval_request_invalid`.
    pub const INTERVAL_REQUEST_INVALID: &str = "predictive_assessment.interval_request_invalid";
    /// Detail `predictive_assessment.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "predictive_assessment.invalid_artifact";
    /// Detail `predictive_assessment.invalid_law`.
    pub const INVALID_LAW: &str = "predictive_assessment.invalid_law";
    /// Detail `predictive_assessment.invalid_observations`.
    pub const INVALID_OBSERVATIONS: &str = "predictive_assessment.invalid_observations";
    /// Detail `predictive_assessment.law_callback_not_built`.
    pub const LAW_CALLBACK_NOT_BUILT: &str = "predictive_assessment.law_callback_not_built";
    /// Detail `predictive_assessment.log_score_infinite`.
    pub const LOG_SCORE_INFINITE: &str = "predictive_assessment.log_score_infinite";
    /// Detail `predictive_assessment.not_a_causal_claim`.
    pub const NOT_A_CAUSAL_CLAIM: &str = "predictive_assessment.not_a_causal_claim";
    /// Detail `predictive_assessment.outcome_kind`.
    pub const OUTCOME_KIND: &str = "predictive_assessment.outcome_kind";
    /// Detail `predictive_assessment.overlap_unknown`.
    pub const OVERLAP_UNKNOWN: &str = "predictive_assessment.overlap_unknown";
    /// Detail `predictive_assessment.provider_identity_incomplete`.
    pub const PROVIDER_IDENTITY_INCOMPLETE: &str = "predictive_assessment.provider_identity_incomplete";
    /// Detail `predictive_assessment.report_replay_mismatch`.
    pub const REPORT_REPLAY_MISMATCH: &str = "predictive_assessment.report_replay_mismatch";
    /// Detail `predictive_assessment.score_not_defined`.
    pub const SCORE_NOT_DEFINED: &str = "predictive_assessment.score_not_defined";
    /// Detail `predictive_assessment.score_not_probability`.
    pub const SCORE_NOT_PROBABILITY: &str = "predictive_assessment.score_not_probability";
    /// Detail `predictive_assessment.serial_lag_below_horizon`.
    pub const SERIAL_LAG_BELOW_HORIZON: &str = "predictive_assessment.serial_lag_below_horizon";
    /// Detail `predictive_assessment.target_mismatch`.
    pub const TARGET_MISMATCH: &str = "predictive_assessment.target_mismatch";
    /// Detail `predictive_assessment.too_few_clusters`.
    pub const TOO_FEW_CLUSTERS: &str = "predictive_assessment.too_few_clusters";
    /// Detail `predictive_assessment.unknown_source`.
    pub const UNKNOWN_SOURCE: &str = "predictive_assessment.unknown_source";
    /// Detail `predictive_assessment.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "predictive_assessment.wrong_contract";
}

/// Details in the `prior_signal` family.
pub mod prior_signal {
    /// Detail `prior_signal.adapted_prior_not_draws`.
    pub const ADAPTED_PRIOR_NOT_DRAWS: &str = "prior_signal.adapted_prior_not_draws";
    /// Detail `prior_signal.artifact_not_in_catalog`.
    pub const ARTIFACT_NOT_IN_CATALOG: &str = "prior_signal.artifact_not_in_catalog";
    /// Detail `prior_signal.coefficient_prior_not_signal`.
    pub const COEFFICIENT_PRIOR_NOT_SIGNAL: &str = "prior_signal.coefficient_prior_not_signal";
    /// Detail `prior_signal.duplicate_source`.
    pub const DUPLICATE_SOURCE: &str = "prior_signal.duplicate_source";
    /// Detail `prior_signal.incompatible_artifact`.
    pub const INCOMPATIBLE_ARTIFACT: &str = "prior_signal.incompatible_artifact";
    /// Detail `prior_signal.incomplete_posterior`.
    pub const INCOMPLETE_POSTERIOR: &str = "prior_signal.incomplete_posterior";
    /// Detail `prior_signal.invalid_parameter`.
    pub const INVALID_PARAMETER: &str = "prior_signal.invalid_parameter";
    /// Detail `prior_signal.invalid_request`.
    pub const INVALID_REQUEST: &str = "prior_signal.invalid_request";
    /// Detail `prior_signal.lineage_missing`.
    pub const LINEAGE_MISSING: &str = "prior_signal.lineage_missing";
    /// Detail `prior_signal.named_state_draws_required`.
    pub const NAMED_STATE_DRAWS_REQUIRED: &str = "prior_signal.named_state_draws_required";
    /// Detail `prior_signal.no_sources`.
    pub const NO_SOURCES: &str = "prior_signal.no_sources";
    /// Detail `prior_signal.nonfinite_mapped_prior`.
    pub const NONFINITE_MAPPED_PRIOR: &str = "prior_signal.nonfinite_mapped_prior";
    /// Detail `prior_signal.observation_lineage_missing`.
    pub const OBSERVATION_LINEAGE_MISSING: &str = "prior_signal.observation_lineage_missing";
    /// Detail `prior_signal.observation_overlap_unknown`.
    pub const OBSERVATION_OVERLAP_UNKNOWN: &str = "prior_signal.observation_overlap_unknown";
    /// Detail `prior_signal.observation_reuse`.
    pub const OBSERVATION_REUSE: &str = "prior_signal.observation_reuse";
    /// Detail `prior_signal.original_posterior_unidentified`.
    pub const ORIGINAL_POSTERIOR_UNIDENTIFIED: &str = "prior_signal.original_posterior_unidentified";
    /// Detail `prior_signal.posterior_draws_required`.
    pub const POSTERIOR_DRAWS_REQUIRED: &str = "prior_signal.posterior_draws_required";
    /// Detail `prior_signal.prior_identity_mismatch`.
    pub const PRIOR_IDENTITY_MISMATCH: &str = "prior_signal.prior_identity_mismatch";
    /// Detail `prior_signal.prior_projection_mismatch`.
    pub const PRIOR_PROJECTION_MISMATCH: &str = "prior_signal.prior_projection_mismatch";
    /// Detail `prior_signal.prior_sources_overlap`.
    pub const PRIOR_SOURCES_OVERLAP: &str = "prior_signal.prior_sources_overlap";
    /// Detail `prior_signal.resolution_invalid`.
    pub const RESOLUTION_INVALID: &str = "prior_signal.resolution_invalid";
    /// Detail `prior_signal.signal_family_mismatch`.
    pub const SIGNAL_FAMILY_MISMATCH: &str = "prior_signal.signal_family_mismatch";
    /// Detail `prior_signal.source_not_in_catalog_or_duplicate`.
    pub const SOURCE_NOT_IN_CATALOG_OR_DUPLICATE: &str = "prior_signal.source_not_in_catalog_or_duplicate";
    /// Detail `prior_signal.source_population_mismatch`.
    pub const SOURCE_POPULATION_MISMATCH: &str = "prior_signal.source_population_mismatch";
    /// Detail `prior_signal.state_quantity_mismatch`.
    pub const STATE_QUANTITY_MISMATCH: &str = "prior_signal.state_quantity_mismatch";
    /// Detail `prior_signal.target_population_mismatch`.
    pub const TARGET_POPULATION_MISMATCH: &str = "prior_signal.target_population_mismatch";
    /// Detail `prior_signal.transport_policy_required`.
    pub const TRANSPORT_POLICY_REQUIRED: &str = "prior_signal.transport_policy_required";
    /// Detail `prior_signal.unlock_list_not_signal`.
    pub const UNLOCK_LIST_NOT_SIGNAL: &str = "prior_signal.unlock_list_not_signal";
    /// Detail `prior_signal.unsupported_signal_provider`.
    pub const UNSUPPORTED_SIGNAL_PROVIDER: &str = "prior_signal.unsupported_signal_provider";
}

/// Details in the `program_binding` family.
pub mod program_binding {
    /// Detail `program_binding.contract_identity_mismatch`.
    pub const CONTRACT_IDENTITY_MISMATCH: &str = "program_binding.contract_identity_mismatch";
    /// Detail `program_binding.dose_grid_changed`.
    pub const DOSE_GRID_CHANGED: &str = "program_binding.dose_grid_changed";
    /// Detail `program_binding.graph_only_identity`.
    pub const GRAPH_ONLY_IDENTITY: &str = "program_binding.graph_only_identity";
    /// Detail `program_binding.invalid_binding`.
    pub const INVALID_BINDING: &str = "program_binding.invalid_binding";
    /// Detail `program_binding.malformed_quantity`.
    pub const MALFORMED_QUANTITY: &str = "program_binding.malformed_quantity";
    /// Detail `program_binding.no_identified_request`.
    pub const NO_IDENTIFIED_REQUEST: &str = "program_binding.no_identified_request";
    /// Detail `program_binding.population_mismatch`.
    pub const POPULATION_MISMATCH: &str = "program_binding.population_mismatch";
    /// Detail `program_binding.quantities_override_mismatch`.
    pub const QUANTITIES_OVERRIDE_MISMATCH: &str = "program_binding.quantities_override_mismatch";
    /// Detail `program_binding.treatment_outcome_substitution`.
    pub const TREATMENT_OUTCOME_SUBSTITUTION: &str = "program_binding.treatment_outcome_substitution";
}

/// Details in the `proposal_arrival` family.
pub mod proposal_arrival {
    /// Detail `proposal_arrival.assignment_mismatch`.
    pub const ASSIGNMENT_MISMATCH: &str = "proposal_arrival.assignment_mismatch";
    /// Detail `proposal_arrival.base_catalog_changed`.
    pub const BASE_CATALOG_CHANGED: &str = "proposal_arrival.base_catalog_changed";
    /// Detail `proposal_arrival.base_catalog_missing`.
    pub const BASE_CATALOG_MISSING: &str = "proposal_arrival.base_catalog_missing";
    /// Detail `proposal_arrival.base_provider_mismatch`.
    pub const BASE_PROVIDER_MISMATCH: &str = "proposal_arrival.base_provider_mismatch";
    /// Detail `proposal_arrival.binding_missing`.
    pub const BINDING_MISSING: &str = "proposal_arrival.binding_missing";
    /// Detail `proposal_arrival.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "proposal_arrival.bounds_exceeded";
    /// Detail `proposal_arrival.cancelled`.
    pub const CANCELLED: &str = "proposal_arrival.cancelled";
    /// Detail `proposal_arrival.candidate_missing`.
    pub const CANDIDATE_MISSING: &str = "proposal_arrival.candidate_missing";
    /// Detail `proposal_arrival.duplicate_original_artifacts`.
    pub const DUPLICATE_ORIGINAL_ARTIFACTS: &str = "proposal_arrival.duplicate_original_artifacts";
    /// Detail `proposal_arrival.expected_proposal_mismatch`.
    pub const EXPECTED_PROPOSAL_MISMATCH: &str = "proposal_arrival.expected_proposal_mismatch";
    /// Detail `proposal_arrival.family_not_supported`.
    pub const FAMILY_NOT_SUPPORTED: &str = "proposal_arrival.family_not_supported";
    /// Detail `proposal_arrival.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "proposal_arrival.identity_mismatch";
    /// Detail `proposal_arrival.independent_joint_study_required`.
    pub const INDEPENDENT_JOINT_STUDY_REQUIRED: &str = "proposal_arrival.independent_joint_study_required";
    /// Detail `proposal_arrival.invalid_request`.
    pub const INVALID_REQUEST: &str = "proposal_arrival.invalid_request";
    /// Detail `proposal_arrival.original_artifacts_required`.
    pub const ORIGINAL_ARTIFACTS_REQUIRED: &str = "proposal_arrival.original_artifacts_required";
    /// Detail `proposal_arrival.original_consumer_refused`.
    pub const ORIGINAL_CONSUMER_REFUSED: &str = "proposal_arrival.original_consumer_refused";
    /// Detail `proposal_arrival.proof_not_bound`.
    pub const PROOF_NOT_BOUND: &str = "proposal_arrival.proof_not_bound";
    /// Detail `proposal_arrival.provider_binding_mismatch`.
    pub const PROVIDER_BINDING_MISMATCH: &str = "proposal_arrival.provider_binding_mismatch";
    /// Detail `proposal_arrival.raw_counts_required`.
    pub const RAW_COUNTS_REQUIRED: &str = "proposal_arrival.raw_counts_required";
    /// Detail `proposal_arrival.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "proposal_arrival.replay_mismatch";
    /// Detail `proposal_arrival.sample_size_mismatch`.
    pub const SAMPLE_SIZE_MISMATCH: &str = "proposal_arrival.sample_size_mismatch";
    /// Detail `proposal_arrival.structural_arrival_refused`.
    pub const STRUCTURAL_ARRIVAL_REFUSED: &str = "proposal_arrival.structural_arrival_refused";
}

/// Details in the `proposal_receipt` family.
pub mod proposal_receipt {
    /// Detail `proposal_receipt.arrival_cancelled`.
    pub const ARRIVAL_CANCELLED: &str = "proposal_receipt.arrival_cancelled";
    /// Detail `proposal_receipt.arrival_empty`.
    pub const ARRIVAL_EMPTY: &str = "proposal_receipt.arrival_empty";
    /// Detail `proposal_receipt.arrival_family_mismatch`.
    pub const ARRIVAL_FAMILY_MISMATCH: &str = "proposal_receipt.arrival_family_mismatch";
    /// Detail `proposal_receipt.arrival_invalid`.
    pub const ARRIVAL_INVALID: &str = "proposal_receipt.arrival_invalid";
    /// Detail `proposal_receipt.arrival_sample_mismatch`.
    pub const ARRIVAL_SAMPLE_MISMATCH: &str = "proposal_receipt.arrival_sample_mismatch";
    /// Detail `proposal_receipt.base_failure_mismatch`.
    pub const BASE_FAILURE_MISMATCH: &str = "proposal_receipt.base_failure_mismatch";
    /// Detail `proposal_receipt.bundle_identity_mismatch`.
    pub const BUNDLE_IDENTITY_MISMATCH: &str = "proposal_receipt.bundle_identity_mismatch";
    /// Detail `proposal_receipt.bundle_inconsistent`.
    pub const BUNDLE_INCONSISTENT: &str = "proposal_receipt.bundle_inconsistent";
    /// Detail `proposal_receipt.candidate_mismatch`.
    pub const CANDIDATE_MISMATCH: &str = "proposal_receipt.candidate_mismatch";
    /// Detail `proposal_receipt.candidate_not_in_bundle`.
    pub const CANDIDATE_NOT_IN_BUNDLE: &str = "proposal_receipt.candidate_not_in_bundle";
    /// Detail `proposal_receipt.candidate_not_in_ranking`.
    pub const CANDIDATE_NOT_IN_RANKING: &str = "proposal_receipt.candidate_not_in_ranking";
    /// Detail `proposal_receipt.candidate_not_in_repair`.
    pub const CANDIDATE_NOT_IN_REPAIR: &str = "proposal_receipt.candidate_not_in_repair";
    /// Detail `proposal_receipt.candidate_not_proposed`.
    pub const CANDIDATE_NOT_PROPOSED: &str = "proposal_receipt.candidate_not_proposed";
    /// Detail `proposal_receipt.candidate_set_mismatch`.
    pub const CANDIDATE_SET_MISMATCH: &str = "proposal_receipt.candidate_set_mismatch";
    /// Detail `proposal_receipt.contract_mismatch`.
    pub const CONTRACT_MISMATCH: &str = "proposal_receipt.contract_mismatch";
    /// Detail `proposal_receipt.contract_required`.
    pub const CONTRACT_REQUIRED: &str = "proposal_receipt.contract_required";
    /// Detail `proposal_receipt.cost_mismatch`.
    pub const COST_MISMATCH: &str = "proposal_receipt.cost_mismatch";
    /// Detail `proposal_receipt.delta_mismatch`.
    pub const DELTA_MISMATCH: &str = "proposal_receipt.delta_mismatch";
    /// Detail `proposal_receipt.derivation_mismatch`.
    pub const DERIVATION_MISMATCH: &str = "proposal_receipt.derivation_mismatch";
    /// Detail `proposal_receipt.duplicate_candidate`.
    pub const DUPLICATE_CANDIDATE: &str = "proposal_receipt.duplicate_candidate";
    /// Detail `proposal_receipt.empty_bundle`.
    pub const EMPTY_BUNDLE: &str = "proposal_receipt.empty_bundle";
    /// Detail `proposal_receipt.hypothetical_not_evidence`.
    pub const HYPOTHETICAL_NOT_EVIDENCE: &str = "proposal_receipt.hypothetical_not_evidence";
    /// Detail `proposal_receipt.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "proposal_receipt.identity_mismatch";
    /// Detail `proposal_receipt.invalid_delta`.
    pub const INVALID_DELTA: &str = "proposal_receipt.invalid_delta";
    /// Detail `proposal_receipt.lineage_mismatch`.
    pub const LINEAGE_MISMATCH: &str = "proposal_receipt.lineage_mismatch";
    /// Detail `proposal_receipt.non_canonical_order`.
    pub const NON_CANONICAL_ORDER: &str = "proposal_receipt.non_canonical_order";
    /// Detail `proposal_receipt.obligation_mismatch`.
    pub const OBLIGATION_MISMATCH: &str = "proposal_receipt.obligation_mismatch";
    /// Detail `proposal_receipt.ranking_digest_mismatch`.
    pub const RANKING_DIGEST_MISMATCH: &str = "proposal_receipt.ranking_digest_mismatch";
    /// Detail `proposal_receipt.repair_digest_mismatch`.
    pub const REPAIR_DIGEST_MISMATCH: &str = "proposal_receipt.repair_digest_mismatch";
    /// Detail `proposal_receipt.signal_mismatch`.
    pub const SIGNAL_MISMATCH: &str = "proposal_receipt.signal_mismatch";
    /// Detail `proposal_receipt.source_digest_mismatch`.
    pub const SOURCE_DIGEST_MISMATCH: &str = "proposal_receipt.source_digest_mismatch";
    /// Detail `proposal_receipt.value_mismatch`.
    pub const VALUE_MISMATCH: &str = "proposal_receipt.value_mismatch";
}

/// Details in the `provider_capability_negotiation` family.
pub mod provider_capability_negotiation {
    /// Detail `provider_capability_negotiation.capability_missing`.
    pub const CAPABILITY_MISSING: &str = "provider_capability_negotiation.capability_missing";
    /// Detail `provider_capability_negotiation.request_mismatch`.
    pub const REQUEST_MISMATCH: &str = "provider_capability_negotiation.request_mismatch";
}

/// Details in the `provider_transport` family.
pub mod provider_transport {
    /// Detail `provider_transport.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "provider_transport.artifact_invalid";
    /// Detail `provider_transport.assumptions_missing`.
    pub const ASSUMPTIONS_MISSING: &str = "provider_transport.assumptions_missing";
    /// Detail `provider_transport.assumptions_not_carried`.
    pub const ASSUMPTIONS_NOT_CARRIED: &str = "provider_transport.assumptions_not_carried";
    /// Detail `provider_transport.blank_field`.
    pub const BLANK_FIELD: &str = "provider_transport.blank_field";
    /// Detail `provider_transport.claim_not_in_target`.
    pub const CLAIM_NOT_IN_TARGET: &str = "provider_transport.claim_not_in_target";
    /// Detail `provider_transport.evidence_missing`.
    pub const EVIDENCE_MISSING: &str = "provider_transport.evidence_missing";
    /// Detail `provider_transport.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "provider_transport.identity_mismatch";
    /// Detail `provider_transport.input_not_external`.
    pub const INPUT_NOT_EXTERNAL: &str = "provider_transport.input_not_external";
    /// Detail `provider_transport.native_target_required`.
    pub const NATIVE_TARGET_REQUIRED: &str = "provider_transport.native_target_required";
    /// Detail `provider_transport.native_trust_claimed`.
    pub const NATIVE_TRUST_CLAIMED: &str = "provider_transport.native_trust_claimed";
    /// Detail `provider_transport.object_invalid`.
    pub const OBJECT_INVALID: &str = "provider_transport.object_invalid";
    /// Detail `provider_transport.object_unsupported`.
    pub const OBJECT_UNSUPPORTED: &str = "provider_transport.object_unsupported";
    /// Detail `provider_transport.population_missing`.
    pub const POPULATION_MISSING: &str = "provider_transport.population_missing";
    /// Detail `provider_transport.population_not_a_source`.
    pub const POPULATION_NOT_A_SOURCE: &str = "provider_transport.population_not_a_source";
    /// Detail `provider_transport.quantity_invalid`.
    pub const QUANTITY_INVALID: &str = "provider_transport.quantity_invalid";
    /// Detail `provider_transport.quantity_not_declared`.
    pub const QUANTITY_NOT_DECLARED: &str = "provider_transport.quantity_not_declared";
    /// Detail `provider_transport.receipt_does_not_cover`.
    pub const RECEIPT_DOES_NOT_COVER: &str = "provider_transport.receipt_does_not_cover";
    /// Detail `provider_transport.regime_missing`.
    pub const REGIME_MISSING: &str = "provider_transport.regime_missing";
    /// Detail `provider_transport.transport_identity_incomplete`.
    pub const TRANSPORT_IDENTITY_INCOMPLETE: &str = "provider_transport.transport_identity_incomplete";
    /// Detail `provider_transport.trust_incomplete`.
    pub const TRUST_INCOMPLETE: &str = "provider_transport.trust_incomplete";
    /// Detail `provider_transport.verification_required`.
    pub const VERIFICATION_REQUIRED: &str = "provider_transport.verification_required";
}

/// Details in the `quadratic_posterior` family.
pub mod quadratic_posterior {
    /// Detail `quadratic_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "quadratic_posterior.band_not_constructed";
    /// Detail `quadratic_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "quadratic_posterior.external_power_prior";
    /// Detail `quadratic_posterior.insufficient_weight`.
    pub const INSUFFICIENT_WEIGHT: &str = "quadratic_posterior.insufficient_weight";
    /// Detail `quadratic_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "quadratic_posterior.mapped_prior";
    /// Detail `quadratic_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "quadratic_posterior.non_positive_scale";
}

/// Details in the `randomized_quantile_effect` family.
pub mod randomized_quantile_effect {
    /// Detail `randomized_quantile_effect.adjustment_not_supported`.
    pub const ADJUSTMENT_NOT_SUPPORTED: &str = "randomized_quantile_effect.adjustment_not_supported";
    /// Detail `randomized_quantile_effect.arm_too_small`.
    pub const ARM_TOO_SMALL: &str = "randomized_quantile_effect.arm_too_small";
    /// Detail `randomized_quantile_effect.scope`.
    pub const SCOPE: &str = "randomized_quantile_effect.scope";
}

/// Details in the `rank_drop_estimate` family.
pub mod rank_drop_estimate {
    /// Detail `rank_drop_estimate.estimator_not_supported`.
    pub const ESTIMATOR_NOT_SUPPORTED: &str = "rank_drop_estimate.estimator_not_supported";
    /// Detail `rank_drop_estimate.propensity_separates`.
    pub const PROPENSITY_SEPARATES: &str = "rank_drop_estimate.propensity_separates";
    /// Detail `rank_drop_estimate.shape_not_supported`.
    pub const SHAPE_NOT_SUPPORTED: &str = "rank_drop_estimate.shape_not_supported";
    /// Detail `rank_drop_estimate.span_not_preserved`.
    pub const SPAN_NOT_PRESERVED: &str = "rank_drop_estimate.span_not_preserved";
}

/// Details in the `recalc` family.
pub mod recalc {
    /// Detail `recalc.adjusted_arm_not_populated`.
    pub const ADJUSTED_ARM_NOT_POPULATED: &str = "recalc.adjusted_arm_not_populated";
    /// Detail `recalc.adjusted_design_rank_deficient`.
    pub const ADJUSTED_DESIGN_RANK_DEFICIENT: &str = "recalc.adjusted_design_rank_deficient";
    /// Detail `recalc.adjusted_fit_refused`.
    pub const ADJUSTED_FIT_REFUSED: &str = "recalc.adjusted_fit_refused";
    /// Detail `recalc.adjusted_glm_boundary_saturated`.
    pub const ADJUSTED_GLM_BOUNDARY_SATURATED: &str = "recalc.adjusted_glm_boundary_saturated";
    /// Detail `recalc.adjusted_glm_not_converged`.
    pub const ADJUSTED_GLM_NOT_CONVERGED: &str = "recalc.adjusted_glm_not_converged";
    /// Detail `recalc.adjusted_glm_separated`.
    pub const ADJUSTED_GLM_SEPARATED: &str = "recalc.adjusted_glm_separated";
    /// Detail `recalc.adjusted_invalid_model`.
    pub const ADJUSTED_INVALID_MODEL: &str = "recalc.adjusted_invalid_model";
    /// Detail `recalc.adjusted_model_data_invalid`.
    pub const ADJUSTED_MODEL_DATA_INVALID: &str = "recalc.adjusted_model_data_invalid";
    /// Detail `recalc.adjusted_prediction_out_of_support`.
    pub const ADJUSTED_PREDICTION_OUT_OF_SUPPORT: &str = "recalc.adjusted_prediction_out_of_support";
    /// Detail `recalc.adjusted_prediction_schema_mismatch`.
    pub const ADJUSTED_PREDICTION_SCHEMA_MISMATCH: &str = "recalc.adjusted_prediction_schema_mismatch";
    /// Detail `recalc.adjusted_receipt_invalid`.
    pub const ADJUSTED_RECEIPT_INVALID: &str = "recalc.adjusted_receipt_invalid";
    /// Detail `recalc.adjustment_not_identified`.
    pub const ADJUSTMENT_NOT_IDENTIFIED: &str = "recalc.adjustment_not_identified";
    /// Detail `recalc.aipw_requires_overlap_diagnostics`.
    pub const AIPW_REQUIRES_OVERLAP_DIAGNOSTICS: &str = "recalc.aipw_requires_overlap_diagnostics";
    /// Detail `recalc.bayesian_basis_prior_unsupported`.
    pub const BAYESIAN_BASIS_PRIOR_UNSUPPORTED: &str = "recalc.bayesian_basis_prior_unsupported";
    /// Detail `recalc.bayesian_effect_unavailable`.
    pub const BAYESIAN_EFFECT_UNAVAILABLE: &str = "recalc.bayesian_effect_unavailable";
    /// Detail `recalc.bayesian_external_compose_unsupported`.
    pub const BAYESIAN_EXTERNAL_COMPOSE_UNSUPPORTED: &str = "recalc.bayesian_external_compose_unsupported";
    /// Detail `recalc.bayesian_fit_refused`.
    pub const BAYESIAN_FIT_REFUSED: &str = "recalc.bayesian_fit_refused";
    /// Detail `recalc.bayesian_inference_unsupported`.
    pub const BAYESIAN_INFERENCE_UNSUPPORTED: &str = "recalc.bayesian_inference_unsupported";
    /// Detail `recalc.bayesian_invalid_data`.
    pub const BAYESIAN_INVALID_DATA: &str = "recalc.bayesian_invalid_data";
    /// Detail `recalc.bayesian_invalid_graph`.
    pub const BAYESIAN_INVALID_GRAPH: &str = "recalc.bayesian_invalid_graph";
    /// Detail `recalc.bayesian_invalid_inference`.
    pub const BAYESIAN_INVALID_INFERENCE: &str = "recalc.bayesian_invalid_inference";
    /// Detail `recalc.bayesian_invalid_query`.
    pub const BAYESIAN_INVALID_QUERY: &str = "recalc.bayesian_invalid_query";
    /// Detail `recalc.bayesian_invalid_summary`.
    pub const BAYESIAN_INVALID_SUMMARY: &str = "recalc.bayesian_invalid_summary";
    /// Detail `recalc.bayesian_prior_invalid`.
    pub const BAYESIAN_PRIOR_INVALID: &str = "recalc.bayesian_prior_invalid";
    /// Detail `recalc.bayesian_prior_likelihood_double_use`.
    pub const BAYESIAN_PRIOR_LIKELIHOOD_DOUBLE_USE: &str = "recalc.bayesian_prior_likelihood_double_use";
    /// Detail `recalc.bayesian_prior_source_conflict`.
    pub const BAYESIAN_PRIOR_SOURCE_CONFLICT: &str = "recalc.bayesian_prior_source_conflict";
    /// Detail `recalc.bayesian_prior_source_unsupported`.
    pub const BAYESIAN_PRIOR_SOURCE_UNSUPPORTED: &str = "recalc.bayesian_prior_source_unsupported";
    /// Detail `recalc.bayesian_prior_source_unverified`.
    pub const BAYESIAN_PRIOR_SOURCE_UNVERIFIED: &str = "recalc.bayesian_prior_source_unverified";
    /// Detail `recalc.bayesian_workspace_limit`.
    pub const BAYESIAN_WORKSPACE_LIMIT: &str = "recalc.bayesian_workspace_limit";
    /// Detail `recalc.blocked_by_refused_dependency`.
    pub const BLOCKED_BY_REFUSED_DEPENDENCY: &str = "recalc.blocked_by_refused_dependency";
    /// Detail `recalc.bootstrap_replicates_unsupported`.
    pub const BOOTSTRAP_REPLICATES_UNSUPPORTED: &str = "recalc.bootstrap_replicates_unsupported";
    /// Detail `recalc.cancelled`.
    pub const CANCELLED: &str = "recalc.cancelled";
    /// Detail `recalc.capability_adapter_unavailable`.
    pub const CAPABILITY_ADAPTER_UNAVAILABLE: &str = "recalc.capability_adapter_unavailable";
    /// Detail `recalc.categorical_role_mismatch`.
    pub const CATEGORICAL_ROLE_MISMATCH: &str = "recalc.categorical_role_mismatch";
    /// Detail `recalc.contrast_out_of_support`.
    pub const CONTRAST_OUT_OF_SUPPORT: &str = "recalc.contrast_out_of_support";
    /// Detail `recalc.contrast_unavailable`.
    pub const CONTRAST_UNAVAILABLE: &str = "recalc.contrast_unavailable";
    /// Detail `recalc.declared_size_mismatch`.
    pub const DECLARED_SIZE_MISMATCH: &str = "recalc.declared_size_mismatch";
    /// Detail `recalc.design_artifact_mismatch`.
    pub const DESIGN_ARTIFACT_MISMATCH: &str = "recalc.design_artifact_mismatch";
    /// Detail `recalc.design_artifact_unverified`.
    pub const DESIGN_ARTIFACT_UNVERIFIED: &str = "recalc.design_artifact_unverified";
    /// Detail `recalc.design_data_unavailable`.
    pub const DESIGN_DATA_UNAVAILABLE: &str = "recalc.design_data_unavailable";
    /// Detail `recalc.design_effect_unavailable`.
    pub const DESIGN_EFFECT_UNAVAILABLE: &str = "recalc.design_effect_unavailable";
    /// Detail `recalc.design_fit_refused`.
    pub const DESIGN_FIT_REFUSED: &str = "recalc.design_fit_refused";
    /// Detail `recalc.design_model_data_invalid`.
    pub const DESIGN_MODEL_DATA_INVALID: &str = "recalc.design_model_data_invalid";
    /// Detail `recalc.design_not_checked`.
    pub const DESIGN_NOT_CHECKED: &str = "recalc.design_not_checked";
    /// Detail `recalc.design_not_identified`.
    pub const DESIGN_NOT_IDENTIFIED: &str = "recalc.design_not_identified";
    /// Detail `recalc.design_point_unavailable`.
    pub const DESIGN_POINT_UNAVAILABLE: &str = "recalc.design_point_unavailable";
    /// Detail `recalc.design_rank_deficient`.
    pub const DESIGN_RANK_DEFICIENT: &str = "recalc.design_rank_deficient";
    /// Detail `recalc.design_receipt_invalid`.
    pub const DESIGN_RECEIPT_INVALID: &str = "recalc.design_receipt_invalid";
    /// Detail `recalc.dr_arm_not_populated`.
    pub const DR_ARM_NOT_POPULATED: &str = "recalc.dr_arm_not_populated";
    /// Detail `recalc.dr_design_rank_deficient`.
    pub const DR_DESIGN_RANK_DEFICIENT: &str = "recalc.dr_design_rank_deficient";
    /// Detail `recalc.dr_effect_unavailable`.
    pub const DR_EFFECT_UNAVAILABLE: &str = "recalc.dr_effect_unavailable";
    /// Detail `recalc.dr_fit_counter_mismatch`.
    pub const DR_FIT_COUNTER_MISMATCH: &str = "recalc.dr_fit_counter_mismatch";
    /// Detail `recalc.dr_fit_refused`.
    pub const DR_FIT_REFUSED: &str = "recalc.dr_fit_refused";
    /// Detail `recalc.dr_glm_boundary_saturated`.
    pub const DR_GLM_BOUNDARY_SATURATED: &str = "recalc.dr_glm_boundary_saturated";
    /// Detail `recalc.dr_glm_not_converged`.
    pub const DR_GLM_NOT_CONVERGED: &str = "recalc.dr_glm_not_converged";
    /// Detail `recalc.dr_glm_separated`.
    pub const DR_GLM_SEPARATED: &str = "recalc.dr_glm_separated";
    /// Detail `recalc.dr_invalid_folds`.
    pub const DR_INVALID_FOLDS: &str = "recalc.dr_invalid_folds";
    /// Detail `recalc.dr_invalid_learner`.
    pub const DR_INVALID_LEARNER: &str = "recalc.dr_invalid_learner";
    /// Detail `recalc.dr_invalid_model`.
    pub const DR_INVALID_MODEL: &str = "recalc.dr_invalid_model";
    /// Detail `recalc.dr_invalid_overlap`.
    pub const DR_INVALID_OVERLAP: &str = "recalc.dr_invalid_overlap";
    /// Detail `recalc.dr_invalid_score`.
    pub const DR_INVALID_SCORE: &str = "recalc.dr_invalid_score";
    /// Detail `recalc.dr_model_data_invalid`.
    pub const DR_MODEL_DATA_INVALID: &str = "recalc.dr_model_data_invalid";
    /// Detail `recalc.dr_not_identified`.
    pub const DR_NOT_IDENTIFIED: &str = "recalc.dr_not_identified";
    /// Detail `recalc.dr_prediction_out_of_support`.
    pub const DR_PREDICTION_OUT_OF_SUPPORT: &str = "recalc.dr_prediction_out_of_support";
    /// Detail `recalc.dr_prediction_schema_mismatch`.
    pub const DR_PREDICTION_SCHEMA_MISMATCH: &str = "recalc.dr_prediction_schema_mismatch";
    /// Detail `recalc.dr_predictor_unavailable`.
    pub const DR_PREDICTOR_UNAVAILABLE: &str = "recalc.dr_predictor_unavailable";
    /// Detail `recalc.dr_receipt_invalid`.
    pub const DR_RECEIPT_INVALID: &str = "recalc.dr_receipt_invalid";
    /// Detail `recalc.dr_scores_unavailable`.
    pub const DR_SCORES_UNAVAILABLE: &str = "recalc.dr_scores_unavailable";
    /// Detail `recalc.dr_source_mismatch`.
    pub const DR_SOURCE_MISMATCH: &str = "recalc.dr_source_mismatch";
    /// Detail `recalc.dr_target_incompatible`.
    pub const DR_TARGET_INCOMPATIBLE: &str = "recalc.dr_target_incompatible";
    /// Detail `recalc.family_model_data_invalid`.
    pub const FAMILY_MODEL_DATA_INVALID: &str = "recalc.family_model_data_invalid";
    /// Detail `recalc.family_not_identified`.
    pub const FAMILY_NOT_IDENTIFIED: &str = "recalc.family_not_identified";
    /// Detail `recalc.family_receipt_invalid`.
    pub const FAMILY_RECEIPT_INVALID: &str = "recalc.family_receipt_invalid";
    /// Detail `recalc.frozen_scores_rejected`.
    pub const FROZEN_SCORES_REJECTED: &str = "recalc.frozen_scores_rejected";
    /// Detail `recalc.incompatible_contrast`.
    pub const INCOMPATIBLE_CONTRAST: &str = "recalc.incompatible_contrast";
    /// Detail `recalc.invalid_adjustment_set`.
    pub const INVALID_ADJUSTMENT_SET: &str = "recalc.invalid_adjustment_set";
    /// Detail `recalc.invalid_categorical_contrast`.
    pub const INVALID_CATEGORICAL_CONTRAST: &str = "recalc.invalid_categorical_contrast";
    /// Detail `recalc.invalid_changed_input`.
    pub const INVALID_CHANGED_INPUT: &str = "recalc.invalid_changed_input";
    /// Detail `recalc.invalid_contrast`.
    pub const INVALID_CONTRAST: &str = "recalc.invalid_contrast";
    /// Detail `recalc.invalid_data`.
    pub const INVALID_DATA: &str = "recalc.invalid_data";
    /// Detail `recalc.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "recalc.invalid_declaration";
    /// Detail `recalc.invalid_design_artifact`.
    pub const INVALID_DESIGN_ARTIFACT: &str = "recalc.invalid_design_artifact";
    /// Detail `recalc.invalid_design_roles`.
    pub const INVALID_DESIGN_ROLES: &str = "recalc.invalid_design_roles";
    /// Detail `recalc.invalid_evidence`.
    pub const INVALID_EVIDENCE: &str = "recalc.invalid_evidence";
    /// Detail `recalc.invalid_graph`.
    pub const INVALID_GRAPH: &str = "recalc.invalid_graph";
    /// Detail `recalc.invalid_initial_state_law`.
    pub const INVALID_INITIAL_STATE_LAW: &str = "recalc.invalid_initial_state_law";
    /// Detail `recalc.invalid_query`.
    pub const INVALID_QUERY: &str = "recalc.invalid_query";
    /// Detail `recalc.invalid_rd_window`.
    pub const INVALID_RD_WINDOW: &str = "recalc.invalid_rd_window";
    /// Detail `recalc.invalid_request`.
    pub const INVALID_REQUEST: &str = "recalc.invalid_request";
    /// Detail `recalc.invalid_target_weights`.
    pub const INVALID_TARGET_WEIGHTS: &str = "recalc.invalid_target_weights";
    /// Detail `recalc.invalid_temporal_provider`.
    pub const INVALID_TEMPORAL_PROVIDER: &str = "recalc.invalid_temporal_provider";
    /// Detail `recalc.invalid_temporal_window`.
    pub const INVALID_TEMPORAL_WINDOW: &str = "recalc.invalid_temporal_window";
    /// Detail `recalc.iv_decision_unavailable`.
    pub const IV_DECISION_UNAVAILABLE: &str = "recalc.iv_decision_unavailable";
    /// Detail `recalc.iv_diagnostics_unavailable`.
    pub const IV_DIAGNOSTICS_UNAVAILABLE: &str = "recalc.iv_diagnostics_unavailable";
    /// Detail `recalc.joint_bayesian_graph_unsupported`.
    pub const JOINT_BAYESIAN_GRAPH_UNSUPPORTED: &str = "recalc.joint_bayesian_graph_unsupported";
    /// Detail `recalc.joint_bayesian_invalid_data`.
    pub const JOINT_BAYESIAN_INVALID_DATA: &str = "recalc.joint_bayesian_invalid_data";
    /// Detail `recalc.joint_bayesian_invalid_query`.
    pub const JOINT_BAYESIAN_INVALID_QUERY: &str = "recalc.joint_bayesian_invalid_query";
    /// Detail `recalc.joint_bayesian_refused`.
    pub const JOINT_BAYESIAN_REFUSED: &str = "recalc.joint_bayesian_refused";
    /// Detail `recalc.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "recalc.limits_exceeded";
    /// Detail `recalc.memory_budget_exceeded`.
    pub const MEMORY_BUDGET_EXCEEDED: &str = "recalc.memory_budget_exceeded";
    /// Detail `recalc.mz_action_out_of_support`.
    pub const MZ_ACTION_OUT_OF_SUPPORT: &str = "recalc.mz_action_out_of_support";
    /// Detail `recalc.mz_invalid_request`.
    pub const MZ_INVALID_REQUEST: &str = "recalc.mz_invalid_request";
    /// Detail `recalc.no_live_state`.
    pub const NO_LIVE_STATE: &str = "recalc.no_live_state";
    /// Detail `recalc.off_grid_request`.
    pub const OFF_GRID_REQUEST: &str = "recalc.off_grid_request";
    /// Detail `recalc.primary_result_is_not_an_effect`.
    pub const PRIMARY_RESULT_IS_NOT_AN_EFFECT: &str = "recalc.primary_result_is_not_an_effect";
    /// Detail `recalc.refused`.
    pub const REFUSED: &str = "recalc.refused";
    /// Detail `recalc.retarget_incompatible`.
    pub const RETARGET_INCOMPATIBLE: &str = "recalc.retarget_incompatible";
    /// Detail `recalc.retarget_not_supported`.
    pub const RETARGET_NOT_SUPPORTED: &str = "recalc.retarget_not_supported";
    /// Detail `recalc.row_count_mismatch`.
    pub const ROW_COUNT_MISMATCH: &str = "recalc.row_count_mismatch";
    /// Detail `recalc.row_ids_mismatch`.
    pub const ROW_IDS_MISMATCH: &str = "recalc.row_ids_mismatch";
    /// Detail `recalc.score_stage_undeclared`.
    pub const SCORE_STAGE_UNDECLARED: &str = "recalc.score_stage_undeclared";
    /// Detail `recalc.score_table_unavailable`.
    pub const SCORE_TABLE_UNAVAILABLE: &str = "recalc.score_table_unavailable";
    /// Detail `recalc.static_action_out_of_support`.
    pub const STATIC_ACTION_OUT_OF_SUPPORT: &str = "recalc.static_action_out_of_support";
    /// Detail `recalc.static_engine_refused`.
    pub const STATIC_ENGINE_REFUSED: &str = "recalc.static_engine_refused";
    /// Detail `recalc.static_graph_unsupported`.
    pub const STATIC_GRAPH_UNSUPPORTED: &str = "recalc.static_graph_unsupported";
    /// Detail `recalc.static_invalid_contrast`.
    pub const STATIC_INVALID_CONTRAST: &str = "recalc.static_invalid_contrast";
    /// Detail `recalc.static_invalid_data`.
    pub const STATIC_INVALID_DATA: &str = "recalc.static_invalid_data";
    /// Detail `recalc.static_invalid_factor`.
    pub const STATIC_INVALID_FACTOR: &str = "recalc.static_invalid_factor";
    /// Detail `recalc.static_invalid_query`.
    pub const STATIC_INVALID_QUERY: &str = "recalc.static_invalid_query";
    /// Detail `recalc.static_not_identified`.
    pub const STATIC_NOT_IDENTIFIED: &str = "recalc.static_not_identified";
    /// Detail `recalc.static_receipt_invalid`.
    pub const STATIC_RECEIPT_INVALID: &str = "recalc.static_receipt_invalid";
    /// Detail `recalc.static_required_factor_missing`.
    pub const STATIC_REQUIRED_FACTOR_MISSING: &str = "recalc.static_required_factor_missing";
    /// Detail `recalc.static_transport_budget_cancel`.
    pub const STATIC_TRANSPORT_BUDGET_CANCEL: &str = "recalc.static_transport_budget_cancel";
    /// Detail `recalc.static_transport_missing_evidence`.
    pub const STATIC_TRANSPORT_MISSING_EVIDENCE: &str = "recalc.static_transport_missing_evidence";
    /// Detail `recalc.static_transport_missing_provider`.
    pub const STATIC_TRANSPORT_MISSING_PROVIDER: &str = "recalc.static_transport_missing_provider";
    /// Detail `recalc.static_transport_non_transportable`.
    pub const STATIC_TRANSPORT_NON_TRANSPORTABLE: &str = "recalc.static_transport_non_transportable";
    /// Detail `recalc.static_transport_not_certified`.
    pub const STATIC_TRANSPORT_NOT_CERTIFIED: &str = "recalc.static_transport_not_certified";
    /// Detail `recalc.static_transport_numerical_failure`.
    pub const STATIC_TRANSPORT_NUMERICAL_FAILURE: &str = "recalc.static_transport_numerical_failure";
    /// Detail `recalc.static_transport_support_failure`.
    pub const STATIC_TRANSPORT_SUPPORT_FAILURE: &str = "recalc.static_transport_support_failure";
    /// Detail `recalc.static_transport_unsupported_evaluator`.
    pub const STATIC_TRANSPORT_UNSUPPORTED_EVALUATOR: &str = "recalc.static_transport_unsupported_evaluator";
    /// Detail `recalc.temporal_artifact_invalid`.
    pub const TEMPORAL_ARTIFACT_INVALID: &str = "recalc.temporal_artifact_invalid";
    /// Detail `recalc.temporal_artifact_mismatch`.
    pub const TEMPORAL_ARTIFACT_MISMATCH: &str = "recalc.temporal_artifact_mismatch";
    /// Detail `recalc.temporal_artifact_seed_mismatch`.
    pub const TEMPORAL_ARTIFACT_SEED_MISMATCH: &str = "recalc.temporal_artifact_seed_mismatch";
    /// Detail `recalc.temporal_budget_cancel`.
    pub const TEMPORAL_BUDGET_CANCEL: &str = "recalc.temporal_budget_cancel";
    /// Detail `recalc.temporal_data_invalid`.
    pub const TEMPORAL_DATA_INVALID: &str = "recalc.temporal_data_invalid";
    /// Detail `recalc.temporal_evaluator_unsupported`.
    pub const TEMPORAL_EVALUATOR_UNSUPPORTED: &str = "recalc.temporal_evaluator_unsupported";
    /// Detail `recalc.temporal_fit_refused`.
    pub const TEMPORAL_FIT_REFUSED: &str = "recalc.temporal_fit_refused";
    /// Detail `recalc.temporal_history_unsupported`.
    pub const TEMPORAL_HISTORY_UNSUPPORTED: &str = "recalc.temporal_history_unsupported";
    /// Detail `recalc.temporal_missing_evidence`.
    pub const TEMPORAL_MISSING_EVIDENCE: &str = "recalc.temporal_missing_evidence";
    /// Detail `recalc.temporal_missing_provider`.
    pub const TEMPORAL_MISSING_PROVIDER: &str = "recalc.temporal_missing_provider";
    /// Detail `recalc.temporal_numerical_failure`.
    pub const TEMPORAL_NUMERICAL_FAILURE: &str = "recalc.temporal_numerical_failure";
    /// Detail `recalc.temporal_scope_unsupported`.
    pub const TEMPORAL_SCOPE_UNSUPPORTED: &str = "recalc.temporal_scope_unsupported";
    /// Detail `recalc.temporal_support_failure`.
    pub const TEMPORAL_SUPPORT_FAILURE: &str = "recalc.temporal_support_failure";
    /// Detail `recalc.unavailable_data`.
    pub const UNAVAILABLE_DATA: &str = "recalc.unavailable_data";
    /// Detail `recalc.unavailable_fit`.
    pub const UNAVAILABLE_FIT: &str = "recalc.unavailable_fit";
    /// Detail `recalc.unavailable_provider`.
    pub const UNAVAILABLE_PROVIDER: &str = "recalc.unavailable_provider";
    /// Detail `recalc.unknown_variable`.
    pub const UNKNOWN_VARIABLE: &str = "recalc.unknown_variable";
    /// Detail `recalc.unsupported_aipw_score_configuration`.
    pub const UNSUPPORTED_AIPW_SCORE_CONFIGURATION: &str = "recalc.unsupported_aipw_score_configuration";
    /// Detail `recalc.unsupported_request`.
    pub const UNSUPPORTED_REQUEST: &str = "recalc.unsupported_request";
    /// Detail `recalc.weighted_overlap_failed`.
    pub const WEIGHTED_OVERLAP_FAILED: &str = "recalc.weighted_overlap_failed";
}

/// Details in the `recalc_receipt` family.
pub mod recalc_receipt {
    /// Detail `recalc_receipt.counts_inconsistent`.
    pub const COUNTS_INCONSISTENT: &str = "recalc_receipt.counts_inconsistent";
    /// Detail `recalc_receipt.fresh_process_reuse`.
    pub const FRESH_PROCESS_REUSE: &str = "recalc_receipt.fresh_process_reuse";
    /// Detail `recalc_receipt.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "recalc_receipt.identity_mismatch";
    /// Detail `recalc_receipt.plan_mismatch`.
    pub const PLAN_MISMATCH: &str = "recalc_receipt.plan_mismatch";
    /// Detail `recalc_receipt.totals_mismatch`.
    pub const TOTALS_MISMATCH: &str = "recalc_receipt.totals_mismatch";
}

/// Details in the `recovery` family.
pub mod recovery {
    /// Detail `recovery.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "recovery.bounds_exceeded";
    /// Detail `recovery.budget`.
    pub const BUDGET: &str = "recovery.budget";
    /// Detail `recovery.empirical_not_supported`.
    pub const EMPIRICAL_NOT_SUPPORTED: &str = "recovery.empirical_not_supported";
    /// Detail `recovery.handoff_mismatch`.
    pub const HANDOFF_MISMATCH: &str = "recovery.handoff_mismatch";
    /// Detail `recovery.invalid_derivation`.
    pub const INVALID_DERIVATION: &str = "recovery.invalid_derivation";
    /// Detail `recovery.invalid_observed_law`.
    pub const INVALID_OBSERVED_LAW: &str = "recovery.invalid_observed_law";
    /// Detail `recovery.invalid_query`.
    pub const INVALID_QUERY: &str = "recovery.invalid_query";
    /// Detail `recovery.missing_margin`.
    pub const MISSING_MARGIN: &str = "recovery.missing_margin";
    /// Detail `recovery.nonrecoverable_witness`.
    pub const NONRECOVERABLE_WITNESS: &str = "recovery.nonrecoverable_witness";
    /// Detail `recovery.positivity`.
    pub const POSITIVITY: &str = "recovery.positivity";
    /// Detail `recovery.unsupported_mechanism`.
    pub const UNSUPPORTED_MECHANISM: &str = "recovery.unsupported_mechanism";
    /// Detail `recovery.witness_unavailable`.
    pub const WITNESS_UNAVAILABLE: &str = "recovery.witness_unavailable";
}

/// Details in the `recovery_chain` family.
pub mod recovery_chain {
    /// Detail `recovery_chain.budget`.
    pub const BUDGET: &str = "recovery_chain.budget";
    /// Detail `recovery_chain.data_identity_mismatch`.
    pub const DATA_IDENTITY_MISMATCH: &str = "recovery_chain.data_identity_mismatch";
    /// Detail `recovery_chain.decision_replay_mismatch`.
    pub const DECISION_REPLAY_MISMATCH: &str = "recovery_chain.decision_replay_mismatch";
    /// Detail `recovery_chain.invalid_derivation`.
    pub const INVALID_DERIVATION: &str = "recovery_chain.invalid_derivation";
    /// Detail `recovery_chain.invalid_observed_law`.
    pub const INVALID_OBSERVED_LAW: &str = "recovery_chain.invalid_observed_law";
    /// Detail `recovery_chain.invalid_query`.
    pub const INVALID_QUERY: &str = "recovery_chain.invalid_query";
    /// Detail `recovery_chain.names_mismatch`.
    pub const NAMES_MISMATCH: &str = "recovery_chain.names_mismatch";
    /// Detail `recovery_chain.nonrecoverable_witness`.
    pub const NONRECOVERABLE_WITNESS: &str = "recovery_chain.nonrecoverable_witness";
    /// Detail `recovery_chain.not_evaluated`.
    pub const NOT_EVALUATED: &str = "recovery_chain.not_evaluated";
    /// Detail `recovery_chain.positivity`.
    pub const POSITIVITY: &str = "recovery_chain.positivity";
    /// Detail `recovery_chain.premises_mismatch`.
    pub const PREMISES_MISMATCH: &str = "recovery_chain.premises_mismatch";
    /// Detail `recovery_chain.recovered_replay_mismatch`.
    pub const RECOVERED_REPLAY_MISMATCH: &str = "recovery_chain.recovered_replay_mismatch";
    /// Detail `recovery_chain.undecodable`.
    pub const UNDECODABLE: &str = "recovery_chain.undecodable";
    /// Detail `recovery_chain.unsupported_mechanism`.
    pub const UNSUPPORTED_MECHANISM: &str = "recovery_chain.unsupported_mechanism";
    /// Detail `recovery_chain.unsupported_semantics`.
    pub const UNSUPPORTED_SEMANTICS: &str = "recovery_chain.unsupported_semantics";
}

/// Details in the `refresh_posterior` family.
pub mod refresh_posterior {
    /// Detail `refresh_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "refresh_posterior.band_not_constructed";
    /// Detail `refresh_posterior.empty_values`.
    pub const EMPTY_VALUES: &str = "refresh_posterior.empty_values";
    /// Detail `refresh_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "refresh_posterior.external_power_prior";
    /// Detail `refresh_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "refresh_posterior.mapped_prior";
    /// Detail `refresh_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "refresh_posterior.non_positive_scale";
}

/// Details in the `repair_artifact` family.
pub mod repair_artifact {
    /// Detail `repair_artifact.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "repair_artifact.bounds_exceeded";
    /// Detail `repair_artifact.budget`.
    pub const BUDGET: &str = "repair_artifact.budget";
    /// Detail `repair_artifact.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "repair_artifact.invalid_artifact";
    /// Detail `repair_artifact.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "repair_artifact.replay_mismatch";
    /// Detail `repair_artifact.unsupported_version`.
    pub const UNSUPPORTED_VERSION: &str = "repair_artifact.unsupported_version";
}

/// Details in the `response_gam_derivative` family.
pub mod response_gam_derivative {
    /// Detail `response_gam_derivative.shared_complete_case_rows`.
    pub const SHARED_COMPLETE_CASE_ROWS: &str = "response_gam_derivative.shared_complete_case_rows";
}

/// Details in the `rollout` family.
pub mod rollout {
    /// Detail `rollout.artifact_id_missing`.
    pub const ARTIFACT_ID_MISSING: &str = "rollout.artifact_id_missing";
    /// Detail `rollout.artifact_kind_mismatch`.
    pub const ARTIFACT_KIND_MISMATCH: &str = "rollout.artifact_kind_mismatch";
    /// Detail `rollout.expected_binding_mismatch`.
    pub const EXPECTED_BINDING_MISMATCH: &str = "rollout.expected_binding_mismatch";
    /// Detail `rollout.finite_draw_state_required`.
    pub const FINITE_DRAW_STATE_REQUIRED: &str = "rollout.finite_draw_state_required";
    /// Detail `rollout.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "rollout.identity_mismatch";
    /// Detail `rollout.invalid_request`.
    pub const INVALID_REQUEST: &str = "rollout.invalid_request";
    /// Detail `rollout.ordered_prior_mismatch`.
    pub const ORDERED_PRIOR_MISMATCH: &str = "rollout.ordered_prior_mismatch";
    /// Detail `rollout.ranking_invalid`.
    pub const RANKING_INVALID: &str = "rollout.ranking_invalid";
    /// Detail `rollout.ranking_source_mismatch`.
    pub const RANKING_SOURCE_MISMATCH: &str = "rollout.ranking_source_mismatch";
    /// Detail `rollout.source_digest_mismatch`.
    pub const SOURCE_DIGEST_MISMATCH: &str = "rollout.source_digest_mismatch";
    /// Detail `rollout.source_evidence_mismatch`.
    pub const SOURCE_EVIDENCE_MISMATCH: &str = "rollout.source_evidence_mismatch";
    /// Detail `rollout.source_standing_mismatch`.
    pub const SOURCE_STANDING_MISMATCH: &str = "rollout.source_standing_mismatch";
    /// Detail `rollout.state_coordinate_mismatch`.
    pub const STATE_COORDINATE_MISMATCH: &str = "rollout.state_coordinate_mismatch";
    /// Detail `rollout.state_interpretation_mismatch`.
    pub const STATE_INTERPRETATION_MISMATCH: &str = "rollout.state_interpretation_mismatch";
    /// Detail `rollout.terminal_decision_mismatch`.
    pub const TERMINAL_DECISION_MISMATCH: &str = "rollout.terminal_decision_mismatch";
    /// Detail `rollout.unweighted_joint_state_required`.
    pub const UNWEIGHTED_JOINT_STATE_REQUIRED: &str = "rollout.unweighted_joint_state_required";
}

/// Details in the `sampled_recovery` family.
pub mod sampled_recovery {
    /// Detail `sampled_recovery.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "sampled_recovery.bounds_exceeded";
    /// Detail `sampled_recovery.cancelled`.
    pub const CANCELLED: &str = "sampled_recovery.cancelled";
    /// Detail `sampled_recovery.component_variance_only`.
    pub const COMPONENT_VARIANCE_ONLY: &str = "sampled_recovery.component_variance_only";
    /// Detail `sampled_recovery.invalid_input`.
    pub const INVALID_INPUT: &str = "sampled_recovery.invalid_input";
    /// Detail `sampled_recovery.native_hook_unavailable`.
    pub const NATIVE_HOOK_UNAVAILABLE: &str = "sampled_recovery.native_hook_unavailable";
    /// Detail `sampled_recovery.producer_missing`.
    pub const PRODUCER_MISSING: &str = "sampled_recovery.producer_missing";
    /// Detail `sampled_recovery.receipt_mismatch`.
    pub const RECEIPT_MISMATCH: &str = "sampled_recovery.receipt_mismatch";
    /// Detail `sampled_recovery.too_many_failed_replicates`.
    pub const TOO_MANY_FAILED_REPLICATES: &str = "sampled_recovery.too_many_failed_replicates";
    /// Detail `sampled_recovery.unrecoverable_pattern`.
    pub const UNRECOVERABLE_PATTERN: &str = "sampled_recovery.unrecoverable_pattern";
}

/// Details in the `sampling_weights` family.
pub mod sampling_weights {
    /// Detail `sampling_weights.invalid`.
    pub const INVALID: &str = "sampling_weights.invalid";
    /// Detail `sampling_weights.route_unsupported`.
    pub const ROUTE_UNSUPPORTED: &str = "sampling_weights.route_unsupported";
}

/// Details in the `scenario_covariance` family.
pub mod scenario_covariance {
    /// Detail `scenario_covariance.cancelled`.
    pub const CANCELLED: &str = "scenario_covariance.cancelled";
    /// Detail `scenario_covariance.consumer_limit_exceeded`.
    pub const CONSUMER_LIMIT_EXCEEDED: &str = "scenario_covariance.consumer_limit_exceeded";
    /// Detail `scenario_covariance.data_identity_mismatch`.
    pub const DATA_IDENTITY_MISMATCH: &str = "scenario_covariance.data_identity_mismatch";
    /// Detail `scenario_covariance.duplicate_scenario_id`.
    pub const DUPLICATE_SCENARIO_ID: &str = "scenario_covariance.duplicate_scenario_id";
    /// Detail `scenario_covariance.exact_enumeration_cap`.
    pub const EXACT_ENUMERATION_CAP: &str = "scenario_covariance.exact_enumeration_cap";
    /// Detail `scenario_covariance.invalid_declaration`.
    pub const INVALID_DECLARATION: &str = "scenario_covariance.invalid_declaration";
    /// Detail `scenario_covariance.invalid_dependence`.
    pub const INVALID_DEPENDENCE: &str = "scenario_covariance.invalid_dependence";
    /// Detail `scenario_covariance.invalid_failure_bound`.
    pub const INVALID_FAILURE_BOUND: &str = "scenario_covariance.invalid_failure_bound";
    /// Detail `scenario_covariance.invalid_functional`.
    pub const INVALID_FUNCTIONAL: &str = "scenario_covariance.invalid_functional";
    /// Detail `scenario_covariance.invalid_row_table`.
    pub const INVALID_ROW_TABLE: &str = "scenario_covariance.invalid_row_table";
    /// Detail `scenario_covariance.no_scenarios`.
    pub const NO_SCENARIOS: &str = "scenario_covariance.no_scenarios";
    /// Detail `scenario_covariance.not_positive_semidefinite`.
    pub const NOT_POSITIVE_SEMIDEFINITE: &str = "scenario_covariance.not_positive_semidefinite";
    /// Detail `scenario_covariance.premises_mismatch`.
    pub const PREMISES_MISMATCH: &str = "scenario_covariance.premises_mismatch";
    /// Detail `scenario_covariance.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "scenario_covariance.replay_mismatch";
    /// Detail `scenario_covariance.table_too_large`.
    pub const TABLE_TOO_LARGE: &str = "scenario_covariance.table_too_large";
    /// Detail `scenario_covariance.too_few_replicates`.
    pub const TOO_FEW_REPLICATES: &str = "scenario_covariance.too_few_replicates";
    /// Detail `scenario_covariance.too_few_rows`.
    pub const TOO_FEW_ROWS: &str = "scenario_covariance.too_few_rows";
    /// Detail `scenario_covariance.too_many_failed_replicates`.
    pub const TOO_MANY_FAILED_REPLICATES: &str = "scenario_covariance.too_many_failed_replicates";
    /// Detail `scenario_covariance.too_many_terms`.
    pub const TOO_MANY_TERMS: &str = "scenario_covariance.too_many_terms";
    /// Detail `scenario_covariance.unknown_dependence`.
    pub const UNKNOWN_DEPENDENCE: &str = "scenario_covariance.unknown_dependence";
    /// Detail `scenario_covariance.unsupported_semantics`.
    pub const UNSUPPORTED_SEMANTICS: &str = "scenario_covariance.unsupported_semantics";
}

/// Details in the `scenario_covariance_posterior` family.
pub mod scenario_covariance_posterior {
    /// Detail `scenario_covariance_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "scenario_covariance_posterior.band_not_constructed";
    /// Detail `scenario_covariance_posterior.empty_matrix`.
    pub const EMPTY_MATRIX: &str = "scenario_covariance_posterior.empty_matrix";
    /// Detail `scenario_covariance_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "scenario_covariance_posterior.external_power_prior";
    /// Detail `scenario_covariance_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "scenario_covariance_posterior.mapped_prior";
    /// Detail `scenario_covariance_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "scenario_covariance_posterior.non_positive_scale";
}

/// Details in the `scenario_invariance` family.
pub mod scenario_invariance {
    /// Detail `scenario_invariance.derivation_inconsistent`.
    pub const DERIVATION_INCONSISTENT: &str = "scenario_invariance.derivation_inconsistent";
    /// Detail `scenario_invariance.non_static_node`.
    pub const NON_STATIC_NODE: &str = "scenario_invariance.non_static_node";
    /// Detail `scenario_invariance.not_estimated`.
    pub const NOT_ESTIMATED: &str = "scenario_invariance.not_estimated";
    /// Detail `scenario_invariance.refused`.
    pub const REFUSED: &str = "scenario_invariance.refused";
    /// Detail `scenario_invariance.wrong_result_type`.
    pub const WRONG_RESULT_TYPE: &str = "scenario_invariance.wrong_result_type";
}

/// Details in the `scenarios` family.
pub mod scenarios {
    /// Detail `scenarios.catalog_selections`.
    pub const CATALOG_SELECTIONS: &str = "scenarios.catalog_selections";
    /// Detail `scenarios.coordinate_mismatch`.
    pub const COORDINATE_MISMATCH: &str = "scenarios.coordinate_mismatch";
    /// Detail `scenarios.count`.
    pub const COUNT: &str = "scenarios.count";
    /// Detail `scenarios.duplicate_or_empty_name`.
    pub const DUPLICATE_OR_EMPTY_NAME: &str = "scenarios.duplicate_or_empty_name";
    /// Detail `scenarios.duplicate_scenario`.
    pub const DUPLICATE_SCENARIO: &str = "scenarios.duplicate_scenario";
    /// Detail `scenarios.empty`.
    pub const EMPTY: &str = "scenarios.empty";
    /// Detail `scenarios.equivalence_class_input`.
    pub const EQUIVALENCE_CLASS_INPUT: &str = "scenarios.equivalence_class_input";
    /// Detail `scenarios.invalid_weights`.
    pub const INVALID_WEIGHTS: &str = "scenarios.invalid_weights";
    /// Detail `scenarios.shared_data_aggregate`.
    pub const SHARED_DATA_AGGREGATE: &str = "scenarios.shared_data_aggregate";
    /// Detail `scenarios.unevaluated_budget`.
    pub const UNEVALUATED_BUDGET: &str = "scenarios.unevaluated_budget";
}

/// Details in the `scientific_bridge` family.
pub mod scientific_bridge {
    /// Detail `scientific_bridge.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "scientific_bridge.artifact_invalid";
    /// Detail `scientific_bridge.association_as_intervention`.
    pub const ASSOCIATION_AS_INTERVENTION: &str = "scientific_bridge.association_as_intervention";
    /// Detail `scientific_bridge.assumption_evidence_missing`.
    pub const ASSUMPTION_EVIDENCE_MISSING: &str = "scientific_bridge.assumption_evidence_missing";
    /// Detail `scientific_bridge.assumption_missing`.
    pub const ASSUMPTION_MISSING: &str = "scientific_bridge.assumption_missing";
    /// Detail `scientific_bridge.blank_field`.
    pub const BLANK_FIELD: &str = "scientific_bridge.blank_field";
    /// Detail `scientific_bridge.cancelled`.
    pub const CANCELLED: &str = "scientific_bridge.cancelled";
    /// Detail `scientific_bridge.chain_stage_not_affine`.
    pub const CHAIN_STAGE_NOT_AFFINE: &str = "scientific_bridge.chain_stage_not_affine";
    /// Detail `scientific_bridge.conditioning_mismatch`.
    pub const CONDITIONING_MISMATCH: &str = "scientific_bridge.conditioning_mismatch";
    /// Detail `scientific_bridge.degenerate_validation`.
    pub const DEGENERATE_VALIDATION: &str = "scientific_bridge.degenerate_validation";
    /// Detail `scientific_bridge.dependence_not_a_covariance`.
    pub const DEPENDENCE_NOT_A_COVARIANCE: &str = "scientific_bridge.dependence_not_a_covariance";
    /// Detail `scientific_bridge.differential_error`.
    pub const DIFFERENTIAL_ERROR: &str = "scientific_bridge.differential_error";
    /// Detail `scientific_bridge.draws_meaning_mismatch`.
    pub const DRAWS_MEANING_MISMATCH: &str = "scientific_bridge.draws_meaning_mismatch";
    /// Detail `scientific_bridge.draws_not_aligned`.
    pub const DRAWS_NOT_ALIGNED: &str = "scientific_bridge.draws_not_aligned";
    /// Detail `scientific_bridge.draws_outside_domain`.
    pub const DRAWS_OUTSIDE_DOMAIN: &str = "scientific_bridge.draws_outside_domain";
    /// Detail `scientific_bridge.draws_unusable`.
    pub const DRAWS_UNUSABLE: &str = "scientific_bridge.draws_unusable";
    /// Detail `scientific_bridge.duplicate_assumption`.
    pub const DUPLICATE_ASSUMPTION: &str = "scientific_bridge.duplicate_assumption";
    /// Detail `scientific_bridge.endpoint_classes`.
    pub const ENDPOINT_CLASSES: &str = "scientific_bridge.endpoint_classes";
    /// Detail `scientific_bridge.estimand_bridge_incomplete`.
    pub const ESTIMAND_BRIDGE_INCOMPLETE: &str = "scientific_bridge.estimand_bridge_incomplete";
    /// Detail `scientific_bridge.estimand_bridge_mismatch`.
    pub const ESTIMAND_BRIDGE_MISMATCH: &str = "scientific_bridge.estimand_bridge_mismatch";
    /// Detail `scientific_bridge.estimand_spend_inconsistent`.
    pub const ESTIMAND_SPEND_INCONSISTENT: &str = "scientific_bridge.estimand_spend_inconsistent";
    /// Detail `scientific_bridge.estimand_units_mismatch`.
    pub const ESTIMAND_UNITS_MISMATCH: &str = "scientific_bridge.estimand_units_mismatch";
    /// Detail `scientific_bridge.experimental_prior_without_estimand_bridge`.
    pub const EXPERIMENTAL_PRIOR_WITHOUT_ESTIMAND_BRIDGE: &str = "scientific_bridge.experimental_prior_without_estimand_bridge";
    /// Detail `scientific_bridge.functional_not_supported`.
    pub const FUNCTIONAL_NOT_SUPPORTED: &str = "scientific_bridge.functional_not_supported";
    /// Detail `scientific_bridge.horizon_mismatch`.
    pub const HORIZON_MISMATCH: &str = "scientific_bridge.horizon_mismatch";
    /// Detail `scientific_bridge.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "scientific_bridge.identity_mismatch";
    /// Detail `scientific_bridge.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "scientific_bridge.interval_not_supported";
    /// Detail `scientific_bridge.interval_scope_outside_measured`.
    pub const INTERVAL_SCOPE_OUTSIDE_MEASURED: &str = "scientific_bridge.interval_scope_outside_measured";
    /// Detail `scientific_bridge.interval_source_uncertainty_required`.
    pub const INTERVAL_SOURCE_UNCERTAINTY_REQUIRED: &str = "scientific_bridge.interval_source_uncertainty_required";
    /// Detail `scientific_bridge.invalid_quantity`.
    pub const INVALID_QUANTITY: &str = "scientific_bridge.invalid_quantity";
    /// Detail `scientific_bridge.joint_law_dimension_limit`.
    pub const JOINT_LAW_DIMENSION_LIMIT: &str = "scientific_bridge.joint_law_dimension_limit";
    /// Detail `scientific_bridge.malformed_mapping`.
    pub const MALFORMED_MAPPING: &str = "scientific_bridge.malformed_mapping";
    /// Detail `scientific_bridge.mapping_not_allowed`.
    pub const MAPPING_NOT_ALLOWED: &str = "scientific_bridge.mapping_not_allowed";
    /// Detail `scientific_bridge.mapping_scope_mismatch`.
    pub const MAPPING_SCOPE_MISMATCH: &str = "scientific_bridge.mapping_scope_mismatch";
    /// Detail `scientific_bridge.mapping_uncertainty_invalid`.
    pub const MAPPING_UNCERTAINTY_INVALID: &str = "scientific_bridge.mapping_uncertainty_invalid";
    /// Detail `scientific_bridge.mapping_uncertainty_missing`.
    pub const MAPPING_UNCERTAINTY_MISSING: &str = "scientific_bridge.mapping_uncertainty_missing";
    /// Detail `scientific_bridge.measurement_model_mismatch`.
    pub const MEASUREMENT_MODEL_MISMATCH: &str = "scientific_bridge.measurement_model_mismatch";
    /// Detail `scientific_bridge.measurement_route_not_supported`.
    pub const MEASUREMENT_ROUTE_NOT_SUPPORTED: &str = "scientific_bridge.measurement_route_not_supported";
    /// Detail `scientific_bridge.model_identity_mismatch`.
    pub const MODEL_IDENTITY_MISMATCH: &str = "scientific_bridge.model_identity_mismatch";
    /// Detail `scientific_bridge.model_score_as_probability`.
    pub const MODEL_SCORE_AS_PROBABILITY: &str = "scientific_bridge.model_score_as_probability";
    /// Detail `scientific_bridge.non_finite_value`.
    pub const NON_FINITE_VALUE: &str = "scientific_bridge.non_finite_value";
    /// Detail `scientific_bridge.nonlinear_law_insufficient`.
    pub const NONLINEAR_LAW_INSUFFICIENT: &str = "scientific_bridge.nonlinear_law_insufficient";
    /// Detail `scientific_bridge.nonlinear_mean_not_expectation`.
    pub const NONLINEAR_MEAN_NOT_EXPECTATION: &str = "scientific_bridge.nonlinear_mean_not_expectation";
    /// Detail `scientific_bridge.outside_attainable_range`.
    pub const OUTSIDE_ATTAINABLE_RANGE: &str = "scientific_bridge.outside_attainable_range";
    /// Detail `scientific_bridge.outside_validated_range`.
    pub const OUTSIDE_VALIDATED_RANGE: &str = "scientific_bridge.outside_validated_range";
    /// Detail `scientific_bridge.population_mismatch`.
    pub const POPULATION_MISMATCH: &str = "scientific_bridge.population_mismatch";
    /// Detail `scientific_bridge.predictive_importance_as_causal_effect`.
    pub const PREDICTIVE_IMPORTANCE_AS_CAUSAL_EFFECT: &str = "scientific_bridge.predictive_importance_as_causal_effect";
    /// Detail `scientific_bridge.propagation_admission_refused`.
    pub const PROPAGATION_ADMISSION_REFUSED: &str = "scientific_bridge.propagation_admission_refused";
    /// Detail `scientific_bridge.propagation_overflow`.
    pub const PROPAGATION_OVERFLOW: &str = "scientific_bridge.propagation_overflow";
    /// Detail `scientific_bridge.propagation_work_limit`.
    pub const PROPAGATION_WORK_LIMIT: &str = "scientific_bridge.propagation_work_limit";
    /// Detail `scientific_bridge.provenance`.
    pub const PROVENANCE: &str = "scientific_bridge.provenance";
    /// Detail `scientific_bridge.proxy_as_construct`.
    pub const PROXY_AS_CONSTRUCT: &str = "scientific_bridge.proxy_as_construct";
    /// Detail `scientific_bridge.quantity_mismatch`.
    pub const QUANTITY_MISMATCH: &str = "scientific_bridge.quantity_mismatch";
    /// Detail `scientific_bridge.regime_mismatch`.
    pub const REGIME_MISMATCH: &str = "scientific_bridge.regime_mismatch";
    /// Detail `scientific_bridge.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "scientific_bridge.replay_mismatch";
    /// Detail `scientific_bridge.scientific_endpoint_as_utility`.
    pub const SCIENTIFIC_ENDPOINT_AS_UTILITY: &str = "scientific_bridge.scientific_endpoint_as_utility";
    /// Detail `scientific_bridge.simulation_draw_limit`.
    pub const SIMULATION_DRAW_LIMIT: &str = "scientific_bridge.simulation_draw_limit";
    /// Detail `scientific_bridge.source_dependence_invalid`.
    pub const SOURCE_DEPENDENCE_INVALID: &str = "scientific_bridge.source_dependence_invalid";
    /// Detail `scientific_bridge.source_dependence_missing`.
    pub const SOURCE_DEPENDENCE_MISSING: &str = "scientific_bridge.source_dependence_missing";
    /// Detail `scientific_bridge.source_moment_not_finite`.
    pub const SOURCE_MOMENT_NOT_FINITE: &str = "scientific_bridge.source_moment_not_finite";
    /// Detail `scientific_bridge.source_not_draws`.
    pub const SOURCE_NOT_DRAWS: &str = "scientific_bridge.source_not_draws";
    /// Detail `scientific_bridge.source_not_point`.
    pub const SOURCE_NOT_POINT: &str = "scientific_bridge.source_not_point";
    /// Detail `scientific_bridge.source_variance_invalid`.
    pub const SOURCE_VARIANCE_INVALID: &str = "scientific_bridge.source_variance_invalid";
    /// Detail `scientific_bridge.stated_preference_as_realized_behavior`.
    pub const STATED_PREFERENCE_AS_REALIZED_BEHAVIOR: &str = "scientific_bridge.stated_preference_as_realized_behavior";
    /// Detail `scientific_bridge.transport_diagram_scope_mismatch`.
    pub const TRANSPORT_DIAGRAM_SCOPE_MISMATCH: &str = "scientific_bridge.transport_diagram_scope_mismatch";
    /// Detail `scientific_bridge.transport_premise_contradicted_by_selection_diagram`.
    pub const TRANSPORT_PREMISE_CONTRADICTED_BY_SELECTION_DIAGRAM: &str = "scientific_bridge.transport_premise_contradicted_by_selection_diagram";
    /// Detail `scientific_bridge.transport_premise_mismatch`.
    pub const TRANSPORT_PREMISE_MISMATCH: &str = "scientific_bridge.transport_premise_mismatch";
    /// Detail `scientific_bridge.transport_premise_missing`.
    pub const TRANSPORT_PREMISE_MISSING: &str = "scientific_bridge.transport_premise_missing";
    /// Detail `scientific_bridge.transport_premise_unused`.
    pub const TRANSPORT_PREMISE_UNUSED: &str = "scientific_bridge.transport_premise_unused";
    /// Detail `scientific_bridge.uncertain_mapping_not_affine`.
    pub const UNCERTAIN_MAPPING_NOT_AFFINE: &str = "scientific_bridge.uncertain_mapping_not_affine";
    /// Detail `scientific_bridge.units_mismatch`.
    pub const UNITS_MISMATCH: &str = "scientific_bridge.units_mismatch";
    /// Detail `scientific_bridge.validation_r_squared_too_low`.
    pub const VALIDATION_R_SQUARED_TOO_LOW: &str = "scientific_bridge.validation_r_squared_too_low";
    /// Detail `scientific_bridge.validation_trials_too_few`.
    pub const VALIDATION_TRIALS_TOO_FEW: &str = "scientific_bridge.validation_trials_too_few";
    /// Detail `scientific_bridge.value_kind_not_carried`.
    pub const VALUE_KIND_NOT_CARRIED: &str = "scientific_bridge.value_kind_not_carried";
    /// Detail `scientific_bridge.value_kind_substitution`.
    pub const VALUE_KIND_SUBSTITUTION: &str = "scientific_bridge.value_kind_substitution";
    /// Detail `scientific_bridge.youden_too_small`.
    pub const YOUDEN_TOO_SMALL: &str = "scientific_bridge.youden_too_small";
}

/// Details in the `sde` family.
pub mod sde {
    /// Detail `sde.action_time_off_grid`.
    pub const ACTION_TIME_OFF_GRID: &str = "sde.action_time_off_grid";
    /// Detail `sde.input_outside_certified_envelope`.
    pub const INPUT_OUTSIDE_CERTIFIED_ENVELOPE: &str = "sde.input_outside_certified_envelope";
    /// Detail `sde.milstein_noncommutative`.
    pub const MILSTEIN_NONCOMMUTATIVE: &str = "sde.milstein_noncommutative";
    /// Detail `sde.unbounded_growth`.
    pub const UNBOUNDED_GROWTH: &str = "sde.unbounded_growth";
}

/// Details in the `selective_discovery` family.
pub mod selective_discovery {
    /// Detail `selective_discovery.analysis_rows_not_the_split_part`.
    pub const ANALYSIS_ROWS_NOT_THE_SPLIT_PART: &str = "selective_discovery.analysis_rows_not_the_split_part";
    /// Detail `selective_discovery.artifact_invalid`.
    pub const ARTIFACT_INVALID: &str = "selective_discovery.artifact_invalid";
    /// Detail `selective_discovery.background_knowledge_contradictory`.
    pub const BACKGROUND_KNOWLEDGE_CONTRADICTORY: &str = "selective_discovery.background_knowledge_contradictory";
    /// Detail `selective_discovery.background_knowledge_not_applicable`.
    pub const BACKGROUND_KNOWLEDGE_NOT_APPLICABLE: &str = "selective_discovery.background_knowledge_not_applicable";
    /// Detail `selective_discovery.background_knowledge_unknown_variable`.
    pub const BACKGROUND_KNOWLEDGE_UNKNOWN_VARIABLE: &str = "selective_discovery.background_knowledge_unknown_variable";
    /// Detail `selective_discovery.components_incomplete`.
    pub const COMPONENTS_INCOMPLETE: &str = "selective_discovery.components_incomplete";
    /// Detail `selective_discovery.contemporaneous_links`.
    pub const CONTEMPORANEOUS_LINKS: &str = "selective_discovery.contemporaneous_links";
    /// Detail `selective_discovery.contract_identity`.
    pub const CONTRACT_IDENTITY: &str = "selective_discovery.contract_identity";
    /// Detail `selective_discovery.contract_mismatch`.
    pub const CONTRACT_MISMATCH: &str = "selective_discovery.contract_mismatch";
    /// Detail `selective_discovery.data_mismatch`.
    pub const DATA_MISMATCH: &str = "selective_discovery.data_mismatch";
    /// Detail `selective_discovery.differences_are_not_selection_targets`.
    pub const DIFFERENCES_ARE_NOT_SELECTION_TARGETS: &str = "selective_discovery.differences_are_not_selection_targets";
    /// Detail `selective_discovery.downstream_interval_not_supported`.
    pub const DOWNSTREAM_INTERVAL_NOT_SUPPORTED: &str = "selective_discovery.downstream_interval_not_supported";
    /// Detail `selective_discovery.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "selective_discovery.identity_mismatch";
    /// Detail `selective_discovery.invalid_contract`.
    pub const INVALID_CONTRACT: &str = "selective_discovery.invalid_contract";
    /// Detail `selective_discovery.multi_population`.
    pub const MULTI_POPULATION: &str = "selective_discovery.multi_population";
    /// Detail `selective_discovery.nothing_retained`.
    pub const NOTHING_RETAINED: &str = "selective_discovery.nothing_retained";
    /// Detail `selective_discovery.pooled_population_data`.
    pub const POOLED_POPULATION_DATA: &str = "selective_discovery.pooled_population_data";
    /// Detail `selective_discovery.pooled_population_search`.
    pub const POOLED_POPULATION_SEARCH: &str = "selective_discovery.pooled_population_search";
    /// Detail `selective_discovery.population_missing`.
    pub const POPULATION_MISSING: &str = "selective_discovery.population_missing";
    /// Detail `selective_discovery.population_repeated`.
    pub const POPULATION_REPEATED: &str = "selective_discovery.population_repeated";
    /// Detail `selective_discovery.populations_missing`.
    pub const POPULATIONS_MISSING: &str = "selective_discovery.populations_missing";
    /// Detail `selective_discovery.raw_data_missing`.
    pub const RAW_DATA_MISSING: &str = "selective_discovery.raw_data_missing";
    /// Detail `selective_discovery.receipt_invalid`.
    pub const RECEIPT_INVALID: &str = "selective_discovery.receipt_invalid";
    /// Detail `selective_discovery.receipt_limit`.
    pub const RECEIPT_LIMIT: &str = "selective_discovery.receipt_limit";
    /// Detail `selective_discovery.receipt_repeated`.
    pub const RECEIPT_REPEATED: &str = "selective_discovery.receipt_repeated";
    /// Detail `selective_discovery.receipts_identity`.
    pub const RECEIPTS_IDENTITY: &str = "selective_discovery.receipts_identity";
    /// Detail `selective_discovery.recovery_claim_not_supported`.
    pub const RECOVERY_CLAIM_NOT_SUPPORTED: &str = "selective_discovery.recovery_claim_not_supported";
    /// Detail `selective_discovery.replay_diverged`.
    pub const REPLAY_DIVERGED: &str = "selective_discovery.replay_diverged";
    /// Detail `selective_discovery.replay_miss`.
    pub const REPLAY_MISS: &str = "selective_discovery.replay_miss";
    /// Detail `selective_discovery.rng_mismatch`.
    pub const RNG_MISMATCH: &str = "selective_discovery.rng_mismatch";
    /// Detail `selective_discovery.route_unknown`.
    pub const ROUTE_UNKNOWN: &str = "selective_discovery.route_unknown";
    /// Detail `selective_discovery.rows`.
    pub const ROWS: &str = "selective_discovery.rows";
    /// Detail `selective_discovery.score_search_not_supported`.
    pub const SCORE_SEARCH_NOT_SUPPORTED: &str = "selective_discovery.score_search_not_supported";
    /// Detail `selective_discovery.search_incomplete`.
    pub const SEARCH_INCOMPLETE: &str = "selective_discovery.search_incomplete";
    /// Detail `selective_discovery.split_fraction_invalid`.
    pub const SPLIT_FRACTION_INVALID: &str = "selective_discovery.split_fraction_invalid";
    /// Detail `selective_discovery.split_mask_mismatch`.
    pub const SPLIT_MASK_MISMATCH: &str = "selective_discovery.split_mask_mismatch";
    /// Detail `selective_discovery.split_not_disjoint`.
    pub const SPLIT_NOT_DISJOINT: &str = "selective_discovery.split_not_disjoint";
    /// Detail `selective_discovery.split_part_empty`.
    pub const SPLIT_PART_EMPTY: &str = "selective_discovery.split_part_empty";
    /// Detail `selective_discovery.split_part_weight_invalid`.
    pub const SPLIT_PART_WEIGHT_INVALID: &str = "selective_discovery.split_part_weight_invalid";
    /// Detail `selective_discovery.split_receipt_mismatch`.
    pub const SPLIT_RECEIPT_MISMATCH: &str = "selective_discovery.split_receipt_mismatch";
    /// Detail `selective_discovery.split_row_excluded`.
    pub const SPLIT_ROW_EXCLUDED: &str = "selective_discovery.split_row_excluded";
    /// Detail `selective_discovery.split_row_out_of_range`.
    pub const SPLIT_ROW_OUT_OF_RANGE: &str = "selective_discovery.split_row_out_of_range";
    /// Detail `selective_discovery.split_rows_mismatch`.
    pub const SPLIT_ROWS_MISMATCH: &str = "selective_discovery.split_rows_mismatch";
    /// Detail `selective_discovery.split_rows_repeated`.
    pub const SPLIT_ROWS_REPEATED: &str = "selective_discovery.split_rows_repeated";
    /// Detail `selective_discovery.split_source_mismatch`.
    pub const SPLIT_SOURCE_MISMATCH: &str = "selective_discovery.split_source_mismatch";
    /// Detail `selective_discovery.split_unsupported_rows`.
    pub const SPLIT_UNSUPPORTED_ROWS: &str = "selective_discovery.split_unsupported_rows";
    /// Detail `selective_discovery.split_weights_mismatch`.
    pub const SPLIT_WEIGHTS_MISMATCH: &str = "selective_discovery.split_weights_mismatch";
    /// Detail `selective_discovery.structure_class`.
    pub const STRUCTURE_CLASS: &str = "selective_discovery.structure_class";
    /// Detail `selective_discovery.structure_marks`.
    pub const STRUCTURE_MARKS: &str = "selective_discovery.structure_marks";
    /// Detail `selective_discovery.structure_nodes`.
    pub const STRUCTURE_NODES: &str = "selective_discovery.structure_nodes";
    /// Detail `selective_discovery.test_identity_missing`.
    pub const TEST_IDENTITY_MISSING: &str = "selective_discovery.test_identity_missing";
    /// Detail `selective_discovery.unaccepted_discovery`.
    pub const UNACCEPTED_DISCOVERY: &str = "selective_discovery.unaccepted_discovery";
    /// Detail `selective_discovery.unresolved_structure`.
    pub const UNRESOLVED_STRUCTURE: &str = "selective_discovery.unresolved_structure";
    /// Detail `selective_discovery.variables_repeated`.
    pub const VARIABLES_REPEATED: &str = "selective_discovery.variables_repeated";
    /// Detail `selective_discovery.weighted_test_without_weights`.
    pub const WEIGHTED_TEST_WITHOUT_WEIGHTS: &str = "selective_discovery.weighted_test_without_weights";
    /// Detail `selective_discovery.weights_not_read_by_test`.
    pub const WEIGHTS_NOT_READ_BY_TEST: &str = "selective_discovery.weights_not_read_by_test";
    /// Detail `selective_discovery.work_inconsistent`.
    pub const WORK_INCONSISTENT: &str = "selective_discovery.work_inconsistent";
}

/// Details in the `sensitivity_decision_composition` family.
pub mod sensitivity_decision_composition {
    /// Detail `sensitivity_decision_composition.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "sensitivity_decision_composition.bounds_exceeded";
    /// Detail `sensitivity_decision_composition.composition_not_supported`.
    pub const COMPOSITION_NOT_SUPPORTED: &str = "sensitivity_decision_composition.composition_not_supported";
    /// Detail `sensitivity_decision_composition.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "sensitivity_decision_composition.invalid_artifact";
    /// Detail `sensitivity_decision_composition.invalid_surface`.
    pub const INVALID_SURFACE: &str = "sensitivity_decision_composition.invalid_surface";
    /// Detail `sensitivity_decision_composition.sampling_not_reported`.
    pub const SAMPLING_NOT_REPORTED: &str = "sensitivity_decision_composition.sampling_not_reported";
    /// Detail `sensitivity_decision_composition.unsupported_adaptation`.
    pub const UNSUPPORTED_ADAPTATION: &str = "sensitivity_decision_composition.unsupported_adaptation";
    /// Detail `sensitivity_decision_composition.unsupported_coordinate`.
    pub const UNSUPPORTED_COORDINATE: &str = "sensitivity_decision_composition.unsupported_coordinate";
    /// Detail `sensitivity_decision_composition.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "sensitivity_decision_composition.wrong_contract";
}

/// Details in the `sensitivity_source` family.
pub mod sensitivity_source {
    /// Detail `sensitivity_source.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "sensitivity_source.invalid_artifact";
    /// Detail `sensitivity_source.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "sensitivity_source.limits_exceeded";
    /// Detail `sensitivity_source.lineage_digest`.
    pub const LINEAGE_DIGEST: &str = "sensitivity_source.lineage_digest";
    /// Detail `sensitivity_source.source_unavailable`.
    pub const SOURCE_UNAVAILABLE: &str = "sensitivity_source.source_unavailable";
    /// Detail `sensitivity_source.surface_mismatch`.
    pub const SURFACE_MISMATCH: &str = "sensitivity_source.surface_mismatch";
}

/// Details in the `sequential_evaluation` family.
pub mod sequential_evaluation {
    /// Detail `sequential_evaluation.accounting_residual`.
    pub const ACCOUNTING_RESIDUAL: &str = "sequential_evaluation.accounting_residual";
    /// Detail `sequential_evaluation.action_channel`.
    pub const ACTION_CHANNEL: &str = "sequential_evaluation.action_channel";
    /// Detail `sequential_evaluation.action_cost`.
    pub const ACTION_COST: &str = "sequential_evaluation.action_cost";
    /// Detail `sequential_evaluation.action_delayed_outcome`.
    pub const ACTION_DELAYED_OUTCOME: &str = "sequential_evaluation.action_delayed_outcome";
    /// Detail `sequential_evaluation.action_index`.
    pub const ACTION_INDEX: &str = "sequential_evaluation.action_index";
    /// Detail `sequential_evaluation.action_label`.
    pub const ACTION_LABEL: &str = "sequential_evaluation.action_label";
    /// Detail `sequential_evaluation.action_not_supported`.
    pub const ACTION_NOT_SUPPORTED: &str = "sequential_evaluation.action_not_supported";
    /// Detail `sequential_evaluation.action_support`.
    pub const ACTION_SUPPORT: &str = "sequential_evaluation.action_support";
    /// Detail `sequential_evaluation.action_transition`.
    pub const ACTION_TRANSITION: &str = "sequential_evaluation.action_transition";
    /// Detail `sequential_evaluation.action_utility`.
    pub const ACTION_UTILITY: &str = "sequential_evaluation.action_utility";
    /// Detail `sequential_evaluation.artifact_id_missing`.
    pub const ARTIFACT_ID_MISSING: &str = "sequential_evaluation.artifact_id_missing";
    /// Detail `sequential_evaluation.artifact_kind_mismatch`.
    pub const ARTIFACT_KIND_MISMATCH: &str = "sequential_evaluation.artifact_kind_mismatch";
    /// Detail `sequential_evaluation.behaviour_support_missing`.
    pub const BEHAVIOUR_SUPPORT_MISSING: &str = "sequential_evaluation.behaviour_support_missing";
    /// Detail `sequential_evaluation.cancelled`.
    pub const CANCELLED: &str = "sequential_evaluation.cancelled";
    /// Detail `sequential_evaluation.channel_binding_mismatch`.
    pub const CHANNEL_BINDING_MISMATCH: &str = "sequential_evaluation.channel_binding_mismatch";
    /// Detail `sequential_evaluation.channel_construct_mismatch`.
    pub const CHANNEL_CONSTRUCT_MISMATCH: &str = "sequential_evaluation.channel_construct_mismatch";
    /// Detail `sequential_evaluation.channel_levels_mismatch`.
    pub const CHANNEL_LEVELS_MISMATCH: &str = "sequential_evaluation.channel_levels_mismatch";
    /// Detail `sequential_evaluation.channel_rates`.
    pub const CHANNEL_RATES: &str = "sequential_evaluation.channel_rates";
    /// Detail `sequential_evaluation.channel_route`.
    pub const CHANNEL_ROUTE: &str = "sequential_evaluation.channel_route";
    /// Detail `sequential_evaluation.commit_has_no_observation`.
    pub const COMMIT_HAS_NO_OBSERVATION: &str = "sequential_evaluation.commit_has_no_observation";
    /// Detail `sequential_evaluation.comparison_kind_mismatch`.
    pub const COMPARISON_KIND_MISMATCH: &str = "sequential_evaluation.comparison_kind_mismatch";
    /// Detail `sequential_evaluation.comparison_label`.
    pub const COMPARISON_LABEL: &str = "sequential_evaluation.comparison_label";
    /// Detail `sequential_evaluation.comparisons`.
    pub const COMPARISONS: &str = "sequential_evaluation.comparisons";
    /// Detail `sequential_evaluation.decision_receipt_mismatch`.
    pub const DECISION_RECEIPT_MISMATCH: &str = "sequential_evaluation.decision_receipt_mismatch";
    /// Detail `sequential_evaluation.delayed_outcome_beyond_horizon`.
    pub const DELAYED_OUTCOME_BEYOND_HORIZON: &str = "sequential_evaluation.delayed_outcome_beyond_horizon";
    /// Detail `sequential_evaluation.differential_channel`.
    pub const DIFFERENTIAL_CHANNEL: &str = "sequential_evaluation.differential_channel";
    /// Detail `sequential_evaluation.empty_log`.
    pub const EMPTY_LOG: &str = "sequential_evaluation.empty_log";
    /// Detail `sequential_evaluation.exact_mismatch`.
    pub const EXACT_MISMATCH: &str = "sequential_evaluation.exact_mismatch";
    /// Detail `sequential_evaluation.history_receipt_mismatch`.
    pub const HISTORY_RECEIPT_MISMATCH: &str = "sequential_evaluation.history_receipt_mismatch";
    /// Detail `sequential_evaluation.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "sequential_evaluation.identity_mismatch";
    /// Detail `sequential_evaluation.importance_weight_overflow`.
    pub const IMPORTANCE_WEIGHT_OVERFLOW: &str = "sequential_evaluation.importance_weight_overflow";
    /// Detail `sequential_evaluation.indicator_as_state`.
    pub const INDICATOR_AS_STATE: &str = "sequential_evaluation.indicator_as_state";
    /// Detail `sequential_evaluation.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "sequential_evaluation.interval_not_supported";
    /// Detail `sequential_evaluation.limit_exceeded`.
    pub const LIMIT_EXCEEDED: &str = "sequential_evaluation.limit_exceeded";
    /// Detail `sequential_evaluation.log_partially_counted`.
    pub const LOG_PARTIALLY_COUNTED: &str = "sequential_evaluation.log_partially_counted";
    /// Detail `sequential_evaluation.logged_trajectory_inconsistent`.
    pub const LOGGED_TRAJECTORY_INCONSISTENT: &str = "sequential_evaluation.logged_trajectory_inconsistent";
    /// Detail `sequential_evaluation.malformed`.
    pub const MALFORMED: &str = "sequential_evaluation.malformed";
    /// Detail `sequential_evaluation.model_actions`.
    pub const MODEL_ACTIONS: &str = "sequential_evaluation.model_actions";
    /// Detail `sequential_evaluation.model_field`.
    pub const MODEL_FIELD: &str = "sequential_evaluation.model_field";
    /// Detail `sequential_evaluation.model_horizon`.
    pub const MODEL_HORIZON: &str = "sequential_evaluation.model_horizon";
    /// Detail `sequential_evaluation.model_initial`.
    pub const MODEL_INITIAL: &str = "sequential_evaluation.model_initial";
    /// Detail `sequential_evaluation.model_mismatch`.
    pub const MODEL_MISMATCH: &str = "sequential_evaluation.model_mismatch";
    /// Detail `sequential_evaluation.model_resource_budget`.
    pub const MODEL_RESOURCE_BUDGET: &str = "sequential_evaluation.model_resource_budget";
    /// Detail `sequential_evaluation.model_states`.
    pub const MODEL_STATES: &str = "sequential_evaluation.model_states";
    /// Detail `sequential_evaluation.model_terminal_utility`.
    pub const MODEL_TERMINAL_UTILITY: &str = "sequential_evaluation.model_terminal_utility";
    /// Detail `sequential_evaluation.model_work`.
    pub const MODEL_WORK: &str = "sequential_evaluation.model_work";
    /// Detail `sequential_evaluation.no_overlap`.
    pub const NO_OVERLAP: &str = "sequential_evaluation.no_overlap";
    /// Detail `sequential_evaluation.observation_impossible`.
    pub const OBSERVATION_IMPOSSIBLE: &str = "sequential_evaluation.observation_impossible";
    /// Detail `sequential_evaluation.observation_recorded_twice`.
    pub const OBSERVATION_RECORDED_TWICE: &str = "sequential_evaluation.observation_recorded_twice";
    /// Detail `sequential_evaluation.off_policy_mismatch`.
    pub const OFF_POLICY_MISMATCH: &str = "sequential_evaluation.off_policy_mismatch";
    /// Detail `sequential_evaluation.past_horizon`.
    pub const PAST_HORIZON: &str = "sequential_evaluation.past_horizon";
    /// Detail `sequential_evaluation.policy_change_refused`.
    pub const POLICY_CHANGE_REFUSED: &str = "sequential_evaluation.policy_change_refused";
    /// Detail `sequential_evaluation.policy_distribution`.
    pub const POLICY_DISTRIBUTION: &str = "sequential_evaluation.policy_distribution";
    /// Detail `sequential_evaluation.policy_incomplete`.
    pub const POLICY_INCOMPLETE: &str = "sequential_evaluation.policy_incomplete";
    /// Detail `sequential_evaluation.policy_mismatch`.
    pub const POLICY_MISMATCH: &str = "sequential_evaluation.policy_mismatch";
    /// Detail `sequential_evaluation.policy_version`.
    pub const POLICY_VERSION: &str = "sequential_evaluation.policy_version";
    /// Detail `sequential_evaluation.record_already_counted`.
    pub const RECORD_ALREADY_COUNTED: &str = "sequential_evaluation.record_already_counted";
    /// Detail `sequential_evaluation.record_identity`.
    pub const RECORD_IDENTITY: &str = "sequential_evaluation.record_identity";
    /// Detail `sequential_evaluation.resource_budget_exceeded`.
    pub const RESOURCE_BUDGET_EXCEEDED: &str = "sequential_evaluation.resource_budget_exceeded";
    /// Detail `sequential_evaluation.return_overflow`.
    pub const RETURN_OVERFLOW: &str = "sequential_evaluation.return_overflow";
    /// Detail `sequential_evaluation.rollout_belief`.
    pub const ROLLOUT_BELIEF: &str = "sequential_evaluation.rollout_belief";
    /// Detail `sequential_evaluation.rollout_mismatch`.
    pub const ROLLOUT_MISMATCH: &str = "sequential_evaluation.rollout_mismatch";
    /// Detail `sequential_evaluation.rollouts`.
    pub const ROLLOUTS: &str = "sequential_evaluation.rollouts";
    /// Detail `sequential_evaluation.session_label`.
    pub const SESSION_LABEL: &str = "sequential_evaluation.session_label";
    /// Detail `sequential_evaluation.static_score_table`.
    pub const STATIC_SCORE_TABLE: &str = "sequential_evaluation.static_score_table";
    /// Detail `sequential_evaluation.stopping_boundary`.
    pub const STOPPING_BOUNDARY: &str = "sequential_evaluation.stopping_boundary";
    /// Detail `sequential_evaluation.temporal_transport_contract_missing`.
    pub const TEMPORAL_TRANSPORT_CONTRACT_MISSING: &str = "sequential_evaluation.temporal_transport_contract_missing";
    /// Detail `sequential_evaluation.trajectory_identity`.
    pub const TRAJECTORY_IDENTITY: &str = "sequential_evaluation.trajectory_identity";
    /// Detail `sequential_evaluation.trajectory_recorded_twice`.
    pub const TRAJECTORY_RECORDED_TWICE: &str = "sequential_evaluation.trajectory_recorded_twice";
    /// Detail `sequential_evaluation.transported_policy_contract_missing`.
    pub const TRANSPORTED_POLICY_CONTRACT_MISSING: &str = "sequential_evaluation.transported_policy_contract_missing";
    /// Detail `sequential_evaluation.value_mismatch`.
    pub const VALUE_MISMATCH: &str = "sequential_evaluation.value_mismatch";
    /// Detail `sequential_evaluation.version`.
    pub const VERSION: &str = "sequential_evaluation.version";
    /// Detail `sequential_evaluation.voi_collect_action`.
    pub const VOI_COLLECT_ACTION: &str = "sequential_evaluation.voi_collect_action";
    /// Detail `sequential_evaluation.voi_commit_actions`.
    pub const VOI_COMMIT_ACTIONS: &str = "sequential_evaluation.voi_commit_actions";
    /// Detail `sequential_evaluation.voi_enumeration_mismatch`.
    pub const VOI_ENUMERATION_MISMATCH: &str = "sequential_evaluation.voi_enumeration_mismatch";
    /// Detail `sequential_evaluation.voi_ranking`.
    pub const VOI_RANKING: &str = "sequential_evaluation.voi_ranking";
    /// Detail `sequential_evaluation.voi_scope`.
    pub const VOI_SCOPE: &str = "sequential_evaluation.voi_scope";
    /// Detail `sequential_evaluation.weighted_estimate_overflow`.
    pub const WEIGHTED_ESTIMATE_OVERFLOW: &str = "sequential_evaluation.weighted_estimate_overflow";
    /// Detail `sequential_evaluation.weighted_sum_overflow`.
    pub const WEIGHTED_SUM_OVERFLOW: &str = "sequential_evaluation.weighted_sum_overflow";
}

/// Details in the `signal_provider` family.
pub mod signal_provider {
    /// Detail `signal_provider.candidate_mismatch`.
    pub const CANDIDATE_MISMATCH: &str = "signal_provider.candidate_mismatch";
    /// Detail `signal_provider.capability_missing`.
    pub const CAPABILITY_MISSING: &str = "signal_provider.capability_missing";
    /// Detail `signal_provider.computational_limit`.
    pub const COMPUTATIONAL_LIMIT: &str = "signal_provider.computational_limit";
    /// Detail `signal_provider.conditional_independence_missing`.
    pub const CONDITIONAL_INDEPENDENCE_MISSING: &str = "signal_provider.conditional_independence_missing";
    /// Detail `signal_provider.invalid_object`.
    pub const INVALID_OBJECT: &str = "signal_provider.invalid_object";
    /// Detail `signal_provider.invalid_request`.
    pub const INVALID_REQUEST: &str = "signal_provider.invalid_request";
    /// Detail `signal_provider.invalid_table`.
    pub const INVALID_TABLE: &str = "signal_provider.invalid_table";
    /// Detail `signal_provider.law_incoherent`.
    pub const LAW_INCOHERENT: &str = "signal_provider.law_incoherent";
    /// Detail `signal_provider.missing_lineage`.
    pub const MISSING_LINEAGE: &str = "signal_provider.missing_lineage";
    /// Detail `signal_provider.observation_quantity_mismatch`.
    pub const OBSERVATION_QUANTITY_MISMATCH: &str = "signal_provider.observation_quantity_mismatch";
    /// Detail `signal_provider.posterior_incoherent`.
    pub const POSTERIOR_INCOHERENT: &str = "signal_provider.posterior_incoherent";
    /// Detail `signal_provider.posterior_quantity_mismatch`.
    pub const POSTERIOR_QUANTITY_MISMATCH: &str = "signal_provider.posterior_quantity_mismatch";
    /// Detail `signal_provider.prior_mismatch`.
    pub const PRIOR_MISMATCH: &str = "signal_provider.prior_mismatch";
    /// Detail `signal_provider.request_fingerprint_mismatch`.
    pub const REQUEST_FINGERPRINT_MISMATCH: &str = "signal_provider.request_fingerprint_mismatch";
    /// Detail `signal_provider.sample_size_invalid`.
    pub const SAMPLE_SIZE_INVALID: &str = "signal_provider.sample_size_invalid";
    /// Detail `signal_provider.sample_size_mismatch`.
    pub const SAMPLE_SIZE_MISMATCH: &str = "signal_provider.sample_size_mismatch";
    /// Detail `signal_provider.trust_mismatch`.
    pub const TRUST_MISMATCH: &str = "signal_provider.trust_mismatch";
    /// Detail `signal_provider.trust_native_claimed`.
    pub const TRUST_NATIVE_CLAIMED: &str = "signal_provider.trust_native_claimed";
}

/// Details in the `smoothed_dose_posterior` family.
pub mod smoothed_dose_posterior {
    /// Detail `smoothed_dose_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "smoothed_dose_posterior.band_not_constructed";
    /// Detail `smoothed_dose_posterior.cancelled`.
    pub const CANCELLED: &str = "smoothed_dose_posterior.cancelled";
    /// Detail `smoothed_dose_posterior.certificate_mismatch`.
    pub const CERTIFICATE_MISMATCH: &str = "smoothed_dose_posterior.certificate_mismatch";
    /// Detail `smoothed_dose_posterior.empty_scores`.
    pub const EMPTY_SCORES: &str = "smoothed_dose_posterior.empty_scores";
    /// Detail `smoothed_dose_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "smoothed_dose_posterior.external_power_prior";
    /// Detail `smoothed_dose_posterior.invalid_graph`.
    pub const INVALID_GRAPH: &str = "smoothed_dose_posterior.invalid_graph";
    /// Detail `smoothed_dose_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "smoothed_dose_posterior.mapped_prior";
    /// Detail `smoothed_dose_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "smoothed_dose_posterior.non_positive_scale";
    /// Detail `smoothed_dose_posterior.not_certified`.
    pub const NOT_CERTIFIED: &str = "smoothed_dose_posterior.not_certified";
    /// Detail `smoothed_dose_posterior.resource_limit`.
    pub const RESOURCE_LIMIT: &str = "smoothed_dose_posterior.resource_limit";
}

/// Details in the `source_evidence` family.
pub mod source_evidence {
    /// Detail `source_evidence.action_mismatch`.
    pub const ACTION_MISMATCH: &str = "source_evidence.action_mismatch";
    /// Detail `source_evidence.coordinate_mismatch`.
    pub const COORDINATE_MISMATCH: &str = "source_evidence.coordinate_mismatch";
    /// Detail `source_evidence.coordinates_unavailable`.
    pub const COORDINATES_UNAVAILABLE: &str = "source_evidence.coordinates_unavailable";
    /// Detail `source_evidence.diagnostic_scope_mismatch`.
    pub const DIAGNOSTIC_SCOPE_MISMATCH: &str = "source_evidence.diagnostic_scope_mismatch";
    /// Detail `source_evidence.forward_binding_mismatch`.
    pub const FORWARD_BINDING_MISMATCH: &str = "source_evidence.forward_binding_mismatch";
    /// Detail `source_evidence.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "source_evidence.invalid_artifact";
    /// Detail `source_evidence.invalid_contract`.
    pub const INVALID_CONTRACT: &str = "source_evidence.invalid_contract";
    /// Detail `source_evidence.invalid_coordinates`.
    pub const INVALID_COORDINATES: &str = "source_evidence.invalid_coordinates";
    /// Detail `source_evidence.invalid_source_artifact`.
    pub const INVALID_SOURCE_ARTIFACT: &str = "source_evidence.invalid_source_artifact";
    /// Detail `source_evidence.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "source_evidence.limits_exceeded";
    /// Detail `source_evidence.point_binding_mismatch`.
    pub const POINT_BINDING_MISMATCH: &str = "source_evidence.point_binding_mismatch";
    /// Detail `source_evidence.resolution_scope_unsupported`.
    pub const RESOLUTION_SCOPE_UNSUPPORTED: &str = "source_evidence.resolution_scope_unsupported";
    /// Detail `source_evidence.response_unavailable`.
    pub const RESPONSE_UNAVAILABLE: &str = "source_evidence.response_unavailable";
    /// Detail `source_evidence.source_binding_mismatch`.
    pub const SOURCE_BINDING_MISMATCH: &str = "source_evidence.source_binding_mismatch";
    /// Detail `source_evidence.source_encoding_failed`.
    pub const SOURCE_ENCODING_FAILED: &str = "source_evidence.source_encoding_failed";
    /// Detail `source_evidence.source_execution_failed`.
    pub const SOURCE_EXECUTION_FAILED: &str = "source_evidence.source_execution_failed";
    /// Detail `source_evidence.source_unverified`.
    pub const SOURCE_UNVERIFIED: &str = "source_evidence.source_unverified";
}

/// Details in the `source_projection` family.
pub mod source_projection {
    /// Detail `source_projection.affine_contract_refused`.
    pub const AFFINE_CONTRACT_REFUSED: &str = "source_projection.affine_contract_refused";
    /// Detail `source_projection.attestation_unavailable`.
    pub const ATTESTATION_UNAVAILABLE: &str = "source_projection.attestation_unavailable";
    /// Detail `source_projection.causal_source_unavailable`.
    pub const CAUSAL_SOURCE_UNAVAILABLE: &str = "source_projection.causal_source_unavailable";
    /// Detail `source_projection.coordinate_mismatch`.
    pub const COORDINATE_MISMATCH: &str = "source_projection.coordinate_mismatch";
    /// Detail `source_projection.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "source_projection.identity_mismatch";
    /// Detail `source_projection.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "source_projection.invalid_artifact";
    /// Detail `source_projection.invalid_request`.
    pub const INVALID_REQUEST: &str = "source_projection.invalid_request";
    /// Detail `source_projection.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "source_projection.limits_exceeded";
    /// Detail `source_projection.mean_binding_mismatch`.
    pub const MEAN_BINDING_MISMATCH: &str = "source_projection.mean_binding_mismatch";
    /// Detail `source_projection.mean_grid_required`.
    pub const MEAN_GRID_REQUIRED: &str = "source_projection.mean_grid_required";
    /// Detail `source_projection.original_consumer_refused`.
    pub const ORIGINAL_CONSUMER_REFUSED: &str = "source_projection.original_consumer_refused";
    /// Detail `source_projection.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "source_projection.replay_mismatch";
    /// Detail `source_projection.response_unavailable`.
    pub const RESPONSE_UNAVAILABLE: &str = "source_projection.response_unavailable";
    /// Detail `source_projection.source_binding_mismatch`.
    pub const SOURCE_BINDING_MISMATCH: &str = "source_projection.source_binding_mismatch";
    /// Detail `source_projection.unsupported_coordinate`.
    pub const UNSUPPORTED_COORDINATE: &str = "source_projection.unsupported_coordinate";
}

/// Details in the `split_effect` family.
pub mod split_effect {
    /// Detail `split_effect.degenerate_variance`.
    pub const DEGENERATE_VARIANCE: &str = "split_effect.degenerate_variance";
    /// Detail `split_effect.design_singular`.
    pub const DESIGN_SINGULAR: &str = "split_effect.design_singular";
    /// Detail `split_effect.discovery_failed`.
    pub const DISCOVERY_FAILED: &str = "split_effect.discovery_failed";
    /// Detail `split_effect.outcome_precedes_treatment`.
    pub const OUTCOME_PRECEDES_TREATMENT: &str = "split_effect.outcome_precedes_treatment";
    /// Detail `split_effect.search_not_finished`.
    pub const SEARCH_NOT_FINISHED: &str = "split_effect.search_not_finished";
    /// Detail `split_effect.source_invalid`.
    pub const SOURCE_INVALID: &str = "split_effect.source_invalid";
    /// Detail `split_effect.source_not_canonical`.
    pub const SOURCE_NOT_CANONICAL: &str = "split_effect.source_not_canonical";
    /// Detail `split_effect.structure_class`.
    pub const STRUCTURE_CLASS: &str = "split_effect.structure_class";
    /// Detail `split_effect.structure_unresolved`.
    pub const STRUCTURE_UNRESOLVED: &str = "split_effect.structure_unresolved";
    /// Detail `split_effect.too_few_rows`.
    pub const TOO_FEW_ROWS: &str = "split_effect.too_few_rows";
}

/// Details in the `structured_refusals` family.
pub mod structured_refusals {
    /// Detail `structured_refusals.malformed_detail`.
    pub const MALFORMED_DETAIL: &str = "structured_refusals.malformed_detail";
    /// Detail `structured_refusals.missing_stage`.
    pub const MISSING_STAGE: &str = "structured_refusals.missing_stage";
    /// Detail `structured_refusals.unregistered_code`.
    pub const UNREGISTERED_CODE: &str = "structured_refusals.unregistered_code";
}

/// Details in the `study_candidate` family.
pub mod study_candidate {
    /// Detail `study_candidate.declarations_required`.
    pub const DECLARATIONS_REQUIRED: &str = "study_candidate.declarations_required";
    /// Detail `study_candidate.sample_below_obligation`.
    pub const SAMPLE_BELOW_OBLIGATION: &str = "study_candidate.sample_below_obligation";
    /// Detail `study_candidate.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "study_candidate.wrong_contract";
}

/// Details in the `study_plan` family.
pub mod study_plan {
    /// Detail `study_plan.arrival_mismatch`.
    pub const ARRIVAL_MISMATCH: &str = "study_plan.arrival_mismatch";
    /// Detail `study_plan.arrival_not_identified`.
    pub const ARRIVAL_NOT_IDENTIFIED: &str = "study_plan.arrival_not_identified";
    /// Detail `study_plan.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "study_plan.bounds_exceeded";
    /// Detail `study_plan.budget`.
    pub const BUDGET: &str = "study_plan.budget";
    /// Detail `study_plan.invalid_artifact`.
    pub const INVALID_ARTIFACT: &str = "study_plan.invalid_artifact";
    /// Detail `study_plan.invalid_candidate`.
    pub const INVALID_CANDIDATE: &str = "study_plan.invalid_candidate";
    /// Detail `study_plan.invalid_query`.
    pub const INVALID_QUERY: &str = "study_plan.invalid_query";
    /// Detail `study_plan.no_failure_to_repair`.
    pub const NO_FAILURE_TO_REPAIR: &str = "study_plan.no_failure_to_repair";
    /// Detail `study_plan.none_certified`.
    pub const NONE_CERTIFIED: &str = "study_plan.none_certified";
    /// Detail `study_plan.theorem_limited`.
    pub const THEOREM_LIMITED: &str = "study_plan.theorem_limited";
}

/// Details in the `survival_effect` family.
pub mod survival_effect {
    /// Detail `survival_effect.arm_not_populated`.
    pub const ARM_NOT_POPULATED: &str = "survival_effect.arm_not_populated";
    /// Detail `survival_effect.cancelled`.
    pub const CANCELLED: &str = "survival_effect.cancelled";
    /// Detail `survival_effect.censoring_assumption_missing`.
    pub const CENSORING_ASSUMPTION_MISSING: &str = "survival_effect.censoring_assumption_missing";
    /// Detail `survival_effect.censoring_assumption_unknown`.
    pub const CENSORING_ASSUMPTION_UNKNOWN: &str = "survival_effect.censoring_assumption_unknown";
    /// Detail `survival_effect.censoring_covariates_missing`.
    pub const CENSORING_COVARIATES_MISSING: &str = "survival_effect.censoring_covariates_missing";
    /// Detail `survival_effect.censoring_support`.
    pub const CENSORING_SUPPORT: &str = "survival_effect.censoring_support";
    /// Detail `survival_effect.horizon_beyond_follow_up`.
    pub const HORIZON_BEYOND_FOLLOW_UP: &str = "survival_effect.horizon_beyond_follow_up";
    /// Detail `survival_effect.horizon_missing`.
    pub const HORIZON_MISSING: &str = "survival_effect.horizon_missing";
    /// Detail `survival_effect.invalid_data`.
    pub const INVALID_DATA: &str = "survival_effect.invalid_data";
    /// Detail `survival_effect.invalid_event`.
    pub const INVALID_EVENT: &str = "survival_effect.invalid_event";
    /// Detail `survival_effect.invalid_floor`.
    pub const INVALID_FLOOR: &str = "survival_effect.invalid_floor";
    /// Detail `survival_effect.invalid_folds`.
    pub const INVALID_FOLDS: &str = "survival_effect.invalid_folds";
    /// Detail `survival_effect.invalid_horizon`.
    pub const INVALID_HORIZON: &str = "survival_effect.invalid_horizon";
    /// Detail `survival_effect.invalid_levels`.
    pub const INVALID_LEVELS: &str = "survival_effect.invalid_levels";
    /// Detail `survival_effect.invalid_request`.
    pub const INVALID_REQUEST: &str = "survival_effect.invalid_request";
    /// Detail `survival_effect.too_few_rows`.
    pub const TOO_FEW_ROWS: &str = "survival_effect.too_few_rows";
    /// Detail `survival_effect.training_fold_without_events`.
    pub const TRAINING_FOLD_WITHOUT_EVENTS: &str = "survival_effect.training_fold_without_events";
    /// Detail `survival_effect.treatment_positivity`.
    pub const TREATMENT_POSITIVITY: &str = "survival_effect.treatment_positivity";
    /// Detail `survival_effect.undeclared_level`.
    pub const UNDECLARED_LEVEL: &str = "survival_effect.undeclared_level";
}

/// Details in the `temporal_counterfactual` family.
pub mod temporal_counterfactual {
    /// Detail `temporal_counterfactual.artifact_changed`.
    pub const ARTIFACT_CHANGED: &str = "temporal_counterfactual.artifact_changed";
    /// Detail `temporal_counterfactual.fit_mismatch`.
    pub const FIT_MISMATCH: &str = "temporal_counterfactual.fit_mismatch";
    /// Detail `temporal_counterfactual.history_name_invalid`.
    pub const HISTORY_NAME_INVALID: &str = "temporal_counterfactual.history_name_invalid";
    /// Detail `temporal_counterfactual.horizon_exceeded`.
    pub const HORIZON_EXCEEDED: &str = "temporal_counterfactual.horizon_exceeded";
    /// Detail `temporal_counterfactual.invalid_graph`.
    pub const INVALID_GRAPH: &str = "temporal_counterfactual.invalid_graph";
    /// Detail `temporal_counterfactual.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "temporal_counterfactual.invalid_identity";
    /// Detail `temporal_counterfactual.invalid_request`.
    pub const INVALID_REQUEST: &str = "temporal_counterfactual.invalid_request";
    /// Detail `temporal_counterfactual.latent_confounding`.
    pub const LATENT_CONFOUNDING: &str = "temporal_counterfactual.latent_confounding";
    /// Detail `temporal_counterfactual.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "temporal_counterfactual.limits_exceeded";
    /// Detail `temporal_counterfactual.non_finite_history`.
    pub const NON_FINITE_HISTORY: &str = "temporal_counterfactual.non_finite_history";
    /// Detail `temporal_counterfactual.refuting_history`.
    pub const REFUTING_HISTORY: &str = "temporal_counterfactual.refuting_history";
    /// Detail `temporal_counterfactual.shared_history_missing`.
    pub const SHARED_HISTORY_MISSING: &str = "temporal_counterfactual.shared_history_missing";
    /// Detail `temporal_counterfactual.time_misaligned`.
    pub const TIME_MISALIGNED: &str = "temporal_counterfactual.time_misaligned";
    /// Detail `temporal_counterfactual.too_many_units`.
    pub const TOO_MANY_UNITS: &str = "temporal_counterfactual.too_many_units";
    /// Detail `temporal_counterfactual.unpaired_histories`.
    pub const UNPAIRED_HISTORIES: &str = "temporal_counterfactual.unpaired_histories";
}

/// Details in the `temporal_interval` family.
pub mod temporal_interval {
    /// Detail `temporal_interval.consumer_unavailable`.
    pub const CONSUMER_UNAVAILABLE: &str = "temporal_interval.consumer_unavailable";
    /// Detail `temporal_interval.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "temporal_interval.interval_not_supported";
    /// Detail `temporal_interval.invalid_estimand`.
    pub const INVALID_ESTIMAND: &str = "temporal_interval.invalid_estimand";
    /// Detail `temporal_interval.retired_method`.
    pub const RETIRED_METHOD: &str = "temporal_interval.retired_method";
    /// Detail `temporal_interval.retired_response_method`.
    pub const RETIRED_RESPONSE_METHOD: &str = "temporal_interval.retired_response_method";
    /// Detail `temporal_interval.studentized_bounds_exceeded`.
    pub const STUDENTIZED_BOUNDS_EXCEEDED: &str = "temporal_interval.studentized_bounds_exceeded";
    /// Detail `temporal_interval.studentized_estimator_not_certified`.
    pub const STUDENTIZED_ESTIMATOR_NOT_CERTIFIED: &str = "temporal_interval.studentized_estimator_not_certified";
    /// Detail `temporal_interval.studentized_score_mismatch`.
    pub const STUDENTIZED_SCORE_MISMATCH: &str = "temporal_interval.studentized_score_mismatch";
    /// Detail `temporal_interval.studentized_unbalanced`.
    pub const STUDENTIZED_UNBALANCED: &str = "temporal_interval.studentized_unbalanced";
    /// Detail `temporal_interval.studentized_zero_variance`.
    pub const STUDENTIZED_ZERO_VARIANCE: &str = "temporal_interval.studentized_zero_variance";
    /// Detail `temporal_interval.too_few_units`.
    pub const TOO_FEW_UNITS: &str = "temporal_interval.too_few_units";
    /// Detail `temporal_interval.too_many_replicates`.
    pub const TOO_MANY_REPLICATES: &str = "temporal_interval.too_many_replicates";
    /// Detail `temporal_interval.unknown_units`.
    pub const UNKNOWN_UNITS: &str = "temporal_interval.unknown_units";
    /// Detail `temporal_interval.unsupported_history`.
    pub const UNSUPPORTED_HISTORY: &str = "temporal_interval.unsupported_history";
}

/// Details in the `temporal_policy_transport` family.
pub mod temporal_policy_transport {
    /// Detail `temporal_policy_transport.history_alphabet_incompatible`.
    pub const HISTORY_ALPHABET_INCOMPATIBLE: &str = "temporal_policy_transport.history_alphabet_incompatible";
    /// Detail `temporal_policy_transport.horizon_mismatch`.
    pub const HORIZON_MISMATCH: &str = "temporal_policy_transport.horizon_mismatch";
    /// Detail `temporal_policy_transport.initial_law_declared_invariant`.
    pub const INITIAL_LAW_DECLARED_INVARIANT: &str = "temporal_policy_transport.initial_law_declared_invariant";
    /// Detail `temporal_policy_transport.invariance_binding_mismatch`.
    pub const INVARIANCE_BINDING_MISMATCH: &str = "temporal_policy_transport.invariance_binding_mismatch";
    /// Detail `temporal_policy_transport.mechanism_declared_twice`.
    pub const MECHANISM_DECLARED_TWICE: &str = "temporal_policy_transport.mechanism_declared_twice";
    /// Detail `temporal_policy_transport.mechanism_status_missing`.
    pub const MECHANISM_STATUS_MISSING: &str = "temporal_policy_transport.mechanism_status_missing";
    /// Detail `temporal_policy_transport.mechanism_time_varying`.
    pub const MECHANISM_TIME_VARYING: &str = "temporal_policy_transport.mechanism_time_varying";
    /// Detail `temporal_policy_transport.mechanism_unknown`.
    pub const MECHANISM_UNKNOWN: &str = "temporal_policy_transport.mechanism_unknown";
    /// Detail `temporal_policy_transport.target_initial_law`.
    pub const TARGET_INITIAL_LAW: &str = "temporal_policy_transport.target_initial_law";
    /// Detail `temporal_policy_transport.target_initial_law_missing`.
    pub const TARGET_INITIAL_LAW_MISSING: &str = "temporal_policy_transport.target_initial_law_missing";
    /// Detail `temporal_policy_transport.target_is_source`.
    pub const TARGET_IS_SOURCE: &str = "temporal_policy_transport.target_is_source";
    /// Detail `temporal_policy_transport.target_period`.
    pub const TARGET_PERIOD: &str = "temporal_policy_transport.target_period";
    /// Detail `temporal_policy_transport.target_support_outside_source`.
    pub const TARGET_SUPPORT_OUTSIDE_SOURCE: &str = "temporal_policy_transport.target_support_outside_source";
    /// Detail `temporal_policy_transport.time_invariance_missing`.
    pub const TIME_INVARIANCE_MISSING: &str = "temporal_policy_transport.time_invariance_missing";
    /// Detail `temporal_policy_transport.value_mismatch`.
    pub const VALUE_MISMATCH: &str = "temporal_policy_transport.value_mismatch";
}

/// Details in the `temporal_refresh` family.
pub mod temporal_refresh {
    /// Detail `temporal_refresh.factor_window_mismatch`.
    pub const FACTOR_WINDOW_MISMATCH: &str = "temporal_refresh.factor_window_mismatch";
    /// Detail `temporal_refresh.horizon_changed`.
    pub const HORIZON_CHANGED: &str = "temporal_refresh.horizon_changed";
    /// Detail `temporal_refresh.intervention_history_changed`.
    pub const INTERVENTION_HISTORY_CHANGED: &str = "temporal_refresh.intervention_history_changed";
    /// Detail `temporal_refresh.invalid_initial_law`.
    pub const INVALID_INITIAL_LAW: &str = "temporal_refresh.invalid_initial_law";
    /// Detail `temporal_refresh.invalid_replacement`.
    pub const INVALID_REPLACEMENT: &str = "temporal_refresh.invalid_replacement";
    /// Detail `temporal_refresh.lag_alignment_changed`.
    pub const LAG_ALIGNMENT_CHANGED: &str = "temporal_refresh.lag_alignment_changed";
    /// Detail `temporal_refresh.no_refresh`.
    pub const NO_REFRESH: &str = "temporal_refresh.no_refresh";
    /// Detail `temporal_refresh.premises_changed`.
    pub const PREMISES_CHANGED: &str = "temporal_refresh.premises_changed";
    /// Detail `temporal_refresh.receipt_digest_mismatch`.
    pub const RECEIPT_DIGEST_MISMATCH: &str = "temporal_refresh.receipt_digest_mismatch";
    /// Detail `temporal_refresh.receipt_mismatch`.
    pub const RECEIPT_MISMATCH: &str = "temporal_refresh.receipt_mismatch";
    /// Detail `temporal_refresh.stale_interval`.
    pub const STALE_INTERVAL: &str = "temporal_refresh.stale_interval";
    /// Detail `temporal_refresh.stale_snapshot`.
    pub const STALE_SNAPSHOT: &str = "temporal_refresh.stale_snapshot";
    /// Detail `temporal_refresh.window_missing`.
    pub const WINDOW_MISSING: &str = "temporal_refresh.window_missing";
}

/// Details in the `temporal_transport` family.
pub mod temporal_transport {
    /// Detail `temporal_transport.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "temporal_transport.bounds_exceeded";
    /// Detail `temporal_transport.checked_obstruction`.
    pub const CHECKED_OBSTRUCTION: &str = "temporal_transport.checked_obstruction";
    /// Detail `temporal_transport.history_budget`.
    pub const HISTORY_BUDGET: &str = "temporal_transport.history_budget";
    /// Detail `temporal_transport.history_outside_support`.
    pub const HISTORY_OUTSIDE_SUPPORT: &str = "temporal_transport.history_outside_support";
    /// Detail `temporal_transport.horizon`.
    pub const HORIZON: &str = "temporal_transport.horizon";
    /// Detail `temporal_transport.interval_requested`.
    pub const INTERVAL_REQUESTED: &str = "temporal_transport.interval_requested";
    /// Detail `temporal_transport.invalid_sequence`.
    pub const INVALID_SEQUENCE: &str = "temporal_transport.invalid_sequence";
    /// Detail `temporal_transport.invalid_spec`.
    pub const INVALID_SPEC: &str = "temporal_transport.invalid_spec";
    /// Detail `temporal_transport.missing_evidence`.
    pub const MISSING_EVIDENCE: &str = "temporal_transport.missing_evidence";
    /// Detail `temporal_transport.search_incomplete`.
    pub const SEARCH_INCOMPLETE: &str = "temporal_transport.search_incomplete";
    /// Detail `temporal_transport.target_law_mismatch`.
    pub const TARGET_LAW_MISMATCH: &str = "temporal_transport.target_law_mismatch";
    /// Detail `temporal_transport.target_law_missing`.
    pub const TARGET_LAW_MISSING: &str = "temporal_transport.target_law_missing";
}

/// Details in the `tier_diagnostics` family.
pub mod tier_diagnostics {
    /// Detail `tier_diagnostics.cancelled`.
    pub const CANCELLED: &str = "tier_diagnostics.cancelled";
    /// Detail `tier_diagnostics.evalue_embeds_interval_limit`.
    pub const EVALUE_EMBEDS_INTERVAL_LIMIT: &str = "tier_diagnostics.evalue_embeds_interval_limit";
    /// Detail `tier_diagnostics.evalue_not_computed`.
    pub const EVALUE_NOT_COMPUTED: &str = "tier_diagnostics.evalue_not_computed";
    /// Detail `tier_diagnostics.no_evalue_interval_method`.
    pub const NO_EVALUE_INTERVAL_METHOD: &str = "tier_diagnostics.no_evalue_interval_method";
    /// Detail `tier_diagnostics.not_a_tiered_average_result`.
    pub const NOT_A_TIERED_AVERAGE_RESULT: &str = "tier_diagnostics.not_a_tiered_average_result";
    /// Detail `tier_diagnostics.overlap_not_carried`.
    pub const OVERLAP_NOT_CARRIED: &str = "tier_diagnostics.overlap_not_carried";
    /// Detail `tier_diagnostics.unknown_scenarios_no_evalue`.
    pub const UNKNOWN_SCENARIOS_NO_EVALUE: &str = "tier_diagnostics.unknown_scenarios_no_evalue";
    /// Detail `tier_diagnostics.unknown_scenarios_no_overlap`.
    pub const UNKNOWN_SCENARIOS_NO_OVERLAP: &str = "tier_diagnostics.unknown_scenarios_no_overlap";
}

/// Details in the `transport_revalidation` family.
pub mod transport_revalidation {
    /// Detail `transport_revalidation.claim_target_mismatch`.
    pub const CLAIM_TARGET_MISMATCH: &str = "transport_revalidation.claim_target_mismatch";
    /// Detail `transport_revalidation.duplicate_claim`.
    pub const DUPLICATE_CLAIM: &str = "transport_revalidation.duplicate_claim";
    /// Detail `transport_revalidation.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "transport_revalidation.identity_mismatch";
    /// Detail `transport_revalidation.provider_trust_not_applicable`.
    pub const PROVIDER_TRUST_NOT_APPLICABLE: &str = "transport_revalidation.provider_trust_not_applicable";
    /// Detail `transport_revalidation.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "transport_revalidation.replay_mismatch";
    /// Detail `transport_revalidation.scene_invalid`.
    pub const SCENE_INVALID: &str = "transport_revalidation.scene_invalid";
    /// Detail `transport_revalidation.source_undeclared`.
    pub const SOURCE_UNDECLARED: &str = "transport_revalidation.source_undeclared";
    /// Detail `transport_revalidation.unknown_claim`.
    pub const UNKNOWN_CLAIM: &str = "transport_revalidation.unknown_claim";
    /// Detail `transport_revalidation.value_digest`.
    pub const VALUE_DIGEST: &str = "transport_revalidation.value_digest";
}

/// Details in the `transport_sensitivity` family.
pub mod transport_sensitivity {
    /// Detail `transport_sensitivity.blank_identity`.
    pub const BLANK_IDENTITY: &str = "transport_sensitivity.blank_identity";
    /// Detail `transport_sensitivity.coordinate_out_of_range`.
    pub const COORDINATE_OUT_OF_RANGE: &str = "transport_sensitivity.coordinate_out_of_range";
    /// Detail `transport_sensitivity.interval_not_composed`.
    pub const INTERVAL_NOT_COMPOSED: &str = "transport_sensitivity.interval_not_composed";
    /// Detail `transport_sensitivity.invalid_bound`.
    pub const INVALID_BOUND: &str = "transport_sensitivity.invalid_bound";
    /// Detail `transport_sensitivity.invalid_grid`.
    pub const INVALID_GRID: &str = "transport_sensitivity.invalid_grid";
    /// Detail `transport_sensitivity.law_response_not_scalar`.
    pub const LAW_RESPONSE_NOT_SCALAR: &str = "transport_sensitivity.law_response_not_scalar";
    /// Detail `transport_sensitivity.no_scalar_family`.
    pub const NO_SCALAR_FAMILY: &str = "transport_sensitivity.no_scalar_family";
    /// Detail `transport_sensitivity.non_finite_point`.
    pub const NON_FINITE_POINT: &str = "transport_sensitivity.non_finite_point";
    /// Detail `transport_sensitivity.not_transported`.
    pub const NOT_TRANSPORTED: &str = "transport_sensitivity.not_transported";
    /// Detail `transport_sensitivity.proportional_at_zero`.
    pub const PROPORTIONAL_AT_ZERO: &str = "transport_sensitivity.proportional_at_zero";
    /// Detail `transport_sensitivity.quantity_not_in_target`.
    pub const QUANTITY_NOT_IN_TARGET: &str = "transport_sensitivity.quantity_not_in_target";
    /// Detail `transport_sensitivity.route_undeclared`.
    pub const ROUTE_UNDECLARED: &str = "transport_sensitivity.route_undeclared";
    /// Detail `transport_sensitivity.surface_refused`.
    pub const SURFACE_REFUSED: &str = "transport_sensitivity.surface_refused";
}

/// Details in the `transported_counterfactual` family.
pub mod transported_counterfactual {
    /// Detail `transported_counterfactual.artifact_changed`.
    pub const ARTIFACT_CHANGED: &str = "transported_counterfactual.artifact_changed";
    /// Detail `transported_counterfactual.cross_world_independence_missing`.
    pub const CROSS_WORLD_INDEPENDENCE_MISSING: &str = "transported_counterfactual.cross_world_independence_missing";
    /// Detail `transported_counterfactual.factor_missing`.
    pub const FACTOR_MISSING: &str = "transported_counterfactual.factor_missing";
    /// Detail `transported_counterfactual.invalid_diagram`.
    pub const INVALID_DIAGRAM: &str = "transported_counterfactual.invalid_diagram";
    /// Detail `transported_counterfactual.invalid_factor`.
    pub const INVALID_FACTOR: &str = "transported_counterfactual.invalid_factor";
    /// Detail `transported_counterfactual.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "transported_counterfactual.invalid_identity";
    /// Detail `transported_counterfactual.invalid_law`.
    pub const INVALID_LAW: &str = "transported_counterfactual.invalid_law";
    /// Detail `transported_counterfactual.invalid_model`.
    pub const INVALID_MODEL: &str = "transported_counterfactual.invalid_model";
    /// Detail `transported_counterfactual.invalid_query`.
    pub const INVALID_QUERY: &str = "transported_counterfactual.invalid_query";
    /// Detail `transported_counterfactual.invalid_request`.
    pub const INVALID_REQUEST: &str = "transported_counterfactual.invalid_request";
    /// Detail `transported_counterfactual.joint_theorem_not_derived`.
    pub const JOINT_THEOREM_NOT_DERIVED: &str = "transported_counterfactual.joint_theorem_not_derived";
    /// Detail `transported_counterfactual.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "transported_counterfactual.limits_exceeded";
    /// Detail `transported_counterfactual.noise_law_not_shared`.
    pub const NOISE_LAW_NOT_SHARED: &str = "transported_counterfactual.noise_law_not_shared";
    /// Detail `transported_counterfactual.nonadditive_mechanism`.
    pub const NONADDITIVE_MECHANISM: &str = "transported_counterfactual.nonadditive_mechanism";
    /// Detail `transported_counterfactual.overlap_failure`.
    pub const OVERLAP_FAILURE: &str = "transported_counterfactual.overlap_failure";
    /// Detail `transported_counterfactual.prerequisite_missing`.
    pub const PREREQUISITE_MISSING: &str = "transported_counterfactual.prerequisite_missing";
    /// Detail `transported_counterfactual.selection_on_mechanism`.
    pub const SELECTION_ON_MECHANISM: &str = "transported_counterfactual.selection_on_mechanism";
}

/// Details in the `transported_decision` family.
pub mod transported_decision {
    /// Detail `transported_decision.calibration_label_inflated`.
    pub const CALIBRATION_LABEL_INFLATED: &str = "transported_decision.calibration_label_inflated";
    /// Detail `transported_decision.coordinate_not_in_target`.
    pub const COORDINATE_NOT_IN_TARGET: &str = "transported_decision.coordinate_not_in_target";
    /// Detail `transported_decision.exclusions_leave_no_comparison`.
    pub const EXCLUSIONS_LEAVE_NO_COMPARISON: &str = "transported_decision.exclusions_leave_no_comparison";
    /// Detail `transported_decision.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "transported_decision.interval_not_supported";
    /// Detail `transported_decision.invalid_value`.
    pub const INVALID_VALUE: &str = "transported_decision.invalid_value";
    /// Detail `transported_decision.law_not_supplied`.
    pub const LAW_NOT_SUPPLIED: &str = "transported_decision.law_not_supplied";
    /// Detail `transported_decision.mixed_transports`.
    pub const MIXED_TRANSPORTS: &str = "transported_decision.mixed_transports";
    /// Detail `transported_decision.native_label_on_transported_value`.
    pub const NATIVE_LABEL_ON_TRANSPORTED_VALUE: &str = "transported_decision.native_label_on_transported_value";
    /// Detail `transported_decision.not_transported`.
    pub const NOT_TRANSPORTED: &str = "transported_decision.not_transported";
    /// Detail `transported_decision.route_needs_typed_source`.
    pub const ROUTE_NEEDS_TYPED_SOURCE: &str = "transported_decision.route_needs_typed_source";
    /// Detail `transported_decision.support_label_inflated`.
    pub const SUPPORT_LABEL_INFLATED: &str = "transported_decision.support_label_inflated";
    /// Detail `transported_decision.support_rule_unmet`.
    pub const SUPPORT_RULE_UNMET: &str = "transported_decision.support_rule_unmet";
    /// Detail `transported_decision.transport_identity_incomplete`.
    pub const TRANSPORT_IDENTITY_INCOMPLETE: &str = "transported_decision.transport_identity_incomplete";
    /// Detail `transported_decision.transport_identity_not_digest`.
    pub const TRANSPORT_IDENTITY_NOT_DIGEST: &str = "transported_decision.transport_identity_not_digest";
    /// Detail `transported_decision.transported_not_accepted`.
    pub const TRANSPORTED_NOT_ACCEPTED: &str = "transported_decision.transported_not_accepted";
    /// Detail `transported_decision.uncertainty_requirement`.
    pub const UNCERTAINTY_REQUIREMENT: &str = "transported_decision.uncertainty_requirement";
}

/// Details in the `transported_decision_artifact` family.
pub mod transported_decision_artifact {
    /// Detail `transported_decision_artifact.bounds_exceeded`.
    pub const BOUNDS_EXCEEDED: &str = "transported_decision_artifact.bounds_exceeded";
    /// Detail `transported_decision_artifact.identity_expected`.
    pub const IDENTITY_EXPECTED: &str = "transported_decision_artifact.identity_expected";
    /// Detail `transported_decision_artifact.invalid`.
    pub const INVALID: &str = "transported_decision_artifact.invalid";
    /// Detail `transported_decision_artifact.law_not_exportable`.
    pub const LAW_NOT_EXPORTABLE: &str = "transported_decision_artifact.law_not_exportable";
    /// Detail `transported_decision_artifact.replay_mismatch`.
    pub const REPLAY_MISMATCH: &str = "transported_decision_artifact.replay_mismatch";
}

/// Details in the `transported_effect` family.
pub mod transported_effect {
    /// Detail `transported_effect.cohort_invalid`.
    pub const COHORT_INVALID: &str = "transported_effect.cohort_invalid";
    /// Detail `transported_effect.cohort_not_numeric`.
    pub const COHORT_NOT_NUMERIC: &str = "transported_effect.cohort_not_numeric";
    /// Detail `transported_effect.decision_source_mismatch`.
    pub const DECISION_SOURCE_MISMATCH: &str = "transported_effect.decision_source_mismatch";
    /// Detail `transported_effect.declaration_invalid`.
    pub const DECLARATION_INVALID: &str = "transported_effect.declaration_invalid";
    /// Detail `transported_effect.feature_schema_mismatch`.
    pub const FEATURE_SCHEMA_MISMATCH: &str = "transported_effect.feature_schema_mismatch";
    /// Detail `transported_effect.identity_mismatch`.
    pub const IDENTITY_MISMATCH: &str = "transported_effect.identity_mismatch";
    /// Detail `transported_effect.interval_not_supported`.
    pub const INTERVAL_NOT_SUPPORTED: &str = "transported_effect.interval_not_supported";
    /// Detail `transported_effect.invalid_request`.
    pub const INVALID_REQUEST: &str = "transported_effect.invalid_request";
    /// Detail `transported_effect.law_not_supplied`.
    pub const LAW_NOT_SUPPLIED: &str = "transported_effect.law_not_supplied";
    /// Detail `transported_effect.link_not_identity`.
    pub const LINK_NOT_IDENTITY: &str = "transported_effect.link_not_identity";
    /// Detail `transported_effect.model_invalid`.
    pub const MODEL_INVALID: &str = "transported_effect.model_invalid";
    /// Detail `transported_effect.model_not_portable`.
    pub const MODEL_NOT_PORTABLE: &str = "transported_effect.model_not_portable";
    /// Detail `transported_effect.modifier_set_mismatch`.
    pub const MODIFIER_SET_MISMATCH: &str = "transported_effect.modifier_set_mismatch";
    /// Detail `transported_effect.no_overlap`.
    pub const NO_OVERLAP: &str = "transported_effect.no_overlap";
    /// Detail `transported_effect.non_finite_result`.
    pub const NON_FINITE_RESULT: &str = "transported_effect.non_finite_result";
    /// Detail `transported_effect.out_of_support`.
    pub const OUT_OF_SUPPORT: &str = "transported_effect.out_of_support";
    /// Detail `transported_effect.outcome_units_required`.
    pub const OUTCOME_UNITS_REQUIRED: &str = "transported_effect.outcome_units_required";
    /// Detail `transported_effect.population_not_different`.
    pub const POPULATION_NOT_DIFFERENT: &str = "transported_effect.population_not_different";
    /// Detail `transported_effect.prediction_refused`.
    pub const PREDICTION_REFUSED: &str = "transported_effect.prediction_refused";
    /// Detail `transported_effect.selection_enters_mechanism`.
    pub const SELECTION_ENTERS_MECHANISM: &str = "transported_effect.selection_enters_mechanism";
    /// Detail `transported_effect.shift_declaration_mismatch`.
    pub const SHIFT_DECLARATION_MISMATCH: &str = "transported_effect.shift_declaration_mismatch";
    /// Detail `transported_effect.shift_declaration_required`.
    pub const SHIFT_DECLARATION_REQUIRED: &str = "transported_effect.shift_declaration_required";
    /// Detail `transported_effect.support_contract_absent`.
    pub const SUPPORT_CONTRACT_ABSENT: &str = "transported_effect.support_contract_absent";
    /// Detail `transported_effect.weights_invalid`.
    pub const WEIGHTS_INVALID: &str = "transported_effect.weights_invalid";
    /// Detail `transported_effect.weights_not_normalized`.
    pub const WEIGHTS_NOT_NORMALIZED: &str = "transported_effect.weights_not_normalized";
}

/// Details in the `transported_policy` family.
pub mod transported_policy {
    /// Detail `transported_policy.diagram_binding_mismatch`.
    pub const DIAGRAM_BINDING_MISMATCH: &str = "transported_policy.diagram_binding_mismatch";
    /// Detail `transported_policy.initial_law_declared_invariant`.
    pub const INITIAL_LAW_DECLARED_INVARIANT: &str = "transported_policy.initial_law_declared_invariant";
    /// Detail `transported_policy.mechanism_declared_twice`.
    pub const MECHANISM_DECLARED_TWICE: &str = "transported_policy.mechanism_declared_twice";
    /// Detail `transported_policy.mechanism_not_invariant`.
    pub const MECHANISM_NOT_INVARIANT: &str = "transported_policy.mechanism_not_invariant";
    /// Detail `transported_policy.mechanism_status_missing`.
    pub const MECHANISM_STATUS_MISSING: &str = "transported_policy.mechanism_status_missing";
    /// Detail `transported_policy.mechanism_unknown`.
    pub const MECHANISM_UNKNOWN: &str = "transported_policy.mechanism_unknown";
    /// Detail `transported_policy.selection_diagram_missing`.
    pub const SELECTION_DIAGRAM_MISSING: &str = "transported_policy.selection_diagram_missing";
    /// Detail `transported_policy.target_initial_law`.
    pub const TARGET_INITIAL_LAW: &str = "transported_policy.target_initial_law";
    /// Detail `transported_policy.target_initial_law_missing`.
    pub const TARGET_INITIAL_LAW_MISSING: &str = "transported_policy.target_initial_law_missing";
    /// Detail `transported_policy.target_is_source`.
    pub const TARGET_IS_SOURCE: &str = "transported_policy.target_is_source";
    /// Detail `transported_policy.target_population`.
    pub const TARGET_POPULATION: &str = "transported_policy.target_population";
    /// Detail `transported_policy.target_support_outside_source`.
    pub const TARGET_SUPPORT_OUTSIDE_SOURCE: &str = "transported_policy.target_support_outside_source";
    /// Detail `transported_policy.target_utility_digest`.
    pub const TARGET_UTILITY_DIGEST: &str = "transported_policy.target_utility_digest";
    /// Detail `transported_policy.target_utility_missing`.
    pub const TARGET_UTILITY_MISSING: &str = "transported_policy.target_utility_missing";
    /// Detail `transported_policy.target_utility_provenance_missing`.
    pub const TARGET_UTILITY_PROVENANCE_MISSING: &str = "transported_policy.target_utility_provenance_missing";
    /// Detail `transported_policy.target_utility_support`.
    pub const TARGET_UTILITY_SUPPORT: &str = "transported_policy.target_utility_support";
    /// Detail `transported_policy.target_utility_value`.
    pub const TARGET_UTILITY_VALUE: &str = "transported_policy.target_utility_value";
    /// Detail `transported_policy.utility_declared_invariant`.
    pub const UTILITY_DECLARED_INVARIANT: &str = "transported_policy.utility_declared_invariant";
    /// Detail `transported_policy.value_mismatch`.
    pub const VALUE_MISMATCH: &str = "transported_policy.value_mismatch";
}

/// Details in the `transported_synthesis` family.
pub mod transported_synthesis {
    /// Detail `transported_synthesis.native_population_blank`.
    pub const NATIVE_POPULATION_BLANK: &str = "transported_synthesis.native_population_blank";
    /// Detail `transported_synthesis.no_contributions`.
    pub const NO_CONTRIBUTIONS: &str = "transported_synthesis.no_contributions";
    /// Detail `transported_synthesis.numeric_refused`.
    pub const NUMERIC_REFUSED: &str = "transported_synthesis.numeric_refused";
    /// Detail `transported_synthesis.overlap_not_modelled`.
    pub const OVERLAP_NOT_MODELLED: &str = "transported_synthesis.overlap_not_modelled";
    /// Detail `transported_synthesis.prior_role_mismatch`.
    pub const PRIOR_ROLE_MISMATCH: &str = "transported_synthesis.prior_role_mismatch";
    /// Detail `transported_synthesis.shared_transport_needs_covariance`.
    pub const SHARED_TRANSPORT_NEEDS_COVARIANCE: &str = "transported_synthesis.shared_transport_needs_covariance";
    /// Detail `transported_synthesis.source_listed_twice`.
    pub const SOURCE_LISTED_TWICE: &str = "transported_synthesis.source_listed_twice";
    /// Detail `transported_synthesis.target_population_mismatch`.
    pub const TARGET_POPULATION_MISMATCH: &str = "transported_synthesis.target_population_mismatch";
    /// Detail `transported_synthesis.transport_identity_incomplete`.
    pub const TRANSPORT_IDENTITY_INCOMPLETE: &str = "transported_synthesis.transport_identity_incomplete";
    /// Detail `transported_synthesis.transport_variance_on_native_source`.
    pub const TRANSPORT_VARIANCE_ON_NATIVE_SOURCE: &str = "transported_synthesis.transport_variance_on_native_source";
    /// Detail `transported_synthesis.transport_variance_undeclared`.
    pub const TRANSPORT_VARIANCE_UNDECLARED: &str = "transported_synthesis.transport_variance_undeclared";
    /// Detail `transported_synthesis.unknown_source`.
    pub const UNKNOWN_SOURCE: &str = "transported_synthesis.unknown_source";
}

/// Details in the `treatment_levels` family.
pub mod treatment_levels {
    /// Detail `treatment_levels.cancelled`.
    pub const CANCELLED: &str = "treatment_levels.cancelled";
    /// Detail `treatment_levels.contrast_level_not_declared`.
    pub const CONTRAST_LEVEL_NOT_DECLARED: &str = "treatment_levels.contrast_level_not_declared";
    /// Detail `treatment_levels.full_population_undeclared_level`.
    pub const FULL_POPULATION_UNDECLARED_LEVEL: &str = "treatment_levels.full_population_undeclared_level";
    /// Detail `treatment_levels.invalid_data`.
    pub const INVALID_DATA: &str = "treatment_levels.invalid_data";
    /// Detail `treatment_levels.invalid_level_set`.
    pub const INVALID_LEVEL_SET: &str = "treatment_levels.invalid_level_set";
    /// Detail `treatment_levels.invalid_request`.
    pub const INVALID_REQUEST: &str = "treatment_levels.invalid_request";
    /// Detail `treatment_levels.level_not_populated`.
    pub const LEVEL_NOT_POPULATED: &str = "treatment_levels.level_not_populated";
    /// Detail `treatment_levels.levels_missing`.
    pub const LEVELS_MISSING: &str = "treatment_levels.levels_missing";
    /// Detail `treatment_levels.sparse_level`.
    pub const SPARSE_LEVEL: &str = "treatment_levels.sparse_level";
    /// Detail `treatment_levels.training_fold_missing_level`.
    pub const TRAINING_FOLD_MISSING_LEVEL: &str = "treatment_levels.training_fold_missing_level";
    /// Detail `treatment_levels.undeclared_level`.
    pub const UNDECLARED_LEVEL: &str = "treatment_levels.undeclared_level";
}

/// Details in the `two_step_posterior` family.
pub mod two_step_posterior {
    /// Detail `two_step_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "two_step_posterior.band_not_constructed";
    /// Detail `two_step_posterior.empty_law`.
    pub const EMPTY_LAW: &str = "two_step_posterior.empty_law";
    /// Detail `two_step_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "two_step_posterior.external_power_prior";
    /// Detail `two_step_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "two_step_posterior.mapped_prior";
    /// Detail `two_step_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "two_step_posterior.non_positive_scale";
}

/// Details in the `typed_mechanism_transport` family.
pub mod typed_mechanism_transport {
    /// Detail `typed_mechanism_transport.cycle`.
    pub const CYCLE: &str = "typed_mechanism_transport.cycle";
    /// Detail `typed_mechanism_transport.dependence_invalid`.
    pub const DEPENDENCE_INVALID: &str = "typed_mechanism_transport.dependence_invalid";
    /// Detail `typed_mechanism_transport.design_not_shared`.
    pub const DESIGN_NOT_SHARED: &str = "typed_mechanism_transport.design_not_shared";
    /// Detail `typed_mechanism_transport.fit_not_converged`.
    pub const FIT_NOT_CONVERGED: &str = "typed_mechanism_transport.fit_not_converged";
    /// Detail `typed_mechanism_transport.fit_refused`.
    pub const FIT_REFUSED: &str = "typed_mechanism_transport.fit_refused";
    /// Detail `typed_mechanism_transport.invariance_contradicted`.
    pub const INVARIANCE_CONTRADICTED: &str = "typed_mechanism_transport.invariance_contradicted";
    /// Detail `typed_mechanism_transport.invariance_level_invalid`.
    pub const INVARIANCE_LEVEL_INVALID: &str = "typed_mechanism_transport.invariance_level_invalid";
    /// Detail `typed_mechanism_transport.parameter_dependence_incomplete`.
    pub const PARAMETER_DEPENDENCE_INCOMPLETE: &str = "typed_mechanism_transport.parameter_dependence_incomplete";
    /// Detail `typed_mechanism_transport.parameter_dependence_unknown`.
    pub const PARAMETER_DEPENDENCE_UNKNOWN: &str = "typed_mechanism_transport.parameter_dependence_unknown";
    /// Detail `typed_mechanism_transport.parent_without_mechanism`.
    pub const PARENT_WITHOUT_MECHANISM: &str = "typed_mechanism_transport.parent_without_mechanism";
    /// Detail `typed_mechanism_transport.parents_changed`.
    pub const PARENTS_CHANGED: &str = "typed_mechanism_transport.parents_changed";
    /// Detail `typed_mechanism_transport.regime_not_declared_invariant`.
    pub const REGIME_NOT_DECLARED_INVARIANT: &str = "typed_mechanism_transport.regime_not_declared_invariant";
    /// Detail `typed_mechanism_transport.selection_target_unknown_variable`.
    pub const SELECTION_TARGET_UNKNOWN_VARIABLE: &str = "typed_mechanism_transport.selection_target_unknown_variable";
    /// Detail `typed_mechanism_transport.shifted_without_target_evidence`.
    pub const SHIFTED_WITHOUT_TARGET_EVIDENCE: &str = "typed_mechanism_transport.shifted_without_target_evidence";
    /// Detail `typed_mechanism_transport.target_specification_differs`.
    pub const TARGET_SPECIFICATION_DIFFERS: &str = "typed_mechanism_transport.target_specification_differs";
    /// Detail `typed_mechanism_transport.transport_identity_mismatch`.
    pub const TRANSPORT_IDENTITY_MISMATCH: &str = "typed_mechanism_transport.transport_identity_mismatch";
    /// Detail `typed_mechanism_transport.unknown_variable`.
    pub const UNKNOWN_VARIABLE: &str = "typed_mechanism_transport.unknown_variable";
}

/// Details in the `vector_posterior` family.
pub mod vector_posterior {
    /// Detail `vector_posterior.band_not_constructed`.
    pub const BAND_NOT_CONSTRUCTED: &str = "vector_posterior.band_not_constructed";
    /// Detail `vector_posterior.external_power_prior`.
    pub const EXTERNAL_POWER_PRIOR: &str = "vector_posterior.external_power_prior";
    /// Detail `vector_posterior.mapped_prior`.
    pub const MAPPED_PRIOR: &str = "vector_posterior.mapped_prior";
    /// Detail `vector_posterior.non_positive_scale`.
    pub const NON_POSITIVE_SCALE: &str = "vector_posterior.non_positive_scale";
}

/// Details in the `vector_treatment` family.
pub mod vector_treatment {
    /// Detail `vector_treatment.adjustment_set_mismatch`.
    pub const ADJUSTMENT_SET_MISMATCH: &str = "vector_treatment.adjustment_set_mismatch";
    /// Detail `vector_treatment.cancelled`.
    pub const CANCELLED: &str = "vector_treatment.cancelled";
    /// Detail `vector_treatment.cluster_ids_required`.
    pub const CLUSTER_IDS_REQUIRED: &str = "vector_treatment.cluster_ids_required";
    /// Detail `vector_treatment.cluster_unread`.
    pub const CLUSTER_UNREAD: &str = "vector_treatment.cluster_unread";
    /// Detail `vector_treatment.collinear_treatments`.
    pub const COLLINEAR_TREATMENTS: &str = "vector_treatment.collinear_treatments";
    /// Detail `vector_treatment.data_not_bound`.
    pub const DATA_NOT_BOUND: &str = "vector_treatment.data_not_bound";
    /// Detail `vector_treatment.degenerate_covariance`.
    pub const DEGENERATE_COVARIANCE: &str = "vector_treatment.degenerate_covariance";
    /// Detail `vector_treatment.duplicate_contrast_coefficient`.
    pub const DUPLICATE_CONTRAST_COEFFICIENT: &str = "vector_treatment.duplicate_contrast_coefficient";
    /// Detail `vector_treatment.duplicate_name`.
    pub const DUPLICATE_NAME: &str = "vector_treatment.duplicate_name";
    /// Detail `vector_treatment.empty_contrast`.
    pub const EMPTY_CONTRAST: &str = "vector_treatment.empty_contrast";
    /// Detail `vector_treatment.export_budget_exceeded`.
    pub const EXPORT_BUDGET_EXCEEDED: &str = "vector_treatment.export_budget_exceeded";
    /// Detail `vector_treatment.invalid_identity`.
    pub const INVALID_IDENTITY: &str = "vector_treatment.invalid_identity";
    /// Detail `vector_treatment.invalid_lag`.
    pub const INVALID_LAG: &str = "vector_treatment.invalid_lag";
    /// Detail `vector_treatment.invalid_request`.
    pub const INVALID_REQUEST: &str = "vector_treatment.invalid_request";
    /// Detail `vector_treatment.lag_required`.
    pub const LAG_REQUIRED: &str = "vector_treatment.lag_required";
    /// Detail `vector_treatment.lag_unread`.
    pub const LAG_UNREAD: &str = "vector_treatment.lag_unread";
    /// Detail `vector_treatment.leverage_one`.
    pub const LEVERAGE_ONE: &str = "vector_treatment.leverage_one";
    /// Detail `vector_treatment.limits_exceeded`.
    pub const LIMITS_EXCEEDED: &str = "vector_treatment.limits_exceeded";
    /// Detail `vector_treatment.memory_budget_exceeded`.
    pub const MEMORY_BUDGET_EXCEEDED: &str = "vector_treatment.memory_budget_exceeded";
    /// Detail `vector_treatment.non_finite_value`.
    pub const NON_FINITE_VALUE: &str = "vector_treatment.non_finite_value";
    /// Detail `vector_treatment.rank_deficient_adjustment`.
    pub const RANK_DEFICIENT_ADJUSTMENT: &str = "vector_treatment.rank_deficient_adjustment";
    /// Detail `vector_treatment.row_count_mismatch`.
    pub const ROW_COUNT_MISMATCH: &str = "vector_treatment.row_count_mismatch";
    /// Detail `vector_treatment.row_snapshot_mismatch`.
    pub const ROW_SNAPSHOT_MISMATCH: &str = "vector_treatment.row_snapshot_mismatch";
    /// Detail `vector_treatment.summary_inconsistent`.
    pub const SUMMARY_INCONSISTENT: &str = "vector_treatment.summary_inconsistent";
    /// Detail `vector_treatment.summary_model_based_only`.
    pub const SUMMARY_MODEL_BASED_ONLY: &str = "vector_treatment.summary_model_based_only";
    /// Detail `vector_treatment.too_few_clusters`.
    pub const TOO_FEW_CLUSTERS: &str = "vector_treatment.too_few_clusters";
    /// Detail `vector_treatment.too_few_rows`.
    pub const TOO_FEW_ROWS: &str = "vector_treatment.too_few_rows";
    /// Detail `vector_treatment.too_few_treatments`.
    pub const TOO_FEW_TREATMENTS: &str = "vector_treatment.too_few_treatments";
    /// Detail `vector_treatment.too_many_treatments`.
    pub const TOO_MANY_TREATMENTS: &str = "vector_treatment.too_many_treatments";
    /// Detail `vector_treatment.treatment_without_variation`.
    pub const TREATMENT_WITHOUT_VARIATION: &str = "vector_treatment.treatment_without_variation";
    /// Detail `vector_treatment.unknown_contrast_coefficient`.
    pub const UNKNOWN_CONTRAST_COEFFICIENT: &str = "vector_treatment.unknown_contrast_coefficient";
    /// Detail `vector_treatment.weights_need_robust_covariance`.
    pub const WEIGHTS_NEED_ROBUST_COVARIANCE: &str = "vector_treatment.weights_need_robust_covariance";
    /// Detail `vector_treatment.wrong_contract`.
    pub const WRONG_CONTRACT: &str = "vector_treatment.wrong_contract";
}
