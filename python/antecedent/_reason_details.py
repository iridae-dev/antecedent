"""Generated from `parity/reason_details.toml`. Do not edit.

Every refusal detail id is an attribute here: `<family>.<NAME>`.
"""

from __future__ import annotations

from typing import Final


class adaptive_inference:  # noqa: N801
    ANALYSIS_INVALID: Final = "adaptive_inference.analysis_invalid"
    ANALYSIS_ROUTE_MISMATCH: Final = "adaptive_inference.analysis_route_mismatch"
    ANYTIME_PREMISE_CONTEXTS: Final = "adaptive_inference.anytime_premise_contexts"
    ANYTIME_SETTINGS_INVALID: Final = "adaptive_inference.anytime_settings_invalid"
    ANYTIME_SETTINGS_MISSING: Final = "adaptive_inference.anytime_settings_missing"
    ARM_NOT_OBSERVED: Final = "adaptive_inference.arm_not_observed"
    ARM_OUT_OF_RANGE: Final = "adaptive_inference.arm_out_of_range"
    ARMS_INVALID: Final = "adaptive_inference.arms_invalid"
    ARTIFACT_SIZE_EXCEEDED: Final = "adaptive_inference.artifact_size_exceeded"
    ASSIGNED_PROBABILITY_ZERO: Final = "adaptive_inference.assigned_probability_zero"
    ASSIGNMENT_NOT_REPLAYED: Final = "adaptive_inference.assignment_not_replayed"
    ASSIGNMENT_PROBABILITIES_MISSING: Final = "adaptive_inference.assignment_probabilities_missing"
    BATCH_INCOMPLETE: Final = "adaptive_inference.batch_incomplete"
    BETWEEN_UNIT_VARIANCE_MISSING: Final = "adaptive_inference.between_unit_variance_missing"
    BORROWING_FIXED_AFTER_FIRST_LOOK: Final = "adaptive_inference.borrowing_fixed_after_first_look"
    CANCELLED: Final = "adaptive_inference.cancelled"
    CELL_INCOHERENT: Final = "adaptive_inference.cell_incoherent"
    CELL_NOT_DELIVERED: Final = "adaptive_inference.cell_not_delivered"
    CHOICE_OUTCOME_NEEDS_CHOICE_CONTRACT: Final = (
        "adaptive_inference.choice_outcome_needs_choice_contract"
    )
    CONTEXTS_INVALID: Final = "adaptive_inference.contexts_invalid"
    CORRELATION_INVALID: Final = "adaptive_inference.correlation_invalid"
    DECLARED_LOOKS_EXCEEDED: Final = "adaptive_inference.declared_looks_exceeded"
    DECLARED_UNITS_EXCEEDED: Final = "adaptive_inference.declared_units_exceeded"
    DELAY_NOT_IGNORABLE: Final = "adaptive_inference.delay_not_ignorable"
    EXACT_LAW_STATES_EXCEEDED: Final = "adaptive_inference.exact_law_states_exceeded"
    EXACT_LAW_UNAVAILABLE: Final = "adaptive_inference.exact_law_unavailable"
    EXACT_LAW_WORK_EXCEEDED: Final = "adaptive_inference.exact_law_work_exceeded"
    EXPOSURE_DIFFERS_FROM_ASSIGNMENT: Final = "adaptive_inference.exposure_differs_from_assignment"
    EXPOSURE_HISTORY_MISSING: Final = "adaptive_inference.exposure_history_missing"
    EXPOSURE_INVALID: Final = "adaptive_inference.exposure_invalid"
    FINAL_ANALYSIS_MISSING: Final = "adaptive_inference.final_analysis_missing"
    FIXED_SAMPLE_DATA_DEPENDENT_STOPPING: Final = (
        "adaptive_inference.fixed_sample_data_dependent_stopping"
    )
    FIXED_SAMPLE_DESIGN_ADAPTIVE: Final = "adaptive_inference.fixed_sample_design_adaptive"
    FIXED_SAMPLE_ON_ADAPTIVE_DATA: Final = "adaptive_inference.fixed_sample_on_adaptive_data"
    GROUP_SEQUENTIAL_ASSIGNMENT_ADAPTIVE: Final = (
        "adaptive_inference.group_sequential_assignment_adaptive"
    )
    GUARANTEE_NOT_TRANSFERABLE: Final = "adaptive_inference.guarantee_not_transferable"
    HISTORICAL_BORROWING_MISSING: Final = "adaptive_inference.historical_borrowing_missing"
    HISTORICAL_PRIOR_INVALID: Final = "adaptive_inference.historical_prior_invalid"
    INDEPENDENCE_ASSUMED_FOR_DEPENDENT_MEASUREMENTS: Final = (
        "adaptive_inference.independence_assumed_for_dependent_measurements"
    )
    INPUT_INVALID: Final = "adaptive_inference.input_invalid"
    INTERIM_ANALYSES_MISSING: Final = "adaptive_inference.interim_analyses_missing"
    INTERVAL_NAMES_NO_QUANTITY: Final = "adaptive_inference.interval_names_no_quantity"
    INVERSION_PREMISE_CONTEXTS: Final = "adaptive_inference.inversion_premise_contexts"
    LEDGER_LIMIT: Final = "adaptive_inference.ledger_limit"
    LOOK_OUT_OF_RANGE: Final = "adaptive_inference.look_out_of_range"
    LOOKS_NOT_ORDERED: Final = "adaptive_inference.looks_not_ordered"
    LOOKS_OVERLAP_UNKNOWN: Final = "adaptive_inference.looks_overlap_unknown"
    LOOKS_SHARE_OBSERVATIONS: Final = "adaptive_inference.looks_share_observations"
    MEASUREMENTS_INVALID: Final = "adaptive_inference.measurements_invalid"
    NO_STAGEWISE_ADJUSTMENT_PROCEDURE: Final = (
        "adaptive_inference.no_stagewise_adjustment_procedure"
    )
    NO_STOPPING_RELIABILITY_PROCEDURE: Final = (
        "adaptive_inference.no_stopping_reliability_procedure"
    )
    NUMERICAL_FAILURE: Final = "adaptive_inference.numerical_failure"
    OUTCOME_AVAILABILITY_INVALID: Final = "adaptive_inference.outcome_availability_invalid"
    OUTCOME_AVAILABILITY_MISSING: Final = "adaptive_inference.outcome_availability_missing"
    OUTCOME_DEPENDENT_DELAY: Final = "adaptive_inference.outcome_dependent_delay"
    OUTCOME_INVALID: Final = "adaptive_inference.outcome_invalid"
    OUTCOME_STRUCTURE_MISSING: Final = "adaptive_inference.outcome_structure_missing"
    PAIRING_NOT_SUPPORTED: Final = "adaptive_inference.pairing_not_supported"
    PARADIGM_NOT_SUPPORTED_FOR_CELL: Final = "adaptive_inference.paradigm_not_supported_for_cell"
    PARADIGM_RULE_MISMATCH: Final = "adaptive_inference.paradigm_rule_mismatch"
    POLICY_HISTORY_INVALID: Final = "adaptive_inference.policy_history_invalid"
    POLICY_HISTORY_MISSING: Final = "adaptive_inference.policy_history_missing"
    POLICY_SELECTION_MISSING: Final = "adaptive_inference.policy_selection_missing"
    POSITIVITY_FLOOR: Final = "adaptive_inference.positivity_floor"
    POSTERIOR_PREMISE_OUTCOME_DEPENDENT_DELAY: Final = (
        "adaptive_inference.posterior_premise_outcome_dependent_delay"
    )
    PROBABILITIES_INVALID: Final = "adaptive_inference.probabilities_invalid"
    PROBABILITY_FLOOR_INVALID: Final = "adaptive_inference.probability_floor_invalid"
    PROCEDURE_LIMIT_EXCEEDED: Final = "adaptive_inference.procedure_limit_exceeded"
    REPEATED_MEASURES_PARADIGM_NOT_DERIVED: Final = (
        "adaptive_inference.repeated_measures_paradigm_not_derived"
    )
    REPEATED_MEASURES_STOPPING_NOT_DERIVED: Final = (
        "adaptive_inference.repeated_measures_stopping_not_derived"
    )
    REPORT_REPLAY_MISMATCH: Final = "adaptive_inference.report_replay_mismatch"
    RETAINED_RECORD_EXCEEDED: Final = "adaptive_inference.retained_record_exceeded"
    RULE_MISMATCH: Final = "adaptive_inference.rule_mismatch"
    SELECTION_NOT_IGNORABLE: Final = "adaptive_inference.selection_not_ignorable"
    SHAPE_MISMATCH: Final = "adaptive_inference.shape_mismatch"
    SIMULATION_MEMORY_EXCEEDED: Final = "adaptive_inference.simulation_memory_exceeded"
    SIMULATION_REPLICATIONS_EXCEEDED: Final = "adaptive_inference.simulation_replications_exceeded"
    SIMULATION_WORK_EXCEEDED: Final = "adaptive_inference.simulation_work_exceeded"
    STOPPING_NOT_REPLAYED: Final = "adaptive_inference.stopping_not_replayed"
    STOPPING_RULE_MISSING: Final = "adaptive_inference.stopping_rule_missing"
    STRUCTURE_INCOMPLETE: Final = "adaptive_inference.structure_incomplete"
    UNIT_OUT_OF_RANGE: Final = "adaptive_inference.unit_out_of_range"
    VARIANCE_INVALID: Final = "adaptive_inference.variance_invalid"
    WITHIN_UNIT_CORRELATION_MISSING: Final = "adaptive_inference.within_unit_correlation_missing"
    WITHIN_UNIT_VARIANCE_MISSING: Final = "adaptive_inference.within_unit_variance_missing"
    WRONG_CONTRACT: Final = "adaptive_inference.wrong_contract"


class adaptive_transport:  # noqa: N801
    ARMS: Final = "adaptive_transport.arms"
    ASSIGNMENT_BELOW_FLOOR: Final = "adaptive_transport.assignment_below_floor"
    ASSIGNMENT_MONITORING_CONTRACT_MISSING: Final = (
        "adaptive_transport.assignment_monitoring_contract_missing"
    )
    DESIGN_NOT_ADAPTIVE: Final = "adaptive_transport.design_not_adaptive"
    DIAGRAM_BINDING_MISMATCH: Final = "adaptive_transport.diagram_binding_mismatch"
    MECHANISM_DECLARED_TWICE: Final = "adaptive_transport.mechanism_declared_twice"
    MECHANISM_STATUS_MISSING: Final = "adaptive_transport.mechanism_status_missing"
    NO_ANALYSED_UNITS: Final = "adaptive_transport.no_analysed_units"
    OUTCOME_MECHANISM_NOT_INVARIANT: Final = "adaptive_transport.outcome_mechanism_not_invariant"
    OUTCOME_MISSING: Final = "adaptive_transport.outcome_missing"
    POPULATION_LABEL: Final = "adaptive_transport.population_label"
    SELECTION_DIAGRAM_MISSING: Final = "adaptive_transport.selection_diagram_missing"
    STRATA: Final = "adaptive_transport.strata"
    STRATUM_ARM_UNOBSERVED: Final = "adaptive_transport.stratum_arm_unobserved"
    STRATUM_LAW_DECLARED_INVARIANT: Final = "adaptive_transport.stratum_law_declared_invariant"
    TARGET_IS_SOURCE: Final = "adaptive_transport.target_is_source"
    TARGET_STRATUM_LAW: Final = "adaptive_transport.target_stratum_law"
    TARGET_STRATUM_LAW_MISSING: Final = "adaptive_transport.target_stratum_law_missing"
    UNIT_STRATA: Final = "adaptive_transport.unit_strata"
    VALUE_MISMATCH: Final = "adaptive_transport.value_mismatch"


class admg_conditional_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "admg_conditional_posterior.band_not_constructed"
    EMPTY_LAW: Final = "admg_conditional_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "admg_conditional_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "admg_conditional_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "admg_conditional_posterior.non_positive_scale"


class admg_transport:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "admg_transport.bounds_exceeded"
    BUDGET: Final = "admg_transport.budget"
    CONSUMER_LIMITS: Final = "admg_transport.consumer_limits"
    COUNTED_LAWS_UNSUPPORTED: Final = "admg_transport.counted_laws_unsupported"
    DATA_IDENTITY_MISMATCH: Final = "admg_transport.data_identity_mismatch"
    EVIDENCE_MISMATCH: Final = "admg_transport.evidence_mismatch"
    INVALID_ARTIFACT: Final = "admg_transport.invalid_artifact"
    INVALID_CATALOG: Final = "admg_transport.invalid_catalog"
    INVALID_DERIVATION: Final = "admg_transport.invalid_derivation"
    INVALID_QUERY: Final = "admg_transport.invalid_query"
    INVALID_REQUEST: Final = "admg_transport.invalid_request"
    MISSING_EVIDENCE: Final = "admg_transport.missing_evidence"
    NOT_CERTIFIED: Final = "admg_transport.not_certified"
    PREMISES_MISMATCH: Final = "admg_transport.premises_mismatch"
    PROVEN_NON_TRANSPORTABLE: Final = "admg_transport.proven_non_transportable"
    REPLAY_MISMATCH: Final = "admg_transport.replay_mismatch"
    SUPPORT_FAILURE: Final = "admg_transport.support_failure"


class aligned_joint_draws:  # noqa: N801
    IDENTITY_EXPECTED: Final = "aligned_joint_draws.identity_expected"
    LEGACY_BINDING: Final = "aligned_joint_draws.legacy_binding"
    LEGACY_SOURCE: Final = "aligned_joint_draws.legacy_source"
    MARGINALS_NOT_JOINT: Final = "aligned_joint_draws.marginals_not_joint"
    OUTCOME_COORDINATE: Final = "aligned_joint_draws.outcome_coordinate"


class batch_retarget:  # noqa: N801
    CANCELLED: Final = "batch_retarget.cancelled"
    CONTRAST_MEMBER_FAILED: Final = "batch_retarget.contrast_member_failed"
    COVARIANCE_UNAVAILABLE: Final = "batch_retarget.covariance_unavailable"
    DUPLICATE_NAME: Final = "batch_retarget.duplicate_name"
    EMPTY_FAMILY: Final = "batch_retarget.empty_family"
    FUNCTIONAL_NOT_SUPPORTED: Final = "batch_retarget.functional_not_supported"
    INCOMPATIBLE_TARGET: Final = "batch_retarget.incompatible_target"
    INVALID_CONTRAST: Final = "batch_retarget.invalid_contrast"
    MAX_T_DRAWS_OUT_OF_RANGE: Final = "batch_retarget.max_t_draws_out_of_range"
    MAX_T_INVALID_LEVEL: Final = "batch_retarget.max_t_invalid_level"
    MIXED_SNAPSHOT: Final = "batch_retarget.mixed_snapshot"
    PARTIAL_FAMILY: Final = "batch_retarget.partial_family"
    POINT_ONLY_MEMBER: Final = "batch_retarget.point_only_member"
    SCORES_MISMATCH: Final = "batch_retarget.scores_mismatch"
    SCORES_UNAVAILABLE: Final = "batch_retarget.scores_unavailable"
    SCORES_UNAVAILABLE_AFTER_ESTIMATE: Final = "batch_retarget.scores_unavailable_after_estimate"
    UNKNOWN_CLAIM: Final = "batch_retarget.unknown_claim"
    WEIGHTED_OVERLAP_FAILED: Final = "batch_retarget.weighted_overlap_failed"


class batch_retarget_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "batch_retarget_posterior.band_not_constructed"
    EMPTY_SCORES: Final = "batch_retarget_posterior.empty_scores"
    EXTERNAL_POWER_PRIOR: Final = "batch_retarget_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "batch_retarget_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "batch_retarget_posterior.non_positive_scale"


class bayesian_transport:  # noqa: N801
    ADMISSION_REFUSED: Final = "bayesian_transport.admission_refused"
    CANCELLED: Final = "bayesian_transport.cancelled"
    FEATURE_MISMATCH: Final = "bayesian_transport.feature_mismatch"
    IDENTIFICATION_NOT_CERTIFIED: Final = "bayesian_transport.identification_not_certified"
    IDENTIFICATION_REQUIRED: Final = "bayesian_transport.identification_required"
    INVALID_INPUT: Final = "bayesian_transport.invalid_input"
    INVALID_PRIOR: Final = "bayesian_transport.invalid_prior"
    MISSING_LAW: Final = "bayesian_transport.missing_law"
    MISSING_SOURCE: Final = "bayesian_transport.missing_source"
    PRIOR_DATA_OVERLAP: Final = "bayesian_transport.prior_data_overlap"
    PRIOR_DIMENSION: Final = "bayesian_transport.prior_dimension"
    SINGULAR_FIT: Final = "bayesian_transport.singular_fit"
    SOURCE_DEPENDENCE: Final = "bayesian_transport.source_dependence"
    SUPPORT_RESOURCE_LIMIT: Final = "bayesian_transport.support_resource_limit"
    TOO_MANY_DRAWS: Final = "bayesian_transport.too_many_draws"
    TOO_MANY_PARAMETERS: Final = "bayesian_transport.too_many_parameters"
    UNSUPPORTED_GRAPH: Final = "bayesian_transport.unsupported_graph"
    WEAK_OVERLAP: Final = "bayesian_transport.weak_overlap"


class binary_recovery_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "binary_recovery_posterior.band_not_constructed"
    EMPTY_LAW: Final = "binary_recovery_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "binary_recovery_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "binary_recovery_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "binary_recovery_posterior.non_positive_scale"


class callback_protocol:  # noqa: N801
    ARTIFACT_INVALID: Final = "callback_protocol.artifact_invalid"
    BLANK_FIELD: Final = "callback_protocol.blank_field"
    CANCELLED: Final = "callback_protocol.cancelled"
    DIAGNOSTIC_ONLY: Final = "callback_protocol.diagnostic_only"
    DUPLICATE_NAME: Final = "callback_protocol.duplicate_name"
    HIDDEN_STATE_UNOBSERVABLE: Final = "callback_protocol.hidden_state_unobservable"
    IDENTIFICATION_NOT_CONFERRED: Final = "callback_protocol.identification_not_conferred"
    IDENTITY_MISMATCH: Final = "callback_protocol.identity_mismatch"
    INTERVAL_NOT_SUPPORTED: Final = "callback_protocol.interval_not_supported"
    INVALID_QUANTITY: Final = "callback_protocol.invalid_quantity"
    INVALID_SUPPORT: Final = "callback_protocol.invalid_support"
    INVOCATION_LIMIT: Final = "callback_protocol.invocation_limit"
    INVOCATION_OUTSIDE_EXECUTION: Final = "callback_protocol.invocation_outside_execution"
    JOINT_LAW_NOT_CONFERRED: Final = "callback_protocol.joint_law_not_conferred"
    MEANING_NOT_FOR_KIND: Final = "callback_protocol.meaning_not_for_kind"
    NATIVE_TRUST_CLAIMED: Final = "callback_protocol.native_trust_claimed"
    NATIVE_TRUST_NOT_CONFERRED: Final = "callback_protocol.native_trust_not_conferred"
    NESTED_INPUT_OVERLAP_UNKNOWN: Final = "callback_protocol.nested_input_overlap_unknown"
    NESTED_INPUT_UNREGISTERED: Final = "callback_protocol.nested_input_unregistered"
    NESTED_INPUTS_MISMATCH: Final = "callback_protocol.nested_inputs_mismatch"
    NESTED_INPUTS_SHARE_EVIDENCE: Final = "callback_protocol.nested_inputs_share_evidence"
    NO_IDEMPOTENCY_CONTRACT: Final = "callback_protocol.no_idempotency_contract"
    NO_OPERATIONS: Final = "callback_protocol.no_operations"
    OPAQUE_DEPENDENCIES: Final = "callback_protocol.opaque_dependencies"
    OPERATION_NOT_DECLARED: Final = "callback_protocol.operation_not_declared"
    OPERATION_NOT_FOR_KIND: Final = "callback_protocol.operation_not_for_kind"
    OUTPUT_NOT_A_LAW: Final = "callback_protocol.output_not_a_law"
    OUTPUT_NOT_FINITE: Final = "callback_protocol.output_not_finite"
    OUTPUT_OUTSIDE_SUPPORT: Final = "callback_protocol.output_outside_support"
    OUTPUT_SHAPE: Final = "callback_protocol.output_shape"
    OUTPUT_TOO_LARGE: Final = "callback_protocol.output_too_large"
    POSTERIOR_GUARANTEE_NOT_SUPPORTED: Final = "callback_protocol.posterior_guarantee_not_supported"
    PROVIDER_FAILED: Final = "callback_protocol.provider_failed"
    PROVIDER_IDENTITY_MISMATCH: Final = "callback_protocol.provider_identity_mismatch"
    PROVIDER_UNAVAILABLE: Final = "callback_protocol.provider_unavailable"
    QUERY_INVALID: Final = "callback_protocol.query_invalid"
    REPLAY_CONTRACT_INCOMPLETE: Final = "callback_protocol.replay_contract_incomplete"
    REQUEST_MISMATCH: Final = "callback_protocol.request_mismatch"
    RETRY_OUTSIDE_POLICY: Final = "callback_protocol.retry_outside_policy"
    SIDE_EFFECTS_NOT_ALLOWED: Final = "callback_protocol.side_effects_not_allowed"
    TOO_MANY_ITEMS: Final = "callback_protocol.too_many_items"
    TRUST_INCOMPLETE: Final = "callback_protocol.trust_incomplete"
    UPDATE_DECLARATION_MISMATCH: Final = "callback_protocol.update_declaration_mismatch"
    WALL_LIMIT_EXCEEDED: Final = "callback_protocol.wall_limit_exceeded"


class categorical_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "categorical_posterior.band_not_constructed"
    EXTERNAL_POWER_PRIOR: Final = "categorical_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "categorical_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "categorical_posterior.non_positive_scale"


class categorical_treatment:  # noqa: N801
    ABSENT_LEVEL: Final = "categorical_treatment.absent_level"
    CANCELLED: Final = "categorical_treatment.cancelled"
    DATA_NOT_BOUND: Final = "categorical_treatment.data_not_bound"
    EXPORT_BUDGET_EXCEEDED: Final = "categorical_treatment.export_budget_exceeded"
    INVALID_IDENTITY: Final = "categorical_treatment.invalid_identity"
    INVALID_LEVEL: Final = "categorical_treatment.invalid_level"
    INVALID_MIN_LEVEL_ROWS: Final = "categorical_treatment.invalid_min_level_rows"
    INVALID_PAIR: Final = "categorical_treatment.invalid_pair"
    INVALID_REQUEST: Final = "categorical_treatment.invalid_request"
    LIMITS_EXCEEDED: Final = "categorical_treatment.limits_exceeded"
    MEMORY_BUDGET_EXCEEDED: Final = "categorical_treatment.memory_budget_exceeded"
    MONOTONICITY_REQUIRES_ORDERED: Final = "categorical_treatment.monotonicity_requires_ordered"
    POSTERIOR_DECLARED_ROWS: Final = "categorical_treatment.posterior_declared_rows"
    ROW_COUNT_MISMATCH: Final = "categorical_treatment.row_count_mismatch"
    SPARSE_LEVEL: Final = "categorical_treatment.sparse_level"
    SUMMARY_INCONSISTENT: Final = "categorical_treatment.summary_inconsistent"
    TOO_FEW_LEVELS: Final = "categorical_treatment.too_few_levels"
    UNDECLARED_LEVEL: Final = "categorical_treatment.undeclared_level"
    UNKNOWN_LEVEL: Final = "categorical_treatment.unknown_level"
    UNKNOWN_REFERENCE: Final = "categorical_treatment.unknown_reference"
    WRONG_CONTRACT: Final = "categorical_treatment.wrong_contract"


class chain_recovery_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "chain_recovery_posterior.band_not_constructed"
    EMPTY_LAW: Final = "chain_recovery_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "chain_recovery_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "chain_recovery_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "chain_recovery_posterior.non_positive_scale"


class cluster_dml:  # noqa: N801
    CANCELLED: Final = "cluster_dml.cancelled"
    CLUSTER_IDS_MISSING: Final = "cluster_dml.cluster_ids_missing"
    DYADIC_GIANT_COMPONENT: Final = "cluster_dml.dyadic_giant_component"
    DYADIC_SHARED_NAMESPACE: Final = "cluster_dml.dyadic_shared_namespace"
    FLEXIBLE_LEARNER_NOT_SUPPORTED: Final = "cluster_dml.flexible_learner_not_supported"
    INVALID_MIN_CLUSTERS: Final = "cluster_dml.invalid_min_clusters"
    INVALID_MIN_COMPONENTS: Final = "cluster_dml.invalid_min_components"
    NO_INTERVAL_METHOD: Final = "cluster_dml.no_interval_method"
    PENALIZED_PROPENSITY: Final = "cluster_dml.penalized_propensity"
    SCOPE: Final = "cluster_dml.scope"
    SECOND_IDS_MISSING: Final = "cluster_dml.second_ids_missing"
    SECOND_IDS_UNEXPECTED: Final = "cluster_dml.second_ids_unexpected"
    TOO_FEW_CLUSTERS: Final = "cluster_dml.too_few_clusters"
    TOO_FEW_COMPONENTS: Final = "cluster_dml.too_few_components"


class clustered_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "clustered_posterior.band_not_constructed"
    EMPTY_SCORES: Final = "clustered_posterior.empty_scores"
    EXTERNAL_POWER_PRIOR: Final = "clustered_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "clustered_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "clustered_posterior.non_positive_scale"


class cohort_decision:  # noqa: N801
    CANCELLED: Final = "cohort_decision.cancelled"
    COHORT_CHANGED: Final = "cohort_decision.cohort_changed"
    COHORT_EMPTY: Final = "cohort_decision.cohort_empty"
    COHORT_INVALID: Final = "cohort_decision.cohort_invalid"
    COHORT_TOO_LARGE: Final = "cohort_decision.cohort_too_large"
    CONTAINER_CHECKSUM_MISMATCH: Final = "cohort_decision.container_checksum_mismatch"
    CONTAINER_INVALID: Final = "cohort_decision.container_invalid"
    CONTRAST_MISMATCH: Final = "cohort_decision.contrast_mismatch"
    CONTRAST_NOT_DECLARED: Final = "cohort_decision.contrast_not_declared"
    EXTRAPOLATIVE_NOT_ACCEPTED: Final = "cohort_decision.extrapolative_not_accepted"
    FEATURE_SCHEMA_MISMATCH: Final = "cohort_decision.feature_schema_mismatch"
    FEATURE_TYPE_MISMATCH: Final = "cohort_decision.feature_type_mismatch"
    FITTED_MODEL_ABSENT: Final = "cohort_decision.fitted_model_absent"
    FITTED_MODEL_INVALID: Final = "cohort_decision.fitted_model_invalid"
    IDENTITY_UNEXPECTED: Final = "cohort_decision.identity_unexpected"
    INTERVAL_LEVEL_INVALID: Final = "cohort_decision.interval_level_invalid"
    INTERVAL_NOT_SUPPORTED: Final = "cohort_decision.interval_not_supported"
    INTERVAL_REQUIRES_NATIVE_FIT: Final = "cohort_decision.interval_requires_native_fit"
    INVALID_REQUEST: Final = "cohort_decision.invalid_request"
    JOINT_LAW_REQUIRED: Final = "cohort_decision.joint_law_required"
    LAW_NOT_SUPPLIED: Final = "cohort_decision.law_not_supplied"
    LAYOUT_INVALID: Final = "cohort_decision.layout_invalid"
    LINK_NOT_IDENTITY: Final = "cohort_decision.link_not_identity"
    MISSING_ARTIFACT_ID: Final = "cohort_decision.missing_artifact_id"
    MODEL_IDENTITY_MISMATCH: Final = "cohort_decision.model_identity_mismatch"
    NATIVE_FIT_REFUSED: Final = "cohort_decision.native_fit_refused"
    NEGATIVE_VARIANCE: Final = "cohort_decision.negative_variance"
    NON_FINITE_RESULT: Final = "cohort_decision.non_finite_result"
    OUTCOME_UNITS_REQUIRED: Final = "cohort_decision.outcome_units_required"
    OUTSIDE_SUPPORT: Final = "cohort_decision.outside_support"
    OVERSIZED: Final = "cohort_decision.oversized"
    PARENT_ATTESTED_NOT_NATIVE: Final = "cohort_decision.parent_attested_not_native"
    PARENT_NOT_VERIFIED: Final = "cohort_decision.parent_not_verified"
    PARENT_QUERY_UNSUPPORTED: Final = "cohort_decision.parent_query_unsupported"
    POPULATION_MISMATCH: Final = "cohort_decision.population_mismatch"
    PREDICTION_REFUSED: Final = "cohort_decision.prediction_refused"
    REPLAY_MISMATCH: Final = "cohort_decision.replay_mismatch"
    SHIFT_DECLARATION_MISMATCH: Final = "cohort_decision.shift_declaration_mismatch"
    SHIFT_DECLARATION_REQUIRED: Final = "cohort_decision.shift_declaration_required"
    SHIFT_DECLARATION_UNNEEDED: Final = "cohort_decision.shift_declaration_unneeded"
    SHIFT_SUPPORT_FAILURE: Final = "cohort_decision.shift_support_failure"
    SOURCE_INVALID: Final = "cohort_decision.source_invalid"
    SUPPORT_CONTRACT_ABSENT: Final = "cohort_decision.support_contract_absent"
    UNITS_MISMATCH: Final = "cohort_decision.units_mismatch"
    UNSUPPORTED_VERSION: Final = "cohort_decision.unsupported_version"
    WEIGHTS_INVALID: Final = "cohort_decision.weights_invalid"
    WEIGHTS_NOT_NORMALIZED: Final = "cohort_decision.weights_not_normalized"
    WRONG_PARENT_IDENTITY: Final = "cohort_decision.wrong_parent_identity"


class compact_export:  # noqa: N801
    CONTAINER_CHECKSUM_MISMATCH: Final = "compact_export.container_checksum_mismatch"
    CONTAINER_INVALID: Final = "compact_export.container_invalid"
    COVARIANCE_ASYMMETRIC: Final = "compact_export.covariance_asymmetric"
    COVARIANCE_NOT_PSD: Final = "compact_export.covariance_not_psd"
    DATA_DIGEST_MISMATCH: Final = "compact_export.data_digest_mismatch"
    DIMENSION_MISMATCH: Final = "compact_export.dimension_mismatch"
    DUPLICATE_TERM: Final = "compact_export.duplicate_term"
    EMPTY_MASK_REGION: Final = "compact_export.empty_mask_region"
    ENCODE: Final = "compact_export.encode"
    IDENTITY_MISMATCH: Final = "compact_export.identity_mismatch"
    IDENTITY_UNEXPECTED: Final = "compact_export.identity_unexpected"
    INPUTS_NOT_CANONICAL: Final = "compact_export.inputs_not_canonical"
    INVALID_DEGREE: Final = "compact_export.invalid_degree"
    INVALID_MASK_REGION: Final = "compact_export.invalid_mask_region"
    INVALID_QUANTITY: Final = "compact_export.invalid_quantity"
    INVALID_QUERY: Final = "compact_export.invalid_query"
    INVALID_REQUEST: Final = "compact_export.invalid_request"
    INVALID_SUPPORT: Final = "compact_export.invalid_support"
    LAYOUT_INVALID: Final = "compact_export.layout_invalid"
    MASK_NOT_CANONICAL: Final = "compact_export.mask_not_canonical"
    MASK_SUPPORT_MISMATCH: Final = "compact_export.mask_support_mismatch"
    MASKED_REGION: Final = "compact_export.masked_region"
    MISSING_ARTIFACT_ID: Final = "compact_export.missing_artifact_id"
    MISSING_QUANTITY: Final = "compact_export.missing_quantity"
    NEGATIVE_VARIANCE: Final = "compact_export.negative_variance"
    NO_TERMS: Final = "compact_export.no_terms"
    NON_FINITE_INPUT: Final = "compact_export.non_finite_input"
    NON_FINITE_RESULT: Final = "compact_export.non_finite_result"
    NON_FINITE_VALUE: Final = "compact_export.non_finite_value"
    OUT_OF_SUPPORT: Final = "compact_export.out_of_support"
    OVERSIZED: Final = "compact_export.oversized"
    PREMISES_DIGEST_MISMATCH: Final = "compact_export.premises_digest_mismatch"
    QUERY_TYPE_MISMATCH: Final = "compact_export.query_type_mismatch"
    SCOPE_MISMATCH: Final = "compact_export.scope_mismatch"
    TERM_SUPPORT_MISMATCH: Final = "compact_export.term_support_mismatch"
    TERMS_NOT_CANONICAL: Final = "compact_export.terms_not_canonical"
    TOO_MANY_INPUTS: Final = "compact_export.too_many_inputs"
    TOO_MANY_LEVELS: Final = "compact_export.too_many_levels"
    TOO_MANY_MASK_REGIONS: Final = "compact_export.too_many_mask_regions"
    TOO_MANY_TERMS: Final = "compact_export.too_many_terms"
    UNKNOWN_LEVEL: Final = "compact_export.unknown_level"
    UNKNOWN_QUANTITY: Final = "compact_export.unknown_quantity"
    UNSUPPORTED_VERSION: Final = "compact_export.unsupported_version"


class compiled_categorical_posterior:  # noqa: N801
    CANCELLED: Final = "compiled_categorical_posterior.cancelled"
    CANDIDATE: Final = "compiled_categorical_posterior.candidate"
    EXECUTION_REFUSED: Final = "compiled_categorical_posterior.execution_refused"
    RESOURCE_LIMIT: Final = "compiled_categorical_posterior.resource_limit"
    SOURCE_INVALID: Final = "compiled_categorical_posterior.source_invalid"


class composite_recalc:  # noqa: N801
    INVALID_REQUEST: Final = "composite_recalc.invalid_request"
    NATIVE_STATE_MISMATCH: Final = "composite_recalc.native_state_mismatch"
    RECEIPT_INVALID: Final = "composite_recalc.receipt_invalid"
    REQUEST_UNSUPPORTED: Final = "composite_recalc.request_unsupported"


class composition_boundary:  # noqa: N801
    ARTIFACT_REBUILD_FAILED: Final = "composition_boundary.artifact_rebuild_failed"
    ATOM_PROBABILITIES_INVALID: Final = "composition_boundary.atom_probabilities_invalid"
    CONFLICTING_ATOMS_NOT_AVERAGED: Final = "composition_boundary.conflicting_atoms_not_averaged"
    COORDINATE_MISSING: Final = "composition_boundary.coordinate_missing"
    COORDINATE_UNSUPPORTED: Final = "composition_boundary.coordinate_unsupported"
    DEPENDENCE_ROUTE_NOT_SUPPORTED: Final = "composition_boundary.dependence_route_not_supported"
    INVALID_INPUT: Final = "composition_boundary.invalid_input"
    MEAN_IS_NOT_A_DISTRIBUTION: Final = "composition_boundary.mean_is_not_a_distribution"
    METADATA_ONLY_NATIVE_CLAIM: Final = "composition_boundary.metadata_only_native_claim"
    NATIVE_REQUIRED: Final = "composition_boundary.native_required"
    NON_AFFINE_NEEDS_JOINT_LAW: Final = "composition_boundary.non_affine_needs_joint_law"
    OPERATION_NOT_DECLARED: Final = "composition_boundary.operation_not_declared"
    OUTCOME_LAW_NEEDS_JOINT_LAW: Final = "composition_boundary.outcome_law_needs_joint_law"
    PAIRED_DRAWS_ACROSS_SOURCES: Final = "composition_boundary.paired_draws_across_sources"
    SHARED_EVIDENCE_NOT_INDEPENDENT: Final = "composition_boundary.shared_evidence_not_independent"
    UNKNOWN_DEPENDENCE_IS_NOT_INDEPENDENCE: Final = (
        "composition_boundary.unknown_dependence_is_not_independence"
    )
    UNKNOWN_INPUT: Final = "composition_boundary.unknown_input"
    UNSUPPORTED_ACTION_NOT_COMPARABLE: Final = (
        "composition_boundary.unsupported_action_not_comparable"
    )
    VERIFICATION_RECEIPT_MISSING: Final = "composition_boundary.verification_receipt_missing"


class composition_bundle:  # noqa: N801
    CALLBACK_UNAVAILABLE: Final = "composition_bundle.callback_unavailable"
    EDGE_DIGEST_MISMATCH: Final = "composition_bundle.edge_digest_mismatch"
    EXPECTED_IDENTITY_MISMATCH: Final = "composition_bundle.expected_identity_mismatch"
    GRAPH_OR_SNAPSHOT_MISMATCH: Final = "composition_bundle.graph_or_snapshot_mismatch"
    INCOMPATIBLE_VERSION: Final = "composition_bundle.incompatible_version"
    INVALID_CONTAINER: Final = "composition_bundle.invalid_container"
    NODE_NOT_FOUND: Final = "composition_bundle.node_not_found"
    OVERSIZED: Final = "composition_bundle.oversized"
    PROVIDER_REQUEST_CHANGED: Final = "composition_bundle.provider_request_changed"
    SWAPPED_EVIDENCE: Final = "composition_bundle.swapped_evidence"
    TAMPERED_QUANTITY: Final = "composition_bundle.tampered_quantity"
    UNKNOWN_NODE_KIND: Final = "composition_bundle.unknown_node_kind"
    UNSUPPORTED_LAW: Final = "composition_bundle.unsupported_law"


class composition_provenance:  # noqa: N801
    DIGEST_MISMATCH: Final = "composition_provenance.digest_mismatch"
    DUPLICATE_LINK: Final = "composition_provenance.duplicate_link"
    INVALID_LINK: Final = "composition_provenance.invalid_link"
    MISSING_STAGE: Final = "composition_provenance.missing_stage"
    UNKNOWN_LINK: Final = "composition_provenance.unknown_link"
    UNRESOLVED_PARENT: Final = "composition_provenance.unresolved_parent"


class composition_verification:  # noqa: N801
    ARTIFACT_INVALID: Final = "composition_verification.artifact_invalid"
    CANCELLED: Final = "composition_verification.cancelled"
    CYCLE: Final = "composition_verification.cycle"
    DUPLICATE_NODE: Final = "composition_verification.duplicate_node"
    EMPTY_UNSUPPORTED: Final = "composition_verification.empty_unsupported"
    GRAPH_INVALID: Final = "composition_verification.graph_invalid"
    IDENTITY_MISMATCH: Final = "composition_verification.identity_mismatch"
    INHERITED_WITHOUT_EDGE: Final = "composition_verification.inherited_without_edge"
    INPUT_RETIRED: Final = "composition_verification.input_retired"
    INVALID_TEXT: Final = "composition_verification.invalid_text"
    LEDGER_MISMATCH: Final = "composition_verification.ledger_mismatch"
    LIMIT_EXCEEDED: Final = "composition_verification.limit_exceeded"
    PROVIDER_TRUST_WITHDRAWN: Final = "composition_verification.provider_trust_withdrawn"
    RECOMPUTE_REQUIRED: Final = "composition_verification.recompute_required"
    REPLAY_MISMATCH: Final = "composition_verification.replay_mismatch"
    SELF_DEPENDENCY: Final = "composition_verification.self_dependency"
    STAGE_NOT_IN_PLAN: Final = "composition_verification.stage_not_in_plan"
    STAGE_UNEXECUTABLE: Final = "composition_verification.stage_unexecutable"
    SUPPORT_WITHDRAWN: Final = "composition_verification.support_withdrawn"
    UNKNOWN_CLAIM: Final = "composition_verification.unknown_claim"
    UNKNOWN_NODE: Final = "composition_verification.unknown_node"
    UNKNOWN_RESULT: Final = "composition_verification.unknown_result"


class conditional_study_ranking:  # noqa: N801
    ACTION_NOT_UNIQUE: Final = "conditional_study_ranking.action_not_unique"
    ARTIFACT_INVALID: Final = "conditional_study_ranking.artifact_invalid"
    INVALID_POLICY: Final = "conditional_study_ranking.invalid_policy"
    ORIGINAL_RANKING_REFUSED: Final = "conditional_study_ranking.original_ranking_refused"
    RESOURCE_LIMIT: Final = "conditional_study_ranking.resource_limit"
    SOURCE_MISMATCH: Final = "conditional_study_ranking.source_mismatch"
    SOURCE_UNAVAILABLE: Final = "conditional_study_ranking.source_unavailable"


class constancy_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "constancy_posterior.band_not_constructed"
    EMPTY_CONTRAST: Final = "constancy_posterior.empty_contrast"
    EXTERNAL_POWER_PRIOR: Final = "constancy_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "constancy_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "constancy_posterior.non_positive_scale"


class coordinate_support:  # noqa: N801
    FUNCTIONAL_NOT_DESCRIBED: Final = "coordinate_support.functional_not_described"
    INVALID_GRID: Final = "coordinate_support.invalid_grid"
    MISSING_EVIDENCE_RELABELED: Final = "coordinate_support.missing_evidence_relabeled"
    POINT_COUNT_MISMATCH: Final = "coordinate_support.point_count_mismatch"
    REGIME_NOT_DESCRIBABLE: Final = "coordinate_support.regime_not_describable"
    SUMMARY_NOT_WORST_LABEL: Final = "coordinate_support.summary_not_worst_label"
    TEMPORAL_PATH_NOT_DESCRIBED: Final = "coordinate_support.temporal_path_not_described"
    UNITS_REQUIRED: Final = "coordinate_support.units_required"
    UNKNOWN_VARIABLE: Final = "coordinate_support.unknown_variable"


class counterfactual_id:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "counterfactual_id.bounds_exceeded"
    BUDGET: Final = "counterfactual_id.budget"
    CONFLICTING_SUBSCRIPTS: Final = "counterfactual_id.conflicting_subscripts"
    COUNTERFACTUAL_HEDGE: Final = "counterfactual_id.counterfactual_hedge"
    GRAPH_OUTSIDE_CONTRACT: Final = "counterfactual_id.graph_outside_contract"
    INTERVAL_REQUESTED: Final = "counterfactual_id.interval_requested"
    INVALID_ARTIFACT: Final = "counterfactual_id.invalid_artifact"
    INVALID_QUERY: Final = "counterfactual_id.invalid_query"
    PATH_SPECIFIC_NOT_SUPPORTED: Final = "counterfactual_id.path_specific_not_supported"
    POSITIVITY_VIOLATION: Final = "counterfactual_id.positivity_violation"
    QUERY_OUTSIDE_CONTRACT: Final = "counterfactual_id.query_outside_contract"


class counterfactual_transport:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "counterfactual_transport.bounds_exceeded"
    BUDGET: Final = "counterfactual_transport.budget"
    CHECKED_OBSTRUCTION: Final = "counterfactual_transport.checked_obstruction"
    COUPLING_UNSUPPORTED: Final = "counterfactual_transport.coupling_unsupported"
    EVIDENCE_INCOMPATIBLE: Final = "counterfactual_transport.evidence_incompatible"
    FACTOR_NOT_DERIVABLE: Final = "counterfactual_transport.factor_not_derivable"
    GRAPH_OUTSIDE_CONTRACT: Final = "counterfactual_transport.graph_outside_contract"
    INVALID_ARTIFACT: Final = "counterfactual_transport.invalid_artifact"
    INVALID_CATALOG: Final = "counterfactual_transport.invalid_catalog"
    INVALID_DERIVATION: Final = "counterfactual_transport.invalid_derivation"
    INVALID_EVIDENCE: Final = "counterfactual_transport.invalid_evidence"
    INVALID_QUERY: Final = "counterfactual_transport.invalid_query"
    INVALID_WITNESS: Final = "counterfactual_transport.invalid_witness"
    MISSING_LAW: Final = "counterfactual_transport.missing_law"
    MISSING_REGIME: Final = "counterfactual_transport.missing_regime"
    NO_SAMPLED_INFERENCE: Final = "counterfactual_transport.no_sampled_inference"
    SELECTED_SAMPLE_LAW: Final = "counterfactual_transport.selected_sample_law"
    SUPPORT_FAILURE: Final = "counterfactual_transport.support_failure"
    ZERO_CONDITIONING_MASS: Final = "counterfactual_transport.zero_conditioning_mass"


class counterfactual_transport_sampled:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "counterfactual_transport_sampled.bounds_exceeded"
    CANCELLED: Final = "counterfactual_transport_sampled.cancelled"
    ENVELOPE_BUDGET: Final = "counterfactual_transport_sampled.envelope_budget"
    ENVELOPE_UNDEFINED: Final = "counterfactual_transport_sampled.envelope_undefined"
    INTERVAL_NOT_COMPUTABLE: Final = "counterfactual_transport_sampled.interval_not_computable"
    INVALID_ARTIFACT: Final = "counterfactual_transport_sampled.invalid_artifact"
    INVALID_COUNTS: Final = "counterfactual_transport_sampled.invalid_counts"
    INVALID_OVERLAP: Final = "counterfactual_transport_sampled.invalid_overlap"
    MISSING_TABLE: Final = "counterfactual_transport_sampled.missing_table"
    OVERLAP_DECLARED_INDEPENDENT: Final = (
        "counterfactual_transport_sampled.overlap_declared_independent"
    )
    OVERLAP_UNKNOWN: Final = "counterfactual_transport_sampled.overlap_unknown"
    SAMPLE_SIZE_BELOW_LIMIT: Final = "counterfactual_transport_sampled.sample_size_below_limit"
    SHARED_UNITS_UNLINKED: Final = "counterfactual_transport_sampled.shared_units_unlinked"
    SPARSITY_ABOVE_LIMIT: Final = "counterfactual_transport_sampled.sparsity_above_limit"
    TABLE_NOT_CITED: Final = "counterfactual_transport_sampled.table_not_cited"
    UNSEEN_CELL: Final = "counterfactual_transport_sampled.unseen_cell"


class cpdag_scenarios:  # noqa: N801
    CANCELLED_NOT_EXPORTABLE: Final = "cpdag_scenarios.cancelled_not_exportable"
    CONSUMER_LIMIT_EXCEEDED: Final = "cpdag_scenarios.consumer_limit_exceeded"
    DATA_IDENTITY_MISMATCH: Final = "cpdag_scenarios.data_identity_mismatch"
    EVIDENCE_IDENTITY_MISMATCH: Final = "cpdag_scenarios.evidence_identity_mismatch"
    EXACT_LAWS_ONLY: Final = "cpdag_scenarios.exact_laws_only"
    NO_EVIDENCE_BINDING: Final = "cpdag_scenarios.no_evidence_binding"
    NOT_A_CPDAG: Final = "cpdag_scenarios.not_a_cpdag"
    PREMISES_MISMATCH: Final = "cpdag_scenarios.premises_mismatch"
    PROVIDER_MISMATCH: Final = "cpdag_scenarios.provider_mismatch"
    REPORT_REPLAY_MISMATCH: Final = "cpdag_scenarios.report_replay_mismatch"
    SELECTION_OR_LATENT: Final = "cpdag_scenarios.selection_or_latent"
    UNSUPPORTED_SEMANTICS: Final = "cpdag_scenarios.unsupported_semantics"


class cross_world:  # noqa: N801
    GRAPH_OUTSIDE_CONTRACT: Final = "cross_world.graph_outside_contract"
    INTERVAL_REQUESTED: Final = "cross_world.interval_requested"
    INVALID_QUERY: Final = "cross_world.invalid_query"
    QUERY_OUTSIDE_CONTRACT: Final = "cross_world.query_outside_contract"
    RECANTING_WITNESS: Final = "cross_world.recanting_witness"


class cross_world_execution:  # noqa: N801
    ARTIFACT_CHANGED: Final = "cross_world_execution.artifact_changed"
    BOUNDS_EXCEEDED: Final = "cross_world_execution.bounds_exceeded"
    CANCELLED: Final = "cross_world_execution.cancelled"
    CELL_NOT_IN_MATRIX: Final = "cross_world_execution.cell_not_in_matrix"
    ELIMINATION_BOUND_EXCEEDED: Final = "cross_world_execution.elimination_bound_exceeded"
    ENVELOPE_MEMBER_STRUCTURE_DIFFERS: Final = (
        "cross_world_execution.envelope_member_structure_differs"
    )
    ENVELOPE_NEEDS_PARAMETER_SET: Final = "cross_world_execution.envelope_needs_parameter_set"
    ENVELOPE_PARAMETER_SET_EMPTY: Final = "cross_world_execution.envelope_parameter_set_empty"
    ENVELOPE_READING_REQUIRED: Final = "cross_world_execution.envelope_reading_required"
    EVIDENCE_OUTSIDE_MODEL_SUPPORT: Final = "cross_world_execution.evidence_outside_model_support"
    EVIDENCE_PROBABILITY_UNDERFLOW: Final = "cross_world_execution.evidence_probability_underflow"
    FINITE_FITTING_NOT_EXECUTED: Final = "cross_world_execution.finite_fitting_not_executed"
    INCOMPATIBLE_HISTORY: Final = "cross_world_execution.incompatible_history"
    INVALID_ARTIFACT: Final = "cross_world_execution.invalid_artifact"
    INVALID_EVIDENCE: Final = "cross_world_execution.invalid_evidence"
    INVALID_IDENTITY: Final = "cross_world_execution.invalid_identity"
    INVALID_QUANTITY: Final = "cross_world_execution.invalid_quantity"
    INVALID_REQUEST: Final = "cross_world_execution.invalid_request"
    INVALID_SCM: Final = "cross_world_execution.invalid_scm"
    INVALID_WORLD: Final = "cross_world_execution.invalid_world"
    JOINT_STATE_BOUND_EXCEEDED: Final = "cross_world_execution.joint_state_bound_exceeded"
    LATENT_CONFOUNDING_NOT_IDENTIFIED: Final = (
        "cross_world_execution.latent_confounding_not_identified"
    )
    MECHANISMS_NOT_DECLARED: Final = "cross_world_execution.mechanisms_not_declared"
    NO_INTERVAL_METHOD: Final = "cross_world_execution.no_interval_method"
    NO_POSTERIOR_METHOD: Final = "cross_world_execution.no_posterior_method"
    NO_TRANSPORTED_ROW_DERIVATION: Final = "cross_world_execution.no_transported_row_derivation"
    POSTERIOR_WEIGHT_UNDERFLOW: Final = "cross_world_execution.posterior_weight_underflow"
    SAMPLED_ABDUCTION_NOT_SUPPORTED: Final = "cross_world_execution.sampled_abduction_not_supported"
    SOFT_INTERVENTION: Final = "cross_world_execution.soft_intervention"
    SPREAD_NEEDS_ENUMERATION: Final = "cross_world_execution.spread_needs_enumeration"


class cross_world_uncertainty:  # noqa: N801
    ACCOUNT_MISMATCH: Final = "cross_world_uncertainty.account_mismatch"
    BOUNDS_EXCEEDED: Final = "cross_world_uncertainty.bounds_exceeded"
    INCOMPATIBLE_HISTORY: Final = "cross_world_uncertainty.incompatible_history"
    NO_TRANSPORTED_PATH_DERIVATION: Final = "cross_world_uncertainty.no_transported_path_derivation"
    RESULT_MISMATCH: Final = "cross_world_uncertainty.result_mismatch"
    SAMPLED_PATH_NOT_SUPPORTED: Final = "cross_world_uncertainty.sampled_path_not_supported"
    UNIT_SAMPLING_UNDECLARED: Final = "cross_world_uncertainty.unit_sampling_undeclared"


class decision_adapters:  # noqa: N801
    BAYES_OVER_IDENTIFIED_SET: Final = "decision_adapters.bayes_over_identified_set"
    CLAIM_DRAWS_TOO_LARGE: Final = "decision_adapters.claim_draws_too_large"
    CLAIMS_SOURCES_MISMATCH: Final = "decision_adapters.claims_sources_mismatch"
    COMPLETION_COUNT_NOT_PROBABILITY: Final = "decision_adapters.completion_count_not_probability"
    CRITERION_NOT_SUPPORTED_FOR_SET: Final = "decision_adapters.criterion_not_supported_for_set"
    DUPLICATE_ACTION: Final = "decision_adapters.duplicate_action"
    EVALUATED_NEEDS_DRAWS: Final = "decision_adapters.evaluated_needs_draws"
    GRAPH_CLAIMS_CARRY_NO_PROBABILITY: Final = "decision_adapters.graph_claims_carry_no_probability"
    INVALID_CLAIMS: Final = "decision_adapters.invalid_claims"
    INVALID_INTERVAL: Final = "decision_adapters.invalid_interval"
    INVALID_PROBABILITY: Final = "decision_adapters.invalid_probability"
    INVALID_UTILITIES: Final = "decision_adapters.invalid_utilities"
    MISSING_ACTION: Final = "decision_adapters.missing_action"
    POINT_CLAIM_IS_SINGLE: Final = "decision_adapters.point_claim_is_single"
    POLICY_MISMATCH: Final = "decision_adapters.policy_mismatch"
    PROBABILITIES_REQUIRED: Final = "decision_adapters.probabilities_required"
    UNEVALUATED_NEEDS_REASON: Final = "decision_adapters.unevaluated_needs_reason"
    UNKNOWN_ACTION: Final = "decision_adapters.unknown_action"
    UNKNOWN_CLAIM_KIND: Final = "decision_adapters.unknown_claim_kind"
    UNKNOWN_CLAIM_STATUS: Final = "decision_adapters.unknown_claim_status"
    UNKNOWN_SUPPORT_STATUS: Final = "decision_adapters.unknown_support_status"


class decision_admissibility:  # noqa: N801
    DUPLICATE_RULE: Final = "decision_admissibility.duplicate_rule"
    INVALID_RULE: Final = "decision_admissibility.invalid_rule"
    UNKNOWN_ACTION: Final = "decision_admissibility.unknown_action"
    UNKNOWN_INPUT: Final = "decision_admissibility.unknown_input"


class decision_claims:  # noqa: N801
    IDENTITY_MISMATCH: Final = "decision_claims.identity_mismatch"
    MISSING_SCENARIO_UTILITY: Final = "decision_claims.missing_scenario_utility"
    PROBABILITIES_NOT_DECLARED: Final = "decision_claims.probabilities_not_declared"
    RANGE_UNAVAILABLE: Final = "decision_claims.range_unavailable"
    UNRESOLVED_SCENARIOS: Final = "decision_claims.unresolved_scenarios"
    UNSUPPORTED_CLAIM_SHAPE: Final = "decision_claims.unsupported_claim_shape"


class decision_contract:  # noqa: N801
    DUPLICATE_ID: Final = "decision_contract.duplicate_id"
    INVALID_DECLARATION: Final = "decision_contract.invalid_declaration"
    INVALID_PARAMETER: Final = "decision_contract.invalid_parameter"
    INVALID_QUANTITY: Final = "decision_contract.invalid_quantity"
    MISSING_SOURCE: Final = "decision_contract.missing_source"
    OUTSIDE_DECISION_SCOPE: Final = "decision_contract.outside_decision_scope"
    UNKNOWN_ACTION: Final = "decision_contract.unknown_action"
    UNKNOWN_INPUT: Final = "decision_contract.unknown_input"


class decision_evaluation:  # noqa: N801
    DISTRIBUTION_MEANING: Final = "decision_evaluation.distribution_meaning"
    JOINT_LAW_REQUIRED: Final = "decision_evaluation.joint_law_required"
    MEAN_SOURCE_INSUFFICIENT: Final = "decision_evaluation.mean_source_insufficient"
    MEAN_SOURCE_NOT_REPLAYABLE: Final = "decision_evaluation.mean_source_not_replayable"
    NON_FINITE_UTILITY: Final = "decision_evaluation.non_finite_utility"
    QUANTITY_NOT_FOUND: Final = "decision_evaluation.quantity_not_found"
    STRUCTURE_INPUTS_REQUIRED: Final = "decision_evaluation.structure_inputs_required"
    UNSUPPORTED_COORDINATE: Final = "decision_evaluation.unsupported_coordinate"


class decision_evidence:  # noqa: N801
    ADJUSTMENT_NOT_ADMISSIBLE: Final = "decision_evidence.adjustment_not_admissible"
    CANCELLED: Final = "decision_evidence.cancelled"
    DEPENDENCE_UNKNOWN: Final = "decision_evidence.dependence_unknown"
    DEPLOYMENT_BRIDGE_MISSING: Final = "decision_evidence.deployment_bridge_missing"
    FRAME_SHAPE: Final = "decision_evidence.frame_shape"
    INCOMPATIBLE_HORIZON: Final = "decision_evidence.incompatible_horizon"
    INCOMPATIBLE_INTERVENTION: Final = "decision_evidence.incompatible_intervention"
    INCOMPATIBLE_POPULATION: Final = "decision_evidence.incompatible_population"
    INTERFERENCE_UNSPECIFIED: Final = "decision_evidence.interference_unspecified"
    INVALID_DECLARATION: Final = "decision_evidence.invalid_declaration"
    RESOURCE_LIMIT: Final = "decision_evidence.resource_limit"
    TOO_MANY_UNKNOWNS: Final = "decision_evidence.too_many_unknowns"
    UNSUPPORTED_ASSIGNMENT_DESIGN: Final = "decision_evidence.unsupported_assignment_design"


class decision_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "decision_posterior.band_not_constructed"
    EMPTY_LAW: Final = "decision_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "decision_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "decision_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "decision_posterior.non_positive_scale"


class decision_reliability:  # noqa: N801
    ARTIFACT: Final = "decision_reliability.artifact"
    ARTIFACT_ID_MISSING: Final = "decision_reliability.artifact_id_missing"
    ARTIFACT_KIND_MISMATCH: Final = "decision_reliability.artifact_kind_mismatch"
    CANCELLED: Final = "decision_reliability.cancelled"
    CANDIDATE_SET_MISMATCH: Final = "decision_reliability.candidate_set_mismatch"
    COMBINED_UNCERTAINTY: Final = "decision_reliability.combined_uncertainty"
    DECISION_MISMATCH: Final = "decision_reliability.decision_mismatch"
    DEPENDENCE_UNKNOWN: Final = "decision_reliability.dependence_unknown"
    DRAWS_DIGEST_MISMATCH: Final = "decision_reliability.draws_digest_mismatch"
    EFFECTIVE_SAMPLE_SIZE_TOO_SMALL: Final = "decision_reliability.effective_sample_size_too_small"
    IDENTITY_MISMATCH: Final = "decision_reliability.identity_mismatch"
    INVALID_INPUT: Final = "decision_reliability.invalid_input"
    JOINT_LAW_REQUIRED: Final = "decision_reliability.joint_law_required"
    MEMORY_BUDGET_EXCEEDED: Final = "decision_reliability.memory_budget_exceeded"
    NUMERICAL: Final = "decision_reliability.numerical"
    PRODUCER_MEAN_DISAGREES: Final = "decision_reliability.producer_mean_disagrees"
    PRODUCER_POINT_ONLY: Final = "decision_reliability.producer_point_only"
    REPORT_MISMATCH: Final = "decision_reliability.report_mismatch"
    ROUTE_NOT_SUPPORTED: Final = "decision_reliability.route_not_supported"
    SEARCH_RECORD_INCONSISTENT: Final = "decision_reliability.search_record_inconsistent"
    SEARCH_RECORD_REQUIRED: Final = "decision_reliability.search_record_required"
    SELECTION_STREAM_MISMATCH: Final = "decision_reliability.selection_stream_mismatch"
    UTILITY_UNIT_MISMATCH: Final = "decision_reliability.utility_unit_mismatch"
    VALIDATION_NOT_INDEPENDENT: Final = "decision_reliability.validation_not_independent"
    VALIDATION_STREAM_MISMATCH: Final = "decision_reliability.validation_stream_mismatch"


class decision_robustness:  # noqa: N801
    CONTRACT_MISMATCH: Final = "decision_robustness.contract_mismatch"
    INVALID_PROFILE: Final = "decision_robustness.invalid_profile"


class decision_structural:  # noqa: N801
    INVALID_ATOMS: Final = "decision_structural.invalid_atoms"
    INVALID_PROBABILITIES: Final = "decision_structural.invalid_probabilities"
    PROBABILITIES_REQUIRED: Final = "decision_structural.probabilities_required"


class descriptive_comparison:  # noqa: N801
    ADMISSION_REFUSED: Final = "descriptive_comparison.admission_refused"
    ARM_NOT_POPULATED: Final = "descriptive_comparison.arm_not_populated"
    ARM_TOO_SMALL: Final = "descriptive_comparison.arm_too_small"
    CANCELLED: Final = "descriptive_comparison.cancelled"
    COLUMN_ATTRIBUTION_NOT_IDENTIFIED: Final = (
        "descriptive_comparison.column_attribution_not_identified"
    )
    ESTIMAND_CODING_MISMATCH: Final = "descriptive_comparison.estimand_coding_mismatch"
    FOLD_EXCEEDS_LEASE: Final = "descriptive_comparison.fold_exceeds_lease"
    GAP_INTERVAL_UNAVAILABLE: Final = "descriptive_comparison.gap_interval_unavailable"
    INVALID_COVARIANCE: Final = "descriptive_comparison.invalid_covariance"
    INVALID_DATA: Final = "descriptive_comparison.invalid_data"
    JOINT_COVARIANCE_UNAVAILABLE: Final = "descriptive_comparison.joint_covariance_unavailable"
    LEARNER_EXCEEDS_LEASE: Final = "descriptive_comparison.learner_exceeds_lease"
    POPULATION_NOT_ALL_OBSERVED: Final = "descriptive_comparison.population_not_all_observed"
    SCALE_NOT_SUPPORTED: Final = "descriptive_comparison.scale_not_supported"
    TRANSFORM_DOMAIN: Final = "descriptive_comparison.transform_domain"


class design_ranking:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "design_ranking.bounds_exceeded"
    CONTRACT_MISMATCH: Final = "design_ranking.contract_mismatch"
    COST_UNITS_MISMATCH: Final = "design_ranking.cost_units_mismatch"
    DESIGN_ERROR: Final = "design_ranking.design_error"
    DIGEST_MISMATCH: Final = "design_ranking.digest_mismatch"
    DUPLICATE_CANDIDATE: Final = "design_ranking.duplicate_candidate"
    EMPTY: Final = "design_ranking.empty"
    IDENTITY_MISMATCH: Final = "design_ranking.identity_mismatch"
    INVALID_ARTIFACT: Final = "design_ranking.invalid_artifact"
    INVALID_REQUEST: Final = "design_ranking.invalid_request"
    RANK_MISMATCH: Final = "design_ranking.rank_mismatch"
    REPLAY_MISMATCH: Final = "design_ranking.replay_mismatch"
    SIGNAL_INCONSISTENT: Final = "design_ranking.signal_inconsistent"
    SIGNAL_MISMATCH: Final = "design_ranking.signal_mismatch"
    SOURCE_DIGEST_MISMATCH: Final = "design_ranking.source_digest_mismatch"
    SOURCE_OVERLAP: Final = "design_ranking.source_overlap"
    SOURCE_OVERLAP_UNKNOWN: Final = "design_ranking.source_overlap_unknown"
    UNSUPPORTED_VERSION: Final = "design_ranking.unsupported_version"


class discovered_transport:  # noqa: N801
    CLASS_ENUMERATION_PARTIAL: Final = "discovered_transport.class_enumeration_partial"
    CLASS_NOT_IDENTIFIED: Final = "discovered_transport.class_not_identified"
    CLASS_TOO_LARGE: Final = "discovered_transport.class_too_large"
    EVIDENCE_INVALID: Final = "discovered_transport.evidence_invalid"
    MEMBER_BOUND_ZERO: Final = "discovered_transport.member_bound_zero"
    MEMBER_NOT_ESTIMABLE: Final = "discovered_transport.member_not_estimable"
    NO_MEMBERS: Final = "discovered_transport.no_members"
    ORIENTATION_CONFLICT: Final = "discovered_transport.orientation_conflict"
    PAG_CLASS_AMBIGUOUS: Final = "discovered_transport.pag_class_ambiguous"
    PAG_SELECTION_BIAS_EDGE: Final = "discovered_transport.pag_selection_bias_edge"
    SEARCH_INCOMPLETE: Final = "discovered_transport.search_incomplete"
    SELECTION_TARGET_UNKNOWN: Final = "discovered_transport.selection_target_unknown"
    TEMPORAL_QUERY_INVALID: Final = "discovered_transport.temporal_query_invalid"
    TEMPORAL_STRUCTURE_UNSUPPORTED: Final = "discovered_transport.temporal_structure_unsupported"
    TRIAL_COMPATIBILITY_INVALID: Final = "discovered_transport.trial_compatibility_invalid"
    TRIAL_REJECTS_EVERY_MEMBER: Final = "discovered_transport.trial_rejects_every_member"


class discretized_scm:  # noqa: N801
    CANCELLED: Final = "discretized_scm.cancelled"
    COMPOSITION_INCOHERENT: Final = "discretized_scm.composition_incoherent"
    DEGENERATE_MARGINAL: Final = "discretized_scm.degenerate_marginal"
    INTERVENTION_INVALID: Final = "discretized_scm.intervention_invalid"
    LATENT_CONFOUNDING: Final = "discretized_scm.latent_confounding"
    LEVELS_OUT_OF_RANGE: Final = "discretized_scm.levels_out_of_range"
    LEVELS_UNDECLARED: Final = "discretized_scm.levels_undeclared"
    NON_GAUSSIAN_MECHANISM: Final = "discretized_scm.non_gaussian_mechanism"
    NUMERICAL: Final = "discretized_scm.numerical"
    POINT_ONLY: Final = "discretized_scm.point_only"
    STATE_SPACE_TOO_LARGE: Final = "discretized_scm.state_space_too_large"
    SUPPORT_NOT_REAL_LINE: Final = "discretized_scm.support_not_real_line"
    UNKNOWN_VARIABLE: Final = "discretized_scm.unknown_variable"


class discretized_scm_artifact:  # noqa: N801
    CLAIMED_NUMBERS_DIFFER: Final = "discretized_scm_artifact.claimed_numbers_differ"
    IDENTITY_MISMATCH: Final = "discretized_scm_artifact.identity_mismatch"
    MALFORMED: Final = "discretized_scm_artifact.malformed"
    NOT_EXPORTABLE: Final = "discretized_scm_artifact.not_exportable"
    REDERIVATION_REFUSED: Final = "discretized_scm_artifact.rederivation_refused"
    VERSION_UNSUPPORTED: Final = "discretized_scm_artifact.version_unsupported"


class distribution_meaning:  # noqa: N801
    INCOMPATIBLE_OPERATION: Final = "distribution_meaning.incompatible_operation"
    LEGACY_POSTERIOR: Final = "distribution_meaning.legacy_posterior"
    OBSERVATIONAL_NOT_INTERVENTIONAL: Final = (
        "distribution_meaning.observational_not_interventional"
    )
    POSTERIOR_INTERVAL: Final = "distribution_meaning.posterior_interval"


class dml:  # noqa: N801
    ARM_NOT_POPULATED: Final = "dml.arm_not_populated"
    CANCELLED: Final = "dml.cancelled"


class dose_grid:  # noqa: N801
    BANDWIDTH_OUTSIDE_RANGE: Final = "dose_grid.bandwidth_outside_range"
    CANCELLED: Final = "dose_grid.cancelled"
    CONSUMER_LIMIT_EXCEEDED: Final = "dose_grid.consumer_limit_exceeded"
    DATA_IDENTITY_MISMATCH: Final = "dose_grid.data_identity_mismatch"
    DERIVATIVE_WITHOUT_CLAIM: Final = "dose_grid.derivative_without_claim"
    GRAPH_NOT_CERTIFIED: Final = "dose_grid.graph_not_certified"
    INSUFFICIENT_LOCAL_WEIGHT: Final = "dose_grid.insufficient_local_weight"
    INVALID_LAG: Final = "dose_grid.invalid_lag"
    INVALID_REQUEST: Final = "dose_grid.invalid_request"
    LEVEL_WITHOUT_CLAIM: Final = "dose_grid.level_without_claim"
    NO_SIMULTANEOUS_BAND_METHOD: Final = "dose_grid.no_simultaneous_band_method"
    NON_FINITE_RESULT: Final = "dose_grid.non_finite_result"
    PREMISES_MISMATCH: Final = "dose_grid.premises_mismatch"
    REFUSED: Final = "dose_grid.refused"
    RESOURCE_LIMIT: Final = "dose_grid.resource_limit"
    RESOURCE_REFUSED: Final = "dose_grid.resource_refused"
    RESULT_REPLAY_MISMATCH: Final = "dose_grid.result_replay_mismatch"
    TOO_FEW_CLUSTERS: Final = "dose_grid.too_few_clusters"
    UNDECODABLE: Final = "dose_grid.undecodable"
    UNSUPPORTED_DOSE: Final = "dose_grid.unsupported_dose"
    UNSUPPORTED_SEMANTICS: Final = "dose_grid.unsupported_semantics"


class dose_grid_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "dose_grid_posterior.band_not_constructed"
    EXTERNAL_POWER_PRIOR: Final = "dose_grid_posterior.external_power_prior"
    INSUFFICIENT_WEIGHT: Final = "dose_grid_posterior.insufficient_weight"
    MAPPED_PRIOR: Final = "dose_grid_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "dose_grid_posterior.non_positive_scale"


class dose_response:  # noqa: N801
    BOOTSTRAP_BELOW_FLOOR: Final = "dose_response.bootstrap_below_floor"
    BOUNDS_EXCEEDED: Final = "dose_response.bounds_exceeded"
    CATE_OR_SIMULTANEOUS: Final = "dose_response.cate_or_simultaneous"
    DECLARED_FOLDS_UNBALANCED: Final = "dose_response.declared_folds_unbalanced"
    DOSE_DENSITY_FLOOR: Final = "dose_response.dose_density_floor"
    DOSE_DENSITY_INVALID: Final = "dose_response.dose_density_invalid"
    DOSE_TOO_DISCRETE: Final = "dose_response.dose_too_discrete"
    ESTIMATED_DOSE_DENSITY: Final = "dose_response.estimated_dose_density"
    GRID_OUTSIDE_DOSE_SUPPORT: Final = "dose_response.grid_outside_dose_support"
    INVALID_DECLARED_ROWS: Final = "dose_response.invalid_declared_rows"
    KERNEL_NOT_SUPPORTED: Final = "dose_response.kernel_not_supported"
    LEARNER_NOT_PORTABLE: Final = "dose_response.learner_not_portable"
    LOCAL_DOSE_ESS: Final = "dose_response.local_dose_ess"
    MEMBERSHIP_OVERLAP: Final = "dose_response.membership_overlap"
    NON_IID_DESIGN: Final = "dose_response.non_iid_design"
    POSTERIOR_DECLARED_ROWS: Final = "dose_response.posterior_declared_rows"
    QUADRATURE_TOLERANCE: Final = "dose_response.quadrature_tolerance"
    TARGET_NOT_SMOOTHED: Final = "dose_response.target_not_smoothed"


class effect_constancy:  # noqa: N801
    ESTIMAND_MISSING: Final = "effect_constancy.estimand_missing"
    INCOMPATIBLE_PARTITIONS: Final = "effect_constancy.incompatible_partitions"
    INVALID_ALPHA: Final = "effect_constancy.invalid_alpha"
    INVALID_COVARIANCE: Final = "effect_constancy.invalid_covariance"
    INVALID_IDENTITY: Final = "effect_constancy.invalid_identity"
    INVALID_REQUEST: Final = "effect_constancy.invalid_request"
    INVALID_STANDARD_ERROR: Final = "effect_constancy.invalid_standard_error"
    LIMITS_EXCEEDED: Final = "effect_constancy.limits_exceeded"
    NON_FINITE_ESTIMATE: Final = "effect_constancy.non_finite_estimate"
    TOO_FEW_PARTITIONS: Final = "effect_constancy.too_few_partitions"
    TOO_MANY_PARTITIONS: Final = "effect_constancy.too_many_partitions"
    UNKNOWN_REFERENCE: Final = "effect_constancy.unknown_reference"
    UNSUPPORTED_PARTITION: Final = "effect_constancy.unsupported_partition"
    WRONG_CONTRACT: Final = "effect_constancy.wrong_contract"


class effect_constancy_consumer:  # noqa: N801
    ARTIFACT_INVALID: Final = "effect_constancy_consumer.artifact_invalid"
    CATALOG_BINDING_INVALID: Final = "effect_constancy_consumer.catalog_binding_invalid"
    CONTRAST_MISSING: Final = "effect_constancy_consumer.contrast_missing"
    DECISION_UNSUPPORTED: Final = "effect_constancy_consumer.decision_unsupported"
    INVALID_REQUEST: Final = "effect_constancy_consumer.invalid_request"
    LIMITS_EXCEEDED: Final = "effect_constancy_consumer.limits_exceeded"
    PARTITION_MISSING: Final = "effect_constancy_consumer.partition_missing"
    QUANTITY_MISMATCH: Final = "effect_constancy_consumer.quantity_mismatch"
    SUPPORT_UNSUPPORTED: Final = "effect_constancy_consumer.support_unsupported"


class evidence_ledger:  # noqa: N801
    CANCELLED: Final = "evidence_ledger.cancelled"
    DIAGONAL_COVARIANCE_SHARED_ROWS: Final = "evidence_ledger.diagonal_covariance_shared_rows"
    DUPLICATE_RESULT: Final = "evidence_ledger.duplicate_result"
    DUPLICATE_ROW: Final = "evidence_ledger.duplicate_row"
    DUPLICATE_SOURCE: Final = "evidence_ledger.duplicate_source"
    IDENTITY_MISMATCH: Final = "evidence_ledger.identity_mismatch"
    INTERVAL_CONSTRUCTION_UNAVAILABLE: Final = "evidence_ledger.interval_construction_unavailable"
    INTERVAL_REPORT_INVALID: Final = "evidence_ledger.interval_report_invalid"
    INVALID_DECLARATION: Final = "evidence_ledger.invalid_declaration"
    KIND_MISMATCH: Final = "evidence_ledger.kind_mismatch"
    LIMIT_EXCEEDED: Final = "evidence_ledger.limit_exceeded"
    LINEAGE_ROWS_MISMATCH: Final = "evidence_ledger.lineage_rows_mismatch"
    NATIVE_AND_TRANSPORTED: Final = "evidence_ledger.native_and_transported"
    OPERATION_ARITY: Final = "evidence_ledger.operation_arity"
    OVERLAP_UNKNOWN: Final = "evidence_ledger.overlap_unknown"
    REENTRY_DERIVED_FROM_USED_DATA: Final = "evidence_ledger.reentry_derived_from_used_data"
    REENTRY_DUPLICATE: Final = "evidence_ledger.reentry_duplicate"
    REENTRY_PRIOR_PLUS_LIKELIHOOD: Final = "evidence_ledger.reentry_prior_plus_likelihood"
    REENTRY_TRANSFORM: Final = "evidence_ledger.reentry_transform"
    REPEATED_SOURCE: Final = "evidence_ledger.repeated_source"
    REPLAY_MISMATCH: Final = "evidence_ledger.replay_mismatch"
    RESULT_NOT_VALID: Final = "evidence_ledger.result_not_valid"
    SHARED_OBSERVATIONS: Final = "evidence_ledger.shared_observations"
    SOURCE_SUPERSEDED: Final = "evidence_ledger.source_superseded"
    SOURCE_WITHDRAWN: Final = "evidence_ledger.source_withdrawn"
    SUPERSEDE_KIND_MISMATCH: Final = "evidence_ledger.supersede_kind_mismatch"
    SYNTHESIS_MISMATCH: Final = "evidence_ledger.synthesis_mismatch"
    TRANSPORT_IDENTITY_MISSING: Final = "evidence_ledger.transport_identity_missing"
    TRANSPORTED_AS_NATIVE: Final = "evidence_ledger.transported_as_native"
    UNKNOWN_RESULT: Final = "evidence_ledger.unknown_result"
    UNKNOWN_SOURCE: Final = "evidence_ledger.unknown_source"
    UPDATE_DROPS_ROWS: Final = "evidence_ledger.update_drops_rows"


class evidence_obligations:  # noqa: N801
    INVALID_OBLIGATION: Final = "evidence_obligations.invalid_obligation"
    WRONG_CONTRACT: Final = "evidence_obligations.wrong_contract"


class evidence_plan:  # noqa: N801
    BUDGET: Final = "evidence_plan.budget"
    COMPILE_REFUSED: Final = "evidence_plan.compile_refused"
    COST_UNITS_MISMATCH: Final = "evidence_plan.cost_units_mismatch"
    INVALID_REQUEST: Final = "evidence_plan.invalid_request"
    NO_FAILURE_TO_PLAN: Final = "evidence_plan.no_failure_to_plan"
    NO_ROUTE: Final = "evidence_plan.no_route"
    NONE_CERTIFIED: Final = "evidence_plan.none_certified"


class evsi:  # noqa: N801
    ACTION_SET_CHANGED: Final = "evsi.action_set_changed"
    BOUND_VIOLATION: Final = "evsi.bound_violation"
    CANCELLED: Final = "evsi.cancelled"
    COST_MAP_REQUIRED: Final = "evsi.cost_map_required"
    COST_UNITS_MISMATCH: Final = "evsi.cost_units_mismatch"
    DECISION_VALUES_INCOHERENT: Final = "evsi.decision_values_incoherent"
    DECISION_VALUES_NEED_DRAWS: Final = "evsi.decision_values_need_draws"
    DESIGN_ERROR: Final = "evsi.design_error"
    DUPLICATE_CANDIDATE: Final = "evsi.duplicate_candidate"
    EXECUTION_RESOURCE_LIMIT: Final = "evsi.execution_resource_limit"
    INVALID_COST: Final = "evsi.invalid_cost"
    INVALID_REQUEST: Final = "evsi.invalid_request"
    PRIOR_STATES_MISMATCH: Final = "evsi.prior_states_mismatch"
    RECEIPT_MISMATCH: Final = "evsi.receipt_mismatch"
    SAMPLE_SIZE_MISMATCH: Final = "evsi.sample_size_mismatch"
    SOURCE_OVERLAP: Final = "evsi.source_overlap"
    SOURCE_OVERLAP_UNKNOWN: Final = "evsi.source_overlap_unknown"
    UNSCORED_CANDIDATE: Final = "evsi.unscored_candidate"


class external_mechanism:  # noqa: N801
    DECLARATION_INVALID: Final = "external_mechanism.declaration_invalid"
    IMPLEMENTATION_CHANGED: Final = "external_mechanism.implementation_changed"
    IMPLEMENTATION_UNAVAILABLE: Final = "external_mechanism.implementation_unavailable"
    RESOLVER_FAILED: Final = "external_mechanism.resolver_failed"
    RESOLVER_MISSING: Final = "external_mechanism.resolver_missing"


class external_object_verification:  # noqa: N801
    DUPLICATE_PROBE: Final = "external_object_verification.duplicate_probe"
    FAILED_PROBE: Final = "external_object_verification.failed_probe"
    INVALID_PROBE: Final = "external_object_verification.invalid_probe"
    MISSING_PROBE: Final = "external_object_verification.missing_probe"


class external_recalc:  # noqa: N801
    ARTIFACT_INVALID: Final = "external_recalc.artifact_invalid"
    ATTEMPT_LIMIT: Final = "external_recalc.attempt_limit"
    CALLBACK_FAILED: Final = "external_recalc.callback_failed"
    CONTRACT_MISMATCH: Final = "external_recalc.contract_mismatch"
    IDEMPOTENCY_CONFLICT: Final = "external_recalc.idempotency_conflict"
    IDEMPOTENCY_UNSUPPORTED: Final = "external_recalc.idempotency_unsupported"
    INVALID_DESCRIPTOR: Final = "external_recalc.invalid_descriptor"
    INVALID_OUTPUT: Final = "external_recalc.invalid_output"
    INVALID_REQUEST: Final = "external_recalc.invalid_request"
    LIMITS_EXCEEDED: Final = "external_recalc.limits_exceeded"
    OUTPUT_UNAVAILABLE: Final = "external_recalc.output_unavailable"
    POLICY_UNSUPPORTED: Final = "external_recalc.policy_unsupported"
    PROVIDER_MISMATCH: Final = "external_recalc.provider_mismatch"
    PROVIDER_UNAVAILABLE: Final = "external_recalc.provider_unavailable"
    QUANTITY_UNSUPPORTED: Final = "external_recalc.quantity_unsupported"
    REPLAY_MISMATCH: Final = "external_recalc.replay_mismatch"
    REPLAY_UNSUPPORTED: Final = "external_recalc.replay_unsupported"
    RETRY_UNSAFE: Final = "external_recalc.retry_unsafe"


class external_response_binding:  # noqa: N801
    COLUMN_QUANTITY: Final = "external_response_binding.column_quantity"
    COLUMN_SUPPORT_LENGTH: Final = "external_response_binding.column_support_length"
    CONTRACT_NOT_IDENTIFIED: Final = "external_response_binding.contract_not_identified"
    COORDINATE_CONDITIONING: Final = "external_response_binding.coordinate_conditioning"
    COORDINATE_FUNCTIONAL: Final = "external_response_binding.coordinate_functional"
    COORDINATE_HORIZON: Final = "external_response_binding.coordinate_horizon"
    COORDINATE_INVALID_COORDINATE: Final = "external_response_binding.coordinate_invalid_coordinate"
    COORDINATE_POPULATION: Final = "external_response_binding.coordinate_population"
    COORDINATE_REGIME: Final = "external_response_binding.coordinate_regime"
    COORDINATE_TRANSFORM: Final = "external_response_binding.coordinate_transform"
    COORDINATE_UNITS: Final = "external_response_binding.coordinate_units"
    COORDINATE_VARIABLE: Final = "external_response_binding.coordinate_variable"
    DIMENSION: Final = "external_response_binding.dimension"
    DISTRIBUTION_MEANING: Final = "external_response_binding.distribution_meaning"
    EMPTY_COLUMN: Final = "external_response_binding.empty_column"
    GRAPH: Final = "external_response_binding.graph"
    INVALID_CONTRACT: Final = "external_response_binding.invalid_contract"
    MALFORMED_ARTIFACT: Final = "external_response_binding.malformed_artifact"
    MALFORMED_CAPABILITY: Final = "external_response_binding.malformed_capability"
    MALFORMED_EQUIVALENCE: Final = "external_response_binding.malformed_equivalence"
    MALFORMED_IDENTIFICATION: Final = "external_response_binding.malformed_identification"
    MALFORMED_MEANING: Final = "external_response_binding.malformed_meaning"
    MALFORMED_PROBE: Final = "external_response_binding.malformed_probe"
    MALFORMED_QUANTITY: Final = "external_response_binding.malformed_quantity"
    MALFORMED_SUPPORT: Final = "external_response_binding.malformed_support"
    MALFORMED_TRUST: Final = "external_response_binding.malformed_trust"
    MISSING_ASSUMPTION: Final = "external_response_binding.missing_assumption"
    MISSING_EVIDENCE: Final = "external_response_binding.missing_evidence"
    NON_FINITE_VALUE: Final = "external_response_binding.non_finite_value"
    OBJECT_KIND: Final = "external_response_binding.object_kind"
    OBJECT_QUANTITIES: Final = "external_response_binding.object_quantities"
    POSTERIOR_KIND: Final = "external_response_binding.posterior_kind"
    UNCHECKED_OBSERVATIONAL_LAW: Final = "external_response_binding.unchecked_observational_law"


class external_scientific_providers:  # noqa: N801
    DUPLICATE_DECLARATION: Final = "external_scientific_providers.duplicate_declaration"
    INVALID_DECLARATION: Final = "external_scientific_providers.invalid_declaration"
    INVALID_IDENTITY: Final = "external_scientific_providers.invalid_identity"


class external_trust_receipt:  # noqa: N801
    RECEIPT_MISMATCH: Final = "external_trust_receipt.receipt_mismatch"


class finite_action_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "finite_action_posterior.band_not_constructed"
    EMPTY_LAW: Final = "finite_action_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "finite_action_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "finite_action_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "finite_action_posterior.non_positive_scale"


class frozen_scores:  # noqa: N801
    IDENTITY_MISMATCH: Final = "frozen_scores.identity_mismatch"


class functional_inverse_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "functional_inverse_posterior.band_not_constructed"
    EMPTY_LAW: Final = "functional_inverse_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "functional_inverse_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "functional_inverse_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "functional_inverse_posterior.non_positive_scale"


class functional_inverse_query:  # noqa: N801
    ARTIFACT_KIND_MISMATCH: Final = "functional_inverse_query.artifact_kind_mismatch"
    BOUNDS_EXCEEDED: Final = "functional_inverse_query.bounds_exceeded"
    FORWARD_ORIGIN_INVALID: Final = "functional_inverse_query.forward_origin_invalid"
    GLOBAL_FEASIBILITY_CLAIM: Final = "functional_inverse_query.global_feasibility_claim"
    INVALID_ARTIFACT: Final = "functional_inverse_query.invalid_artifact"
    WRONG_CONTRACT: Final = "functional_inverse_query.wrong_contract"


class functional_source:  # noqa: N801
    EXPECTED_IDENTITY_MISMATCH: Final = "functional_source.expected_identity_mismatch"
    INVALID_ARTIFACT: Final = "functional_source.invalid_artifact"
    INVALID_CONTRACT: Final = "functional_source.invalid_contract"
    INVALID_SOURCE: Final = "functional_source.invalid_source"
    LIMITS_EXCEEDED: Final = "functional_source.limits_exceeded"
    MISSING_ACTION: Final = "functional_source.missing_action"
    NATIVE_SOURCE_UNAVAILABLE: Final = "functional_source.native_source_unavailable"
    ORIGINAL_LAW_REQUIRED: Final = "functional_source.original_law_required"
    OUTPUT_MISMATCH: Final = "functional_source.output_mismatch"
    SOURCE_MISMATCH: Final = "functional_source.source_mismatch"


class graph_falsification:  # noqa: N801
    CI_P_VALUE_MISSING: Final = "graph_falsification.ci_p_value_missing"
    CI_RESULT_COUNT: Final = "graph_falsification.ci_result_count"
    CI_TEST_NOT_FREQUENTIST: Final = "graph_falsification.ci_test_not_frequentist"
    DATA_NOT_OBSERVED: Final = "graph_falsification.data_not_observed"
    DUPLICATE_VARIABLE: Final = "graph_falsification.duplicate_variable"
    EMPTY_DATA: Final = "graph_falsification.empty_data"
    INVALID_LEVEL: Final = "graph_falsification.invalid_level"
    INVALID_PERMUTATIONS: Final = "graph_falsification.invalid_permutations"
    NODE_NOT_STATIC: Final = "graph_falsification.node_not_static"
    TOO_FEW_NODES: Final = "graph_falsification.too_few_nodes"
    WORK_LIMIT: Final = "graph_falsification.work_limit"


class graph_imputation:  # noqa: N801
    ATOM_FAILURE: Final = "graph_imputation.atom_failure"
    ATOM_NOT_LOWERED: Final = "graph_imputation.atom_not_lowered"
    CANCELLED: Final = "graph_imputation.cancelled"
    CANDIDATE_REFUSED: Final = "graph_imputation.candidate_refused"
    COMPILED: Final = "graph_imputation.compiled"
    EXECUTION_REFUSED: Final = "graph_imputation.execution_refused"
    INTERVAL_UNSUPPORTED: Final = "graph_imputation.interval_unsupported"
    INVALID_SOURCE: Final = "graph_imputation.invalid_source"
    LOWERING: Final = "graph_imputation.lowering"
    POINT_UNSUPPORTED: Final = "graph_imputation.point_unsupported"
    RESOURCE_LIMIT: Final = "graph_imputation.resource_limit"
    SEARCH_STOPPED: Final = "graph_imputation.search_stopped"
    SUPPORTED_GRAPH_UNIDENTIFIED: Final = "graph_imputation.supported_graph_unidentified"
    UNIDENTIFIED: Final = "graph_imputation.unidentified"
    UNSCORABLE: Final = "graph_imputation.unscorable"
    UNSUPPORTED: Final = "graph_imputation.unsupported"
    WRONG_DISPATCH: Final = "graph_imputation.wrong_dispatch"


class graph_selection_imputation:  # noqa: N801
    COLUMNS: Final = "graph_selection_imputation.columns"
    DATA: Final = "graph_selection_imputation.data"
    DOMAIN: Final = "graph_selection_imputation.domain"
    INTERVAL_UNMEASURED: Final = "graph_selection_imputation.interval_unmeasured"
    MODEL: Final = "graph_selection_imputation.model"
    RESULT: Final = "graph_selection_imputation.result"
    VARIABLES: Final = "graph_selection_imputation.variables"


class identification_repair:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "identification_repair.bounds_exceeded"
    BUDGET: Final = "identification_repair.budget"
    COST_UNITS_MISMATCH: Final = "identification_repair.cost_units_mismatch"
    INVALID_REQUEST: Final = "identification_repair.invalid_request"
    NOT_A_FAILURE: Final = "identification_repair.not_a_failure"
    WRONG_CONTRACT: Final = "identification_repair.wrong_contract"


class initial_state:  # noqa: N801
    INVALID_LAW: Final = "initial_state.invalid_law"
    INVALID_PREMISES: Final = "initial_state.invalid_premises"
    SUPPORT_GAP: Final = "initial_state.support_gap"
    TARGET_LAW_MISSING: Final = "initial_state.target_law_missing"
    TOO_MANY_STATES: Final = "initial_state.too_many_states"


class initial_state_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "initial_state_posterior.band_not_constructed"
    EMPTY_LAW: Final = "initial_state_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "initial_state_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "initial_state_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "initial_state_posterior.non_positive_scale"


class interface_contract:  # noqa: N801
    ARTIFACT_INVALID: Final = "interface_contract.artifact_invalid"
    BLANK_FIELD: Final = "interface_contract.blank_field"
    CALIBRATION_RECORD_BLANK: Final = "interface_contract.calibration_record_blank"
    CALIBRATION_WITHOUT_INTERVAL: Final = "interface_contract.calibration_without_interval"
    CONDITIONING_MISMATCH: Final = "interface_contract.conditioning_mismatch"
    CYCLIC_COMPOSITION: Final = "interface_contract.cyclic_composition"
    DEPENDENCE_UNKNOWN: Final = "interface_contract.dependence_unknown"
    DUPLICATE_COMPONENT: Final = "interface_contract.duplicate_component"
    DUPLICATE_PORT: Final = "interface_contract.duplicate_port"
    FUNCTIONAL_MISMATCH: Final = "interface_contract.functional_mismatch"
    HORIZON_MISMATCH: Final = "interface_contract.horizon_mismatch"
    IDENTITY_MISMATCH: Final = "interface_contract.identity_mismatch"
    INHERITED_INPUT_UNKNOWN: Final = "interface_contract.inherited_input_unknown"
    INVALID_DIGEST: Final = "interface_contract.invalid_digest"
    INVALID_QUANTITY: Final = "interface_contract.invalid_quantity"
    INVALID_SUPPORT: Final = "interface_contract.invalid_support"
    MEASUREMENT_ASSUMPTION_MISMATCH: Final = "interface_contract.measurement_assumption_mismatch"
    MISSING_OPERATIONS: Final = "interface_contract.missing_operations"
    OPERATION_UNAVAILABLE: Final = "interface_contract.operation_unavailable"
    POINT_ONLY_WITH_UNCERTAINTY: Final = "interface_contract.point_only_with_uncertainty"
    POPULATION_MISMATCH: Final = "interface_contract.population_mismatch"
    PORT_MULTIPLY_FED: Final = "interface_contract.port_multiply_fed"
    POSTERIOR_WITHOUT_POSTERIOR_MEANING: Final = (
        "interface_contract.posterior_without_posterior_meaning"
    )
    PROVENANCE_MISSING: Final = "interface_contract.provenance_missing"
    REGIME_MISMATCH: Final = "interface_contract.regime_mismatch"
    SEMANTIC_IDENTITY_MISMATCH: Final = "interface_contract.semantic_identity_mismatch"
    SHARED_EVIDENCE: Final = "interface_contract.shared_evidence"
    SUPPORT_MISMATCH: Final = "interface_contract.support_mismatch"
    TRANSFORM_MISMATCH: Final = "interface_contract.transform_mismatch"
    UNCERTAINTY_MEANING_MISMATCH: Final = "interface_contract.uncertainty_meaning_mismatch"
    UNCERTAINTY_MEANING_MISSING: Final = "interface_contract.uncertainty_meaning_missing"
    UNCERTAINTY_MEANING_REQUIRED: Final = "interface_contract.uncertainty_meaning_required"
    UNCERTAINTY_MEANING_UNUSED: Final = "interface_contract.uncertainty_meaning_unused"
    UNCERTAINTY_PARADIGM_MISMATCH: Final = "interface_contract.uncertainty_paradigm_mismatch"
    UNITS_MISMATCH: Final = "interface_contract.units_mismatch"
    UNKNOWN_COMPONENT: Final = "interface_contract.unknown_component"
    UNKNOWN_PORT: Final = "interface_contract.unknown_port"
    UNKNOWN_PROVENANCE_STAGE: Final = "interface_contract.unknown_provenance_stage"
    VARIABLE_MISMATCH: Final = "interface_contract.variable_mismatch"


class intrinsic_influence:  # noqa: N801
    CORRELATED_NOISE: Final = "intrinsic_influence.correlated_noise"
    DRAW_BUDGET_EXCEEDED: Final = "intrinsic_influence.draw_budget_exceeded"
    INVALID_SETTINGS: Final = "intrinsic_influence.invalid_settings"
    MECHANISM_UNFITTED: Final = "intrinsic_influence.mechanism_unfitted"
    NOISE_INDEPENDENCE_UNDECLARED: Final = "intrinsic_influence.noise_independence_undeclared"
    NOISE_NOT_ROW_INDEPENDENT: Final = "intrinsic_influence.noise_not_row_independent"
    STRUCTURE_MISMATCH: Final = "intrinsic_influence.structure_mismatch"
    TARGET_NOT_CONTINUOUS: Final = "intrinsic_influence.target_not_continuous"


class inverse:  # noqa: N801
    CANCELLED: Final = "inverse.cancelled"
    FORWARD_NOT_MEAN_RESPONSE: Final = "inverse.forward_not_mean_response"
    FORWARD_NOT_POINT_IDENTIFIED: Final = "inverse.forward_not_point_identified"
    FORWARD_NOT_SUPPORTED: Final = "inverse.forward_not_supported"
    INVALID_ACTION: Final = "inverse.invalid_action"
    INVALID_FORWARD: Final = "inverse.invalid_forward"
    INVALID_TARGET: Final = "inverse.invalid_target"
    OBSERVATIONAL_SCENARIOS: Final = "inverse.observational_scenarios"
    PROBABILITY_TARGET: Final = "inverse.probability_target"
    QUANTILE_TARGET: Final = "inverse.quantile_target"


class inverse_forward_source:  # noqa: N801
    COMPOSED_COORDINATE_IS_OUTCOME_LAW: Final = (
        "inverse_forward_source.composed_coordinate_is_outcome_law"
    )
    COMPOSED_FAMILY_MISMATCH: Final = "inverse_forward_source.composed_family_mismatch"
    COMPOSED_LAW_NOT_MATERIALIZED: Final = "inverse_forward_source.composed_law_not_materialized"
    COMPOSED_LEVEL_CODING: Final = "inverse_forward_source.composed_level_coding"
    COMPOSED_QUERY_RESULT_MISMATCH: Final = "inverse_forward_source.composed_query_result_mismatch"
    COMPOSED_RESPONSE_INVALID: Final = "inverse_forward_source.composed_response_invalid"
    INTERVAL_NOT_SUPPORTED: Final = "inverse_forward_source.interval_not_supported"
    ORIGIN_INVALID: Final = "inverse_forward_source.origin_invalid"
    PER_ACTION_NOT_SEALABLE: Final = "inverse_forward_source.per_action_not_sealable"
    POINT_ONLY_LAW_REQUESTED: Final = "inverse_forward_source.point_only_law_requested"
    TRANSPORTED_NOT_ACCEPTED: Final = "inverse_forward_source.transported_not_accepted"


class inverse_observational_scenarios:  # noqa: N801
    CONDITION_ON_OUTCOME: Final = "inverse_observational_scenarios.condition_on_outcome"
    COORDINATE_MISSING: Final = "inverse_observational_scenarios.coordinate_missing"
    COORDINATE_UNSUPPORTED: Final = "inverse_observational_scenarios.coordinate_unsupported"
    EVENT_HAS_NO_MASS: Final = "inverse_observational_scenarios.event_has_no_mass"
    INVALID_PARAMETER: Final = "inverse_observational_scenarios.invalid_parameter"
    INVALID_SCENARIOS: Final = "inverse_observational_scenarios.invalid_scenarios"
    JOINT_LAW_REQUIRED: Final = "inverse_observational_scenarios.joint_law_required"
    MEANING_NOT_OBSERVATIONAL: Final = "inverse_observational_scenarios.meaning_not_observational"
    NON_FINITE_DRAW: Final = "inverse_observational_scenarios.non_finite_draw"
    RESULT_MISMATCH: Final = "inverse_observational_scenarios.result_mismatch"
    UNKNOWN_SCENARIO: Final = "inverse_observational_scenarios.unknown_scenario"


class joint_cells:  # noqa: N801
    CANCELLED: Final = "joint_cells.cancelled"
    CELL_EMPTY: Final = "joint_cells.cell_empty"
    CELL_ESS_BELOW_MINIMUM: Final = "joint_cells.cell_ess_below_minimum"
    COLUMNS: Final = "joint_cells.columns"
    COMPONENT_COUNT: Final = "joint_cells.component_count"
    CONDITIONAL_UNSUPPORTED: Final = "joint_cells.conditional_unsupported"
    CONFIG: Final = "joint_cells.config"
    CONTRAST_CELL_UNSUPPORTED: Final = "joint_cells.contrast_cell_unsupported"
    DERIVED_CONSTITUENT_COPY_IN_ADJUSTMENT: Final = (
        "joint_cells.derived_constituent_copy_in_adjustment"
    )
    DERIVED_CONSTITUENT_IN_ADJUSTMENT: Final = "joint_cells.derived_constituent_in_adjustment"
    DERIVED_DECLARATION_INVALID: Final = "joint_cells.derived_declaration_invalid"
    DERIVED_DESCENDANT_IN_ADJUSTMENT: Final = "joint_cells.derived_descendant_in_adjustment"
    DERIVED_DUPLICATE_CONSTITUENTS: Final = "joint_cells.derived_duplicate_constituents"
    DERIVED_EXCLUSION_NOT_IN_ADJUSTMENT: Final = "joint_cells.derived_exclusion_not_in_adjustment"
    DERIVED_EXCLUSION_RULE_MISMATCH: Final = "joint_cells.derived_exclusion_rule_mismatch"
    DERIVED_ILLEGAL_VALUE: Final = "joint_cells.derived_illegal_value"
    DERIVED_INTERVENTION_NOT_DEFINED: Final = "joint_cells.derived_intervention_not_defined"
    DERIVED_NOT_A_JOINT_CELL: Final = "joint_cells.derived_not_a_joint_cell"
    DERIVED_OUTCOME_IN_ADJUSTMENT: Final = "joint_cells.derived_outcome_in_adjustment"
    DERIVED_RANK_DEFICIENT: Final = "joint_cells.derived_rank_deficient"
    DERIVED_TREATMENT_TRACKED_BY_COVARIATES: Final = (
        "joint_cells.derived_treatment_tracked_by_covariates"
    )
    DERIVED_UNKNOWN_COLUMN: Final = "joint_cells.derived_unknown_column"
    LEARNER: Final = "joint_cells.learner"
    LEARNER_UNAVAILABLE: Final = "joint_cells.learner_unavailable"
    ML_LEARNER_NOT_DECLARED: Final = "joint_cells.ml_learner_not_declared"
    NO_INTERVAL_METHOD: Final = "joint_cells.no_interval_method"
    NO_ML_INTERVAL_METHOD: Final = "joint_cells.no_ml_interval_method"
    NO_ROWS: Final = "joint_cells.no_rows"
    NO_SUPPORTED_CELL: Final = "joint_cells.no_supported_cell"
    NORMALIZATION_FAILED: Final = "joint_cells.normalization_failed"
    NOT_BINARY: Final = "joint_cells.not_binary"
    NUISANCE_PROVIDER: Final = "joint_cells.nuisance_provider"
    ORDERING: Final = "joint_cells.ordering"
    OUTCOME_MODEL_UNSUPPORTED: Final = "joint_cells.outcome_model_unsupported"


class joint_law_propagation:  # noqa: N801
    ALIGNMENT_MISMATCH: Final = "joint_law_propagation.alignment_mismatch"
    ARTIFACT_INVALID: Final = "joint_law_propagation.artifact_invalid"
    ARTIFACT_REFUSED: Final = "joint_law_propagation.artifact_refused"
    BOUNDS_EXCEEDED: Final = "joint_law_propagation.bounds_exceeded"
    CANCELLED: Final = "joint_law_propagation.cancelled"
    COVARIANCE_INSUFFICIENT_FOR_NONLINEAR: Final = (
        "joint_law_propagation.covariance_insufficient_for_nonlinear"
    )
    COVERAGE_RECORD_CONSTRUCTION_MISMATCH: Final = (
        "joint_law_propagation.coverage_record_construction_mismatch"
    )
    COVERAGE_RECORD_COORDINATE_MISMATCH: Final = (
        "joint_law_propagation.coverage_record_coordinate_mismatch"
    )
    DEPENDENCE_UNKNOWN: Final = "joint_law_propagation.dependence_unknown"
    ENVELOPE_IS_NOT_A_POINT: Final = "joint_law_propagation.envelope_is_not_a_point"
    ENVELOPE_NOT_AVAILABLE: Final = "joint_law_propagation.envelope_not_available"
    EVIDENCE_OVERLAP_UNKNOWN: Final = "joint_law_propagation.evidence_overlap_unknown"
    HORIZON_COMPOSITION: Final = "joint_law_propagation.horizon_composition"
    HORIZON_DRAWS_DOSE: Final = "joint_law_propagation.horizon_draws_dose"
    HORIZON_DRAWS_LAYOUT: Final = "joint_law_propagation.horizon_draws_layout"
    HORIZON_DRAWS_MEAN_CURVE: Final = "joint_law_propagation.horizon_draws_mean_curve"
    HORIZON_DRAWS_NOT_JOINT: Final = "joint_law_propagation.horizon_draws_not_joint"
    INDEX_PAIRING_REFUSED: Final = "joint_law_propagation.index_pairing_refused"
    INVALID_REQUEST: Final = "joint_law_propagation.invalid_request"
    MIXTURE_WEIGHTS_UNKNOWN: Final = "joint_law_propagation.mixture_weights_unknown"
    NO_INTERVAL_CONSTRUCTION: Final = "joint_law_propagation.no_interval_construction"
    NO_POSTERIOR_FOR_FIXED_LAW_INPUTS: Final = (
        "joint_law_propagation.no_posterior_for_fixed_law_inputs"
    )
    NONLINEAR_NEEDS_ALIGNED_JOINT_LAW: Final = (
        "joint_law_propagation.nonlinear_needs_aligned_joint_law"
    )
    SHARED_EVIDENCE_DECLARED_INDEPENDENT: Final = (
        "joint_law_propagation.shared_evidence_declared_independent"
    )
    SPREAD_NEEDS_SUPPORTED_DEPENDENCE: Final = (
        "joint_law_propagation.spread_needs_supported_dependence"
    )
    STRUCTURE_NOT_VERIFIED: Final = "joint_law_propagation.structure_not_verified"
    TRANSPORT_PIECES_INCOMPLETE: Final = "joint_law_propagation.transport_pieces_incomplete"
    TRANSPORTED_DEPENDENCE_REQUIRED: Final = "joint_law_propagation.transported_dependence_required"


class joint_sensitivity:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "joint_sensitivity.bounds_exceeded"
    BUDGET: Final = "joint_sensitivity.budget"
    BUDGET_COUPLING: Final = "joint_sensitivity.budget_coupling"
    CANCELLED: Final = "joint_sensitivity.cancelled"
    CONSUMER_LIMITS: Final = "joint_sensitivity.consumer_limits"
    DATA_IDENTITY_MISMATCH: Final = "joint_sensitivity.data_identity_mismatch"
    DUPLICATE_FACTOR: Final = "joint_sensitivity.duplicate_factor"
    FACTOR_COUNT: Final = "joint_sensitivity.factor_count"
    FIXED_GRAPH_CONDITIONAL: Final = "joint_sensitivity.fixed_graph_conditional"
    FIXED_GRAPH_PARENT: Final = "joint_sensitivity.fixed_graph_parent"
    FORMULA_INCOMPATIBLE: Final = "joint_sensitivity.formula_incompatible"
    INCOMPLETE_KERNEL: Final = "joint_sensitivity.incomplete_kernel"
    INVALID_FRACTION: Final = "joint_sensitivity.invalid_fraction"
    INVALID_THRESHOLD: Final = "joint_sensitivity.invalid_threshold"
    INVALID_TOLERANCE: Final = "joint_sensitivity.invalid_tolerance"
    NO_INTERVAL_METHOD: Final = "joint_sensitivity.no_interval_method"
    PREMISES_MISMATCH: Final = "joint_sensitivity.premises_mismatch"
    PROVIDER_MISMATCH: Final = "joint_sensitivity.provider_mismatch"
    REPLAY_MISMATCH: Final = "joint_sensitivity.replay_mismatch"
    SOURCE_TARGET_DISCREPANCY: Final = "joint_sensitivity.source_target_discrepancy"
    STAGE_ACCOUNTING: Final = "joint_sensitivity.stage_accounting"
    TREATMENT_FACTOR: Final = "joint_sensitivity.treatment_factor"
    UNION_LABELLED_CI: Final = "joint_sensitivity.union_labelled_ci"
    UNKNOWN_FACTOR: Final = "joint_sensitivity.unknown_factor"
    UNSUPPORTED_DOMAIN: Final = "joint_sensitivity.unsupported_domain"


class latent_class:  # noqa: N801
    CANCELLED: Final = "latent_class.cancelled"
    CONSTANT_OUTCOME: Final = "latent_class.constant_outcome"
    DATA_REQUIRED: Final = "latent_class.data_required"
    DEGENERATE_CLASS: Final = "latent_class.degenerate_class"
    DESIGN_SUPPORT_TOO_SMALL: Final = "latent_class.design_support_too_small"
    EXPORT_BUDGET_EXCEEDED: Final = "latent_class.export_budget_exceeded"
    INVALID_CONFIG: Final = "latent_class.invalid_config"
    INVALID_REQUEST: Final = "latent_class.invalid_request"
    LENGTH_MISMATCH: Final = "latent_class.length_mismatch"
    LIKELIHOOD_NOT_MONOTONE: Final = "latent_class.likelihood_not_monotone"
    LIMITS_EXCEEDED: Final = "latent_class.limits_exceeded"
    MEMORY_BUDGET_EXCEEDED: Final = "latent_class.memory_budget_exceeded"
    NON_FINITE_INPUT: Final = "latent_class.non_finite_input"
    NOT_CONVERGED: Final = "latent_class.not_converged"
    RANDOMIZATION_NOT_DECLARED: Final = "latent_class.randomization_not_declared"
    REPORT_REPLAY_MISMATCH: Final = "latent_class.report_replay_mismatch"
    TOO_FEW_ROWS: Final = "latent_class.too_few_rows"
    TREATMENT_CONSTANT: Final = "latent_class.treatment_constant"
    WEAK_CLASS: Final = "latent_class.weak_class"
    WRONG_CONTRACT: Final = "latent_class.wrong_contract"


class latent_class_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "latent_class_posterior.band_not_constructed"
    EMPTY_DESIGN: Final = "latent_class_posterior.empty_design"
    EXTERNAL_POWER_PRIOR: Final = "latent_class_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "latent_class_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "latent_class_posterior.non_positive_scale"


class learned_continuous_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "learned_continuous_posterior.band_not_constructed"
    EMPTY_SCORES: Final = "learned_continuous_posterior.empty_scores"
    EXTERNAL_POWER_PRIOR: Final = "learned_continuous_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "learned_continuous_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "learned_continuous_posterior.non_positive_scale"


class learned_joint_transport:  # noqa: N801
    ADMISSION_REFUSED: Final = "learned_joint_transport.admission_refused"
    CANCELLED: Final = "learned_joint_transport.cancelled"
    FEATURE_MISMATCH: Final = "learned_joint_transport.feature_mismatch"
    FIT_FAILED: Final = "learned_joint_transport.fit_failed"
    IDENTIFICATION_NOT_CERTIFIED: Final = "learned_joint_transport.identification_not_certified"
    IDENTIFICATION_REQUIRED: Final = "learned_joint_transport.identification_required"
    INVALID_BASIS: Final = "learned_joint_transport.invalid_basis"
    INVALID_INPUT: Final = "learned_joint_transport.invalid_input"
    INVALID_PRIOR: Final = "learned_joint_transport.invalid_prior"
    MISSING_LAW: Final = "learned_joint_transport.missing_law"
    MISSING_SOURCE: Final = "learned_joint_transport.missing_source"
    PRIOR_DATA_OVERLAP: Final = "learned_joint_transport.prior_data_overlap"
    PRIOR_DIMENSION: Final = "learned_joint_transport.prior_dimension"
    RANK_DEFICIENT: Final = "learned_joint_transport.rank_deficient"
    SOURCE_DEPENDENCE: Final = "learned_joint_transport.source_dependence"
    SUPPORT_RESOURCE_LIMIT: Final = "learned_joint_transport.support_resource_limit"
    TOO_MANY_DRAWS: Final = "learned_joint_transport.too_many_draws"
    TOO_MANY_PARAMETERS: Final = "learned_joint_transport.too_many_parameters"
    UNSUPPORTED_GRAPH: Final = "learned_joint_transport.unsupported_graph"
    WEAK_OVERLAP: Final = "learned_joint_transport.weak_overlap"


class learned_transport:  # noqa: N801
    BOOTSTRAP_BELOW_FLOOR: Final = "learned_transport.bootstrap_below_floor"
    BOOTSTRAP_NOT_SUPPORTED: Final = "learned_transport.bootstrap_not_supported"
    CATE_REQUESTED: Final = "learned_transport.cate_requested"
    MEMBERSHIP_OVERLAP: Final = "learned_transport.membership_overlap"
    NON_IID_DESIGN: Final = "learned_transport.non_iid_design"
    TREATMENT_OVERLAP: Final = "learned_transport.treatment_overlap"


class mapped_posterior_transfer:  # noqa: N801
    COEFFICIENT_MAP: Final = "mapped_posterior_transfer.coefficient_map"
    PARAMETER_POSTERIOR: Final = "mapped_posterior_transfer.parameter_posterior"
    RESIDUAL_VARIANCE: Final = "mapped_posterior_transfer.residual_variance"


class matched_case_control:  # noqa: N801
    ABSOLUTE_RISK_NOT_IDENTIFIED: Final = "matched_case_control.absolute_risk_not_identified"
    BUDGET: Final = "matched_case_control.budget"
    ESTIMAND_NOT_SUPPORTED: Final = "matched_case_control.estimand_not_supported"
    ESTIMATE_NOT_FINITE: Final = "matched_case_control.estimate_not_finite"
    INVALID_DATA: Final = "matched_case_control.invalid_data"
    INVALID_LEVEL: Final = "matched_case_control.invalid_level"
    NO_INFORMATIVE_SETS: Final = "matched_case_control.no_informative_sets"
    SAMPLING_DESIGN: Final = "matched_case_control.sampling_design"


class measured_inference:  # noqa: N801
    ARTIFACT_MISMATCH: Final = "measured_inference.artifact_mismatch"
    BOUNDS_EXCEEDED: Final = "measured_inference.bounds_exceeded"
    CONSUMER_BOUNDS_EXCEEDED: Final = "measured_inference.consumer_bounds_exceeded"
    INVALID_EXPECTATION: Final = "measured_inference.invalid_expectation"
    INVALID_REPORT: Final = "measured_inference.invalid_report"
    MEMORY_BUDGET_EXCEEDED: Final = "measured_inference.memory_budget_exceeded"
    SOURCE_REPORT_NOT_SUPPORTED: Final = "measured_inference.source_report_not_supported"
    UNSUPPORTED_ROUTE: Final = "measured_inference.unsupported_route"


class measured_interval:  # noqa: N801
    DRAWS_NOT_CONVERGED: Final = "measured_interval.draws_not_converged"
    INVALID_SOURCE: Final = "measured_interval.invalid_source"
    NO_INTERVAL_OF_THIS_KIND: Final = "measured_interval.no_interval_of_this_kind"
    SOURCE_MISMATCH: Final = "measured_interval.source_mismatch"
    VARIANCE_ASSUMPTION_NOT_MET: Final = "measured_interval.variance_assumption_not_met"


class measured_transport:  # noqa: N801
    INPUT_BOUNDS_EXCEEDED: Final = "measured_transport.input_bounds_exceeded"
    SAMPLER_MISMATCH: Final = "measured_transport.sampler_mismatch"


class measurement_model:  # noqa: N801
    ADMISSION_REFUSED: Final = "measurement_model.admission_refused"
    ARM_MISMATCH: Final = "measurement_model.arm_mismatch"
    ARTIFACT_INVALID: Final = "measurement_model.artifact_invalid"
    ARTIFACT_REFUSED: Final = "measurement_model.artifact_refused"
    BINARY_OUTCOME_CONTINUOUS_ERROR: Final = "measurement_model.binary_outcome_continuous_error"
    BLANK_FIELD: Final = "measurement_model.blank_field"
    BOOTSTRAP_UNSTABLE: Final = "measurement_model.bootstrap_unstable"
    CANCELLED: Final = "measurement_model.cancelled"
    CONFUSION_ILL_CONDITIONED: Final = "measurement_model.confusion_ill_conditioned"
    CONSTRUCT_MISMATCH: Final = "measurement_model.construct_mismatch"
    CONTINUOUS_EXPORT: Final = "measurement_model.continuous_export"
    DESIGN_SINGULAR: Final = "measurement_model.design_singular"
    DIFFERENTIAL_AXIS_UNSUPPORTED: Final = "measurement_model.differential_axis_unsupported"
    DIFFERENTIAL_DECLARED: Final = "measurement_model.differential_declared"
    DIFFERENTIAL_POPULATION_UNESTABLISHED: Final = (
        "measurement_model.differential_population_unestablished"
    )
    DIFFERENTIAL_UNDECLARED: Final = "measurement_model.differential_undeclared"
    ERROR_VARIANCE_NOT_IDENTIFIED: Final = "measurement_model.error_variance_not_identified"
    INDICATOR_AS_CONSTRUCT: Final = "measurement_model.indicator_as_construct"
    INDICATOR_KIND: Final = "measurement_model.indicator_kind"
    INDICATOR_MISMATCH: Final = "measurement_model.indicator_mismatch"
    INDICATOR_OBSERVED: Final = "measurement_model.indicator_observed"
    INPUT_INVALID: Final = "measurement_model.input_invalid"
    INSUFFICIENT_DRAWS: Final = "measurement_model.insufficient_draws"
    INSUFFICIENT_REPLICATES: Final = "measurement_model.insufficient_replicates"
    INTERVAL_LEVEL: Final = "measurement_model.interval_level"
    INTERVAL_REQUIRES_ESTIMATE: Final = "measurement_model.interval_requires_estimate"
    INTERVENTION_MEANING_NOT_CARRIED: Final = "measurement_model.intervention_meaning_not_carried"
    INVALID_MODEL: Final = "measurement_model.invalid_model"
    INVARIANCE_NOT_DECLARED: Final = "measurement_model.invariance_not_declared"
    INVARIANCE_PREMISE_REQUIRED: Final = "measurement_model.invariance_premise_required"
    INVARIANCE_SOURCE_UNKNOWN: Final = "measurement_model.invariance_source_unknown"
    INVARIANCE_TARGET_MISMATCH: Final = "measurement_model.invariance_target_mismatch"
    LEARNED_COORDINATES_NOT_INTERVENABLE: Final = (
        "measurement_model.learned_coordinates_not_intervenable"
    )
    MAPPING_ENDPOINTS: Final = "measurement_model.mapping_endpoints"
    NO_MODEL: Final = "measurement_model.no_model"
    NOTHING_TO_TRANSPORT: Final = "measurement_model.nothing_to_transport"
    OPERATION_NOT_SUPPORTED: Final = "measurement_model.operation_not_supported"
    OUTCOME_MODEL_NOT_FITTED: Final = "measurement_model.outcome_model_not_fitted"
    PARAMETER: Final = "measurement_model.parameter"
    POPULATION_MISMATCH: Final = "measurement_model.population_mismatch"
    POSTERIOR_BOUND_TO_SNAPSHOT: Final = "measurement_model.posterior_bound_to_snapshot"
    POSTERIOR_CAPABILITY: Final = "measurement_model.posterior_capability"
    POSTERIOR_DECLARATION: Final = "measurement_model.posterior_declaration"
    POSTERIOR_IDENTITY: Final = "measurement_model.posterior_identity"
    POSTERIOR_KIND: Final = "measurement_model.posterior_kind"
    POSTERIOR_NATIVE_TRUST: Final = "measurement_model.posterior_native_trust"
    POSTERIOR_QUANTITY: Final = "measurement_model.posterior_quantity"
    POSTERIOR_RECEIPT: Final = "measurement_model.posterior_receipt"
    PROXY_AS_CONSTRUCT: Final = "measurement_model.proxy_as_construct"
    REGIME_MISMATCH: Final = "measurement_model.regime_mismatch"
    RELIABILITY_TOO_SMALL: Final = "measurement_model.reliability_too_small"
    REPRESENTATION_UNMAPPED: Final = "measurement_model.representation_unmapped"
    ROUTE_MISMATCH: Final = "measurement_model.route_mismatch"
    ROUTE_PARAMETERS: Final = "measurement_model.route_parameters"
    STRUCTURE_PARAMETERS: Final = "measurement_model.structure_parameters"
    TRANSPORT_RECEIPT_MISMATCH: Final = "measurement_model.transport_receipt_mismatch"
    UNKNOWN_LEVEL: Final = "measurement_model.unknown_level"
    UNMODELLED_PROXY: Final = "measurement_model.unmodelled_proxy"
    UNRECOGNIZED: Final = "measurement_model.unrecognized"
    VARIANCE_NOT_POSITIVE: Final = "measurement_model.variance_not_positive"
    YOUDEN_TOO_SMALL: Final = "measurement_model.youden_too_small"


class mechanism_composition:  # noqa: N801
    ARTIFACT_INVALID: Final = "mechanism_composition.artifact_invalid"
    ARTIFACT_KIND_MISMATCH: Final = "mechanism_composition.artifact_kind_mismatch"
    CANCELLED: Final = "mechanism_composition.cancelled"
    CONDITIONING_EXTRA_CONDITIONER: Final = "mechanism_composition.conditioning_extra_conditioner"
    CONDITIONING_PARENT_NOT_CONDITIONED: Final = (
        "mechanism_composition.conditioning_parent_not_conditioned"
    )
    CYCLE_DIRECTED_CYCLE: Final = "mechanism_composition.cycle_directed_cycle"
    FAMILY_MIXED: Final = "mechanism_composition.family_mixed"
    IDENTITY_DECLARATION_UNKNOWN_VARIABLE: Final = (
        "mechanism_composition.identity_declaration_unknown_variable"
    )
    IDENTITY_DUPLICATE_MECHANISM: Final = "mechanism_composition.identity_duplicate_mechanism"
    IDENTITY_EDGE_UNKNOWN_VARIABLE: Final = "mechanism_composition.identity_edge_unknown_variable"
    IDENTITY_PARENT_WITHOUT_MECHANISM: Final = (
        "mechanism_composition.identity_parent_without_mechanism"
    )
    INTERVAL_INVALID: Final = "mechanism_composition.interval_invalid"
    LATENT_FAMILY_UNSUPPORTED: Final = "mechanism_composition.latent_family_unsupported"
    LATENT_LOADING_UNDECLARED: Final = "mechanism_composition.latent_loading_undeclared"
    LATENT_SHARED_WITH_ABSENT_MECHANISM: Final = (
        "mechanism_composition.latent_shared_with_absent_mechanism"
    )
    LATENT_SHARING_NOT_DECLARED: Final = "mechanism_composition.latent_sharing_not_declared"
    LATENT_SHARING_NOT_RECIPROCAL: Final = "mechanism_composition.latent_sharing_not_reciprocal"
    LATENT_TOO_MANY: Final = "mechanism_composition.latent_too_many"
    NUMBERS_UNBOUND: Final = "mechanism_composition.numbers_unbound"
    NUMERICAL_FAILURE: Final = "mechanism_composition.numerical_failure"
    PARENTS_DECLARED_GRAPH_MISMATCH: Final = "mechanism_composition.parents_declared_graph_mismatch"
    POPULATION_INVARIANCE_UNSTATED: Final = "mechanism_composition.population_invariance_unstated"
    POPULATION_LATENT_BLOCKS_SELECTION_ADMISSIBILITY: Final = (
        "mechanism_composition.population_latent_blocks_selection_admissibility"
    )
    POPULATION_PARENT_CONTEXT_DIFFERS_FROM_FIT: Final = (
        "mechanism_composition.population_parent_context_differs_from_fit"
    )
    POPULATION_SELECTION_ARROW_INTO_MECHANISM: Final = (
        "mechanism_composition.population_selection_arrow_into_mechanism"
    )
    POPULATION_UNDECLARED_TRANSPORT: Final = "mechanism_composition.population_undeclared_transport"
    POSTERIOR_DEPENDENCE_CROSS_BLOCK_DUPLICATE: Final = (
        "mechanism_composition.posterior_dependence_cross_block_duplicate"
    )
    POSTERIOR_DEPENDENCE_CROSS_COVARIANCE_MISSING: Final = (
        "mechanism_composition.posterior_dependence_cross_covariance_missing"
    )
    POSTERIOR_DEPENDENCE_EVIDENCE_OVERLAP_UNKNOWN: Final = (
        "mechanism_composition.posterior_dependence_evidence_overlap_unknown"
    )
    POSTERIOR_DEPENDENCE_INDEPENDENCE_UNJUSTIFIED: Final = (
        "mechanism_composition.posterior_dependence_independence_unjustified"
    )
    POSTERIOR_DEPENDENCE_SHARED_EVIDENCE_DECLARED_INDEPENDENT: Final = (
        "mechanism_composition.posterior_dependence_shared_evidence_declared_independent"
    )
    POSTERIOR_DEPENDENCE_UNKNOWN: Final = "mechanism_composition.posterior_dependence_unknown"
    PREMISES_MISMATCH: Final = "mechanism_composition.premises_mismatch"
    PROVENANCE_DECLARED_MECHANISM_UNACKNOWLEDGED: Final = (
        "mechanism_composition.provenance_declared_mechanism_unacknowledged"
    )
    PROVENANCE_MISSING_ARTIFACT: Final = "mechanism_composition.provenance_missing_artifact"
    PROVENANCE_MISSING_EVIDENCE: Final = "mechanism_composition.provenance_missing_evidence"
    QUANTITY_CONDITIONED_EVENT: Final = "mechanism_composition.quantity_conditioned_event"
    QUANTITY_CONDITIONING_MISMATCH: Final = "mechanism_composition.quantity_conditioning_mismatch"
    QUANTITY_FUNCTIONAL_MISMATCH: Final = "mechanism_composition.quantity_functional_mismatch"
    QUANTITY_HORIZON_MISMATCH: Final = "mechanism_composition.quantity_horizon_mismatch"
    QUANTITY_INVALID_COORDINATE: Final = "mechanism_composition.quantity_invalid_coordinate"
    QUANTITY_NOT_A_REALIZATION: Final = "mechanism_composition.quantity_not_a_realization"
    QUANTITY_ROLE_MISMATCH: Final = "mechanism_composition.quantity_role_mismatch"
    REGIME_PARENT_CONTEXT_DIFFERS_FROM_FIT: Final = (
        "mechanism_composition.regime_parent_context_differs_from_fit"
    )
    REGIME_SELECTION_DIAGRAM_NOT_APPLICABLE: Final = (
        "mechanism_composition.regime_selection_diagram_not_applicable"
    )
    REGIME_UNDECLARED_INVARIANCE: Final = "mechanism_composition.regime_undeclared_invariance"
    REQUEST_UNSUPPORTED: Final = "mechanism_composition.request_unsupported"
    RESULT_MISMATCH: Final = "mechanism_composition.result_mismatch"
    SHAPE_BLANK_COORDINATE: Final = "mechanism_composition.shape_blank_coordinate"
    SHAPE_BLANK_IDENTITY: Final = "mechanism_composition.shape_blank_identity"
    SHAPE_COEFFICIENT_PARENT_MISMATCH: Final = (
        "mechanism_composition.shape_coefficient_parent_mismatch"
    )
    SHAPE_DUPLICATE_PARENT: Final = "mechanism_composition.shape_duplicate_parent"
    SHAPE_EMPTY_COMPOSITION: Final = "mechanism_composition.shape_empty_composition"
    SHAPE_INVALID_DOMAIN: Final = "mechanism_composition.shape_invalid_domain"
    SHAPE_INVALID_LATENTS: Final = "mechanism_composition.shape_invalid_latents"
    SHAPE_INVALID_TABLE: Final = "mechanism_composition.shape_invalid_table"
    SHAPE_MECHANISM_COUNT: Final = "mechanism_composition.shape_mechanism_count"
    SHAPE_NEGATIVE_SCALE: Final = "mechanism_composition.shape_negative_scale"
    SHAPE_NON_FINITE_PARAMETER: Final = "mechanism_composition.shape_non_finite_parameter"
    SHAPE_TOO_MANY_MECHANISMS: Final = "mechanism_composition.shape_too_many_mechanisms"
    SHAPE_TOO_MANY_PARENTS: Final = "mechanism_composition.shape_too_many_parents"
    SLOT_NOT_LINEAR_GAUSSIAN: Final = "mechanism_composition.slot_not_linear_gaussian"
    SUPPORT_DISJOINT: Final = "mechanism_composition.support_disjoint"
    SUPPORT_INTERVENTION_OUTSIDE_DOMAIN: Final = (
        "mechanism_composition.support_intervention_outside_domain"
    )
    SUPPORT_KIND_MISMATCH: Final = "mechanism_composition.support_kind_mismatch"
    SUPPORT_MISSING_PARENT_CONFIGURATION: Final = (
        "mechanism_composition.support_missing_parent_configuration"
    )
    SUPPORT_NOT_CONTAINED: Final = "mechanism_composition.support_not_contained"
    TRANSFORM_MISMATCH: Final = "mechanism_composition.transform_mismatch"
    UNCERTAINTY_ASSEMBLED_NOT_PSD: Final = "mechanism_composition.uncertainty_assembled_not_psd"
    UNCERTAINTY_CROSS_BLOCK_SHAPE: Final = "mechanism_composition.uncertainty_cross_block_shape"
    UNCERTAINTY_CROSS_BLOCK_UNKNOWN_MECHANISM: Final = (
        "mechanism_composition.uncertainty_cross_block_unknown_mechanism"
    )
    UNCERTAINTY_FAMILY_UNSUPPORTED: Final = "mechanism_composition.uncertainty_family_unsupported"
    UNCERTAINTY_JOINT_PARAMETER_SET_MISMATCH: Final = (
        "mechanism_composition.uncertainty_joint_parameter_set_mismatch"
    )
    UNCERTAINTY_MARGINAL_INCONSISTENT_WITH_JOINT: Final = (
        "mechanism_composition.uncertainty_marginal_inconsistent_with_joint"
    )
    UNCERTAINTY_MARGINAL_MISSING: Final = "mechanism_composition.uncertainty_marginal_missing"
    UNCERTAINTY_MEANING_MISMATCH: Final = "mechanism_composition.uncertainty_meaning_mismatch"
    UNCERTAINTY_NOT_PSD: Final = "mechanism_composition.uncertainty_not_psd"
    UNCERTAINTY_NOT_SYMMETRIC: Final = "mechanism_composition.uncertainty_not_symmetric"
    UNCERTAINTY_PARAMETER_SET_MISMATCH: Final = (
        "mechanism_composition.uncertainty_parameter_set_mismatch"
    )
    UNCERTAINTY_SHAPE: Final = "mechanism_composition.uncertainty_shape"
    UNITS_MISMATCH: Final = "mechanism_composition.units_mismatch"


class mechanism_discrepancy:  # noqa: N801
    DEGENERATE_COVARIANCE: Final = "mechanism_discrepancy.degenerate_covariance"
    DEPENDENCE_UNKNOWN: Final = "mechanism_discrepancy.dependence_unknown"
    DUPLICATE_NAME: Final = "mechanism_discrepancy.duplicate_name"
    INCOMPARABLE_MEASUREMENTS: Final = "mechanism_discrepancy.incomparable_measurements"
    INCONSISTENT_SUMMARY: Final = "mechanism_discrepancy.inconsistent_summary"
    INVALID_ALPHA: Final = "mechanism_discrepancy.invalid_alpha"
    INVALID_ARTIFACT: Final = "mechanism_discrepancy.invalid_artifact"
    INVALID_IDENTITY: Final = "mechanism_discrepancy.invalid_identity"
    INVALID_POWER: Final = "mechanism_discrepancy.invalid_power"
    INVALID_REQUEST: Final = "mechanism_discrepancy.invalid_request"
    NO_COMPARED_COEFFICIENTS: Final = "mechanism_discrepancy.no_compared_coefficients"
    NON_FINITE_VALUE: Final = "mechanism_discrepancy.non_finite_value"
    RANK_DEFICIENT_DESIGN: Final = "mechanism_discrepancy.rank_deficient_design"
    ROW_COUNT_MISMATCH: Final = "mechanism_discrepancy.row_count_mismatch"
    SAMPLE_TOO_SMALL: Final = "mechanism_discrepancy.sample_too_small"
    TOO_MANY_PARENTS: Final = "mechanism_discrepancy.too_many_parents"
    WRONG_CONTRACT: Final = "mechanism_discrepancy.wrong_contract"


class mechanism_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "mechanism_posterior.band_not_constructed"
    EMPTY_SUMMARY: Final = "mechanism_posterior.empty_summary"
    EXTERNAL_POWER_PRIOR: Final = "mechanism_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "mechanism_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "mechanism_posterior.non_positive_scale"


class mechanism_transport:  # noqa: N801
    DIAGRAM_SOURCE_MISMATCH: Final = "mechanism_transport.diagram_source_mismatch"
    DIAGRAM_TARGET_MISMATCH: Final = "mechanism_transport.diagram_target_mismatch"
    DUPLICATE_MECHANISM: Final = "mechanism_transport.duplicate_mechanism"
    FAMILY_UNSUPPORTED: Final = "mechanism_transport.family_unsupported"
    ORIGIN_NOT_FITTED: Final = "mechanism_transport.origin_not_fitted"
    SELECTION_ARROW_INTO_LATENT: Final = "mechanism_transport.selection_arrow_into_latent"
    TARGET_MECHANISM_NOT_NATIVE: Final = "mechanism_transport.target_mechanism_not_native"
    TRANSPORT_IDENTITY_MISMATCH: Final = "mechanism_transport.transport_identity_mismatch"


class mediation_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "mediation_posterior.band_not_constructed"
    EMPTY_DESIGN: Final = "mediation_posterior.empty_design"
    EXTERNAL_POWER_PRIOR: Final = "mediation_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "mediation_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "mediation_posterior.non_positive_scale"


class missing_data:  # noqa: N801
    COVARIATES_INCOMPLETE: Final = "missing_data.covariates_incomplete"
    MULTIPLE_IMPUTATION: Final = "missing_data.multiple_imputation"
    POLICY_NOT_APPLICABLE: Final = "missing_data.policy_not_applicable"
    POLICY_REQUIRED: Final = "missing_data.policy_required"


class mixed_search:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "mixed_search.bounds_exceeded"
    BUDGET: Final = "mixed_search.budget"
    INVALID_DERIVATION: Final = "mixed_search.invalid_derivation"
    INVALID_QUERY: Final = "mixed_search.invalid_query"
    MISSING_JOINT: Final = "mixed_search.missing_joint"
    NAMED_ROUTE: Final = "mixed_search.named_route"
    NOT_CERTIFIED: Final = "mixed_search.not_certified"
    POSTERIOR_AS_LAW: Final = "mixed_search.posterior_as_law"


class mnar_fixed_shift:  # noqa: N801
    CANCELLED: Final = "mnar_fixed_shift.cancelled"
    CONSTANT_OUTCOME: Final = "mnar_fixed_shift.constant_outcome"
    DEGENERATE_INTERVAL: Final = "mnar_fixed_shift.degenerate_interval"
    DUPLICATE_SUBJECT: Final = "mnar_fixed_shift.duplicate_subject"
    INSUFFICIENT_OBSERVATIONS: Final = "mnar_fixed_shift.insufficient_observations"
    INTERVAL_NOT_SUPPORTED: Final = "mnar_fixed_shift.interval_not_supported"
    INVALID_DECLARATION: Final = "mnar_fixed_shift.invalid_declaration"
    INVALID_OUTCOME: Final = "mnar_fixed_shift.invalid_outcome"
    INVALID_SHIFT: Final = "mnar_fixed_shift.invalid_shift"
    INVALID_SOURCE: Final = "mnar_fixed_shift.invalid_source"
    LOWERING: Final = "mnar_fixed_shift.lowering"
    PREPARATION: Final = "mnar_fixed_shift.preparation"
    PREPARATION_CANCELLED: Final = "mnar_fixed_shift.preparation_cancelled"
    PREPARATION_RESOURCE_LIMIT: Final = "mnar_fixed_shift.preparation_resource_limit"
    SOURCE_ADMISSION_CANCELLED: Final = "mnar_fixed_shift.source_admission_cancelled"
    SOURCE_ADMISSION_RESOURCE_LIMIT: Final = "mnar_fixed_shift.source_admission_resource_limit"


class mnar_sensitivity:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "mnar_sensitivity.bounds_exceeded"
    COMPOSITION_NOT_SUPPORTED: Final = "mnar_sensitivity.composition_not_supported"
    COVARIATES_NOT_USED: Final = "mnar_sensitivity.covariates_not_used"
    COVARIATES_REQUIRED: Final = "mnar_sensitivity.covariates_required"
    DESIGN_SINGULAR: Final = "mnar_sensitivity.design_singular"
    INSUFFICIENT_OBSERVED: Final = "mnar_sensitivity.insufficient_observed"
    INVALID_BOOTSTRAP: Final = "mnar_sensitivity.invalid_bootstrap"
    INVALID_DATA: Final = "mnar_sensitivity.invalid_data"
    INVALID_DECISION: Final = "mnar_sensitivity.invalid_decision"
    INVALID_LEVEL: Final = "mnar_sensitivity.invalid_level"
    INVALID_LOADING: Final = "mnar_sensitivity.invalid_loading"
    INVALID_TOLERANCE: Final = "mnar_sensitivity.invalid_tolerance"
    MECHANISM_NOT_DECLARED: Final = "mnar_sensitivity.mechanism_not_declared"
    RANGE_EXCLUDES_MAR: Final = "mnar_sensitivity.range_excludes_mar"
    UNBOUNDED_RANGE: Final = "mnar_sensitivity.unbounded_range"
    UNSUPPORTED_ESTIMATOR: Final = "mnar_sensitivity.unsupported_estimator"
    UNSUPPORTED_MECHANISM: Final = "mnar_sensitivity.unsupported_mechanism"
    UNSUPPORTED_QUERY: Final = "mnar_sensitivity.unsupported_query"


class msm_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "msm_posterior.band_not_constructed"
    EMPTY_LAW: Final = "msm_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "msm_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "msm_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "msm_posterior.non_positive_scale"


class msm_sensitivity:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "msm_sensitivity.bounds_exceeded"
    COMPOSITION_NOT_SUPPORTED: Final = "msm_sensitivity.composition_not_supported"
    INVALID_THRESHOLD: Final = "msm_sensitivity.invalid_threshold"
    INVALID_TOLERANCE: Final = "msm_sensitivity.invalid_tolerance"
    LAMBDA_BELOW_ONE: Final = "msm_sensitivity.lambda_below_one"
    LAMBDA_RANGE_EMPTY: Final = "msm_sensitivity.lambda_range_empty"
    NO_INTERVAL_METHOD: Final = "msm_sensitivity.no_interval_method"
    OUTCOME_LAW: Final = "msm_sensitivity.outcome_law"
    POSITIVITY: Final = "msm_sensitivity.positivity"
    STRATUM_MASS: Final = "msm_sensitivity.stratum_mass"


class multi_source_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "multi_source_posterior.band_not_constructed"
    EMPTY_LAW: Final = "multi_source_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "multi_source_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "multi_source_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "multi_source_posterior.non_positive_scale"


class multiple_imputation:  # noqa: N801
    ADMISSION_REFUSED: Final = "multiple_imputation.admission_refused"
    AMBIGUOUS_VARIABLE_DOMAIN: Final = "multiple_imputation.ambiguous_variable_domain"
    BINARY_UNDER_JOINT_NORMAL: Final = "multiple_imputation.binary_under_joint_normal"
    CANCELLED: Final = "multiple_imputation.cancelled"
    CLUSTERS_REQUIRED: Final = "multiple_imputation.clusters_required"
    COLUMN_LENGTH: Final = "multiple_imputation.column_length"
    CONDITIONAL_MODEL_NOT_FITTED: Final = "multiple_imputation.conditional_model_not_fitted"
    DEPENDENT_SERIES: Final = "multiple_imputation.dependent_series"
    DYNAMICS_JOINT_NORMAL: Final = "multiple_imputation.dynamics_joint_normal"
    DYNAMICS_LAGS: Final = "multiple_imputation.dynamics_lags"
    GRAPH_POSTERIOR_ATOMS: Final = "multiple_imputation.graph_posterior_atoms"
    GRAPH_POSTERIOR_FREQUENTIST: Final = "multiple_imputation.graph_posterior_frequentist"
    GRAPH_POSTERIOR_QUERY: Final = "multiple_imputation.graph_posterior_query"
    GRAPH_POSTERIOR_SERIES: Final = "multiple_imputation.graph_posterior_series"
    INDEX: Final = "multiple_imputation.index"
    INTERACTION_VARIABLE: Final = "multiple_imputation.interaction_variable"
    INTERACTIONS_JOINT_NORMAL: Final = "multiple_imputation.interactions_joint_normal"
    MIXED_FITS: Final = "multiple_imputation.mixed_fits"
    MULTI_HORIZON_POSTERIOR_NOT_JOINT: Final = (
        "multiple_imputation.multi_horizon_posterior_not_joint"
    )
    NON_FINITE: Final = "multiple_imputation.non_finite"
    NOT_AT_RANDOM: Final = "multiple_imputation.not_at_random"
    NOT_IDENTIFIED: Final = "multiple_imputation.not_identified"
    POOL_DF: Final = "multiple_imputation.pool_df"
    POOL_NON_FINITE: Final = "multiple_imputation.pool_non_finite"
    POOL_SHAPE: Final = "multiple_imputation.pool_shape"
    SAMPLING_DESIGN: Final = "multiple_imputation.sampling_design"
    SPEC_RANGE: Final = "multiple_imputation.spec_range"
    STACK_SHAPE: Final = "multiple_imputation.stack_shape"
    TOO_FEW_IMPUTATIONS: Final = "multiple_imputation.too_few_imputations"
    UNITS_LABELS: Final = "multiple_imputation.units_labels"
    UNITS_SHAPE: Final = "multiple_imputation.units_shape"
    VARIABLE_COUNT: Final = "multiple_imputation.variable_count"
    VARIABLE_DOMAIN: Final = "multiple_imputation.variable_domain"
    WITHIN_SINGULAR: Final = "multiple_imputation.within_singular"


class mz_transport:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "mz_transport.bounds_exceeded"
    BUDGET: Final = "mz_transport.budget"
    CHECKED_OBSTRUCTION: Final = "mz_transport.checked_obstruction"
    EMPIRICAL_COUNTS_REQUIRED: Final = "mz_transport.empirical_counts_required"
    FABRICATED_JOINT: Final = "mz_transport.fabricated_joint"
    FUNCTIONAL_GRAPH_MISMATCH: Final = "mz_transport.functional_graph_mismatch"
    INVALID_CATALOG: Final = "mz_transport.invalid_catalog"
    INVALID_DERIVATION: Final = "mz_transport.invalid_derivation"
    INVALID_OBSTRUCTION: Final = "mz_transport.invalid_obstruction"
    INVALID_QUERY: Final = "mz_transport.invalid_query"
    MISSING_JOINT_REGIME: Final = "mz_transport.missing_joint_regime"
    SEARCH_INCOMPLETE: Final = "mz_transport.search_incomplete"


class native_attempt:  # noqa: N801
    INVALID_CALLABLE: Final = "native_attempt.invalid_callable"


class native_claims:  # noqa: N801
    CALIBRATION_NOT_SUPPORTED: Final = "native_claims.calibration_not_supported"
    CONTRACT_MISMATCH: Final = "native_claims.contract_mismatch"
    GRID_MISMATCH: Final = "native_claims.grid_mismatch"
    INVALID_CONTEXT: Final = "native_claims.invalid_context"
    JOINT_LAW_INVALID: Final = "native_claims.joint_law_invalid"
    NATIVE_STATE_UNAVAILABLE: Final = "native_claims.native_state_unavailable"
    NO_SOURCE_REQUIREMENT: Final = "native_claims.no_source_requirement"
    NO_SUPPORTED_COORDINATE: Final = "native_claims.no_supported_coordinate"
    POINT_SUPPORT_MISSING: Final = "native_claims.point_support_missing"
    PROGRAM_MISMATCH: Final = "native_claims.program_mismatch"
    PROJECTION_MISMATCH: Final = "native_claims.projection_mismatch"
    PROVENANCE_MISMATCH: Final = "native_claims.provenance_mismatch"
    RESPONSE_NOT_POINT_IDENTIFIED: Final = "native_claims.response_not_point_identified"
    SOURCE_NOT_SUPPLIED: Final = "native_claims.source_not_supplied"
    UNITS_CONFLICT: Final = "native_claims.units_conflict"
    UNKNOWN_VARIABLE: Final = "native_claims.unknown_variable"
    UNSUPPORTED_ESTIMAND: Final = "native_claims.unsupported_estimand"


class native_distribution:  # noqa: N801
    AUTHORITY_REQUIRED: Final = "native_distribution.authority_required"


class nested_markov:  # noqa: N801
    BAYESIAN: Final = "nested_markov.bayesian"
    BAYESIAN_ARTIFACT: Final = "nested_markov.bayesian_artifact"
    BAYESIAN_ARTIFACT_LIMITS: Final = "nested_markov.bayesian_artifact_limits"
    BAYESIAN_IDENTITY_MISMATCH: Final = "nested_markov.bayesian_identity_mismatch"
    BAYESIAN_INVALID_OPTIONS: Final = "nested_markov.bayesian_invalid_options"
    BAYESIAN_INVALID_PRIOR: Final = "nested_markov.bayesian_invalid_prior"
    BAYESIAN_SAMPLER_SCOPE: Final = "nested_markov.bayesian_sampler_scope"
    CANCELLED: Final = "nested_markov.cancelled"
    CONSTRAINT_VIOLATED: Final = "nested_markov.constraint_violated"
    FISHER_ARTIFACT_LIMITS: Final = "nested_markov.fisher_artifact_limits"
    FISHER_IDENTITY_MISMATCH: Final = "nested_markov.fisher_identity_mismatch"
    FISHER_INVALID_COVARIANCE: Final = "nested_markov.fisher_invalid_covariance"
    FISHER_INVALID_LEVEL: Final = "nested_markov.fisher_invalid_level"
    FISHER_INVALID_OPTIONS: Final = "nested_markov.fisher_invalid_options"
    FISHER_SAMPLING_DESIGN: Final = "nested_markov.fisher_sampling_design"
    FISHER_SINGULAR_INFORMATION: Final = "nested_markov.fisher_singular_information"
    FIT_NOT_CONVERGED: Final = "nested_markov.fit_not_converged"
    FITTED_BOUNDARY: Final = "nested_markov.fitted_boundary"
    IDENTIFICATION_UNAVAILABLE: Final = "nested_markov.identification_unavailable"
    INVALID_COUNTS: Final = "nested_markov.invalid_counts"
    INVALID_INPUT: Final = "nested_markov.invalid_input"
    INVALID_OPTIONS: Final = "nested_markov.invalid_options"
    ITERATION_BOUND: Final = "nested_markov.iteration_bound"
    MEASURED_SCOPE: Final = "nested_markov.measured_scope"
    NATIVE_HOOK_UNAVAILABLE: Final = "nested_markov.native_hook_unavailable"
    NOT_NORMALIZED: Final = "nested_markov.not_normalized"
    OUTSIDE_BINARY_PILOT: Final = "nested_markov.outside_binary_pilot"
    POSITIVITY: Final = "nested_markov.positivity"
    PRODUCER_MISSING: Final = "nested_markov.producer_missing"


class nonlinear_mediation:  # noqa: N801
    CANCELLED: Final = "nonlinear_mediation.cancelled"
    CONFOUNDING: Final = "nonlinear_mediation.confounding"
    CROSS_WORLD_INDEPENDENCE_NOT_DECLARED: Final = (
        "nonlinear_mediation.cross_world_independence_not_declared"
    )
    DATA_REQUIRED: Final = "nonlinear_mediation.data_required"
    DEGENERATE_MEDIATOR: Final = "nonlinear_mediation.degenerate_mediator"
    DEGENERATE_MEDIATOR_RESIDUAL: Final = "nonlinear_mediation.degenerate_mediator_residual"
    DEGENERATE_OUTCOME: Final = "nonlinear_mediation.degenerate_outcome"
    DEGREE_ABOVE_LIMIT: Final = "nonlinear_mediation.degree_above_limit"
    EXPORT_BUDGET_EXCEEDED: Final = "nonlinear_mediation.export_budget_exceeded"
    IDENTITY_VIOLATION: Final = "nonlinear_mediation.identity_violation"
    INTEGRATION_ERROR: Final = "nonlinear_mediation.integration_error"
    INTERVENTIONAL_EFFECTS_NOT_SUPPORTED: Final = (
        "nonlinear_mediation.interventional_effects_not_supported"
    )
    INVALID_DEGREE: Final = "nonlinear_mediation.invalid_degree"
    INVALID_NODES: Final = "nonlinear_mediation.invalid_nodes"
    INVALID_REQUEST: Final = "nonlinear_mediation.invalid_request"
    INVALID_SUPPORT_BOUND: Final = "nonlinear_mediation.invalid_support_bound"
    INVALID_TOLERANCE: Final = "nonlinear_mediation.invalid_tolerance"
    LENGTH_MISMATCH: Final = "nonlinear_mediation.length_mismatch"
    LIMITS_EXCEEDED: Final = "nonlinear_mediation.limits_exceeded"
    MEMORY_BUDGET_EXCEEDED: Final = "nonlinear_mediation.memory_budget_exceeded"
    NODES_ABOVE_LIMIT: Final = "nonlinear_mediation.nodes_above_limit"
    NON_FINITE_INPUT: Final = "nonlinear_mediation.non_finite_input"
    NON_FINITE_OPTION: Final = "nonlinear_mediation.non_finite_option"
    OVERLAP: Final = "nonlinear_mediation.overlap"
    QUADRATURE_NODES_NOT_CONVERGED: Final = "nonlinear_mediation.quadrature_nodes_not_converged"
    QUADRATURE_WEIGHTS_INVALID: Final = "nonlinear_mediation.quadrature_weights_invalid"
    RANK_DEFICIENT: Final = "nonlinear_mediation.rank_deficient"
    REPORT_REPLAY_MISMATCH: Final = "nonlinear_mediation.report_replay_mismatch"
    TOO_FEW_ROWS: Final = "nonlinear_mediation.too_few_rows"
    TREATMENT_INDUCED_CONFOUNDING: Final = "nonlinear_mediation.treatment_induced_confounding"
    TREATMENT_NOT_BINARY: Final = "nonlinear_mediation.treatment_not_binary"
    WRONG_CONTRACT: Final = "nonlinear_mediation.wrong_contract"


class observation_query:  # noqa: N801
    ARM_MISMATCH: Final = "observation_query.arm_mismatch"
    ARTIFACT_INVALID: Final = "observation_query.artifact_invalid"
    ARTIFACT_REFUSED: Final = "observation_query.artifact_refused"
    BLANK_FIELD: Final = "observation_query.blank_field"
    CANCELLED: Final = "observation_query.cancelled"
    CONSTRUCT_MISMATCH: Final = "observation_query.construct_mismatch"
    CONSTRUCT_OBSERVED_DIRECTLY: Final = "observation_query.construct_observed_directly"
    COUNT_FITTED_BOUNDS_EXCEEDED: Final = "observation_query.count_fitted_bounds_exceeded"
    COUNT_FITTED_REFUSED: Final = "observation_query.count_fitted_refused"
    DECLARED_MODEL_HAS_NO_SAMPLING_INTERVAL: Final = (
        "observation_query.declared_model_has_no_sampling_interval"
    )
    DIFFERENTIAL_ERROR_UNSUPPORTED: Final = "observation_query.differential_error_unsupported"
    EVIDENCE_COVARIANCE_SINGULAR: Final = "observation_query.evidence_covariance_singular"
    INDICATOR_KIND: Final = "observation_query.indicator_kind"
    INDICATOR_MISMATCH: Final = "observation_query.indicator_mismatch"
    INTERVENTION_IN_CONDITIONING_SLOT: Final = "observation_query.intervention_in_conditioning_slot"
    INVALID_DECLARATION: Final = "observation_query.invalid_declaration"
    INVALID_INTERVAL: Final = "observation_query.invalid_interval"
    INVALID_REQUEST: Final = "observation_query.invalid_request"
    LEVEL_UNREACHABLE: Final = "observation_query.level_unreachable"
    LEVELS_FOR_CONTINUOUS_MODEL: Final = "observation_query.levels_for_continuous_model"
    MEASURED_AS_CONSTRUCT: Final = "observation_query.measured_as_construct"
    MEASURED_BACKGROUND: Final = "observation_query.measured_background"
    MEASURED_SIGNAL_NOT_REDRAWN: Final = "observation_query.measured_signal_not_redrawn"
    MEASUREMENT_ROUTE: Final = "observation_query.measurement_route"
    MIXED_CONDITIONING_AND_INTERVENTION: Final = (
        "observation_query.mixed_conditioning_and_intervention"
    )
    MULTIPLE_MEASURED_SIGNALS: Final = "observation_query.multiple_measured_signals"
    NO_CALIBRATED_PROBABILITY: Final = "observation_query.no_calibrated_probability"
    NO_PARAMETER_DRAWS: Final = "observation_query.no_parameter_draws"
    OUTCOME_ASSIGNED: Final = "observation_query.outcome_assigned"
    POPULATION_MISMATCH: Final = "observation_query.population_mismatch"
    POSITIVE_LEVEL: Final = "observation_query.positive_level"
    REAL_VALUES_FOR_FINITE_MODEL: Final = "observation_query.real_values_for_finite_model"
    REGIME_MISMATCH: Final = "observation_query.regime_mismatch"
    REPEATED_SLOT: Final = "observation_query.repeated_slot"
    REQUEST_UNSUPPORTED: Final = "observation_query.request_unsupported"
    UNKNOWN_LEVEL: Final = "observation_query.unknown_level"
    UNKNOWN_QUANTITY: Final = "observation_query.unknown_quantity"
    UNKNOWN_VARIABLE: Final = "observation_query.unknown_variable"
    VALUE_NOT_FINITE: Final = "observation_query.value_not_finite"
    ZERO_PROBABILITY_EVIDENCE: Final = "observation_query.zero_probability_evidence"


class paradigm_parity:  # noqa: N801
    PARADIGM_NOT_SUPPORTED: Final = "paradigm_parity.paradigm_not_supported"


class parent_set_effect:  # noqa: N801
    CONFLICT_RESOLUTION_BUDGET: Final = "parent_set_effect.conflict_resolution_budget"
    DISCOVERY_FAILED: Final = "parent_set_effect.discovery_failed"
    KNOWLEDGE_INVALID: Final = "parent_set_effect.knowledge_invalid"
    LOG_LINEAR_INPUT_NOT_POSITIVE: Final = "parent_set_effect.log_linear_input_not_positive"
    OUTCOME_PRECEDES_TREATMENT: Final = "parent_set_effect.outcome_precedes_treatment"
    PARENT_LIMIT_BOUND: Final = "parent_set_effect.parent_limit_bound"
    SEARCH_NOT_FINISHED: Final = "parent_set_effect.search_not_finished"
    SOURCE_INVALID: Final = "parent_set_effect.source_invalid"
    SOURCE_NOT_CANONICAL: Final = "parent_set_effect.source_not_canonical"
    STRUCTURE_CLASS: Final = "parent_set_effect.structure_class"
    TRANSFORMATION_UNDECLARED: Final = "parent_set_effect.transformation_undeclared"
    TREATMENT_ORIENTATION_DEPENDS_ON_CONFLICT: Final = (
        "parent_set_effect.treatment_orientation_depends_on_conflict"
    )
    TREATMENT_PARENTS_AMBIGUOUS: Final = "parent_set_effect.treatment_parents_ambiguous"


class penalized_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "penalized_posterior.band_not_constructed"
    EMPTY_SCORES: Final = "penalized_posterior.empty_scores"
    EXTERNAL_POWER_PRIOR: Final = "penalized_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "penalized_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "penalized_posterior.non_positive_scale"


class penalized_propensity:  # noqa: N801
    CANCELLED: Final = "penalized_propensity.cancelled"
    EFFECT_POSTERIOR: Final = "penalized_propensity.effect_posterior"
    FALLBACK_NOT_SUPPORTED: Final = "penalized_propensity.fallback_not_supported"
    FALLBACK_WITH_PENALTY: Final = "penalized_propensity.fallback_with_penalty"
    GRID_ALLOCATION: Final = "penalized_propensity.grid_allocation"
    INVALID_PENALTY: Final = "penalized_propensity.invalid_penalty"
    SCOPE: Final = "penalized_propensity.scope"
    SELECTION_NOT_SUPPORTED: Final = "penalized_propensity.selection_not_supported"


class policy_value:  # noqa: N801
    ACTION_NOT_POPULATED: Final = "policy_value.action_not_populated"
    ACTION_OUTSIDE_SUPPORT: Final = "policy_value.action_outside_support"
    ACTION_PROPENSITY_BELOW_FLOOR: Final = "policy_value.action_propensity_below_floor"
    CANCELLED: Final = "policy_value.cancelled"
    EVALUATION_FRACTION_MISSING: Final = "policy_value.evaluation_fraction_missing"
    EVALUATION_MISSING: Final = "policy_value.evaluation_missing"
    EVALUATION_UNKNOWN: Final = "policy_value.evaluation_unknown"
    INVALID_DATA: Final = "policy_value.invalid_data"
    INVALID_EVALUATION_FRACTION: Final = "policy_value.invalid_evaluation_fraction"
    INVALID_FOLDS: Final = "policy_value.invalid_folds"
    INVALID_KNOWN_PROPENSITY: Final = "policy_value.invalid_known_propensity"
    INVALID_PROPENSITY_FLOOR: Final = "policy_value.invalid_propensity_floor"
    INVALID_REQUEST: Final = "policy_value.invalid_request"
    INVALID_SUPPORT: Final = "policy_value.invalid_support"
    OBJECTIVE_MISSING: Final = "policy_value.objective_missing"
    OBJECTIVE_UNKNOWN: Final = "policy_value.objective_unknown"
    POLICY_ACTIONS_MISSING: Final = "policy_value.policy_actions_missing"
    POLICY_LEARNED_ON_EVALUATION_ROWS: Final = "policy_value.policy_learned_on_evaluation_rows"
    POLICY_ORIGIN_MISSING: Final = "policy_value.policy_origin_missing"
    POLICY_ORIGIN_UNKNOWN: Final = "policy_value.policy_origin_unknown"
    REFERENCE_OUTSIDE_SUPPORT: Final = "policy_value.reference_outside_support"
    SPARSE_ACTION: Final = "policy_value.sparse_action"
    SUPPORT_MISSING: Final = "policy_value.support_missing"
    TOO_FEW_EVALUATION_ROWS: Final = "policy_value.too_few_evaluation_rows"
    TOO_FEW_ROWS: Final = "policy_value.too_few_rows"
    TRAINING_SPLIT_MISSING_ACTION: Final = "policy_value.training_split_missing_action"
    UNDECLARED_LEVEL: Final = "policy_value.undeclared_level"


class predictive_assessment:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "predictive_assessment.bounds_exceeded"
    CANCELLED: Final = "predictive_assessment.cancelled"
    DESIGN_INVALID: Final = "predictive_assessment.design_invalid"
    DIAGNOSTIC_NOT_DEFINED: Final = "predictive_assessment.diagnostic_not_defined"
    HORIZON_MISMATCH: Final = "predictive_assessment.horizon_mismatch"
    IID_NOT_SUPPORTED_FOR_DEPENDENT_DESIGN: Final = (
        "predictive_assessment.iid_not_supported_for_dependent_design"
    )
    IN_SAMPLE: Final = "predictive_assessment.in_sample"
    INTERVAL_DEPENDENCE_UNDECLARED: Final = "predictive_assessment.interval_dependence_undeclared"
    INTERVAL_LAW_FAMILY_MIXED: Final = "predictive_assessment.interval_law_family_mixed"
    INTERVAL_NOT_SUPPORTED: Final = "predictive_assessment.interval_not_supported"
    INTERVAL_REQUEST_INVALID: Final = "predictive_assessment.interval_request_invalid"
    INVALID_ARTIFACT: Final = "predictive_assessment.invalid_artifact"
    INVALID_LAW: Final = "predictive_assessment.invalid_law"
    INVALID_OBSERVATIONS: Final = "predictive_assessment.invalid_observations"
    LAW_CALLBACK_NOT_BUILT: Final = "predictive_assessment.law_callback_not_built"
    LOG_SCORE_INFINITE: Final = "predictive_assessment.log_score_infinite"
    NOT_A_CAUSAL_CLAIM: Final = "predictive_assessment.not_a_causal_claim"
    OUTCOME_KIND: Final = "predictive_assessment.outcome_kind"
    OVERLAP_UNKNOWN: Final = "predictive_assessment.overlap_unknown"
    PROVIDER_IDENTITY_INCOMPLETE: Final = "predictive_assessment.provider_identity_incomplete"
    REPORT_REPLAY_MISMATCH: Final = "predictive_assessment.report_replay_mismatch"
    SCORE_NOT_DEFINED: Final = "predictive_assessment.score_not_defined"
    SCORE_NOT_PROBABILITY: Final = "predictive_assessment.score_not_probability"
    SERIAL_LAG_BELOW_HORIZON: Final = "predictive_assessment.serial_lag_below_horizon"
    TARGET_MISMATCH: Final = "predictive_assessment.target_mismatch"
    TOO_FEW_CLUSTERS: Final = "predictive_assessment.too_few_clusters"
    UNKNOWN_SOURCE: Final = "predictive_assessment.unknown_source"
    WRONG_CONTRACT: Final = "predictive_assessment.wrong_contract"


class prior_signal:  # noqa: N801
    ADAPTED_PRIOR_NOT_DRAWS: Final = "prior_signal.adapted_prior_not_draws"
    ARTIFACT_NOT_IN_CATALOG: Final = "prior_signal.artifact_not_in_catalog"
    COEFFICIENT_PRIOR_NOT_SIGNAL: Final = "prior_signal.coefficient_prior_not_signal"
    DUPLICATE_SOURCE: Final = "prior_signal.duplicate_source"
    INCOMPATIBLE_ARTIFACT: Final = "prior_signal.incompatible_artifact"
    INCOMPLETE_POSTERIOR: Final = "prior_signal.incomplete_posterior"
    INVALID_PARAMETER: Final = "prior_signal.invalid_parameter"
    INVALID_REQUEST: Final = "prior_signal.invalid_request"
    LINEAGE_MISSING: Final = "prior_signal.lineage_missing"
    NAMED_STATE_DRAWS_REQUIRED: Final = "prior_signal.named_state_draws_required"
    NO_SOURCES: Final = "prior_signal.no_sources"
    NONFINITE_MAPPED_PRIOR: Final = "prior_signal.nonfinite_mapped_prior"
    OBSERVATION_LINEAGE_MISSING: Final = "prior_signal.observation_lineage_missing"
    OBSERVATION_OVERLAP_UNKNOWN: Final = "prior_signal.observation_overlap_unknown"
    OBSERVATION_REUSE: Final = "prior_signal.observation_reuse"
    ORIGINAL_POSTERIOR_UNIDENTIFIED: Final = "prior_signal.original_posterior_unidentified"
    POSTERIOR_DRAWS_REQUIRED: Final = "prior_signal.posterior_draws_required"
    PRIOR_IDENTITY_MISMATCH: Final = "prior_signal.prior_identity_mismatch"
    PRIOR_PROJECTION_MISMATCH: Final = "prior_signal.prior_projection_mismatch"
    PRIOR_SOURCES_OVERLAP: Final = "prior_signal.prior_sources_overlap"
    RESOLUTION_INVALID: Final = "prior_signal.resolution_invalid"
    SIGNAL_FAMILY_MISMATCH: Final = "prior_signal.signal_family_mismatch"
    SOURCE_NOT_IN_CATALOG_OR_DUPLICATE: Final = "prior_signal.source_not_in_catalog_or_duplicate"
    SOURCE_POPULATION_MISMATCH: Final = "prior_signal.source_population_mismatch"
    STATE_QUANTITY_MISMATCH: Final = "prior_signal.state_quantity_mismatch"
    TARGET_POPULATION_MISMATCH: Final = "prior_signal.target_population_mismatch"
    TRANSPORT_POLICY_REQUIRED: Final = "prior_signal.transport_policy_required"
    UNLOCK_LIST_NOT_SIGNAL: Final = "prior_signal.unlock_list_not_signal"
    UNSUPPORTED_SIGNAL_PROVIDER: Final = "prior_signal.unsupported_signal_provider"


class program_binding:  # noqa: N801
    CONTRACT_IDENTITY_MISMATCH: Final = "program_binding.contract_identity_mismatch"
    DOSE_GRID_CHANGED: Final = "program_binding.dose_grid_changed"
    GRAPH_ONLY_IDENTITY: Final = "program_binding.graph_only_identity"
    INVALID_BINDING: Final = "program_binding.invalid_binding"
    MALFORMED_QUANTITY: Final = "program_binding.malformed_quantity"
    NO_IDENTIFIED_REQUEST: Final = "program_binding.no_identified_request"
    POPULATION_MISMATCH: Final = "program_binding.population_mismatch"
    QUANTITIES_OVERRIDE_MISMATCH: Final = "program_binding.quantities_override_mismatch"
    TREATMENT_OUTCOME_SUBSTITUTION: Final = "program_binding.treatment_outcome_substitution"


class proposal_arrival:  # noqa: N801
    ASSIGNMENT_MISMATCH: Final = "proposal_arrival.assignment_mismatch"
    BASE_CATALOG_CHANGED: Final = "proposal_arrival.base_catalog_changed"
    BASE_CATALOG_MISSING: Final = "proposal_arrival.base_catalog_missing"
    BASE_PROVIDER_MISMATCH: Final = "proposal_arrival.base_provider_mismatch"
    BINDING_MISSING: Final = "proposal_arrival.binding_missing"
    BOUNDS_EXCEEDED: Final = "proposal_arrival.bounds_exceeded"
    CANCELLED: Final = "proposal_arrival.cancelled"
    CANDIDATE_MISSING: Final = "proposal_arrival.candidate_missing"
    DUPLICATE_ORIGINAL_ARTIFACTS: Final = "proposal_arrival.duplicate_original_artifacts"
    EXPECTED_PROPOSAL_MISMATCH: Final = "proposal_arrival.expected_proposal_mismatch"
    FAMILY_NOT_SUPPORTED: Final = "proposal_arrival.family_not_supported"
    IDENTITY_MISMATCH: Final = "proposal_arrival.identity_mismatch"
    INDEPENDENT_JOINT_STUDY_REQUIRED: Final = "proposal_arrival.independent_joint_study_required"
    INVALID_REQUEST: Final = "proposal_arrival.invalid_request"
    ORIGINAL_ARTIFACTS_REQUIRED: Final = "proposal_arrival.original_artifacts_required"
    ORIGINAL_CONSUMER_REFUSED: Final = "proposal_arrival.original_consumer_refused"
    PROOF_NOT_BOUND: Final = "proposal_arrival.proof_not_bound"
    PROVIDER_BINDING_MISMATCH: Final = "proposal_arrival.provider_binding_mismatch"
    RAW_COUNTS_REQUIRED: Final = "proposal_arrival.raw_counts_required"
    REPLAY_MISMATCH: Final = "proposal_arrival.replay_mismatch"
    SAMPLE_SIZE_MISMATCH: Final = "proposal_arrival.sample_size_mismatch"
    STRUCTURAL_ARRIVAL_REFUSED: Final = "proposal_arrival.structural_arrival_refused"


class proposal_receipt:  # noqa: N801
    ARRIVAL_CANCELLED: Final = "proposal_receipt.arrival_cancelled"
    ARRIVAL_EMPTY: Final = "proposal_receipt.arrival_empty"
    ARRIVAL_FAMILY_MISMATCH: Final = "proposal_receipt.arrival_family_mismatch"
    ARRIVAL_INVALID: Final = "proposal_receipt.arrival_invalid"
    ARRIVAL_SAMPLE_MISMATCH: Final = "proposal_receipt.arrival_sample_mismatch"
    BASE_FAILURE_MISMATCH: Final = "proposal_receipt.base_failure_mismatch"
    BUNDLE_IDENTITY_MISMATCH: Final = "proposal_receipt.bundle_identity_mismatch"
    BUNDLE_INCONSISTENT: Final = "proposal_receipt.bundle_inconsistent"
    CANDIDATE_MISMATCH: Final = "proposal_receipt.candidate_mismatch"
    CANDIDATE_NOT_IN_BUNDLE: Final = "proposal_receipt.candidate_not_in_bundle"
    CANDIDATE_NOT_IN_RANKING: Final = "proposal_receipt.candidate_not_in_ranking"
    CANDIDATE_NOT_IN_REPAIR: Final = "proposal_receipt.candidate_not_in_repair"
    CANDIDATE_NOT_PROPOSED: Final = "proposal_receipt.candidate_not_proposed"
    CANDIDATE_SET_MISMATCH: Final = "proposal_receipt.candidate_set_mismatch"
    CONTRACT_MISMATCH: Final = "proposal_receipt.contract_mismatch"
    CONTRACT_REQUIRED: Final = "proposal_receipt.contract_required"
    COST_MISMATCH: Final = "proposal_receipt.cost_mismatch"
    DELTA_MISMATCH: Final = "proposal_receipt.delta_mismatch"
    DERIVATION_MISMATCH: Final = "proposal_receipt.derivation_mismatch"
    DUPLICATE_CANDIDATE: Final = "proposal_receipt.duplicate_candidate"
    EMPTY_BUNDLE: Final = "proposal_receipt.empty_bundle"
    HYPOTHETICAL_NOT_EVIDENCE: Final = "proposal_receipt.hypothetical_not_evidence"
    IDENTITY_MISMATCH: Final = "proposal_receipt.identity_mismatch"
    INVALID_DELTA: Final = "proposal_receipt.invalid_delta"
    LINEAGE_MISMATCH: Final = "proposal_receipt.lineage_mismatch"
    NON_CANONICAL_ORDER: Final = "proposal_receipt.non_canonical_order"
    OBLIGATION_MISMATCH: Final = "proposal_receipt.obligation_mismatch"
    RANKING_DIGEST_MISMATCH: Final = "proposal_receipt.ranking_digest_mismatch"
    REPAIR_DIGEST_MISMATCH: Final = "proposal_receipt.repair_digest_mismatch"
    SIGNAL_MISMATCH: Final = "proposal_receipt.signal_mismatch"
    SOURCE_DIGEST_MISMATCH: Final = "proposal_receipt.source_digest_mismatch"
    VALUE_MISMATCH: Final = "proposal_receipt.value_mismatch"


class provider_capability_negotiation:  # noqa: N801
    CAPABILITY_MISSING: Final = "provider_capability_negotiation.capability_missing"
    REQUEST_MISMATCH: Final = "provider_capability_negotiation.request_mismatch"


class provider_transport:  # noqa: N801
    ARTIFACT_INVALID: Final = "provider_transport.artifact_invalid"
    ASSUMPTIONS_MISSING: Final = "provider_transport.assumptions_missing"
    ASSUMPTIONS_NOT_CARRIED: Final = "provider_transport.assumptions_not_carried"
    BLANK_FIELD: Final = "provider_transport.blank_field"
    CLAIM_NOT_IN_TARGET: Final = "provider_transport.claim_not_in_target"
    EVIDENCE_MISSING: Final = "provider_transport.evidence_missing"
    IDENTITY_MISMATCH: Final = "provider_transport.identity_mismatch"
    INPUT_NOT_EXTERNAL: Final = "provider_transport.input_not_external"
    NATIVE_TARGET_REQUIRED: Final = "provider_transport.native_target_required"
    NATIVE_TRUST_CLAIMED: Final = "provider_transport.native_trust_claimed"
    OBJECT_INVALID: Final = "provider_transport.object_invalid"
    OBJECT_UNSUPPORTED: Final = "provider_transport.object_unsupported"
    POPULATION_MISSING: Final = "provider_transport.population_missing"
    POPULATION_NOT_A_SOURCE: Final = "provider_transport.population_not_a_source"
    QUANTITY_INVALID: Final = "provider_transport.quantity_invalid"
    QUANTITY_NOT_DECLARED: Final = "provider_transport.quantity_not_declared"
    RECEIPT_DOES_NOT_COVER: Final = "provider_transport.receipt_does_not_cover"
    REGIME_MISSING: Final = "provider_transport.regime_missing"
    TRANSPORT_IDENTITY_INCOMPLETE: Final = "provider_transport.transport_identity_incomplete"
    TRUST_INCOMPLETE: Final = "provider_transport.trust_incomplete"
    VERIFICATION_REQUIRED: Final = "provider_transport.verification_required"


class quadratic_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "quadratic_posterior.band_not_constructed"
    EXTERNAL_POWER_PRIOR: Final = "quadratic_posterior.external_power_prior"
    INSUFFICIENT_WEIGHT: Final = "quadratic_posterior.insufficient_weight"
    MAPPED_PRIOR: Final = "quadratic_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "quadratic_posterior.non_positive_scale"


class rank_drop_estimate:  # noqa: N801
    ESTIMATOR_NOT_SUPPORTED: Final = "rank_drop_estimate.estimator_not_supported"
    PROPENSITY_SEPARATES: Final = "rank_drop_estimate.propensity_separates"
    SHAPE_NOT_SUPPORTED: Final = "rank_drop_estimate.shape_not_supported"
    SPAN_NOT_PRESERVED: Final = "rank_drop_estimate.span_not_preserved"


class recalc:  # noqa: N801
    ADJUSTED_ARM_NOT_POPULATED: Final = "recalc.adjusted_arm_not_populated"
    ADJUSTED_DESIGN_RANK_DEFICIENT: Final = "recalc.adjusted_design_rank_deficient"
    ADJUSTED_FIT_REFUSED: Final = "recalc.adjusted_fit_refused"
    ADJUSTED_GLM_BOUNDARY_SATURATED: Final = "recalc.adjusted_glm_boundary_saturated"
    ADJUSTED_GLM_NOT_CONVERGED: Final = "recalc.adjusted_glm_not_converged"
    ADJUSTED_GLM_SEPARATED: Final = "recalc.adjusted_glm_separated"
    ADJUSTED_INVALID_MODEL: Final = "recalc.adjusted_invalid_model"
    ADJUSTED_MODEL_DATA_INVALID: Final = "recalc.adjusted_model_data_invalid"
    ADJUSTED_PREDICTION_OUT_OF_SUPPORT: Final = "recalc.adjusted_prediction_out_of_support"
    ADJUSTED_PREDICTION_SCHEMA_MISMATCH: Final = "recalc.adjusted_prediction_schema_mismatch"
    ADJUSTED_RECEIPT_INVALID: Final = "recalc.adjusted_receipt_invalid"
    ADJUSTMENT_NOT_IDENTIFIED: Final = "recalc.adjustment_not_identified"
    AIPW_REQUIRES_OVERLAP_DIAGNOSTICS: Final = "recalc.aipw_requires_overlap_diagnostics"
    BAYESIAN_BASIS_PRIOR_UNSUPPORTED: Final = "recalc.bayesian_basis_prior_unsupported"
    BAYESIAN_EFFECT_UNAVAILABLE: Final = "recalc.bayesian_effect_unavailable"
    BAYESIAN_EXTERNAL_COMPOSE_UNSUPPORTED: Final = "recalc.bayesian_external_compose_unsupported"
    BAYESIAN_FIT_REFUSED: Final = "recalc.bayesian_fit_refused"
    BAYESIAN_INFERENCE_UNSUPPORTED: Final = "recalc.bayesian_inference_unsupported"
    BAYESIAN_INVALID_DATA: Final = "recalc.bayesian_invalid_data"
    BAYESIAN_INVALID_GRAPH: Final = "recalc.bayesian_invalid_graph"
    BAYESIAN_INVALID_INFERENCE: Final = "recalc.bayesian_invalid_inference"
    BAYESIAN_INVALID_QUERY: Final = "recalc.bayesian_invalid_query"
    BAYESIAN_INVALID_SUMMARY: Final = "recalc.bayesian_invalid_summary"
    BAYESIAN_PRIOR_INVALID: Final = "recalc.bayesian_prior_invalid"
    BAYESIAN_PRIOR_LIKELIHOOD_DOUBLE_USE: Final = "recalc.bayesian_prior_likelihood_double_use"
    BAYESIAN_PRIOR_SOURCE_CONFLICT: Final = "recalc.bayesian_prior_source_conflict"
    BAYESIAN_PRIOR_SOURCE_UNSUPPORTED: Final = "recalc.bayesian_prior_source_unsupported"
    BAYESIAN_PRIOR_SOURCE_UNVERIFIED: Final = "recalc.bayesian_prior_source_unverified"
    BAYESIAN_WORKSPACE_LIMIT: Final = "recalc.bayesian_workspace_limit"
    BLOCKED_BY_REFUSED_DEPENDENCY: Final = "recalc.blocked_by_refused_dependency"
    BOOTSTRAP_REPLICATES_UNSUPPORTED: Final = "recalc.bootstrap_replicates_unsupported"
    CANCELLED: Final = "recalc.cancelled"
    CAPABILITY_ADAPTER_UNAVAILABLE: Final = "recalc.capability_adapter_unavailable"
    CATEGORICAL_ROLE_MISMATCH: Final = "recalc.categorical_role_mismatch"
    CONTRAST_OUT_OF_SUPPORT: Final = "recalc.contrast_out_of_support"
    CONTRAST_UNAVAILABLE: Final = "recalc.contrast_unavailable"
    DECLARED_SIZE_MISMATCH: Final = "recalc.declared_size_mismatch"
    DESIGN_ARTIFACT_MISMATCH: Final = "recalc.design_artifact_mismatch"
    DESIGN_ARTIFACT_UNVERIFIED: Final = "recalc.design_artifact_unverified"
    DESIGN_DATA_UNAVAILABLE: Final = "recalc.design_data_unavailable"
    DESIGN_EFFECT_UNAVAILABLE: Final = "recalc.design_effect_unavailable"
    DESIGN_FIT_REFUSED: Final = "recalc.design_fit_refused"
    DESIGN_MODEL_DATA_INVALID: Final = "recalc.design_model_data_invalid"
    DESIGN_NOT_CHECKED: Final = "recalc.design_not_checked"
    DESIGN_NOT_IDENTIFIED: Final = "recalc.design_not_identified"
    DESIGN_POINT_UNAVAILABLE: Final = "recalc.design_point_unavailable"
    DESIGN_RANK_DEFICIENT: Final = "recalc.design_rank_deficient"
    DESIGN_RECEIPT_INVALID: Final = "recalc.design_receipt_invalid"
    DR_ARM_NOT_POPULATED: Final = "recalc.dr_arm_not_populated"
    DR_DESIGN_RANK_DEFICIENT: Final = "recalc.dr_design_rank_deficient"
    DR_EFFECT_UNAVAILABLE: Final = "recalc.dr_effect_unavailable"
    DR_FIT_COUNTER_MISMATCH: Final = "recalc.dr_fit_counter_mismatch"
    DR_FIT_REFUSED: Final = "recalc.dr_fit_refused"
    DR_GLM_BOUNDARY_SATURATED: Final = "recalc.dr_glm_boundary_saturated"
    DR_GLM_NOT_CONVERGED: Final = "recalc.dr_glm_not_converged"
    DR_GLM_SEPARATED: Final = "recalc.dr_glm_separated"
    DR_INVALID_FOLDS: Final = "recalc.dr_invalid_folds"
    DR_INVALID_LEARNER: Final = "recalc.dr_invalid_learner"
    DR_INVALID_MODEL: Final = "recalc.dr_invalid_model"
    DR_INVALID_OVERLAP: Final = "recalc.dr_invalid_overlap"
    DR_INVALID_SCORE: Final = "recalc.dr_invalid_score"
    DR_MODEL_DATA_INVALID: Final = "recalc.dr_model_data_invalid"
    DR_NOT_IDENTIFIED: Final = "recalc.dr_not_identified"
    DR_PREDICTION_OUT_OF_SUPPORT: Final = "recalc.dr_prediction_out_of_support"
    DR_PREDICTION_SCHEMA_MISMATCH: Final = "recalc.dr_prediction_schema_mismatch"
    DR_PREDICTOR_UNAVAILABLE: Final = "recalc.dr_predictor_unavailable"
    DR_RECEIPT_INVALID: Final = "recalc.dr_receipt_invalid"
    DR_SCORES_UNAVAILABLE: Final = "recalc.dr_scores_unavailable"
    DR_SOURCE_MISMATCH: Final = "recalc.dr_source_mismatch"
    DR_TARGET_INCOMPATIBLE: Final = "recalc.dr_target_incompatible"
    FAMILY_MODEL_DATA_INVALID: Final = "recalc.family_model_data_invalid"
    FAMILY_NOT_IDENTIFIED: Final = "recalc.family_not_identified"
    FAMILY_RECEIPT_INVALID: Final = "recalc.family_receipt_invalid"
    FROZEN_SCORES_REJECTED: Final = "recalc.frozen_scores_rejected"
    INCOMPATIBLE_CONTRAST: Final = "recalc.incompatible_contrast"
    INVALID_ADJUSTMENT_SET: Final = "recalc.invalid_adjustment_set"
    INVALID_CATEGORICAL_CONTRAST: Final = "recalc.invalid_categorical_contrast"
    INVALID_CHANGED_INPUT: Final = "recalc.invalid_changed_input"
    INVALID_CONTRAST: Final = "recalc.invalid_contrast"
    INVALID_DATA: Final = "recalc.invalid_data"
    INVALID_DECLARATION: Final = "recalc.invalid_declaration"
    INVALID_DESIGN_ARTIFACT: Final = "recalc.invalid_design_artifact"
    INVALID_DESIGN_ROLES: Final = "recalc.invalid_design_roles"
    INVALID_EVIDENCE: Final = "recalc.invalid_evidence"
    INVALID_GRAPH: Final = "recalc.invalid_graph"
    INVALID_INITIAL_STATE_LAW: Final = "recalc.invalid_initial_state_law"
    INVALID_QUERY: Final = "recalc.invalid_query"
    INVALID_RD_WINDOW: Final = "recalc.invalid_rd_window"
    INVALID_REQUEST: Final = "recalc.invalid_request"
    INVALID_TARGET_WEIGHTS: Final = "recalc.invalid_target_weights"
    INVALID_TEMPORAL_PROVIDER: Final = "recalc.invalid_temporal_provider"
    INVALID_TEMPORAL_WINDOW: Final = "recalc.invalid_temporal_window"
    IV_DECISION_UNAVAILABLE: Final = "recalc.iv_decision_unavailable"
    IV_DIAGNOSTICS_UNAVAILABLE: Final = "recalc.iv_diagnostics_unavailable"
    JOINT_BAYESIAN_GRAPH_UNSUPPORTED: Final = "recalc.joint_bayesian_graph_unsupported"
    JOINT_BAYESIAN_INVALID_DATA: Final = "recalc.joint_bayesian_invalid_data"
    JOINT_BAYESIAN_INVALID_QUERY: Final = "recalc.joint_bayesian_invalid_query"
    JOINT_BAYESIAN_REFUSED: Final = "recalc.joint_bayesian_refused"
    LIMITS_EXCEEDED: Final = "recalc.limits_exceeded"
    MEMORY_BUDGET_EXCEEDED: Final = "recalc.memory_budget_exceeded"
    MZ_ACTION_OUT_OF_SUPPORT: Final = "recalc.mz_action_out_of_support"
    MZ_INVALID_REQUEST: Final = "recalc.mz_invalid_request"
    NO_LIVE_STATE: Final = "recalc.no_live_state"
    OFF_GRID_REQUEST: Final = "recalc.off_grid_request"
    PRIMARY_RESULT_IS_NOT_AN_EFFECT: Final = "recalc.primary_result_is_not_an_effect"
    REFUSED: Final = "recalc.refused"
    RETARGET_INCOMPATIBLE: Final = "recalc.retarget_incompatible"
    RETARGET_NOT_SUPPORTED: Final = "recalc.retarget_not_supported"
    ROW_COUNT_MISMATCH: Final = "recalc.row_count_mismatch"
    ROW_IDS_MISMATCH: Final = "recalc.row_ids_mismatch"
    SCORE_STAGE_UNDECLARED: Final = "recalc.score_stage_undeclared"
    SCORE_TABLE_UNAVAILABLE: Final = "recalc.score_table_unavailable"
    STATIC_ACTION_OUT_OF_SUPPORT: Final = "recalc.static_action_out_of_support"
    STATIC_ENGINE_REFUSED: Final = "recalc.static_engine_refused"
    STATIC_GRAPH_UNSUPPORTED: Final = "recalc.static_graph_unsupported"
    STATIC_INVALID_CONTRAST: Final = "recalc.static_invalid_contrast"
    STATIC_INVALID_DATA: Final = "recalc.static_invalid_data"
    STATIC_INVALID_FACTOR: Final = "recalc.static_invalid_factor"
    STATIC_INVALID_QUERY: Final = "recalc.static_invalid_query"
    STATIC_NOT_IDENTIFIED: Final = "recalc.static_not_identified"
    STATIC_RECEIPT_INVALID: Final = "recalc.static_receipt_invalid"
    STATIC_REQUIRED_FACTOR_MISSING: Final = "recalc.static_required_factor_missing"
    STATIC_TRANSPORT_BUDGET_CANCEL: Final = "recalc.static_transport_budget_cancel"
    STATIC_TRANSPORT_MISSING_EVIDENCE: Final = "recalc.static_transport_missing_evidence"
    STATIC_TRANSPORT_MISSING_PROVIDER: Final = "recalc.static_transport_missing_provider"
    STATIC_TRANSPORT_NON_TRANSPORTABLE: Final = "recalc.static_transport_non_transportable"
    STATIC_TRANSPORT_NOT_CERTIFIED: Final = "recalc.static_transport_not_certified"
    STATIC_TRANSPORT_NUMERICAL_FAILURE: Final = "recalc.static_transport_numerical_failure"
    STATIC_TRANSPORT_SUPPORT_FAILURE: Final = "recalc.static_transport_support_failure"
    STATIC_TRANSPORT_UNSUPPORTED_EVALUATOR: Final = "recalc.static_transport_unsupported_evaluator"
    TEMPORAL_ARTIFACT_INVALID: Final = "recalc.temporal_artifact_invalid"
    TEMPORAL_ARTIFACT_MISMATCH: Final = "recalc.temporal_artifact_mismatch"
    TEMPORAL_ARTIFACT_SEED_MISMATCH: Final = "recalc.temporal_artifact_seed_mismatch"
    TEMPORAL_BUDGET_CANCEL: Final = "recalc.temporal_budget_cancel"
    TEMPORAL_DATA_INVALID: Final = "recalc.temporal_data_invalid"
    TEMPORAL_EVALUATOR_UNSUPPORTED: Final = "recalc.temporal_evaluator_unsupported"
    TEMPORAL_FIT_REFUSED: Final = "recalc.temporal_fit_refused"
    TEMPORAL_HISTORY_UNSUPPORTED: Final = "recalc.temporal_history_unsupported"
    TEMPORAL_MISSING_EVIDENCE: Final = "recalc.temporal_missing_evidence"
    TEMPORAL_MISSING_PROVIDER: Final = "recalc.temporal_missing_provider"
    TEMPORAL_NUMERICAL_FAILURE: Final = "recalc.temporal_numerical_failure"
    TEMPORAL_SCOPE_UNSUPPORTED: Final = "recalc.temporal_scope_unsupported"
    TEMPORAL_SUPPORT_FAILURE: Final = "recalc.temporal_support_failure"
    UNAVAILABLE_DATA: Final = "recalc.unavailable_data"
    UNAVAILABLE_FIT: Final = "recalc.unavailable_fit"
    UNAVAILABLE_PROVIDER: Final = "recalc.unavailable_provider"
    UNKNOWN_VARIABLE: Final = "recalc.unknown_variable"
    UNSUPPORTED_AIPW_SCORE_CONFIGURATION: Final = "recalc.unsupported_aipw_score_configuration"
    UNSUPPORTED_REQUEST: Final = "recalc.unsupported_request"
    WEIGHTED_OVERLAP_FAILED: Final = "recalc.weighted_overlap_failed"


class recalc_receipt:  # noqa: N801
    COUNTS_INCONSISTENT: Final = "recalc_receipt.counts_inconsistent"
    FRESH_PROCESS_REUSE: Final = "recalc_receipt.fresh_process_reuse"
    IDENTITY_MISMATCH: Final = "recalc_receipt.identity_mismatch"
    PLAN_MISMATCH: Final = "recalc_receipt.plan_mismatch"
    TOTALS_MISMATCH: Final = "recalc_receipt.totals_mismatch"


class recovery:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "recovery.bounds_exceeded"
    BUDGET: Final = "recovery.budget"
    EMPIRICAL_NOT_SUPPORTED: Final = "recovery.empirical_not_supported"
    HANDOFF_MISMATCH: Final = "recovery.handoff_mismatch"
    INVALID_DERIVATION: Final = "recovery.invalid_derivation"
    INVALID_OBSERVED_LAW: Final = "recovery.invalid_observed_law"
    INVALID_QUERY: Final = "recovery.invalid_query"
    MISSING_MARGIN: Final = "recovery.missing_margin"
    NONRECOVERABLE_WITNESS: Final = "recovery.nonrecoverable_witness"
    POSITIVITY: Final = "recovery.positivity"
    UNSUPPORTED_MECHANISM: Final = "recovery.unsupported_mechanism"
    WITNESS_UNAVAILABLE: Final = "recovery.witness_unavailable"


class recovery_chain:  # noqa: N801
    BUDGET: Final = "recovery_chain.budget"
    DATA_IDENTITY_MISMATCH: Final = "recovery_chain.data_identity_mismatch"
    DECISION_REPLAY_MISMATCH: Final = "recovery_chain.decision_replay_mismatch"
    INVALID_DERIVATION: Final = "recovery_chain.invalid_derivation"
    INVALID_OBSERVED_LAW: Final = "recovery_chain.invalid_observed_law"
    INVALID_QUERY: Final = "recovery_chain.invalid_query"
    NAMES_MISMATCH: Final = "recovery_chain.names_mismatch"
    NONRECOVERABLE_WITNESS: Final = "recovery_chain.nonrecoverable_witness"
    NOT_EVALUATED: Final = "recovery_chain.not_evaluated"
    POSITIVITY: Final = "recovery_chain.positivity"
    PREMISES_MISMATCH: Final = "recovery_chain.premises_mismatch"
    RECOVERED_REPLAY_MISMATCH: Final = "recovery_chain.recovered_replay_mismatch"
    UNDECODABLE: Final = "recovery_chain.undecodable"
    UNSUPPORTED_MECHANISM: Final = "recovery_chain.unsupported_mechanism"
    UNSUPPORTED_SEMANTICS: Final = "recovery_chain.unsupported_semantics"


class refresh_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "refresh_posterior.band_not_constructed"
    EMPTY_VALUES: Final = "refresh_posterior.empty_values"
    EXTERNAL_POWER_PRIOR: Final = "refresh_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "refresh_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "refresh_posterior.non_positive_scale"


class repair_artifact:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "repair_artifact.bounds_exceeded"
    BUDGET: Final = "repair_artifact.budget"
    INVALID_ARTIFACT: Final = "repair_artifact.invalid_artifact"
    REPLAY_MISMATCH: Final = "repair_artifact.replay_mismatch"
    UNSUPPORTED_VERSION: Final = "repair_artifact.unsupported_version"


class rollout:  # noqa: N801
    ARTIFACT_ID_MISSING: Final = "rollout.artifact_id_missing"
    ARTIFACT_KIND_MISMATCH: Final = "rollout.artifact_kind_mismatch"
    EXPECTED_BINDING_MISMATCH: Final = "rollout.expected_binding_mismatch"
    FINITE_DRAW_STATE_REQUIRED: Final = "rollout.finite_draw_state_required"
    IDENTITY_MISMATCH: Final = "rollout.identity_mismatch"
    INVALID_REQUEST: Final = "rollout.invalid_request"
    ORDERED_PRIOR_MISMATCH: Final = "rollout.ordered_prior_mismatch"
    RANKING_INVALID: Final = "rollout.ranking_invalid"
    RANKING_SOURCE_MISMATCH: Final = "rollout.ranking_source_mismatch"
    SOURCE_DIGEST_MISMATCH: Final = "rollout.source_digest_mismatch"
    SOURCE_EVIDENCE_MISMATCH: Final = "rollout.source_evidence_mismatch"
    SOURCE_STANDING_MISMATCH: Final = "rollout.source_standing_mismatch"
    STATE_COORDINATE_MISMATCH: Final = "rollout.state_coordinate_mismatch"
    STATE_INTERPRETATION_MISMATCH: Final = "rollout.state_interpretation_mismatch"
    TERMINAL_DECISION_MISMATCH: Final = "rollout.terminal_decision_mismatch"
    UNWEIGHTED_JOINT_STATE_REQUIRED: Final = "rollout.unweighted_joint_state_required"


class sampled_recovery:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "sampled_recovery.bounds_exceeded"
    CANCELLED: Final = "sampled_recovery.cancelled"
    COMPONENT_VARIANCE_ONLY: Final = "sampled_recovery.component_variance_only"
    INVALID_INPUT: Final = "sampled_recovery.invalid_input"
    NATIVE_HOOK_UNAVAILABLE: Final = "sampled_recovery.native_hook_unavailable"
    PRODUCER_MISSING: Final = "sampled_recovery.producer_missing"
    RECEIPT_MISMATCH: Final = "sampled_recovery.receipt_mismatch"
    TOO_MANY_FAILED_REPLICATES: Final = "sampled_recovery.too_many_failed_replicates"
    UNRECOVERABLE_PATTERN: Final = "sampled_recovery.unrecoverable_pattern"


class sampling_weights:  # noqa: N801
    INVALID: Final = "sampling_weights.invalid"
    ROUTE_UNSUPPORTED: Final = "sampling_weights.route_unsupported"


class scenario_covariance:  # noqa: N801
    CANCELLED: Final = "scenario_covariance.cancelled"
    CONSUMER_LIMIT_EXCEEDED: Final = "scenario_covariance.consumer_limit_exceeded"
    DATA_IDENTITY_MISMATCH: Final = "scenario_covariance.data_identity_mismatch"
    DUPLICATE_SCENARIO_ID: Final = "scenario_covariance.duplicate_scenario_id"
    EXACT_ENUMERATION_CAP: Final = "scenario_covariance.exact_enumeration_cap"
    INVALID_DECLARATION: Final = "scenario_covariance.invalid_declaration"
    INVALID_DEPENDENCE: Final = "scenario_covariance.invalid_dependence"
    INVALID_FAILURE_BOUND: Final = "scenario_covariance.invalid_failure_bound"
    INVALID_FUNCTIONAL: Final = "scenario_covariance.invalid_functional"
    INVALID_ROW_TABLE: Final = "scenario_covariance.invalid_row_table"
    NO_SCENARIOS: Final = "scenario_covariance.no_scenarios"
    NOT_POSITIVE_SEMIDEFINITE: Final = "scenario_covariance.not_positive_semidefinite"
    PREMISES_MISMATCH: Final = "scenario_covariance.premises_mismatch"
    REPLAY_MISMATCH: Final = "scenario_covariance.replay_mismatch"
    TABLE_TOO_LARGE: Final = "scenario_covariance.table_too_large"
    TOO_FEW_REPLICATES: Final = "scenario_covariance.too_few_replicates"
    TOO_FEW_ROWS: Final = "scenario_covariance.too_few_rows"
    TOO_MANY_FAILED_REPLICATES: Final = "scenario_covariance.too_many_failed_replicates"
    TOO_MANY_TERMS: Final = "scenario_covariance.too_many_terms"
    UNKNOWN_DEPENDENCE: Final = "scenario_covariance.unknown_dependence"
    UNSUPPORTED_SEMANTICS: Final = "scenario_covariance.unsupported_semantics"


class scenario_covariance_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "scenario_covariance_posterior.band_not_constructed"
    EMPTY_MATRIX: Final = "scenario_covariance_posterior.empty_matrix"
    EXTERNAL_POWER_PRIOR: Final = "scenario_covariance_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "scenario_covariance_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "scenario_covariance_posterior.non_positive_scale"


class scenario_invariance:  # noqa: N801
    DERIVATION_INCONSISTENT: Final = "scenario_invariance.derivation_inconsistent"
    NON_STATIC_NODE: Final = "scenario_invariance.non_static_node"
    NOT_ESTIMATED: Final = "scenario_invariance.not_estimated"
    REFUSED: Final = "scenario_invariance.refused"
    WRONG_RESULT_TYPE: Final = "scenario_invariance.wrong_result_type"


class scenarios:  # noqa: N801
    CATALOG_SELECTIONS: Final = "scenarios.catalog_selections"
    COORDINATE_MISMATCH: Final = "scenarios.coordinate_mismatch"
    COUNT: Final = "scenarios.count"
    DUPLICATE_OR_EMPTY_NAME: Final = "scenarios.duplicate_or_empty_name"
    DUPLICATE_SCENARIO: Final = "scenarios.duplicate_scenario"
    EMPTY: Final = "scenarios.empty"
    EQUIVALENCE_CLASS_INPUT: Final = "scenarios.equivalence_class_input"
    INVALID_WEIGHTS: Final = "scenarios.invalid_weights"
    SHARED_DATA_AGGREGATE: Final = "scenarios.shared_data_aggregate"
    UNEVALUATED_BUDGET: Final = "scenarios.unevaluated_budget"


class scientific_bridge:  # noqa: N801
    ARTIFACT_INVALID: Final = "scientific_bridge.artifact_invalid"
    ASSOCIATION_AS_INTERVENTION: Final = "scientific_bridge.association_as_intervention"
    ASSUMPTION_EVIDENCE_MISSING: Final = "scientific_bridge.assumption_evidence_missing"
    ASSUMPTION_MISSING: Final = "scientific_bridge.assumption_missing"
    BLANK_FIELD: Final = "scientific_bridge.blank_field"
    CHAIN_STAGE_NOT_AFFINE: Final = "scientific_bridge.chain_stage_not_affine"
    CONDITIONING_MISMATCH: Final = "scientific_bridge.conditioning_mismatch"
    DEGENERATE_VALIDATION: Final = "scientific_bridge.degenerate_validation"
    DEPENDENCE_NOT_A_COVARIANCE: Final = "scientific_bridge.dependence_not_a_covariance"
    DIFFERENTIAL_ERROR: Final = "scientific_bridge.differential_error"
    DRAWS_MEANING_MISMATCH: Final = "scientific_bridge.draws_meaning_mismatch"
    DRAWS_NOT_ALIGNED: Final = "scientific_bridge.draws_not_aligned"
    DRAWS_OUTSIDE_DOMAIN: Final = "scientific_bridge.draws_outside_domain"
    DRAWS_UNUSABLE: Final = "scientific_bridge.draws_unusable"
    DUPLICATE_ASSUMPTION: Final = "scientific_bridge.duplicate_assumption"
    ENDPOINT_CLASSES: Final = "scientific_bridge.endpoint_classes"
    ESTIMAND_BRIDGE_INCOMPLETE: Final = "scientific_bridge.estimand_bridge_incomplete"
    ESTIMAND_BRIDGE_MISMATCH: Final = "scientific_bridge.estimand_bridge_mismatch"
    ESTIMAND_SPEND_INCONSISTENT: Final = "scientific_bridge.estimand_spend_inconsistent"
    ESTIMAND_UNITS_MISMATCH: Final = "scientific_bridge.estimand_units_mismatch"
    EXPERIMENTAL_PRIOR_WITHOUT_ESTIMAND_BRIDGE: Final = (
        "scientific_bridge.experimental_prior_without_estimand_bridge"
    )
    FUNCTIONAL_NOT_SUPPORTED: Final = "scientific_bridge.functional_not_supported"
    HORIZON_MISMATCH: Final = "scientific_bridge.horizon_mismatch"
    IDENTITY_MISMATCH: Final = "scientific_bridge.identity_mismatch"
    INTERVAL_NOT_SUPPORTED: Final = "scientific_bridge.interval_not_supported"
    INTERVAL_SCOPE_OUTSIDE_MEASURED: Final = "scientific_bridge.interval_scope_outside_measured"
    INTERVAL_SOURCE_UNCERTAINTY_REQUIRED: Final = (
        "scientific_bridge.interval_source_uncertainty_required"
    )
    INVALID_QUANTITY: Final = "scientific_bridge.invalid_quantity"
    MALFORMED_MAPPING: Final = "scientific_bridge.malformed_mapping"
    MAPPING_NOT_ALLOWED: Final = "scientific_bridge.mapping_not_allowed"
    MAPPING_SCOPE_MISMATCH: Final = "scientific_bridge.mapping_scope_mismatch"
    MAPPING_UNCERTAINTY_INVALID: Final = "scientific_bridge.mapping_uncertainty_invalid"
    MAPPING_UNCERTAINTY_MISSING: Final = "scientific_bridge.mapping_uncertainty_missing"
    MEASUREMENT_MODEL_MISMATCH: Final = "scientific_bridge.measurement_model_mismatch"
    MEASUREMENT_ROUTE_NOT_SUPPORTED: Final = "scientific_bridge.measurement_route_not_supported"
    MODEL_IDENTITY_MISMATCH: Final = "scientific_bridge.model_identity_mismatch"
    MODEL_SCORE_AS_PROBABILITY: Final = "scientific_bridge.model_score_as_probability"
    NON_FINITE_VALUE: Final = "scientific_bridge.non_finite_value"
    NONLINEAR_LAW_INSUFFICIENT: Final = "scientific_bridge.nonlinear_law_insufficient"
    NONLINEAR_MEAN_NOT_EXPECTATION: Final = "scientific_bridge.nonlinear_mean_not_expectation"
    OUTSIDE_ATTAINABLE_RANGE: Final = "scientific_bridge.outside_attainable_range"
    OUTSIDE_VALIDATED_RANGE: Final = "scientific_bridge.outside_validated_range"
    POPULATION_MISMATCH: Final = "scientific_bridge.population_mismatch"
    PREDICTIVE_IMPORTANCE_AS_CAUSAL_EFFECT: Final = (
        "scientific_bridge.predictive_importance_as_causal_effect"
    )
    PROPAGATION_OVERFLOW: Final = "scientific_bridge.propagation_overflow"
    PROVENANCE: Final = "scientific_bridge.provenance"
    PROXY_AS_CONSTRUCT: Final = "scientific_bridge.proxy_as_construct"
    QUANTITY_MISMATCH: Final = "scientific_bridge.quantity_mismatch"
    REGIME_MISMATCH: Final = "scientific_bridge.regime_mismatch"
    REPLAY_MISMATCH: Final = "scientific_bridge.replay_mismatch"
    SCIENTIFIC_ENDPOINT_AS_UTILITY: Final = "scientific_bridge.scientific_endpoint_as_utility"
    SOURCE_DEPENDENCE_INVALID: Final = "scientific_bridge.source_dependence_invalid"
    SOURCE_DEPENDENCE_MISSING: Final = "scientific_bridge.source_dependence_missing"
    SOURCE_MOMENT_NOT_FINITE: Final = "scientific_bridge.source_moment_not_finite"
    SOURCE_NOT_DRAWS: Final = "scientific_bridge.source_not_draws"
    SOURCE_NOT_POINT: Final = "scientific_bridge.source_not_point"
    SOURCE_VARIANCE_INVALID: Final = "scientific_bridge.source_variance_invalid"
    STATED_PREFERENCE_AS_REALIZED_BEHAVIOR: Final = (
        "scientific_bridge.stated_preference_as_realized_behavior"
    )
    TRANSPORT_DIAGRAM_SCOPE_MISMATCH: Final = "scientific_bridge.transport_diagram_scope_mismatch"
    TRANSPORT_PREMISE_CONTRADICTED_BY_SELECTION_DIAGRAM: Final = (
        "scientific_bridge.transport_premise_contradicted_by_selection_diagram"
    )
    TRANSPORT_PREMISE_MISMATCH: Final = "scientific_bridge.transport_premise_mismatch"
    TRANSPORT_PREMISE_MISSING: Final = "scientific_bridge.transport_premise_missing"
    TRANSPORT_PREMISE_UNUSED: Final = "scientific_bridge.transport_premise_unused"
    UNCERTAIN_MAPPING_NOT_AFFINE: Final = "scientific_bridge.uncertain_mapping_not_affine"
    UNITS_MISMATCH: Final = "scientific_bridge.units_mismatch"
    VALIDATION_R_SQUARED_TOO_LOW: Final = "scientific_bridge.validation_r_squared_too_low"
    VALIDATION_TRIALS_TOO_FEW: Final = "scientific_bridge.validation_trials_too_few"
    VALUE_KIND_NOT_CARRIED: Final = "scientific_bridge.value_kind_not_carried"
    VALUE_KIND_SUBSTITUTION: Final = "scientific_bridge.value_kind_substitution"
    YOUDEN_TOO_SMALL: Final = "scientific_bridge.youden_too_small"


class sde:  # noqa: N801
    ACTION_TIME_OFF_GRID: Final = "sde.action_time_off_grid"
    INPUT_OUTSIDE_CERTIFIED_ENVELOPE: Final = "sde.input_outside_certified_envelope"
    MILSTEIN_NONCOMMUTATIVE: Final = "sde.milstein_noncommutative"
    UNBOUNDED_GROWTH: Final = "sde.unbounded_growth"


class selective_discovery:  # noqa: N801
    ANALYSIS_ROWS_NOT_THE_SPLIT_PART: Final = "selective_discovery.analysis_rows_not_the_split_part"
    ARTIFACT_INVALID: Final = "selective_discovery.artifact_invalid"
    BACKGROUND_KNOWLEDGE_CONTRADICTORY: Final = (
        "selective_discovery.background_knowledge_contradictory"
    )
    BACKGROUND_KNOWLEDGE_NOT_APPLICABLE: Final = (
        "selective_discovery.background_knowledge_not_applicable"
    )
    BACKGROUND_KNOWLEDGE_UNKNOWN_VARIABLE: Final = (
        "selective_discovery.background_knowledge_unknown_variable"
    )
    COMPONENTS_INCOMPLETE: Final = "selective_discovery.components_incomplete"
    CONTEMPORANEOUS_LINKS: Final = "selective_discovery.contemporaneous_links"
    CONTRACT_IDENTITY: Final = "selective_discovery.contract_identity"
    CONTRACT_MISMATCH: Final = "selective_discovery.contract_mismatch"
    DATA_MISMATCH: Final = "selective_discovery.data_mismatch"
    DIFFERENCES_ARE_NOT_SELECTION_TARGETS: Final = (
        "selective_discovery.differences_are_not_selection_targets"
    )
    DOWNSTREAM_INTERVAL_NOT_SUPPORTED: Final = (
        "selective_discovery.downstream_interval_not_supported"
    )
    IDENTITY_MISMATCH: Final = "selective_discovery.identity_mismatch"
    INVALID_CONTRACT: Final = "selective_discovery.invalid_contract"
    MULTI_POPULATION: Final = "selective_discovery.multi_population"
    NOTHING_RETAINED: Final = "selective_discovery.nothing_retained"
    POOLED_POPULATION_DATA: Final = "selective_discovery.pooled_population_data"
    POOLED_POPULATION_SEARCH: Final = "selective_discovery.pooled_population_search"
    POPULATION_MISSING: Final = "selective_discovery.population_missing"
    POPULATION_REPEATED: Final = "selective_discovery.population_repeated"
    POPULATIONS_MISSING: Final = "selective_discovery.populations_missing"
    RAW_DATA_MISSING: Final = "selective_discovery.raw_data_missing"
    RECEIPT_INVALID: Final = "selective_discovery.receipt_invalid"
    RECEIPT_LIMIT: Final = "selective_discovery.receipt_limit"
    RECEIPT_REPEATED: Final = "selective_discovery.receipt_repeated"
    RECEIPTS_IDENTITY: Final = "selective_discovery.receipts_identity"
    RECOVERY_CLAIM_NOT_SUPPORTED: Final = "selective_discovery.recovery_claim_not_supported"
    REPLAY_DIVERGED: Final = "selective_discovery.replay_diverged"
    REPLAY_MISS: Final = "selective_discovery.replay_miss"
    RNG_MISMATCH: Final = "selective_discovery.rng_mismatch"
    ROUTE_UNKNOWN: Final = "selective_discovery.route_unknown"
    ROWS: Final = "selective_discovery.rows"
    SCORE_SEARCH_NOT_SUPPORTED: Final = "selective_discovery.score_search_not_supported"
    SEARCH_INCOMPLETE: Final = "selective_discovery.search_incomplete"
    SPLIT_FRACTION_INVALID: Final = "selective_discovery.split_fraction_invalid"
    SPLIT_MASK_MISMATCH: Final = "selective_discovery.split_mask_mismatch"
    SPLIT_NOT_DISJOINT: Final = "selective_discovery.split_not_disjoint"
    SPLIT_PART_EMPTY: Final = "selective_discovery.split_part_empty"
    SPLIT_PART_WEIGHT_INVALID: Final = "selective_discovery.split_part_weight_invalid"
    SPLIT_RECEIPT_MISMATCH: Final = "selective_discovery.split_receipt_mismatch"
    SPLIT_ROW_EXCLUDED: Final = "selective_discovery.split_row_excluded"
    SPLIT_ROW_OUT_OF_RANGE: Final = "selective_discovery.split_row_out_of_range"
    SPLIT_ROWS_MISMATCH: Final = "selective_discovery.split_rows_mismatch"
    SPLIT_ROWS_REPEATED: Final = "selective_discovery.split_rows_repeated"
    SPLIT_SOURCE_MISMATCH: Final = "selective_discovery.split_source_mismatch"
    SPLIT_UNSUPPORTED_ROWS: Final = "selective_discovery.split_unsupported_rows"
    SPLIT_WEIGHTS_MISMATCH: Final = "selective_discovery.split_weights_mismatch"
    STRUCTURE_CLASS: Final = "selective_discovery.structure_class"
    STRUCTURE_MARKS: Final = "selective_discovery.structure_marks"
    STRUCTURE_NODES: Final = "selective_discovery.structure_nodes"
    TEST_IDENTITY_MISSING: Final = "selective_discovery.test_identity_missing"
    UNACCEPTED_DISCOVERY: Final = "selective_discovery.unaccepted_discovery"
    UNRESOLVED_STRUCTURE: Final = "selective_discovery.unresolved_structure"
    VARIABLES_REPEATED: Final = "selective_discovery.variables_repeated"
    WEIGHTED_TEST_WITHOUT_WEIGHTS: Final = "selective_discovery.weighted_test_without_weights"
    WEIGHTS_NOT_READ_BY_TEST: Final = "selective_discovery.weights_not_read_by_test"
    WORK_INCONSISTENT: Final = "selective_discovery.work_inconsistent"


class sensitivity_decision_composition:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "sensitivity_decision_composition.bounds_exceeded"
    COMPOSITION_NOT_SUPPORTED: Final = "sensitivity_decision_composition.composition_not_supported"
    INVALID_ARTIFACT: Final = "sensitivity_decision_composition.invalid_artifact"
    INVALID_SURFACE: Final = "sensitivity_decision_composition.invalid_surface"
    SAMPLING_NOT_REPORTED: Final = "sensitivity_decision_composition.sampling_not_reported"
    UNSUPPORTED_ADAPTATION: Final = "sensitivity_decision_composition.unsupported_adaptation"
    UNSUPPORTED_COORDINATE: Final = "sensitivity_decision_composition.unsupported_coordinate"
    WRONG_CONTRACT: Final = "sensitivity_decision_composition.wrong_contract"


class sensitivity_source:  # noqa: N801
    INVALID_ARTIFACT: Final = "sensitivity_source.invalid_artifact"
    LIMITS_EXCEEDED: Final = "sensitivity_source.limits_exceeded"
    LINEAGE_DIGEST: Final = "sensitivity_source.lineage_digest"
    SOURCE_UNAVAILABLE: Final = "sensitivity_source.source_unavailable"
    SURFACE_MISMATCH: Final = "sensitivity_source.surface_mismatch"


class sequential_evaluation:  # noqa: N801
    ACCOUNTING_RESIDUAL: Final = "sequential_evaluation.accounting_residual"
    ACTION_CHANNEL: Final = "sequential_evaluation.action_channel"
    ACTION_COST: Final = "sequential_evaluation.action_cost"
    ACTION_DELAYED_OUTCOME: Final = "sequential_evaluation.action_delayed_outcome"
    ACTION_INDEX: Final = "sequential_evaluation.action_index"
    ACTION_LABEL: Final = "sequential_evaluation.action_label"
    ACTION_NOT_SUPPORTED: Final = "sequential_evaluation.action_not_supported"
    ACTION_SUPPORT: Final = "sequential_evaluation.action_support"
    ACTION_TRANSITION: Final = "sequential_evaluation.action_transition"
    ACTION_UTILITY: Final = "sequential_evaluation.action_utility"
    ARTIFACT_ID_MISSING: Final = "sequential_evaluation.artifact_id_missing"
    ARTIFACT_KIND_MISMATCH: Final = "sequential_evaluation.artifact_kind_mismatch"
    BEHAVIOUR_SUPPORT_MISSING: Final = "sequential_evaluation.behaviour_support_missing"
    CANCELLED: Final = "sequential_evaluation.cancelled"
    CHANNEL_BINDING_MISMATCH: Final = "sequential_evaluation.channel_binding_mismatch"
    CHANNEL_CONSTRUCT_MISMATCH: Final = "sequential_evaluation.channel_construct_mismatch"
    CHANNEL_LEVELS_MISMATCH: Final = "sequential_evaluation.channel_levels_mismatch"
    CHANNEL_RATES: Final = "sequential_evaluation.channel_rates"
    CHANNEL_ROUTE: Final = "sequential_evaluation.channel_route"
    COMMIT_HAS_NO_OBSERVATION: Final = "sequential_evaluation.commit_has_no_observation"
    COMPARISON_KIND_MISMATCH: Final = "sequential_evaluation.comparison_kind_mismatch"
    COMPARISON_LABEL: Final = "sequential_evaluation.comparison_label"
    COMPARISONS: Final = "sequential_evaluation.comparisons"
    DECISION_RECEIPT_MISMATCH: Final = "sequential_evaluation.decision_receipt_mismatch"
    DELAYED_OUTCOME_BEYOND_HORIZON: Final = "sequential_evaluation.delayed_outcome_beyond_horizon"
    DIFFERENTIAL_CHANNEL: Final = "sequential_evaluation.differential_channel"
    EMPTY_LOG: Final = "sequential_evaluation.empty_log"
    EXACT_MISMATCH: Final = "sequential_evaluation.exact_mismatch"
    HISTORY_RECEIPT_MISMATCH: Final = "sequential_evaluation.history_receipt_mismatch"
    IDENTITY_MISMATCH: Final = "sequential_evaluation.identity_mismatch"
    IMPORTANCE_WEIGHT_OVERFLOW: Final = "sequential_evaluation.importance_weight_overflow"
    INDICATOR_AS_STATE: Final = "sequential_evaluation.indicator_as_state"
    INTERVAL_NOT_SUPPORTED: Final = "sequential_evaluation.interval_not_supported"
    LIMIT_EXCEEDED: Final = "sequential_evaluation.limit_exceeded"
    LOG_PARTIALLY_COUNTED: Final = "sequential_evaluation.log_partially_counted"
    LOGGED_TRAJECTORY_INCONSISTENT: Final = "sequential_evaluation.logged_trajectory_inconsistent"
    MALFORMED: Final = "sequential_evaluation.malformed"
    MODEL_ACTIONS: Final = "sequential_evaluation.model_actions"
    MODEL_FIELD: Final = "sequential_evaluation.model_field"
    MODEL_HORIZON: Final = "sequential_evaluation.model_horizon"
    MODEL_INITIAL: Final = "sequential_evaluation.model_initial"
    MODEL_MISMATCH: Final = "sequential_evaluation.model_mismatch"
    MODEL_RESOURCE_BUDGET: Final = "sequential_evaluation.model_resource_budget"
    MODEL_STATES: Final = "sequential_evaluation.model_states"
    MODEL_TERMINAL_UTILITY: Final = "sequential_evaluation.model_terminal_utility"
    MODEL_WORK: Final = "sequential_evaluation.model_work"
    NO_OVERLAP: Final = "sequential_evaluation.no_overlap"
    OBSERVATION_IMPOSSIBLE: Final = "sequential_evaluation.observation_impossible"
    OBSERVATION_RECORDED_TWICE: Final = "sequential_evaluation.observation_recorded_twice"
    OFF_POLICY_MISMATCH: Final = "sequential_evaluation.off_policy_mismatch"
    PAST_HORIZON: Final = "sequential_evaluation.past_horizon"
    POLICY_CHANGE_REFUSED: Final = "sequential_evaluation.policy_change_refused"
    POLICY_DISTRIBUTION: Final = "sequential_evaluation.policy_distribution"
    POLICY_INCOMPLETE: Final = "sequential_evaluation.policy_incomplete"
    POLICY_MISMATCH: Final = "sequential_evaluation.policy_mismatch"
    POLICY_VERSION: Final = "sequential_evaluation.policy_version"
    RECORD_ALREADY_COUNTED: Final = "sequential_evaluation.record_already_counted"
    RECORD_IDENTITY: Final = "sequential_evaluation.record_identity"
    RESOURCE_BUDGET_EXCEEDED: Final = "sequential_evaluation.resource_budget_exceeded"
    RETURN_OVERFLOW: Final = "sequential_evaluation.return_overflow"
    ROLLOUT_BELIEF: Final = "sequential_evaluation.rollout_belief"
    ROLLOUT_MISMATCH: Final = "sequential_evaluation.rollout_mismatch"
    ROLLOUTS: Final = "sequential_evaluation.rollouts"
    SESSION_LABEL: Final = "sequential_evaluation.session_label"
    STATIC_SCORE_TABLE: Final = "sequential_evaluation.static_score_table"
    STOPPING_BOUNDARY: Final = "sequential_evaluation.stopping_boundary"
    TEMPORAL_TRANSPORT_CONTRACT_MISSING: Final = (
        "sequential_evaluation.temporal_transport_contract_missing"
    )
    TRAJECTORY_IDENTITY: Final = "sequential_evaluation.trajectory_identity"
    TRAJECTORY_RECORDED_TWICE: Final = "sequential_evaluation.trajectory_recorded_twice"
    TRANSPORTED_POLICY_CONTRACT_MISSING: Final = (
        "sequential_evaluation.transported_policy_contract_missing"
    )
    VALUE_MISMATCH: Final = "sequential_evaluation.value_mismatch"
    VERSION: Final = "sequential_evaluation.version"
    VOI_COLLECT_ACTION: Final = "sequential_evaluation.voi_collect_action"
    VOI_COMMIT_ACTIONS: Final = "sequential_evaluation.voi_commit_actions"
    VOI_ENUMERATION_MISMATCH: Final = "sequential_evaluation.voi_enumeration_mismatch"
    VOI_RANKING: Final = "sequential_evaluation.voi_ranking"
    VOI_SCOPE: Final = "sequential_evaluation.voi_scope"
    WEIGHTED_ESTIMATE_OVERFLOW: Final = "sequential_evaluation.weighted_estimate_overflow"
    WEIGHTED_SUM_OVERFLOW: Final = "sequential_evaluation.weighted_sum_overflow"


class signal_provider:  # noqa: N801
    CANDIDATE_MISMATCH: Final = "signal_provider.candidate_mismatch"
    CAPABILITY_MISSING: Final = "signal_provider.capability_missing"
    COMPUTATIONAL_LIMIT: Final = "signal_provider.computational_limit"
    CONDITIONAL_INDEPENDENCE_MISSING: Final = "signal_provider.conditional_independence_missing"
    INVALID_OBJECT: Final = "signal_provider.invalid_object"
    INVALID_REQUEST: Final = "signal_provider.invalid_request"
    INVALID_TABLE: Final = "signal_provider.invalid_table"
    LAW_INCOHERENT: Final = "signal_provider.law_incoherent"
    MISSING_LINEAGE: Final = "signal_provider.missing_lineage"
    OBSERVATION_QUANTITY_MISMATCH: Final = "signal_provider.observation_quantity_mismatch"
    POSTERIOR_INCOHERENT: Final = "signal_provider.posterior_incoherent"
    POSTERIOR_QUANTITY_MISMATCH: Final = "signal_provider.posterior_quantity_mismatch"
    PRIOR_MISMATCH: Final = "signal_provider.prior_mismatch"
    REQUEST_FINGERPRINT_MISMATCH: Final = "signal_provider.request_fingerprint_mismatch"
    SAMPLE_SIZE_INVALID: Final = "signal_provider.sample_size_invalid"
    SAMPLE_SIZE_MISMATCH: Final = "signal_provider.sample_size_mismatch"
    TRUST_MISMATCH: Final = "signal_provider.trust_mismatch"
    TRUST_NATIVE_CLAIMED: Final = "signal_provider.trust_native_claimed"


class smoothed_dose_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "smoothed_dose_posterior.band_not_constructed"
    CANCELLED: Final = "smoothed_dose_posterior.cancelled"
    CERTIFICATE_MISMATCH: Final = "smoothed_dose_posterior.certificate_mismatch"
    EMPTY_SCORES: Final = "smoothed_dose_posterior.empty_scores"
    EXTERNAL_POWER_PRIOR: Final = "smoothed_dose_posterior.external_power_prior"
    INVALID_GRAPH: Final = "smoothed_dose_posterior.invalid_graph"
    MAPPED_PRIOR: Final = "smoothed_dose_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "smoothed_dose_posterior.non_positive_scale"
    NOT_CERTIFIED: Final = "smoothed_dose_posterior.not_certified"
    RESOURCE_LIMIT: Final = "smoothed_dose_posterior.resource_limit"


class source_evidence:  # noqa: N801
    ACTION_MISMATCH: Final = "source_evidence.action_mismatch"
    COORDINATE_MISMATCH: Final = "source_evidence.coordinate_mismatch"
    COORDINATES_UNAVAILABLE: Final = "source_evidence.coordinates_unavailable"
    DIAGNOSTIC_SCOPE_MISMATCH: Final = "source_evidence.diagnostic_scope_mismatch"
    FORWARD_BINDING_MISMATCH: Final = "source_evidence.forward_binding_mismatch"
    INVALID_ARTIFACT: Final = "source_evidence.invalid_artifact"
    INVALID_CONTRACT: Final = "source_evidence.invalid_contract"
    INVALID_COORDINATES: Final = "source_evidence.invalid_coordinates"
    INVALID_SOURCE_ARTIFACT: Final = "source_evidence.invalid_source_artifact"
    LIMITS_EXCEEDED: Final = "source_evidence.limits_exceeded"
    POINT_BINDING_MISMATCH: Final = "source_evidence.point_binding_mismatch"
    RESOLUTION_SCOPE_UNSUPPORTED: Final = "source_evidence.resolution_scope_unsupported"
    RESPONSE_UNAVAILABLE: Final = "source_evidence.response_unavailable"
    SOURCE_BINDING_MISMATCH: Final = "source_evidence.source_binding_mismatch"
    SOURCE_ENCODING_FAILED: Final = "source_evidence.source_encoding_failed"
    SOURCE_EXECUTION_FAILED: Final = "source_evidence.source_execution_failed"
    SOURCE_UNVERIFIED: Final = "source_evidence.source_unverified"


class source_projection:  # noqa: N801
    AFFINE_CONTRACT_REFUSED: Final = "source_projection.affine_contract_refused"
    ATTESTATION_UNAVAILABLE: Final = "source_projection.attestation_unavailable"
    CAUSAL_SOURCE_UNAVAILABLE: Final = "source_projection.causal_source_unavailable"
    COORDINATE_MISMATCH: Final = "source_projection.coordinate_mismatch"
    IDENTITY_MISMATCH: Final = "source_projection.identity_mismatch"
    INVALID_ARTIFACT: Final = "source_projection.invalid_artifact"
    INVALID_REQUEST: Final = "source_projection.invalid_request"
    LIMITS_EXCEEDED: Final = "source_projection.limits_exceeded"
    MEAN_BINDING_MISMATCH: Final = "source_projection.mean_binding_mismatch"
    MEAN_GRID_REQUIRED: Final = "source_projection.mean_grid_required"
    ORIGINAL_CONSUMER_REFUSED: Final = "source_projection.original_consumer_refused"
    REPLAY_MISMATCH: Final = "source_projection.replay_mismatch"
    RESPONSE_UNAVAILABLE: Final = "source_projection.response_unavailable"
    SOURCE_BINDING_MISMATCH: Final = "source_projection.source_binding_mismatch"
    UNSUPPORTED_COORDINATE: Final = "source_projection.unsupported_coordinate"


class split_effect:  # noqa: N801
    DEGENERATE_VARIANCE: Final = "split_effect.degenerate_variance"
    DESIGN_SINGULAR: Final = "split_effect.design_singular"
    DISCOVERY_FAILED: Final = "split_effect.discovery_failed"
    OUTCOME_PRECEDES_TREATMENT: Final = "split_effect.outcome_precedes_treatment"
    SEARCH_NOT_FINISHED: Final = "split_effect.search_not_finished"
    SOURCE_INVALID: Final = "split_effect.source_invalid"
    SOURCE_NOT_CANONICAL: Final = "split_effect.source_not_canonical"
    STRUCTURE_CLASS: Final = "split_effect.structure_class"
    STRUCTURE_UNRESOLVED: Final = "split_effect.structure_unresolved"
    TOO_FEW_ROWS: Final = "split_effect.too_few_rows"


class structured_refusals:  # noqa: N801
    MALFORMED_DETAIL: Final = "structured_refusals.malformed_detail"
    MISSING_STAGE: Final = "structured_refusals.missing_stage"
    UNREGISTERED_CODE: Final = "structured_refusals.unregistered_code"


class study_candidate:  # noqa: N801
    DECLARATIONS_REQUIRED: Final = "study_candidate.declarations_required"
    SAMPLE_BELOW_OBLIGATION: Final = "study_candidate.sample_below_obligation"
    WRONG_CONTRACT: Final = "study_candidate.wrong_contract"


class study_plan:  # noqa: N801
    ARRIVAL_MISMATCH: Final = "study_plan.arrival_mismatch"
    ARRIVAL_NOT_IDENTIFIED: Final = "study_plan.arrival_not_identified"
    BOUNDS_EXCEEDED: Final = "study_plan.bounds_exceeded"
    BUDGET: Final = "study_plan.budget"
    INVALID_ARTIFACT: Final = "study_plan.invalid_artifact"
    INVALID_CANDIDATE: Final = "study_plan.invalid_candidate"
    INVALID_QUERY: Final = "study_plan.invalid_query"
    NO_FAILURE_TO_REPAIR: Final = "study_plan.no_failure_to_repair"
    NONE_CERTIFIED: Final = "study_plan.none_certified"
    THEOREM_LIMITED: Final = "study_plan.theorem_limited"


class survival_effect:  # noqa: N801
    ARM_NOT_POPULATED: Final = "survival_effect.arm_not_populated"
    CANCELLED: Final = "survival_effect.cancelled"
    CENSORING_ASSUMPTION_MISSING: Final = "survival_effect.censoring_assumption_missing"
    CENSORING_ASSUMPTION_UNKNOWN: Final = "survival_effect.censoring_assumption_unknown"
    CENSORING_COVARIATES_MISSING: Final = "survival_effect.censoring_covariates_missing"
    CENSORING_SUPPORT: Final = "survival_effect.censoring_support"
    HORIZON_BEYOND_FOLLOW_UP: Final = "survival_effect.horizon_beyond_follow_up"
    HORIZON_MISSING: Final = "survival_effect.horizon_missing"
    INVALID_DATA: Final = "survival_effect.invalid_data"
    INVALID_EVENT: Final = "survival_effect.invalid_event"
    INVALID_FLOOR: Final = "survival_effect.invalid_floor"
    INVALID_FOLDS: Final = "survival_effect.invalid_folds"
    INVALID_HORIZON: Final = "survival_effect.invalid_horizon"
    INVALID_LEVELS: Final = "survival_effect.invalid_levels"
    INVALID_REQUEST: Final = "survival_effect.invalid_request"
    TOO_FEW_ROWS: Final = "survival_effect.too_few_rows"
    TRAINING_FOLD_WITHOUT_EVENTS: Final = "survival_effect.training_fold_without_events"
    TREATMENT_POSITIVITY: Final = "survival_effect.treatment_positivity"
    UNDECLARED_LEVEL: Final = "survival_effect.undeclared_level"


class temporal_counterfactual:  # noqa: N801
    ARTIFACT_CHANGED: Final = "temporal_counterfactual.artifact_changed"
    FIT_MISMATCH: Final = "temporal_counterfactual.fit_mismatch"
    HISTORY_NAME_INVALID: Final = "temporal_counterfactual.history_name_invalid"
    HORIZON_EXCEEDED: Final = "temporal_counterfactual.horizon_exceeded"
    INVALID_GRAPH: Final = "temporal_counterfactual.invalid_graph"
    INVALID_IDENTITY: Final = "temporal_counterfactual.invalid_identity"
    INVALID_REQUEST: Final = "temporal_counterfactual.invalid_request"
    LATENT_CONFOUNDING: Final = "temporal_counterfactual.latent_confounding"
    LIMITS_EXCEEDED: Final = "temporal_counterfactual.limits_exceeded"
    NON_FINITE_HISTORY: Final = "temporal_counterfactual.non_finite_history"
    REFUTING_HISTORY: Final = "temporal_counterfactual.refuting_history"
    SHARED_HISTORY_MISSING: Final = "temporal_counterfactual.shared_history_missing"
    TIME_MISALIGNED: Final = "temporal_counterfactual.time_misaligned"
    TOO_MANY_UNITS: Final = "temporal_counterfactual.too_many_units"
    UNPAIRED_HISTORIES: Final = "temporal_counterfactual.unpaired_histories"


class temporal_interval:  # noqa: N801
    CONSUMER_UNAVAILABLE: Final = "temporal_interval.consumer_unavailable"
    INTERVAL_NOT_SUPPORTED: Final = "temporal_interval.interval_not_supported"
    INVALID_ESTIMAND: Final = "temporal_interval.invalid_estimand"
    RETIRED_METHOD: Final = "temporal_interval.retired_method"
    RETIRED_RESPONSE_METHOD: Final = "temporal_interval.retired_response_method"
    STUDENTIZED_BOUNDS_EXCEEDED: Final = "temporal_interval.studentized_bounds_exceeded"
    STUDENTIZED_ESTIMATOR_NOT_CERTIFIED: Final = (
        "temporal_interval.studentized_estimator_not_certified"
    )
    STUDENTIZED_SCORE_MISMATCH: Final = "temporal_interval.studentized_score_mismatch"
    STUDENTIZED_UNBALANCED: Final = "temporal_interval.studentized_unbalanced"
    STUDENTIZED_ZERO_VARIANCE: Final = "temporal_interval.studentized_zero_variance"
    TOO_FEW_UNITS: Final = "temporal_interval.too_few_units"
    TOO_MANY_REPLICATES: Final = "temporal_interval.too_many_replicates"
    UNKNOWN_UNITS: Final = "temporal_interval.unknown_units"
    UNSUPPORTED_HISTORY: Final = "temporal_interval.unsupported_history"


class temporal_policy_transport:  # noqa: N801
    HISTORY_ALPHABET_INCOMPATIBLE: Final = "temporal_policy_transport.history_alphabet_incompatible"
    HORIZON_MISMATCH: Final = "temporal_policy_transport.horizon_mismatch"
    INITIAL_LAW_DECLARED_INVARIANT: Final = (
        "temporal_policy_transport.initial_law_declared_invariant"
    )
    INVARIANCE_BINDING_MISMATCH: Final = "temporal_policy_transport.invariance_binding_mismatch"
    MECHANISM_DECLARED_TWICE: Final = "temporal_policy_transport.mechanism_declared_twice"
    MECHANISM_STATUS_MISSING: Final = "temporal_policy_transport.mechanism_status_missing"
    MECHANISM_TIME_VARYING: Final = "temporal_policy_transport.mechanism_time_varying"
    MECHANISM_UNKNOWN: Final = "temporal_policy_transport.mechanism_unknown"
    TARGET_INITIAL_LAW: Final = "temporal_policy_transport.target_initial_law"
    TARGET_INITIAL_LAW_MISSING: Final = "temporal_policy_transport.target_initial_law_missing"
    TARGET_IS_SOURCE: Final = "temporal_policy_transport.target_is_source"
    TARGET_PERIOD: Final = "temporal_policy_transport.target_period"
    TARGET_SUPPORT_OUTSIDE_SOURCE: Final = "temporal_policy_transport.target_support_outside_source"
    TIME_INVARIANCE_MISSING: Final = "temporal_policy_transport.time_invariance_missing"
    VALUE_MISMATCH: Final = "temporal_policy_transport.value_mismatch"


class temporal_refresh:  # noqa: N801
    FACTOR_WINDOW_MISMATCH: Final = "temporal_refresh.factor_window_mismatch"
    HORIZON_CHANGED: Final = "temporal_refresh.horizon_changed"
    INTERVENTION_HISTORY_CHANGED: Final = "temporal_refresh.intervention_history_changed"
    INVALID_INITIAL_LAW: Final = "temporal_refresh.invalid_initial_law"
    INVALID_REPLACEMENT: Final = "temporal_refresh.invalid_replacement"
    LAG_ALIGNMENT_CHANGED: Final = "temporal_refresh.lag_alignment_changed"
    NO_REFRESH: Final = "temporal_refresh.no_refresh"
    PREMISES_CHANGED: Final = "temporal_refresh.premises_changed"
    RECEIPT_DIGEST_MISMATCH: Final = "temporal_refresh.receipt_digest_mismatch"
    RECEIPT_MISMATCH: Final = "temporal_refresh.receipt_mismatch"
    STALE_INTERVAL: Final = "temporal_refresh.stale_interval"
    STALE_SNAPSHOT: Final = "temporal_refresh.stale_snapshot"
    WINDOW_MISSING: Final = "temporal_refresh.window_missing"


class temporal_transport:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "temporal_transport.bounds_exceeded"
    CHECKED_OBSTRUCTION: Final = "temporal_transport.checked_obstruction"
    HISTORY_BUDGET: Final = "temporal_transport.history_budget"
    HISTORY_OUTSIDE_SUPPORT: Final = "temporal_transport.history_outside_support"
    HORIZON: Final = "temporal_transport.horizon"
    INTERVAL_REQUESTED: Final = "temporal_transport.interval_requested"
    INVALID_SEQUENCE: Final = "temporal_transport.invalid_sequence"
    INVALID_SPEC: Final = "temporal_transport.invalid_spec"
    MISSING_EVIDENCE: Final = "temporal_transport.missing_evidence"
    SEARCH_INCOMPLETE: Final = "temporal_transport.search_incomplete"
    TARGET_LAW_MISMATCH: Final = "temporal_transport.target_law_mismatch"
    TARGET_LAW_MISSING: Final = "temporal_transport.target_law_missing"


class tier_diagnostics:  # noqa: N801
    CANCELLED: Final = "tier_diagnostics.cancelled"
    EVALUE_EMBEDS_INTERVAL_LIMIT: Final = "tier_diagnostics.evalue_embeds_interval_limit"
    EVALUE_NOT_COMPUTED: Final = "tier_diagnostics.evalue_not_computed"
    NO_EVALUE_INTERVAL_METHOD: Final = "tier_diagnostics.no_evalue_interval_method"
    NOT_A_TIERED_AVERAGE_RESULT: Final = "tier_diagnostics.not_a_tiered_average_result"
    OVERLAP_NOT_CARRIED: Final = "tier_diagnostics.overlap_not_carried"
    UNKNOWN_SCENARIOS_NO_EVALUE: Final = "tier_diagnostics.unknown_scenarios_no_evalue"
    UNKNOWN_SCENARIOS_NO_OVERLAP: Final = "tier_diagnostics.unknown_scenarios_no_overlap"


class transport_revalidation:  # noqa: N801
    CLAIM_TARGET_MISMATCH: Final = "transport_revalidation.claim_target_mismatch"
    DUPLICATE_CLAIM: Final = "transport_revalidation.duplicate_claim"
    IDENTITY_MISMATCH: Final = "transport_revalidation.identity_mismatch"
    PROVIDER_TRUST_NOT_APPLICABLE: Final = "transport_revalidation.provider_trust_not_applicable"
    REPLAY_MISMATCH: Final = "transport_revalidation.replay_mismatch"
    SCENE_INVALID: Final = "transport_revalidation.scene_invalid"
    SOURCE_UNDECLARED: Final = "transport_revalidation.source_undeclared"
    UNKNOWN_CLAIM: Final = "transport_revalidation.unknown_claim"
    VALUE_DIGEST: Final = "transport_revalidation.value_digest"


class transport_sensitivity:  # noqa: N801
    BLANK_IDENTITY: Final = "transport_sensitivity.blank_identity"
    COORDINATE_OUT_OF_RANGE: Final = "transport_sensitivity.coordinate_out_of_range"
    INTERVAL_NOT_COMPOSED: Final = "transport_sensitivity.interval_not_composed"
    INVALID_BOUND: Final = "transport_sensitivity.invalid_bound"
    INVALID_GRID: Final = "transport_sensitivity.invalid_grid"
    LAW_RESPONSE_NOT_SCALAR: Final = "transport_sensitivity.law_response_not_scalar"
    NO_SCALAR_FAMILY: Final = "transport_sensitivity.no_scalar_family"
    NON_FINITE_POINT: Final = "transport_sensitivity.non_finite_point"
    NOT_TRANSPORTED: Final = "transport_sensitivity.not_transported"
    PROPORTIONAL_AT_ZERO: Final = "transport_sensitivity.proportional_at_zero"
    QUANTITY_NOT_IN_TARGET: Final = "transport_sensitivity.quantity_not_in_target"
    ROUTE_UNDECLARED: Final = "transport_sensitivity.route_undeclared"
    SURFACE_REFUSED: Final = "transport_sensitivity.surface_refused"


class transported_counterfactual:  # noqa: N801
    ARTIFACT_CHANGED: Final = "transported_counterfactual.artifact_changed"
    CROSS_WORLD_INDEPENDENCE_MISSING: Final = (
        "transported_counterfactual.cross_world_independence_missing"
    )
    FACTOR_MISSING: Final = "transported_counterfactual.factor_missing"
    INVALID_DIAGRAM: Final = "transported_counterfactual.invalid_diagram"
    INVALID_FACTOR: Final = "transported_counterfactual.invalid_factor"
    INVALID_IDENTITY: Final = "transported_counterfactual.invalid_identity"
    INVALID_LAW: Final = "transported_counterfactual.invalid_law"
    INVALID_MODEL: Final = "transported_counterfactual.invalid_model"
    INVALID_QUERY: Final = "transported_counterfactual.invalid_query"
    INVALID_REQUEST: Final = "transported_counterfactual.invalid_request"
    JOINT_THEOREM_NOT_DERIVED: Final = "transported_counterfactual.joint_theorem_not_derived"
    LIMITS_EXCEEDED: Final = "transported_counterfactual.limits_exceeded"
    NOISE_LAW_NOT_SHARED: Final = "transported_counterfactual.noise_law_not_shared"
    NONADDITIVE_MECHANISM: Final = "transported_counterfactual.nonadditive_mechanism"
    OVERLAP_FAILURE: Final = "transported_counterfactual.overlap_failure"
    PREREQUISITE_MISSING: Final = "transported_counterfactual.prerequisite_missing"
    SELECTION_ON_MECHANISM: Final = "transported_counterfactual.selection_on_mechanism"


class transported_decision:  # noqa: N801
    CALIBRATION_LABEL_INFLATED: Final = "transported_decision.calibration_label_inflated"
    COORDINATE_NOT_IN_TARGET: Final = "transported_decision.coordinate_not_in_target"
    EXCLUSIONS_LEAVE_NO_COMPARISON: Final = "transported_decision.exclusions_leave_no_comparison"
    INTERVAL_NOT_SUPPORTED: Final = "transported_decision.interval_not_supported"
    INVALID_VALUE: Final = "transported_decision.invalid_value"
    LAW_NOT_SUPPLIED: Final = "transported_decision.law_not_supplied"
    MIXED_TRANSPORTS: Final = "transported_decision.mixed_transports"
    NATIVE_LABEL_ON_TRANSPORTED_VALUE: Final = (
        "transported_decision.native_label_on_transported_value"
    )
    NOT_TRANSPORTED: Final = "transported_decision.not_transported"
    ROUTE_NEEDS_TYPED_SOURCE: Final = "transported_decision.route_needs_typed_source"
    SUPPORT_LABEL_INFLATED: Final = "transported_decision.support_label_inflated"
    SUPPORT_RULE_UNMET: Final = "transported_decision.support_rule_unmet"
    TRANSPORT_IDENTITY_INCOMPLETE: Final = "transported_decision.transport_identity_incomplete"
    TRANSPORT_IDENTITY_NOT_DIGEST: Final = "transported_decision.transport_identity_not_digest"
    TRANSPORTED_NOT_ACCEPTED: Final = "transported_decision.transported_not_accepted"
    UNCERTAINTY_REQUIREMENT: Final = "transported_decision.uncertainty_requirement"


class transported_decision_artifact:  # noqa: N801
    BOUNDS_EXCEEDED: Final = "transported_decision_artifact.bounds_exceeded"
    IDENTITY_EXPECTED: Final = "transported_decision_artifact.identity_expected"
    INVALID: Final = "transported_decision_artifact.invalid"
    LAW_NOT_EXPORTABLE: Final = "transported_decision_artifact.law_not_exportable"
    REPLAY_MISMATCH: Final = "transported_decision_artifact.replay_mismatch"


class transported_effect:  # noqa: N801
    COHORT_INVALID: Final = "transported_effect.cohort_invalid"
    COHORT_NOT_NUMERIC: Final = "transported_effect.cohort_not_numeric"
    DECISION_SOURCE_MISMATCH: Final = "transported_effect.decision_source_mismatch"
    DECLARATION_INVALID: Final = "transported_effect.declaration_invalid"
    FEATURE_SCHEMA_MISMATCH: Final = "transported_effect.feature_schema_mismatch"
    IDENTITY_MISMATCH: Final = "transported_effect.identity_mismatch"
    INTERVAL_NOT_SUPPORTED: Final = "transported_effect.interval_not_supported"
    INVALID_REQUEST: Final = "transported_effect.invalid_request"
    LAW_NOT_SUPPLIED: Final = "transported_effect.law_not_supplied"
    LINK_NOT_IDENTITY: Final = "transported_effect.link_not_identity"
    MODEL_INVALID: Final = "transported_effect.model_invalid"
    MODEL_NOT_PORTABLE: Final = "transported_effect.model_not_portable"
    MODIFIER_SET_MISMATCH: Final = "transported_effect.modifier_set_mismatch"
    NO_OVERLAP: Final = "transported_effect.no_overlap"
    NON_FINITE_RESULT: Final = "transported_effect.non_finite_result"
    OUT_OF_SUPPORT: Final = "transported_effect.out_of_support"
    OUTCOME_UNITS_REQUIRED: Final = "transported_effect.outcome_units_required"
    POPULATION_NOT_DIFFERENT: Final = "transported_effect.population_not_different"
    PREDICTION_REFUSED: Final = "transported_effect.prediction_refused"
    SELECTION_ENTERS_MECHANISM: Final = "transported_effect.selection_enters_mechanism"
    SHIFT_DECLARATION_MISMATCH: Final = "transported_effect.shift_declaration_mismatch"
    SHIFT_DECLARATION_REQUIRED: Final = "transported_effect.shift_declaration_required"
    SUPPORT_CONTRACT_ABSENT: Final = "transported_effect.support_contract_absent"
    WEIGHTS_INVALID: Final = "transported_effect.weights_invalid"
    WEIGHTS_NOT_NORMALIZED: Final = "transported_effect.weights_not_normalized"


class transported_policy:  # noqa: N801
    DIAGRAM_BINDING_MISMATCH: Final = "transported_policy.diagram_binding_mismatch"
    INITIAL_LAW_DECLARED_INVARIANT: Final = "transported_policy.initial_law_declared_invariant"
    MECHANISM_DECLARED_TWICE: Final = "transported_policy.mechanism_declared_twice"
    MECHANISM_NOT_INVARIANT: Final = "transported_policy.mechanism_not_invariant"
    MECHANISM_STATUS_MISSING: Final = "transported_policy.mechanism_status_missing"
    MECHANISM_UNKNOWN: Final = "transported_policy.mechanism_unknown"
    SELECTION_DIAGRAM_MISSING: Final = "transported_policy.selection_diagram_missing"
    TARGET_INITIAL_LAW: Final = "transported_policy.target_initial_law"
    TARGET_INITIAL_LAW_MISSING: Final = "transported_policy.target_initial_law_missing"
    TARGET_IS_SOURCE: Final = "transported_policy.target_is_source"
    TARGET_POPULATION: Final = "transported_policy.target_population"
    TARGET_SUPPORT_OUTSIDE_SOURCE: Final = "transported_policy.target_support_outside_source"
    TARGET_UTILITY_DIGEST: Final = "transported_policy.target_utility_digest"
    TARGET_UTILITY_MISSING: Final = "transported_policy.target_utility_missing"
    TARGET_UTILITY_PROVENANCE_MISSING: Final = (
        "transported_policy.target_utility_provenance_missing"
    )
    TARGET_UTILITY_SUPPORT: Final = "transported_policy.target_utility_support"
    TARGET_UTILITY_VALUE: Final = "transported_policy.target_utility_value"
    UTILITY_DECLARED_INVARIANT: Final = "transported_policy.utility_declared_invariant"
    VALUE_MISMATCH: Final = "transported_policy.value_mismatch"


class transported_synthesis:  # noqa: N801
    NATIVE_POPULATION_BLANK: Final = "transported_synthesis.native_population_blank"
    NO_CONTRIBUTIONS: Final = "transported_synthesis.no_contributions"
    NUMERIC_REFUSED: Final = "transported_synthesis.numeric_refused"
    OVERLAP_NOT_MODELLED: Final = "transported_synthesis.overlap_not_modelled"
    PRIOR_ROLE_MISMATCH: Final = "transported_synthesis.prior_role_mismatch"
    SHARED_TRANSPORT_NEEDS_COVARIANCE: Final = (
        "transported_synthesis.shared_transport_needs_covariance"
    )
    SOURCE_LISTED_TWICE: Final = "transported_synthesis.source_listed_twice"
    TARGET_POPULATION_MISMATCH: Final = "transported_synthesis.target_population_mismatch"
    TRANSPORT_IDENTITY_INCOMPLETE: Final = "transported_synthesis.transport_identity_incomplete"
    TRANSPORT_VARIANCE_ON_NATIVE_SOURCE: Final = (
        "transported_synthesis.transport_variance_on_native_source"
    )
    TRANSPORT_VARIANCE_UNDECLARED: Final = "transported_synthesis.transport_variance_undeclared"
    UNKNOWN_SOURCE: Final = "transported_synthesis.unknown_source"


class treatment_levels:  # noqa: N801
    CANCELLED: Final = "treatment_levels.cancelled"
    CONTRAST_LEVEL_NOT_DECLARED: Final = "treatment_levels.contrast_level_not_declared"
    FULL_POPULATION_UNDECLARED_LEVEL: Final = "treatment_levels.full_population_undeclared_level"
    INVALID_DATA: Final = "treatment_levels.invalid_data"
    INVALID_LEVEL_SET: Final = "treatment_levels.invalid_level_set"
    INVALID_REQUEST: Final = "treatment_levels.invalid_request"
    LEVEL_NOT_POPULATED: Final = "treatment_levels.level_not_populated"
    LEVELS_MISSING: Final = "treatment_levels.levels_missing"
    SPARSE_LEVEL: Final = "treatment_levels.sparse_level"
    TRAINING_FOLD_MISSING_LEVEL: Final = "treatment_levels.training_fold_missing_level"
    UNDECLARED_LEVEL: Final = "treatment_levels.undeclared_level"


class two_step_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "two_step_posterior.band_not_constructed"
    EMPTY_LAW: Final = "two_step_posterior.empty_law"
    EXTERNAL_POWER_PRIOR: Final = "two_step_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "two_step_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "two_step_posterior.non_positive_scale"


class typed_mechanism_transport:  # noqa: N801
    CYCLE: Final = "typed_mechanism_transport.cycle"
    DEPENDENCE_INVALID: Final = "typed_mechanism_transport.dependence_invalid"
    DESIGN_NOT_SHARED: Final = "typed_mechanism_transport.design_not_shared"
    FIT_NOT_CONVERGED: Final = "typed_mechanism_transport.fit_not_converged"
    FIT_REFUSED: Final = "typed_mechanism_transport.fit_refused"
    INVARIANCE_CONTRADICTED: Final = "typed_mechanism_transport.invariance_contradicted"
    INVARIANCE_LEVEL_INVALID: Final = "typed_mechanism_transport.invariance_level_invalid"
    PARAMETER_DEPENDENCE_INCOMPLETE: Final = (
        "typed_mechanism_transport.parameter_dependence_incomplete"
    )
    PARAMETER_DEPENDENCE_UNKNOWN: Final = "typed_mechanism_transport.parameter_dependence_unknown"
    PARENT_WITHOUT_MECHANISM: Final = "typed_mechanism_transport.parent_without_mechanism"
    PARENTS_CHANGED: Final = "typed_mechanism_transport.parents_changed"
    REGIME_NOT_DECLARED_INVARIANT: Final = "typed_mechanism_transport.regime_not_declared_invariant"
    SELECTION_TARGET_UNKNOWN_VARIABLE: Final = (
        "typed_mechanism_transport.selection_target_unknown_variable"
    )
    SHIFTED_WITHOUT_TARGET_EVIDENCE: Final = (
        "typed_mechanism_transport.shifted_without_target_evidence"
    )
    TARGET_SPECIFICATION_DIFFERS: Final = "typed_mechanism_transport.target_specification_differs"
    TRANSPORT_IDENTITY_MISMATCH: Final = "typed_mechanism_transport.transport_identity_mismatch"
    UNKNOWN_VARIABLE: Final = "typed_mechanism_transport.unknown_variable"


class vector_posterior:  # noqa: N801
    BAND_NOT_CONSTRUCTED: Final = "vector_posterior.band_not_constructed"
    EXTERNAL_POWER_PRIOR: Final = "vector_posterior.external_power_prior"
    MAPPED_PRIOR: Final = "vector_posterior.mapped_prior"
    NON_POSITIVE_SCALE: Final = "vector_posterior.non_positive_scale"


class vector_treatment:  # noqa: N801
    ADJUSTMENT_SET_MISMATCH: Final = "vector_treatment.adjustment_set_mismatch"
    CANCELLED: Final = "vector_treatment.cancelled"
    CLUSTER_IDS_REQUIRED: Final = "vector_treatment.cluster_ids_required"
    CLUSTER_UNREAD: Final = "vector_treatment.cluster_unread"
    COLLINEAR_TREATMENTS: Final = "vector_treatment.collinear_treatments"
    DATA_NOT_BOUND: Final = "vector_treatment.data_not_bound"
    DEGENERATE_COVARIANCE: Final = "vector_treatment.degenerate_covariance"
    DUPLICATE_CONTRAST_COEFFICIENT: Final = "vector_treatment.duplicate_contrast_coefficient"
    DUPLICATE_NAME: Final = "vector_treatment.duplicate_name"
    EMPTY_CONTRAST: Final = "vector_treatment.empty_contrast"
    EXPORT_BUDGET_EXCEEDED: Final = "vector_treatment.export_budget_exceeded"
    INVALID_IDENTITY: Final = "vector_treatment.invalid_identity"
    INVALID_LAG: Final = "vector_treatment.invalid_lag"
    INVALID_REQUEST: Final = "vector_treatment.invalid_request"
    LAG_REQUIRED: Final = "vector_treatment.lag_required"
    LAG_UNREAD: Final = "vector_treatment.lag_unread"
    LEVERAGE_ONE: Final = "vector_treatment.leverage_one"
    LIMITS_EXCEEDED: Final = "vector_treatment.limits_exceeded"
    MEMORY_BUDGET_EXCEEDED: Final = "vector_treatment.memory_budget_exceeded"
    NON_FINITE_VALUE: Final = "vector_treatment.non_finite_value"
    RANK_DEFICIENT_ADJUSTMENT: Final = "vector_treatment.rank_deficient_adjustment"
    ROW_COUNT_MISMATCH: Final = "vector_treatment.row_count_mismatch"
    ROW_SNAPSHOT_MISMATCH: Final = "vector_treatment.row_snapshot_mismatch"
    SUMMARY_INCONSISTENT: Final = "vector_treatment.summary_inconsistent"
    SUMMARY_MODEL_BASED_ONLY: Final = "vector_treatment.summary_model_based_only"
    TOO_FEW_CLUSTERS: Final = "vector_treatment.too_few_clusters"
    TOO_FEW_ROWS: Final = "vector_treatment.too_few_rows"
    TOO_FEW_TREATMENTS: Final = "vector_treatment.too_few_treatments"
    TOO_MANY_TREATMENTS: Final = "vector_treatment.too_many_treatments"
    TREATMENT_WITHOUT_VARIATION: Final = "vector_treatment.treatment_without_variation"
    UNKNOWN_CONTRAST_COEFFICIENT: Final = "vector_treatment.unknown_contrast_coefficient"
    WEIGHTS_NEED_ROBUST_COVARIANCE: Final = "vector_treatment.weights_need_robust_covariance"
    WRONG_CONTRACT: Final = "vector_treatment.wrong_contract"


ALL: Final[tuple[str, ...]] = (
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
    "graph_selection_imputation.interval_unmeasured",
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
    "mnar_sensitivity.bounds_exceeded",
    "mnar_sensitivity.composition_not_supported",
    "mnar_sensitivity.covariates_not_used",
    "mnar_sensitivity.covariates_required",
    "mnar_sensitivity.design_singular",
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
    "nested_markov.outside_binary_pilot",
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
    "scientific_bridge.propagation_overflow",
    "scientific_bridge.provenance",
    "scientific_bridge.proxy_as_construct",
    "scientific_bridge.quantity_mismatch",
    "scientific_bridge.regime_mismatch",
    "scientific_bridge.replay_mismatch",
    "scientific_bridge.scientific_endpoint_as_utility",
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
)

CODES: Final[dict[str, tuple[str, ...]]] = {
    "adaptive_inference.analysis_invalid": ("invalid_argument",),
    "adaptive_inference.analysis_route_mismatch": ("adaptive_assumption_violated",),
    "adaptive_inference.anytime_premise_contexts": ("adaptive_assumption_violated",),
    "adaptive_inference.anytime_settings_invalid": ("invalid_argument",),
    "adaptive_inference.anytime_settings_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.arm_not_observed": ("adaptive_assumption_violated",),
    "adaptive_inference.arm_out_of_range": ("invalid_argument",),
    "adaptive_inference.arms_invalid": ("invalid_argument",),
    "adaptive_inference.artifact_size_exceeded": ("route_not_supported",),
    "adaptive_inference.assigned_probability_zero": ("invalid_argument",),
    "adaptive_inference.assignment_not_replayed": ("adaptive_assumption_violated",),
    "adaptive_inference.assignment_probabilities_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.batch_incomplete": ("adaptive_assumption_violated",),
    "adaptive_inference.between_unit_variance_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.borrowing_fixed_after_first_look": ("adaptive_assumption_violated",),
    "adaptive_inference.cancelled": ("cancelled_no_claim",),
    "adaptive_inference.cell_incoherent": ("route_not_supported",),
    "adaptive_inference.cell_not_delivered": ("route_not_supported",),
    "adaptive_inference.choice_outcome_needs_choice_contract": ("invalid_argument",),
    "adaptive_inference.contexts_invalid": ("invalid_argument",),
    "adaptive_inference.correlation_invalid": ("invalid_argument",),
    "adaptive_inference.declared_looks_exceeded": ("invalid_argument",),
    "adaptive_inference.declared_units_exceeded": ("invalid_argument",),
    "adaptive_inference.delay_not_ignorable": ("adaptive_assumption_violated",),
    "adaptive_inference.exact_law_states_exceeded": ("invalid_argument",),
    "adaptive_inference.exact_law_unavailable": ("adaptive_assumption_violated",),
    "adaptive_inference.exact_law_work_exceeded": ("invalid_argument",),
    "adaptive_inference.exposure_differs_from_assignment": ("adaptive_assumption_violated",),
    "adaptive_inference.exposure_history_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.exposure_invalid": ("invalid_argument",),
    "adaptive_inference.final_analysis_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.fixed_sample_data_dependent_stopping": ("adaptive_assumption_violated",),
    "adaptive_inference.fixed_sample_design_adaptive": ("adaptive_assumption_violated",),
    "adaptive_inference.fixed_sample_on_adaptive_data": ("adaptive_design_ignored",),
    "adaptive_inference.group_sequential_assignment_adaptive": ("adaptive_assumption_violated",),
    "adaptive_inference.guarantee_not_transferable": ("adaptive_assumption_violated",),
    "adaptive_inference.historical_borrowing_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.historical_prior_invalid": ("invalid_argument",),
    "adaptive_inference.independence_assumed_for_dependent_measurements": (
        "adaptive_assumption_violated",
    ),
    "adaptive_inference.input_invalid": ("invalid_argument",),
    "adaptive_inference.interim_analyses_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.interval_names_no_quantity": ("quantity_semantics_mismatch",),
    "adaptive_inference.inversion_premise_contexts": ("adaptive_assumption_violated",),
    "adaptive_inference.ledger_limit": ("invalid_argument",),
    "adaptive_inference.look_out_of_range": ("invalid_argument",),
    "adaptive_inference.looks_not_ordered": ("invalid_argument",),
    "adaptive_inference.looks_overlap_unknown": ("evidence_overlap_unknown",),
    "adaptive_inference.looks_share_observations": ("evidence_double_counted",),
    "adaptive_inference.measurements_invalid": ("invalid_argument",),
    "adaptive_inference.no_stagewise_adjustment_procedure": ("route_not_supported",),
    "adaptive_inference.no_stopping_reliability_procedure": ("estimator_inference_mismatch",),
    "adaptive_inference.numerical_failure": ("invalid_argument",),
    "adaptive_inference.outcome_availability_invalid": ("invalid_argument",),
    "adaptive_inference.outcome_availability_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.outcome_dependent_delay": ("adaptive_assumption_violated",),
    "adaptive_inference.outcome_invalid": ("invalid_argument",),
    "adaptive_inference.outcome_structure_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.pairing_not_supported": ("route_not_supported",),
    "adaptive_inference.paradigm_not_supported_for_cell": ("route_not_supported",),
    "adaptive_inference.paradigm_rule_mismatch": ("adaptive_assumption_violated",),
    "adaptive_inference.policy_history_invalid": ("invalid_argument",),
    "adaptive_inference.policy_history_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.policy_selection_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.positivity_floor": ("adaptive_assumption_violated",),
    "adaptive_inference.posterior_premise_outcome_dependent_delay": (
        "adaptive_assumption_violated",
    ),
    "adaptive_inference.probabilities_invalid": ("invalid_argument",),
    "adaptive_inference.probability_floor_invalid": ("invalid_argument",),
    "adaptive_inference.procedure_limit_exceeded": ("invalid_argument",),
    "adaptive_inference.repeated_measures_paradigm_not_derived": ("adaptive_assumption_violated",),
    "adaptive_inference.repeated_measures_stopping_not_derived": ("invalid_argument",),
    "adaptive_inference.report_replay_mismatch": ("invalid_argument",),
    "adaptive_inference.retained_record_exceeded": ("invalid_argument",),
    "adaptive_inference.rule_mismatch": ("invalid_argument",),
    "adaptive_inference.selection_not_ignorable": ("adaptive_assumption_violated",),
    "adaptive_inference.shape_mismatch": ("invalid_argument",),
    "adaptive_inference.simulation_memory_exceeded": ("invalid_argument",),
    "adaptive_inference.simulation_replications_exceeded": ("invalid_argument",),
    "adaptive_inference.simulation_work_exceeded": ("invalid_argument",),
    "adaptive_inference.stopping_not_replayed": ("adaptive_assumption_violated",),
    "adaptive_inference.stopping_rule_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.structure_incomplete": ("adaptive_assumption_violated",),
    "adaptive_inference.unit_out_of_range": ("invalid_argument",),
    "adaptive_inference.variance_invalid": ("invalid_argument",),
    "adaptive_inference.within_unit_correlation_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.within_unit_variance_missing": ("adaptive_contract_incomplete",),
    "adaptive_inference.wrong_contract": ("route_not_supported",),
    "adaptive_transport.arms": ("invalid_argument",),
    "adaptive_transport.assignment_below_floor": ("adaptive_assumption_violated",),
    "adaptive_transport.assignment_monitoring_contract_missing": ("adaptive_contract_incomplete",),
    "adaptive_transport.design_not_adaptive": ("route_not_supported",),
    "adaptive_transport.diagram_binding_mismatch": ("transport_missing_evidence",),
    "adaptive_transport.mechanism_declared_twice": ("invalid_argument",),
    "adaptive_transport.mechanism_status_missing": ("transport_missing_evidence",),
    "adaptive_transport.no_analysed_units": ("transport_support_failure",),
    "adaptive_transport.outcome_mechanism_not_invariant": ("transport_missing_evidence",),
    "adaptive_transport.outcome_missing": ("adaptive_contract_incomplete",),
    "adaptive_transport.population_label": ("invalid_argument",),
    "adaptive_transport.selection_diagram_missing": ("transport_missing_evidence",),
    "adaptive_transport.strata": ("invalid_argument",),
    "adaptive_transport.stratum_arm_unobserved": ("transport_support_failure",),
    "adaptive_transport.stratum_law_declared_invariant": ("invalid_argument",),
    "adaptive_transport.target_is_source": ("invalid_argument",),
    "adaptive_transport.target_stratum_law": ("invalid_argument",),
    "adaptive_transport.target_stratum_law_missing": ("transport_missing_evidence",),
    "adaptive_transport.unit_strata": ("invalid_argument",),
    "adaptive_transport.value_mismatch": ("invalid_argument",),
    "admg_conditional_posterior.band_not_constructed": ("route_not_supported",),
    "admg_conditional_posterior.empty_law": ("invalid_argument",),
    "admg_conditional_posterior.external_power_prior": ("route_not_supported",),
    "admg_conditional_posterior.mapped_prior": ("route_not_supported",),
    "admg_conditional_posterior.non_positive_scale": ("invalid_argument",),
    "admg_transport.bounds_exceeded": ("route_not_supported",),
    "admg_transport.budget": ("transport_budget_cancel",),
    "admg_transport.consumer_limits": ("route_not_supported",),
    "admg_transport.counted_laws_unsupported": ("route_not_supported",),
    "admg_transport.data_identity_mismatch": ("transport_not_certified",),
    "admg_transport.evidence_mismatch": ("transport_not_certified",),
    "admg_transport.invalid_artifact": ("invalid_argument",),
    "admg_transport.invalid_catalog": ("invalid_argument",),
    "admg_transport.invalid_derivation": ("transport_not_certified",),
    "admg_transport.invalid_query": ("invalid_argument",),
    "admg_transport.invalid_request": ("invalid_argument",),
    "admg_transport.missing_evidence": ("transport_missing_evidence",),
    "admg_transport.not_certified": ("transport_not_certified",),
    "admg_transport.premises_mismatch": ("transport_not_certified",),
    "admg_transport.proven_non_transportable": ("transport_proven_non_transportable",),
    "admg_transport.replay_mismatch": ("transport_not_certified",),
    "admg_transport.support_failure": ("transport_support_failure",),
    "aligned_joint_draws.identity_expected": ("quantity_semantics_mismatch",),
    "aligned_joint_draws.legacy_binding": ("quantity_semantics_mismatch",),
    "aligned_joint_draws.legacy_source": ("quantity_semantics_mismatch",),
    "aligned_joint_draws.marginals_not_joint": ("joint_law_required",),
    "aligned_joint_draws.outcome_coordinate": ("quantity_semantics_mismatch",),
    "batch_retarget.cancelled": ("cancelled_no_claim",),
    "batch_retarget.contrast_member_failed": ("route_not_supported",),
    "batch_retarget.covariance_unavailable": ("route_not_supported",),
    "batch_retarget.duplicate_name": ("invalid_argument",),
    "batch_retarget.empty_family": ("invalid_argument",),
    "batch_retarget.functional_not_supported": ("route_not_supported",),
    "batch_retarget.incompatible_target": ("invalid_argument",),
    "batch_retarget.invalid_contrast": ("invalid_argument",),
    "batch_retarget.max_t_draws_out_of_range": ("invalid_argument",),
    "batch_retarget.max_t_invalid_level": ("invalid_argument",),
    "batch_retarget.mixed_snapshot": ("row_weights_bound_to_snapshot",),
    "batch_retarget.partial_family": ("route_not_supported",),
    "batch_retarget.point_only_member": ("route_not_supported",),
    "batch_retarget.scores_mismatch": ("invalid_argument",),
    "batch_retarget.scores_unavailable": ("score_table_unavailable",),
    "batch_retarget.scores_unavailable_after_estimate": ("score_table_unavailable",),
    "batch_retarget.unknown_claim": ("invalid_argument",),
    "batch_retarget.weighted_overlap_failed": ("positivity_violated",),
    "batch_retarget_posterior.band_not_constructed": ("route_not_supported",),
    "batch_retarget_posterior.empty_scores": ("invalid_argument",),
    "batch_retarget_posterior.external_power_prior": ("route_not_supported",),
    "batch_retarget_posterior.mapped_prior": ("route_not_supported",),
    "batch_retarget_posterior.non_positive_scale": ("invalid_argument",),
    "bayesian_transport.admission_refused": ("transport_budget_cancel",),
    "bayesian_transport.cancelled": ("transport_budget_cancel",),
    "bayesian_transport.feature_mismatch": ("invalid_argument",),
    "bayesian_transport.identification_not_certified": ("transport_missing_evidence",),
    "bayesian_transport.identification_required": ("transport_missing_evidence",),
    "bayesian_transport.invalid_input": ("invalid_argument",),
    "bayesian_transport.invalid_prior": ("invalid_argument",),
    "bayesian_transport.missing_law": ("joint_law_required",),
    "bayesian_transport.missing_source": ("transport_missing_evidence",),
    "bayesian_transport.prior_data_overlap": ("invalid_argument",),
    "bayesian_transport.prior_dimension": ("prior_dimension_mismatch",),
    "bayesian_transport.singular_fit": ("design_rank_deficient",),
    "bayesian_transport.source_dependence": ("sampling_dependence_unknown",),
    "bayesian_transport.support_resource_limit": ("execution_resource_limit",),
    "bayesian_transport.too_many_draws": ("invalid_argument",),
    "bayesian_transport.too_many_parameters": ("invalid_argument",),
    "bayesian_transport.unsupported_graph": ("route_not_supported",),
    "bayesian_transport.weak_overlap": ("transport_support_failure",),
    "binary_recovery_posterior.band_not_constructed": ("route_not_supported",),
    "binary_recovery_posterior.empty_law": ("invalid_argument",),
    "binary_recovery_posterior.external_power_prior": ("route_not_supported",),
    "binary_recovery_posterior.mapped_prior": ("route_not_supported",),
    "binary_recovery_posterior.non_positive_scale": ("invalid_argument",),
    "callback_protocol.artifact_invalid": ("invalid_argument",),
    "callback_protocol.blank_field": ("callback_protocol_invalid",),
    "callback_protocol.cancelled": ("cancelled_no_claim",),
    "callback_protocol.diagnostic_only": ("callback_reuse_ineligible",),
    "callback_protocol.duplicate_name": ("callback_protocol_invalid",),
    "callback_protocol.hidden_state_unobservable": ("callback_reuse_ineligible",),
    "callback_protocol.identification_not_conferred": ("callback_protocol_invalid",),
    "callback_protocol.identity_mismatch": ("route_not_supported",),
    "callback_protocol.interval_not_supported": ("uncertainty_representation_insufficient",),
    "callback_protocol.invalid_quantity": ("callback_protocol_invalid",),
    "callback_protocol.invalid_support": ("callback_protocol_invalid",),
    "callback_protocol.invocation_limit": ("callback_limit_exceeded",),
    "callback_protocol.invocation_outside_execution": ("callback_side_effect_forbidden",),
    "callback_protocol.joint_law_not_conferred": ("joint_law_required",),
    "callback_protocol.meaning_not_for_kind": ("callback_protocol_invalid",),
    "callback_protocol.native_trust_claimed": ("callback_protocol_invalid",),
    "callback_protocol.native_trust_not_conferred": ("callback_protocol_invalid",),
    "callback_protocol.nested_input_overlap_unknown": ("sampling_dependence_unknown",),
    "callback_protocol.nested_input_unregistered": ("sampling_dependence_unknown",),
    "callback_protocol.nested_inputs_mismatch": ("callback_protocol_invalid",),
    "callback_protocol.nested_inputs_share_evidence": ("evidence_double_counted",),
    "callback_protocol.no_idempotency_contract": ("callback_reuse_ineligible",),
    "callback_protocol.no_operations": ("callback_protocol_invalid",),
    "callback_protocol.opaque_dependencies": ("callback_reuse_ineligible",),
    "callback_protocol.operation_not_declared": ("external_capability_missing",),
    "callback_protocol.operation_not_for_kind": ("callback_protocol_invalid",),
    "callback_protocol.output_not_a_law": ("callback_output_invalid",),
    "callback_protocol.output_not_finite": ("callback_output_invalid",),
    "callback_protocol.output_outside_support": ("callback_output_invalid",),
    "callback_protocol.output_shape": ("callback_output_invalid",),
    "callback_protocol.output_too_large": ("callback_limit_exceeded",),
    "callback_protocol.posterior_guarantee_not_supported": (
        "uncertainty_representation_insufficient",
    ),
    "callback_protocol.provider_failed": ("callback_provider_unavailable",),
    "callback_protocol.provider_identity_mismatch": ("external_binding_mismatch",),
    "callback_protocol.provider_unavailable": ("callback_provider_unavailable",),
    "callback_protocol.query_invalid": ("callback_output_invalid",),
    "callback_protocol.replay_contract_incomplete": ("callback_protocol_invalid",),
    "callback_protocol.request_mismatch": ("external_binding_mismatch",),
    "callback_protocol.retry_outside_policy": ("callback_side_effect_forbidden",),
    "callback_protocol.side_effects_not_allowed": ("callback_side_effect_forbidden",),
    "callback_protocol.too_many_items": ("callback_protocol_invalid",),
    "callback_protocol.trust_incomplete": ("callback_protocol_invalid",),
    "callback_protocol.update_declaration_mismatch": ("callback_protocol_invalid",),
    "callback_protocol.wall_limit_exceeded": ("callback_limit_exceeded",),
    "categorical_posterior.band_not_constructed": ("route_not_supported",),
    "categorical_posterior.external_power_prior": ("route_not_supported",),
    "categorical_posterior.mapped_prior": ("route_not_supported",),
    "categorical_posterior.non_positive_scale": ("invalid_argument",),
    "categorical_treatment.absent_level": ("arm_not_populated",),
    "categorical_treatment.cancelled": ("cancelled_no_claim",),
    "categorical_treatment.data_not_bound": ("route_not_supported",),
    "categorical_treatment.export_budget_exceeded": ("route_not_supported",),
    "categorical_treatment.invalid_identity": ("invalid_argument",),
    "categorical_treatment.invalid_level": ("invalid_argument",),
    "categorical_treatment.invalid_min_level_rows": ("invalid_argument",),
    "categorical_treatment.invalid_pair": ("invalid_argument",),
    "categorical_treatment.invalid_request": ("invalid_argument",),
    "categorical_treatment.limits_exceeded": ("invalid_argument",),
    "categorical_treatment.memory_budget_exceeded": ("route_not_supported",),
    "categorical_treatment.monotonicity_requires_ordered": ("route_not_supported",),
    "categorical_treatment.posterior_declared_rows": ("route_not_supported",),
    "categorical_treatment.row_count_mismatch": ("route_not_supported",),
    "categorical_treatment.sparse_level": ("arm_not_populated",),
    "categorical_treatment.summary_inconsistent": ("invalid_argument",),
    "categorical_treatment.too_few_levels": ("invalid_argument",),
    "categorical_treatment.undeclared_level": ("invalid_argument",),
    "categorical_treatment.unknown_level": ("invalid_argument",),
    "categorical_treatment.unknown_reference": ("invalid_argument",),
    "categorical_treatment.wrong_contract": ("route_not_supported",),
    "chain_recovery_posterior.band_not_constructed": ("route_not_supported",),
    "chain_recovery_posterior.empty_law": ("invalid_argument",),
    "chain_recovery_posterior.external_power_prior": ("route_not_supported",),
    "chain_recovery_posterior.mapped_prior": ("route_not_supported",),
    "chain_recovery_posterior.non_positive_scale": ("invalid_argument",),
    "cluster_dml.cancelled": ("cancelled_no_claim",),
    "cluster_dml.cluster_ids_missing": ("required_option_missing",),
    "cluster_dml.dyadic_giant_component": ("dyadic_dependence_not_supported",),
    "cluster_dml.dyadic_shared_namespace": ("dyadic_dependence_not_supported",),
    "cluster_dml.flexible_learner_not_supported": ("route_not_supported",),
    "cluster_dml.invalid_min_clusters": ("invalid_argument",),
    "cluster_dml.invalid_min_components": ("invalid_argument",),
    "cluster_dml.no_interval_method": ("cluster_interval_not_supported",),
    "cluster_dml.penalized_propensity": ("route_not_supported",),
    "cluster_dml.scope": ("route_not_supported",),
    "cluster_dml.second_ids_missing": ("required_option_missing",),
    "cluster_dml.second_ids_unexpected": ("invalid_argument",),
    "cluster_dml.too_few_clusters": ("too_few_clusters",),
    "cluster_dml.too_few_components": ("too_few_clusters",),
    "clustered_posterior.band_not_constructed": ("route_not_supported",),
    "clustered_posterior.empty_scores": ("invalid_argument",),
    "clustered_posterior.external_power_prior": ("route_not_supported",),
    "clustered_posterior.mapped_prior": ("route_not_supported",),
    "clustered_posterior.non_positive_scale": ("invalid_argument",),
    "cohort_decision.cancelled": ("invalid_argument",),
    "cohort_decision.cohort_changed": ("invalid_argument",),
    "cohort_decision.cohort_empty": ("invalid_argument",),
    "cohort_decision.cohort_invalid": ("invalid_argument",),
    "cohort_decision.cohort_too_large": ("invalid_argument",),
    "cohort_decision.container_checksum_mismatch": ("invalid_argument",),
    "cohort_decision.container_invalid": ("invalid_argument",),
    "cohort_decision.contrast_mismatch": ("quantity_semantics_mismatch",),
    "cohort_decision.contrast_not_declared": ("quantity_semantics_mismatch",),
    "cohort_decision.extrapolative_not_accepted": ("not_supported",),
    "cohort_decision.feature_schema_mismatch": ("schema_mismatch",),
    "cohort_decision.feature_type_mismatch": ("schema_mismatch",),
    "cohort_decision.fitted_model_absent": ("route_not_supported",),
    "cohort_decision.fitted_model_invalid": ("invalid_argument",),
    "cohort_decision.identity_unexpected": ("invalid_argument",),
    "cohort_decision.interval_level_invalid": ("invalid_argument",),
    "cohort_decision.interval_not_supported": ("route_not_supported",),
    "cohort_decision.interval_requires_native_fit": ("route_not_supported",),
    "cohort_decision.invalid_request": ("invalid_argument",),
    "cohort_decision.joint_law_required": ("joint_law_required",),
    "cohort_decision.law_not_supplied": ("decision_contract_unsatisfied",),
    "cohort_decision.layout_invalid": ("invalid_argument",),
    "cohort_decision.link_not_identity": ("quantity_semantics_mismatch",),
    "cohort_decision.missing_artifact_id": ("invalid_argument",),
    "cohort_decision.model_identity_mismatch": ("invalid_argument",),
    "cohort_decision.native_fit_refused": ("invalid_argument",),
    "cohort_decision.negative_variance": ("invalid_argument",),
    "cohort_decision.non_finite_result": ("invalid_argument",),
    "cohort_decision.outcome_units_required": ("invalid_argument",),
    "cohort_decision.outside_support": ("invalid_argument",),
    "cohort_decision.oversized": ("invalid_argument",),
    "cohort_decision.parent_attested_not_native": ("invalid_argument",),
    "cohort_decision.parent_not_verified": ("invalid_argument",),
    "cohort_decision.parent_query_unsupported": ("invalid_argument",),
    "cohort_decision.population_mismatch": ("quantity_semantics_mismatch",),
    "cohort_decision.prediction_refused": ("invalid_argument",),
    "cohort_decision.replay_mismatch": ("invalid_argument",),
    "cohort_decision.shift_declaration_mismatch": ("transport_not_certified",),
    "cohort_decision.shift_declaration_required": ("transport_missing_evidence",),
    "cohort_decision.shift_declaration_unneeded": ("quantity_semantics_mismatch",),
    "cohort_decision.shift_support_failure": ("transport_support_failure",),
    "cohort_decision.source_invalid": ("invalid_argument",),
    "cohort_decision.support_contract_absent": ("invalid_argument",),
    "cohort_decision.units_mismatch": ("quantity_semantics_mismatch",),
    "cohort_decision.unsupported_version": ("invalid_argument",),
    "cohort_decision.weights_invalid": ("invalid_argument",),
    "cohort_decision.weights_not_normalized": ("invalid_argument",),
    "cohort_decision.wrong_parent_identity": ("invalid_argument",),
    "compact_export.container_checksum_mismatch": ("invalid_argument",),
    "compact_export.container_invalid": ("invalid_argument",),
    "compact_export.covariance_asymmetric": ("invalid_argument",),
    "compact_export.covariance_not_psd": ("invalid_argument",),
    "compact_export.data_digest_mismatch": ("invalid_argument",),
    "compact_export.dimension_mismatch": ("invalid_argument",),
    "compact_export.duplicate_term": ("invalid_argument",),
    "compact_export.empty_mask_region": ("invalid_argument",),
    "compact_export.encode": ("invalid_argument",),
    "compact_export.identity_mismatch": ("invalid_argument",),
    "compact_export.identity_unexpected": ("invalid_argument",),
    "compact_export.inputs_not_canonical": ("invalid_argument",),
    "compact_export.invalid_degree": ("invalid_argument",),
    "compact_export.invalid_mask_region": ("invalid_argument",),
    "compact_export.invalid_quantity": ("invalid_argument",),
    "compact_export.invalid_query": ("invalid_argument",),
    "compact_export.invalid_request": ("invalid_argument",),
    "compact_export.invalid_support": ("invalid_argument",),
    "compact_export.layout_invalid": ("invalid_argument",),
    "compact_export.mask_not_canonical": ("invalid_argument",),
    "compact_export.mask_support_mismatch": ("invalid_argument",),
    "compact_export.masked_region": ("invalid_argument",),
    "compact_export.missing_artifact_id": ("invalid_argument",),
    "compact_export.missing_quantity": ("invalid_argument",),
    "compact_export.negative_variance": ("invalid_argument",),
    "compact_export.no_terms": ("invalid_argument",),
    "compact_export.non_finite_input": ("invalid_argument",),
    "compact_export.non_finite_result": ("invalid_argument",),
    "compact_export.non_finite_value": ("invalid_argument",),
    "compact_export.out_of_support": ("invalid_argument",),
    "compact_export.oversized": ("invalid_argument",),
    "compact_export.premises_digest_mismatch": ("invalid_argument",),
    "compact_export.query_type_mismatch": ("invalid_argument",),
    "compact_export.scope_mismatch": ("invalid_argument",),
    "compact_export.term_support_mismatch": ("invalid_argument",),
    "compact_export.terms_not_canonical": ("invalid_argument",),
    "compact_export.too_many_inputs": ("invalid_argument",),
    "compact_export.too_many_levels": ("invalid_argument",),
    "compact_export.too_many_mask_regions": ("invalid_argument",),
    "compact_export.too_many_terms": ("invalid_argument",),
    "compact_export.unknown_level": ("invalid_argument",),
    "compact_export.unknown_quantity": ("invalid_argument",),
    "compact_export.unsupported_version": ("invalid_argument",),
    "compiled_categorical_posterior.cancelled": ("cancelled_no_claim",),
    "compiled_categorical_posterior.candidate": ("invalid_argument",),
    "compiled_categorical_posterior.execution_refused": ("invalid_argument",),
    "compiled_categorical_posterior.resource_limit": ("execution_resource_limit",),
    "compiled_categorical_posterior.source_invalid": ("invalid_argument",),
    "composite_recalc.invalid_request": ("invalid_argument",),
    "composite_recalc.native_state_mismatch": ("invalid_argument",),
    "composite_recalc.receipt_invalid": ("invalid_argument",),
    "composite_recalc.request_unsupported": ("route_not_supported",),
    "composition_boundary.artifact_rebuild_failed": ("invalid_argument",),
    "composition_boundary.atom_probabilities_invalid": ("decision_contract_unsatisfied",),
    "composition_boundary.conflicting_atoms_not_averaged": ("scenario_aggregate_not_supported",),
    "composition_boundary.coordinate_missing": ("decision_contract_unsatisfied",),
    "composition_boundary.coordinate_unsupported": ("decision_contract_unsatisfied",),
    "composition_boundary.dependence_route_not_supported": ("sampling_dependence_unknown",),
    "composition_boundary.invalid_input": ("invalid_argument",),
    "composition_boundary.mean_is_not_a_distribution": ("decision_contract_unsatisfied",),
    "composition_boundary.metadata_only_native_claim": ("attested_not_reverifiable",),
    "composition_boundary.native_required": ("attested_not_reverifiable",),
    "composition_boundary.non_affine_needs_joint_law": ("decision_contract_unsatisfied",),
    "composition_boundary.operation_not_declared": ("invalid_argument",),
    "composition_boundary.outcome_law_needs_joint_law": ("decision_contract_unsatisfied",),
    "composition_boundary.paired_draws_across_sources": ("joint_law_required",),
    "composition_boundary.shared_evidence_not_independent": ("scenario_aggregate_not_supported",),
    "composition_boundary.unknown_dependence_is_not_independence": ("sampling_dependence_unknown",),
    "composition_boundary.unknown_input": ("invalid_argument",),
    "composition_boundary.unsupported_action_not_comparable": ("decision_contract_unsatisfied",),
    "composition_boundary.verification_receipt_missing": ("external_verification_failed",),
    "composition_bundle.callback_unavailable": ("external_capability_missing",),
    "composition_bundle.edge_digest_mismatch": ("external_binding_mismatch",),
    "composition_bundle.expected_identity_mismatch": ("external_binding_mismatch",),
    "composition_bundle.graph_or_snapshot_mismatch": ("external_binding_mismatch",),
    "composition_bundle.incompatible_version": ("schema_mismatch",),
    "composition_bundle.invalid_container": ("invalid_argument",),
    "composition_bundle.node_not_found": ("invalid_argument",),
    "composition_bundle.oversized": ("invalid_argument",),
    "composition_bundle.provider_request_changed": ("external_binding_mismatch",),
    "composition_bundle.swapped_evidence": ("external_binding_mismatch",),
    "composition_bundle.tampered_quantity": ("quantity_semantics_mismatch",),
    "composition_bundle.unknown_node_kind": ("invalid_argument",),
    "composition_bundle.unsupported_law": ("joint_law_required",),
    "composition_provenance.digest_mismatch": ("external_binding_mismatch",),
    "composition_provenance.duplicate_link": ("invalid_argument",),
    "composition_provenance.invalid_link": ("invalid_argument",),
    "composition_provenance.missing_stage": ("external_binding_mismatch",),
    "composition_provenance.unknown_link": ("external_binding_mismatch",),
    "composition_provenance.unresolved_parent": ("external_binding_mismatch",),
    "composition_verification.artifact_invalid": ("invalid_argument",),
    "composition_verification.cancelled": ("cancelled_no_claim",),
    "composition_verification.cycle": ("invalid_argument",),
    "composition_verification.duplicate_node": ("invalid_argument",),
    "composition_verification.empty_unsupported": ("invalid_argument",),
    "composition_verification.graph_invalid": ("invalid_argument",),
    "composition_verification.identity_mismatch": ("route_not_supported",),
    "composition_verification.inherited_without_edge": ("invalid_argument",),
    "composition_verification.input_retired": ("claim_support_withdrawn",),
    "composition_verification.invalid_text": ("invalid_argument",),
    "composition_verification.ledger_mismatch": ("invalid_argument",),
    "composition_verification.limit_exceeded": ("invalid_argument",),
    "composition_verification.provider_trust_withdrawn": ("claim_support_withdrawn",),
    "composition_verification.recompute_required": ("claim_revalidation_required",),
    "composition_verification.replay_mismatch": ("invalid_argument",),
    "composition_verification.self_dependency": ("invalid_argument",),
    "composition_verification.stage_not_in_plan": ("invalid_argument",),
    "composition_verification.stage_unexecutable": ("claim_support_withdrawn",),
    "composition_verification.support_withdrawn": ("claim_support_withdrawn",),
    "composition_verification.unknown_claim": ("invalid_argument",),
    "composition_verification.unknown_node": ("invalid_argument",),
    "composition_verification.unknown_result": ("invalid_argument",),
    "conditional_study_ranking.action_not_unique": ("route_not_supported",),
    "conditional_study_ranking.artifact_invalid": ("invalid_argument",),
    "conditional_study_ranking.invalid_policy": ("invalid_argument",),
    "conditional_study_ranking.original_ranking_refused": ("invalid_argument",),
    "conditional_study_ranking.resource_limit": ("invalid_argument",),
    "conditional_study_ranking.source_mismatch": ("invalid_argument",),
    "conditional_study_ranking.source_unavailable": ("route_not_supported",),
    "constancy_posterior.band_not_constructed": ("route_not_supported",),
    "constancy_posterior.empty_contrast": ("invalid_argument",),
    "constancy_posterior.external_power_prior": ("route_not_supported",),
    "constancy_posterior.mapped_prior": ("route_not_supported",),
    "constancy_posterior.non_positive_scale": ("invalid_argument",),
    "coordinate_support.functional_not_described": ("route_not_supported",),
    "coordinate_support.invalid_grid": ("invalid_argument",),
    "coordinate_support.missing_evidence_relabeled": ("quantity_semantics_mismatch",),
    "coordinate_support.point_count_mismatch": ("invalid_argument",),
    "coordinate_support.regime_not_describable": ("route_not_supported",),
    "coordinate_support.summary_not_worst_label": ("quantity_semantics_mismatch",),
    "coordinate_support.temporal_path_not_described": ("route_not_supported",),
    "coordinate_support.units_required": ("invalid_argument",),
    "coordinate_support.unknown_variable": ("unknown_variable",),
    "counterfactual_id.bounds_exceeded": ("route_not_supported",),
    "counterfactual_id.budget": ("transport_budget_cancel",),
    "counterfactual_id.conflicting_subscripts": ("route_not_supported",),
    "counterfactual_id.counterfactual_hedge": ("route_not_supported",),
    "counterfactual_id.graph_outside_contract": ("route_not_supported",),
    "counterfactual_id.interval_requested": ("estimator_inference_mismatch",),
    "counterfactual_id.invalid_artifact": ("invalid_argument",),
    "counterfactual_id.invalid_query": ("invalid_argument",),
    "counterfactual_id.path_specific_not_supported": ("route_not_supported",),
    "counterfactual_id.positivity_violation": ("invalid_argument",),
    "counterfactual_id.query_outside_contract": ("route_not_supported",),
    "counterfactual_transport.bounds_exceeded": ("route_not_supported",),
    "counterfactual_transport.budget": ("transport_budget_cancel",),
    "counterfactual_transport.checked_obstruction": ("transport_proven_non_transportable",),
    "counterfactual_transport.coupling_unsupported": ("route_not_supported",),
    "counterfactual_transport.evidence_incompatible": ("transport_numerical_failure",),
    "counterfactual_transport.factor_not_derivable": ("transport_not_certified",),
    "counterfactual_transport.graph_outside_contract": ("route_not_supported",),
    "counterfactual_transport.invalid_artifact": ("invalid_argument",),
    "counterfactual_transport.invalid_catalog": ("invalid_argument",),
    "counterfactual_transport.invalid_derivation": ("transport_not_certified",),
    "counterfactual_transport.invalid_evidence": ("invalid_argument",),
    "counterfactual_transport.invalid_query": ("invalid_argument",),
    "counterfactual_transport.invalid_witness": ("transport_not_certified",),
    "counterfactual_transport.missing_law": ("transport_missing_provider",),
    "counterfactual_transport.missing_regime": ("transport_missing_evidence",),
    "counterfactual_transport.no_sampled_inference": ("route_not_supported",),
    "counterfactual_transport.selected_sample_law": ("route_not_supported",),
    "counterfactual_transport.support_failure": ("transport_support_failure",),
    "counterfactual_transport.zero_conditioning_mass": ("transport_support_failure",),
    "counterfactual_transport_sampled.bounds_exceeded": ("route_not_supported",),
    "counterfactual_transport_sampled.cancelled": ("transport_budget_cancel",),
    "counterfactual_transport_sampled.envelope_budget": ("transport_budget_cancel",),
    "counterfactual_transport_sampled.envelope_undefined": ("transport_support_failure",),
    "counterfactual_transport_sampled.interval_not_computable": ("transport_support_failure",),
    "counterfactual_transport_sampled.invalid_artifact": ("invalid_argument",),
    "counterfactual_transport_sampled.invalid_counts": ("invalid_argument",),
    "counterfactual_transport_sampled.invalid_overlap": ("invalid_argument",),
    "counterfactual_transport_sampled.missing_table": ("transport_missing_provider",),
    "counterfactual_transport_sampled.overlap_declared_independent": ("evidence_double_counted",),
    "counterfactual_transport_sampled.overlap_unknown": ("sampling_dependence_unknown",),
    "counterfactual_transport_sampled.sample_size_below_limit": ("transport_support_failure",),
    "counterfactual_transport_sampled.shared_units_unlinked": ("route_not_supported",),
    "counterfactual_transport_sampled.sparsity_above_limit": ("transport_support_failure",),
    "counterfactual_transport_sampled.table_not_cited": ("invalid_argument",),
    "counterfactual_transport_sampled.unseen_cell": ("transport_support_failure",),
    "cpdag_scenarios.cancelled_not_exportable": ("cancelled_no_claim",),
    "cpdag_scenarios.consumer_limit_exceeded": ("transport_budget_cancel",),
    "cpdag_scenarios.data_identity_mismatch": ("invalid_argument",),
    "cpdag_scenarios.evidence_identity_mismatch": ("invalid_argument",),
    "cpdag_scenarios.exact_laws_only": ("route_not_supported",),
    "cpdag_scenarios.no_evidence_binding": ("invalid_argument",),
    "cpdag_scenarios.not_a_cpdag": ("invalid_argument",),
    "cpdag_scenarios.premises_mismatch": ("invalid_argument",),
    "cpdag_scenarios.provider_mismatch": ("invalid_argument",),
    "cpdag_scenarios.report_replay_mismatch": ("invalid_argument",),
    "cpdag_scenarios.selection_or_latent": ("route_not_supported",),
    "cpdag_scenarios.unsupported_semantics": ("route_not_supported",),
    "cross_world.graph_outside_contract": ("route_not_supported",),
    "cross_world.interval_requested": ("estimator_inference_mismatch",),
    "cross_world.invalid_query": ("invalid_argument",),
    "cross_world.query_outside_contract": ("route_not_supported",),
    "cross_world.recanting_witness": ("cross_world_not_identified",),
    "cross_world_execution.artifact_changed": ("route_not_supported",),
    "cross_world_execution.bounds_exceeded": ("invalid_argument",),
    "cross_world_execution.cancelled": ("transport_budget_cancel",),
    "cross_world_execution.cell_not_in_matrix": ("route_not_supported",),
    "cross_world_execution.elimination_bound_exceeded": ("route_not_supported",),
    "cross_world_execution.envelope_member_structure_differs": ("invalid_argument",),
    "cross_world_execution.envelope_needs_parameter_set": ("route_not_supported",),
    "cross_world_execution.envelope_parameter_set_empty": ("invalid_argument",),
    "cross_world_execution.envelope_reading_required": ("invalid_argument",),
    "cross_world_execution.evidence_outside_model_support": ("route_not_supported",),
    "cross_world_execution.evidence_probability_underflow": ("route_not_supported",),
    "cross_world_execution.finite_fitting_not_executed": ("route_not_supported",),
    "cross_world_execution.incompatible_history": ("route_not_supported",),
    "cross_world_execution.invalid_artifact": ("invalid_argument",),
    "cross_world_execution.invalid_evidence": ("invalid_argument",),
    "cross_world_execution.invalid_identity": ("invalid_argument",),
    "cross_world_execution.invalid_quantity": ("invalid_argument",),
    "cross_world_execution.invalid_request": ("invalid_argument",),
    "cross_world_execution.invalid_scm": ("invalid_argument",),
    "cross_world_execution.invalid_world": ("invalid_argument",),
    "cross_world_execution.joint_state_bound_exceeded": ("route_not_supported",),
    "cross_world_execution.latent_confounding_not_identified": ("cross_world_not_identified",),
    "cross_world_execution.mechanisms_not_declared": ("cross_world_not_identified",),
    "cross_world_execution.no_interval_method": ("sampling_dependence_unknown",),
    "cross_world_execution.no_posterior_method": ("uncertainty_representation_insufficient",),
    "cross_world_execution.no_transported_row_derivation": ("route_not_supported",),
    "cross_world_execution.posterior_weight_underflow": ("route_not_supported",),
    "cross_world_execution.sampled_abduction_not_supported": ("estimator_inference_mismatch",),
    "cross_world_execution.soft_intervention": ("route_not_supported",),
    "cross_world_execution.spread_needs_enumeration": ("route_not_supported",),
    "cross_world_uncertainty.account_mismatch": ("invalid_argument",),
    "cross_world_uncertainty.bounds_exceeded": ("invalid_argument",),
    "cross_world_uncertainty.incompatible_history": ("route_not_supported",),
    "cross_world_uncertainty.no_transported_path_derivation": ("transport_not_certified",),
    "cross_world_uncertainty.result_mismatch": ("invalid_argument",),
    "cross_world_uncertainty.sampled_path_not_supported": ("estimator_inference_mismatch",),
    "cross_world_uncertainty.unit_sampling_undeclared": ("sampling_dependence_unknown",),
    "decision_adapters.bayes_over_identified_set": ("decision_contract_unsatisfied",),
    "decision_adapters.claim_draws_too_large": ("invalid_argument",),
    "decision_adapters.claims_sources_mismatch": ("invalid_argument",),
    "decision_adapters.completion_count_not_probability": ("decision_contract_unsatisfied",),
    "decision_adapters.criterion_not_supported_for_set": ("decision_contract_unsatisfied",),
    "decision_adapters.duplicate_action": ("invalid_argument",),
    "decision_adapters.evaluated_needs_draws": ("invalid_argument",),
    "decision_adapters.graph_claims_carry_no_probability": ("invalid_argument",),
    "decision_adapters.invalid_claims": ("invalid_argument",),
    "decision_adapters.invalid_interval": ("invalid_argument",),
    "decision_adapters.invalid_probability": ("invalid_argument",),
    "decision_adapters.invalid_utilities": ("invalid_argument",),
    "decision_adapters.missing_action": ("invalid_argument",),
    "decision_adapters.point_claim_is_single": ("invalid_argument",),
    "decision_adapters.policy_mismatch": ("decision_contract_unsatisfied",),
    "decision_adapters.probabilities_required": ("decision_contract_unsatisfied",),
    "decision_adapters.unevaluated_needs_reason": ("invalid_argument",),
    "decision_adapters.unknown_action": ("invalid_argument",),
    "decision_adapters.unknown_claim_kind": ("invalid_argument",),
    "decision_adapters.unknown_claim_status": ("invalid_argument",),
    "decision_adapters.unknown_support_status": ("invalid_argument",),
    "decision_admissibility.duplicate_rule": ("invalid_argument",),
    "decision_admissibility.invalid_rule": ("invalid_argument",),
    "decision_admissibility.unknown_action": ("invalid_argument",),
    "decision_admissibility.unknown_input": ("invalid_argument",),
    "decision_claims.identity_mismatch": ("invalid_argument",),
    "decision_claims.missing_scenario_utility": ("decision_contract_unsatisfied",),
    "decision_claims.probabilities_not_declared": ("decision_contract_unsatisfied",),
    "decision_claims.range_unavailable": ("decision_contract_unsatisfied",),
    "decision_claims.unresolved_scenarios": ("decision_contract_unsatisfied",),
    "decision_claims.unsupported_claim_shape": ("invalid_argument",),
    "decision_contract.duplicate_id": ("invalid_argument",),
    "decision_contract.invalid_declaration": ("invalid_argument",),
    "decision_contract.invalid_parameter": ("invalid_argument",),
    "decision_contract.invalid_quantity": ("quantity_semantics_mismatch",),
    "decision_contract.missing_source": ("decision_contract_unsatisfied",),
    "decision_contract.outside_decision_scope": ("quantity_semantics_mismatch",),
    "decision_contract.unknown_action": ("invalid_argument",),
    "decision_contract.unknown_input": ("invalid_argument",),
    "decision_evaluation.distribution_meaning": ("distribution_meaning_mismatch",),
    "decision_evaluation.joint_law_required": ("joint_law_required",),
    "decision_evaluation.mean_source_insufficient": ("decision_contract_unsatisfied",),
    "decision_evaluation.mean_source_not_replayable": ("route_not_supported",),
    "decision_evaluation.non_finite_utility": ("decision_contract_unsatisfied",),
    "decision_evaluation.quantity_not_found": ("quantity_semantics_mismatch",),
    "decision_evaluation.structure_inputs_required": ("route_not_supported",),
    "decision_evaluation.unsupported_coordinate": ("quantity_semantics_mismatch",),
    "decision_evidence.adjustment_not_admissible": (),
    "decision_evidence.cancelled": (),
    "decision_evidence.dependence_unknown": (),
    "decision_evidence.deployment_bridge_missing": (),
    "decision_evidence.frame_shape": (),
    "decision_evidence.incompatible_horizon": (),
    "decision_evidence.incompatible_intervention": (),
    "decision_evidence.incompatible_population": ("decision_evidence_scope_mismatch",),
    "decision_evidence.interference_unspecified": (),
    "decision_evidence.invalid_declaration": (),
    "decision_evidence.resource_limit": (),
    "decision_evidence.too_many_unknowns": (),
    "decision_evidence.unsupported_assignment_design": (),
    "decision_posterior.band_not_constructed": ("route_not_supported",),
    "decision_posterior.empty_law": ("invalid_argument",),
    "decision_posterior.external_power_prior": ("route_not_supported",),
    "decision_posterior.mapped_prior": ("route_not_supported",),
    "decision_posterior.non_positive_scale": ("invalid_argument",),
    "decision_reliability.artifact": ("invalid_argument",),
    "decision_reliability.artifact_id_missing": ("invalid_argument",),
    "decision_reliability.artifact_kind_mismatch": ("invalid_argument",),
    "decision_reliability.cancelled": ("cancelled_no_claim",),
    "decision_reliability.candidate_set_mismatch": ("invalid_argument",),
    "decision_reliability.combined_uncertainty": ("uncertainty_components_not_combined",),
    "decision_reliability.decision_mismatch": ("invalid_argument",),
    "decision_reliability.dependence_unknown": ("sampling_dependence_unknown",),
    "decision_reliability.draws_digest_mismatch": ("invalid_argument",),
    "decision_reliability.effective_sample_size_too_small": ("route_not_supported",),
    "decision_reliability.identity_mismatch": ("invalid_argument",),
    "decision_reliability.invalid_input": ("invalid_argument",),
    "decision_reliability.joint_law_required": ("joint_law_required",),
    "decision_reliability.memory_budget_exceeded": ("invalid_argument",),
    "decision_reliability.numerical": ("invalid_argument",),
    "decision_reliability.producer_mean_disagrees": ("distribution_meaning_mismatch",),
    "decision_reliability.producer_point_only": ("joint_law_required",),
    "decision_reliability.report_mismatch": ("invalid_argument",),
    "decision_reliability.route_not_supported": ("route_not_supported",),
    "decision_reliability.search_record_inconsistent": ("invalid_argument",),
    "decision_reliability.search_record_required": ("candidate_search_record_required",),
    "decision_reliability.selection_stream_mismatch": ("invalid_argument",),
    "decision_reliability.utility_unit_mismatch": ("invalid_argument",),
    "decision_reliability.validation_not_independent": ("validation_stream_not_independent",),
    "decision_reliability.validation_stream_mismatch": ("invalid_argument",),
    "decision_robustness.contract_mismatch": ("decision_contract_unsatisfied",),
    "decision_robustness.invalid_profile": ("invalid_argument",),
    "decision_structural.invalid_atoms": ("invalid_argument",),
    "decision_structural.invalid_probabilities": ("decision_contract_unsatisfied",),
    "decision_structural.probabilities_required": ("decision_contract_unsatisfied",),
    "descriptive_comparison.admission_refused": (),
    "descriptive_comparison.arm_not_populated": ("arm_not_populated",),
    "descriptive_comparison.arm_too_small": ("invalid_argument",),
    "descriptive_comparison.cancelled": ("cancelled_no_claim",),
    "descriptive_comparison.column_attribution_not_identified": ("effect_not_identified",),
    "descriptive_comparison.estimand_coding_mismatch": ("route_not_supported",),
    "descriptive_comparison.fold_exceeds_lease": (),
    "descriptive_comparison.gap_interval_unavailable": ("route_not_supported",),
    "descriptive_comparison.invalid_covariance": ("invalid_argument",),
    "descriptive_comparison.invalid_data": ("invalid_argument",),
    "descriptive_comparison.joint_covariance_unavailable": ("required_option_missing",),
    "descriptive_comparison.learner_exceeds_lease": (),
    "descriptive_comparison.population_not_all_observed": ("population_not_estimable",),
    "descriptive_comparison.scale_not_supported": ("route_not_supported",),
    "descriptive_comparison.transform_domain": ("invalid_argument",),
    "design_ranking.bounds_exceeded": ("route_not_supported",),
    "design_ranking.contract_mismatch": ("design_signal_invalid",),
    "design_ranking.cost_units_mismatch": ("design_cost_units_mismatch",),
    "design_ranking.design_error": ("design_signal_invalid",),
    "design_ranking.digest_mismatch": ("design_signal_invalid",),
    "design_ranking.duplicate_candidate": ("design_signal_invalid",),
    "design_ranking.empty": ("invalid_argument",),
    "design_ranking.identity_mismatch": ("design_signal_invalid",),
    "design_ranking.invalid_artifact": ("invalid_argument",),
    "design_ranking.invalid_request": ("invalid_argument",),
    "design_ranking.rank_mismatch": ("design_signal_invalid",),
    "design_ranking.replay_mismatch": ("design_signal_invalid",),
    "design_ranking.signal_inconsistent": ("design_signal_invalid",),
    "design_ranking.signal_mismatch": ("design_signal_invalid",),
    "design_ranking.source_digest_mismatch": ("design_signal_invalid",),
    "design_ranking.source_overlap": ("design_signal_invalid",),
    "design_ranking.source_overlap_unknown": ("evidence_overlap_unknown",),
    "design_ranking.unsupported_version": ("route_not_supported",),
    "discovered_transport.class_enumeration_partial": ("discovered_structure_unresolved",),
    "discovered_transport.class_not_identified": ("discovered_structure_unresolved",),
    "discovered_transport.class_too_large": ("discovered_structure_unresolved",),
    "discovered_transport.evidence_invalid": ("invalid_argument",),
    "discovered_transport.member_bound_zero": ("discovered_structure_unresolved",),
    "discovered_transport.member_not_estimable": ("discovered_structure_unresolved",),
    "discovered_transport.no_members": ("discovered_structure_unresolved",),
    "discovered_transport.orientation_conflict": ("discovered_structure_unresolved",),
    "discovered_transport.pag_class_ambiguous": ("discovered_structure_unresolved",),
    "discovered_transport.pag_selection_bias_edge": ("discovered_structure_unresolved",),
    "discovered_transport.search_incomplete": ("discovered_structure_unresolved",),
    "discovered_transport.selection_target_unknown": ("discovered_structure_unresolved",),
    "discovered_transport.temporal_query_invalid": ("discovered_structure_unresolved",),
    "discovered_transport.temporal_structure_unsupported": ("discovered_structure_unresolved",),
    "discovered_transport.trial_compatibility_invalid": ("discovered_structure_unresolved",),
    "discovered_transport.trial_rejects_every_member": ("discovered_structure_unresolved",),
    "discretized_scm.cancelled": ("cancelled_no_claim",),
    "discretized_scm.composition_incoherent": ("mechanism_composition_incoherent",),
    "discretized_scm.degenerate_marginal": ("route_not_supported",),
    "discretized_scm.intervention_invalid": ("invalid_argument",),
    "discretized_scm.latent_confounding": ("route_not_supported",),
    "discretized_scm.levels_out_of_range": ("invalid_argument",),
    "discretized_scm.levels_undeclared": ("invalid_argument",),
    "discretized_scm.non_gaussian_mechanism": ("route_not_supported",),
    "discretized_scm.numerical": ("invalid_argument",),
    "discretized_scm.point_only": ("uncertainty_representation_insufficient",),
    "discretized_scm.state_space_too_large": ("invalid_argument",),
    "discretized_scm.support_not_real_line": ("route_not_supported",),
    "discretized_scm.unknown_variable": ("invalid_argument",),
    "discretized_scm_artifact.claimed_numbers_differ": ("invalid_argument",),
    "discretized_scm_artifact.identity_mismatch": ("invalid_argument",),
    "discretized_scm_artifact.malformed": ("invalid_argument",),
    "discretized_scm_artifact.not_exportable": ("route_not_supported",),
    "discretized_scm_artifact.rederivation_refused": ("invalid_argument",),
    "discretized_scm_artifact.version_unsupported": ("schema_mismatch",),
    "distribution_meaning.incompatible_operation": ("distribution_meaning_mismatch",),
    "distribution_meaning.legacy_posterior": ("distribution_meaning_mismatch",),
    "distribution_meaning.observational_not_interventional": ("quantity_semantics_mismatch",),
    "distribution_meaning.posterior_interval": ("distribution_meaning_mismatch",),
    "dml.arm_not_populated": ("arm_not_populated",),
    "dml.cancelled": ("cancelled_no_claim",),
    "dose_grid.bandwidth_outside_range": ("invalid_argument",),
    "dose_grid.cancelled": ("transport_budget_cancel",),
    "dose_grid.consumer_limit_exceeded": ("invalid_argument",),
    "dose_grid.data_identity_mismatch": ("invalid_argument",),
    "dose_grid.derivative_without_claim": ("option_not_applicable",),
    "dose_grid.graph_not_certified": ("effect_not_identified",),
    "dose_grid.insufficient_local_weight": ("positivity_violated",),
    "dose_grid.invalid_lag": ("invalid_argument",),
    "dose_grid.invalid_request": ("invalid_argument",),
    "dose_grid.level_without_claim": ("option_not_applicable",),
    "dose_grid.no_simultaneous_band_method": ("route_not_supported",),
    "dose_grid.non_finite_result": ("transport_numerical_failure",),
    "dose_grid.premises_mismatch": ("invalid_argument",),
    "dose_grid.refused": ("invalid_argument",),
    "dose_grid.resource_limit": ("transport_budget_cancel",),
    "dose_grid.resource_refused": ("transport_budget_cancel",),
    "dose_grid.result_replay_mismatch": ("invalid_argument",),
    "dose_grid.too_few_clusters": ("invalid_argument",),
    "dose_grid.undecodable": ("invalid_argument",),
    "dose_grid.unsupported_dose": ("positivity_violated",),
    "dose_grid.unsupported_semantics": ("route_not_supported",),
    "dose_grid_posterior.band_not_constructed": ("route_not_supported",),
    "dose_grid_posterior.external_power_prior": ("route_not_supported",),
    "dose_grid_posterior.insufficient_weight": ("positivity_violated",),
    "dose_grid_posterior.mapped_prior": ("route_not_supported",),
    "dose_grid_posterior.non_positive_scale": ("invalid_argument",),
    "dose_response.bootstrap_below_floor": ("estimator_inference_mismatch",),
    "dose_response.bounds_exceeded": ("route_not_supported",),
    "dose_response.cate_or_simultaneous": ("route_not_supported",),
    "dose_response.declared_folds_unbalanced": ("invalid_argument",),
    "dose_response.dose_density_floor": ("transport_support_failure",),
    "dose_response.dose_density_invalid": ("transport_support_failure",),
    "dose_response.dose_too_discrete": ("treatment_support_too_discrete",),
    "dose_response.estimated_dose_density": ("route_not_supported",),
    "dose_response.grid_outside_dose_support": ("transport_support_failure",),
    "dose_response.invalid_declared_rows": ("invalid_argument",),
    "dose_response.kernel_not_supported": ("route_not_supported",),
    "dose_response.learner_not_portable": ("route_not_supported",),
    "dose_response.local_dose_ess": ("transport_support_failure",),
    "dose_response.membership_overlap": ("transport_support_failure",),
    "dose_response.non_iid_design": ("sampling_dependence_unknown",),
    "dose_response.posterior_declared_rows": ("route_not_supported",),
    "dose_response.quadrature_tolerance": ("transport_numerical_failure",),
    "dose_response.target_not_smoothed": ("route_not_supported",),
    "effect_constancy.estimand_missing": ("invalid_argument",),
    "effect_constancy.incompatible_partitions": ("route_not_supported",),
    "effect_constancy.invalid_alpha": ("invalid_argument",),
    "effect_constancy.invalid_covariance": ("invalid_argument",),
    "effect_constancy.invalid_identity": ("invalid_argument",),
    "effect_constancy.invalid_request": ("invalid_argument",),
    "effect_constancy.invalid_standard_error": ("invalid_argument",),
    "effect_constancy.limits_exceeded": ("invalid_argument",),
    "effect_constancy.non_finite_estimate": ("invalid_argument",),
    "effect_constancy.too_few_partitions": ("invalid_argument",),
    "effect_constancy.too_many_partitions": ("invalid_argument",),
    "effect_constancy.unknown_reference": ("invalid_argument",),
    "effect_constancy.unsupported_partition": ("route_not_supported",),
    "effect_constancy.wrong_contract": ("route_not_supported",),
    "effect_constancy_consumer.artifact_invalid": ("invalid_argument",),
    "effect_constancy_consumer.catalog_binding_invalid": ("invalid_argument",),
    "effect_constancy_consumer.contrast_missing": ("invalid_argument",),
    "effect_constancy_consumer.decision_unsupported": ("invalid_argument",),
    "effect_constancy_consumer.invalid_request": ("invalid_argument",),
    "effect_constancy_consumer.limits_exceeded": ("invalid_argument",),
    "effect_constancy_consumer.partition_missing": ("invalid_argument",),
    "effect_constancy_consumer.quantity_mismatch": ("invalid_argument",),
    "effect_constancy_consumer.support_unsupported": ("route_not_supported",),
    "evidence_ledger.cancelled": ("cancelled_no_claim",),
    "evidence_ledger.diagonal_covariance_shared_rows": ("evidence_double_counted",),
    "evidence_ledger.duplicate_result": ("invalid_argument",),
    "evidence_ledger.duplicate_row": ("evidence_double_counted",),
    "evidence_ledger.duplicate_source": ("evidence_double_counted",),
    "evidence_ledger.identity_mismatch": ("route_not_supported",),
    "evidence_ledger.interval_construction_unavailable": ("route_not_supported",),
    "evidence_ledger.interval_report_invalid": ("invalid_argument",),
    "evidence_ledger.invalid_declaration": ("invalid_argument",),
    "evidence_ledger.kind_mismatch": ("invalid_argument",),
    "evidence_ledger.limit_exceeded": ("invalid_argument",),
    "evidence_ledger.lineage_rows_mismatch": ("invalid_argument",),
    "evidence_ledger.native_and_transported": ("evidence_double_counted",),
    "evidence_ledger.operation_arity": ("invalid_argument",),
    "evidence_ledger.overlap_unknown": ("evidence_overlap_unknown",),
    "evidence_ledger.reentry_derived_from_used_data": ("evidence_double_counted",),
    "evidence_ledger.reentry_duplicate": ("evidence_double_counted",),
    "evidence_ledger.reentry_prior_plus_likelihood": ("evidence_double_counted",),
    "evidence_ledger.reentry_transform": ("evidence_double_counted",),
    "evidence_ledger.repeated_source": ("evidence_double_counted",),
    "evidence_ledger.replay_mismatch": ("invalid_argument",),
    "evidence_ledger.result_not_valid": ("evidence_source_withdrawn",),
    "evidence_ledger.shared_observations": ("evidence_double_counted",),
    "evidence_ledger.source_superseded": ("evidence_source_withdrawn",),
    "evidence_ledger.source_withdrawn": ("evidence_source_withdrawn",),
    "evidence_ledger.supersede_kind_mismatch": ("invalid_argument",),
    "evidence_ledger.synthesis_mismatch": ("invalid_argument",),
    "evidence_ledger.transport_identity_missing": ("invalid_argument",),
    "evidence_ledger.transported_as_native": ("invalid_argument",),
    "evidence_ledger.unknown_result": ("invalid_argument",),
    "evidence_ledger.unknown_source": ("invalid_argument",),
    "evidence_ledger.update_drops_rows": ("invalid_argument",),
    "evidence_obligations.invalid_obligation": ("invalid_argument",),
    "evidence_obligations.wrong_contract": ("transport_missing_evidence",),
    "evidence_plan.budget": ("transport_budget_cancel",),
    "evidence_plan.compile_refused": ("invalid_argument",),
    "evidence_plan.cost_units_mismatch": ("design_cost_units_mismatch",),
    "evidence_plan.invalid_request": ("invalid_argument",),
    "evidence_plan.no_failure_to_plan": ("invalid_argument",),
    "evidence_plan.no_route": ("invalid_argument",),
    "evidence_plan.none_certified": ("transport_not_certified",),
    "evsi.action_set_changed": ("decision_contract_unsatisfied",),
    "evsi.bound_violation": ("design_signal_invalid",),
    "evsi.cancelled": ("cancelled_no_claim",),
    "evsi.cost_map_required": ("design_cost_units_mismatch",),
    "evsi.cost_units_mismatch": ("design_cost_units_mismatch",),
    "evsi.decision_values_incoherent": ("design_signal_invalid",),
    "evsi.decision_values_need_draws": ("design_signal_invalid",),
    "evsi.design_error": ("design_signal_invalid",),
    "evsi.duplicate_candidate": ("design_signal_invalid",),
    "evsi.execution_resource_limit": ("execution_resource_limit",),
    "evsi.invalid_cost": ("invalid_argument",),
    "evsi.invalid_request": ("invalid_argument",),
    "evsi.prior_states_mismatch": ("design_signal_invalid",),
    "evsi.receipt_mismatch": ("design_signal_invalid",),
    "evsi.sample_size_mismatch": ("design_signal_invalid",),
    "evsi.source_overlap": ("design_signal_invalid",),
    "evsi.source_overlap_unknown": ("evidence_overlap_unknown",),
    "evsi.unscored_candidate": ("design_signal_invalid",),
    "external_mechanism.declaration_invalid": ("mechanism_declaration_invalid",),
    "external_mechanism.implementation_changed": ("mechanism_implementation_changed",),
    "external_mechanism.implementation_unavailable": ("mechanism_implementation_unavailable",),
    "external_mechanism.resolver_failed": ("mechanism_resolver_failed",),
    "external_mechanism.resolver_missing": ("mechanism_resolver_missing",),
    "external_object_verification.duplicate_probe": ("invalid_argument",),
    "external_object_verification.failed_probe": ("external_verification_failed",),
    "external_object_verification.invalid_probe": ("invalid_argument",),
    "external_object_verification.missing_probe": ("external_verification_failed",),
    "external_recalc.artifact_invalid": ("invalid_argument",),
    "external_recalc.attempt_limit": ("invalid_argument",),
    "external_recalc.callback_failed": ("invalid_argument",),
    "external_recalc.contract_mismatch": ("invalid_argument",),
    "external_recalc.idempotency_conflict": ("invalid_argument",),
    "external_recalc.idempotency_unsupported": ("invalid_argument",),
    "external_recalc.invalid_descriptor": ("invalid_argument",),
    "external_recalc.invalid_output": ("invalid_argument",),
    "external_recalc.invalid_request": ("invalid_argument",),
    "external_recalc.limits_exceeded": ("invalid_argument",),
    "external_recalc.output_unavailable": ("invalid_argument",),
    "external_recalc.policy_unsupported": ("route_not_supported",),
    "external_recalc.provider_mismatch": ("invalid_argument",),
    "external_recalc.provider_unavailable": ("invalid_argument",),
    "external_recalc.quantity_unsupported": ("route_not_supported",),
    "external_recalc.replay_mismatch": ("invalid_argument",),
    "external_recalc.replay_unsupported": ("route_not_supported",),
    "external_recalc.retry_unsafe": ("route_not_supported",),
    "external_response_binding.column_quantity": ("external_binding_mismatch",),
    "external_response_binding.column_support_length": ("invalid_argument",),
    "external_response_binding.contract_not_identified": ("effect_not_identified",),
    "external_response_binding.coordinate_conditioning": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_functional": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_horizon": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_invalid_coordinate": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_population": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_regime": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_transform": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_units": ("quantity_semantics_mismatch",),
    "external_response_binding.coordinate_variable": ("quantity_semantics_mismatch",),
    "external_response_binding.dimension": ("external_binding_mismatch",),
    "external_response_binding.distribution_meaning": ("distribution_meaning_mismatch",),
    "external_response_binding.empty_column": ("invalid_argument",),
    "external_response_binding.graph": ("external_binding_mismatch",),
    "external_response_binding.invalid_contract": ("invalid_argument",),
    "external_response_binding.malformed_artifact": ("invalid_argument",),
    "external_response_binding.malformed_capability": ("invalid_argument",),
    "external_response_binding.malformed_equivalence": ("invalid_argument",),
    "external_response_binding.malformed_identification": ("invalid_argument",),
    "external_response_binding.malformed_meaning": ("invalid_argument",),
    "external_response_binding.malformed_probe": ("invalid_argument",),
    "external_response_binding.malformed_quantity": ("invalid_argument",),
    "external_response_binding.malformed_support": ("invalid_argument",),
    "external_response_binding.malformed_trust": ("invalid_argument",),
    "external_response_binding.missing_assumption": ("external_binding_mismatch",),
    "external_response_binding.missing_evidence": ("external_binding_mismatch",),
    "external_response_binding.non_finite_value": ("invalid_argument",),
    "external_response_binding.object_kind": ("external_binding_mismatch",),
    "external_response_binding.object_quantities": ("external_binding_mismatch",),
    "external_response_binding.posterior_kind": ("external_binding_mismatch",),
    "external_response_binding.unchecked_observational_law": ("external_binding_mismatch",),
    "external_scientific_providers.duplicate_declaration": ("invalid_argument",),
    "external_scientific_providers.invalid_declaration": ("invalid_argument",),
    "external_scientific_providers.invalid_identity": ("invalid_argument",),
    "external_trust_receipt.receipt_mismatch": ("external_verification_failed",),
    "finite_action_posterior.band_not_constructed": ("route_not_supported",),
    "finite_action_posterior.empty_law": ("invalid_argument",),
    "finite_action_posterior.external_power_prior": ("route_not_supported",),
    "finite_action_posterior.mapped_prior": ("route_not_supported",),
    "finite_action_posterior.non_positive_scale": ("invalid_argument",),
    "frozen_scores.identity_mismatch": ("route_not_supported",),
    "functional_inverse_posterior.band_not_constructed": ("route_not_supported",),
    "functional_inverse_posterior.empty_law": ("invalid_argument",),
    "functional_inverse_posterior.external_power_prior": ("route_not_supported",),
    "functional_inverse_posterior.mapped_prior": ("route_not_supported",),
    "functional_inverse_posterior.non_positive_scale": ("invalid_argument",),
    "functional_inverse_query.artifact_kind_mismatch": ("invalid_argument",),
    "functional_inverse_query.bounds_exceeded": ("inverse_functional_unsupported",),
    "functional_inverse_query.forward_origin_invalid": ("inverse_functional_unsupported",),
    "functional_inverse_query.global_feasibility_claim": ("inverse_functional_unsupported",),
    "functional_inverse_query.invalid_artifact": ("invalid_argument",),
    "functional_inverse_query.wrong_contract": ("inverse_functional_unsupported",),
    "functional_source.expected_identity_mismatch": ("invalid_argument",),
    "functional_source.invalid_artifact": ("invalid_argument",),
    "functional_source.invalid_contract": ("invalid_argument",),
    "functional_source.invalid_source": ("invalid_argument",),
    "functional_source.limits_exceeded": ("invalid_argument",),
    "functional_source.missing_action": ("invalid_argument",),
    "functional_source.native_source_unavailable": ("invalid_argument",),
    "functional_source.original_law_required": ("route_not_supported",),
    "functional_source.output_mismatch": ("invalid_argument",),
    "functional_source.source_mismatch": ("invalid_argument",),
    "graph_falsification.ci_p_value_missing": ("diagnostic_not_available",),
    "graph_falsification.ci_result_count": ("diagnostic_not_available",),
    "graph_falsification.ci_test_not_frequentist": ("route_not_supported",),
    "graph_falsification.data_not_observed": ("invalid_argument",),
    "graph_falsification.duplicate_variable": ("invalid_argument",),
    "graph_falsification.empty_data": ("invalid_argument",),
    "graph_falsification.invalid_level": ("invalid_argument",),
    "graph_falsification.invalid_permutations": ("invalid_argument",),
    "graph_falsification.node_not_static": ("route_not_supported",),
    "graph_falsification.too_few_nodes": ("invalid_argument",),
    "graph_falsification.work_limit": (),
    "graph_imputation.atom_failure": (),
    "graph_imputation.atom_not_lowered": (),
    "graph_imputation.cancelled": ("cancelled_no_claim",),
    "graph_imputation.candidate_refused": ("invalid_argument",),
    "graph_imputation.compiled": ("invalid_argument",),
    "graph_imputation.execution_refused": ("invalid_argument",),
    "graph_imputation.interval_unsupported": ("route_not_supported",),
    "graph_imputation.invalid_source": (),
    "graph_imputation.lowering": ("invalid_argument",),
    "graph_imputation.point_unsupported": (),
    "graph_imputation.resource_limit": ("execution_resource_limit",),
    "graph_imputation.search_stopped": ("cancelled_no_claim", "execution_resource_limit"),
    "graph_imputation.supported_graph_unidentified": (),
    "graph_imputation.unidentified": (),
    "graph_imputation.unscorable": (),
    "graph_imputation.unsupported": ("route_not_supported",),
    "graph_imputation.wrong_dispatch": (),
    "graph_selection_imputation.columns": ("invalid_argument",),
    "graph_selection_imputation.data": ("invalid_argument",),
    "graph_selection_imputation.domain": ("invalid_argument",),
    "graph_selection_imputation.interval_unmeasured": ("not_supported",),
    "graph_selection_imputation.model": ("invalid_argument",),
    "graph_selection_imputation.result": ("invalid_argument",),
    "graph_selection_imputation.variables": ("invalid_argument",),
    "identification_repair.bounds_exceeded": ("invalid_argument",),
    "identification_repair.budget": ("transport_budget_cancel",),
    "identification_repair.cost_units_mismatch": ("design_cost_units_mismatch",),
    "identification_repair.invalid_request": ("invalid_argument",),
    "identification_repair.not_a_failure": ("invalid_argument",),
    "identification_repair.wrong_contract": ("transport_not_certified",),
    "initial_state.invalid_law": ("invalid_argument",),
    "initial_state.invalid_premises": ("invalid_argument",),
    "initial_state.support_gap": ("transport_support_failure",),
    "initial_state.target_law_missing": ("transport_missing_evidence",),
    "initial_state.too_many_states": ("route_not_supported",),
    "initial_state_posterior.band_not_constructed": ("route_not_supported",),
    "initial_state_posterior.empty_law": ("invalid_argument",),
    "initial_state_posterior.external_power_prior": ("route_not_supported",),
    "initial_state_posterior.mapped_prior": ("route_not_supported",),
    "initial_state_posterior.non_positive_scale": ("invalid_argument",),
    "interface_contract.artifact_invalid": ("invalid_argument",),
    "interface_contract.blank_field": ("interface_contract_invalid",),
    "interface_contract.calibration_record_blank": ("interface_contract_invalid",),
    "interface_contract.calibration_without_interval": ("interface_contract_invalid",),
    "interface_contract.conditioning_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.cyclic_composition": ("interface_contract_cyclic_composition",),
    "interface_contract.dependence_unknown": ("interface_contract_assumption_mismatch",),
    "interface_contract.duplicate_component": ("interface_contract_invalid",),
    "interface_contract.duplicate_port": ("interface_contract_invalid",),
    "interface_contract.functional_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.horizon_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.identity_mismatch": ("route_not_supported",),
    "interface_contract.inherited_input_unknown": ("interface_contract_invalid",),
    "interface_contract.invalid_digest": ("interface_contract_invalid",),
    "interface_contract.invalid_quantity": ("interface_contract_invalid",),
    "interface_contract.invalid_support": ("interface_contract_invalid",),
    "interface_contract.measurement_assumption_mismatch": (
        "interface_contract_assumption_mismatch",
    ),
    "interface_contract.missing_operations": ("interface_contract_invalid",),
    "interface_contract.operation_unavailable": ("interface_contract_operation_unavailable",),
    "interface_contract.point_only_with_uncertainty": ("interface_contract_invalid",),
    "interface_contract.population_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.port_multiply_fed": ("interface_contract_invalid",),
    "interface_contract.posterior_without_posterior_meaning": ("interface_contract_invalid",),
    "interface_contract.provenance_missing": ("interface_contract_assumption_mismatch",),
    "interface_contract.regime_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.semantic_identity_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.shared_evidence": ("interface_contract_assumption_mismatch",),
    "interface_contract.support_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.transform_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.uncertainty_meaning_mismatch": ("interface_contract_uncertainty_mismatch",),
    "interface_contract.uncertainty_meaning_missing": ("interface_contract_uncertainty_mismatch",),
    "interface_contract.uncertainty_meaning_required": ("interface_contract_invalid",),
    "interface_contract.uncertainty_meaning_unused": ("interface_contract_invalid",),
    "interface_contract.uncertainty_paradigm_mismatch": (
        "interface_contract_uncertainty_mismatch",
    ),
    "interface_contract.units_mismatch": ("interface_contract_quantity_mismatch",),
    "interface_contract.unknown_component": ("interface_contract_invalid",),
    "interface_contract.unknown_port": ("interface_contract_invalid",),
    "interface_contract.unknown_provenance_stage": ("interface_contract_invalid",),
    "interface_contract.variable_mismatch": ("interface_contract_quantity_mismatch",),
    "intrinsic_influence.correlated_noise": ("route_not_supported",),
    "intrinsic_influence.draw_budget_exceeded": ("route_not_supported",),
    "intrinsic_influence.invalid_settings": ("invalid_argument",),
    "intrinsic_influence.mechanism_unfitted": ("route_not_supported",),
    "intrinsic_influence.noise_independence_undeclared": ("route_not_supported",),
    "intrinsic_influence.noise_not_row_independent": ("route_not_supported",),
    "intrinsic_influence.structure_mismatch": ("invalid_argument",),
    "intrinsic_influence.target_not_continuous": ("route_not_supported",),
    "inverse.cancelled": ("transport_budget_cancel",),
    "inverse.forward_not_mean_response": ("route_not_supported",),
    "inverse.forward_not_point_identified": ("route_not_supported",),
    "inverse.forward_not_supported": ("not_supported",),
    "inverse.invalid_action": ("invalid_argument",),
    "inverse.invalid_forward": ("invalid_argument",),
    "inverse.invalid_target": ("invalid_argument",),
    "inverse.observational_scenarios": ("inverse_functional_unsupported",),
    "inverse.probability_target": ("inverse_functional_unsupported",),
    "inverse.quantile_target": ("inverse_functional_unsupported",),
    "inverse_forward_source.composed_coordinate_is_outcome_law": ("invalid_argument",),
    "inverse_forward_source.composed_family_mismatch": ("mechanism_composition_incoherent",),
    "inverse_forward_source.composed_law_not_materialized": ("decision_contract_unsatisfied",),
    "inverse_forward_source.composed_level_coding": ("mechanism_composition_incoherent",),
    "inverse_forward_source.composed_query_result_mismatch": ("invalid_argument",),
    "inverse_forward_source.composed_response_invalid": ("mechanism_composition_incoherent",),
    "inverse_forward_source.interval_not_supported": ("uncertainty_representation_insufficient",),
    "inverse_forward_source.origin_invalid": ("invalid_argument",),
    "inverse_forward_source.per_action_not_sealable": ("invalid_argument",),
    "inverse_forward_source.point_only_law_requested": ("decision_contract_unsatisfied",),
    "inverse_forward_source.transported_not_accepted": ("decision_contract_unsatisfied",),
    "inverse_observational_scenarios.condition_on_outcome": ("quantity_semantics_mismatch",),
    "inverse_observational_scenarios.coordinate_missing": ("quantity_semantics_mismatch",),
    "inverse_observational_scenarios.coordinate_unsupported": ("quantity_semantics_mismatch",),
    "inverse_observational_scenarios.event_has_no_mass": ("invalid_argument",),
    "inverse_observational_scenarios.invalid_parameter": ("invalid_argument",),
    "inverse_observational_scenarios.invalid_scenarios": ("invalid_argument",),
    "inverse_observational_scenarios.joint_law_required": ("joint_law_required",),
    "inverse_observational_scenarios.meaning_not_observational": ("distribution_meaning_mismatch",),
    "inverse_observational_scenarios.non_finite_draw": ("invalid_argument",),
    "inverse_observational_scenarios.result_mismatch": ("invalid_argument",),
    "inverse_observational_scenarios.unknown_scenario": ("invalid_argument",),
    "joint_cells.cancelled": ("cancelled_no_claim",),
    "joint_cells.cell_empty": ("arm_not_populated",),
    "joint_cells.cell_ess_below_minimum": ("joint_cell_unsupported",),
    "joint_cells.columns": ("invalid_argument",),
    "joint_cells.component_count": ("invalid_argument",),
    "joint_cells.conditional_unsupported": ("joint_cell_unsupported",),
    "joint_cells.config": ("invalid_argument",),
    "joint_cells.contrast_cell_unsupported": ("joint_cell_unsupported",),
    "joint_cells.derived_constituent_copy_in_adjustment": ("derived_treatment_invalid",),
    "joint_cells.derived_constituent_in_adjustment": ("derived_treatment_invalid",),
    "joint_cells.derived_declaration_invalid": ("derived_treatment_invalid",),
    "joint_cells.derived_descendant_in_adjustment": ("derived_treatment_invalid",),
    "joint_cells.derived_duplicate_constituents": ("derived_treatment_invalid",),
    "joint_cells.derived_exclusion_not_in_adjustment": ("derived_treatment_invalid",),
    "joint_cells.derived_exclusion_rule_mismatch": ("derived_treatment_invalid",),
    "joint_cells.derived_illegal_value": ("derived_treatment_invalid",),
    "joint_cells.derived_intervention_not_defined": ("derived_treatment_invalid",),
    "joint_cells.derived_not_a_joint_cell": ("derived_treatment_invalid",),
    "joint_cells.derived_outcome_in_adjustment": ("derived_treatment_invalid",),
    "joint_cells.derived_rank_deficient": ("design_rank_deficient",),
    "joint_cells.derived_treatment_tracked_by_covariates": ("derived_treatment_invalid",),
    "joint_cells.derived_unknown_column": ("invalid_argument",),
    "joint_cells.learner": ("invalid_argument",),
    "joint_cells.learner_unavailable": ("route_not_supported",),
    "joint_cells.ml_learner_not_declared": ("ml_nuisance_not_supported",),
    "joint_cells.no_interval_method": ("penalized_interval_not_supported",),
    "joint_cells.no_ml_interval_method": ("ml_nuisance_not_supported",),
    "joint_cells.no_rows": ("invalid_argument",),
    "joint_cells.no_supported_cell": ("joint_cell_unsupported",),
    "joint_cells.normalization_failed": ("route_not_supported",),
    "joint_cells.not_binary": ("invalid_argument",),
    "joint_cells.nuisance_provider": ("invalid_argument",),
    "joint_cells.ordering": ("invalid_argument",),
    "joint_cells.outcome_model_unsupported": ("joint_cell_unsupported",),
    "joint_law_propagation.alignment_mismatch": ("joint_law_required",),
    "joint_law_propagation.artifact_invalid": ("invalid_argument",),
    "joint_law_propagation.artifact_refused": ("invalid_argument",),
    "joint_law_propagation.bounds_exceeded": ("route_not_supported",),
    "joint_law_propagation.cancelled": ("cancelled_no_claim",),
    "joint_law_propagation.covariance_insufficient_for_nonlinear": ("joint_law_required",),
    "joint_law_propagation.coverage_record_construction_mismatch": ("quantity_semantics_mismatch",),
    "joint_law_propagation.coverage_record_coordinate_mismatch": ("quantity_semantics_mismatch",),
    "joint_law_propagation.dependence_unknown": ("sampling_dependence_unknown",),
    "joint_law_propagation.envelope_is_not_a_point": ("uncertainty_representation_insufficient",),
    "joint_law_propagation.envelope_not_available": ("route_not_supported",),
    "joint_law_propagation.evidence_overlap_unknown": ("sampling_dependence_unknown",),
    "joint_law_propagation.horizon_composition": ("invalid_argument", "option_not_applicable"),
    "joint_law_propagation.horizon_draws_dose": ("invalid_argument",),
    "joint_law_propagation.horizon_draws_layout": ("invalid_argument",),
    "joint_law_propagation.horizon_draws_mean_curve": ("option_not_applicable",),
    "joint_law_propagation.horizon_draws_not_joint": ("joint_law_required",),
    "joint_law_propagation.index_pairing_refused": ("joint_law_required",),
    "joint_law_propagation.invalid_request": ("invalid_argument",),
    "joint_law_propagation.mixture_weights_unknown": ("invalid_argument",),
    "joint_law_propagation.no_interval_construction": ("uncertainty_representation_insufficient",),
    "joint_law_propagation.no_posterior_for_fixed_law_inputs": (
        "uncertainty_representation_insufficient",
    ),
    "joint_law_propagation.nonlinear_needs_aligned_joint_law": ("joint_law_required",),
    "joint_law_propagation.shared_evidence_declared_independent": ("evidence_double_counted",),
    "joint_law_propagation.spread_needs_supported_dependence": (
        "uncertainty_representation_insufficient",
    ),
    "joint_law_propagation.structure_not_verified": ("route_not_supported",),
    "joint_law_propagation.transport_pieces_incomplete": ("joint_law_required",),
    "joint_law_propagation.transported_dependence_required": ("joint_law_required",),
    "joint_sensitivity.bounds_exceeded": ("route_not_supported",),
    "joint_sensitivity.budget": ("transport_budget_cancel",),
    "joint_sensitivity.budget_coupling": ("route_not_supported",),
    "joint_sensitivity.cancelled": ("transport_budget_cancel",),
    "joint_sensitivity.consumer_limits": ("route_not_supported",),
    "joint_sensitivity.data_identity_mismatch": ("transport_not_certified",),
    "joint_sensitivity.duplicate_factor": ("invalid_argument",),
    "joint_sensitivity.factor_count": ("route_not_supported",),
    "joint_sensitivity.fixed_graph_conditional": ("route_not_supported",),
    "joint_sensitivity.fixed_graph_parent": ("route_not_supported",),
    "joint_sensitivity.formula_incompatible": ("transport_not_certified",),
    "joint_sensitivity.incomplete_kernel": ("transport_support_failure",),
    "joint_sensitivity.invalid_fraction": ("invalid_argument",),
    "joint_sensitivity.invalid_threshold": ("invalid_argument",),
    "joint_sensitivity.invalid_tolerance": ("invalid_argument",),
    "joint_sensitivity.no_interval_method": ("route_not_supported",),
    "joint_sensitivity.premises_mismatch": ("transport_not_certified",),
    "joint_sensitivity.provider_mismatch": ("transport_missing_provider",),
    "joint_sensitivity.replay_mismatch": ("transport_not_certified",),
    "joint_sensitivity.source_target_discrepancy": ("route_not_supported",),
    "joint_sensitivity.stage_accounting": ("invalid_argument",),
    "joint_sensitivity.treatment_factor": ("invalid_argument",),
    "joint_sensitivity.union_labelled_ci": ("estimator_inference_mismatch",),
    "joint_sensitivity.unknown_factor": ("invalid_argument",),
    "joint_sensitivity.unsupported_domain": ("route_not_supported",),
    "latent_class.cancelled": ("cancelled_no_claim",),
    "latent_class.constant_outcome": ("population_not_estimable",),
    "latent_class.data_required": ("route_not_supported",),
    "latent_class.degenerate_class": ("population_not_estimable",),
    "latent_class.design_support_too_small": ("design_rank_deficient",),
    "latent_class.export_budget_exceeded": ("route_not_supported",),
    "latent_class.invalid_config": ("invalid_argument",),
    "latent_class.invalid_request": ("invalid_argument",),
    "latent_class.length_mismatch": ("invalid_argument",),
    "latent_class.likelihood_not_monotone": ("mechanism_fit_not_converged",),
    "latent_class.limits_exceeded": ("invalid_argument",),
    "latent_class.memory_budget_exceeded": ("route_not_supported",),
    "latent_class.non_finite_input": ("invalid_argument",),
    "latent_class.not_converged": ("mechanism_fit_not_converged",),
    "latent_class.randomization_not_declared": ("required_option_missing",),
    "latent_class.report_replay_mismatch": ("invalid_argument",),
    "latent_class.too_few_rows": ("invalid_argument",),
    "latent_class.treatment_constant": ("effect_not_identified",),
    "latent_class.weak_class": ("population_not_estimable",),
    "latent_class.wrong_contract": ("route_not_supported",),
    "latent_class_posterior.band_not_constructed": ("route_not_supported",),
    "latent_class_posterior.empty_design": ("invalid_argument",),
    "latent_class_posterior.external_power_prior": ("route_not_supported",),
    "latent_class_posterior.mapped_prior": ("route_not_supported",),
    "latent_class_posterior.non_positive_scale": ("invalid_argument",),
    "learned_continuous_posterior.band_not_constructed": ("route_not_supported",),
    "learned_continuous_posterior.empty_scores": ("invalid_argument",),
    "learned_continuous_posterior.external_power_prior": ("route_not_supported",),
    "learned_continuous_posterior.mapped_prior": ("route_not_supported",),
    "learned_continuous_posterior.non_positive_scale": ("invalid_argument",),
    "learned_joint_transport.admission_refused": ("transport_budget_cancel",),
    "learned_joint_transport.cancelled": ("transport_budget_cancel",),
    "learned_joint_transport.feature_mismatch": ("invalid_argument",),
    "learned_joint_transport.fit_failed": ("transport_numerical_failure",),
    "learned_joint_transport.identification_not_certified": ("transport_missing_evidence",),
    "learned_joint_transport.identification_required": ("transport_missing_evidence",),
    "learned_joint_transport.invalid_basis": ("invalid_argument",),
    "learned_joint_transport.invalid_input": ("invalid_argument",),
    "learned_joint_transport.invalid_prior": ("invalid_argument",),
    "learned_joint_transport.missing_law": ("joint_law_required",),
    "learned_joint_transport.missing_source": ("transport_missing_evidence",),
    "learned_joint_transport.prior_data_overlap": ("invalid_argument",),
    "learned_joint_transport.prior_dimension": ("prior_dimension_mismatch",),
    "learned_joint_transport.rank_deficient": ("design_rank_deficient",),
    "learned_joint_transport.source_dependence": ("sampling_dependence_unknown",),
    "learned_joint_transport.support_resource_limit": ("execution_resource_limit",),
    "learned_joint_transport.too_many_draws": ("invalid_argument",),
    "learned_joint_transport.too_many_parameters": ("invalid_argument",),
    "learned_joint_transport.unsupported_graph": ("route_not_supported",),
    "learned_joint_transport.weak_overlap": ("transport_support_failure",),
    "learned_transport.bootstrap_below_floor": ("estimator_inference_mismatch",),
    "learned_transport.bootstrap_not_supported": ("estimator_inference_mismatch",),
    "learned_transport.cate_requested": ("route_not_supported",),
    "learned_transport.membership_overlap": ("transport_support_failure",),
    "learned_transport.non_iid_design": ("sampling_dependence_unknown",),
    "learned_transport.treatment_overlap": ("transport_support_failure",),
    "mapped_posterior_transfer.coefficient_map": ("quantity_semantics_mismatch",),
    "mapped_posterior_transfer.parameter_posterior": ("distribution_meaning_mismatch",),
    "mapped_posterior_transfer.residual_variance": ("prior_transfer_not_hydrated",),
    "matched_case_control.absolute_risk_not_identified": ("effect_not_identified",),
    "matched_case_control.budget": ("transport_budget_cancel",),
    "matched_case_control.estimand_not_supported": ("route_not_supported",),
    "matched_case_control.estimate_not_finite": ("route_not_supported",),
    "matched_case_control.invalid_data": ("invalid_argument",),
    "matched_case_control.invalid_level": ("invalid_argument",),
    "matched_case_control.no_informative_sets": ("effect_not_identified",),
    "matched_case_control.sampling_design": ("invalid_argument",),
    "measured_inference.artifact_mismatch": ("invalid_argument",),
    "measured_inference.bounds_exceeded": ("route_not_supported",),
    "measured_inference.consumer_bounds_exceeded": (),
    "measured_inference.invalid_expectation": (),
    "measured_inference.invalid_report": ("invalid_argument",),
    "measured_inference.memory_budget_exceeded": (),
    "measured_inference.source_report_not_supported": (),
    "measured_inference.unsupported_route": ("route_not_supported",),
    "measured_interval.draws_not_converged": ("posterior_draws_not_converged",),
    "measured_interval.invalid_source": ("invalid_argument",),
    "measured_interval.no_interval_of_this_kind": ("route_not_supported",),
    "measured_interval.source_mismatch": ("invalid_argument",),
    "measured_interval.variance_assumption_not_met": ("route_not_supported",),
    "measured_transport.input_bounds_exceeded": ("invalid_argument",),
    "measured_transport.sampler_mismatch": ("invalid_argument",),
    "measurement_model.admission_refused": ("invalid_argument",),
    "measurement_model.arm_mismatch": ("measurement_binding_mismatch",),
    "measurement_model.artifact_invalid": ("invalid_argument",),
    "measurement_model.artifact_refused": ("measurement_binding_mismatch",),
    "measurement_model.binary_outcome_continuous_error": ("route_not_supported",),
    "measurement_model.blank_field": ("measurement_model_invalid",),
    "measurement_model.bootstrap_unstable": ("measurement_not_identified",),
    "measurement_model.cancelled": ("cancelled_no_claim",),
    "measurement_model.confusion_ill_conditioned": ("measurement_not_identified",),
    "measurement_model.construct_mismatch": ("measurement_binding_mismatch",),
    "measurement_model.continuous_export": ("route_not_supported",),
    "measurement_model.design_singular": ("design_rank_deficient",),
    "measurement_model.differential_axis_unsupported": ("differential_error_not_pooled",),
    "measurement_model.differential_declared": ("differential_error_not_pooled",),
    "measurement_model.differential_population_unestablished": ("differential_error_not_pooled",),
    "measurement_model.differential_undeclared": ("differential_error_not_pooled",),
    "measurement_model.error_variance_not_identified": ("measurement_not_identified",),
    "measurement_model.indicator_as_construct": ("proxy_offered_as_construct",),
    "measurement_model.indicator_kind": ("measurement_model_invalid",),
    "measurement_model.indicator_mismatch": ("measurement_binding_mismatch",),
    "measurement_model.indicator_observed": ("measurement_model_invalid",),
    "measurement_model.input_invalid": ("invalid_argument",),
    "measurement_model.insufficient_draws": ("measurement_not_identified",),
    "measurement_model.insufficient_replicates": ("measurement_not_identified",),
    "measurement_model.interval_level": ("invalid_argument",),
    "measurement_model.interval_requires_estimate": ("route_not_supported",),
    "measurement_model.intervention_meaning_not_carried": ("representation_mapping_required",),
    "measurement_model.invalid_model": ("measurement_model_invalid",),
    "measurement_model.invariance_not_declared": ("transport_not_certified",),
    "measurement_model.invariance_premise_required": ("transport_missing_evidence",),
    "measurement_model.invariance_source_unknown": ("transport_not_certified",),
    "measurement_model.invariance_target_mismatch": ("transport_not_certified",),
    "measurement_model.learned_coordinates_not_intervenable": ("representation_mapping_required",),
    "measurement_model.mapping_endpoints": ("representation_mapping_required",),
    "measurement_model.no_model": ("measurement_model_invalid",),
    "measurement_model.nothing_to_transport": ("route_not_supported",),
    "measurement_model.operation_not_supported": ("route_not_supported",),
    "measurement_model.outcome_model_not_fitted": ("measurement_not_identified",),
    "measurement_model.parameter": ("measurement_model_invalid",),
    "measurement_model.population_mismatch": ("measurement_binding_mismatch",),
    "measurement_model.posterior_bound_to_snapshot": ("route_not_supported",),
    "measurement_model.posterior_capability": ("measurement_posterior_not_trusted",),
    "measurement_model.posterior_declaration": ("measurement_posterior_not_trusted",),
    "measurement_model.posterior_identity": ("measurement_posterior_not_trusted",),
    "measurement_model.posterior_kind": ("measurement_posterior_not_trusted",),
    "measurement_model.posterior_native_trust": ("measurement_posterior_not_trusted",),
    "measurement_model.posterior_quantity": ("measurement_posterior_not_trusted",),
    "measurement_model.posterior_receipt": ("measurement_posterior_not_trusted",),
    "measurement_model.proxy_as_construct": ("proxy_offered_as_construct",),
    "measurement_model.regime_mismatch": ("measurement_binding_mismatch",),
    "measurement_model.reliability_too_small": ("measurement_not_identified",),
    "measurement_model.representation_unmapped": ("representation_mapping_required",),
    "measurement_model.route_mismatch": ("route_not_supported",),
    "measurement_model.route_parameters": ("measurement_model_invalid",),
    "measurement_model.structure_parameters": ("differential_error_not_pooled",),
    "measurement_model.transport_receipt_mismatch": ("measurement_binding_mismatch",),
    "measurement_model.unknown_level": ("invalid_argument",),
    "measurement_model.unmodelled_proxy": ("proxy_offered_as_construct",),
    "measurement_model.unrecognized": ("measurement_model_invalid",),
    "measurement_model.variance_not_positive": ("measurement_not_identified",),
    "measurement_model.youden_too_small": ("measurement_not_identified",),
    "mechanism_composition.artifact_invalid": ("invalid_argument",),
    "mechanism_composition.artifact_kind_mismatch": ("invalid_argument",),
    "mechanism_composition.cancelled": ("cancelled_no_claim",),
    "mechanism_composition.conditioning_extra_conditioner": ("mechanism_composition_incoherent",),
    "mechanism_composition.conditioning_parent_not_conditioned": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.cycle_directed_cycle": ("graph_invalid",),
    "mechanism_composition.family_mixed": ("mechanism_composition_incoherent",),
    "mechanism_composition.identity_declaration_unknown_variable": ("quantity_semantics_mismatch",),
    "mechanism_composition.identity_duplicate_mechanism": ("quantity_semantics_mismatch",),
    "mechanism_composition.identity_edge_unknown_variable": ("quantity_semantics_mismatch",),
    "mechanism_composition.identity_parent_without_mechanism": ("quantity_semantics_mismatch",),
    "mechanism_composition.interval_invalid": ("invalid_argument",),
    "mechanism_composition.latent_family_unsupported": ("mechanism_composition_incoherent",),
    "mechanism_composition.latent_loading_undeclared": ("mechanism_composition_incoherent",),
    "mechanism_composition.latent_shared_with_absent_mechanism": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.latent_sharing_not_declared": ("mechanism_composition_incoherent",),
    "mechanism_composition.latent_sharing_not_reciprocal": ("mechanism_composition_incoherent",),
    "mechanism_composition.latent_too_many": ("mechanism_composition_incoherent",),
    "mechanism_composition.numbers_unbound": ("invalid_argument",),
    "mechanism_composition.numerical_failure": ("invalid_argument",),
    "mechanism_composition.parents_declared_graph_mismatch": ("graph_invalid",),
    "mechanism_composition.population_invariance_unstated": ("quantity_semantics_mismatch",),
    "mechanism_composition.population_latent_blocks_selection_admissibility": (
        "quantity_semantics_mismatch",
    ),
    "mechanism_composition.population_parent_context_differs_from_fit": (
        "quantity_semantics_mismatch",
    ),
    "mechanism_composition.population_selection_arrow_into_mechanism": (
        "quantity_semantics_mismatch",
    ),
    "mechanism_composition.population_undeclared_transport": ("quantity_semantics_mismatch",),
    "mechanism_composition.posterior_dependence_cross_block_duplicate": ("joint_law_required",),
    "mechanism_composition.posterior_dependence_cross_covariance_missing": ("joint_law_required",),
    "mechanism_composition.posterior_dependence_evidence_overlap_unknown": ("joint_law_required",),
    "mechanism_composition.posterior_dependence_independence_unjustified": ("joint_law_required",),
    "mechanism_composition.posterior_dependence_shared_evidence_declared_independent": (
        "joint_law_required",
    ),
    "mechanism_composition.posterior_dependence_unknown": ("joint_law_required",),
    "mechanism_composition.premises_mismatch": ("mechanism_composition_incoherent",),
    "mechanism_composition.provenance_declared_mechanism_unacknowledged": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.provenance_missing_artifact": ("mechanism_composition_incoherent",),
    "mechanism_composition.provenance_missing_evidence": ("mechanism_composition_incoherent",),
    "mechanism_composition.quantity_conditioned_event": ("quantity_semantics_mismatch",),
    "mechanism_composition.quantity_conditioning_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_composition.quantity_functional_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_composition.quantity_horizon_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_composition.quantity_invalid_coordinate": ("quantity_semantics_mismatch",),
    "mechanism_composition.quantity_not_a_realization": ("quantity_semantics_mismatch",),
    "mechanism_composition.quantity_role_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_composition.regime_parent_context_differs_from_fit": (
        "quantity_semantics_mismatch",
    ),
    "mechanism_composition.regime_selection_diagram_not_applicable": (
        "quantity_semantics_mismatch",
    ),
    "mechanism_composition.regime_undeclared_invariance": ("quantity_semantics_mismatch",),
    "mechanism_composition.request_unsupported": ("route_not_supported",),
    "mechanism_composition.result_mismatch": ("mechanism_composition_incoherent",),
    "mechanism_composition.shape_blank_coordinate": ("invalid_argument",),
    "mechanism_composition.shape_blank_identity": ("invalid_argument",),
    "mechanism_composition.shape_coefficient_parent_mismatch": ("invalid_argument",),
    "mechanism_composition.shape_duplicate_parent": ("invalid_argument",),
    "mechanism_composition.shape_empty_composition": ("invalid_argument",),
    "mechanism_composition.shape_invalid_domain": ("invalid_argument",),
    "mechanism_composition.shape_invalid_latents": ("invalid_argument",),
    "mechanism_composition.shape_invalid_table": ("invalid_argument",),
    "mechanism_composition.shape_mechanism_count": ("invalid_argument",),
    "mechanism_composition.shape_negative_scale": ("invalid_argument",),
    "mechanism_composition.shape_non_finite_parameter": ("invalid_argument",),
    "mechanism_composition.shape_too_many_mechanisms": ("invalid_argument",),
    "mechanism_composition.shape_too_many_parents": ("invalid_argument",),
    "mechanism_composition.slot_not_linear_gaussian": ("mechanism_composition_incoherent",),
    "mechanism_composition.support_disjoint": ("mechanism_composition_incoherent",),
    "mechanism_composition.support_intervention_outside_domain": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.support_kind_mismatch": ("mechanism_composition_incoherent",),
    "mechanism_composition.support_missing_parent_configuration": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.support_not_contained": ("mechanism_composition_incoherent",),
    "mechanism_composition.transform_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_composition.uncertainty_assembled_not_psd": ("mechanism_composition_incoherent",),
    "mechanism_composition.uncertainty_cross_block_shape": ("mechanism_composition_incoherent",),
    "mechanism_composition.uncertainty_cross_block_unknown_mechanism": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.uncertainty_family_unsupported": ("mechanism_composition_incoherent",),
    "mechanism_composition.uncertainty_joint_parameter_set_mismatch": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.uncertainty_marginal_inconsistent_with_joint": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.uncertainty_marginal_missing": ("mechanism_composition_incoherent",),
    "mechanism_composition.uncertainty_meaning_mismatch": ("mechanism_composition_incoherent",),
    "mechanism_composition.uncertainty_not_psd": ("mechanism_composition_incoherent",),
    "mechanism_composition.uncertainty_not_symmetric": ("mechanism_composition_incoherent",),
    "mechanism_composition.uncertainty_parameter_set_mismatch": (
        "mechanism_composition_incoherent",
    ),
    "mechanism_composition.uncertainty_shape": ("mechanism_composition_incoherent",),
    "mechanism_composition.units_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_discrepancy.degenerate_covariance": ("invalid_argument",),
    "mechanism_discrepancy.dependence_unknown": ("route_not_supported",),
    "mechanism_discrepancy.duplicate_name": ("invalid_argument",),
    "mechanism_discrepancy.incomparable_measurements": ("route_not_supported",),
    "mechanism_discrepancy.inconsistent_summary": ("invalid_argument",),
    "mechanism_discrepancy.invalid_alpha": ("invalid_argument",),
    "mechanism_discrepancy.invalid_artifact": ("invalid_argument",),
    "mechanism_discrepancy.invalid_identity": ("invalid_argument",),
    "mechanism_discrepancy.invalid_power": ("invalid_argument",),
    "mechanism_discrepancy.invalid_request": ("invalid_argument",),
    "mechanism_discrepancy.no_compared_coefficients": ("invalid_argument",),
    "mechanism_discrepancy.non_finite_value": ("invalid_argument",),
    "mechanism_discrepancy.rank_deficient_design": ("invalid_argument",),
    "mechanism_discrepancy.row_count_mismatch": ("invalid_argument",),
    "mechanism_discrepancy.sample_too_small": ("invalid_argument",),
    "mechanism_discrepancy.too_many_parents": ("invalid_argument",),
    "mechanism_discrepancy.wrong_contract": ("route_not_supported",),
    "mechanism_posterior.band_not_constructed": ("route_not_supported",),
    "mechanism_posterior.empty_summary": ("invalid_argument",),
    "mechanism_posterior.external_power_prior": ("route_not_supported",),
    "mechanism_posterior.mapped_prior": ("route_not_supported",),
    "mechanism_posterior.non_positive_scale": ("invalid_argument",),
    "mechanism_transport.diagram_source_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_transport.diagram_target_mismatch": ("quantity_semantics_mismatch",),
    "mechanism_transport.duplicate_mechanism": ("quantity_semantics_mismatch",),
    "mechanism_transport.family_unsupported": ("mechanism_composition_incoherent",),
    "mechanism_transport.origin_not_fitted": ("mechanism_composition_incoherent",),
    "mechanism_transport.selection_arrow_into_latent": ("quantity_semantics_mismatch",),
    "mechanism_transport.target_mechanism_not_native": ("quantity_semantics_mismatch",),
    "mechanism_transport.transport_identity_mismatch": ("quantity_semantics_mismatch",),
    "mediation_posterior.band_not_constructed": ("route_not_supported",),
    "mediation_posterior.empty_design": ("invalid_argument",),
    "mediation_posterior.external_power_prior": ("route_not_supported",),
    "mediation_posterior.mapped_prior": ("route_not_supported",),
    "mediation_posterior.non_positive_scale": ("invalid_argument",),
    "missing_data.covariates_incomplete": ("missing_policy_not_applicable",),
    "missing_data.multiple_imputation": ("missing_policy_not_applicable",),
    "missing_data.policy_not_applicable": ("missing_policy_not_applicable",),
    "missing_data.policy_required": ("missing_policy_required",),
    "mixed_search.bounds_exceeded": ("route_not_supported",),
    "mixed_search.budget": ("transport_budget_cancel",),
    "mixed_search.invalid_derivation": ("transport_not_certified",),
    "mixed_search.invalid_query": ("invalid_argument",),
    "mixed_search.missing_joint": ("transport_missing_evidence",),
    "mixed_search.named_route": ("route_not_supported",),
    "mixed_search.not_certified": ("transport_not_certified",),
    "mixed_search.posterior_as_law": ("route_not_supported",),
    "mnar_fixed_shift.cancelled": ("execution_cancelled",),
    "mnar_fixed_shift.constant_outcome": ("invalid_argument",),
    "mnar_fixed_shift.degenerate_interval": ("invalid_argument",),
    "mnar_fixed_shift.duplicate_subject": ("invalid_argument",),
    "mnar_fixed_shift.insufficient_observations": ("invalid_argument",),
    "mnar_fixed_shift.interval_not_supported": ("invalid_argument",),
    "mnar_fixed_shift.invalid_declaration": ("invalid_argument",),
    "mnar_fixed_shift.invalid_outcome": ("invalid_argument",),
    "mnar_fixed_shift.invalid_shift": ("invalid_argument",),
    "mnar_fixed_shift.invalid_source": ("invalid_argument",),
    "mnar_fixed_shift.lowering": ("invalid_argument",),
    "mnar_fixed_shift.preparation": ("invalid_argument",),
    "mnar_fixed_shift.preparation_cancelled": ("execution_cancelled",),
    "mnar_fixed_shift.preparation_resource_limit": ("execution_resource_limit",),
    "mnar_fixed_shift.source_admission_cancelled": ("execution_cancelled",),
    "mnar_fixed_shift.source_admission_resource_limit": ("execution_resource_limit",),
    "mnar_sensitivity.bounds_exceeded": ("route_not_supported",),
    "mnar_sensitivity.composition_not_supported": ("route_not_supported",),
    "mnar_sensitivity.covariates_not_used": ("invalid_argument",),
    "mnar_sensitivity.covariates_required": ("invalid_argument",),
    "mnar_sensitivity.design_singular": ("invalid_argument",),
    "mnar_sensitivity.insufficient_observed": ("invalid_argument",),
    "mnar_sensitivity.invalid_bootstrap": ("invalid_argument",),
    "mnar_sensitivity.invalid_data": ("invalid_argument",),
    "mnar_sensitivity.invalid_decision": ("invalid_argument",),
    "mnar_sensitivity.invalid_level": ("invalid_argument",),
    "mnar_sensitivity.invalid_loading": ("invalid_argument",),
    "mnar_sensitivity.invalid_tolerance": ("invalid_argument",),
    "mnar_sensitivity.mechanism_not_declared": ("required_option_missing",),
    "mnar_sensitivity.range_excludes_mar": ("invalid_argument",),
    "mnar_sensitivity.unbounded_range": ("invalid_argument",),
    "mnar_sensitivity.unsupported_estimator": ("route_not_supported",),
    "mnar_sensitivity.unsupported_mechanism": ("route_not_supported",),
    "mnar_sensitivity.unsupported_query": ("route_not_supported",),
    "msm_posterior.band_not_constructed": ("route_not_supported",),
    "msm_posterior.empty_law": ("invalid_argument",),
    "msm_posterior.external_power_prior": ("route_not_supported",),
    "msm_posterior.mapped_prior": ("route_not_supported",),
    "msm_posterior.non_positive_scale": ("invalid_argument",),
    "msm_sensitivity.bounds_exceeded": ("route_not_supported",),
    "msm_sensitivity.composition_not_supported": ("route_not_supported",),
    "msm_sensitivity.invalid_threshold": ("invalid_argument",),
    "msm_sensitivity.invalid_tolerance": ("invalid_argument",),
    "msm_sensitivity.lambda_below_one": ("invalid_argument",),
    "msm_sensitivity.lambda_range_empty": ("invalid_argument",),
    "msm_sensitivity.no_interval_method": ("uncertainty_representation_insufficient",),
    "msm_sensitivity.outcome_law": ("invalid_argument",),
    "msm_sensitivity.positivity": ("route_not_supported",),
    "msm_sensitivity.stratum_mass": ("invalid_argument",),
    "multi_source_posterior.band_not_constructed": ("route_not_supported",),
    "multi_source_posterior.empty_law": ("invalid_argument",),
    "multi_source_posterior.external_power_prior": ("route_not_supported",),
    "multi_source_posterior.mapped_prior": ("route_not_supported",),
    "multi_source_posterior.non_positive_scale": ("invalid_argument",),
    "multiple_imputation.admission_refused": ("invalid_argument",),
    "multiple_imputation.ambiguous_variable_domain": ("invalid_argument",),
    "multiple_imputation.binary_under_joint_normal": ("missing_policy_not_applicable",),
    "multiple_imputation.cancelled": ("cancelled_no_claim",),
    "multiple_imputation.clusters_required": ("missing_policy_not_applicable",),
    "multiple_imputation.column_length": ("invalid_argument",),
    "multiple_imputation.conditional_model_not_fitted": ("missing_policy_not_applicable",),
    "multiple_imputation.dependent_series": ("missing_policy_not_applicable",),
    "multiple_imputation.dynamics_joint_normal": ("missing_policy_not_applicable",),
    "multiple_imputation.dynamics_lags": ("invalid_argument",),
    "multiple_imputation.graph_posterior_atoms": ("missing_policy_not_applicable",),
    "multiple_imputation.graph_posterior_frequentist": ("missing_policy_not_applicable",),
    "multiple_imputation.graph_posterior_query": ("missing_policy_not_applicable",),
    "multiple_imputation.graph_posterior_series": ("missing_policy_not_applicable",),
    "multiple_imputation.index": ("invalid_argument",),
    "multiple_imputation.interaction_variable": ("invalid_argument",),
    "multiple_imputation.interactions_joint_normal": ("missing_policy_not_applicable",),
    "multiple_imputation.mixed_fits": ("invalid_argument",),
    "multiple_imputation.multi_horizon_posterior_not_joint": ("missing_policy_not_applicable",),
    "multiple_imputation.non_finite": ("invalid_argument",),
    "multiple_imputation.not_at_random": ("missing_policy_not_applicable",),
    "multiple_imputation.not_identified": ("missing_policy_not_applicable",),
    "multiple_imputation.pool_df": ("invalid_argument",),
    "multiple_imputation.pool_non_finite": ("invalid_argument",),
    "multiple_imputation.pool_shape": ("invalid_argument",),
    "multiple_imputation.sampling_design": ("invalid_argument",),
    "multiple_imputation.spec_range": ("invalid_argument",),
    "multiple_imputation.stack_shape": ("invalid_argument",),
    "multiple_imputation.too_few_imputations": ("invalid_argument",),
    "multiple_imputation.units_labels": ("invalid_argument",),
    "multiple_imputation.units_shape": ("invalid_argument",),
    "multiple_imputation.variable_count": ("invalid_argument",),
    "multiple_imputation.variable_domain": ("invalid_argument",),
    "multiple_imputation.within_singular": ("invalid_argument",),
    "mz_transport.bounds_exceeded": ("route_not_supported",),
    "mz_transport.budget": ("transport_budget_cancel",),
    "mz_transport.checked_obstruction": ("transport_proven_non_transportable",),
    "mz_transport.empirical_counts_required": ("transport_missing_provider",),
    "mz_transport.fabricated_joint": ("transport_not_certified",),
    "mz_transport.functional_graph_mismatch": ("invalid_argument",),
    "mz_transport.invalid_catalog": ("invalid_argument",),
    "mz_transport.invalid_derivation": ("transport_not_certified",),
    "mz_transport.invalid_obstruction": ("transport_not_certified",),
    "mz_transport.invalid_query": ("invalid_argument",),
    "mz_transport.missing_joint_regime": ("transport_missing_evidence",),
    "mz_transport.search_incomplete": ("transport_not_certified",),
    "native_attempt.invalid_callable": ("invalid_argument",),
    "native_claims.calibration_not_supported": ("invalid_argument",),
    "native_claims.contract_mismatch": ("invalid_argument",),
    "native_claims.grid_mismatch": ("quantity_semantics_mismatch",),
    "native_claims.invalid_context": ("invalid_argument",),
    "native_claims.joint_law_invalid": ("invalid_argument",),
    "native_claims.native_state_unavailable": ("invalid_argument",),
    "native_claims.no_source_requirement": ("invalid_argument",),
    "native_claims.no_supported_coordinate": ("quantity_semantics_mismatch",),
    "native_claims.point_support_missing": ("quantity_semantics_mismatch",),
    "native_claims.program_mismatch": ("invalid_argument",),
    "native_claims.projection_mismatch": ("invalid_argument",),
    "native_claims.provenance_mismatch": ("invalid_argument",),
    "native_claims.response_not_point_identified": ("effect_not_identified",),
    "native_claims.source_not_supplied": ("decision_contract_unsatisfied",),
    "native_claims.units_conflict": ("quantity_semantics_mismatch",),
    "native_claims.unknown_variable": ("unknown_variable",),
    "native_claims.unsupported_estimand": ("route_not_supported",),
    "native_distribution.authority_required": ("invalid_argument",),
    "nested_markov.bayesian": ("invalid_argument",),
    "nested_markov.bayesian_artifact": ("invalid_argument",),
    "nested_markov.bayesian_artifact_limits": ("invalid_argument",),
    "nested_markov.bayesian_identity_mismatch": ("invalid_argument",),
    "nested_markov.bayesian_invalid_options": ("invalid_argument",),
    "nested_markov.bayesian_invalid_prior": ("invalid_argument",),
    "nested_markov.bayesian_sampler_scope": ("route_not_supported",),
    "nested_markov.cancelled": ("transport_budget_cancel",),
    "nested_markov.constraint_violated": ("transport_numerical_failure",),
    "nested_markov.fisher_artifact_limits": ("invalid_argument",),
    "nested_markov.fisher_identity_mismatch": ("invalid_argument",),
    "nested_markov.fisher_invalid_covariance": ("invalid_argument",),
    "nested_markov.fisher_invalid_level": ("invalid_argument",),
    "nested_markov.fisher_invalid_options": ("invalid_argument",),
    "nested_markov.fisher_sampling_design": ("invalid_argument",),
    "nested_markov.fisher_singular_information": ("invalid_argument",),
    "nested_markov.fit_not_converged": ("transport_numerical_failure",),
    "nested_markov.fitted_boundary": ("transport_numerical_failure",),
    "nested_markov.identification_unavailable": ("effect_not_identified",),
    "nested_markov.invalid_counts": ("invalid_argument",),
    "nested_markov.invalid_input": ("invalid_argument",),
    "nested_markov.invalid_options": ("invalid_argument",),
    "nested_markov.iteration_bound": ("invalid_argument",),
    "nested_markov.measured_scope": ("route_not_supported",),
    "nested_markov.native_hook_unavailable": ("not_supported",),
    "nested_markov.not_normalized": ("transport_numerical_failure",),
    "nested_markov.outside_binary_pilot": ("route_not_supported",),
    "nested_markov.positivity": ("transport_support_failure",),
    "nested_markov.producer_missing": ("not_supported",),
    "nonlinear_mediation.cancelled": ("cancelled_no_claim",),
    "nonlinear_mediation.confounding": ("effect_not_identified",),
    "nonlinear_mediation.cross_world_independence_not_declared": ("cross_world_not_identified",),
    "nonlinear_mediation.data_required": ("route_not_supported",),
    "nonlinear_mediation.degenerate_mediator": ("invalid_argument",),
    "nonlinear_mediation.degenerate_mediator_residual": ("invalid_argument",),
    "nonlinear_mediation.degenerate_outcome": ("invalid_argument",),
    "nonlinear_mediation.degree_above_limit": ("invalid_argument",),
    "nonlinear_mediation.export_budget_exceeded": ("route_not_supported",),
    "nonlinear_mediation.identity_violation": ("mechanism_fit_not_converged",),
    "nonlinear_mediation.integration_error": ("mechanism_fit_not_converged",),
    "nonlinear_mediation.interventional_effects_not_supported": ("route_not_supported",),
    "nonlinear_mediation.invalid_degree": ("invalid_argument",),
    "nonlinear_mediation.invalid_nodes": ("invalid_argument",),
    "nonlinear_mediation.invalid_request": ("invalid_argument",),
    "nonlinear_mediation.invalid_support_bound": ("invalid_argument",),
    "nonlinear_mediation.invalid_tolerance": ("invalid_argument",),
    "nonlinear_mediation.length_mismatch": ("invalid_argument",),
    "nonlinear_mediation.limits_exceeded": ("invalid_argument",),
    "nonlinear_mediation.memory_budget_exceeded": ("route_not_supported",),
    "nonlinear_mediation.nodes_above_limit": ("invalid_argument",),
    "nonlinear_mediation.non_finite_input": ("invalid_argument",),
    "nonlinear_mediation.non_finite_option": ("invalid_argument",),
    "nonlinear_mediation.overlap": ("arm_not_populated",),
    "nonlinear_mediation.quadrature_nodes_not_converged": ("mechanism_fit_not_converged",),
    "nonlinear_mediation.quadrature_weights_invalid": ("mechanism_fit_not_converged",),
    "nonlinear_mediation.rank_deficient": ("design_rank_deficient",),
    "nonlinear_mediation.report_replay_mismatch": ("invalid_argument",),
    "nonlinear_mediation.too_few_rows": ("design_rank_deficient",),
    "nonlinear_mediation.treatment_induced_confounding": ("cross_world_not_identified",),
    "nonlinear_mediation.treatment_not_binary": ("invalid_argument",),
    "nonlinear_mediation.wrong_contract": ("route_not_supported",),
    "observation_query.arm_mismatch": ("measurement_binding_mismatch",),
    "observation_query.artifact_invalid": ("invalid_argument",),
    "observation_query.artifact_refused": ("invalid_argument",),
    "observation_query.blank_field": ("invalid_argument",),
    "observation_query.cancelled": ("cancelled_no_claim",),
    "observation_query.construct_mismatch": ("measurement_binding_mismatch",),
    "observation_query.construct_observed_directly": ("invalid_argument",),
    "observation_query.count_fitted_bounds_exceeded": ("invalid_argument",),
    "observation_query.count_fitted_refused": ("invalid_argument",),
    "observation_query.declared_model_has_no_sampling_interval": (
        "uncertainty_representation_insufficient",
    ),
    "observation_query.differential_error_unsupported": ("differential_error_not_pooled",),
    "observation_query.evidence_covariance_singular": ("invalid_argument",),
    "observation_query.indicator_kind": ("invalid_argument",),
    "observation_query.indicator_mismatch": ("measurement_binding_mismatch",),
    "observation_query.intervention_in_conditioning_slot": ("quantity_semantics_mismatch",),
    "observation_query.invalid_declaration": ("invalid_argument",),
    "observation_query.invalid_interval": ("invalid_argument",),
    "observation_query.invalid_request": ("invalid_argument",),
    "observation_query.level_unreachable": ("evidence_cannot_reach_level",),
    "observation_query.levels_for_continuous_model": ("invalid_argument",),
    "observation_query.measured_as_construct": ("proxy_offered_as_construct",),
    "observation_query.measured_background": ("route_not_supported",),
    "observation_query.measured_signal_not_redrawn": ("route_not_supported",),
    "observation_query.measurement_route": ("route_not_supported",),
    "observation_query.mixed_conditioning_and_intervention": ("route_not_supported",),
    "observation_query.multiple_measured_signals": ("route_not_supported",),
    "observation_query.no_calibrated_probability": ("uncertainty_representation_insufficient",),
    "observation_query.no_parameter_draws": ("invalid_argument",),
    "observation_query.outcome_assigned": ("invalid_argument",),
    "observation_query.population_mismatch": ("measurement_binding_mismatch",),
    "observation_query.positive_level": ("invalid_argument",),
    "observation_query.real_values_for_finite_model": ("invalid_argument",),
    "observation_query.regime_mismatch": ("measurement_binding_mismatch",),
    "observation_query.repeated_slot": ("invalid_argument",),
    "observation_query.request_unsupported": ("route_not_supported",),
    "observation_query.unknown_level": ("invalid_argument",),
    "observation_query.unknown_quantity": ("invalid_argument",),
    "observation_query.unknown_variable": ("unknown_variable",),
    "observation_query.value_not_finite": ("invalid_argument",),
    "observation_query.zero_probability_evidence": ("invalid_argument",),
    "paradigm_parity.paradigm_not_supported": ("route_not_supported",),
    "parent_set_effect.conflict_resolution_budget": ("discovered_structure_unresolved",),
    "parent_set_effect.discovery_failed": ("invalid_argument",),
    "parent_set_effect.knowledge_invalid": ("invalid_argument",),
    "parent_set_effect.log_linear_input_not_positive": ("invalid_argument",),
    "parent_set_effect.outcome_precedes_treatment": ("invalid_argument",),
    "parent_set_effect.parent_limit_bound": ("discovered_structure_unresolved",),
    "parent_set_effect.search_not_finished": ("discovered_structure_unresolved",),
    "parent_set_effect.source_invalid": ("invalid_argument",),
    "parent_set_effect.source_not_canonical": ("invalid_argument",),
    "parent_set_effect.structure_class": ("invalid_argument",),
    "parent_set_effect.transformation_undeclared": ("invalid_argument",),
    "parent_set_effect.treatment_orientation_depends_on_conflict": (
        "discovered_structure_unresolved",
    ),
    "parent_set_effect.treatment_parents_ambiguous": ("discovered_structure_unresolved",),
    "penalized_posterior.band_not_constructed": ("route_not_supported",),
    "penalized_posterior.empty_scores": ("invalid_argument",),
    "penalized_posterior.external_power_prior": ("route_not_supported",),
    "penalized_posterior.mapped_prior": ("route_not_supported",),
    "penalized_posterior.non_positive_scale": ("invalid_argument",),
    "penalized_propensity.cancelled": ("cancelled_no_claim",),
    "penalized_propensity.effect_posterior": ("route_not_supported",),
    "penalized_propensity.fallback_not_supported": ("nuisance_fallback_not_supported",),
    "penalized_propensity.fallback_with_penalty": ("invalid_argument",),
    "penalized_propensity.grid_allocation": ("execution_resource_limit",),
    "penalized_propensity.invalid_penalty": ("invalid_argument",),
    "penalized_propensity.scope": ("route_not_supported",),
    "penalized_propensity.selection_not_supported": ("selection_inference_not_supported",),
    "policy_value.action_not_populated": ("arm_not_populated",),
    "policy_value.action_outside_support": ("positivity_violated",),
    "policy_value.action_propensity_below_floor": ("positivity_violated",),
    "policy_value.cancelled": ("cancelled_no_claim",),
    "policy_value.evaluation_fraction_missing": ("required_option_missing",),
    "policy_value.evaluation_missing": ("required_option_missing",),
    "policy_value.evaluation_unknown": ("invalid_argument",),
    "policy_value.invalid_data": ("invalid_argument",),
    "policy_value.invalid_evaluation_fraction": ("invalid_argument",),
    "policy_value.invalid_folds": ("invalid_argument",),
    "policy_value.invalid_known_propensity": ("invalid_argument",),
    "policy_value.invalid_propensity_floor": ("invalid_argument",),
    "policy_value.invalid_request": ("invalid_argument",),
    "policy_value.invalid_support": ("invalid_argument",),
    "policy_value.objective_missing": ("required_option_missing",),
    "policy_value.objective_unknown": ("invalid_argument",),
    "policy_value.policy_actions_missing": ("required_option_missing",),
    "policy_value.policy_learned_on_evaluation_rows": ("policy_learned_on_evaluation_rows",),
    "policy_value.policy_origin_missing": ("required_option_missing",),
    "policy_value.policy_origin_unknown": ("invalid_argument",),
    "policy_value.reference_outside_support": ("invalid_argument",),
    "policy_value.sparse_action": ("arm_not_populated",),
    "policy_value.support_missing": ("required_option_missing",),
    "policy_value.too_few_evaluation_rows": ("invalid_argument",),
    "policy_value.too_few_rows": ("invalid_argument",),
    "policy_value.training_split_missing_action": ("arm_not_populated",),
    "policy_value.undeclared_level": ("treatment_level_undeclared",),
    "predictive_assessment.bounds_exceeded": ("route_not_supported",),
    "predictive_assessment.cancelled": ("cancelled_no_claim",),
    "predictive_assessment.design_invalid": ("invalid_argument",),
    "predictive_assessment.diagnostic_not_defined": ("invalid_argument",),
    "predictive_assessment.horizon_mismatch": ("quantity_semantics_mismatch",),
    "predictive_assessment.iid_not_supported_for_dependent_design": (
        "sampling_dependence_unknown",
    ),
    "predictive_assessment.in_sample": ("evidence_double_counted",),
    "predictive_assessment.interval_dependence_undeclared": ("sampling_dependence_unknown",),
    "predictive_assessment.interval_law_family_mixed": ("route_not_supported",),
    "predictive_assessment.interval_not_supported": ("quantity_semantics_mismatch",),
    "predictive_assessment.interval_request_invalid": ("invalid_argument",),
    "predictive_assessment.invalid_artifact": ("invalid_argument",),
    "predictive_assessment.invalid_law": ("invalid_argument",),
    "predictive_assessment.invalid_observations": ("invalid_argument",),
    "predictive_assessment.law_callback_not_built": ("route_not_supported",),
    "predictive_assessment.log_score_infinite": ("invalid_argument",),
    "predictive_assessment.not_a_causal_claim": ("route_not_supported",),
    "predictive_assessment.outcome_kind": ("invalid_argument",),
    "predictive_assessment.overlap_unknown": ("evidence_overlap_unknown",),
    "predictive_assessment.provider_identity_incomplete": ("invalid_argument",),
    "predictive_assessment.report_replay_mismatch": ("invalid_argument",),
    "predictive_assessment.score_not_defined": ("route_not_supported",),
    "predictive_assessment.score_not_probability": ("route_not_supported",),
    "predictive_assessment.serial_lag_below_horizon": ("sampling_dependence_unknown",),
    "predictive_assessment.target_mismatch": ("quantity_semantics_mismatch",),
    "predictive_assessment.too_few_clusters": ("too_few_clusters",),
    "predictive_assessment.unknown_source": ("invalid_argument",),
    "predictive_assessment.wrong_contract": ("route_not_supported",),
    "prior_signal.adapted_prior_not_draws": ("invalid_argument",),
    "prior_signal.artifact_not_in_catalog": ("design_signal_invalid",),
    "prior_signal.coefficient_prior_not_signal": ("design_signal_invalid",),
    "prior_signal.duplicate_source": ("design_signal_invalid",),
    "prior_signal.incompatible_artifact": ("design_signal_invalid",),
    "prior_signal.incomplete_posterior": ("design_signal_invalid",),
    "prior_signal.invalid_parameter": ("invalid_argument",),
    "prior_signal.invalid_request": ("invalid_argument",),
    "prior_signal.lineage_missing": ("design_signal_invalid",),
    "prior_signal.named_state_draws_required": ("design_signal_invalid",),
    "prior_signal.no_sources": ("design_signal_invalid",),
    "prior_signal.nonfinite_mapped_prior": ("invalid_argument",),
    "prior_signal.observation_lineage_missing": ("design_signal_invalid",),
    "prior_signal.observation_overlap_unknown": ("evidence_overlap_unknown",),
    "prior_signal.observation_reuse": ("design_signal_invalid",),
    "prior_signal.original_posterior_unidentified": ("design_signal_invalid",),
    "prior_signal.posterior_draws_required": ("design_signal_invalid",),
    "prior_signal.prior_identity_mismatch": ("invalid_argument",),
    "prior_signal.prior_projection_mismatch": ("invalid_argument",),
    "prior_signal.prior_sources_overlap": ("design_signal_invalid",),
    "prior_signal.resolution_invalid": ("invalid_argument",),
    "prior_signal.signal_family_mismatch": ("design_signal_invalid",),
    "prior_signal.source_not_in_catalog_or_duplicate": ("design_signal_invalid",),
    "prior_signal.source_population_mismatch": ("invalid_argument",),
    "prior_signal.state_quantity_mismatch": ("invalid_argument",),
    "prior_signal.target_population_mismatch": ("invalid_argument",),
    "prior_signal.transport_policy_required": ("transport_policy_required",),
    "prior_signal.unlock_list_not_signal": ("design_signal_invalid",),
    "prior_signal.unsupported_signal_provider": ("route_not_supported",),
    "program_binding.contract_identity_mismatch": ("external_binding_mismatch",),
    "program_binding.dose_grid_changed": ("quantity_semantics_mismatch",),
    "program_binding.graph_only_identity": ("external_binding_mismatch",),
    "program_binding.invalid_binding": ("invalid_argument",),
    "program_binding.malformed_quantity": ("invalid_argument",),
    "program_binding.no_identified_request": ("invalid_argument",),
    "program_binding.population_mismatch": ("quantity_semantics_mismatch",),
    "program_binding.quantities_override_mismatch": ("quantity_semantics_mismatch",),
    "program_binding.treatment_outcome_substitution": ("external_binding_mismatch",),
    "proposal_arrival.assignment_mismatch": ("invalid_argument",),
    "proposal_arrival.base_catalog_changed": ("invalid_argument",),
    "proposal_arrival.base_catalog_missing": ("invalid_argument",),
    "proposal_arrival.base_provider_mismatch": ("invalid_argument",),
    "proposal_arrival.binding_missing": ("invalid_argument",),
    "proposal_arrival.bounds_exceeded": ("route_not_supported",),
    "proposal_arrival.cancelled": ("route_not_supported",),
    "proposal_arrival.candidate_missing": ("invalid_argument",),
    "proposal_arrival.duplicate_original_artifacts": ("invalid_argument",),
    "proposal_arrival.expected_proposal_mismatch": ("invalid_argument",),
    "proposal_arrival.family_not_supported": ("route_not_supported",),
    "proposal_arrival.identity_mismatch": ("invalid_argument",),
    "proposal_arrival.independent_joint_study_required": ("route_not_supported",),
    "proposal_arrival.invalid_request": ("invalid_argument",),
    "proposal_arrival.original_artifacts_required": ("invalid_argument",),
    "proposal_arrival.original_consumer_refused": ("invalid_argument",),
    "proposal_arrival.proof_not_bound": ("invalid_argument",),
    "proposal_arrival.provider_binding_mismatch": ("invalid_argument",),
    "proposal_arrival.raw_counts_required": ("invalid_argument",),
    "proposal_arrival.replay_mismatch": ("invalid_argument",),
    "proposal_arrival.sample_size_mismatch": ("invalid_argument",),
    "proposal_arrival.structural_arrival_refused": ("invalid_argument",),
    "proposal_receipt.arrival_cancelled": ("transport_budget_cancel",),
    "proposal_receipt.arrival_empty": ("invalid_argument",),
    "proposal_receipt.arrival_family_mismatch": ("design_signal_invalid",),
    "proposal_receipt.arrival_invalid": ("invalid_argument",),
    "proposal_receipt.arrival_sample_mismatch": ("design_signal_invalid",),
    "proposal_receipt.base_failure_mismatch": ("design_signal_invalid",),
    "proposal_receipt.bundle_identity_mismatch": ("design_signal_invalid",),
    "proposal_receipt.bundle_inconsistent": ("design_signal_invalid",),
    "proposal_receipt.candidate_mismatch": ("design_signal_invalid",),
    "proposal_receipt.candidate_not_in_bundle": ("design_signal_invalid",),
    "proposal_receipt.candidate_not_in_ranking": ("design_signal_invalid",),
    "proposal_receipt.candidate_not_in_repair": ("design_signal_invalid",),
    "proposal_receipt.candidate_not_proposed": ("design_signal_invalid",),
    "proposal_receipt.candidate_set_mismatch": ("design_signal_invalid",),
    "proposal_receipt.contract_mismatch": ("design_signal_invalid",),
    "proposal_receipt.contract_required": ("invalid_argument",),
    "proposal_receipt.cost_mismatch": ("design_signal_invalid",),
    "proposal_receipt.delta_mismatch": ("design_signal_invalid",),
    "proposal_receipt.derivation_mismatch": ("design_signal_invalid",),
    "proposal_receipt.duplicate_candidate": ("invalid_argument",),
    "proposal_receipt.empty_bundle": ("invalid_argument",),
    "proposal_receipt.hypothetical_not_evidence": ("transport_missing_evidence",),
    "proposal_receipt.identity_mismatch": ("design_signal_invalid",),
    "proposal_receipt.invalid_delta": ("invalid_argument",),
    "proposal_receipt.lineage_mismatch": ("design_signal_invalid",),
    "proposal_receipt.non_canonical_order": ("design_signal_invalid",),
    "proposal_receipt.obligation_mismatch": ("design_signal_invalid",),
    "proposal_receipt.ranking_digest_mismatch": ("design_signal_invalid",),
    "proposal_receipt.repair_digest_mismatch": ("design_signal_invalid",),
    "proposal_receipt.signal_mismatch": ("design_signal_invalid",),
    "proposal_receipt.source_digest_mismatch": ("design_signal_invalid",),
    "proposal_receipt.value_mismatch": ("design_signal_invalid",),
    "provider_capability_negotiation.capability_missing": ("external_capability_missing",),
    "provider_capability_negotiation.request_mismatch": ("external_binding_mismatch",),
    "provider_transport.artifact_invalid": ("invalid_argument",),
    "provider_transport.assumptions_missing": ("interface_contract_invalid",),
    "provider_transport.assumptions_not_carried": ("interface_contract_assumption_mismatch",),
    "provider_transport.blank_field": ("interface_contract_invalid",),
    "provider_transport.claim_not_in_target": ("interface_contract_quantity_mismatch",),
    "provider_transport.evidence_missing": ("interface_contract_invalid",),
    "provider_transport.identity_mismatch": ("route_not_supported",),
    "provider_transport.input_not_external": ("interface_contract_assumption_mismatch",),
    "provider_transport.native_target_required": ("interface_contract_assumption_mismatch",),
    "provider_transport.native_trust_claimed": ("interface_contract_invalid",),
    "provider_transport.object_invalid": ("interface_contract_invalid",),
    "provider_transport.object_unsupported": ("interface_contract_invalid",),
    "provider_transport.population_missing": ("interface_contract_invalid",),
    "provider_transport.population_not_a_source": ("interface_contract_quantity_mismatch",),
    "provider_transport.quantity_invalid": ("interface_contract_invalid",),
    "provider_transport.quantity_not_declared": ("external_binding_mismatch",),
    "provider_transport.receipt_does_not_cover": ("external_verification_failed",),
    "provider_transport.regime_missing": ("interface_contract_invalid",),
    "provider_transport.transport_identity_incomplete": ("interface_contract_invalid",),
    "provider_transport.trust_incomplete": ("interface_contract_invalid",),
    "provider_transport.verification_required": ("external_verification_failed",),
    "quadratic_posterior.band_not_constructed": ("route_not_supported",),
    "quadratic_posterior.external_power_prior": ("route_not_supported",),
    "quadratic_posterior.insufficient_weight": ("positivity_violated",),
    "quadratic_posterior.mapped_prior": ("route_not_supported",),
    "quadratic_posterior.non_positive_scale": ("invalid_argument",),
    "rank_drop_estimate.estimator_not_supported": ("route_not_supported",),
    "rank_drop_estimate.propensity_separates": ("rank_drop_not_supported",),
    "rank_drop_estimate.shape_not_supported": ("route_not_supported",),
    "rank_drop_estimate.span_not_preserved": ("rank_drop_not_supported",),
    "recalc.adjusted_arm_not_populated": ("arm_not_populated",),
    "recalc.adjusted_design_rank_deficient": ("design_rank_deficient",),
    "recalc.adjusted_fit_refused": ("route_not_supported",),
    "recalc.adjusted_glm_boundary_saturated": ("route_not_supported",),
    "recalc.adjusted_glm_not_converged": ("route_not_supported",),
    "recalc.adjusted_glm_separated": ("route_not_supported",),
    "recalc.adjusted_invalid_model": ("route_not_supported",),
    "recalc.adjusted_model_data_invalid": ("invalid_argument",),
    "recalc.adjusted_prediction_out_of_support": ("route_not_supported",),
    "recalc.adjusted_prediction_schema_mismatch": ("invalid_argument",),
    "recalc.adjusted_receipt_invalid": ("invalid_argument",),
    "recalc.adjustment_not_identified": ("effect_not_identified",),
    "recalc.aipw_requires_overlap_diagnostics": ("invalid_argument",),
    "recalc.bayesian_basis_prior_unsupported": ("route_not_supported",),
    "recalc.bayesian_effect_unavailable": ("invalid_argument",),
    "recalc.bayesian_external_compose_unsupported": ("route_not_supported",),
    "recalc.bayesian_fit_refused": ("route_not_supported",),
    "recalc.bayesian_inference_unsupported": ("route_not_supported",),
    "recalc.bayesian_invalid_data": ("invalid_argument",),
    "recalc.bayesian_invalid_graph": ("invalid_argument",),
    "recalc.bayesian_invalid_inference": ("invalid_argument",),
    "recalc.bayesian_invalid_query": ("invalid_argument",),
    "recalc.bayesian_invalid_summary": ("invalid_argument",),
    "recalc.bayesian_prior_invalid": ("invalid_argument",),
    "recalc.bayesian_prior_likelihood_double_use": ("invalid_argument",),
    "recalc.bayesian_prior_source_conflict": ("invalid_argument",),
    "recalc.bayesian_prior_source_unsupported": ("route_not_supported",),
    "recalc.bayesian_prior_source_unverified": ("route_not_supported",),
    "recalc.bayesian_workspace_limit": ("invalid_argument",),
    "recalc.blocked_by_refused_dependency": ("not_executed",),
    "recalc.bootstrap_replicates_unsupported": ("invalid_argument",),
    "recalc.cancelled": ("cancelled_no_claim",),
    "recalc.capability_adapter_unavailable": ("route_not_supported",),
    "recalc.categorical_role_mismatch": ("invalid_argument",),
    "recalc.contrast_out_of_support": ("route_not_supported",),
    "recalc.contrast_unavailable": ("invalid_argument",),
    "recalc.declared_size_mismatch": ("invalid_argument",),
    "recalc.design_artifact_mismatch": ("invalid_argument",),
    "recalc.design_artifact_unverified": ("route_not_supported",),
    "recalc.design_data_unavailable": ("score_table_unavailable",),
    "recalc.design_effect_unavailable": ("invalid_argument",),
    "recalc.design_fit_refused": ("route_not_supported",),
    "recalc.design_model_data_invalid": ("invalid_argument",),
    "recalc.design_not_checked": ("effect_not_identified",),
    "recalc.design_not_identified": ("effect_not_identified",),
    "recalc.design_point_unavailable": ("invalid_argument",),
    "recalc.design_rank_deficient": ("design_rank_deficient",),
    "recalc.design_receipt_invalid": ("invalid_argument",),
    "recalc.dr_arm_not_populated": ("arm_not_populated",),
    "recalc.dr_design_rank_deficient": ("design_rank_deficient",),
    "recalc.dr_effect_unavailable": ("invalid_argument",),
    "recalc.dr_fit_counter_mismatch": ("invalid_argument",),
    "recalc.dr_fit_refused": ("route_not_supported",),
    "recalc.dr_glm_boundary_saturated": ("route_not_supported",),
    "recalc.dr_glm_not_converged": ("route_not_supported",),
    "recalc.dr_glm_separated": ("route_not_supported",),
    "recalc.dr_invalid_folds": ("invalid_argument",),
    "recalc.dr_invalid_learner": ("invalid_argument",),
    "recalc.dr_invalid_model": ("invalid_argument",),
    "recalc.dr_invalid_overlap": ("invalid_argument",),
    "recalc.dr_invalid_score": ("invalid_argument",),
    "recalc.dr_model_data_invalid": ("invalid_argument",),
    "recalc.dr_not_identified": ("effect_not_identified",),
    "recalc.dr_prediction_out_of_support": ("route_not_supported",),
    "recalc.dr_prediction_schema_mismatch": ("invalid_argument",),
    "recalc.dr_predictor_unavailable": ("route_not_supported",),
    "recalc.dr_receipt_invalid": ("invalid_argument",),
    "recalc.dr_scores_unavailable": ("route_not_supported",),
    "recalc.dr_source_mismatch": ("invalid_argument",),
    "recalc.dr_target_incompatible": ("route_not_supported",),
    "recalc.family_model_data_invalid": ("invalid_argument",),
    "recalc.family_not_identified": ("effect_not_identified",),
    "recalc.family_receipt_invalid": ("invalid_argument",),
    "recalc.frozen_scores_rejected": ("invalid_argument",),
    "recalc.incompatible_contrast": ("route_not_supported",),
    "recalc.invalid_adjustment_set": ("invalid_argument",),
    "recalc.invalid_categorical_contrast": ("invalid_argument",),
    "recalc.invalid_changed_input": ("invalid_argument",),
    "recalc.invalid_contrast": ("invalid_argument",),
    "recalc.invalid_data": ("invalid_argument",),
    "recalc.invalid_declaration": ("invalid_argument",),
    "recalc.invalid_design_artifact": ("invalid_argument",),
    "recalc.invalid_design_roles": ("invalid_argument",),
    "recalc.invalid_evidence": ("invalid_argument",),
    "recalc.invalid_graph": ("invalid_argument",),
    "recalc.invalid_initial_state_law": ("invalid_argument",),
    "recalc.invalid_query": ("invalid_argument",),
    "recalc.invalid_rd_window": ("invalid_argument",),
    "recalc.invalid_request": ("invalid_argument",),
    "recalc.invalid_target_weights": ("invalid_argument",),
    "recalc.invalid_temporal_provider": ("invalid_argument",),
    "recalc.invalid_temporal_window": ("invalid_argument",),
    "recalc.iv_decision_unavailable": ("route_not_supported",),
    "recalc.iv_diagnostics_unavailable": ("invalid_argument",),
    "recalc.joint_bayesian_graph_unsupported": ("invalid_argument",),
    "recalc.joint_bayesian_invalid_data": ("invalid_argument",),
    "recalc.joint_bayesian_invalid_query": ("invalid_argument",),
    "recalc.joint_bayesian_refused": ("route_not_supported",),
    "recalc.limits_exceeded": ("invalid_argument",),
    "recalc.memory_budget_exceeded": ("invalid_argument",),
    "recalc.mz_action_out_of_support": ("route_not_supported",),
    "recalc.mz_invalid_request": ("invalid_argument",),
    "recalc.no_live_state": ("score_table_unavailable",),
    "recalc.off_grid_request": ("route_not_supported",),
    "recalc.primary_result_is_not_an_effect": ("invalid_argument",),
    "recalc.refused": ("not_executed",),
    "recalc.retarget_incompatible": ("construction_not_supported",),
    "recalc.retarget_not_supported": ("invalid_argument",),
    "recalc.row_count_mismatch": ("invalid_argument",),
    "recalc.row_ids_mismatch": ("invalid_argument",),
    "recalc.score_stage_undeclared": ("invalid_argument",),
    "recalc.score_table_unavailable": ("score_table_unavailable",),
    "recalc.static_action_out_of_support": ("route_not_supported",),
    "recalc.static_engine_refused": ("route_not_supported",),
    "recalc.static_graph_unsupported": ("route_not_supported",),
    "recalc.static_invalid_contrast": ("invalid_argument",),
    "recalc.static_invalid_data": ("invalid_argument",),
    "recalc.static_invalid_factor": ("invalid_argument",),
    "recalc.static_invalid_query": ("invalid_argument",),
    "recalc.static_not_identified": ("effect_not_identified",),
    "recalc.static_receipt_invalid": ("invalid_argument",),
    "recalc.static_required_factor_missing": ("route_not_supported",),
    "recalc.static_transport_budget_cancel": ("transport_budget_cancel",),
    "recalc.static_transport_missing_evidence": ("transport_missing_evidence",),
    "recalc.static_transport_missing_provider": ("transport_missing_provider",),
    "recalc.static_transport_non_transportable": ("transport_proven_non_transportable",),
    "recalc.static_transport_not_certified": ("transport_not_certified",),
    "recalc.static_transport_numerical_failure": ("transport_numerical_failure",),
    "recalc.static_transport_support_failure": ("transport_support_failure",),
    "recalc.static_transport_unsupported_evaluator": ("transport_unsupported_evaluator",),
    "recalc.temporal_artifact_invalid": ("invalid_argument",),
    "recalc.temporal_artifact_mismatch": ("invalid_argument",),
    "recalc.temporal_artifact_seed_mismatch": ("invalid_argument",),
    "recalc.temporal_budget_cancel": ("transport_budget_cancel",),
    "recalc.temporal_data_invalid": ("invalid_argument",),
    "recalc.temporal_evaluator_unsupported": ("transport_unsupported_evaluator",),
    "recalc.temporal_fit_refused": ("route_not_supported",),
    "recalc.temporal_history_unsupported": ("route_not_supported",),
    "recalc.temporal_missing_evidence": ("transport_missing_evidence",),
    "recalc.temporal_missing_provider": ("transport_missing_provider",),
    "recalc.temporal_numerical_failure": ("transport_numerical_failure",),
    "recalc.temporal_scope_unsupported": ("route_not_supported",),
    "recalc.temporal_support_failure": ("transport_support_failure",),
    "recalc.unavailable_data": ("score_table_unavailable",),
    "recalc.unavailable_fit": ("score_table_unavailable",),
    "recalc.unavailable_provider": ("external_capability_missing",),
    "recalc.unknown_variable": ("invalid_argument",),
    "recalc.unsupported_aipw_score_configuration": ("invalid_argument",),
    "recalc.unsupported_request": ("route_not_supported",),
    "recalc.weighted_overlap_failed": ("invalid_argument",),
    "recalc_receipt.counts_inconsistent": ("route_not_supported",),
    "recalc_receipt.fresh_process_reuse": ("score_table_unavailable",),
    "recalc_receipt.identity_mismatch": ("route_not_supported",),
    "recalc_receipt.plan_mismatch": ("route_not_supported",),
    "recalc_receipt.totals_mismatch": ("route_not_supported",),
    "recovery.bounds_exceeded": ("route_not_supported",),
    "recovery.budget": ("transport_budget_cancel",),
    "recovery.empirical_not_supported": ("route_not_supported",),
    "recovery.handoff_mismatch": ("invalid_argument",),
    "recovery.invalid_derivation": ("transport_not_certified",),
    "recovery.invalid_observed_law": ("invalid_argument",),
    "recovery.invalid_query": ("invalid_argument",),
    "recovery.missing_margin": ("transport_missing_evidence",),
    "recovery.nonrecoverable_witness": ("transport_proven_non_transportable",),
    "recovery.positivity": ("transport_support_failure",),
    "recovery.unsupported_mechanism": ("route_not_supported",),
    "recovery.witness_unavailable": ("transport_not_certified",),
    "recovery_chain.budget": ("transport_budget_cancel",),
    "recovery_chain.data_identity_mismatch": ("invalid_argument",),
    "recovery_chain.decision_replay_mismatch": ("invalid_argument",),
    "recovery_chain.invalid_derivation": ("transport_not_certified",),
    "recovery_chain.invalid_observed_law": ("invalid_argument",),
    "recovery_chain.invalid_query": ("invalid_argument",),
    "recovery_chain.names_mismatch": ("invalid_argument",),
    "recovery_chain.nonrecoverable_witness": ("transport_proven_non_transportable",),
    "recovery_chain.not_evaluated": ("invalid_argument",),
    "recovery_chain.positivity": ("transport_support_failure",),
    "recovery_chain.premises_mismatch": ("invalid_argument",),
    "recovery_chain.recovered_replay_mismatch": ("invalid_argument",),
    "recovery_chain.undecodable": ("invalid_argument",),
    "recovery_chain.unsupported_mechanism": ("route_not_supported",),
    "recovery_chain.unsupported_semantics": ("route_not_supported",),
    "refresh_posterior.band_not_constructed": ("route_not_supported",),
    "refresh_posterior.empty_values": ("invalid_argument",),
    "refresh_posterior.external_power_prior": ("route_not_supported",),
    "refresh_posterior.mapped_prior": ("route_not_supported",),
    "refresh_posterior.non_positive_scale": ("invalid_argument",),
    "repair_artifact.bounds_exceeded": ("route_not_supported",),
    "repair_artifact.budget": ("transport_budget_cancel",),
    "repair_artifact.invalid_artifact": ("transport_not_certified",),
    "repair_artifact.replay_mismatch": ("transport_not_certified",),
    "repair_artifact.unsupported_version": ("transport_not_certified",),
    "rollout.artifact_id_missing": ("invalid_argument",),
    "rollout.artifact_kind_mismatch": ("invalid_argument",),
    "rollout.expected_binding_mismatch": ("invalid_argument",),
    "rollout.finite_draw_state_required": ("route_not_supported",),
    "rollout.identity_mismatch": ("invalid_argument",),
    "rollout.invalid_request": ("invalid_argument",),
    "rollout.ordered_prior_mismatch": ("invalid_argument",),
    "rollout.ranking_invalid": ("invalid_argument",),
    "rollout.ranking_source_mismatch": ("invalid_argument",),
    "rollout.source_digest_mismatch": ("invalid_argument",),
    "rollout.source_evidence_mismatch": ("invalid_argument",),
    "rollout.source_standing_mismatch": ("invalid_argument",),
    "rollout.state_coordinate_mismatch": ("invalid_argument",),
    "rollout.state_interpretation_mismatch": ("route_not_supported",),
    "rollout.terminal_decision_mismatch": ("invalid_argument",),
    "rollout.unweighted_joint_state_required": ("route_not_supported",),
    "sampled_recovery.bounds_exceeded": ("route_not_supported",),
    "sampled_recovery.cancelled": ("transport_budget_cancel",),
    "sampled_recovery.component_variance_only": ("estimator_inference_mismatch",),
    "sampled_recovery.invalid_input": ("invalid_argument",),
    "sampled_recovery.native_hook_unavailable": ("not_supported",),
    "sampled_recovery.producer_missing": ("not_supported",),
    "sampled_recovery.receipt_mismatch": ("invalid_argument",),
    "sampled_recovery.too_many_failed_replicates": ("route_not_supported",),
    "sampled_recovery.unrecoverable_pattern": ("route_not_supported",),
    "sampling_weights.invalid": ("sampling_weights_invalid",),
    "sampling_weights.route_unsupported": ("option_not_applicable",),
    "scenario_covariance.cancelled": ("cancelled_no_claim",),
    "scenario_covariance.consumer_limit_exceeded": ("execution_resource_limit",),
    "scenario_covariance.data_identity_mismatch": ("invalid_argument",),
    "scenario_covariance.duplicate_scenario_id": ("invalid_argument",),
    "scenario_covariance.exact_enumeration_cap": ("execution_resource_limit",),
    "scenario_covariance.invalid_declaration": ("invalid_argument",),
    "scenario_covariance.invalid_dependence": ("invalid_argument",),
    "scenario_covariance.invalid_failure_bound": ("invalid_argument",),
    "scenario_covariance.invalid_functional": ("invalid_argument",),
    "scenario_covariance.invalid_row_table": ("invalid_argument",),
    "scenario_covariance.no_scenarios": ("invalid_argument",),
    "scenario_covariance.not_positive_semidefinite": ("transport_numerical_failure",),
    "scenario_covariance.premises_mismatch": ("invalid_argument",),
    "scenario_covariance.replay_mismatch": ("invalid_argument",),
    "scenario_covariance.table_too_large": ("execution_resource_limit",),
    "scenario_covariance.too_few_replicates": ("invalid_argument",),
    "scenario_covariance.too_few_rows": ("invalid_argument",),
    "scenario_covariance.too_many_failed_replicates": ("transport_numerical_failure",),
    "scenario_covariance.too_many_terms": ("execution_resource_limit",),
    "scenario_covariance.unknown_dependence": ("route_not_supported",),
    "scenario_covariance.unsupported_semantics": ("route_not_supported",),
    "scenario_covariance_posterior.band_not_constructed": ("route_not_supported",),
    "scenario_covariance_posterior.empty_matrix": ("invalid_argument",),
    "scenario_covariance_posterior.external_power_prior": ("route_not_supported",),
    "scenario_covariance_posterior.mapped_prior": ("route_not_supported",),
    "scenario_covariance_posterior.non_positive_scale": ("invalid_argument",),
    "scenario_invariance.derivation_inconsistent": ("invalid_argument",),
    "scenario_invariance.non_static_node": ("invalid_argument",),
    "scenario_invariance.not_estimated": ("not_executed",),
    "scenario_invariance.refused": ("invalid_argument",),
    "scenario_invariance.wrong_result_type": ("invalid_argument",),
    "scenarios.catalog_selections": ("invalid_argument",),
    "scenarios.coordinate_mismatch": ("schema_mismatch",),
    "scenarios.count": ("route_not_supported",),
    "scenarios.duplicate_or_empty_name": ("invalid_argument",),
    "scenarios.duplicate_scenario": ("invalid_argument",),
    "scenarios.empty": ("invalid_argument",),
    "scenarios.equivalence_class_input": ("route_not_supported",),
    "scenarios.invalid_weights": ("invalid_argument",),
    "scenarios.shared_data_aggregate": ("scenario_aggregate_not_supported",),
    "scenarios.unevaluated_budget": ("transport_budget_cancel",),
    "scientific_bridge.artifact_invalid": ("invalid_argument",),
    "scientific_bridge.association_as_intervention": ("bridge_substitution_refused",),
    "scientific_bridge.assumption_evidence_missing": ("bridge_assumption_missing",),
    "scientific_bridge.assumption_missing": ("bridge_assumption_missing",),
    "scientific_bridge.blank_field": ("scientific_bridge_invalid",),
    "scientific_bridge.chain_stage_not_affine": ("route_not_supported",),
    "scientific_bridge.conditioning_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.degenerate_validation": ("bridge_mapping_not_supported",),
    "scientific_bridge.dependence_not_a_covariance": ("bridge_mapping_not_supported",),
    "scientific_bridge.differential_error": ("differential_error_not_pooled",),
    "scientific_bridge.draws_meaning_mismatch": ("distribution_meaning_mismatch",),
    "scientific_bridge.draws_not_aligned": ("joint_law_required",),
    "scientific_bridge.draws_outside_domain": ("bridge_mapping_not_supported",),
    "scientific_bridge.draws_unusable": ("scientific_bridge_invalid",),
    "scientific_bridge.duplicate_assumption": ("scientific_bridge_invalid",),
    "scientific_bridge.endpoint_classes": ("scientific_bridge_invalid",),
    "scientific_bridge.estimand_bridge_incomplete": ("scientific_bridge_invalid",),
    "scientific_bridge.estimand_bridge_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.estimand_spend_inconsistent": ("bridge_binding_mismatch",),
    "scientific_bridge.estimand_units_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.experimental_prior_without_estimand_bridge": ("bridge_assumption_missing",),
    "scientific_bridge.functional_not_supported": ("bridge_mapping_not_supported",),
    "scientific_bridge.horizon_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.identity_mismatch": ("route_not_supported",),
    "scientific_bridge.interval_not_supported": ("route_not_supported",),
    "scientific_bridge.interval_scope_outside_measured": ("route_not_supported",),
    "scientific_bridge.interval_source_uncertainty_required": ("required_option_missing",),
    "scientific_bridge.invalid_quantity": ("scientific_bridge_invalid",),
    "scientific_bridge.malformed_mapping": ("scientific_bridge_invalid",),
    "scientific_bridge.mapping_not_allowed": ("scientific_bridge_invalid",),
    "scientific_bridge.mapping_scope_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.mapping_uncertainty_invalid": ("scientific_bridge_invalid",),
    "scientific_bridge.mapping_uncertainty_missing": ("bridge_assumption_missing",),
    "scientific_bridge.measurement_model_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.measurement_route_not_supported": ("bridge_mapping_not_supported",),
    "scientific_bridge.model_identity_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.model_score_as_probability": ("bridge_substitution_refused",),
    "scientific_bridge.non_finite_value": ("scientific_bridge_invalid",),
    "scientific_bridge.nonlinear_law_insufficient": ("bridge_mapping_not_supported",),
    "scientific_bridge.nonlinear_mean_not_expectation": ("bridge_mapping_not_supported",),
    "scientific_bridge.outside_attainable_range": ("bridge_mapping_not_supported",),
    "scientific_bridge.outside_validated_range": ("bridge_mapping_not_supported",),
    "scientific_bridge.population_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.predictive_importance_as_causal_effect": ("bridge_substitution_refused",),
    "scientific_bridge.propagation_overflow": ("scientific_bridge_invalid",),
    "scientific_bridge.provenance": ("scientific_bridge_invalid",),
    "scientific_bridge.proxy_as_construct": ("proxy_offered_as_construct",),
    "scientific_bridge.quantity_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.regime_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.replay_mismatch": ("invalid_argument",),
    "scientific_bridge.scientific_endpoint_as_utility": ("bridge_substitution_refused",),
    "scientific_bridge.source_dependence_invalid": ("scientific_bridge_invalid",),
    "scientific_bridge.source_dependence_missing": ("sampling_dependence_unknown",),
    "scientific_bridge.source_moment_not_finite": ("scientific_bridge_invalid",),
    "scientific_bridge.source_not_draws": ("bridge_mapping_not_supported",),
    "scientific_bridge.source_not_point": ("bridge_mapping_not_supported",),
    "scientific_bridge.source_variance_invalid": ("scientific_bridge_invalid",),
    "scientific_bridge.stated_preference_as_realized_behavior": ("bridge_substitution_refused",),
    "scientific_bridge.transport_diagram_scope_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.transport_premise_contradicted_by_selection_diagram": (
        "bridge_assumption_missing",
    ),
    "scientific_bridge.transport_premise_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.transport_premise_missing": ("bridge_assumption_missing",),
    "scientific_bridge.transport_premise_unused": ("bridge_binding_mismatch",),
    "scientific_bridge.uncertain_mapping_not_affine": ("bridge_mapping_not_supported",),
    "scientific_bridge.units_mismatch": ("bridge_binding_mismatch",),
    "scientific_bridge.validation_r_squared_too_low": ("bridge_mapping_not_supported",),
    "scientific_bridge.validation_trials_too_few": ("bridge_mapping_not_supported",),
    "scientific_bridge.value_kind_not_carried": ("bridge_mapping_not_supported",),
    "scientific_bridge.value_kind_substitution": ("bridge_substitution_refused",),
    "scientific_bridge.youden_too_small": ("measurement_not_identified",),
    "sde.action_time_off_grid": ("route_not_supported",),
    "sde.input_outside_certified_envelope": ("route_not_supported",),
    "sde.milstein_noncommutative": ("route_not_supported",),
    "sde.unbounded_growth": ("route_not_supported",),
    "selective_discovery.analysis_rows_not_the_split_part": ("invalid_argument",),
    "selective_discovery.artifact_invalid": ("invalid_argument",),
    "selective_discovery.background_knowledge_contradictory": ("invalid_argument",),
    "selective_discovery.background_knowledge_not_applicable": ("option_not_applicable",),
    "selective_discovery.background_knowledge_unknown_variable": ("unknown_variable",),
    "selective_discovery.components_incomplete": ("discovery_statistics_invalid",),
    "selective_discovery.contemporaneous_links": ("route_not_supported",),
    "selective_discovery.contract_identity": ("discovery_statistics_invalid",),
    "selective_discovery.contract_mismatch": ("discovery_statistics_invalid",),
    "selective_discovery.data_mismatch": ("discovery_statistics_invalid",),
    "selective_discovery.differences_are_not_selection_targets": (
        "discovered_structure_unresolved",
    ),
    "selective_discovery.downstream_interval_not_supported": ("selection_inference_not_supported",),
    "selective_discovery.identity_mismatch": ("discovery_statistics_invalid",),
    "selective_discovery.invalid_contract": ("invalid_argument",),
    "selective_discovery.multi_population": ("route_not_supported",),
    "selective_discovery.nothing_retained": ("invalid_argument",),
    "selective_discovery.pooled_population_data": ("invalid_argument",),
    "selective_discovery.pooled_population_search": ("route_not_supported",),
    "selective_discovery.population_missing": ("invalid_argument",),
    "selective_discovery.population_repeated": ("invalid_argument",),
    "selective_discovery.populations_missing": ("invalid_argument",),
    "selective_discovery.raw_data_missing": ("discovery_statistics_invalid",),
    "selective_discovery.receipt_invalid": ("discovery_statistics_invalid",),
    "selective_discovery.receipt_limit": ("discovery_statistics_invalid",),
    "selective_discovery.receipt_repeated": ("discovery_statistics_invalid",),
    "selective_discovery.receipts_identity": ("discovery_statistics_invalid",),
    "selective_discovery.recovery_claim_not_supported": ("discovered_structure_unresolved",),
    "selective_discovery.replay_diverged": ("discovery_statistics_invalid",),
    "selective_discovery.replay_miss": ("discovery_statistics_invalid",),
    "selective_discovery.rng_mismatch": ("invalid_argument",),
    "selective_discovery.route_unknown": ("discovery_statistics_invalid",),
    "selective_discovery.rows": ("discovery_statistics_invalid",),
    "selective_discovery.score_search_not_supported": ("route_not_supported",),
    "selective_discovery.search_incomplete": ("discovered_structure_unresolved",),
    "selective_discovery.split_fraction_invalid": ("invalid_argument",),
    "selective_discovery.split_mask_mismatch": ("invalid_argument",),
    "selective_discovery.split_not_disjoint": ("invalid_argument",),
    "selective_discovery.split_part_empty": ("invalid_argument",),
    "selective_discovery.split_part_weight_invalid": ("invalid_argument",),
    "selective_discovery.split_receipt_mismatch": ("invalid_argument",),
    "selective_discovery.split_row_excluded": ("invalid_argument",),
    "selective_discovery.split_row_out_of_range": ("invalid_argument",),
    "selective_discovery.split_rows_mismatch": ("invalid_argument",),
    "selective_discovery.split_rows_repeated": ("invalid_argument",),
    "selective_discovery.split_source_mismatch": ("invalid_argument",),
    "selective_discovery.split_unsupported_rows": ("route_not_supported",),
    "selective_discovery.split_weights_mismatch": ("invalid_argument",),
    "selective_discovery.structure_class": ("discovery_statistics_invalid",),
    "selective_discovery.structure_marks": ("discovery_statistics_invalid",),
    "selective_discovery.structure_nodes": ("discovery_statistics_invalid",),
    "selective_discovery.test_identity_missing": ("invalid_argument",),
    "selective_discovery.unaccepted_discovery": ("discovered_structure_unresolved",),
    "selective_discovery.unresolved_structure": ("discovered_structure_unresolved",),
    "selective_discovery.variables_repeated": ("discovery_statistics_invalid",),
    "selective_discovery.weighted_test_without_weights": ("option_not_applicable",),
    "selective_discovery.weights_not_read_by_test": ("option_not_applicable",),
    "selective_discovery.work_inconsistent": ("discovery_statistics_invalid",),
    "sensitivity_decision_composition.bounds_exceeded": ("route_not_supported",),
    "sensitivity_decision_composition.composition_not_supported": ("route_not_supported",),
    "sensitivity_decision_composition.invalid_artifact": ("invalid_argument",),
    "sensitivity_decision_composition.invalid_surface": ("invalid_argument",),
    "sensitivity_decision_composition.sampling_not_reported": ("not_supported",),
    "sensitivity_decision_composition.unsupported_adaptation": ("route_not_supported",),
    "sensitivity_decision_composition.unsupported_coordinate": ("quantity_semantics_mismatch",),
    "sensitivity_decision_composition.wrong_contract": ("decision_contract_unsatisfied",),
    "sensitivity_source.invalid_artifact": ("invalid_argument",),
    "sensitivity_source.limits_exceeded": ("invalid_argument",),
    "sensitivity_source.lineage_digest": ("invalid_argument",),
    "sensitivity_source.source_unavailable": ("invalid_argument",),
    "sensitivity_source.surface_mismatch": ("invalid_argument",),
    "sequential_evaluation.accounting_residual": ("invalid_argument",),
    "sequential_evaluation.action_channel": ("invalid_argument",),
    "sequential_evaluation.action_cost": ("invalid_argument",),
    "sequential_evaluation.action_delayed_outcome": ("invalid_argument",),
    "sequential_evaluation.action_index": ("invalid_argument",),
    "sequential_evaluation.action_label": ("invalid_argument",),
    "sequential_evaluation.action_not_supported": ("history_support_violated",),
    "sequential_evaluation.action_support": ("invalid_argument",),
    "sequential_evaluation.action_transition": ("invalid_argument",),
    "sequential_evaluation.action_utility": ("invalid_argument",),
    "sequential_evaluation.artifact_id_missing": ("invalid_argument",),
    "sequential_evaluation.artifact_kind_mismatch": ("invalid_argument",),
    "sequential_evaluation.behaviour_support_missing": ("history_support_violated",),
    "sequential_evaluation.cancelled": ("cancelled_no_claim",),
    "sequential_evaluation.channel_binding_mismatch": ("measurement_binding_mismatch",),
    "sequential_evaluation.channel_construct_mismatch": ("measurement_binding_mismatch",),
    "sequential_evaluation.channel_levels_mismatch": ("measurement_binding_mismatch",),
    "sequential_evaluation.channel_rates": ("route_not_supported",),
    "sequential_evaluation.channel_route": ("route_not_supported",),
    "sequential_evaluation.commit_has_no_observation": ("invalid_argument",),
    "sequential_evaluation.comparison_kind_mismatch": ("invalid_argument",),
    "sequential_evaluation.comparison_label": ("invalid_argument",),
    "sequential_evaluation.comparisons": ("invalid_argument",),
    "sequential_evaluation.decision_receipt_mismatch": ("invalid_argument",),
    "sequential_evaluation.delayed_outcome_beyond_horizon": ("invalid_argument",),
    "sequential_evaluation.differential_channel": ("differential_error_not_pooled",),
    "sequential_evaluation.empty_log": ("invalid_argument",),
    "sequential_evaluation.exact_mismatch": ("invalid_argument",),
    "sequential_evaluation.history_receipt_mismatch": ("invalid_argument",),
    "sequential_evaluation.identity_mismatch": ("route_not_supported",),
    "sequential_evaluation.importance_weight_overflow": ("invalid_argument",),
    "sequential_evaluation.indicator_as_state": ("proxy_offered_as_construct",),
    "sequential_evaluation.interval_not_supported": ("uncertainty_representation_insufficient",),
    "sequential_evaluation.limit_exceeded": ("invalid_argument",),
    "sequential_evaluation.log_partially_counted": ("evidence_double_counted",),
    "sequential_evaluation.logged_trajectory_inconsistent": ("invalid_argument",),
    "sequential_evaluation.malformed": ("invalid_argument",),
    "sequential_evaluation.model_actions": ("invalid_argument",),
    "sequential_evaluation.model_field": ("invalid_argument",),
    "sequential_evaluation.model_horizon": ("invalid_argument",),
    "sequential_evaluation.model_initial": ("invalid_argument",),
    "sequential_evaluation.model_mismatch": ("invalid_argument",),
    "sequential_evaluation.model_resource_budget": ("invalid_argument",),
    "sequential_evaluation.model_states": ("invalid_argument",),
    "sequential_evaluation.model_terminal_utility": ("invalid_argument",),
    "sequential_evaluation.model_work": ("invalid_argument",),
    "sequential_evaluation.no_overlap": ("history_support_violated",),
    "sequential_evaluation.observation_impossible": ("invalid_argument",),
    "sequential_evaluation.observation_recorded_twice": ("evidence_double_counted",),
    "sequential_evaluation.off_policy_mismatch": ("invalid_argument",),
    "sequential_evaluation.past_horizon": ("invalid_argument",),
    "sequential_evaluation.policy_change_refused": ("route_not_supported",),
    "sequential_evaluation.policy_distribution": ("invalid_argument",),
    "sequential_evaluation.policy_incomplete": ("invalid_argument",),
    "sequential_evaluation.policy_mismatch": ("invalid_argument",),
    "sequential_evaluation.policy_version": ("invalid_argument",),
    "sequential_evaluation.record_already_counted": ("evidence_double_counted",),
    "sequential_evaluation.record_identity": ("invalid_argument",),
    "sequential_evaluation.resource_budget_exceeded": ("invalid_argument",),
    "sequential_evaluation.return_overflow": ("invalid_argument",),
    "sequential_evaluation.rollout_belief": ("invalid_argument",),
    "sequential_evaluation.rollout_mismatch": ("invalid_argument",),
    "sequential_evaluation.rollouts": ("invalid_argument",),
    "sequential_evaluation.session_label": ("invalid_argument",),
    "sequential_evaluation.static_score_table": ("route_not_supported",),
    "sequential_evaluation.stopping_boundary": ("invalid_argument",),
    "sequential_evaluation.temporal_transport_contract_missing": ("transport_missing_evidence",),
    "sequential_evaluation.trajectory_identity": ("invalid_argument",),
    "sequential_evaluation.trajectory_recorded_twice": ("evidence_double_counted",),
    "sequential_evaluation.transported_policy_contract_missing": ("transport_missing_evidence",),
    "sequential_evaluation.value_mismatch": ("invalid_argument",),
    "sequential_evaluation.version": ("invalid_argument",),
    "sequential_evaluation.voi_collect_action": ("invalid_argument",),
    "sequential_evaluation.voi_commit_actions": ("invalid_argument",),
    "sequential_evaluation.voi_enumeration_mismatch": ("invalid_argument",),
    "sequential_evaluation.voi_ranking": ("invalid_argument",),
    "sequential_evaluation.voi_scope": ("route_not_supported",),
    "sequential_evaluation.weighted_estimate_overflow": ("invalid_argument",),
    "sequential_evaluation.weighted_sum_overflow": ("invalid_argument",),
    "signal_provider.candidate_mismatch": ("design_signal_invalid",),
    "signal_provider.capability_missing": ("external_capability_missing",),
    "signal_provider.computational_limit": ("design_signal_invalid",),
    "signal_provider.conditional_independence_missing": ("design_signal_invalid",),
    "signal_provider.invalid_object": ("external_binding_mismatch",),
    "signal_provider.invalid_request": ("invalid_argument",),
    "signal_provider.invalid_table": ("design_signal_invalid",),
    "signal_provider.law_incoherent": ("design_signal_invalid",),
    "signal_provider.missing_lineage": ("design_signal_invalid",),
    "signal_provider.observation_quantity_mismatch": ("quantity_semantics_mismatch",),
    "signal_provider.posterior_incoherent": ("design_signal_invalid",),
    "signal_provider.posterior_quantity_mismatch": ("quantity_semantics_mismatch",),
    "signal_provider.prior_mismatch": ("design_signal_invalid",),
    "signal_provider.request_fingerprint_mismatch": ("external_binding_mismatch",),
    "signal_provider.sample_size_invalid": ("design_signal_invalid",),
    "signal_provider.sample_size_mismatch": ("design_signal_invalid",),
    "signal_provider.trust_mismatch": ("external_binding_mismatch",),
    "signal_provider.trust_native_claimed": ("external_binding_mismatch",),
    "smoothed_dose_posterior.band_not_constructed": ("route_not_supported",),
    "smoothed_dose_posterior.cancelled": ("transport_budget_cancel",),
    "smoothed_dose_posterior.certificate_mismatch": ("invalid_argument",),
    "smoothed_dose_posterior.empty_scores": ("invalid_argument",),
    "smoothed_dose_posterior.external_power_prior": ("route_not_supported",),
    "smoothed_dose_posterior.invalid_graph": ("invalid_argument",),
    "smoothed_dose_posterior.mapped_prior": ("route_not_supported",),
    "smoothed_dose_posterior.non_positive_scale": ("invalid_argument",),
    "smoothed_dose_posterior.not_certified": ("invalid_argument",),
    "smoothed_dose_posterior.resource_limit": ("transport_budget_cancel",),
    "source_evidence.action_mismatch": ("invalid_argument",),
    "source_evidence.coordinate_mismatch": ("invalid_argument",),
    "source_evidence.coordinates_unavailable": ("invalid_argument",),
    "source_evidence.diagnostic_scope_mismatch": ("invalid_argument",),
    "source_evidence.forward_binding_mismatch": ("invalid_argument",),
    "source_evidence.invalid_artifact": ("invalid_argument",),
    "source_evidence.invalid_contract": ("invalid_argument",),
    "source_evidence.invalid_coordinates": ("invalid_argument",),
    "source_evidence.invalid_source_artifact": ("invalid_argument",),
    "source_evidence.limits_exceeded": ("invalid_argument",),
    "source_evidence.point_binding_mismatch": ("invalid_argument",),
    "source_evidence.resolution_scope_unsupported": ("invalid_argument",),
    "source_evidence.response_unavailable": ("invalid_argument",),
    "source_evidence.source_binding_mismatch": ("invalid_argument",),
    "source_evidence.source_encoding_failed": ("invalid_argument",),
    "source_evidence.source_execution_failed": ("invalid_argument",),
    "source_evidence.source_unverified": ("invalid_argument",),
    "source_projection.affine_contract_refused": ("decision_contract_unsatisfied",),
    "source_projection.attestation_unavailable": ("invalid_argument",),
    "source_projection.causal_source_unavailable": ("invalid_argument",),
    "source_projection.coordinate_mismatch": ("invalid_argument",),
    "source_projection.identity_mismatch": ("invalid_argument",),
    "source_projection.invalid_artifact": ("invalid_argument",),
    "source_projection.invalid_request": ("invalid_argument",),
    "source_projection.limits_exceeded": ("invalid_argument",),
    "source_projection.mean_binding_mismatch": ("invalid_argument",),
    "source_projection.mean_grid_required": ("invalid_argument",),
    "source_projection.original_consumer_refused": ("invalid_argument",),
    "source_projection.replay_mismatch": ("invalid_argument",),
    "source_projection.response_unavailable": ("invalid_argument",),
    "source_projection.source_binding_mismatch": ("invalid_argument",),
    "source_projection.unsupported_coordinate": ("invalid_argument",),
    "split_effect.degenerate_variance": ("invalid_argument",),
    "split_effect.design_singular": ("design_rank_deficient",),
    "split_effect.discovery_failed": ("invalid_argument",),
    "split_effect.outcome_precedes_treatment": ("invalid_argument",),
    "split_effect.search_not_finished": ("discovered_structure_unresolved",),
    "split_effect.source_invalid": ("invalid_argument",),
    "split_effect.source_not_canonical": ("invalid_argument",),
    "split_effect.structure_class": ("invalid_argument",),
    "split_effect.structure_unresolved": ("discovered_structure_unresolved",),
    "split_effect.too_few_rows": ("invalid_argument",),
    "structured_refusals.malformed_detail": ("invalid_argument",),
    "structured_refusals.missing_stage": ("invalid_argument",),
    "structured_refusals.unregistered_code": ("invalid_argument",),
    "study_candidate.declarations_required": ("design_signal_invalid",),
    "study_candidate.sample_below_obligation": ("design_signal_invalid",),
    "study_candidate.wrong_contract": ("design_signal_invalid",),
    "study_plan.arrival_mismatch": ("invalid_argument",),
    "study_plan.arrival_not_identified": ("transport_not_certified",),
    "study_plan.bounds_exceeded": ("route_not_supported",),
    "study_plan.budget": ("transport_budget_cancel",),
    "study_plan.invalid_artifact": ("transport_not_certified",),
    "study_plan.invalid_candidate": ("invalid_argument",),
    "study_plan.invalid_query": ("invalid_argument",),
    "study_plan.no_failure_to_repair": ("invalid_argument",),
    "study_plan.none_certified": ("transport_not_certified",),
    "study_plan.theorem_limited": ("transport_proven_non_transportable",),
    "survival_effect.arm_not_populated": ("arm_not_populated",),
    "survival_effect.cancelled": ("cancelled_no_claim",),
    "survival_effect.censoring_assumption_missing": ("required_option_missing",),
    "survival_effect.censoring_assumption_unknown": ("invalid_argument",),
    "survival_effect.censoring_covariates_missing": ("invalid_argument",),
    "survival_effect.censoring_support": ("positivity_violated",),
    "survival_effect.horizon_beyond_follow_up": ("positivity_violated",),
    "survival_effect.horizon_missing": ("required_option_missing",),
    "survival_effect.invalid_data": ("invalid_argument",),
    "survival_effect.invalid_event": ("invalid_argument",),
    "survival_effect.invalid_floor": ("invalid_argument",),
    "survival_effect.invalid_folds": ("invalid_argument",),
    "survival_effect.invalid_horizon": ("invalid_argument",),
    "survival_effect.invalid_levels": ("invalid_argument",),
    "survival_effect.invalid_request": ("invalid_argument",),
    "survival_effect.too_few_rows": ("invalid_argument",),
    "survival_effect.training_fold_without_events": ("arm_not_populated",),
    "survival_effect.treatment_positivity": ("positivity_violated",),
    "survival_effect.undeclared_level": ("treatment_level_undeclared",),
    "temporal_counterfactual.artifact_changed": ("route_not_supported",),
    "temporal_counterfactual.fit_mismatch": ("invalid_argument",),
    "temporal_counterfactual.history_name_invalid": ("invalid_argument",),
    "temporal_counterfactual.horizon_exceeded": ("invalid_argument",),
    "temporal_counterfactual.invalid_graph": ("invalid_argument",),
    "temporal_counterfactual.invalid_identity": ("invalid_argument",),
    "temporal_counterfactual.invalid_request": ("invalid_argument",),
    "temporal_counterfactual.latent_confounding": ("route_not_supported",),
    "temporal_counterfactual.limits_exceeded": ("invalid_argument",),
    "temporal_counterfactual.non_finite_history": ("invalid_argument",),
    "temporal_counterfactual.refuting_history": ("route_not_supported",),
    "temporal_counterfactual.shared_history_missing": ("route_not_supported",),
    "temporal_counterfactual.time_misaligned": ("route_not_supported",),
    "temporal_counterfactual.too_many_units": ("invalid_argument",),
    "temporal_counterfactual.unpaired_histories": ("route_not_supported",),
    "temporal_interval.consumer_unavailable": ("route_not_supported",),
    "temporal_interval.interval_not_supported": ("route_not_supported",),
    "temporal_interval.invalid_estimand": ("invalid_argument",),
    "temporal_interval.retired_method": ("route_not_supported",),
    "temporal_interval.retired_response_method": ("route_not_supported",),
    "temporal_interval.studentized_bounds_exceeded": ("route_not_supported",),
    "temporal_interval.studentized_estimator_not_certified": ("route_not_supported",),
    "temporal_interval.studentized_score_mismatch": ("route_not_supported",),
    "temporal_interval.studentized_unbalanced": ("route_not_supported",),
    "temporal_interval.studentized_zero_variance": ("route_not_supported",),
    "temporal_interval.too_few_units": ("too_few_clusters",),
    "temporal_interval.too_many_replicates": ("route_not_supported",),
    "temporal_interval.unknown_units": ("route_not_supported",),
    "temporal_interval.unsupported_history": ("route_not_supported",),
    "temporal_policy_transport.history_alphabet_incompatible": ("transport_missing_evidence",),
    "temporal_policy_transport.horizon_mismatch": ("route_not_supported",),
    "temporal_policy_transport.initial_law_declared_invariant": ("invalid_argument",),
    "temporal_policy_transport.invariance_binding_mismatch": ("transport_missing_evidence",),
    "temporal_policy_transport.mechanism_declared_twice": ("invalid_argument",),
    "temporal_policy_transport.mechanism_status_missing": ("transport_missing_evidence",),
    "temporal_policy_transport.mechanism_time_varying": ("transport_missing_evidence",),
    "temporal_policy_transport.mechanism_unknown": ("invalid_argument",),
    "temporal_policy_transport.target_initial_law": ("invalid_argument",),
    "temporal_policy_transport.target_initial_law_missing": ("transport_missing_evidence",),
    "temporal_policy_transport.target_is_source": ("invalid_argument",),
    "temporal_policy_transport.target_period": ("invalid_argument",),
    "temporal_policy_transport.target_support_outside_source": ("transport_support_failure",),
    "temporal_policy_transport.time_invariance_missing": ("transport_missing_evidence",),
    "temporal_policy_transport.value_mismatch": ("invalid_argument",),
    "temporal_refresh.factor_window_mismatch": ("invalid_argument",),
    "temporal_refresh.horizon_changed": ("route_not_supported",),
    "temporal_refresh.intervention_history_changed": ("route_not_supported",),
    "temporal_refresh.invalid_initial_law": ("invalid_argument",),
    "temporal_refresh.invalid_replacement": ("invalid_argument",),
    "temporal_refresh.lag_alignment_changed": ("route_not_supported",),
    "temporal_refresh.no_refresh": ("invalid_argument",),
    "temporal_refresh.premises_changed": ("route_not_supported",),
    "temporal_refresh.receipt_digest_mismatch": ("route_not_supported",),
    "temporal_refresh.receipt_mismatch": ("route_not_supported",),
    "temporal_refresh.stale_interval": ("route_not_supported",),
    "temporal_refresh.stale_snapshot": ("route_not_supported",),
    "temporal_refresh.window_missing": ("invalid_argument",),
    "temporal_transport.bounds_exceeded": ("route_not_supported",),
    "temporal_transport.checked_obstruction": ("transport_proven_non_transportable",),
    "temporal_transport.history_budget": ("transport_budget_cancel",),
    "temporal_transport.history_outside_support": ("transport_support_failure",),
    "temporal_transport.horizon": ("route_not_supported",),
    "temporal_transport.interval_requested": ("estimator_inference_mismatch",),
    "temporal_transport.invalid_sequence": ("invalid_argument",),
    "temporal_transport.invalid_spec": ("invalid_argument",),
    "temporal_transport.missing_evidence": ("transport_missing_evidence",),
    "temporal_transport.search_incomplete": ("transport_not_certified",),
    "temporal_transport.target_law_mismatch": ("transport_missing_evidence",),
    "temporal_transport.target_law_missing": ("transport_missing_evidence",),
    "tier_diagnostics.cancelled": ("cancelled_no_claim",),
    "tier_diagnostics.evalue_embeds_interval_limit": ("diagnostic_not_available",),
    "tier_diagnostics.evalue_not_computed": ("diagnostic_not_available",),
    "tier_diagnostics.no_evalue_interval_method": ("diagnostic_not_available",),
    "tier_diagnostics.not_a_tiered_average_result": ("invalid_argument",),
    "tier_diagnostics.overlap_not_carried": ("diagnostic_not_available",),
    "tier_diagnostics.unknown_scenarios_no_evalue": ("diagnostic_not_available",),
    "tier_diagnostics.unknown_scenarios_no_overlap": ("diagnostic_not_available",),
    "transport_revalidation.claim_target_mismatch": ("quantity_semantics_mismatch",),
    "transport_revalidation.duplicate_claim": ("invalid_argument",),
    "transport_revalidation.identity_mismatch": ("route_not_supported",),
    "transport_revalidation.provider_trust_not_applicable": ("invalid_argument",),
    "transport_revalidation.replay_mismatch": ("invalid_argument",),
    "transport_revalidation.scene_invalid": ("invalid_argument",),
    "transport_revalidation.source_undeclared": ("transport_missing_evidence",),
    "transport_revalidation.unknown_claim": ("invalid_argument",),
    "transport_revalidation.value_digest": ("invalid_argument",),
    "transport_sensitivity.blank_identity": ("invalid_argument",),
    "transport_sensitivity.coordinate_out_of_range": ("invalid_argument",),
    "transport_sensitivity.interval_not_composed": ("route_not_supported",),
    "transport_sensitivity.invalid_bound": ("invalid_argument",),
    "transport_sensitivity.invalid_grid": ("invalid_argument",),
    "transport_sensitivity.law_response_not_scalar": ("route_not_supported",),
    "transport_sensitivity.no_scalar_family": ("route_not_supported",),
    "transport_sensitivity.non_finite_point": ("invalid_argument",),
    "transport_sensitivity.not_transported": ("invalid_argument",),
    "transport_sensitivity.proportional_at_zero": ("invalid_argument",),
    "transport_sensitivity.quantity_not_in_target": ("quantity_semantics_mismatch",),
    "transport_sensitivity.route_undeclared": ("route_not_supported",),
    "transport_sensitivity.surface_refused": ("invalid_argument",),
    "transported_counterfactual.artifact_changed": ("route_not_supported",),
    "transported_counterfactual.cross_world_independence_missing": ("route_not_supported",),
    "transported_counterfactual.factor_missing": ("transport_missing_evidence",),
    "transported_counterfactual.invalid_diagram": ("invalid_argument",),
    "transported_counterfactual.invalid_factor": ("invalid_argument",),
    "transported_counterfactual.invalid_identity": ("invalid_argument",),
    "transported_counterfactual.invalid_law": ("invalid_argument",),
    "transported_counterfactual.invalid_model": ("invalid_argument",),
    "transported_counterfactual.invalid_query": ("invalid_argument",),
    "transported_counterfactual.invalid_request": ("invalid_argument",),
    "transported_counterfactual.joint_theorem_not_derived": ("route_not_supported",),
    "transported_counterfactual.limits_exceeded": ("invalid_argument",),
    "transported_counterfactual.noise_law_not_shared": ("route_not_supported",),
    "transported_counterfactual.nonadditive_mechanism": ("route_not_supported",),
    "transported_counterfactual.overlap_failure": ("transport_support_failure",),
    "transported_counterfactual.prerequisite_missing": ("transport_missing_evidence",),
    "transported_counterfactual.selection_on_mechanism": ("route_not_supported",),
    "transported_decision.calibration_label_inflated": ("transport_not_certified",),
    "transported_decision.coordinate_not_in_target": ("quantity_semantics_mismatch",),
    "transported_decision.exclusions_leave_no_comparison": ("decision_contract_unsatisfied",),
    "transported_decision.interval_not_supported": ("not_supported",),
    "transported_decision.invalid_value": ("invalid_argument",),
    "transported_decision.law_not_supplied": ("decision_contract_unsatisfied",),
    "transported_decision.mixed_transports": ("transport_not_certified",),
    "transported_decision.native_label_on_transported_value": ("transport_not_certified",),
    "transported_decision.not_transported": ("invalid_argument",),
    "transported_decision.route_needs_typed_source": ("invalid_argument",),
    "transported_decision.support_label_inflated": ("transport_not_certified",),
    "transported_decision.support_rule_unmet": ("transport_support_failure",),
    "transported_decision.transport_identity_incomplete": ("transport_missing_evidence",),
    "transported_decision.transport_identity_not_digest": ("transport_missing_evidence",),
    "transported_decision.transported_not_accepted": ("decision_contract_unsatisfied",),
    "transported_decision.uncertainty_requirement": ("decision_contract_unsatisfied",),
    "transported_decision_artifact.bounds_exceeded": ("invalid_argument",),
    "transported_decision_artifact.identity_expected": ("decision_contract_unsatisfied",),
    "transported_decision_artifact.invalid": ("invalid_argument",),
    "transported_decision_artifact.law_not_exportable": ("route_not_supported",),
    "transported_decision_artifact.replay_mismatch": ("invalid_argument",),
    "transported_effect.cohort_invalid": ("invalid_argument",),
    "transported_effect.cohort_not_numeric": ("invalid_argument",),
    "transported_effect.decision_source_mismatch": ("transport_not_certified",),
    "transported_effect.declaration_invalid": ("invalid_argument",),
    "transported_effect.feature_schema_mismatch": ("schema_mismatch",),
    "transported_effect.identity_mismatch": ("transport_not_certified",),
    "transported_effect.interval_not_supported": ("route_not_supported",),
    "transported_effect.invalid_request": ("invalid_argument",),
    "transported_effect.law_not_supplied": ("decision_contract_unsatisfied",),
    "transported_effect.link_not_identity": ("quantity_semantics_mismatch",),
    "transported_effect.model_invalid": ("invalid_argument",),
    "transported_effect.model_not_portable": ("invalid_argument",),
    "transported_effect.modifier_set_mismatch": ("transport_not_certified",),
    "transported_effect.no_overlap": ("transport_support_failure",),
    "transported_effect.non_finite_result": ("invalid_argument",),
    "transported_effect.out_of_support": ("transport_support_failure",),
    "transported_effect.outcome_units_required": ("invalid_argument",),
    "transported_effect.population_not_different": ("quantity_semantics_mismatch",),
    "transported_effect.prediction_refused": ("invalid_argument",),
    "transported_effect.selection_enters_mechanism": ("transport_not_certified",),
    "transported_effect.shift_declaration_mismatch": ("transport_not_certified",),
    "transported_effect.shift_declaration_required": ("transport_missing_evidence",),
    "transported_effect.support_contract_absent": ("invalid_argument",),
    "transported_effect.weights_invalid": ("invalid_argument",),
    "transported_effect.weights_not_normalized": ("invalid_argument",),
    "transported_policy.diagram_binding_mismatch": ("transport_missing_evidence",),
    "transported_policy.initial_law_declared_invariant": ("invalid_argument",),
    "transported_policy.mechanism_declared_twice": ("invalid_argument",),
    "transported_policy.mechanism_not_invariant": ("transport_missing_evidence",),
    "transported_policy.mechanism_status_missing": ("transport_missing_evidence",),
    "transported_policy.mechanism_unknown": ("invalid_argument",),
    "transported_policy.selection_diagram_missing": ("transport_missing_evidence",),
    "transported_policy.target_initial_law": ("invalid_argument",),
    "transported_policy.target_initial_law_missing": ("transport_missing_evidence",),
    "transported_policy.target_is_source": ("invalid_argument",),
    "transported_policy.target_population": ("invalid_argument",),
    "transported_policy.target_support_outside_source": ("transport_support_failure",),
    "transported_policy.target_utility_digest": ("invalid_argument",),
    "transported_policy.target_utility_missing": ("transport_missing_evidence",),
    "transported_policy.target_utility_provenance_missing": ("transport_missing_evidence",),
    "transported_policy.target_utility_support": ("invalid_argument",),
    "transported_policy.target_utility_value": ("invalid_argument",),
    "transported_policy.utility_declared_invariant": ("invalid_argument",),
    "transported_policy.value_mismatch": ("invalid_argument",),
    "transported_synthesis.native_population_blank": ("invalid_argument",),
    "transported_synthesis.no_contributions": ("invalid_argument",),
    "transported_synthesis.numeric_refused": ("invalid_argument",),
    "transported_synthesis.overlap_not_modelled": ("invalid_argument",),
    "transported_synthesis.prior_role_mismatch": ("invalid_argument",),
    "transported_synthesis.shared_transport_needs_covariance": ("joint_law_required",),
    "transported_synthesis.source_listed_twice": ("invalid_argument",),
    "transported_synthesis.target_population_mismatch": ("quantity_semantics_mismatch",),
    "transported_synthesis.transport_identity_incomplete": ("transport_missing_evidence",),
    "transported_synthesis.transport_variance_on_native_source": ("invalid_argument",),
    "transported_synthesis.transport_variance_undeclared": ("transport_missing_evidence",),
    "transported_synthesis.unknown_source": ("invalid_argument",),
    "treatment_levels.cancelled": ("cancelled_no_claim",),
    "treatment_levels.contrast_level_not_declared": ("invalid_argument",),
    "treatment_levels.full_population_undeclared_level": ("treatment_level_undeclared",),
    "treatment_levels.invalid_data": ("invalid_argument",),
    "treatment_levels.invalid_level_set": ("invalid_argument",),
    "treatment_levels.invalid_request": ("invalid_argument",),
    "treatment_levels.level_not_populated": ("arm_not_populated",),
    "treatment_levels.levels_missing": ("invalid_argument",),
    "treatment_levels.sparse_level": ("arm_not_populated",),
    "treatment_levels.training_fold_missing_level": ("arm_not_populated",),
    "treatment_levels.undeclared_level": ("treatment_level_undeclared",),
    "two_step_posterior.band_not_constructed": ("route_not_supported",),
    "two_step_posterior.empty_law": ("invalid_argument",),
    "two_step_posterior.external_power_prior": ("route_not_supported",),
    "two_step_posterior.mapped_prior": ("route_not_supported",),
    "two_step_posterior.non_positive_scale": ("invalid_argument",),
    "typed_mechanism_transport.cycle": ("mechanism_composition_incoherent",),
    "typed_mechanism_transport.dependence_invalid": ("uncertainty_representation_insufficient",),
    "typed_mechanism_transport.design_not_shared": ("quantity_semantics_mismatch",),
    "typed_mechanism_transport.fit_not_converged": ("mechanism_fit_not_converged",),
    "typed_mechanism_transport.fit_refused": ("mechanism_composition_incoherent",),
    "typed_mechanism_transport.invariance_contradicted": ("quantity_semantics_mismatch",),
    "typed_mechanism_transport.invariance_level_invalid": ("mechanism_composition_incoherent",),
    "typed_mechanism_transport.parameter_dependence_incomplete": (
        "uncertainty_representation_insufficient",
    ),
    "typed_mechanism_transport.parameter_dependence_unknown": (
        "uncertainty_representation_insufficient",
    ),
    "typed_mechanism_transport.parent_without_mechanism": ("mechanism_composition_incoherent",),
    "typed_mechanism_transport.parents_changed": ("quantity_semantics_mismatch",),
    "typed_mechanism_transport.regime_not_declared_invariant": ("quantity_semantics_mismatch",),
    "typed_mechanism_transport.selection_target_unknown_variable": ("quantity_semantics_mismatch",),
    "typed_mechanism_transport.shifted_without_target_evidence": ("transport_missing_evidence",),
    "typed_mechanism_transport.target_specification_differs": ("quantity_semantics_mismatch",),
    "typed_mechanism_transport.transport_identity_mismatch": ("quantity_semantics_mismatch",),
    "typed_mechanism_transport.unknown_variable": ("mechanism_composition_incoherent",),
    "vector_posterior.band_not_constructed": ("route_not_supported",),
    "vector_posterior.external_power_prior": ("route_not_supported",),
    "vector_posterior.mapped_prior": ("route_not_supported",),
    "vector_posterior.non_positive_scale": ("invalid_argument",),
    "vector_treatment.adjustment_set_mismatch": ("route_not_supported",),
    "vector_treatment.cancelled": ("cancelled_no_claim",),
    "vector_treatment.cluster_ids_required": ("invalid_argument",),
    "vector_treatment.cluster_unread": ("invalid_argument",),
    "vector_treatment.collinear_treatments": ("design_rank_deficient",),
    "vector_treatment.data_not_bound": ("route_not_supported",),
    "vector_treatment.degenerate_covariance": ("invalid_argument",),
    "vector_treatment.duplicate_contrast_coefficient": ("invalid_argument",),
    "vector_treatment.duplicate_name": ("invalid_argument",),
    "vector_treatment.empty_contrast": ("invalid_argument",),
    "vector_treatment.export_budget_exceeded": ("route_not_supported",),
    "vector_treatment.invalid_identity": ("invalid_argument",),
    "vector_treatment.invalid_lag": ("invalid_argument",),
    "vector_treatment.invalid_request": ("invalid_argument",),
    "vector_treatment.lag_required": ("invalid_argument",),
    "vector_treatment.lag_unread": ("invalid_argument",),
    "vector_treatment.leverage_one": ("invalid_argument",),
    "vector_treatment.limits_exceeded": ("invalid_argument",),
    "vector_treatment.memory_budget_exceeded": ("route_not_supported",),
    "vector_treatment.non_finite_value": ("invalid_argument",),
    "vector_treatment.rank_deficient_adjustment": ("design_rank_deficient",),
    "vector_treatment.row_count_mismatch": ("route_not_supported",),
    "vector_treatment.row_snapshot_mismatch": ("route_not_supported",),
    "vector_treatment.summary_inconsistent": ("invalid_argument",),
    "vector_treatment.summary_model_based_only": ("route_not_supported",),
    "vector_treatment.too_few_clusters": ("invalid_argument",),
    "vector_treatment.too_few_rows": ("invalid_argument",),
    "vector_treatment.too_few_treatments": ("invalid_argument",),
    "vector_treatment.too_many_treatments": ("invalid_argument",),
    "vector_treatment.treatment_without_variation": ("design_rank_deficient",),
    "vector_treatment.unknown_contrast_coefficient": ("invalid_argument",),
    "vector_treatment.weights_need_robust_covariance": ("option_not_applicable",),
    "vector_treatment.wrong_contract": ("route_not_supported",),
}
