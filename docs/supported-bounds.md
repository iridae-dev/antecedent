# Supported input bounds

These limits belong to the exact implementation and record shown. They bound the
listed input or default consumer scope; they do not authorize inference or
calibration. Python preflight limits can be stricter than the native API. A caller
can supply a separate execution/search budget where the API explicitly allows it;
a default budget is not a universal hard cap. Semantic-only policies do not acquire
a coordinate limit from an unrelated distribution artifact. Each value in the
generated table is the integer `bounds` entry of that record in
`parity/promotion.toml`. `parity/limits.toml` names the implementation place, the
source guard, the original refusal path and an original executing fixture for that
family. A fixture entry is not a claim that every listed cap was independently
stress-tested.

Measured inference adds exact protocol restrictions to these general native
input limits. A native maximum does not mean its entire range was measured:
for example, nested Fisher uses 1000..4000 observations, the fixed-prior nested
Bayesian adapter uses 2000..8000 and original graph labels at most 256 UTF-8
bytes, direct balanced studentized temporal response uses 75..200 units, and
whole-row BCa uses 1000..4000 observations with B2000. Current method/prior/
functional records are required separately for every reported scalar. See the
[population and uncertainty guide](population-time-uncertainty.md) for all
protocol checks; underlying engine bounds do not confer measured authority.

`python3 scripts/check_limits_agreement.py` regenerates the table between the
markers below and compares it. Prose outside those markers is not read.

<!-- generated:supported-bounds:begin -->
| Record | Bound | Value | Implementation |
| --- | --- | ---: | --- |
| distribution.joint_alignment | max_draws | 100000 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| distribution.joint_alignment | max_coordinates | 1024 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| transport.cpdag_completion | max_nodes | 6 | `crates/antecedent-identify/src/sid/cpdag_completion.rs` |
| transport.cpdag_completion | max_completions | 64 | `crates/antecedent-identify/src/sid/scenarios.rs` |
| transport.shared_scenario_covariance | max_completions | 64 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| transport.shared_scenario_covariance | max_replicates | 2000 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| transport.shared_scenario_covariance | max_exact_rows | 24 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| transport.shared_scenario_covariance | max_exact_compositions | 4000000 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| transport.shared_scenario_covariance | max_table_rows | 10000 | `crates/antecedent-io/src/scenario_covariance_artifact.rs` |
| transport.shared_scenario_covariance | max_table_columns | 64 | `crates/antecedent-io/src/scenario_covariance_artifact.rs` |
| transport.joint_bayesian | max_draws | 100000 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| transport.joint_bayesian | max_model_parameters | 256 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| identify.binary_nested_markov | max_observed | 6 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| identify.binary_nested_markov | max_iterations | 1000000 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| temporal.dependent_interval | max_horizon | 2 | `crates/antecedent-identify/src/sid/temporal_sequence.rs` |
| temporal.dependent_interval | max_replicates | 2000 | `crates/antecedent-estimate/src/temporal_dependent_interval.rs` |
| temporal.dependent_interval | max_units | 100000 | `crates/antecedent-estimate/src/temporal_dependent_interval.rs` |
| temporal.uncertain_initial_state | max_initial_states | 64 | `crates/antecedent-estimate/src/temporal_initial_state.rs` |
| temporal.uncertain_initial_state | max_horizon | 2 | `crates/antecedent-identify/src/sid/temporal_sequence.rs` |
| temporal.new_period_refresh | max_horizon | 2 | `crates/antecedent-estimate/src/temporal_refresh.rs` |
| temporal.new_period_refresh | max_history_states | 4096 | `crates/antecedent-identify/src/sid/temporal_sequence.rs` |
| counterfactual.temporal_fixed_population | max_horizon | 2 | `crates/antecedent-counterfactual/src/temporal_cross_world.rs` |
| counterfactual.temporal_fixed_population | max_units | 100000 | `crates/antecedent-counterfactual/src/temporal_cross_world.rs` |
| recovery.sampled_observation | min_replicates | 2000 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| recovery.sampled_observation | max_replicates | 2000 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| recovery.sampled_observation | max_rows | 1000000 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| recovery.sampled_observation | max_binary_variables | 6 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| distribution.aligned_joint_draws | max_draws | 100000 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| distribution.aligned_joint_draws | max_coordinates | 1024 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| distribution.aligned_joint_draws | max_payload_bytes | 16777216 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| distribution.mapped_posterior_transfer | max_draws | 100000 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| distribution.mapped_posterior_transfer | max_coordinates | 1024 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| distribution.mapped_posterior_transfer | max_payload_bytes | 16777216 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| temporal.effect_constancy | max_partitions | 1024 | `crates/antecedent-estimate/src/effect_constancy.rs` |
| decision.contract | max_actions | 1024 | `crates/antecedent-design/src/decision_contract.rs` |
| decision.contract | max_inputs | 256 | `crates/antecedent-design/src/decision_contract.rs` |
| decision.contract | max_utility_depth | 64 | `crates/antecedent-design/src/decision_contract.rs` |
| decision.functionals | max_actions | 1024 | `crates/antecedent-design/src/decision_contract.rs` |
| decision.functionals | max_inputs | 256 | `crates/antecedent-design/src/decision_contract.rs` |
| decision.functionals | max_utility_depth | 64 | `crates/antecedent-design/src/decision_contract.rs` |
| design.evidence_obligations | max_coordinates | 1024 | `crates/antecedent-contract/src/evidence_obligation.rs` |
| design.identification_repair | max_candidates | 16 | `crates/antecedent-design/src/repair.rs` |
| design.identification_repair | max_subset_size | 4 | `crates/antecedent-design/src/repair.rs` |
| design.identification_repair | max_operations | 100000 | `crates/antecedent-design/src/repair.rs` |
| design.identification_repair | max_memory_bytes | 268435456 | `crates/antecedent-design/src/repair.rs` |
| estimate.vector_treatment | max_treatments | 64 | `crates/antecedent-estimate/src/vector_treatment.rs` |
| estimate.vector_treatment | hc_replay_row_cap | 4096 | `crates/antecedent-io/src/vector_treatment_artifact.rs` |
| estimate.categorical_treatment | hc_replay_row_cap | 4096 | `crates/antecedent-io/src/categorical_treatment_artifact.rs` |
| estimate.latent_class_effects | max_classes | 4 | `crates/antecedent-estimate/src/latent_class_effects.rs` |
| recalculation.selective | max_external_branches | 8 | `crates/antecedent-core/src/recalc.rs` |
| recalculation.receipt | max_bytes | 262144 | `crates/antecedent-io-core/src/recalc_receipt_artifact.rs` |
| recalculation.receipt | max_stages | 64 | `crates/antecedent-io-core/src/recalc_receipt_artifact.rs` |
| composition.bundle | max_nodes | 1024 | `crates/antecedent-design/src/composition_bundle.rs` |
| composition.bundle | max_edges | 8192 | `crates/antecedent-design/src/composition_bundle.rs` |
| transport.scenario_invariance | max_scenarios | 64 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| counterfactual.transported_additive | max_covariates | 64 | `crates/antecedent-io-core/src/transported_counterfactual_artifact.rs` |
| counterfactual.transported_additive | max_mechanisms | 256 | `crates/antecedent-io-core/src/transported_counterfactual_artifact.rs` |
| counterfactual.transported_additive | max_support_points | 65536 | `crates/antecedent-io-core/src/transported_counterfactual_artifact.rs` |
| counterfactual.transported_additive | max_declarations | 256 | `crates/antecedent-io-core/src/transported_counterfactual_artifact.rs` |
| counterfactual.transported_additive | max_artifact_bytes | 67108864 | `python/src/transported_counterfactual_api.rs` |
| transport.learned_joint | max_draws | 100000 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| transport.learned_joint | max_model_parameters | 256 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| transport.learned_joint | max_basis_degree | 6 | `crates/antecedent-learn/src/bayesian_basis_regression.rs` |
| sensitivity.mechanism_discrepancy | max_bytes | 16777216 | `crates/antecedent-io/src/mechanism_discrepancy_artifact.rs` |
| sensitivity.marginal_structural_model | max_strata | 4096 | `crates/antecedent-validate/src/msm_sensitivity.rs` |
| sensitivity.marginal_structural_model | max_grid_points | 1024 | `crates/antecedent-validate/src/msm_sensitivity.rs` |
| recalculation.frozen_scores_artifact | max_bytes | 134217728 | `crates/antecedent-io-core/src/frozen_scores_artifact.rs` |
| recalculation.frozen_scores_artifact | max_rows | 2097152 | `crates/antecedent-io-core/src/frozen_scores_artifact.rs` |
| recalculation.frozen_scores_artifact | max_columns | 64 | `crates/antecedent-io-core/src/frozen_scores_artifact.rs` |
| recalculation.operation_inventory | max_external_branches | 8 | `crates/antecedent-core/src/recalc.rs` |
| recalculation.adjusted | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| recalculation.adjusted | max_columns | 256 | `python/src/recalc_bounds.rs` |
| recalculation.adjusted | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| recalculation.adjusted | max_glm_iterations | 1000 | `python/src/recalc_adjusted_api.rs` |
| recalculation.adjusted | max_json_bytes | 1048576 | `python/src/recalc_api.rs` |
| recalculation.doubly_robust | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| recalculation.doubly_robust | max_columns | 256 | `python/src/recalc_bounds.rs` |
| recalculation.doubly_robust | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| recalculation.doubly_robust | max_json_bytes | 1048576 | `python/src/recalc_api.rs` |
| recalculation.doubly_robust | max_folds | 20 | `python/src/recalc_dr_api.rs` |
| identify.binary_nested_markov_fisher | max_observed | 6 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| identify.binary_nested_markov_fisher | max_iterations | 1000000 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| recalculation.static_response | max_rows | 100000 | `crates/antecedent/src/analysis/recalc_static.rs` |
| recalculation.static_response | max_values | 1000000 | `crates/antecedent/src/analysis/recalc_static.rs` |
| recalculation.static_response | max_factor_cells | 1000000 | `python/src/recalc_static_api.rs` |
| recalculation.static_response | max_variables | 12 | `python/src/recalc_static_api.rs` |
| recalculation.static_response | max_levels_per_axis | 64 | `crates/antecedent/src/analysis/recalc_static.rs` |
| recalculation.static_response | min_support_points | 2 | `crates/antecedent/src/analysis/recalc_static.rs` |
| recalculation.static_response | max_support_points | 64 | `crates/antecedent/src/analysis/recalc_static.rs` |
| recalculation.static_response | max_selected_actions | 64 | `crates/antecedent/src/analysis/recalc_static.rs` |
| recalculation.design_specific | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| recalculation.design_specific | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| recalculation.bayesian | max_rows | 100000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| recalculation.bayesian | max_values | 1000000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| recalculation.bayesian | max_variables | 128 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| recalculation.bayesian | min_draws | 2 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| recalculation.bayesian | max_draws | 100000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| recalculation.bayesian | max_workspace_values | 16000000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| recalculation.bayesian | max_prior_artifact_bytes | 16000000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| recalculation.temporal | max_variables | 5 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| recalculation.temporal | max_periods | 2 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| recalculation.temporal | max_units_rust | 8192 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| recalculation.temporal | max_units_python | 4096 | `python/src/recalc_temporal_api.rs` |
| recalculation.temporal | max_histories_rust | 131072 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| recalculation.temporal | max_histories_python | 100000 | `python/src/recalc_temporal_api.rs` |
| recalculation.temporal | max_source_support_rows | 4096 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| composition.native_authority | max_draws | 100000 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| composition.native_authority | max_coordinates | 1024 | `crates/antecedent-io-core/src/distribution_artifact.rs` |
| composition.rollout_state | max_artifact_bytes | 33554432 | `crates/antecedent-design/src/rollout_artifact.rs` |
| composition.rollout_state | max_states | 65536 | `crates/antecedent-design/src/rollout_artifact.rs` |
| composition.rollout_state | max_quantities | 1024 | `crates/antecedent-design/src/rollout_artifact.rs` |
| composition.rollout_state | max_actions | 65536 | `crates/antecedent-design/src/rollout_artifact.rs` |
| composition.rollout_state | max_utility_cells | 4194304 | `crates/antecedent-design/src/rollout_artifact.rs` |
| recalculation.external_callback | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| recalculation.external_callback | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| recalculation.external_callback | max_variables | 64 | `crates/antecedent/src/analysis/recalc_external.rs` |
| recalculation.external_callback | max_actions | 64 | `crates/antecedent/src/analysis/recalc_external.rs` |
| recalculation.external_callback | max_model_parameters | 128 | `crates/antecedent/src/analysis/recalc_external.rs` |
| recalculation.external_callback | max_branches | 8 | `crates/antecedent-core/src/recalc.rs` |
| recalculation.external_callback | max_attempts | 256 | `crates/antecedent/src/analysis/recalc_external.rs` |
| recalculation.external_callback | max_artifact_bytes | 6291456 | `crates/antecedent-io-core/src/external_callback_artifact.rs` |
| recalculation.external_callback | max_response_bytes | 1048576 | `python/src/recalc_external_api.rs` |
| temporal.effect_constancy_consumers | max_partitions | 1024 | `crates/antecedent-estimate/src/effect_constancy.rs` |
| temporal.effect_constancy_consumers | max_catalog_sources | 1024 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| temporal.effect_constancy_consumers | max_actions | 64 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| temporal.effect_constancy_consumers | max_inputs | 16 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| temporal.effect_constancy_consumers | max_utility_nodes | 2048 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| temporal.effect_constancy_consumers | max_utility_depth | 64 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| temporal.effect_constancy_consumers | max_artifact_bytes | 67108864 | `python/src/effect_constancy_review_api.rs` |
| temporal.effect_constancy_consumers | max_prior_artifact_bytes | 16777216 | `python/src/effect_constancy_review_api.rs` |
| design.proposal_arrival | max_artifact_bytes | 33554432 | `crates/antecedent/src/analysis/proposal_arrival.rs` |
| design.proposal_arrival | max_laws | 64 | `python/src/proposal_arrival_api.rs` |
| design.proposal_arrival | max_cells_per_law | 65536 | `python/src/proposal_arrival_api.rs` |
| design.proposal_arrival | max_variables | 12 | `python/src/proposal_arrival_api.rs` |
| design.proposal_arrival | default_evaluation_operations | 100000 | `python/antecedent/proposal_arrival.py` |
| design.proposal_arrival | default_evaluation_depth | 128 | `python/antecedent/proposal_arrival.py` |
| design.proposal_arrival | max_evaluation_operations | 1000000 | `crates/antecedent/src/analysis/proposal_arrival.rs` |
| design.proposal_arrival | max_evaluation_depth | 256 | `crates/antecedent/src/analysis/proposal_arrival.rs` |
| recalculation.composite | max_native_variables | 12 | `python/src/recalc_composite_api.rs` |
| recalculation.composite | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| recalculation.composite | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| recalculation.composite | max_actions | 64 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| recalculation.composite | max_utility_nodes | 2048 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| recalculation.composite | external_branches | 2 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| recalculation.composite | utility_depth_exclusive | 64 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| composition.source_evidence | max_artifact_bytes | 16777216 | `crates/antecedent-design/src/source_evidence.rs` |
| composition.source_evidence | max_logical_bytes | 16777216 | `crates/antecedent-design/src/source_evidence.rs` |
| composition.source_evidence | max_projection_bytes | 1048576 | `crates/antecedent-design/src/source_evidence.rs` |
| composition.source_evidence | max_actions | 4096 | `crates/antecedent-design/src/source_evidence.rs` |
| composition.source_projection | max_artifact_bytes | 18874368 | `crates/antecedent-design/src/source_projection_artifact.rs` |
| composition.source_lineage | max_functional_artifact_bytes | 16777216 | `crates/antecedent-design/src/functional_source.rs` |
| sensitivity.source_lineage | max_artifact_bytes | 16777216 | `crates/antecedent/src/analysis/sensitivity_source.rs` |
| design.conditional_study_ranking | max_actions | 64 | `crates/antecedent/src/analysis/conditional_study_ranking.rs` |
| design.conditional_study_ranking | max_candidates_per_action | 128 | `crates/antecedent/src/analysis/conditional_study_ranking.rs` |
| design.conditional_study_ranking | max_candidates | 512 | `crates/antecedent/src/analysis/conditional_study_ranking.rs` |
| design.conditional_study_ranking | max_name_bytes | 256 | `crates/antecedent/src/analysis/conditional_study_ranking.rs` |
| design.conditional_study_ranking | max_artifact_bytes | 4194304 | `crates/antecedent/src/analysis/conditional_study_ranking.rs` |
| design.conditional_study_ranking | max_policy_json_bytes | 262144 | `python/src/recalc_composite_api.rs` |
| composition.source_projection | max_action_id_bytes | 256 | `crates/antecedent-design/src/source_projection_artifact.rs` |
| sensitivity.marginal_structural_model | max_sample_rows | 1000000 | `python/antecedent/msm_sensitivity.py` |
| sensitivity.marginal_structural_model | max_outcomes_per_arm | 256 | `python/antecedent/msm_sensitivity.py` |
| decision.contract | max_effect_unit_bytes | 256 | `python/src/scalar_decision_api.rs` |
| inference.measured_scalar | max_bytes | 67108864 | `crates/antecedent-io/src/measured_inference.rs` |
| counterfactual.transport_general | max_observed | 8 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_levels | 4 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_worlds | 6 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_atoms | 16 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_network_nodes | 24 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_sources | 4 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_controllable_per_source | 4 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_regimes | 64 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_sum_cells | 65536 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_operations | 400000 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_general | max_depth | 256 | `crates/antecedent-identify/src/sid/counterfactual_transport.rs` |
| counterfactual.transport_sampled | max_tables | 64 | `crates/antecedent-estimate/src/counterfactual_transport_sampled.rs` |
| counterfactual.transport_sampled | max_table_cells | 65536 | `crates/antecedent-estimate/src/counterfactual_transport_sampled.rs` |
| counterfactual.transport_sampled | max_envelope_evaluations | 65536 | `crates/antecedent-estimate/src/counterfactual_transport_sampled.rs` |
| mechanism_composition.acyclic_joint_model | max_mechanisms | 64 | `crates/antecedent-model/src/compose/descriptor.rs` |
| mechanism_composition.acyclic_joint_model | max_parents_per_mechanism | 16 | `crates/antecedent-model/src/compose/descriptor.rs` |
| mechanism_composition.acyclic_joint_model | max_latents | 32 | `crates/antecedent-model/src/compose/descriptor.rs` |
| mechanism_composition.acyclic_joint_model | max_finite_states | 1048576 | `crates/antecedent-model/src/compose/descriptor.rs` |
| mechanism_composition.acyclic_joint_model | max_sample_cells | 67108864 | `crates/antecedent-model/src/compose.rs` |
<!-- generated:supported-bounds:end -->

The typed native plan/objective ranker preserves its original native scoring scope.
The portable study-artifact `max_candidates` budget applies to signal-provider
study rankings and artifact consumers; it is not a universal limit on original
native plan scoring. No portable-artifact or calibrated EVSI guarantee transfers
to graph-channel entropy, model-gap heuristics, Gram-SE reduction or callable
utilities.

Empirical MSM requires an explicit supplemental sample, bounded by
`sensitivity.marginal_structural_model.max_strata` and
`sensitivity.marginal_structural_model.max_outcomes_per_arm` in the table above.
Both treatment arms must appear in each stratum. These bounds do not validate
sampling uncertainty or certify equality with the original analysis snapshot.
Scalar effect units label the original outcome scale and do not perform a unit
conversion. `inference.measured_scalar.max_bytes` is the original measured
envelope decode/source bound; caller limits may only tighten it.
