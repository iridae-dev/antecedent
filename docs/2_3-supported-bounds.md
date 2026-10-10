# Supported 2.3 input bounds

These limits belong to the exact implementation and record shown. They bound the
listed input or default consumer scope; they do not authorize inference or
calibration. Python preflight limits can be stricter than the native API. A caller
can supply a separate execution/search budget where the API explicitly allows it;
a default budget is not a universal hard cap. Semantic-only policies do not acquire
a coordinate limit from an unrelated distribution artifact. The accompanying
`parity/limits_2_3.toml` compares each numeric declaration with its source guard,
original refusal path and an original executing fixture for that family. A fixture
entry is not a claim that every listed cap was independently stress-tested.

Measured inference adds exact protocol restrictions to these general native
input limits. A native maximum does not mean its entire range was measured:
for example, nested Fisher uses 1000..4000 observations, the fixed-prior nested
Bayesian adapter uses 2000..8000 and original graph labels at most 256 UTF-8
bytes, direct balanced studentized temporal response uses 75..200 units, and
whole-row BCa uses 1000..4000 observations with B2000. Current method/prior/
functional records are required separately for every reported scalar. See the
[population and uncertainty guide](2_3-population-time-uncertainty.md) for all
protocol checks; historical candidate bounds do not confer measured authority.

Run `python3 scripts/check_limits_agreement.py --release 2.3 --strict` to check
these statements; the original 2.2 check remains the default.

| Record | Bound | Value | Implementation |
| --- | --- | ---: | --- |
| 2.3A.F15.joint_distribution_semantics | max_draws | 100000 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F15.joint_distribution_semantics | max_coordinates | 1024 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.X2.cpdag_completion_scenarios | max_nodes | 6 | `crates/antecedent-identify/src/sid/cpdag_completion.rs` |
| 2.3A.X2.cpdag_completion_scenarios | max_completions | 64 | `crates/antecedent-identify/src/sid/scenarios.rs` |
| 2.3A.X2.shared_scenario_covariance | max_completions | 64 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| 2.3A.X2.shared_scenario_covariance | max_replicates | 2000 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| 2.3A.X2.shared_scenario_covariance | max_exact_rows | 24 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| 2.3A.X2.shared_scenario_covariance | max_exact_compositions | 4000000 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| 2.3A.X2.shared_scenario_covariance | max_table_rows | 10000 | `crates/antecedent-io/src/scenario_covariance_artifact.rs` |
| 2.3A.X2.shared_scenario_covariance | max_table_columns | 64 | `crates/antecedent-io/src/scenario_covariance_artifact.rs` |
| 2.3A.X4.joint_bayesian_transport | max_draws | 100000 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| 2.3A.X4.joint_bayesian_transport | max_model_parameters | 256 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| 2.3A.X4.binary_nested_markov_pilot | max_observed | 6 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| 2.3A.X4.binary_nested_markov_pilot | max_iterations | 1000000 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| 2.3A.X5.dependent_temporal_interval | max_horizon | 2 | `crates/antecedent-identify/src/sid/temporal_sequence.rs` |
| 2.3A.X5.dependent_temporal_interval | max_replicates | 2000 | `crates/antecedent-estimate/src/temporal_dependent_interval.rs` |
| 2.3A.X5.dependent_temporal_interval | max_units | 100000 | `crates/antecedent-estimate/src/temporal_dependent_interval.rs` |
| 2.3A.X5.uncertain_initial_state | max_initial_states | 64 | `crates/antecedent-estimate/src/temporal_initial_state.rs` |
| 2.3A.X5.uncertain_initial_state | max_horizon | 2 | `crates/antecedent-identify/src/sid/temporal_sequence.rs` |
| 2.3A.X5.new_period_refresh | max_horizon | 2 | `crates/antecedent-estimate/src/temporal_refresh.rs` |
| 2.3A.X5.new_period_refresh | max_history_states | 4096 | `crates/antecedent-identify/src/sid/temporal_sequence.rs` |
| 2.3A.X8.temporal_fixed_population_counterfactual | max_horizon | 2 | `crates/antecedent-counterfactual/src/temporal_cross_world.rs` |
| 2.3A.X8.temporal_fixed_population_counterfactual | max_units | 100000 | `crates/antecedent-counterfactual/src/temporal_cross_world.rs` |
| 2.3A.X10.sampled_observation_recovery | min_replicates | 2000 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| 2.3A.X10.sampled_observation_recovery | max_replicates | 2000 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| 2.3A.X10.sampled_observation_recovery | max_rows | 1000000 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| 2.3A.X10.sampled_observation_recovery | max_binary_variables | 6 | `crates/antecedent-estimate/src/recovery_sampled.rs` |
| 2.3A.F1.aligned_joint_draws | max_draws | 100000 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F1.aligned_joint_draws | max_coordinates | 1024 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F1.aligned_joint_draws | max_payload_bytes | 16777216 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F19.mapped_posterior_transfer | max_draws | 100000 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F19.mapped_posterior_transfer | max_coordinates | 1024 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F19.mapped_posterior_transfer | max_payload_bytes | 16777216 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F18.temporal_effect_constancy | max_partitions | 1024 | `crates/antecedent-estimate/src/effect_constancy.rs` |
| 2.3B.F4.decision_contract | max_actions | 1024 | `crates/antecedent-design/src/decision_contract.rs` |
| 2.3B.F4.decision_contract | max_inputs | 256 | `crates/antecedent-design/src/decision_contract.rs` |
| 2.3B.F4.decision_contract | max_utility_depth | 64 | `crates/antecedent-design/src/decision_contract.rs` |
| 2.3B.F5.decision_functionals | max_actions | 1024 | `crates/antecedent-design/src/decision_contract.rs` |
| 2.3B.F5.decision_functionals | max_inputs | 256 | `crates/antecedent-design/src/decision_contract.rs` |
| 2.3B.F5.decision_functionals | max_utility_depth | 64 | `crates/antecedent-design/src/decision_contract.rs` |
| 2.3B.F9.evidence_obligations | max_coordinates | 1024 | `crates/antecedent-core/src/evidence_obligation.rs` |
| 2.3B.F13.identification_repair | max_candidates | 16 | `crates/antecedent-design/src/repair.rs` |
| 2.3B.F13.identification_repair | max_subset_size | 4 | `crates/antecedent-design/src/repair.rs` |
| 2.3B.F13.identification_repair | max_operations | 100000 | `crates/antecedent-design/src/repair.rs` |
| 2.3B.F13.identification_repair | max_memory_bytes | 268435456 | `crates/antecedent-design/src/repair.rs` |
| 2.3B.B4.vector_treatment_coefficients | max_treatments | 64 | `crates/antecedent-estimate/src/vector_treatment.rs` |
| 2.3B.B4.vector_treatment_coefficients | hc_replay_row_cap | 4096 | `crates/antecedent-io/src/vector_treatment_artifact.rs` |
| 2.3B.B4.categorical_treatment_contrasts | hc_replay_row_cap | 4096 | `crates/antecedent-io/src/categorical_treatment_artifact.rs` |
| 2.3B.B4.latent_class_effects | max_classes | 4 | `crates/antecedent-estimate/src/latent_class_effects.rs` |
| 2.3C.C2.selective_recalculation | max_external_branches | 8 | `crates/antecedent-core/src/recalc.rs` |
| 2.3C.C2.recalc_receipt_artifact | max_bytes | 262144 | `crates/antecedent-io/src/recalc_receipt_artifact.rs` |
| 2.3C.C2.recalc_receipt_artifact | max_stages | 64 | `crates/antecedent-io/src/recalc_receipt_artifact.rs` |
| 2.3C.C3.composition_bundle | max_nodes | 1024 | `crates/antecedent-design/src/composition_bundle.rs` |
| 2.3C.C3.composition_bundle | max_edges | 8192 | `crates/antecedent-design/src/composition_bundle.rs` |
| 2.3A.X2.scenario_invariance_report | max_scenarios | 64 | `crates/antecedent-estimate/src/scenario_covariance.rs` |
| 2.3A.X8.transported_additive_counterfactual | max_covariates | 64 | `crates/antecedent-io/src/transported_counterfactual_artifact.rs` |
| 2.3A.X8.transported_additive_counterfactual | max_mechanisms | 256 | `crates/antecedent-io/src/transported_counterfactual_artifact.rs` |
| 2.3A.X8.transported_additive_counterfactual | max_support_points | 65536 | `crates/antecedent-io/src/transported_counterfactual_artifact.rs` |
| 2.3A.X8.transported_additive_counterfactual | max_declarations | 256 | `crates/antecedent-io/src/transported_counterfactual_artifact.rs` |
| 2.3A.X8.transported_additive_counterfactual | max_artifact_bytes | 67108864 | `python/src/transported_counterfactual_api.rs` |
| 2.3A.X4.learned_joint_transport | max_draws | 100000 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| 2.3A.X4.learned_joint_transport | max_model_parameters | 256 | `crates/antecedent-estimate/src/joint_bayesian_transport.rs` |
| 2.3A.X4.learned_joint_transport | max_basis_degree | 6 | `crates/antecedent-learn/src/bayesian_basis_regression.rs` |
| 2.3B.B3.mechanism_discrepancy_diagnostic | max_bytes | 16777216 | `crates/antecedent-io/src/mechanism_discrepancy_artifact.rs` |
| 2.3B.B3.msm_sensitivity | max_strata | 4096 | `crates/antecedent-validate/src/msm_sensitivity.rs` |
| 2.3B.B3.msm_sensitivity | max_grid_points | 1024 | `crates/antecedent-validate/src/msm_sensitivity.rs` |
| 2.3C.C2.frozen_scores_artifact | max_bytes | 134217728 | `crates/antecedent-io/src/frozen_scores_artifact.rs` |
| 2.3C.C2.frozen_scores_artifact | max_rows | 2097152 | `crates/antecedent-io/src/frozen_scores_artifact.rs` |
| 2.3C.C2.frozen_scores_artifact | max_columns | 64 | `crates/antecedent-io/src/frozen_scores_artifact.rs` |
| 2.3C.C2.operation_capability_inventory | max_external_branches | 8 | `crates/antecedent-core/src/recalc.rs` |
| 2.3C.C2.adjusted_recalculation | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.adjusted_recalculation | max_columns | 256 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.adjusted_recalculation | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.adjusted_recalculation | max_glm_iterations | 1000 | `python/src/recalc_adjusted_api.rs` |
| 2.3C.C2.adjusted_recalculation | max_json_bytes | 1048576 | `python/src/recalc_api.rs` |
| 2.3C.C2.dr_recalculation | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.dr_recalculation | max_columns | 256 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.dr_recalculation | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.dr_recalculation | max_json_bytes | 1048576 | `python/src/recalc_api.rs` |
| 2.3C.C2.dr_recalculation | max_folds | 20 | `python/src/recalc_dr_api.rs` |
| 2.3A.X4.binary_nested_markov_fisher | max_observed | 6 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| 2.3A.X4.binary_nested_markov_fisher | max_iterations | 1000000 | `crates/antecedent-estimate/src/nested_markov_binary.rs` |
| 2.3C.C2.static_recalculation | max_rows | 100000 | `crates/antecedent/src/analysis/recalc_static.rs` |
| 2.3C.C2.static_recalculation | max_values | 1000000 | `crates/antecedent/src/analysis/recalc_static.rs` |
| 2.3C.C2.static_recalculation | max_factor_cells | 1000000 | `python/src/recalc_static_api.rs` |
| 2.3C.C2.static_recalculation | max_variables | 12 | `python/src/recalc_static_api.rs` |
| 2.3C.C2.static_recalculation | max_levels_per_axis | 64 | `crates/antecedent/src/analysis/recalc_static.rs` |
| 2.3C.C2.static_recalculation | min_support_points | 2 | `crates/antecedent/src/analysis/recalc_static.rs` |
| 2.3C.C2.static_recalculation | max_support_points | 64 | `crates/antecedent/src/analysis/recalc_static.rs` |
| 2.3C.C2.static_recalculation | max_selected_actions | 64 | `crates/antecedent/src/analysis/recalc_static.rs` |
| 2.3C.C2.design_recalculation | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.design_recalculation | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.bayesian_recalculation | max_rows | 100000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| 2.3C.C2.bayesian_recalculation | max_values | 1000000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| 2.3C.C2.bayesian_recalculation | max_variables | 128 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| 2.3C.C2.bayesian_recalculation | min_draws | 2 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| 2.3C.C2.bayesian_recalculation | max_draws | 100000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| 2.3C.C2.bayesian_recalculation | max_workspace_values | 16000000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| 2.3C.C2.bayesian_recalculation | max_prior_artifact_bytes | 16000000 | `crates/antecedent/src/analysis/recalc_bayesian.rs` |
| 2.3C.C2.temporal_recalculation | max_variables | 5 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| 2.3C.C2.temporal_recalculation | max_periods | 2 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| 2.3C.C2.temporal_recalculation | max_units_rust | 8192 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| 2.3C.C2.temporal_recalculation | max_units_python | 4096 | `python/src/recalc_temporal_api.rs` |
| 2.3C.C2.temporal_recalculation | max_histories_rust | 131072 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| 2.3C.C2.temporal_recalculation | max_histories_python | 100000 | `python/src/recalc_temporal_api.rs` |
| 2.3C.C2.temporal_recalculation | max_source_support_rows | 4096 | `crates/antecedent/src/analysis/recalc_temporal.rs` |
| 2.3A.F15.native_authority_boundary | max_draws | 100000 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3A.F15.native_authority_boundary | max_coordinates | 1024 | `crates/antecedent-io/src/distribution_artifact.rs` |
| 2.3C.C3.rollout_state_binding | max_artifact_bytes | 33554432 | `crates/antecedent-design/src/rollout_artifact.rs` |
| 2.3C.C3.rollout_state_binding | max_states | 65536 | `crates/antecedent-design/src/rollout_artifact.rs` |
| 2.3C.C3.rollout_state_binding | max_quantities | 1024 | `crates/antecedent-design/src/rollout_artifact.rs` |
| 2.3C.C3.rollout_state_binding | max_actions | 65536 | `crates/antecedent-design/src/rollout_artifact.rs` |
| 2.3C.C3.rollout_state_binding | max_utility_cells | 4194304 | `crates/antecedent-design/src/rollout_artifact.rs` |
| 2.3C.C2.external_callback_recalculation | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.external_callback_recalculation | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| 2.3C.C2.external_callback_recalculation | max_variables | 64 | `crates/antecedent/src/analysis/recalc_external.rs` |
| 2.3C.C2.external_callback_recalculation | max_actions | 64 | `crates/antecedent/src/analysis/recalc_external.rs` |
| 2.3C.C2.external_callback_recalculation | max_model_parameters | 128 | `crates/antecedent/src/analysis/recalc_external.rs` |
| 2.3C.C2.external_callback_recalculation | max_branches | 8 | `crates/antecedent-core/src/recalc.rs` |
| 2.3C.C2.external_callback_recalculation | max_attempts | 256 | `crates/antecedent/src/analysis/recalc_external.rs` |
| 2.3C.C2.external_callback_recalculation | max_artifact_bytes | 6291456 | `crates/antecedent-io/src/external_callback_artifact.rs` |
| 2.3C.C2.external_callback_recalculation | max_response_bytes | 1048576 | `python/src/recalc_external_api.rs` |
| 2.3A.F18.effect_constancy_consumers | max_partitions | 1024 | `crates/antecedent-estimate/src/effect_constancy.rs` |
| 2.3A.F18.effect_constancy_consumers | max_catalog_sources | 1024 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| 2.3A.F18.effect_constancy_consumers | max_actions | 64 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| 2.3A.F18.effect_constancy_consumers | max_inputs | 16 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| 2.3A.F18.effect_constancy_consumers | max_utility_nodes | 2048 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| 2.3A.F18.effect_constancy_consumers | max_utility_depth | 64 | `crates/antecedent/src/analysis/effect_constancy_consumers.rs` |
| 2.3A.F18.effect_constancy_consumers | max_artifact_bytes | 67108864 | `python/src/effect_constancy_review_api.rs` |
| 2.3A.F18.effect_constancy_consumers | max_prior_artifact_bytes | 16777216 | `python/src/effect_constancy_review_api.rs` |
| 2.3B.X6.proposal_arrival | max_artifact_bytes | 33554432 | `crates/antecedent/src/analysis/proposal_arrival.rs` |
| 2.3B.X6.proposal_arrival | max_laws | 64 | `python/src/proposal_arrival_api.rs` |
| 2.3B.X6.proposal_arrival | max_cells_per_law | 65536 | `python/src/proposal_arrival_api.rs` |
| 2.3B.X6.proposal_arrival | max_variables | 12 | `python/src/proposal_arrival_api.rs` |
| 2.3B.X6.proposal_arrival | default_evaluation_operations | 100000 | `python/antecedent/proposal_arrival.py` |
| 2.3B.X6.proposal_arrival | default_evaluation_depth | 128 | `python/antecedent/proposal_arrival.py` |
| 2.3B.X6.proposal_arrival | max_evaluation_operations | 1000000 | `crates/antecedent/src/analysis/proposal_arrival.rs` |
| 2.3B.X6.proposal_arrival | max_evaluation_depth | 256 | `crates/antecedent/src/analysis/proposal_arrival.rs` |
| 2.3C.C4.composite_recalculation | max_native_variables | 12 | `python/src/recalc_composite_api.rs` |
| 2.3C.C4.composite_recalculation | max_rows | 100000 | `python/src/recalc_bounds.rs` |
| 2.3C.C4.composite_recalculation | max_values | 1000000 | `python/src/recalc_bounds.rs` |
| 2.3C.C4.composite_recalculation | max_actions | 64 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| 2.3C.C4.composite_recalculation | max_utility_nodes | 2048 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| 2.3C.C4.composite_recalculation | external_branches | 2 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| 2.3C.C4.composite_recalculation | utility_depth_exclusive | 64 | `crates/antecedent/src/analysis/recalc_composite.rs` |
| 2.3A.A0.source_evidence | max_artifact_bytes | 16777216 | `crates/antecedent-design/src/source_evidence.rs` |
| 2.3A.A0.source_evidence | max_logical_bytes | 16777216 | `crates/antecedent-design/src/source_evidence.rs` |
| 2.3A.A0.source_evidence | max_projection_bytes | 1048576 | `crates/antecedent-design/src/source_evidence.rs` |
| 2.3A.A0.source_evidence | max_actions | 4096 | `crates/antecedent-design/src/source_evidence.rs` |
| 2.3C.C3.source_projection | max_artifact_bytes | 18874368 | `crates/antecedent-design/src/source_projection_artifact.rs` |
| 2.3C.C3.source_projection | max_action_id_bytes | 256 | `crates/antecedent-design/src/source_projection_artifact.rs` |
| 2.3A.A0.composition_source_lineage | max_functional_artifact_bytes | 16777216 | `crates/antecedent-design/src/functional_source.rs` |
| 2.3A.A0.sensitivity_source_lineage | max_artifact_bytes | 16777216 | `crates/antecedent/src/analysis/sensitivity_source.rs` |

| 2.3C.C4.conditional_study_ranking | max_actions | 64 | Bounded original conditional ranking and full source replay; excess refuses before scientific work. |
| 2.3C.C4.conditional_study_ranking | max_candidates_per_action | 128 | Bounded original conditional ranking and full source replay; excess refuses before scientific work. |
| 2.3C.C4.conditional_study_ranking | max_candidates | 512 | Bounded original conditional ranking and full source replay; excess refuses before scientific work. |
| 2.3C.C4.conditional_study_ranking | max_name_bytes | 256 | Bounded original conditional ranking and full source replay; excess refuses before scientific work. |
| 2.3C.C4.conditional_study_ranking | max_artifact_bytes | 4,194,304 | Bounded original conditional ranking and full source replay; excess refuses before scientific work. |
| 2.3C.C4.conditional_study_ranking | max_policy_json_bytes | 262,144 | Bounded original conditional ranking and full source replay; excess refuses before scientific work. |

The typed native plan/objective ranker preserves its original native scoring scope.
The F14 portable study-artifact `max_candidates` budget applies to signal-provider
study rankings and artifact consumers; it is not a universal limit on original
native plan scoring. No portable-artifact or calibrated EVSI guarantee transfers
to graph-channel entropy, model-gap heuristics, Gram-SE reduction or callable
utilities.

Additional public adapter bounds:

| Record | Bound | Value | Implementation |
| --- | --- | ---: | --- |
| 2.3B.B3.msm_sensitivity | max_sample_rows | 1000000 | `python/antecedent/msm_sensitivity.py` |
| 2.3B.B3.msm_sensitivity | max_outcomes_per_arm | 256 | `python/antecedent/msm_sensitivity.py` |
| 2.3B.F4.decision_contract | max_effect_unit_bytes | 256 | `python/src/scalar_decision_api.rs` |

Empirical MSM requires an explicit supplemental sample, at most 4096 finite strata
and 256 distinct outcomes per arm/stratum. Both treatment arms must appear in each
stratum. These bounds do not validate sampling uncertainty or certify equality
with the original analysis snapshot. Scalar effect units label the original
outcome scale and do not perform a unit conversion.

| 2.3A.F15.measured_scalar_inference | max_bytes | 67,108,864 | Original measured envelope decode/source bound; caller limits may only tighten it. |
