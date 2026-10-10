# 2.2 baseline for the 2.3 work

Frozen source: annotated `v2.2.0` at `00e020885770cc397f6d2c996fcea96e51cea5ea`. This snapshot reads the tag, not the 2.3 working tree.

The 2.2 release is [published](https://github.com/iridae-dev/antecedent/releases/tag/v2.2.0).
Its [CI run](https://github.com/iridae-dev/antecedent/actions/runs/37429694924) completed successfully on the tag commit, including required Rust, Python, dependency, wheel and domain jobs. The [publish-release run](https://github.com/iridae-dev/antecedent/actions/runs/37434348369) passed its tag/version/CI verification, wheel provenance, docs bundle and publication jobs on the same SHA. The workspace and Python package at the tag both declare `2.2.0`; the checked-in release notes describe the licensed analytic learned-continuous interval and the failed percentile method separately. These records establish the release cut; they do not separately attest a local invocation of `gate_release_candidate.sh`.

At 2.3 kickoff, `generate_calibration_backlog.py --check` reports **40 unmeasured cells / 18 distinct coordinates**. `check_release_claims.py --final` and `check_limits_agreement.py --strict` pass against the 2.2 registry and notes.

## Immutable file snapshot

| File at v2.2.0 | SHA-256 |
| --- | --- |
| `parity/promotion_2_2.toml` | `78315785beca65cefc140461c59b340901661a4ee77e019c9580f3fb9b75b7f3` |
| `parity/support_licensed.toml` | `64022b8f6d7fcd2c955c15263273470d8a6e16986b3892a8e565f5b4e25cb053` |
| `parity/support_closed.toml` | `a73e1bdcf119b4033633cb28b5182728d8815b50b5bc67f578de4fb098bfcfc7` |
| `parity/transport_stages.toml` | `3234929e95e588b8c536b7752472297a3e5dc33e3c3b8ffd053db82602641fab` |
| `parity/transport_coverage.md` | `68ca6346ea778ec46bc33121cee47f263299f9b75c1f682bb72b492e6796cbb3` |
| `parity/counterfactual_coverage.md` | `2b6ee3f7376a7c1b21a7cabd950ae0fc2ce229dd4ff66877dd13ffae68ffe572` |
| `parity/coverage_records.toml` | `a48e90605ae05111ad41d970119dd1922c1d6147ddb05fd07bc7f592174c12d6` |
| `parity/calibration_backlog.md` | `5f0ef3eebc7c333a2476b4c8a60df9cc629457fdbad869795e4bd81f743d5fb4` |

## Closed-route disposition

A carryover candidate requires its own 2.3 theorem, provider, value, refusal, artifact and calibration gates before opening. A deliberate refusal keeps its 2.2 reason code. No route is declared superseded at kickoff. A listed candidate remains closed today.

Reviewed set: 66 routes; 29 carryover candidates; 37 deliberate refusals; 0 superseded.

| 2.2 record | Closed route | Reason code | Disposition | Next review |
| --- | --- | --- | --- | --- |
| `transport.multi_source_limited_experiment` | `antecedent.transport.multi_source_z_transport.uncertainty_joint_bootstrap` | `cell_not_licensed` | carryover candidate | C0 X1 |
| `transport.finite_scenario_envelope` | `antecedent.PreparedTransportScenarios.aggregate_interval` | `scenario_aggregate_not_licensed` | carryover candidate | A1 X2 |
| `transport.finite_scenario_envelope` | `antecedent.transport.advanced.PreparedTransportScenariosStage.aggregate_interval` | `scenario_aggregate_not_licensed` | carryover candidate | A1 X2 |
| `transport.finite_scenario_envelope` | `transport.scenarios.aggregate_inference` | `scenario_aggregate_not_licensed` | carryover candidate | A1 X2 |
| `transport.finite_scenario_envelope` | `transport.scenarios.equivalence_class_native` | `route_not_supported` | carryover candidate | A1 X2 |
| `transport.learned_continuous_trial` | `antecedent.transport.learned_continuous.uncertainty_joint_outer_bootstrap` | `estimator_inference_mismatch` | deliberate refusal | retain boundary |
| `transport.two_step_temporal` | `antecedent.transport.advanced.PreparedTemporalTransportStage.interval` | `estimator_inference_mismatch` | carryover candidate | A4 X5 |
| `transport.two_step_temporal` | `transport.temporal.uncertainty` | `estimator_inference_mismatch` | carryover candidate | A4 X5 |
| `counterfactual.path_specific_edge` | `antecedent.cross_world.uncertainty` | `estimator_inference_mismatch` | deliberate refusal | retain boundary |
| `counterfactual.path_specific_edge` | `antecedent.cross_world.structure_outside_contract` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `counterfactual.path_specific_edge` | `antecedent.cross_world.recanting_witness_edge_sets` | `cross_world_not_identified` | deliberate refusal | retain boundary |
| `counterfactual.path_specific_edge` | `antecedent.cross_world.query_outside_contract` | `route_not_supported` | deliberate refusal | retain boundary |
| `identify.mixed_source_proof_search` | `antecedent.transport.advanced.MixedSourceStage.prepare_empirical` | `cell_not_licensed` | carryover candidate | B2 X9 |
| `transport.admg_conditional` | `antecedent.transport.advanced.AdmgConditionalTransportStage.prepare_empirical` | `cell_not_licensed` | carryover candidate | B1 X4 |
| `recovery.binary_observation` | `antecedent.transport.advanced.ObservationRecoveryStage.prepare_empirical` | `cell_not_licensed` | carryover candidate | A6 X10 |
| `transport.smoothed_dose_response` | `antecedent.PreparedSmoothedDose.interval` | `cell_not_licensed` | carryover candidate | C0 X4 |
| `transport.smoothed_dose_response` | `antecedent.transport.advanced.PreparedSmoothedDose.interval` | `cell_not_licensed` | carryover candidate | C0 X4 |
| `transport.smoothed_dose_response` | `antecedent.transport.smoothed_dose.uncertainty_joint_outer_bootstrap` | `cell_not_licensed` | carryover candidate | C0 X4 |
| `sensitivity.joint_uncertainty` | `antecedent.PreparedZTransport.joint_mechanism_sensitivity_interval` | `cell_not_licensed` | carryover candidate | C0 X3 |
| `sensitivity.joint_uncertainty` | `antecedent.transport.advanced.joint_mechanism_sensitivity_interval` | `cell_not_licensed` | carryover candidate | C0 X3 |
| `sensitivity.joint_uncertainty` | `antecedent.transport.joint_sensitivity.uncertainty_conservative_endpoint_bootstrap` | `cell_not_licensed` | carryover candidate | C0 X3 |
| `counterfactual.admg_identification` | `antecedent.counterfactual_id.uncertainty` | `estimator_inference_mismatch` | deliberate refusal | retain boundary |
| `counterfactual.admg_identification` | `antecedent.counterfactual_id.structure_outside_contract` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `counterfactual.admg_identification` | `antecedent.counterfactual_id.query_outside_contract` | `route_not_supported` | deliberate refusal | retain boundary |
| `counterfactual.admg_identification` | `antecedent.counterfactual_id.path_specific_admg` | `route_not_supported` | carryover candidate | B2 X8 |
| `counterfactual.admg_identification` | `antecedent.counterfactual_id.not_identified_by_id_star` | `route_not_supported` | deliberate refusal | retain boundary |
| `estimate.matched_case_control_odds_ratio` | `antecedent_estimate.parse_matched_estimand` | `effect_not_identified` | deliberate refusal | retain boundary |
| `estimate.matched_case_control_odds_ratio` | `antecedent_estimate.refuse_matched_interval` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `estimate.matched_case_control_odds_ratio` | `antecedent.matched.risk_scale_estimand` | `effect_not_identified` | deliberate refusal | retain boundary |
| `estimate.matched_case_control_odds_ratio` | `antecedent.matched.interval` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `decision.finite_action_inverse` | `antecedent.inverse.probability_target` | `cell_not_licensed` | carryover candidate | B4 F7 |
| `decision.finite_action_inverse` | `antecedent.inverse.quantile_target` | `cell_not_licensed` | carryover candidate | B4 F7 |
| `decision.finite_action_inverse` | `antecedent.inverse.observational_scenarios` | `cell_not_licensed` | carryover candidate | B4 F7 |
| `decision.finite_action_inverse` | `antecedent.inverse.allowed_unlicensed_forward` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `estimate.penalized_propensity_aipw` | `antecedent_estimate.PropensityNuisance.lasso_full_sample` | `selection_inference_not_licensed` | deliberate refusal | retain boundary |
| `estimate.penalized_propensity_aipw` | `antecedent.estimators.PropensityPenalty.lasso_full_sample` | `selection_inference_not_licensed` | deliberate refusal | retain boundary |
| `estimate.penalized_propensity_aipw` | `antecedent_estimate.PropensityNuisance.ml_fallback` | `nuisance_fallback_not_licensed` | deliberate refusal | retain boundary |
| `estimate.clustered_dml_aipw` | `antecedent_estimate.AipwAte.cluster_dml_interval` | `cluster_interval_not_licensed` | carryover candidate | B1 X4 |
| `estimate.clustered_dml_aipw` | `antecedent_estimate.ClusterDml.few_clusters` | `too_few_clusters` | deliberate refusal | retain boundary |
| `estimate.clustered_dml_aipw` | `antecedent_estimate.ClusterDml.dyadic` | `dyadic_dependence_not_licensed` | carryover candidate | B1 X4 |
| `estimate.clustered_dml_aipw` | `antecedent.estimators.ClusterDml.dyad_structure` | `dyadic_dependence_not_licensed` | carryover candidate | B1 X4 |
| `estimate.clustered_dml_aipw` | `antecedent_estimate.ClusterDml.few_components` | `too_few_clusters` | deliberate refusal | retain boundary |
| `estimate.clustered_dml_aipw` | `antecedent_estimate.flexible_learner_cluster_option` | `route_not_supported` | deliberate refusal | retain boundary |
| `estimate.clustered_dml_aipw` | `antecedent.estimators.DML.cluster_option` | `route_not_supported` | deliberate refusal | retain boundary |
| `estimate.descriptive_comparison` | `antecedent_estimate.refuse_transform_interval` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `estimate.descriptive_comparison` | `antecedent_estimate.refuse_column_attribution` | `effect_not_identified` | deliberate refusal | retain boundary |
| `estimate.descriptive_comparison` | `antecedent.descriptive.attribute_gap_to_columns` | `effect_not_identified` | deliberate refusal | retain boundary |
| `estimate.descriptive_comparison` | `antecedent.descriptive.interval` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `estimate.tier_diagnostics` | `antecedent.tier_diagnostics.unknown_scenarios` | `diagnostic_not_available` | deliberate refusal | retain boundary |
| `estimate.tier_diagnostics` | `antecedent.tier_diagnostics.refuter_evalue` | `diagnostic_not_available` | deliberate refusal | retain boundary |
| `estimate.tier_diagnostics` | `antecedent.tier_diagnostics.evalue_interval` | `cell_not_licensed` | deliberate refusal | retain boundary |
| `estimate.tier_diagnostics` | `antecedent.tier_diagnostics.non_tiered_result` | `invalid_argument` | deliberate refusal | retain boundary |
| `estimate.factorized_joint_cells` | `antecedent_estimate.refuse_joint_inference` | `penalized_interval_not_licensed` | carryover candidate | B4 vector treatment |
| `estimate.factorized_joint_cells` | `antecedent_estimate.refuse_joint_learner_inference` | `ml_nuisance_not_licensed` | carryover candidate | B4 vector treatment |
| `estimate.factorized_joint_cells` | `antecedent.derived.interval` | `penalized_interval_not_licensed` | carryover candidate | B4 vector treatment |
| `estimate.factorized_joint_cells` | `antecedent.derived.ml_interval` | `ml_nuisance_not_licensed` | carryover candidate | B4 vector treatment |
| `estimate.batch_retarget_covariance` | `antecedent.PreparedBatch.partial_family` | `cell_not_licensed` | carryover candidate | B4 vector treatment |
| `estimate.batch_retarget_covariance` | `antecedent.PreparedBatch.mixed_snapshot` | `row_weights_bound_to_snapshot` | deliberate refusal | retain boundary |
| `estimate.batch_retarget_covariance` | `antecedent.PreparedBatch.scores_unavailable` | `score_table_unavailable` | deliberate refusal | retain boundary |
| `estimate.dml_score_covariance` | `antecedent_estimate.DmlAte.partially_linear_scores` | `score_table_unavailable` | deliberate refusal | retain boundary |
| `estimate.dml_score_covariance` | `antecedent_estimate.DmlAte.trimmed_scores` | `score_table_unavailable` | deliberate refusal | retain boundary |
| `estimate.preflight_rank_drop` | `antecedent.estimate_with_rank_drop.other_estimator` | `route_not_supported` | deliberate refusal | retain boundary |
| `estimate.preflight_rank_drop` | `antecedent.estimate_with_rank_drop.joint_cell` | `route_not_supported` | carryover candidate | B4 vector treatment |
| `estimate.preflight_rank_drop` | `antecedent.estimate_with_rank_drop.treatment_alias` | `rank_drop_not_licensed` | deliberate refusal | retain boundary |
| `estimate.preflight_rank_drop` | `antecedent.estimate_with_rank_drop.separating_propensity` | `rank_drop_not_licensed` | deliberate refusal | retain boundary |
| `estimate.preflight_rank_drop` | `antecedent.estimate_with_rank_drop.span_not_preserved` | `rank_drop_not_licensed` | deliberate refusal | retain boundary |
